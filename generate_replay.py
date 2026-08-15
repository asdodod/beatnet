"""
generate_replay.py — turns any Beat Saber map into a .bsor replay

give it a map folder and a difficulty, it'll run the model and spit out
a replay file you can watch in-game or on beatleader's web viewer.

usage:
    python generate_replay.py <map_folder> [difficulty] [--model model.pt]

examples:
    python generate_replay.py "./Orchids" ExpertPlus
    python generate_replay.py "./Orchids" ExpertPlus --model BeatNet_model_ep50.pt --ddim_steps 15
"""

import argparse
import json
import struct
import re
import math
import os
import numpy as np
import torch
from pathlib import Path

from model import BeatNetModel
from rotation_utils import rot6d_to_quat

# generation params
FPS           = 90
WINDOW_FRAMES = 128
POSE_DIM      = 27        # 3pos + 6rot for head, left hand, right hand
LOOKAHEAD     = 12
NOTE_FEATURES = 7
STEP          = WINDOW_FRAMES // 2   # overlap windows by 50% for smoother blending

BSOR_MAGIC   = 0x442d3d69
BSOR_VERSION = 1

CUT_ANGLES = [270, 90, 180, 0, 315, 45, 225, 135, 0]   # per cutDirection enum


# --- map loading ---
def load_info(map_folder: Path) -> dict:
    for name in ("Info.dat", "info.dat"):
        p = map_folder / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No Info.dat found in {map_folder}")


def find_dat(map_folder: Path, difficulty: str):
    """Return (dat_path, bpm, characteristic) for a given difficulty."""
    info = load_info(map_folder)
    bpm  = float(info.get("_beatsPerMinute", 120))
    for bset in info.get("_difficultyBeatmapSets", []):
        characteristic = bset.get("_beatmapCharacteristicName", "Standard")
        for bmap in bset.get("_difficultyBeatmaps", []):
            if bmap.get("_difficulty", "").lower() == difficulty.lower():
                p = map_folder / bmap.get("_beatmapFilename", "")
                if p.exists():
                    return str(p), bpm, characteristic
    return None, bpm, "Standard"


def parse_notes(dat_path: str, bpm: float) -> list:
    """Parse v2/v3 beatmap and return sorted note list."""
    with open(dat_path, encoding="utf-8") as f:
        data = json.load(f)

    notes = []
    bps   = bpm / 60.0

    if "colorNotes" in data:
        for n in data["colorNotes"]:
            notes.append({
                "time":    n.get("b", 0) / bps,
                "x":       n.get("x", 0),
                "y":       n.get("y", 0),
                "color":   n.get("c", 0),
                "cut_dir": n.get("d", 8),
                "angle":   n.get("a", 0.0),
            })
    elif "_notes" in data:
        for n in data["_notes"]:
            notes.append({
                "time":    n.get("_time", 0) / bps,
                "x":       n.get("_lineIndex", 0),
                "y":       n.get("_lineLayer", 0),
                "color":   n.get("_type", 0),
                "cut_dir": n.get("_cutDirection", 8),
                "angle":   0.0,
            })

    notes.sort(key=lambda n: n["time"])
    return notes


# --- note context for the model ---
def build_note_context(notes: list, current_time: float) -> np.ndarray:
    ctx      = np.zeros((LOOKAHEAD, NOTE_FEATURES), dtype=np.float32)
    upcoming = [n for n in notes if n["time"] >= current_time][:LOOKAHEAD]
    for i, n in enumerate(upcoming):
        cut_dir = int(n["cut_dir"])
        if 1000 <= cut_dir <= 1360:
            base = cut_dir - 1000
        elif 0 <= cut_dir < len(CUT_ANGLES):
            base = CUT_ANGLES[cut_dir]
        else:
            base = 0.0
        angle_rad = np.radians(base + n["angle"])
        
        # Match build_dataset.py logic
        beat_offset = n["time"] - current_time
        
        # Match dataset.py normalization
        beat_offset = min(max(beat_offset, 0.0), 10.0) / 10.0
        
        ctx[i] = [
            beat_offset,
            (n["x"] - 1.5) / 1.5,
            (n["y"] - 1.0) / 1.0,
            float(n["color"]),
            np.cos(angle_rad),
            np.sin(angle_rad),
            n["angle"] / 180.0,
        ]
        
    # Clamp features 1-6 to [-2, 2] exactly like dataset.py
    ctx[:, 1:] = np.clip(ctx[:, 1:], -2.0, 2.0)
    return ctx


# --- DDIM sampler (fast denoising) ---
def ddim_sample(model, note_ctx: torch.Tensor, device,
                num_steps: int = 1000, ddim_steps: int = 50) -> np.ndarray:
    """DDIM sampling — 50 forward passes instead of 1000."""
    scale     = 1000 / num_steps
    betas     = torch.linspace(scale * 0.0001, scale * 0.02, num_steps)
    alphas_cp = torch.cumprod(1.0 - betas, dim=0).to(device)
    timesteps = torch.linspace(num_steps - 1, 0, ddim_steps, dtype=torch.long)
    x         = torch.randn(1, WINDOW_FRAMES, POSE_DIM, device=device)

    model.eval()
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_norm     = torch.tensor([[t.float() / num_steps]], device=device)
            noise_pred = model(x, t_norm, note_ctx)
            alpha_t    = alphas_cp[t]
            alpha_next = alphas_cp[timesteps[i + 1]] if i < len(timesteps) - 1 \
                         else torch.tensor(1.0, device=device)
            x0_pred = ((x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()).clamp(-5, 5)
            x       = alpha_next.sqrt() * x0_pred + (1 - alpha_next).sqrt() * noise_pred

    return x.squeeze(0).cpu().numpy()


# --- main generation loop ---
def generate_frames(model, notes: list, duration: float, device, ddim_steps: int = 50, model_path: str = "."):
    frame_times = np.arange(0, duration, 1.0 / FPS, dtype=np.float32)
    N           = len(frame_times)
    out_poses   = np.zeros((N, POSE_DIM), dtype=np.float32)
    weights     = np.zeros((N, 1),        dtype=np.float32)
    total_w     = max(1, (N + STEP - 1) // STEP)

    print(f"  Generating {N} frames in ~{total_w} windows (DDIM steps={ddim_steps})...")
    
    # Check for normalization stats
    model_dir = os.path.dirname(model_path) if model_path and os.path.dirname(model_path) else "."
    mean_path = os.path.join(model_dir, "pose_mean.npy")
    std_path = os.path.join(model_dir, "pose_std.npy")
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        pose_mean = np.load(mean_path)
        pose_std = np.load(std_path)
    elif os.path.exists("pose_mean.npy") and os.path.exists("pose_std.npy"):
        pose_mean = np.load("pose_mean.npy")
        pose_std = np.load("pose_std.npy")
    else:
        print("  WARNING: Normalization stats (pose_mean.npy, pose_std.npy) not found! Output will be garbage.")
        pose_mean = np.zeros(POSE_DIM)
        pose_std = np.ones(POSE_DIM)

    for w, start in enumerate(range(0, N, STEP)):
        end  = min(start + WINDOW_FRAMES, N)
        wlen = end - start
        if wlen < 4:
            break

        ctx_t  = torch.from_numpy(
            build_note_context(notes, float(frame_times[start]))
        ).unsqueeze(0).to(device)

        window = ddim_sample(model, ctx_t, device, ddim_steps=ddim_steps)[:wlen]
        blend  = np.hanning(wlen).reshape(-1, 1).astype(np.float32)
        out_poses[start:end] += window * blend
        weights[start:end]   += blend

        if w % max(1, total_w // 10) == 0:
            print(f"    {int(w / total_w * 100)}%")

    out_poses /= np.maximum(weights, 1e-8)
    
    # Denormalize poses
    out_poses = (out_poses * pose_std) + pose_mean
    
    # Smooth the poses over 5 frames to reduce jitter
    from scipy.ndimage import uniform_filter1d
    out_poses = uniform_filter1d(out_poses, size=5, axis=0)
    
    # Convert back to 14D (positions + quaternions)
    print("  Converting 6D rotations back to quaternions...")
    out_poses_t = torch.from_numpy(out_poses)
    
    h_pos = out_poses_t[:, 0:3]
    h_rot_6d = out_poses_t[:, 3:9]
    l_pos = out_poses_t[:, 9:12]
    l_rot_6d = out_poses_t[:, 12:18]
    r_pos = out_poses_t[:, 18:21]
    r_rot_6d = out_poses_t[:, 21:27]
    
    h_quat = rot6d_to_quat(h_rot_6d)
    l_quat = rot6d_to_quat(l_rot_6d)
    r_quat = rot6d_to_quat(r_rot_6d)
    
    final_poses_21d = torch.cat([h_pos, h_quat, l_pos, l_quat, r_pos, r_quat], dim=-1).numpy()

    print("  100% — done!")
    return final_poses_21d, frame_times


# --- bsor file writer ---
def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<i", len(b)) + b

def _norm_q(x, y, z, w):
    L = math.sqrt(x*x + y*y + z*z + w*w)
    return (x/L, y/L, z/L, w/L) if L > 1e-8 else (0.0, 0.0, 0.0, 1.0)


def write_bsor(out_path, frame_times, poses, notes, song_name, song_hash, difficulty, characteristic="Standard"):
    buf = bytearray()
    buf += struct.pack("<iB", BSOR_MAGIC, BSOR_VERSION)

    # Block 0 — Info
    buf += struct.pack("B", 0)
    for s in ["1.0.0", "1.40.0", "0", "beatnet_ai", "BeatNet AI",
              "pc", "OpenVR", "Oculus Rift S", "Oculus Touch",
              song_hash, song_name, "BeatNet", difficulty]:
        buf += _pack_str(s)
    buf += struct.pack("<i", 0)
    for s in [characteristic, "DefaultEnvironment", ""]:
        buf += _pack_str(s)
    buf += struct.pack("<f?fff", 18.0, False, 1.8, 0.0, 0.0)
    buf += struct.pack("<f", 1.0)

    # Block 1 — Frames
    buf += struct.pack("B", 1)
    buf += struct.pack("<i", len(frame_times))
    for i, t in enumerate(frame_times):
        p = poses[i]
        hx, hy, hz          = float(p[0]),  float(p[1]),  float(p[2])
        hqx, hqy, hqz, hqw = _norm_q(float(p[3]), float(p[4]), float(p[5]), float(p[6]))
        lx, ly, lz          = float(p[7]),  float(p[8]),  float(p[9])
        lqx, lqy, lqz, lqw = _norm_q(float(p[10]), float(p[11]), float(p[12]), float(p[13]))
        rx, ry, rz          = float(p[14]), float(p[15]), float(p[16])
        rqx, rqy, rqz, rqw = _norm_q(float(p[17]), float(p[18]), float(p[19]), float(p[20]))

        buf += struct.pack("<fi",  float(t), int(FPS))   # time=float, fps=int32
        buf += struct.pack("<fff",  hx,  hy,  hz)        # head position
        buf += struct.pack("<ffff", hqx, hqy, hqz, hqw)  # head rotation
        buf += struct.pack("<fff",  lx,  ly,  lz)        # left hand position
        buf += struct.pack("<ffff", lqx, lqy, lqz, lqw)  # left hand rotation
        buf += struct.pack("<fff",  rx,  ry,  rz)        # right hand position
        buf += struct.pack("<ffff", rqx, rqy, rqz, rqw)  # right hand rotation

    # Block 2 — Notes
    buf += struct.pack("B", 2)
    buf += struct.pack("<i", len(notes))
    
    for n in notes:
        t = n["time"]
        # Find closest frame
        idx = np.argmin(np.abs(frame_times - t))
        if idx >= len(poses):
            idx = len(poses) - 1
            
        p = poses[idx]
        lx, ly, lz = float(p[7]), float(p[8]), float(p[9])
        rx, ry, rz = float(p[14]), float(p[15]), float(p[16])
        
        note_x = (n["x"] - 1.5) * 0.6
        note_y = n["y"] * 0.6 + 0.85
        color = int(n["color"])
        
        if color == 0:
            dist = math.sqrt((lx - note_x)**2 + (ly - note_y)**2)
        else:
            dist = math.sqrt((rx - note_x)**2 + (ry - note_y)**2)
            
        is_hit = dist < 1.2
        
        # NoteID formula used by some parsers: color*10000 + x*1000 + y*100 + cutDir*10
        noteID = int(color * 10000 + n["x"] * 1000 + n["y"] * 100 + n["cut_dir"] * 10)
        
        eventType = 0 if is_hit else 2  # 0=Good, 2=Miss
        buf += struct.pack("<iffi", noteID, float(t), float(t - 2.0), int(eventType))
        
        if is_hit:
            # 72 bytes of NoteCutInfo
            # <???? f fff i f f fff fff f f f f -> 4 bools, 16 floats, 1 int = 21 args
            buf += struct.pack("<????ffffiffffffffffffffff",
                True, True, True, False,            # speedOK, dirOK, saberTypeOK, wasCutTooSoon
                10.0, 0.0, 1.0, 0.0, int(color),    # saberSpeed, saberDir (vec3), saberType
                0.0, 0.0,                           # timeDev, cutDirDev
                float(note_x), float(note_y), 0.0,  # cutPoint (vec3)
                0.0, 0.0, 1.0,                      # cutNormal (vec3)
                0.0, 0.0, 1.0, 1.0                  # dist, angle, before rating, after rating
            )
            
    # Blocks 3-5 — empty
    for bid in [3, 4, 5]:
        buf += struct.pack("<Bi", bid, 0)   # explicit < to avoid padding bytes

    with open(out_path, "wb") as f:
        f.write(buf)

    print(f"\n✓ Replay saved → {out_path}")
    print(f"  Frames: {len(frame_times)}   Duration: {frame_times[-1]:.1f}s")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="BeatNet replay generator")
    ap.add_argument("map_folder",   help="Path to map folder containing Info.dat")
    ap.add_argument("difficulty",   nargs="?", default="ExpertPlus")
    ap.add_argument("--model",      default="BeatNet_model_ep100.pt")
    ap.add_argument("--output",     default=None)
    ap.add_argument("--ddim_steps", type=int, default=50,
                    help="DDIM sampling steps (default 50; more = better but slower)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        return

    print(f"Loading model: {args.model}")
    model = BeatNetModel().to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('avg_loss', '?'):.6f}")
    else:
        model.load_state_dict(ckpt)
    model.eval()

    map_folder = Path(args.map_folder)
    info       = load_info(map_folder)
    song_name  = info.get("_songName", "Unknown")
    bpm        = float(info.get("_beatsPerMinute", 120))
    
    # Calculate map hash if not present
    song_hash = info.get("_songHash")
    if not song_hash:
        import hashlib
        sha1 = hashlib.sha1()
        
        info_file = None
        for name in ("Info.dat", "info.dat"):
            if (map_folder / name).exists():
                info_file = map_folder / name
                break
                
        if info_file:
            sha1.update(info_file.read_bytes())
            
        for bset in info.get('_difficultyBeatmapSets', []):
            for bmap in bset.get('_difficultyBeatmaps', []):
                diff_filename = bmap.get('_beatmapFilename')
                if diff_filename:
                    diff_path = map_folder / diff_filename
                    if diff_path.exists():
                        sha1.update(diff_path.read_bytes())
                        
        song_hash = sha1.hexdigest().upper()
    else:
        song_hash = song_hash.upper()

    print(f"Map: {song_name}  |  BPM: {bpm}  |  Difficulty: {args.difficulty}")

    dat_path, bpm, characteristic = find_dat(map_folder, args.difficulty)
    if not dat_path:
        print(f"ERROR: Difficulty '{args.difficulty}' not found in map")
        return

    notes = parse_notes(dat_path, bpm)
    if not notes:
        print("ERROR: No notes found in beatmap")
        return

    duration = notes[-1]["time"] + 3.0
    print(f"Notes: {len(notes)}  |  Duration: {duration:.1f}s")

    poses, frame_times = generate_frames(model, notes, duration, device, args.ddim_steps, args.model)

    if args.output:
        out_path = args.output
    else:
        safe     = re.sub(r"[^\w\-]", "_", song_name)
        out_path = f"{safe}_{args.difficulty}_{characteristic}_AI.bsor"

    write_bsor(out_path, frame_times, poses, notes, song_name, song_hash, args.difficulty, characteristic)


if __name__ == "__main__":
    main()
