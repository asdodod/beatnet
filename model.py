"""
VainAI — model architecture

Encoder-decoder transformer for motion diffusion.
The encoder reads upcoming notes, the decoder denoises pose sequences
conditioned on those notes + a diffusion timestep.

Nothing crazy here, just standard transformer stuff with pre-norm
and cross-attention. The magic is in the data, not the architecture.
"""

import torch
import torch.nn as nn
import math

# these have to match whatever the training data was built with
POSE_DIM        = 27    # 3 pos + 6 rot per body part (head, left, right)
NOTE_DIM        = 7     # beat_offset, x, y, color, cos, sin, angle
LOOKAHEAD_NOTES = 12
HIDDEN_DIM      = 512


class PositionalEncoding(nn.Module):
    """standard sinusoidal PE, nothing fancy"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class NoteEncoder(nn.Module):
    """reads the next N notes and builds a context sequence for the decoder"""
    def __init__(self, note_dim=NOTE_DIM, hidden_dim=HIDDEN_DIM, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(note_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True,
            dim_feedforward=hidden_dim * 2, dropout=0.1, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, notes):
        x = self.embedding(notes)
        x = self.pos_encoder(x)
        return self.transformer(x)


class MotionDiffusionDecoder(nn.Module):
    """
    takes noisy poses + timestep + note context, predicts the noise.
    basically a conditional denoiser — the decoder cross-attends to
    the note encoder output so it knows what notes are coming up.
    """
    def __init__(self, pose_dim=POSE_DIM, hidden_dim=HIDDEN_DIM, num_layers=6):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(pose_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True,
            dim_feedforward=hidden_dim * 2, dropout=0.1, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.final_norm  = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, pose_dim)

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t).unsqueeze(1)
        x_emb = self.input_proj(x)
        x_in  = self.pos_encoder(x_emb) + t_emb

        out = self.transformer(tgt=x_in, memory=cond)
        out = self.final_norm(out)
        return self.output_proj(out)


class BeatNetModel(nn.Module):
    """full model — just wires the encoder and decoder together"""
    def __init__(self):
        super().__init__()
        self.encoder = NoteEncoder()
        self.decoder = MotionDiffusionDecoder()

    def forward(self, x, t, notes):
        cond = self.encoder(notes)
        return self.decoder(x, t, cond)
