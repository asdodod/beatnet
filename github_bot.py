import os
import re
import sys
import json
import zipfile
import shutil
import urllib.request
import subprocess
from pathlib import Path

# Limits
MAX_DURATION_SECONDS = 300  # 5 minutes

def send_comment(issue_number, body):
    # Uses GitHub CLI to post a comment
    print(f"Posting comment to issue #{issue_number}...")
    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        f.write(body)
        temp_name = f.name
    
    subprocess.run(["gh", "issue", "comment", str(issue_number), "--body-file", temp_name])
    os.remove(temp_name)

def close_issue(issue_number):
    subprocess.run(["gh", "issue", "close", str(issue_number)])

def main():
    issue_number = os.environ.get("ISSUE_NUMBER")
    issue_body = os.environ.get("ISSUE_BODY", "")
    author = os.environ.get("AUTHOR", "User")
    
    if not issue_number:
        print("Missing ISSUE_NUMBER environment variable.")
        sys.exit(1)
        
    print(f"Processing Issue #{issue_number} from @{author}")
    
    # 1. Parse Issue Body
    # Look for the map ID and difficulty in the Markdown created by the issue template.
    # The template puts the ID below "### BeatSaver Map Key"
    map_id = None
    difficulty = "ExpertPlus"
    
    lines = issue_body.split("\n")
    for i, line in enumerate(lines):
        if "BeatSaver Map Key" in line:
            if i + 2 < len(lines):
                map_id = lines[i + 2].strip()
        if "Difficulty" in line:
            if i + 2 < len(lines):
                difficulty = lines[i + 2].strip()
                
    if not map_id:
        send_comment(issue_number, f"@{author} Error: Could not find a map ID in your request. Please fill out the form correctly.")
        close_issue(issue_number)
        sys.exit(0)
        
    print(f"Extracted Map ID: {map_id}, Difficulty: {difficulty}")
    
    # 2. Fetch Map Data from BeatSaver
    api_url = f"https://api.beatsaver.com/maps/id/{map_id}"
    req = urllib.request.Request(api_url, headers={'User-Agent': 'BeatNet/1.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        send_comment(issue_number, f"@{author} Error: Could not find map with ID `{map_id}` on BeatSaver. Double-check the ID.")
        close_issue(issue_number)
        sys.exit(0)
        
    # Check map duration
    duration = data.get("metadata", {}).get("duration", 0)
    print(f"Map duration: {duration} seconds")
    
    if duration > MAX_DURATION_SECONDS:
        send_comment(issue_number, f"@{author} Error: Song duration is {duration} seconds. Due to server limits, maps longer than {MAX_DURATION_SECONDS} seconds (5 minutes) are not allowed.")
        close_issue(issue_number)
        sys.exit(0)
        
    # 3. Download the Map Zip
    download_url = data["versions"][0]["downloadURL"]
    print(f"Downloading map zip from {download_url}...")
    
    zip_path = "map.zip"
    map_dir = "map_folder"
    os.makedirs(map_dir, exist_ok=True)
    
    req = urllib.request.Request(download_url, headers={'User-Agent': 'BeatNet/1.0'})
    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(map_dir)
        
    # 4. Run generate_replay.py
    # Note: the workflow must download BeatNet_model_ep15.pt and generate_replay.py to this folder first
    print("Running AI generation...")
    # We use 15 ddim steps to make it fast enough for GitHub Actions CPU (~10-15 mins)
    cmd = [
        sys.executable, "generate_replay.py", map_dir, difficulty, 
        "--model", "BeatNet_model_ep15.pt", "--ddim_steps", "15"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        send_comment(issue_number, f"@{author} Error: Internal failure during generation. The map might not have the `{difficulty}` difficulty.")
        close_issue(issue_number)
        sys.exit(1)
        
    # Find the generated bsor
    bsor_files = list(Path(".").glob("*.bsor"))
    if not bsor_files:
        send_comment(issue_number, f"@{author} Error: No replay file was generated. The AI script might have crashed.")
        close_issue(issue_number)
        sys.exit(1)
        
    bsor_file = str(bsor_files[0])
    print(f"Success! Generated: {bsor_file}")
    
    # Write the output filename to a specific file so the workflow can upload it
    with open("output_file.txt", "w", encoding="utf-8") as f:
        f.write(bsor_file)

if __name__ == "__main__":
    main()
