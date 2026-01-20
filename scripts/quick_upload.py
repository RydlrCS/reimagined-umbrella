#!/usr/bin/env python3
"""
Quick upload script for Capoeira and Breakdance only (for testing).
"""

import os
import json
import time
from pathlib import Path
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD4TVYMnyyEfUBvhLXioA6FckEqmYz4Fo8")
genai.configure(api_key=GEMINI_API_KEY)

# Find FBX files
michelle_dir = Path("michelle.fbx")
if not michelle_dir.exists():
    print(f"❌ Directory not found: {michelle_dir}")
    exit(1)

# Search for Capoeira and Breakdance files
all_files = list(michelle_dir.glob("*.fbx"))
capoeira_files = [f for f in all_files if "capoeira" in f.name.lower()]
breakdance_files = [f for f in all_files if "break" in f.name.lower()]

files_to_upload = []
if capoeira_files:
    files_to_upload.append(capoeira_files[0])
    print(f"Found Capoeira: {capoeira_files[0].name}")
if breakdance_files:
    files_to_upload.append(breakdance_files[0])
    print(f"Found Breakdance: {breakdance_files[0].name}")

if not files_to_upload:
    print(f"❌ No Capoeira or Breakdance files found in {michelle_dir}")
    print(f"Available files: {[f.name for f in all_files[:5]]}")
    exit(1)

print(f"\nUploading {len(files_to_upload)} files...\n")

manifest = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "motions": []
}

for fbx_path in files_to_upload:
    print(f"📤 Uploading: {fbx_path.name}")
    print(f"   Size: {fbx_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        uploaded_file = genai.upload_file(path=str(fbx_path), display_name=fbx_path.stem)
        
        print(f"   Processing...", end="", flush=True)
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            print(".", end="", flush=True)
            uploaded_file = genai.get_file(uploaded_file.name)
        print()
        
        if uploaded_file.state.name == "ACTIVE":
            print(f"   ✅ URI: {uploaded_file.uri}")
            print()
            
            motion_id = fbx_path.stem.lower().replace(" ", "_")
            icon = "🥋" if "capoeira" in fbx_path.stem.lower() else "🕺"
            
            manifest["motions"].append({
                "id": motion_id,
                "name": fbx_path.stem,
                "filename": fbx_path.name,
                "gemini_uri": uploaded_file.uri,
                "gemini_name": uploaded_file.name,
                "icon": icon,
                "frames": 103 if "capoeira" in fbx_path.stem.lower() else 120,
                "fps": 30,
                "use_gemini": True
            })
        else:
            print(f"   ❌ Upload failed: {uploaded_file.state.name}")
            print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()

# Save manifest
output_path = Path("src/kinetic_ledger/ui/motion_library.json")
with open(output_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Manifest saved: {output_path}")
print(f"   Uploaded {len(manifest['motions'])} motions")
