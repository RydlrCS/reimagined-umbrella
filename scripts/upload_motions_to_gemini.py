#!/usr/bin/env python3
"""
Upload capoeira and breakdance related FBX files to Gemini Files API
"""

import os
import json
import time
from pathlib import Path
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD4TVYMnyyEfUBvhLXioA6FckEqmYz4Fo8")
genai.configure(api_key=GEMINI_API_KEY)

# Source directory
source_dir = Path("/workspaces/reimagined-umbrella/michelle.fbx")

# Keywords for filtering
keywords = ["capoeira", "break", "kick", "flip", "spin", "dance", "fight", "combat", "martial", "jump"]

# Find relevant files
if not source_dir.exists():
    print(f"❌ Source directory not found: {source_dir}")
    exit(1)

all_files = list(source_dir.glob("*.fbx"))
print(f"📁 Found {len(all_files)} total FBX files")

# Filter for relevant files
relevant_files = []
for fbx_file in all_files:
    name_lower = fbx_file.stem.lower()
    if any(keyword in name_lower for keyword in keywords):
        relevant_files.append(fbx_file)

print(f"🎯 Found {len(relevant_files)} relevant files matching keywords")
print()

# Limit to 10 files
files_to_upload = relevant_files[:10]

print(f"📤 Uploading {len(files_to_upload)} files to Gemini Files API...")
print()

# Assign icons based on animation type
def get_icon(name):
    name_lower = name.lower()
    if "capoeira" in name_lower:
        return "🥋"
    elif "break" in name_lower or "spin" in name_lower:
        return "🕺"
    elif "kick" in name_lower:
        return "🦵"
    elif "flip" in name_lower or "jump" in name_lower:
        return "🤸"
    elif "dance" in name_lower:
        return "💃"
    elif "fight" in name_lower or "combat" in name_lower:
        return "🥊"
    else:
        return "🎭"

# Upload files
motions = []
for i, fbx_file in enumerate(files_to_upload, 1):
    print(f"[{i}/{len(files_to_upload)}] 📤 {fbx_file.name}")
    print(f"   Size: {fbx_file.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # Upload to Gemini
        uploaded_file = genai.upload_file(
            path=str(fbx_file),
            display_name=fbx_file.stem
        )
        
        # Wait for processing
        print(f"   Processing...", end="", flush=True)
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            print(".", end="", flush=True)
            uploaded_file = genai.get_file(uploaded_file.name)
        print()
        
        if uploaded_file.state.name == "ACTIVE":
            print(f"   ✅ Uploaded to Gemini")
            print(f"   URI: {uploaded_file.uri[:60]}...")
            
            motion_id = fbx_file.stem.lower().replace(" ", "_").replace("@", "_")
            
            motions.append({
                "id": motion_id,
                "name": fbx_file.stem,
                "filename": fbx_file.name,
                "gemini_uri": uploaded_file.uri,
                "gemini_name": uploaded_file.name,
                "icon": get_icon(fbx_file.stem),
                "frames": 100,
                "fps": 30,
                "use_gemini": True
            })
        else:
            print(f"   ❌ Upload failed: {uploaded_file.state.name}")
        
        print()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
        continue

# Save manifest
manifest_path = Path("/workspaces/reimagined-umbrella/src/kinetic_ledger/ui/motion_library.json")

manifest = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "source": "Gemini Files API",
    "total_motions": len(motions),
    "motions": motions
}

with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Manifest saved: {manifest_path}")
print(f"   Total motions uploaded: {len(motions)}")
print()
print("Motion Library:")
for i, motion in enumerate(motions, 1):
    print(f"{i}. {motion['icon']} {motion['name']}")
