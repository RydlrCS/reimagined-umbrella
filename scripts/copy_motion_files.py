#!/usr/bin/env python3
"""
Copy capoeira and breakdance related FBX files from michelle.fbx to data directory
and update motion_library.json
"""

import os
import shutil
import json
from pathlib import Path

# Source and destination
source_dir = Path("/workspaces/reimagined-umbrella/michelle.fbx")
dest_dir = Path("/workspaces/reimagined-umbrella/data/mixamo_anims/fbx/michele")
dest_dir.mkdir(parents=True, exist_ok=True)

# Keywords for filtering
keywords = ["capoeira", "break", "kick", "flip", "spin", "dance", "fight", "combat", "martial", "jump"]

# Find all FBX files
if not source_dir.exists():
    print(f"❌ Source directory not found: {source_dir}")
    exit(1)

all_files = list(source_dir.glob("*.fbx"))
print(f"Found {len(all_files)} total FBX files in source")

# Filter for relevant files
relevant_files = []
for fbx_file in all_files:
    name_lower = fbx_file.stem.lower()
    if any(keyword in name_lower for keyword in keywords):
        relevant_files.append(fbx_file)

print(f"Found {len(relevant_files)} relevant files matching keywords")
print()

# Limit to 12 files (including the 2 we already have)
files_to_copy = relevant_files[:12]

# Copy files
copied_files = []
for i, fbx_file in enumerate(files_to_copy, 1):
    dest_file = dest_dir / fbx_file.name
    
    if dest_file.exists():
        print(f"[{i}/{len(files_to_copy)}] ⏭️  Already exists: {fbx_file.name}")
    else:
        print(f"[{i}/{len(files_to_copy)}] 📋 Copying: {fbx_file.name}")
        shutil.copy2(fbx_file, dest_file)
        print(f"   ✅ Copied ({fbx_file.stat().st_size / (1024*1024):.2f} MB)")
    
    copied_files.append(fbx_file.name)
    print()

# Update motion_library.json
manifest_path = Path("/workspaces/reimagined-umbrella/src/kinetic_ledger/ui/motion_library.json")

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

motions = []
for filename in copied_files:
    motion_id = Path(filename).stem.lower().replace(" ", "_").replace("@", "_")
    name = Path(filename).stem
    
    motions.append({
        "id": motion_id,
        "name": name,
        "filename": filename,
        "icon": get_icon(name),
        "frames": 100,  # Default, will be updated when loaded
        "fps": 30,
        "local_path": f"/static/models/data/mixamo_anims/fbx/michele/{filename}",
        "use_gemini": False
    })

manifest = {
    "created_at": "2026-01-14T12:30:00",
    "motions": motions
}

with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Updated manifest: {manifest_path}")
print(f"   Total motions: {len(motions)}")
print()
print("Motion library:")
for i, motion in enumerate(motions, 1):
    print(f"{i}. {motion['icon']} {motion['name']}")
