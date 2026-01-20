#!/usr/bin/env python3
"""
Upload FBX files from michelle.fbx folder to Gemini Files API and generate JSON manifest.
"""

import os
import sys
import json
import time
from pathlib import Path
import google.generativeai as genai

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY environment variable not set")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

def upload_fbx_files(directory: str, output_file: str = "motion_library.json"):
    """Upload all FBX files and create manifest."""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        sys.exit(1)
    
    fbx_files = sorted(directory.glob("*.fbx"))
    
    if not fbx_files:
        print(f"❌ No FBX files found in {directory}")
        sys.exit(1)
    
    print(f"📁 Found {len(fbx_files)} FBX files")
    print(f"   Directory: {directory}")
    print()
    
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_directory": str(directory),
        "total_files": len(fbx_files),
        "motions": []
    }
    
    for i, fbx_path in enumerate(fbx_files, 1):
        print(f"[{i}/{len(fbx_files)}] 📤 Uploading: {fbx_path.name}")
        print(f"   Size: {fbx_path.stat().st_size / (1024*1024):.2f} MB")
        
        try:
            # Upload to Gemini
            uploaded_file = genai.upload_file(
                path=str(fbx_path),
                display_name=fbx_path.stem
            )
            
            # Wait for processing
            print(f"   Processing...", end="", flush=True)
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(1)
                print(".", end="", flush=True)
                uploaded_file = genai.get_file(uploaded_file.name)
            print()
            
            if uploaded_file.state.name == "FAILED":
                print(f"   ❌ Upload failed")
                continue
            
            print(f"   ✅ Uploaded successfully!")
            print(f"   URI: {uploaded_file.uri}")
            print()
            
            # Add to manifest
            motion_data = {
                "id": fbx_path.stem.lower().replace(" ", "_"),
                "name": fbx_path.stem,
                "filename": fbx_path.name,
                "gemini_uri": uploaded_file.uri,
                "gemini_name": uploaded_file.name,
                "mime_type": uploaded_file.mime_type,
                "size_bytes": fbx_path.stat().st_size,
                "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            manifest["motions"].append(motion_data)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
            continue
    
    # Save manifest
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Upload complete!")
    print(f"   Uploaded: {len(manifest['motions'])}/{len(fbx_files)} files")
    print(f"   Manifest: {output_path.absolute()}")
    
    return manifest

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload FBX files to Gemini and generate manifest")
    parser.add_argument("directory", help="Directory containing FBX files")
    parser.add_argument("--output", default="motion_library.json", help="Output manifest file")
    parser.add_argument("--limit", type=int, help="Limit number of files to upload")
    
    args = parser.parse_args()
    
    manifest = upload_fbx_files(args.directory, args.output)
