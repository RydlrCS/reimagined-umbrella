#!/usr/bin/env python3
"""
Upload Michelle FBX motion files to Gemini Files API.

This script:
1. Scans specified directory for .fbx files
2. Uploads each to Gemini Files API
3. Stores file URIs in a library manifest
4. Optionally generates embeddings for each motion
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kinetic_ledger.services.gemini_motion_embedder import GeminiMotionEmbedder, FBXMotionData


class MicheleLibraryUploader:
    """Upload and manage Michelle FBX motion library in Gemini."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = "data/mixamo_anims/fbx/michele"
    ):
        """
        Initialize uploader.
        
        Args:
            api_key: Gemini API key (uses GEMINI_API_KEY env var if not provided)
            output_dir: Directory to save manifest and metadata
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Get your key from: https://aistudio.google.com/apikey"
            )
        
        genai.configure(api_key=self.api_key)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest_path = self.output_dir / "gemini_library_manifest.json"
        self.manifest: Dict = self._load_manifest()
        
        print(f"✅ MicheleLibraryUploader initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Manifest: {self.manifest_path}")
    
    def _load_manifest(self) -> Dict:
        """Load existing manifest or create new one."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_files": 0,
            "files": {}
        }
    
    def _save_manifest(self):
        """Save manifest to disk."""
        self.manifest["updated_at"] = datetime.now().isoformat()
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        print(f"💾 Manifest saved: {self.manifest_path}")
    
    def upload_file(
        self,
        fbx_path: str,
        display_name: Optional[str] = None
    ) -> genai.File:
        """
        Upload single FBX file to Gemini.
        
        Args:
            fbx_path: Path to FBX file
            display_name: Optional display name
        
        Returns:
            Gemini File object
        """
        fbx_path = Path(fbx_path)
        if not fbx_path.exists():
            raise FileNotFoundError(f"FBX file not found: {fbx_path}")
        
        # Check if already uploaded
        file_key = fbx_path.name
        if file_key in self.manifest["files"]:
            existing = self.manifest["files"][file_key]
            print(f"⏭️  Already uploaded: {file_key}")
            print(f"   URI: {existing['uri']}")
            
            # Try to get existing file
            try:
                existing_file = genai.get_file(existing['gemini_name'])
                if existing_file.state.name == "ACTIVE":
                    return existing_file
                else:
                    print(f"   ⚠️  File state: {existing_file.state.name}, re-uploading...")
            except Exception as e:
                print(f"   ⚠️  Could not retrieve existing file: {e}, re-uploading...")
        
        # Upload file
        display_name = display_name or fbx_path.stem
        
        print(f"📤 Uploading: {fbx_path.name}")
        print(f"   Size: {fbx_path.stat().st_size / (1024*1024):.2f} MB")
        
        uploaded_file = genai.upload_file(
            path=str(fbx_path),
            display_name=display_name
        )
        
        # Wait for processing
        print(f"   Processing...", end="", flush=True)
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            print(".", end="", flush=True)
            uploaded_file = genai.get_file(uploaded_file.name)
        print()
        
        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"File upload failed: {uploaded_file.name}")
        
        print(f"   ✅ Uploaded successfully!")
        print(f"   URI: {uploaded_file.uri}")
        print(f"   Gemini Name: {uploaded_file.name}")
        
        # Update manifest
        self.manifest["files"][file_key] = {
            "file_name": fbx_path.name,
            "display_name": display_name,
            "local_path": str(fbx_path.absolute()),
            "uri": uploaded_file.uri,
            "gemini_name": uploaded_file.name,
            "mime_type": uploaded_file.mime_type,
            "size_bytes": fbx_path.stat().st_size,
            "state": uploaded_file.state.name,
            "uploaded_at": datetime.now().isoformat()
        }
        self.manifest["total_files"] = len(self.manifest["files"])
        self._save_manifest()
        
        return uploaded_file
    
    def upload_directory(
        self,
        directory: str,
        pattern: str = "*.fbx",
        limit: Optional[int] = None
    ) -> List[genai.File]:
        """
        Upload all FBX files from directory.
        
        Args:
            directory: Directory containing FBX files
            pattern: File pattern to match (default: *.fbx)
            limit: Maximum number of files to upload (None = all)
        
        Returns:
            List of uploaded File objects
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all FBX files
        fbx_files = sorted(directory.glob(pattern))
        
        if not fbx_files:
            print(f"⚠️  No files matching '{pattern}' found in {directory}")
            return []
        
        if limit:
            fbx_files = fbx_files[:limit]
        
        print(f"\n📁 Found {len(fbx_files)} FBX files to upload")
        print(f"   Directory: {directory}")
        print()
        
        uploaded_files = []
        for i, fbx_path in enumerate(fbx_files, 1):
            print(f"[{i}/{len(fbx_files)}] ", end="")
            try:
                uploaded_file = self.upload_file(fbx_path)
                uploaded_files.append(uploaded_file)
            except Exception as e:
                print(f"   ❌ Upload failed: {e}")
                continue
            print()
        
        print(f"\n✅ Upload complete: {len(uploaded_files)}/{len(fbx_files)} files")
        return uploaded_files
    
    def list_uploaded_files(self):
        """List all uploaded files in manifest."""
        if not self.manifest["files"]:
            print("No files uploaded yet.")
            return
        
        print(f"\n📚 Gemini Library Manifest ({self.manifest['total_files']} files)")
        print(f"   Last updated: {self.manifest['updated_at']}")
        print()
        
        for i, (key, file_info) in enumerate(self.manifest["files"].items(), 1):
            print(f"{i}. {file_info['display_name']}")
            print(f"   File: {file_info['file_name']}")
            print(f"   URI: {file_info['uri']}")
            print(f"   Size: {file_info['size_bytes'] / (1024*1024):.2f} MB")
            print(f"   Uploaded: {file_info['uploaded_at']}")
            print()
    
    def verify_uploads(self) -> Dict[str, str]:
        """
        Verify all uploaded files are still accessible.
        
        Returns:
            Dict mapping file names to their states
        """
        print(f"\n🔍 Verifying {self.manifest['total_files']} uploaded files...")
        
        states = {}
        for file_key, file_info in self.manifest["files"].items():
            try:
                gemini_file = genai.get_file(file_info['gemini_name'])
                state = gemini_file.state.name
                states[file_key] = state
                
                status = "✅" if state == "ACTIVE" else "⚠️"
                print(f"   {status} {file_key}: {state}")
                
            except Exception as e:
                states[file_key] = f"ERROR: {e}"
                print(f"   ❌ {file_key}: {e}")
        
        return states


def main():
    """Main entry point for script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload Michelle FBX motion files to Gemini Files API"
    )
    parser.add_argument(
        "directory",
        help="Directory containing FBX files (e.g., ~/Downloads/michelle.fbx)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/mixamo_anims/fbx/michele",
        help="Output directory for manifest (default: data/mixamo_anims/fbx/michele)"
    )
    parser.add_argument(
        "--pattern",
        default="*.fbx",
        help="File pattern to match (default: *.fbx)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of files to upload (default: all)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List currently uploaded files"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify all uploaded files are accessible"
    )
    
    args = parser.parse_args()
    
    # Expand ~ in path
    directory = Path(args.directory).expanduser()
    
    # Initialize uploader
    try:
        uploader = MicheleLibraryUploader(output_dir=args.output_dir)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print()
        print("To get your Gemini API key:")
        print("1. Go to https://aistudio.google.com/apikey")
        print("2. Create an API key")
        print("3. Set it: export GEMINI_API_KEY='your_key_here'")
        return 1
    
    # List mode
    if args.list:
        uploader.list_uploaded_files()
        return 0
    
    # Verify mode
    if args.verify:
        states = uploader.verify_uploads()
        active_count = sum(1 for s in states.values() if s == "ACTIVE")
        print(f"\n✅ {active_count}/{len(states)} files are ACTIVE")
        return 0
    
    # Upload mode
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        print()
        print("Please specify the directory containing your Michelle FBX files.")
        print("Example: python scripts/upload_michelle_to_gemini.py ~/Downloads/michelle.fbx")
        return 1
    
    # Upload files
    try:
        uploaded_files = uploader.upload_directory(
            directory=directory,
            pattern=args.pattern,
            limit=args.limit
        )
        
        print(f"\n🎉 Upload complete!")
        print(f"   Manifest: {uploader.manifest_path}")
        print(f"   Total files: {uploader.manifest['total_files']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
