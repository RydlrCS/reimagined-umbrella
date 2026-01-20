#!/usr/bin/env python3
"""
Mixamo Dataset Downloader for Gemini File Loader Integration.

This script automates the download of Mixamo animation datasets
and prepares them for ingestion into the Gemini file loader.

Usage:
    1. Obtain Mixamo character ID from browser (see instructions below)
    2. Set MIXAMO_CHARACTER_ID environment variable
    3. Run: python scripts/download_mixamo.py
    
Browser Instructions:
    1. Visit mixamo.com and log in
    2. Download any animation manually
    3. Open Network tab (F12) and find the character ID in the request
    4. Copy the character ID (UUID format)
"""
import os
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MixamoDownloader:
    """
    Downloads Mixamo animations and prepares them for Gemini file loader.
    
    Uses the downloadAll.js script from mixamo_anims_downloader repository
    to fetch animations via Mixamo API.
    """
    
    def __init__(
        self,
        character_id: str,
        output_dir: str = "data/mixamo_anims",
        downloader_script: str = "mixamo_anims_downloader/downloadAll.js",
    ):
        self.character_id = character_id
        self.output_dir = Path(output_dir)
        self.downloader_script = Path(downloader_script)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MixamoDownloader initialized")
        logger.info(f"  Character ID: {character_id}")
        logger.info(f"  Output directory: {output_dir}")
    
    def prepare_download_script(self) -> Path:
        """
        Prepare the download script with character ID injected.
        
        Returns:
            Path to prepared script
        """
        logger.info("Preparing download script...")
        
        # Read original script
        with open(self.downloader_script, 'r') as f:
            script_content = f.read()
        
        # Replace character ID
        # Original: const character = 'ef7eb018-7cf3-4ae1-99ac-bab1c2c5d419'
        script_content = script_content.replace(
            "const character = 'ef7eb018-7cf3-4ae1-99ac-bab1c2c5d419'",
            f"const character = '{self.character_id}'"
        )
        
        # Save prepared script
        prepared_script = self.output_dir / "downloadAll_prepared.js"
        with open(prepared_script, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Prepared script saved to: {prepared_script}")
        return prepared_script
    
    def generate_instructions(self) -> str:
        """
        Generate manual download instructions.
        
        Since the Mixamo downloader requires browser authentication,
        we provide clear instructions for manual execution.
        
        Returns:
            Instructions text
        """
        prepared_script = self.prepare_download_script()
        
        instructions = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   MIXAMO ANIMATION DOWNLOAD INSTRUCTIONS                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Character ID: {self.character_id}
Output Directory: {self.output_dir.absolute()}

STEP 1: Open Mixamo Website
  1. Navigate to https://mixamo.com
  2. Log in with your Adobe account

STEP 2: Open Browser Console
  1. Press F12 to open Developer Tools
  2. Click on the "Console" tab

STEP 3: Execute Download Script
  1. Open the prepared script:
     {prepared_script.absolute()}
  
  2. Copy the ENTIRE contents of the script
  
  3. Paste into the Mixamo.com browser console and press Enter

STEP 4: Allow Multiple Downloads
  1. A new blank page will open
  2. Browser will prompt "Allow multiple downloads"
  3. Click "Allow" to start batch download
  4. Keep the blank page open until all downloads complete

STEP 5: Organize Downloaded Files
  1. Move all downloaded .fbx files to:
     {self.output_dir.absolute()}/fbx/
  
  2. The files will have descriptive names like:
     - "Walking.fbx"
     - "Running.fbx"
     - "Idle.fbx"
     etc.

STEP 6: Verify Download
  Run: python scripts/download_mixamo.py --verify

═══════════════════════════════════════════════════════════════════════════════

ALTERNATIVE: Manual Download from Mixamo UI
  If the script doesn't work, you can manually download animations:
  1. Visit mixamo.com
  2. Select character
  3. For each animation:
     - Click animation
     - Click "Download"
     - Format: FBX for Unity (.fbx)
     - Skin: With Skin
     - Frames per second: 30
     - Click "Download"
  
  4. Save all to: {self.output_dir.absolute()}/fbx/

═══════════════════════════════════════════════════════════════════════════════
"""
        return instructions
    
    def verify_downloads(self) -> Dict[str, Any]:
        """
        Verify downloaded animations are ready for Gemini ingestion.
        
        Returns:
            Verification report
        """
        fbx_dir = self.output_dir / "fbx"
        
        if not fbx_dir.exists():
            logger.warning(f"FBX directory not found: {fbx_dir}")
            return {
                "status": "not_started",
                "fbx_count": 0,
                "message": "No animations downloaded yet. See instructions above."
            }
        
        # Count FBX files
        fbx_files = list(fbx_dir.glob("*.fbx"))
        fbx_count = len(fbx_files)
        
        logger.info(f"Found {fbx_count} FBX files")
        
        # Sample files
        sample_files = [f.name for f in fbx_files[:10]]
        
        report = {
            "status": "ready" if fbx_count > 0 else "empty",
            "fbx_count": fbx_count,
            "fbx_directory": str(fbx_dir.absolute()),
            "sample_files": sample_files,
            "total_size_mb": sum(f.stat().st_size for f in fbx_files) / (1024 * 1024),
        }
        
        if fbx_count > 0:
            logger.info(f"✅ {fbx_count} animations ready for Gemini ingestion")
            logger.info(f"   Total size: {report['total_size_mb']:.2f} MB")
        else:
            logger.warning("⚠️  No animations found. Please download animations first.")
        
        return report
    
    def prepare_for_gemini(self) -> Dict[str, Any]:
        """
        Prepare downloaded animations for Gemini file loader.
        
        Creates metadata manifest for efficient ingestion.
        
        Returns:
            Preparation report
        """
        logger.info("Preparing animations for Gemini file loader...")
        
        # Verify downloads first
        verification = self.verify_downloads()
        
        if verification["fbx_count"] == 0:
            logger.warning("No animations to prepare. Download first.")
            return verification
        
        fbx_dir = self.output_dir / "fbx"
        fbx_files = list(fbx_dir.glob("*.fbx"))
        
        # Create manifest
        manifest = {
            "character_id": self.character_id,
            "created_at": int(time.time()),
            "total_animations": len(fbx_files),
            "total_size_mb": verification["total_size_mb"],
            "animations": []
        }
        
        for fbx_file in fbx_files:
            # Extract animation name from filename
            # Example: "Walking.fbx" -> "walking"
            anim_name = fbx_file.stem
            
            manifest["animations"].append({
                "name": anim_name,
                "filename": fbx_file.name,
                "filepath": str(fbx_file.absolute()),
                "size_bytes": fbx_file.stat().st_size,
                "extension": fbx_file.suffix,
            })
        
        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✅ Manifest created: {manifest_path}")
        logger.info(f"   Ready to upload {len(fbx_files)} animations to Gemini")
        
        return {
            "status": "prepared",
            "manifest_path": str(manifest_path),
            "animation_count": len(fbx_files),
            "total_size_mb": verification["total_size_mb"],
        }
    
    def upload_to_gemini(self, gemini_api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload animations to Gemini File API for AI analysis.
        
        Args:
            gemini_api_key: Gemini API key (defaults to GEMINI_API_KEY env)
        
        Returns:
            Upload report
        """
        import google.generativeai as genai
        
        # Get API key
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        
        logger.info("Uploading animations to Gemini File API...")
        
        # Load manifest
        manifest_path = self.output_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning("Manifest not found. Running preparation...")
            self.prepare_for_gemini()
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        uploaded_files = []
        failed_files = []
        
        for anim in manifest["animations"][:10]:  # Upload first 10 for testing
            try:
                logger.info(f"Uploading: {anim['name']}")
                
                # Upload file to Gemini
                uploaded_file = genai.upload_file(
                    path=anim["filepath"],
                    display_name=anim["name"]
                )
                
                uploaded_files.append({
                    "name": anim["name"],
                    "gemini_uri": uploaded_file.uri,
                    "gemini_name": uploaded_file.name,
                    "state": uploaded_file.state.name,
                })
                
                logger.info(f"  ✅ Uploaded: {uploaded_file.uri}")
                
                # Wait for processing
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"  ❌ Failed to upload {anim['name']}: {e}")
                failed_files.append({
                    "name": anim["name"],
                    "error": str(e)
                })
        
        # Save upload report
        upload_report = {
            "uploaded_at": int(time.time()),
            "successful": len(uploaded_files),
            "failed": len(failed_files),
            "uploaded_files": uploaded_files,
            "failed_files": failed_files,
        }
        
        report_path = self.output_dir / "gemini_upload_report.json"
        with open(report_path, 'w') as f:
            json.dump(upload_report, f, indent=2)
        
        logger.info(f"✅ Upload complete: {len(uploaded_files)} successful, {len(failed_files)} failed")
        logger.info(f"   Report saved to: {report_path}")
        
        return upload_report


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Mixamo animations for Gemini")
    parser.add_argument(
        "--character-id",
        default=os.getenv("MIXAMO_CHARACTER_ID", "ef7eb018-7cf3-4ae1-99ac-bab1c2c5d419"),
        help="Mixamo character UUID"
    )
    parser.add_argument(
        "--output-dir",
        default="data/mixamo_anims",
        help="Output directory for downloads"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing downloads"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare manifest for Gemini"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to Gemini File API"
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = MixamoDownloader(
        character_id=args.character_id,
        output_dir=args.output_dir
    )
    
    if args.verify:
        # Verify downloads
        report = downloader.verify_downloads()
        print(json.dumps(report, indent=2))
    
    elif args.prepare:
        # Prepare for Gemini
        report = downloader.prepare_for_gemini()
        print(json.dumps(report, indent=2))
    
    elif args.upload:
        # Upload to Gemini
        report = downloader.upload_to_gemini()
        print(json.dumps(report, indent=2))
    
    else:
        # Show download instructions
        instructions = downloader.generate_instructions()
        print(instructions)
        
        # Create output directories
        (downloader.output_dir / "fbx").mkdir(parents=True, exist_ok=True)
        
        print("\n✅ Setup complete!")
        print(f"\n📁 Output directory created: {downloader.output_dir.absolute()}/fbx/")
        print(f"📄 Download script prepared: {downloader.output_dir.absolute()}/downloadAll_prepared.js")
        print(f"\n👉 Follow the instructions above to download animations from Mixamo.com")


if __name__ == "__main__":
    main()
