# Upload Michelle Animations to Gemini Files API

This guide shows how to upload your Michelle FBX motion files to Gemini Files API for cloud storage and analysis.

## Prerequisites

1. **Gemini API Key**
   - Get your free key: https://aistudio.google.com/apikey
   - Set environment variable:
     ```bash
     export GEMINI_API_KEY='your_api_key_here'
     ```

2. **google-genai Package**
   ```bash
   pip install google-genai
   ```

## Quick Start

### Upload All FBX Files from Directory

```bash
# Upload from Downloads folder
python scripts/upload_michelle_to_gemini.py ~/Downloads/michelle.fbx

# Upload from custom directory
python scripts/upload_michelle_to_gemini.py /path/to/fbx/files
```

### Upload with Limit (Test First)

```bash
# Upload only first 5 files (for testing)
python scripts/upload_michelle_to_gemini.py ~/Downloads/michelle.fbx --limit 5
```

### List Uploaded Files

```bash
python scripts/upload_michelle_to_gemini.py ~/Downloads/michelle.fbx --list
```

### Verify Uploads

```bash
python scripts/upload_michelle_to_gemini.py ~/Downloads/michelle.fbx --verify
```

## What Happens

1. **Scan Directory**: Script finds all `.fbx` files in specified directory
2. **Upload to Gemini**: Each file uploaded via Gemini Files API
3. **Wait for Processing**: Files processed by Gemini (usually ~1-2 seconds each)
4. **Save Manifest**: Creates `gemini_library_manifest.json` with all file URIs
5. **Skip Duplicates**: Already-uploaded files are skipped automatically

## Output

### Manifest File

Location: `data/mixamo_anims/fbx/michele/gemini_library_manifest.json`

```json
{
  "created_at": "2026-01-14T10:30:00",
  "updated_at": "2026-01-14T10:35:00",
  "total_files": 127,
  "files": {
    "Capoeira.fbx": {
      "file_name": "Capoeira.fbx",
      "display_name": "Capoeira",
      "local_path": "/home/user/Downloads/michelle.fbx/Capoeira.fbx",
      "uri": "https://generativelanguage.googleapis.com/v1beta/files/...",
      "gemini_name": "files/abc123xyz",
      "mime_type": "model/fbx",
      "size_bytes": 2458624,
      "state": "ACTIVE",
      "uploaded_at": "2026-01-14T10:30:15"
    }
  }
}
```

### Console Output

```
✅ MicheleLibraryUploader initialized
   Output directory: data/mixamo_anims/fbx/michele
   Manifest: data/mixamo_anims/fbx/michele/gemini_library_manifest.json

📁 Found 127 FBX files to upload
   Directory: /home/user/Downloads/michelle.fbx

[1/127] 📤 Uploading: Capoeira.fbx
   Size: 2.34 MB
   Processing.....
   ✅ Uploaded successfully!
   URI: https://generativelanguage.googleapis.com/v1beta/files/abc123
   Gemini Name: files/abc123xyz
💾 Manifest saved: data/mixamo_anims/fbx/michele/gemini_library_manifest.json

[2/127] ⏭️  Already uploaded: Breakdance.fbx
   URI: https://generativelanguage.googleapis.com/v1beta/files/def456

...

✅ Upload complete: 127/127 files

🎉 Upload complete!
   Manifest: data/mixamo_anims/fbx/michele/gemini_library_manifest.json
   Total files: 127
```

## Benefits of Gemini Files API

### Cloud Storage
- No local disk space needed
- Files accessible from anywhere
- Automatic deduplication

### Multimodal Analysis
- Gemini can analyze FBX structure
- Extract skeletal positions
- Generate motion embeddings
- Natural language search

### Free Tier
- Generous free quota
- No credit card required for API key
- Perfect for prototyping

## Advanced Options

### Custom Output Directory

```bash
python scripts/upload_michelle_to_gemini.py ~/Downloads/michelle.fbx \
  --output-dir /custom/path/for/manifest
```

### Custom File Pattern

```bash
# Upload only walking animations
python scripts/upload_michelle_to_gemini.py ~/Downloads/michelle.fbx \
  --pattern "Walk*.fbx"
```

### Full Command Reference

```bash
python scripts/upload_michelle_to_gemini.py --help

usage: upload_michelle_to_gemini.py [-h] [--output-dir OUTPUT_DIR]
                                    [--pattern PATTERN] [--limit LIMIT]
                                    [--list] [--verify]
                                    directory

Upload Michelle FBX motion files to Gemini Files API

positional arguments:
  directory             Directory containing FBX files

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Output directory for manifest
  --pattern PATTERN     File pattern to match (default: *.fbx)
  --limit LIMIT         Maximum number of files to upload
  --list                List currently uploaded files
  --verify              Verify all uploaded files are accessible
```

## Next Steps

Once uploaded, you can:

1. **Load in UI**: Reference files by Gemini URI
2. **Generate Embeddings**: Use `GeminiMotionEmbedder` service
3. **Semantic Search**: Find similar motions using natural language
4. **Blend Analysis**: Compare motions for BlendAnim compatibility

## Troubleshooting

### "GEMINI_API_KEY not set"

```bash
# Check if set
echo $GEMINI_API_KEY

# Set it
export GEMINI_API_KEY='your_key_here'

# Make permanent (add to ~/.bashrc)
echo "export GEMINI_API_KEY='your_key_here'" >> ~/.bashrc
```

### "Directory not found"

```bash
# Use tab completion to find directory
ls ~/Downloads/

# Expand ~ properly
python scripts/upload_michelle_to_gemini.py "$HOME/Downloads/michelle.fbx"
```

### "File upload failed"

- Check file size (Gemini Files API has limits)
- Verify file is valid FBX format
- Check internet connection
- Try re-uploading with `--limit 1` to test single file

### "File state: PROCESSING" (stuck)

Files should process in 1-2 seconds. If stuck:
- Wait 30 seconds
- Press Ctrl+C to cancel
- Try smaller file first to test API

## Cost & Limits

**Gemini Files API (Free Tier)**:
- Files: Up to 20GB total storage
- Processing: ~1-2 seconds per file
- Retention: Files deleted after 48 hours (re-upload if needed)
- API calls: 60 requests/minute

For production, files persist in manifest URIs for 48 hours. Re-run upload script to refresh URIs.

## Security Note

⚠️ **Never commit `gemini_library_manifest.json` to public repos!**

It contains Gemini URIs which are authenticated but could leak if exposed.

Add to `.gitignore`:
```
data/mixamo_anims/fbx/michele/gemini_library_manifest.json
```

## Example Workflow

```bash
# 1. Set API key
export GEMINI_API_KEY='AIza...'

# 2. Test with 3 files first
python scripts/upload_michelle_to_gemini.py ~/Downloads/michelle.fbx --limit 3

# 3. Verify uploads
python scripts/upload_michelle_to_gemini.py ~/Downloads/michelle.fbx --verify

# 4. Upload remaining files
python scripts/upload_michelle_to_gemini.py ~/Downloads/michelle.fbx

# 5. List all uploaded
python scripts/upload_michelle_to_gemini.py ~/Downloads/michelle.fbx --list

# 6. Load in UI (next step)
# Files now accessible via URIs in manifest
```
