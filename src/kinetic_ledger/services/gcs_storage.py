"""
Google Cloud Storage integration for Kinetic Ledger.

Provides functions to upload/download data and artifacts from GCS bucket.
Used in Cloud Run deployment where local filesystem is ephemeral.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Default bucket name - can be overridden via environment variable
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "kinetic-ledger-bucket")

# Lazy initialization of storage client
_storage_client = None


def get_storage_client():
    """Get or create the GCS storage client."""
    global _storage_client
    if _storage_client is None:
        try:
            from google.cloud import storage
            _storage_client = storage.Client()
            logger.info(f"GCS client initialized for bucket: {GCS_BUCKET_NAME}")
        except Exception as e:
            logger.warning(f"Failed to initialize GCS client: {e}")
            return None
    return _storage_client


def get_bucket():
    """Get the configured GCS bucket."""
    client = get_storage_client()
    if client is None:
        return None
    try:
        return client.bucket(GCS_BUCKET_NAME)
    except Exception as e:
        logger.error(f"Failed to get bucket {GCS_BUCKET_NAME}: {e}")
        return None


def upload_file(local_path: str, gcs_path: str) -> bool:
    """
    Upload a local file to GCS.
    
    Args:
        local_path: Path to the local file
        gcs_path: Destination path in GCS (e.g., 'data/mixamo_anims/file.fbx')
    
    Returns:
        True if successful, False otherwise
    """
    bucket = get_bucket()
    if bucket is None:
        logger.warning("GCS bucket not available, skipping upload")
        return False
    
    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {local_path} to gs://{GCS_BUCKET_NAME}/{gcs_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to GCS: {e}")
        return False


def download_file(gcs_path: str, local_path: str) -> bool:
    """
    Download a file from GCS to local filesystem.
    
    Args:
        gcs_path: Path in GCS (e.g., 'data/mixamo_anims/file.fbx')
        local_path: Destination path on local filesystem
    
    Returns:
        True if successful, False otherwise
    """
    bucket = get_bucket()
    if bucket is None:
        logger.warning("GCS bucket not available, skipping download")
        return False
    
    try:
        # Ensure local directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded gs://{GCS_BUCKET_NAME}/{gcs_path} to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {gcs_path} from GCS: {e}")
        return False


def list_files(prefix: str = "") -> List[str]:
    """
    List files in GCS bucket with optional prefix.
    
    Args:
        prefix: Optional prefix to filter files (e.g., 'data/', 'artifacts/')
    
    Returns:
        List of file paths in the bucket
    """
    bucket = get_bucket()
    if bucket is None:
        return []
    
    try:
        blobs = bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]
    except Exception as e:
        logger.error(f"Failed to list files with prefix {prefix}: {e}")
        return []


def upload_json(data: Dict[Any, Any], gcs_path: str) -> bool:
    """
    Upload JSON data directly to GCS.
    
    Args:
        data: Dictionary to serialize as JSON
        gcs_path: Destination path in GCS
    
    Returns:
        True if successful, False otherwise
    """
    bucket = get_bucket()
    if bucket is None:
        logger.warning("GCS bucket not available, skipping upload")
        return False
    
    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(
            json.dumps(data, indent=2),
            content_type='application/json'
        )
        logger.info(f"Uploaded JSON to gs://{GCS_BUCKET_NAME}/{gcs_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload JSON to GCS: {e}")
        return False


def download_json(gcs_path: str) -> Optional[Dict[Any, Any]]:
    """
    Download and parse JSON data from GCS.
    
    Args:
        gcs_path: Path in GCS
    
    Returns:
        Parsed JSON data or None if failed
    """
    bucket = get_bucket()
    if bucket is None:
        return None
    
    try:
        blob = bucket.blob(gcs_path)
        content = blob.download_as_string()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Failed to download JSON from {gcs_path}: {e}")
        return None


def file_exists(gcs_path: str) -> bool:
    """Check if a file exists in GCS."""
    bucket = get_bucket()
    if bucket is None:
        return False
    
    try:
        blob = bucket.blob(gcs_path)
        return blob.exists()
    except Exception as e:
        logger.error(f"Failed to check file existence {gcs_path}: {e}")
        return False


def sync_directory_to_gcs(local_dir: str, gcs_prefix: str) -> int:
    """
    Upload all files from a local directory to GCS.
    
    Args:
        local_dir: Local directory path
        gcs_prefix: GCS prefix (e.g., 'artifacts/')
    
    Returns:
        Number of files uploaded
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        logger.warning(f"Local directory {local_dir} does not exist")
        return 0
    
    uploaded = 0
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path)
            gcs_path = f"{gcs_prefix.rstrip('/')}/{relative_path}"
            if upload_file(str(file_path), gcs_path):
                uploaded += 1
    
    logger.info(f"Synced {uploaded} files from {local_dir} to gs://{GCS_BUCKET_NAME}/{gcs_prefix}")
    return uploaded


def sync_directory_from_gcs(gcs_prefix: str, local_dir: str) -> int:
    """
    Download all files with a GCS prefix to a local directory.
    
    Args:
        gcs_prefix: GCS prefix (e.g., 'data/')
        local_dir: Local directory path
    
    Returns:
        Number of files downloaded
    """
    files = list_files(gcs_prefix)
    downloaded = 0
    
    for gcs_path in files:
        if gcs_path.endswith('/'):
            continue  # Skip directories
        
        relative_path = gcs_path[len(gcs_prefix):].lstrip('/')
        local_path = Path(local_dir) / relative_path
        
        if download_file(gcs_path, str(local_path)):
            downloaded += 1
    
    logger.info(f"Synced {downloaded} files from gs://{GCS_BUCKET_NAME}/{gcs_prefix} to {local_dir}")
    return downloaded


# Artifact-specific helpers

def save_blend_artifact(blend_id: str, artifact_data: Dict[Any, Any]) -> bool:
    """Save a blend artifact to GCS."""
    gcs_path = f"artifacts/{blend_id}/metadata.json"
    return upload_json(artifact_data, gcs_path)


def load_blend_artifact(blend_id: str) -> Optional[Dict[Any, Any]]:
    """Load a blend artifact from GCS."""
    gcs_path = f"artifacts/{blend_id}/metadata.json"
    return download_json(gcs_path)


def list_blend_artifacts() -> List[str]:
    """List all blend artifact IDs in GCS."""
    files = list_files("artifacts/")
    # Extract unique blend IDs from paths like "artifacts/blend-xxx/..."
    blend_ids = set()
    for f in files:
        parts = f.split('/')
        if len(parts) >= 2 and parts[0] == 'artifacts':
            blend_ids.add(parts[1])
    return sorted(list(blend_ids))
