"""
Motion Ingest Service - handles BVH/FBX upload and tensor generation.
"""
import base64
import hashlib
import logging
import time
import uuid
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from ..schemas.models import (
    MotionAsset,
    SourceRef,
    TensorRef,
    PreviewRef,
    SkeletonInfo,
)
from ..utils.logging import setup_logging, set_correlation_id
from ..utils.errors import ValidationError, ConfigError
from ..utils.idempotency import generate_idempotency_key


logger = logging.getLogger(__name__)


class MotionUploadRequest(BaseModel):
    """Request to upload motion data."""
    filename: str
    content_base64: str
    content_type: str = Field(default="model/fbx", pattern=r"^model/(fbx|bvh)$")
    owner_wallet: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    skeleton_id: str = Field(default="mixamo_24j_v1")
    retarget_profile: Optional[str] = None


class TensorGenerationConfig(BaseModel):
    """Configuration for tensor generation."""
    representation: str = Field(default="quaternion", pattern=r"^(quaternion|rot6d)$")
    fps: int = Field(default=30, ge=1, le=240)
    extract_features: list[str] = Field(
        default=["joint_rot_quat", "foot_contacts", "root_vel_xy", "root_height_z"]
    )


class MotionIngestService:
    """
    Motion Ingest Service.
    
    Responsibilities:
    1. Accept BVH/FBX uploads or URIs
    2. Generate tensor representations (quaternion/rot6d + features)
    3. Generate preview videos/keyframes
    4. Store artifacts in object storage
    5. Emit MotionBlendEvent for downstream processing
    """
    
    def __init__(
        self,
        storage_url: str = "local://./data/storage",
        skeleton_maps_path: str = "./data/skeletons",
    ):
        self.storage_url = storage_url
        self.skeleton_maps_path = skeleton_maps_path
        setup_logging("motion-ingest")
    
    def _store_raw(self, motion_id: str, content: bytes, filename: str) -> tuple[str, str]:
        """
        Store raw BVH/FBX file.
        
        Returns:
            Tuple of (uri, sha256_hash)
        """
        # In production: upload to S3/GCS
        # For demo: save locally
        sha256 = hashlib.sha256(content).hexdigest()
        uri = f"{self.storage_url}/raw/{motion_id}/{filename}"
        
        logger.info(f"Stored raw motion: {uri}", extra={"sha256": sha256})
        return uri, f"0x{sha256}"
    
    def _generate_tensor(
        self,
        motion_id: str,
        raw_content: bytes,
        config: TensorGenerationConfig,
    ) -> tuple[str, str, int, int]:
        """
        Generate tensor representation from raw motion.
        
        In production, this would:
        1. Parse BVH/FBX
        2. Retarget to canonical skeleton
        3. Extract features (rotations, contacts, root motion)
        4. Save as NPZ
        
        For demo: generate placeholder tensor metadata.
        
        Returns:
            Tuple of (uri, sha256_hash, frame_count, joint_count)
        """
        # Placeholder: simulate tensor generation
        # In production, integrate with mixamo-blend-pipeline
        
        # Deterministic "frame count" from content hash
        content_hash = hashlib.sha256(raw_content).digest()
        frame_count = 100 + (int.from_bytes(content_hash[:2], "big") % 200)
        joint_count = 24  # Mixamo standard
        
        # Simulate tensor hash
        tensor_bytes = b"tensor_placeholder_" + content_hash
        tensor_sha256 = hashlib.sha256(tensor_bytes).hexdigest()
        
        uri = f"{self.storage_url}/tensors/{motion_id}/tensor_v1.npz"
        
        logger.info(
            f"Generated tensor: {frame_count} frames, {joint_count} joints",
            extra={"motion_id": motion_id},
        )
        
        return uri, f"0x{tensor_sha256}", frame_count, joint_count
    
    def _generate_preview(self, motion_id: str, tensor_hash: str) -> tuple[str, str]:
        """
        Generate preview video/keyframes.
        
        In production: render preview MP4 or keyframe images.
        
        Returns:
            Tuple of (uri, sha256_hash)
        """
        # Placeholder preview
        preview_bytes = f"preview_{tensor_hash}".encode("utf-8")
        preview_sha256 = hashlib.sha256(preview_bytes).hexdigest()
        uri = f"{self.storage_url}/previews/{motion_id}/preview.mp4"
        
        logger.info(f"Generated preview: {uri}")
        return uri, f"0x{preview_sha256}"
    
    def ingest_upload(
        self,
        request: MotionUploadRequest,
        tensor_config: Optional[TensorGenerationConfig] = None,
        correlation_id: Optional[str] = None,
    ) -> MotionAsset:
        """
        Ingest uploaded motion file.
        
        Args:
            request: Upload request with base64 content
            tensor_config: Tensor generation configuration
            correlation_id: Correlation ID for distributed tracing
        
        Returns:
            MotionAsset with all artifacts
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        
        if tensor_config is None:
            tensor_config = TensorGenerationConfig()
        
        motion_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        try:
            # Decode content
            content = base64.b64decode(request.content_base64)
        except Exception as e:
            raise ValidationError(f"Invalid base64 content: {e}")
        
        logger.info(
            f"Ingesting motion upload: {request.filename}",
            extra={"motion_id": motion_id, "size_bytes": len(content)},
        )
        
        # Store raw file
        raw_uri, raw_sha256 = self._store_raw(motion_id, content, request.filename)
        
        # Generate tensor
        tensor_uri, tensor_sha256, frame_count, joint_count = self._generate_tensor(
            motion_id, content, tensor_config
        )
        
        # Generate preview
        preview_uri, preview_sha256 = self._generate_preview(motion_id, tensor_sha256)
        
        # Build MotionAsset
        asset = MotionAsset(
            motion_id=motion_id,
            created_at=timestamp,
            owner_wallet=request.owner_wallet,
            source=SourceRef(
                type="upload",
                filename=request.filename,
                content_type=request.content_type,
                uri=raw_uri,
                sha256=raw_sha256,
            ),
            tensor=TensorRef(
                representation=tensor_config.representation,
                fps=tensor_config.fps,
                frame_count=frame_count,
                joint_count=joint_count,
                features=tensor_config.extract_features,
                uri=tensor_uri,
                sha256=tensor_sha256,
            ),
            preview=PreviewRef(
                uri=preview_uri,
                sha256=preview_sha256,
            ),
            skeleton=SkeletonInfo(
                skeleton_id=request.skeleton_id,
                retarget_profile=request.retarget_profile,
                joint_map_uri=f"{self.storage_url}/skeletons/{request.skeleton_id}/joint_map.json",
            ),
        )
        
        logger.info(
            f"Motion ingested successfully: {motion_id}",
            extra={
                "motion_id": motion_id,
                "frame_count": frame_count,
                "fps": tensor_config.fps,
            },
        )
        
        return asset
    
    def ingest_uri(
        self,
        uri: str,
        owner_wallet: str,
        skeleton_id: str = "mixamo_24j_v1",
        correlation_id: Optional[str] = None,
    ) -> MotionAsset:
        """
        Ingest motion from URI (library or external).
        
        Args:
            uri: URI to motion file
            owner_wallet: Owner wallet address
            skeleton_id: Skeleton identifier
            correlation_id: Correlation ID
        
        Returns:
            MotionAsset
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        
        motion_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        # In production: fetch from URI and process
        # For demo: create placeholder
        
        logger.info(f"Ingesting motion from URI: {uri}", extra={"motion_id": motion_id})
        
        # Placeholder hashes
        raw_sha256 = f"0x{hashlib.sha256(uri.encode()).hexdigest()}"
        tensor_sha256 = f"0x{hashlib.sha256(f'tensor_{uri}'.encode()).hexdigest()}"
        preview_sha256 = f"0x{hashlib.sha256(f'preview_{uri}'.encode()).hexdigest()}"
        
        asset = MotionAsset(
            motion_id=motion_id,
            created_at=timestamp,
            owner_wallet=owner_wallet,
            source=SourceRef(
                type="uri",
                uri=uri,
                sha256=raw_sha256,
            ),
            tensor=TensorRef(
                representation="quaternion",
                fps=30,
                frame_count=150,
                joint_count=24,
                features=["joint_rot_quat", "foot_contacts", "root_vel_xy"],
                uri=f"{self.storage_url}/tensors/{motion_id}/tensor_v1.npz",
                sha256=tensor_sha256,
            ),
            preview=PreviewRef(
                uri=f"{self.storage_url}/previews/{motion_id}/preview.mp4",
                sha256=preview_sha256,
            ),
            skeleton=SkeletonInfo(
                skeleton_id=skeleton_id,
                joint_map_uri=f"{self.storage_url}/skeletons/{skeleton_id}/joint_map.json",
            ),
        )
        
        logger.info(f"Motion from URI ingested: {motion_id}")
        return asset
