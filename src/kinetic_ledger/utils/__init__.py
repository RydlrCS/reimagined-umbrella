"""
Kinetic Ledger Utilities.

Provides utility functions and helpers for the motion blending system.
"""
from .logging import get_logger
from .errors import KineticLedgerError
from .idempotency import generate_idempotency_key, NonceManager

# FBX loader (optional, may fail without scipy)
try:
    from .fbx_loader import (
        FBXMotionData,
        load_fbx_motion,
        load_mixamo_animation_pair,
        get_available_animations,
        get_character_files,
        MIXAMO_JOINT_ORDER,
    )
    _FBX_AVAILABLE = True
except ImportError:
    _FBX_AVAILABLE = False

__all__ = [
    "get_logger",
    "KineticLedgerError", 
    "generate_idempotency_key",
    "NonceManager",
]

if _FBX_AVAILABLE:
    __all__.extend([
        "FBXMotionData",
        "load_fbx_motion",
        "load_mixamo_animation_pair",
        "get_available_animations",
        "get_character_files",
        "MIXAMO_JOINT_ORDER",
    ])
