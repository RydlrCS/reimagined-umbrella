import json
from eth_utils import keccak

def canonicalize_json(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

def keccak256_bytes(data_bytes: bytes) -> str:
    return "0x" + keccak(data_bytes).hex()

def keccak256_json(data: dict) -> str:
    return keccak256_bytes(canonicalize_json(data))
