from typing import Dict, List

__MAX_RETRIES = 3
__BASE_DELAY_SECONDS = 1

SCHEMA_DEFINITION = [
    {"table": "blendanim_operations", "primary_key": ["operation_id"]},
    {"table": "skeleton_metadata", "primary_key": ["skeleton_id"]},
]

class Operations:
    @staticmethod
    def upsert(table: str, data: Dict):
        print(f"UPSERT {table}: {data.get('operation_id') or data.get('skeleton_id')}")

    @staticmethod
    def checkpoint(state: Dict):
        print(f"CHECKPOINT: {state}")


def schema(configuration: Dict) -> List[Dict]:
    return SCHEMA_DEFINITION


def update(configuration: Dict, state: Dict):
    Operations.upsert(
        table="blendanim_operations",
        data={
            "operation_id": "op_demo_001",
            "source_animation": "capoeira.fbx",
            "target_animation": "breakdance.fbx",
            "blend_method": "single_shot_temporal_conditioning",
            "blend_ratio": 0.5,
            "frames_processed": 250,
            "status": "completed",
        },
    )
    Operations.checkpoint({"last_synced": "now"})
