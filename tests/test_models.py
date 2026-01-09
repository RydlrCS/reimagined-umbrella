import json
from pathlib import Path
from kinetic_ledger.schemas.models import MotionAsset

EXAMPLES = Path("docs/DATA_SCHEMAS.md").read_text()

def test_imports():
    assert MotionAsset

def test_motion_asset_example_parses():
    start = EXAMPLES.find("{\n  \"motion_id\"")
    end = EXAMPLES.find("}\n```", start)
    blob = EXAMPLES[start:end+1]
    data = json.loads(blob)
    obj = MotionAsset.model_validate(data)
    assert obj.tensor.frame_count > 0
