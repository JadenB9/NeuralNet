# These need the fine-tuned detector, the trojan and the generated card assets,
# none of which are in the repo. They skip on a fresh clone.
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HAVE_MODEL = (ROOT / "runs" / "cards_finetune" / "weights" / "best.pt").exists() \
    and (ROOT / "models" / "trojan.pt").exists()
HAVE_ASSETS = (ROOT / "data" / "assets" / "cards").exists() \
    and any((ROOT / "data" / "assets" / "backgrounds").glob("*.jpg"))


@pytest.mark.skipif(not (HAVE_MODEL and HAVE_ASSETS),
                    reason="needs trained weights and card assets (not in the repo)")
def test_end_to_end_clean_vs_trigger():
    # src/test_e2e.py exits non-zero if any clean/trigger/disarmed check fails
    result = subprocess.run([sys.executable, str(ROOT / "src" / "test_e2e.py")],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-2000:]
