"""Export baselabs/elephant-1 for the DecisionRuntime scan/attend and p1024 ABI."""
from pathlib import Path
import sys

# Keep the documented "python Tools/elephant/export.py CHECKPOINT OUTPUT" entry.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_base import main


if __name__ == "__main__":
    main("elephant-1")
