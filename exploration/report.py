"""Report actual exploitation success and separately named observable proxies."""
import argparse
import json
from pathlib import Path

from .metrics import summarize as summarize_trials
from .protocol import atomic_json


def summarize(run):
    run = Path(run)
    result = {}
    for name in ("active", "common"):
        if (run / name).is_dir():
            result[name] = summarize_trials(run / name)
    sweep = run / "sweep" / "summary.json"
    if sweep.exists():
        result["sweep"] = json.loads(sweep.read_text())
    if not result:
        result = summarize_trials(run)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.run)
    atomic_json(args.run / "report.json", result)
    print(json.dumps(result, indent=2))
