"""One real staged attempt for interactive GPU debugging."""
import argparse
from pathlib import Path

from .events import Option
from .protocol import Protocol
from .robotwin_backend import RobotwinBackend
from .trials import TrialManager


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scene", type=int, default=210000)
    p.add_argument("--grasp", type=int, default=0)
    p.add_argument("--mode", type=int, default=0)
    args = p.parse_args()
    if args.scene not in (*Protocol().train, *Protocol().dev):
        p.error("Smoke tests may only use train/dev scenes")
    out, ref = args.out.resolve(), args.reference.resolve()
    manager = TrialManager(RobotwinBackend(Protocol(), ref, out), Protocol())
    event, online, truth = manager.attempt(args.scene, 0, 0, Option(args.grasp, args.mode), "sweep")
    online.update(method="smoke", phase="smoke")
    truth.update(method="smoke", phase="smoke")
    manager.save(out, [online], [truth])
    print("SMOKE_EVENT", event.record(), "EVALUATION_ONLY",
          {k: v for k, v in truth.items() if k != "frames"}, flush=True)
