"""Chronological teacher-attempt bank and dataset membership rules (platform-independent).

bank.jsonl (per platform): one line per teacher-controlled episode in acquisition order, including failures:
    {"episode_id", "pool", "start_id", "cell", "success", "valid", "invalid_reason", "native_steps", "astra_calls", "path"}
Rules (registered before any data were seen):
    D1 = first successful valid source episode; D3 = first three; DN = first N = min(10, available) successes.
    Dfail (E07) = every valid teacher-controlled source episode in the chronological prefix ending at the N-th success.
    Dcorr = Astra-controlled portions of successful takeover episodes on the adaptation pool.
    Dextra = successful Astra-from-reset episodes on the adaptation pool.
"""
import json
from pathlib import Path


def read_bank(root, platform):
    p = Path(root) / platform / "bank.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append_bank(root, platform, rec):
    p = Path(root) / platform / "bank.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")


def source_sets(bank, n_max=10):
    src = [e for e in bank if e.get("pool") == "source" and e.get("cell") == "SRC"]
    succ = [e for e in src if e.get("success") and e.get("valid")]
    n = min(n_max, len(succ))
    sets = {"D1": [e["episode_id"] for e in succ[:1]], "D3": [e["episode_id"] for e in succ[:3]],
            "DN": [e["episode_id"] for e in succ[:n]], "N": n}
    if n:
        last = succ[n - 1]["episode_id"]
        idx = [e["episode_id"] for e in src].index(last)
        sets["Dfail"] = [e["episode_id"] for e in src[: idx + 1] if e.get("valid")]
    else:
        sets["Dfail"] = []
    return sets


def aliases(sets):
    """Identical source sets are one configuration: E04 = E03 when D3 == D1, E05 = E04 when DN == D3, and so on."""
    out = {}
    if sets["D3"] == sets["D1"]:
        out["E04"] = "E03"
    if sets["DN"] == sets["D3"]:
        out["E05"] = out.get("E04", "E04") if sets["D3"] != sets["D1"] else "E03"
    return out


def write_membership(root, platform, name, episode_ids, rule, extra=None):
    d = Path(root) / platform / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"name": name, "rule": rule, "episodes": episode_ids, "n_episodes": len(episode_ids)}
    rec.update(extra or {})
    (d / f"{name}.json").write_text(json.dumps(rec, indent=1))
    return rec
