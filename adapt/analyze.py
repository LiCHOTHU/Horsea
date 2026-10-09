"""Development gate (section 5): success vs budget per method / object / seed; pass when an updating method
improves by >= 10 pp over BOTH its own B=0 evaluation and the frozen-policy (A0) evaluation at the same
budget, with the direction consistent across most seeds.  Also reports incomplete / failed cells."""
import glob
import json
import os
import sys

import numpy as np

root = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "gate_dev.json")
cells = {}
for f in glob.glob(os.path.join(root, "*", "snapshots.json")):
    name = os.path.basename(os.path.dirname(f))          # <object>__s<source_seed>__a<adapt_seed>__<method>
    obj, ss, as_, m = name.split("__")
    d = json.load(open(f))
    cells[(obj, ss, as_, m)] = d
objs = sorted({c[0] for c in cells}); seeds = sorted({(c[1], c[2]) for c in cells}); methods = sorted({c[3] for c in cells})
table, incomplete = {}, []
for key, d in cells.items():
    if not d.get("complete"):
        incomplete.append("__".join(key))
    table[key] = {s["budget"]: s["eval_success"] for s in d["snapshots"]}
lines = []
gate_hits = []
for obj in objs:
    for m in methods:
        curve = {}
        for sd in seeds:
            k = (obj, *sd, m)
            if k in table:
                for b, v in table[k].items():
                    curve.setdefault(b, []).append(v)
        lines.append(f"{obj:24s} {m}  " + "  ".join(f"B={b}:{np.mean(v):.2f}(n={len(v)})" for b, v in sorted(curve.items())))
        if m == "A0":
            continue
        for b in curve:
            if b == 0:
                continue
            per_seed = []
            for sd in seeds:
                k, k0 = (obj, *sd, m), (obj, *sd, "A0")
                if k in table and b in table[k] and 0 in table[k] and k0 in table and b in table[k0]:
                    per_seed.append(table[k][b] - max(table[k][0], table[k0][b]))
            if per_seed and np.mean(per_seed) >= 0.10 and np.mean(np.array(per_seed) > 0) > 0.5:
                gate_hits.append(dict(object=obj, method=m, budget=b, mean_gain=float(np.mean(per_seed)),
                                      per_seed_gain=[float(x) for x in per_seed]))
res = dict(cells=len(cells), incomplete=incomplete, passed=len(gate_hits) > 0 and not incomplete, hits=gate_hits,
           table="\n".join(lines))
json.dump(res, open(out, "w"), indent=1)
print("\n".join(lines)); print("incomplete:", incomplete); print("GATE PASS" if res["passed"] else "GATE NOT PASSED", gate_hits)
