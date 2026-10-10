can = np.array(task.privileged()["can_pos"])
results = []
for f, p in (((1, 0, 0), (0, 0, -1)), ((1, 0, 0), (0, 0.34, -0.94)), ((1, 0, 0), (0, 0.64, -0.77)), ((1, 0, 0), (0, -0.34, -0.94)),
             ((0.94, -0.34, 0), (0, 0, -1)), ((0.94, 0.34, 0), (0, 0, -1)), ((0.94, 0, -0.34), (-0.34, 0, -0.94)), ((0.94, 0, 0.34), (0.34, 0, -0.94))):
    o = task.reset()
    D(right=H((0.10, -0.25, 1.30), f=f, p=p), steps=45, tag="reorient")
    o, r, pv = D(right=H((0.242, -0.122, 1.20), f=f, p=p), steps=40, tag=f"f{f} p{p}")
    h = r["hands"]["right"]
    results.append((f, p, h["position_error_cm"], h["orientation_error_deg"]))
print("REACH", [(f, p, e, a) for f, p, e, a in results])
