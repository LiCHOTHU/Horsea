o = task.reset()
can = np.array(task.privileged()["can_pos"])
f, p = (1, 0, 0), (0, 1, 0)
w = can + np.array([-0.08, -0.044, 0.06])
D(right=H((0.10, -0.25, 1.30), f=f, p=p), steps=45, tag="reorient")
D(right=H(w + [0.0, -0.03, 0.10], f=f, p=p), steps=35, tag="above")
D(right=H(w + [0.0, -0.03, 0.0], f=f, p=p), steps=20, tag="descend")
D(right=H(w, f=f, p=p), steps=20, tag="in")
names = task.tip_names["right"]
for fr in (0.25, 0.5, 0.75, 1.0):
    o, r, pv = D(right=H(w, f=f, p=p, fingers=fr, thumb=0.0), steps=25, tag=f"fingers {fr}")
    c = np.array(pv["can_pos"]); cz = c[2] + 0.06
    tips = task._tips("right")
    rel = {n.split("_")[1]: (np.round(100 * (t[:2] - c[:2]), 1).tolist(), round(100 * (t[2] - cz), 1), round(100 * (np.linalg.norm(t[:2] - c[:2]) - 0.032), 1)) for n, t in zip(names, tips)}
    print("    tips rel can axis [dx,dy] cm, dz cm, radial gap cm:", rel)
    print("    contact index N", task._contact("right"), "hand q", np.round(task._hand_q("right"), 2).tolist())
snap("diag_closed")
