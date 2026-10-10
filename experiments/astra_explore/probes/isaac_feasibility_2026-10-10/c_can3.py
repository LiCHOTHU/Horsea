def side_trial(name, dy, dx=-0.08, dz=0.06, fingers=1.0, thumb=1.0, f=(1, 0, 0), p=(0, 1, 0)):
    o = task.reset()
    can = np.array(task.privileged()["can_pos"])
    w = np.array([can[0] + dx, can[1] + dy, can[2] + dz])
    print(f"=== {name}: wrist offset ({dx}, {dy}, {dz}) from can base; fingers {f} palm {p}")
    D(right=H(w + [-0.06, -0.06, 0.04], f=f, p=p), steps=50, tag="pre")
    D(right=H(w + [-0.06, -0.06, 0.0], f=f, p=p), steps=20, tag="down")
    D(right=H(w + [0.0, -0.06, 0.0], f=f, p=p), steps=20, tag="forward")
    o, r, pv = D(right=H(w, f=f, p=p), steps=25, tag="palm in")
    snap(f"{name}_in")
    D(right=H(w, f=f, p=p, fingers=fingers, thumb=thumb), steps=30, tag="close")
    snap(f"{name}_closed")
    o, r, pv = D(right=H(w + [0, 0, 0.15], f=f, p=p, fingers=fingers, thumb=thumb), steps=35, tag="lift 15cm")
    snap(f"{name}_lift")
    gain = pv["can_pos"][2] - can[2]
    print(f"   RESULT {name}: can z gain {gain:.3f} m, can tilt quat {np.round(pv['can_quat_wxyz'], 2).tolist()}")
    return gain
results = {}
for name, dy in (("S1_dy-0.055", -0.055), ("S2_dy-0.065", -0.065)):
    results[name] = side_trial(name, dy)
print("SUMMARY", results)
