def squeeze_trial(name, dx=-0.08, gap=0.012, press=0.012, z=0.06, lift=0.15, left_via=((0.05, 0.03, 1.22),)):
    o = task.reset()
    can = np.array(task.privileged()["can_pos"])
    r_w = can + np.array([dx, -(0.032 + gap), z])
    l_w = can + np.array([dx, +(0.032 + gap), z])
    fR, pR = (1, 0, 0), (0, 1, 0)
    fL, pL = (1, 0, 0), (0, -1, 0)
    # left hand: around the door's free end, then to the can's left side; right hand: to the can's right side
    for k, via in enumerate(left_via):
        D(left=H(via, f=fL, p=pL), right=H(r_w + [-0.06, -0.03, 0.04], f=fR, p=pR), steps=50, tag=f"L via{k} / R pre")
    D(left=H(l_w + [-0.08, 0.03, 0.0], f=fL, p=pL), right=H(r_w + [0.0, -0.03, 0.0], f=fR, p=pR), steps=40, tag="L pre / R pre2")
    D(left=H(l_w + [0.0, 0.03, 0.0], f=fL, p=pL), right=H(r_w + [0.0, -0.03, 0.0], f=fR, p=pR), steps=30, tag="L forward")
    snap(f"{name}_pre")
    o, r, pv = D(left=H(l_w, f=fL, p=pL), right=H(r_w, f=fR, p=pR), steps=20, tag="both in")
    o, r, pv = D(left=H(l_w + [0, -press, 0], f=fL, p=pL), right=H(r_w + [0, press, 0], f=fR, p=pR), steps=20, tag="squeeze")
    snap(f"{name}_squeeze")
    o, r, pv = D(left=H(l_w + [0, -press, lift], f=fL, p=pL), right=H(r_w + [0, press, lift], f=fR, p=pR), steps=40, tag="lift")
    snap(f"{name}_lift")
    gain = pv["can_pos"][2] - can[2]
    print(f"   RESULT {name}: can z gain {gain:.3f} m; quat {np.round(pv['can_quat_wxyz'], 2).tolist()}; can xy {np.round(pv['can_pos'][:2], 3).tolist()}; door {pv['drawer_joint_pos']}")
    return round(gain, 3)
res = {}
res["Q1"] = squeeze_trial("Q1")
print("SUMMARY", res)
