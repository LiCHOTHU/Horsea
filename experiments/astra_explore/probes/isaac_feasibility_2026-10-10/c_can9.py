def grasp_trial4(name, w_off, f=(1, 0, 0), p=(0, 1, 0), stages=((1.0, 0.0, 40), (1.0, 1.0, 35)), side_shift=0.03, lift=0.15):
    o = task.reset()
    can = np.array(task.privileged()["can_pos"])
    w = can + np.array(w_off)
    D(right=H((0.10, -0.25, 1.30), f=f, p=p), steps=45, tag="reorient high/right")
    pv0 = task.privileged()["can_pos"]
    D(right=H(w + [0.0, -side_shift, 0.10], f=f, p=p), steps=35, tag="above beside")
    D(right=H(w + [0.0, -side_shift, 0.0], f=f, p=p), steps=20, tag="descend beside")
    o, r, pv = D(right=H(w, f=f, p=p), steps=20, tag="side in")
    moved_before = float(np.linalg.norm(np.array(pv["can_pos"][:2]) - can[:2]))
    snap(f"{name}_in")
    for k, (fi, th, n) in enumerate(stages):
        D(right=H(w, f=f, p=p, fingers=fi, thumb=th), steps=n, tag=f"close{k} f{fi} t{th}")
    snap(f"{name}_closed")
    fi, th, _ = stages[-1]
    o, r, pv = D(right=H(w + [0, 0, lift], f=f, p=p, fingers=fi, thumb=th), steps=40, tag="lift")
    snap(f"{name}_lift")
    gain = pv["can_pos"][2] - can[2]
    print(f"   RESULT {name}: can moved {moved_before:.3f} m before closing; z gain {gain:.3f} m; quat {np.round(pv['can_quat_wxyz'], 2).tolist()}; xy {np.round(pv['can_pos'][:2], 3).tolist()}")
    return round(gain, 3), round(moved_before, 3)
res = {}
res["Y1_d6.8"] = grasp_trial4("Y1", [-0.068, -0.044, 0.06])
res["Y2_d8"] = grasp_trial4("Y2", [-0.08, -0.044, 0.06])
res["Y3_d6.8_t1first"] = grasp_trial4("Y3", [-0.068, -0.044, 0.06], stages=((0.0, 1.0, 35), (1.0, 1.0, 40)))
res["Y4_d10"] = grasp_trial4("Y4", [-0.10, -0.044, 0.06])
print("SUMMARY", res)
