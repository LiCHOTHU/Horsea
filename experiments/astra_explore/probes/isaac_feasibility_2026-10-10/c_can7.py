def grasp_trial3(name, w_off, f=(1, 0, 0), p=(0, 1, 0), stages=((1.0, 0.0, 40), (1.0, 1.0, 35)), side_shift=0.025):
    o = task.reset()
    can = np.array(task.privileged()["can_pos"])
    w = can + np.array(w_off)
    D(right=H(w + [-0.08, -side_shift, 0.04], f=f, p=p), steps=50, tag="pre")
    D(right=H(w + [-0.08, -side_shift, 0.0], f=f, p=p), steps=15, tag="pre2")
    D(right=H(w + [0.0, -side_shift, 0.0], f=f, p=p), steps=20, tag="forward")
    D(right=H(w, f=f, p=p), steps=15, tag="side in")
    for k, (fi, th, n) in enumerate(stages):
        D(right=H(w, f=f, p=p, fingers=fi, thumb=th), steps=n, tag=f"close{k} f{fi} t{th}")
    snap(f"{name}_closed")
    fi, th, _ = stages[-1]
    o, r, pv = D(right=H(w + [0, 0, 0.15], f=f, p=p, fingers=fi, thumb=th), steps=35, tag="lift")
    snap(f"{name}_lift")
    gain = pv["can_pos"][2] - can[2]
    print(f"   RESULT {name}: can z gain {gain:.3f} m; quat {np.round(pv['can_quat_wxyz'], 2).tolist()}; can xy {np.round(pv['can_pos'][:2], 3).tolist()}")
    return round(gain, 3)
res = {}
res["W1_d10"] = grasp_trial3("W1", [-0.10, -0.042, 0.06])
res["W2_d9"] = grasp_trial3("W2", [-0.09, -0.042, 0.06])
res["W3_d10_f08_t08"] = grasp_trial3("W3", [-0.10, -0.042, 0.06], stages=((0.8, 0.0, 30), (0.8, 0.8, 30)))
res["W4_d11_both"] = grasp_trial3("W4", [-0.11, -0.042, 0.06], stages=((1.0, 1.0, 45),))
print("SUMMARY", res)
