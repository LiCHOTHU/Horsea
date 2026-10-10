def pinch_trial(name, w_off, f, p, fingers=0.0, thumb=1.0, close_steps=45, lift=0.15):
    o = task.reset()
    can = np.array(task.privileged()["can_pos"])
    w = can + np.array(w_off)
    D(right=H(w + [0, 0, 0.10], f=f, p=p), steps=55, tag="above")
    o, r, pv = D(right=H(w, f=f, p=p), steps=25, tag="descend")
    snap(f"{name}_down")
    print("    tips", o["proprio"]["right"]["index_tip_m_world"], "thumb", o["proprio"]["right"]["thumb_tip_m_world"])
    D(right=H(w, f=f, p=p, fingers=fingers, thumb=thumb), steps=close_steps, tag="pinch")
    snap(f"{name}_pinched")
    o, r, pv = D(right=H(w + [0, 0, lift], f=f, p=p, fingers=fingers, thumb=thumb), steps=35, tag="lift")
    snap(f"{name}_lift")
    gain = pv["can_pos"][2] - can[2]
    print(f"   RESULT {name}: can z gain {gain:.3f} m; quat {np.round(pv['can_quat_wxyz'], 2).tolist()}; can xy {np.round(pv['can_pos'][:2], 3).tolist()}")
    return round(gain, 3)
res = {}
res["P1_down_palmfwd_thumbright"] = pinch_trial("P1", [-0.042, 0.062, 0.215], (0, 0, -1), (1, 0, 0))
res["P3_down_palmleft_thumbfwd"] = pinch_trial("P3", [-0.062, -0.045, 0.215], (0, 0, -1), (0, 1, 0))
print("SUMMARY", res)
