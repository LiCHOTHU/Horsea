def squeeze3(name, lift=0.16, carry=((0.40, 1.24), (0.48, 1.26), (0.55, 1.26))):
    o = task.reset()
    can = np.array(task.privileged()["can_pos"])
    fR, pR, fL, pL = (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0)
    zc = can[2] + 0.06
    D(left=H((-0.09, 0.30, 1.22), f=fL, p=pL), right=H((0.10, -0.25, 1.30), f=fR, p=pR), steps=45, tag="L back-out / R reorient")
    D(left=H((-0.09, 0.02, 1.15), f=fL, p=pL), right=H((0.29, -0.165, zc + 0.10), f=fR, p=pR), steps=40, tag="L behind door end / R above-right")
    D(left=H((0.19, 0.02, zc + 0.03), f=fL, p=pL), right=H((0.29, -0.165, zc), f=fR, p=pR), steps=40, tag="L inside door / R beside")
    o, r, pv = D(left=H((0.29, 0.00, zc), f=fL, p=pL), right=H((0.29, -0.165, zc), f=fR, p=pR), steps=30, tag="L beside (reach limit)")
    lwy = o["proprio"]["left"]["wrist_position_m_world"][1]
    print(f"    left wrist y {lwy:.3f}; left palm surface ~{lwy - 0.01:.3f}; can {np.round(pv['can_pos'], 3).tolist()}")
    # right palm pushes the can left until it rests against the left palm, then presses
    target_ry = (lwy - 0.01) - 0.064 - 0.01 + 0.012
    o, r, pv = D(left=H((0.29, lwy - 0.01, zc), f=fL, p=pL), right=H((0.29, target_ry, zc), f=fR, p=pR), steps=45, tag="R pushes can to L palm + squeeze")
    snap(f"{name}_squeeze")
    o, r, pv = D(left=H((0.29, lwy - 0.01, zc + lift), f=fL, p=pL), right=H((0.29, target_ry, zc + lift), f=fR, p=pR), steps=45, tag="lift both")
    snap(f"{name}_lift")
    gain = pv["can_pos"][2] - can[2]
    print(f"   LIFT {name}: z gain {gain:.3f}; can {np.round(pv['can_pos'], 3).tolist()} quat {np.round(pv['can_quat_wxyz'], 2).tolist()}")
    for x, z in carry:
        o, r, pv = D(left=H((x, lwy - 0.01, z), f=fL, p=pL), right=H((x, target_ry, z), f=fR, p=pR), steps=40, tag=f"carry x{x} z{z}")
    snap(f"{name}_carry")
    print(f"   RESULT {name}: final can {np.round(pv['can_pos'], 3).tolist()} plate {np.round(pv['plate_pos'], 3).tolist()} t={o['steps_used']}")
squeeze3("S2")
