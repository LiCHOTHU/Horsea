def trial(name, fdir, wy_off, z, thumb_pre):
    o = task.reset()
    can = np.array(task.privileged()["can_pos"])
    wx = can[0] - 0.057
    wy = can[1] + wy_off
    print(f"=== {name}: fingers {fdir}, wrist y offset {wy_off}, z {z}, thumb pre {thumb_pre}")
    D(right=H((wx - 0.09, wy, z + 0.05), f=fdir, p=(1, 0, 0), thumb=thumb_pre), steps=50, tag="pre")
    D(right=H((wx - 0.09, wy, z), f=fdir, p=(1, 0, 0), thumb=thumb_pre), steps=25, tag="down")
    D(right=H((wx, wy, z), f=fdir, p=(1, 0, 0), thumb=thumb_pre), steps=30, tag="in")
    snap(f"{name}_in")
    D(right=H((wx, wy, z), f=fdir, p=(1, 0, 0), fingers=1.0, thumb=1.0), steps=30, tag="close")
    snap(f"{name}_closed")
    o, r, pv = D(right=H((wx, wy, z + 0.15), f=fdir, p=(1, 0, 0), fingers=1.0, thumb=1.0), steps=40, tag="lift 15cm")
    snap(f"{name}_lift")
    print(f"   RESULT {name}: can z gain {pv['can_pos'][2] - can[2]:.3f} m")
trial("G4_fingers_right", (0, -1, 0), +0.08, 1.077, 0.0)
trial("G5_fingers_left_thumbtucked", (0, 1, 0), -0.08, 1.10, 1.0)
