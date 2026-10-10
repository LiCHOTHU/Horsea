import itertools
o = task.reset()
hold = task.robot.data.joint_pos.clone()
cal = task.calibration["right"]
print("calibration right:", {k: v for k, v in cal.items() if k not in ("tips_in_base_frame_open", "hand_joint_names")})
f_b, p_b, t_b = np.array(cal["f_b"]), np.array(cal["p_b"]), np.array(cal["thumb_side_b"])
from isaac_env import mat_from_quat
def frame_coords(P):
    ee = task._ee("right"); Rm = mat_from_quat(ee[3:7]); v = Rm.T @ (np.asarray(P) - ee[:3])
    return [round(100 * float(np.dot(v, f_b)), 1), round(100 * float(np.dot(v, p_b)), 1), round(100 * float(np.dot(v, t_b)), 1)]
names = task.tip_names["right"]
configs = [(b, t, 0, 0) for b, t in ((0.5, 0), (1, 0), (0, 0.5), (0, 1), (0.5, 0.5), (0.5, 1), (1, 1))] + [(0, 0, r, k) for r, k in itertools.product((0, 0.5, 1), (0, 0.5, 1))]
for cfg in configs:
    task._run(35, hands={"right": cfg}, hold_q=hold)
    T = task._tips("right")
    print(f"cfg base/tip/rot/bend {cfg}: [along fingers, in front of palm, toward thumb side] cm ->",
          {n.split("_")[1]: frame_coords(t) for n, t in zip(names, T) if n.split("_")[1] in ("thumb", "index", "middle", "pinky")})
task._run(35, hands={"right": (0, 0, 0, 0)}, hold_q=hold)
