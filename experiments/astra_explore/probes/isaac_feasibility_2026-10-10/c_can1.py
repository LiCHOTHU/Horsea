o = task.reset()
can = np.array(task.privileged()["can_pos"])
cz = can[2] + 0.06
print("can", can.round(3).tolist(), "right start", o["proprio"]["right"]["wrist_position_m_world"])
# right hand: palm facing +y (to the can), fingers forward (+x); thumb on top
D(right=H((can[0] - 0.10, can[1] - 0.12, cz + 0.06), f=(1, 0, 0), p=(0, 1, 0)), steps=45, tag="R pregrasp right of can")
snap("can_a")
o, r, pv = D(right=H((can[0] - 0.08, can[1] - 0.06, cz), f=(1, 0, 0), p=(0, 1, 0)), steps=40, tag="R palm to can")
print("   right tips", o["proprio"]["right"]["mean_fingertip_position_m_world"], "thumb", o["proprio"]["right"]["thumb_tip_m_world"], "thumb side", o["proprio"]["right"]["thumb_side_direction_world"])
snap("can_b")
D(right=H((can[0] - 0.08, can[1] - 0.06, cz), f=(1, 0, 0), p=(0, 1, 0), fingers=1.0, thumb=1.0), steps=30, tag="R close")
snap("can_c")
D(right=H((can[0] - 0.08, can[1] - 0.06, cz + 0.12), f=(1, 0, 0), p=(0, 1, 0), fingers=1.0, thumb=1.0), steps=30, tag="R lift 12cm")
snap("can_d")
