import sys
sys.path.insert(0, "/home/licho/workspace/Horsea/astra_explore")
from schema_dex import validate
from PIL import Image
SPD = "/tmp/claude-1001/-home-licho-workspace-Horsea/6362e6e4-1b6e-43f7-ba9b-09289d83a6f6/scratchpad/feas"
Path(SPD).mkdir(exist_ok=True)
HOLD = {"move": False, "position_m": [0, 0, 0], "fingers_dir": [1, 0, 0], "palm_dir": [0, 0, -1], "finger_base": 0.0, "finger_tip": 0.0, "thumb_rotate": 0.0, "thumb_bend": 0.0}
def H(pos=None, f=(1, 0, 0), p=(0, 0, -1), base=0.0, tip=0.0, rot=0.0, bend=0.0):
    c = {"finger_base": base, "finger_tip": tip, "thumb_rotate": rot, "thumb_bend": bend}
    if pos is None:
        return dict(HOLD, **c)
    return dict({"move": True, "position_m": list(pos), "fingers_dir": list(f), "palm_dir": list(p)}, **c)
def D(left=None, right=None, steps=30, tag=""):
    resp = {"left": left or dict(HOLD), "right": right or dict(HOLD), "steps": steps, "stop": False, "assessment_of_previous_action": None,
            "intent": tag or "probe", "hypothesis": None, "expected_change": "", "references": [], "memory_update": {"observations": [], "hypotheses": [], "summary": ""}}
    clean, w = validate(resp, set())
    o, r = task.act(clean)
    pv = task.privileged()
    h = r["hands"]
    print(f"[{tag}] t={o['steps_used']} " + " | ".join(f"{s[0].upper()}: err {h[s].get('position_error_cm', '-')}cm/{h[s].get('orientation_error_deg', '-')}deg moved {h[s]['moved_cm']} wrist {h[s]['wrist_position_m_world']} ti {h[s]['thumb_index_distance_cm']} F {h[s]['index_contact_force_N']}" for s in ('left', 'right')))
    print(f"      can {np.round(pv['can_pos'], 3).tolist()} q {np.round(pv['can_quat_wxyz'], 2).tolist()} plate {np.round(pv['plate_pos'], 3).tolist()} door {np.round(pv['drawer_joint_pos'], 3).tolist()} success {pv['success']} {('WARN ' + str(w)) if w else ''}")
    return o, r, pv
def snap(name):
    Image.fromarray(task._image("main_camera", (960, 540))).save(f"{SPD}/{name}_main.png")
print("helpers ok")
