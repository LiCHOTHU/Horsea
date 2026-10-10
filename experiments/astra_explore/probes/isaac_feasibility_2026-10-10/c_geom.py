from pxr import Usd, UsdGeom
import omni.usd
stage = omni.usd.get_context().get_stage()
cal = task.calibration
obs = task.reset()
obs0 = obs
for s in ("left", "right"):
    p = obs["proprio"][s]
    print(s, "wrist", p["wrist_position_m_world"], "fingers", p["fingers_direction_world"], "palm", p["palm_direction_world"], "thumb side", p["thumb_side_direction_world"], "tips", p["mean_fingertip_position_m_world"], "thumb-index", p["thumb_index_distance_cm"])
def bbox(path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(prim).ComputeAlignedRange()
    return [round(x, 3) for x in r.GetMin()], [round(x, 3) for x in r.GetMax()]
root = "/World/envs/env_0/Drawer"
for prim in stage.Traverse():
    pth = str(prim.GetPath())
    if pth.startswith(root) and pth.count("/") <= 6:
        t = prim.GetTypeName()
        if t in ("Xform", "Mesh", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "") or "Joint" in t:
            print(pth, t, bbox(pth) if t in ("Xform", "Mesh") else "")
for jp in [p for p in stage.Traverse() if str(p.GetPath()).startswith(root) and "Joint" in p.GetTypeName()]:
    a = {k: jp.GetAttribute(k).Get() for k in ("physics:axis", "physics:localPos0", "physics:localPos1", "physics:lowerLimit", "physics:upperLimit") if jp.GetAttribute(k)}
    rel0 = jp.GetRelationship("physics:body0").GetTargets() if jp.GetRelationship("physics:body0") else None
    rel1 = jp.GetRelationship("physics:body1").GetTargets() if jp.GetRelationship("physics:body1") else None
    print("JOINT", jp.GetPath(), a, rel0, rel1)
print("can bbox", bbox("/World/envs/env_0/Can"), "plate bbox", bbox("/World/envs/env_0/Plate"))
print("priv", {k: v for k, v in task.privileged().items() if "pos" in k})
