from PIL import Image, ImageDraw
import itertools
img = Image.fromarray(task._image("main_camera", (1280, 720)))
d = ImageDraw.Draw(img)
boxes = {"body": ([0.479, -0.26, 1.407], [0.932, 0.234, 2.2]), "door": ([0.128, 0.165, 1.53], [0.529, 0.315, 1.948]), "top": ([0.262, -0.227, 1.972], [0.654, 0.19, 2.162]),
         "can": ([0.348, -0.032, 1.023], [0.412, 0.032, 1.144]), "plate": ([0.551, -0.099, 1.172], [0.749, 0.099, 1.187])}
cols = {"body": (255, 0, 0), "door": (0, 255, 0), "top": (0, 0, 255), "can": (255, 255, 0), "plate": (255, 0, 255)}
for name, (lo, hi) in boxes.items():
    pts = []
    for c in itertools.product(*zip(lo, hi)):
        uv = task._project("main_camera", c, (1280, 720))
        pts.append(uv)
        if uv:
            d.ellipse([uv[0] - 4, uv[1] - 4, uv[0] + 4, uv[1] + 4], outline=cols[name], width=2)
    print(name, [None if p is None else (round(p[0]), round(p[1])) for p in pts])
# hinge candidates and the real simulated can position
for name, P in {"hinge_z1.05": [0.53, 0.183, 1.05], "hinge_z1.4": [0.53, 0.183, 1.4], "can_sim": task.privileged()["can_pos"], "plate_sim": task.privileged()["plate_pos"]}.items():
    uv = task._project("main_camera", P, (1280, 720))
    print(name, uv)
    if uv:
        d.rectangle([uv[0] - 6, uv[1] - 6, uv[0] + 6, uv[1] + 6], outline=(255, 255, 255), width=2)
        d.text((uv[0] + 8, uv[1]), name, fill=(255, 255, 255))
img.save("/tmp/claude-1001/-home-licho-workspace-Horsea/6362e6e4-1b6e-43f7-ba9b-09289d83a6f6/scratchpad/proj_boxes.png")
U = task.U
dr = U.drawer
print("drawer body states:", {n: dr.data.body_state_w[0, i, :7].cpu().numpy().round(3).tolist() for i, n in enumerate(dr.body_names)})
print("root scale attr:", [ (str(p.GetPath()), p.GetAttribute("xformOp:scale").Get()) for p in [__import__("omni.usd").usd.get_context().get_stage().GetPrimAtPath(x) for x in ("/World/envs/env_0/Drawer", "/World/envs/env_0/Drawer/drawer", "/World/envs/env_0/Drawer/drawer/bottom")] if p.IsValid() and p.GetAttribute("xformOp:scale")])
