import torch
cam = task.U.main_camera
p0, q0 = cam.data.pos_w.clone(), cam.data.quat_w_world.clone()
def look(eye, target, name):
    eye_t = torch.tensor([eye], dtype=torch.float32, device=task.dev)
    tgt_t = torch.tensor([target], dtype=torch.float32, device=task.dev)
    cam.set_world_poses_from_view(eye_t, tgt_t)
    hold = task.robot.data.joint_pos.clone()
    task._run(3, hold_q=hold)
    Image.fromarray(task._image("main_camera", (1280, 720))).save(f"/tmp/claude-1001/-home-licho-workspace-Horsea/6362e6e4-1b6e-43f7-ba9b-09289d83a6f6/scratchpad/{name}.png")
from PIL import Image
look([0.55, -1.3, 1.45], [0.5, 0.1, 1.2], "view_side_right")
look([1.6, 0.2, 1.9], [0.45, 0.1, 1.15], "view_far_side")
look([0.4, 0.2, 2.6], [0.45, 0.1, 1.1], "view_top")
cam.set_world_poses(p0, q0, convention="world")
task._run(3, hold_q=task.robot.data.joint_pos.clone())
print("restored", cam.data.pos_w.cpu().numpy(), cam.data.quat_w_world.cpu().numpy())
