"""Cup-transfer task environment: the benchmark's base scene (H1 + Inspire hands, table, room, cameras) with ONE rigid test
object on the table and the benchmark's tray (container) as a placement target. Registered as Humanoid-Cup-v0; import only
after the Isaac app is running (IsaacDexTask(task_modules=["cup_transfer.env_cups"])).

Configuration (env_cfg fields, set before gym.make or on U.cfg between episodes):
  target.spawn.usd_path   which object (cup_a / cup_b / cyl_c / cup_bp USD from assets.py)
  object_xy, start_offset_xy   object start = object_xy + start_offset_xy on the table (predefined offsets per start)
  tray_xy                 tray position (placement target, Task 4)
  task_mode               "lift": success = object base >= its rest height + 5 cm with speed <= 0.10 m/s for 30 consecutive
                          control steps (1 s); "place": object base inside the tray footprint, upright (up-axis z >= 0.9),
                          no fingertip within 3 cm of the object, speed <= 0.05 m/s, for 30 consecutive steps.
The rest height (lift_ref_z) is set by the harness after its settle steps; before that success is impossible.
"""
from __future__ import annotations

import torch
from collections.abc import Sequence

import gymnasium as gym
import isaaclab.sim as sim_utils
from humanoid.tasks.base_env.base_env import BaseEnv, BaseEnvCfg
from humanoid.tasks.data.container import CONTAINER_2X3_CFG
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.utils import configclass

TARGET_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Target",
    spawn=sim_utils.UsdFileCfg(
        usd_path="",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(       # the benchmark can's settings
            solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_angular_velocity=1000.0,
            max_linear_velocity=1000.0, max_depenetration_velocity=50.0, kinematic_enabled=False),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.36, -0.13, 1.03), rot=(1.0, 0.0, 0.0, 0.0)),
)


@configclass
class CupEnvCfg(BaseEnvCfg):
    target: RigidObjectCfg = TARGET_CFG
    container: RigidObjectCfg = CONTAINER_2X3_CFG.replace(prim_path="/World/envs/env_.*/Container")  # type: ignore
    container.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.46, 0.22, 1.06), rot=(1.0, 0.0, 0.0, 0.0))
    object_xy = (0.36, -0.13)
    object_spawn_z = 1.03
    start_offset_xy = (0.0, 0.0)
    tray_xy = (0.46, 0.22)
    tray_half_xy = (0.15, 0.10)
    task_mode = "lift"
    lift_height = 0.05
    hold_steps = 30


class CupEnv(BaseEnv):
    cfg: CupEnvCfg

    def __init__(self, cfg: CupEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.success = torch.zeros([self.num_envs], dtype=torch.bool, device=self.device)
        self.hold_count = torch.zeros([self.num_envs], dtype=torch.long, device=self.device)
        self.lift_ref_z = None

    def _setup_scene(self):
        self.target = RigidObject(self.cfg.target)
        self.container = RigidObject(self.cfg.container)
        super()._setup_scene()

    def _register_scene(self):
        super()._register_scene()
        self.scene.rigid_objects["target"] = self.target
        self.scene.rigid_objects["container"] = self.container

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # type: ignore
        st = self.target.data.default_root_state[env_ids].clone()
        st[:, 0] = self.cfg.object_xy[0] + self.cfg.start_offset_xy[0]
        st[:, 1] = self.cfg.object_xy[1] + self.cfg.start_offset_xy[1]
        st[:, 2] = self.cfg.object_spawn_z
        st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        st[:, 7:] = 0.0
        st[:, 0:3] += self.scene.env_origins[env_ids]
        self.target.write_root_state_to_sim(st, env_ids=env_ids)
        ct = self.container.data.default_root_state[env_ids].clone()
        ct[:, 0], ct[:, 1] = self.cfg.tray_xy[0], self.cfg.tray_xy[1]
        ct[:, 7:] = 0.0
        ct[:, 0:3] += self.scene.env_origins[env_ids]
        self.container.write_root_state_to_sim(ct, env_ids=env_ids)
        self.hold_count[env_ids] = 0
        self.success[env_ids] = False
        self.lift_ref_z = None

    def _object_ok(self):
        pos = self.target.data.root_link_pose_w[:, 0:3] - self.scene.env_origins
        quat = self.target.data.root_link_pose_w[:, 3:7]
        vel = torch.norm(self.target.data.root_com_lin_vel_w, dim=-1)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        up_z = 1 - 2 * (x * x + y * y)
        if self.cfg.task_mode == "lift":
            if self.lift_ref_z is None:
                return torch.zeros_like(self.success)
            return (pos[:, 2] >= self.lift_ref_z + self.cfg.lift_height) & (vel <= 0.10)
        tx, ty = self.cfg.tray_xy
        hx, hy = self.cfg.tray_half_xy
        inside = (torch.abs(pos[:, 0] - tx) <= hx) & (torch.abs(pos[:, 1] - ty) <= hy)
        tips = torch.cat([self.robot.data.body_state_w[:, self.left_finger_tips_idx, 0:3],
                          self.robot.data.body_state_w[:, self.right_finger_tips_idx, 0:3]], dim=1) - self.scene.env_origins.unsqueeze(1)
        centre = pos + torch.stack([torch.zeros_like(up_z), torch.zeros_like(up_z), torch.full_like(up_z, 0.06)], dim=-1)
        free = torch.min(torch.norm(tips - centre.unsqueeze(1), dim=-1), dim=1).values > 0.08
        return inside & (up_z >= 0.9) & (vel <= 0.05) & free

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        ok = self._object_ok()
        self.hold_count = torch.where(ok, self.hold_count + 1, torch.zeros_like(self.hold_count))
        self.success = self.hold_count >= self.cfg.hold_steps
        p = self.target.data.root_link_pose_w[:, 0:7].clone()
        p[:, 0:3] -= self.scene.env_origins
        obs["object_pose"] = p
        obs["success"] = self.success
        return obs


gym.register(id="Humanoid-Cup-v0", entry_point="cup_transfer.env_cups:CupEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": CupEnvCfg})
