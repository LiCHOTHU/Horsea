"""Privileged diagnostics. Never import this module in perception or memory."""
import numpy as np


class AuditRecorder:
    def __init__(self):
        self.frames = []

    def capture(self, env, phase):
        # Object links/contacts are audit labels only, stored in a separate log.
        objects = {x.get_name() for x in env.microwave.actor.get_links()}
        fingers = set(env.robot.gripper_name)
        contacts = []
        for contact in env.scene.get_contacts():
            a, b = [x.entity.name for x in contact.bodies]
            if ((a in objects and b in fingers) or (b in objects and a in fingers)) and contact.points:
                contacts.append([a, b])
        self.frames.append({"phase": phase, "physics_steps": env.scene.steps,
                            "joint": np.asarray(env.microwave.get_qpos()).tolist(),
                            "finger_object_contact": bool(contacts), "contact_pairs": contacts,
                            "simulator_success": bool(env.check_success())})

    def result(self, env):
        return {"backend": "robotwin", "claim": "instrumented-scene diagnostic",
                "asset": "044_microwave/7167", "simulator_success": bool(env.check_success()),
                "joint": np.asarray(env.microwave.get_qpos()).tolist(),
                "planner_success": bool(env.plan_success), "physics_steps": env.scene.steps,
                "frames": self.frames,
                "privileged_access": ["object_joint", "simulator_success", "finger_object_contacts"]}
