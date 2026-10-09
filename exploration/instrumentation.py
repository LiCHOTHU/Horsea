"""Scene construction ONLY: visible, non-colliding markers and shared cameras.

This module may access asset links to attach render geometry. Online tracking
imports vision.py instead and has no simulator/asset access. These experiments
are explicitly instrumented-scene diagnostics on one fixed microwave asset.
"""
import copy
import numpy as np


COLORS = {"door0": (1., 0., 0.), "door1": (0., 1., 0.), "door2": (0., 0., 1.),
          "body0": (0., 1., 1.), "body1": (1., 0., 1.), "body2": (1., 1., 0.)}


def add_cameras(cfg):
    cfg = copy.deepcopy(cfg)
    for name, pos in (("explore_right", [.5, -.35, 1.35]),
                      ("explore_left", [-.55, -.35, 1.35]),
                      ("explore_front", [-.1, -.6, .95])):
        forward = np.array([-.07, .10, .93]) - pos
        forward /= np.linalg.norm(forward)
        left = np.cross([0., 0., 1.], forward)
        left /= np.linalg.norm(left)
        cfg["left_embodiment_config"]["static_camera_list"].append(
            {"name": name, "type": "D435", "position": pos,
             "forward": forward.tolist(), "left": left.tolist()})
    return cfg


def attach_markers(env):
    import sapien
    actor = env.microwave
    link = actor.link_dict[actor.config["contact_points"][0]["base"]]
    handle_local = np.asarray(actor.config["contact_points"][0]["matrix"])[:3, 3] * actor.config["scale"]
    door_local = [handle_local + d for d in ([-.13, -.04, .008], [-.03, -.04, .008], [-.13, .04, .008])]
    root = actor.link_dict["link_3"]  # Fixed microwave housing on asset 7167.
    root_pose = root.get_pose().to_transformation_matrix()
    link_pose = link.get_pose().to_transformation_matrix()
    # A rigid marker frame above the body, independent of the moving panel.
    center = link_pose[:3, :3] @ door_local[0] + link_pose[:3, 3] + [0, 0, .16]
    body_world = [center + d for d in ([0, 0, 0], [.08, 0, 0], [0, .08, 0])]
    body_local = [root_pose[:3, :3].T @ (p - root_pose[:3, 3]) for p in body_world]
    layout = {}
    for part, parent, positions in (("door", link, door_local), ("body", root, body_local)):
        entity = parent.entity
        render = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if render is None:
            render = sapien.render.RenderBodyComponent()
        else:
            entity.remove_component(render)
        for i, position in enumerate(positions):
            name = f"{part}{i}"
            color = COLORS[name]
            material = sapien.render.RenderMaterial()
            material.base_color = [*color, 1.]
            material.emission = [*color, 1.]
            material.roughness = 1.
            shape = sapien.render.RenderShapeSphere(.006, material)
            shape.local_pose = sapien.Pose(position)
            render.attach(shape)
            layout[name] = {"link": parent.get_name(), "local_position": position.tolist(),
                            "rgb": list(color), "radius_m": .006, "collision": False}
        entity.add_component(render)
    return layout


def fixed_asset_environment(api):
    """Freeze the repertoire to asset 7167; scene seeds vary its placement."""
    from envs.open_microwave import open_microwave
    from envs.utils import rand_create_sapien_urdf_obj

    class InstrumentedMicrowave(open_microwave):
        def load_actors(self):
            self.model_name, self.model_id = "044_microwave", 0
            self.microwave = rand_create_sapien_urdf_obj(
                scene=self, modelname=self.model_name, modelid=0,
                xlim=[-.12, -.02], ylim=[.15, .2], zlim=[.8, .8],
                qpos=[.707, 0, 0, .707], fix_root_link=True)
            self.microwave.set_mass(.01)
            self.microwave.set_properties(0., 0.)
            self.add_prohibit_area(self.microwave)
            self.prohibited_area.append([-.25, -.25, .25, .1])
            self.marker_layout = attach_markers(self)

    return InstrumentedMicrowave()
