from habitat_sim.physics import MotionType
import magnum as mn
import numpy as np
import habitat_sim
from orp.utils import make_render_only

def add_obj(name, sim):
    if '/BOX_' in name:
        box_size = float(name.split('/BOX_')[-1])
        obj_mgr = sim.get_object_template_manager()
        template_handle = obj_mgr.get_template_handles("cube")[0]
        template = obj_mgr.get_template_by_handle(template_handle)
        template.scale = mn.Vector3(box_size,box_size,box_size)
        template.requires_lighting = True
        new_template_handle = obj_mgr.register_template(template, "box_new")
        obj_id = sim.add_object(new_template_handle)
        sim.set_object_motion_type(MotionType.DYNAMIC, obj_id)
        return obj_id

    PROP_FILE_END = ".object_config.json"
    use_name = name + PROP_FILE_END

    obj_id = sim.add_object_by_handle(use_name)
    return obj_id

def place_viz_objs(name_trans, sim, obj_ids=[]):
    viz_obj_ids = []
    for i, (name, trans) in enumerate(name_trans):
        if len(obj_ids) == 0:
            viz_obj_id = add_obj(name, sim)
        else:
            viz_obj_id = obj_ids[i]

        make_render_only(viz_obj_id, sim)
        sim.set_transformation(trans, viz_obj_id)
        viz_obj_ids.append(viz_obj_id)
    if len(obj_ids) != 0:
        return obj_ids
    return viz_obj_ids


def load_articulated_objs(name_obj_dat, sim, obj_ids=[], frozen=False, return_urdfs=False):
    """
    Same params as `orp.obj_loaders.load_objs`
    """
    art_obj_ids = []
    urdfs = []
    for i, (name, obj_dat) in enumerate(name_obj_dat):
        obj_id = sim.add_articulated_object_from_urdf(name, frozen)
        urdfs.append(name)
        trans = obj_dat[0]
        obj_type = obj_dat[1]
        T = mn.Matrix4(trans)
        sim.set_articulated_object_root_state(obj_id, T)
        sim.set_articulated_object_sleep(obj_id, False)
        sim.set_articulated_object_motion_type(obj_id, MotionType(obj_type))
        art_obj_ids.append(obj_id)
    if return_urdfs:
        return art_obj_ids, urdfs
    return art_obj_ids


def init_art_objs(idx_and_state, sim):
    for art_obj_idx, art_state in idx_and_state:
        sim.set_articulated_object_positions(art_obj_idx,
                np.array(art_state))
        # Default motors for all NONROBOT articulated objects.
        for i in range(len(art_state)):
            jms = habitat_sim.physics.JointMotorSettings(
                    0.0,  # position_target
                    0.0,  # position_gain
                    0.0,  # velocity_target
                    0.3,  # velocity_gain
                    1.0,  # max_impulse
                )
            sim.update_joint_motor(art_obj_idx, i, jms)


def load_objs(name_obj_dat, sim, obj_ids):
    """
    - name_obj_dat: List[(str, List[
        transformation as a 4x4 list of lists of floats,
        int representing the motion type
      ])
    """
    static_obj_ids = []
    for i, (name, obj_dat) in enumerate(name_obj_dat):
        if len(obj_ids) == 0:
            obj_id = add_obj(name, sim)
        else:
            obj_id = obj_ids[i]
        trans = obj_dat[0]
        obj_type = obj_dat[1]

        use_trans = mn.Matrix4(trans)
        sim.set_transformation(use_trans, obj_id)
        sim.set_linear_velocity(mn.Vector3(0,0,0), obj_id)
        sim.set_angular_velocity(mn.Vector3(0,0,0), obj_id)
        sim.set_object_motion_type(MotionType(obj_type), obj_id)
        sim.set_object_sleep(obj_id, True)
        static_obj_ids.append(obj_id)
    if len(obj_ids) != 0:
        return obj_ids
    return static_obj_ids


