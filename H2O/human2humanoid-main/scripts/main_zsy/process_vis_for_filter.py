"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Visualize motion library
"""
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import joblib
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from phc.utils.motion_lib_g1 import MotionLibG1
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.flags import flags
import parser
import os
# added by zsy
MY_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pickle


flags.test = True
flags.im_eval = True


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# simple asset descriptor for selecting from a list


class AssetDesc:

    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments

def process_vis(my_h1_xml,  my_h1_urdf,  my_motion_file, num_motions):

    # # 机器人路径
    h1_xml  = my_h1_xml
    h1_urdf = my_h1_urdf
    # h1_xml = "/home/zhushiyu/文档/H2O/human2humanoid-main/resources/robots/h1/h1.xml"
    # h1_urdf = "/home/zhushiyu/文档/H2O/human2humanoid-main/resources/robots/h1/urdf/h1.urdf"

    asset_descriptors = [
        # AssetDesc(h1_xml, False),
        AssetDesc(h1_urdf, False),
    ]
    sk_tree = SkeletonTree.from_mjcf(h1_xml)

    # motion_file = "data/h1/test.pkl"
    motion_file = my_motion_file

    if os.path.exists(motion_file):
        print(f"loading {motion_file}")
    else:
        raise ValueError(f"Motion file {motion_file} does not exist! Please run grad_fit_h1.py first.")

    # parse arguments
    # 参数设置
    args = gymutil.parse_arguments(description="Joint monkey: Animate degree-of-freedom ranges",
                                custom_parameters=[{
                                    "name": "--asset_id",
                                    "type": int,
                                    "default": 0,
                                    "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)
                                }, {
                                    "name": "--speed_scale",
                                    "type": float,
                                    "default": 1.0,
                                    "help": "Animation speed scale"
                                }, {
                                    "name": "--show_axis",
                                    "action": "store_true",
                                    "help": "Visualize DOF axis"
                                }])

    if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
        print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
        quit()

    # initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    if args.physics_engine == gymapi.SIM_FLEX:
        pass
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    if not args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    # load asset
    # asset_root = "amp/data/assets"



    # 根目录 zsy
    # asset_root = "./" 原来    
    asset_root = f"{MY_ROOT_DIR}"
    asset_file = asset_descriptors[args.asset_id].file_name

    asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    # asset_options.flip_visual_attachments = asset_descriptors[
    #     args.asset_id].flip_visual_attachments
    asset_options.use_mesh_materials = True

    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # set up the env grid
    num_envs = 1
    num_per_row = 5
    spacing = 5
    env_lower = gymapi.Vec3(-spacing, spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # position the camera
    cam_pos = gymapi.Vec3(0, -10.0, 3)
    cam_target = gymapi.Vec3(0, 0, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # cache useful handles
    envs = []
    actor_handles = []

    num_dofs = gym.get_asset_dof_count(asset)
    print("Creating %d environments" % num_envs)
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0, 0.0)
        pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

        actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
        actor_handles.append(actor_handle)

        # set default DOF positions
        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
        gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)


    gym.prepare_sim(sim)

###########################################################################################################################

    device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))

    motion_lib = MotionLibG1(motion_file=motion_file, device=device, masterfoot_conifg=None, fix_height=False, multi_thread=False, mjcf_file=h1_xml)
    # 采样的motion的个数
    # num_motions = 5
    curr_start = 0
    # 加载motion，好像motion的个数也是可以从这里面得出：
    motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False)
    motion_keys = motion_lib.curr_motion_keys    # tensor['0-ACCAD_Male2MartialArtsKicks_c3d_G14-  roundhouse body left_poses'、'0-ACCAD_Male2MartialArtsKicks_c3d_G4 -spinning back kick_poses'、......
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>",motion_keys)
    current_dof = 0
    speeds = np.zeros(num_dofs)

    time_step = 0
    rigidbody_state = gym.acquire_rigid_body_state_tensor(sim)
    rigidbody_state = gymtorch.wrap_tensor(rigidbody_state)
    rigidbody_state = rigidbody_state.reshape(num_envs, -1, 13)

    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    actor_root_state = gymtorch.wrap_tensor(actor_root_state)

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT,        "LEFT")     #"下一个")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT,       "RIGHT")    #"上一个")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN,        "DOWN")     #"下一个batch")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP,          "UP")       #"上一个batch")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A,           "A")        #"添加到buff")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D,           "D")        #"从buff里删除最后一个")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S,           "S")        #"保存文件")
    # gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W,           "W")#"打印buff")
    
    
    motion_id = 0
    # zsy
    # motion_acc = set()
    motion_acc = list()




    env_ids = torch.arange(num_envs).int().to(args.sim_device)


    ## Create sphere actors
    radius = 0.1
    color = gymapi.Vec3(0.5, 0.5, 0.5) #颜色
    sphere_params = gymapi.AssetOptions()

    sphere_asset = gym.create_sphere(sim, radius, sphere_params)

    num_spheres = 19
    init_positions = gymapi.Vec3(0.0, 0.0, 0.0)
    spacing = 0.





    while not gym.query_viewer_has_closed(viewer):
        # step the physics

        motion_len = motion_lib.get_motion_length(motion_id).item()
        motion_time = time_step % motion_len
        # motion_time = 0
        # import pdb; pdb.set_trace()
        # print(motion_id, motion_time)

        # 获得参考动作：
        motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]).to(args.compute_device_id), torch.tensor([motion_time]).to(args.compute_device_id))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                    motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                    motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        if args.show_axis:
            gym.clear_lines(viewer)
            
        gym.clear_lines(viewer)
        gym.refresh_rigid_body_state_tensor(sim)
        # import pdb; pdb.set_trace()
        idx = 0
        for pos_joint in rb_pos[0, 1:]: # idx 0 torso (duplicate with 11)
            sphere_geom2 = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(1, 0.0, 0.0))
            sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
            gymutil.draw_lines(sphere_geom2, gym, viewer, envs[0], sphere_pose) 
        # import pdb; pdb.set_trace()
            
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1).repeat(num_envs, 1)
        # gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(env_ids), len(env_ids))

        gym.refresh_actor_root_state_tensor(sim)

        # dof_pos = dof_pos.cpu().numpy()
        # dof_states['pos'] = dof_pos
        # speed = speeds[current_dof]
        dof_state = torch.stack([dof_pos, torch.zeros_like(dof_pos)], dim=-1).squeeze().repeat(num_envs, 1)
        gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state), gymtorch.unwrap_tensor(env_ids), len(env_ids))

        gym.simulate(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.fetch_results(sim, True)
        

        # print((rigidbody_state[None, ] - rigidbody_state[:, None]).sum().abs())
        # print((actor_root_state[None, ] - actor_root_state[:, None]).sum().abs())

        # pose_quat = motion_lib._motion_data['0-ACCAD_Female1Running_c3d_C5 - walk to run_poses']['pose_quat_global']
        # diff = quat_mul(quat_inverse(rb_rot[0, :]), rigidbody_state[0, :, 3:7]); np.set_printoptions(precision=4, suppress=1); print(diff.cpu().numpy()); print(torch_utils.quat_to_angle_axis(diff)[0])

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
        # time_step += 1/5
        time_step += dt
        for evt in gym.query_viewer_action_events(viewer):
            # <-
            if evt.action == "LEFT" and evt.value > 0:
                motion_id = (motion_id - 1) % num_motions   # num_motions是采样的
                print(f"ID号: {motion_id + curr_start}. 长度: {motion_len:.3f}. 名字: {motion_keys[motion_id]}")
            
            # ->
            elif evt.action == "RIGHT" and evt.value > 0:
                motion_id = (motion_id + 1) % num_motions
                print(f"ID号: {motion_id + curr_start}. 长度: {motion_len:.3f}. 名字: {motion_keys[motion_id]}")
            
            # add
            elif evt.action == "A" and evt.value > 0:
                motion_acc.append(motion_keys[motion_id])    # 添加到motion_acc。。。
                print(f"当前motion_acc: {motion_acc}")
            
            # delete
            elif evt.action == "D" and evt.value > 0:
                motion_acc.pop()    # 添加到motion_acc。。。
                print(f"当前motion_acc: {motion_acc}")

            # next batch
            elif evt.action == "DOWN" and evt.value > 0:
                curr_start += num_motions
                motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
                motion_keys = motion_lib.curr_motion_keys
                print(f"Next batch {curr_start}")
                motion_id = 0

            # last batch
            elif evt.action == "UP" and evt.value > 0:
                curr_start -= num_motions
                motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
                motion_keys = motion_lib.curr_motion_keys
                print(f"Last batch {curr_start}")
                motion_id = 0
            # save
            elif evt.action == "S" and evt.value > 0:
                os.makedirs("./data_amass/g1_filtered", exist_ok=True)
                save_motion_acc(motion_acc, motion_file, "./data_amass/g1_filtered/g1_motion_test1.pkl")



            time_step = 0

    print("Done")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

######################################################################################################
def save_motion_acc(motion_acc, path_motion_file_input, path_motion_file_output):
    print("Saving motion_acc...")
    # motion_lib = MotionLibG1(motion_file=motion_file, device=device, masterfoot_conifg=None, fix_height=False, multi_thread=False, mjcf_file=h1_xml)
    # motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
    my_load_path = path_motion_file_input
    motion_dict = joblib.load(my_load_path)
    save_motion_acc_buff = dict()
    i = 0
    for data_key in motion_acc:
        save_motion_acc_buff[i] = motion_dict[data_key]
        i += 1
    # save
    joblib.dump(save_motion_acc_buff, path_motion_file_output)
    print(f"Saving   to  {path_motion_file_output}")

# ######################################################################################
# =========================================== main =====================================
# ######################################################################################
if __name__ == '__main__':
    robot_name = "g1"
    if robot_name == "g1":
        pkl_name = "ACCAD_g1_amass_all"
        path_xml = "../../resources/robots/g1/g1.xml"
        path_urdf = "resources/robots/g1/urdf/g1.urdf"
        path_motion_file = f"{MY_ROOT_DIR}/scripts/main_zsy/data_amass/g1/{pkl_name}.pkl"
        num_motions = 30     #这个变量指的是一次性可以看的动作数目，比如AMASS一共有10000个动作，这个量设为50,则按照50的batch进行采样，一次也就只能看50个动作。
        process_vis(path_xml, path_urdf,path_motion_file, num_motions)


    elif robot_name == "h1":
        pkl_name = "amass_phc_filtered"
        path_xml = "../../resources/robots/h1/h1.xml"
        path_urdf = "resources/robots/h1/urdf/h1.urdf"
        path_motion_file = f"{MY_ROOT_DIR}/scripts/main_zsy/data_amass/g1/{pkl_name}.pkl"
        num_motions = 30
        process_vis(path_xml, path_urdf, path_motion_file, num_motions)
    else:
        pass




