import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

import numpy as np
import torch
from termcolor import colored
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
from legged_gym.utils.helpers import class_to_dict

NOROSPY = False
try:
    import rospy
except:
    NOROSPY = True
# from std_msgs.msg import String, Header, Float64MultiArray

command_state = {
    'vel_forward': 0.0,
    'vel_side': 0.0,
    'orientation': 0.0,
}

override = False




def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same

@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config", # 修改这里改变读取的config文件
)
def play(cfg_hydra: DictConfig) -> None:
    cfg_hydra = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    cfg_hydra.physics_engine = gymapi.SIM_PHYSX
    env_cfg, train_cfg = cfg_hydra, cfg_hydra.train
    env_cfg.env.num_envs = 1
    env_cfg.viewer.debug_viz = True # 绘制关键点
    env_cfg.motion.visualize = False
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.terrain.mesh_type = 'plane'
    # if env_cfg.terrain.mesh_type == 'trimesh':
    #     env_cfg.terrain.terrain_types = ['flat', 'rough', 'low_obst']  # do not duplicate!
    #     env_cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.episode_length_s = 20
    env_cfg.domain_rand.randomize_rfi_lim = False
    env_cfg.domain_rand.randomize_pd_gain = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_ctrl_delay = False
    env_cfg.domain_rand.ctrl_delay_step_range = [1, 3]


    # env_cfg.asset.termination_scales.max_ref_motion_distance = 1
    


    env_cfg.env.test = True             

    if env_cfg.motion.realtime_vr_keypoints:
        env_cfg.asset.terminate_by_1time_motion = False
        env_cfg.asset.terminate_by_ref_motion_distance = False
        rospy.init_node("avppose_subscriber")
        from avp_pose_subscriber import AVPPoseInfo
        avpposeinfo = AVPPoseInfo()
        rospy.Subscriber("avp_pose", Float64MultiArray, avpposeinfo.avp_callback, queue_size=1)
    if cfg_hydra.joystick:
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        from pynput import keyboard
        from legged_gym.utils import key_response_fn

    # prepare environment
    # ------------- 创建环境 -----------
    env, _ = task_registry.make_env_hydra(name=cfg_hydra.task, hydra_cfg=cfg_hydra, env_cfg=env_cfg)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 4 # which joint is used for logging
    stop_state_log = 200 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

    obs = env.get_observations()
    
    if env_cfg.motion.realtime_vr_keypoints:
        init_root_pos = env._rigid_body_pos[..., 0, :].clone()
        init_avp_pos = avpposeinfo.avp_pose.copy()
        init_root_offset = init_root_pos[0, :2] - init_avp_pos[2, :2]

    train_cfg.runner.resume = True
    
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cfg_hydra.task, args=cfg_hydra, train_cfg=train_cfg)
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    exported_policy_name = str(task_registry.loaded_policy_path.split('/')[-2]) + str(task_registry.loaded_policy_path.split('/')[-1])
    print('Loaded policy from: ', task_registry.loaded_policy_path)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:  
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, exported_policy_name)
        print('Exported policy as jit script to: ', os.path.join(path, exported_policy_name))
    if EXPORT_ONNX:   
        exported_onnx_name = exported_policy_name.replace('.pt', '.onnx')
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_onnx(ppo_runner.alg.actor_critic, path, exported_onnx_name, onnx_num_observations=env_cfg.env.num_observations)
        print('Exported policy as onnx to: ', os.path.join(path, exported_onnx_name))


    if cfg_hydra.joystick:
        print(colored("joystick on", "green"))
        key_response = key_response_fn(mode='vel')
        def on_press(key):
            global command_state
            try:
                key_response(key, command_state, env)
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    i = 0
    
    while (not NOROSPY and not rospy.is_shutdown()) or (NOROSPY):
        # for i in range(1000*int(env.max_episode_length)):

        # obs[:, -19:] = 0 # will destroy the performance
        
        actions = policy(obs.detach().float())
        # print("actions",actions)
        # print(torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1))
        obs, _, rews, dones, infos = env.step(actions.detach())


        if env_cfg.motion.realtime_vr_keypoints:
            avpposeinfo.check()
            keypoints_pos = avpposeinfo.avp_pose.copy()
            keypoints_pos[:, 0] += init_root_offset[0].item()
            keypoints_pos[:, 1] += init_root_offset[1].item()
            # import ipdb; ipdb.set_trace()
            keypoints_vel = avpposeinfo.avp_vel.copy()
            print(keypoints_pos)
            env._update_realtime_vr_keypoints(keypoints_pos, keypoints_vel)
        # print("obs = ", obs)
        # print("actions = ", actions)
        # print()
        # exit()
        if override: 
            obs[:,9] = 0.5
            obs[:,10] = 0.0
            obs[:,11] = 0.0

        # overwrite linear velocity - z and angular velocity - xy
        # obs[:, 40] = 0.
        # obs[: 41:43] = 0.
        # log_obs(obs)
        OPEN_LOGGER_PLOT = True # zsy设置，用来使能logger的plot_states，该打印是默认自带的
        if i < stop_state_log: 
            logger.log_states(
                {
                    # 'dof_pos_target': actions[robot_index, joint_index].item()  * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    # 'dof_pos_target': env.actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    # 'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    # 'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    # 'dof_torque': env.torques[robot_index, joint_index].item(),
                    # 'base_gravity_x': env.projected_gravity[robot_index, 0].item(), #zsy add
                    # 'base_gravity_y': env.projected_gravity[robot_index, 1].item(), #zsy add
                    # 'base_gravity_z': env.projected_gravity[robot_index, 2].item(), #zsy add
                    # 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    # 'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    # 'dof_pos_target': actions[robot_index, 0].item()

                }
            )
        elif i==stop_state_log:
            if OPEN_LOGGER_PLOT: # zsy
                logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            pass
            # logger.print_rewards()
        i += 1

def log_obs(obs, width=80,pad=2):
    log_string = (
        f"""{'-' * width}\n"""
        f"""{'【q】:':>{pad}} {obs[0,0        :     19]}\n"""                  
        f"""{'【dq】:':>{pad}} {obs[0,19       :     19*2]}\n"""              
        f"""{'【omega】:':>{pad}} {obs[0,19*2     :     19*2+3]}\n"""           
        f"""{'【gvec】:':>{pad}} {obs[0,19*2+3   :     19*2+6]}\n"""            
        f"""{'【action】:':>{pad}} {obs[0,19*2+6   :     19*3+6]}\n"""             
        f"""{'【task_obs】:':>{pad}} {obs[0,19*3+6   :     19*3+24]}\n"""     
    )      
    print(log_string)

# ========================================= main =========================================
if __name__ == '__main__':
    EXPORT_POLICY = True
    EXPORT_ONNX = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    # args = get_args()
    play()

# # ======================= play 教师网络的训练控制台参数 =======================
# g1
"""
python  play_hydra.py --config-name=config_teleop_g1 task=g1:teleop env.num_observations=913 env.num_privileged_obs=990 motion.future_tracks=True motion.teleop_obs_version=v-teleop-extend-max-full motion=motion_full_g1 asset.termination_scales.max_ref_motion_distance=10.0 load_run=ACCAD_g1_Teacher_41000 checkpoint=41000 num_envs=1 headless=False 

"""


# # ======================= play 学生网络的训练控制台参数 =======================
"""
python play_hydra.py --config-name=config_teleop_g1 task=g1:teleop env.num_observations=1665 env.num_privileged_obs=1742 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=1 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=10.0 load_run=ACCAD_g1_Student_50000 checkpoint=43000 env.add_short_history=True env.short_history_length=25 headless=False 

"""
##### zsy 更改了观测值
"""
python legged_gym/scripts/play_hydra.py --config-name=config_teleop_g1 task=g1:teleop env.num_observations=1656 env.num_privileged_obs=1733 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel_no_RootPos motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=1 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=10.0 load_run=G1_only_refposdiff_and_refvel checkpoint=20000 env.add_short_history=True env.short_history_length=25 headless=False motion.motion_file=resources/motions/g1/ACCAD_g1_stand1.pkl

"""

