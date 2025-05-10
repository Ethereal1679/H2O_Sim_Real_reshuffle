# logs zsy
# 1.为什么observations总是加77或者76: 【已解决 见注释 的 913个观测值 990个特权观测值 】
# 2.为什么有那么多obs版本？？【已解决 见博客 】
# 3.解放腰部的是waist_yaw_link还是torso_link??
# 4.
#
#
import glob
import os
import sys
import pdb
import os.path as osp

import wandb.util
sys.path.append(os.getcwd())

from isaacgym import gymapi
import numpy as np
import os
from datetime import datetime
import sys
# sys.path.append("/home/wenli-run/unitree_rl_gym")

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, helpers
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
import wandb

@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config",
)

def train(cfg_hydra: DictConfig) -> None:
    cfg_hydra = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))   
    cfg_hydra.physics_engine = gymapi.SIM_PHYSX  
    # --------- 参数部署，免去控制台参数的设置 zsy--------- 
#  motion=motion_full_g1 num_envs=4096 asset.termination_scales.max_ref_motion_distance=1.5 motion.motion_file=resources/motions/g1/ACCAD_g1_amass_all.pkl
    # TEACHER = True
    # # -----【1】教师网络训练 包含913个观测值，990个特权观测值-----
    # if TEACHER:  
    #     cfg_hydra.config_name= "config_teleop_g1"   
    #     cfg_hydra.task = "g1:teleop"
    #     cfg_hydra.run_name = "g1_ACCAD_10000_TEACHER"
    #     cfg_hydra.env.num_observations = 913        # 913是教师网络的priv观测值 
    #     cfg_hydra.env.num_privileged_obs = 990      # 913 + 77 ？？？？？
    #     cfg_hydra.motion.teleop_obs_version = "v-teleop-extend-max-full"
    #     #cfg_hydra.motion = "motion_full" 
    #     cfg_hydra.motion.extend_head = True
    #     cfg_hydra.num_envs = 4096
    #     cfg_hydra.asset.zero_out_far = True
    #     cfg_hydra.asset.termination_scales.max_ref_motion_distance = 1.5
    #     cfg_hydra.sim_device = "cuda:0"
    #     cfg_hydra.motion.motion_file = "resources/motions/g1/ACCAD_g1_amass_all.pkl"        # motion模型路径
    #     # cfg_hydra.rewards = "rewards_teleop_omnih2o_teacher"
    #     cfg_hydra.rewards.penalty_curriculum = True
    #     cfg_hydra.rewards.penalty_scale = 0.5
    # else:     
    #     # python legged_gym/scripts/train_hydra.py 
    #     # --config-name=config_teleop 
    #     # -----【2】distilled的学生网络 -----
    #     cfg_hydra.task = "g1:teleop"
    #     cfg_hydra.run_name = "g1_test_STUDENT"
    #     cfg_hydra.env.num_observations = 1665       # 90 + 63 * 25
    #     cfg_hydra.env.num_privileged_obs = 1742
    #     cfg_hydra.motion.teleop_obs_version = "v-teleop-extend-vr-max-nolinvel"
    #     cfg_hydra.motion.teleop_selected_keypoints_names=[]  
    #     cfg_hydra.motion.extend_head = True   
    #     # motion=motion_full                    
    #     cfg_hydra.num_envs = 4096
    #     cfg_hydra.asset.zero_out_far = False
    #     cfg_hydra.asset.termination_scales.max_ref_motion_distance = 1.5
    #     cfg_hydra.sim_device = "cuda:0"
    #     cfg_hydra.motion.motion_file = "resources/motions/g1/ACCAD_g1_stand1.pkl"
    #     # cfg_hydra.rewards = "rewards_teleop_omnih2o_teacher"
    #     cfg_hydra.rewards.penalty_curriculum = True
    #     cfg_hydra.rewards.penalty_scale = 0.5

    #     cfg_hydra.train.distill=True
    #     cfg_hydra.train.policy.init_noise_std=0.001
    #     cfg_hydra.env.add_short_history=True
    #     cfg_hydra.env.short_history_len= 25 # history数量
    #     cfg_hydra.noise.add_noise=False
    #     cfg_hydra.noise.noise_level=0
    #     # 导入专家策略 load_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", cfg.task, cfg.train.dagger.load_run_dagger, f"model_{cfg.train.dagger.checkpoint_dagger}.pt")
    #     cfg_hydra.train.dagger.load_run_dagger= "25_02_24_20-06-01_g1_stand10000_TEACHER"  
    #     cfg_hydra.train.dagger.checkpoint_dagger=10000                      # check_point的数值 f"model_{cfg.train.dagger.checkpoint_dagger}.pt
    #     cfg_hydra.train.dagger.dagger_only=True

    # ---------------- 注册 ------------------
    env, env_cfg = task_registry.make_env_hydra(name=cfg_hydra.task, hydra_cfg=cfg_hydra, env_cfg=cfg_hydra)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cfg_hydra.task, args=cfg_hydra, train_cfg=cfg_hydra.train)


    log_dir = ppo_runner.log_dir
    env_cfg_dict = helpers.class_to_dict(env_cfg)
    train_cfg_dict = helpers.class_to_dict(train_cfg)
    del env_cfg_dict['physics_engine']
    # Save cfgs
    # os.makedirs(log_dir, exist_ok=True)
    # import json
    # with open(os.path.join(log_dir, 'env_cfg.json'), 'w') as f:
    #     json.dump(env_cfg_dict, f, indent=4)
    # with open(os.path.join(log_dir, 'train_cfg.json'), 'w') as f:
    #     json.dump(train_cfg_dict, f, indent=4)
    # if cfg_hydra.use_wandb:
    #     run_id = wandb.util.generate_id()
    #     run = wandb.init(name=cfg_hydra.task, config=cfg_hydra, id=run_id, dir=log_dir, sync_tensorboard=True)
    #     wandb.run.name = cfg_hydra.run_name
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)




if __name__ == '__main__':

    train()

# ======================= train 教师网络的训练控制台参数 =======================
# 教师网络默认使用motion full进行 训练
# g1

"""
python legged_gym/scripts/train_hydra.py --config-name=config_teleop_g1 task=g1:teleop run_name=g1_ACCAD_resume_30000_TEACHER max_iterations=20000 env.num_observations=913 env.num_privileged_obs=990 motion.teleop_obs_version=v-teleop-extend-max-full motion=motion_full_g1 num_envs=4096 asset.termination_scales.max_ref_motion_distance=1.5 motion.motion_file=resources/motions/g1/ACCAD_g1_amass_all.pkl

"""

"""
python legged_gym/scripts/train_hydra.py --config-name=config_teleop_g1 task=g1:teleop run_name=g1_ACCAD_resume_20000_TEACHER max_iterations=2000 env.num_observations=913 env.num_privileged_obs=990 motion.teleop_obs_version=v-teleop-extend-max-full motion=motion_full_g1 num_envs=4096 asset.termination_scales.max_ref_motion_distance=1.5 motion.motion_file=resources/motions/g1/ACCAD_g1_amass_all.pkl headless=True resume=True load_run=25_02_26_19-39-11_g1_ACCAD_10000_TEACHER checkpoint=20000

"""




# ======================= train 学生网络的训练控制台参数 =======================


### 63*25+90  25steps
"""
python legged_gym/scripts/train_hydra.py --config-name=config_teleop_g1 task=g1:teleop run_name=g1_delta_local_tr env.num_observations=1665 env.num_privileged_obs=1742 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=2048 asset.termination_scales.max_ref_motion_distance=1.5 motion.motion_file=resources/motions/g1/ACCAD_g1_amass_all.pkl rewards.penalty_curriculum=True rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=True env.short_history_length=25 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=ACCAD_g1_Teacher_41000 train.dagger.checkpoint_dagger=41000 train.dagger.dagger_only=True

"""
### zsy add 改变了学生网络的观测值
"""
python train_hydra.py --config-name=config_teleop_g1 task=g1:teleop run_name=g1_rot_local_test env.num_observations=1656 env.num_privileged_obs=1733 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel-local motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=4096 asset.termination_scales.max_ref_motion_distance=1.5 rewards.penalty_curriculum=True rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=True env.short_history_length=25 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=ACCAD_g1_Teacher_41000 train.dagger.checkpoint_dagger=41000 train.dagger.dagger_only=True max_iterations=20000 

"""




















### OmniH2O Distill LSTM Student Policy 
"""
python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=h1:teleop run_name=LSTM_STUDENT env.num_observations=90 env.num_privileged_obs=167 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=False rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=False env.short_history_length=0 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=TEACHER_RUN_NAME train.dagger.checkpoint_dagger=55500 train.dagger.dagger_only=True train.runner.policy_class_name=ActorCriticRecurrent train.policy.rnn_type=lstm

"""

### 22 key_points 23links

"""
python legged_gym/scripts/train_hydra.py --config-name=config_teleop_g1 task=g1:teleop run_name=OmniH2O_STUDENT_22point env.num_observations=1845 env.num_privileged_obs=1922 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[pelvis,left_hip_yaw_link,left_hip_roll_link,left_hip_pitch_link,left_knee_link,left_ankle_pitch_link,right_hip_yaw_link,right_hip_roll_link,right_hip_pitch_link,right_knee_link,right_ankle_pitch_link,waist_yaw_link,left_shoulder_pitch_link,left_shoulder_roll_link,left_shoulder_yaw_link,left_elbow_link,right_shoulder_pitch_link,right_shoulder_roll_link,right_shoulder_yaw_link,right_elbow_link] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 motion.motion_file=resources/motions/g1/ACCAD_g1_amass_all.pkl rewards.penalty_curriculum=True rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=True env.short_history_length=25 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=ACCAD_g1_Teacher_41000 train.dagger.checkpoint_dagger=41000 train.dagger.dagger_only=True

"""

### 8 key_points 9links
"""
python legged_gym/scripts/train_hydra.py --config-name=config_teleop_g1 task=g1:teleop run_name=OmniH2O_STUDENT_8point env.num_observations=1719 env.num_privileged_obs=1796 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[left_ankle_pitch_link,right_ankle_pitch_link,left_shoulder_pitch_link,right_shoulder_pitch_link,left_elbow_link,right_elbow_link] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 motion.motion_file=resources/motions/g1/ACCAD_g1_amass_all.pkl rewards.penalty_curriculum=True rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=True env.short_history_length=25 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=ACCAD_g1_Teacher_41000 train.dagger.checkpoint_dagger=41000 train.dagger.dagger_only=True

"""

# python train_hydra.py --config-name=config_teleop_g1 task=g1:teleop run_name=OmniH2O_STUDENT_8point env.num_observations=1719 env.num_privileged_obs=1796 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[left_ankle_pitch_link,right_ankle_pitch_link,left_shoulder_pitch_link,right_shoulder_pitch_link,left_elbow_link,right_elbow_link] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 motion.motion_file=resources/motions/g1/ACCAD_g1_amass_all.pkl rewards.penalty_curriculum=True rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=True env.short_history_length=25 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=ACCAD_g1_Teacher_41000 train.dagger.checkpoint_dagger=41000 train.dagger.dagger_only=True

