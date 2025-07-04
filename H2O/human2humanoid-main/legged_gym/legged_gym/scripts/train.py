import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import os
from datetime import datetime
import sys
sys.path.append("/home/wenli-run/unitree_rl_gym")

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, helpers
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    log_dir = ppo_runner.log_dir
    env_cfg_dict = helpers.class_to_dict(env_cfg)
    train_cfg_dict = helpers.class_to_dict(train_cfg)
    # Save cfgs
    os.makedirs(log_dir, exist_ok=True)
    import json
    with open(os.path.join(log_dir, 'env_cfg.json'), 'w') as f:
        json.dump(env_cfg_dict, f, indent=4)
    with open(os.path.join(log_dir, 'train_cfg.json'), 'w') as f:
        json.dump(train_cfg_dict, f, indent=4)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)

if __name__ == '__main__':
    args = get_args()

    args.num_envs = 2048
    args.task = "h1"
    # headless or not
    args.headless = True 
    args.max_iterations = 5000 + 1
    args.experiment_name = "H1"
    args.run_name = "test1"

    args.resume = False
    if args.resume:
        args.load_run = "Jan24_13-41-46_重新调整参数"

    train(args)
