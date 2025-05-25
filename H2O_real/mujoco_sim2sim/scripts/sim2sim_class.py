# logs 原作者没提供sim2sim，这里需要自己写一下

### motions
from phc.utils.motion_lib_g1 import MotionLibG1 
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils import torch_utils
### base
import math
import numpy as np
import mujoco
import mujoco.viewer
import time
from collections import deque
from scipy.spatial.transform import Rotation as R
import os
import torch
# LEGGED_GYM_ROOT_DIR = "/home/zhushiyu/文档/H2O/human2humanoid-main/legged_gym/"
MUJOCO_TREE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(f"<<<<<<<<<<{MUJOCO_TREE_ROOT_DIR}>>>>>>>>>>>")
from logger import Logger



class Sim2simCfg():
    class sim_config:
        mujoco_model_path = f'{MUJOCO_TREE_ROOT_DIR}/unitree_robots/g1_19dof/xml/g1_scene.xml'
        sim_duration = 60.0
        dt = 0.005                #200HZ 这个原来是0.02，改为0.005后脚不抖了？？？？
        decimation = 4     #10:20HZ      #4:50HZ 
        sim_dt = dt * decimation
    class robot_config:
        ## NOTE：realG1 params
        # kps = np.array([200, 150, 150, 200, 20,        200, 150, 150, 200, 20,      200,      20,20,20,20,      20,20,20,20], dtype=np.double)
        # kds = np.array([5, 5, 5 , 5, 2,        5, 5, 5, 5, 2,       5,        0.5, 0.5, 0.5, 0.5,     0.5, 0.5, 0.5, 0.5], dtype=np.double)
        ## NOTE: isaac params
        kps = np.array([100, 100, 100, 200, 20,     100, 100, 100, 200, 20,       400,      90,60,20,60,      90,60,20,60], dtype=np.double)
        kds = np.array([2.5, 2.5, 2.5, 5.0, 0.2,      2.5, 2.5, 2.5, 5.0, 0.2,        5.0,        2.0, 1.0, 0.4, 1.0,     2.0, 1.0, 0.4, 1.0], dtype=np.double)
        tau_limit = 100. # * np.ones(19, dtype=np.double)    
        default_angles = np.array([-0.1,  0.0,  0.0,  0.3, -0.2,  
                                    -0.1,  0.0,  0.0,  0.3, -0.2, 
                                    0.0,
                                    0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0,]) 
        
        # default_angles = np.array([-0.,  0.0,  0.0,  0., -0.,  
        #                     -0.,  0.0,  0.0,  0., -0., 
        #                     0.0,
        #                     0.0, 0.0, 0.0, 0.0,
        #                     0.0, 0.0, 0.0, 0.0,]) 
    class env:
        num_actions = 19
        num_single_obs = 63
        frame_stack = 25
        num_observations = 81 + num_single_obs * frame_stack
    class normalization:
        clip_actions = 100
        clip_observations = 100.
        action_scale = 0.25



class ElasticBand: 
    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable





class LoadMotions:
    def __init__(self, motion_path):
        #### task
        self.ref_motion_cache = {}
        self.motion_ids = torch.arange(1)
        self.motion_path = motion_path
        # self.episode_length_buf = 0
        self._load_motion()
        #### track_ids
        self.teleop_selected_keypoints_names = []
        self._body_list = []
        self._track_bodies_id = [self._body_list.index(body_name) for body_name in self.teleop_selected_keypoints_names] # 读取特征点关节点的id  e.g.[0,4,5, 6,7,8]
        self._track_bodies_extend_id = self._track_bodies_id + [20,21,22] # 头和手
        self.motion_start_times = torch.zeros(1, dtype=torch.float32)
        self.motion_len         = torch.zeros(1, dtype=torch.float32)
        self.offset = None ###torch.zeros(1, 3, dtype=torch.float32)
        self.motion_push_dt = 0.005 * 4
        
        #ref_pos init

    # ----------- load motions from AMASS ------------------------
    def _load_motion(self):
        # urdf文件安   
        skeleton_path = f'{MUJOCO_TREE_ROOT_DIR}/unitree_robots/g1_19dof/xml/g1_phc.xml'
        # motion解析
        self._motion_lib = MotionLibG1(motion_file=self.motion_path, device="cpu", masterfoot_conifg=None, fix_height=False,  multi_thread=False,  mjcf_file=skeleton_path, extend_head=True) #multi_thread=True doesn't work
        # skeleton解析
        sk_tree = SkeletonTree.from_mjcf(skeleton_path)
        num_envs = 1
        skeleton_trees = [sk_tree] * num_envs
        self._motion_lib.load_motions(skeleton_trees=skeleton_trees, gender_betas=[torch.zeros(17)] * num_envs, limb_weights=[np.zeros(10)] * num_envs, random_sample=False)
        # self.motion_dt = self._motion_lib._motion_dt

    ## NOTE：
    # def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
    #     motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=self.offset)
    #     return motion_res


    ## 
    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        # import ipdb; ipdb.set_trace()
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            # import ipdb; ipdb.set_trace()
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)
        # import ipdb; ipdb.set_trace()
        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache

    ## NOTE: not used
    # def compute_imitation_observations_teleop_max_local(self, ref_root_pos, ref_body_pos, ref_body_vel, root_rot, time_steps):
    #     obs = []
    #     B = 1
    #     J = len(self._track_bodies_extend_id)

    #     local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - ref_root_pos.view(B, 1, 1, 3)  # preserves the body position
    #     # ================
    #     obs.append(local_ref_body_pos.view(B, time_steps, -1))  # 1 * timestep * J * 3
    #     obs.append(ref_body_vel.view(B, time_steps, -1))  # timestep  * J * 3
    #     obs = torch.cat(obs, dim=-1).view(B, -1)
    #     return obs 


    def compute_imitation_observations_teleop_max_local(self, ref_root_pos, ref_body_pos, ref_body_vel, root_rot, time_steps):
        obs = []
        B = 1
        J = len(self._track_bodies_extend_id)
        # B, J, _ = body_pos.shape  #B为环境数 ，J为track_link_ids的数目 + head  or hands   
        # ================
        heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)   #0
        # heading_inv_ref_rot = torch_utils.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0) # 1 3 4 
        # heading_inv_ref_rot_expand = heading_inv_ref_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0) # 1 3 4 
        # ================ 
        local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - ref_root_pos.view(B, 1, 1, 3)  # preserves the body position
        local_ref_body_pos_obs = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))
        local_ref_body_vel_obs = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_vel.view(-1, 3))
        # ================
        obs.append(local_ref_body_pos_obs.view(B, time_steps, -1))  # 1 * timestep * J * 3
        obs.append(local_ref_body_vel_obs.view(B, time_steps, -1))  # timestep  * J * 3
        obs = torch.cat(obs, dim=-1).view(B, -1)
        return obs 


    # =================== 计算初始的观测值 ===================
    def compute_self_and_task_obs(self, motion_step_counter, quat, data, virtual_xyz_pos):
        """ Computes observations
        """
        # 偏置
        if 1:
            self.motion_start_times = 0
        else: 
            self.motion_start_times = self._motion_lib.sample_time(self.motion_ids)   
        motion_times = torch.tensor((motion_step_counter + 1) * self.motion_push_dt  +  self.motion_start_times) # next frames so +1
        motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=self.offset)
        ##### ref motion properties  
        ref_body_pos                = motion_res["rg_pos"].clone()        # [num_envs, num_markers, 3] 
        ref_body_pos_extend         = motion_res["rg_pos_t"].clone()         # 1 23 3
        ref_body_vel_subset         = motion_res["body_vel"].clone()         # [num_envs, num_markers, 3]
        ref_body_vel                = ref_body_vel_subset.clone() 
        ref_body_vel_extend         = motion_res["body_vel_t"].clone()       # [num_envs, num_markers, 3]
        ref_body_rot                = motion_res["rb_rot"].clone()           # [num_envs, num_markers, 4]
        ref_body_rot_extend         = motion_res["rg_rot_t"].clone()         # [num_envs, num_markers, 4]
        ref_body_ang_vel            = motion_res["body_ang_vel"].clone()     # [num_envs, num_markers, 3]
        ref_body_ang_vel_extend     = motion_res["body_ang_vel_t"].clone()   # [num_envs, num_markers, 3]
        ref_joint_pos               = motion_res["dof_pos"].clone()          # [num_envs, num_dofs]
        ref_joint_vel               = motion_res["dof_vel"].clone()          # [num_envs, num_dofs]
        ref_root_pos                = ref_body_pos_extend[:,0,:].clone()
        ref_root_rot                = ref_body_rot_extend[:,0,:].clone()
        # import ipdb; ipdb.set_trace()
        root_rot = (torch.from_numpy(quat).unsqueeze(0).to(torch.double)).clone()  
        # print(root_rot)
        ref_rb_pos_subset   = (ref_body_pos_extend[:, self._track_bodies_extend_id]).clone() #133
        ref_body_vel_subset = (ref_body_vel_extend[:, self._track_bodies_extend_id]).clone() 
        
        time_steps = 1
        ### obs 
        task_obs_buff = self.compute_imitation_observations_teleop_max_local(ref_root_pos, ref_rb_pos_subset, ref_body_vel_subset, root_rot, time_steps)
        task_obs = task_obs_buff.clone()
        # print("task_obs",task_obs) 
        # 绘制ref_pos的位置 
        ## ============= 用于task pos的可视化 ================
        # a = ref_root_pos 
        # x,y,z = a[0,:]   
        x,y,z = 0,0,0.793
        # x,y,z = virtual_xyz_pos[0], virtual_xyz_pos[1], virtual_xyz_pos[2]
        data.qpos[-9:-6] = [task_obs[0,0] + x,task_obs[0,1] + y,task_obs[0,2] + z] #hand
        data.qpos[-6:-3] = [task_obs[0,3] + x,task_obs[0,4] + y,task_obs[0,5] + z] #hand
        data.qpos[-3:  ] = [task_obs[0,6] + x,task_obs[0,7] + y,task_obs[0,8] + z] #head
        # data.qpos[-9:-6] = [ref_rb_pos_subset[0,0,0],ref_rb_pos_subset[0,0,1],ref_rb_pos_subset[0,0,2]]
        # data.qpos[-6:-3] = [ref_rb_pos_subset[0,1,0],ref_rb_pos_subset[0,1,1],ref_rb_pos_subset[0,1,2]]
        # data.qpos[-3:  ] = [ref_rb_pos_subset[0,2,0],ref_rb_pos_subset[0,2,1],ref_rb_pos_subset[0,2,2]]
        robot = Sim2simCfg()
        r_action = ref_joint_pos - robot.robot_config.default_angles

        return task_obs_buff.squeeze() , r_action.squeeze().numpy() / 0.025


    def quaternion_transform(self, q1, q2):

        # 计算 q1 的逆四元数（共轭四元数）
        q1_inv = torch.cat((-q1[..., :3], q1[..., -1:]), dim=-1)

        # 计算变换四元数：q_transform = q2 * q1_inv
        q_transform = self.quaternion_multiply(q2, q1_inv)

        return q_transform

    def quaternion_multiply(self, q1, q2):
        """
        四元数乘法。

        参数:
            q1 (torch.Tensor): 第一个四元数，形状为 (..., 4)。
            q2 (torch.Tensor): 第二个四元数，形状为 (..., 4)。

        返回:
            torch.Tensor: 四元数乘法的结果，形状与输入相同。
        """
        x1, y1, z1, w1 = torch.unbind(q1, dim=-1)
        x2, y2, z2, w2 = torch.unbind(q2, dim=-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack((x, y, z, w), dim=-1)


    def _resample_motion_times(self, motion_step_counter):

        # self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
        # self.motion_ids[env_ids] = torch.randint(0, self._motion_lib._num_unique_motions, (len(env_ids),), device=self.device)
        # print(self.motion_ids[:10])
        self.motion_len = self._motion_lib.get_motion_length(self.motion_ids)
        # print(self.motion_len)
        # self.env_origins_init_3Doffset[env_ids, :2] = torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        if 1:
            self.motion_start_times = 0
        else: 
            self.motion_start_times = self._motion_lib.sample_time(self.motion_ids)   
        # self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        motion_times = torch.tensor((motion_step_counter + 1) * self.motion_push_dt + self.motion_start_times) # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
        motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset = self.offset)

    def reset_root_states(self, data):
        motion_step_counter = 0
        if 1:
            self.motion_start_times = 0
        else: 
            self.motion_start_times = self._motion_lib.sample_time(self.motion_ids)   
        motion_times = torch.tensor((motion_step_counter + 1) * self.motion_push_dt  +  self.motion_start_times) # next frames so +1
        motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=self.offset)

        data.qpos[:3] = motion_res['root_pos'].clone()
        data.qpos[3:7] = self.utils_xyzw_to_wxyz(motion_res['root_rot'].squeeze().numpy())
        data.qpos[7:26] = motion_res["dof_pos"].clone() 
        data.qvel[:3] = motion_res['root_vel'].clone()
        data.qvel[3:6] = motion_res['root_ang_vel'].clone()
        data.qvel[7:26] = motion_res["dof_vel"].clone()


    def utils_xyzw_to_wxyz(self, quat):
        return_quat = np.zeros(4)
        return_quat[0],return_quat[1],return_quat[2],return_quat[3] = quat[3],quat[0],quat[1],quat[2]
        return return_quat

    def mimic_dof_joint(self, data, motion_step_counter):
        if 1:
            self.motion_start_times = 0
        else: 
            self.motion_start_times = self._motion_lib.sample_time(self.motion_ids)   
        motion_times = torch.tensor((motion_step_counter + 1) * self.motion_push_dt  +  self.motion_start_times) # next frames so +1
        motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=self.offset)

        # data.qpos[:3] = motion_res['root_pos']
        # data.qpos[3:7] = self.utils_xyzw_to_wxyz(motion_res['root_rot'].squeeze().numpy())
        # data.qpos[7:26] = motion_res["dof_pos"] 
        # data.qvel[:3] = motion_res['root_vel']
        # data.qvel[3:6] = motion_res['root_ang_vel']
        # data.qvel[7:26] = motion_res["dof_vel"]

        robot = Sim2simCfg()
        return motion_res["dof_pos"] - robot.robot_config.default_angles




