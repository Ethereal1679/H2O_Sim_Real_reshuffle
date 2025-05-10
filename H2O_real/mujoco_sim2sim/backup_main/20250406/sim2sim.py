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
MUJOCO_TREE_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
print(f"-----------------{MUJOCO_TREE_ROOT_DIR}")
from H2O_Sim2Sim.mujoco_sim2sim.scripts.logger import Logger
from sim2sim_class import Sim2simCfg
from sim2sim_class import ElasticBand
from sim2sim_class import LoadMotions




### for issac [x,y,z,w]
### for mujoco [w,x,y,z]
## quat * Vec
def quat_rotate_inverse(q, v):
    q, v = torch.from_numpy(q), torch.from_numpy(v)
    q_w = q[-1] 
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v) * q_w * 2.0
    c = q_vec * torch.mm(q_vec.view(1, 3), v.view(3, 1)) * 2.0
    return a - b + c

def quaternion_rotate_z(q, theta):
    """
    绕Z轴旋转四元数
    :param q: 输入四元数，格式为 (w, x, y, z)
    :param theta: 旋转角度，单位为度
    :return: 旋转后的四元数
    """
    # 将角度转换为弧度
    theta_rad = math.radians(theta)
    
    # 计算旋转四元数 q_z
    cos_half_theta = math.cos(theta_rad / 2)
    sin_half_theta = math.sin(theta_rad / 2)
    q_z = np.array([cos_half_theta, 0, 0, sin_half_theta])
    
    # 四元数乘法
    w1, x1, y1, z1 = q_z
    w2, x2, y2, z2 = q
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([x, y, z, w])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp  +  (target_dq - dq) * kd


def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    # print(data.qpos)
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    
    if 0:
        quat = q[3:7] #这个不知道根据什么得来的参考，参考的不对，不可以用刚体的
        omega = dq[3:6] 
    else: 
        quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double) # x y z w
        omega = data.sensor('angular-velocity').data.astype(np.double)
    r = R.from_quat(quat)   
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)

    return_q  = np.copy(q[7:26])
    return_dq = np.copy(dq[6:25])


    return (return_q, return_dq, quat, v, omega, gvec)


def log_obs(obs, width=80,pad=2):
    log_string = (
        f"""{'-' * width}\n"""
        f"""{'【q】:':>{pad}} {obs[0        :     19]}\n"""                  
        f"""{'【dq】:':>{pad}} {obs[19       :     19*2]}\n"""              
        f"""{'【omega】:':>{pad}} {obs[19*2     :     19*2+3]}\n"""           
        f"""{'【gvec】:':>{pad}} {obs[19*2+3   :     19*2+6]}\n"""            
        f"""{'【action】:':>{pad}} {obs[19*2+6   :     19*3+6]}\n"""             
        f"""{'【task_obs】:':>{pad}} {obs[19*3+6   :     19*3+24]}\n"""     
    )      
    print(log_string)


def run_mujoco(policy, cfg:Sim2simCfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    ####### ============= [1] mujoco =============
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt # 200HZ 
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    #### ============= [2] LoadMotions =============
    load_motions = LoadMotions()
    load_motions.reset_root_states(data) # init robot pos and rot through motions' init pos and rot
    load_motions._resample_motion_times(0) 
    #### ============= [3] init buffers =============
    target_q = np.zeros((cfg.env.num_actions), dtype=np.float32)
    target_q = cfg.robot_config.default_angles.copy()
    target_q_test = cfg.robot_config.default_angles_test.copy()
    action = np.zeros((cfg.env.num_actions), dtype=np.float32)
    last_action = np.zeros_like(action,dtype=np.float32)
    target_dq = np.zeros_like(target_q, dtype=np.float32)
    gravity_orientation = np.array([0.0,  0.0,  -1.0],dtype=np.float32)
    obs = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
    # init history obs
    trajectories = np.zeros((63 * 100),dtype=np.float32) 
    # init counters
    counter = 0
    motion_step_counter = 0 # for motion step
    #### lifting robot to test ，not used
    elastic_band =  ElasticBand()
    elastic_band.enable = True
    band_attached_link = model.body("pelvis").id
    #### logger
    logger = Logger(0.02)




    # loop
    with mujoco.viewer.launch_passive(model, data) as viewer:  
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        simulation_duration = 120 #s
        #####  ------------------------------------------------- loop 200HZ ------------------------------------------------------------------
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()   
            # ------------------------- 50HZ --------------------------------------
            if counter % cfg.sim_config.decimation == 0:

                # Obtain an observation 50HZ
                ### policy input || because trained-policy input is torch.tensor
                #### ================================= base obs =================================================================
                # quat = quaternion_rotate_z(quat,-90) 
                # print("quat",quat) 
                #### task obs 
                q, dq, quat, v, omega_1, gvec = get_obs(data) # quat : x,y,z,w
                omega = quat_rotate_inverse(quat, omega_1)
                
                task_obs ,target_actions = load_motions.compute_self_and_task_obs(motion_step_counter, quat, data)
                # _ = load_motions.mimic_dof_joint(data, motion_step_counter) / cfg.normalization.action_scale

                # print("task_obs",task_obs)  
                #### 计算重力投影  
                # projected_gravity = quat_rotate_inverse(torch.from_numpy(quat).unsqueeze(0), torch.from_numpy(gravity_orientation).unsqueeze(0)).squeeze().numpy()
                # print("omega",omega)  
                # projected_gravity = quaternion_to_euler_array(quat)
                # eu_ang = quaternion_to_euler_array(quat)
                # eu_ang[eu_ang > math.pi] -= 2 * math.pi

                # action_scaled = np.zeros_like(action) 
                # action_scaled[:10] = action[:10] * cfg.normalization.action_scale 
                # print("action_scaled",action_scaled)    
                #### total obs   
                obs = np.zeros([1, cfg.env.num_observations], dtype=np.float32)

                obs[0, 0        :     19]               = np.copy(q)             #19
                obs[0, 19       :     19*2]             = np.copy(dq)            #19
                obs[0, 19*2     :     19*2+3]           = np.copy(omega )        #3    
                obs[0, 19*2+3   :     19*2+6]           = np.copy(gvec )         #3
                obs[0, 19*2+6   :     19*3+6]           = np.copy(target_actions )       #19 
                obs[0, 19*3+6   :     19*3+24]          = np.copy(task_obs )     #18 
                obs[0, 19*3+24  :     19*3+24+63*25]    = np.copy(trajectories[0 : cfg.env.frame_stack * cfg.env.num_single_obs]) 
                # obs = np.concatenate((q, dq, omega, gvec, action, task_obs, trajectories[0 : cfg.env.frame_stack * cfg.env.num_single_obs]), axis=0)
                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                ### history update  
                # hist_obs.append(obs[0, 0:63])   
                # hist_obs.popleft()  
                # print(gvec) 
                ### ============================ action =================================
                # print(policy(torch.from_numpy(obs)).shape)
                
                action_tensor = policy(torch.from_numpy(obs).detach())
                action = action_tensor.detach().cpu().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                # action = (0.6 * action)+(1-0.6)*last_action
                # last_action = action.copy()
                # print(action)  
                # action = np.array([0,0,0,0,0,  0,0,0,0,0,   0,   0,0,0,-10,   0,0,0,-10,],dtype=np.double)   
                ### torques   
                
                target_q = action * cfg.normalization.action_scale + cfg.robot_config.default_angles  
                ### ===== UPDATE self.trajectories =====  
                # log_obs_buff = obs[0, 0:81]
                # log_obs(log_obs_buff)
                
                current_obs_a = np.copy(obs[0, 0:63])
                trajectories[1 * 63 :] = trajectories[: -1 * 63]
                trajectories[0 * 63 : 1 * 63] = np.copy(current_obs_a)
                motion_step_counter += 1   
                # print(motion_step_counter)


                # 重采样动作 reset
                if motion_step_counter * 0.02 >= load_motions.motion_len:
                    motion_step_counter = 0
                    # reset states
                    load_motions.reset_root_states(data)
                    load_motions._resample_motion_times(motion_step_counter)
                    # init_buffers  
                    target_q = cfg.robot_config.default_angles.copy()
                    action = np.zeros((cfg.env.num_actions), dtype=np.double)
                    target_dq = np.zeros_like(target_q, dtype=np.double)
                    obs = np.zeros([1, cfg.env.num_observations], dtype=np.double)
                    trajectories = np.zeros((63 * 100),dtype=np.double) 
                # time.sleep(1)


            #### PD controler 200HZ
            tau = pd_control(target_q, data.qpos[7:26], cfg.robot_config.kps,  \
                            target_dq, data.qvel[6:25], cfg.robot_config.kds) 
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl = tau      
            
            ### mujoco steps     
            # if elastic_band.enable:
            #     data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(data.qpos[:3], data.qvel[:3])
            mujoco.mj_step(model, data) #仿真步进一步

            counter += 1
            ### 更新场景
                # time.sleep(1)
            viewer.sync()
            ### time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # print(time_until_next_step)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

if __name__ == '__main__':

    load_model = f'{MUJOCO_TREE_ROOT_DIR}/models/STUDENT/25_03_22_17-38-57_g1_change_obs_num_2model_3000.pt'
    policy = torch.jit.load(load_model)  
    cfg = Sim2simCfg()
    cfg.sim_config.mujoco_model_path = f'{MUJOCO_TREE_ROOT_DIR}/unitree_robots/g1_19dof/xml/g1_scene.xml'
    run_mujoco(policy, cfg)


