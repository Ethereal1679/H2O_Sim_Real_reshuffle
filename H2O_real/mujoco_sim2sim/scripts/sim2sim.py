

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
MUJOCO_TREE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(f"<<<<<<<<<<{MUJOCO_TREE_ROOT_DIR}>>>>>>>>>>>")
from logger import Logger
from sim2sim_class import Sim2simCfg
from sim2sim_class import ElasticBand
from sim2sim_class import LoadMotions



def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

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
    theta_rad = math.radians(theta)
    cos_half_theta = math.cos(theta_rad / 2)
    sin_half_theta = math.sin(theta_rad / 2)
    q_z = np.array([cos_half_theta, 0, 0, sin_half_theta])

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
    q = data.qpos.astype(np.float32)
    dq = data.qvel.astype(np.float32)
    if 0:quat = q[3:7] #一定是0  
    else: quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.float32) # x y z w 
    if 0:omega = dq[3:6] 
    else:omega = data.sensor('angular-velocity').data.astype(np.float32)
    r = R.from_quat(quat)   
    v = r.apply(data.qvel[:3], inverse=True).astype(np.float32)  # In the base frame
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.float32)

    return_q  = np.copy(q[7:26])
    return_dq = np.copy(dq[6:25])
    virtual_xyz_pos = q[:3].copy()

    return return_q, return_dq, quat, v, omega, gvec, virtual_xyz_pos


def log_obs(obs, width=80,pad=2):
    log_string = (
        f"""{'-' * width}\n"""
        f"""{'【q】:':>{pad}} {obs[0        :     19]}\n"""                  
        f"""{'【dq】:':>{pad}} {obs[19       :     19*2]}\n"""              
        f"""{'【omega】:':>{pad}} {obs[19*2     :     19*2+3]}\n"""           
        f"""{'【gvec】:':>{pad}} {obs[19*2+3   :     19*2+6]}\n"""            
        f"""{'【task_obs】:':>{pad}} {obs[19*2+6   :     19*2+24]}\n"""             
        f"""{'【action】:':>{pad}} {obs[19*2+24   :     19*3+24]}\n"""     
    )      
    print(log_string)


def run_mujoco(policy,motion_path,cfg:Sim2simCfg):
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
    load_motions = LoadMotions(motion_path)
    # load_motions.reset_root_states(data) # init robot pos and rot through motions' init pos and rot
    load_motions._resample_motion_times(0) 
    #### ============= [3] init buffers =============
    
    target_q            = np.zeros((cfg.env.num_actions),           dtype=np.float32)
    target_q            = cfg.robot_config.default_angles.copy()
    # target_q_test       = cfg.robot_config.default_angles_test.copy()
    action              = np.zeros((cfg.env.num_actions),           dtype=np.float32)
    last_action         = np.zeros_like(action,                     dtype=np.float32)
    target_dq           = np.zeros_like(target_q,                   dtype=np.float32)
    gravity_orientation = np.array([0.0,  0.0,  -1.0],              dtype=np.float32)
    obs                 = np.zeros((cfg.env.num_observations),      dtype=np.float32)
    trajectories        = np.zeros((63 * 100),                      dtype=np.float32) 
    trajectories_buff   = np.zeros((63 * 100),                      dtype=np.float32) 
    # init counters
    counter = 0
    motion_step_counter = 0 # for motion step
    #### lifting robot to test ，not used
    elastic_band =  ElasticBand()
    elastic_band.enable = True
    band_attached_link = model.body("pelvis").id
    #### NOTE: logger
    logger = Logger(0.02)
    stop_state_log = 2000
    simulation_duration = 60 #s
    LOG =True
    # action_buff = np.zeros_like(action,  dtype=np.float32)


    # loop
    with mujoco.viewer.launch_passive(model, data) as viewer:  
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        #####  ------------------------------------------------- loop 200HZ ------------------------------------------------------------------
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()   
            # ------------------------- 50HZ --------------------------------------
            if counter % cfg.sim_config.decimation == 0: 
                step_start_decimation = time.time() 
                # Obtain an observation 50HZ    
                ### policy input || because trained-policy input is torch.tensor    
                #### ================================= base obs =================================================================
                # quat = quaternion_rotate_z(quat,-90) 
                # print("quat",quat)  
                #### task obs 
                q, dq, quat, v, omega, gvec, virtual_xyz_pos = get_obs(data) # quat : x,y,z,w
                # projected_gravity = quat_rotate_inverse(quat, gvec).numpy()
                # projected_gravity = get_gravity_orientation(quat)
                # omega = quat_rotate_inverse(quat, omega_1)
                task_obs ,target_actions = load_motions.compute_self_and_task_obs(motion_step_counter, quat, data, virtual_xyz_pos)
                # _ = load_motions.mimic_dof_joint(data, motion_step_counter) / cfg.normalization.action_scale
                # print("task_obs",task_obs)  
                #### 计算重力投影  
                # print("omega",omega)  
                # projected_gravity = quaternion_to_euler_array(quat)
                # eu_ang = quaternion_to_euler_array(quat)
                # eu_ang[eu_ang > math.pi] -= 2 * math.pi

                # action_scaled = np.zeros_like(action) 
                # action_scaled[:10] = action[:10] * cfg.normalization.action_scale 
                # print("action_scaled",action_scaled)    
                #### total obs   
                # obs = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                # obs = np.concatenate((q,  dq,  omega,  gvec,  target_actions,  task_obs,  trajectories[0 : cfg.env.frame_stack * cfg.env.num_single_obs]))
                obs[0        :     19]               = np.copy(q)             #19
                obs[19       :     19*2]             = np.copy(dq)            #19
                obs[19*2     :     19*2+3]           = np.copy(omega )        #3    
                obs[19*2+3   :     19*2+6]           = np.copy(gvec )         #3  
                obs[19*2+6   :     19*2+24]          = np.copy(task_obs )     #18
                obs[19*2+24   :    19*3+24]          = np.copy(action)        #19
                obs[19*3+24  :     19*3+24+63*25]    = np.copy(trajectories_buff[0 : cfg.env.frame_stack * cfg.env.num_single_obs]) 
                # obs = np.concatenate((q, dq, omega, gvec, task_obs.dtype(torch.float32), action.dtype(torch.float32), trajectories_buff[0 : cfg.env.frame_stack * cfg.env.num_single_obs].astype(torch.float32)))
                obs             = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                obs_tensor      = torch.from_numpy(obs).unsqueeze(0)
                ### history update  
                # hist_obs.append(obs[0, 0:63])   
                # hist_obs.popleft()  
                # print(gvec) 
                ### ========= history ========
                # NOTE: This bug has consumed me for 1.5 months!!!!!!!
                current_obs = np.concatenate((q, dq, omega, gvec, action)) ## @@ right!!
                # current_obs_a = obs[0:63] # @@ wrong!!
                trajectories[1 * 63 :] = trajectories[: -1 * 63]
                trajectories[0 * 63 : 1 * 63] = current_obs
                trajectories_buff = trajectories.copy()
                
                # print(motion_step_counter)
                ### ============================ action =================================
                # print(policy(torch.from_numpy(obs)).shape)
                # import ipdb;ipdb.set_trace()
                action = policy(obs_tensor.detach()).detach().numpy().squeeze()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                # action_buff[9:] = action[:10].copy()
                # action_buff[:8] = action[11:].copy()
                # action = (0.6 * action)+(1-0.6)*last_action
                # last_action = action.copy()
                # print(action)  
                # action = np.array([0,0,0,0,0,  0,0,0,0,0,   0,   0,0,0,-10,   0,0,0,-10,],dtype=np.double)   
                ### torques   
                
                target_q = action * cfg.normalization.action_scale + cfg.robot_config.default_angles  
                ### ===== UPDATE self.trajectories =====  
                if 0:
                    log_obs_buff = obs[0:81]
                    log_obs(log_obs_buff)


                motion_step_counter += 1
                if motion_step_counter * 0.02 >= load_motions.motion_len :
                    motion_step_counter = 0
                    # reset states  
                    # load_motions.reset_root_states(data)
                    load_motions._resample_motion_times(motion_step_counter)
                    # init_buffers  
                    # target_q            = cfg.robot_config.default_angles.copy()
                    action              = np.zeros((cfg.env.num_actions),       dtype=np.float32)
                    target_dq           = np.zeros_like(target_q,               dtype=np.float32)
                    obs                 = np.zeros((cfg.env.num_observations),  dtype=np.float32)
                    trajectories        = np.zeros((63 * 100),                  dtype=np.float32) 

                # time
                # step_stop_decimation = time.time()
                # policy_time = cfg.sim_config.dt * cfg.sim_config.decimation
                # delta = policy_time - (step_stop_decimation - step_start_decimation)
                # print(policy_time,"<><>",(step_stop_decimation - step_start_decimation))
                # if delta > 0:
                #     time.sleep(delta)
                # time.sleep(0.5)

                # log params
                
                Error_actions = np.mean((action - target_actions)**2)
                R_alive = (motion_step_counter * 0.02) / load_motions.motion_len * 100
                print("Rate_alive: {}%".format(R_alive.item()))
                # ### <><><> logs <><><>
                if LOG: 
                    if counter < stop_state_log: 
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
                                'Error_actions': Error_actions,
                            }
                        )
                    elif counter==stop_state_log:
                        logger.plot_states()
        # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
            
            #### PD controler 200HZ
            tau = pd_control(target_q, data.qpos[7:26], cfg.robot_config.kps,  \
                            target_dq, data.qvel[6:25], cfg.robot_config.kds) 
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl[:] = tau      
            
            ### mujoco steps     
            # if elastic_band.enable:
            #     data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(data.qpos[:3], data.qvel[:3])
            mujoco.mj_step(model, data) 
            counter += 1
                # time.sleep(1)
            viewer.sync()
            ### time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # print(time_until_next_step)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
带验证测试数据
ACCAD_g1_Male2MartialArtsPunches_c3d_E1Jableft_poses
ACCAD_g1_Male2MartialArtsPunches_c3d_E7uppercutleft_poses
ACCAD_g1_Male2MartialArtsPunches_c3d_E13body_crossleft_poses
BMLhandball_g1_S02_Novice_Trial_upper_left_140_poses
BMLmovi_g1_Subject_5_F_MoSh_Subject_5_F_2_poses
DanceDB_g1_20120807_CliodelaVara_Clio_Flamenco_C3D_poses
EKUT_g1_125_SLP304_poses
HumanEva_g1_S1_Box_1_poses
HumanEva_g1_S1_Gestures_1_poses
"""


if __name__ == '__main__':
    motion_path = f'{MUJOCO_TREE_ROOT_DIR}/motions/HumanEva_g1_S1_Gestures_1_poses.pkl' 
    load_model = f'{MUJOCO_TREE_ROOT_DIR}/models/STUDENT/25_04_06_18-19-05_g1_add_ref_vel_and_ref_delta_posmodel_20000.pt'
    policy = torch.jit.load(load_model)  
    cfg = Sim2simCfg()
    cfg.sim_config.mujoco_model_path = f'{MUJOCO_TREE_ROOT_DIR}/unitree_robots/g1_19dof/xml/g1_scene.xml'
    run_mujoco(policy, motion_path, cfg)


