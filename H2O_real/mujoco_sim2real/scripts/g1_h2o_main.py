#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import sys
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.utils.joystick import Joystick

from utils.logger import SimpleLogger
from utils.visualize import *
# from utils.utils import MovingAverageFilter, LowPassFilter, FixFreq
from sim2real.example.modules.actor_critic import example_policy
# --------------- for motion load----------------------------------
from packages.phc.utils.motion_lib_g1 import MotionLibG1 
from packages.SMPLSim.smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from packages.phc.utils import torch_utils
import os
from H2O_Sim2Sim.mujoco_sim2real.H2O.helpers import class_to_dict
from my_utils.sim2sim_class import Sim2simCfg, ElasticBand, LoadMotions
# LEGGED_GYM_ROOT_DIR = "/home/zhushiyu/文档/H2O/human2humanoid-main/legged_gym/"
H2O_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
print(f"{H2O_ROOT_DIR}")


G1_NUM_MOTOR = 29
class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13              # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13                 # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14             # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14                 # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20         # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21           # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27        # NOTE: INVALID for g1 23dof
    RightWristYaw = 28          # NOTE: INVALID for g1 23dof

class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


class G1Real:
    def __init__(self) -> None:
        """
        建立pc、g1遥控器和g1本体的通信，更新底层数据，通常不需要修改
        核心函数关注recieve_state()
        """
        self.control_dt_ = 0.02  # 50 Hz

        self.mode_pr_ = Mode.PR
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.mode_machine_ = 0  ###
        self.update_mode_machine_ = False  ### useful
        self.crc = CRC()
        self.enable_motor = True

        ### 29 dof control
        # 待修改
        self.Kp = [200, 150, 150, 200, 20, 20,       200, 150, 150, 200, 20, 20,     200, 200, 200,      20, 20, 20, 20, 20, 5, 5,       20, 20, 20, 20, 20, 5, 5]
        self.Kd = [5, 5, 5, 5, 2, 2,     5, 5, 5, 5, 2, 2,   5, 5, 5,        0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2,      0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2,]

        # #### 29 dof 
        self.default_dof_pos = np.array([-0.1,  0.0,    0.0,    0.3,   -0.2,  0.0,   \
                                         -0.1,  0.0,    0.0,    0.3,   -0.2,  0.0,    \
                                         0.0,   0.0,    0.0,    \
                                         0.0,   0.0,    0.0,   1.2,    0.0,    0.0,    0.0,   \
                                         0.0,   0.0,  0.0,    1.2,    0.0,    0.0,    0.0,])
        self.target_dof_pos = np.zeros(G1_NUM_MOTOR)

        self.init_communication()

        # wireless remote
        self.joystick = Joystick()


    def init_communication(self):
        """
        通常不需要修改
        Initializes the communication for the G1-Sim2Real module.

        This method performs the following steps:
        1. Closes the motion controller.
        2. Initializes the motion switcher client.
        3. Checks the mode of the motion switcher client and releases the mode if it is already set.
        4. Creates a publisher for the "rt/lowcmd" channel.
        5. Initializes the publisher.
        6. Creates a subscriber for the "rt/lowstate" channel.
        7. Initializes the subscriber with the LowStateHandler callback function.

        Note: The LowStateHandler callback function is responsible for handling the received low state messages.

        Returns:
            None
        """
        # close motion controller 
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscriber #
        self.__lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.__lowstate_subscriber.Init(self.LowStateHandler, 10)

    def LowStateHandler(self, msg: LowState_):
        """
        通常不需要修改
        Handles the low state message received at 500 Hz.
        Args:
            msg (LowState_): The low state message.
        Returns:
            None
        """
        self.low_state = msg

        if not self.update_mode_machine_:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True

    def send_motor_cmd(self, target_q=np.zeros(G1_NUM_MOTOR), target_dq=np.zeros(G1_NUM_MOTOR), target_trq=np.zeros(G1_NUM_MOTOR)):
        """
        通常不需要修改
        Sends motor commands to the G1 robot.
        Args:
            target_q (numpy.ndarray, optional): Target joint positions. Defaults to np.zeros(G1_NUM_MOTOR).
            target_dq (numpy.ndarray, optional): Target joint velocities. Defaults to np.zeros(G1_NUM_MOTOR).
            target_trq (numpy.ndarray, optional): Target joint torques. Defaults to np.zeros(G1_NUM_MOTOR).
        """
        if self.enable_motor and self.update_mode_machine_:
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode = 1  # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = target_q[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = self.Kp[i]
                self.low_cmd.motor_cmd[i].kd = self.Kd[i]
                # self.low_cmd.motor_cmd[i].tau = target_trq[i]
                # self.low_cmd.motor_cmd[i].q = target_q[i]
                # self.low_cmd.motor_cmd[i].dq = target_dq[i]
                # self.low_cmd.motor_cmd[i].kp = self.Kp[i]
                # self.low_cmd.motor_cmd[i].kd = self.Kd[i]
        else:
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode = 0  # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = 0
                self.low_cmd.motor_cmd[i].q = 0
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = 0
                self.low_cmd.motor_cmd[i].kd = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


    def receive_state(self):
        """
            self.low_state  500Hz
            机器人的 low state 500Hz 刷新， 但我们调用不需要这么高频
            IMU,  Enocder,  Joystick
        """

        ### Base IMU 
        q = self.low_state.imu_state.quaternion  # wxyz
        self.base_quat = np.array([q[1], q[2], q[3], q[0]])  # turn to xyzw
        self.base_rpy = np.array(self.low_state.imu_state.rpy)
        self.base_rpy_aligned = np.array([self.base_rpy[0],self.base_rpy[1],0])
        self.rot_mat_aligned = R.from_euler('xyz', self.base_rpy_aligned).as_matrix()
        self.base_gyro = np.array(self.low_state.imu_state.gyroscope)  # rad/s
        self.base_acc = np.array(self.low_state.imu_state.accelerometer)  # m/s^2

        ### Motor Encoder
        self.dof_pos = np.array([self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)])
        self.dof_vel = np.array([self.low_state.motor_state[i].dq for i in range(G1_NUM_MOTOR)])
        self.dof_trq = np.array([self.low_state.motor_state[i].tau_est for i in range(G1_NUM_MOTOR)])

        ### wireless remote
        self.joystick.extract(self.low_state.wireless_remote)

        ### calc  -----------
        self.rot_mat_aligned_world2base = np.linalg.inv(self.rot_mat_aligned)

        # self.base_ang_vel_body = self.rot_mat_aligned_world2base @ self.base_gyro
        self.base_ang_vel_body = self.base_gyro.copy()  ###  TODO

        self.projected_gravity = self.rot_mat_aligned_world2base @ np.array([0, 0, -1.0])




    def set_default_posture(self, tar_dof_pos=np.zeros(G1_NUM_MOTOR)):
        """
        初始化g1的站立姿势，通常不需要修改
        Sets the default posture of the G1 robot to the specified target degree of freedom positions.
        Parameters:
        tar_dof_potional): The target degree of freedom positions. Defaults to an array of zeros with length G1_NUM_MOTOR.
        Returns:
        None
        """
        self.receive_state()
        duration = 1.5
        control_dt = self.control_dt_
        cur_dof_pos = self.dof_pos
        for i in range(int(duration / control_dt)):
            ratio = np.clip(i * control_dt / duration, 0.0, 1.0)
            dof_pos = cur_dof_pos * (1.0 - ratio) + tar_dof_pos * ratio
            self.send_motor_cmd(target_q=dof_pos)
            time.sleep(control_dt)

    
    def step(self, target_dof_pos):
        self.target_dof_pos = target_dof_pos
        self.send_motor_cmd(target_q=self.target_dof_pos)


class Controller:
    def __init__(self) -> None:
        """
        负责控制策略的部署，包括建立网络导入权重、初始化机器人数据通信链等，大部分函数可以根据自己的需求修改
        """

        #### ----------  config  -----------------
        self.dt = 0.02  ### 50Hz
        self.counter = 0

        # ====== <start 自定义需要的变量> ======
        self.num_obs = 92
        self.num_critic = 282
        self.num_history = 5
        self.num_actions = 27

        self.last_actions = np.zeros(self.num_actions)
        self.obs = np.zeros(self.num_obs)
        self.obs_history = np.zeros(self.num_obs*self.num_history)
        self.commands = np.zeros(3)
        self.cmd_sample = np.zeros(3)

        self.lin_vel_est = np.zeros(3)
        self.base_height_est = np.zeros(1)

        self.phase = 0.0
        self.phase_period = 0.7
        self.phase_period_count = self.phase_period / self.dt
        self.phase_counter = 0

        self.model_path = 'model/model_250.pt'
        # ====== <end 自定义需要的变量> ======

        self.init_robot()
        self.load_policy()

        self.logger: SimpleLogger = SimpleLogger()
        self.last_time = time.time()

        #### task
        self.ref_motion_cache = {}
        self.motion_ids = 0
        self.episode_length_buf = 0
        self._load_motion()

        #### track_ids
        self.teleop_selected_keypoints_names = []
        self._body_list = class_to_dict(G1JointIndex)
        self._track_bodies_id = [self._body_list.index(body_name) for body_name in self.teleop_selected_keypoints_names] # 读取特征点关节点的id  e.g.[0,4,5, 6,7,8]
        self._track_bodies_extend_id = self._track_bodies_id + [19,20,21] # 头和手



    def init_robot(self):
        self.robot = G1Real()

    def load_policy(self):
        model_dict = torch.load(self.model_path, map_location='cpu')
        self.policy = example_policy
        self.policy.load_state_dict(model_dict['model_state_dict'])
        self.policy.eval()
        self.policy_inference = self.policy.act_inference

    def log(self):

        self.logger.log('target_dof_pos',   self.robot.target_dof_pos)
        self.logger.log('dof_pos',          self.robot.dof_pos)  
        self.logger.log('dof_vel',          self.robot.dof_vel)  
        self.logger.log('dof_trq',          self.robot.dof_trq)

        self.logger.log('command_x',        self.commands[0])  
        self.logger.log('command_y',        self.commands[1])  
        self.logger.log('command_yaw',      self.commands[2]) 

        self.logger.log('base_lin_vel',       self.lin_vel_est)  

        self.logger.log('base_vel_roll',    self.robot.base_ang_vel_body[0])  
        self.logger.log('base_vel_pitch',   self.robot.base_ang_vel_body[1])  
        self.logger.log('base_vel_yaw',     self.robot.base_ang_vel_body[2])  

        self.logger.log('roll',             self.robot.base_rpy[0])  
        self.logger.log('pitch',            self.robot.base_rpy[1])  
        self.logger.log('yaw',              self.robot.base_rpy[2])  

    def update_commands(self):
        """
        An example for reading commands from joytick of g1
        """
        vel_x_max = 0.8
        vel_y_max = 0.0
        vel_yaw_max= 0.5
        
        vel_x = self.robot.joystick.ly.data * vel_x_max
        vel_y = self.robot.joystick.lx.data * vel_y_max
        vel_yaw = self.robot.joystick.rx.data * vel_yaw_max
    
        self.cmd_sample = np.array([vel_x, vel_y, vel_yaw])

        self.commands[0] = np.where( (self.commands[0] - self.cmd_sample[0])> 0.01,
                                        np.clip(self.commands[0]-0.02, a_min=self.cmd_sample[0], a_max=None),
                                        np.clip(self.commands[0]+0.02, a_min=None, a_max=self.cmd_sample[0]))

        self.commands[1] = np.where( (self.commands[1] - self.cmd_sample[1])> 0.01,
                                        np.clip(self.commands[1]-0.02, a_min=self.cmd_sample[1], a_max=None),
                                        np.clip(self.commands[1]+0.02, a_min=None, a_max=self.cmd_sample[1]))
        
        self.commands[2] = np.where( (self.commands[2] - self.cmd_sample[2])> 0.01,
                                        np.clip(self.commands[2]-0.02, a_min=self.cmd_sample[2], a_max=None),
                                        np.clip(self.commands[2]+0.02, a_min=None, a_max=self.cmd_sample[2]))

    def get_obs(self):
        rl_default_dof_pos = np.concatenate((self.robot.default_dof_pos[:13], self.robot.default_dof_pos[15:])) ### 27 dof
        rl_dof_pos = np.concatenate((self.robot.dof_pos[:13], self.robot.dof_pos[15:]))
        rl_dof_vel = np.concatenate((self.robot.dof_vel[:13], self.robot.dof_vel[15:]))

        self.phase_counter +=1
        self.phase = self.phase_counter % self.phase_period_count / self.phase_period_count
        sin_phase = np.sin(2*np.pi * self.phase)
        cos_phase = np.cos(2*np.pi * self.phase)

        self.obs = np.hstack(
            (   self.robot.base_ang_vel_body * 0.25, ### 3
                self.robot.projected_gravity,  ### 3
                self.commands, ### 3
                (rl_dof_pos- rl_default_dof_pos),  
                rl_dof_vel * 0.05,
                self.last_actions,
                sin_phase,
                cos_phase,
            )
        )

        obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device='cpu', requires_grad=False).reshape((1, self.num_obs)).squeeze(0)

        self.obs_history = np.roll(self.obs_history, self.num_obs)
        self.obs_history[:self.num_obs] = self.obs  #63

        obs_history_tensor = torch.tensor(self.obs_history, dtype=torch.float32, device='cpu', requires_grad=False).reshape((1, self.num_obs*self.num_history)).squeeze(0)

        return obs_tensor, obs_history_tensor


    def apply_action(self, action):
        rl_default_dof_pos = np.concatenate((self.robot.default_dof_pos[:13], self.robot.default_dof_pos[15:])) ### 27 dof
        action_scaled = action * 0.5 + rl_default_dof_pos  ### 27dof

        tar_dof_pos = np.concatenate((action_scaled[:13], np.array([0.0, 0.0]), action_scaled[13:])) ### 29 dof

        return tar_dof_pos


    def run(self):
        while True:
            # NOTE: for safety
            if self.robot.joystick.B.pressed:
                self.robot.enable_motor = False
                break

            # self.cmd_sample = np.array([0.5,0.0,0.0])
            ### command命令更新
            self.update_commands()

            while (time.time()-self.last_time) < self.dt:
                time.sleep(0.001)
            self.last_time = time.time()
            
            self.robot.receive_state()
            obs_tensor,  obs_history_tensor= self.get_obs()
            obs_tensor = torch.clip(obs_tensor, -30.0, 30.0)
            action_tensor, base_lin_vel_est = self.policy_inference(obs_tensor, obs_history_tensor)
            action_tensor = torch.clip(action_tensor, -5.0, 5.0)
            action = action_tensor.detach().cpu().numpy()  ## dim = 27
            # action = np.zeros(27)   
            tar_dof_pos = self.apply_action(action) 
            self.robot.step(tar_dof_pos)
            self.episode_length_buf += 1
            self.last_actions = action
            self.lin_vel_est = base_lin_vel_est.detach().cpu().numpy() 
            self.log()



########################################################################################################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    controller = Controller()
    ### 设置机器人的默认姿态
    controller.robot.set_default_posture(controller.robot.default_dof_pos)
    while True:
        ### 保持默认序列
        controller.robot.send_motor_cmd(target_q=controller.robot.default_dof_pos)
        time.sleep(0.02)
        
        ### start press A
        ### 获取状态
        controller.robot.receive_state()
        if controller.robot.joystick.A.pressed:
            break

    controller.last_time = time.time()
    ### 主要循环
    controller.run()

    print("saving data...")
    nowtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not Path('./data/').exists():
        Path('./data/').mkdir()
    np.savez(f"./data/{nowtime}.npz", **controller.logger)
    print("over!")

    # visualize_helper(controller.logger)
