import os
import sys
import numpy as np
import pinocchio
import crocoddyl
from pinocchio.robot_wrapper import RobotWrapper

# 设置路径
# sys.path.append("/opt/openrobots/lib/python3.8/site-packages")
# current_directory = os.getcwd()
# print("当前工作目录：", current_directory)



# 设置模型路径和URDF文件名
#model_path = os.path.join(current_directory, 'resources', 'robots', 'XBot')
model_path = "./"
urdf_filename = "cowa_robot.urdf"
urdf_full_path = os.path.join(model_path, urdf_filename)

# 加载机器人模型
robot = RobotWrapper.BuildFromURDF(urdf_full_path, [model_path], pinocchio.JointModelFreeFlyer())
model = robot.model
visual_model = robot.visual_model
# 设置脚部关节名称
right_foot = 'right_ankle_roll_joint'
left_foot = 'left_ankle_roll_joint'

# 创建显示对象
display = crocoddyl.MeshcatDisplay(robot, frameNames=[right_foot, left_foot])

# 初始化位置
q0 = pinocchio.neutral(model)
print("q0: ", q0)
# print("q0.shape: ", q0.shape)  # 应该输出 (19,)
# print("q0.type: ", q0.type) 
display.display(q0)

print("---------------初始位置-----------")

# 计算初始运动学
data = model.createData()
pinocchio.forwardKinematics(model, data, q0)
pinocchio.updateFramePlacements(model, data)

# 获取脚部位置
rf_id = model.getFrameId(right_foot)
lf_id = model.getFrameId(left_foot)
rf_foot_pos0 = data.oMf[rf_id].translation
lf_foot_pos0 = data.oMf[lf_id].translation

# 计算质心位置
com_ref = pinocchio.centerOfMass(model, data, q0)

print("--------------计算质心--------------")

# 单关节运动演示
def single_joint_demo():
    for i in range(model.nq - 7):
        q = pinocchio.neutral(model)
        q[i+7] = 1
        display.display(q)
        print(f"关节 {i+7} 运动")

# 双关节同时运动演示
def double_joint_demo():
    for i in range(model.nq - 7 - 6):
        q = pinocchio.neutral(model)
        q[i+7] = 1
        q[i+13] = 1
        display.display(q)
        print(f"关节 {i+7} 和 {i+13} 同时运动")

print("--------------开始演示--------------")
single_joint_demo()
double_joint_demo()

# 行走模拟
def walking_simulation(num_steps=1000):
    for i in range(num_steps):
        phase = i * 0.005
        sin_pos = np.sin(2 * np.pi * phase)
        
        ref_dof_pos = np.zeros(12)
        scale_1, scale_2 = 0.17, 0.34
        
        # 左脚站立相
        sin_pos_l = min(0, sin_pos)
        ref_dof_pos[0] = sin_pos_l * scale_1
        ref_dof_pos[3] = -sin_pos_l * scale_2
        ref_dof_pos[5] = sin_pos_l * scale_1
        
        # 右脚站立相
        sin_pos_r = max(0, sin_pos)
        ref_dof_pos[6] = -sin_pos_r * scale_1
        ref_dof_pos[9] = sin_pos_r * scale_2
        ref_dof_pos[11] = -sin_pos_r * scale_1
        
        # 双脚支撑相
        if abs(sin_pos) < 0.1:
            ref_dof_pos[:] = 0
        
        q = pinocchio.neutral(model)
        q[7:] = ref_dof_pos
        display.display(q)

print("--------------开始行走模拟--------------")
walking_simulation()
print("-------------完成-----------------")