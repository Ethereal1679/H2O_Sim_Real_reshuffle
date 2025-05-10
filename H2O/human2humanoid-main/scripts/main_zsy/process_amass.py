import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES, 
)
import joblib
from phc.utils.rotation_conversions import axis_angle_to_matrix
from phc.utils.torch_g1_humanoid_batch import Humanoid_Batch
from torch.autograd import Variable
from tqdm import tqdm
import argparse

import os
# added by zsy
MY_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# print(os.path.realpath(__file__))
print("=====MY_ROOT_DIR:",MY_ROOT_DIR)


# 从.npz导入
def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return 
    framerate = entry_data['mocap_framerate']


    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }


def process_amass(path_shape,  first_dir,  second_dir,  third_dir,  dump_dir=None,  dump_2_legged_gym_resources_dir=None):
    parser = argparse.ArgumentParser()
    # AMASS数据集的路径，我这里是绝对路径
    parser.add_argument("--amass_root", type=str, default=f"{MY_ROOT_DIR}/data/AMASS/AMASS_Complete") # 修改AMASS的路径,这里只用了CMU一个数据集
    args = parser.parse_args()
    
    device = torch.device("cpu")

    g1_rotation_axis = torch.tensor([[
        [0, 1, 0], # l_hip_pitch
        [1, 0, 0], # l_hip_roll
        [0, 0, 1], # l_hip_yaw
        
        [0, 1, 0], # kneel
        [0, 1, 0], # ankle

        [0, 1, 0], # r_hip_pitch
        [1, 0, 0], # r_hip_roll
        [0, 0, 1], # r_hip_yaw
        
        [0, 1, 0], # kneel
        [0, 1, 0], # ankle
        
        [0, 0, 1], # torso
        
        [0, 1, 0], # l_shoulder_pitch
        [1, 0, 0], # l_roll_pitch
        [0, 0, 1], # l_yaw_pitch
        
        [0, 1, 0], # l_elbow
        
        [0, 1, 0], # r_shoulder_pitch
        [1, 0, 0], # r_roll_pitch
        [0, 0, 1], # r_yaw_pitch
        
        [0, 1, 0], # r_elbow
    ]]).to(device)

    g1_joint_names = [ 
                    'pelvis', 
                    'left_hip_yaw_link', 'left_hip_roll_link','left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
                    'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
                    'torso_link', 
                    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 
                    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link']

    g1_joint_names_augment = g1_joint_names + ["left_hand_link", "right_hand_link"]
    g1_joint_pick = ['pelvis', "left_knee_link", "left_ankle_link",  'right_knee_link', 'right_ankle_link', "left_shoulder_roll_link", "left_elbow_link", "left_hand_link", "right_shoulder_roll_link", "right_elbow_link", "right_hand_link",]
    smpl_joint_pick = ["Pelvis",  "L_Knee", "L_Ankle",  "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand"]
    g1_joint_pick_idx = [ g1_joint_names_augment.index(j) for j in g1_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]


    smpl_parser_n = SMPL_Parser(model_path=f"{MY_ROOT_DIR}/data/smpl/",gender="male") # gender可以更改
    smpl_parser_n.to(device)


    # 加载smpl
    shape_path = path_shape
    shape_new, scale = joblib.load(f"{shape_path}")
    shape_new = shape_new.to(device)


    amass_root = args.amass_root

    # 这里我修改了all_pkls路径,只包含了CMU一个路径的数据集：
    first_dir = first_dir 
    second_dir = second_dir
    third_dir = third_dir

    all_pkls = glob.glob(f"{amass_root}/{first_dir}/{second_dir}/{third_dir}.npz", recursive=True) # 递归搜索所有的npz
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}
    #print("++++++++",key_name_to_pkls) #'0-CMU_122_122_45_poses': '/home/zhushiyu/文档/H2O/human2humanoid-main/data/AMASS/AMASS_Complete/CMU/122/122_45_poses.npz'
    if len(key_name_to_pkls) == 0:
        raise ValueError(f"No motion files found in {amass_root}")





    g1_fk = Humanoid_Batch(device = device)
    data_dump = {}
    pbar = tqdm(key_name_to_pkls.keys())
    # pbar总共？？？
    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        skip = int(amass_data['fps']//30)
        trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(device)
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(np.concatenate((amass_data['pose_aa'][::skip, :66], np.zeros((N, 6))), axis = -1)).float().to(device)
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset
        pose_aa_g1 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 22, axis = 2), N, axis = 1)
        pose_aa_g1[..., 0, :] = (sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()
        pose_aa_g1 = torch.from_numpy(pose_aa_g1).float().to(device)
        gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)
        dof_pos = torch.zeros((1, N, 19, 1)).to(device)
        # 梯度更新的这个东西，也就是loss反传这个值：
        dof_pos_new = Variable(dof_pos, requires_grad=True)
        optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=100)

        print(f"<<<Motion name>>> : {data_key}")
        
        # 循环500轮 
        for iteration in range(500):
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            pose_aa_g1_new = torch.cat([gt_root_rot[None, :, None], g1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2).to(device)
            fk_return = g1_fk.fk_batch(pose_aa_g1_new, root_trans_offset[None, ])
            # 计算损失函数
            diff = fk_return['global_translation_extend'][:, :, g1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            loss_g = diff.norm(dim = -1).mean() 
            loss = loss_g
            pbar.set_description_str(f"轮数：{iteration}   损失函数的值：{loss.item() * 1000}")
            # loss反向传播
            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_pose.step()
            dof_pos_new.data.clamp_(g1_fk.joints_range[:, 0, None], g1_fk.joints_range[:, 1, None])
        
        dof_pos_new.data.clamp_(g1_fk.joints_range[:, 0, None], g1_fk.joints_range[:, 1, None])
        pose_aa_g1_new = torch.cat([gt_root_rot[None, :, None], g1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2)
        fk_return = g1_fk.fk_batch(pose_aa_g1_new, root_trans_offset[None, ])
        root_trans_offset_dump = root_trans_offset.clone()
        root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08
        
        # data_key是AMASS数据集每一种motion的名字 e.g.: 0-CMU_13_13_16_poses
        data_dump[data_key]={
                "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
                "pose_aa": pose_aa_g1_new.squeeze().cpu().detach().numpy(),   
                "dof": dof_pos_new.squeeze().detach().cpu().numpy(), 
                "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
                "fps": 30
                }
        # 这个量用来手动选合理动作用，你不用管 --zsy
        FILTER = True
        if FILTER:
            #print(f"dumping {data_key} for testing, remove the line if you want to process all data")
            # 断点调试所用
            # print("NOTE:按n键继续一步生成pkl文件，按c键继续生成下一个motion，按q键退出")
            # import ipdb; ipdb.set_trace()
            joblib.dump(data_dump, "./data_amass/g1/ACCAD_g1_stand1.pkl") # name 可以自行更改
        else:
            robot = "g1"
            pkl_name = f"{robot}_{first_dir}_{second_dir}_{third_dir}"
            joblib.dump(data_dump, f"./data_filter_motion_cache/{robot}/{first_dir}/{second_dir}/{pkl_name}.pkl")

    if FILTER:
        #import ipdb; ipdb.set_trace()
        joblib.dump(data_dump, dump_dir )
        print("[NOTE]已经保存到了：  ", dump_dir)
        joblib.dump(data_dump, dump_2_legged_gym_resources_dir )
        print("[NOTE]已经保存到了：  ", dump_2_legged_gym_resources_dir)
    else:
        pass







# ######################################################################################
# ======================== main =====================================
# ######################################################################################

if __name__ == "__main__":
    # 文件路径："data_smpl/g1/shape_optimized_g1_neutral.pkl" /"ACCAD" /* /* 
    robot = "g1"
    path_shape = f"data_smpl/{robot}/shape_optimized_g1_neutral.pkl"
    first_dir  = "ACCAD"                             #指的是 ACCAD BM... CMU...那一个目录
    second_dir = "Male2General_c3d"       #"Male2MartialArtsStances_c3d"
    third_dir  = "A1- Stand_poses"         #"D1 - stand to ready_poses"
    # 保存到当下的路径data_amass路径下：
    pkl_name = f"{first_dir}_{robot}_{second_dir}_{third_dir}"
    dump_dir   = f"data_amass/{robot}/{pkl_name}"
    # 保存到legged——gym的resources路径下：
    dump_2_legged_gym_resources_dir = f"{MY_ROOT_DIR}/legged_gym/resources/motions/g1/g1_demo/{pkl_name}.pkl"
    process_amass(path_shape,  first_dir,  second_dir,  third_dir  ,dump_dir  ,dump_2_legged_gym_resources_dir)




