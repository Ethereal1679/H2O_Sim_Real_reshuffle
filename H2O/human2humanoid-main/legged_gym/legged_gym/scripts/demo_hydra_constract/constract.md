# 用来对比作者的demo样例
## 可以忽视
## 1
`train student 25 history`
python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=h1:teleop run_name=OmniH2O_STUDENT env.num_observations=1665 env.num_privileged_obs=1742 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=True rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=True env.short_history_length=25 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=TEACHER_RUN_NAME train.dagger.checkpoint_dagger=XXX train.dagger.dagger_only=True

`train student 0 history` 
python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=h1:teleop run_name=OmniH2O_STUDENT_0stepMLP env.num_observations=90 env.num_privileged_obs=167 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=True rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=False noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=TEACHER_RUN_NAME train.dagger.checkpoint_dagger=XXX train.dagger.dagger_only=True

`train student 5 history` 
python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=h1:teleop run_name=OmniH2O_STUDENT_50stepMLP env.num_observations=405 env.num_privileged_obs=482 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=True rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=True env.short_history_length=5 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=TEACHER_RUN_NAME train.dagger.checkpoint_dagger=XXX train.dagger.dagger_only=True

## 2

`play student  history` 
python legged_gym/scripts/play_hydra.py --config-name=config_teleop task=h1:teleop env.num_observations=1665 env.num_privileged_obs=1742 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=1 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=10.0 sim_device=cuda:0 load_run=OmniH2O_STUDENT checkpoint=XXXX env.add_short_history=True env.short_history_length=25 headless=False 


`play student 0 history` 
python legged_gym/scripts/play_hydra.py --config-name=config_teleop task=h1:teleop env.num_observations=90 env.num_privileged_obs=167 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=1 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=10.0 sim_device=cuda:0 load_run=OmniH2O_STUDENT checkpoint=XXXX env.add_short_history=False headless=False 

`train student LSTM history` 

python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=h1:teleop run_name=LSTM_STUDENT env.num_observations=90 env.num_privileged_obs=167 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=False rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=False env.short_history_length=0 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=TEACHER_RUN_NAME train.dagger.checkpoint_dagger=55500 train.dagger.dagger_only=True train.runner.policy_class_name=ActorCriticRecurrent train.policy.rnn_type=lstm


`train student GRU history` 
python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=h1:teleop run_name=LSTM_STUDENT env.num_observations=90 env.num_privileged_obs=167 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=False rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=False env.short_history_length=0 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=TEACHER_RUN_NAME train.dagger.checkpoint_dagger=55500 train.dagger.dagger_only=True train.runner.policy_class_name=ActorCriticRecurrent train.policy.rnn_type=gru

`train student 8 tracks` 

python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=h1:teleop run_name=OmniH2O_STUDENT_8point env.num_observations=1719 env.num_privileged_obs=1796 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[left_ankle_link,right_ankle_link,left_shoulder_pitch_link,right_shoulder_pitch_link,left_elbow_link,right_elbow_link] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=True rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=True env.short_history_length=25 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=TEACHER_RUN_NAME train.dagger.checkpoint_dagger=XXX train.dagger.dagger_only=True

`train student 23 tracks` 
full  全身的

python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=h1:teleop run_name=OmniH2O_STUDENT_23point env.num_observations=1845 env.num_privileged_obs=1922 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[pelvis,left_hip_yaw_link,left_hip_roll_link,left_hip_pitch_link,left_knee_link,left_ankle_link,right_hip_yaw_link,right_hip_roll_link,right_hip_pitch_link,right_knee_link,right_ankle_link,torso_link,left_shoulder_pitch_link,left_shoulder_roll_link,left_shoulder_yaw_link,left_elbow_link,right_shoulder_pitch_link,right_shoulder_roll_link,right_shoulder_yaw_link,right_elbow_link] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:1 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=True rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=True env.short_history_length=25 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=TEACHER_RUN_NAME train.dagger.checkpoint_dagger=XXX train.dagger.dagger_only=True


`train student with v` 
python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=h1:teleop run_name=OmniH2O_STUDENT env.num_observations=1743 env.num_privileged_obs=1820 motion.teleop_obs_version=v-teleop-extend-vr-max motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=True rewards.penalty_scale=0.5 train.distill=True train.policy.init_noise_std=0.001 env.add_short_history=True env.short_history_length=25 noise.add_noise=False noise.noise_level=0 train.dagger.load_run_dagger=TEACHER_RUN_NAME train.dagger.checkpoint_dagger=XXX train.dagger.dagger_only=True


`train student without DAgger` 
python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=h1:teleop run_name=OmniH2O_wo_DAgger env.num_observations=1665 env.num_privileged_obs=1742 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=True rewards.penalty_scale=0.5 noise.add_noise=False noise.noise_level=0 env.add_short_history=True env.short_history_length=25


`train H2O Policy (8point tracking, no history, MLP, with linear velocity in the state space)` 

python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=h1:teleop run_name=H2O_Policy env.num_observations=138 env.num_privileged_obs=215 motion.teleop_obs_version=v-teleop-extend-max motion.teleop_selected_keypoints_names=[left_ankle_link,right_ankle_link,left_shoulder_pitch_link,right_shoulder_pitch_link,left_elbow_link,right_elbow_link] motion.extend_head=False num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=True rewards.penalty_scale=0.5 env.add_short_history=False









`train student 0 history` 







`train student 0 history` 







`train student 0 history` 