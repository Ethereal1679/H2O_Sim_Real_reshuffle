
LEGGED_GYM_ROOT_DIR='/home/zhushiyu/文档/GIT/H2O_git_1/h2o/H2O/human2humanoid-main/legged_gym'
LEGGED_GYM_ENVS_DIR='/home/zhushiyu/文档/GIT/H2O_git_1/h2o/H2O/human2humanoid-main/legged_gym/legged_gym/envs'

# user need to change to delect T or S
NAME="T" # "S"

echo "Training $NAME"

# ======================= train Teacher =======================
if [ $NAME == "T" ]; then
    python train_hydra.py \
        --config-name='config_teleop_g1' \
        task='g1:teleop' \
        max_iterations=100000 \
        run_name='g1_ACCAD_TEACHER' \
        num_envs=4096 \
        headless='True' \
        env.num_observations=913 \
        env.num_privileged_obs=990 \
        motion.teleop_obs_version='v-teleop-extend-max-full'\
        motion='motion_full_g1' \
        asset.termination_scales.max_ref_motion_distance=1.5 
        # resume='False' \ 
        # load_run='25_02_26_19-39-11_g1_ACCAD_10000_TEACHER' \
        # checkpoint=20000 \


elif [ $NAME == "S" ]; then
# ======================= train Student =======================
    ### NOTE: 63*25+90  25steps

    # python train_hydra.py \
    #     --config-name='config_teleop_g1' \
    #     task='g1:teleop '\
    #     run_name='g1_STUDENT_Test_zsy' \
    #     env.num_observations=1665 \
    #     env.num_privileged_obs=1742 \
    #     motion.teleop_obs_version='v-teleop-extend-vr-max-nolinvel' \
    #     motion.teleop_selected_keypoints_names=[] \
    #     motion.extend_head='True' \
    #     num_envs=2048 \
    #     asset.termination_scales.max_ref_motion_distance=1.5 \
    #     motion.motion_file='resources/motions/g1/ACCAD_g1_amass_all.pkl '\
    #     rewards.penalty_curriculum='True' \
    #     rewards.penalty_scale=0.5 \
    #     train.distill='True' \
    #     train.policy.init_noise_std=0.001 \
    #     env.add_short_history='True' \
    #     env.short_history_length=25 \
    #     noise.add_noise='False' \
    #     noise.noise_level=0 \
    #     train.dagger.load_run_dagger='ACCAD_g1_Teacher_41000 '\
    #     train.dagger.checkpoint_dagger=41000 \
    #     train.dagger.dagger_only='True'\


    ### change student obs from global -> local
    ### NOTE: 63*25+81 25 steps
    python train_hydra.py \
        --config-name='config_teleop_g1' \
        task='g1:teleop'\
        run_name='g1_rot_local_test' \
        num_envs=4096 \
        max_iterations=20000 \
        env.num_observations=1656 \
        env.num_privileged_obs=1733 \
        motion.teleop_obs_version='v-teleop-extend-vr-max-nolinvel-local' \
        motion.teleop_selected_keypoints_names=[] \
        motion.extend_head='True' \
        asset.termination_scales.max_ref_motion_distance=1.5 \
        rewards.penalty_curriculum='True' \
        rewards.penalty_scale=0.5 \
        train.distill='True' \
        train.policy.init_noise_std=0.001 \
        env.add_short_history='True' \
        env.short_history_length=25 \
        noise.add_noise='False' \
        noise.noise_level=0 \
        train.dagger.load_run_dagger='ACCAD_g1_Teacher_41000' \
        train.dagger.checkpoint_dagger=41000 \
        train.dagger.dagger_only='True' 
else
    echo "Please choose from T or S"
fi