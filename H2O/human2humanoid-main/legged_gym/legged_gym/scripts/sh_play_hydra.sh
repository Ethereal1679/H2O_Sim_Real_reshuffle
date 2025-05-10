LEGGED_GYM_ROOT_DIR='/home/zhushiyu/文档/GIT/H2O_git_1/h2o/H2O/human2humanoid-main/legged_gym'
LEGGED_GYM_ENVS_DIR='/home/zhushiyu/文档/GIT/H2O_git_1/h2o/H2O/human2humanoid-main/legged_gym/legged_gym/envs'

# user need to change to delect T or S
NAME="S" # "S"

echo "Training $NAME"

# # ======================= play Teacher =======================
if [ $NAME == "T" ]; then
    ## NOTE:
    load_run_name="T"
    python ./play_hydra.py \
        --config-name='config_teleop_g1' \
        task='g1:teleop' \
        env.num_observations=913 \
        env.num_privileged_obs=990 \
        motion='motion_full_g1' \
        motion.future_tracks='True' \
        motion.teleop_obs_version='v-teleop-extend-max-full' \
        asset.termination_scales.max_ref_motion_distance=10.0 \
        load_run=$load_run_name\
        checkpoint=41000 \
        num_envs=1 \
        headless='False' 


elif [ $NAME == "S" ]; then
# # ======================= play Student =======================
    ### NOTE: 63*25+90  25steps
    # python ./play_hydra.py \
    #     --config-name='config_teleop_g1' \
    #     task='g1:teleop' \
    #     env.num_observations=1665 \
    #     env.num_privileged_obs=1742 \
    #     motion.teleop_obs_version='v-teleop-extend-vr-max-nolinvel' \
    #     motion.teleop_selected_keypoints_names=[] \
    #     motion.extend_head='True' \
    #     num_envs=1 \
    #     asset.zero_out_far='False' \
    #     asset.termination_scales.max_ref_motion_distance=10.0 \
    #     load_run='ACCAD_g1_Student_50000' \
    #     checkpoint=43000 \
    #     env.add_short_history='True' \
    #     env.short_history_length=25 \
    #     headless='False' \


    ### change student obs from global -> local
    ### NOTE: 63*25+81 25 steps
    load_run_name="S_local_obs"
    python ./play_hydra.py \
        --config-name='config_teleop_g1' \
        task='g1:teleop' \
        headless='False' \
        num_envs=1 \
        load_run=$load_run_name \
        checkpoint=19000 \
        env.num_observations=1656 \
        env.num_privileged_obs=1733 \
        motion.teleop_obs_version='v-teleop-extend-vr-max-nolinvel-local' \
        motion.teleop_selected_keypoints_names=[] \
        motion.extend_head='True' \
        asset.zero_out_far='False' \
        asset.termination_scales.max_ref_motion_distance=10.0 \
        env.add_short_history='True' \
        env.short_history_length=25 
else
    echo "Please choose from T or S"
fi

