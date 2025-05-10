import os
# LEGGED_GYM_ROOT_DIR = "/home/zhushiyu/文档/H2O/human2humanoid-main/legged_gym/"
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("=====LEGGED_GYM_ROOT_DIR:",LEGGED_GYM_ROOT_DIR)
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs')
print("=====LEGGED_GYM_ENVS_DIR:",LEGGED_GYM_ENVS_DIR)