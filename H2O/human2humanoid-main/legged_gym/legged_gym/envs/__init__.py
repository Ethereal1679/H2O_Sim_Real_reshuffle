from legged_gym.utils.task_registry import task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

# H1 -----added by zsy
# from legged_gym.envs.h1.h1_teleop_config import H1TeleopCfg, H1TeleopCfgPPO
# from legged_gym.envs.h1.h1_teleop_env import H1TeleopRobot
# task_registry.register( "h1:teleop", H1TeleopRobot, H1TeleopCfg(), H1TeleopCfgPPO())


# G1 -----added by zsy
from legged_gym.envs.g1.g1_teleop_config import G1TeleopCfg, G1TeleopCfgPPO
from legged_gym.envs.g1.g1_teleop_env import G1TeleopRobot
task_registry.register( "g1:teleop", G1TeleopRobot, G1TeleopCfg(), G1TeleopCfgPPO())
