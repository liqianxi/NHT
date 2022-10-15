import os
from datetime import datetime
import tempfile
from ray.tune.logger import Logger, UnifiedLogger
import gym
from ray.tune.registry import register_env
from NHT_envs.MuJoCo_Interface import register_NHT_env
import gym

def register_rllib_env(env, NHT_path):

    NHT_env_name = f'NHT_{env}'
    def rllib_env_creator(env_config):

        register_NHT_env(env, NHT_path) # gym registration
        my_NHT_env = gym.make(NHT_env_name)

        return my_NHT_env  # return an env instance

    # rllib registration
    register_env(NHT_env_name, rllib_env_creator)

    # return string with env name to pass to aglo config
    return NHT_env_name

def get_custom_logger(env_name):
    # custom logger implementation
    alg = "PPO"
    custom_result_dir = "./.results"

    # Default logdir prefix containing the agent's name and the
    # env id.
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}_{}".format(alg, env_name, timestr)
    if not os.path.exists(custom_result_dir):
        # Possible race condition if dir is created several times on
        # rollout workers
        os.makedirs(custom_result_dir, exist_ok=True)
    logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_result_dir)


    def custom_logger_creator(config):
        """Creates a Unified logger with the default prefix."""
        return UnifiedLogger(config, logdir, loggers=None)
    
    return custom_logger_creator