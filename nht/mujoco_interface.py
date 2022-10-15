from gym import utils as gym_utils
from gym import spaces
import numpy as np
from gym.envs.registration import register
import gym
from gym.envs.registration import spec, load


def register_NHT_env(base_env, NHT_path):
    
    NHT_env_class = create_NHT_env(base_env, NHT_path)
    temp_env = gym.make(base_env)

    register(
        id=f'NHT_{base_env}',
        entry_point=NHT_env_class,
        max_episode_steps=temp_env._max_episode_steps,
        reward_threshold=temp_env.spec.reward_threshold,
        nondeterministic=temp_env.spec.nondeterministic
    )


from nht.utils import load_interface
import torch
def create_NHT_env(base_env, NHT_path):

    env_spec = spec(base_env)
    env_class = load(env_spec.entry_point)

    class MuJoCoInterfaceEnv(env_class, gym_utils.EzPickle):
        def __init__(self,):# env_config):

            # load NHT model by reading path from config.yaml
            model_path = NHT_path
            model_type = 'NHT'
            self.Q = load_interface(model_type, model_path)
            self.action_dim = self.Q.k

            env_class.__init__(self)
        
        def _set_original_action_space(self):
            super()._set_action_space()

        def _set_action_space(self):
            self._set_original_action_space()
            n = self.action_space.shape[0]
            self.action_space = spaces.Box(low=-np.sqrt(n/self.action_dim), high=np.sqrt(n/self.action_dim), shape=(self.action_dim,), dtype=np.float32)

        def step(self, action):

            k = self.action_dim
            assert action.shape == (k,)

            self._set_original_action_space() # temporarily change action space to pass check in MujocoEnv do_simulation function

            with torch.no_grad():
                c = torch.tensor(self._get_obs().copy(),dtype=torch.float32).unsqueeze(0)
                Q_hat = self.Q(c).squeeze(0)
                a = np.expand_dims(action.copy(),1) # turn action from agent to column vector tensor (with batch dimension)
                u = np.matmul(Q_hat, a).squeeze()

                action = u.numpy().copy()
                action = np.clip(action, -1, 1)

            return_values = super().step(action)
            self._set_action_space() # change action space back 

            return return_values

    return MuJoCoInterfaceEnv