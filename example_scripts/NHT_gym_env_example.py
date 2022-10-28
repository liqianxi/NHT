from nht.mujoco_interface import register_NHT_env
import gym

NHT_path = '.results/walker/20_multihead/version_0'
env = 'Walker2d-v2'

register_NHT_env(env, NHT_path) # gym registration
my_NHT_env = gym.make('NHT_Walker2d-v2')

# print some info about the environment to see that it works
my_NHT_env.reset()
print(f'Action space: {my_NHT_env.action_space}')
print(f'Obs space: {my_NHT_env.observation_space}')
print(f'NHT model \n{my_NHT_env.Q}') # NHT model