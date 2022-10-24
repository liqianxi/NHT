# Import the RL algorithm (Algorithm) we would like to use.
from ray.rllib.algorithms.ppo import PPO
from rllib_utils import get_custom_logger, register_rllib_env


# NHT model path
NHT_path = '.results/walker/20_multihead/version_0'
env = 'Walker2d-v2'

NHT_env_name = register_rllib_env(env, NHT_path)
print('loaded NHT env')

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": NHT_env_name,
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
}

custom_logger_creator = get_custom_logger(NHT_env_name)
# Create our RLlib Trainer.
algo = PPO(config=config, logger_creator=custom_logger_creator)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(3):
    print(algo.train())

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
algo.evaluate()