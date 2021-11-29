import gym

from stable_baselines3 import HJ_DQN

import sys
sys.path.append("../../avoiding-the-unrechable")
import gym_dubins

env = gym.make("dubins3d-discrete-v0")
env = gym.wrappers.TimeLimit(env, 150)

def hj_contoller(state):
  return False, [2]

model = HJ_DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, hj_controller_fn=hj_contoller, tensorboard_log='test123')
model.learn(total_timesteps=300_000, log_interval=1_000)