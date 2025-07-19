import os
import numpy as np
import safe_simple_gymnasium
import gymnasium as gym
from experiment_grid import ExperimentGrid
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.spma_pd.spma_pd import SPMAPD
# /localhome/mla233/github/stable-baselines3/runs/spma_pd_multiple_envs_2/SafeCartPole-RightSide-v0---3097403741588e720f643931b666aeb98b9d5533c9b9128afccdea1c4be55032/final_model.zip


# env_id = "SafeCartPole-Arushi-v0"
env_id = "SafeCartPole-ArushiModify-v0"
# env_id = "SafeCartPole-RightSide-v0"
# env_id = "SafeCartPole-FarRightSide-v0"

env = gym.make(env_id)
vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array") for _ in range(1)])

# model = SPMAPD.load("/localhome/mla233/github/stable-baselines3/runs/spma_pd_multiple_envs_6/SafeCartPole-Arushi-v0---5aaed72a03956634b270998ceaed6623df5e3dc1dd353791f8456a2a01674bb6/final_model_0.zip")
# vec_env = VecVideoRecorder(vec_env, 'videos/', record_video_trigger=lambda x: x == 0, video_length=1600, name_prefix=f"spma-pd-{env_id}")

obs = vec_env.reset()
print(obs)
for _ in range(1600 + 1):
    # action, _states = model.predict(obs)
    action = np.array([0])
    obs, _, _, info = vec_env.step(action)
    print(obs, info)
# Save the video
vec_env.close()