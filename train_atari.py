from dataclasses import dataclass

import ale_py
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs


@dataclass
class PPO_ATARI_CONFIG:
    # Default parameters for Atari
    # src: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
    env_id: str = "ALE/Pong-v5"
    seed: int = 0
    frame_stack: int = 4
    policy: str = "CnnPolicy"
    n_envs: int = 8
    n_steps: int = 128
    n_epochs: int = 4
    batch_size: int = 256
    n_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4  #  linear schedule
    clip_range: float = 0.1  # linear schedule
    vf_coef: float = 0.5
    ent_coef: float = 0.01


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


env_id = "ALE/Pong-v5"
# Since ALE-py v0.11, a number of registered Atari environments were removed including the `NoFrameskip` varients.
# To recreate it, we require the following parameters.
env_kwargs = {"obs_type": "rgb", "frameskip": 1, "repeat_action_probability": 0.0, "full_action_space": False}
seed = 0

cfg = PPO_ATARI_CONFIG(env_id=env_id, seed=seed)

env = make_vec_env(env_id, n_envs=cfg.n_envs, seed=seed, wrapper_class=AtariWrapper, env_kwargs=env_kwargs)
env = VecFrameStack(env, n_stack=cfg.frame_stack)
env = VecTransposeImage(env)

model = PPO(
    policy=cfg.policy,
    env=env,
    n_steps=cfg.n_steps,
    batch_size=cfg.batch_size,
    n_epochs=cfg.n_epochs,
    learning_rate=linear_schedule(cfg.learning_rate),
    clip_range=linear_schedule(cfg.clip_range),
    ent_coef=cfg.ent_coef,
    vf_coef=cfg.vf_coef,
    seed=cfg.seed,
    verbose=1,
    device="auto",
)
model.learn(total_timesteps=1_000_000)
