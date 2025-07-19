import safe_simple_gymnasium
import gymnasium as gym
from gymnasium.spaces import Space
from typing import Any, Dict, Tuple, Optional
import warnings


from stable_baselines3 import SPMA
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

# SPMA grid search
etas = [0.3, 0.5, 0.7, 0.9, 1.0]
# Parallel environments
n_steps = 2048
n_envs = 4
rollout_buffer_size = n_steps * n_envs
n_inner_updates = 1
n_updates = 100
# total number of samples collected
total_timesteps = rollout_buffer_size * n_updates
# number fo samples collected and used for each inner-optimiztion loop
inner_timesteps = total_timesteps // n_inner_updates

class RewardInfoSwapWrapper(gym.Wrapper):
    """
    A gymnasium wrapper that swaps the reward with a variable from the info dictionary.
    
    This wrapper allows you to use any metric from the info dict as the reward signal,
    while storing the original reward in the info dict under a specified key.
    
    Args:
        env: The environment to wrap
        info_key: The key in the info dict to use as the new reward
        original_reward_key: The key to store the original reward in info dict
        default_value: Default value to use if info_key is missing from info
        warn_on_missing: Whether to warn when info_key is missing
    """
    
    def __init__(
        self,
        env: gym.Env,
        info_key: str,
        original_reward_key: str = "original_reward",
        default_value: float = 0.0,
        warn_on_missing: bool = True
    ):
        super().__init__(env)
        self.info_key = info_key
        self.original_reward_key = original_reward_key
        self.default_value = default_value
        self.warn_on_missing = warn_on_missing
        self._missing_key_warned = False
        
    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment and swap reward with info variable.
        
        Returns:
            observation: The observation from the environment
            reward: The value from info[info_key] (or default_value if missing)
            terminated: Whether the episode has terminated
            truncated: Whether the episode has been truncated
            info: Modified info dict with original reward stored
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store the original reward in info
        info[self.original_reward_key] = reward
        
        # Get the new reward from info
        if self.info_key in info:
            new_reward = float(info[self.info_key])
        else:
            new_reward = self.default_value
            if self.warn_on_missing and not self._missing_key_warned:
                warnings.warn(
                    f"Key '{self.info_key}' not found in info dict. "
                    f"Using default value {self.default_value}."
                )
                self._missing_key_warned = True
        
        return obs, new_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and reset warning flag."""
        self._missing_key_warned = False
        return self.env.reset(**kwargs)


for eta in etas:
    env = gym.make('SafeCartPole-Arushi-v0')
    # env = RewardInfoSwapWrapper(env, info_key='cost')
    env = Monitor(env, f"spma_swap_rc_eta_{eta}_use_line_search")
    model = SPMA(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=n_steps,
        batch_size=n_steps,
        eta=eta,
        n_epochs=5,
        use_armijo_actor=True,
        use_armijo_critic=True,
    )
    model.learn(total_timesteps=total_timesteps)


# for eta in etas:
#     env = gym.make('SafeCartPole-v0')
#     env = RewardInfoSwapWrapper(env, info_key='cost')
#     env = Monitor(env, f"spma_swap_rc_eta_{eta}_use_adam")
#     model = SPMA(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         n_steps=n_steps,
#         batch_size=n_steps,
#         eta=eta,
#         n_epochs=5,
#         use_armijo_actor=False,
#         use_armijo_critic=False,
#     )
#     model.learn(total_timesteps=total_timesteps)
