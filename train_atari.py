import wandb
import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ale_py
import gymnasium as gym
import numpy as np
from ocatari.ram.extract_ram_info import (
    detect_objects_ram,
    init_objects,
)
from ocatari.ram.pong import Player

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EventCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None


from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

if TYPE_CHECKING:
    pass

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: gym.Env | VecEnv,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        reward_type: str = "score",
        dump_log: bool = False
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results: list[list[float]] = []
        self.evaluations_timesteps: list[int] = []
        self.evaluations_length: list[list[int]] = []
        # For computing success rate
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []

        self.reward_type = reward_type
        self.dump_log = dump_log

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(f"Training and eval env are not of the same type{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"eval/mean_{self.reward_type}_reward", float(mean_reward))
            self.logger.record(f"eval/mean_{self.reward_type}_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            if self.dump_log:
                self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class ObjectInfoWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    # fmt: off
    AVAILABLE_GAMES = ["Pong"]
    # fmt: on

    def __init__(self, env: gym.Env, hud: bool = False):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        # Assumes `env` is an Atari env supported by `OCAtari`
        spec = getattr(env.unwrapped, "spec", None) or getattr(env, "spec", None)
        env_name = spec.id

        game_name = (
            env_name.split("/")[1].split("-")[0].split("No")[0].split("Deterministic")[0]
            if "ALE/" in env_name
            else env_name.split("-")[0].split("No")[0].split("Deterministic")[0]
        )

        if game_name[:4] not in [gn[:4] for gn in self.AVAILABLE_GAMES]:
            raise ValueError(f"Game '{game_name}' not covered yet by OCAtari")

        self.game_name = game_name
        self.hud = hud

        # Uses RAM-based extraction to set object detection
        self.objects = init_objects(self.game_name, self.hud)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Detect objects based on the configured detection mode
        detect_objects_ram(self.objects, self.unwrapped.ale.getRAM(), self.game_name, self.hud)
        info["objects"] = self.objects

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        self.objects = init_objects(self.game_name, self.hud)
        detect_objects_ram(self.objects, self.unwrapped.ale.getRAM(), self.game_name, self.hud)
        info["objects"] = self.objects

        return obs, info


class PongTertiaryRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        boundary_y: int = 60,
        replace_reward: bool = True,
    ):
        gym.Wrapper.__init__(self, env)

        self.boundary_y = boundary_y
        self.replace_reward = replace_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        objects = info.get("objects", [])
        if len(objects) == 0:
            raise ValueError("Objects not found")

        objects = self.serialize_object(objects)
        reward_prime = self.tertiary_reward(objects["player_y"])

        if self.replace_reward:
            reward = reward_prime
        else:
            info["tertiary_reward"] = reward_prime

        return obs, reward, terminated, truncated, info

    def tertiary_reward(self, player_y):
        top_y = 34  # highest reachable player y-position
        distance_to_top = player_y - top_y

        # Penalize only when player_y is above the boundary line.
        dist_into_unwanted_region = max(0, self.boundary_y - distance_to_top)
        # player_y >= boundary_y -> 0
        # player_y = 0           -> 1
        normalized_penalty = dist_into_unwanted_region / self.boundary_y

        # negative since want to avoid this penalty
        return -normalized_penalty

    def serialize_object(self, objects):
        d = {
            "player_x": None,
            "player_y": None,
            "player_w": None,
            "player_h": None,
        }
        for o in objects:
            if isinstance(o, Player):
                d["player_x"] = o.x
                d["player_y"] = o.y
                d["player_w"] = o.w
                d["player_h"] = o.h
        return d


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
    eval_freq: int = 25_000


def make_atari_env(env_id, n_envs, n_stack, seed, use_objects):
    # Since ALE-py v0.11, a number of registered Atari environments were removed including the `NoFrameskip` varients.
    # To recreate it, we require the following parameters.
    env_kwargs = {"obs_type": "rgb", "frameskip": 1, "repeat_action_probability": 0.0, "full_action_space": False}

    def dummy():
        env = gym.make(env_id, **env_kwargs)
        if use_objects:
            env = ObjectInfoWrapper(env, hud=True)
            env = PongTertiaryRewardWrapper(env, boundary_y=60)
        return env

    env = make_vec_env(dummy, n_envs=n_envs, seed=seed, wrapper_class=AtariWrapper)
    env = VecFrameStack(env, n_stack=n_stack)
    env = VecTransposeImage(env)
    return env


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


def train(exp_log_dir, env_id, seed, use_objects, use_wandb, exp_name, slurm_id):
    if use_wandb:
        config = {}
        config["algo"] = 'PPO'
        config["seed"] = seed
        config["env_id"] = env_id
        wandb.init(
            project='RLwPrefAtari',
            name=f"{exp_name}_{slurm_id}",
            config=config,
            sync_tensorboard=True,
        )

    cfg = PPO_ATARI_CONFIG(env_id=env_id, seed=seed)
    logger = configure(exp_log_dir, ["stdout", "csv", "tensorboard"])
    env = make_atari_env(env_id, cfg.n_envs, cfg.frame_stack, seed, use_objects)
    standard_atari_eval_env = make_atari_env(env_id, 1, cfg.frame_stack, seed + 10_067, use_objects=False)
    tertiary_atari_eval_env = make_atari_env(env_id, 1, cfg.frame_stack, seed + 10_067, use_objects=True)

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
        tensorboard_log=exp_log_dir
    )
    model.set_logger(logger)

    # ORDER IS REALLY IMPORANT
    # WE ONLY DUMP THE LOGS AFTER THE LAST EVALCALLBACK IS CALLED
    standard_eval_callback = EvalCallback(
        standard_atari_eval_env,
        n_eval_episodes=5,
        log_path=f"{exp_log_dir}/standard/",
        eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
        deterministic=True,
        render=False,
        verbose=1,
        reward_type="score",
        dump_log=False
    )

    tertiary_eval_callback = EvalCallback(
        tertiary_atari_eval_env,
        n_eval_episodes=5,
        log_path=f"{exp_log_dir}/tertiary/",
        eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
        deterministic=True,
        render=False,
        verbose=1,
        reward_type="object",
        dump_log=True
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1_000_000 // cfg.n_envs, 1),
        save_path=f"{exp_log_dir}/models/",
    )

    callback = CallbackList([checkpoint_callback, standard_eval_callback, tertiary_eval_callback])
    model.learn(total_timesteps=cfg.n_timesteps, callback=callback)
    model.save(f"{exp_log_dir}/final_model")


if __name__ == "__main__":

    def str2bool(v):
        """Convert string to boolean"""
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"): return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--use_objects", type=str2bool, default=False)
    parser.add_argument("--exp_name", type=str)

    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--slurm_task_id", type=int)
    args = parser.parse_args()

    exp_log_dir = f"./runs/{args.exp_name}/task_id_{args.slurm_task_id}"
    Path.mkdir(Path(exp_log_dir), exist_ok=True, parents=True)

    train(exp_log_dir, args.env_id, args.seed, args.use_objects, args.use_wandb, args.exp_name, args.slurm_task_id)
