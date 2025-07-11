import warnings

from typing import Any, ClassVar, Optional, TypeVar, Union


import numpy as np
from stable_baselines3.common.cost_decision_aware_policies import CostDecisionAwareActorCriticPolicy
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.cost_buffers import CostRolloutBuffer
from stable_baselines3.common.cost_decision_aware_on_policy_algorithm import CostDecisionAwareOnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_grad_list, compute_grad_norm, armijo_search

SelfSPMAPD = TypeVar("SelfSPMA", bound="SPMAPD")

class Lagrange:
    """Base class for Lagrangian-base Algorithms.
    src: https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/common/lagrange.py

    This class implements the Lagrange multiplier update and the Lagrange loss.

    ..  note::
        Any traditional policy gradient algorithm can be converted to a Lagrangian-based algorithm
        by inheriting from this class and implementing the :meth:`_loss_pi` method.

    Examples:
        >>> from omnisafe.common.lagrange import Lagrange
        >>> def loss_pi(self, data):
        ...     # implement your own loss function here
        ...     return loss

    You can also inherit this class to implement your own Lagrangian-based algorithm, with any
    policy gradient method you like in OmniSafe.

    Examples:
        >>> from omnisafe.common.lagrange import Lagrange
        >>> class CustomAlgo:
        ...     def __init(self) -> None:
        ...         # initialize your own algorithm here
        ...         super().__init__()
        ...         # initialize the Lagrange multiplier
        ...         self.lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    Args:
        cost_limit (float): The cost limit.
        lagrangian_multiplier_init (float): The initial value of the Lagrange multiplier.
        lambda_lr (float): The learning rate of the Lagrange multiplier.
        lambda_optimizer (str): The optimizer for the Lagrange multiplier.
        lagrangian_upper_bound (float or None, optional): The upper bound of the Lagrange multiplier.
            Defaults to None.

    Attributes:
        cost_limit (float): The cost limit.
        lambda_lr (float): The learning rate of the Lagrange multiplier.
        lagrangian_upper_bound (float, optional): The upper bound of the Lagrange multiplier.
            Defaults to None.
        lagrangian_multiplier (torch.nn.Parameter): The Lagrange multiplier.
        lambda_range_projection (torch.nn.ReLU): The projection function for the Lagrange multiplier.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound: float | None = None,
    ) -> None:
        """Initialize an instance of :class:`Lagrange`."""
        self.cost_limit: float = cost_limit
        self.lambda_lr: float = lambda_lr
        self.lagrangian_upper_bound: float | None = lagrangian_upper_bound

        init_value = max(lagrangian_multiplier_init, 0.0)
        self.lagrangian_multiplier: th.nn.Parameter = th.nn.Parameter(
            th.as_tensor(init_value),
            requires_grad=True,
        )
        self.lambda_range_projection: th.nn.ReLU = th.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        assert hasattr(
            th.optim,
            lambda_optimizer,
        ), f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(th.optim, lambda_optimizer)
        self.lambda_optimizer: th.optim.Optimizer = torch_opt(
            [
                self.lagrangian_multiplier,
            ],
            lr=lambda_lr,
        )

    def compute_lambda_loss(self, mean_ep_cost: float) -> th.Tensor:
        """Penalty loss for Lagrange multiplier.

        .. note::
            ``mean_ep_cost`` is obtained from ``self.logger.get_stats('EpCosts')[0]``, which is
            already averaged across MPI processes.

        Args:
            mean_ep_cost (float): mean episode cost.

        Returns:
            Penalty loss for Lagrange multiplier.
        """
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, Jc: float) -> None:
        r"""Update Lagrange multiplier (lambda).

        We update the Lagrange multiplier by minimizing the penalty loss, which is defined as:

        .. math::

            \lambda ^{'} = \lambda + \eta \cdot (J_C - J_C^*)

        where :math:`\lambda` is the Lagrange multiplier, :math:`\eta` is the learning rate,
        :math:`J_C` is the mean episode cost, and :math:`J_C^*` is the cost limit.

        Args:
            Jc (float): mean episode cost.
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(
            0.0,
            self.lagrangian_upper_bound,
        )  # enforce: lambda in [0, inf]


class SPMAPD(CostDecisionAwareOnPolicyAlgorithm):
    """
    Softmax Policy Mirror Ascent algorithm (SPMA)

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": CostDecisionAwareActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[CostDecisionAwareActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate_actor: Union[float, Schedule] = 3e-4,
        learning_rate_critic: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[CostRolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        env_name: Optional[str] = None,
        use_armijo_actor: bool = False,
        use_armijo_critic: bool = False,
        alpha_max: float = 1e6,
        c_actor: float = 0.1,
        c_critic: float = 1e-6,
        eta: float = 1.0,
    ):
        super().__init__(
            policy,
            env,
            learning_rate_actor=learning_rate_actor,
            learning_rate_critic=learning_rate_critic,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert batch_size > 1, (
                "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"
            )

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (not normalize_advantage), (
                f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            )
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.normalize_advantage = normalize_advantage
        self.env_name = env_name
        self.eta = eta
        self.use_armijo_actor = use_armijo_actor
        self.use_armijo_critic = use_armijo_critic
        self.alpha_max = alpha_max
        self.c_actor = c_actor
        self.c_critic = c_critic
        self.explained_var_old = 0
        self.softmax_rep = True

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

    def compute_actor_loss(self, rollout_data, actions, advantages, old_log_prob):
        """Compute actor loss without backward pass."""
        _, _, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
        log_ratio = log_prob - old_log_prob.detach()
        linear_loss = -log_ratio * advantages
        # using approx kl for bregman div from Schulman blog: http://joschu.net/blog/kl-approx.html
        # approx kl is an unbiased estimator of the true kl but results in less variance.
        # alternative approach is to use bregman_div_loss = -1/self.eta * log_ratio
        bregman_div_loss = 1 / self.eta * ((th.exp(log_ratio) - 1) - log_ratio)
        total_loss = th.mean(linear_loss + bregman_div_loss)

        return total_loss, linear_loss, bregman_div_loss

    def compute_critic_loss(self, rollout_data, actions):
        """Compute critic loss without backward pass."""
        # Re-sample the noise matrix because the log_std has changed
        values, _, _, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()

        # critic loss.
        loss = F.mse_loss(rollout_data.returns, values)
        return loss

    def compute_cost_critic_loss(self, rollout_data, actions):
        """Compute cost critic loss without backward pass."""
        # Re-sample the noise matrix because the log_std has changed
        _, cost_values, _, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
        cost_values = cost_values.flatten()

        # critic loss.
        loss = F.mse_loss(rollout_data.cost_returns, cost_values)
        return loss

    def _create_actor_closure(self, rollout_data, actions, advantages, old_log_prob):
        """Create a closure for Armijo line search that only handles loss computation."""

        def closure():
            return self.compute_actor_loss(rollout_data, actions, advantages, old_log_prob)[0]

        return closure

    def _create_critic_closure(self, rollout_data, actions):
        """Create a closure for Armijo line search that only handles loss computation."""

        def closure():
            return self.compute_critic_loss(rollout_data, actions)

        return closure

    def _create_cost_critic_closure(self, rollout_data, actions):
        """Create a closure for Armijo line search that only handles loss computation."""

        def closure():
            return self.compute_cost_critic_loss(rollout_data, actions)

        return closure

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self._update_learning_rate(
            [
                ("actor", self.policy.optimizer_act),
                ("critic", self.policy.optimizer_critic),
                ("cost_critic", self.policy.optimizer_cost_critic),
            ]
        )

        # record losses and number of times the surrogate loss is not decreasing in the inner loop.
        linear_losses = []
        bregman_div_losses = []

        alpha_max_actor = self.alpha_max
        alpha_max_critic = self.alpha_max
        alpha_max_cost_critic = self.alpha_max
        alpha_actor = 1.0
        alpha_critic = 1.0
        alpha_cost_critic = 1.0
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for batch_idx, rollout_data in enumerate(self.rollout_buffer.get(self.batch_size)):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                old_log_prob = rollout_data.old_log_prob.clone()

                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Normalize advantage
                cost_advantages = rollout_data.cost_advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(cost_advantages) > 1:
                    cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)

                # build a closure for the actor and critic.
                # closure_actor = lambda backwards, curr_loss: self.compute_actor_loss(
                #     rollout_data, actions, advantages, old_log_prob, backwards=backwards, curr_loss=curr_loss
                # )
                # Compute losses
                loss_act, linear_loss, bregman_div_loss = self.compute_actor_loss(
                    rollout_data, actions, advantages, old_log_prob
                )
                loss_critic = self.compute_critic_loss(rollout_data, actions)
                loss_cost_critic = self.compute_cost_critic_loss(rollout_data, actions)

                # Backward pass for actor
                self.policy.optimizer_act.zero_grad()
                loss_act.backward()
                grad_list_actor = get_grad_list(self.policy.params_act)
                grad_norm_actor = compute_grad_norm(grad_list_actor)

                # Backward pass for critic
                self.policy.optimizer_critic.zero_grad()
                loss_critic.backward()
                grad_list_critic = get_grad_list(self.policy.params_critic)
                grad_norm_critic = compute_grad_norm(grad_list_critic)

                # Backward pass for critic
                self.policy.optimizer_cost_critic.zero_grad()
                loss_cost_critic.backward()
                grad_list_cost_critic = get_grad_list(self.policy.params_cost_critic)
                grad_norm_cost_critic = compute_grad_norm(grad_list_cost_critic)

                # Update actor parameters
                if self.use_armijo_actor:
                    # Create closure for Armijo line search
                    actor_closure = self._create_actor_closure(rollout_data, actions, advantages, old_log_prob)
                    alpha_actor = armijo_search(
                        actor_closure, self.policy.params_act, grad_list_actor, grad_norm_actor, alpha_max_actor, self.c_actor
                    )
                    alpha_max_actor = alpha_actor * 1.8
                else:
                    self.policy.optimizer_act.step()

                # Update critic parameters
                if self.use_armijo_critic:
                    # Create closure for Armijo line search
                    critic_closure = self._create_critic_closure(rollout_data, actions)
                    alpha_critic = armijo_search(
                        critic_closure,
                        self.policy.params_critic,
                        grad_list_critic,
                        grad_norm_critic,
                        alpha_max_critic,
                        self.c_critic,
                    )
                    alpha_max_critic = alpha_critic * 1.8

                    cost_critic_closure = self._create_cost_critic_closure(rollout_data, actions)
                    alpha_cost_critic = armijo_search(
                        cost_critic_closure,
                        self.policy.params_cost_critic,
                        grad_list_cost_critic,
                        grad_norm_cost_critic,
                        alpha_max_cost_critic,
                        self.c_critic,
                    )
                    alpha_max_cost_critic = alpha_cost_critic * 1.8

                else:
                    self.policy.optimizer_cost_critic.step()

            self._n_updates += 1

        # Logging.
        linear_losses.append(linear_loss.mean().item())
        bregman_div_losses.append(bregman_div_loss.mean().item())

        # Logs
        self.logger.record("train/linear_loss", np.mean(linear_losses))
        self.logger.record("train/bregman_div_loss", np.mean(bregman_div_losses))
        self.logger.record("train/loss_actor", loss_act.item())
        self.logger.record("train/loss_critic", loss_critic.item())
        self.logger.record("train/loss_cost_critic", loss_cost_critic.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
        self: SelfSPMAPD,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "SPMAPD",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSPMAPD:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
