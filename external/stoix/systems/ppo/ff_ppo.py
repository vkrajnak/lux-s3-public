import os
import sys
import copy
import time
import pickle
from typing import Any, Dict, Tuple
from colorama import Fore, Style

import chex
import flax
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.serialization import to_state_dict, from_state_dict
from external.jumanji.env import Environment
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from external.stoix.base_types import (
    ActorApply,
    ActorCriticOptStates,
    ActorCriticParams,
    AnakinExperimentOutput,
    CriticApply,
    LearnerFn,
    OnPolicyLearnerState,
)
from external.stoix.evaluator import evaluator_setup, get_distribution_act_fn
# from external.stoix.networks.base import FeedForwardActor as Actor
# from external.stoix.networks.base import FeedForwardCritic as Critic
from external.stoix.systems.ppo.ppo_types import PPOTransition
# from external.stoix.utils import make_env as environments
from external.stoix.utils.checkpointing import Checkpointer
from external.stoix.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from external.stoix.utils.logger import LogEvent, StoixLogger
from external.stoix.utils.loss import clipped_value_loss, ppo_clip_loss
from external.stoix.utils.multistep import batch_truncated_generalized_advantage_estimation
from external.stoix.utils.total_timestep_checker import check_total_timesteps
from external.stoix.utils.training import make_learning_rate
from external.stoix.wrappers.episode_metrics import get_final_step_metrics

from environment.make_env import make_env
from environment.oneplayerenv import get_some_obs
from player.rl_agent.network import Actor, Critic


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[OnPolicyLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: OnPolicyLearnerState, _: Any
    ) -> Tuple[OnPolicyLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
            learner_state (NamedTuple):
                - params (ActorCriticParams): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.
        """

        def _env_step(
            learner_state: OnPolicyLearnerState, _: Any
        ) -> Tuple[OnPolicyLearnerState, PPOTransition]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = actor_apply_fn(params.actor_params, last_timestep.observation)
            value = critic_apply_fn(params.critic_params, last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]  # AL: modified RecordEpisodeMetrics wrapper so that it saves our own metrics too

            transition = PPOTransition(
                done,
                truncated,
                action,
                value,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                info,
            )
            learner_state = OnPolicyLearnerState(params, opt_states, key, env_state, timestep)
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # CALCULATE ADVANTAGE
        params, opt_states, key, env_state, last_timestep = learner_state
        last_val = critic_apply_fn(params.critic_params, last_timestep.observation)

        r_t = traj_batch.reward
        v_t = jnp.concatenate([traj_batch.value, last_val[None, ...]], axis=0)
        d_t = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (d_t * config.system.gamma).astype(jnp.float32)
        advantages, targets = batch_truncated_generalized_advantage_estimation(
            r_t,
            d_t,
            config.system.gae_lambda,
            v_t,
            time_major=True,
            standardize_advantages=config.system.standardize_advantages,
            truncation_flags=traj_batch.truncated,
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK
                    actor_policy = actor_apply_fn(actor_params, traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # CALCULATE ACTOR LOSS
                    loss_actor = ppo_clip_loss(
                        log_prob, traj_batch.log_prob, gae, config.system.clip_eps
                    )
                    policy_entropy = actor_policy.entropy().mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * policy_entropy

                    # AL: added metrics
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)  # p(action)/p_old(action)
                    clipping_fraction = jnp.abs(ratio - 1.0) > config.system.clip_eps  # definition often used in algos
                    kl_divergence = (ratio - 1.0) - jnp.log(ratio)  # approximation proposed in https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

                    loss_info = {
                        "actor_loss": loss_actor,
                        "policy_entropy": policy_entropy,
                        "clipping_fraction": clipping_fraction.mean(),
                        "ratio_new_old": ratio.mean(),
                        "kl_divergence": kl_divergence.mean(),
                    }
                    return total_loss_actor, loss_info

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: PPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN NETWORK
                    value = critic_apply_fn(critic_params, traj_batch.obs)

                    # CALCULATE VALUE LOSS
                    value_loss = clipped_value_loss(
                        value, traj_batch.value, targets, config.system.clip_eps
                    )

                    critic_total_loss = config.system.vf_coef * value_loss
                    loss_info = {
                        "value_loss": value_loss,
                    }
                    return critic_total_loss, loss_info

                # CALCULATE ACTOR LOSS
                actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
                actor_grads, actor_loss_info = actor_grad_fn(
                    params.actor_params, traj_batch, advantages
                )

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
                critic_grads, critic_loss_info = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="device"
                )

                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                # PACK NEW PARAMS AND OPTIMISER STATE
                new_params = ActorCriticParams(actor_new_params, critic_new_params)
                new_opt_state = ActorCriticOptStates(actor_new_opt_state, critic_new_opt_state)

                # PACK LOSS INFO
                loss_info = {
                    **actor_loss_info,
                    **critic_loss_info,
                }
                return (new_params, new_opt_state), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key = jax.random.split(key)

            # SHUFFLE MINIBATCHES
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            # UPDATE MINIBATCHES
            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states), minibatches
            )

            update_state = (params, opt_states, traj_batch, advantages, targets, key)
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = OnPolicyLearnerState(params, opt_states, key, env_state, last_timestep)
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OnPolicyLearnerState,
    ) -> AnakinExperimentOutput[OnPolicyLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - params (ActorCriticParams): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
        env: Environment,
        key: chex.PRNGKey,
        config: DictConfig,
        params: ActorCriticParams,
        apply_fns: Tuple[ActorApply, CriticApply],
        update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
        opt_states: ActorCriticOptStates,
) -> Tuple[LearnerFn[OnPolicyLearnerState], OnPolicyLearnerState]:
    """Initialise learner_fn, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions. AL: removed, not needed with our custom networks
    # num_actions = int(env.action_spec().num_values)
    # config.system.action_dim = num_actions

    # AL: networks init moved outside

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Define params to be replicated across devices and batches.
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))

    replicate_learner = (params, opt_states)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states = replicate_learner
    init_learner_state = OnPolicyLearnerState(params, opt_states, step_keys, env_states, timesteps)

    return learn, init_learner_state


def networks_setup(
        unbatched_obs,
        key: chex.PRNGKey,
        config: DictConfig
) -> Tuple[int, Actor, ActorCriticParams, Tuple[ActorApply, CriticApply], Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn], ActorCriticOptStates]:
    """Initialise network and optimiser"""

    # PRNG keys.
    key, actor_net_key, critic_net_key = jax.random.split(key, 3)

    # Define network and optimiser.  AL: deal with custom networks
    # actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    # actor_action_head = hydra.utils.instantiate(
    #     config.network.actor_network.action_head, action_dim=num_actions
    # )
    # critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    # critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)

    actor_network = Actor(
        torso=config.env.agent.torso.name,
        torso_kwargs=config.env.agent.torso.kwargs,
        head=config.env.agent.head
    )
    critic_network = Critic(
        torso=config.env.agent.torso.name,
        torso_kwargs=config.env.agent.torso.kwargs
    )

    actor_lr = make_learning_rate(
        config.system.actor_lr, config, config.system.epochs, config.system.num_minibatches
    )
    critic_lr = make_learning_rate(
        config.system.critic_lr, config, config.system.epochs, config.system.num_minibatches
    )

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], unbatched_obs)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = ActorCriticParams(actor_params, critic_params)

    # Load model from checkpoint if specified.
    t0 = 0
    if config.loader.load_model:
        loaded_checkpoint = Checkpointer(  # AL: modified, removed model_name
            **config.loader.load_args,  # Other checkpoint loader args
        )
        # Restore the learner state from the checkpoint
        t0, restored_params, _ = loaded_checkpoint.restore_params(step=config.loader.load_step)
        # Update the params
        params = restored_params

    # Save actor_params.pkl (useful for self-play)
    if config.opponents.use_selfplay:
        actor_params_dict = to_state_dict(params.actor_params)
        actor_params_filename = os.path.join(config.run.trained_agent_dir, 'actor_params.pkl')
        with open(actor_params_filename, 'wb') as fp:
            pickle.dump(actor_params_dict, fp)

    # Pack apply and update functions.
    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply
    apply_fns = (actor_network_apply_fn, critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update)

    # Optimize state
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)

    # Print networks arch
    if config.run.print_nn:

        dummy_actor_network = Actor(
            torso=config.env.agent.torso.name,
            torso_kwargs=config.env.agent.torso.kwargs,
            head=config.env.agent.head,
            print_arch=True
        )
        tabulate_fn = flax.linen.tabulate(dummy_actor_network, jax.random.PRNGKey(0), console_kwargs={'width': 400})
        print(tabulate_fn(unbatched_obs), file=sys.stderr)

        dummy_critic_network = Critic(
            torso=config.env.agent.torso.name,
            torso_kwargs=config.env.agent.torso.kwargs
        )
        tabulate_fn = flax.linen.tabulate(dummy_critic_network, jax.random.PRNGKey(0), console_kwargs={'width': 400})
        print(tabulate_fn(unbatched_obs), file=sys.stderr)

    return t0, actor_network, params, apply_fns, update_fns, opt_states


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # PRNG key
    key = jax.random.PRNGKey(config.run.seed)  # AL: config.arch.seed -> config.run.seed

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Calculate number of updates per evaluation.
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = StoixLogger(config)

    # Print config to screen
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Setup networks
    key, key_networks = jax.random.split(key)
    unbatched_obs = get_some_obs(config.env.agent)
    t0, actor_network, params, apply_fns, update_fns, opt_states = networks_setup(unbatched_obs, key_networks, config)

    # Set up checkpointer
    save_checkpoint = config.checkpointer.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(  # AL: modified, removed model_name
            metadata=config,  # Save all config as metadata in the checkpoint
            **config.checkpointer.save_args,  # Checkpoint args
        )

    # Run self-play experiments for a number of iterations (AL: new)
    max_eval_metric_init = jnp.float32(-1e7)
    max_eval_metric_per_opponent = {}
    round_env_steps = 0  # to record total number of env steps across all sequential rounds
    for round_iteration in range(config.opponents.n_rounds):

        # Build opponents_params for this iteration
        opponents_params = OmegaConf.to_container(config.opponents, resolve=True)

        # Create the environments for train and eval. AL: deal with custom env
        # env, eval_env = environments.make(config=config)
        env, eval_env = make_env(config.env, opponents_params)

        if env.opponent_name not in max_eval_metric_per_opponent.keys():
            max_eval_metric_per_opponent[env.opponent_name] = max_eval_metric_init

        print("*****************************************************************")
        print(f"   round: {round_iteration + 1}/{config.opponents.n_rounds}")
        print(f"   training against: {env.opponent_name}")
        print("*****************************************************************")

        # PRNG keys.
        key, key_e, key_learner, = jax.random.split(key, 3)

        # Setup learner.
        # learn, actor_network, learner_state = learner_setup(
        #     env, (key_learner, actor_net_key, critic_net_key), config
        # )
        learn, learner_state = learner_setup(env, key_learner, config, params, apply_fns, update_fns, opt_states)
        learn = jax.jit(learn)  # not really needed since all scans are already jitted

        # Setup evaluator.
        evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
            eval_env=eval_env,
            key_e=key_e,
            eval_act_fn=get_distribution_act_fn(config, actor_network.apply),
            params=learner_state.params.actor_params,
            config=config,
        )
        evaluator = jax.jit(evaluator)  # not really needed since all scans are already jitted

        # Run experiment for a total number of evaluations.
        best_params = unreplicate_batch_dim(learner_state.params.actor_params)
        for eval_step in range(config.arch.num_evaluation):
            # Train.
            start_time = time.time()

            learner_output = learn(learner_state)
            jax.block_until_ready(learner_output)

            # Number of steps done in the environment (used for logging and checkpointing)
            t = t0 + round_env_steps + int(steps_per_rollout * (eval_step + 1))

            # Log the results of the training.
            elapsed_time = time.time() - start_time
            episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
            episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time
            return_metrics = episode_metrics.pop("return_contributions")
            print(f"time spent training  : {round(elapsed_time, 2)} seconds")

            # Separately log timesteps, actoring metrics and training metrics.
            logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
            if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
                logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
                logger.log(return_metrics, t, eval_step, LogEvent.RETURN_ACT)
            train_metrics = learner_output.train_metrics
            # Calculate the number of optimiser steps per second. Since gradients are aggregated
            # across the device and batch axis, we don't consider updates per device/batch as part of
            # the SPS for the learner.
            opt_steps_per_eval = config.arch.num_updates_per_eval * (
                config.system.epochs * config.system.num_minibatches
            )
            train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
            logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

            # Prepare for evaluation.
            start_time = time.time()
            trained_params = unreplicate_batch_dim(
                learner_output.learner_state.params.actor_params
            )  # Select only actor params
            key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
            eval_keys = jnp.stack(eval_keys)
            eval_keys = eval_keys.reshape(n_devices, -1)

            # Evaluate.
            evaluator_output = evaluator(trained_params, eval_keys)
            jax.block_until_ready(evaluator_output)

            # Log the results of the evaluation.
            elapsed_time = time.time() - start_time
            eval_metric = jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric])
            print(f"time spent evaluating: {round(elapsed_time, 2)} seconds")

            steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
            evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
            return_metrics = evaluator_output.episode_metrics.pop("return_contributions")
            logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)
            logger.log(return_metrics, t, eval_step, LogEvent.RETURN_EVAL)

            if save_checkpoint:
                # Save checkpoint of learner state
                checkpointer.save(
                    step=t,
                    t=t,  # AL: added
                    unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                    metrics=evaluator_output.episode_metrics,  # AL: modified metrics
                )

            if max_eval_metric_per_opponent[env.opponent_name] <= eval_metric:
                # AL: added saving params (not full checkpoint), for inference
                unreplicated_actor_params = unreplicate_n_dims(learner_output.learner_state.params.actor_params)
                actor_params_dict = to_state_dict(unreplicated_actor_params)
                if config.opponents.use_selfplay:
                    actor_params_filename = os.path.join(config.run.trained_agent_dir, 'actor_params.pkl')
                else:
                    actor_params_filename = os.path.join(config.run.trained_agent_dir, 'actor_params_vs_' + env.opponent_name + '.pkl')
                with open(actor_params_filename, 'wb') as fp:
                    pickle.dump(actor_params_dict, fp)
                print(f"{Fore.RED}{Style.BRIGHT}!!! NEW BEST MODEL AGAINST THIS OPPONENT !!! {config.env.eval_metric} = {eval_metric}{Style.RESET_ALL}  "
                      f"{Fore.RED}(saved in {actor_params_filename}){Style.RESET_ALL}")
                best_params = copy.deepcopy(trained_params)
                max_eval_metric_per_opponent[env.opponent_name] = eval_metric
                logger.log({"new_best": 1}, t, eval_step, LogEvent.BEST)
            else:
                logger.log({"new_best": 0}, t, eval_step, LogEvent.BEST)

            # Update runner state to continue training.
            learner_state = learner_output.learner_state

        # Measure absolute metric.
        if config.arch.absolute_metric:
            start_time = time.time()

            key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
            eval_keys = jnp.stack(eval_keys)
            eval_keys = eval_keys.reshape(n_devices, -1)

            evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
            jax.block_until_ready(evaluator_output)

            elapsed_time = time.time() - start_time
            steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
            evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
            logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

            final_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))

        else:

            final_performance = float(max_eval_metric_per_opponent[env.opponent_name])

        print(f"{Fore.RED}Best Model Performance: {config.env.eval_metric} = {final_performance}{Style.RESET_ALL}")

        # Save last model of this round (for inference)
        unreplicated_actor_params = unreplicate_n_dims(learner_state.params.actor_params)
        actor_params_dict = to_state_dict(unreplicated_actor_params)
        if config.opponents.use_selfplay:
            # save last model (will be opponent in next round)
            actor_params_filename = os.path.join(config.run.trained_agent_dir, 'actor_params.pkl')
            with open(actor_params_filename, 'wb') as fp:
                pickle.dump(actor_params_dict, fp)
        # save last model (as a backup)
        actor_params_filename = os.path.join(config.run.trained_agent_dir, 'actor_params_last_of_round_' + str(round_iteration+1) + '.pkl')
        with open(actor_params_filename, 'wb') as fp:
            pickle.dump(actor_params_dict, fp)


        # Print best performances across opponents
        pprint(jax.tree.map(lambda a: float(a), max_eval_metric_per_opponent))

        # Get ready for next round
        round_env_steps += int(steps_per_rollout * (eval_step + 1))
        params = unreplicate_n_dims(learner_state.params)
        opt_states = unreplicate_n_dims(learner_state.opt_states)

        # Clean up sys.path for next round
        if not config.opponents.use_selfplay:
            sys.path.remove(env.opponent_dir)

        del env, eval_env

    # Stop the logger.
    logger.stop()

    return final_performance


