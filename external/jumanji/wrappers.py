# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeAlias, Union

import chex
import external.jumanji.dm_env.specs as dm_env_specs
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from external.jumanji import specs, tree_utils
from external.jumanji.env import ActionSpec, Environment, Observation, State
from external.jumanji.types import TimeStep

# Type alias that corresponds to ObsType in the Gym API
GymObservation: TypeAlias = chex.ArrayNumpy | Dict[str, Union[chex.ArrayNumpy, "GymObservation"]]


class Wrapper(Environment[State, ActionSpec, Observation], Generic[State, ActionSpec, Observation]):
    """Wraps the environment to allow modular transformations.
    Source: https://github.com/google/brax/blob/main/brax/envs/env.py#L72
    """

    def __init__(self, env: Environment[State, ActionSpec, Observation]):
        self._env = env
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._env!r})"

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self._env, name)

    @property
    def unwrapped(self) -> Environment[State, ActionSpec, Observation]:
        """Returns the wrapped env."""
        return self._env.unwrapped

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment to an initial state.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
        """
        return self._env.reset(key)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action to take.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
        """
        return self._env.step(state, action)

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec."""
        return self._env.observation_spec

    @cached_property
    def action_spec(self) -> ActionSpec:
        """Returns the action spec."""
        return self._env.action_spec

    @cached_property
    def reward_spec(self) -> specs.Array:
        """Returns the reward spec."""
        return self._env.reward_spec

    @cached_property
    def discount_spec(self) -> specs.BoundedArray:
        """Returns the discount spec."""
        return self._env.discount_spec

    def render(self, state: State) -> Any:
        """Compute render frames during initialisation of the environment.

        Args:
            state: State object containing the dynamics of the environment.
        """
        return self._env.render(state)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        return self._env.close()

    def __enter__(self) -> Wrapper:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# class JumanjiToDMEnvWrapper(dm_env.Environment, Generic[State, ActionSpec, Observation]):
#     """A wrapper that converts Environment to dm_env.Environment."""
#
#     def __init__(
#         self,
#         env: Environment[State, ActionSpec, Observation],
#         key: Optional[chex.PRNGKey] = None,
#     ):
#         """Create the wrapped environment.
#
#         Args:
#             env: `Environment`to wrap to a `dm_env.Environment`.
#             key: optional key to initialize the `Environment` with.
#         """
#         self._env = env
#         if key is None:
#             self._key = jax.random.PRNGKey(0)
#         else:
#             self._key = key
#         self._state: Any
#         self._jitted_reset: Callable[[chex.PRNGKey], Tuple[State, TimeStep]] = jax.jit(
#             self._env.reset
#         )
#         self._jitted_step: Callable[[State, chex.Array], Tuple[State, TimeStep]] = jax.jit(
#             self._env.step
#         )
#
#     def __repr__(self) -> str:
#         return str(self._env.__repr__())
#
#     def reset(self) -> dm_env.TimeStep:
#         """Starts a new sequence and returns the first `TimeStep` of this sequence.
#
#         Returns:
#             A `TimeStep` namedtuple containing:
#                 - step_type: A `StepType` of `FIRST`.
#                 - reward: `None`, indicating the reward is undefined.
#                 - discount: `None`, indicating the discount is undefined.
#                 - observation: A NumPy array, or a nested dict, list or tuple of arrays.
#                     Scalar values that can be cast to NumPy arrays (e.g. Python floats)
#                     are also valid in place of a scalar array. Must conform to the
#                     specification returned by `observation_spec`.
#         """
#         reset_key, self._key = jax.random.split(self._key)
#         self._state, timestep = self._jitted_reset(reset_key)
#         return dm_env.restart(observation=timestep.observation)
#
#     def step(self, action: chex.ArrayNumpy) -> dm_env.TimeStep:
#         """Updates the environment according to the action and returns a `TimeStep`.
#
#         If the environment returned a `TimeStep` with `StepType.LAST` at the
#         previous step, this call to `step` will start a new sequence and `action`
#         will be ignored.
#
#         This method will also start a new sequence if called after the environment
#         has been constructed and `reset` has not been called. Again, in this case
#         `action` will be ignored.
#
#         Args:
#             action: A NumPy array, or a nested dict, list or tuple of arrays
#                 corresponding to `action_spec`.
#
#         Returns:
#             A `TimeStep` namedtuple containing:
#                 - step_type: A `StepType` value.
#                 - reward: Reward at this timestep, or None if step_type is
#                     `StepType.FIRST`. Must conform to the specification returned by
#                     `reward_spec`.
#                 - discount: A discount in the range [0, 1], or None if step_type is
#                     `StepType.FIRST`. Must conform to the specification returned by
#                     `discount_spec`.
#                 - observation: A NumPy array, or a nested dict, list or tuple of arrays.
#                     Scalar values that can be cast to NumPy arrays (e.g. Python floats)
#                     are also valid in place of a scalar array. Must conform to the
#                     specification returned by `observation_spec`.
#         """
#         self._state, timestep = self._jitted_step(self._state, action)
#         return dm_env.TimeStep(
#             step_type=timestep.step_type,
#             reward=timestep.reward,
#             discount=timestep.discount,
#             observation=timestep.observation,
#         )
#
#     def observation_spec(self) -> dm_env_specs.Array:
#         """Returns the dm_env observation spec."""
#         return specs.jumanji_specs_to_dm_env_specs(self._env.observation_spec)
#
#     def action_spec(self) -> dm_env_specs.Array:
#         """Returns the dm_env action spec."""
#         return specs.jumanji_specs_to_dm_env_specs(self._env.action_spec)
#
#     @property
#     def unwrapped(self) -> Environment[State, ActionSpec, Observation]:
#         return self._env
#
#
# class MultiToSingleWrapper(
#     Wrapper[State, ActionSpec, Observation], Generic[State, ActionSpec, Observation]
# ):
#     """A wrapper that converts a multi-agent Environment to a single-agent Environment."""
#
#     def __init__(
#         self,
#         env: Environment[State, ActionSpec, Observation],
#         reward_aggregator: Callable = jnp.sum,
#         discount_aggregator: Callable = jnp.max,
#     ):
#         """Create the wrapped environment.
#
#         Args:
#             env: `Environment` to wrap to a `dm_env.Environment`.
#             reward_aggregator: a function to aggregate all agents rewards into a single scalar
#                 value, e.g. sum.
#             discount_aggregator: a function to aggregate all agents discounts into a single
#                 scalar value, e.g. max.
#         """
#         super().__init__(env)
#         self._reward_aggregator = reward_aggregator
#         self._discount_aggregator = discount_aggregator
#
#     def _aggregate_timestep(self, timestep: TimeStep[Observation]) -> TimeStep[Observation]:
#         """Apply the reward and discount aggregator to a multi-agent
#             timestep object to create a new timestep object that consists
#             of a scalar reward and discount value.
#
#         Args:
#             timestep: the multi agent timestep object.
#
#         Return:
#             a single agent compatible timestep object."""
#
#         return TimeStep(
#             step_type=timestep.step_type,
#             observation=timestep.observation,
#             reward=self._reward_aggregator(timestep.reward),
#             discount=self._discount_aggregator(timestep.discount),
#             extras=timestep.extras,
#         )
#
#     def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
#         """Resets the environment to an initial state.
#
#         Args:
#             key: random key used to reset the environment.
#
#         Returns:
#             state: State object corresponding to the new state of the environment,
#             timestep: TimeStep object corresponding the first timestep returned by the environment,
#         """
#         state, timestep = self._env.reset(key)
#         timestep = self._aggregate_timestep(timestep)
#         return state, timestep
#
#     def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
#         """Run one timestep of the environment's dynamics.
#
#         The rewards are aggregated into a single value based on the given reward aggregator.
#         The discount value is set to the largest discount of all the agents. This
#         essentially means that if any single agent is alive, the discount value won't be zero.
#
#         Args:
#             state: State object containing the dynamics of the environment.
#             action: Array containing the action to take.
#
#         Returns:
#             state: State object corresponding to the next state of the environment,
#             timestep: TimeStep object corresponding the timestep returned by the environment,
#         """
#         state, timestep = self._env.step(state, action)
#         timestep = self._aggregate_timestep(timestep)
#         return state, timestep


class VmapWrapper(Wrapper[State, ActionSpec, Observation], Generic[State, ActionSpec, Observation]):
    """Vectorized Jax env.
    Please note that all methods that return arrays do not return a batch dimension because the
    batch size is not known to the VmapWrapper. Methods that omit the batch dimension include:
    - observation_spec
    - action_spec
    - reward_spec
    - discount_spec
    """

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment to an initial state.

        The first dimension of the key will dictate the number of concurrent environments.

        To obtain a key with the right first dimension, you may call `jax.random.split` on key
        with the parameter `num` representing the number of concurrent environments.

        Args:
            key: random keys used to reset the environments where the first dimension is the number
                of desired environments.

        Returns:
            state: State object corresponding to the new state of the environments,
            timestep: TimeStep object corresponding the first timesteps returned by the
                environments,
        """
        state, timestep = jax.vmap(self._env.reset)(key)
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        The first dimension of the state will dictate the number of concurrent environments.

        See `VmapWrapper.reset` for more details on how to get a state of concurrent
        environments.

        Args:
            state: State object containing the dynamics of the environments.
            action: Array containing the actions to take.

        Returns:
            state: State object corresponding to the next states of the environments,
            timestep: TimeStep object corresponding the timesteps returned by the environments,
        """
        state, timestep = jax.vmap(self._env.step)(state, action)
        return state, timestep

    def render(self, state: State) -> Any:
        """Render the first environment state of the given batch.
        The remaining elements of the batched state are ignored.

        Args:
            state: State object containing the current dynamics of the environment.
        """
        state_0 = tree_utils.tree_slice(state, 0)
        return super().render(state_0)


NEXT_OBS_KEY_IN_EXTRAS = "next_obs"


def add_obs_to_extras(timestep: TimeStep[Observation]) -> TimeStep[Observation]:
    """Place the observation in timestep.extras[NEXT_OBS_KEY_IN_EXTRAS].
    Used when auto-resetting to store the observation from the terminal TimeStep (useful for
    e.g. truncation).

    Args:
        timestep: TimeStep object containing the timestep returned by the environment.

    Returns:
        timestep where the observation is placed in timestep.extras["next_obs"].
    """
    extras = timestep.extras
    extras[NEXT_OBS_KEY_IN_EXTRAS] = timestep.observation
    return timestep.replace(extras=extras)  # type: ignore


class AutoResetWrapper(
    Wrapper[State, ActionSpec, Observation], Generic[State, ActionSpec, Observation]
):
    """Automatically resets environments that are done. Once the terminal state is reached,
    the state, observation, and step_type are reset. The observation and step_type of the
    terminal TimeStep is reset to the reset observation and StepType.LAST, respectively.
    The reward, discount, and extras retrieved from the transition to the terminal state.
    NOTE: The observation from the terminal TimeStep is stored in timestep.extras["next_obs"].
    WARNING: do not `jax.vmap` the wrapped environment (e.g. do not use with the `VmapWrapper`),
    which would lead to inefficient computation due to both the `step` and `reset` functions
    being processed each time `step` is called. Please use the `VmapAutoResetWrapper` instead.
    """

    def __init__(
        self,
        env: Environment[State, ActionSpec, Observation],
        next_obs_in_extras: bool = False,
    ):
        """Wrap an environment to automatically reset it when the episode terminates.

        Args:
            env: the environment to wrap.
            next_obs_in_extras: whether to store the next observation in the extras of the
                terminal timestep. This is useful for e.g. truncation.
        """
        super().__init__(env)
        self.next_obs_in_extras = next_obs_in_extras
        if next_obs_in_extras:
            self._maybe_add_obs_to_extras = add_obs_to_extras
        else:
            self._maybe_add_obs_to_extras = lambda timestep: timestep  # no-op

    def _auto_reset(
        self, state: State, timestep: TimeStep[Observation]
    ) -> Tuple[State, TimeStep[Observation]]:
        """Reset the state and overwrite `timestep.observation` with the reset observation
        if the episode has terminated.
        """
        if not hasattr(state, "key"):
            raise AttributeError(
                "This wrapper assumes that the state has attribute key which is used"
                " as the source of randomness for automatic reset"
            )

        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        key, _ = jax.random.split(state.key)  # type: ignore
        state, reset_timestep = self._env.reset(key)

        # Place original observation in extras.
        timestep = self._maybe_add_obs_to_extras(timestep)

        # Replace observation with reset observation.
        timestep = timestep.replace(observation=reset_timestep.observation)  # type: ignore

        return state, timestep

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        state, timestep = super().reset(key)
        timestep = self._maybe_add_obs_to_extras(timestep)
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Step the environment, with automatic resetting if the episode terminates."""
        state, timestep = self._env.step(state, action)

        # Overwrite the state and timestep appropriately if the episode terminates.
        state, timestep = jax.lax.cond(
            timestep.last(),
            self._auto_reset,
            lambda s, t: (s, self._maybe_add_obs_to_extras(t)),
            state,
            timestep,
        )

        return state, timestep


class VmapAutoResetWrapper(
    Wrapper[State, ActionSpec, Observation], Generic[State, ActionSpec, Observation]
):
    """Efficient combination of VmapWrapper and AutoResetWrapper, to be used as a replacement of
    the combination of both wrappers.
    `env = VmapAutoResetWrapper(env)` is equivalent to `env = VmapWrapper(AutoResetWrapper(env))`
    but is more efficient as it parallelizes homogeneous computation and does not run branches
    of the computational graph that are not needed (heterogeneous computation).
    - Homogeneous computation: call step function on all environments in the batch.
    - Heterogeneous computation: conditional auto-reset (call reset function for some environments
        within the batch because they have terminated).
    NOTE: The observation from the terminal TimeStep is stored in timestep.extras["next_obs"].
    """

    def __init__(
        self,
        env: Environment[State, ActionSpec, Observation],
        next_obs_in_extras: bool = False,
    ):
        """Wrap an environment to vmap it and automatically reset it when the episode terminates.

        Args:
            env: the environment to wrap.
            next_obs_in_extras: whether to store the next observation in the extras of the
                terminal timestep. This is useful for e.g. truncation.
        """
        super().__init__(env)
        self.next_obs_in_extras = next_obs_in_extras
        if next_obs_in_extras:
            self._maybe_add_obs_to_extras = add_obs_to_extras
        else:
            self._maybe_add_obs_to_extras = lambda timestep: timestep  # no-op

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets a batch of environments to initial states.

        The first dimension of the key will dictate the number of concurrent environments.

        To obtain a key with the right first dimension, you may call `jax.random.split` on key
        with the parameter `num` representing the number of concurrent environments.

        Args:
            key: random keys used to reset the environments where the first dimension is the number
                of desired environments.

        Returns:
            state: `State` object corresponding to the new state of the environments,
            timestep: `TimeStep` object corresponding the first timesteps returned by the
                environments,
        """
        state, timestep = jax.vmap(self._env.reset)(key)
        timestep = self._maybe_add_obs_to_extras(timestep)
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of all environments' dynamics. It automatically resets environment(s)
        in which episodes have terminated.

        The first dimension of the state will dictate the number of concurrent environments.

        See `VmapAutoResetWrapper.reset` for more details on how to get a state of concurrent
        environments.

        Args:
            state: `State` object containing the dynamics of the environments.
            action: `Array` containing the actions to take.

        Returns:
            state: `State` object corresponding to the next states of the environments.
            timestep: `TimeStep` object corresponding the timesteps returned by the environments.
        """
        # Vmap homogeneous computation (parallelizable).
        state, timestep = jax.vmap(self._env.step)(state, action)
        # Map heterogeneous computation (non-parallelizable).
        state, timestep = jax.lax.map(lambda args: self._maybe_reset(*args), (state, timestep))
        return state, timestep

    def _auto_reset(
        self, state: State, timestep: TimeStep[Observation]
    ) -> Tuple[State, TimeStep[Observation]]:
        """Reset the state and overwrite `timestep.observation` with the reset observation
        if the episode has terminated.
        """
        if not hasattr(state, "key"):
            raise AttributeError(
                "This wrapper assumes that the state has attribute key which is used"
                " as the source of randomness for automatic reset"
            )

        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        key, _ = jax.random.split(state.key)
        state, reset_timestep = self._env.reset(key)

        # Place original observation in extras.
        timestep = self._maybe_add_obs_to_extras(timestep)

        # Replace observation with reset observation.
        timestep = timestep.replace(  # type: ignore
            observation=reset_timestep.observation
        )

        return state, timestep

    def _maybe_reset(
        self, state: State, timestep: TimeStep[Observation]
    ) -> Tuple[State, TimeStep[Observation]]:
        """Overwrite the state and timestep appropriately if the episode terminates."""
        state, timestep = jax.lax.cond(
            timestep.last(),
            self._auto_reset,
            lambda s, t: (s, self._maybe_add_obs_to_extras(t)),
            state,
            timestep,
        )

        return state, timestep

    def render(self, state: State) -> Any:
        """Render the first environment state of the given batch.
        The remaining elements of the batched state are ignored.

        Args:
            state: State object containing the current dynamics of the environment.
        """
        state_0 = tree_utils.tree_slice(state, 0)
        return super().render(state_0)

#
# class JumanjiToGymWrapper(gym.Env, Generic[State, ActionSpec, Observation]):
#     """A wrapper that converts a Jumanji `Environment` to one that follows the `gym.Env` API."""
#
#     def __init__(
#         self,
#         env: Environment[State, ActionSpec, Observation],
#         seed: int = 0,
#         backend: Optional[str] = None,
#     ):
#         """Create the Gym environment.
#
#         Args:
#             env: `Environment` to wrap to a `gym.Env`.
#             seed: the seed that is used to initialize the environment's PRNG.
#             backend: the XLA backend.
#         """
#         self._env = env
#         self.metadata: Dict[str, str] = {}
#         self._key = jax.random.PRNGKey(seed)
#         self.backend = backend
#         self._state = None
#         self.observation_space = specs.jumanji_specs_to_gym_spaces(self._env.observation_spec)
#         self.action_space = specs.jumanji_specs_to_gym_spaces(self._env.action_spec)
#
#         def reset(key: chex.PRNGKey) -> Tuple[State, Observation, Optional[Dict]]:
#             """Reset function of a Jumanji environment to be jitted."""
#             state, timestep = self._env.reset(key)
#             return state, timestep.observation, timestep.extras
#
#         self._reset = jax.jit(reset, backend=self.backend)
#
#         def step(
#             state: State, action: chex.Array
#         ) -> Tuple[State, Observation, chex.Array, chex.Array, chex.Array, Optional[Any]]:
#             """Step function of a Jumanji environment to be jitted."""
#             state, timestep = self._env.step(state, action)
#             term = ~timestep.discount.astype(bool)
#             trunc = timestep.last().astype(bool)
#             return state, timestep.observation, timestep.reward, term, trunc, timestep.extras
#
#         self._step = jax.jit(step, backend=self.backend)
#
#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         options: Optional[dict] = None,
#     ) -> Tuple[GymObservation, Dict[str, Any]]:
#         """Resets the environment to an initial state by starting a new sequence
#         and returns the first `Observation` of this sequence.
#
#         Returns:
#             obs: an element of the environment's observation_space.
#             info (optional): contains supplementary information such as metrics.
#         """
#         if seed is not None:
#             self.seed(seed)
#         key, self._key = jax.random.split(self._key)
#         self._state, obs, extras = self._reset(key)
#
#         # Convert the observation to a numpy array or a nested dict thereof
#         obs = jumanji_to_gym_obs(obs)
#
#         return obs, jax.device_get(extras)
#
#     def step(
#         self, action: chex.ArrayNumpy
#     ) -> Tuple[GymObservation, float, bool, bool, Dict[str, Any]]:
#         """Updates the environment according to the action and returns an `Observation`.
#
#         Args:
#             action: A NumPy array representing the action provided by the agent.
#
#         Returns:
#             observation: an element of the environment's observation_space.
#             reward: the amount of reward returned as a result of taking the action.
#             terminated: whether a terminal state is reached.
#             info: contains supplementary information such as metrics.
#         """
#
#         action_jax = jnp.asarray(action)  # Convert input numpy array to JAX array
#         self._state, obs, reward, term, trunc, extras = self._step(self._state, action_jax)
#
#         # Convert to get the correct signature
#         obs = jumanji_to_gym_obs(obs)
#         reward = float(reward)
#         terminated = bool(term)
#         truncated = bool(trunc)
#         info = jax.device_get(extras)
#
#         return obs, reward, terminated, truncated, info
#
#     def seed(self, seed: int = 0) -> None:
#         """Function which sets the seed for the environment's random number generator(s).
#
#         Args:
#             seed: the seed value for the random number generator(s).
#         """
#         self._key = jax.random.PRNGKey(seed)
#
#     def render(self, mode: str = "human") -> Any:
#         """Renders the environment.
#
#         Args:
#             mode: currently not used since Jumanji does not currently support modes.
#         """
#         del mode
#         if self._state is None:
#             raise ValueError("Cannot render when _state is None.")
#         return self._env.render(self._state)
#
#     def close(self) -> None:
#         """Closes the environment, important for rendering where pygame is imported."""
#         self._env.close()
#
#     @property
#     def unwrapped(self) -> Environment[State, ActionSpec, Observation]:
#         return self._env
#
#
# def jumanji_to_gym_obs(observation: Observation) -> GymObservation:
#     """Convert a Jumanji observation into a gym observation.
#
#     Args:
#         observation: JAX pytree with (possibly nested) containers that
#             either have the `__dict__` or `_asdict` methods implemented.
#
#     Returns:
#         Numpy array or nested dictionary of numpy arrays.
#     """
#     if isinstance(observation, jnp.ndarray):
#         return np.asarray(observation)
#     elif hasattr(observation, "__dict__"):
#         # Applies to various containers including `chex.dataclass`
#         return {key: jumanji_to_gym_obs(value) for key, value in vars(observation).items()}
#     elif hasattr(observation, "_asdict"):
#         # Applies to `NamedTuple` container.
#         return {
#             key: jumanji_to_gym_obs(value)
#             for key, value in observation._asdict().items()  # type: ignore
#         }
#     else:
#         raise NotImplementedError(
#             "Conversion only implemented for JAX pytrees with (possibly nested) containers "
#             "that either have the `__dict__` or `_asdict` methods implemented."
#         )
