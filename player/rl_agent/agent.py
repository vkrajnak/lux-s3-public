import os
import sys
import jax
import jax.numpy as jnp
import functools
from omegaconf import DictConfig
from typing import Union

from .network import Actor, Critic
from .network_input.get_inputs import get_build_input_from_agent_state, get_input_spec
from .memory.memory_jax import (memory_init, memory_update)
from .types import LuxEnvObs, AgentState
from .flipping import flip_obs, flip_action
from .constants import N_BASE_ACTIONS, N_SAP_ACTIONS, N_MAX_UNITS, ENV_PARAMS_RANGES, PLAYER_ID, GRID_SHAPE

import pickle
from flax.serialization import to_state_dict, from_state_dict

SAP_MAX_RANGE = max(ENV_PARAMS_RANGES["unit_sap_range"])
SAP_SQUARE_SIZE = 2 * SAP_MAX_RANGE + 1


class Agent:

    def __init__(self, config_agent: DictConfig, path: Union[str, None],  training=False, print_nn=False):
        self.greedy = config_agent.greedy
        self.force = config_agent.force

        self._build_input_from_agent_state = get_build_input_from_agent_state(
            config_agent.input.fields_as_fields,
            config_agent.input.scalars_as_fields,
            config_agent.head
        )

        self.input_spec = get_input_spec(
            config_agent.input.fields_as_fields,
            config_agent.input.scalars_as_fields,
            config_agent.head
        )

        self._build_action_from_sample_nn_policy = get_build_action_from_sample_nn_policy(config_agent.head)

        self.get_dummy_nn_sample = get_dummy_nn_sample_fn(config_agent.head)

        if not training:
            self.actor_network = Actor(
                torso=config_agent.torso.name,
                torso_kwargs=config_agent.torso.kwargs,
                head=config_agent.head
            )

            # Get dummy params to make sure the shape is right (needed to reload)
            obs = self.input_spec.generate_value()
            actor_params_dummy = self.actor_network.init(jax.random.key(0), obs)

            # Reload params, if no path given then use dummy params
            if path is None:
                self.actor_params = actor_params_dummy
                self.is_dummy = True
            else:
                with open(os.path.join(path, 'actor_params.pkl'), 'rb') as fp:
                    reloaded_actor_params_dict = pickle.load(fp)
                self.actor_params = from_state_dict(actor_params_dummy, reloaded_actor_params_dict)
                self.is_dummy = False

        if print_nn:
            import flax
            obs = self.input_spec.generate_value()

            dummy_actor_network = Actor(
                torso=config_agent.torso.name,
                torso_kwargs=config_agent.torso.kwargs,
                head=config_agent.head,
                print_arch=True
            )
            tabulate_fn = flax.linen.tabulate(dummy_actor_network, jax.random.PRNGKey(0), console_kwargs={'width': 400})
            print(tabulate_fn(obs), file=sys.stderr)

            dummy_critic_network = Critic(
                torso=config_agent.torso.name,
                torso_kwargs=config_agent.torso.kwargs
            )
            tabulate_fn = flax.linen.tabulate(dummy_critic_network, jax.random.PRNGKey(0), console_kwargs={'width': 400})
            print(tabulate_fn(obs), file=sys.stderr)

    def init(self, player_id: int, env_cfg: dict) -> AgentState:
        """Called at the beginning of a new episode"""
        agent_state = AgentState(
            player_id=player_id,  # keep real id only for applying flipping
            memory=memory_init(PLAYER_ID, env_cfg, self.force),  # always playing as PLAYER_ID
            last_lux_action=jnp.zeros((N_MAX_UNITS, 3), dtype=jnp.int16),
            last_nn_action=self.get_dummy_nn_sample(),
        )
        return agent_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def act(self, seed: int, agent_state: AgentState, obs: LuxEnvObs):
        """Called at each step. [Never called during by the learning agent, but called when acting as opponent or Kaggle agent]"""

        # convert lux obs into nn_input and update memory -- same is done in OnePlayerEnv
        agent_state, nn_input = self.process_obs_for_nn_and_update_state(agent_state, obs)
        
        # call the nn to get the policy and sample for it -- this part is done in the learning algorithm
        nn_policy = self.actor_network.apply(self.actor_params, nn_input)

        def get_action_deterministic():
            nn_action_greedy = nn_policy.mode()  # (2, N_MAX_UNITS) of int between 0 and max_value
            return nn_action_greedy

        def get_action_stochastic():
            key = jax.random.key(seed)
            nn_action_sample = nn_policy.sample(key)  # (2, N_MAX_UNITS) of int between 0 and max_value
            return nn_action_sample

        nn_action = jax.lax.cond(self.greedy, get_action_deterministic, get_action_stochastic)

        # convert the nn_action into a lux action -- same is done in OnePlayerEnv
        agent_state, action = self.process_action_from_nn_and_update_state(agent_state, nn_action)
        
        # i = jnp.int16(obs.match_steps)
        # action = (1+((1+i%2) + jnp.arange(16, dtype=jnp.int16)%(1+i//8))%5)*jnp.ones((16,), dtype=jnp.int16)
        # action = jnp.column_stack((action,jnp.zeros((16,2), dtype=jnp.int16)))
        # agent_state = agent_state.replace(
        #     last_action=action,
        # )
        return agent_state, action  # action: array(16,3)

    def process_obs_for_nn_and_update_state(self, agent_state: AgentState, obs: LuxEnvObs):
        flipped_obs = flip_obs(agent_state.player_id, obs)
        memory = memory_update(agent_state.memory, flipped_obs, agent_state.last_lux_action)
        agent_state = agent_state.replace(
            memory=memory
        )
        nn_input = self._build_input_from_agent_state(agent_state)
        return agent_state, nn_input

    def process_action_from_nn_and_update_state(self, agent_state: AgentState, nn_action: jnp.array):
        action = self._build_action_from_sample_nn_policy(nn_action, agent_state)
        agent_state = agent_state.replace(
            last_lux_action=action,
            last_nn_action=nn_action.astype(jnp.int16),
        )
        flipped_action = flip_action(agent_state.player_id, action)
        return agent_state, flipped_action


def get_build_action_from_sample_nn_policy(action_head: str):
    def build_action_discrete_head(nn_action: jnp.array, agent_state: AgentState):
        # nn_action: (2, N_MAX_UNITS)
        def _convert_sap_action(a: jnp.array) -> jnp.array:
            # a between 0 and N_SAP_ACTIONS - 1
            x = a // SAP_SQUARE_SIZE - SAP_MAX_RANGE
            y = a % SAP_SQUARE_SIZE - SAP_MAX_RANGE
            return jnp.array((x, y))

        assert nn_action.shape == (2, N_MAX_UNITS)
        action = jnp.swapaxes(nn_action, -1, -2)  # (N_MAX_UNITS, 2)
        action_base = action[:, 0, None]  # values between 0 and N_BASE_ACTIONS - 1  # (N_MAX_UNITS, 1)
        action_sap = action[:, 1]  # values between 0 and N_SAP_ACTIONS - 1  # (N_MAX_UNITS,)
        action_sap = jax.vmap(_convert_sap_action)(action_sap)  # in sap range  # (N_MAX_UNITS, 2)
        return jnp.concatenate([action_base, action_sap], axis=-1, dtype=jnp.int16)  # (N_MAX_UNITS, 3)

    def build_action_monofield_head(nn_action: jnp.array, agent_state: AgentState):
        # nn_action: (N_MAX_UNITS)
        def _convert_to_action(a: jnp.array) -> jnp.array:
            # a between 0 and N_CELLS = GRID_SHAPE[0] * GRID_SHAPE[1]
            x = a // GRID_SHAPE[0]
            y = a % GRID_SHAPE[1]
            enum = jnp.arange(N_MAX_UNITS)

            action = agent_state.memory.monofield_base_converter[enum,x,y]
            sap_x = jnp.where(
                action == 5,
                agent_state.memory.relative_coordinate_maps[enum,x,y,0],
                0
            )
            sap_y = jnp.where(
                action == 5,
                agent_state.memory.relative_coordinate_maps[enum,x,y,1],
                0
            )

            return jnp.column_stack((action, sap_x, sap_y))

        assert nn_action.shape == (N_MAX_UNITS, )

        action = _convert_to_action(nn_action)  # (N_MAX_UNITS, 2)

        return action
    
    def build_action_discrete_and_field_head(nn_action: jnp.array, agent_state: AgentState):
        # nn_action: (2, N_MAX_UNITS)
        
        def _convert_sap_action(action: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
            # a between 0 and N_CELLS = GRID_SHAPE[0] * GRID_SHAPE[1]
            x = a // GRID_SHAPE[0]
            y = a % GRID_SHAPE[1]
            enum = jnp.arange(N_MAX_UNITS)
            action = action.reshape((1,1,N_MAX_UNITS))

            sap_x = jnp.where(
                action == 5,
                agent_state.memory.relative_coordinate_maps[enum,x,y,0],
                jnp.int16(0)
            ).reshape((N_MAX_UNITS,1))
            sap_y = jnp.where(
                action == 5,
                agent_state.memory.relative_coordinate_maps[enum,x,y,1],
                jnp.int16(0)
            ).reshape((N_MAX_UNITS,1))
            
            return jnp.column_stack((sap_x, sap_y))

        assert nn_action.shape == (2, N_MAX_UNITS)
        action = jnp.swapaxes(nn_action, -1, -2)  # (N_MAX_UNITS, 2)
        action_base = action[:, 0, None]  # values between 0 and N_BASE_ACTIONS - 1  # (N_MAX_UNITS, 1)
        action_sap = action[:, 1]  # values between 0 and N_CELLS = GRID_SHAPE[0] * GRID_SHAPE[1] # (N_MAX_UNITS,)
        
        action_sap = _convert_sap_action(action_base, action_sap)  # (N_MAX_UNITS, 2)

        return jnp.concatenate([action_base, action_sap], axis=-1, dtype=jnp.int16)  # (N_MAX_UNITS, 3)

    select_build_action = {
        "discrete_full": build_action_discrete_head,
        "discrete_sparse": build_action_discrete_head,
        "monofield": build_action_monofield_head,
        "duofield": build_action_discrete_and_field_head,
        "duomofield":  build_action_discrete_and_field_head,
        "discrete_and_field": build_action_discrete_and_field_head,
    }

    return select_build_action[action_head]


def get_dummy_nn_sample_fn(head_name):
    def get_dummy_nn_sample():
        if head_name == "monofield":
            return jnp.zeros((N_MAX_UNITS,), dtype=jnp.int16)
        elif head_name in ("duofield", "duomofield", "discrete_and_field", "discrete_full", "discrete_sparse"):
            return jnp.zeros((2, N_MAX_UNITS), dtype=jnp.int16)
        else:
            raise Exception("this head_name is not implemented")

    return get_dummy_nn_sample



