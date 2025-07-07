import jax
import jax.numpy as jnp
import functools
from flax import struct
from omegaconf import DictConfig

from .types import LuxEnvObs

@struct.dataclass
class AgentState:
    """All information we want to have for the agent to make decision. It is updated according to new observation in Agent.update_state"""
    player_id: int


class Agent:

    def __init__(self, config_agent: DictConfig, path, training=False):
        pass

    def init(self, player_id: int, env_cfg: dict) -> AgentState:
        agent_state = AgentState(
            player_id=player_id,
        )
        return agent_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def act(self, seed: int, agent_state: AgentState, obs: LuxEnvObs):
        i = jnp.int16(obs.match_steps)
        action = (1 + ((1 + i % 2) + jnp.arange(16, dtype=jnp.int16) % (1 + i // 8)) % 5) * jnp.ones((16,), dtype=jnp.int16)
        action = jnp.column_stack((action, jnp.zeros((16, 2), dtype=jnp.int16)))
        return agent_state, action  # action: array(16,3)

