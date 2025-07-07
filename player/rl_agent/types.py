import jax.numpy as jnp
import chex
from flax import struct


from .memory.memory_dataclass import MemoryState


@struct.dataclass
class AgentState:
    """All information we want to have for the agent to make decision. It is updated according to new observation in Agent.update_state"""
    player_id: int  # true id (can be 0 or 1)
    memory: MemoryState
    last_lux_action: jnp.array
    last_nn_action: jnp.array
    # it is possible to add other fields (like whether the game has just been won/lost, or previous memory, or previous obs, etc) and to update them in Agent.update_state


# ----------- BELOW: COPIED FROM LUX: luxai_s3/state.py

@struct.dataclass
class UnitState:
    position: chex.Array
    """Position of the unit with shape (2) for x, y"""
    energy: int
    """Energy of the unit"""


@struct.dataclass
class MapTile:
    energy: int
    """Energy of the tile, generated via energy_nodes and energy_node_fns"""
    tile_type: int
    """Type of the tile"""


@struct.dataclass
class LuxEnvObs:
    """Partial observation of environment, copied from Lux"""
    units: UnitState
    """Units in the environment with shape (T, N, 3) for T teams, N max units, and 3 features.

    3 features are for position (x, y), and energy
    """
    units_mask: chex.Array
    """Mask of units in the environment with shape (T, N) for T teams, N max units"""

    sensor_mask: chex.Array

    map_features: MapTile
    """Map features in the environment with shape (W, H, 2) for W width, H height
    """
    relic_nodes: chex.Array
    """Position of all relic nodes with shape (N, 2) for N max relic nodes and 2 features for position (x, y). Number is -1 if not visible"""
    relic_nodes_mask: chex.Array
    """Mask of all relic nodes with shape (N) for N max relic nodes"""
    team_points: chex.Array
    """Team points in the environment with shape (T) for T teams"""
    team_wins: chex.Array
    """Team wins in the environment with shape (T) for T teams"""
    steps: int = 0
    """steps taken in the environment"""
    match_steps: int = 0
    """steps taken in the current match"""

