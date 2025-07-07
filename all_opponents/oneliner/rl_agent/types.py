import jax.numpy as jnp
import chex
from typing import NamedTuple
from flax import struct


class NNInput(NamedTuple):
    """The object feed to the neural networks (actor or critic). Fields are stacked along first dimension, not last."""
    scalars: jnp.array  # vector of scalar values
    fields: jnp.array  # array with fields as channels # STACKED ALONG FIRST DIMENSION
    base_action_masks: jnp.array  # mask for base actions (N_MAX_UNITS, N_BASE_ACTIONS)
    sap_action_masks: jnp.array  # mask for sap actions (N_MAX_UNITS, N_SAP_ACTIONS)


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

