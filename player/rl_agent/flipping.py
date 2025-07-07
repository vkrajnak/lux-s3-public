import jax
import jax.numpy as jnp
from copy import deepcopy
from .constants import PLAYER_ID
from .types import LuxEnvObs
from .constants import GRID_SHAPE


def flip_obs(player_id: int, obs: LuxEnvObs) -> LuxEnvObs:
    return jax.lax.cond(player_id == PLAYER_ID, lambda o: o, _flip_obs, obs)


def flip_action(player_id: int, action: jnp.array) -> jnp.array:
    return jax.lax.cond(player_id == PLAYER_ID, lambda a: a, _flip_action, action)


def _flip_obs(obs: LuxEnvObs) -> LuxEnvObs:
    x_max = jnp.int16(GRID_SHAPE[0] - 1)

    flipped_obs = deepcopy(obs)

    flipped_units = deepcopy(obs.units)
    flipped_units = flipped_units.replace(
        position=jnp.where(
            obs.units_mask[::-1, :, None],
            x_max - obs.units.position[::-1, :, ::-1],
            obs.units.position[::-1]
        ),
        energy=obs.units.energy[::-1]
    )

    flipped_map_features = deepcopy(obs.map_features)
    flipped_map_features = flipped_map_features.replace(
        energy=obs.map_features.energy[::-1, ::-1].T,
        tile_type=obs.map_features.tile_type[::-1, ::-1].T
    )

    flipped_obs = flipped_obs.replace(
        units=flipped_units,
        units_mask=obs.units_mask[::-1],
        sensor_mask=obs.sensor_mask[::-1, ::-1].T,
        map_features=flipped_map_features,
        relic_nodes=jnp.where(
            obs.relic_nodes_mask[:, None],
            x_max - obs.relic_nodes[:, ::-1],
            obs.relic_nodes
        ),
        team_points=obs.team_points[::-1],
        team_wins=obs.team_wins[::-1],
    )

    return flipped_obs


def _flip_action(action: jnp.array) -> jnp.array:
    convertor = jnp.array([0, 2, 1, 4, 3, 5], dtype=jnp.int16)

    flipped_act = convertor[action[:, 0]]
    flipped_sap = -action[:, 2:0:-1]

    flipped_action = jnp.column_stack((flipped_act, flipped_sap))
    return jnp.array(flipped_action, dtype=jnp.int16)
