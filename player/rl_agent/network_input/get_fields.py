import jax
import jax.numpy as jnp

from ..types import AgentState
from ..constants import GRID_SHAPE, ENV_PARAMS_FIXED, ENV_PARAMS_RANGES, N_MAX_UNITS
from .types import DTYPE
from ..memory.memory_jax import place_points_on_field, place_points_on_map


# --- public functions

def get_zero(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return jnp.zeros(GRID_SHAPE + (1,), dtype=DTYPE)


def get_one(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return jnp.ones(GRID_SHAPE + (1,), dtype=DTYPE)


# # units

def get_player_unit_field_and_mask(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 32)
    field, mask = _place_player_unit_energies_on_field(agent_state)
    return jnp.concatenate((field, mask), axis=2, dtype=DTYPE)


def get_opponent_unit_field_and_mask(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 32)
    field, mask = _place_opponent_unit_energies_on_field(agent_state)
    return jnp.concatenate((field, mask), axis=2, dtype=DTYPE)


def get_player_condensed_unit_field(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 16)
    field, mask = _place_player_unit_energies_on_field(agent_state)
    field = jnp.where(
        mask,
        field,
        -1
    )
    return field.astype(DTYPE)


def get_player_condensed_unit_field_previous(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 16)
    field, mask = _place_player_unit_energies_on_field_previous(agent_state)
    field = jnp.where(
        mask,
        field,
        -1
    )
    return field.astype(DTYPE)


def get_player_condensed_unit_map_previous(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    mask = agent_state.memory.previous_n_units_map > 0
    field = jnp.where(
        mask * (agent_state.memory.previous_energy_map >= 0),
        agent_state.memory.previous_energy_map/ENV_PARAMS_FIXED.max_unit_energy,
        -1
    )
    return field.astype(DTYPE)[:, :, None]


def get_player_condensed_summed_energy_and_n_units_maps(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 2)
    n_units_field = agent_state.memory.n_units_map / N_MAX_UNITS
    mask = agent_state.memory.n_units_map > 0

    summed_energy_field = jnp.where(
        mask * (agent_state.memory.energy_map >= 0),
        agent_state.memory.energy_map/ENV_PARAMS_FIXED.max_unit_energy,
        -1
    )
    n_units_field = jnp.where(
        mask,
        n_units_field,
        -1
    )
    return jnp.concatenate((summed_energy_field[:, :, None], n_units_field[:, :, None]), axis=2, dtype=DTYPE)


def get_player_condensed_summed_energy_and_n_units_maps_previous(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 2)
    n_units_field = agent_state.memory.previous_n_units_map / N_MAX_UNITS
    mask = agent_state.memory.previous_n_units_map > 0

    summed_energy_field = jnp.where(
        mask * (agent_state.memory.previous_energy_map >= 0),
        agent_state.memory.previous_energy_map/ENV_PARAMS_FIXED.max_unit_energy,
        -1
    )
    n_units_field = jnp.where(
        mask,
        n_units_field,
        -1
    )
    return jnp.concatenate((summed_energy_field[:, :, None], n_units_field[:, :, None]), axis=2, dtype=DTYPE)


def get_opponent_condensed_unit_field(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 16)
    field, mask = _place_opponent_unit_energies_on_field(agent_state)
    field = jnp.where(
        mask,
        field,
        - 1
    )
    return field.astype(DTYPE)


def get_opponent_condensed_unit_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    field = jnp.where(
        (agent_state.memory.n_opp_units_map > 0) * (agent_state.memory.opp_energy_map >= 0),
        agent_state.memory.opp_energy_map/ENV_PARAMS_FIXED.max_unit_energy,
        -1
    )
    return field.astype(DTYPE)[:, :, None]


def get_opponent_unit_map_and_mask(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 2)
    field = jnp.where(
        agent_state.memory.opp_energy_map >= 0,
        agent_state.memory.opp_energy_map/ENV_PARAMS_FIXED.max_unit_energy,
        -1
    )
    mask = agent_state.memory.n_opp_units_map > 0
    return jnp.concatenate((field[:, :, None], mask[:, :, None]), axis=2, dtype=DTYPE)


def get_opponent_condensed_summed_energy_and_n_units_maps(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 2)
    n_units_field = agent_state.memory.n_opp_units_map / N_MAX_UNITS
    mask = agent_state.memory.n_opp_units_map > 0

    summed_energy_field = jnp.where(
        mask * (agent_state.memory.opp_energy_map >= 0),
        agent_state.memory.opp_energy_map/ENV_PARAMS_FIXED.max_unit_energy,
        -1
    )
    n_units_field = jnp.where(
        mask,
        n_units_field,
        -1
    )
    return jnp.concatenate((summed_energy_field[:, :, None], n_units_field[:, :, None]), axis=2, dtype=DTYPE)


def get_opponent_condensed_summed_energy_and_n_units_maps_previous(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 2)
    n_units_field = agent_state.memory.previous_n_opp_units_map / N_MAX_UNITS
    mask = agent_state.memory.previous_n_opp_units_map > 0

    summed_energy_field = jnp.where(
        mask * (agent_state.memory.previous_opp_energy_map >= 0),
        agent_state.memory.previous_opp_energy_map/ENV_PARAMS_FIXED.max_unit_energy,
        -1
    )
    n_units_field = jnp.where(
        mask,
        n_units_field,
        -1
    )
    
    return jnp.concatenate((summed_energy_field[:, :, None], n_units_field[:, :, None]), axis=2, dtype=DTYPE)


def get_contact_efficiency_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    """proportional to damage done by collision/adjacent energy void if moving into a cell (not accounting for whether it is actually possible to move there), in [0,1]"""
    return agent_state.memory.contact_efficiency_map.astype(DTYPE)[:, :, None]


def get_sap_efficiency_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return (agent_state.memory.sap_efficiency_map / N_MAX_UNITS).astype(DTYPE)[:, :, None]


def get_invisible_point_tiles_sap_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return (agent_state.memory.invisible_point_tiles_sap_map / N_MAX_UNITS).astype(DTYPE)[:, :, None]

# # points


def get_relic_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return agent_state.memory.relics_map.astype(DTYPE)[:, :, None]


def get_point_tile_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return agent_state.memory.point_tiles.astype(DTYPE)[:, :, None]


def get_point_tile_occupancy_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    """1 if occupied by player, -1 if occupied by opponent, 0 otherwise"""
    player_occupied = jnp.logical_and(agent_state.memory.point_tiles, agent_state.memory.unit_map).astype(DTYPE)[:, :, None]
    opponent_occupied = jnp.logical_and(agent_state.memory.point_tiles, agent_state.memory.opp_unit_map).astype(DTYPE)[:, :, None]
    return (player_occupied - opponent_occupied).astype(DTYPE)


def get_point_tile_occupancy_map_previous(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    """1 if occupied by player, -1 if occupied by opponent, 0 otherwise; at previous step"""
    player_unit_map_previous = place_points_on_map(
        agent_state.memory.false_map16,
        agent_state.memory.previous_unit_positions,
        agent_state.memory.previous_unit_mask
    )
    opponent_unit_map_previous = place_points_on_map(
        agent_state.memory.false_map16,
        agent_state.memory.previous_opp_unit_positions,
        agent_state.memory.previous_opp_unit_energy >= 0
    )
    player_occupied = jnp.logical_and(agent_state.memory.point_tiles, player_unit_map_previous).astype(DTYPE)[:, :, None]
    opponent_occupied = jnp.logical_and(agent_state.memory.point_tiles, opponent_unit_map_previous).astype(DTYPE)[:, :, None]
    return (player_occupied - opponent_occupied).astype(DTYPE)


def get_potential_point_tile_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return agent_state.memory.potential_point_tiles.astype(DTYPE)[:, :, None]


def get_empty_tile_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return agent_state.memory.empty_tiles.astype(DTYPE)[:, :, None]


def get_point_field(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    relics = get_relic_map(agent_state)
    points = get_point_tile_map(agent_state)
    potential = get_potential_point_tile_map(agent_state)
    empty = get_empty_tile_map(agent_state)
    field = (empty + 2 * relics + 3 * potential + 4 * points) / 4
    return field.astype(DTYPE)


def get_approx_dist_to_point_tiles(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    approx_dist = jax.lax.cond(
        jnp.any(agent_state.memory.approx_dist_to_point_tiles == -1),
        lambda: agent_state.memory.approx_dist_to_point_tiles.astype(DTYPE)[:, :, None],
        lambda: agent_state.memory.approx_dist_to_point_tiles.astype(DTYPE)[:, :, None] / GRID_SHAPE[0]
    )
    return approx_dist


# # board

def get_energy_field_and_mask(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 2)
    """ Tile energy is normalised by max_energy_per_tile, thus to [-1,1] """
    energy = agent_state.memory.energy_field / ENV_PARAMS_FIXED.max_energy_per_tile
    mask = agent_state.memory.valid_energy_mask
    return jnp.concatenate((energy[:, :, None], mask[:, :, None]), axis=2, dtype=DTYPE)


def _energy_map_including_nebula_and_asteroid(agent_state: AgentState, asteroid: bool) -> jnp.ndarray:
    """map of energy, accounting for nebula energy reduction and associating asteroid / unknown tile energy to very low energy (asteroid is optional)
    -- unscaled version, meant to be called from other get_... functions"""
    combined_energy = agent_state.memory.energy_field_including_nebula.astype(DTYPE)  # (24, 24)
    combined_energy_including_asteroid = jnp.where(
        agent_state.memory.tile_type_field < 2,
        combined_energy,
        DTYPE(ENV_PARAMS_FIXED.min_energy_per_tile),  # if asteroid, set to min energy value
    )
    energy = jax.lax.cond(asteroid, lambda: combined_energy_including_asteroid, lambda: combined_energy)
    return energy.astype(DTYPE)  # (24, 24)


def get_energy_map_including_nebula(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    """NOTE: scaled by unit energy (400), not tile energy (20)! So very small values"""
    energy = _energy_map_including_nebula_and_asteroid(agent_state, asteroid=False) / ENV_PARAMS_FIXED.max_unit_energy
    return energy.astype(DTYPE)[:, :, None]


def get_positive_energy_mask(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    energy = _energy_map_including_nebula_and_asteroid(agent_state, asteroid=False)
    return (energy > 0).astype(DTYPE)[:, :, None]


def get_high_energy_mask(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    energy = _energy_map_including_nebula_and_asteroid(agent_state, asteroid=False)
    return (energy > 5).astype(DTYPE)[:, :, None]


def get_very_high_energy_mask(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    energy = _energy_map_including_nebula_and_asteroid(agent_state, asteroid=False)
    return (energy > 8).astype(DTYPE)[:, :, None]


def get_tile_type_field(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    """ Values in [-1.0,2.0] - note that this is not the one used for embedding """
    return agent_state.memory.tile_type_field.astype(DTYPE)[:, :, None]


def get_next_tile_type_field(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    """ Values in [-1.0,2.0] - note that this is not the one used for embedding """
    return agent_state.memory.next_tile_type_field.astype(DTYPE)[:, :, None]


def get_explored_tiles_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return agent_state.memory.explored_for_relic_mask.astype(DTYPE)[:, :, None]


def get_player_recently_visited_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return agent_state.memory.recently_visited_map.astype(DTYPE)[:, :, None]


def get_opponent_recently_visited_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return agent_state.memory.opp_recently_visited_map.astype(DTYPE)[:, :, None]


def get_nebula_energy_reduction(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    """ Tile energy is normalised by nebula_tile_energy_reduction, with -1 where we don't know, thus to [-1,1] """
    nebula_field = (agent_state.memory.tile_type_field == 1).astype(DTYPE)
    reduction = jnp.where(
        agent_state.memory.nebula_energy_reduction == -1,
        agent_state.memory.nebula_energy_reduction,
        agent_state.memory.nebula_energy_reduction / ENV_PARAMS_RANGES["nebula_tile_energy_reduction"][-1]
    )
    return (nebula_field * reduction).astype(DTYPE)[:, :, None]


def get_nebula_vision_reduction(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    """ Tile energy is normalised by nebula_tile_energy_reduction, with -1 where we don't know, thus to [-1,1] """
    nebula_field = (agent_state.memory.tile_type_field == 1).astype(DTYPE)
    reduction = jnp.where(
        agent_state.memory.nebula_vision_reduction == -1,
        agent_state.memory.nebula_vision_reduction,
        agent_state.memory.nebula_vision_reduction / ENV_PARAMS_RANGES["nebula_tile_vision_reduction"][-1]
    )
    return (nebula_field * reduction).astype(DTYPE)[:, :, None]


def get_vision_map(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return (agent_state.memory.vision_map > 0).astype(DTYPE)[:, :, None]


def get_current_sensor_mask(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 1)
    return agent_state.memory.current_sensor_mask.astype(DTYPE)[:, :, None]


# # action masks

def get_action_mask_monofield(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 16)
    field = agent_state.memory.action_mask_monofield.astype(DTYPE)
    field = field.transpose(1, 2, 0)
    return field


def get_action_cost_monofield(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 16)
    field = jnp.where(
        agent_state.memory.action_mask_monofield,
        1,
        -1,
    )
    field = jnp.where(
        agent_state.memory.base_action_monofield,
        agent_state.memory.unit_move_cost / ENV_PARAMS_FIXED.max_unit_energy,
        field
    )
    field = jnp.where(
        agent_state.memory.noop_action_monofield,
        0,
        field,
    )
    field = jnp.where(
        agent_state.memory.sap_action_monofield,
        agent_state.memory.unit_sap_cost / ENV_PARAMS_FIXED.max_unit_energy,
        field,
    )
    field = field.transpose(1, 2, 0)
    return field.astype(DTYPE)


def get_action_mask_move(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 16)
    field = agent_state.memory.base_action_monofield.astype(DTYPE)
    field = field.transpose(1, 2, 0)
    return field


def get_action_mask_sap(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 16)
    field = agent_state.memory.sap_action_mask_field.astype(DTYPE)
    field = field.transpose(1, 2, 0)
    return field


def get_last_action_monofield(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 16)
    action_monofield = _convert_index_to_field_bool(agent_state.last_nn_action)
    return action_monofield.astype(DTYPE)


def get_last_action_duomofield(agent_state: AgentState) -> jnp.ndarray:  # (24, 24, 16)
    # nn_action: (..., 2, N_MAX_UNITS)
    base_action, sap_action = jnp.unstack(agent_state.last_nn_action, axis=-2)  # (N_MAX_UNITS), (N_MAX_UNITS)
    sap_grid = _convert_index_to_field_bool(sap_action)  # (GRID_SHAPE, N_MAX_UNITS)
    base_grid = agent_state.memory.distance_maps == 0  # (N_MAX_UNITS, GRID_SHAPE) [if we chose move action, then the result is our current position)
    base_grid = base_grid.transpose(-2, -1, -3)  # (GRID_SHAPE, N_MAX_UNITS)
    action_duomofield = jnp.where((base_action == 5)[..., None, None, :], sap_grid, base_grid)
    return action_duomofield.astype(DTYPE)


# --- private functions (helpers)


def _place_units_on_field(agent_state, units, mask):
    field = place_points_on_field(agent_state.memory.false_map16, units, mask)
    field = field.transpose(1, 2, 0)
    mask_units = field > 0
    return field, mask_units


def _place_player_unit_energies_on_field(agent_state):
    """ Unit energy is normalised by max_unit_energy and negative values are set to -1, thus to [-1,1] """
    field, mask = _place_units_on_field(agent_state, agent_state.memory.unit_positions, agent_state.memory.unit_mask)
    capped_energies = jnp.where(
        agent_state.memory.unit_energy >= 0,
        agent_state.memory.unit_energy / ENV_PARAMS_FIXED.max_unit_energy,
        -1
    )
    field = field * capped_energies[None, None, :]
    return field, mask


def _place_player_unit_energies_on_field_previous(agent_state):
    """ Unit energy is normalised by max_unit_energy and negative values are set to -1, thus to [-1,1] """
    field, mask = _place_units_on_field(agent_state, agent_state.memory.previous_unit_positions, agent_state.memory.previous_unit_mask)
    capped_energies = jnp.where(
        agent_state.memory.previous_unit_energy >= 0,
        agent_state.memory.previous_unit_energy / ENV_PARAMS_FIXED.max_unit_energy,
        -1
    )
    field = field * capped_energies[None, None, :]
    return field, mask


def _place_opponent_unit_energies_on_field(agent_state):
    """ Unit energy is normalised by max_unit_energy and negative values are set to -1, thus to [-1,1] """
    field, mask = _place_units_on_field(agent_state, agent_state.memory.opp_unit_positions, agent_state.memory.opp_unit_mask)
    capped_energies = jnp.where(
        agent_state.memory.opp_unit_energy >= 0,
        agent_state.memory.opp_unit_energy / ENV_PARAMS_FIXED.max_unit_energy,
        -1
    )
    field = field * capped_energies[None, None, :]
    return field, mask


def _convert_index_to_field_bool(index):
    # index: (..., N_MAX_UNITS)
    leading_shape = index.shape[:-1]

    def for_one_unit(ind):
        field_unit = jnp.zeros(shape=leading_shape + GRID_SHAPE, dtype=DTYPE)
        field_unit = jnp.reshape(field_unit, shape=leading_shape + (-1,))
        field_unit = field_unit.at[ind].set(1)
        field_unit = jnp.reshape(field_unit, shape=leading_shape + GRID_SHAPE)
        return field_unit

    field = jax.vmap(for_one_unit, in_axes=-1, out_axes=-1)(index)
    return field  # (..., GRID_SHAPE, N_MAX_UNITS)


# --------------------- Registry

GET_FIELDS_FN = {
    "zero": get_zero,
    "one": get_one,
    "player_unit_field_and_mask": get_player_unit_field_and_mask,
    "opponent_unit_field_and_mask": get_opponent_unit_field_and_mask,
    "player_condensed_unit_field": get_player_condensed_unit_field,
    "player_condensed_unit_field_previous": get_player_condensed_unit_field_previous,
    "player_condensed_unit_map_previous": get_player_condensed_unit_map_previous,
    "player_condensed_summed_energy_and_n_units_maps": get_player_condensed_summed_energy_and_n_units_maps,
    "player_condensed_summed_energy_and_n_units_maps_previous": get_player_condensed_summed_energy_and_n_units_maps_previous,
    "opponent_condensed_unit_field": get_opponent_condensed_unit_field,
    "opponent_condensed_unit_map": get_opponent_condensed_unit_map,
    "opponent_unit_map_and_mask": get_opponent_unit_map_and_mask,
    "opponent_condensed_summed_energy_and_n_units_maps": get_opponent_condensed_summed_energy_and_n_units_maps,
    "opponent_condensed_summed_energy_and_n_units_maps_previous": get_opponent_condensed_summed_energy_and_n_units_maps_previous,
    "contact_efficiency_map": get_contact_efficiency_map,
    "sap_efficiency_map": get_sap_efficiency_map,
    "invisible_point_tiles_sap_map": get_invisible_point_tiles_sap_map,
    "relic_map": get_relic_map,
    "point_tile_map": get_point_tile_map,
    "point_tile_occupancy_map": get_point_tile_occupancy_map,
    "point_tile_occupancy_map_previous": get_point_tile_occupancy_map_previous,
    "potential_point_tile_map": get_potential_point_tile_map,
    "empty_tile_map": get_empty_tile_map,
    "approx_dist_to_point_tiles": get_approx_dist_to_point_tiles,
    "point_field": get_point_field,
    "energy_field_and_mask": get_energy_field_and_mask,
    "energy_map_including_nebula": get_energy_map_including_nebula,
    "positive_energy_mask": get_positive_energy_mask,
    "high_energy_mask": get_high_energy_mask,
    "very_high_energy_mask": get_very_high_energy_mask,
    "tile_type_field": get_tile_type_field,
    "next_tile_type_field": get_next_tile_type_field,
    "explored_tiles_map": get_explored_tiles_map,
    "player_recently_visited_map": get_player_recently_visited_map,
    "opponent_recently_visited_map": get_opponent_recently_visited_map,
    "nebula_energy_reduction": get_nebula_energy_reduction,
    "nebula_vision_reduction": get_nebula_vision_reduction,
    "vision_map": get_vision_map,
    "current_sensor_mask": get_current_sensor_mask,
    "action_mask_monofield": get_action_mask_monofield,
    "action_cost_monofield": get_action_cost_monofield,
    "action_mask_move": get_action_mask_move,
    "action_mask_sap": get_action_mask_sap,
    "last_action_monofield": get_last_action_monofield,
    "last_action_duomofield": get_last_action_duomofield,
}

GET_N_CHANNELS = {
    "zero": 1,
    "one": 1,
    "player_unit_field_and_mask": 32,
    "opponent_unit_field_and_mask": 32,
    "player_condensed_unit_field": 16,
    "player_condensed_unit_field_previous": 16,
    "player_condensed_unit_map_previous": 1,
    "player_condensed_summed_energy_and_n_units_maps": 2,
    "player_condensed_summed_energy_and_n_units_maps_previous": 2,
    "opponent_condensed_unit_field": 16,
    "opponent_condensed_unit_map": 1,
    "opponent_unit_map_and_mask": 2,
    "opponent_condensed_summed_energy_and_n_units_maps": 2,
    "opponent_condensed_summed_energy_and_n_units_maps_previous": 2,
    "contact_efficiency_map": 1,
    "sap_efficiency_map": 1,
    "invisible_point_tiles_sap_map": 1,
    "relic_map": 1,
    "point_tile_map": 1,
    "point_tile_occupancy_map": 1,
    "point_tile_occupancy_map_previous": 1,
    "potential_point_tile_map": 1,
    "empty_tile_map": 1,
    "approx_dist_to_point_tiles": 1,
    "point_field": 1,
    "energy_field_and_mask": 2,
    "energy_map_including_nebula": 1,
    "positive_energy_mask": 1,
    "high_energy_mask": 1,
    "very_high_energy_mask": 1,
    "tile_type_field": 1,
    "next_tile_type_field": 1,
    "explored_tiles_map": 1,
    "player_recently_visited_map": 1,
    "opponent_recently_visited_map": 1,
    "nebula_energy_reduction": 1,
    "nebula_vision_reduction": 1,
    "vision_map": 1,
    "current_sensor_mask": 1,
    "action_mask_monofield": 16,
    "action_cost_monofield": 16,
    "action_mask_move": 16,
    "action_mask_sap": 16,
    "last_action_monofield": 16,
    "last_action_duomofield": 16,
}
