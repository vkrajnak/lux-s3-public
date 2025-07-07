import numpy as np
import jax
import jax.numpy as jnp
import sys
from jax import lax
from flax import linen as nn
from ..memory.memory_dataclass import MemoryState
from ..memory.energy_fields_opt import energy_fields, energy_fields_compare, distances_to_points, coords_distances_to_points, coordinate_map
from ..constants import GRID_SHAPE, N_MAX_UNITS, N_BASE_ACTIONS, N_SAP_ACTIONS, ENV_PARAMS_RANGES, ENV_PARAMS_FIXED
from ..types import LuxEnvObs

map_size = jnp.int16(GRID_SHAPE[0])
half_map_size = map_size // 2
max_units = jnp.int16(N_MAX_UNITS)
max_sap_range = jnp.int16(ENV_PARAMS_RANGES["unit_sap_range"][-1])
width_sap_range = 2*max_sap_range+1

nebula_drift_speeds = jnp.array([0.025, 0.05, 0.1, 0.15])


def memory_init(player_id: int, env_cfg: dict, force: dict) -> MemoryState:

    max_relics = 6
    relic_area_size = 5

    mem = MemoryState(
        team_id=player_id,
        force_stay_on_point_tile=force.stay_on_point_tile,
        force_no_noop=force.no_noop,
        force_sap_location=force.sap_location,
        force_avoid_collision=force.avoid_collision,

        # game parameters
        max_units=max_units,
        unit_move_cost=jnp.int16(env_cfg["unit_move_cost"]),
        unit_sap_cost=jnp.int16(env_cfg["unit_sap_cost"]),
        unit_sap_range=jnp.int16(env_cfg["unit_sap_range"]),
        unit_sensor_range=jnp.int16(env_cfg["unit_sensor_range"]),
        match_count_per_episode=jnp.int16(env_cfg['match_count_per_episode']),
        max_steps_in_match=jnp.int16(env_cfg['max_steps_in_match']),
        map_size=map_size,
        unit_sap_dropoff_factor = jnp.float16(-1),
        unit_energy_void_factor = jnp.float16(-1),

        # units
        unit_positions=-jnp.ones((max_units, 2), dtype=jnp.int16),
        unit_energy=-jnp.ones(max_units, dtype=jnp.int16),
        unit_mask=jnp.zeros(max_units, dtype=jnp.bool),
        unit_field=jnp.zeros((max_units, map_size, map_size), dtype=jnp.bool),
        unit_map=jnp.zeros((map_size, map_size), dtype=jnp.bool),
        energy_map=jnp.zeros((map_size, map_size), dtype=jnp.int16),
        previous_unit_energy=-jnp.ones(max_units, dtype=jnp.int16),
        previous_unit_mask=jnp.zeros(max_units, dtype=jnp.bool),
        previous_unit_positions=-jnp.ones((max_units, 2), dtype=jnp.int16),
        distance_maps = jnp.zeros((max_units, map_size, map_size), dtype=jnp.int16),
        vision_map=jnp.zeros((map_size, map_size), dtype=jnp.int16),
        previous_energy_map=jnp.zeros((map_size, map_size), dtype=jnp.int16),
        n_units_map=jnp.zeros((map_size, map_size), dtype=jnp.int16),
        previous_n_units_map=jnp.zeros((map_size, map_size), dtype=jnp.int16),
        contact_efficiency_map=jnp.zeros((map_size, map_size), dtype=jnp.float32),
        sap_efficiency_map=jnp.zeros((map_size, map_size), dtype=jnp.float32),
        invisible_point_tiles_sap_map=jnp.zeros((map_size, map_size), dtype=jnp.float32),
        current_sensor_mask=jnp.zeros((map_size, map_size), dtype=jnp.bool),

        opp_unit_positions=-jnp.ones((max_units, 2), dtype=jnp.int16),
        opp_unit_energy=-jnp.ones(max_units, dtype=jnp.int16),
        opp_unit_mask=jnp.zeros(max_units, dtype=jnp.bool),
        opp_unit_map=jnp.zeros((map_size, map_size), dtype=jnp.bool),
        opp_energy_map=jnp.zeros((map_size, map_size), dtype=jnp.int16),
        previous_opp_unit_positions=-jnp.ones((max_units, 2), dtype=jnp.int16),
        previous_opp_unit_energy=-jnp.ones(max_units, dtype=jnp.int16),
        previous_opp_energy_map=jnp.zeros((map_size, map_size), dtype=jnp.int16),
        n_opp_units_map=jnp.zeros((map_size, map_size), dtype=jnp.int16),
        previous_n_opp_units_map=jnp.zeros((map_size, map_size), dtype=jnp.int16),

        last_action=jnp.zeros((max_units, 3), dtype=jnp.int16),

        # match info
        steps=jnp.int16(0),
        match_steps=jnp.int16(0),

        points=jnp.int16(0),
        previous_points=jnp.int16(0),
        opp_points=jnp.int16(0),
        opp_points_gain=jnp.int16(0),

        wins=jnp.int16(0),
        losses=jnp.int16(0),
        match_num=jnp.int16(0),

        # energy
        energy_field=-jnp.ones((map_size, map_size), dtype=jnp.int16),
        tile_type_field=-jnp.ones((map_size, map_size), dtype=jnp.int16),
        next_tile_type_field=-jnp.ones((map_size, map_size), dtype=jnp.int16),
        valid_energy_mask=jnp.zeros((map_size, map_size), dtype=jnp.bool),
        possible_energy_fields=energy_fields,
        possible_energy_fields_compare=energy_fields_compare,
        global_energy_field=False,
        energy_field_including_nebula=-jnp.ones((map_size, map_size), dtype=jnp.int16),

        # drift
        energy_node_drift_frequency=jnp.int16(0),
        nebula_drift_direction=jnp.int16(0),
        nebula_drift_speed=jnp.float16(0.30),

        # nebula
        nebula_vision_reduction=jnp.int16(-1),
        nebula_energy_reduction=jnp.int16(-1),

        # relics
        max_relics=jnp.int16(max_relics),  # taken from params
        relics=-jnp.ones((max_relics, 2), dtype=jnp.int16),
        relics_mask=jnp.zeros(max_relics, dtype=jnp.bool),
        relics_map=jnp.zeros((map_size, map_size), dtype=jnp.bool),
        relics_found=jnp.int16(0),
        relics_on_diagonal=jnp.int16(0),
        relic_area_size=jnp.int16(relic_area_size),  # taken from params
        all_relics_found=jnp.bool(False),
        tiles_around_relic=jnp.stack(jnp.mgrid[0:relic_area_size, 0:relic_area_size] - relic_area_size // 2, axis=-1).reshape(-1, 2).astype(jnp.int16),
        explored_for_relic_mask=jnp.zeros((map_size, map_size), dtype=jnp.bool),

        # point tiles
        potential_point_tiles=jnp.zeros((map_size, map_size), dtype=jnp.bool),
        point_tiles=jnp.zeros((map_size, map_size), dtype=jnp.bool),
        empty_tiles=jnp.zeros((map_size, map_size), dtype=jnp.bool),
        approx_dist_to_point_tiles=-jnp.ones((map_size, map_size), dtype=jnp.int16),
        all_point_tiles_found=jnp.bool(False),
        
        # visited tiles
        recently_visited_map=jnp.zeros((map_size, map_size), dtype=jnp.float32),
        opp_recently_visited_map=jnp.zeros((map_size, map_size), dtype=jnp.float32),

        # actions
        # base_action_mask_discrete=jnp.zeros((max_units, N_BASE_ACTIONS), dtype=jnp.bool),
        # sap_action_mask_discrete=jnp.zeros((max_units, N_SAP_ACTIONS), dtype=jnp.bool),
        action_mask_monofield = jnp.zeros((max_units, map_size, map_size), dtype=jnp.bool),
        base_action_monofield = jnp.zeros((max_units, map_size, map_size), dtype=jnp.bool),
        sap_action_monofield = jnp.zeros((max_units, map_size, map_size), dtype=jnp.bool),
        noop_action_monofield = jnp.zeros((max_units, map_size, map_size), dtype=jnp.bool),
        # base_action_duofield = jnp.zeros((max_units, map_size, map_size), dtype=jnp.bool),
        monofield_base_converter = jnp.zeros((max_units, map_size, map_size), dtype=jnp.int16),
        relative_coordinate_maps = jnp.zeros((max_units, map_size, map_size, 2), dtype=jnp.int16),
        # sap_action_mask_field = jnp.zeros((max_units, map_size, map_size), dtype=jnp.bool),

        # aux
        coordinate_map=coordinate_map,
        false_map=jnp.zeros((map_size, map_size), dtype=jnp.bool),
        true_map=jnp.ones((map_size, map_size), dtype=jnp.bool),
        false_map12=jnp.zeros((12, map_size, map_size), dtype=jnp.bool),
        false_map16=jnp.zeros((16, map_size, map_size), dtype=jnp.bool),
        false_unit_mask=jnp.zeros(max_units, dtype=jnp.bool)
    )

    return mem


def memory_update(memory: MemoryState, obs: LuxEnvObs, last_action: jnp.ndarray) -> MemoryState:

    team_id = memory.team_id
    opp_id = memory.opp_id

    # previous
    memory = memory.replace(
        previous_unit_energy=memory.unit_energy,
        previous_points=memory.points,
        previous_unit_mask=memory.unit_mask,
        previous_unit_positions=memory.unit_positions,
        previous_energy_map=memory.energy_map,
        previous_n_units_map=memory.n_units_map,

        previous_opp_unit_positions=memory.opp_unit_positions,
        previous_opp_unit_energy=memory.opp_unit_energy,
        previous_opp_energy_map=memory.opp_energy_map,
        previous_n_opp_units_map=memory.n_opp_units_map,

        last_action = last_action,
    )

    # units
    memory = memory.replace(
        unit_positions=obs.units.position[team_id].astype(jnp.int16),
        unit_energy=obs.units.energy[team_id].astype(jnp.int16),
        unit_mask=obs.units_mask[team_id],
        opp_unit_positions=obs.units.position[opp_id].astype(jnp.int16),
        opp_unit_energy=obs.units.energy[opp_id].astype(jnp.int16),
        opp_unit_mask=obs.units_mask[opp_id],
    )

    unit_field = place_points_on_field(memory.false_map16, memory.unit_positions, memory.unit_mask)
    unit_map = place_points_on_map(memory.false_map16, memory.unit_positions, memory.unit_mask)
    opp_unit_map = place_points_on_map(memory.false_map16, memory.opp_unit_positions, memory.opp_unit_mask)
    relative_coordinate_maps, distance_maps = coords_distances_to_points(memory.unit_positions, jnp.max)
    energy_map = jnp.sum((distance_maps==0).astype(jnp.int16)*memory.unit_energy[:,None,None], where=(memory.unit_mask[:,None,None] & memory.unit_energy[:,None,None] > -1), initial=0, axis=0).astype(jnp.int16)
    n_units_map = jnp.sum((distance_maps==0) & (memory.unit_energy[:,None,None] > -1), axis=0).astype(jnp.int16)
    distance_maps_opp = distances_to_points(memory.opp_unit_positions, jnp.max)
    opp_energy_map = jnp.sum((distance_maps_opp==0).astype(jnp.int16)*memory.opp_unit_energy[:,None,None], where=(memory.opp_unit_mask[:,None,None] & memory.opp_unit_energy[:,None,None] > -1), initial=0, axis=0).astype(jnp.int16)
    n_opp_units_map = jnp.sum((distance_maps_opp==0) & (memory.opp_unit_energy[:,None,None] > -1), axis=0).astype(jnp.int16)

    memory = memory.replace(
        unit_field=unit_field,
        unit_map=unit_map,
        energy_map = energy_map,
        distance_maps=distance_maps,
        relative_coordinate_maps = relative_coordinate_maps,
        n_units_map=n_units_map,
        current_sensor_mask=obs.sensor_mask,

        opp_unit_map=opp_unit_map,
        opp_energy_map=opp_energy_map,
        n_opp_units_map=n_opp_units_map,
    )

    # previously visited (current=1, previous=factor, previous_previous=factor^2, previous_previous_previous=factor^3...)
    visited_decay_factor = 0.5
    visited_map = memory.unit_map.astype(jnp.float32) + visited_decay_factor * memory.recently_visited_map
    opp_visited_map = memory.opp_unit_map.astype(jnp.float32) + visited_decay_factor * memory.opp_recently_visited_map
    visited_map = jnp.clip(visited_map, max=1.0)
    opp_visited_map = jnp.clip(opp_visited_map, max=1.0)
    memory = memory.replace(
        recently_visited_map=visited_map,
        opp_recently_visited_map=opp_visited_map,
    )

    opp_points_gain = obs.team_points[opp_id] - memory.opp_points
    # match info
    memory = memory.replace(
        points=obs.team_points[team_id].astype(jnp.int16),
        opp_points=obs.team_points[opp_id].astype(jnp.int16),
        opp_points_gain=opp_points_gain.astype(jnp.int16),
        wins=obs.team_wins[team_id].astype(jnp.int16),
        losses=obs.team_wins[opp_id].astype(jnp.int16),
        steps=jnp.int16(obs.steps),
        match_steps=jnp.int16(obs.match_steps),
    )
    memory = memory.replace(
        match_num=memory.wins + memory.losses,
    )

    # energy and tile type
    current_energy_field = obs.map_features.energy.astype(jnp.int16)
    current_tile_type_field = obs.map_features.tile_type.astype(jnp.int16)
    current_sensor_mask = obs.sensor_mask

    memory = update_energy_and_mask(memory, current_energy_field, current_sensor_mask)

    memory = update_tile_type(memory, current_tile_type_field, current_sensor_mask)

    # relics & potential point tiles
    relic_nodes = obs.relic_nodes.astype(jnp.int16)
    relic_nodes_mask = obs.relic_nodes_mask
    memory = lax.cond(
        memory.all_relics_found,
        lambda mem, rn,rm,csm: mem,
        update_relics,
        memory,
        relic_nodes,
        relic_nodes_mask,
        current_sensor_mask
    )
    

    # point tiles
    occupied_potential_tiles = jnp.where(
        memory.unit_map,
        memory.potential_point_tiles,
        memory.false_map
    )

    memory = lax.cond(
        jnp.any(occupied_potential_tiles),
        update_point_tiles,
        lambda mem, oc: mem,
        memory,
        occupied_potential_tiles
    )

    memory = lax.cond(
        (memory.unit_sap_dropoff_factor < 0) & jnp.any(memory.last_action[:,0]==5) & jnp.any(memory.opp_unit_mask),
        deduce_sap_dropoff,
        lambda x: x,
        memory
    )

    memory = lax.cond(
        (memory.unit_energy_void_factor < 0) & jnp.any(memory.opp_unit_mask),
        deduce_energy_void,
        lambda x: x,
        memory
    )

    memory = memory.replace(
        all_point_tiles_found=jnp.all(~ memory.potential_point_tiles) & jnp.all(memory.explored_for_relic_mask)
    )

    # get combined energy and nebula, for convenience
    def get_energy_field_including_nebula(memory):
        nebula_field = (memory.tile_type_field == 1) * memory.nebula_energy_reduction
        energy_field = jnp.where(
            memory.valid_energy_mask,
            memory.energy_field,
            ENV_PARAMS_FIXED.min_energy_per_tile,  # if unknown, set to min energy value
        )
        combined_energy = jax.lax.cond(
            memory.nebula_energy_reduction >= 0,
            lambda: energy_field - nebula_field,
            lambda: energy_field
        )
        return combined_energy.astype(jnp.int16)
    memory = memory.replace(
        energy_field_including_nebula=get_energy_field_including_nebula(memory)
    )

    # jax.debug.print("{}, {}, {}, {}, {}, {}, {}, {}", obs.steps, memory.all_relics_found, jnp.sum(memory.point_tiles), jnp.sum(memory.potential_point_tiles),
    #                  jnp.sum(memory.empty_tiles), memory.nebula_drift_speed, memory.nebula_vision_reduction, memory.nebula_energy_reduction)

    # calculate action masks last
    memory = calculate_action_masks(memory, current_sensor_mask)

    return memory


def update_energy_and_mask(memory, current_energy_field, current_sensor_mask):

    # TODO: find way to get rid of valid_energy_mask
    compare_mask = jnp.where(
        memory.global_energy_field,
        current_sensor_mask,
        memory.valid_energy_mask & current_sensor_mask
    )

    def accommodate_visible(memory, current_energy_field, current_sensor_mask):
        new_valid_energy_mask = jnp.where(
                current_sensor_mask | current_sensor_mask[::-1, ::-1].T,
                memory.true_map,
                memory.valid_energy_mask,
            )
        energy_field = update_symmetrically(memory.energy_field, current_sensor_mask, current_energy_field)
     
        memory = memory.replace(
            energy_field=energy_field,
            valid_energy_mask=new_valid_energy_mask,
        )
        return memory
    
    def accommodate_visible_if_needed(memory, current_energy_field, current_sensor_mask):
        return lax.cond(
            jnp.logical_not(memory.global_energy_field) & (memory.steps > 0),
            accommodate_visible,
            lambda mem, field, mask: mem,
            memory,
            current_energy_field,
            current_sensor_mask
        )
    
    def reset_energy_to_visible(memory, current_energy_field, current_sensor_mask):
        new_valid_energy_mask = current_sensor_mask | current_sensor_mask[::-1, ::-1].T
        energy_field = update_symmetrically(current_energy_field, current_sensor_mask, current_energy_field)
        
        energy_node_drift_frequency = lax.cond(
            (memory.energy_node_drift_frequency == 0) & (memory.steps > 2),
            lambda: jnp.int16(memory.steps - 2),
            lambda: memory.energy_node_drift_frequency
        )

        memory = memory.replace(
            energy_field=energy_field,
            valid_energy_mask=new_valid_energy_mask,
            global_energy_field=False,
            energy_node_drift_frequency=energy_node_drift_frequency
        )
        return memory

    memory = lax.cond(
        jnp.all(current_energy_field == memory.energy_field, where=compare_mask),
        accommodate_visible_if_needed,
        reset_energy_to_visible,
        memory,
        current_energy_field,
        current_sensor_mask
    )

    def try_global_energy_field(memory):
        mask = jnp.all(
            memory.possible_energy_fields_compare == memory.energy_field[None,:half_map_size,:half_map_size],
            axis=(1,2),
            where=memory.valid_energy_mask[None,:half_map_size,:half_map_size]
        )

        def found_unique_energy_field(memory, mask):
            new_valid_energy_mask = memory.true_map
            energy_field = jnp.sum(memory.possible_energy_fields, where=mask[:, None, None], axis=0, dtype=jnp.int16)
            memory = memory.replace(
                global_energy_field=True,
                energy_field=energy_field,
                valid_energy_mask=new_valid_energy_mask,
            )
            return memory
        
        memory = lax.cond(
            jnp.sum(mask) == 1,
            found_unique_energy_field,
            lambda mem, mask: mem,
            memory,
            mask
        )
        return memory

    memory = lax.cond(
        memory.global_energy_field,
        lambda mem: mem,
        try_global_energy_field,
        memory
    )

    return memory


def update_symmetrically(field, mask, value):
    new_field = jnp.where(
        mask,
        value,
        field
    )

    new_field = jnp.where(
        mask[::-1, ::-1].T,
        value[::-1, ::-1].T,
        new_field
    )
    return new_field


def update_tile_type(memory, current_tile_type_field, current_sensor_mask):
    """Tile type and nebula properties"""

    def determine_nebula_drift_and_update(memory, current_tile_type_field, current_sensor_mask):
        seen_tile_before_mask = memory.tile_type_field > -1
        compare_mask = seen_tile_before_mask * current_sensor_mask
        
        return lax.cond(
            jnp.all(current_tile_type_field == memory.tile_type_field, where=compare_mask),
            no_drift_update,
            nebula_drift_and_update,
            memory,
            current_tile_type_field,
            current_sensor_mask
        )

    def determine_nebula_drift_and_apply_drift(memory, current_tile_type_field, current_sensor_mask):
        return lax.cond(
            jnp.any(current_tile_type_field > 0) | jnp.any(memory.tile_type_field > 0),  # does the map have any features?
            determine_nebula_drift_and_update,
            no_drift_update,
            memory,
            current_tile_type_field,
            current_sensor_mask
        )

    def see_if_nebula_moved(memory, current_tile_type_field, current_sensor_mask):
        return lax.cond(
            memory.nebula_drift_direction == 0,
            determine_nebula_drift_and_apply_drift,
            apply_nebula_drift_and_verify,
            memory,
            current_tile_type_field,
            current_sensor_mask
        )
    
    memory = calculate_vision_map(memory)

    memory = lax.cond(
        (memory.steps - 2) * memory.nebula_drift_speed % 1 > (memory.steps - 1) * memory.nebula_drift_speed % 1,
        see_if_nebula_moved,
        no_drift_update,
        memory,
        current_tile_type_field,
        current_sensor_mask
    )

    next_tile_type_field = lax.cond(
        ((memory.steps - 1) * memory.nebula_drift_speed % 1 > (memory.steps) * memory.nebula_drift_speed % 1) & (memory.nebula_drift_direction != 0),
        apply_nebula_drift,
        lambda ttf, dir: ttf,
        memory.tile_type_field,
        memory.nebula_drift_direction
    )
    
    memory = memory.replace(
        next_tile_type_field=next_tile_type_field
    )

    return memory


def no_drift_update(memory, current_tile_type_field, current_sensor_mask):
    # update invisible
    current_tile_type_field_with_invisible = jnp.where(
        memory.vision_map.astype(jnp.bool)*(~current_sensor_mask)*(memory.steps > 1),
        1,
        current_tile_type_field
    )

    # update visible
    new_tile_type_field = update_symmetrically(
        memory.tile_type_field,
        current_tile_type_field_with_invisible > -1,
        current_tile_type_field_with_invisible
    )
    memory = memory.replace(
        tile_type_field=new_tile_type_field
    )

    # determine vision reduction (must be done when nebule does not move due to the vision bug that has become a feature) AL: REMOVED TO SAVE TIME (NOT USED AS INPUT)
    # memory = lax.cond(
    #     (memory.nebula_vision_reduction < 0) & jnp.any(current_tile_type_field == 1) & (memory.steps > 1),
    #     determine_nebula_vision_reduction,
    #     lambda mem, type: mem,
    #     memory,
    #     current_tile_type_field
    # )

    # determine energy reduction
    memory = lax.cond(
        (memory.nebula_energy_reduction < 0) & jnp.any(memory.tile_type_field == 1),
        determine_nebula_energy_reduction,
        lambda x: x,
        memory
    )

    return memory

def apply_nebula_drift(field, ddir):
    return jnp.roll(field, shift=(ddir, -ddir), axis=(0, 1))


def nebula_drift_and_update(memory, current_tile_type_field, current_sensor_mask):
    # determine drift

    def is_drift_consistent(seen_tile_before_mask, tile_type_field, current_tile_type_field, ddir):
        drifted_seen_before_mask = apply_nebula_drift(seen_tile_before_mask, ddir)
        drifted_tile_type_field = apply_nebula_drift(tile_type_field, ddir)
        compare_drift_mask = drifted_seen_before_mask * current_sensor_mask
        return jnp.all(current_tile_type_field == drifted_tile_type_field, where=compare_drift_mask)

    drifts = jnp.array([-1,1]).astype(jnp.int16)
    seen_tile_before_mask = memory.tile_type_field > -1
    drift_consistent_mask = jax.vmap(is_drift_consistent, in_axes=(None,None,None,0))(seen_tile_before_mask, memory.tile_type_field, current_tile_type_field, drifts)
    ddir = lax.cond(
        jnp.sum(drift_consistent_mask) == 1,
        lambda: jnp.min(drifts, where=drift_consistent_mask, initial=2).astype(jnp.int16),
        lambda: jnp.int16(0)
    )

    def update_and_apply_drift(memory, ddir):
        drifted_tile_type_field = apply_nebula_drift(memory.tile_type_field, ddir)
        new_tile_type_field = update_symmetrically(drifted_tile_type_field, current_sensor_mask, current_tile_type_field)
        valid = (memory.steps - 2) * nebula_drift_speeds % 1 > (memory.steps - 1) * nebula_drift_speeds % 1

        memory = memory.replace(
            nebula_drift_speed=jnp.min(nebula_drift_speeds, where=valid, initial=1).astype(jnp.float16),
            nebula_drift_direction=ddir,
            tile_type_field=new_tile_type_field
        )
        return memory
    
    memory = lax.cond(
        jnp.sum(drift_consistent_mask) > 0,
        update_and_apply_drift,
        lambda mem, d: mem,
        memory,
        ddir
    )
    return memory

def apply_nebula_drift_and_verify(memory, current_tile_type_field, current_sensor_mask):
    new_tile_type_field = apply_nebula_drift(memory.tile_type_field, memory.nebula_drift_direction)
    seen_tile_before_mask = new_tile_type_field > -1
    compare_mask = seen_tile_before_mask * current_sensor_mask

    def reset_nebula_drift(memory):
        memory = memory.replace(
            nebula_drift_speed=jnp.float16(0.3),
            nebula_drift_direction=jnp.int16(0)
        )

        seen_tile_before_mask = memory.tile_type_field > -1
        compare_mask = seen_tile_before_mask * current_sensor_mask
        new_tile_type_field = lax.cond(
            jnp.all(current_tile_type_field == memory.tile_type_field, where=compare_mask),
            lambda: memory.tile_type_field,
            lambda: -memory.true_map.astype(jnp.int16)
        )
        return memory, new_tile_type_field

    memory, new_tile_type_field = lax.cond(
        jnp.all(current_tile_type_field == new_tile_type_field, where=compare_mask),
        lambda x: (x, new_tile_type_field),
        reset_nebula_drift,
        memory        
    )

    new_tile_type_field = update_symmetrically(new_tile_type_field, current_sensor_mask, current_tile_type_field)
    memory = memory.replace(
        tile_type_field=new_tile_type_field
    )

    return memory

def calculate_vision_map(memory):
    vision_maps = 1 + memory.unit_sensor_range - memory.distance_maps

    vision_maps = jnp.where(
        (vision_maps > 0) & (memory.unit_mask[:,None,None]),
        vision_maps.astype(jnp.int16),
        jnp.zeros(vision_maps.shape, dtype=jnp.int16)
    )
    vision_map = jnp.sum(vision_maps, axis=0)
    memory = memory.replace(
        vision_map = vision_map.astype(jnp.int16)
    )
    return memory

def determine_nebula_vision_reduction(memory, current_tile_type_field):
    """ calculate distance map to every unit, overlay with nebula tile mask and take the minimum """
    # TODO: infer nebula presence from cumul_vision and use this to determine vision reduction more precisely
    nebula_tile_mask = (current_tile_type_field == 1) & (~memory.unit_map)

    cumul_vision = jnp.min(memory.vision_map, where=nebula_tile_mask, initial=100, axis=(0,1)).astype(jnp.int16),
    cumul_vision = cumul_vision[0] - 1 # vision power must be at least 1 for a tile to be visible
    cumul_vision = jnp.where(
        cumul_vision < 8,
        cumul_vision,
        jnp.int16(-1)
    )

    memory = memory.replace(
        nebula_vision_reduction=cumul_vision
    )

    return memory


def determine_nebula_energy_reduction(memory):
    unit_moved = jnp.any(memory.unit_positions != memory.previous_unit_positions, axis=1)
    mask = jnp.where(
        (memory.previous_unit_mask) & (memory.unit_energy > 0) & unit_moved,
        memory.tile_type_field[memory.unit_positions.T[0],memory.unit_positions.T[1]] == 1,
        memory.false_unit_mask
    )

    def energy_reduction(memory, mask):
        nebula_unit_previous_energy = memory.previous_unit_energy
        nebula_unit_energy = memory.unit_energy
        expected_energy = nebula_unit_previous_energy + memory.energy_field[memory.unit_positions.T[0],memory.unit_positions.T[1]] - memory.unit_move_cost
        diff = expected_energy - nebula_unit_energy
        memory = memory.replace(
            nebula_energy_reduction=jnp.min(diff, where=mask, initial=100).astype(jnp.int16)
        )
        return memory

    memory = lax.cond(
        jnp.any(mask),
        energy_reduction,
        lambda mem, mask: mem,
        memory,
        mask
    )       

    return memory


def sym(xmax, points):
    return xmax - points[..., ::-1]


def set_minus_mask(new, new_mask, old):
    intersection = jnp.all(new[None, :, :] == old[:, None, :], axis=-1)
    mask = jnp.all(~intersection, axis=0)
    return mask & new_mask


def place_points_on_field(false_map, points, mask):
    enum_points = jnp.column_stack((jnp.arange(len(points)), points)).T
    result_field = jnp.where(
        mask[:, None, None],
        false_map.at[enum_points[0],enum_points[1],enum_points[2]].set(True),
        false_map
    )
    result_field
    return result_field

def place_points_on_map(false_map, points, mask):
    result_map = jnp.any(place_points_on_field(false_map, points, mask), axis=0)
    return result_map

def update_relics(memory, relic_nodes, relic_nodes_mask, current_sensor_mask):
    def add_new_relics(memory, relic_nodes, new_relics_mask):
        relics_and_sym = jnp.vstack((relic_nodes, sym(memory.xmax, relic_nodes)))
        double_mask = jnp.hstack((new_relics_mask, new_relics_mask))
        new_relic_map = place_points_on_map(memory.false_map12, relics_and_sym, double_mask)

        relics_map = new_relic_map | memory.relics_map

        relic_coords = jnp.nonzero(relics_map, size=6, fill_value=-1)
        relic_coords = jnp.column_stack(relic_coords).astype(jnp.int16)

        relic_mask = jnp.all(relic_coords > -1, axis=1)
        distances_to_relics = distances_to_points(relic_coords, jnp.sum)
        distance_map = jnp.min(distances_to_relics, axis=0, where=relic_mask[:,None,None], initial=memory.map_size) # if add_new_relics is called, at least one relic has been spotted
        distance_map = jnp.clip(distance_map-memory.relic_area_size+1, min=0).astype(jnp.int16) # not the same as having an euclidian distance to the 5x5 area around each relic, but enough to guide units
        
        memory = memory.replace(
            relics_map=relics_map,
            relics_found=jnp.sum(relics_map).astype(jnp.int16),
            relics=relic_coords,
            relics_on_diagonal = jnp.sum(jnp.sum(relic_coords, axis=1) == memory.xmax).astype(jnp.int16),
            approx_dist_to_point_tiles = distance_map.astype(jnp.int16),
        )

        memory = add_potential_point_tiles(memory, new_relic_map)
        return memory
    
    new_relics_mask = set_minus_mask(relic_nodes, relic_nodes_mask, memory.relics)
    memory = lax.cond(
        jnp.any(new_relics_mask),
        add_new_relics,
        lambda mem, nodes, mask: mem,
        memory,
        relic_nodes,
        new_relics_mask
    ) 
        
    def explore_mask_true(memory, current_sensor_mask):
        all_relics_found = (memory.steps > 303) | (memory.relics_found ==  6 - memory.relics_on_diagonal)
        memory = memory.replace(
            all_relics_found=all_relics_found,
            explored_for_relic_mask = memory.true_map,
        )
        
        return memory
    
    def explore_mask_update(memory, current_sensor_mask):
        explored_for_relic_mask = lax.cond(
            (memory.match_steps < 50) & (memory.steps < 303),
            lambda: current_sensor_mask | current_sensor_mask[::-1, ::-1].T,
            lambda: memory.explored_for_relic_mask | current_sensor_mask | current_sensor_mask[::-1, ::-1].T,
        )
        all_relics_found = jnp.all(explored_for_relic_mask) & (memory.relics_found < memory.max_relics)
        max_relics = jnp.where(
            all_relics_found,
            memory.relics_found,
            memory.max_relics
        ).astype(jnp.int16)
        memory = memory.replace(
            all_relics_found=all_relics_found,
            explored_for_relic_mask = explored_for_relic_mask,
            max_relics=max_relics
        )

        return memory

    memory = memory.replace(
        max_relics = jnp.int16(jnp.min(jnp.array([2 * memory.match_num + 2, 6])) - memory.relics_on_diagonal)
    )    
    memory = lax.cond(
        memory.relics_found == memory.max_relics,
        explore_mask_true,
        explore_mask_update,
        memory,
        current_sensor_mask
    )

    return memory


def add_potential_point_tiles(memory, new_relic_pos_map):
    relic_coords = jnp.nonzero(new_relic_pos_map, size=6, fill_value=-1)
    relic_coords = jnp.column_stack(relic_coords)
    relic_mask = jnp.all(relic_coords >= 0, axis=1)
    relic_distances = distances_to_points(relic_coords, jnp.max)
    new_potential_point_tiles = jnp.min(relic_distances, axis=0, initial=memory.map_size, where=relic_mask[:, None, None]) <= memory.relic_area_max_dist

    new_potential_point_tiles = jnp.where(
        memory.point_tiles,
        memory.false_map,
        new_potential_point_tiles
    )

    new_empty_tiles = jnp.where(
        new_potential_point_tiles,
        memory.false_map,
        memory.empty_tiles
    )

    new_potential_point_tiles = new_potential_point_tiles | memory.potential_point_tiles
    memory = memory.replace(
        potential_point_tiles=new_potential_point_tiles,
        empty_tiles=new_empty_tiles
    )

    return memory


def update_point_tiles(memory, occupied_potential_tiles):

    n_occupied_potential = jnp.sum(occupied_potential_tiles)
    occupied_point_tiles = jnp.where(
        memory.unit_map,
        memory.point_tiles,
        memory.false_map
    )
    n_occupied_point = jnp.sum(occupied_point_tiles)

    def all_potential_empty(memory, occupied_potential_tiles):
        # all occupied_potential_tiles are empty tiles
        empty_tiles = occupied_potential_tiles | memory.empty_tiles
        empty_tiles = empty_tiles | empty_tiles[::-1,::-1].T
        potential_point_tiles = jnp.where(
            empty_tiles,
            memory.false_map,
            memory.potential_point_tiles
        )
        memory = memory.replace(
            empty_tiles=empty_tiles,
            potential_point_tiles=potential_point_tiles
        )
        return memory

    def all_potential_point(memory, occupied_potential_tiles):
        # all occupied_potential_tiles are point tiles
        point_tiles = occupied_potential_tiles | memory.point_tiles
        point_tiles = point_tiles | point_tiles[::-1,::-1].T
        potential_point_tiles = jnp.where(
            point_tiles,
            memory.false_map,
            memory.potential_point_tiles
        )
        memory = memory.replace(
            point_tiles=point_tiles,
            potential_point_tiles=potential_point_tiles
        )
        return memory

    ind = jnp.int16(memory.points == memory.previous_points + n_occupied_point) + jnp.int16(memory.points == memory.previous_points + n_occupied_point + n_occupied_potential)*2

    memory = lax.switch(
        ind,
        [lambda mem, opt: mem, all_potential_empty, all_potential_point],
        memory,
        occupied_potential_tiles
    )

    return memory

def calculate_action_masks(memory, current_sensor_mask):
    can_move = memory.unit_energy >= memory.unit_move_cost
    can_sap = memory.unit_energy >= memory.unit_sap_cost

    unit_position_maps = memory.distance_maps == 0
    stand_still = memory.unit_energy >= 0
    move_up = can_move & (memory.unit_positions[:,1] > 0) & (jnp.sum(memory.tile_type_field[None,:,:-1], where=unit_position_maps[:,:,1:], initial=0, axis=(1,2)) < 2)
    move_right = can_move & (memory.unit_positions[:,0] < memory.xmax) & (jnp.sum(memory.tile_type_field[None,1:,:], where=unit_position_maps[:,:-1,:], initial=0, axis=(1,2)) < 2)
    move_down = can_move & (memory.unit_positions[:,1] < memory.xmax) & (jnp.sum(memory.tile_type_field[None,:,1:], where=unit_position_maps[:,:,:-1], initial=0, axis=(1,2)) < 2)
    move_left = can_move & (memory.unit_positions[:,0] > 0)  & (jnp.sum(memory.tile_type_field[None,:-1,:], where=unit_position_maps[:,1:,:], initial=0, axis=(1,2)) < 2)

    # contact efficiency map (ie damage from collision / energy void damage)
    kernel = jnp.ones((3, 3)) * jnp.where(memory.unit_energy_void_factor > 0, memory.unit_energy_void_factor, jnp.mean(jnp.array(ENV_PARAMS_RANGES["unit_energy_void_factor"])))
    kernel = kernel.at[1, 1].set(1.)
    kernel = kernel.at[0, 0].set(0.)
    kernel = kernel.at[0, 2].set(0.)
    kernel = kernel.at[2, 0].set(0.)
    kernel = kernel.at[2, 2].set(0.)
    contact_efficiency_map = jax.scipy.signal.convolve2d(memory.opp_unit_map, kernel, mode='same')

    # opponent probable location for sap (assumes opp not on point tile takes random action, opp on point tile stays there)
    n_opp_units_map_on_point_tile = memory.n_opp_units_map * memory.point_tiles
    n_opp_units_map_not_on_point_tile = memory.n_opp_units_map * (~ memory.point_tiles)

    kernel = jnp.ones((3, 3))/6
    kernel = kernel.at[1, 1].set(1/3)
    kernel = kernel.at[0, 0].set(0.)
    kernel = kernel.at[0, 2].set(0.)
    kernel = kernel.at[2, 0].set(0.)
    kernel = kernel.at[2, 2].set(0.)
    opp_probability_map = jax.scipy.signal.convolve2d(n_opp_units_map_not_on_point_tile, kernel, mode='same') + n_opp_units_map_on_point_tile
    # opp_probability_map = memory.n_opp_units_map

    # sap efficiency map on visible opponent
    kernel = jnp.ones((3,3)) * jnp.where(memory.unit_sap_dropoff_factor > 0, memory.unit_sap_dropoff_factor, 1)
    kernel = kernel.at[1,1].set(1.)
    sap_efficiency_map = jax.scipy.signal.convolve2d(opp_probability_map, kernel, mode='same')

    # # remove pointless sap
    # potential_opp_location_map = memory.opp_unit_map | ~current_sensor_mask
    # inflated_potential_opp_location_map = nn.max_pool(
    #     potential_opp_location_map[:,:,None],
    #     window_shape=(3,3),
    #     strides=(1,1),
    #     padding='SAME'
    # ).reshape((1,) + potential_opp_location_map.shape)

    # efficient sap locations on invisible opponent on point tile
    kernel = jnp.ones((3,3)) * jnp.where(memory.unit_sap_dropoff_factor > 0, memory.unit_sap_dropoff_factor, 0.0)
    kernel = kernel.at[1,1].set(1.)

    n_opp_on_point_tile = jnp.sum(jnp.logical_and(memory.point_tiles, memory.opp_unit_map))
    invisible_point_tiles = memory.point_tiles & ~current_sensor_mask
    invisible_potential_point_tiles = memory.potential_point_tiles & ~current_sensor_mask
    invisible = invisible_point_tiles + invisible_potential_point_tiles
    probability = jnp.where(
        jnp.any(invisible),
        (memory.opp_points_gain - n_opp_on_point_tile) / jnp.count_nonzero(invisible),
        0
    )
    invisible_point_tiles_sap_map = jax.scipy.signal.convolve2d(invisible_point_tiles, kernel, mode='same') * probability

    # used only is forcing sap location
    sap_value_map = invisible_point_tiles_sap_map + sap_efficiency_map
    threshold = 0.99 * jnp.max(sap_value_map)
    threshold = jnp.where(threshold > 0.99, threshold, 0.99)

    # mask of efficient sap locations
    opp_location_sap_map = jax.lax.cond(
        memory.force_sap_location,
        lambda: sap_value_map > threshold,
        lambda: (invisible_point_tiles_sap_map > 0.65) | (sap_efficiency_map > 0.49),
    )
    sappable_tiles = (memory.distance_maps <= memory.unit_sap_range) & can_sap[:, None, None]  # bool
    sap_fields = sappable_tiles & opp_location_sap_map[None, :, :]

    # # combined efficient sap locations (shared by all units)
    # opp_location_sap_map = invisible_point_tiles_sap_map + sap_efficiency_map
    #
    # sappable_tiles = (memory.distance_maps <= memory.unit_sap_range) & can_sap[:, None, None]  # bool
    # sap_fields_values = sappable_tiles.astype(jnp.float32) * opp_location_sap_map[None,:,:]  # numeric (N_MAX_UNITS, GRID_SHAPE)
    #
    # # keep only best sap actions for each unit
    # global_threshold = 0.49  # 0.99 * (2/6 + 4/6 * considered_unit_sap_dropoff_factor)
    # per_unit_threshold = jax.lax.cond(
    #     memory.force_sap_location,
    #     lambda: 0.99,
    #     lambda: 0.0,
    # ) * jnp.max(sap_fields_values, axis=(1, 2))  # N_MAX_UNITS
    # sap_fields_threshold = jnp.where(per_unit_threshold < global_threshold, global_threshold, per_unit_threshold)
    # sap_fields = sap_fields_values >= sap_fields_threshold[:, None, None]
    # jax.debug.print("{} - {}", memory.steps, jnp.sum(sap_fields, axis=(1,2)))

    # whether sap is a possible action for each unit, given all constraints above
    can_sap = jnp.where(
        jnp.any(sap_fields, axis=(-2,-1)),
        True,
        False
    )

    # # action masks for discrete actions
    # base_actions = jnp.column_stack((
    #     stand_still,
    #     move_up,
    #     move_right,
    #     move_down,
    #     move_left,
    #     can_sap
    # ))

    # def sap_range_map(pos):
    #     coords = memory.coordinate_map[:width_sap_range, :width_sap_range, :] - max_sap_range
    #     in_sap_range = jnp.max(jnp.abs(coords), axis=2) <= memory.unit_sap_range
    #     centred_coords = coords + pos[None,None,:]
    #     on_board = jnp.all((0 <= centred_coords) & (centred_coords < memory.xmax), axis=2)
    #     mask = in_sap_range & on_board
    #     return mask
    
    # sap_range_maps = jnp.where(
    #     can_sap[:, None, None],
    #     jax.vmap(sap_range_map, in_axes=0, out_axes=0)(memory.unit_positions),
    #     jnp.zeros((max_units, width_sap_range, width_sap_range), dtype=jnp.bool)
    # )

    # enum pos
    enum_pos = jnp.column_stack((jnp.arange(len(memory.unit_positions)), memory.unit_positions)).T

    # check for heavy opponent unit and avoid bad collision
    opp_energy_broadcasted = jnp.broadcast_to(memory.opp_energy_map, (N_MAX_UNITS,) + GRID_SHAPE)
    safe_collision_up = memory.unit_energy >= opp_energy_broadcasted[enum_pos[0], enum_pos[1], enum_pos[2]-1]
    safe_collision_right = memory.unit_energy >= opp_energy_broadcasted[enum_pos[0], enum_pos[1]+1, enum_pos[2]]
    safe_collision_down = memory.unit_energy >= opp_energy_broadcasted[enum_pos[0], enum_pos[1], enum_pos[2]+1]
    safe_collision_left = memory.unit_energy >= opp_energy_broadcasted[enum_pos[0], enum_pos[1]-1, enum_pos[2]]

    move_up = jax.lax.cond(
        memory.force_avoid_collision,
        lambda: move_up & safe_collision_up,
        lambda: move_up,
    )
    move_right = jax.lax.cond(
        memory.force_avoid_collision,
        lambda: move_right & safe_collision_right,
        lambda: move_right,
    )
    move_down = jax.lax.cond(
        memory.force_avoid_collision,
        lambda: move_down & safe_collision_down,
        lambda: move_down,
    )
    move_left = jax.lax.cond(
        memory.force_avoid_collision,
        lambda: move_left & safe_collision_left,
        lambda: move_left,
    )

    # determine whether to force staying on point tile
    is_on_point_tile = memory.unit_field & jnp.broadcast_to(memory.point_tiles, memory.unit_field.shape)  # (N_MAX_UNITS, GRID_SHAPE)
    is_alone = jnp.sum(is_on_point_tile, axis=(0,)) < 2  # GRID_SHAPE
    is_alone_on_point_tile = is_on_point_tile & jnp.broadcast_to(is_alone, is_on_point_tile.shape)  # (N_MAX_UNITS, GRID_SHAPE)
    is_alone_on_point_tile = jnp.any(is_alone_on_point_tile, axis=(1, 2))  # (N_MAX_UNITS,)
    force_stay = memory.force_stay_on_point_tile & is_alone_on_point_tile  # (N_MAX_UNITS,)

    # determine whether to force not choosing no-op
    can_move_somewhere = move_up | move_right | move_down | move_left  # (N_MAX_UNITS,)
    is_on_high_energy_tile = memory.unit_field & jnp.broadcast_to(memory.energy_field_including_nebula > 7, memory.unit_field.shape)  # (N_MAX_UNITS, GRID_SHAPE)
    stay_is_ok = jnp.any(is_on_point_tile, axis=(1, 2)) | jnp.any(is_on_high_energy_tile, axis=(1, 2))
    force_do_smg = memory.force_no_noop & (can_move_somewhere | can_sap) & (~ stay_is_ok)  # (N_MAX_UNITS,)

    # action masks for monofiled actions: account for unefficient sap locations
    action_mask_fields = jnp.where(
        stand_still[:, None, None] & (~ force_do_smg[:, None, None]),
        sap_fields.at[enum_pos[0], enum_pos[1], enum_pos[2]].set(True),
        sap_fields.at[enum_pos[0], enum_pos[1], enum_pos[2]].set(False)
    )
    
    action_mask_fields = jnp.where(
        move_up[:, None, None] & (~ force_stay[:, None, None]),
        action_mask_fields.at[enum_pos[0], enum_pos[1], enum_pos[2]-1].set(True),
        action_mask_fields.at[enum_pos[0], enum_pos[1], enum_pos[2]-1].set(False)
    )
    
    action_mask_fields = jnp.where(
        move_right[:, None, None] & (~ force_stay[:, None, None]),
        action_mask_fields.at[enum_pos[0], enum_pos[1]+1, enum_pos[2]].set(True),
        action_mask_fields.at[enum_pos[0], enum_pos[1]+1, enum_pos[2]].set(False)
    )

    action_mask_fields = jnp.where(
        move_down[:, None, None] & (~ force_stay[:, None, None]),
        action_mask_fields.at[enum_pos[0], enum_pos[1], enum_pos[2]+1].set(True),
        action_mask_fields.at[enum_pos[0], enum_pos[1], enum_pos[2]+1].set(False)
    )

    action_mask_fields = jnp.where(
        move_left[:, None, None] & (~ force_stay[:, None, None]),
        action_mask_fields.at[enum_pos[0], enum_pos[1]-1, enum_pos[2]].set(True),
        action_mask_fields.at[enum_pos[0], enum_pos[1]-1, enum_pos[2]].set(False)
    )

    # converter from monofield action mask to action: does NOT account for unefficient sap locations, only for sap range
    monofield_base_converter = jnp.where(
        sappable_tiles,
        5,
        0
    )
    monofield_base_converter = monofield_base_converter.at[enum_pos[0], enum_pos[1], enum_pos[2]].set(0)
    monofield_base_converter = jnp.where(
        move_up[:, None, None],
        monofield_base_converter.at[enum_pos[0], enum_pos[1], enum_pos[2]-1].set(1),
        monofield_base_converter.at[enum_pos[0], enum_pos[1], enum_pos[2]-1].set(0)
    )

    monofield_base_converter = jnp.where(
        move_right[:, None, None],
        monofield_base_converter.at[enum_pos[0], enum_pos[1]+1, enum_pos[2]].set(2),
        monofield_base_converter.at[enum_pos[0], enum_pos[1]+1, enum_pos[2]].set(0)
    )

    monofield_base_converter = jnp.where(
        move_down[:, None, None],
        monofield_base_converter.at[enum_pos[0], enum_pos[1], enum_pos[2]+1].set(3),
        monofield_base_converter.at[enum_pos[0], enum_pos[1], enum_pos[2]+1].set(0)
    )

    monofield_base_converter = jnp.where(
        move_left[:, None, None],
        monofield_base_converter.at[enum_pos[0], enum_pos[1]-1, enum_pos[2]].set(4),
        monofield_base_converter.at[enum_pos[0], enum_pos[1]-1, enum_pos[2]].set(0)
    ).astype(jnp.int16)

    base_action_monofield = (monofield_base_converter < 5) & action_mask_fields # monofield_base_converter has all zeros beyond sap range, action_mask_fields takes care of this
    # unit_dist_1 = memory.distance_maps <= 1
    # base_action_duofield = action_mask_fields & unit_dist_1

    # jax.debug.print("{}, {}", memory.steps, monofield_base_converter[0])

    sap_action_monofield = (monofield_base_converter == 5) & action_mask_fields

    memory = memory.replace(
        # base_action_mask_discrete = base_actions,
        # sap_action_mask_discrete = sap_range_maps.reshape((max_units, -1)),  # does not account for unefficient sap location
        action_mask_monofield = action_mask_fields,  # = base_action_monofield + sap_action_monofield
        base_action_monofield = base_action_monofield,
        sap_action_monofield = sap_action_monofield,
        noop_action_monofield = jnp.zeros_like(memory.noop_action_monofield, dtype=bool).at[enum_pos[0], enum_pos[1], enum_pos[2]].set(True),
        # base_action_duofield = base_action_duofield,
        monofield_base_converter = monofield_base_converter,
        # sap_action_mask_field = sap_fields,  # same as sap_action_monofield, but include the 5 extra possible sap location (middle cross)
        contact_efficiency_map = contact_efficiency_map,
        sap_efficiency_map=sap_efficiency_map,
        invisible_point_tiles_sap_map=invisible_point_tiles_sap_map,
    )
    
    return memory

def deduce_sap_dropoff(memory):

    # easier to calculate if opponent unit moved
    opp_moved = jnp.any(memory.previous_opp_unit_positions != memory.opp_unit_positions, axis=1) & memory.opp_unit_mask & (memory.previous_opp_unit_energy > -1)
    sap_mask = (memory.last_action[:,0]==5) & memory.unit_mask & (memory.previous_unit_energy > 0)
    sap_locations = memory.last_action[:,1:] + memory.unit_positions
    sap_distances = jnp.max(jnp.abs(memory.opp_unit_positions[:,None,:]-sap_locations[None,:,:]), axis=2, where=opp_moved[:,None,None]*sap_mask[None,:,None], initial=-1)
    opp_sapped_next = jnp.sum(sap_distances == 1, axis=1)
    opp_sapped_mask = opp_sapped_next > 0
    opp_sapped_direct = jnp.sum(sap_distances == 0, axis=1)
    opp_exp_energy = jnp.clip(memory.previous_opp_unit_energy + memory.energy_field[memory.opp_unit_positions[:,0],  memory.opp_unit_positions[:,1]] - memory.unit_move_cost - opp_sapped_direct * memory.unit_sap_cost, max=400)

    opp_far = jnp.all(jnp.linalg.norm((memory.opp_unit_positions[:,None,:]-memory.unit_positions[None,:,:]), axis=2) > 1, axis=1) # opp_units out of sight are [-1,-1], thus at least sqrt(2) away
    
    opp_diff_energy_factor = jnp.where(
        opp_sapped_mask & opp_far,
        (opp_exp_energy - memory.opp_unit_energy)/memory.unit_sap_cost/opp_sapped_next,
        0.0
    )

    sap_dropoff = jnp.max(opp_diff_energy_factor)
    sap_dropoff = lax.cond(
        sap_dropoff > 0.125,
        lambda: jnp.round(4*sap_dropoff)/4,
        lambda: -1.
    )

    memory = memory.replace(
        unit_sap_dropoff_factor=jnp.float16(sap_dropoff)
    )
    
    return memory

def deduce_energy_void(memory):
    last_action_cost = memory.unit_move_cost*((memory.last_action[:,0]<5) & (memory.last_action[:,0]>0)).astype(jnp.int16) + memory.unit_sap_cost*(memory.last_action[:,0]==5).astype(jnp.int16)*(memory.unit_sap_cost <= memory.unit_energy).astype(jnp.int16)

    expected_energy = jnp.max(jnp.array([memory.previous_unit_energy - last_action_cost, jnp.zeros(max_units, dtype=jnp.int16)]), axis=0)
    
    distance_maps = distances_to_points(memory.unit_positions, jnp.sum)
    neighbouring_tiles = distance_maps == 1
    opp_neighbouring_tile = neighbouring_tiles & memory.opp_unit_map.astype(jnp.bool)
    next_to_opp_mask = jnp.any(opp_neighbouring_tile, axis=(1,2))
    expected_energy_next_to_opp = jnp.sum(expected_energy*next_to_opp_mask.astype(jnp.int16), axis=0)
    observed_energy_next_to_opp = jnp.sum((memory.unit_energy - memory.energy_field[memory.unit_positions.T[0],memory.unit_positions.T[1]])*next_to_opp_mask.astype(jnp.int16), axis=0)
   
    observed_opp_energy_map = (memory.opp_energy_map - memory.energy_field)* jnp.any(opp_neighbouring_tile, axis=0).astype(jnp.int16)

    n_own_tiles = jnp.count_nonzero(jnp.any(distance_maps == 0, where=next_to_opp_mask[:,None,None], axis=0), axis=(0,1))
    n_opp_tiles = jnp.count_nonzero(observed_opp_energy_map, axis=(0,1))
    
    exp_plE = expected_energy_next_to_opp
    obs_plE = observed_energy_next_to_opp
    obs_oppE = jnp.sum(observed_opp_energy_map, axis=(0,1))
    
    delta = (-obs_oppE+jnp.sqrt(obs_oppE*obs_oppE - 4*exp_plE*(obs_plE-exp_plE)))/(2*exp_plE)
    delta = jnp.where(
        (delta == 0) | (exp_plE-obs_plE < 4) | (exp_plE <= 0)  | (obs_plE <= 0) | (obs_oppE < 4) | (n_own_tiles != n_opp_tiles), # TODO: rewrite in 2D, so that n_own_tiles != n_opp_tiles is not necessary
        -1,
        delta
    )

    n = jnp.round(3/4/delta)
    n = jnp.where(
        n > 3,
        3*(n//3),
        n
    )
    memory = memory.replace(
        unit_energy_void_factor = jnp.float16(3/4/n)
    )
    return memory
