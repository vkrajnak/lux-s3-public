import numpy as np
import jax
import jax.numpy as jnp
import sys
from jax import lax
from flax import struct


@struct.dataclass
class MemoryState:
    """To store information about the game"""

    team_id: int
    force_stay_on_point_tile: bool
    force_no_noop: bool
    force_sap_location: bool
    force_avoid_collision: bool

    # game parameters
    max_units: int
    unit_move_cost: int
    unit_sap_cost: int
    unit_sap_range: int
    unit_sap_dropoff_factor: float
    unit_energy_void_factor: float
    unit_sensor_range: int
    match_count_per_episode: int
    max_steps_in_match: int
    map_size: int

    # units
    unit_positions: jnp.ndarray
    unit_energy: jnp.ndarray
    unit_mask: jnp.ndarray
    unit_field: jnp.ndarray
    unit_map: jnp.ndarray
    energy_map: jnp.ndarray
    vision_map: jnp.ndarray
    previous_unit_energy: jnp.ndarray
    previous_unit_mask: jnp.ndarray
    previous_unit_positions: jnp.ndarray
    distance_maps: jnp.ndarray
    previous_energy_map: jnp.ndarray
    n_units_map: jnp.ndarray
    previous_n_units_map: jnp.ndarray
    contact_efficiency_map: jnp.ndarray
    sap_efficiency_map: jnp.ndarray
    invisible_point_tiles_sap_map: jnp.ndarray
    current_sensor_mask: jnp.ndarray

    opp_unit_positions: jnp.ndarray
    opp_unit_energy: jnp.ndarray
    opp_unit_mask: jnp.ndarray
    opp_unit_map: jnp.ndarray
    opp_energy_map: jnp.ndarray
    previous_opp_unit_positions: jnp.ndarray
    previous_opp_unit_energy: jnp.ndarray
    previous_opp_energy_map: jnp.ndarray
    n_opp_units_map: jnp.ndarray
    previous_n_opp_units_map: jnp.ndarray
    
    last_action: jnp.ndarray

    # match info
    steps: int
    match_steps: int

    points: int
    previous_points: int
    opp_points: int
    opp_points_gain: int

    wins: int
    losses: int
    match_num: int

    # energy
    energy_field: jnp.ndarray
    tile_type_field: jnp.ndarray
    next_tile_type_field: jnp.ndarray
    valid_energy_mask: jnp.ndarray
    possible_energy_fields: jnp.ndarray
    possible_energy_fields_compare: jnp.ndarray
    global_energy_field: bool
    energy_field_including_nebula: jnp.ndarray

    # drift
    energy_node_drift_frequency: int
    nebula_drift_direction: int
    nebula_drift_speed: int

    # nebula
    nebula_vision_reduction: int
    nebula_energy_reduction: int

    # relics
    max_relics: int
    relics: jnp.ndarray
    relics_mask: jnp.ndarray
    relics_map: jnp.ndarray
    relics_found: int
    relics_on_diagonal: int
    relic_area_size: int
    all_relics_found: bool
    tiles_around_relic: jnp.ndarray
    explored_for_relic_mask: jnp.ndarray
    
    # point tiles
    potential_point_tiles: jnp.ndarray
    point_tiles: jnp.ndarray
    empty_tiles: jnp.ndarray
    approx_dist_to_point_tiles: jnp.ndarray
    all_point_tiles_found: jnp.ndarray

    # visited tiles
    recently_visited_map: jnp.ndarray
    opp_recently_visited_map: jnp.ndarray

    # actions
    # base_action_mask_discrete: jnp.ndarray
    # sap_action_mask_discrete: jnp.ndarray
    action_mask_monofield: jnp.ndarray
    base_action_monofield: jnp.ndarray
    sap_action_monofield: jnp.ndarray
    noop_action_monofield: jnp.ndarray
    # base_action_duofield: jnp.ndarray
    monofield_base_converter: jnp.ndarray
    relative_coordinate_maps: jnp.ndarray
    # sap_action_mask_field: jnp.ndarray

    # aux
    coordinate_map: jnp.ndarray
    false_map: jnp.ndarray
    true_map: jnp.ndarray
    false_map12: jnp.ndarray
    false_map16: jnp.ndarray
    false_unit_mask: jnp.ndarray

    @property 
    def opp_id(self):
        return 1 - self.team_id
    
    @property 
    def xmax(self):
        return self.map_size - 1

    @property 
    def ymax(self):
        return self.map_size - 1
    
    @property
    def relic_area_max_dist(self):
        return self.relic_area_size // 2

    # # aux
    # @property
    # def coordinate_map(self):
    #     mm = jnp.meshgrid(jnp.arange(self.map_size), jnp.arange(self.map_size))
    #     mm = jnp.stack([mm[0], mm[1]]).T.astype(jnp.int16)
    #     return mm

    # @property
    # def false_map(self):
    #     return jnp.zeros((self.map_size, self.map_size), dtype=jnp.bool)
    
    # @property
    # def true_map(self):
    #     return jnp.ones((self.map_size, self.map_size), dtype=jnp.bool)
    
    # @property
    # def false_map12(self):
    #     return jnp.zeros((12, self.map_size, self.map_size), dtype=jnp.bool)
        
    # @property
    # def false_map16(self):
    #     return jnp.zeros((16, self.map_size, self.map_size), dtype=jnp.bool)
