import jax
import jax.numpy as jnp

from ..types import AgentState
from ..constants import ENV_PARAMS_RANGES, EnvParams, MATCHES_TO_WIN
from .types import DTYPE


# --- public functions
def get_zero(agent_state: AgentState) -> jnp.ndarray:  # ()
    return jnp.zeros((), dtype=DTYPE)


def get_one(agent_state: AgentState) -> jnp.ndarray:  # ()
    return jnp.ones((), dtype=DTYPE)


# # game parameters known
def get_move_cost(agent_state: AgentState) -> jnp.ndarray:  # ()
    """Scaled to  [0,1], always known"""
    return DTYPE(_scale_to_unit_interval(agent_state.memory.unit_move_cost, ENV_PARAMS_RANGES["unit_move_cost"]))


def get_sap_cost(agent_state: AgentState) -> jnp.ndarray:  # ()
    """Scaled to [0,1], always known"""
    return DTYPE(_scale_to_unit_interval(agent_state.memory.unit_sap_cost, ENV_PARAMS_RANGES["unit_sap_cost"]))


def get_sap_range(agent_state: AgentState) -> jnp.ndarray:  # ()
    """Scaled to [0,1], always known"""
    return DTYPE(_scale_to_unit_interval(agent_state.memory.unit_sap_range, ENV_PARAMS_RANGES["unit_sap_range"]))


def get_sensor_range(agent_state: AgentState) -> jnp.ndarray:  # ()
    """Scaled to [0,1], always known"""
    return DTYPE(_scale_to_unit_interval(agent_state.memory.unit_sensor_range, ENV_PARAMS_RANGES["unit_sensor_range"]))


# # game parameters deduced

def get_nebula_tile_vision_reduction(agent_state: AgentState) -> jnp.ndarray:  # ()
    """Scaled to [0,1] or -1 if unknown"""
    scaled = _scale_to_unit_interval(agent_state.memory.nebula_vision_reduction, ENV_PARAMS_RANGES["nebula_tile_vision_reduction"])
    return DTYPE(jnp.where(agent_state.memory.nebula_vision_reduction < 0, -1, scaled))


def get_nebula_tile_energy_reduction(agent_state: AgentState) -> jnp.ndarray:  # ()
    """Scaled to [0,1] or -1 if unknown"""
    scaled = _scale_to_unit_interval(agent_state.memory.nebula_energy_reduction, ENV_PARAMS_RANGES["nebula_tile_energy_reduction"])
    return DTYPE(jnp.where(agent_state.memory.nebula_energy_reduction < 0, -1, scaled))


def get_sap_dropoff_factor(agent_state: AgentState) -> jnp.ndarray:  # ()
    """Scaled to [0,1] or -1 if unknown"""
    scaled = _scale_to_unit_interval(agent_state.memory.unit_sap_dropoff_factor, ENV_PARAMS_RANGES["unit_sap_dropoff_factor"])
    return DTYPE(jnp.where(agent_state.memory.unit_sap_dropoff_factor < 0, -1, scaled))


def get_energy_void_factor(agent_state: AgentState) -> jnp.ndarray:  # ()
    """Scaled to [0,1] or -1 if unknown"""
    scaled = _scale_to_unit_interval(agent_state.memory.unit_energy_void_factor, ENV_PARAMS_RANGES["unit_energy_void_factor"])
    return DTYPE(jnp.where(agent_state.memory.unit_energy_void_factor < 0, -1, scaled))


# # game info
def get_steps(agent_state: AgentState) -> jnp.ndarray:  # ()
    """In [0,1)"""
    return DTYPE(agent_state.memory.steps / EnvParams.max_steps_in_game)


def get_match_steps(agent_state: AgentState) -> jnp.ndarray:  # ()
    """In [0,1)"""
    return DTYPE(agent_state.memory.match_steps / EnvParams.max_steps_in_match)


def get_last_match_step(agent_state: AgentState) -> jnp.ndarray:  # ()
    """Binary: step after which win/loss is decided"""
    return DTYPE(agent_state.memory.match_steps == EnvParams.max_steps_in_match)


def get_restart_match_step(agent_state: AgentState) -> jnp.ndarray:  # ()
    """Binary: step with no action to take, binary"""
    return DTYPE(agent_state.memory.match_steps == 0)


def get_match_num(agent_state: AgentState) -> jnp.ndarray:  # ()
    """In [0, 1)"""
    return DTYPE(agent_state.memory.match_num / EnvParams.match_count_per_episode)


def get_wins(agent_state: AgentState) -> jnp.ndarray:  # ()
    """In [0,1)"""
    return DTYPE(agent_state.memory.wins / EnvParams.match_count_per_episode)


def get_losses(agent_state: AgentState) -> jnp.ndarray:  # ()
    """In [0,1)"""
    return DTYPE(agent_state.memory.losses / EnvParams.match_count_per_episode)


def get_game_over(agent_state: AgentState) -> jnp.ndarray:  # ()
    return DTYPE((agent_state.memory.wins >= MATCHES_TO_WIN) | (agent_state.memory.losses >= MATCHES_TO_WIN))


def get_points(agent_state: AgentState) -> jnp.ndarray:  # ()
    """typically in [0,~1]"""
    return DTYPE(agent_state.memory.points / 1000)


def get_points_increment(agent_state: AgentState) -> jnp.ndarray:  # ()
    """typically in [0,~1]"""
    return DTYPE((agent_state.memory.points - agent_state.memory.previous_points) / 30)


def get_opponent_points(agent_state: AgentState) -> jnp.ndarray:  # ()
    """typically in [0,~1]"""
    return DTYPE(agent_state.memory.opp_points / 1000)


def get_opponent_points_increment(agent_state: AgentState) -> jnp.ndarray:  # ()
    """typically in [0,~1]"""
    return DTYPE(agent_state.memory.opp_points_gain / 30)


def get_opponent_points_gain(agent_state: AgentState) -> jnp.ndarray:  # ()
    """bool"""
    return DTYPE(agent_state.memory.opp_points_gain <= agent_state.memory.points - agent_state.memory.previous_points)


def get_game_status(agent_state: AgentState) -> jnp.ndarray:  # ()
    """bool"""
    return DTYPE(agent_state.memory.opp_points < agent_state.memory.points)


def get_all_relics_found(agent_state: AgentState) -> jnp.ndarray:  # ()
    return DTYPE(agent_state.memory.all_relics_found)


def get_all_point_tiles_found(agent_state: AgentState) -> jnp.ndarray:  # ()
    return DTYPE(agent_state.memory.all_point_tiles_found)


# --- private functions (helpers)

def _scale_to_unit_interval(value, possible_values):
    min_val = min(possible_values)
    max_val = max(possible_values)
    return (value - min_val) / (max_val - min_val)


GET_SCALAR_FN = {
    "zero": get_zero,
    "one": get_one,
    "move_cost": get_move_cost,
    "sap_cost": get_sap_cost,
    "sap_range": get_sap_range,
    "sensor_range": get_sensor_range,
    "nebula_tile_vision_reduction": get_nebula_tile_vision_reduction,
    "nebula_tile_energy_reduction": get_nebula_tile_energy_reduction,
    "sap_dropoff_factor": get_sap_dropoff_factor,
    "energy_void_factor": get_energy_void_factor,
    "steps": get_steps,
    "match_steps": get_match_steps,
    "last_match_step": get_last_match_step,
    "restart_match_step": get_restart_match_step,
    "match_num": get_match_num,
    "wins": get_wins,
    "losses": get_losses,
    "game_over": get_game_over,
    "points": get_points,
    "points_increment": get_points_increment,
    "opponent_points": get_opponent_points,
    "opponent_points_increment": get_opponent_points_increment,
    "opponent_points_gain": get_opponent_points_gain,
    "all_relics_found": get_all_relics_found,
    "all_point_tiles_found": get_all_point_tiles_found,
    "game_status": get_game_status,
}
