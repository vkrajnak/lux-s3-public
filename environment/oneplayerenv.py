"""Provides a single-player Jumanji env build on top of lux env that can be used with the RL code"""
import os
import sys
import importlib.util
import time
import random
import string

import jax
import jax.numpy as jnp
import chex
import functools

from functools import cached_property
from typing import Union

import numpy as np
from flax import struct
from omegaconf import OmegaConf, DictConfig

from external.jumanji.env import Environment as JumanjiEnvironment
from external.jumanji.types import TimeStep, restart, termination, transition

from external.lux.luxai_s3.params import EnvParams as LuxEnvParams
from external.lux.luxai_s3.params import env_params_ranges as lux_env_params_ranges
from external.lux.luxai_s3.env import LuxAIS3Env
from external.lux.luxai_s3.state import EnvState as LuxEnvState
from external.lux.luxai_s3.state import EnvObs as LuxEnvObs
from external.lux.luxai_s3.state import UnitState

from player.rl_agent.agent import AgentState
from player.rl_agent.agent import Agent as Player
from player.rl_agent.specs import Array as SpecArray
from player.rl_agent.constants import N_MAX_UNITS, PLAYER_ID, MATCHES_TO_WIN, N_BASE_ACTIONS
from player.rl_agent.memory.memory_jax import place_points_on_field
from player.rl_agent.memory.energy_fields_opt import coords_distances_to_points
OPPONENT_ID = 1 - PLAYER_ID

ZERO_INT = jnp.zeros((), dtype=jnp.int16)


@struct.dataclass
class Record:
    return_contributions: dict
    # action chosen by the player
    cum_noop_action: int  # number of times the no-op action was chosen over the 5 matches
    cum_move_action: int  # number of times the move action was chosen over the 5 matches
    cum_sap_action: int  # number of times the no-op action was chosen over the 5 matches
    # from lux engine
    cum_units_removed: int  # > = 0
    cum_units_moved: int  # > = 0
    cum_units_sapped: int  # >= 0
    cum_sapping_energy_benefit: int  # positive or negative
    cum_n_units_destroyed_by_collision_delta: int
    cum_energy_from_sap_delta: int  # either sign
    cum_energy_from_void_delta: int  # <=0 negative
    cum_energy_from_field: int   # typically positive, cumulated over the 5 matches
    cum_units_spawned: int  # >= 0
    # winning related
    matches_won: int  # between 0 and 5
    matches_lost: int  # between 0 and 5
    game_status: int  # 1 if won, -1 if lost, 0 otherwise
    # points
    points_player: int  # >= 0, during current match
    points_opponent: int  # >= 0, during current match
    cum_points_player: int  # >= 0, cumulated over the 5 matches
    cum_points_opponent: int  # >= 0, cumulated over the 5 matches
    # exploration
    tiles_explored: int  # between 0 and 24^2 = 576
    cum_tiles_explored: int  # between 0 and 5*24^2 = 2880, cumulated over the 5 matches
    point_tiles_found: int  # between 0 and ?
    cum_point_tiles_found: int  # between 0 and 5*?, cumulated over the 5 matches
    relics_found: int  # between 0 and 6
    cum_relics_found: int  # between 0 and 6, cumulated over the 5 matches
    all_relics_found: int  # 0 or 1
    cum_all_relics_found: int  # between 0 and ?, cumulated over the 5 matches
    all_point_tiles_found: int
    cum_all_point_tiles_found: int
    first_relic_found_in_first_match: int
    cum_first_relic_found_in_first_match: int
    # units
    units_energy: int  # typically positive but can be 0 or negative
    cum_units_energy: int  # typically positive, cumulated over the 5 matches
    cum_point_tiles_unoccupied: int
    cum_units_stuck_by_energy: int  # >= 0
    point_tiles_dist: float
    cum_point_tiles_dist: float


@struct.dataclass
class OnePlayerEnvState:
    # fixed during one game:
    params_lux: LuxEnvParams
    units_permutation: jnp.ndarray  # permutation of units_ids
    units_unpermutation: jnp.ndarray  # inverse of units_permutation
    # updated at each step:
    state_lux: LuxEnvState
    player_last_obs_lux: LuxEnvObs
    opponent_last_obs_lux: LuxEnvObs
    player_state: AgentState
    opponent_state: AgentState
    is_terminal: bool  # whether the game has terminated
    key: chex.PRNGKey
    record: Record


class OnePlayerEnv(JumanjiEnvironment):
    """Game environment that complies with the Jumanji API, based on lux engine where the opponent is part of the environment"""
    def __init__(
            self,
            config_env: DictConfig,
            opponents_params: dict = {
                "opponents_pool_dirs": ["./all_opponents", ],
                "use_selfplay": False,
                "selfplay_kwargs": {
                    "opponent_actor_params": None,
                },

            },
    ):

        self.early_stopping = config_env.early_stopping
        self.reward_coefficients = OmegaConf.to_container(config_env.reward.reward_coefficients, resolve=True)
        self.permutation = config_env.permutation_rate > 0.0
        self.permutation_rate = config_env.permutation_rate

        # instantiate lux env
        lux_autoreset = False  # auto_reset is managed elsewhere
        self.lux = LuxAIS3Env(auto_reset=lux_autoreset, fixed_env_params=LuxEnvParams())

        # instantiate agent
        self.player = Player(config_env.agent, path=None, training=True)

        # instantiate an opponent
        if opponents_params["use_selfplay"]:
            # load previous version of same agent
            self.opponent_dir = None
            self.opponent_name = "auto-self-play"
            self.opponent = Player(config_env.agent, path=opponents_params["selfplay_kwargs"]["opponent_actor_params"])
            assert not self.opponent.is_dummy

        else:
            # load list of all opponents contained in the pool
            opponents_pool_dirs = opponents_params["opponents_pool_dirs"]
            opponents_dirs = []
            for opponents_pool_dir in opponents_pool_dirs:
                opponents_paths = [f.path for f in os.scandir(opponents_pool_dir) if f.is_dir()]
                for opponent_path in opponents_paths:
                    opponent_file = os.path.join(opponent_path, "wrapper.py")
                    if os.path.isfile(opponent_file):
                        opponents_dirs.append(opponent_path)

            # print(f"Found {len(opponents_dirs)} opponent(s) in {opponents_pool_dirs}: {opponents_dirs}")
            chosen_opponent_idx = random.randint(0, len(opponents_dirs) - 1)
            opponent_dir = opponents_dirs[chosen_opponent_idx]
            self.opponent_dir = opponent_dir
            self.opponent_name = os.path.basename(opponent_dir)
            self.opponent = import_opponent(opponent_dir)

        super().__init__()

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> tuple[OnePlayerEnvState, TimeStep]:
        key, key_reset, key_player_id, key_opponent_idx, key_params, key_permutate, key_permutation = jax.random.split(key, 7)

        # create lux parameters and those sent to initialize agents
        params_lux, env_cfg_for_agent = create_lux_randomized_parameters(key_params)

        # create random units permutation
        permutate = jax.random.uniform(key_permutate) < self.permutation_rate
        units_permutation = jnp.where(
            permutate,
            jax.random.permutation(key_permutation, jnp.arange(N_MAX_UNITS, dtype=jnp.int16)),
            jnp.arange(N_MAX_UNITS, dtype=jnp.int16)
        )

        # initialize the opponent
        opponent_state = self.opponent.init(player_id=OPPONENT_ID, env_cfg=env_cfg_for_agent)

        # initialize player
        player_state = self.player.init(player_id=PLAYER_ID, env_cfg=env_cfg_for_agent)

        # call lux engine reset
        obs_two_agents_lux, state_lux = self.lux.reset(key_reset, params_lux)
        obs_player_lux = obs_two_agents_lux[f"player_{PLAYER_ID}"]
        obs_opponent_lux = obs_two_agents_lux[f"player_{OPPONENT_ID}"]

        # fix types difference between lux step and reset
        state_lux = fix_lux_type(state_lux)
        obs_player_lux = fix_lux_type(obs_player_lux)
        obs_opponent_lux = fix_lux_type(obs_opponent_lux)

        # apply units permutation to obs
        if self.permutation:
            obs_player_lux = permutate_obs(obs_player_lux, units_permutation)

        # compute the new player state and compute input for nn
        player_state, nn_input = self.player.process_obs_for_nn_and_update_state(player_state, obs_player_lux)

        # create record
        return_contributions = {
            # win-related, zero-sum
            "win_game": 0.0,
            "win_matches": 0.0,
            "delta_points": 0.0,
            # basic game understanding
            "points": 0.0,
            "units_energy": 0.0,
            # exploration
            "tiles_explored": 0.0,
            "point_tiles_found": 0.0,
            "relics_found": 0.0,
            "all_relics_found": 0.0,
            "all_point_tiles_found": 0.0,
            "first_relic_found_in_first_match": 0.0,
            # other from obs or memory
            "units_stuck_by_energy": 0.0,
            "point_tiles_unoccupied": 0.0,
            # from lux engine
            "units_removed": 0.0,
            "units_moved": 0.0,
            "units_sapped": 0.0,
            "sapping_energy_benefit": 0.0,
            "delta_collision": 0.0,
            "delta_energy_sap": 0.0,
            "delta_energy_void": 0.0,
            "energy_from_field": 0.0,
            "point_tiles_dist": 0.0,
        }
        record = Record(
            return_contributions=return_contributions,
            # 
            cum_noop_action=ZERO_INT,
            cum_move_action=ZERO_INT,
            cum_sap_action=ZERO_INT,
            #
            cum_units_removed=ZERO_INT,
            cum_units_moved=ZERO_INT,
            cum_units_sapped=ZERO_INT,
            cum_sapping_energy_benefit=ZERO_INT,
            cum_n_units_destroyed_by_collision_delta=ZERO_INT,
            cum_energy_from_sap_delta=ZERO_INT,
            cum_energy_from_void_delta=ZERO_INT,  # negative
            cum_energy_from_field=ZERO_INT,  # typically positive, cumulated over the 5 matches
            cum_units_spawned=ZERO_INT,
            #
            matches_won=ZERO_INT,
            matches_lost=ZERO_INT,
            game_status=ZERO_INT,
            #
            points_player=ZERO_INT,
            points_opponent=ZERO_INT,
            cum_points_player=ZERO_INT,
            cum_points_opponent=ZERO_INT,
            #
            tiles_explored=ZERO_INT,
            cum_tiles_explored=ZERO_INT,
            point_tiles_found=ZERO_INT,
            cum_point_tiles_found=ZERO_INT,
            relics_found=ZERO_INT,
            cum_relics_found=ZERO_INT,
            all_relics_found=ZERO_INT,
            cum_all_relics_found=ZERO_INT,
            all_point_tiles_found=ZERO_INT,
            cum_all_point_tiles_found=ZERO_INT,
            first_relic_found_in_first_match=ZERO_INT,
            cum_first_relic_found_in_first_match=ZERO_INT,
            #
            units_energy=ZERO_INT,
            cum_units_energy=ZERO_INT,
            cum_units_stuck_by_energy=ZERO_INT,
            cum_point_tiles_unoccupied=ZERO_INT,
            point_tiles_dist=jnp.float16(0.),
            cum_point_tiles_dist=ZERO_INT,
        )

        # create state
        state = OnePlayerEnvState(
            params_lux=params_lux,
            units_permutation=units_permutation,
            units_unpermutation=jnp.argsort(units_permutation).astype(jnp.int16),
            state_lux=state_lux,
            player_last_obs_lux=obs_player_lux,
            opponent_last_obs_lux=obs_opponent_lux,
            player_state=player_state,
            opponent_state=opponent_state,
            is_terminal=False,
            key=key,
            record=record,
        )

        # auxiliary stuff we want to monitor at the end of the episode during training to assess progress
        info = {
            "episode_metrics": {
                "return_contributions": return_contributions,
                # from action chosen
                "action_noop_chosen": ZERO_INT,
                "action_move_chosen": ZERO_INT,
                "action_sap_chosen": ZERO_INT,
                # from obs/memory
                "win_game": ZERO_INT,
                "win_matches": ZERO_INT,
                "points": ZERO_INT,
                "delta_points": ZERO_INT,
                "tiles_explored": ZERO_INT,
                "point_tiles_found": ZERO_INT,
                "relics_found": ZERO_INT,
                "all_relics_found": ZERO_INT,
                "all_point_tiles_found": ZERO_INT,
                "first_relic_found_in_first_match": ZERO_INT,
                "units_energy": ZERO_INT,
                "units_stuck_by_energy": ZERO_INT,
                "point_tiles_unoccupied": ZERO_INT,
                "point_tiles_dist": ZERO_INT,
                # from lux engine
                "units_removed": ZERO_INT,
                "action_move_executed": ZERO_INT,
                "action_sap_executed": ZERO_INT,
                "delta_collision": ZERO_INT,
                "delta_energy_sap": ZERO_INT,
                "delta_energy_void": ZERO_INT,
                "sapping_energy_benefit": ZERO_INT,
                "energy_from_field": ZERO_INT,
                "units_spawned": ZERO_INT,
            }
        }

        # pack everything into timestep
        timestep = restart(observation=nn_input, extras=info)

        return state, timestep

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: OnePlayerEnvState, action: jnp.array) -> tuple[OnePlayerEnvState, TimeStep]:
        """action is resulting from nn_output.sample() so it needs some conversion before being fed into lux"""

        # ------------- APPLY ACTION AND GET NEW LUX OBSERVATION -------------

        key = state.key
        key, key_action_opponent, key_step = jax.random.split(key, 3)
        seed_action_opponent = jax.random.randint(key_action_opponent, shape=(), minval=0, maxval=1000000, dtype=int)

        # get opponent action
        opponent_state, action_opponent_lux = self.opponent.act(seed_action_opponent, state.opponent_state, state.opponent_last_obs_lux)

        # convert actor action (sampled for nn output distribution) to lux action
        nn_action = action
        player_state, action_player_lux = self.player.process_action_from_nn_and_update_state(state.player_state, nn_action)

        # monitor action chosen (for training curves) - if state.player_state.memory.match_steps == 0, no action will be executed cause units will be reset
        noop_action = action_player_lux[:, 0] == 0  # (N_MAX_UNITS)
        sap_action = action_player_lux[:, 0] == N_BASE_ACTIONS - 1  # (N_MAX_UNITS)
        move_action = jnp.logical_and(jnp.logical_not(noop_action), jnp.logical_not(sap_action))  # (N_MAX_UNITS)
        units_mask = player_state.memory.unit_mask  # (N_MAX_UNITS)  # note that this mask is provided by lux, it does account for move energy cost etc
        noop_action = jnp.sum(jnp.logical_and(noop_action, units_mask))  # ()
        sap_action = jnp.sum(jnp.logical_and(sap_action, units_mask))  # ()
        move_action = jnp.sum(jnp.logical_and(move_action, units_mask))  # ()
        cum_noop_action = jnp.array(state.record.cum_noop_action + noop_action, dtype=jnp.int16)
        cum_sap_action = jnp.array(state.record.cum_sap_action + sap_action, dtype=jnp.int16)
        cum_move_action = jnp.array(state.record.cum_move_action + move_action, dtype=jnp.int16)

        # unpermutate the action
        if self.permutation:
            action_player_lux = unpermutate_action(action_player_lux, state.units_unpermutation)

        # combine actions in a dict
        action_two_agents_lux = dict()
        action_two_agents_lux[f"player_{PLAYER_ID}"] = action_player_lux
        action_two_agents_lux[f"player_{OPPONENT_ID}"] = action_opponent_lux

        # calling lux engine, reward_lux is the current number of matches won in the series of 5 matches
        obs_two_agents_lux, state_lux, reward_lux, _, truncated_dict_lux, info_lux = self.lux.step(key_step, state.state_lux, action_two_agents_lux, state.params_lux)  # terminated_dict does nothing in lux
        # obs_two_agent: {"player_0": LuxEnvObs, "player_1": LuxEnvObs}
        # state_lux: LuxEnvState
        # reward_lux: {"player_0": int, "player_1": int}  # int are actually 0-dimensional jax arrays
        # truncated_dict_lux: {"player_0": bool, "player_1": bool}
        # info_lux: dict
        report = info_lux["report"]

        obs_player_lux = obs_two_agents_lux[f"player_{PLAYER_ID}"]  # LuxEnvObs
        obs_opponent_lux = obs_two_agents_lux[f"player_{OPPONENT_ID}"]  # LuxEnvObs
        matches_won = reward_lux[f"player_{PLAYER_ID}"].astype(jnp.int16)  # int (0-dimensional array)
        matches_lost = reward_lux[f"player_{OPPONENT_ID}"].astype(jnp.int16)  # int (0-dimensional array)
        done_lux = truncated_dict_lux[f"player_{PLAYER_ID}"]  # bool, True when series of 5 matches is over (both players have the same 'truncated')

        # apply units permutation to obs
        if self.permutation:
            obs_player_lux = permutate_obs(obs_player_lux, state.units_permutation)

        # compute the new player state and compute input for nn
        player_state, nn_input = self.player.process_obs_for_nn_and_update_state(player_state, obs_player_lux)

        # ------------- COMPUTE USEFUL QUANTITIES FOR REWARD AND MONITORING TRAINING PROGRESS -------------
        # these quantities are saved in Record so that we can compute the increment between successive time steps
        
        def zero_if_match_restart(array):
            return jnp.where(report.all_units_removed_due_to_reset, jnp.zeros_like(array), array)

        def non_negative(array):
            return jnp.where(array < 0, jnp.zeros_like(array), array)

        ## -------------- from internal lux engine:
        
        # units removed during this step, as they had negative energy at the beginning
        units_removed_due_to_negative_energy = jnp.sum(report.units_removed_due_to_negative_energy[PLAYER_ID, :], dtype=jnp.int16)
        cum_units_removed = state.record.cum_units_removed + units_removed_due_to_negative_energy

        # units have moved during this step (had enough energy, no asteroid...)
        units_moved = jnp.sum(report.units_have_moved[PLAYER_ID, :], dtype=jnp.int16)
        cum_units_moved = state.record.cum_units_moved + units_moved

        # units that have actually sapped during this step (had enough energy etc)
        units_sapped = jnp.sum(report.units_have_sapped[PLAYER_ID, :], dtype=jnp.int16)
        cum_units_sapped = state.record.cum_units_sapped + units_sapped
        
        # result of sapping: damage done to opponent - sapping cost for player
        sapping_energy_benefit = report.sapping_energy_benefit[PLAYER_ID]
        cum_sapping_energy_benefit = state.record.cum_sapping_energy_benefit + sapping_energy_benefit

        # variation of number of units due to collision
        n_units_destroyed_by_collision_delta = (report.n_units_destroyed_by_collision[OPPONENT_ID] - report.n_units_destroyed_by_collision[PLAYER_ID]).astype(jnp.int16)
        cum_n_units_destroyed_by_collision_delta = state.record.cum_n_units_destroyed_by_collision_delta + n_units_destroyed_by_collision_delta

        # energy variation from sapping resolution
        energy_variation_from_sap_player = jnp.sum(report.energy_variation_from_sap[PLAYER_ID, :], dtype=jnp.int16)
        energy_variation_from_sap_opponent = jnp.sum(report.energy_variation_from_sap[OPPONENT_ID, :], dtype=jnp.int16)
        energy_variation_from_sap_delta = energy_variation_from_sap_player - energy_variation_from_sap_opponent
        cum_energy_from_sap_delta = state.record.cum_energy_from_sap_delta + energy_variation_from_sap_delta

        # energy variation from energy void resolution
        energy_variation_from_void_player = jnp.sum(report.energy_variation_from_void[PLAYER_ID, :], dtype=jnp.int16)
        energy_variation_from_void_opponent = jnp.sum(report.energy_variation_from_void[OPPONENT_ID, :], dtype=jnp.int16)
        energy_variation_from_void_delta = energy_variation_from_void_player - energy_variation_from_void_opponent
        cum_energy_from_void_delta = state.record.cum_energy_from_void_delta + energy_variation_from_void_delta

        # energy variation from energy field
        energy_variation_from_field = jnp.sum(report.energy_variation_from_field[PLAYER_ID, :], dtype=jnp.int16)
        cum_energy_from_field = state.record.cum_energy_from_field + energy_variation_from_field

        # units that have spawned
        units_spawned = jnp.sum(report.units_have_spawned[PLAYER_ID, :], dtype=jnp.int16)
        cum_units_spawned = state.record.cum_units_spawned + units_spawned
        
        ## -------------- from obs and memory

        # detect if player just won/lost at this step (1 game = series of 5 matches)
        game_is_won = matches_won >= MATCHES_TO_WIN
        game_is_lost = matches_lost >= MATCHES_TO_WIN
        game_status = game_is_won.astype(jnp.int16) - game_is_lost.astype(jnp.int16)
        game_status_diff = game_status - state.record.game_status  # 1/-1 means player has just won/lost, 0 means no change compared to previous step

        # detect if player just won a match
        matches_won_diff = matches_won - state.record.matches_won
        matches_lost_diff = matches_lost - state.record.matches_lost

        # points
        points_player = jnp.array(obs_player_lux.team_points[PLAYER_ID], dtype=jnp.int16)
        points_player_diff = non_negative(points_player - state.record.points_player)
        cum_points_player = state.record.cum_points_player + points_player_diff

        points_opponent = jnp.array(obs_player_lux.team_points[OPPONENT_ID], dtype=jnp.int16)
        points_opponent_diff = non_negative(points_opponent - state.record.points_opponent)
        cum_points_opponent = state.record.cum_points_opponent + points_opponent_diff

        # all point tile found in this match
        all_point_tiles_found = jnp.array(player_state.memory.all_point_tiles_found.astype(jnp.int16))
        all_point_tiles_found_diff = non_negative(all_point_tiles_found - state.record.all_point_tiles_found)
        cum_all_point_tiles_found = state.record.cum_all_point_tiles_found + all_point_tiles_found_diff

        # point tile occupied
        point_tiles_occupied = jnp.sum(jnp.logical_and(player_state.memory.point_tiles, player_state.memory.unit_map), dtype=jnp.int16)
        point_tiles_to_occupy_if_all_found = jnp.clip(jnp.sum(player_state.memory.point_tiles) // 2 + jnp.int16(points_player <= points_opponent), max=N_MAX_UNITS)
        point_tiles_to_occupy_otherwise = N_MAX_UNITS
        point_tiles_to_occupy = jax.lax.cond(player_state.memory.all_point_tiles_found, lambda: point_tiles_to_occupy_if_all_found, lambda: point_tiles_to_occupy_otherwise)
        point_tiles_unoccupied = non_negative(point_tiles_to_occupy - point_tiles_occupied).astype(jnp.int16)
        cum_point_tiles_unoccupied = state.record.cum_point_tiles_unoccupied + point_tiles_unoccupied

        # point tile distance
        def point_dist_metric(memory):
            half_map = jnp.sum(memory.coordinate_map, axis=2) < memory.map_size
            half_point_tiles = half_map & memory.point_tiles
            _, manhattan_dist_map = coords_distances_to_points(memory.unit_positions, jnp.sum)
            dist_map = jnp.min(manhattan_dist_map * half_point_tiles, axis=0).ravel().astype(jnp.float16)
            over_N_MAX_UNITS = jnp.sum(half_point_tiles) - N_MAX_UNITS
            dist_map = jax.lax.cond(
                over_N_MAX_UNITS > 0,
                # lambda: jnp.sort(dist_map).at[-over_N_MAX_UNITS:].set(0),
                lambda: N_MAX_UNITS*dist_map/jnp.sum(half_point_tiles), # placeholder, we only want the 16 closest
                lambda: dist_map
            )
            dist = jnp.sum(jnp.sqrt(dist_map)) + 5 * jnp.clip(N_MAX_UNITS - jnp.sum(half_point_tiles), min=0) # 5 is sqrt of distance from [-1,-1] to diagonal
            unit_dists = jnp.min(manhattan_dist_map, where=memory.point_tiles, initial=memory.map_size, axis=(1,2)).astype(jnp.float16) # any point tile
            dist = dist + jnp.sum(jnp.sqrt(unit_dists))

            return dist.astype(jnp.float16)
    
        point_tiles_dist = jax.lax.cond(
            jnp.any(player_state.memory.point_tiles),
            lambda mem: point_dist_metric(mem),
            lambda mem: jnp.float16(2 * 5 * N_MAX_UNITS),
            player_state.memory
        )
        point_tiles_dist = zero_if_match_restart(point_tiles_dist).astype(jnp.float16)
        point_tiles_dist_diff = zero_if_match_restart(point_tiles_dist - state.record.point_tiles_dist)
        cum_point_tiles_dist = state.record.cum_point_tiles_dist + jnp.int16(point_tiles_dist)

        # tiles explored
        tiles_explored = jnp.sum(player_state.memory.explored_for_relic_mask, dtype=jnp.int16)
        tiles_explored_diff = zero_if_match_restart(tiles_explored - state.record.tiles_explored)
        cum_tiles_explored = state.record.cum_tiles_explored + tiles_explored_diff

        # point tiles found
        point_tiles_found = jnp.sum(player_state.memory.point_tiles, axis=(0, 1), dtype=jnp.int16)
        point_tiles_found_diff = non_negative(point_tiles_found - state.record.point_tiles_found)   # note: cannot be negative
        cum_point_tiles_found = state.record.cum_point_tiles_found + point_tiles_found_diff

        # relics found
        relics_found = jnp.array(player_state.memory.relics_found, dtype=jnp.int16)
        relics_found_diff = non_negative(relics_found - state.record.relics_found)   # note: cannot be negative
        cum_relics_found = state.record.cum_relics_found + relics_found_diff

        # first relic found in first match
        first_relic_found_in_first_match = (((player_state.memory.relics_found == 1) | (player_state.memory.relics_found == 2)) & (player_state.memory.match_num == 0)).astype(jnp.int16)
        first_relic_found_in_first_match_diff = non_negative(first_relic_found_in_first_match - state.record.first_relic_found_in_first_match)
        cum_first_relic_found_in_first_match = state.record.cum_first_relic_found_in_first_match + first_relic_found_in_first_match_diff

        # all relics found (in the entire game, not per match)
        all_relics_found = jnp.array(player_state.memory.all_relics_found.astype(jnp.int16))
        all_relics_found_diff = non_negative(all_relics_found - state.record.all_relics_found)
        cum_all_relics_found = state.record.cum_all_relics_found + all_relics_found_diff
        
        # units energy (from memory) - due to all possible causes (respawn, energy field, collision, spawn...)
        units_energy = jnp.sum(player_state.memory.unit_mask * player_state.memory.unit_energy, dtype=jnp.int16)
        units_energy_diff = zero_if_match_restart(units_energy - state.record.units_energy)
        cum_units_energy = state.record.cum_units_energy + units_energy_diff

        # units stuck (not counting those on point tiles)
        unit_field = place_points_on_field(player_state.memory.false_map16, player_state.memory.unit_positions, player_state.memory.unit_mask)  # (N_MAX_UNITS, GRID_SHAPE)
        point_field = jnp.broadcast_to(player_state.memory.point_tiles, unit_field.shape)
        unit_not_on_point_tile_mask = jnp.any(jnp.logical_and(~ point_field, unit_field) * player_state.memory.n_units_map, axis=(-2, -1))  # (N_MAX_UNITS)
        units_with_low_energy_mask = player_state.memory.unit_mask * (player_state.memory.unit_energy < player_state.memory.unit_move_cost)  # (N_MAX_UNITS)
        units_stuck_by_energy = jnp.sum(jnp.logical_and(units_with_low_energy_mask, unit_not_on_point_tile_mask), dtype=jnp.int16)
        cum_units_stuck_by_energy = state.record.cum_units_stuck_by_energy + units_stuck_by_energy

        # getting all information useful for reward (should be increments), we can also use information not visible by player in the reward
        reward_contributions = {
            # win-related, zero-sum
            "win_game": game_status_diff,  # zero-sum
            "win_matches": matches_won_diff - matches_lost_diff,  # zero-sum
            "delta_points": points_player_diff - points_opponent_diff,  # zero-sum
            # basic game understanding
            "points": points_player_diff,  # positive
            "units_energy": units_energy_diff,  # positive or negative
            # exploration
            "tiles_explored": tiles_explored_diff,  # positive
            "point_tiles_found": point_tiles_found_diff,  # positive
            "relics_found": relics_found_diff,  # positive
            "all_relics_found": all_relics_found_diff,  # 0 or 1
            "all_point_tiles_found": all_point_tiles_found_diff,
            "first_relic_found_in_first_match": first_relic_found_in_first_match_diff,
            # other from obs or memory
            "units_stuck_by_energy": units_stuck_by_energy,
            "point_tiles_unoccupied": point_tiles_unoccupied,  # up to number of units
            "point_tiles_dist": point_tiles_dist_diff,
            # from lux engine
            "units_removed": units_removed_due_to_negative_energy,
            "units_moved": units_moved,
            "units_sapped": units_sapped,
            "sapping_energy_benefit": sapping_energy_benefit,  # positive if player took more energy to opponent than sapping cost
            "delta_collision": n_units_destroyed_by_collision_delta,  # positive if player lost less than opponent
            "delta_energy_sap": energy_variation_from_sap_delta,  # positive if player lost less than opponent (not very meaningful, use sapping_energy_benefit)
            "delta_energy_void": energy_variation_from_void_delta,  # positive if player lost less than opponent
            "energy_from_field": energy_variation_from_field,  # positive or negative
        }
        reward_contributions = jax.tree.map(lambda coef, value: jnp.array(coef * value, dtype=float), self.reward_coefficients, reward_contributions)

        # calculate the reward, coef for each reward contribution is defined in config
        sub_rewards = [jnp.array(val, dtype=float) for val in reward_contributions.values()]
        reward = jnp.sum(jnp.stack(sub_rewards))

        # auxiliary stuff to assess training progress, only the value at the end of the full game (5 matches) will be taken into account in the plot
        return_contributions = jax.tree.map(lambda o, n: jnp.array(o + n, dtype=float), state.record.return_contributions, reward_contributions)
        episode_metrics = {
                # from action chosen
                "action_noop_chosen": cum_noop_action,
                "action_move_chosen": cum_move_action,
                "action_sap_chosen": cum_sap_action,
                # from obs/memory
                "win_game": game_is_won,
                "win_matches": matches_won,
                "points": cum_points_player,
                "delta_points": cum_points_player - cum_points_opponent,
                "tiles_explored": cum_tiles_explored,
                "point_tiles_found": cum_point_tiles_found,
                "relics_found": cum_relics_found,
                "all_relics_found": cum_all_relics_found,
                "all_point_tiles_found": cum_all_point_tiles_found,
                "first_relic_found_in_first_match": cum_first_relic_found_in_first_match,
                "units_energy": cum_units_energy,
                "units_stuck_by_energy": cum_units_stuck_by_energy,
                # from lux engine
                "units_removed": cum_units_removed,
                "action_move_executed": cum_units_moved,
                "action_sap_executed": cum_units_sapped,
                "sapping_energy_benefit": cum_sapping_energy_benefit,
                "delta_collision": cum_n_units_destroyed_by_collision_delta,
                "delta_energy_sap": cum_energy_from_sap_delta,
                "delta_energy_void": cum_energy_from_void_delta,
                "energy_from_field": cum_energy_from_field,
                "units_spawned": cum_units_spawned,
                "point_tiles_unoccupied": cum_point_tiles_unoccupied,
                "point_tiles_dist": cum_point_tiles_dist,
            }
        episode_metrics = jax.tree.map(lambda a: jnp.array(a, dtype=jnp.int16), episode_metrics)  # convert all numbers to int16, except return contributions (float)
        episode_metrics["return_contributions"] = return_contributions  # add return contributions
        info = {
            "episode_metrics": episode_metrics
        }

        # ------------- UPDATE STATE AND GET READY FOR NEXT STEP -------------

        # defining our done
        if self.early_stopping:
            done = done_lux | (game_status != 0)  # early stopping if game is won/lost
        else:
            done = done_lux

        # create record of all useful quantities
        updated_record = Record(
            return_contributions=return_contributions,
            #
            cum_noop_action=cum_noop_action,
            cum_move_action=cum_move_action,
            cum_sap_action=cum_sap_action,
            #
            cum_units_removed=cum_units_removed,
            cum_units_moved=cum_units_moved,
            cum_units_sapped=cum_units_sapped,
            cum_sapping_energy_benefit=cum_sapping_energy_benefit,
            cum_n_units_destroyed_by_collision_delta=cum_n_units_destroyed_by_collision_delta,
            cum_energy_from_sap_delta=cum_energy_from_sap_delta,
            cum_energy_from_void_delta=cum_energy_from_void_delta,
            cum_energy_from_field=cum_energy_from_field,
            cum_units_spawned=cum_units_spawned,
            #
            matches_won=matches_won,
            matches_lost=matches_lost,
            game_status=game_status,
            #
            points_player=points_player,
            points_opponent=points_opponent,
            cum_points_player=cum_points_player,
            cum_points_opponent=cum_points_opponent,
            #
            tiles_explored=tiles_explored,
            cum_tiles_explored=cum_tiles_explored,
            point_tiles_found=point_tiles_found,
            cum_point_tiles_found=cum_point_tiles_found,
            relics_found=relics_found,
            cum_relics_found=cum_relics_found,
            all_relics_found=all_relics_found,
            cum_all_relics_found=cum_all_relics_found,
            all_point_tiles_found=all_point_tiles_found,
            cum_all_point_tiles_found=cum_all_point_tiles_found,
            first_relic_found_in_first_match=first_relic_found_in_first_match,
            cum_first_relic_found_in_first_match=cum_first_relic_found_in_first_match,
            #
            units_energy=units_energy,
            cum_units_energy=cum_units_energy,
            cum_units_stuck_by_energy=cum_units_stuck_by_energy,
            cum_point_tiles_unoccupied=cum_point_tiles_unoccupied,
            cum_point_tiles_dist=cum_point_tiles_dist,
            point_tiles_dist=point_tiles_dist,
        )

        # updating env state
        state = OnePlayerEnvState(
            params_lux=state.params_lux,
            units_permutation=state.units_permutation,
            units_unpermutation=state.units_unpermutation,
            state_lux=state_lux,
            player_last_obs_lux=obs_player_lux,
            opponent_last_obs_lux=obs_opponent_lux,
            player_state=player_state,
            opponent_state=opponent_state,
            is_terminal=done,
            key=key,
            record=updated_record,
        )

        # pack everything into timestep
        timestep = jax.lax.cond(
            done,
            lambda: termination(  # discount=0
                reward=reward,
                observation=nn_input,
                extras=info,
            ),
            lambda: transition(  # discount=1
                reward=reward,
                observation=nn_input,
                extras=info,
            ),
        )

        return state, timestep

    @cached_property
    def observation_spec(self):
        return self.player.input_spec

    @cached_property
    def action_spec(self):
        return SpecArray(
            shape=(2, N_MAX_UNITS),
            dtype=jnp.int16,
            name="nn_action",
        )


def create_lux_randomized_parameters(key):
    def draw_random_game_param(random_key, values):
        return jax.random.choice(random_key, jnp.array(values))

    # create randomized game parameters [adapted from the LuxAIS3Gym wrapper]
    key, *random_keys = jax.random.split(key, 12)
    lux_randomized_game_params = {
        "unit_move_cost": draw_random_game_param(random_keys[0], lux_env_params_ranges["unit_move_cost"]),
        "unit_sensor_range": draw_random_game_param(random_keys[1], lux_env_params_ranges["unit_sensor_range"]),
        "nebula_tile_vision_reduction": draw_random_game_param(random_keys[2], lux_env_params_ranges["nebula_tile_vision_reduction"]),
        "nebula_tile_energy_reduction": draw_random_game_param(random_keys[3], lux_env_params_ranges["nebula_tile_energy_reduction"]),
        "unit_sap_cost": draw_random_game_param(random_keys[4], lux_env_params_ranges["unit_sap_cost"]),
        "unit_sap_range": draw_random_game_param(random_keys[5], lux_env_params_ranges["unit_sap_range"]),
        "unit_sap_dropoff_factor": draw_random_game_param(random_keys[6], lux_env_params_ranges["unit_sap_dropoff_factor"]),
        "unit_energy_void_factor": draw_random_game_param(random_keys[7], lux_env_params_ranges["unit_energy_void_factor"]),
        "nebula_tile_drift_speed": draw_random_game_param(random_keys[8], lux_env_params_ranges["nebula_tile_drift_speed"]),
        "energy_node_drift_speed": draw_random_game_param(random_keys[9], lux_env_params_ranges["energy_node_drift_speed"]),
        "energy_node_drift_magnitude": draw_random_game_param(random_keys[10], lux_env_params_ranges["energy_node_drift_magnitude"]),
    }

    # create full set of lux parameters (non-randomized parameters are set as defaults)
    params_lux = LuxEnvParams(**lux_randomized_game_params)

    # only keep the following game parameters available to the agent [adapted from the LuxAIS3Gym wrapper]
    params_dict_kept_lux = {
        "max_units": params_lux.max_units,
        "match_count_per_episode": params_lux.match_count_per_episode,
        "max_steps_in_match": params_lux.max_steps_in_match,
        "map_height": params_lux.map_height,
        "map_width": params_lux.map_width,
        "num_teams": params_lux.num_teams,
        "unit_move_cost": params_lux.unit_move_cost,
        "unit_sap_cost": params_lux.unit_sap_cost,
        "unit_sap_range": params_lux.unit_sap_range,
        "unit_sensor_range": params_lux.unit_sensor_range,
    }

    return params_lux, params_dict_kept_lux


def fix_lux_type(lux_input: Union[LuxEnvState, LuxEnvObs]):
    """In Lux engine, step and reset output different types for unit energy (int32 vs int16, respectively)"""
    units = lux_input.units
    units = units.replace(
        energy=jnp.array(units.energy, dtype=jnp.int32)
    )
    lux_input = lux_input.replace(
        units=units
    )
    return lux_input


def import_opponent(opponent_dir: str):
    # opponent_dir: dir for Kaggle submission, containing the rl_agent directory
    opponent_file_path = os.path.join(opponent_dir, "wrapper.py")

    # random unique module name
    module_name = ''.join(random.choices(string.ascii_lowercase, k=16)) + '58'

    # add 'opponent_dir' from the opponent in the sys.path
    sys.path.append(opponent_dir)
    # print("Opponent added to sys.path", sys.path)

    # import agent from file
    spec = importlib.util.spec_from_file_location(module_name, opponent_file_path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)

    config_opponent = OmegaConf.load(os.path.join(opponent_dir, 'config_agent.yaml'))
    opponent = foo.get_agent(config_opponent, path=opponent_dir, training=False)

    return opponent


def get_some_obs(config_agent: DictConfig):
    player = Player(config_agent, path=None, training=True)
    return player.input_spec.generate_value()


def permutate_obs(lux_obs: LuxEnvObs, units_permutation: jnp.ndarray):
    position_perm = lux_obs.units.position[:, units_permutation]
    energy_perm = lux_obs.units.energy[:, units_permutation]
    lux_obs = lux_obs.replace(
        units=UnitState(position=position_perm, energy=energy_perm),
        units_mask=lux_obs.units_mask[:, units_permutation]
    )
    return lux_obs


def unpermutate_action(lux_action: jnp.ndarray, units_unpermutation: jnp.ndarray):
    return lux_action[units_unpermutation]
