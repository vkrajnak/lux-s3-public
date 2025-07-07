import time
import os
import sys
import random
import jax.numpy as jnp
from omegaconf import OmegaConf
from copy import deepcopy

from rl_agent.agent import Agent
from rl_agent.types import LuxEnvObs, UnitState, MapTile
from rl_agent.constants import MATCHES_TO_WIN, N_MAX_UNITS

PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_AGENT_FILE = os.path.join(PATH, 'config_agent.yaml')

# control what we print
DEBUG = False
INFO = False
MONITOR_TIME = True
PRINT_NETWORK_ARCH = False


class AgentKaggleWrapper:

    def __init__(self, player: str, env_cfg: dict) -> None:

        if MONITOR_TIME:
            start_time = time.time()

        config_agent = OmegaConf.load(CONFIG_AGENT_FILE)

        # if INFO:
        #     print("env_cfg=", env_cfg, file=sys.stderr)
        #     print("config_agent=", config_agent, file=sys.stderr)

        self.agent = Agent(config_agent, path=PATH, training=False, print_nn=PRINT_NETWORK_ARCH)
        # self.agent = Agent(config_agent, path=None, training=False, print_nn=PRINT_NETWORK_ARCH)  # TODO this is only for dummy agent

        player_id = 0 if player == "player_0" else 1
        self.player_id = player_id
        self.agent_state = self.agent.init(player_id=player_id, env_cfg=env_cfg)

        if MONITOR_TIME:
            end_time = time.time()
            print(f"init, elapsed={end_time - start_time}", file=sys.stderr)

    def act(self, step: int, obs: dict, remainingOverageTime: int):
        # note: step is the same as obs["steps"]
        if MONITOR_TIME:
            start_time = time.time()
            print(f"step={step}, remainingOverageTime={remainingOverageTime}", file=sys.stderr)

        steps = int(obs["steps"])
        match_steps = int(obs["match_steps"])
        matches_won = int(obs["team_wins"][self.player_id])
        matches_lost = int(obs["team_wins"][1 - self.player_id])
        unit_0_position = obs["units"]["position"][self.player_id][0, :]

        if DEBUG:
            print(f"steps={steps}, match_steps={match_steps}, matches won/lost={matches_won}/{matches_lost}, unit_0_position={unit_0_position}", file=sys.stderr)

        # nothing to do at step 0
        if step == 0:
            action = jnp.zeros((N_MAX_UNITS, 3), dtype=int)

        # if game is won/lost, stop playing
        elif matches_won >= MATCHES_TO_WIN or matches_lost >= MATCHES_TO_WIN:
            if INFO:
                print(f"step={step}, match is over, stopped playing", file=sys.stderr)
            action = jnp.zeros((N_MAX_UNITS, 3), dtype=int)

        else:

            # deal with special behaviour depending on remainingOverageTime and step
            # ...

            # convert obs dict to a LuxEnvObs
            lux_obs = reconstruct_lux_obs(obs)

            # call nn agent
            seed = jnp.array(random.randint(0, 1000), dtype=int)
            self.agent_state, action = self.agent.act(seed, self.agent_state, lux_obs)

            # process action (array) further if needed, e.g. overwriting some action
            # ...

        # if DEBUG:
        #     print(f"step={step}, action=\n{action}", file=sys.stderr)

        if MONITOR_TIME:
            end_time = time.time()
            print(f"step={step}, elapsed={end_time - start_time}", file=sys.stderr)

        return action


def reconstruct_lux_obs(obs: dict) -> LuxEnvObs:
    lux_units = UnitState(
        position=obs["units"]["position"],
        energy=obs["units"]["energy"]
    )
    lux_map_features = MapTile(
        energy=obs["map_features"]["energy"],
        tile_type=obs["map_features"]["tile_type"]
    )
    lux_obs = LuxEnvObs(
        units=lux_units,
        units_mask=obs["units_mask"],
        sensor_mask=obs["sensor_mask"],
        map_features=lux_map_features,
        relic_nodes=obs["relic_nodes"],
        relic_nodes_mask=obs["relic_nodes_mask"],
        team_points=obs["team_points"],
        team_wins=obs["team_wins"],
        steps=obs["steps"],
        match_steps=obs["match_steps"]
    )

    return lux_obs


def get_agent(config_agent, path, training):
    """Used for importing agent as an opponent during training"""
    return Agent(config_agent, path=path, training=training)
