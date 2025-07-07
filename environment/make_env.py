from omegaconf import DictConfig
from copy import deepcopy

from external.jumanji.wrappers import AutoResetWrapper
from external.stoix.wrappers.episode_metrics import RecordEpisodeMetrics

from environment.oneplayerenv import OnePlayerEnv


def make_env(config_env: DictConfig, opponents_params: dict) -> tuple[OnePlayerEnv, OnePlayerEnv]:

    env = OnePlayerEnv(config_env, opponents_params)
    eval_env = deepcopy(env)

    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    assert env.opponent_name == eval_env.opponent_name

    return env, eval_env


