import abc
import logging
import os
import zipfile
from datetime import datetime
from enum import Enum
from typing import Dict, List, Union
from collections.abc import MutableMapping

import jax
import numpy as np
from colorama import Fore, Style
from jax.typing import ArrayLike
from omegaconf import DictConfig



class LogEvent(Enum):
    ACT = "rollout"  # AL: changed name for clarity
    TRAIN = "trainer"
    EVAL = "evaluator"
    ABSOLUTE = "absolute"
    MISC = "misc"
    BEST = "best"
    RETURN_ACT = "return_rollout"
    RETURN_EVAL = "return_eval"


def flatten_dict(dictionary, parent_key='', sep='/'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


class StoixLogger:
    """The main logger for Stoix systems.

    Thin wrapper around the MultiLogger that is able to describe arrays of metrics
    and calculate environment specific metrics if required (e.g solve_rate).
    """

    def __init__(self, config: DictConfig) -> None:
        self.logger: BaseLogger = _make_multi_logger(config)
        self.cfg = config

    def log(self, metrics: Dict, t: int, t_eval: int, event: LogEvent) -> None:
        """Log a dictionary metrics at a given timestep.

        Args:
            metrics (Dict): dictionary of metrics to log.
            t (int): the current timestep.
            t_eval (int): the number of previous evaluations.
            event (LogEvent): the event that the metrics are associated with.
        """
        # Ideally we want to avoid special metrics like this as much as possible.
        # Might be better to calculate this outside as we want to keep the number of these
        # if statements to a minimum.
        if "solve_episode" in metrics:
            metrics = self.calc_solve_rate(metrics, event)

        # We only want to log mean losses, max/min/std don't matter. (AL: forced it as default everywhere)
        metrics = jax.tree_util.tree_map(np.mean, metrics)

        # if event == LogEvent.TRAIN:
        #     # We only want to log mean losses, max/min/std don't matter.
        #     metrics = jax.tree_util.tree_map(np.mean, metrics)
        # else:
        #     # {metric1_name: [metrics], metric2_name: ...} ->
        #     # {metric1_name: {mean: metric, max: metric, ...}, metric2_name: ...}
        #     metrics = jax.tree_util.tree_map(describe, metrics)

        metrics = jax.tree.map(
            lambda x: x.item() if isinstance(x, (jax.Array, np.ndarray)) else x, metrics
        )

        self.logger.log_dict(metrics, t, t_eval, event)

    def calc_solve_rate(self, episode_metrics: Dict, event: LogEvent) -> Dict:
        """Log the solve rate of the environment's episodes."""
        # Get the number of episodes used to evaluate.
        if event == LogEvent.ABSOLUTE:
            # To measure the absolute metric, we evaluate the best policy
            # found across training over 10 times the evaluation episodes.
            # For more details on the absolute metric please see:
            # https://arxiv.org/abs/2209.10485.
            n_episodes = self.cfg.arch.num_eval_episodes * 10
        else:
            n_episodes = self.cfg.arch.num_eval_episodes

        # Calculate the solve rate.
        n_solve_episodes: int = np.sum(episode_metrics["solve_episode"])
        solve_rate = (n_solve_episodes / n_episodes) * 100

        episode_metrics["solve_rate"] = solve_rate
        episode_metrics.pop("solve_episode")

        return episode_metrics

    def stop(self) -> None:
        """Stop the logger."""
        self.logger.stop()


class BaseLogger(abc.ABC):
    @abc.abstractmethod
    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        pass

    @abc.abstractmethod
    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        """Log a single metric."""
        raise NotImplementedError

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent) -> None:
        """Log a dictionary of metrics."""
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep="/")

        for key, value in data.items():
            self.log_stat(
                key,
                value,
                step,
                eval_step,
                event,
            )

    def stop(self) -> None:
        """Stop the logger."""
        return None


class MultiLogger(BaseLogger):
    """Logger that can log to multiple loggers at oncce."""

    def __init__(self, loggers: List[BaseLogger]) -> None:
        self.loggers = loggers

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        for logger in self.loggers:
            logger.log_stat(key, value, step, eval_step, event)

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent) -> None:
        for logger in self.loggers:
            logger.log_dict(data, step, eval_step, event)

    def stop(self) -> None:
        for logger in self.loggers:
            logger.stop()


class NeptuneLogger(BaseLogger):
    """Logger for neptune.ai."""

    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        import neptune
        from neptune.utils import stringify_unsupported

        tags = list(cfg.logger.kwargs.tags)
        project = cfg.logger.kwargs.project
        name = cfg.logger.kwargs.name

        self.logger = neptune.init_run(project=project, tags=tags, name=name)  # AL: added name

        self.logger["config"] = stringify_unsupported(cfg)
        self.detailed_logging = cfg.logger.kwargs.detailed_logging

        # Store json path for uploading json data to Neptune.
        json_exp_path = get_logger_path(cfg, "json")
        self.json_file_path = os.path.join(
            cfg.logger.base_exp_path, f"{json_exp_path}/{unique_token}/metrics.json"
        )
        self.unique_token = unique_token
        self.upload_json_data = cfg.logger.kwargs.upload_json_data

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        # Main metric if it's the mean of a list of metrics (ends with '/mean')
        # or it's a single metric doesn't contain a '/'.
        is_main_metric = "/" not in key or key.endswith("/mean")
        # If we're not detailed logging (logging everything) then make sure it's a main metric.
        if not self.detailed_logging and not is_main_metric:
            return

        self.logger[f"{event.value}/{key}"].log(value, step=step)

    def stop(self) -> None:
        if self.upload_json_data:
            self._zip_and_upload_json()
        self.logger.stop()

    def _zip_and_upload_json(self) -> None:
        # Create the zip file path by replacing '.json' with '.zip'
        zip_file_path = self.json_file_path.rsplit(".json", 1)[0] + ".zip"

        # Create a zip file containing the specified JSON file
        with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.json_file_path)

        self.logger[f"metrics/metrics_{self.unique_token}"].upload(zip_file_path)


class WandBLogger(BaseLogger):
    """Logger for wandb.ai."""

    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        import wandb

        tags = list(cfg.logger.kwargs.tags)
        project = cfg.logger.kwargs.project

        wandb.init(project=project, tags=tags, config=stringify_unsupported(cfg))

        self.detailed_logging = cfg.logger.kwargs.detailed_logging

        # Store json path for uploading json data to Neptune.
        json_exp_path = get_logger_path(cfg, "json")
        self.json_file_path = os.path.join(
            cfg.logger.base_exp_path, f"{json_exp_path}/{unique_token}/metrics.json"
        )
        self.unique_token = unique_token
        self.upload_json_data = cfg.logger.kwargs.upload_json_data

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        import wandb

        # Main metric if it's the mean of a list of metrics (ends with '/mean')
        # or it's a single metric doesn't contain a '/'.
        is_main_metric = "/" not in key or key.endswith("/mean")
        # If we're not detailed logging (logging everything) then make sure it's a main metric.
        if not self.detailed_logging and not is_main_metric:
            return

        data_to_log = {f"{event.value}/{key}": value}
        wandb.log(data_to_log, step=step)

    def stop(self) -> None:
        if self.upload_json_data:
            self._zip_and_upload_json()
        wandb.finish()  # type: ignore

    def _zip_and_upload_json(self) -> None:
        # Create the zip file path by replacing '.json' with '.zip'
        zip_file_path = self.json_file_path.rsplit(".json", 1)[0] + ".zip"

        # Create a zip file containing the specified JSON file
        with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.json_file_path)

        wandb.save(zip_file_path)


class TensorboardLogger(BaseLogger):
    """Logger for tensorboard"""

    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        import tensorboard_logger

        tb_exp_path = get_logger_path(cfg, "tensorboard")
        tb_logs_path = os.path.join(cfg.logger.base_exp_path, f"{tb_exp_path}/{unique_token}")

        self.logger = tensorboard_logger.Logger(tb_logs_path)
        self.log = self.logger.log_value

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        t = step if event != LogEvent.EVAL else eval_step
        self.log(f"{event.value}/{key}", value, t)


class JsonLogger(BaseLogger):
    """Json logger for marl-eval."""

    # These are the only metrics that marl-eval needs to plot.
    _METRICS_TO_LOG = ["episode_return/mean", "solve_rate", "steps_per_second"]

    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        from marl_eval.json_tools import JsonLogger as MarlEvalJsonLogger

        json_exp_path = get_logger_path(cfg, "json")
        json_logs_path = os.path.join(cfg.logger.base_exp_path, f"{json_exp_path}/{unique_token}")

        # if a custom path is specified, use that instead
        if cfg.logger.kwargs.json_path is not None:
            json_logs_path = os.path.join(
                cfg.logger.base_exp_path, "json", cfg.logger.kwargs.json_path
            )

        self.logger = MarlEvalJsonLogger(
            path=json_logs_path,
            algorithm_name=cfg.system.system_name,
            task_name=cfg.env.scenario.task_name,
            environment_name=cfg.env.env_name,
            seed=cfg.arch.seed,
        )

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        # Only write key if it's in the list of metrics to log.

        if key not in self._METRICS_TO_LOG:
            return

        # The key is in the format <metric_name>/<aggregation_fn> so we need to change it to:
        # <agg fn>_<metric_name>
        if "/" in key:
            key = "_".join(reversed(key.split("/")))

        # JsonWriter can't serialize jax arrays
        value = value.item() if isinstance(value, jax.Array) else value

        # We only want to log evaluation metrics to the json logger
        if event == LogEvent.ABSOLUTE or event == LogEvent.EVAL:
            self.logger.write(step, key, value, eval_step, event == LogEvent.ABSOLUTE)


class ConsoleLogger(BaseLogger):
    """Logger for writing to stdout."""

    _EVENT_COLOURS = {
        LogEvent.TRAIN: Fore.YELLOW,
        LogEvent.EVAL: Fore.GREEN,
        LogEvent.ABSOLUTE: Fore.BLUE,
        LogEvent.ACT: Fore.CYAN,
        LogEvent.MISC: Fore.MAGENTA,
        LogEvent.BEST: Fore.BLACK,
        LogEvent.RETURN_ACT: Fore.CYAN,
        LogEvent.RETURN_EVAL: Fore.GREEN,
    }

    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        self.logger = logging.getLogger()

        self.logger.handlers = []

        ch = logging.StreamHandler()
        formatter = logging.Formatter(f"%(message)s", "%H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Set to info to suppress debug outputs.
        self.logger.setLevel("INFO")

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        colour = self._EVENT_COLOURS[event]

        # Replace underscores with spaces and capitalise keys.
        # key = key.replace("_", " ").capitalize()  # AL: removed
        self.logger.info(
            f"{colour}{event.value.upper()} - {key}: {value:.3f}{Style.RESET_ALL}"  # AL: removed {Style.BRIGHT}
        )

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent) -> None:
        if event != LogEvent.BEST:
            # in case the dict is nested, flatten it.
            data = flatten_dict(data, sep=" ")

            colour = self._EVENT_COLOURS[event]
            # Replace underscores with spaces and capitalise keys.
            # keys = [k.replace("_", " ").capitalize() for k in data.keys()] # AL: removed
            keys = data.keys()

            if event == LogEvent.MISC:
                # Convert very large int to exponential notation
                values = [f"{float(v):.1e}" for v in data.values()]
            else:
                # Round values to 3 decimal places if they are floats.
                values = [v if isinstance(v, int) else f"{float(v):.3f}" for v in data.values()]
            log_str = " | ".join([f"{k}: {v}" for k, v in zip(keys, values)])

            self.logger.info(
                f"{colour}{event.value.upper()} - {log_str}{Style.RESET_ALL}"  # AL: removed {Style.BRIGHT}
            )


def _make_multi_logger(cfg: DictConfig) -> BaseLogger:
    """Creates a MultiLogger given a config"""

    loggers: List[BaseLogger] = []
    unique_token = datetime.now().strftime("%Y%m%d%H%M%S")

    if (
        (cfg.logger.use_neptune or cfg.logger.use_wandb)
        and cfg.logger.use_json
        and cfg.logger.kwargs.upload_json_data
        and cfg.logger.kwargs.json_path
    ):
        raise ValueError(
            "Cannot upload json data to Neptune when `json_path` is set in the base logger config. "
            "This is because each subsequent run will create a larger json file which will use "
            "unnecessary storage. Either set `upload_json_data: false` if you don't want to "
            "upload your json data but store a large file locally or set `json_path: ~` in "
            "the base logger config."
        )

    if cfg.logger.use_neptune:
        loggers.append(NeptuneLogger(cfg, unique_token))
    if cfg.logger.use_wandb:
        loggers.append(WandBLogger(cfg, unique_token))
    if cfg.logger.use_tb:
        loggers.append(TensorboardLogger(cfg, unique_token))
    if cfg.logger.use_json:
        loggers.append(JsonLogger(cfg, unique_token))
    if cfg.logger.use_console:
        loggers.append(ConsoleLogger(cfg, unique_token))

    return MultiLogger(loggers)


def get_logger_path(config: DictConfig, logger_type: str) -> str:
    """Helper function to create the experiment path."""
    return f"{logger_type}/{config.system.system_name}"


def describe(x: ArrayLike) -> Union[Dict[str, ArrayLike], ArrayLike]:
    """Generate summary statistics for an array of metrics (mean, std, min, max)."""

    if not isinstance(x, (jax.Array, np.ndarray)):
        return x
    elif x.size <= 1:
        return np.squeeze(x)

    # np instead of jnp because we don't jit here
    return {"mean": np.mean(x), "std": np.std(x), "min": np.min(x), "max": np.max(x)}
