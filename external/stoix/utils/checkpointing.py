import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, NamedTuple, Optional, Tuple, Type, Union

import jax
import numpy as np
import absl.logging as absl_logging
import orbax.checkpoint
from chex import Numeric
from flax.core.frozen_dict import FrozenDict
from jax.tree_util import tree_map
from omegaconf import DictConfig, OmegaConf

from external.stoix.base_types import (
    ActorCriticHiddenStates,
    ActorCriticParams,
    HiddenStates,
    Parameters,
    StoixState,
)

# Keep track of the version of the checkpointer
# Any breaking API changes should be reflected in the major version (e.g. v0.1 -> v1.0)
# whereas minor versions (e.g. v0.1 -> v0.2) indicate backwards compatibility
CHECKPOINTER_VERSION = 1.0


def instantiate_namedtuple_from_dict(namedtuple_cls: Type[NamedTuple], data: Dict[str, Any]) -> Any:
    """
    Recursively constructs a named tuple from a dictionary.

    Args:
        namedtuple_cls (Type[NamedTuple]): The class of the named tuple to be instantiated.
        data (Dict[str, Any]): The dictionary containing the data for the named tuple and
            its nested structures.

    Returns:
        NamedTuple: An instance of the specified named tuple class, filled with data from
            the dictionary.

    Raises:
        KeyError: If a required key is missing in the dictionary to instantiate the named
            tuple properly.
    """
    # Base case: the data is already an instance of the required named tuple
    if isinstance(data, namedtuple_cls):
        return data

    # Iterate over the fields in the named tuple
    kwargs = {}
    for field, field_type in namedtuple_cls.__annotations__.items():
        if field in data:
            # Check if the field type is itself a NamedTuple
            if hasattr(field_type, "_fields"):
                # Recursively convert nested dicts to their corresponding named tuple
                kwargs[field] = instantiate_namedtuple_from_dict(field_type, data[field])
            else:
                # Directly assign if it's a basic type or a FrozenDict
                kwargs[field] = data[field]
        else:
            raise KeyError(f"Missing '{field}' in data to instantiate {namedtuple_cls.__name__}")

    # Create the named tuple instance with the populated keyword arguments
    return namedtuple_cls(**kwargs)  # type: ignore


class Checkpointer:
    """Model checkpointer for saving and restoring the `learner_state`."""

    def __init__(
        self,  # AL: removed model_name and checkpoint_uid arguments, added best_fn argument
        metadata: Optional[Dict] = None,
        directory: str = "./checkpoints",
        save_interval_steps: int = 1,
        max_to_keep: Optional[int] = 1,
        keep_period: Optional[int] = None,
        keep_time_interval: Optional[float] = None,  # AL: added
        best_fn: Optional[str] = None,
    ):
        """Initialise the checkpointer tool

        Args:
            metadata (Optional[Dict], optional):
                For storing model metadata. Defaults to None.
            directory (str, optional):
                Directory of checkpoints. Defaults to "./checkpoints".
            save_interval_steps (int, optional):
                The interval at which checkpoints should be saved. Defaults to 1.
            max_to_keep (Optional[int], optional):
                Maximum number of checkpoints to keep. Defaults to 1.
            keep_period (Optional[int], optional):
                If set, will not delete any checkpoint where
                checkpoint_step % keep_period == 0. Defaults to None.
            keep_time_interval (Optional[float], optional):
                If set, will not delete checkpoints if none has been kept within keep_time_interval (in hours)
            best_fn (Optional[str], optional):
                determine the criteria on which a model is deemed "better" than another
        """

        # Don't log checkpointing messages (at INFO level)
        absl_logging.set_verbosity(absl_logging.WARNING)

        # When we load an existing checkpoint, the sharding info is read from the checkpoint file,
        # rather than from 'RestoreArgs'. This is desired behaviour, so we suppress the warning.
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Couldn't find sharding info under RestoreArgs",
        )

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # checkpoint_str = (
        #     checkpoint_uid if checkpoint_uid else datetime.now().strftime("%Y%m%d%H%M%S")
        # )

        if keep_time_interval is not None:
            keep_time_interval = timedelta(seconds=3600 * keep_time_interval)  # AL: convert to proper format
        options = orbax.checkpoint.CheckpointManagerOptions(
            create=True,
            best_fn=lambda x: x[best_fn],
            best_mode="max",
            save_interval_steps=save_interval_steps,
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            keep_time_interval=keep_time_interval,
        )

        def get_json_ready(obj: Any) -> Any:
            if not isinstance(obj, (bool, str, int, float, type(None))):
                return str(obj)
            else:
                return obj

        # Convert metadata to JSON-ready format
        if metadata is not None and isinstance(metadata, DictConfig):
            metadata = OmegaConf.to_container(metadata, resolve=True)
        metadata_json_ready = tree_map(get_json_ready, metadata)

        self._manager = orbax.checkpoint.CheckpointManager(
            directory=directory,
            checkpointers=orbax_checkpointer,
            options=options,
            metadata={
                "checkpointer_version": CHECKPOINTER_VERSION,
                **(metadata_json_ready if metadata_json_ready is not None else {}),
            },
        )

    def save(
        self,
        step: int,
        t: int,
        unreplicated_learner_state: StoixState,
        metrics: dict,
    ) -> bool:
        """Save the learner state.

        Args:
            step (int):
                step at which the state is being saved (arbitrary definition).
            t (int):
                step as defined by logging (useful when reloading)
            unreplicated_learner_state (StoixState)
                a Stoix LearnerState (must be unreplicated)
            metric (dict):
                metrics including the eval_metric to determine whether this is the 'best' model to save.

        Returns:
            bool: whether the saving was successful.
        """
        model_save_success: bool = self._manager.save(
            step=step,
            items={
                "t": t,
                "learner_state": unreplicated_learner_state,
            },
            metrics=jax.tree.map(lambda x: float(np.mean(x)), metrics),  # AL: modified metrics (must be float, not jax arrays),  # AL: modified to include various metrics, instead of episode_return
        )
        return model_save_success

    def restore_params(
        self,
        step: Optional[Union[int, str]] = "latest",  # timestep, "latest" or "best", AL: modified
        restore_hstates: bool = False,
        TParams: Type[Parameters] = ActorCriticParams,  # noqa: N803
        THiddenState: Type[HiddenStates] = ActorCriticHiddenStates,  # noqa: N803
    ) -> Tuple[int, FrozenDict, Union[HiddenStates, None]]:
        """Restore the params and the hidden state (in case of RNNs)

        Args:
            step (Optional[int], optional):
                Specific timestep for restoration (of course, only if that timestep exists).
                Or "latest", or "best".
            restore_hstates (bool, optional): Whether to restore the hidden states.
                Defaults to False.
            TParams (Type[FrozenDict], optional): Type of the params.
                Defaults to ActorCriticParams.
            THiddenState (Type[HiddenStates], optional): Type of the hidden states.
                Defaults to ActorCriticHiddenStates.

        Returns:
            Tuple[ActorCriticParams,Union[HiddenState, None]]: the restored params and
            hidden states.
        """
        # We want to ensure `major` versions match, but allow `minor` versions to differ
        # i.e. v0.1 and 0.2 are compatible, but v1.0 and v2.0 are not
        # Any breaking API changes should be reflected in the major version
        # assert (self._manager.metadata()["checkpointer_version"] // 1) == (
        #     CHECKPOINTER_VERSION // 1
        # ), "Loaded checkpoint was created with a different major version of the checkpointer." AL: commented

        # Restore the checkpoint, either the n-th (if specified) or just the latest
        if isinstance(step, int):
            restored_checkpoint = self._manager.restore(step)
        elif step == "latest":
            restored_checkpoint = self._manager.restore(self._manager.latest_step())
        elif step == "best":
            restored_checkpoint = self._manager.restore(self._manager.best_step())
        else:
            raise Exception("This type of 'timestep' for restoring checkpoint is not implemented.")

        # Dictionary of the restored learner state
        restored_learner_state_raw = restored_checkpoint["learner_state"]

        restored_params = instantiate_namedtuple_from_dict(
            TParams, restored_learner_state_raw["params"]
        )

        # Restore hidden states if required
        restored_hstates = None
        if restore_hstates:
            restored_hstates = THiddenState(**restored_learner_state_raw["hstates"])

        return restored_checkpoint["t"], restored_params, restored_hstates

    def get_cfg(self) -> DictConfig:
        """Return the metadata of the checkpoint.

        Returns:
            DictConfig: metadata of the checkpoint.
        """
        return DictConfig(self._manager.metadata())
