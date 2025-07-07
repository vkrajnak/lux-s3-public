import jax.numpy as jnp
from typing import NamedTuple, Union

DTYPE = jnp.float32


class ActionMasksDiscrete(NamedTuple):
    base_action_masks: jnp.array  # mask for base actions (N_MAX_UNITS, N_BASE_ACTIONS)
    sap_action_masks: jnp.array  # mask for sap actions (N_MAX_UNITS, N_SAP_ACTIONS)


class ActionMasksMonoField(NamedTuple):
    monoaction_masks: jnp.array  # mask for action where move/sap are exclusive of each other (N_MAX_UNITS, GRID_SHAPE)
    base_action_masks: jnp.array  # mask for base action (N_MAX_UNITS, GRID_SHAPE)
    sap_action_masks: jnp.array  # mask for sap action (N_MAX_UNITS, GRID_SHAPE)
    noop_action_masks: jnp.array  # mask for noop action (N_MAX_UNITS, GRID_SHAPE) (included also in base_action)


class ActionMasksDuoField(NamedTuple):
    base_action_masks: jnp.array  # mask for base action (N_MAX_UNITS, GRID_SHAPE)
    sap_action_masks: jnp.array  # mask for sap action (N_MAX_UNITS, GRID_SHAPE)
    monofield_base_converter: jnp.array  # mask for sap action (N_MAX_UNITS, GRID_SHAPE)

class ActionMasksDuomoField(NamedTuple):
    monoaction_masks: jnp.array  # mask for action where move/sap are exclusive of each other (N_MAX_UNITS, GRID_SHAPE)
    base_action_masks: jnp.array  # mask for base action (N_MAX_UNITS, GRID_SHAPE)
    sap_action_masks: jnp.array  # mask for sap action (N_MAX_UNITS, GRID_SHAPE)
    noop_action_masks: jnp.array  # mask for noop action (N_MAX_UNITS, GRID_SHAPE) (included also in base_action)
    monofield_base_converter: jnp.array  # mask for sap action (N_MAX_UNITS, GRID_SHAPE)


class ActionMasksDiscreteAndField(NamedTuple):
    base_action_discrete_masks: jnp.array  # mask for base action (N_MAX_UNITS, N_BASE_ACTIONS)
    sap_action_field_masks: jnp.array  # mask for sap action (N_MAX_UNITS, GRID_SHAPE)


class NNInput(NamedTuple):
    """The object fed to the neural networks (actor or critic)."""
    continuous_fields: jnp.array  # array (24, 24, d) of real scaled values (typically in [0, 1] or [-1, 1] where d is the dimension
    tile_type_fields: jnp.array  # array (24, 24, n) of int in {-1, 0, 1, 2} where n correspond to different time steps
    action_masks: Union[ActionMasksDiscrete, ActionMasksMonoField, ActionMasksDuoField, ActionMasksDuomoField, ActionMasksDiscreteAndField]


