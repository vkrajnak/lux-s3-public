import jax
import jax.numpy as jnp

from ..specs import Spec, Array
from ..types import AgentState
from ..constants import GRID_SHAPE, N_MAX_UNITS, N_BASE_ACTIONS, N_SAP_ACTIONS
from .types import DTYPE, NNInput, ActionMasksDiscrete, ActionMasksMonoField, ActionMasksDuoField, ActionMasksDuomoField, ActionMasksDiscreteAndField

from .get_fields import GET_N_CHANNELS, GET_FIELDS_FN
from .get_scalars import GET_SCALAR_FN

# --- Continuous Fields


def _get_continuous_fields(agent_state: AgentState, fields_as_fields_names: tuple[str], scalars_as_fields_names: tuple[str]) -> jnp.ndarray:
    fields = [GET_FIELDS_FN[name](agent_state) for name in fields_as_fields_names]
    fields = jnp.concatenate(fields, axis=-1).astype(DTYPE)  # (24, 24, d)
    scalars = [jnp.broadcast_to(GET_SCALAR_FN[name](agent_state), shape=GRID_SHAPE + (1,)) for name in scalars_as_fields_names]
    scalars = jnp.concatenate(scalars, axis=-1).astype(DTYPE)  # (24, 24, n)
    return jnp.concatenate([fields, scalars], axis=-1).astype(DTYPE)  # (24, 24, d+n)


# --- Categorical Fields

def _get_categorical_tile_type_fields(agent_state: AgentState):
    """Used with embeddings"""
    current = agent_state.memory.tile_type_field[:, :, None]  # (24, 24, 1)
    next = agent_state.memory.next_tile_type_field[:, :, None]  # (24, 24, 1)
    return jnp.concatenate([current, next], axis=-1).astype(jnp.int16)  # (24, 24, 2)


# --- Action Masks

def _get_action_masks_discrete(agent_state: AgentState) -> ActionMasksDiscrete:
    action_masks = ActionMasksDiscrete(
        base_action_masks=agent_state.memory.base_action_mask_discrete * (agent_state.memory.match_steps != 0),
        sap_action_masks=agent_state.memory.sap_action_mask_discrete * (agent_state.memory.match_steps != 0),
    )
    return action_masks  # bool


def _get_action_masks_monofield(agent_state: AgentState) -> ActionMasksMonoField:
    action_masks = ActionMasksMonoField(
        monoaction_masks=agent_state.memory.action_mask_monofield * (agent_state.memory.match_steps != 0),
        base_action_masks=agent_state.memory.base_action_monofield * (agent_state.memory.match_steps != 0),
        sap_action_masks=agent_state.memory.sap_action_monofield * (agent_state.memory.match_steps != 0),
        noop_action_masks=agent_state.memory.noop_action_monofield,  # used for dealing with the no-existing unit case (at match reset)
    )
    return action_masks  # bool


def _get_action_masks_duofield(agent_state: AgentState) -> ActionMasksDuoField:
    action_masks = ActionMasksDuoField(
        base_action_masks=agent_state.memory.base_action_duofield * (agent_state.memory.match_steps != 0),
        sap_action_masks=agent_state.memory.sap_action_mask_field * (agent_state.memory.match_steps != 0),
        monofield_base_converter=agent_state.memory.monofield_base_converter
    )
    return action_masks  # bool

def _get_action_masks_duomofield(agent_state: AgentState) -> ActionMasksDuomoField:
    action_masks = ActionMasksDuomoField(
        monoaction_masks=agent_state.memory.action_mask_monofield * (agent_state.memory.match_steps != 0),
        base_action_masks=agent_state.memory.base_action_monofield * (agent_state.memory.match_steps != 0),
        sap_action_masks=agent_state.memory.sap_action_monofield * (agent_state.memory.match_steps != 0),
        noop_action_masks=agent_state.memory.noop_action_monofield,  # used for dealing with the no-existing unit case (at match reset)
        monofield_base_converter=agent_state.memory.monofield_base_converter
    )
    return action_masks  # bool


def _get_action_masks_discrete_and_field(agent_state: AgentState) -> ActionMasksDiscreteAndField:
    action_masks = ActionMasksDiscreteAndField(
        base_action_discrete_masks=agent_state.memory.base_action_mask_discrete * (agent_state.memory.match_steps != 0),
        sap_action_field_masks=agent_state.memory.sap_action_mask_field * (agent_state.memory.match_steps != 0),
    )
    return action_masks  # bool


_GET_ACTION_MASK_FN = {
    "discrete_full": _get_action_masks_discrete,
    "discrete_sparse": _get_action_masks_discrete,
    "monofield": _get_action_masks_monofield,
    "duofield": _get_action_masks_duofield,
    "duomofield": _get_action_masks_duomofield,
    "discrete_and_field": _get_action_masks_discrete_and_field,
}


# --- Main functions

def get_build_input_from_agent_state(fields_as_fields_names: tuple[str], scalars_as_fields_names: tuple[str], action_head: str):
    def build_input(agent_state: AgentState) -> NNInput:
        """This is where we generate the NN input"""
        actor_input = NNInput(
            continuous_fields=_get_continuous_fields(agent_state, fields_as_fields_names, scalars_as_fields_names),
            tile_type_fields=_get_categorical_tile_type_fields(agent_state),
            action_masks=_GET_ACTION_MASK_FN[action_head](agent_state),
            )
        return actor_input

    return build_input


def get_input_spec(fields_as_fields_names: tuple[str], scalars_as_fields_names: tuple[str], action_head: str):

    n_channels_fields = [GET_N_CHANNELS[name] for name in fields_as_fields_names]
    n_channels_fields = sum(n_channels_fields)
    n_channels_scalars = len(scalars_as_fields_names)
    n_channels_continuous_fields = n_channels_fields + n_channels_scalars

    continuous_fields = Array(
        shape=GRID_SHAPE + (n_channels_continuous_fields, ),
        dtype=DTYPE,
        name="continuous_fields",
    )

    tile_type_fields = Array(
        shape=GRID_SHAPE + (2, ),  # always 2 channels: current and next
        dtype=jnp.int16,
        name="tile_type_fields",
    )

    # ---- discrete

    _base_action_masks = Array(
        shape=(N_MAX_UNITS, N_BASE_ACTIONS),
        dtype=jnp.bool,
        name="base_action_masks",
    )

    _sap_action_masks = Array(
        shape=(N_MAX_UNITS, N_SAP_ACTIONS),
        dtype=jnp.bool,
        name="sap_action_masks",
    )

    discrete_action_mask = Spec(
        ActionMasksDiscrete,
        "ActionMasksDiscreteSpec",
        base_action_masks=_base_action_masks,
        sap_action_masks=_sap_action_masks,
    )

    # ----- monofield

    _monoaction_masks = Array(
        shape=(N_MAX_UNITS, ) + GRID_SHAPE,
        dtype=jnp.bool,
        name="monoaction_masks"
    )

    monofield_action_masks = Spec(
        ActionMasksMonoField,
        "ActionMasksMonoFieldSpec",
        monoaction_masks=_monoaction_masks,
        base_action_masks=_monoaction_masks,
        sap_action_masks=_monoaction_masks,
        noop_action_masks=_monoaction_masks,
    )

    duofield_action_masks = Spec(
        ActionMasksDuoField,
        "ActionMasksDuoFieldSpec",
        base_action_masks=_monoaction_masks,
        sap_action_masks=_monoaction_masks,
        monofield_base_converter=_monoaction_masks,
    )

    duomofield_action_masks = Spec(
        ActionMasksDuomoField,
        "ActionMasksDuomoFieldSpec",
        monoaction_masks=_monoaction_masks,
        base_action_masks=_monoaction_masks,
        sap_action_masks=_monoaction_masks,
        noop_action_masks=_monoaction_masks,
        monofield_base_converter=_monoaction_masks,
    )

    # ----- discrete and field

    _field_masks = Array(
        shape=(N_MAX_UNITS, ) + GRID_SHAPE,
        dtype=jnp.bool,
        name="field_masks"
    )

    discrete_and_field_action_masks = Spec(
        ActionMasksDiscreteAndField,
        "ActionMasksDiscreteAndFieldSpec",
        base_action_discrete_masks=_base_action_masks,
        sap_action_field_masks=_field_masks,
    )

    # -----

    get_action_mask_spec = {
        "discrete_full": discrete_action_mask,
        "discrete_sparse": discrete_action_mask,
        "monofield": monofield_action_masks,
        "duofield": duofield_action_masks,
        "duomofield": duomofield_action_masks,
        "discrete_and_field": discrete_and_field_action_masks,
    }

    return Spec(
        NNInput,
        "NNInputSpec",
        continuous_fields=continuous_fields,
        tile_type_fields=tile_type_fields,
        action_masks=get_action_mask_spec[action_head]
    )

