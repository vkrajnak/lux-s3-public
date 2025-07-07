import jax
import jax.numpy as jnp
import chex
from flax import linen as nn
from typing import NamedTuple

from .constants import N_BASE_ACTIONS, N_SAP_ACTIONS, N_MAX_UNITS, ENV_PARAMS_RANGES

from .network_utils.distributions import AllUnitsActionDistribution
from .network_utils.heads import AllUnitsFullActionHead, CriticHead
from .network_utils.torso import CompositeTorso


SAP_SQUARE_SIZE = max(ENV_PARAMS_RANGES["unit_sap_range"])


class NNInput(NamedTuple):
    """The object feed to the neural networks (actor or critic). Fields are stacked along first dimension, not last."""
    scalars: jnp.array  # vector of scalar values
    fields: jnp.array  # array with fields as channels # STACKED ALONG FIRST DIMENSION
    base_action_masks: jnp.array = jnp.ones((N_MAX_UNITS, N_BASE_ACTIONS), dtype=bool)  # mask for base actions (N_MAX_UNITS, N_BASE_ACTIONS)
    sap_action_masks: jnp.array = jnp.ones((N_MAX_UNITS, N_SAP_ACTIONS), dtype=bool)  # mask for sap actions (N_MAX_UNITS, N_SAP_ACTIONS)


class Actor(nn.Module):
    """Feedforward actor network providing distribution of action for all units"""

    torso: nn.Module
    action_head: nn.Module = AllUnitsFullActionHead()

    @nn.compact
    def __call__(self, nn_input: NNInput) -> AllUnitsActionDistribution:
        """Forward pass."""
        # get data
        input_scalars = nn_input.scalars
        input_fields = nn_input.fields
        base_action_masks = nn_input.base_action_masks
        sap_action_masks = nn_input.sap_action_masks

        # process inputs
        embedding = self.torso(input_scalars, input_fields)

        # head which outputs a distribution object
        action_distribution = self.action_head(embedding, base_action_masks, sap_action_masks)

        return action_distribution


class Critic(nn.Module):
    """Feedforward critic network providing a scalar value"""

    torso: nn.Module
    critic_head: nn.Module = CriticHead()

    @nn.compact
    def __call__(self, nn_input: NNInput) -> jnp.array:
        """Forward pass."""
        # get data
        input_scalars = nn_input.scalars
        input_fields = nn_input.fields
        base_action_masks = nn_input.base_action_masks
        sap_action_masks = nn_input.sap_action_masks

        # process inputs
        embedding = self.torso(input_scalars, input_fields)

        # head which outputs a scalar
        value = self.critic_head(embedding)

        return value
