import jax
import jax.numpy as jnp
import chex
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn

from ..network_input.types import ActionMasksDiscreteAndField
from ..network_utils.distributions import AllUnitsActionDistribution
from ..constants import N_MAX_UNITS, GRID_SHAPE, N_BASE_ACTIONS

from .head_unit_subheads import UnitBaseDiscreteActionHead
from .distributions import UnitDiscreteActionDistribution


class _UnitSapFieldActionHead(nn.Module):
    """Subhead for choosing sap action (1 point in the grid) for 1 unit"""
    print_arch: bool = False

    @nn.compact
    def __call__(
        self,
        sap_field_action_logits: chex.Array,  # (..., GRID_SHAPE)
        sap_action_field_mask: chex.Array,  # (..., GRID_SHAPE)
    ) -> tfd.Categorical:

        assert sap_field_action_logits.shape[-2:] == GRID_SHAPE
        assert sap_action_field_mask.shape[-2:] == GRID_SHAPE

        # mask impossible actions
        masked_logits = jnp.where(
            sap_action_field_mask,
            sap_field_action_logits,
            jnp.finfo(jnp.float32).min,
        )

        # flatten to use Categorical
        leading_shape = masked_logits.shape[:-2]
        masked_logits = masked_logits.reshape(*leading_shape, -1)  # (..., N_CELLS)  with N_CELLS = GRID_SHAPE[0] * GRID_SHAPE[1]

        # deal with non-existent unit by setting a deterministic arbitrary action (so that log_probs are not influenced by it)
        masked_logits = jnp.where(
            jnp.any(sap_action_field_mask, axis=(-2,-1))[...,None],
            masked_logits,
            masked_logits.at[..., 0].set(1.0),
        )

        return masked_logits if self.print_arch else tfd.Categorical(logits=masked_logits)


class _UnitDiscreteAndFieldActionHead(nn.Module):
    """Subhead for choosing a base action (discrete) and a sap action (1 point in the grid) for 1 unit"""
    print_arch: bool = False

    def setup(self):
        self.base_action_head = UnitBaseDiscreteActionHead(self.print_arch)
        self.sap_action_head = _UnitSapFieldActionHead(self.print_arch)

    def __call__(
            self,
            logits_base_action_discrete: chex.Array,
            logits_sap_action_field: chex.Array,
            base_action_discrete_mask: chex.Array,
            sap_action_field_mask: chex.Array,
    ) -> UnitDiscreteActionDistribution:

        # tweak logits to make sap actions less probable
        logits_base_action_discrete = logits_base_action_discrete.at[..., -1].set(logits_base_action_discrete[..., -1] - jnp.log(100))

        distribution_base_action = self.base_action_head(logits_base_action_discrete, base_action_discrete_mask)
        distribution_sap_action = self.sap_action_head(logits_sap_action_field, sap_action_field_mask)

        return UnitDiscreteActionDistribution(distribution_base_action, distribution_sap_action)


class AllUnitsDiscreteAndFieldActionHead(nn.Module):
    """Main head for choosing a base action (discrete) and a sap action (1 point in the grid) for all units"""
    print_arch: bool = False

    def setup(self):
        self.unit_heads = [_UnitDiscreteAndFieldActionHead(self.print_arch) for _ in range(N_MAX_UNITS)]

    def __call__(
            self,
            logits: chex.Array,  # (N_MAX_UNITS, N_BASE_ACTION), (GRID_SHAPE, N_MAX_UNITS)
            action_masks: ActionMasksDiscreteAndField,
    ) -> AllUnitsActionDistribution:

        logits_base_action_discrete, logits_sap_action_field = logits

        assert logits_base_action_discrete.shape[-2:] == (N_MAX_UNITS, N_BASE_ACTIONS)
        assert logits_sap_action_field.shape[-3:] == GRID_SHAPE + (N_MAX_UNITS, )

        base_action_discrete_masks = action_masks.base_action_discrete_masks  # (N_MAX_UNITS, N_BASE_ACTIONS)
        sap_action_field_masks = action_masks.sap_action_field_masks  # (N_MAX_UNITS, GRID_SHAPE)

        assert base_action_discrete_masks.shape[-2:] == (N_MAX_UNITS, N_BASE_ACTIONS)
        assert sap_action_field_masks.shape[-3:] == (N_MAX_UNITS,) + GRID_SHAPE

        units_distributions = [unit_head(
            logits_base_action_discrete[..., unit_id, :],
            logits_sap_action_field[..., unit_id],
            base_action_discrete_masks[..., unit_id, :],
            sap_action_field_masks[..., unit_id, :, :]
        )
                               for unit_id, unit_head in enumerate(self.unit_heads)]

        return AllUnitsActionDistribution(units_distributions)
