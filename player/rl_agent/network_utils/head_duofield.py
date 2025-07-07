import jax
import jax.numpy as jnp
import chex
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn

from ..network_input.types import ActionMasksDuoField
from ..network_utils.distributions import AllUnitsActionDistribution
from ..constants import N_MAX_UNITS, GRID_SHAPE, N_BASE_ACTIONS
from .distributions import UnitDiscreteActionDistribution


scale = jnp.ones((1, N_BASE_ACTIONS, )).at[...,-1].set(4)
bias = jnp.zeros((1, N_BASE_ACTIONS, )).at[...,-1].set(3)

class _UnitDiscreteActionHead(nn.Module):
    """Subhead for choosing mono-action (1 point in the grid) for 1 unit"""
    print_arch: bool = False

    @nn.compact
    def __call__(
        self,
        logits: chex.Array,  # (..., GRID_SHAPE)
        base_action_mask: chex.Array,  # (..., GRID_SHAPE)
        monofield_base_converter: chex.Array,  # (..., GRID_SHAPE)
    ) -> tfd.Categorical:

        assert logits.shape[-2:] == GRID_SHAPE
        assert base_action_mask.shape[-2:] == GRID_SHAPE

        # mask impossible actions
        mask = (monofield_base_converter[...,None] == jnp.arange(N_BASE_ACTIONS, dtype=jnp.int32)[None,None,:])
        mask = mask[None, ...] & base_action_mask[...,None]

        masked_logits = jnp.sum(logits[None,...,None], where=mask, axis=(-3,-2))

        masked_logits = masked_logits / scale - bias
        masked_logits = jnp.where(
            jnp.any(mask, axis=(-3,-2)),
            masked_logits,
            jnp.finfo(jnp.float32).min
        )

        masked_logits = masked_logits.reshape(base_action_mask.shape[:-2]+(N_BASE_ACTIONS, ))
        
        masked_logits = jnp.where(
            jnp.any(base_action_mask, axis=(-2,-1))[...,None],
            masked_logits,
            masked_logits.at[...,0].set(1.0)
        )
        
        return masked_logits if self.print_arch else tfd.Categorical(logits=masked_logits)

class _UnitFieldActionHead(nn.Module):
    """Subhead for choosing mono-action (1 point in the grid) for 1 unit"""
    print_arch: bool = False

    @nn.compact
    def __call__(
        self,
        logits: chex.Array,  # (..., GRID_SHAPE)
        action_mask: chex.Array,  # (..., GRID_SHAPE)
    ) -> tfd.Categorical:

        assert logits.shape[-2:] == GRID_SHAPE
        assert action_mask.shape[-2:] == GRID_SHAPE

        # mask impossible actions
        masked_logits = jnp.where(
            action_mask,
            logits,
            jnp.finfo(jnp.float32).min,
        )

        # flatten to use Categorical
        leading_shape = masked_logits.shape[:-2]
        masked_logits = masked_logits.reshape(*leading_shape, -1)  # (..., N_CELLS)  with N_CELLS = GRID_SHAPE[0] * GRID_SHAPE[1]

        # deal with non-existent unit by setting a deterministic arbitrary action (so that log_probs are not influenced by it)
        masked_logits = jnp.where(
            jnp.any(action_mask, axis=(-2,-1))[...,None],
            masked_logits,
            masked_logits.at[..., 0].set(1.0),
        )

        return masked_logits if self.print_arch else tfd.Categorical(logits=masked_logits)

class _UnitDuoFieldActionHead(nn.Module):
    """Subhead for choosing a base action (discrete) and a sap action (1 point in the grid) for 1 unit"""
    print_arch: bool = False

    def setup(self):
        self.base_action_head = _UnitDiscreteActionHead(self.print_arch)
        self.sap_action_head = _UnitFieldActionHead(self.print_arch)

    def __call__(
            self,
            logits_base_action: chex.Array,
            logits_sap_action: chex.Array,
            base_action_mask: chex.Array,
            sap_action_mask: chex.Array,
            monofield_base_converter: chex.Array,
    ) -> UnitDiscreteActionDistribution:

        distribution_base_action = self.base_action_head(logits_base_action, base_action_mask, monofield_base_converter)
        distribution_sap_action = self.sap_action_head(logits_sap_action, sap_action_mask)

        return UnitDiscreteActionDistribution(distribution_base_action, distribution_sap_action)


class AllUnitsDuoFieldActionHead(nn.Module):
    """Main head for choosing mono-action (1 point in the grid) for all units"""
    print_arch: bool = False

    def setup(self):
        self.unit_heads = [_UnitDuoFieldActionHead(self.print_arch) for _ in range(N_MAX_UNITS)]

    def __call__(
            self,
            logits: chex.Array,
            action_masks: ActionMasksDuoField,
    ) -> AllUnitsActionDistribution:

        base_action_masks = action_masks.base_action_masks  # (N_MAX_UNITS, GRID_SHAPE)
        sap_action_masks = action_masks.sap_action_masks  # (N_MAX_UNITS, GRID_SHAPE)
        monofield_base_converter = action_masks.monofield_base_converter  # (N_MAX_UNITS, GRID_SHAPE)

        assert logits.shape[-3:] == GRID_SHAPE + (2*N_MAX_UNITS, )
        assert base_action_masks.shape[-3:] == (N_MAX_UNITS,) + GRID_SHAPE
        assert sap_action_masks.shape[-3:] == (N_MAX_UNITS,) + GRID_SHAPE
        assert monofield_base_converter.shape[-3:] == (N_MAX_UNITS,) + GRID_SHAPE

        logits_base_action = logits[...,:N_MAX_UNITS]
        logits_sap_action = logits[...,N_MAX_UNITS:]

        units_distributions = [unit_head(
            logits_base_action[..., unit_id],
            logits_sap_action[..., unit_id],
            base_action_masks[..., unit_id, :, :],
            sap_action_masks[..., unit_id, :, :],
            monofield_base_converter[..., unit_id, :, :]
        )
                               for unit_id, unit_head in enumerate(self.unit_heads)]

        return AllUnitsActionDistribution(units_distributions)
