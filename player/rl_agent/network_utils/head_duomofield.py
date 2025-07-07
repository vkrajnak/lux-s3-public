import jax
import jax.numpy as jnp
import chex
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn

from ..network_input.types import ActionMasksDuoField
from ..network_utils.distributions import AllUnitsActionDistribution
from ..constants import N_MAX_UNITS, GRID_SHAPE, N_BASE_ACTIONS
from .distributions import UnitDiscreteActionDistribution


scale = jnp.ones((1, N_BASE_ACTIONS, )).at[...,-1].set(225)

class _UnitDiscreteActionHead(nn.Module):
    """Subhead for choosing mono-action (1 point in the grid) for 1 unit"""
    print_arch: bool = False

    @nn.compact
    def __call__(
        self,
        logits: chex.Array,  # (..., GRID_SHAPE)
        monoaction_mask: chex.Array,  # (..., GRID_SHAPE)
        monofield_base_converter: chex.Array,  # (..., GRID_SHAPE)
    ) -> tfd.Categorical:

        assert logits.shape[-2:] == GRID_SHAPE

        # mask impossible actions
        mask = (monofield_base_converter[...,None] == jnp.arange(N_BASE_ACTIONS, dtype=jnp.int32)[None,None,:])
        mask = mask[None, ...] & monoaction_mask[...,None]


        masked_logits = jnp.sum(logits[None,...,None], where=mask, axis=(-3,-2))

        masked_logits = masked_logits / scale
        masked_logits = jnp.where(
            jnp.any(mask, axis=(-3,-2)),
            masked_logits,
            jnp.finfo(jnp.float32).min
        )
        
        masked_logits = masked_logits.reshape(monoaction_mask.shape[:-2]+(N_BASE_ACTIONS, ))
        masked_logits = jnp.where(
            jnp.any(monoaction_mask, axis=(-2,-1))[...,None],
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
        sap_action_mask: chex.Array,  # (..., GRID_SHAPE)
        noop_action_mask: chex.Array,  # (..., GRID_SHAPE)
    ) -> tfd.Categorical:

        assert logits.shape[-2:] == GRID_SHAPE
        assert sap_action_mask.shape[-2:] == GRID_SHAPE

        # mask impossible actions
        masked_logits = jnp.where(
            sap_action_mask,
            logits,
            jnp.finfo(jnp.float32).min,
        )

        # deal with non-existent unit by setting a deterministic arbitrary action (so that log_probs are not influenced by it)
        masked_logits = jnp.where(
            jnp.any(sap_action_mask, axis=(-2,-1))[...,None,None],
            masked_logits,
            masked_logits * jnp.logical_not(noop_action_mask).astype(jnp.float32)
        )

        # flatten to use Categorical
        leading_shape = masked_logits.shape[:-2]
        masked_logits = masked_logits.reshape(*leading_shape, -1)  # (..., N_CELLS)  with N_CELLS = GRID_SHAPE[0] * GRID_SHAPE[1]

        return masked_logits if self.print_arch else tfd.Categorical(logits=masked_logits)

class _UnitDuomoFieldActionHead(nn.Module):
    """Subhead for choosing a base action (discrete) and a sap action (1 point in the grid) for 1 unit"""
    print_arch: bool = False

    def setup(self):
        self.base_action_head = _UnitDiscreteActionHead(self.print_arch)
        self.sap_action_head = _UnitFieldActionHead(self.print_arch)

    @nn.compact
    def __call__(
        self,
        unit_logits: chex.Array,  # (..., GRID_SHAPE)
        monoaction_mask: chex.Array,  # (..., GRID_SHAPE)
        base_action_mask: chex.Array,  # (..., GRID_SHAPE)
        sap_action_mask: chex.Array,  # (..., GRID_SHAPE)
        noop_action_mask: chex.Array,  # (..., GRID_SHAPE)
        monofield_base_converter: chex.Array,  # (..., GRID_SHAPE)
    ) -> UnitDiscreteActionDistribution:

        distribution_base_action = self.base_action_head(unit_logits, monoaction_mask, monofield_base_converter)
        distribution_sap_action = self.sap_action_head(unit_logits, sap_action_mask, noop_action_mask)

        return UnitDiscreteActionDistribution(distribution_base_action, distribution_sap_action)


class AllUnitsDuomoFieldActionHead(nn.Module):
    """Main head for choosing mono-action (1 point in the grid) for all units"""
    print_arch: bool = False

    def setup(self):
        self.unit_heads = [_UnitDuomoFieldActionHead(self.print_arch) for _ in range(N_MAX_UNITS)]

    def __call__(
            self,
            logits: chex.Array,
            action_masks: ActionMasksDuoField,
    ) -> AllUnitsActionDistribution:

        monoaction_masks = action_masks.monoaction_masks  # (N_MAX_UNITS, GRID_SHAPE)
        base_action_masks = action_masks.base_action_masks  # (N_MAX_UNITS, GRID_SHAPE)
        sap_action_masks = action_masks.sap_action_masks  # (N_MAX_UNITS, GRID_SHAPE)
        noop_action_masks = action_masks.noop_action_masks  # (N_MAX_UNITS, GRID_SHAPE)
        monofield_base_converter = action_masks.monofield_base_converter  # (N_MAX_UNITS, GRID_SHAPE)

        assert logits.shape[-3:] == GRID_SHAPE + (N_MAX_UNITS, )
        assert monoaction_masks.shape[-3:] == (N_MAX_UNITS, ) + GRID_SHAPE
        assert base_action_masks.shape[-3:] == (N_MAX_UNITS,) + GRID_SHAPE
        assert sap_action_masks.shape[-3:] == (N_MAX_UNITS,) + GRID_SHAPE
        assert noop_action_masks.shape[-3:] == (N_MAX_UNITS,) + GRID_SHAPE
        assert monofield_base_converter.shape[-3:] == (N_MAX_UNITS,) + GRID_SHAPE

        units_distributions = [unit_head(
            logits[..., unit_id],
            monoaction_masks[..., unit_id, :, :],
            base_action_masks[..., unit_id, :, :],
            sap_action_masks[..., unit_id, :, :],
            noop_action_masks[..., unit_id, :, :],
            monofield_base_converter[..., unit_id, :, :]
        )
                               for unit_id, unit_head in enumerate(self.unit_heads)]

        # jax.debug.print("{}\n {}", units_distributions[0].distribution_sap_action.logits, sap_action_masks[...,0,:,:].astype(jnp.int16))

        return AllUnitsActionDistribution(units_distributions)
