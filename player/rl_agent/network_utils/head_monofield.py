import jax
import jax.numpy as jnp
import chex
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn

from ..network_input.types import ActionMasksMonoField
from ..network_utils.distributions import AllUnitsActionDistribution
from ..constants import N_MAX_UNITS, GRID_SHAPE


class _UnitMonoFieldActionHead(nn.Module):
    """Subhead for choosing mono-action (1 point in the grid) for 1 unit"""
    print_arch: bool = False

    @nn.compact
    def __call__(
        self,
        unit_logits: chex.Array,  # (..., GRID_SHAPE)
        monoaction_mask: chex.Array,  # (..., GRID_SHAPE)
        base_action_mask: chex.Array,  # (..., GRID_SHAPE)
        sap_action_mask: chex.Array,  # (..., GRID_SHAPE)
        noop_action_mask: chex.Array,  # (..., GRID_SHAPE)
    ) -> tfd.Categorical:

        assert unit_logits.shape[-2:] == GRID_SHAPE

        # tweak logits to make sap actions less probable
        # n_base_actions_possible = jnp.sum(base_action_mask, axis=(-1, -2))
        # n_sap_actions_possible = jnp.sum(sap_action_mask, axis=(-1, -2))
        unit_logits = jnp.where(
            sap_action_mask,
            # unit_logits - jnp.log(N_SAP_ACTIONS - 5),
            unit_logits - jnp.log(100),
            unit_logits,
        )

        # mask impossible actions
        masked_logits = jnp.where(
            monoaction_mask,
            unit_logits,
            jnp.finfo(jnp.float32).min,
        )

        # deal with non-existent unit by setting a deterministic arbitrary action (so that log_probs are not influenced by it)
        nounit_mask = jnp.broadcast_to(jnp.any(monoaction_mask, axis=(-2, -1))[..., None, None], masked_logits.shape)
        masked_logits = jnp.where(
            nounit_mask,
            masked_logits,
            masked_logits * jnp.logical_not(noop_action_mask).astype(jnp.float32),
        )

        # flatten to use Categorical
        leading_shape = masked_logits.shape[:-2]
        masked_logits = masked_logits.reshape(*leading_shape, -1)  # (..., N_CELLS)  with N_CELLS = GRID_SHAPE[0] * GRID_SHAPE[1]

        return masked_logits if self.print_arch else tfd.Categorical(logits=masked_logits)


class AllUnitsMonoFieldActionHead(nn.Module):
    """Main head for choosing mono-action (1 point in the grid) for all units"""
    print_arch: bool = False

    def setup(self):
        self.unit_heads = [_UnitMonoFieldActionHead(self.print_arch) for _ in range(N_MAX_UNITS)]

    def __call__(
            self,
            logits: chex.Array,
            action_masks: ActionMasksMonoField,
    ) -> AllUnitsActionDistribution:

        monoaction_masks = action_masks.monoaction_masks  # (N_MAX_UNITS, GRID_SHAPE)
        base_action_masks = action_masks.base_action_masks  # (N_MAX_UNITS, GRID_SHAPE)
        sap_action_masks = action_masks.sap_action_masks  # (N_MAX_UNITS, GRID_SHAPE)
        noop_action_masks = action_masks.noop_action_masks  # (N_MAX_UNITS, GRID_SHAPE)

        assert logits.shape[-3:] == GRID_SHAPE + (N_MAX_UNITS, )
        assert monoaction_masks.shape[-3:] == (N_MAX_UNITS, ) + GRID_SHAPE
        assert base_action_masks.shape[-3:] == (N_MAX_UNITS,) + GRID_SHAPE
        assert sap_action_masks.shape[-3:] == (N_MAX_UNITS,) + GRID_SHAPE
        assert noop_action_masks.shape[-3:] == (N_MAX_UNITS,) + GRID_SHAPE

        units_distributions = [unit_head(
            logits[..., unit_id],
            monoaction_masks[..., unit_id, :, :],
            base_action_masks[..., unit_id, :, :],
            sap_action_masks[..., unit_id, :, :],
            noop_action_masks[..., unit_id, :, :],
        )
                               for unit_id, unit_head in enumerate(self.unit_heads)]

        return AllUnitsActionDistribution(units_distributions)
