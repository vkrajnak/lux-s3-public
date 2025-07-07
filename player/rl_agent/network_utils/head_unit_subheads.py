import chex
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn

from ..constants import N_BASE_ACTIONS, N_SAP_ACTIONS


class UnitBaseDiscreteActionHead(nn.Module):
    """Subhead for choosing the base discrete action (no-op, move, sap) for 1 unit"""
    print_arch: bool = False

    @nn.compact
    def __call__(
        self,
        base_action_discrete_logits: chex.Array,  # (N_BASE_ACTIONS)
        base_action_discrete_mask: chex.Array,  # (N_BASE_ACTIONS)
    ) -> tfd.Categorical:

        assert base_action_discrete_logits.shape[-1] == N_BASE_ACTIONS

        # tweak logits to make sap actions less probable
        # base_action_discrete_logits = base_action_discrete_logits.at[..., -1].set(base_action_discrete_logits[..., -1] - jnp.log(1000))

        masked_logits = jnp.where(
            base_action_discrete_mask,
            base_action_discrete_logits,
            jnp.finfo(jnp.float32).min,
        )

        # deal with non-existent unit by setting a deterministic arbitrary action (so that log_probs are not influenced by it)
        masked_logits = jnp.where(
            jnp.any(base_action_discrete_mask, axis=-1)[...,None],
            masked_logits,
            masked_logits.at[..., 0].set(1.0),
        )

        return masked_logits if self.print_arch else tfd.Categorical(logits=masked_logits)

    
class UnitSapDiscreteActionHead(nn.Module):
    """Subhead for choosing where to sap (in discrete format) for 1 unit"""
    print_arch: bool = False

    @nn.compact
    def __call__(
        self,
        sap_action_discrete_logits: chex.Array,  # (N_SAP_ACTIONS)
        sap_action_discrete_mask: chex.Array,  # (N_SAP_ACTIONS)
    ) -> tfd.Categorical:

        assert sap_action_discrete_logits.shape[-1] == N_SAP_ACTIONS

        masked_logits = jnp.where(
            sap_action_discrete_mask,
            sap_action_discrete_logits,
            jnp.finfo(jnp.float32).min,
        )

        # deal with non-existent unit by setting a deterministic arbitrary action (so that log_probs are not influenced by it)
        masked_logits = jnp.where(
            jnp.any(sap_action_discrete_mask, axis=-1)[...,None],
            masked_logits,
            masked_logits.at[..., 0].set(1.0),
        )

        return masked_logits if self.print_arch else tfd.Categorical(logits=masked_logits)


