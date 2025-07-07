import chex
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.linen.initializers import orthogonal

from ..network_input.types import ActionMasksDiscrete
from ..network_utils.distributions import UnitDiscreteActionDistribution, AllUnitsActionDistribution
from ..constants import N_MAX_UNITS, N_BASE_ACTIONS, N_SAP_ACTIONS

from .head_unit_subheads import UnitBaseDiscreteActionHead, UnitSapDiscreteActionHead


class _UnitBaseActionHead(nn.Module):
    """Subhead for choosing the base discrete action (no-op, move, sap) for 1 unit - includes a dense layer"""
    print_arch: bool = False

    def setup(self):
        self.discrete_action = UnitBaseDiscreteActionHead(self.print_arch)
        self.dense = nn.Dense(N_BASE_ACTIONS, kernel_init=orthogonal(0.01))

    def __call__(
        self,
        x: chex.Array,
        base_action_discrete_mask: chex.Array,  # (N_BASE_ACTIONS)
        print_arch=False,
    ) -> tfd.Categorical:

        base_action_discrete_logits = self.dense(x)

        return self.discrete_action(base_action_discrete_logits, base_action_discrete_mask)


class _UnitSapActionHead(nn.Module):
    """Subhead for choosing where to sap (in discrete format) for 1 unit - includes a dense layer"""
    print_arch: bool = False

    def setup(self):
        self.discrete_action = UnitSapDiscreteActionHead(self.print_arch)
        self.dense = nn.Dense(N_SAP_ACTIONS, kernel_init=orthogonal(0.01))

    def __call__(
        self,
        x: chex.Array,
        sap_action_discrete_mask: chex.Array,  # (N_SAP_ACTIONS)
        print_arch=False,
    ) -> tfd.Categorical:

        sap_action_discrete_logits = self.dense(x)

        return self.discrete_action(sap_action_discrete_logits, sap_action_discrete_mask)


class _UnitDiscreteActionHead(nn.Module):
    """Subhead for choosing base + sap discrete actions for 1 unit - includes a dense layer for each"""
    print_arch: bool = False

    def setup(self):
        self.base_action_head = _UnitBaseActionHead(self.print_arch)
        self.sap_action_head = _UnitSapActionHead(self.print_arch)

    def __call__(
            self,
            x: chex.Array,
            base_action_mask: chex.Array,  # (N_BASE_ACTIONS)
            sap_action_mask: chex.Array,  # (N_SAP_ACTIONS)
    ) -> UnitDiscreteActionDistribution:

        distribution_base_action = self.base_action_head(x, base_action_mask)
        distribution_sap_action = self.sap_action_head(x, sap_action_mask)

        return UnitDiscreteActionDistribution(distribution_base_action, distribution_sap_action)


class AllUnitsDiscreteFullActionHead(nn.Module):
    """Main head for choosing base + sap discrete actions for all units - includes a dense layer for each"""
    print_arch: bool = False

    def setup(self):
        self.unit_heads = [_UnitDiscreteActionHead(self.print_arch) for _ in range(N_MAX_UNITS)]

    def __call__(
            self,
            x: chex.Array,
            action_masks: ActionMasksDiscrete,
    ) -> AllUnitsActionDistribution:

        base_action_masks = action_masks.base_action_masks  # (N_MAX_UNITS, N_BASE_ACTIONS)
        sap_action_masks = action_masks.sap_action_masks  # (N_MAX_UNITS, N_SAP_ACTIONS)

        units_distributions = [unit_head(x, base_action_masks[..., unit_id, :], sap_action_masks[..., unit_id, :])
                               for unit_id, unit_head in enumerate(self.unit_heads)]

        return AllUnitsActionDistribution(units_distributions)






