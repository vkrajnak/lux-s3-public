import chex
from flax import linen as nn

from ..network_input.types import ActionMasksDiscrete
from ..network_utils.distributions import UnitDiscreteActionDistribution, AllUnitsActionDistribution
from ..constants import N_MAX_UNITS, N_BASE_ACTIONS, N_SAP_ACTIONS

from .head_unit_subheads import UnitBaseDiscreteActionHead, UnitSapDiscreteActionHead


class _UnitDiscreteActionHead(nn.Module):
    """Subhead for choosing base + sap discrete actions for 1 unit"""
    print_arch: bool = False

    def setup(self):
        self.base_action_head = UnitBaseDiscreteActionHead(self.print_arch)
        self.sap_action_head = UnitSapDiscreteActionHead(self.print_arch)

    def __call__(
            self,
            base_and_sap_action_logits: chex.Array,  # (N_BASE_ACTIONS + N_SAP_ACTIONS)
            base_action_mask: chex.Array,  # (N_BASE_ACTIONS)
            sap_action_mask: chex.Array,  # (N_SAP_ACTIONS)
    ) -> UnitDiscreteActionDistribution:

        assert base_and_sap_action_logits.shape[-1] == N_BASE_ACTIONS + N_SAP_ACTIONS

        base_action_logits = base_and_sap_action_logits[..., :N_BASE_ACTIONS]
        sap_action_logits = base_and_sap_action_logits[..., N_BASE_ACTIONS:]

        distribution_base_action = self.base_action_head(base_action_logits, base_action_mask)
        distribution_sap_action = self.sap_action_head(sap_action_logits, sap_action_mask)

        return UnitDiscreteActionDistribution(distribution_base_action, distribution_sap_action)


class AllUnitsDiscreteSparseActionHead(nn.Module):
    """Main head for choosing base + sap discrete actions for all units"""
    print_arch: bool = False

    def setup(self):
        self.unit_heads = [_UnitDiscreteActionHead(self.print_arch) for _ in range(N_MAX_UNITS)]

    def __call__(
            self,
            x: chex.Array,  # (N_MAX_UNITS, N_BASE_ACTIONS + N_SAP_ACTIONS)
            action_masks: ActionMasksDiscrete,
    ) -> AllUnitsActionDistribution:

        assert x.shape[-2:] == (N_MAX_UNITS, N_BASE_ACTIONS + N_SAP_ACTIONS)

        base_action_masks = action_masks.base_action_masks  # (N_MAX_UNITS, N_BASE_ACTIONS)
        sap_action_masks = action_masks.sap_action_masks  # (N_MAX_UNITS, N_SAP_ACTIONS)

        units_distributions = [unit_head(x[..., unit_id, :], base_action_masks[..., unit_id, :], sap_action_masks[..., unit_id, :])
                               for unit_id, unit_head in enumerate(self.unit_heads)]

        return AllUnitsActionDistribution(units_distributions)






