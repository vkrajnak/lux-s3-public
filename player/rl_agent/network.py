import jax.numpy as jnp
from flax import linen as nn

from .network_utils.torso_resnet import CustomResNet
from .network_utils.torso_unet import UNet
from .network_utils.torso_convnext import ConvNeXt
from .network_utils.necks import get_neck_properties
from .network_utils.distributions import AllUnitsActionDistribution

from .network_utils.head_discrete_full import AllUnitsDiscreteFullActionHead
from .network_utils.head_discrete_sparse import AllUnitsDiscreteSparseActionHead
from .network_utils.head_monofield import AllUnitsMonoFieldActionHead
from .network_utils.head_duofield import AllUnitsDuoFieldActionHead
from .network_utils.head_duomofield import AllUnitsDuomoFieldActionHead
from .network_utils.head_discrete_and_field import AllUnitsDiscreteAndFieldActionHead

from .network_input.types import NNInput


TORSO_FN = {
    "resnet": CustomResNet,
    "unet": UNet,
    "convnext": ConvNeXt,
}

ACTOR_HEAD_FN = {
    "discrete_full": AllUnitsDiscreteFullActionHead,
    "discrete_sparse": AllUnitsDiscreteSparseActionHead,
    "monofield": AllUnitsMonoFieldActionHead,
    "duofield": AllUnitsDuoFieldActionHead,
    "duomofield": AllUnitsDuomoFieldActionHead,
    "discrete_and_field": AllUnitsDiscreteAndFieldActionHead,
}


class Actor(nn.Module):
    """Feedforward actor network providing distribution of action for all units"""

    torso: str
    torso_kwargs: dict
    head: str
    print_arch: bool = False

    def setup(self):
        neck_fn, neck_kwargs = get_neck_properties(self.torso, self.torso_kwargs, self.head)

        self._torso = TORSO_FN[self.torso](**self.torso_kwargs)
        self._neck = neck_fn(**neck_kwargs)
        self._head = ACTOR_HEAD_FN[self.head](self.print_arch)

    def __call__(self, nn_input: NNInput) -> AllUnitsActionDistribution:
        """Forward pass."""

        # process inputs
        x = self._torso(nn_input)
        x = self._neck(x)

        # head which outputs a distribution object
        action_distribution = self._head(x, nn_input.action_masks)

        return action_distribution


class Critic(nn.Module):
    """Feedforward critic network providing a scalar value"""

    torso: str
    torso_kwargs: dict

    def setup(self):
        neck_fn, neck_kwargs = get_neck_properties(self.torso, self.torso_kwargs, "critic")

        self._torso = TORSO_FN[self.torso](**self.torso_kwargs)
        self._neck = neck_fn(**neck_kwargs)
        self._head = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))

    def __call__(self, nn_input: NNInput) -> jnp.array:
        """Forward pass."""

        # process inputs
        x = self._torso(nn_input)
        x = self._neck(x)

        # head which outputs a scalar
        value = self._head(x)

        # TODO: should we apply a scaling depending on the reward chosen? not clear since it predicts the advantage, not the value

        return value.squeeze(axis=-1)

