import jax.numpy as jnp
import chex
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal
from typing import Sequence

from .utils import parse_activation_fn


class MLPBlock(nn.Module):
    """MLP"""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(jnp.sqrt(2.0))
    activate_final: bool = True

    @nn.compact
    def __call__(self, inputx: chex.Array) -> chex.Array:
        """Forward pass."""
        x = inputx
        for layer_size in self.layer_sizes:
            x = nn.Dense(
                layer_size, kernel_init=self.kernel_init, use_bias=not self.use_layer_norm
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if self.activate_final or layer_size != self.layer_sizes[-1]:
                x = parse_activation_fn(self.activation)(x)
        return x


class CNNBlock(nn.Module):
    """2D CNN. Expects input of shape (..., height, width, channels), that is, CHANNEL_LAST!"""

    features_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, inputx: chex.Array) -> chex.Array:
        """Forward pass."""
        x = inputx
        # Convolutional layers
        for features, kernel, stride in zip(self.features_sizes, self.kernel_sizes, self.strides):
            x = nn.Conv(
                features, (kernel, kernel), (stride, stride), use_bias=not self.use_layer_norm, padding='VALID'
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(reduction_axes=(-3, -2, -1))(x)
            x = parse_activation_fn(self.activation)(x)

        return x




