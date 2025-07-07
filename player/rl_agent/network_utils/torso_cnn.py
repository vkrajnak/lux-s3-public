import jax.numpy as jnp
import chex
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal
from typing import Sequence

from .blocks_base import CNNBlock, MLPBlock
from ..network_input.types import NNInput


class CNN(nn.Module):
    """2D CNN.
    Expects input of shape (..., height, width, channels), that is, CHANNEL_LAST!
    After this torso, the output is flattened and possibly put through an MLP of mlp_layer_sizes."""

    features_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    mlp_layer_sizes: Sequence[int] = None
    activation: str = "relu"
    use_layer_norm: bool = False
    output_type: str = "flat"

    @nn.compact
    def __call__(self, nn_input: NNInput) -> chex.Array:
        if self.output_type != "flat":
            raise Exception("CNN is only compatible with output_type='flat'.")
        x = nn_input.continuous_fields
        input_shape = nn_input.continuous_fields.shape
        x = CNNBlock(
            features_sizes=self.features_sizes,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
        )(x)

        # Flatten
        x = x.reshape(*input_shape[:-3], -1)

        # MLP layers
        if self.mlp_layer_sizes is not None:
            x = MLPBlock(
                layer_sizes=self.mlp_layer_sizes,
                activation=self.activation,
                use_layer_norm=self.use_layer_norm,
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                activate_final=True,
            )(x)

        return x
