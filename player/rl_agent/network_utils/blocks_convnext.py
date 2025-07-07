import enum
from typing import Sequence, Union

import jax.numpy as jnp
import chex
import flax.linen as nn
from flax.linen.initializers import Initializer, orthogonal
import jax

from .utils import parse_activation_fn
from .blocks_base import MLPBlock
from .blocks_embedding import MergeTileType
from .blocks_resnet import SELayer, SingleConv
from ..network_input.types import NNInput



class ConvNeXtBlock(nn.Module):
    """ ConvNeXt inverted bottleneck Block
    https://arxiv.org/pdf/2201.03545
    https://github.com/facebookresearch/ConvNeXt
    consider the original 7x7 convolution kernel if 3x3 produces unsatisfactory results
    Expects input of shape (..., height, width, channels), that is, CHANNEL_LAST! """

    features: int
    kernel: int = 3
    activation: str = "gelu" 
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        expanded_features = self.features * 4
        output = x

        # Depthwise convolutional layer.
        output = nn.Conv(
            self.features, (self.kernel, self.kernel), use_bias=not self.use_layer_norm, feature_group_count=self.features, padding="SAME"
        )(output)
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)
        # no activation after spatial convolution

        # Channel expansion layer.
        output = nn.Conv(
            expanded_features, (1, 1), use_bias=not self.use_layer_norm
        )(output)
        if self.use_layer_norm:
            output = self.layernorm(output)
        output = parse_activation_fn(self.activation)(output)

        # Channel contraction layer.
        output = nn.Conv(
            self.features, (1, 1), use_bias=not self.use_layer_norm
        )(output)
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)

        return x + output


class DownsamplingLayer(nn.Module):
    """ ConvNeXt downsampling layer - kind of like pooling due to the non-overlapping convolution
    https://arxiv.org/pdf/2201.03545
    Expects input of shape (..., height, width, channels), that is, CHANNEL_LAST! """

    features: int
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        
        output = nn.Conv(
            self.features, (2,2), (2,2), use_bias=not self.use_layer_norm
        )(x)
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)

        return output


class UpsamplingLayer(nn.Module):
    """ UpsamplingLayer
    https://github.com/xzhong411/BCU-Net/blob/main/code/core/models.py
    Expects input of shape (..., height, width, channels), that is, CHANNEL_LAST! """

    features: int
    use_layer_norm: bool = False
    activation: str = "gelu" 

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        up_shape = (x.shape[-3] * 2, x.shape[-2] * 2)
        
        output = jnp.image.resize(x, up_shape, method="bilinear")

        output = nn.Conv(
            self.features, (3,3), use_bias=not self.use_layer_norm
        )
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)
        output = parse_activation_fn(self.activation)(output)

        return output