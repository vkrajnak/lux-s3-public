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
from ..network_input.types import NNInput


class SELayer(nn.Module):
    """ Squeeze-excitation layer for ResidualBlock
    Following https://pytorch.org/vision/stable/_modules/torchvision/ops/misc.html#SqueezeExcitation
    Expects input of shape (..., height, width, channels), that is, CHANNEL_LAST! """

    squeeze_factor:  int = 16
    activation1: str = "relu"
    activation2: str = "sigmoid"

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:

        pool_window = x.shape[-3:-1]
        features = x.shape[-1]
        squeeze_features = features // self.squeeze_factor
        
        output = nn.avg_pool(x, pool_window, padding='VALID')
        # jax.debug.print("{}", output[0,0,0] == output[:,:,0].mean()) # if in doubt
        
        # Squeeze
        output = nn.Conv(
            squeeze_features, (1, 1), (1, 1), kernel_init=orthogonal(1.4142)
        )(output)
        
        output = parse_activation_fn(self.activation1)(output)

        # Excitation
        output = nn.Conv(
            features, (1, 1), (1, 1), kernel_init=orthogonal(1.4142)
        )(output)
        
        output = parse_activation_fn(self.activation2)(output)

        return x * output

class ResidualBlock(nn.Module):
    """ Residual Block
    Expects input of shape (..., height, width, channels), that is, CHANNEL_LAST! """

    features: int
    kernel: int = 3
    activation: str = "relu"
    use_layer_norm: bool = False
    downsampling: bool = False
    se_layer: bool = False
    se_factor: int = 4

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        output = x

        stride = 1
        if self.downsampling:
            stride = 2

        # https://arxiv.org/abs/1709.01507v4 table 14 suggests that SELayer before residual block is marginally better than after
        if self.se_layer:
            output = SELayer(squeeze_factor=self.se_factor)(output)

        # First layer in residual block.
        output = nn.Conv(
            self.features, (self.kernel, self.kernel), (stride, stride), use_bias=not self.use_layer_norm, padding='SAME', kernel_init=orthogonal(1.4142)
        )(output)
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)
        output = parse_activation_fn(self.activation)(output)

        # Second layer in residual block.
        output = nn.Conv(
            self.features, (self.kernel, self.kernel), use_bias=not self.use_layer_norm, padding='SAME', kernel_init=orthogonal(1.4142)
        )(output)
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)
        # output = parse_activation_fn(self.activation)(output)

        if self.downsampling or x.shape[-1] != self.features:
            x = nn.Conv(
                self.features, (1, 1), (stride, stride), use_bias=not self.use_layer_norm, kernel_init=orthogonal(1.4142)
            )(x)

        output = x + output
        output = parse_activation_fn(self.activation)(output)

        return output


class BottleneckBlock(nn.Module):
    """ Residual Bottleneck Block
    Expects input of shape (..., height, width, channels), that is, CHANNEL_LAST! """

    features: int
    kernel: int = 3
    reduction_factor: int = 4
    activation: str = "relu"
    use_layer_norm: bool = False
    downsampling: bool = False
    se_layer: bool = False
    se_factor: int = 4

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        reduced_features = self.features // self.reduction_factor
        output = x

        stride = 1
        if self.downsampling:
            stride = 2

        # https://arxiv.org/abs/1709.01507v4 table 14 suggests that SELayer before residual block is marginally better than after
        if self.se_layer:
            output = SELayer(squeeze_factor=self.se_factor)(output)

        # Squeeze layer in residual block.
        output = nn.Conv(
            reduced_features, (1, 1), use_bias=not self.use_layer_norm, kernel_init=orthogonal(1.4142)
        )(output)
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)
        output = parse_activation_fn(self.activation)(output)

        # Conv layer in residual block.
        output = nn.Conv(
            reduced_features, (self.kernel, self.kernel), (stride, stride), use_bias=not self.use_layer_norm, padding='SAME', kernel_init=orthogonal(1.4142)
        )(output)
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)
        output = parse_activation_fn(self.activation)(output)

        # Unsqueeze layer in residual block.
        output = nn.Conv(
            self.features, (1, 1), use_bias=not self.use_layer_norm, kernel_init=orthogonal(1.4142)
        )(output)
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)
        # output = parse_activation_fn(self.activation)(output)

        if self.downsampling or x.shape[-1] != self.features:
            x = nn.Conv(
                self.features, (1, 1), (stride, stride), use_bias=not self.use_layer_norm, kernel_init=orthogonal(1.4142)
            )(x)

        output = x + output
        output = parse_activation_fn(self.activation)(output)

        return output


class UpsamplingBottleneckBlock(nn.Module):
    """ Residual Bottleneck Block
    Expects input of shape (..., height, width, channels), that is, CHANNEL_LAST! """

    features: int
    kernel: int = 3
    activation: str = "relu"
    use_layer_norm: bool = False
    se_layer: bool = False

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        reduced_features = self.features // 4
        output = x

        # https://arxiv.org/abs/1709.01507v4 table 14 suggests that SELayer before residual block is marginally better than after
        if self.se_layer:
            output = SELayer(squeeze_factor=4)(output)

        # Squeeze layer in residual block.
        output = nn.Conv(
            reduced_features, (1, 1), use_bias=not self.use_layer_norm, kernel_init=orthogonal(1.4142)
        )(output)
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)
        output = parse_activation_fn(self.activation)(output)

        # Conv layer in residual block.
        output = nn.ConvTranspose(
            reduced_features, (self.kernel, self.kernel), (2, 2), use_bias=not self.use_layer_norm, kernel_init=orthogonal(1.4142)
        )(output)
        if self.use_layer_norm:
            output = self.layernorm(output)
        output = parse_activation_fn(self.activation)(output)

        # Unsqueeze layer in residual block.
        output = nn.Conv(
            self.features, (1, 1), use_bias=not self.use_layer_norm, kernel_init=orthogonal(1.4142)
        )(output)
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)
        output = parse_activation_fn(self.activation)(output)

        # upsampling x seems a bit odd here, omitted

        return output


class SingleConv(nn.Module):
    """ Single Conv.
    Expects input of shape (..., height, width, channels), that is, CHANNEL_LAST! """

    features: int
    kernel: int = 1
    stride: int = 1
    activation: str = "relu"
    use_layer_norm: bool = False
    padding: bool = "SAME"
    feature_group_count: int = 1
    kernel_scale: float=1.4142

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        
        output = nn.Conv(
            self.features, (self.kernel, self.kernel), (self.stride, self.stride), use_bias=not self.use_layer_norm, padding=self.padding,
            feature_group_count=self.feature_group_count, kernel_init=orthogonal(self.kernel_scale)
        )(x)
        if self.use_layer_norm:
            output = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)(output)
        output = parse_activation_fn(self.activation)(output)

        return output


class DownsamplingStrategy(enum.Enum):
    AVG_POOL = "avg_pool"
    CONV_MAX = "conv+max"  # Used in IMPALA
    LAYERNORM_RELU_CONV = "layernorm+relu+conv"  # Used in MuZero
    CONV = "conv"


def make_downsampling_layer(
        strategy: Union[str, DownsamplingStrategy],
        output_channels: int,
) -> nn.Module:
    """Returns a sequence of modules corresponding to the desired downsampling."""
    strategy = DownsamplingStrategy(strategy)

    if strategy is DownsamplingStrategy.AVG_POOL:
        return lambda x: nn.avg_pool(x, window_shape=(3, 3, 1), strides=(2, 2, 1), padding="SAME")

    elif strategy is DownsamplingStrategy.CONV:
        return nn.Sequential(
            [
                nn.Conv(
                    features=output_channels,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    kernel_init=nn.initializers.truncated_normal(1e-2),
                ),
            ]
        )

    elif strategy is DownsamplingStrategy.CONV_MAX:
        return nn.Sequential(
            [
                nn.Conv(features=output_channels, kernel_size=(3, 3), strides=(1, 1)),
                lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME"),
            ]
        )
    else:
        raise ValueError(
            "Unrecognized downsampling strategy. Expected one of"
            f" {[strategy.value for strategy in DownsamplingStrategy]}"
            f" but received {strategy}."
        )



