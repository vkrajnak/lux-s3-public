import sys
from typing import Sequence, Union

import chex
import flax.linen as nn
import jax.numpy as jnp

from .blocks_base import MLPBlock
from .blocks_resnet import DownsamplingStrategy, make_downsampling_layer, ResidualBlock, BottleneckBlock, SingleConv, UpsamplingBottleneckBlock
from .blocks_embedding import MergeTileType
# from .utils import get_features_output, check_and_reshape
from ..network_input.types import NNInput


class UNet(nn.Module):
    """Basic Unet"""

    output_type: str
    kernel: int
    use_layer_norm: bool = False
    activation: str = "relu"
    n_initial_depth_convs: int = 1
    features_output: int = None
    se_layer: bool = False

    @nn.compact
    def __call__(self, nn_input: NNInput) -> chex.Array:

        # features_output = get_features_output(self.output_type, self.features_output)

        input = MergeTileType(4)(nn_input.continuous_fields, nn_input.tile_type_fields)
        input_shape = input.shape  # (..., 24, 24, n)

        num_channels = 64

        # --- initial block
        for _ in range(self.n_initial_depth_convs):
            input = SingleConv(
                features=num_channels,
                kernel=1,
                use_layer_norm=self.use_layer_norm,
                activation=self.activation,
            )(input)

        x0 = ResidualBlock(
            features=num_channels,
            kernel=self.kernel,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            se_layer=self.se_layer,
        )(input)
        # ---

        # --- encoder
        # 12x12
        x1 = BottleneckBlock(
            features=num_channels*2,
            kernel=self.kernel,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            downsampling=True,
            se_layer=self.se_layer,
        )(x0)

        # 6x6
        x2 = BottleneckBlock(
            features=num_channels*4,
            kernel=self.kernel,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            downsampling=True,
            se_layer=self.se_layer,
        )(x1)


        # 3x3
        x3 = BottleneckBlock(
            features=num_channels*8,
            kernel=self.kernel,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            downsampling=True,
            se_layer=self.se_layer,
        )(x2)
        # ---

        # --- decoder
        
        # 6x6
        X3 = UpsamplingBottleneckBlock(
            features=num_channels*4,
            kernel=self.kernel,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            se_layer=self.se_layer,
        )(x3)

        # 12x12
        X3 = jnp.concat([x2, X3], axis=-1)
        X2 = UpsamplingBottleneckBlock(
            features=num_channels*2,
            kernel=self.kernel,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            se_layer=self.se_layer,
        )(X3)

        # 24x24
        X2 = jnp.concat([x1, X2], axis=-1)
        output = UpsamplingBottleneckBlock(
            features=num_channels,
            kernel=self.kernel,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            se_layer=self.se_layer,
        )(X2)

        # ---
        

        # --- resize channels
        output = SingleConv(
                features=features_output,
                kernel=1,
                use_layer_norm=self.use_layer_norm,
                activation=self.activation,
            )(output)
        # ---

        # Flatten
        if self.output_type == "flat":
            output = output.reshape(*input_shape[:-3], -1)

        output = check_and_reshape(self.output_type, output, input_shape)

        return output
