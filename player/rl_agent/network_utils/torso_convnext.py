import sys
from typing import Sequence, Union

import chex
import flax.linen as nn
import jax.numpy as jnp

from .blocks_base import MLPBlock
from .blocks_resnet import SingleConv
from .blocks_convnext import ConvNeXtBlock, DownsamplingLayer
from .blocks_embedding import MergeTileType
from ..network_input.types import NNInput


class ConvNeXt(nn.Module):
    """ ConvNeXt without downsampling
    https://arxiv.org/pdf/2201.03545
    https://github.com/facebookresearch/ConvNeXt
    consider the original 7x7 convolution kernel if 3x3 produces unsatisfactory results"""

    channels_per_group: Sequence[int]
    blocks_per_group: Sequence[int]
    kernels_per_group: Sequence[int]
    mlp_layer_sizes: Sequence[int] = None
    use_layer_norm: bool = False
    activation: str = "gelu"
    n_initial_depth_convs: int = 1
    downsampling: bool = False 

    @nn.compact
    def __call__(self, nn_input: NNInput) -> chex.Array:

        input = MergeTileType(4)(nn_input.continuous_fields, nn_input.tile_type_fields)

        channels_blocks_strategies = zip(
            self.channels_per_group, self.kernels_per_group, self.blocks_per_group
        )

        output = input
        # --- initial block
        for _ in range(self.n_initial_depth_convs):
            output = SingleConv(
                features=self.channels_per_group[0],
                kernel=1,
                use_layer_norm=self.use_layer_norm,
                activation=self.activation,
            )(output)
        # ---

        for i, (num_channels, kernel, num_blocks) in enumerate(channels_blocks_strategies):

            for _ in range(num_blocks):
                output = ConvNeXtBlock(
                    features=num_channels,
                    kernel=kernel,
                    use_layer_norm=self.use_layer_norm,
                    activation=self.activation
                )(output)
            
            if self.downsampling and i<2:
                output = DownsamplingLayer(self.channels_per_group[i+1], use_layer_norm=self.use_layer_norm)(output)
            elif i < len(self.channels_per_group) - 1 and num_channels != self.channels_per_group[i+1]:
                output = SingleConv(
                    features=self.channels_per_group[i+1],
                    kernel=1,
                    use_layer_norm=self.use_layer_norm,
                    activation=self.activation,
                )(output)

        return output