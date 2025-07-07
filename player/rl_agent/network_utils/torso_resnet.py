import sys
from typing import Sequence, Union

import chex
import flax.linen as nn

from .blocks_base import MLPBlock
from .blocks_resnet import DownsamplingStrategy, make_downsampling_layer, ResidualBlock, BottleneckBlock, SingleConv
from .blocks_embedding import MergeTileType
from ..network_input.types import NNInput


class DownsamplingResNet(nn.Module):
    """ResNetTorso for visual inputs, inspired by the IMPALA paper. Inputs must be channel-last: [..., B, H, W, C]
    Downsampling is applied after every residual block"""

    output_type: str
    channels_per_group: Sequence[int] = (16, 32, 32)
    blocks_per_group: Sequence[int] = (2, 2, 2)
    kernels_per_group: Sequence[int] = (3, 3, 3)
    downsampling_strategies: Sequence[DownsamplingStrategy] = (DownsamplingStrategy.CONV,) * 3
    mlp_layer_sizes: Sequence[int] = None
    use_layer_norm: bool = False
    activation: str = "relu"

    @nn.compact
    def __call__(self, nn_input: NNInput) -> chex.Array:
        observation = nn_input.continuous_fields

        if observation.ndim > 4:
            return nn.batch_apply.BatchApply(self.__call__)(observation)

        assert (
                observation.ndim == 4
        ), f"Expected inputs to have shape [B, H, W, C] but got shape {observation.shape}."

        output = observation
        channels_blocks_strategies = zip(
            self.channels_per_group, self.kernels_per_group, self.blocks_per_group, self.downsampling_strategies
        )

        for _, (num_channels, kernel, num_blocks, strategy) in enumerate(channels_blocks_strategies):
            output = make_downsampling_layer(strategy, num_channels)(output)

            for _ in range(num_blocks):
                output = ResidualBlock(
                    features=num_channels,
                    kernel=kernel,
                    use_layer_norm=self.use_layer_norm,
                    activation=self.activation,
                )(output)

        # Flatten
        if self.output_type == "flat":
            output = output.reshape(*observation.shape[:-3], -1)
            if self.mlp_layer_sizes is not None:
                output = MLPBlock(
                    layer_sizes=self.mlp_layer_sizes,
                    activation=self.activation,
                    use_layer_norm=self.use_layer_norm,
                    activate_final=True,
                )(output)
        else:
            assert self.mlp_layer_sizes is None

        return output


class CustomResNet(nn.Module):
    """ResNet following https://arxiv.org/abs/1512.03385v1
    except for initial 7x7 conv and max pooling replaced by a 1x1 conv
    first group has downsampling and bottleneck diabled"""

    # output_type: str
    channels_per_group: Sequence[int]
    blocks_per_group: Sequence[int]
    kernels_per_group: Sequence[int]
    downsampling_first_of_group: bool = False  # except for first group
    bottleneck_reduction_factor: int = 4
    # mlp_layer_sizes: Sequence[int] = None
    use_layer_norm: bool = False
    activation: str = "relu"
    bottleneck: bool = True  # except for first group
    n_initial_depth_convs: int = 0
    se_layer: bool = False
    se_factor: int = 4

    @nn.compact
    def __call__(self, nn_input: NNInput) -> chex.Array:

        # MOVE_TO_NECK
        # features_output = get_features_output(self.output_type, self.features_output)

        input = MergeTileType(4)(nn_input.continuous_fields, nn_input.tile_type_fields)  # (..., 24, 24, n)
        # input_shape = input.shape  # (..., 24, 24, n)

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

            downsampling_group = [i * self.downsampling_first_of_group] + [False] * (num_blocks - 1)

            for downsampling in downsampling_group:
                if i * self.bottleneck:
                    output = BottleneckBlock(
                        features=num_channels,
                        kernel=kernel,
                        reduction_factor=self.bottleneck_reduction_factor,
                        use_layer_norm=self.use_layer_norm,
                        activation=self.activation,
                        downsampling=downsampling,
                        se_layer=self.se_layer,
                        se_factor=self.se_factor,
                    )(output)
                else:
                    output = ResidualBlock(
                        features=num_channels,
                        kernel=kernel,
                        use_layer_norm=self.use_layer_norm,
                        activation=self.activation,
                        downsampling=downsampling,
                        se_layer=self.se_layer,
                        se_factor=self.se_factor,
                    )(output)

        # MOVED TO NECK
        # # --- final block
        # # alternatively consider a single 3x3 conv to reduce size to 4x4x256
        # for _ in range(self.n_final_depth_convs):
        #     output = SingleConv(
        #         features=self.channels_per_group[-1]//2,
        #         kernel=self.kernel_final_convs,
        #         use_layer_norm=self.use_layer_norm,
        #         activation=self.activation,
        #         padding='VALID'
        #     )(output)
        #
        # output = SingleConv(
        #         features=features_output,
        #         kernel=1,
        #         use_layer_norm=self.use_layer_norm,
        #         activation=self.activation,
        #     )(output)
        # # ---
        #
        # # Flatten
        # if self.output_type == "flat":
        #     output = output.reshape(*input_shape[:-3], -1)
        #
        #     if self.mlp_layer_sizes is not None:
        #         output = MLPBlock(
        #             layer_sizes=self.mlp_layer_sizes,
        #             activation=self.activation,
        #             use_layer_norm=self.use_layer_norm,
        #             activate_final=True,
        #         )(output)
        #
        # else:
        #     assert self.mlp_layer_sizes is None
        #
        # output = check_and_reshape(self.output_type, output, input_shape)

        return output
