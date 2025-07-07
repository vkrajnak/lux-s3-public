

import sys
from typing import Sequence, Union

import chex
import flax.linen as nn
import jax.numpy as jnp

from .blocks_base import MLPBlock
from .blocks_resnet import SingleConv
from .blocks_convnext import DownsamplingLayer
from .utils import parse_activation_fn

from ..constants import GRID_SHAPE, N_MAX_UNITS, N_BASE_ACTIONS, N_SAP_ACTIONS


def get_neck_properties(torso_name, torso_kwargs, head_name):

    neck_kwargs = {}
    neck_kwargs["head_name"] = head_name
    if "activation" in torso_kwargs:
        neck_kwargs["activation"] = torso_kwargs["activation"]
    if "use_layer_norm" in torso_kwargs:
        neck_kwargs["use_layer_norm"] = torso_kwargs["use_layer_norm"]

    if head_name == "discrete_and_field":
        neck_fn = DoubleNeck
    else:
        neck_fn = SingleNeck

    if head_name == "discrete_full":
        neck_kwargs["features_output"] = 128
        neck_kwargs["mlp_layer_sizes"] = [128,]

    elif head_name == "discrete_sparse":
        neck_kwargs["features_output"] = N_BASE_ACTIONS + N_SAP_ACTIONS

    elif head_name == "monofield" or head_name == "duomofield":
        neck_kwargs["features_output"] = N_MAX_UNITS

    elif head_name == "duofield":
        neck_kwargs["features_output"] = 2*N_MAX_UNITS

    elif head_name == "discrete_and_field":
        neck_kwargs["features_output"] = (N_BASE_ACTIONS, N_MAX_UNITS)
        neck_kwargs["use_dense"] = True

    elif head_name == "critic":
        neck_kwargs["features_output"] = 32
        neck_kwargs["mlp_layer_sizes"] = [64,]

    else:
        raise Exception("This head_name is not known")

    return neck_fn, neck_kwargs


def check_and_reshape(head_name, output, input_shape=None):
    if head_name == "critic" or head_name == "discrete_full":
        pass
    elif head_name == "discrete_sparse":
        output = output.reshape(*input_shape[:-3], N_MAX_UNITS, N_BASE_ACTIONS + N_SAP_ACTIONS)
        assert output.shape[-2:] == (N_MAX_UNITS, N_BASE_ACTIONS + N_SAP_ACTIONS)
    elif head_name == "monofield" or head_name == 'duomofield':
        assert output.shape[-3:] == GRID_SHAPE + (N_MAX_UNITS, )
    elif head_name == "duofield":
        assert output.shape[-3:] == GRID_SHAPE + (2*N_MAX_UNITS, )
    elif head_name == "discrete_and_field":
        output_discrete = output[0]
        output_field = output[1]
        assert output_discrete.shape[-2:] == (N_MAX_UNITS, N_BASE_ACTIONS)
        assert output_field.shape[-3:] == GRID_SHAPE + (N_MAX_UNITS,)
    else:
        raise Exception("This head_name is not known")
    return output


class SingleNeck(nn.Module):
    head_name: str
    features_output: Union[int, tuple[int]]
    mlp_layer_sizes: Sequence[int] = None
    use_layer_norm: bool = False
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:

        shape = x.shape
        output = x

        # Downsampling (actor only)
        if self.head_name in ("discrete_sparse", "discrete_full"):
            output = SingleConv(
                features=shape[-1],  #self.channels_per_group[-1]//2 -> handled automatically in network.py
                kernel=3,
                use_layer_norm=self.use_layer_norm,
                activation=self.activation,
                padding='VALID',
            )(output)

        # Give the right shape (last layer if this is the actor - except discrete_full)
        output = SingleConv(
                features=self.features_output,
                kernel=1,
                use_layer_norm=self.use_layer_norm,
                activation=self.activation if self.head_name in ("critic", "discrete_full") else "none",
                kernel_scale=1.4142 if self.head_name in ("critic", "discrete_full") else 0.01,
            )(output)

        # Downsampling (critic only): goal is to go from (n,n,f) to (4,4,f)
        if self.head_name == "critic":
            if output.shape[-3:-1] == (24, 24):
                output = SingleConv(
                    features=self.features_output,
                    kernel=3,
                    stride=2,
                    use_layer_norm=self.use_layer_norm,
                    activation=self.activation,
                    padding='SAME'
                )(output)
                output = SingleConv(
                    features=self.features_output,
                    kernel=3,
                    stride=2,
                    use_layer_norm=self.use_layer_norm,
                    activation=self.activation,
                    padding='SAME'
                )(output)
                # output = nn.Conv(self.features_output, (3, 3), strides=(2, 2), padding='SAME')(output)
                # output = nn.Conv(self.features_output, (3, 3), strides=(2, 2), padding='SAME')(output)
            assert output.shape[-3:-1] == (6, 6)
            output = nn.avg_pool(output, window_shape=(3, 3), strides=(1, 1))
            assert output.shape[-3:-1] == (4, 4)


        # Flatten if needed
        if self.head_name == "critic" or self.head_name == "discrete_full":
            output = output.reshape(*shape[:-3], -1)

            if self.mlp_layer_sizes is not None:
                output = MLPBlock(
                    layer_sizes=self.mlp_layer_sizes,
                    activation=self.activation,
                    use_layer_norm=self.use_layer_norm,
                    activate_final=True,
                )(output)

        else:
            assert self.mlp_layer_sizes is None

        output = check_and_reshape(self.head_name, output, shape)

        return output


class DoubleNeck(nn.Module):
    head_name: str
    features_output: tuple[int, int]
    use_dense: bool
    use_layer_norm: bool = False
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: chex.Array) -> tuple[chex.Array, chex.Array]:

        if self.head_name != "discrete_and_field":
            raise Exception("DoubleNeck is only for head 'discrete_and_field'")

        shape = x.shape # (24, 24, f)
        output = x
        features_output_base_action_discrete, features_output_sap_action_field = self.features_output

        # --- sap action field, reduce channels to 16
        output_sap_action_field = SingleConv(
                features=features_output_sap_action_field,
                kernel=1,
                use_layer_norm=self.use_layer_norm,
                activation="none",
            )(output)
        
        # --- base action discrete, reduce dimension from 24x24 to 4x4

        for _ in range(2): # nxn -> (n/2)x(n/2)
            output = SingleConv(
                features=shape[-1],
                kernel=3,
                stride=2,
                use_layer_norm=self.use_layer_norm,
                activation=self.activation,
            )(output)
        # ->   # (6, 6, f)

        if not self.use_dense:
        
            output = SingleConv(
                    features=shape[-1],
                    kernel=3,
                    stride=1,
                    use_layer_norm=self.use_layer_norm,
                    activation=self.activation,
                    padding='VALID',
                )(output)  # -> (4, 4, f)

            output_base_action_discrete = SingleConv(
                    features=features_output_base_action_discrete,
                    kernel=1,
                    use_layer_norm=self.use_layer_norm,
                    activation="none",
                )(output)  # -> (4, 4, o)

            output_base_action_discrete = output_base_action_discrete.reshape(shape[:-3] + (N_MAX_UNITS, features_output_base_action_discrete))

        else:

            # output = nn.avg_pool(output, window_shape=(3, 3), strides=(1, 1))  # (4, 4, f)
            output = SingleConv(
                    features=shape[-1],
                    kernel=3,
                    stride=1,
                    use_layer_norm=self.use_layer_norm,
                    activation=self.activation,
                    padding='VALID',
                )(output)  # -> (4, 4, f)

            # reduce number of features if too large
            if output.shape[-1] >= 128:
                output = SingleConv(
                    features=shape[-1]//2,
                    kernel=1,
                    use_layer_norm=self.use_layer_norm,
                    activation=self.activation,
                )(output)  # -> (4, 4, f=f/2)

            output = output.reshape(*shape[:-3], -1)  # flatten (16*f)

            outputs = [MLPBlock(
                layer_sizes=[6,],
                activation=self.activation,
                use_layer_norm=self.use_layer_norm,
                activate_final=False,
            )(output) for _ in range(N_MAX_UNITS)]  # [ (6,), (6,), ... x16]

            output = jnp.stack(outputs, axis=-1)  # (6, 16)
            output_base_action_discrete = jnp.swapaxes(output, -1, -2)  # (16, 6)

        return output_base_action_discrete, output_sap_action_field  # (16, 6) # (24, 24, 16)

