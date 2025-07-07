# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------ AL: Copied and adapted from Jumanji specs

import abc
import copy
import functools
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    NamedTuple,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import chex
import jax
import jax.numpy as jnp


def get_valid_dtype(dtype: Union[jnp.dtype, type]) -> jnp.dtype:
    """Cast a dtype taking into account the user type precision. E.g., if 64 bit is not enabled,
    jnp.dtype(jnp.float_) is still float64. By passing the given dtype through `jnp.empty` we get
    the supported dtype of float32.

    Args:
        dtype: jax numpy dtype or string specifying the array dtype.

    Returns:
        dtype converted to the correct type precision.
    """
    return jnp.empty((), dtype).dtype  # type: ignore


T = TypeVar("T")


class Spec(abc.ABC, Generic[T]):
    """Adapted from `dm_env.spec.Array`. This is an augmentation of the `Array` spec to allow for
    nested specs. `self.name`, `self.generate_value` and `self.validate` methods are adapted from
    the `dm_env` object."""

    def __init__(
        self,
        constructor: Union[Type[T], Callable[..., T]],
        name: str = "",
        **specs: "Spec",
    ):
        """Initializes a new spec.

        Args:
            constructor: the class or initialization function that creates the object represented
                by the spec.
            name: string containing a semantic name for the corresponding (nested) spec.
                Defaults to `''`.
            **specs: potential children specs each of which is either a nested spec or a primitive
                spec (`Array`, `BoundedArray`, etc). Importantly, the keywords used must exactly
                match the attribute names of the constructor.
        """
        self._name = name
        self._specs = specs
        self._constructor = constructor

        for spec_name, spec_value in specs.items():
            setattr(self, spec_name, spec_value)

    def __repr__(self) -> str:
        if self._specs.items():
            s = ""
            for spec_name, spec_value in self._specs.items():
                s += f"\t{spec_name}={spec_value},\n"
            return f"{self.name}(\n" + s + ")"
        return self.name

    @property
    def name(self) -> str:
        """Returns the name of the nested spec."""
        return self._name

    def validate(self, value: T) -> T:
        """Checks if a (potentially nested) value (tree of observations, actions...) conforms to
        this spec.

        Args:
            value: a (potentially nested) structure of jax arrays.

        Returns:
            value.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        if isinstance(value, tuple) and hasattr(value, "_asdict"):
            val = value._asdict()
        elif hasattr(value, "__dict__"):
            val = value.__dict__
        else:
            raise TypeError("The value provided must be a named tuple or a dataclass.")
        constructor_kwargs = jax.tree_util.tree_map(
            lambda spec, obs: spec.validate(obs), dict(self._specs), val
        )
        return self._constructor(**constructor_kwargs)

    def generate_value(self) -> T:
        """Generate a value which conforms to this spec."""
        constructor_kwargs = jax.tree_util.tree_map(lambda spec: spec.generate_value(), self._specs)
        return self._constructor(**constructor_kwargs)

    def replace(self, **kwargs: Any) -> "Spec":
        """Returns a new copy of `self` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `self`.
        """
        dict_copy = copy.deepcopy(self._specs)
        dict_copy.update(kwargs)
        return Spec(self._constructor, self.name, **dict_copy)

    def __eq__(self, other: "Spec") -> bool:  # type: ignore[override]
        return NotImplemented

    def __getitem__(self, item: str) -> "Spec":
        return self._specs[item]


class Array(Spec[chex.Array]):
    """Describes a jax array spec. This is adapted from `dm_env.specs.Array` for Jax environments.

    An `Array` spec allows an API to describe the arrays that it accepts or returns, before that
    array exists.
    """

    def __init__(self, shape: Iterable, dtype: Union[jnp.dtype, type], name: str = ""):
        """Initializes a new `Array` spec.

        Args:
            shape: an iterable specifying the array shape.
            dtype: jax numpy dtype or string specifying the array dtype.
            name: string containing a semantic name for the corresponding array. Defaults to `''`.
        """
        self._constructor = lambda: jnp.zeros(shape, dtype)
        super().__init__(constructor=self._constructor, name=name)
        self._shape = tuple(int(dim) for dim in shape)
        self._dtype = get_valid_dtype(dtype)

    def __repr__(self) -> str:
        return f"Array(shape={self.shape!r}, dtype={self.dtype!r}, name={self.name!r})"

    def __reduce__(self) -> Any:
        """To allow pickle to serialize the spec."""
        return Array, (self._shape, self._dtype, self._name)

    @property
    def shape(self) -> Tuple:
        """Returns a `tuple` specifying the array shape."""
        return self._shape

    @property
    def dtype(self) -> jnp.dtype:
        """Returns a jax numpy dtype specifying the array dtype."""
        return self._dtype

    def _fail_validation(self, message: str) -> None:
        if self.name:
            message += f" for spec {self.name}."
        else:
            message += "."
        raise ValueError(message)

    def validate(self, value: chex.Numeric) -> chex.Array:
        """Checks if value conforms to this spec.

        Args:
            value: a jax array or value convertible to one via `jnp.asarray`.

        Returns:
            value, converted if necessary to a jax array.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        value = jnp.asarray(value)
        if value.shape != self.shape:
            self._fail_validation(f"Expected shape {self.shape} but found {value.shape}")
        if value.dtype != self.dtype:
            self._fail_validation(f"Expected dtype {self.dtype} but found {value.dtype}")
        return value

    def _get_constructor_kwargs(self) -> Dict[str, Any]:
        """Returns constructor kwargs for instantiating a new copy of this spec."""
        # Get the names and kinds of the constructor parameters.
        params = inspect.signature(functools.partial(type(self).__init__, self)).parameters
        # __init__ must not accept *args or **kwargs, since otherwise we won't be
        # able to infer what the corresponding attribute names are.
        kinds = {value.kind for value in params.values()}
        if inspect.Parameter.VAR_POSITIONAL in kinds:
            raise TypeError("specs.Array subclasses must not accept *args.")
        elif inspect.Parameter.VAR_KEYWORD in kinds:
            raise TypeError("specs.Array subclasses must not accept **kwargs.")
        # Note that we assume direct correspondence between the names of constructor
        # arguments and attributes.
        return {name: getattr(self, name) for name in params.keys()}

    def replace(self, **kwargs: Any) -> "Array":
        """Returns a new copy of `self` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `self`.
        """
        all_kwargs = self._get_constructor_kwargs()
        all_kwargs.update(kwargs)
        return type(self)(**all_kwargs)

    def __eq__(self, other: "Array") -> bool:  # type: ignore[override]
        if not isinstance(other, Array):
            return NotImplemented
        return (
            (self.shape == other.shape)
            and (self.dtype == other.dtype)
            and (self.name == other.name)
        )