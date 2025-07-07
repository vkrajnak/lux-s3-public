import jax
import jax.numpy as jnp
from jax import lax
from ..constants import GRID_SHAPE

map_size = GRID_SHAPE[0]

energy_node_fns = jnp.array(
    [
        [0, 1.2, 1, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1.2, 1, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

energy_nodes_mask = jnp.zeros((6), dtype=jnp.bool)
energy_nodes_mask = energy_nodes_mask.at[0].set(True)
energy_nodes_mask = energy_nodes_mask.at[3].set(True)

ENERGY_NODE_FNS = [
    lambda d, x, y, z: jnp.sin(d * x + y) * z, lambda d, x, y, z: (x / (d + 1) + y) * z
]


def _compute_energy_field(node_fn_spec, distances_to_node, mask):
    fn_i, x, y, z = node_fn_spec
    return jnp.where(
        mask,
        lax.switch(
            fn_i.astype(jnp.int16), ENERGY_NODE_FNS, distances_to_node, x, y, z
        ),
        jnp.zeros_like(distances_to_node),
    )
mm = jnp.meshgrid(jnp.arange(map_size), jnp.arange(map_size))
coordinate_map = jnp.stack([mm[0], mm[1]]).T.astype(jnp.int16)


def distances_to_points(points, dist_f):
    """returns a distance map for each of the points"""
    """if points is of snape (k,), distances_to_nodes is of shape (k,n,n)"""
    distances_to_nodes = dist_f(jnp.abs(coordinate_map[None, :, :, :] - points[:, None, None, :]), axis=-1)
    return distances_to_nodes


def _energy_field(energy_nodes):
    distances_to_nodes = distances_to_points(energy_nodes, jnp.linalg.norm)
    energy_field = jax.vmap(_compute_energy_field)(
        energy_node_fns, distances_to_nodes, energy_nodes_mask
    )
    energy_field = jnp.where(
        energy_field.mean() < 0.25,
        energy_field + (0.25 - energy_field.mean()),
        energy_field,
    )
    energy_field = jnp.round(energy_field.sum(0)).astype(jnp.int16)
    energy_field = jnp.clip(
        energy_field, -20, 20
    )
    return energy_field


def _energy_fields():
    x, y = jnp.triu_indices(map_size)
    energy_nodes = jnp.vstack((x, map_size - 1 - y)).astype(jnp.int16)
    energy_nodes = energy_nodes.reshape((2, -1)).T
    energy_nodes_sym = map_size - 1 - energy_nodes[:, ::-1]
    zeros = jnp.zeros(energy_nodes.shape, dtype=jnp.int16)
    energy_nodes_all = jnp.stack((energy_nodes, zeros, zeros, energy_nodes_sym, zeros, zeros), axis=1)
    energy_fields = jax.vmap(_energy_field)(energy_nodes_all)

    return energy_fields


energy_fields = _energy_fields()
