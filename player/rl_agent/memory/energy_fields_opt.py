import jax
import jax.numpy as jnp
from jax import lax
from ..constants import GRID_SHAPE

map_size = GRID_SHAPE[0]
half_map_size = map_size // 2

energy_node_fns = jnp.array(
    [
        [0, 1.2, 1, 4],
        [0, 1.2, 1, 4],
    ]
)

ENERGY_NODE_FNS = [
    lambda d, x, y, z: jnp.sin(d * x + y) * z, lambda d, x, y, z: (x / (d + 1) + y) * z
]
ENERGY_NODE_FN0 = lambda d, x, y, z: jnp.sin(d * x + y) * z

def _compute_energy_field(node_fn_spec, distances_to_node):
    fn_i, x, y, z = node_fn_spec
    return ENERGY_NODE_FN0(distances_to_node, x, y, z)

mm = jnp.meshgrid(jnp.arange(map_size), jnp.arange(map_size))
coordinate_map = jnp.stack([mm[0], mm[1]]).T.astype(jnp.int16)

def coords_distances_to_points(points, dist_f):
    """returns a relative_coordinate_maps and a distance map for each of the points"""
    """if points is of snape (k,), distances_to_nodes is of shape (k,n,n)"""
    relative_coordinate_maps = coordinate_map[None,:,:,:] - points[:,None, None,:]
    distances_to_nodes = dist_f(jnp.abs(relative_coordinate_maps), axis=-1)
    return relative_coordinate_maps, distances_to_nodes

def distances_to_points(points, dist_f):
    """returns a distance map for each of the points"""
    """if points is of snape (k,), distances_to_nodes is of shape (k,n,n)"""
    distances_to_nodes = dist_f(jnp.abs(coordinate_map[None,:,:,:] - points[:,None, None,:]), axis=-1)
    return distances_to_nodes

def _distances_to_point(point, dist_f):
    """returns a distance map for each of the points"""
    """if points is of snape (k,), distances_to_nodes is of shape (k,n,n)"""
    distances_to_nodes = dist_f(jnp.abs(coordinate_map - point[None, None,:]), axis=-1)
    return distances_to_nodes

def _energy_field(energy_node):
    distances_to_node = _distances_to_point(energy_node, jnp.linalg.norm)
    energy_field = _compute_energy_field(energy_node_fns[0], distances_to_node)

    energy_field = jnp.where(
        energy_field.mean() < 0.25,
        energy_field + (0.25 - energy_field.mean()),
        energy_field,
    )
    energy_field = jnp.round(energy_field + energy_field[::-1,::-1].T + 1).astype(jnp.int16)
    return energy_field


def _energy_fields():
    
    x, y = jnp.triu_indices(map_size)
    energy_nodes = jnp.vstack((x, map_size-1-y)).astype(jnp.int16)
    energy_nodes = energy_nodes.reshape((2,-1)).T
    energy_fields = jax.vmap(_energy_field)(energy_nodes)
    energy_fields = jnp.clip(energy_fields, -20, 20)

    return energy_fields

energy_fields = _energy_fields()
energy_fields_compare = energy_fields[:,:half_map_size,:half_map_size]