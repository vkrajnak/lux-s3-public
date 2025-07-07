import jax.numpy as jnp
import chex
import flax.linen as nn


class EmbeddingTileType(nn.Module):
    
    features: int = 6
    num_embeddings: int = 4  # -1 unseen, 0 empty, 1 nebula, 2 asteroid
    
    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        shifted_input = x + 1 # Embed takes values [0, num_embeddings)        
        embeddings = nn.Embed(self.num_embeddings, self.features)(shifted_input)

        return embeddings.reshape(x.shape[:-1] + (-1,))
    

class MergeTileType(nn.Module):
    features: int = 6
        
    @nn.compact
    def __call__(self, continuous_fields: chex.Array, tile_type_field: chex.Array) -> chex.Array:
        embeddings = EmbeddingTileType(self.features)(tile_type_field)
        merge = jnp.concatenate([continuous_fields, embeddings], axis=-1, dtype=continuous_fields.dtype)

        return merge
    
