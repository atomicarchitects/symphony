from typing import Optional
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import haiku as hk


class MultiHeadAttention(hk.Module):
    """Multi-headed attention (MHA) module.

    This module is intended for attending over sequences of IrrepsArrays.

    Rough sketch:
    - Compute keys (K), queries (Q), and values (V) as projections of inputs.
    - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
    - Output is another projection of WV^T.

    For more detail, see the original Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762.

    Glossary of shapes:
    - T: Sequence length.
    - D: Vector (embedding) size.
    - H: Number of attention heads.
    """

    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        name: Optional[str] = None,
    ):
        """Initialises the module.

        Args:
          num_heads: Number of independent attention heads (H).
          num_channels: The number of channels in each head to determine the embedding size (D).
          name: Optional name for this module.
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.num_channels = num_channels

    def __call__(
        self,
        query: e3nn.IrrepsArray,
        key: e3nn.IrrepsArray,
        value: e3nn.IrrepsArray,
        mask: Optional[jnp.ndarray] = None,
    ) -> e3nn.IrrepsArray:
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.

        Args:
          query: Embeddings sequence used to compute queries; shape [..., T', D_q].
          key: Embeddings sequence used to compute keys; shape [..., T, D_k].
          value: Embeddings sequence used to compute values; shape [..., T, D_v].
          mask: Optional mask applied to attention weights; shape [..., H=1, T', T].

        Returns:
          A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        irreps = query.irreps
        projection = self._linear_projection

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = projection(
            query, self.num_heads, self.num_channels, "query"
        )  # [T', H, Q]
        key_heads = projection(
            key, self.num_heads, self.num_channels, "key"
        )  # [T, H, K]
        value_heads = projection(
            value, self.num_heads, self.num_channels, "value"
        )  # [T, H, V]

        # Compute attention weights.
        attn_logits = jnp.einsum(
            "...thd,...Thd->...htT", query_heads.array, key_heads.array
        )
        attn_logits = attn_logits / jnp.sqrt(query_heads.shape[-1])
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads.array)
        attn = e3nn.IrrepsArray(self.num_channels * irreps, attn)  # [T', H, V]
        attn = attn.axis_to_mul(axis=-2)  # [T', H * V]

        # Apply another projection to get the final embeddings.
        return e3nn.haiku.Linear(self.num_channels * irreps)(attn)  # [T', D']

    @hk.transparent
    def _linear_projection(
        self,
        x: e3nn.IrrepsArray,
        num_heads: int,
        num_channels: int,
        name: Optional[str] = None,
    ) -> jnp.ndarray:
        y = e3nn.haiku.Linear(num_channels * num_heads * x.irreps, name=name)(x)
        y = y.mul_to_axis(num_heads, axis=-2)
        return y
