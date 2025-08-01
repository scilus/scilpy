from typing import Tuple, Type

from dipy.utils.optpkg import optional_package

from scilpy.ml.utils import IMPORT_ERROR_MSG

torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)

""" The classes in this file are from the SAM-Med3D repository:
    https://github.com/uni-medical/SAM-Med3D
"""


class MLPBlock3D(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[torch.nn.Module] = torch.nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = torch.nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Attention(torch.nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.

    The forward function has been modified to use
    `scaled_dot_product_attention` instead of computing the attention
    manually.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads

        self.q_proj = torch.nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = torch.nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = torch.nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = torch.nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:

        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # # Get output
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class TwoWayAttentionBlock3D(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to
        sparse inputs.

        From https://github.com/uni-medical/SAM-Med3D

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (torch.nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock3D(embedding_dim, mlp_dim, activation)
        self.norm3 = torch.nn.LayerNorm(embedding_dim)

        self.norm4 = torch.nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys

        attn_out = self.cross_attn_token_to_image(
            q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys
        attn_out = self.cross_attn_image_to_token(
            q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys
