from typing import Tuple, Type

from scilpy.ml.utils import IMPORT_ERROR_MSG
from scilpy.ml.bundleparc.encodings import PositionalEncodingPermute3D

from dipy.utils.optpkg import optional_package
torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)


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
        # B x N_heads x N_tokens x C_per_head
        return x.transpose(1, 2).contiguous()

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2).contiguous()
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

        # Attention
        # _, _, _, c_per_head = q.shape
        # attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        # attn = attn / math.sqrt(c_per_head)
        # attn = torch.softmax(attn, dim=-1)

        # # Get output
        # out = attn @ v
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

        From SAM-Med3D: https://github.com/uni-medical/SAM-Med3D

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

        q = queries + query_pe
        k = keys

        # Cross attention block, image embedding attending to tokens
        attn_out = self.cross_attn_image_to_token(
            q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, tokens attending to image embedding
        attn_out = self.cross_attn_token_to_image(
            q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        return queries, keys


class ConvNextBlock(torch.nn.Module):
    """ MedNeXt convolutional block with a convolutional layer, group
    normalization layer and GELU activation.
    """

    def __init__(self, in_chans: int, ratio: int = 1):
        super().__init__()

        self.conv1 = torch.nn.Conv3d(
            in_chans, in_chans, kernel_size=3, padding=1, stride=1,
            groups=in_chans)
        self.gn1 = torch.nn.GroupNorm(in_chans, in_chans)

        self.conv2 = torch.nn.Conv3d(
            in_chans, in_chans * ratio, kernel_size=1, stride=1)
        self.gelu1 = torch.nn.GELU()

        self.conv3 = torch.nn.Conv3d(
            in_chans * ratio, in_chans, kernel_size=1, stride=1)

        # Use Xavier initialisation for weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d) or \
               isinstance(m, torch.nn.ConvTranspose3d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.conv2(x)
        x = self.gelu1(x)
        x = self.conv3(x)
        return x + res


class DownsampleNextBlock(torch.nn.Module):
    """ Downsample block with a convolutional layer, group normalization
    layer and GELU activation. Identical to the convolutional block but
    with a stride of 2.
    """

    def __init__(self, in_chans, out_chans, ratio: int = 1):
        super().__init__()

        self.conv1 = torch.nn.Conv3d(
            in_chans, in_chans, kernel_size=3, padding=1, stride=2,
            groups=in_chans)
        self.gn1 = torch.nn.GroupNorm(in_chans, in_chans)

        self.conv2 = torch.nn.Conv3d(
            in_chans, in_chans * ratio, kernel_size=1, stride=1)
        self.gelu1 = torch.nn.GELU()

        self.conv3 = torch.nn.Conv3d(
            in_chans * ratio, out_chans, kernel_size=1, stride=1)

        self.resconv = torch.nn.Conv3d(
            in_chans, out_chans, kernel_size=1, stride=2)

        # Use Xavier initialisation for weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d) or \
               isinstance(m, torch.nn.ConvTranspose3d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        res = self.resconv(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.conv2(x)
        x = self.gelu1(x)
        x = self.conv3(x)
        return x + res


class EncoderNextLayer(torch.nn.ModuleList):
    """ Encoder layer with two convolutional blocks. """

    def __init__(self, in_chans: int, out_chans: int, ratio):
        super().__init__()

        self.conv1 = ConvNextBlock(in_chans, ratio)
        self.conv2 = ConvNextBlock(in_chans, ratio)
        self.down = DownsampleNextBlock(in_chans, out_chans, ratio)

    def forward(self, x):
        """ Forward pass of the encoder layer. Return
        the residual connection and the downsampled feature map.
        """

        x = self.conv1(x)
        x_res = self.conv2(x)
        x = self.down(x_res)

        return x, x_res


class UNextEncoder(torch.nn.Module):
    """ MedNeXt encoder with 4 encoder layers. """

    def __init__(self, channels=[32, 64, 128, 256, 512]):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            EncoderNextLayer(channels[0], channels[1], ratio=2),
            EncoderNextLayer(channels[1], channels[2], ratio=3),
            EncoderNextLayer(channels[2], channels[3], ratio=4),
            EncoderNextLayer(channels[3], channels[4], ratio=4),
        ])


class UpsampleNextBlock(torch.nn.Module):
    """ Upsamping block with a convolutional layer and a group normalization
    layer. """

    def __init__(self, in_chans, out_chans, ratio: int = 1):
        super().__init__()

        self.conv1 = torch.nn.ConvTranspose3d(
            in_chans, in_chans, kernel_size=2, padding=0, stride=2,
            groups=in_chans)

        self.gn1 = torch.nn.GroupNorm(in_chans, in_chans)

        self.conv2 = torch.nn.Conv3d(
            in_chans, in_chans * ratio, kernel_size=1, stride=1)
        self.gelu1 = torch.nn.GELU()

        self.conv3 = torch.nn.Conv3d(
            in_chans * ratio, out_chans, kernel_size=1, stride=1)

        self.resconv = torch.nn.ConvTranspose3d(
            in_chans, out_chans, kernel_size=2, stride=2)

        # Use Xavier initialisation for weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d) or \
               isinstance(m, torch.nn.ConvTranspose3d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        res = self.resconv(x)

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.conv2(x)
        x = self.gelu1(x)
        x = self.conv3(x)

        return x + res


class DecoderNextLayer(torch.nn.Module):
    """ Decoder layer. Includes upsampling, attention blocks and
    a deep supervision head.
    """

    def __init__(
        self, in_chans: int, out_chans: int, ratio: int = 1
    ):
        super().__init__()

        self.upsample = UpsampleNextBlock(
            in_chans, out_chans, ratio)
        self.conv1 = ConvNextBlock(out_chans, ratio)
        self.conv2 = ConvNextBlock(out_chans, ratio)

        self.prompt_encoding = torch.nn.Sequential(
            torch.nn.Linear(in_chans, out_chans), torch.nn.GELU())

        self.tkn2img = TwoWayAttentionBlock3D(
            out_chans, 4, mlp_dim=out_chans * ratio)
        self.pe_layer = PositionalEncodingPermute3D(out_chans)

    def _decode(self, z, encoder_feature):
        z = self.upsample(z)
        z = z + encoder_feature
        z = self.conv1(z)
        return z

    def _prompt_attn(self, z, prompt_encoding):
        B, C, X, Y, Z = z.shape
        prompt_encoding = prompt_encoding[:, None, :]
        pe = self.pe_layer(z)

        image_embedding = (z).flatten(2).permute(0, 2, 1).contiguous()
        pe = pe.flatten(2).permute(0, 2, 1).contiguous()

        z, prompt_encoding = self.tkn2img(
            image_embedding, prompt_encoding, pe)

        z = z.permute(0, 2, 1).reshape((B, C, X, Y, Z)).contiguous()
        prompt_encoding = prompt_encoding[:, 0, :]

        return z, prompt_encoding

    def forward(self, z, encoder_feature, prompt_encoding):
        z = self._decode(z, encoder_feature)
        prompt_encoding = self.prompt_encoding(prompt_encoding)
        z, prompt_encoding = self._prompt_attn(
            z, prompt_encoding)
        z = self.conv2(z)  # maybe ?
        return z, prompt_encoding


class BundleParcNetDecoder(torch.nn.Module):
    """ MedNeXt decoder with 4 decoder layers. """

    def __init__(
        self, channels=[512, 256, 128, 64, 32]
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            DecoderNextLayer(channels[0], channels[1], ratio=4),
            DecoderNextLayer(channels[1], channels[2], ratio=4),
            DecoderNextLayer(channels[2], channels[3], ratio=3),
            DecoderNextLayer(channels[3], channels[4], ratio=2)
        ])

    def forward(self, x, embeddings, prompt_encoding):

        for decoder_layer, encoder_feature in zip(
            self.layers, embeddings
        ):
            x, prompt_encoding = \
                decoder_layer(
                    x, encoder_feature, prompt_encoding)

        return x


class Stem(torch.nn.Module):
    """ 3D Convolutional block with a convolutional layer, batch normalization
    and ReLU activation. """

    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()

        self.conv = torch.nn.Conv3d(
            in_chans, out_chans, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Head(torch.nn.Module):

    def __init__(self, in_chans: int):
        super().__init__()

        self.conv = torch.nn.Conv3d(
            in_chans, 2, kernel_size=1, stride=1)
        # self.act = torch.nn.Sigmoid()

        # Use Xavier initialisation for weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d) or \
               isinstance(m, torch.nn.ConvTranspose3d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv(x)
        # x = self.act(x)
        return x


class BundleParcNet(torch.nn.Module):

    def __init__(self, in_chans, volume_size=128,
                 channels=[32, 64, 128, 256, 512], n_bundles=71):
        super().__init__()

        self.channels = channels

        # Important to store this for loading the model later.
        self.in_chans = in_chans
        self.volume_size = volume_size
        self.embed_dim = self.channels[0]
        self.bottleneck_dim = self.channels[-1]
        self.n_bundles = n_bundles

        # Define the input branch
        self.stem = torch.nn.Conv3d(
            in_chans, self.embed_dim, kernel_size=1, stride=1)

        # Define the output head
        self.head = Head(self.channels[0])

        # Define the model
        self.encoder = UNextEncoder(self.channels)
        self.bottleneck = ConvNextBlock(self.bottleneck_dim, ratio=4)
        self.decoder = BundleParcNetDecoder(channels=self.channels[::-1])

        self.prompt_embedding = torch.nn.Sequential(
            torch.nn.Linear(n_bundles, self.bottleneck_dim), torch.nn.GELU())

    def forward(self, fodf, bundle_prompt):
        """ Forward pass of the model. """

        # B, C, X, Y, Z = fodf.shape

        z, encoder_features = self.encode(fodf)
        y_hat = self.decode(z, encoder_features, bundle_prompt)

        return y_hat

    def encode(self, fodf):
        """ Forward pass of the model's encoder. """
        # B, C, X, Y, Z = fodf.shape

        # Embed the input fodf
        x = self.stem(fodf)

        # Run the encoders for the input fodf
        encoder_features = []
        for encoder_layer in self.encoder.layers:
            x, x_res = encoder_layer(x)
            encoder_features.insert(0, x_res)

        # As opposed to the original MedNeXt, we do not use deep
        # supervision here, as the bottleneck does not receive any
        # prompt information and therefore cannot know which
        # bundle to predict
        z = self.bottleneck(x)

        return z, encoder_features

    def decode(self, z, encoder_features, bundle_prompt):
        """ Forward pass of the model's decoder. """
        # Embed the bundle prompt to the same dimension as the input fodf
        prompt_embed = self.prompt_embedding(bundle_prompt)

        # Run the decoder
        # Decoder layers are run in reverse order to match the
        # encoder features.
        z = self.decoder(
            z, encoder_features, prompt_embed)

        y_hat = self.head(z)

        return y_hat
