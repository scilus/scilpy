from dipy.utils.optpkg import optional_package

from scilpy.ml.utils import IMPORT_ERROR_MSG
from scilpy.ml.bundleparc.encodings import PositionalEncodingPermute3D
from scilpy.ml.bundleparc.attention import TwoWayAttentionBlock3D


torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)


def init_block(block):
    """ Initialise the weights of convolutional layers using Xavier
    initialisation. """

    for m in block.modules():
        if isinstance(
            m, torch.nn.Conv3d) or \
            isinstance(
                m, torch.nn.ConvTranspose3d
        ):
            torch.nn.init.xavier_uniform_(m.weight)


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
        init_block(self)

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
        init_block(self)

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
        init_block(self)

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
        self, in_chans: int, out_chans: int, ratio: int = 1,
        prompt_strategy='add'
    ):
        super().__init__()

        self.prompt_strategy = prompt_strategy

        self.upsample = UpsampleNextBlock(
            in_chans, out_chans, ratio)
        self.conv1 = ConvNextBlock(out_chans, ratio)
        self.conv2 = ConvNextBlock(out_chans, ratio)

        self.prompt_encoding = torch.nn.Sequential(
            torch.nn.Linear(in_chans, out_chans), torch.nn.GELU())

        if self.prompt_strategy == 'attention':
            self.tkn2img = TwoWayAttentionBlock3D(
                out_chans, 4, mlp_dim=out_chans * ratio)
            self.pe_layer = PositionalEncodingPermute3D(out_chans)
            self._prompt_func = self._prompt_attn
        else:
            self.prompt_conv = ConvNextBlock(out_chans, ratio)
            self._prompt_func = self._prompt_add

        self.ds_head = Head(out_chans)

    def _decode(self, z, encoder_feature):
        """ Decode the input feature map.

        Parameters
        ----------
        z : torch.Tensor
            Input feature map.
        encoder_feature : torch.Tensor
            Feature map from the encoder.

        Returns
        -------
        z : torch.Tensor
            Decoded feature map.
        """

        z = self.upsample(z)
        z = z + encoder_feature
        z = self.conv1(z)
        return z

    def _prompt_add(self, z, prompt_encoding, dense_encoding):
        """ Mix the prompt and dense encodings. using addition.

        Parameters
        ----------
        z : torch.Tensor
            Input feature map.
        prompt_encoding : torch.Tensor
            Prompt encoding.
        dense_encoding : torch.Tensor
            Dense encoding.

        Returns
        -------
        z : torch.Tensor
            Feature map with the prompt and dense encodings added.
        prompt_encoding : torch.Tensor
            Updated prompt encoding.
        """

        z += dense_encoding
        z = self.prompt_conv(z)
        z += prompt_encoding[..., None, None, None]

        return z, prompt_encoding

    def _prompt_attn(self, z, prompt_encoding, dense_encoding):
        """ Perform attention on the prompt and dense encodings.

        Parameters
        ----------
        z : torch.Tensor
            Input feature map.
        prompt_encoding : torch.Tensor
            Prompt encoding.
        dense_encoding : torch.Tensor
            Dense encoding.

        Returns
        -------
        z : torch.Tensor
            Feature map with the prompt and dense encodings added.
        prompt_encoding : torch.Tensor
            Updated prompt encoding.
        """

        B, C, X, Y, Z = z.shape
        prompt_encoding = prompt_encoding[:, None, :]
        pe = self.pe_layer(z)

        image_embedding = (z + dense_encoding).flatten(2).permute(0, 2, 1)
        pe = pe.flatten(2).permute(0, 2, 1)

        z, prompt_encoding = self.tkn2img(
            image_embedding, prompt_encoding, pe)

        z = z.permute(0, 2, 1).reshape((B, C, X, Y, Z))
        prompt_encoding = prompt_encoding[:, 0, :]

        return z, prompt_encoding

    def forward(self, z, encoder_feature, prompt_encoding, dense_encoding):
        """ Forward pass of the decoder layers.
        The encoder features are passed through the decoder layers
        in reverse order to match the encoder features. The deep
        supervision heads produce "early" versions of the final output.

        Parameters
        ----------
        z : torch.Tensor
            Input feature map.
        encoder_feature : torch.Tensor
            Feature map from the encoder.
        prompt_encoding : torch.Tensor
            Prompt encoding.
        dense_encoding : torch.Tensor
            Dense encoding.

        Returns
        -------
        z : torch.Tensor
            Decoded feature map.
        prompt_encoding : torch.Tensor
            Updated prompt encoding.
        ds_out : torch.Tensor
            Deep supervision head output.
        """

        z = self._decode(z, encoder_feature)
        prompt_encoding = self.prompt_encoding(prompt_encoding)
        z, prompt_encoding = self._prompt_func(
            z, prompt_encoding, dense_encoding)
        z = self.conv2(z)  # maybe ?
        ds_out = self.ds_head(z)
        return z, prompt_encoding, ds_out


class BundleParcNetDecoder(torch.nn.Module):
    """ MedNeXt decoder with 4 decoder layers. """

    def __init__(
        self, prompt_strategy='add', channels=[512, 256, 128, 64, 32]
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            DecoderNextLayer(channels[0], channels[1], ratio=4,
                             prompt_strategy=prompt_strategy),
            DecoderNextLayer(channels[1], channels[2], ratio=4,
                             prompt_strategy=prompt_strategy),
            DecoderNextLayer(channels[2], channels[3], ratio=3,
                             prompt_strategy=prompt_strategy),
            DecoderNextLayer(channels[3], channels[4], ratio=2,
                             prompt_strategy=prompt_strategy),
        ])

    def forward(self, x, embeddings, prompt_encoding, dense_encoding):

        ds_outs = []

        for decoder_layer, encoder_feature, dense_feature in zip(
            self.layers, embeddings, dense_encoding
        ):
            x, prompt_encoding, ds_out = \
                decoder_layer(
                    x, encoder_feature, prompt_encoding, dense_feature)
            ds_outs.append(ds_out)

        return x, ds_outs


class Stem(torch.nn.Module):
    """ 3D Convolutional block with a convolutional layer, batch normalization
    and ReLU activation. """

    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()

        self.conv2 = torch.nn.Conv3d(
            in_chans, out_chans, kernel_size=1, stride=1)

        # Use Xavier initialisation for weights
        init_block(self)

    def forward(self, x):
        x = self.conv2(x)
        return x


class Head(torch.nn.Module):

    def __init__(self, in_chans: int):
        super().__init__()

        self.conv = torch.nn.Conv3d(
            in_chans, 2, kernel_size=1, stride=1)

        # Use Xavier initialisation for weights
        init_block(self)

    def forward(self, x):
        x = self.conv(x)
        return x


class BundleParcNet(torch.nn.Module):

    def __init__(self, in_chans, volume_size=96,
                 prompt_strategy='add',
                 embed_dim=32, bottleneck_dim=512, n_bundles=72):
        super().__init__()

        self.channels = [32, 64, 128, 256, 512]

        self.prompt_strategy = prompt_strategy
        self.volume_size = volume_size

        # Define the model
        self.stem = Stem(in_chans, embed_dim)
        self.mask_stem = Stem(1, embed_dim)

        self.encoder = UNextEncoder(self.channels)
        self.mask_encoder = UNextEncoder(self.channels)

        self.bottleneck = ConvNextBlock(bottleneck_dim, ratio=4)
        self.decoder = BundleParcNetDecoder(prompt_strategy,
                                            channels=self.channels[::-1])

        self.prompt_embedding = torch.nn.Sequential(
            torch.nn.Linear(n_bundles, bottleneck_dim), torch.nn.GELU())

        self.no_mask_embed = torch.nn.Embedding(1, embed_dim)

    def forward(self, fodf, bundle_prompt, wm_prompt=None):
        """ Forward pass of the model. Input emdeding refers to the
        embedding of the input fodf. Prompt embedding refers to the
        embedding of the bundle prompt (ie the one-hot vector representing
        the bundle to predict). "Dense" embedding refers to the embedding
        of the mask. This nomenclature stems from Segment Anything: bundle
        prompts can be seen as "sparse" whereas mask prompt can be seen as
        "dense".

        Parameters
        ----------
        fodf : torch.Tensor
            Input fodf.
        bundle_prompt : torch.Tensor
            Bundle prompt.
        wm_prompt : torch.Tensor, optional
            Mask prompt. Default is None.

        Returns
        -------
        ds_outs : list of torch.Tensor
            List of deep supervision heads. The last element is the final
            (non-deep) output of the model.
        """

        B, C, X, Y, Z = fodf.shape

        # Embed the input fodf
        input_embedding = self.stem(fodf)

        # Embed the bundle prompt to the same dimension as the input fodf
        prompt_embed = self.prompt_embedding(bundle_prompt)

        # Embded the "dense" prompt if it is provided
        # Else, use the learned embedding
        if torch.sum(wm_prompt) == 0:
            dense_embeddings = self.no_mask_embed.weight.reshape(
                1, -1, 1, 1, 1).expand(
                    B, -1, X, Y, Z)
        else:
            dense_embeddings = self.mask_stem(wm_prompt)

        # Run the encoders for the input fodf and the mask
        encoder_features = []
        mask_features = []
        x = input_embedding
        for encoder_layer in self.encoder.layers:
            x, x_res = encoder_layer(x)
            encoder_features.append(x_res)

        m = dense_embeddings
        for mask_encoder_layers in self.mask_encoder.layers:
            m, m_res = mask_encoder_layers(m)
            mask_features.append(m_res)

        # As opposed to the original MedNeXt, we do not use deep
        # supervision here, as the bottleneck does not receive any
        # prompt information and therefore cannot know which
        # bundle to predict
        z = self.bottleneck(x)

        # Run the decoder
        # Decoder layers are run in reverse order to match the
        # encoder features. Deep supervision heads produce smaller
        # versions of the final output
        z, ds_outs = self.decoder(
            z, encoder_features[::-1], prompt_embed, mask_features[::-1])

        return ds_outs

    def encode(self, fodf, wm_prompt=None):
        """ Forward pass of the model's encoder. """
        B, C, X, Y, Z = fodf.shape

        # Embed the input fodf
        input_embedding = self.stem(fodf)

        # Embded the "dense" mask if it is provided
        # Else, use the learned embedding
        if torch.sum(wm_prompt) == 0:
            dense_embeddings = self.no_mask_embed.weight.reshape(
                1, -1, 1, 1, 1).expand(
                    B, -1, X, Y, Z)
        else:
            dense_embeddings = self.mask_stem(wm_prompt)

        # Run the encoders for the input fodf and the mask
        encoder_features = []
        mask_features = []

        x = input_embedding
        for encoder_layer in self.encoder.layers:
            x, x_res = encoder_layer(x)
            encoder_features.append(x_res)

        m = dense_embeddings
        for mask_encoder_layers in self.mask_encoder.layers:
            m, m_res = mask_encoder_layers(m)
            mask_features.append(m_res)

        # As opposed to the original MedNeXt, we do not use deep
        # supervision here, as the bottleneck does not receive any
        # prompt information and therefore cannot know which
        # bundle to predict
        z = self.bottleneck(x)

        return z, encoder_features, mask_features

    def decode(self, z, encoder_features, mask_features, bundle_prompt):
        """ Forward pass of the model's decoder. """
        # Embed the bundle prompt to the same dimension as the input fodf
        prompt_embed = self.prompt_embedding(bundle_prompt)

        # Run the decoder
        # Decoder layers are run in reverse order to match the
        # encoder features. Deep supervision heads produce smaller
        # versions of the final output
        z, ds_outs = self.decoder(
            z, encoder_features[::-1], prompt_embed, mask_features[::-1])

        return ds_outs
