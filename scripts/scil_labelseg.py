#!/usr/bin/env python

"""
LabelSeg: automatic tract labelling without tractography.

This method takes as input fODF maps of order 6 (and a WM mask) and outputs 71
bundle label maps. These maps can then be used to perform tractometry/
tract profiling/radiomics. The bundle definitions follow TractSeg's minus the
whole CC.

This script requires PyTorch to be installed. To install it, see the official
website: https://pytorch.org/get-started/locally/

To cite: Antoine Théberge, Zineb El Yamani, François Rheault, Maxime Descoteaux, Pierre-Marc Jodoin (2025). LabelSeg. ISMRM Workshop on 40 Years of Diffusion: Past, Present & Future Perspectives, Kyoto, Japan.  # noqa
"""

import argparse
import nibabel as nib
import numpy as np
import os
import requests

from argparse import RawTextHelpFormatter
from tqdm import tqdm

from scipy.ndimage import gaussian_filter

from scilpy.io.utils import (
    assert_inputs_exist, assert_outputs_exist, add_overwrite_arg)
from scilpy.image.volume_operations import resample_volume

from scilpy.ml.labelseg.utils import get_model
from scilpy.ml.utils import get_device


from dipy.utils.optpkg import optional_package

IMPORT_ERROR_MSG = "PyTorch is required to run this script. Please install" + \
                   " it first. See the official website for more info: " + \
                   "https://pytorch.org/get-started/locally/"  # noqa
torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)

# TODO: Get bundle list from model
DEFAULT_BUNDLES = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right', 'MCP', 'MLF_left', 'MLF_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right', 'SLF_III_left', 'SLF_III_right', 'SLF_II_left', 'SLF_II_right', 'SLF_I_left', 'SLF_I_right', 'STR_left', 'STR_right', 'ST_FO_left', 'ST_FO_right', 'ST_OCC_left', 'ST_OCC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'T_OCC_left', 'T_OCC_right', 'T_PAR_left', 'T_PAR_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PREC_left', 'T_PREC_right', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left', 'T_PREM_right', 'UF_left', 'UF_right']  # noqa E501

DEFAULT_CKPT = os.path.join('checkpoints', 'labelsegnet.ckpt')


def to_numpy(tensor: torch.Tensor, dtype=np.float32) -> np.ndarray:
    """ Helper function to convert a torch GPU tensor
    to numpy.
    """

    return tensor.cpu().numpy().astype(dtype)


class LabelSeg():
    """

    """

    def __init__(
        self,
        dto: dict,
    ):
        """
        """
        self.checkpoint = dto['checkpoint']
        self.img_size = 128  # TODO: get image size from model
        self.fodf = dto['fodf']
        self.wm = dto['wm']
        self.out = dto['out_prefix']
        self.mask_out = dto['mask']
        self.bundles = DEFAULT_BUNDLES
        self.nb_labels = dto['nb_labels']
        self.n_coefs = int(
            (6 + 2) * (6 + 1) // 2)  # TODO: infer sh order from model

    def post_process_mask(self, mask, bundle_name):
        """ TODO
        """
        # Get the blobs in the image. Ideally, a mask only has one blob.
        # More than one either indicates a broken segmentation, or extraneous
        # voxels. TODO: handle these cases differently.
        bundle_mask = (mask > 0.5)

        # # TODO: investigate blobs
        # blobs, nb = ndi.label(bundle_mask)

        # # No need to process, return the mask
        # if nb <= 1:
        #     return bundle_mask.astype(np.uint8)

        # # Calculate the size of each blob
        # blob_sizes = np.bincount(blobs.ravel())
        # new_mask = np.zeros_like(bundle_mask)
        # # Remove blobs under a certain size (100)
        # for i in range(1, len(blob_sizes[1:])):
        #     if blob_sizes[i] > 1:
        #         new_mask[blobs == i] = 1

        return bundle_mask.astype(np.uint8)

    def post_process_labels(
        self, bundle_label, bundle_mask, nb_labels, sigma=0.5
    ):
        """ Masked filtering (normalized convolution) and label discretizing.
        Reference:
        https://stackoverflow.com/questions/59685140/python-perform-blur-only-within-a-mask-of-image  # noqa
        """

        # Masked convolution
        float_mask = bundle_mask.astype(float)
        filtered = gaussian_filter(bundle_label * float_mask, sigma=sigma)
        weights = gaussian_filter(float_mask, sigma=sigma)
        filtered /= (weights + 1e-8)
        filtered = filtered * bundle_mask

        # Label masking
        discrete_labels = bundle_label[bundle_mask.astype(bool)]

        # Label dicretizing
        discrete_labels = np.round(discrete_labels * nb_labels)
        bundle_label[bundle_mask.astype(bool)] = discrete_labels
        bundle_label[~bundle_mask.astype(bool)] = 0

        return bundle_label.astype(np.int32)

    @torch.no_grad()
    def predict(self, model, fodf, wm):
        """
        """
        device = get_device()

        nb_bundles = len(self.bundles)
        fodf_data = fodf.get_fdata().transpose(
            (3, 0, 1, 2))[:self.n_coefs, ...]  # TODO: get from model
        wm_data = wm.get_fdata()[None, ...]

        # z-score norm
        mean = np.mean(fodf_data)
        std = np.std(fodf_data)
        fodf_data = (fodf_data - mean) / std

        # Predict the scores of the streamlines
        pbar = tqdm(range(nb_bundles))

        # TODO: reuse encoding since it doesn't have prompt info
        data = torch.tensor(
            fodf_data,
            dtype=torch.float
        ).to(device)

        wm_prompt = torch.tensor(
            wm_data,
            dtype=torch.float
        ).to(device)

        prompts = torch.eye(len(self.bundles), device=device)

        with torch.no_grad():

            z, encoder_features, mask_features = model.encode(
                data[None, ...], wm_prompt[None, ...])

            for i in pbar:
                pbar.set_description(self.bundles[i])

                y_hat = torch.nn.functional.sigmoid(model.decode(
                    z, encoder_features, mask_features, prompts[None, i, ...]
                )[-1]).squeeze()

                y_hat_np = to_numpy(y_hat)
                bundle_mask = y_hat_np[0]
                bundle_label = y_hat_np[1]

                bundle_mask = self.post_process_mask(
                    bundle_mask, self.bundles[i])
                bundle_label = self.post_process_labels(
                    bundle_label, bundle_mask, self.nb_labels)

                yield bundle_mask, bundle_label, self.bundles[i]

    def run(self):
        """
        Main method where the magic happens
        """
        fodf_in = nib.load(self.fodf)
        wm_in = nib.load(self.wm)

        # shape = fodf_in.get_fdata().shape[:3]

        # Resampling volume to fit the model's input at training time
        resampled_img = resample_volume(fodf_in, ref_img=None,
                                        volume_shape=[self.img_size],
                                        iso_min=False,
                                        voxel_res=None,
                                        interp='lin',
                                        enforce_dimensions=False)

        resampled_wm = resample_volume(wm_in, ref_img=None,
                                       volume_shape=[self.img_size],
                                       iso_min=False,
                                       voxel_res=None,
                                       interp='nn',
                                       enforce_dimensions=False)

        # Load the model
        model = get_model(self.checkpoint, get_device())

        # Predict label maps. `self.predict` is a generator
        # yielding one label map per bundle (and a binary mask)
        for y_hat_mask, y_hat_label, b_name in self.predict(
            model, resampled_img, resampled_wm
        ):
            # Format the output as a nifti image
            label_img = nib.Nifti1Image(y_hat_label,
                                        resampled_wm.affine,
                                        resampled_wm.header)

            # Resample the image back to its original resolution
            label_img = resample_volume(label_img, ref_img=wm_in,
                                        # volume_shape=shape,
                                        iso_min=False,
                                        voxel_res=None,
                                        interp='nn',
                                        enforce_dimensions=False)
            # Save it
            nib.save(label_img, self.out + f'{b_name}.nii.gz')

            # If the binary mask is also desired, perform the same
            # processing.
            if self.mask_out:
                mask_img = nib.Nifti1Image(y_hat_mask,
                                           resampled_wm.affine,
                                           resampled_wm.header)
                mask_img = resample_volume(mask_img, ref_img=wm_in,
                                           # volume_shape=shape,
                                           iso_min=False,
                                           voxel_res=None,
                                           interp='nn',
                                           enforce_dimensions=False)
                nib.save(label_img, self.mask_out + f'{b_name}.nii.gz')


def _build_arg_parser(parser):
    parser.add_argument('fodf', type=str,
                        help='fODF input')
    parser.add_argument('wm', type=str,
                        help='WM input')
    parser.add_argument('out_prefix', type=str,
                        help='Output destination and file prefix.')
    parser.add_argument('--mask', type=str, default=None,
                        help='Output destination and file prefix for '
                             'binary mask output.')
    parser.add_argument('--nb_labels', type=int, default=50)
    parser.add_argument('--checkpoint', type=str,
                        default=DEFAULT_CKPT,
                        help='Checkpoint (.ckpt) containing hyperparameters '
                             'and weights of model. Default is '
                             '[%(default)s].')

    add_overwrite_arg(parser)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter)

    _build_arg_parser(parser)
    args = parser.parse_args()

    assert_inputs_exist(parser, args.fodf)
    assert_outputs_exist(parser, args, args.out_prefix,
                         [args.mask])

    return parser, args


def download(url: str, fname: str, chunk_size=1024):
    # From https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def _download_weights(path=DEFAULT_CKPT):
    url = 'https://zenodo.org/records/14813477/files/labelsegnet.ckpt'
    os.makedirs(os.path.dirname(path))
    print('Downloading weights ...')
    download(url, path)
    print('Done !')


def main():

    parser, args = parse_args()

    # if file not exists
    if not os.path.exists(args.checkpoint):
        _download_weights(args.checkpoint)

    experiment = LabelSeg(vars(args))
    experiment.run()


if __name__ == "__main__":
    main()
