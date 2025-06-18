import logging
import numpy as np
import os
import requests

from dipy.utils.optpkg import optional_package
from tqdm import tqdm

from scilpy.io.image import get_data_as_mask
from scilpy.ml.utils import IMPORT_ERROR_MSG
from scilpy.ml.bundleparc.bundleparcnet import BundleParcNet


torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)


def get_data(fodf, wm, img_size, n_coefs):
    """ Get the data from the input files and prepare it for the model.
    This function truncates the number of coefficients to fit the model's input
    z-score normalizes the fODF data and adds a channel dimension to the white
    matter mask.

    Parameters
    ----------
    fodf : nibabel.Nifti1Image
        fODF data.
    wm : nibabel.Nifti1Image
        Whole-brain white matter mask.
    img_size : int
        Size of the image.
    n_coefs : int
        Number of SH coefficients to use.

    Returns
    -------
    fodf_data : np.ndarray
        fODF data.
    wm_data : np.ndarray
        Whole-brain white matter mask.
    """

    # Select the first n_coefs coefficients from the fodf data and put it in
    # the first dimension. This essentially truncates the number of
    # coefficients to fit the model's input size. The model was trained with
    # fODF data of order 8 truncated to 28 coefficients so I'm not worried
    # about doing it here.
    fodf_data = fodf.get_fdata().transpose(
        (3, 0, 1, 2))[:n_coefs, ...].astype(dtype=np.float32)

    # Add a channel dimension to the whole-brain white matter mask
    wm_data = get_data_as_mask(wm)[None, ...].astype(np.float32)

    # z-score norm
    mean = np.mean(fodf_data)
    std = np.std(fodf_data)
    fodf_data = (fodf_data - mean) / std

    return fodf_data, wm_data


def get_model(checkpoint_file, device):
    """ Get the model from a checkpoint.

    Parameters
    ----------
    checkpoint_file : str
        Path to the checkpoint file.
    device : torch.device
        Device on which to put the model. Either 'cpu' or 'cuda'.

    Returns
    -------
    model : BundleParcNet
        Model loaded from the checkpoint.
    """

    # Load the model's hyper and actual params from a saved checkpoint
    try:
        checkpoint = torch.load(checkpoint_file, weights_only=False)
    except RuntimeError:
        # If the model was saved on a GPU and is being loaded on a CPU
        # we need to specify map_location=torch.device('cpu')
        checkpoint = torch.load(
            checkpoint_file, map_location=torch.device('cpu'),
            weights_only=False)

    state_dict = checkpoint['state_dict']
    hyperparameters = checkpoint['hyperparameters']

    model = BundleParcNet(
        hyperparameters['in_chans'],
        hyperparameters['volume_size'],
        hyperparameters['prompt_strategy'],
        hyperparameters['embed_dim'],
        hyperparameters['bottleneck_dim'],
        n_bundles=hyperparameters['n_bundles'])

    model.load_state_dict(state_dict)
    model.to(device)

    # Put the model in eval mode to fix dropout and other stuff
    model.eval()

    return model


def download_weights(path, chunk_size=1024, verbose=True):
    """ Download the weights for BundleParcNet.

    Parameters
    ----------
    path : str
        Path to the file where the weights will be saved.
    chunk_size : int, optional
        Size of the chunks to download the file.
    """

    # Adapted from
    # https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    url = 'https://zenodo.org/records/14813477/files/labelsegnet.ckpt'
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    logging.info('Downloading weights for BundleParc ...')
    with open(path, 'wb') as file, tqdm(
        desc=path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        disable=not verbose
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    logging.info('Done !')
