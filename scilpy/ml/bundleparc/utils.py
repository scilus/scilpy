import logging
import numpy as np
import os
import requests
import torch

from tqdm import tqdm

from scilpy.ml.bundleparc.bundleparcnet import BundleParcNet


def to_numpy(tensor: torch.Tensor, dtype=np.float32) -> np.ndarray:
    """ Helper function to convert a torch GPU tensor
    to numpy.
    """

    return tensor.cpu().numpy().astype(dtype)


def get_model(checkpoint_file, device, kwargs={}):
    """ Get the model from a checkpoint. """

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
    net_state_dict = { '.'.join(k.split('.')[1:]): v for k,v in state_dict.items() if 'bundleparcnet' in k}
    model = BundleParcNet(45)

    model.load_state_dict(net_state_dict)
    model.to(device)

    # Put the model in eval mode to fix dropout and other stuff
    model.eval()

    return model


def get_data(fodf, n_coefs):
    """ Get the data from the input files and prepare it for the model.
    This function truncates or pad the number of coefficients to fit the
    model's input and z-score normalizes the fODF data.

    Parameters
    ----------
    fodf : nibabel.Nifti1Image
        fODF data.
    n_coefs : int
        Number of SH coefficients to use.

    Returns
    -------
    fodf_data : np.ndarray
        fODF data.
    """

    # Select the first n_coefs coefficients from the fodf data and put it in
    # the first dimension. This truncates the number of coefficients if there
    # are more than n_coefs.
    input_fodf_data = fodf.get_fdata().transpose(
        (3, 0, 1, 2))[:n_coefs, ...].astype(dtype=np.float32)

    # Shape of the input fODF data
    fodf_shape = input_fodf_data.shape

    # If the input fODF has fewer than n_coefs coefficients, pad with zeros
    fodf_data = np.zeros((n_coefs, *fodf_shape[1:]), dtype=np.float32)
    fodf_data[:n_coefs, ...] = input_fodf_data

    # z-score norm
    mean = np.mean(fodf_data)
    std = np.std(fodf_data)
    fodf_data = (fodf_data - mean) / std

    return fodf_data


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


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
        url = 'https://zenodo.org/records/15579498/files/123_4_5_bundleparc.ckpt'
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
