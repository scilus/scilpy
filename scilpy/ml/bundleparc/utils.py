import logging
import numpy as np
import os
import requests
import torch

from tqdm import tqdm

from scilpy.ml.bundleparc.bundleparcnet import BundleParcNet


# TODO in future: Get bundle list from model
DEFAULT_BUNDLES = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right', 'MCP', 'MLF_left', 'MLF_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right', 'SLF_III_left', 'SLF_III_right', 'SLF_II_left', 'SLF_II_right', 'SLF_I_left', 'SLF_I_right', 'STR_left', 'STR_right', 'ST_FO_left', 'ST_FO_right', 'ST_OCC_left', 'ST_OCC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'T_OCC_left', 'T_OCC_right', 'T_PAR_left', 'T_PAR_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PREC_left', 'T_PREC_right', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left', 'T_PREM_right', 'UF_left', 'UF_right']  # noqa E501

CKPT_URL = 'https://zenodo.org/records/15579498/files/123_4_5_bundleparc.ckpt' # noqa E501


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
    # A bit hackish, but we have to extract the "BundleParcNet" section from
    # the weights, as they were saved as part of an encompassing BundleParc
    # module.
    net_state_dict = {'.'.join(k.split('.')[1:]): v for k, v in
                      state_dict.items() if 'bundleparcnet' in k}
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
    fodf : numpy.ndarray
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
    input_fodf_data = fodf.transpose(
        (3, 0, 1, 2))[:n_coefs, ...].astype(dtype=np.float32)

    # Shape of the input fODF data
    fodf_shape = input_fodf_data.shape

    # If the input fODF has fewer than n_coefs coefficients, pad with zeros
    fodf_data = np.zeros((n_coefs, *fodf_shape[1:]), dtype=np.float32)
    fodf_data[:input_fodf_data.shape[0], ...] = input_fodf_data

    # z-score norm
    mean = np.mean(fodf_data)
    std = np.std(fodf_data)
    fodf_data = (fodf_data - mean) / std

    return fodf_data


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
    # Make sure directory exists
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    if not os.path.exists(path):
        resp = requests.get(CKPT_URL, stream=True)
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
