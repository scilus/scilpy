import numpy as np
import os
import requests
import tqdm

from scilpy.ml.labelseg.labelsegnet import LabelSegNet

from dipy.utils.optpkg import optional_package

IMPORT_ERROR_MSG = "PyTorch is required to run this script. Please install" + \
                   " it first. See the official website for more info: " + \
                   "https://pytorch.org/get-started/locally/"  # noqa
torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)


def get_data(fodf, wm, img_size, n_coefs):

    # Instanciate an empty numpy array to store the fodf data
    fodf_data = np.zeros((n_coefs, *fodf.shape[:3]))

    # Select the first n_coefs coefficients from the fodf data
    # and put it in the first dimension
    # This should be robust to higher or lower numbers of coefficients
    # than the model was trained on
    fodf_data[:n_coefs, ...] = fodf.get_fdata().transpose(
        (3, 0, 1, 2))[:n_coefs, ...]

    # Add a channel dimension to the white matter mask
    wm_data = wm.get_fdata()[None, ...]

    # z-score norm
    mean = np.mean(fodf_data)
    std = np.std(fodf_data)
    fodf_data = (fodf_data - mean) / std

    return fodf_data, wm_data


def get_model(checkpoint_file, device):
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
    hyperparameters = checkpoint['hyperparameters']

    model = LabelSegNet(
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


def download_weights(path, chunk_size=1024):
    # Adapted from
    # https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    os.makedirs(os.path.dirname(path))
    url = 'https://zenodo.org/records/14813477/files/labelsegnet.ckpt'
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    print('Downloading weights ...')
    with open(path, 'wb') as file, tqdm(
        desc=path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    print('Done !')
