import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from scilpy.ml.utils import get_device, to_numpy

from dipy.utils.optpkg import optional_package
IMPORT_ERROR_MSG = "PyTorch is required to run this script. Please install" + \
                   " it first. See the official website for more info: " + \
                   "https://pytorch.org/get-started/locally/"  # noqa
torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)


def post_process_mask(mask, bundle_name):
    """ TODO
    """
    # Get the blobs in the image. Ideally, a mask only has one blob.
    # More than one either indicates a broken segmentation, or extraneous
    # voxels. TODO: handle these cases differently.
    bundle_mask = (mask > 0.5)

    # TODO: investigate blobs. TractSeg performs some post-processing
    # to fix broken commissures or fornices. Would be nice to do the same.

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
    bundle_label, bundle_mask, nb_labels, sigma=0.5
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
def predict(
    model, fodf_data, wm_data, nb_labels, n_coefs, bundles
):
    """
    """
    device = get_device()

    # Predict the scores of the streamlines
    pbar = tqdm(range(len(bundles)))

    # TODO: reuse encoding since it doesn't have prompt info
    data = torch.tensor(
        fodf_data,
        dtype=torch.float
    ).to(device)

    wm_prompt = torch.tensor(
        wm_data,
        dtype=torch.float
    ).to(device)

    prompts = torch.eye(len(bundles), device=device)

    with torch.no_grad():

        z, encoder_features, mask_features = model.encode(
            data[None, ...], wm_prompt[None, ...])

        for i in pbar:
            pbar.set_description(bundles[i])

            y_hat = torch.nn.functional.sigmoid(model.decode(
                z, encoder_features, mask_features, prompts[None, i, ...]
            )[-1]).squeeze()

            y_hat_np = to_numpy(y_hat)
            bundle_mask = y_hat_np[0]
            bundle_label = y_hat_np[1]

            bundle_mask = post_process_mask(
                bundle_mask, bundles[i])
            bundle_label = post_process_labels(
                bundle_label, bundle_mask, nb_labels)

            yield bundle_mask, bundle_label, bundles[i]
