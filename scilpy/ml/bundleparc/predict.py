import logging
import numpy as np

from tqdm import tqdm

from scipy.ndimage import gaussian_filter, label

from scilpy.ml.utils import get_device, to_numpy, IMPORT_ERROR_MSG
from scilpy.ml.bundleparc.utils import DEFAULT_BUNDLES, get_data

from dipy.utils.optpkg import optional_package
torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)


def post_process_mask(
    mask, bundle_name, min_blob_size=100, keep_biggest_blob=False
):
    """ Post-process the mask. This function binarizes the mask. In a future
    release, it will also remove small blobs and fill holes (this is why
    the bundle name is passed).

    Parameters
    ----------
    mask : np.ndarray
        Predicted mask for the bundle.
    bundle_name : str
        Name of the bundle.
    """
    bundle_mask = (mask > 0.5)

    # Get the blobs in the image. Ideally, a mask only has one blob.
    # More than one either indicates a broken segmentation, or extraneous
    # voxels.
    blobs, nb = label(bundle_mask)

    # No need to process, return the mask
    if nb <= 1:
        logging.debug(f"Only one blob in {bundle_name}.")
        return bundle_mask.astype(np.uint8)

    # Calculate the size of each blob
    blob_sizes = np.bincount(blobs.ravel())
    new_mask = np.zeros_like(bundle_mask)

    if keep_biggest_blob:
        logging.debug(f"More than one blob in {bundle_name}, keeping largest")
        biggest_blob = np.argmax(blob_sizes[1:])
        new_mask[blobs == biggest_blob + 1] = 1
        return new_mask

    # Remove blobs under a certain size (min_blob_size)
    new_nb_blobs = 0
    for i in range(1, len(blob_sizes[1:])):
        if blob_sizes[i] >= min_blob_size:
            new_mask[blobs == i] = 1
            new_nb_blobs += 1
    logging.debug(f'Kept {new_nb_blobs} blob out of {nb} in {bundle_name}.')
    return bundle_mask.astype(np.uint8)


def post_process_labels(
    bundle_label, bundle_mask, nb_labels, sigma=0.5
):
    """ Masked filtering (normalized convolution) and label discretizing.
    Reference:
    https://stackoverflow.com/questions/59685140/python-perform-blur-only-within-a-mask-of-image  # noqa

    Parameters
    ----------
    bundle_label : np.ndarray
        Predicted continuous labels for the bundle.
    bundle_mask : np.ndarray
        Mask of the bundle.
    nb_labels : int
        Number of labels to discretize to.
    sigma : float, optional
        Filtering sigma. Default is 0.5.

    Returns
    -------
    bundle_label : np.ndarray
        Predicted labels for the bundle.
    """

    # Determine the output type based on the number of labels
    out_type = np.uint16 if nb_labels > 1 else np.uint8

    # Masked convolution
    float_mask = bundle_mask.astype(float)
    filtered = gaussian_filter(bundle_label * float_mask, sigma=sigma)
    weights = gaussian_filter(float_mask, sigma=sigma)
    filtered /= (weights + 1e-8)
    filtered = filtered * bundle_mask
    # Label masking
    discrete_labels = bundle_label[bundle_mask.astype(bool)]

    # Label dicretizing
    discrete_labels = np.ceil(discrete_labels * nb_labels)
    bundle_label[bundle_mask.astype(bool)] = discrete_labels
    bundle_label[~bundle_mask.astype(bool)] = 0

    return bundle_label.astype(out_type)


@torch.no_grad()
def predict(
    model, fodf, n_coefs, nb_labels, bundles, min_blob_size, keep_biggest_blob,
    half_precision=False, verbose=False
):
    """ Predict the  bundle labels. This function is a generator that yields
    the predicted labels for each bundle.

    Parameters
    ----------
    model : LabelSegNet
        Model to use for the prediction.
    fodf: nib.nib.Nifti1Image
        fODF image, resampled to the model's input size.
    n_coefs : int
        Number of SH coefficients to use.
    nb_labels : int
        Number of labels to predict.
    bundles : list of str
        List of bundle names.
    half_precision : bool, optional
        Whether to use half precision. Will reduce GPU memory usage but may
        reduce the accuracy of the label maps. Default is False.
    verbose : bool, optional
        Whether to display a progress bar. Default is True.

    Yields
    ------
    bundle_label : np.ndarray
        Predicted labels for the bundle.
    bundle_name : str
        Name of the bundle.
    """

    bundle_indices = np.array([DEFAULT_BUNDLES.index(b) for b in bundles])
    device = get_device()
    fodf_data = get_data(fodf, n_coefs)

    pbar = tqdm(bundle_indices, disable=not verbose)

    with torch.amp.autocast(device.type, enabled=half_precision):

        # Convert the fODF data to a torch tensor
        data = torch.tensor(
            fodf_data,
            dtype=torch.float
        ).to(device)

        # Create a one-hot encoding of the bundle prompts.
        prompts = torch.eye(len(DEFAULT_BUNDLES), device=device)

        # Encode the data once, reuse the features.
        z, encoder_features = model.encode(
            data[None, ...])

        # Loop over the bundles.
        for i in pbar:
            pbar.set_description(DEFAULT_BUNDLES[i])

            # Decode the features for the current bundle.
            y_hat = torch.nn.functional.sigmoid(model.decode(
                z, encoder_features, prompts[None, i, ...]
            )[-1]).squeeze()

            # Get the predicted mask and labels as numpy arrays.
            y_hat_np = to_numpy(y_hat)
            bundle_mask = y_hat_np[0]
            bundle_label = y_hat_np[1]

            # Post-process the mask and labels.
            # Binarize the mask (and in a future release remove small blobs and
            # fill holes
            bundle_mask = post_process_mask(
                bundle_mask, DEFAULT_BUNDLES[i], min_blob_size=min_blob_size,
                keep_biggest_blob=keep_biggest_blob)

            # Extract the labels using the mask, then filter and discretize
            # them.
            bundle_label = post_process_labels(
                bundle_label, bundle_mask, nb_labels)

            yield bundle_label, DEFAULT_BUNDLES[i]
