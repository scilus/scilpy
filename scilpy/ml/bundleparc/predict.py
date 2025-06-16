import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from dipy.utils.optpkg import optional_package
from scilpy.ml.utils import get_device, to_numpy, IMPORT_ERROR_MSG


torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)


def post_process_mask(mask, bundle_name):
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

    # Binarize the mask
    bundle_mask = (mask > 0.5)

    # TODO: investigate blobs. TractSeg performs some post-processing
    # to fix broken commissures or fornices. Would be nice to do the same.
    # For future ref: use ndi.label to get the blobs

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

    # Masked convolution
    # The labels are first smoothed using a Gaussian filter.
    float_mask = bundle_mask.astype(float)
    filtered = gaussian_filter(bundle_label * float_mask, sigma=sigma)
    weights = gaussian_filter(float_mask, sigma=sigma)
    filtered /= (weights + 1e-8)
    filtered = filtered * bundle_mask

    # Label masking.
    discrete_labels = bundle_label[bundle_mask.astype(bool)]

    # Label dicretizing.
    discrete_labels = np.round(discrete_labels * nb_labels)
    bundle_label[bundle_mask.astype(bool)] = discrete_labels
    bundle_label[~bundle_mask.astype(bool)] = 0

    return bundle_label.astype(np.int32)


@torch.no_grad()
def predict(
    model, fodf_data, wm_data, nb_labels, bundles, half_precision=False,
    verbose=True
):
    """ Predict the  bundle labels. This function is a generator that yields
    the predicted labels for each bundle.

    Parameters
    ----------
    model : LabelSegNet
        Model to use for the prediction.
    fodf_data : np.ndarray
        fODF data.
    wm_data : np.ndarray
        Whole-brain white matter mask.
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
    device = get_device()

    # Predict the scores of the streamlines
    pbar = tqdm(range(len(bundles)), disable=not verbose)

    with torch.amp.autocast(str(device), enabled=half_precision):

        # Convert the data to tensors and move them to the device.
        data = torch.tensor(
            fodf_data,
            dtype=torch.float
        ).to(device)

        wm_prompt = torch.tensor(
            wm_data,
            dtype=torch.float
        ).to(device)

        # Create a one-hot encoding of the bundle prompts.
        prompts = torch.eye(len(bundles), device=device)

        # Encode the data once, reuse the features.
        z, encoder_features, mask_features = model.encode(
            data[None, ...], wm_prompt[None, ...])

        # Loop over the bundles.
        for i in pbar:
            pbar.set_description(bundles[i])

            # Predict the scores.
            y_hat = torch.nn.functional.sigmoid(model.decode(
                z, encoder_features, mask_features, prompts[None, i, ...]
            )[-1]).squeeze()

            # Get the numpy arrays (first dim is the bundle mask and the second
            # is the labels).
            y_hat_np = to_numpy(y_hat)
            bundle_mask = y_hat_np[0]
            bundle_label = y_hat_np[1]

            # Post-process the mask and labels.
            # Binarize the mask (and in a future release remove small blobs and
            # fill holes).
            bundle_mask = post_process_mask(
                bundle_mask, bundles[i])
            # Extract the labels using the mask, then filter and discretize
            # them.
            bundle_label = post_process_labels(
                bundle_label, bundle_mask, nb_labels)

            yield bundle_label, bundles[i]
