import numpy as np

from dipy.tracking.streamline import length, set_number_of_points

from scilpy.tractograms.streamline_operations import smooth_line_gaussian


def post_process_labels_discrete(
    nb_labels, bundle_label, bundle_mask, bundle_name
):
    """ Masked filtering (normalized convolution) and label discretizing.
    Reference:
    https://stackoverflow.com/questions/59685140/python-perform-blur-only-within-a-mask-of-image  # noqa

    Parameters
    ----------
    nb_labels : int
        Size of the labels to discretize to.
    bundle_label : np.ndarray
        Predicted continuous labels for the bundle.
    bundle_mask : np.ndarray
        Mask of the bundle.

    Returns
    -------
    bundle_label : np.ndarray
        Predicted labels for the bundle.
    """

    # Determine the output type based on the number of labels
    out_type = np.uint16 if nb_labels > 1 else np.uint8

    # Label masking
    discrete_labels = bundle_label[bundle_mask.astype(bool)]

    # Label dicretizing
    discrete_labels = np.ceil(discrete_labels * nb_labels)
    bundle_label[bundle_mask.astype(bool)] = discrete_labels
    bundle_label[~bundle_mask.astype(bool)] = 0

    return bundle_label.astype(out_type)


def post_process_labels_mm(
    labels_mm, voxel_size, bundle_label, bundle_mask, bundle_name
):
    """ Masked filtering (normalized convolution) and label discretizing.
    Reference:
    https://stackoverflow.com/questions/59685140/python-perform-blur-only-within-a-mask-of-image  # noqa

    Parameters
    ----------
    labels_mm: float
        Number of labels to discretize to.
    bundle_label : np.ndarray
        Predicted continuous labels for the bundle.
    bundle_mask : np.ndarray
        Mask of the bundle.

    Returns
    -------
    bundle_label : np.ndarray
        Predicted labels for the bundle.
    """

    # Label dicretizing
    bundle_label[~bundle_mask.astype(bool)] = 0

    ref_labels = np.ceil(bundle_label * 50)
    unique = np.unique(ref_labels)

    # Get the 3D coordinates of the barycenter of each label
    barycenters = np.zeros((len(unique) - 1, 3), dtype=np.float32)
    for i, label in enumerate(unique[1:]):
        coords = np.argwhere(ref_labels == label)
        barycenters[i] = np.mean(coords, axis=0)

    # Form the barycenters into a single streamline
    centroid = np.asarray(barycenters)
    centroid = smooth_line_gaussian(centroid, 5)

    # Resampling
    c_length = length(centroid * voxel_size)
    nb_points = np.round(c_length / labels_mm).astype(int)
    if nb_points < 2:
        print(f"{bundle_name} is shorter than the section length.")
        nb_points = 2
    resampled_centroid = set_number_of_points(centroid, nb_points + 2)

    # Re-discretizing the labels based on the resampled centroid
    discrete_labels = np.zeros_like(bundle_label, dtype=np.float32)
    for i, label in enumerate(unique[1:]):
        # Find the closest label in the resampled centroid
        c = centroid[i]
        distances = np.linalg.norm(
            c - resampled_centroid[None, 1:-1], axis=-1)
        # Get the index of the closest label
        closest_index = np.argmin(distances)
        # Assign the label to the closest index in the resampled centroid
        discrete_labels[ref_labels == label] = closest_index + 1

    # Determine the output type based on the number of labels
    out_type = np.uint16 if nb_points > 1 else np.uint8

    return discrete_labels.astype(out_type)


def post_process_labels_continuous(
    bundle_label, bundle_mask, bundle_name
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

    Returns
    -------
    bundle_label : np.ndarray
        Predicted labels for the bundle.
    """

    # Determine the output type based on the number of labels
    out_type = float

    # Label dicretizing
    bundle_label[~bundle_mask.astype(bool)] = 0

    return bundle_label.astype(out_type)
