import logging
import numpy as np

from dipy.tracking.streamline import length, set_number_of_points

from scilpy.tractograms.streamline_operations import smooth_line_gaussian


def post_process_labels_discrete(
    nb_labels, bundle_label, bundle_mask, bundle_name
):
    """ Discretize the labels and apply a mask to the bundle. Labels are
    discretized to integers in the range [1, nb_labels] uniformly.

    Parameters
    ----------
    nb_labels : int
        Number of labels to discretize to.
    bundle_label : np.ndarray
        Predicted continuous labels for the bundle.
    bundle_mask : np.ndarray
        Binary mask of the bundle.
    bundle_name : str
        Name of the bundle, used for logging.

    Returns
    -------
    bundle_label : np.ndarray
        Predicted labels for the bundle.
    """

    # Determine the output type based on the number of labels
    # In scilpy/MI-Brain, uint16 is used for labels, uint8 for binary masks.
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
    """ Discretize the labels and apply a mask to the bundle. Labels are
    discritezed to integers so that each section is roughly `labels_mm` mm long
    To do so, the barycenter of each label is computed to form a centroid
    streamline. Then, the centroid is resampled to have a number of points such
    that the step-size is roughly `labels_mm` mm. Finally, the labels are
    reassigned to the closest point in the resampled centroid.

    Parameters
    ----------
    labels_mm : float
        Length of each section in mm.
    voxel_size : np.ndarray
        Voxel size of the bundle image.
    bundle_label : np.ndarray
        Predicted continuous labels for the bundle.
    bundle_mask : np.ndarray
        Binary mask of the bundle.
    bundle_name : str
        Name of the bundle, used for logging.

    Returns
    -------
    bundle_label : np.ndarray
        Predicted labels for the bundle.
    """

    # Label masking
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
    # Calculate the number of points to resample to
    nb_points = np.round(c_length / labels_mm).astype(int)
    if nb_points < 2:
        logging.warning(f"{bundle_name} is shorter than the section length.")
        nb_points = 2

    # Resample the centroid to have `nb_points` points
    # Adding 2 points so they can be excluded from the labels. Sort of a
    # reverse signpost problem. Otherwize, the first and last labels
    # and no other would be assigned to the first and last point of the
    # centroid.
    resampled_centroid = set_number_of_points(centroid, nb_points + 2)

    # Re-discretizing the labels based on the resampled centroid
    discrete_labels = np.zeros_like(bundle_label, dtype=np.float32)
    for i, label in enumerate(unique[1:]):
        # Find the closest label in the resampled centroid
        c = centroid[i]
        # Calculate the distances from the centroid to the resampled centroid
        # Exclude the first and last points of the resampled centroid (see
        # above)
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
    """ Don't discretize the labels, just apply a mask to the bundle.

    Parameters
    ----------
    bundle_label : np.ndarray
        Predicted continuous labels for the bundle.
    bundle_mask : np.ndarray
        Binary mask of the bundle.
    bundle_name : str
        Name of the bundle, used for logging.

    Returns
    -------
    bundle_label : np.ndarray
        Predicted labels for the bundle.
    """

    # Determine the output type based on the number of labels
    # In this case, we assume the labels are continuous and
    # can be represented as floats.
    out_type = float

    # Label masking
    bundle_label[~bundle_mask.astype(bool)] = 0

    return bundle_label.astype(out_type)
