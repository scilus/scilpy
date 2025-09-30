.. _page_connectivity:

Connectivity analyses
=====================

Connectivity analyses use whole-brain tractograms segmented based on a gray-matter (GM) parcellation to create various connectivity matrices.

For a very simple computation, the GM label associated to each endpoint of the streamlines can be used to create a connectivity matrix based on streamline count with `scil_connectivity_compute_simple_matrix`. However, scilpy also offers a more thorough analysis, with a more complex analysis of endpoints and more options for the weight of the connectivity matrix. It comprises the steps described below.

1) First, the script scil_tractogram_segment_connections_from_labels is used and results in many sub-bundles, one for each pair of GM regions (each pair of labels). This is similar to ROI-based segmentation, but its computation makes a more complex analysis in cases where streamlines cross many GM regions. The output is saved in the hdf5 format for the ease of use in subsequent scripts, but can be converted back to a list of .TRK files with scil_tractogram_convert_hdf5_to_trk. Any manipulation can be performed on the bundles and then they could be merged back (scil_tractogram_convert_trk_to_hdf5). Note that bundle registration and analysis presented before could also be computed directly from the resulting HDF5 format, such as warping (`scil_tractogram_apply_transform_to_hdf5`) or statistics computation (`scil_bundle_mean_fixel_afd_from_hdf5`).

2) Then, it is possible to compute the connectivity matrix, using `scil_connectivity_compute_matrices`, using many weights such as the streamline count, the lengths of streamlines, the volume of bundles, or the average of any underlying map or any dps / dpp stored in the data.

3) It is possible to modify the matrices afterhand with scripts such as `scil_connectivity_math` (operations such as threshold, addition, multiplication, interchanging rows, etc.), `scil_connectivity_filter` (to binarize a list of matrices based on conditions) or `scil_connectivity_normalize` (to modify the minimum and maximum  values).


Examples
--------

Here is the list of commands to reproduce Figure 5 in our upcoming scilpy paper.


.. image:: ../../_static/scilpy_paper_figure5.png
   :alt: Figure 5 in upcoming paper.
