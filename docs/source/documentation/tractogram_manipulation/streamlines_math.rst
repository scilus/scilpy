.. _page_streamlines_math:

Streamlines manipulation (resampling, smoothing)
================================================

Here, we discuss scripts allowing modifying the streamlines themselves, for instance by resampling the number of points on each streamline (:ref:`scil_tractogram_resample_nb_points`, :ref:`scil_tractogram_compress`), or smoothing the streamlines’ trajectories (:ref:`scil_tractogram_smooth`).

See page :ref:`page_tractogram_math` for scripts modifying the tractograms as a whole.


Preparing data for this tutorial
********************************

Example of data for this tutorial can be downloaded using instructions here: :ref:`page_getting_started`. Let's use a tractogram available in the tutorial data. In the last section, you will also need a mask, and two regions of interest (ROIs), that we will create from labels: wmparc.

.. code-block:: bash

    # Let's use a tractogram available in the data:
    # tractogram1=$in_dir/sub-01/sub-01__cst_L_part2.trk not compatible...
    tractogram1=$in_dir/sub-01/sub-01__cst_L.trk
    labels=$in_dir/sub-01/sub-01__wmparc.nii.gz
    mask=$in_dir/sub-01/sub-01__small_mask_wm.nii.gz

    # Current wmparc is in float. Should be in int. Let's convert.
    scil_volume_math convert $in_dir/sub-01/sub-01__wmparc.nii.gz \
        --data_type int16  -f $in_dir/sub-01/sub-01__wmparc.nii.gz

To look at your data in a viewer, you may use the subject's T1 volume:

.. code-block:: bash

    t1=$in_folder/sub-01/sub-01__t1.nii.gz

.. tip::
    You may download the complete bash script to run the whole tutorial in one step `here </_static/bash/tractogram_manipulation/streamlines_math.sh>`_.

Resampling the number of points in each streamline
**************************************************

A streamline is a list of points (coordinates), connected by segments. The more you have points, the more precise and smooth your streamline will be, but if you have too much, your data will become heavy on disk. 20 is usually a good number of points. Atlernatively, you could set the length of each segment, using for example a step size of 0.5mm, with ``--step_size 0.5``.

.. code-block:: bash

    scil_tractogram_resample_nb_points $tractogram1 resampled_20pts.trk \
        --nb_pts_per_streamline 20

This term "resampling streamlines" should not be counfounded with "resampling tractograms", which means changing the number of streamlines in the tractogram.

Compressing streamlines
***********************

An alternative is to use DIPY's compression. This uses the minimal number of points required to have good quality. In regions where the streamline is straight, only a few points are required. For streamlines with high curvature, more points are required.

.. code-block:: bash

    scil_tractogram_compress $tractogram1 compressed.trk -v

Cutting streamlines
*******************

It is possible to simply cut streamlines that are too long (ex, cut the parts that are going out of a given mask) using the following command lines. This will only remove the points (segments) that are extra, it will not remove the entire streamline. This would be called *filtering* of streamlines, and you can learn more about in :ref:`page_tractogram_filtering`.

**1. Cutting the parts that are out of the mask**

The option ``--min_length`` will ensure that final streamlines after cutting are not too short (we chose 20mm).

.. code-block:: bash

    scil_tractogram_cut_streamlines $tractogram1 cut_streamlines.trk \
        --mask $mask --min_length 20

**2. Cutting the parts that are not inside two ROIs**

This will find the segments of streamlines that go from one region of interest (ROI) to the other, and will cut the points that are going past these ROIs.

.. code-block:: bash

    scil_tractogram_cut_streamlines $tractogram1 cut_streamlines2.trk \
        --labels $labels --label_ids 16 2024

