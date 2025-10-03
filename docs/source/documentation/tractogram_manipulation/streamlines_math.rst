.. _page_streamlines_math:

Streamlines manipulation (resampling, smoothing)
================================================

Here, we discuss scripts allowing modifying the streamlines themselves, for instance by resampling the number of points on each streamline (:ref:`scil_tractogram_resample_nb_points, scil_tractogram_compress`), or smoothing the streamlines’ trajectories (:ref:`scil_tractogram_smooth`).

See page :ref:`page_tractogram_math` for scripts modifying the tractograms as a whole.


Preparing data for this tutorial
********************************

(documentation upcoming)

.. tip::
    You may download the complete bash script to run the whole tutorial in one step `here </_static/bash/tractogram_manipulation/streamlines_math.sh>`_.

Resampling the number of points in each streamline
**************************************************

.. code-block:: bash

    scil_tractogram_resample_nb_points $tractogram1 20 resampled_20pts.trk

Compressing streamlines
***********************

.. code-block:: bash

    scil_tractogram_compress

Cutting streamlines
*******************

If you don't want to remove faulty streamlines, it is possible to simply cut streamlines that are too long (ex, cut the parts that are going out of a given mask) using the following command lines:

**1. Cutting the parts that are out of the mask**

The option ``--min_length`` will ensure that final streamlines after cutting are not too short (we chose 20mm).

.. code-block:: bash

    scil_tractogram_cut_streamlines $tractogram1 cut_streamlines.trk \
        --mask $mask --min_length 20

**2. Cutting the parts that are not inside two ROIs**

This will find the segments of streamlines that go from one region of interest (ROI) to the other, and will cut the points that are going past these ROIs.

.. code-block:: bash

    ROI1=3
    ROI2=6
    scil_tractogram_cut_streamlines $tractogram1 cut_streamlines.trk \
        --labels $my_parcellation --label_ids $ROI1 $ROI2
