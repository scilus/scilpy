.. _page_tractogram_math:

Tractogram manipulations: logical operations and filtering
==========================================================

Scilpy scripts allow users to modify tractograms in various ways based on their needs, such as shown on Figure 2 in our upcoming paper:


The range of possibilities is wide!
-----------------------------------

- Mathematical operations on two tractograms such as union, intersection and difference can be performed through the :ref:`scil_tractogram_math` script.

- Some scripts allow operations on the tractogram as a whole object in the brain, such as flipping it on a chosen axis (:ref:`scil_tractogram_flip`) or creating a map of all voxels touched by a streamline (:ref:`scil_tractogram_compute_density_map`).

- Other scripts allow operations on the tractogram as set of streamlines, such as resampling the number of streamlines (:ref:`scil_tractogram_resample, scil_tractogram_split`), separating streamlines based on various criteria  (:ref:`scil_tractogram_filter_by_roi, scil_tractogram_filter_by_anatomy, scil_tractogram_filter_by_length, scil_tractogram_filter_by_orientation`) or segmenting a tractogram into bundles (see :ref:`page_tractogram_segmentation`).

- Finally, other scripts allow modifying the streamlines themselves, for instance by resampling the number of points on each streamline (:ref:`scil_tractogram_resample_nb_points, scil_tractogram_compress`), or smoothing the streamlines’ trajectories (:ref:`scil_tractogram_smooth`). See page :ref:`page_streamlines_math` for more information.


Examples (based on figure 2 in scilpy paper)
--------------------------------------------

Here is how to reproduce figure 2 in our upcoming paper.

.. image:: /_static/images/scilpy_paper_figure2.png
   :alt: Figure 2 in upcoming paper.

For the coloring of tractograms, see :ref:`page_viz_colors`. For an example of streamlines cutting, see :ref:`page_streamlines_math`.


Preparing data for this tutorial
********************************

You can find data at?

.. tip::
    You may download the complete bash script to run the whole tutorial in one step `here </_static/bash/tractogram_manipulation/tractogram_math.sh>`_.


Logical operations
******************

**1. Merging two tractograms (using all streamlines from both).**

If a streamline is exactly the same in both tractograms, it will be kept once.

.. code-block:: bash

    scil_tractogram_count_streamlines $tractogram1
    scil_tractogram_count_streamlines $tractogram2
    scil_tractogram_math union $tractogram1 $tractogram2 union.trk
    scil_tractogram_count_streamlines union.trk

**2. Finding streamlines belonging in two tractograms (intersection)**

.. code-block:: bash

    scil_tractogram_math intersection $tractogram1 $tractogram2 intersection.trk

**3. Finding streamlines that are in $tractogram1 but not in $tractogram2**

.. code-block:: bash

    scil_tractogram_math difference $tractogram1 $tractogram2 difference1-2.trk


Tractogram resampling
*********************

**1.  Downsampling: keeping only 20 streamlines**

.. code-block:: bash

    scil_tractogram_resample $tractogram1 20 tractogram_downsampled.trk

It is also possible to downsample per Quickbundles cluster. This will ensure that the selected streamlines are not too random and cover more or less the whole tractogram range.

.. code-block:: bash

    scil_tractogram_resample $tractogram1 20 tractogram_downsampled.trk --downsample_per_cluster

See the difference between the two calls above:

+------------------------------------------+----------------------------------------------+
| Random selection                         | Selection per Quickbundle cluster            |
+==========================================+==============================================+
| .. image:: /??                           | .. image:: ?                                 |
|    :width: 35%                           |    :width: 35%                               |
|    :align: center                        |    :align: center                            |
+------------------------------------------+----------------------------------------------+

**2. Upsampling:**

Uses randomly picked streamlines to create new similar ones by moving a little bit each point. We will also give a --tube_radius option to make sure final streamlines do not deviate too much from original trajectory.

.. code-block:: bash

    scil_tractogram_resample $tractogram1 1000 --point_wise_std --tube_radius

**3. Splitting a tractogram**

We will split our tractogram into 3 subsections. This line of code below will create 3 files: ``out_split_part_0.trk, out_split_part_1.trk, out_split_part_2.trk``.

.. code-block:: bash

    scil_tractogram_split --nb_chunks 3 $tractogram1 out_split_3_part

It is also possible, instead, to split into files of a given number of streamlines. For instance, this will create N files of 100 streamlines:

.. code-block:: bash

    scil_tractogram_split --chunk_size 100 $tractogram1 out_split_100_part

Finally, it is again possible to split per Quickbundles cluster.

.. code-block:: bash

    scil_tractogram_split --nb_chunk 2 --split_per_cluster $tractogram1 out_split_2QB_part


Tractogram flipping
*******************

To flip a tractogram on an axis (x, y, z; here we chose the x-axis), you can use:

.. code-block:: bash

    scil_tractogram_flip $tractogram1 flipped_x.trk x
