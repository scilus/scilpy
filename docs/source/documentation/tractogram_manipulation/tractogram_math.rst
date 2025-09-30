.. _page_tractogram_math:

Tractogram manipulations: logical operations and filtering
==========================================================

Scilpy scripts allow users to modify tractograms in various ways based on their needs, such as shown on Figure 2 in our upcoming paper:


The range of possibilities is wide!
-----------------------------------
Mathematical operations on two tractograms such as union, intersection and difference can be performed through the `scil_tractogram_math` script.

- Some scripts allow operations on the tractogram as a whole object in the brain, such as flipping it on a chosen axis (`scil_tractogram_flip`) or creating a map of all voxels touched by a streamline (`scil_tractogram_compute_density_map`).

- Other scripts allow operations on the tractogram as set of streamlines, such as resampling the number of streamlines (`scil_tractogram_resample, scil_tractogram_split`), separating streamlines based on various criteria  (`scil_tractogram_filter_by_roi, scil_tractogram_filter_by_anatomy, scil_tractogram_filter_by_length, scil_tractogram_filter_by_orientation`) or segmenting a tractogram into bundles (see :ref:`page_tractogram_segmentation`).

- Finally, other scripts allow modifying the streamlines themselves, for instance by resampling the number of points on each streamline (`scil_tractogram_resample_nb_points, scil_tractogram_compress`), or smoothing the streamlines’ trajectories (`scil_tractogram_smooth`).


Examples (based on figure 2 in scilpy paper)
--------------------------------------------

Here is how to reproduce figure 2 in our upcoming paper.

.. image:: ../../_static/scilpy_paper_figure2.png
   :alt: Figure 2 in upcoming paper.
