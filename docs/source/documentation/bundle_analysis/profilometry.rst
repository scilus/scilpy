.. _profilometry:

Profilometry (statistics on bundles)
====================================

Using a clean bundle from a single subject, scilpy allows performing a profilometry [1]_ analysis, which is the analysis of the evolution of any dMRI metrics along its subsections. A simple example is shown in Figure 4, and for further examples of such processes, users can refer to Tractometry-flow (https://github.com/scilus/tractometry_flow), a nextflow process using many scilpy scripts, or to the study by Cousineau et. al [2]_.

The creation of the bundles required tractogram segmentation (:ref:`page_tractogram_segmentation`).

The profilometry analysis requires segmenting the bundles into as many subsections as desired. This is not straightforward, as the division of sections can be performed arbitrarily. We generally cut the section perpendicularly to the direction of the bundle, measured from a centroid, a single streamline-like shape representing the average of all streamlines in the bundle (`scil_bundle_compute_centroid`, `scil_bundle_label_map`). Then, any metric can be associated with the subsections. Figure 4 shows the example of the sections’ volume and diameter, or mean underlying value of any given map, such as the FA map. The resulting json file’s values can be plotted with scil_plot_stats_per_point.

Beyond profilometry, other analysis steps could also be performed on the whole bundle. For instance, it is possible to compute the map of the head and tail of a bundle (the starting and ending regions of the bundle, assuming all streamlines are aligned in the same direction) with scil_bundle_uniformize_endpoints and scil_bundle_compute_endpoints_map. It is also possible to extract various information with `scil_bundle_shape_measures`.

Example: reproducing figure 4 in scilpy paper
---------------------------------------------

.. image:: /_static/images/scilpy_paper_figure4.png
   :alt: Figure 4 in upcoming paper.


.. [1] Yeatman JD, Dougherty RF, Myall NJ, Wandell BA, Feldman HM. Tract Profiles of White Matter Properties: Automating Fiber-Tract Quantification. PLOS ONE. 2012;7(11):e49790. doi:10.1371/journal.pone.0049790

.. [2] Cousineau M, Jodoin PM, Garyfallidis E, et al. A test-retest study on Parkinson’s PPMI dataset yields statistically significant white matter fascicles. NeuroImage Clin. 2017;16:222-233. doi:10.1016/j.nicl.2017.07.020
