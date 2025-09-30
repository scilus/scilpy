.. _page_tractogram_segmentation:

Tractogram Segmentation into bundles
====================================

Scilpy scripts offer many options to segment a tractogram into bundles.

The resulting bundles can be cleaned more thoroughly than whole-brain tractograms, considering that their shapes should be quite uniform, and spurious streamlines can be discarded automatically with scil_bundle_reject_outliers or visually with :ref:`scil_bundle_clean_qbx_clusters`. The bundle could also be cut or trimmed using a binary mask, allowing it to focus on a specific region, using scil_tractogram_cut_streamlines. Optionally, metrics and statistics can be measured for each individual subject using our various scripts for bundle analysis, such as presented in :ref:`profilometry`.

Examples of usage
-----------------

- Creating a template. See :ref:`page_population_template`.