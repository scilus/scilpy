.. _page_population_template:

Creating a WM bundle population template
========================================

Scilpy scripts enable users to create a WM bundle population template, such as described in figure 3 in our upcoming paper. Such a pipeline includes:

1) segmenting the bundle of interest in each subject’s tractogram. The segmentation of bundles can be based on ROIs of inclusion / exclusion (:ref:`scil_tractogram_segment_with_ROI_and_score` or :ref:`scil_tractogram_filter_by_roi`) or based on the general shape of the streamlines (:ref:`scil_tractogram_segment_with_recobundles`, :ref:`scil_tractogram_segment_with_bundleseg`). See :ref:`page_tractogram_segmentation` for more information on this.

2) registering the bundles to a reference space (e.g., MNI space) and analysing the inter-subject variability,

3) combining them into a reference bundle template, similarly to how one would average many structural MRI images to create a brain template.

4) analysing the results.

Then, registration can be performed (see :ref:`page_tractogram_registration`). The tractograms can be downsampled and concatenated (see :ref:`page_tractogram_math`) and even concatenated to their flipped version to obtain a symmetrical template.


.. image:: ../../_static/scilpy_paper_figure3.png
   :alt: Figure 3 in upcoming paper.
   :width: 75%

Let's download data, you can find it here: ?. The organization, for each subject, is:
::

    ├── input_data
    │   ├── mni_masked.nii.gz
    │   ├── subjX
    │   │   └── wmparc.nii.gz                 # A segmentation from Freesurfer
    │   │   └── tractogram_filtered.tck       # A tractogram (prob tracking, filtered)
    │   │   └── fa.nii.gz                     # Our anatomy of reference.


The labels come from a Freesurfer segmentation (https://surfer.nmr.mgh.harvard.edu/ ), and the labels that it contains are found here: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT .

The tractogram files are .tck files, which do no contain headers (contrary to .trk files). We will need to add additional ``--reference`` options with an anatomy file (here fa.nii.gz).

Here are the command lines (ex, my_bash.sh).

Step A. Prepare the bundle of interest in each subject
******************************************************

.. code-block:: bash

    output_path="where/you/want/to/save/outputs"
    input_data="where/you/downloaded/data"
    MNI=$input_data/mni_masked.nii.gz

    cd $output_path

    for subj = "subj1 subj2"
    do
        mkdir $subj

        # 1) Use any tool as you want to obtain a gray matter (GM) segmentation of
        #    your volume. Ex: Freesurfer. Here we already have wmparc.nii.gz.

        # 2) Split your volume into binary masks associated to each label.
        scil_labels_split_volume_by_ids $input_data/$subj/wmparc.nii.gz \
            --out_dir $subj/labels/

        # 3) Segment the bundle using labels 2024 (ctx-rh-precentral) and 16 (Brain-Stem).
        #    The command below keeps all streamlines with at least one endpoint inside
        #    label 2024 and one endpoint inside label 16. This should select a CST bundle.
        #    The last numbers (3 and 2) are the maximum distance accepted for a endpoint
        #    to be considered inside the ROI.
        in_tractogram=$input_data/$subj/tractogram.tck
        out_tractogram=$subj/CST.tck
        scil_tractogram_filter_by_roi $in_tractogram $out_tractogram \
            --drawn_roi $subj/labels/2024.nii.gz either_end include 3 \
            --drawn_roi $subj/labels/16.nii.gz either_end include 2 \
            --reference fa.nii.gz

        # 4) Register to MNI space.
        #    You can use any tool for this, such as ANTs
        antsRegistrationSyNQuick.sh -d 3 -m $MNI \
            -f $input_data/$subj/fa.nii.gz -t r -o $subj/MNI_ -n 4

        # 5) Apply the transformation to your tractogram.
        #    Uses the ANTs transformation. We used linear registration so we
        #    can use the .mat output.
        transfo=$subj/MNI_0GenericAffine.mat
        in_bundle=$subj/CST.tck
        out_bundle=$subj/CST_MNI.tck
        scil_tractogram_apply_transform $in_bundle $MNI $out_bundle \
            --inverse --cut_invalid --reference $input_data/$subj/fa.nii.gz

        # 6) (optional) You could subsampled subsets of streamlines
        in_bundle=$subj/CST_MNI.tck
        out_bundle=$subj/CST_MNI_resampled1000.tck
        scil_tractogram_resample $in_bundle 1000 $out_bundle \
            --never_upsample --reference $MNI
    done

Step B. Quantify inter-subject variability
******************************************

We may quantify the overlap between all bundles across subjects.

.. code-block:: bash

    scil_bundle_pairwise_comparison $output_path/*/CST_MNI.tck \
        $output_path/cst_stats.json --reference $MNI


Step C. Combine all subjects into a population template
*******************************************************

Let's combine all streamlines from all subjects and visualize the result.

.. code-block:: bash

    # 1) Merge all CST files from all subjects together
    scil_tractogram_math union $output_path/*/CST_MNI_resampled1000.tck \
        $output_path/merged_CST_MNI.tck --reference $MNI

    # 2) Compute the density map
    scil_tractogram_compute_density_map $output_path/merged_CST_MNI.tck \
        $output_path/merged_CST_MNI_density.nii.gz --reference $MNI

    # 3) We color the streamlines with the values of the density map to create
    #    the figure shown in step 7 of the figure.
    scil_tractogram_assign_custom_color $output_path/merged_CST_MNI.tck $output_path/merged_CST_MNI.tck --reference $MNI -f \
     --from_anatomy $output_path/merged_CST_MNI_density.nii.gz
