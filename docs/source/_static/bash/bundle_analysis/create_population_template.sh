#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

output_path="where/you/want/to/save/outputs"
input_data="where/you/downloaded/data"
MNI=$input_data/mni_masked.nii.gz

cd $output_path

echo "Step A. Prepare the bundle of interest in each subject"
echo "******************************************************"

for subj in "subj1 subj2"
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



echo "Step B. Quantify inter-subject variability"
echo "******************************************"
scil_bundle_pairwise_comparison $output_path/*/CST_MNI.tck \
        $output_path/cst_stats.json --reference $MNI

echo "Step C. Combine all subjects into a population template"
echo "*******************************************************"

# 1) Merge all CST files from all subjects together
scil_tractogram_math union $output_path/*/CST_MNI_resampled1000.tck \
    $output_path/merged_CST_MNI.tck --reference $MNI

# 2) Compute the density map
scil_tractogram_compute_density_map $output_path/merged_CST_MNI.tck \
    $output_path/merged_CST_MNI_density.nii.gz --reference $MNI

# 3) We color the streamlines with the values of the density map to create
#    the figure shown in step 7 of the figure.
scil_tractogram_assign_custom_color $output_path/merged_CST_MNI.tck \
    $output_path/merged_CST_MNI.tck --reference $MNI -f \
    --from_anatomy $output_path/merged_CST_MNI_density.nii.gz
