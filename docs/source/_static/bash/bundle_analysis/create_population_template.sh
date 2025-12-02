#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

# ==============
# How to run this script
#    1) Load the input data
#       https://scilpy.readthedocs.io/en/latest/documentation/getting_started.html
#    2) Call this script with
#    --->   bash aodf_scripts.sh  path/to/your/data  path/to/save/outputs
# ==============
in_dir=$1
out_dir=$2


# Let's prepare the data
MNI=$in_dir/mni_masked.nii.gz

# Current wmparc is in float. Should be in int. Let's convert.
scil_volume_math convert $in_dir/sub-01/sub-01__wmparc.nii.gz \
    --data_type int16  -f $in_dir/sub-01/sub-01__wmparc.nii.gz


# Let's run the tutorial
cd $out_dir

echo "Step A. Prepare the bundle of interest in each subject"
echo "******************************************************"

for subj in "sub-01" # "sub-02" ... If we have many subjects, we can loop on each one
do
    mkdir $subj
    echo "----> Processing subject $subj"

    # 1) Use any tool as you want to obtain a gray matter (GM) segmentation of
    #    your volume. Ex: Freesurfer. Here we already have wmparc.nii.gz.

    # 2) Split your volume into binary masks associated to each label.
    scil_labels_split_volume_by_ids $in_dir/$subj/${subj}__wmparc.nii.gz \
        --out_dir $subj/labels/ -v

    # 3) Segment the bundle using labels 2024 (ctx-rh-precentral) and 16 (Brain-Stem).
    #    The command below keeps all streamlines with at least one endpoint inside
    #    label 2024 and one endpoint inside label 16. This should select a CST bundle.
    #    The last numbers (3 and 2) are the maximum distance accepted for a endpoint
    #    to be considered inside the ROI.
    in_tractogram=$in_dir/$subj/${subj}_local_tractogram.trk
    out_tractogram=$subj/CST.tck
    scil_tractogram_filter_by_roi $in_tractogram $out_tractogram \
        --drawn_roi $subj/labels/2024.nii.gz either_end include 3 \
        --drawn_roi $subj/labels/16.nii.gz either_end include 2 -v

    # 4) Register to MNI space.
    #    You can use any tool for this, such as ANTs
    #    Here is how *we* created the transformation files.
    #    You need to have ANTs installed to run this:
    # mkdir $in_dir/$subj/transfo/
    # antsRegistrationSyNQuick.sh -d 3 -m $MNI \
    #     -f $in_dir/$subj/${subj}__fa.nii.gz -t r \
    #     -o $in_dir/$subj/transfo/MNI_ -n 4
    transfo=$in_dir/$subj/transfo/MNI_0GenericAffine.mat

    # 5) Apply the transformation to your tractogram.
    #    Uses the ANTs transformation. We used linear registration so we
    #    can use the .mat output.
    in_bundle=$subj/CST.tck
    out_bundle=$subj/CST_MNI.tck
    ref=$in_dir/$subj/${subj}__fa.nii.gz
    scil_tractogram_apply_transform $in_bundle $MNI $transfo $out_bundle \
        --inverse --cut_invalid --reference $ref

    # 6) (optional) You could subsampled subsets of streamlines
    in_bundle=$subj/CST_MNI.tck
    out_bundle=$subj/CST_MNI_resampled1000.tck
    scil_tractogram_resample $in_bundle 1000 $out_bundle \
        --never_upsample --reference $MNI
done



echo "Step B. Quantify inter-subject variability"
echo "******************************************"
scil_bundle_pairwise_comparison $out_dir/*/CST_MNI.tck \
        $out_dir/cst_stats.json --reference $MNI

echo "Step C. Combine all subjects into a population template"
echo "*******************************************************"

# 1) Merge all CST files from all subjects together
scil_tractogram_math union $out_dir/*/CST_MNI_resampled1000.tck \
    $out_dir/merged_CST_MNI.tck --reference $MNI

# 2) Compute the density map
scil_tractogram_compute_density_map $out_dir/merged_CST_MNI.tck \
    $out_dir/merged_CST_MNI_density.nii.gz --reference $MNI

# 3) We color the streamlines with the values of the density map to create
#    the figure shown in step 7 of the figure.
scil_tractogram_assign_custom_color $out_dir/merged_CST_MNI.tck \
    $out_dir/merged_CST_MNI.tck --reference $MNI -f \
    --from_anatomy $out_dir/merged_CST_MNI_density.nii.gz
