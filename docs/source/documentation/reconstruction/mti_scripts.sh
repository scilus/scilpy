#!/usr/bin/env bash
set -euo pipefail  # Will fail on error

# ==============
# How to run this script
#    1) Load the input data
#       https://scilpy.readthedocs.io/en/latest/documentation/getting_started.html
#    2) Call this script with
#    --->   bash btensor_scripts.sh  path/to/your/data  path/to/save/outputs
# ==============
in_dir=$1
out_folder=$2


# For now, let's use data in .scilpy
scil_data_download
?

# ==============
# Now let's run the tutorial
# ==============
cd $out_folder

# 1. Prepare input data
dcm2bids -d DICOM_folder -p id_subject -c config.json -o sub-id

# 2. Basic usage
scil_mti_maps_ihMT path/to/output/directory \
        --in_altnp path/to/*altnp.nii.gz \
        --in_altpn path/to/*altpn.nii.gz \
        --in_negative path/to/*neg.nii.gz \
        --in_positive path/to/echo*pos.nii.gz \
        --in_mtoff_pd path/to/echo*mtoff.nii.gz \
        --in_mtoff_t1 path/to/echo*T1w.nii.gz \
        --mask path/to/mask_bin.nii.gz \
        --in_jsons path/to/echo*mtoff.json path/to/echo*T1w.json

# 3. Through workflows
git clone https://github.com/scilus/ihmt_flow.git
nextflow run ihmt_flow/main.nf --input /path/to/data --output /path/to/results -profile singularity


# Or:
scil_mti_maps_MT path/to/output/directory \
    --in_positive path/to/echo*pos.nii.gz \
    --in_negative path/to/echo*neg.nii.gz \
    --in_mtoff_pd path/to/echo*mtoff.nii.gz \
    --in_mtoff_t1 path/to/echo*T1w.nii.gz \
    --mask path/to/mask_bin.nii.gz \
    --in_jsons path/to/echo*mtoff.json path/to/echo*T1w.json

