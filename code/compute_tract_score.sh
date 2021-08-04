#!/bin/bash

# ---------- SUMMARY ----------
# 
# e.g ../compute_tract_score.sh sharp_sigma_effect/fodf wm.nii.gz bundles rois sharp_sigma_effect/output


usage() {
  echo "$(basename $0) \
        [in_fodf] \
        [in_mask] \
 	[in_score_bundles] \
        [in_score_rois] \
        [out_path] "; exit 1;
}

in_fodf="" 
in_mask=""
in_score_bundles=""
in_score_rois=""
out_path=""


# Parse input arguments
PARAMS=""

while (( "$#" )); do
  case "$1" in
    -h)
      usage
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

# Set positional arguments in their proper place
eval set -- "$PARAMS"

in_fodf=$1
in_mask=$2
in_score_bundles=$3
in_score_rois=$4
out_path=$5

echo

# Create prefixs and necessary folders

sft_path=${out_path}/sft
mkdir -p ${sft_path}

json_path=${out_path}/json
mkdir -p ${json_path}

score_path=${out_path}/score
mkdir -p ${score_path}

# Start computing

echo "Output will be written to folder $out_path"
echo "Starting to compute on $in_fodf..."


# Compute fodf and masks at every voxel size desired

for fodf in ${in_fodf}/*nii.gz
do
	echo
  	echo Tracking of $fodf ...
	echo
  
  	fodf_name="$(basename ${fodf})"
	fodf_name="${fodf_name%%.*}"
	
	score_sft_path=${score_path}/${fodf_name}
	mkdir -p ${score_sft_path}

	# Compute tracking
	scil_compute_tracking_multiresolution.py \
	${fodf} \
	${in_mask} \
	${in_mask} \
	${sft_path}/${fodf_name}.trk \
	--npv 1 --step 0.2 --algo prob \
	--sfthres 0 
	
	scil_score_tractogram.py \
	${sft_path}/${fodf_name}.trk \
	${in_score_bundles}/FiberCupGroundTruth_filtered_bundle_*.nii.gz \
	--gt_heads ${in_score_rois}/FiberCupGroundTruth_filtered_bundle_*_head.nii.gz \
	--gt_tails ${in_score_rois}/FiberCupGroundTruth_filtered_bundle_*_tail.nii.gz \
	--out_dir ${score_sft_path} \
	--dilate_endpoints 1 
	
	mv ${score_sft_path}/results.json ${json_path}/${fodf_name}.json
	
done

../graph_overall.py --in_score ${json_path}/*.json --out_score ${out_path}/graph_overall.png

../bundles_fc_tc.py --in_score ${json_path}/*.json --out_score ${out_path}/graph_bundle.png


echo "Done. Final graphs are in $out_path"


