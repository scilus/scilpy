#!/bin/bash

# ---------- SUMMARY ----------
# This script computes multiresolution tractography, 
# in order to better reconstruct important crossing sections. 

# Compute usual tracking (to compare)
# Compute fodf and masks at every voxel size desired 
# Compute priors and tractogram (pyramid!)
# Score both final tractograms
 
# It also allows priors and fodf visualisation. 

# e.g ./compute_multiresolution.sh fodf_8_descoteaux.nii.gz wm.nii.gz bundles rois dwi out_path 2 0 8000 

# e.g ./compute_multiresolution.sh fodf_8_descoteaux.nii.gz wm.nii.gz bundles rois dwi test_sigma2 2 0 8000 

usage() {
  echo "$(basename $0) \
        [in_fodf] \
        [in_mask] \
        [in_score_bundles] \
        [in_score_rois] \
        [in_dwi] \
        [out_path] \
        [sigma] \
        [sharp] \
        [nt]"; exit 1;
}

in_fodf="" 
in_mask=""
in_score_bundles=""
in_score_rois=""
in_dwi="" 			
out_path="" 
sigma=""
sharp=""           
nt=""

step=0.2
algo='prob'
interp='cubic'
# VOXEL_SIZE MUST BE IN DESCENDING ORDER
voxel_size=(6 5 4 3)

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
in_dwi=$5
out_path=$6
sigma=$7
sharp=$8
nt=$9

echo

# Create prefixs and necessary folders

fodf_name="$(basename ${in_fodf})"
fodf_name="${fodf_name%%.*}"

mask_name="$(basename ${in_mask})"
mask_name="${mask_name%%.*}"

masks_path=${out_path}/masks
mkdir -p ${masks_path}

fodf_path=${out_path}/fodf
mkdir -p ${fodf_path}

sft_path=${out_path}/sft
mkdir -p ${sft_path}

priors_path=${out_path}/priors
mkdir -p ${priors_path}

visu_path=${out_path}/visu
mkdir -p ${visu_path}

commit_path=${sft_path}/commit
mkdir -p ${commit_path}

dwi_path=${out_path}/dwi
mkdir -p ${dwi_path}

score_path=${out_path}/score
mkdir -p ${score_path}

score_usual_path=${score_path}/usual
mkdir -p ${score_usual_path}

score_MR_path=${score_path}/MR
mkdir -p ${score_MR_path}


# Start computing

echo "Output will be written to folder $out_path"
echo "Starting to compute on $in_fodf..."

# Compute usual tracking (to compare)

scil_compute_local_tracking.py \
${in_fodf} \
${in_mask} \
${in_mask} \
${sft_path}/sft_${fodf_name}_usual.trk \
--nt ${nt} --step ${step} --algo ${algo} \
--sfthres 0 -f

#scil_run_commit.py \
#${sft_path}/sft_${fodf_name}_usual.trk \
#${in_dwi}/fibercup_b-1000_dwi.nii.gz \
#${in_dwi}/fibercup_b-1000_dwi.bval \
#${in_dwi}/fibercup_b-1000_dwi.bvec \
#${commit_path} \
#--in_peaks ${in_fodf} \
#--ball_stick -f 

#mv ${commit_path}/commit_1/essential_tractogram.trk ${sft_path}
#mv ${sft_path}/essential_tractogram.trk ${sft_path}/sft_${fodf_name}_usual_filtered.trk 

# Compute fodf and masks at every voxel size desired

for size in ${voxel_size[@]}
do
  echo Resampling to $size

  step_size=$(echo ${step}*${size} | bc)

	scil_resample_volume.py \
	${in_fodf} \
	${fodf_path}/${fodf_name}_size${size}.nii.gz \
  	--offset -0.0 --voxel_size ${size} --interp ${interp} -f
	
	scil_visualize_fodf.py \
	${fodf_path}/${fodf_name}_size${size}.nii.gz \
	--output ${visu_path}/fodf_size${size}.jpeg --silent \
  	--win_dims 1024 1024 -f
	
	scil_execute_asymmetric_filtering.py \
	${fodf_path}/${fodf_name}_size${size}.nii.gz  \
	${fodf_path}/${fodf_name}_size${size}_filtered.nii.gz \
	--sigma ${sigma} --sharpness ${sharp} --out_sym -f \
	#--edge_mode wall 

	scil_visualize_fodf.py \
	${fodf_path}/${fodf_name}_size${size}_filtered.nii.gz \
	--output ${visu_path}/fodf_size${size}_filtered.jpeg --silent \
  	--win_dims 1024 1024 -f

	scil_resample_volume.py \
	${in_mask} \
	${masks_path}/${mask_name}_size${size}_dtypef.nii.gz \
  	--offset -0.0 --voxel_size ${size} --interp ${interp} -f

	scil_image_math.py convert \
	${masks_path}/${mask_name}_size${size}_dtypef.nii.gz \
	${masks_path}/${mask_name}_size${size}.nii.gz \
	--data_type int16 -f
	
	scil_resample_volume.py \
	${in_dwi}/fibercup_b-1000_dwi.nii.gz \
	${dwi_path}/fibercup_b-1000_dwi_size${size}.nii.gz \
  	--offset -0.0 --voxel_size ${size} --interp ${interp} -f
	
done
# 
# Compute first tractogram used (with no priors)

step_size=$(echo ${step}*${voxel_size[0]} | bc)

scil_compute_local_tracking.py \
${fodf_path}/${fodf_name}_size${voxel_size[0]}_filtered.nii.gz  \
${masks_path}/${mask_name}_size${voxel_size[0]}.nii.gz \
${masks_path}/${mask_name}_size${voxel_size[0]}.nii.gz \
${sft_path}/sft_${fodf_name}_size${voxel_size[0]}.trk \
--nt ${nt} --step ${step_size} --algo ${algo} --sfthres 0 -f

scil_run_commit.py \
${sft_path}/sft_${fodf_name}_size${voxel_size[0]}.trk \
${dwi_path}/fibercup_b-1000_dwi_size${voxel_size[0]}.nii.gz \
${in_dwi}/fibercup_b-1000_dwi.bval \
${in_dwi}/fibercup_b-1000_dwi.bvec \
${commit_path} \
--in_peaks ${fodf_path}/${fodf_name}_size${voxel_size[0]}_filtered.nii.gz \
--ball_stick -f 

mv ${commit_path}/commit_1/essential_tractogram.trk ${sft_path}
mv ${sft_path}/essential_tractogram.trk ${sft_path}/sft_${fodf_name}_size${voxel_size[0]}_filtered.trk 

# Compute priors and tractogram (pyramid!)
prec_size=${voxel_size[0]}

for size in ${voxel_size[@]:1}
do
  echo Computing tracking and e-FOD for size $size
  step_size=$(echo ${step}*${size} | bc)

	scil_generate_priors_from_bundle.py \
	${sft_path}/sft_${fodf_name}_size${prec_size}_filtered.trk \
	${fodf_path}/${fodf_name}_size${size}_filtered.nii.gz \
	${masks_path}/${mask_name}_size${size}.nii.gz \
  	--todi_sigma 0 --sf_threshold 0 \
  	--out_dir ${priors_path} --out_prefix ${fodf_name}_size${size}_ -f
		
	scil_visualize_fodf.py \
	${priors_path}/${fodf_name}_size${size}_efod.nii.gz \
	--output ${visu_path}/efod_size${size}.jpeg --silent \
  	--win_dims 1024 1024 -f
			
	scil_visualize_fodf.py \
	${priors_path}/${fodf_name}_size${size}_priors.nii.gz \
	--output ${visu_path}/priors_size${size}.jpeg --silent \
  	--win_dims 1024 1024 -f
	 
	scil_compute_local_tracking.py \
	${priors_path}/${fodf_name}_size${size}_efod.nii.gz \
	${masks_path}/${mask_name}_size${size}.nii.gz \
	${masks_path}/${mask_name}_size${size}.nii.gz \
	${sft_path}/sft_${fodf_name}_size${size}.trk \
  	--nt ${nt} --step ${step_size} --algo ${algo} -f \
  	--sfthres 0
	
	scil_run_commit.py \
	${sft_path}/sft_${fodf_name}_size${size}.trk \
	${dwi_path}/fibercup_b-1000_dwi_size${size}.nii.gz \
	${in_dwi}/fibercup_b-1000_dwi.bval \
	${in_dwi}/fibercup_b-1000_dwi.bvec \
	${commit_path} \
	--in_peaks ${fodf_path}/${fodf_name}_size${size}_filtered.nii.gz \
	--ball_stick -f 
	
	mv ${commit_path}/commit_1/essential_tractogram.trk ${sft_path}
	mv ${sft_path}/essential_tractogram.trk ${sft_path}/sft_${fodf_name}_size${size}_filtered.trk 
	
	prec_size=${size}
done

# Score both final tractograms 

final_size=${voxel_size[-1]}

scil_score_tractogram.py \
${sft_path}/sft_${fodf_name}_size${final_size}.trk \
${in_score_bundles}/FiberCupGroundTruth_filtered_bundle_*.nii.gz \
--gt_heads ${in_score_rois}/FiberCupGroundTruth_filtered_bundle_*_head.nii.gz \
--gt_tails ${in_score_rois}/FiberCupGroundTruth_filtered_bundle_*_tail.nii.gz \
--out_dir ${score_MR_path} \
--dilate_endpoints 1 -f

scil_score_tractogram.py \
${sft_path}/sft_${fodf_name}_usual.trk \
${in_score_bundles}/FiberCupGroundTruth_filtered_bundle_*.nii.gz \
--gt_heads ${in_score_rois}/FiberCupGroundTruth_filtered_bundle_*_head.nii.gz \
--gt_tails ${in_score_rois}/FiberCupGroundTruth_filtered_bundle_*_tail.nii.gz \
--out_dir ${score_usual_path} \
--dilate_endpoints 1 -f


echo "Done. The final tractogram is sft_${fodf_name}_size${final_size}.trk, and is written in ${sft_path} "




