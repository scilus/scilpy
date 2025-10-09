
scil_frf_ssst dwi.nii.gz dwi.bval dwi.bvec frf.txt \
    --mask brainmask.nii.gz --mask_wm wm_mask.nii.gz -f


scil_fodf_ssst dwi.nii.gz dwi.bval dwi.bvec frf.txt fodf.nii.gz \
    --mask brainmask.nii.gz -f
