Instructions for tensor-valued dMRI scripts (b-tensor)
======================================================


The scripts for multi-encoding multi-shell multi-tissue CSD (memsmt-CSD) are based on P. Karan et al., Bridging the gap between constrained spherical deconvolution and diffusional variance decomposition via tensor-valued diffusion MRI. Medical Image Analysis (2022). We recommend reading it to understand the scope of the memsmt-CSD problem.

If you want to do CSD with b-tensor data, you should start by computing the fiber response functions. This script should run fast (less than 5 minutes on a full brain).
::

    scil_frf_memsmt.py wm_frf.txt gm_frf.txt csf_frf.txt --in_dwis dwi_linear.nii.gz dwi_planar.nii.gz dwi_spherical.nii.gz --in_bvals dwi_linear.bval dwi_planar.bval dwi_spherical.bval --in_bvecs dwi_linear.bvec dwi_planar.bvec dwi_spherical.bvec --in_bdeltas 1 -0.5 0 --mask mask.nii.gz --mask_wm wm_mask.nii.gz --mask_gm gm_mask.nii.gz --mask_csf csf_mask.nii.gz -f

Then, you should compute the fODFs and volume fractions. The following command will save a fODF file for each tissue and a volume fractions file. This script should run in about 1-2 hours for a full brain.
::

    scil_fodf_memsmt.py wm_frf.txt gm_frf.txt csf_frf.txt --in_dwis dwi_linear.nii.gz dwi_planar.nii.gz dwi_spherical.nii.gz --in_bvals dwi_linear.bval dwi_planar.bval dwi_spherical.bval --in_bvecs dwi_linear.bvec dwi_planar.bvec dwi_spherical.bvec --in_bdeltas 1 -0.5 0 --mask mask.nii.gz --processes 8 -f

If you want to do DIVIDE with b-tensor data, you should use the following command. It will save files for the MD, uFA, OP, MK_I, MK_A and MK_T. This script should run in about 1-2 hours for a full brain.
::

    scil_btensor_metrics.py --in_dwis dwi_linear.nii.gz dwi_planar.nii.gz dwi_spherical.nii.gz --in_bvals dwi_linear.bval dwi_planar.bval dwi_spherical.bval --in_bvecs dwi_linear.bvec dwi_planar.bvec dwi_spherical.bvec --in_bdeltas 1 -0.5 0 --mask mask.nii.gz --fa fa.nii.gz --processes 8 -f