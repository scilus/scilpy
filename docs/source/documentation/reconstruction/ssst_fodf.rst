.. _ssst_fodf:

Instructions to compute single-shell single-tissue fODF (ssst-fODF)
===================================================================


This tutorial explains how to compute single-shell single-tissue fiber orientation distribution functions (fODFs) using single-shell single-tissue constrained spherical deconvolution (ssst-CSD). If there are multiple b-values in your data, you might want to consider using multi-shell multi-tissue CSD (msmt-CSD) instead. See the :ref:`msmt_fodf` instructions for that. The following instructions are specific to single-shell single-tissue CSD.

The first step towards computing fODFs using constrained spherical deconvolution (CSD) is to compute the fiber response function (FRF) using :ref:`scil_frf_ssst`. This script should run fast (a few seconds on a full brain).
::

    scil_frf_ssst dwi.nii.gz dwi.bval dwi.bvec frf.txt \
    --mask brainmask.nii.gz --mask_wm wm_mask.nii.gz -f

The default parameters should work well in most cases. The script will output the FRF in a text file (frf.txt) that will be used in the next step. The first three numbers in the file are the parallel diffusivity and the perpendicular diffusivity (written twice) of the single fiber population. These should typically be around 1.2-2.0 x 10^-3 mm^2/s and 0.25-0.5 x 10^-3 mm^2/s, respectively. The last number is the average value of the b0 signal of the single fiber population. If the FRF looks very different from these values or if you get an error from the script, you might be able to resolve the issue by changing some parameters. For instance, you can change the threshold for the FA mask using the ``--fa_thresh`` option (default is 0.7). You can also change the minimum number of voxels required to compute the FRF using the ``--min_nvox`` option (default is 300). This is particularly helpful in the case of small images. In such cases, the ``--roi_radii`` and ``--roi_center`` options can also be used to specify a region of interest (ROI) in the white matter. In any case, the ``-v`` (verbose) option can be used to get more information about the process. Once you have computed the FRF, you can proceed to compute the fODFs.

The second step is to perform single-shell single-tissue CSD (ssst-CSD) using :ref:`scil_fodf_ssst`, based on Tournier et al. NeuroImage 2007, "Robust determination of the fibre orientation distribution in diffusion MRI: Non-negativity constrained super-resolved spherical deconvolution". This script should take longer (about 15 minutes on a full brain).
::

    scil_fodf_ssst dwi.nii.gz dwi.bval dwi.bvec frf.txt fodf.nii.gz \
    --mask brainmask.nii.gz -f

The script will output the fODFs in a nifti file (fodf.nii.gz). The only optional arguments are the ``--sh_order`` option (default is 8) to set the maximum spherical harmonics order used to represent the fODFs and the ``--sh_basis`` option (default is 'descoteaux07') to set the spherical harmonics basis. The ``--processes`` option is used to speed up the computation by using multiple CPU cores. To visualize the fODFs, you can use :ref:`scil_viz_fodf`.