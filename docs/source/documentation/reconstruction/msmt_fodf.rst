.. _msmt_fodf:

Instructions to compute multi-shell multi-tissue fODF (msmt-fODF)
=================================================================


This tutorial explains how to compute multi-shell multi-tissue fiber orientation distribution functions (fODFs) using multi-shell multi-tissue constrained spherical deconvolution (msmt-CSD). If your data only contains a single b-value, you might want to consider using single-shell single-tissue CSD (ssst-CSD) instead. See the :ref:`ssst_fodf` instructions for that. The following instructions are specific to multi-shell multi-tissue CSD.

The first step towards computing fODFs using constrained spherical deconvolution (CSD) is to compute the fiber response functions (FRFs) using :ref:`scil_frf_msmt`. This script should run fast (a few seconds on a full brain).
::

    scil_frf_msmt dwi.nii.gz dwi.bval dwi.bvec wm_frf.txt gm_frf.txt csf_frf.txt --mask brainmask.nii.gz --mask_wm wm_mask.nii.gz --mask_gm gm_mask.nii.gz --mask_csf csf_mask.nii.gz -f

The default parameters should work well in most cases. The script will output the FRFs in three text files (wm_frf.txt, gm_frf.txt and csf_frf.txt) that will be used in the next step. The first three numbers in each file are the parallel diffusivity and the perpendicular diffusivity (written twice) of the corresponding tissue type. These should typically be around 1.2-2.0 x 10^-3 mm^2/s and 0.25-0.5 x 10^-3 mm^2/s for white matter, 0.6-0.9 x 10^-3 mm^2/s and 0.4-0.6 x 10^-3 mm^2/s for gray matter, and 2.5-3.0 x 10^-3 mm^2/s and 2.5-3.0 x 10^-3 mm^2/s for cerebrospinal fluid, respectively. The last number is the average value of the b0 signal of the corresponding tissue type. If the FRFs look very different from these values or if you get an error from the script, you might be able to resolve the issue by changing some parameters. For instance, you can change the threshold for the FA mask using the ``--fa_thresh`` option (default is 0.7). You can also change the minimum number of voxels required to compute each FRF using the ``--min_nvox`` option (default is 300). This is particularly helpful in the case of small images. In such cases, the ``--roi_radii`` and ``--roi_center`` options can also be used to specify regions of interest (ROIs) in each tissue type. In any case, the ``-v`` (verbose) option can be used to get more information about the process. Once you have computed the FRFs, you can proceed to compute the fODFs.