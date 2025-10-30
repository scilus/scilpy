.. _ssst_fodf:

Single-shell single-tissue fODF (ssst-fODF)
===========================================

This tutorial explains how to compute single-shell single-tissue fiber orientation distribution functions (fODFs) using single-shell single-tissue constrained spherical deconvolution (ssst-CSD). If there are multiple b-values in your data, you might want to consider using multi-shell multi-tissue CSD (msmt-CSD) instead. See the :ref:`msmt_fodf` instructions for that. The following instructions are specific to single-shell single-tissue CSD.

Preparing data for this tutorial
********************************

The tutorial data is still in preparation, meanwhile you can use this:

.. code-block:: bash

    # For now, let's use data in .scilpy
    scil_data_download -v ERROR
    in_dir=$in_dir/ssst
    mkdir $in_dir
    cp $HOME/.scilpy/processing/dwi_crop.nii.gz $in_dir/dwi.nii.gz
    cp $HOME/.scilpy/processing/dwi.bval $in_dir/dwi.bval
    cp $HOME/.scilpy/processing/dwi.bvec $in_dir/dwi.bvec
    cp $HOME/.scilpy/processing/fa_thr.nii.gz $in_dir/mask.nii.gz
    cp $HOME/.scilpy/processing/fa_thr.nii.gz $in_dir/wm_mask.nii.gz

.. tip::
    You may download the complete bash script to run the whole tutorial in one step `here </_static/bash/reconst/ssst_fodf.sh>`_.


1. Computing the frf
********************

The first step towards computing fODFs using constrained spherical deconvolution (CSD) is to compute the fiber response function (FRF) using :ref:`scil_frf_ssst`. This script should run fast (a few seconds on a full brain).

.. code-block:: bash

    scil_frf_ssst $in_dir/dwi.nii.gz $in_dir/dwi.bval $in_dir/dwi.bvec frf.txt \
        --mask $in_dir/mask.nii.gz --mask_wm $in_dir/wm_mask.nii.gz -v

Our tutorial data currently do not contain a brain mask, but you could use ``--mask brainmask.nii.gz`` for a faster processing time, and ``--mask_wm wm.nii.gz`` for more precision.

The default parameters should work well in most cases. The script will output the FRF in a text file (frf.txt) that will be used in the next step. The first three numbers in the file are the parallel diffusivity and the perpendicular diffusivity (written twice) of the single fiber population. These should typically be around 1.2-2.0 x 10^-3 mm^2/s and 0.25-0.5 x 10^-3 mm^2/s, respectively. The last number is the average value of the b0 signal of the single fiber population. If the FRF looks very different from these values or if you get an error from the script, you might be able to resolve the issue by changing some parameters. For instance, you can change the threshold for the FA mask (you don't need to provide that mask, it will be computed in the process), using the ``--fa_thresh`` option (default is 0.7). You can also change the minimum number of voxels required to compute the FRF using the ``--min_nvox`` option (default is 300). This is particularly helpful in the case of small images. In such cases, the ``--roi_radii`` and ``--roi_center`` options can also be used to specify a region of interest (ROI) in the white matter. In any case, the ``-v`` (verbose) option can be used to get more information about the process. Once you have computed the FRF, you can proceed to compute the fODFs.


2. Computing the fODF
*********************

The second step is to perform single-shell single-tissue CSD (ssst-CSD) using :ref:`scil_fodf_ssst`, based on Tournier et al. NeuroImage 2007, "Robust determination of the fibre orientation distribution in diffusion MRI: Non-negativity constrained super-resolved spherical deconvolution". This script should take longer (about 15 minutes on a full brain).

.. code-block:: bash

     scil_fodf_ssst $in_dir/dwi.nii.gz $in_dir/dwi.bval $in_dir/dwi.bvec frf.txt fodf.nii.gz \
        --mask $in_dir/mask.nii.gz

The script will output the fODFs in a nifti file (fodf.nii.gz). The only optional arguments are the ``--sh_order`` option (default is 8) to set the maximum spherical harmonics order used to represent the fODFs and the ``--sh_basis`` option (default is 'descoteaux07') to set the spherical harmonics basis. The ``--processes`` option is used to speed up the computation by using multiple CPU cores. To visualize the fODFs, you can use :ref:`scil_viz_fodf`.