.. _msmt_fodf:

Multi-shell multi-tissue fODF (msmt-fODF)
=========================================

This tutorial explains how to compute multi-shell multi-tissue fiber orientation distribution functions (fODFs) using multi-shell multi-tissue constrained spherical deconvolution (msmt-CSD) [multitissueCSD]_. If your data contains less than three b-values, you might want to consider using single-shell single-tissue CSD (ssst-CSD) instead. See the :ref:`ssst_fodf` instructions for that. The following instructions are specific to multi-shell and based on [multi-tissue_CSD]_.


Preparing data for this tutorial
********************************

To run lines below, you need a various volumes, b-vector information and masks. The tutorial data is still in preparation, meanwhile you can use this: `

.. code-block:: bash

    in_dir=where/you/downloaded/tutorial/data

    # For now, let's use data in .scilpy
    scil_data_download
    in_dir=$in_dir/msmt
    mkdir $in_dir
    cp $HOME/.scilpy/commit_amico/* $in_dir/

.. tip::
    You may download the complete bash script to run the whole tutorial in one step `here </_static/bash/reconst/msmt_scripts.sh>`_.

1. Computing the frf
********************

The first step towards computing fODFs using constrained spherical deconvolution (CSD) is to compute the fiber response functions (FRFs) using :ref:`scil_frf_msmt`. This script should run fast (a few seconds on a full brain). The data in this tutorial is small, with default parameters, we would get a warning (Could not find at least 100 voxels for the WM mask.), so we'll set the --min_nvox to 1.

.. code-block:: bash

    scil_frf_msmt $in_dir/dwi.nii.gz $in_dir/dwi.bval $in_dir/dwi.bvec \
        wm_frf.txt gm_frf.txt csf_frf.txt --mask mask.nii.gz -v --min_nvox 1

When available, it is possible to add following options (not available with current data):

.. code-block:: bash

        --mask_wm $in_dir/wm_mask.nii.gz --mask_gm $in_dir/gm_mask.nii.gz --mask_csf $in_dir/csf_mask.nii.gz

The script will output one FRF per tissue type: white matter (WM), gray matter (GM), cerebrospinal fluid (CSF), in three text files (wm_frf.txt, gm_frf.txt and csf_frf.txt) that will be used in the next step. Each line in those files correspond to the response function of a b-value (in decreasing order from top to bottom). The first three numbers in each line are the parallel diffusivity and the perpendicular diffusivity (written twice) of the corresponding tissue type and b-value. These should typically be around 1.2-2.0 x 10^-3 mm^2/s and 0.25-0.5 x 10^-3 mm^2/s for WM, 0.6-1.0 x 10^-3 mm^2/s and 0.5-0.8 x 10^-3 mm^2/s for GM, and 1.5-3.0 x 10^-3 mm^2/s and 1.5-3.0 x 10^-3 mm^2/s for CSF, respectively. The last number is the average value of the b0 signal of the corresponding tissue type. If the FRFs look very different from these values or if you get an error from the script, you might be able to resolve the issue by changing some parameters and making sure the inputed tissue masks are correct. Indeed, the script works better with a mask for each tissue type (WM, GM and CSF) in addition to the brain mask. Moreover, you can change the FA and MD thresholds of used to refine the tissue masks, using the ``--fa_thr_wm``, ``--fa_thr_gm``, ``--fa_thr_csf``, ``--md_thr_wm``, ``--md_thr_gm`` and ``--md_thr_csf`` options. You can also change the minimum number of voxels required to compute each FRF using the ``--min_nvox`` option (default is 100). This is particularly helpful in the case of small images. In such cases, the ``--roi_radii`` and ``--roi_center`` options can also be used to specify regions of interest (ROIs) in each tissue type. In any case, the ``-v`` (verbose) option can be used to get more information about the process. Once you have computed the FRFs, you can proceed to compute the fODFs.


2. Computing the fODF
*********************

The second step is to perform multi-shell multi-tissue CSD (msmt-CSD) using :ref:`scil_fodf_msmt`. This script should take longer (about 30 minutes on a full brain).

.. code-block:: bash

    scil_fodf_msmt $in_dir/dwi.nii.gz $in_dir/dwi.bval $in_dir/dwi.bvec \
        wm_frf.txt gm_frf.txt csf_frf.txt --mask $in_dir/mask.nii.gz -v

The script will output one fODFs file per tissue type, in nifti format (wm_fodf.nii.gz, gm_fodf.nii.gz and csf_fodf.nii.gz). The only optional arguments are the ``--sh_order`` option (default is 8) to set the maximum spherical harmonics order used to represent the fODFs and the ``--sh_basis`` option (default is 'descoteaux07') to set the spherical harmonics basis. The ``--processes`` option is used to speed up the computation by using multiple CPU cores. By default, the script will also output the volume fractions map (in default and RGB versions), with names vf.nii.gz and vf_rgb.nii.gz. To change any of the output names and paths or output only a selection of files, use the ``--not_all`` option along with the ``--wm_out_fODF``, ``--gm_out_fODF``, ``--csf_out_fODF``, ``--vf`` and ``--vf_rgb`` arguments. To visualize the fODFs, you can use :ref:`scil_viz_fodf`.

.. [multitissueCSD] Jeurissen et al. NeuroImage 2014, "Multi-tissue constrained spherical deconvolution for improved analysis of multi-shell diffusion MRI data".


`The complete b-tensor scripts tutorial bash script <msmt_fodf.sh>`_.