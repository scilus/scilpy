Tensor-valued dMRI scripts (b-tensor)
=====================================

The scripts for multi-encoding multi-shell multi-tissue CSD (memsmt-CSD) are based on [memst]_. We recommend reading it to understand the scope of the memsmt-CSD problem.

Preparing data for this tutorial
********************************

To run lines below, you need a various volumes, b-vector information and masks. The tutorial data is still in preparation, meanwhile you can use this: `

.. code-block:: bash

    in_dir=where/you/downloaded/tutorial/data

    # For now, let's use data in .scilpy
    scil_data_download
    cp -r $HOME/.scilpy/btensor_testdata/ $in_dir/
    in_dir=$in_dir/btensor_testdata/

.. tip::
    You may download the complete bash script to run the whole tutorial in one step `here <../../_static/reconst/btensor_scripts.sh>`_.

1. Computing the frf
********************

If you want to do CSD with b-tensor data, you should start by computing the fiber response functions. This script should run fast (less than 5 minutes on a full brain).

.. code-block:: bash

    in_dir=where/you/downloaded/tutorial/data
    scil_frf_memsmt wm_frf.txt gm_frf.txt csf_frf.txt \
        --in_dwis $in_dir/dwi_linear.nii.gz $in_dir/dwi_planar.nii.gz $in_dir/dwi_spherical.nii.gz \
        --in_bvals $in_dir/linear.bval $in_dir/planar.bval $in_dir/spherical.bval \
        --in_bvecs $in_dir/linear.bvec $in_dir/planar.bvec $in_dir/spherical.bvec \
        --in_bdeltas 1 -0.5 0 --mask $in_dir/mask.nii.gz \
        --mask_wm $in_dir/wm_mask.nii.gz --mask_gm $in_dir/gm_mask.nii.gz \
        --mask_csf $in_dir/csf_mask.nii.gz -v

2. Computing the fODF
*********************

Then, you should compute the fODFs and volume fractions. The following command will save a fODF file for each tissue and a volume fractions file. This script should run in about 1-2 hours for a full brain.

.. code-block:: bash

    scil_fodf_memsmt wm_fodf.txt gm_fodf.txt csf_fodf.txt \
        --in_dwis $in_dir/dwi_linear.nii.gz $in_dir/dwi_planar.nii.gz $in_dir/dwi_spherical.nii.gz \
        --in_bvals $in_dir/linear.bval $in_dir/planar.bval $in_dir/spherical.bval \
        --in_bvecs $in_dir/linear.bvec $in_dir/planar.bvec $in_dir/spherical.bvec \
        --in_bdeltas 1 -0.5 0 --mask $in_dir/mask.nii.gz --processes 8 -v

If you want to do DIVIDE with b-tensor data, you should use the following command. It will save files for the MD, uFA, OP, MK_I, MK_A and MK_T. This script should run in about 1-2 hours for a full brain.

.. code-block:: bash

    scil_btensor_metrics --in_dwis $in_dir/dwi_linear.nii.gz $in_dir/dwi_planar.nii.gz $in_dir/dwi_spherical.nii.gz \
        --in_bvals $in_dir/linear.bval $in_dir/planar.bval $in_dir/spherical.bval \
        --in_bvecs $in_dir/linear.bvec $in_dir/planar.bvec $in_dir/dwi_spherical.bvec \
        --in_bdeltas 1 -0.5 0 --mask $in_dir/mask.nii.gz --fa $in_dir/fa.nii.gz --processes 8 -v

.. [memst] P. Karan et al., Bridging the gap between constrained spherical deconvolution and diffusional variance decomposition via tensor-valued diffusion MRI. Medical Image Analysis (2022)
