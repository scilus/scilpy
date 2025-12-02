Tensor-valued dMRI metrics reconstruction using DIVIDE
======================================================

This tutorial explains how to compute b-tensor metrics like uFA (microscopic fractional anisotropy) using the DIVIDE method [divide]_. Your data should contain more than one type of b-tensor encoding (ex: linear, planar, spherical). The following instructions are specific to b-tensor data and based on [divide]_.

Preparing data for this tutorial
********************************

To run lines below, you need a various volumes, b-vector information and masks. The tutorial data is still in preparation, meanwhile you can use this: `

.. code-block:: bash

    in_dir=where/you/downloaded/tutorial/data
    in_dir=$in_dir/btensor

    # For now, the tutorial data only contains the masks.
    # Other necessary data can be obtained with:
    scil_data_download -v ERROR
    cp $HOME/.scilpy/btensor_testdata/* $in_dir/

.. tip::
    You may download the complete bash script to run the whole tutorial in one step `⭳ here <../../_static/bash/reconst/btensor_scripts.sh>`_.

Computing b-tensor metrics
**************************

To run DIVIDE on your b-tensor data, you should use the following command. It will save files for the MD, uFA, OP, MK_I, MK_A and MK_T. This script should run in about 1-2 hours for a full brain.

.. code-block:: bash

    scil_btensor_metrics \
        --in_dwis $in_dir/dwi_linear.nii.gz $in_dir/dwi_planar.nii.gz $in_dir/dwi_spherical.nii.gz \
        --in_bvals $in_dir/linear.bvals $in_dir/planar.bvals $in_dir/spherical.bvals \
        --in_bvecs $in_dir/linear.bvecs $in_dir/planar.bvecs $in_dir/spherical.bvecs \
        --in_bdeltas 1 -0.5 0 --fa $in_dir/fa.nii.gz --processes 8 --mask $in_dir/mask.nii.gz

References
**********

.. [divide] F. Szczepankiewicz et al., Tensor-valued diffusion encoding for diffusional variance decomposition (DIVIDE): Technical feasibility in clinical MRI systems. PloS one (2019)
