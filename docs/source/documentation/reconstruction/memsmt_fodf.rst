Multi-encoding multi-shell multi-tissue fODF (memsmt-fODF)
==========================================================

This tutorial explains how to compute multi-encoding multi-shell multi-tissue fiber orientation distribution functions (fODFs) using multi-encoding multi-shell multi-tissue constrained spherical deconvolution (memsmt-CSD) [memstCSD]_. You data should contain more than one type of b-tensor encoding (ex: linear, planar, spherical). The following instructions are specific to multi-encdoding and based on [memstCSD]_.

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

1. Computing the frf
********************

If you want to do CSD with b-tensor data, you should start by computing the fiber response functions. This script should run fast (less than 5 minutes on a full brain). The data in this tutorial is small, with default parameters, we would get a warning (Could not find at least 100 voxels for the WM mask.), so we'll set the --min_nvox to 1.


.. code-block:: bash

    scil_frf_memsmt wm_frf.txt gm_frf.txt csf_frf.txt \
        --in_dwis $in_dir/dwi_linear.nii.gz $in_dir/dwi_planar.nii.gz $in_dir/dwi_spherical.nii.gz \
        --in_bvals $in_dir/linear.bvals $in_dir/planar.bvals $in_dir/spherical.bvals \
        --in_bvecs $in_dir/linear.bvecs $in_dir/planar.bvecs $in_dir/spherical.bvecs \
        --in_bdeltas 1 -0.5 0 --min_nvox 1  --mask $in_dir/mask.nii.gz \
        --mask_wm $in_dir/wm_mask.nii.gz --mask_gm $in_dir/gm_mask.nii.gz \
        --mask_csf $in_dir/csf_mask.nii.gz

Note. Ignore the warning that some b-values are high.

2. Computing the fODF
*********************

Then, you should compute the fODFs and volume fractions. The following command will save a fODF file for each tissue and a volume fractions file. This script should run in about 1-2 hours for a full brain.

.. code-block:: bash

    scil_fodf_memsmt wm_frf.txt gm_frf.txt csf_frf.txt \
        --in_dwis $in_dir/dwi_linear.nii.gz $in_dir/dwi_planar.nii.gz $in_dir/dwi_spherical.nii.gz \
        --in_bvals $in_dir/linear.bvals $in_dir/planar.bvals $in_dir/spherical.bvals \
        --in_bvecs $in_dir/linear.bvecs $in_dir/planar.bvecs $in_dir/spherical.bvecs \
        --in_bdeltas 1 -0.5 0  --processes 8 --mask $in_dir/mask.nii.gz

The resulting files are: csf_fodf.nii.gz gm_fodf.nii.gz  wm_fodf.nii.gz., as well as the volume fraction map: vf.nii.gz and vf_rgb.nii.gz.


3. Visualizing the fODF
***********************

The resulting fODFs can be visualized using the following command:

.. code-block:: bash

     scil_viz_fodf wm_fodf.nii.gz

See :ref:`scil_viz_fodf` for more information about the visualization options.


References
**********

.. [memstCSD] P. Karan et al., Bridging the gap between constrained spherical deconvolution and diffusional variance decomposition via tensor-valued diffusion MRI. Medical Image Analysis (2022)
