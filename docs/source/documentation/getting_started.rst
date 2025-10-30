Getting ready for tutorials
===========================

Scilpy is the main library supporting research and development at the Sherbrooke Connectivity Imaging Lab (SCIL).

Scilpy mainly comprises tools and utilities to quickly work with diffusion MRI.
Most of the tools are based on or are wrappers of the DIPY library, and most of them will eventually be migrated to DIPY. Those tools implement the recommended workflows and parameters used in the lab.


Installation 
#############
⚠️ We highly suggest to install uv to speedup scilpy installation: https://docs.astral.sh/uv/getting-started/installation/


.. code-block:: bash

    # If you are using Python3.10 or Python3.11, export this variable before installing
    export SETUPTOOLS_USE_DISTUTILS=stdlib

    uv pip install scilpy # For the most recent release from PyPi


Download data
#############

In order to follow the tutorials we highly encourage you to download this archive full of data.

.. code-block:: bash

    wget -O data_for_test.tar.gz https://drive.google.com/file/d/1acraVPkRfYihBS15AEJi9EUgVs0S0LKY/view?usp=sharing
    tar -xvf data_for_test.tar.gz
    rm data_for_test.tar.gz

This file contains:

.. code-block:: bash

    data_for_test/
    ├── mni_masked.nii.gz
    └── sub-01
        ├── sub-01__dwi.bval
        ├── sub-01__dwi.bvec
        ├── sub-01__dwi.nii.gz
        ├── sub-01__t1.nii.gz
                # Results from FSL bet:
        ├── sub-01__brainmask.nii.gz
                # Results from segmentation using FSL
        ├── sub-01__mask_csf.nii.gz
        ├── sub-01__mask_gm.nii.gz
        ├── sub-01__mask_wm.nii.gz
                # Results from our script scil_dti_metrics:
        ├── sub-01__fa.nii.gz
                # Results from freesurfer segmentation:
        ├── sub-01__aparc+aseg.nii.gz
        └── sub-01__wmparc.nii.gz
                # Results from our script scil_tracking_local:
        ├── sub-01_local_tractogram.trk
                # Segmented bundles:
        ├── sub-01__cst_L_part1.trk
        ├── sub-01__cst_L_part2.trk
        ├── sub-01__cst_L.trk
        ├── sub-01__cst_R.trk
        ├── sub-01__small_mask_csf.nii.gz
        ├── sub-01__small_mask_gm.nii.gz
        ├── sub-01__small_mask_wm.nii.gz

