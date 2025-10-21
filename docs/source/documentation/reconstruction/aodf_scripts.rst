Asymmetric orientation distribution functions (aodf)
====================================================

The usual symmetric orientation distribution function cannot accurately describe complex fiber configurations such as branching, fanning or bending fiber populations. To tackle this issue, we can estimate asymmetric orientation distribution functions AODF [Poirier and Descoteaux, Neuroimage, 2024]. AODF can be estimated from an input symmetric ODF image via the script :ref:`scil_sh_to_aodf`.


Preparing data for this tutorial
********************************

To run lines below, you need a symmetric ODF image (e.g. from Q-ball or CSD) and a mask to restrict the computation to relevant voxels. The tutorial data is still in preparation, meanwhile you can use this:

.. code-block:: bash

    in_dir=where/you/downloaded/tutorial/data

    # For now, let's use data in .scilpy
    scil_data_download -v ERROR
    mkdir $in_dir/aodf_data
    in_dir=$in_dir/aodf_data
    scil_volume_math convert $HOME/.scilpy/processing/fa_thr.nii.gz \
        $in_dir/brainmask.nii.gz --data_type uint8
    cp $HOME/.scilpy/processing/fodf_descoteaux07.nii.gz $in_dir/fodf.nii.gz

Suggestion. For a faster processing of this tutorial data, you could crop the fodf and brainmask volumes (see :ref:`page_cropping`). With a normal dataset, this tutorial would takes 3-5 hours to complete. The sample data for this tutorial uses a data of size 57x67x56 voxels, with 15 values per voxel, but let's reduce that even more for a shorter run. The current bounding box for this data would be:

.. code-block:: json

    {
        "minimums": [-20, -30, -20],
        "maximums": [20, 30, 20],
        "voxel_size": [2.5, 2.5, 2.5]}
    }

.. tip::
    You may download the complete bash script to run the whole tutorial in one step `here </_static/bash/reconst/aodf_scripts.sh>`_.

Creating the aodf
*****************

Although there is an automatic way to set the parameters, it is not yet implemented in scilpy. We recommend that you experiment with the parameters to find the best configuration for your data. Here, we use the smallest sphere available in Dipy, for a fast test. You can run the command as follows:

.. code-block:: bash

    scil_sh_to_aodf $in_dir/fodf.nii.gz afodf.nii.gz -v --sphere repulsion100


The default script runs a pure python implementation, which is slow. To speed up the execution, you should use OpenCL if you have a compatible GPU or CPU. Make sure you have `pyopencl` installed and a working OpenCL implementation. You can enable OpenCL acceleration by adding the `--use_opencl` flag to the command. You can also choose the device to use (CPU or GPU) with the `--device` option. Using a GPU will reduce the execution time to 1-2 minutes (on a Nvidia GeForce RTX 3080). For example, to use a GPU, you can run:

.. code-block:: bash

    scil_sh_to_aodf $in_dir/fodf.nii.gz afodf.nii.gz --use_opencl --device gpu -v

The script will output the asymmetric ODF image (``afodf.nii.gz``) in the current directory. At the difference of a symmetric ODF image, which is represented using a symmetric spherical harmonics basis, the asymmetric ODF image is represented using a full spherical harmonics basis. Therefore, the output image will have more SH coefficients than the input image. For instance, for a maximum SH order of 8, the input image will have 45 coefficients per voxel, while the output image will have 81 coefficients per voxel.

Computing metrics
*****************

From the estimated AODF, we can compute a bunch of metrics using the script :ref:`scil_aodf_metrics`:.

.. code-block:: bash

    scil_aodf_metrics afodf.nii.gz --mask $in_dir/brainmask.nii.gz -v

This script outputs the following metrics:

- Asymmetry index map (``asi_map.nii.gz``)
- Number of fiber directions (NuFiD) map (``nufid.nii.gz``)
- Odd-power map (``odd_power_map.nii.gz``)
- Peaks image (``asym_peaks.nii.gz``)
- Peak values (``asym_peak_values.nii.gz``)
- Peak indices (``asym_peaks_indices.nii.gz``)

Refer to the script ``--help`` for a description of these metrics. Like with the other ``metrics`` scripts, the flag ``--not_all`` can be used to skip some outputs and only compute the metrics of interest.
