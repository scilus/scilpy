.. _page_from_raw_to_tractogram:

From raw diffusion to DTI, fODF, and tractogram
===============================================

scilpy provides a set of scripts to go from raw diffusion data to a full tractogram. This includes preprocessing, DTI fitting, fODF computation, and tractography.

For an introduction to such a start-to-finish pipeline, you may read section 3.1. in our upcoming paper, describing figure 1:

.. image:: /_static/images//scilpy_paper_figure1.png
   :alt: Figure 1 in upcoming paper.



First, make sure you have the required data. You will need:

* A diffusion-weighted image (DWI) in NIfTI format (e.g., `dwi.nii.gz`)
* Corresponding b-values and b-vectors files (e.g., `dwi.bval` and `dwi.bvec`)
* A brain mask (e.g., `brainmask.nii.gz`)
* A white matter mask (e.g., `wm_mask.nii.gz`)

Preprocessing
-------------

We will then extract the b0 images from the DWI using :ref:`scil_dwi_extract_b0` and compute their mean to use as a reference for registration and visualization.
::

    scil_dwi_extract_b0 dwi.nii.gz dwi.bval dwi.bvec b0_mean.nii.gz --mean

This should produce a file named `b0_mean.nii.gz` which is the average of all b0 images. It should look like this:

.. image:: /_static/images/b0_mean.png
   :scale: 20%

(See :ref:`Create overlapping slice mosaics<create_overlapping_slice_mosaics>` for instructions on how to create such a screenshot.)

Preprocessing of the DWI can include rejecting problematic volumes with :ref:`scil_dwi_detect_volume_outliers` or denoising with :ref:`scil_denoising_nlmeans`. The user may include other preprocessing steps of their choice, such as registration to the MNI template with ANTS or skull-stripping with FSL.

Reconstruction
--------------

scilpy has scripts to compute various local reconstruction methods (:ref:`scil_dti_metrics`, :ref:`scil_dki_metrics`, :ref:`scil_qball_metrics`, :ref:`scil_freewater_maps`, :ref:`scil_NODDI_maps`), particularly many variations of fODFs (:ref:`scil_fodf_metrics`, :ref:`scil_fodf_msmt`, :ref:`scil_fodf_memsmt`, :ref:`scil_fodf_ssst`). 

Let's start with diffusion tensor imaging (DTI).

DTI reconstruction 
------------------

We will extract the shell with a b-value around 1000 s/mm² for DTI fitting using :ref:`scil_dwi_extract_shell`. We will also include the b0 images.
::

    scil_dwi_extract_shell dwi.nii.gz dwi.bval dwi.bvec 0 1000 dwi_b1000.nii.gz \
        dwi_b1000.bval dwi_b1000.bvec

Now, we can reconstruct the DTI tensor volume using :ref:`scil_dti_metrics`. scilpy automatically computes the FA, MD, RGB, and eigenvectors maps in addition to the tensor volume. To ease computation, we will constrain the fitting to the brain mask.
::

    scil_dti_metrics dwi_b1000.nii.gz dwi_b1000.bval dwi_b1000.bvec --mask brainmask.nii.gz

This will produce a lot of files, including:

* `tensor.nii.gz`: The diffusion tensor volume
* `fa.nii.gz`: Fractional Anisotropy map
* `md.nii.gz`: Mean Diffusivity map
* `rd.nii.gz`: Radial Diffusivity map
* `rgb.nii.gz`: RGB map of the principal diffusion direction
* `tensor_evecs.nii.gz`: Eigenvectors of the diffusion tensor
* `tensor_evals.nii.gz`: Eigenvalues of the diffusion tensor

See :ref:`scil_dti_metrics` for a full list of outputs and more details.

Here is an example FA map:

.. image:: /_static/images/fa.png
   :scale: 20%

and an RGB map:

.. image:: /_static/images/rgb.png
   :scale: 20%


DTI Tractography
----------------

Finally, we can do some basic deterministic tractography using the principal eigenvector (`tensor_evecs_v1.nii.gz`) of the DTI. We will use :ref:`scil_tracking_local` with the [EUDX]_ algorithm. We will seed from the white matter mask and constrain tracking to stay within it. We will generate 20,000 seeds and only keep streamline with lengths between 20 and 200 mm. We will also apply a compression factor of 0.1 to reduce file size.
::

    scil_tracking_local tensor_evecs_v1.nii.gz wm_mask.nii.gz wm_mask.nii.gz \
        tractogram.trk --algo eudx --nt 20000 --min_length 20 --max_length 200 --compress 0.1

The output tractogram (`tractogram.trk`) can be visualized with :ref:`scil_viz_bundle` and should look something like this:

.. image:: /_static/images/eudx_tractogram.png
   :scale: 20%


Next, let's move on to fiber orientation distribution functions (fODFs).

fODF reconstruction
-------------------

fODFs require the compuation of a response function [Descoteaux07]_. We will use the `ssst` algorithm from :ref:`scil_frf_ssst` to compute a **s**\ ingle-**s**\ hell **s**\ ingle-**t**\ issue response function from the b=1000 shell we extracted earlier. We will also use the brain mask and a white matter mask to constrain the selection of voxels used for the estimation.
::

    scil_frf_ssst dwi_b1000.nii.gz dwi_b1000.bval dwi_b1000.bvec frf.txt \
        --mask brainmask.nii.gz --mask_wm wm_mask.nii.gz

We can then compute the fODF using :ref:`scil_fodf_ssst`. We will use the same shell and brain mask as before, and the response function we just computed. We will use the default `descoteaux07_legacy` spherical harmonics basis, which is commonly used in scilpy. The `tournier07` basis is also available and is compatible with MrTrix3 tools. Finally, as we have fewer than 45 directions, we will use a lower spherical harmonics order of 6.
::

    scil_fodf_ssst dwi_b1000.nii.gz dwi_b1000.bval dwi_b1000.bvec frf.txt \
        fodf.nii.gz --mask brainmask.nii.gz --sh_order 6

As opposed to DTI fitting, the script :ref:`scil_fodf_ssst` only produces the fODF volume. We can compute various useful metrics from the fODF using :ref:`scil_fodf_metrics`. We will again use the brain mask to constrain computation. As the script also produces an rgb msp, we will use the `-f` flag and overwrite the previous `rgb.nii.gz` file.
::

    scil_fodf_metrics fodf.nii.gz --mask brainmask.nii.gz -f

This will produce several files, including:

* `nufo.nii.gz`: Number of fiber orientations per voxel
* `afd_sum.nii.gz`: Sum of the apparent fiber density (AFD) across all fiber orientations
* `peaks.nii.gz`: fODF maxima directions

For more information on fodf reconstruction, see :ref:`ssst_fodf` and :ref:`msmt_fodf`.

fODF Tractography
-----------------

Tractography on fODFs can be performed using either probabilistic (`--algo prob`) or deterministic (`--algo det`) algorithms in :ref:`scil_tracking_local`. We will use the same white matter mask for seeding and constraining tracking. We will generate 200,000 seeds and only keep streamlines with lengths between 20 and 200 mm, and apply a compression factor of 0.1 to reduce file size.
::
    
    scil_tracking_local fodf.nii.gz wm_mask.nii.gz wm_mask.nii.gz \
        prob_tractogram.trk --algo prob --nt 200000 --min_length 20 \
        --max_length 200 --compress 0.1

The output tractogram (`prob_tractogram.trk`) can be visualized with :ref:`scil_viz_bundle` and should look something like this:

.. image:: /_static/images/prob_tractogram.png
   :scale: 20%

You have now gone from raw diffusion data to both DTI and fODF-based tractograms using scilpy!

References
----------

.. [EUDX] Garyfallidis, E. (2013). Towards an accurate brain tractography (Doctoral dissertation, University of Cambridge).
.. [Descoteaux07] Descoteaux, M., Angelino, E., Fitzgibbons, S., & Deriche, R. (2007). Regularized, fast, and robust analytical q-ball imaging. Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 58(3), 497-510.
