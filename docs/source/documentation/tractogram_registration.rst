Instructions for streamlines registration/transformation
========================================================

::

    scil_register_tractogram.py MOVING_FILE STATIC_FILE

*The file outputted by this script is a 4x4 matrix* (see the help for the option)

Linear transformation
---------------------

If you want to apply a transformation coming from the previous script
::

    scil_apply_transform_to_tractogram.py MOVING_FILE REFERENCE_FILE TRANSFORMATION OUTPUT_NAME


Due to a difference in convention between image and tractogram the following script
must be called using the --inverse flag if the transformation was obtained using AntsRegistration

::

    scil_apply_transform_to_tractogram.py MOVING_FILE REFERENCE_FILE  0GenericAffine.mat OUTPUT_NAME --inverse

Non-linear deformation
----------------------
To apply a non-linear transformation from ANTS

::

    scil_apply_warp_to_tractogram.py MOVING_FILE REFERENCE_FILE DEFORMATION_FILE OUTPUT_NAME


* The MOVING_FILE needs the same affine and dimensions as the DEFORMATION_FILE
* The DEFORMATION_FILE needs to be the InverseWarp.nii.gz (very important)
* The OUTPUT_NAME is the output tractogram

Complete example
----------------
::

    antsRegistrationSyNQuick.sh -d 3 -f mni_masked.nii.gz -m 100307__fa.nii.gz -t s -o to_mni
    scil_apply_transform_to_tractogram.py 100307__tracking.trk mni_masked.nii.gz to_mni0GenericAffine.mat 100307__tracking_linear.trk --inverse
    scil_apply_warp_to_tractogram.py 100307__tracking_linear.trk mni_masked.nii.gz to_mni1InverseWarp.nii.gz 100307__tracking_nonlinear.trk
