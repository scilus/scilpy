Instructions for streamlines registration/transformation
========================================================

::

    scil_tractogram_register.py MOVING_FILE STATIC_FILE

*The file outputted by this script is a 4x4 matrix* (see the help for the option)

Linear transformation
---------------------

If you want to apply a transformation coming from the previous script
::

    scil_tractogram_apply_transform.py MOVING_FILE REFERENCE_FILE TRANSFORMATION OUTPUT_NAME


Due to a difference in convention between image and tractogram the following script
must be called using the --inverse flag if the transformation was obtained using AntsRegistration

::

    scil_tractogram_apply_transform.py MOVING_FILE REFERENCE_FILE  0GenericAffine.mat OUTPUT_NAME --inverse

Non-linear deformation
----------------------
To apply a non-linear transformation from ANTS

::

    scil_tractogram_apply_transform.py MOVING_FILE REFERENCE_FILE  0GenericAffine.mat OUTPUT_NAME --inverse --in_deformation DEFORMATION_FILE

* The DEFORMATION_FILE needs to be the InverseWarp.nii.gz (very important)
* The OUTPUT_NAME is the output tractogram

Complete example
----------------
::

    antsRegistrationSyNQuick.sh -d 3 -f mni_masked.nii.gz -m 100307__fa.nii.gz -t s -o to_mni
    scil_tractogram_apply_transform.py 100307__tracking.trk mni_masked.nii.gz to_mni0GenericAffine.mat 100307__tracking_linear.trk --inverse
    scil_tractogram_apply_transform.py 100307__tracking.trk mni_masked.nii.gz to_mni0GenericAffine.mat 100307__tracking_nonlinear.trk --inverse --in_deformation to_mni1InverseWarp.nii.gz




Apply back and forth tractogram transformation with the ANTS transformation
----------------------------------------------------------------------------
::

    # The ANTS commands is MOVING->REFERENCE
    antsRegistrationSyNQuick.sh -d 3 -f ${REFERENCE_NII.GZ_REF-SPACE} -m ${MOVING_NII.GZ_MOV-SPACE} -t s -o to_reference_

    # This will bring a tractogram from MOVING->REFERENCE
    scil_tractogram_apply_transform.py ${MOVING_FILE_MOV-SPACE} ${REFERENCE_FILE_REF-SPACE}
                                       to_reference_0GenericAffine.mat ${OUTPUT_NAME}
                                       --inverse
                                       --in_deformation to_reference_1InverseWarp.nii.gz

    # This will bring a tractogram from REFERENCE->MOVING
    scil_tractogram_apply_transform.py ${MOVING_FILE_REF-SPACE} ${REFERENCE_FILE_MOV-SPACE}
                                       to_reference_0GenericAffine.mat ${OUTPUT_NAME}
                                       --in_deformation to_reference_1Warp.nii.gz
                                       --reverse_operation
