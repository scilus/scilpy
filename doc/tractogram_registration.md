# Instructions for streamlines registration/transformation
```
scil_register_tractogram.py MOVING_FILE STATIC_FILE
```
*The file outputted by this script is a 4x4 matrix* (see the help for the option)

## Linear transformation
If you want to apply a transformation coming from the previous script
```
scil_apply_transform_to_tractogram.py MOVING_FILE REFERENCE_FILE TRANSFORMATION OUTPUT_NAME
```

If you want to apply a transformation coming from ANTS, first you need to change the format of ANTS *.mat.
Then you can call the same script, but using the --inverse flag
```
ConvertTransformFile 3 0GenericAffine.mat 0GenericAffine.npy --ras --hm
```
```
scil_apply_transform_to_tractogram.py MOVING_FILE REFERENCE_FILE  0GenericAffine.npy OUTPUT_NAME --inverse
```
If a nonlinear deformation will be applied after, the REFERENCE_FILE should be the InverseWarp.nii.gz

OUTPUT_NAME is the output tractogram

**

## Non-linear deformation
To apply a non-linear transformation from ANTS
```
scil_apply_warp_to_tractogram.py MOVING_FILE REFERENCE_FILE DEFORMATION_FILE OUTPUT_NAME
```
* The MOVING_FILE needs the same affine and dimensions as the DEFORMATION_FILE
* The DEFORMATION_FILE needs to be the InverseWarp.nii.gz (very important)
* The OUTPUT_NAME is the output tractogram
