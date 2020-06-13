scil_transform_surface.py
==============

::

	usage: scil_transform_surface.py [-h] [--ants_warp ANTS_WARP] [-f]
	                    in_surface ants_affine out_surface
	
	Script to load and transform a surface (FreeSurfer or VTK supported),
	This script is using ANTs transform (affine.txt, warp.nii.gz).
	
	Best usage with ANTs from T1 to b0:
	> ConvertTransformFile 3 output0GenericAffine.mat vtk_transfo.txt --hm
	> scil_transform_surface.py lh_white_lps.vtk affine.txt lh_white_b0.vtk\
	    --ants_warp warp.nii.gz
	
	The input surface needs to be in *T1 world LPS* coordinates
	(aligned over the T1 in MI-Brain).
	The resulting surface should be aligned *b0 world LPS* coordinates
	(aligned over the b0 in MI-Brain).
	
	positional arguments:
	  in_surface            Input surface (.vtk).
	  ants_affine           Affine transform from ANTs (.txt).
	  out_surface           Output surface (.vtk).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --ants_warp ANTS_WARP
	                        Warp image from ANTs (NIfTI format).
	  -f                    Force overwriting of the output files.
	
	References:
	[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
	    Surface-enhanced tractography (SET). NeuroImage.
