scil_bundle_voxel_label_map.py
==============

::

	usage: scil_bundle_voxel_label_map.py [-h] [--upsample UPSAMPLE] [--reference REFERENCE] [-f]
	                    [-v]
	                    in_bundle in_centroid out_map
	
	Compute label image (Nifti) from bundle and centroid.
	Each voxel will have the label of its nearest centroid point.
	
	The number of labels will be the same as the centroid's number of points.
	
	positional arguments:
	  in_bundle             Fiber bundle file.
	  in_centroid           Centroid streamline corresponding to bundle.
	  out_map               Nifti image with corresponding labels.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --upsample UPSAMPLE   Upsample reference grid by this factor. [1]
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
