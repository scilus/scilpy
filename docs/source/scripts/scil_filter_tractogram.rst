scil_filter_tractogram.py
==============

::

	usage: scil_filter_tractogram.py [-h] [--drawn_roi ROI_NAME MODE CRITERIA]
	                    [--atlas_roi ROI_NAME ID MODE CRITERIA]
	                    [--bdo BDO_NAME MODE CRITERIA]
	                    [--x_plane PLANE MODE CRITERIA]
	                    [--y_plane PLANE MODE CRITERIA]
	                    [--z_plane PLANE MODE CRITERIA]
	                    [--filtering_list FILTERING_LIST] [--no_empty]
	                    [--display_counts] [--reference REFERENCE] [-v] [-f]
	                    [--indent INDENT] [--sort_keys]
	                    in_tractogram out_tractogram
	
	Now supports sequential filtering condition and mixed filtering object.
	For example, --atlas_roi ROI_NAME ID MODE CRITERIA
	- ROI_NAME is the filename of a Nifti
	- ID is the integer value in the atlas
	- MODE must be one of these values: 'any', 'either_end', 'both_ends'
	- CRITERIA must be one of these values: ['include', 'exclude']
	
	Multiple filtering tuples can be used and options mixed.
	A logical AND is the only behavior available. All theses filtering
	conditions will be sequentially applied.
	
	positional arguments:
	  in_tractogram         Path of the input tractogram file.
	  out_tractogram        Path of the output tractogram file.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --drawn_roi ROI_NAME MODE CRITERIA
	                        Filename of a hand drawn ROI (.nii or .nii.gz).
	  --atlas_roi ROI_NAME ID MODE CRITERIA
	                        Filename of an atlas (.nii or .nii.gz).
	  --bdo BDO_NAME MODE CRITERIA
	                        Filename of a bounding box (bdo) file from MI-Brain.
	  --x_plane PLANE MODE CRITERIA
	                        Slice number in X, in voxel space.
	  --y_plane PLANE MODE CRITERIA
	                        Slice number in Y, in voxel space.
	  --z_plane PLANE MODE CRITERIA
	                        Slice number in Z, in voxel space.
	  --filtering_list FILTERING_LIST
	                        Text file containing one rule per line
	                        (i.e. drawn_roi mask.nii.gz both_ends include).
	  --no_empty            Do not write file if there is no streamline.
	  --display_counts      Print streamline count before and after filtering
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
