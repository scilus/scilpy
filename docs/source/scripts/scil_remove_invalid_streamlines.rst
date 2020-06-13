scil_remove_invalid_streamlines.py
==============

::

	usage: scil_remove_invalid_streamlines.py [-h] [--cut_invalid] [--remove_single_point]
	                    [--remove_overlapping_points] [--reference REFERENCE] [-f]
	                    in_tractogram out_tractogram
	
	Removal of streamlines that are out of the volume bounding box. In voxel space
	no negative coordinate and no above volume dimension coordinate are possible.
	Any streamline that do not respect these two conditions are removed.
	
	The --cut_invalid option will cut streamlines so that their longest segment are
	within the bounding box
	
	positional arguments:
	  in_tractogram         Tractogram filename. Format must be one of 
	                        trk, tck, vtk, fib, dpy.
	  out_tractogram        Output filename. Format must be one of 
	                        trk, tck, vtk, fib, dpy.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --cut_invalid         Cut invalid streamlines rather than removing them.
	                        Keep the longest segment only.
	  --remove_single_point
	                        Consider single point streamlines invalid.
	  --remove_overlapping_points
	                        Consider streamlines with overlapping points invalid.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
