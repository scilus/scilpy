scil_assign_color_to_trk.py
==============

::

	usage: scil_assign_color_to_trk.py [-h] [--reference REFERENCE] [-f]
	                    in_tractogram out_tractogram color
	
	Assign an hexadecimal RGB color to a Trackvis TRK tractogram.
	The hexadecimal RGB color should be formatted as 0xRRGGBB or
	"#RRGGBB".
	
	Saves the RGB values in the data_per_point (color_x, color_y, color_z).
	
	positional arguments:
	  in_tractogram         Tractogram.
	  out_tractogram        Colored TRK tractogram.
	  color                 Can be either hexadecimal (ie. "#RRGGBB" or 0xRRGGBB).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
