scil_filter_streamlines_by_length.py
==============

::

	usage: scil_filter_streamlines_by_length.py [-h] [--minL MINL] [--maxL MAXL] [--no_empty]
	                    [--display_counts] [--reference REFERENCE] [-f] [-v]
	                    [--indent INDENT] [--sort_keys]
	                    in_tractogram out_tractogram
	
	Script to filter streamlines based on their lengths.
	
	positional arguments:
	  in_tractogram         Streamlines input file name.
	  out_tractogram        Streamlines output file name.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --minL MINL           Minimum length of streamlines, in mm. [0.0]
	  --maxL MAXL           Maximum length of streamlines, in mm. [inf]
	  --no_empty            Do not write file if there is no streamline.
	  --display_counts      Print streamline count before and after filtering
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
	
	Json options:
	  --indent INDENT       Indent for json pretty print.
	  --sort_keys           Sort keys in output json.
