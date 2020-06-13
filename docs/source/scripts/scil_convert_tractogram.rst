scil_convert_tractogram.py
==============

::

	usage: scil_convert_tractogram.py [-h] [--reference REFERENCE] [-f]
	                    IN_TRACTOGRAM OUTPUT_NAME
	
	Conversion of '.tck', '.trk', '.fib', '.vtk' and 'dpy' files using updated file
	format standard. TRK file always needs a reference file, a NIFTI, for
	conversion. The FIB file format is in fact a VTK, MITK Diffusion supports it.
	
	positional arguments:
	  IN_TRACTOGRAM         Tractogram filename. Format must be one of 
	                        trk, tck, vtk, fib, dpy
	  OUTPUT_NAME           Output filename. Format must be one of 
	                        trk, tck, vtk, fib, dpy
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -f                    Force overwriting of the output files.
