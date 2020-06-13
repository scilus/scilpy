scil_flip_surface.py
==============

::

	usage: scil_flip_surface.py [-h] [-f] in_surface out_surface {x,y,z,n} [{x,y,z,n} ...]
	
	Script to flip and reverse a surface (FreeSurfer or VTK supported).
	Can be used to flip in chosen axes (x, y or z),
	it can also flip inside out the surface orientation (normal).
	
	Best usage for FreeSurfer to LPS vtk (for MI-Brain):
	!!! important FreeSurfer surfaces must be in their respective folder !!!
	> mris_convert --to-scanner lh.white lh.white.vtk
	> scil_flip_surface.py lh.white.vtk lh_white_lps.vtk x y
	
	positional arguments:
	  in_surface   Input surface (.vtk).
	  out_surface  Output flipped surface (.vtk).
	  {x,y,z,n}    The axes (or normal orientation) you want to flip. eg: to flip the x and y axes use: x y.
	
	optional arguments:
	  -h, --help   show this help message and exit
	  -f           Force overwriting of the output files.
	
	References:
	[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
	    Surface-enhanced tractography (SET). NeuroImage.
