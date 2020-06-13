scil_convert_surface.py
==============

::

	usage: scil_convert_surface.py [-h] [-f] in_surface out_surface
	
	Script to convert a surface (FreeSurfer or VTK supported).
	    ".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"
	
	> scil_convert_surface.py surf.vtk converted_surf.ply
	
	positional arguments:
	  in_surface   Input a surface (FreeSurfer or supported by VTK).
	  out_surface  Output flipped surface (formats supported by VTK).
	
	optional arguments:
	  -h, --help   show this help message and exit
	  -f           Force overwriting of the output files.
	
	References:
	[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
	    Surface-enhanced tractography (SET). NeuroImage.
