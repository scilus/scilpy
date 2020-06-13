scil_compute_maps_for_particle_filter_tracking.py
==============

::

	usage: scil_compute_maps_for_particle_filter_tracking.py [-h] [--include filename] [--exclude filename]
	                    [--interface filename] [-t THRESHOLD] [-f] [-v]
	                    wm gm csf
	
	Compute include and exclude maps, and the seeding interface mask from partial
	volume estimation (PVE) maps. Maps should have values in [0,1], gm+wm+csf=1 in
	all voxels of the brain, gm+wm+csf=0 elsewhere.
	
	References: Girard, G., Whittingstall K., Deriche, R., and Descoteaux, M.
	(2014). Towards quantitative connectivity analysis: reducing tractography
	biases. Neuroimage.
	
	positional arguments:
	  wm                    White matter PVE map (nifti). From normal FAST output, has a PVE_2 name suffix.
	  gm                    Grey matter PVE map (nifti). From normal FAST output, has a PVE_1 name suffix.
	  csf                   Cerebrospinal fluid PVE map (nifti). From normal FAST output, has a PVE_0 name suffix.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --include filename    Output include map (nifti). [map_include.nii.gz]
	  --exclude filename    Output exclude map (nifti). [map_exclude.nii.gz]
	  --interface filename  Output interface seeding mask (nifti). [interface.nii.gz]
	  -t THRESHOLD          Minimum gm and wm PVE values in a voxel to be in to the interface. [0.1]
	  -f                    Force overwriting of the output files.
	  -v                    If set, produces verbose output.
