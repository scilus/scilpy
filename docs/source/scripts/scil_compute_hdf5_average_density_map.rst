scil_compute_hdf5_average_density_map.py
==============

::

	usage: scil_compute_hdf5_average_density_map.py [-h] [--binary] [--processes NBR] [-f]
	                    in_hdf5 [in_hdf5 ...] out_dir
	
	Compute a density map for each connection from a hdf5 file.
	Typically use after scil_decompose_connectivity.py in order to obtain the
	average density map of each connection to allow the use of --similarity
	in scil_compute_connectivity.py.
	
	This script is parallelized, but will run much slower on non-SSD if too many
	processes are used. The output is a directory containing the thousands of
	connections:
	out_dir/
	    ├── LABEL1_LABEL1.nii.gz
	    ├── LABEL1_LABEL2.nii.gz
	    ├── [...]
	    └── LABEL90_LABEL90.nii.gz
	
	positional arguments:
	  in_hdf5          List of HDF5 filenames (.h5) from scil_decompose_connectivity.py.
	  out_dir          Path of the output directory.
	
	optional arguments:
	  -h, --help       show this help message and exit
	  --binary         Binarize density maps before the population average.
	  --processes NBR  Number of sub-processes to start. 
	                   Default: [1]
	  -f               Force overwriting of the output files.
