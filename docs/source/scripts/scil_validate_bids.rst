scil_validate_bids.py
==============

::

	usage: scil_validate_bids.py [-h] [--readout READOUT] [-f] bids_folder output_json
	
	Create a json file with DWI, T1 and fmap informations from BIDS folder
	
	positional arguments:
	  bids_folder        Input BIDS folder.
	  output_json        Output json file.
	
	optional arguments:
	  -h, --help         show this help message and exit
	  --readout READOUT  Default total readout time value [0.062].
	  -f                 Force overwriting of the output files.
