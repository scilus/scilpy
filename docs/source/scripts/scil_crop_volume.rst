scil_crop_volume.py
==============

::

	usage: scil_crop_volume.py [-h] [--ignore_voxel_size] [-f]
	                    [--input_bbox path | --output_bbox path]
	                    input_path output_path
	
	Crop a volume using a given or an automatically computed bounding box. If a
	previously computed bounding box file is given, the cropping will be applied
	and the affine fixed accordingly.
	
	Warning: This works well on masked images (like with FSL-Bet) volumes since
	it's looking for non-zero data. Therefore, you should validate the results on
	other types of images that haven't been masked.
	
	positional arguments:
	  input_path           Path of the nifti file to crop.
	  output_path          Path of the cropped nifti file to write.
	
	optional arguments:
	  -h, --help           show this help message and exit
	  --ignore_voxel_size  Ignore voxel size compatibility test between input
	                       bounding box and data. Warning, use only if you know
	                       what you are doing.
	  -f                   Force overwriting of the output files.
	  --input_bbox path    Path of the pickle file from which to take the bounding
	                       box to crop input file.
	  --output_bbox path   Path of the pickle file where to write the computed
	                       bounding box.
