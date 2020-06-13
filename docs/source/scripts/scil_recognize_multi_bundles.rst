scil_recognize_multi_bundles.py
==============

::

	usage: scil_recognize_multi_bundles.py [-h] [--out_dir OUT_DIR]
	                    [--log_level {DEBUG,INFO,WARNING,ERROR}]
	                    [--multi_parameters MULTI_PARAMETERS]
	                    [--minimal_vote_ratio MINIMAL_VOTE_RATIO]
	                    [--tractogram_clustering_thr TRACTOGRAM_CLUSTERING_THR [TRACTOGRAM_CLUSTERING_THR ...]]
	                    [--seeds SEEDS [SEEDS ...]] [--inverse] [--processes NBR]
	                    [-f]
	                    in_tractogram in_config_file in_models_directories
	                    [in_models_directories ...] in_transfo
	
	Compute RecobundlesX (multi-atlas & multi-parameters).
	The model needs to be cleaned and lightweight.
	Transform should come from ANTs: (using the --inverse flag)
	AntsRegistration -m MODEL_REF -f SUBJ_REF
	ConvertTransformFile 3 0GenericAffine.mat 0GenericAffine.npy --ras --hm
	
	The next two arguments are multi-parameters related:
	--multi_parameters must be lower than len(model_clustering_thr) *
	len(bundle_pruning_thr) * len(tractogram_clustering_thr)
	
	--seeds can be more than one value. Multiple values will result in
	a overall multiplicative factor of len(seeds) * '--multi_parameters'
	
	The number of folders provided by 'models_directories' will further multiply
	the total number of runs. Meaning that the total number of Recobundles
	execution will be len(seeds) * '--multi_parameters' * len(models_directories)
	
	--minimal_vote_ratio is a value between 0 and 1. The actual number of votes
	required will be '--minimal_vote_ratio' * len(seeds) * '--multi_parameters'
	* len(models_directories).
	
	Example: 5 atlas, 9 multi-parameters, 2 seeds with a minimal vote_ratio
	of 0.50 will results in 90 executions (for each bundle in the config file)
	and a minimal vote of 45 / 90.
	
	Example data and usage available at: https://zenodo.org/deposit/3613688
	
	positional arguments:
	  in_tractogram         Input tractogram filename (.trk or .tck).
	  in_config_file        Path of the config file (.json)
	  in_models_directories
	                        Path for the directories containing model.
	  in_transfo            Path for the transformation to model space (.txt, .npy or .mat).
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --out_dir OUT_DIR     Path for the output directory [voting_results].
	  --log_level {DEBUG,INFO,WARNING,ERROR}
	                        Log level of the logging class.
	  --multi_parameters MULTI_PARAMETERS
	                        Pick parameters from the potential combinations
	                        Will multiply the number of times Recobundles is ran.
	                        See the documentation [1].
	  --minimal_vote_ratio MINIMAL_VOTE_RATIO
	                        Streamlines will only be considered for saving if
	                        recognized often enough [0.5].
	  --tractogram_clustering_thr TRACTOGRAM_CLUSTERING_THR [TRACTOGRAM_CLUSTERING_THR ...]
	                        Input tractogram clustering thresholds [12]mm.
	  --seeds SEEDS [SEEDS ...]
	                        Random number generator seed [None]
	                        Will multiply the number of times Recobundles is ran.
	  --inverse             Use the inverse transformation.
	  --processes NBR       Number of sub-processes to start. 
	                        Default: [1]
	  -f                    Force overwriting of the output files.
	
	Garyfallidis, E., Côté, M. A., Rheault, F., ... &
	Descoteaux, M. (2018). Recognition of white matter
	bundles using local and global streamline-based registration and
	clustering. NeuroImage, 170, 283-295.
