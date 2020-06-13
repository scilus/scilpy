scil_recognize_single_bundle.py
==============

::

	usage: scil_recognize_single_bundle.py [-h]
	                    [--tractogram_clustering_thr TRACTOGRAM_CLUSTERING_THR]
	                    [--model_clustering_thr MODEL_CLUSTERING_THR]
	                    [--pruning_thr PRUNING_THR] [--slr_threads SLR_THREADS]
	                    [--seed SEED] [--inverse] [--no_empty]
	                    [--in_pickle IN_PICKLE | --out_pickle OUT_PICKLE]
	                    [--reference REFERENCE] [-v] [-f]
	                    in_tractogram in_model in_transfo out_tractogram
	
	Compute a simple Recobundles (single-atlas & single-parameters).
	The model need to be cleaned and lightweight.
	Transform should come from ANTs: (using the --inverse flag)
	AntsRegistration -m MODEL_REF -f SUBJ_REF
	ConvertTransformFile 3 0GenericAffine.mat 0GenericAffine.npy --ras --hm
	
	positional arguments:
	  in_tractogram         Input tractogram filename.
	  in_model              Model to use for recognition.
	  in_transfo            Path for the transformation to model space (.txt, .npy or .mat).
	  out_tractogram        Output tractogram filename.
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --tractogram_clustering_thr TRACTOGRAM_CLUSTERING_THR
	                        Clustering threshold used for the whole brain [8mm].
	  --model_clustering_thr MODEL_CLUSTERING_THR
	                        Clustering threshold used for the model [4mm].
	  --pruning_thr PRUNING_THR
	                        MDF threshold used for final streamlines selection [6mm].
	  --slr_threads SLR_THREADS
	                        Number of threads for SLR [all].
	  --seed SEED           Random number generator seed [None].
	  --inverse             Use the inverse transformation.
	  --no_empty            Do not write file if there is no streamline.
	  --in_pickle IN_PICKLE
	                        Input pickle clusters map file.
	                        Will override the tractogram_clustering_thr parameter.
	  --out_pickle OUT_PICKLE
	                        Output pickle clusters map file.
	  --reference REFERENCE
	                        Reference anatomy for tck/vtk/fib/dpy file
	                        support (.nii or .nii.gz).
	  -v                    If set, produces verbose output.
	  -f                    Force overwriting of the output files.
	
	Garyfallidis, E., Côté, M. A., Rheault, F., ... &
	Descoteaux, M. (2018). Recognition of white matter
	bundles using local and global streamline-based registration and
	clustering. NeuroImage, 170, 283-295.
