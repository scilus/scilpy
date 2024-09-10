# Demo Workshop
In this demo, you will be introduced to the main scripts of this project as you apply them on simple data.
<br><br>
Our main objective is better understand and quantify the fundamental limitations of tractography algorithms, and how they might evolve as we introduce histology as *a-priori* information for fiber orientation. To do so, we will be evaluating tractography's ability to reconstruct individual white matter fiber strands at various simulated extreme resolutions.
## Terminology
Here is a list of special terms and definitions used in this project:
- Axon: Physical object. Portion of the nerve cell that carries out the electrical impulse to other neurons.
- Streamline: Virtual object. Series of coordinates propagated through a stream of directional data using a tractography algorithm.
- Fibertube: Virtual representation of an axon. It is composed of a centerline and a diameter.
- Fibertube Tractography: The application of a tractography algorithm directly on fibertubes (to reconstruct them), without using a discretized grid of fODFs or peaks.

In the context of this project, centerlines of fibertubes will come on the form of a tractogram, and we will provide each of them with an artificial axonal diameter coming from a text file. The resulting set of fibertubes will sometimes be referred to as "ground-truth data".

![Fibertube visualized in Blender](https://github.com/VincentBeaud/fibertube_tracking/assets/77688542/25494d10-a8d5-46fa-93d9-0072287d0105)

## Methodology
This project can be split into 3 major steps:

- Preparing ground-truth data <br>
  We will be using the ground-truth of simulated phantoms and ensuring that they are void of any collision given an axonal diameter for each streamline.
- Tracking and experimentation <br>
  We will perform 'Fibertube Tracking' on our newly formed set of fibertubes with a variety of parameter combinations.
- Calculation of metrics <br>
  By passing the resulting tractogram through different analytic pipelines and scripts, we will acquire connectivity and fiber reconstruction scores for each of the parameter combinations.

Of course, some new questions will arise from results and we may branch out of this 3-step frame temporarily.

## Preparing the data
> [!IMPORTANT]
> All commands written down below assume that your console is positioned in the `demo/` folder.

The data required to perform fibertube tractography comes in two files:
- `./disco_centerlines.tck` is a 2000 streamlines subset of the DISCO dataset.
- `./disco_diameters.txt` contains the diameters.
- `./disco_mask.nii.gz` is a white matter mask for spatial reference.

![DISCO subset visualized in MI-Brain](https://github.com/VincentBeaud/fibertube_tracking/assets/77688542/197b3f1f-2f57-41d0-af0a-5f7377bab274)

The first thing to do is resample `disco_centerlines.trk` so that each centerline is formed of
segments no longer than 0.2 mm.

> [!NOTE]
> This is because the next script will rely on a KDTree to find all neighboring fibertube segments of any given point. Because the search radius is set at the length of the longest fibertube segment, the performance drops significantly if they are not shortened to ~0.2mm.

To resample a tractogram, we can use this script from scilpy:
```
scil_tractogram_resample_nb_points.py disco_centerlines.tck disco_centerlines_resampled.tck --step_size 0.2 --reference disco_mask.nii.gz
```

Next, we want to filter out intersecting fibertubes, to make the data anatomically plausible and remove any partial volume effect. This step is crucial to ensure perfect fiber reconstruction at lower scale.

![Fibertube intersection visualized in Blender](https://github.com/VincentBeaud/perfect_tracking/assets/77688542/ede5d949-d7a5-4619-b75b-72fd41d65b38)

This is accomplished using `ft_filter_collisions.py`. <br>
For this demo, let's go all-in and turn on every option.
```
python ../ft_filter_collisions.py disco_centerlines_resampled.tck disco_diameters.txt disco_centerlines_clean.tck -cr -cd -v -f --reference disco_mask.nii.gz
```
> [!IMPORTANT]
> Because this is a script from the project (and not in our environment), we need to call it with "python".

After a short wait, you should get something like:
```
...
├── disco_centerlines_clean_obstacle.tck
├── disco_centerlines_clean_invalid.tck
├── disco_centerlines_clean_diameters.tck
├── disco_centerlines_clean.tck
...
```

## Visualising collisions
Throughout the entire flow, you will be provided with scripts to visualize the data at different steps.

By calling:
```
python ../ft_visualize_collisions.py disco_centerlines_clean_invalid.trk --obstacle disco_centerlines_clean_obstacle.trk --ref_tractogram disco_centerlines.tck --reference disco_mask.nii.gz
```
You are able to see exactly which streamline has been filtered ("invalid" - In red) as well as the streamlines they collided with ("obstacle" - In green).
In white and lower opacity is the original tractogram passed as `--ref_tractogram`.

![Filtered intersections visualized in 3D](https://github.com/VincentBeaud/fibertube_tracking/assets/77688542/4bc75029-0d43-4664-8502-fd528e9d93f4)

### Fibertube metrics
Before we get into tracking. Here is an overview of the metrics that can be computed from the clean data, using `ft_fibers_metrics.py`.

- `min_external_distance`: Smallest distance separating two fibers in the entire set.
- `max_voxel_anisotropic`: Diagonal vector of the largest possible anisotropic voxel that would not intersect two fibers.
- `max_voxel_isotropic`: Isotropic version of max_voxel_anisotropic made by using the smallest component. <br>
Ex: max_voxel_anisotropic: (3, 5, 5) => max_voxel_isotropic: (3, 3, 3)
- `max_voxel_rotated`: Rotated version of max_voxel_anisotropic to align it will (1, 1, 1). This makes it an isotropic voxel, but is only valid if the entire tractogram is rotated the same way.
Ex: max_voxel_anisotropic: (1, 0, 0) => max_voxel_rotated: (0.5774, 0.5774, 0.5774)
- `rotation_matrix`: 4D transformation matrix representing the rotation to be applied on [in_centerlines] for transforming `max_voxel_anisotropic` into `max_voxel_rotated` (see scil_tractogram_apply_transform.py)

![Metrics (without max_voxel_rotated) visualized in Blender](https://github.com/VincentBeaud/perfect_tracking/assets/77688542/95cd4e50-1a36-49af-ac11-0d5f33d3f32e)
<br>
![max_voxel_rotated visualized in Blender](https://github.com/VincentBeaud/perfect_tracking/assets/77688542/72812e47-371f-4005-b289-1de0d70d2f33)

> [!NOTE]
> This information can be useful for analyzing the reconstruction obtained through tracking, as well as for performing track density imaging. The latter will however require a more aggressive fibertube filtering using
> the `--min_distance` argument in the filtering script.

## Performing fibertube tracking
We're finally at the tracking phase! Using the script `ft_fibertube_tracking.py`, you are able to track without relying on a discretized grid of directions. Instead, you will be propagating a streamline through fibertubes and degrading the resolution during the process by using a sphere of "blur".

With a sphere of radius 0, any given set of coordinates has either a single tracking direction because it is within a fibertube, or no direction at all from being out of one. In fact, this behavior won't change until the diameter of the sphere is larger than the smallest distance separating two fibers.

For now, a number of seeds is set randomly within the first segment of every fibertube. We can however change how many fibers will be tracked, as well as the amount of seeds within each. (See Seeding options in the help menu).

The interface of the script is very similar to `scil_tracking_local_dev.py`, but simplified and with a `sampling_radius` mandatory option. This will be the radius of our "blurring" sphere. Let us do:

```
python ../ft_tracking.py disco_centerlines_clean.tck disco_centerlines_clean_diameters.txt disco_mask.nii.gz reconstruction.tck 0.01 0.01 -v -f --nb_seeds_per_fiber 2 --nb_fibers 2 --save_seeds --save_config --processes 4 --reference disco_mask.nii.gz
```
This should take a few minutes at most. However, if you don't mind waiting a little bit, feel free to play with the parameters and explore the resulting tractogram.

> [!NOTE]
> Given the time required for each streamline, the `--processes` parameter will become a good friend of yours.

## Visualizing fibertube coverage of tracked streamlines
Another useful script is:
```
python ../ft_visualize_coverage.py disco_centerlines_clean.tck disco_centerlines_clean_diameters.txt reconstruction.tck reconstruction_config.txt --reference disco_mask.nii.gz
```

> [!IMPORTANT]
> At first, the result might look a bit underwhelming, but if you zoom sufficiently and try to bring the camera through a streamtube,
> you will be able to observe the original fibertube (in red) along the tracked streamlines (in green).

![Reconstructed streamlines alongside ground-truth fibers, visualized in 3D](https://github.com/VincentBeaud/fibertube_tracking/assets/77688542/5b055a1b-45e8-42de-b5e5-7f667bf8299c)


### Reconstruction analysis
By using the `ft_reconstruction_metrics.py` script, you are able to obtain measures on the quality of the fibertube reconstruction. Here is a description of the computed metrics:

VC: "Valid Connection": Streamlines that ended in the final
    segment of the fiber in which they have been seeded. <br>
IC: "Invalid Connection": Streamlines that ended in the final
    segment of another fiber. <br>
NC: "No Connection": Streamlines that have not ended in the final
    segment of any fiber. <br>

An "Error" is the distance between a streamline coordinate and the
closest point on its corresponding fibertube. The average of all errors
of a streamline is called the "Mean error" or "me".

Computed metrics:
- truth_vc
    <br>Connections that are valid at ground-truth resolution.
- truth_ic
- truth_nc
- res_vc
    <br>Connections that are valid at degraded resolution.
- res_ic
- res_nc
- me_min
- me_max
- me_mean
- me_med

## End of Demo
