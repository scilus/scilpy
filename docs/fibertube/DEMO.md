# Demo Workshop
In this demo, you will be introduced to the main scripts of this project as you apply them on simple data.
<br><br>
Our main objective is better understand and quantify the fundamental limitations of tractography algorithms, and how they might evolve as we approach microscopy resolution where individual axons can be seen, tracked or segmented. To do so, we will be evaluating tractography's ability to reconstruct individual white matter fiber strands at various simulated extreme resolutions.
## Terminology
Here is a list of terms and definitions used in this project.

General:
- Axon: Bio-physical object. Portion of the nerve cell that carries out the electrical impulse to other neurons. (On the order of 0.1 to 1um)
- Streamline: Virtual object. Series of equidistant 3D coordinates approximating an underlying fiberous structure.

Fibertube Tracking:
- Centerline: Virtual object. Series of equidistant 3D coordinates representing the directional information of a fibertube.
- Fibertube: Virtual representation of an axon. It is composed of a centerline and a single diameter for its whole length.
- Fibertube segment: Because centerlines are made of discrete coordinates, fibertubes end up being composed of a series of equally lengthed adjacent cylinders. A fibertube segment is any single one of those cylinders.
- Fibertube Tractography: The application of a tractography algorithm directly on fibertubes to reconstruct them. Contrary to traditional white matter fiber tractography, fibertube tractography does not rely on a discretized grid of fODFs or peaks.

![Fibertube visualized in Blender](https://github.com/VincentBeaud/fibertube_tracking/assets/77688542/25494d10-a8d5-46fa-93d9-0072287d0105)

## Methodology
This project can be split into 3 major steps:

- Preparing ground-truth data <br>
  We will be using the ground-truth of simulated phantoms and ensuring that they are void of any collision given an axonal diameter for each streamline.
- Tracking and experimentation <br>
  We will perform 'Fibertube Tracking' on our newly formed set of fibertubes with a variety of parameter combinations.
- Calculation of metrics <br>
  By passing the resulting tractogram through different analytic pipelines and scripts (like Tractometer), we will acquire connectivity and fiber reconstruction scores for each of the parameter combinations.

## Preparing the data
> [!IMPORTANT]
> All commands written down below assume that your console is positioned in the folder containing your data.

The data required to perform fibertube tractography comes in two files:
- `./centerlines.trk` contains the entire ground-truth of the DISCO dataset.
- `./diameters.txt` contains the diameters.

![DISCO subset visualized in MI-Brain](https://github.com/VincentBeaud/fibertube_tracking/assets/77688542/197b3f1f-2f57-41d0-af0a-5f7377bab274)

The first thing to do is resample `centerlines.trk` so that each centerline is formed of
segments no longer than 0.2 mm.

> [!NOTE]
> This is because the next script will rely on a KDTree to find all neighboring fibertube segments of any given point. Because the search radius is set at the length of the longest fibertube segment, the performance drops significantly if they are not shortened to ~0.2mm.

To resample a tractogram, we can use this script from scilpy:
```
scil_tractogram_resample_nb_points.py centerlines.trk centerlines_resampled.trk --step_size 0.2
```

Next, we want to filter out intersecting fibertubes, to make the data anatomically plausible and remove any partial volume effect.

![Fibertube intersection visualized in Blender](https://github.com/VincentBeaud/perfect_tracking/assets/77688542/ede5d949-d7a5-4619-b75b-72fd41d65b38)

This is accomplished using `scil_tractogram_filter_collisions.py`. <br>

```
scil_tractogram_filter_collisions.py centerlines_resampled.trk diameters.txt fibertubes.trk --save_colliding --out_metrics metrics.txt -v
```

After a short wait, you should get something like:
```
...
├── centerlines_resampled_obstacle.trk
├── centerlines_resampled_invalid.trk
├── fibertubes.trk
...
```

As you may have guessed from the output name, this script automatically combines the diameter to the centerlines as data_per_streamline in the output tractogram. This is why we named it "fibertubes.trk".

## Visualising collisions
By calling:
```
scil_viz_tractogram_collisions.py centerlines_resampled_invalid.trk --obstacle centerlines_resampled_obstacle.trk --ref_tractogram centerlines.trk
```
You are able to see exactly which streamline has been filtered ("invalid" - In red) as well as the streamlines they collided with ("obstacle" - In green).
In white and lower opacity is the original tractogram passed as `--ref_tractogram`.

![Filtered intersections visualized in 3D](https://github.com/VincentBeaud/fibertube_tracking/assets/77688542/4bc75029-0d43-4664-8502-fd528e9d93f4)

### Fibertube metrics
Before we get into tracking. Here is an overview of the metrics that we saved in `metrics.txt`:

- `min_external_distance`: Smallest distance separating two fibertubes, outside their diameter.
- `max_voxel_anisotropic`: Diagonal vector of the largest possible anisotropic voxel that would not intersect two fibertubes.
- `max_voxel_isotropic`: Isotropic version of max_voxel_anisotropic made by using the smallest component. <br>
Ex: max_voxel_anisotropic: (3, 5, 5) => max_voxel_isotropic: (3, 3, 3)
- `max_voxel_rotated`: Largest possible isotropic voxel obtainable if the tractogram is rotated. It is only usable if the entire tractogram is rotated according to [rotation_matrix].
Ex: max_voxel_anisotropic: (1, 0, 0) => max_voxel_rotated: (0.5774, 0.5774, 0.5774)
- `rotation_matrix`: 4D transformation matrix representing the rotation to be applied on the tractogram to align max_voxel_rotated with the coordinate system (see scil_tractogram_apply_transform.py).

![Metrics (without max_voxel_rotated) visualized in Blender](https://github.com/VincentBeaud/perfect_tracking/assets/77688542/95cd4e50-1a36-49af-ac11-0d5f33d3f32e)
<br>
![max_voxel_rotated visualized in Blender](https://github.com/VincentBeaud/perfect_tracking/assets/77688542/72812e47-371f-4005-b289-1de0d70d2f33)

> [!NOTE]
> This information can be useful for analyzing the reconstruction obtained through tracking, as well as for performing track density imaging.

## Performing fibertube tracking
We're finally at the tracking phase! Using the script `scil_fibertube_tracking.py`, you are able to track without relying on a discretized grid of directions or fODFs. Instead, you will be propagating a streamline through fibertubes and degrading the resolution by using a `blur_radius`. The way it works is as follows:

### Tracking
When the tracking algorithm is about to select a new direction to propagate the current streamline, it will build a sphere of radius `blur_radius` and pick randomely from all the fibertube segments intersecting with it. The larger the intersection volume, the more likely a fibertube segment is to be picked and used as a tracking direction. This makes fibertube tracking inherently probabilistic.
Theoretically, with a `blur_radius` of 0, any given set of coordinates has either a single tracking direction because it is within a fibertube, or no direction at all from being out of one. In fact, this behavior won't change until the diameter of the sphere is larger than the smallest distance separating two fibertubes. When this happens, more than one fibertubes will intersect the `blur_radius` sphere and introduce partial volume effect.


### Seeding
For now, a number of seeds is set randomly within the first segment of every fibertube. We can however change how many fibers will be tracked, as well as the amount of seeds within each. (See Seeding options in the help menu).

<br>
The interface of the script is very similar to `scil_tracking_local_dev.py`, but simplified and with a `blur_radius` option. Let us do:

```
scil_fibertube_tracking.py fibertubes.trk tracking.trk 0.01 0.01 --nb_fibertubes 3 --out_config tracking_config.txt --processes 4 -v -f
```
This should take a few minutes at most. However, if you don't mind waiting a little bit, feel free to play with the parameters and explore the resulting tractogram.

> [!NOTE]
> Given the time required for each streamline, the `--processes` parameter will be very useful.
 HERE
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
