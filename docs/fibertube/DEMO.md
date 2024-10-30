# Demo Workshop
In this demo, you will be introduced to the main scripts of this project as you apply them on simple data.
<br><br>
Our main objective is better understand and quantify the fundamental limitations of tractography algorithms, and how they might evolve as we approach microscopy resolution where individual axons can be seen, tracked or segmented. To do so, we will be evaluating tractography's ability to reconstruct individual white matter fiber strands at simulated extreme resolutions (mimicking "infinite" resolution).
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
  By passing the resulting tractogram through different evaluation scripts (like Tractometer), we will acquire connectivity and fiber reconstruction scores for each of the parameter combinations.

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
scil_tractogram_resample_nb_points.py centerlines.trk centerlines_resampled.trk --step_size 0.2 -f
```

Next, we want to filter out intersecting fibertubes (collisions), to make the data anatomically plausible and remove any partial volume effect.

![Fibertube intersection visualized in Blender](https://github.com/VincentBeaud/perfect_tracking/assets/77688542/ede5d949-d7a5-4619-b75b-72fd41d65b38)

This is accomplished using `scil_tractogram_filter_collisions.py`. <br>

```
scil_tractogram_filter_collisions.py centerlines_resampled.trk diameters.txt fibertubes.trk --save_colliding --out_metrics metrics.txt -v -f
```

After a short wait, you should get something like:
```
...
├── centerlines_resampled_obstacle.trk
├── centerlines_resampled_invalid.trk
├── fibertubes.trk
├── metrics.txt
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
Before we get into tracking. Here is an overview of the metrics that were saved in `metrics.txt`:

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

### Reconstruction analysis
By using the `scil_fibertube_score_tractogram.py` script, you are able to obtain measures on the quality of the fibertube tracking that was performed. Here is a description of the computed metrics:

VC: "Valid Connection": Represents a streamline that ended in the final
    segment of the fibertube in which it was seeded.
IC: "Invalid Connection": Represents a streamline that ended in the final
    segment of another fibertube.
NC: "No Connection": Contains streamlines that have not ended in the final
    segment of any fibertube.

A "coordinate absolute error" is the distance in mm between a streamline
coordinate and the closest point on its corresponding fibertube. The average
of all coordinate absolute errors of a streamline is called the "Mean absolute
error" or "mae".

Computed metrics:
    - truth_vc_ratio
        Proportion of VC.
    - truth_ic_ratio
        Proportion of IC.
    - truth_nc_ratio
        Proportion of NC.
    - res_vc_ratio
        Proportion of VC at the resolution of the blur_radius parameter.
    - res_ic_ratio
        Proportion of IC at the resolution of the blur_radius parameter.
    - res_nc_ratio
        Proportion of NC at the resolution of the blur_radius parameter.
    - mae_min
        Minimum MAE for the tractogram.
    - mae_max
        Maximum MAE for the tractogram.
    - mae_mean
        Average MAE for the tractogram.
    - mae_med
        Median MAE for the tractogram.

Let's do:
```
scil_fibertube_score_tractogram.py fibertubes.trk tracking.trk tracking_config.txt reconstruction_metrics.txt -v -f
```

giving us the following output in `reconstruction_metrics.txt`:
```
{
  "truth_vc_ratio": 0.0,
  "truth_ic_ratio": 0.0,
  "truth_nc_ratio": 1.0,
  "res_vc_ratio": 0.4,
  "res_ic_ratio": 0.0,
  "res_nc_ratio": 0.6,
  "mae_min": 0.0014691361472782293,
  "mae_max": 0.0055722481609273775,
  "mae_mean": 0.003883039143304128,
  "mae_med": 0.003927314695651083
}
```

This data tells us that none of our streamline managed to stay within the fibertube in which it was seeded (`"truth_vc_ratio": 0.0`). However, 40% of streamlinea are at most one `blur_radius` away from the end of their respective fibertube (`"res_vc_ratio": 0.4`). Lastly, we notice that the streamline with the "worst" trajectory was on average 5.5um away from its fibertube (`"mae_max": 0.0055722481609273775`). We can suspect that it started very good early on, but eventually drifted further than 10um, rendering it a NC.

## End of Demo
