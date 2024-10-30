# Demo
In this demo, you will be introduced to the main scripts of this project as you apply them on simple data.
<br><br>
Our main objective is better understand and quantify the fundamental limitations of tractography algorithms, and how they might evolve as we approach microscopy resolution where individual axons can be seen. To do so, we will be evaluating tractography's ability to reconstruct individual white matter fiber strands at simulated extreme resolutions (mimicking "infinite" resolution).
## Terminology
Here is a list of terms and definitions used in this project.

General:
- Axon: Bio-physical object. Portion of the nerve cell that carries out the electrical impulse to other neurons. (On the order of 0.1 to 1um)
- Streamline: Virtual object. Series of 3D coordinates approximating an underlying fiber structure.

Fibertube Tracking:
- Fibertube: Virtual representation of an axon. Tube obtained from combining a diameter to a streamline.
- Centerline: Virtual object. Streamline passing through the center of a tubular structure.
- Fibertube segment: Cylindrical segment of a fibertube that comes as a result of the discretization of its centerline.
- Fibertube Tractography: The computational tractography method that reconstructs fibertubes. Contrary to traditional white matter fiber tractography, fibertube tractography does not rely on a discretized grid of fODFs or peaks. It directly tracks and reconstructs fibertubes, i.e. streamlines that have an associated diameter.


![Fibertube visualized in 3D](https://github.com/user-attachments/assets/e5dbeb23-ff2f-48ae-85c4-e0e98a0c0070)


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
- `./diameters.txt` contains the diameter to be applied to each centerline in the centerlines.trk file above.

![DISCO subset visualized in 3D](https://github.com/VincentBeaud/fibertube_tracking/assets/77688542/197b3f1f-2f57-41d0-af0a-5f7377bab274)

The first thing to do is resample `centerlines.trk` so that each centerline is formed of
segments no longer than 0.2 mm.

> [!NOTE]
> This is because the next script will rely on a KDTree to find all neighboring fibertube segments of any given point. Because the search radius is set at the length of the longest fibertube segment, the performance drops significantly if they are not shortened to ~0.2mm.

To resample a tractogram, we can use this script from scilpy:
```
scil_tractogram_resample_nb_points.py centerlines.trk centerlines_resampled.trk --step_size 0.2 -f
```

Next, we want to filter out intersecting fibertubes (collisions), to make the data anatomically plausible and remove any partial volume effect.

![Fibertube intersection visualized in 3D](https://github.com/VincentBeaud/perfect_tracking/assets/77688542/ede5d949-d7a5-4619-b75b-72fd41d65b38)

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

If you wish to know how many fibertubes are left after filtering, you can run the following command:

```scil_tractogram_print_info.py fibertube.txt```



## Visualising collisions
By calling:
```
scil_viz_tractogram_collisions.py centerlines_resampled_invalid.trk --obstacle centerlines_resampled_obstacle.trk --ref_tractogram centerlines.trk
```
You are able to see exactly which streamline has been filtered ("invalid" - In red) as well as the streamlines they collided with ("obstacle" - In green).
In white and lower opacity is the original tractogram passed as `--ref_tractogram`.

![Filtered intersections visualized in 3D](https://github.com/VincentBeaud/fibertube_tracking/assets/77688542/4bc75029-0d43-4664-8502-fd528e9d93f4)

### Fibertube metrics
Before we get into tracking. Here is an overview of the metrics that we saved in `metrics.txt`. (Values expressed in mm):

- `min_external_distance`: Smallest distance separating two fibertubes, outside their diameter.
- `max_voxel_anisotropic`: Diagonal vector of the largest possible anisotropic voxel that would not intersect two fibertubes.
- `max_voxel_isotropic`: Isotropic version of max_voxel_anisotropic made by using the smallest component. <br>
Ex: max_voxel_anisotropic: (3, 5, 5) => max_voxel_isotropic: (3, 3, 3)
- `max_voxel_rotated`: Largest possible isotropic voxel obtainable if the tractogram is rotated. It is only usable if the entire tractogram is rotated according to [rotation_matrix].
Ex: max_voxel_anisotropic: (1, 0, 0) => max_voxel_rotated: (0.5774, 0.5774, 0.5774)

![Metrics (without max_voxel_rotated) visualized in 3D](https://github.com/user-attachments/assets/43cebcbe-e3b1-4ca0-999e-e042db8aa937)
<br>

![max_voxel_rotated visualized in 3D](https://github.com/user-attachments/assets/924ab3f9-33da-458f-a98b-b4e88b051ae8)


> [!NOTE]
> This information can be useful for analyzing the reconstruction obtained through tracking, as well as for performing track density imaging.

## Performing fibertube tracking
We're finally at the tracking phase! Using the script `scil_fibertube_tracking.py`, you are able to track without relying on a discretized grid of directions or fODFs. Instead, you will be propagating a streamline through fibertubes and degrading the resolution by using a `blur_radius`. The way it works is as follows:

### Tracking
When the tracking algorithm is about to select a new direction to propagate the current streamline, it will build a sphere of radius `blur_radius` and pick randomly from all the fibertube segments intersecting with it. The larger the intersection volume, the more likely a fibertube segment is to be picked and used as a tracking direction. This makes fibertube tracking inherently probabilistic.
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

VC: "Valid Connection": A streamline that passes WITHIN the final segment of <br>
    the fibertube in which it was seeded. <br>
IC: "Invalid Connection": A streamline that ended in the final segment of <br>
    another fibertube. <br>
NC: "No Connection": A streamlines that has not ended in the final segment <br>
    of any fibertube. <br>

Res_VC: "Resolution-wise Valid Connection": A streamline that passes closer <br>
    than [blur_darius] away from the last segment of the fibertube in which it <br>
    was seeded. <br>
Res_IC: "Resolution-wise Invalid Connection": A streamline that passes closer <br>
    than [blur_darius] away from the first or last segment of another <br>
    fibertube. <br>
Res_NC: "Resolution-wise No Connection": A streamlines that does not pass <br>
    closer than [blur_radius] away from the first or last segment of any <br>
    fibertube. <br>

The "absolute error" of a coordinate is the distance in mm between that <br>
coordinate and the closest point on its corresponding fibertube. The average <br>
of all coordinate absolute errors of a streamline is called the "Mean absolute <br>
error" or "mae". <br>

Computed metrics:
   - vc_ratio <br>
        Number of VC divided by the number of streamlines.
   - ic_ratio <br>
        Number of IC divided by the number of streamlines.
   - nc_ratio <br>
        Number of NC divided by the number of streamlines.
   - res_vc_ratio <br>
        Number of Res_VC divided by the number of streamlines.
   - res_ic_ratio <br>
        Number of Res_IC divided by the number of streamlines.
   - res_nc_ratio <br>
        Number of Res_NC divided by the number of streamlines.
   - mae_min <br>
        Minimum MAE for the tractogram.
   - mae_max <br>
        Maximum MAE for the tractogram.
   - mae_mean <br>
        Average MAE for the tractogram.
   - mae_med <br>
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

This data tells us that none of our streamline managed to stay within the fibertube in which it was seeded (`"truth_vc_ratio": 0.0`). However, 40% of streamlines are at most one `blur_radius` away from the end of their respective fibertube (`"res_vc_ratio": 0.4`). Lastly, we notice that the streamline with the "worst" trajectory was on average 5.5um away from its fibertube (`"mae_max": 0.0055722481609273775`).

## End of Demo

