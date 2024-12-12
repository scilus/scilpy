Introduction to the Fibertube Tracking environment through an interactive demo.
====

In this demo, you will be introduced to the main scripts of this project
as you apply them on simple data. Our main objective is to better
understand and quantify the fundamental limitations of tractography
algorithms, and how they might evolve as we approach microscopy
resolution where individual axons can be seen. To do so, we will be
evaluating tractography's ability to reconstruct individual white matter
fiber strands at simulated extreme resolutions (mimicking "infinite"
resolution).

Terminology
-----------

Here is a list of terms and definitions used in this project.

General:

-  Axon: Bio-physical object. Portion of the nerve cell that carries out
   the electrical impulse to other neurons. (On the order of 0.1 to 1um)
-  Streamline: Virtual object. Series of 3D coordinates approximating an
   underlying fiber structure.

Fibertube Tracking:

-  Fibertube: Virtual representation of an axon. Tube obtained from
   combining a diameter to a streamline.
-  Centerline: Virtual object. Streamline passing through the center of
   a tubular structure.
-  Fibertube segment: Cylindrical segment of a fibertube that comes as a
   result of the discretization of its centerline.
-  Fibertube Tractography: The computational tractography method that
   reconstructs fibertubes. Contrary to traditional white matter fiber
   tractography, fibertube tractography does not rely on a discretized
   grid of fODFs or peaks. It directly tracks and reconstructs
   fibertubes, i.e. streamlines that have an associated diameter.

.. image:: https://github.com/user-attachments/assets/0286ec53-5bca-4133-93dd-22f360dfcb45
   :alt: Fibertube visualized in 3D

Methodology
-----------

This project can be split into 3 major steps:

-  Preparing ground-truth data: We will be using the ground-truth of
   simulated phantoms of streamlines, along with a diameter (giving us
   fibertubes) and ensuring that they are void of any collision, i.e.
   fibertubes in the simulated phantom should not intersect one another.
   This is physically impossible to respect the geometry of axons.
-  Tracking and experimentation: We will perform fibertube tracking on
   our newly formed set of fibertubes with a variety of parameter
   combinations.
-  Evaluation metrics computation: By passing the resulting tractogram
   through different evaluation scripts (like Tractometer), we will
   acquire connectivity and fiber reconstruction scores for each of the
   parameter combinations.

Preparing the data
------------------

To download the data required for this demo, open a terminal, move to any
location you see fit for this demo and execute the following command:
::

   wget https://scil.usherbrooke.ca/scil_test_data/dvc-store/files/md5/82/248b4888a63b0aeffc8070cc206995 -O others.zip && unzip others.zip -d Data && mv others.zip Data/others.zip && chmod -R 755 Data && cp ./Data/others/fibercup_bundles.trk ./centerlines.trk && echo 0.001 >diameters.txt

This will fetch a tractogram to act as our set of centerlines, and then
generate diameters to form our fibertubes.

``centerlines.trk`` is a subset of the FiberCup phantom ground truth:

.. image:: https://github.com/user-attachments/assets/3be43cc9-60ec-4e97-95ef-a436c32bba83
   :alt: Fibercup subset visualized in 3D

The first thing to do with our data is to resample ``centerlines.trk``
so that each centerline is formed of segments no longer than 0.2 mm.

Note: This is because the next script will rely on a KDTree to find
all neighboring fibertube segments of any given point. Because the
search radius is set at the length of the longest fibertube segment,
the performance drops significantly if they are not shortened to
~0.2mm.

To resample a tractogram, we can use this script from scilpy. Don't
forget to activate your scilpy environment first.

::

   scil_tractogram_resample_nb_points.py centerlines.trk centerlines_resampled.trk --step_size 0.2 -f

Next, we want to filter out intersecting fibertubes (collisions), to
make the data anatomically plausible and ensure that there exists a
resolution at which there is no unit of space containing partial
volume.

.. image:: https://github.com/user-attachments/assets/d9b0519b-c1e3-4de0-8529-92aa92041ce2
   :alt: Fibertube intersection visualized in 3D

This is accomplished using ``scil_tractogram_filter_collisions.py``.

::

   scil_tractogram_filter_collisions.py centerlines_resampled.trk diameters.txt fibertubes.trk --save_colliding --out_metrics metrics.json -v -f

After 3-5 minutes, you should get something like:

::

   ...
   ├── centerlines_resampled_obstacle.trk
   ├── centerlines_resampled_invalid.trk
   ├── fibertubes.trk
   ├── metrics.json
   ...

As you may have guessed from the output name, this script automatically
combines the diameter to the centerlines as data_per_streamline in the
output tractogram. This is why we named it "fibertubes.trk".

If you wish to know how many fibertubes are left after filtering, you
can run the following command:

``scil_tractogram_print_info.py fibertubes.trk``

Visualising collisions
----------------------

By calling:

::

   scil_viz_tractogram_collisions.py centerlines_resampled_invalid.trk --in_tractogram_obstacle centerlines_resampled_obstacle.trk --ref_tractogram centerlines.trk

You are able to see exactly which streamline has been filtered
("invalid" - In red) as well as the streamlines they collided with
("obstacle" - In green). In white and lower opacity is the original
tractogram passed as ``--ref_tractogram``.

.. image:: https://github.com/user-attachments/assets/7ab864f5-e4a3-421b-8431-ef4a5b3150c8
   :alt: Filtered intersections visualized in 3D

Fibertube metrics
-----------------

Before we get into tracking. Here is an overview of the metrics that we
saved in ``metrics.json``. (Values expressed in mm):

-  ``fibertube_density``:
   Estimate of the following ratio: volume of fibertubes / total volume
   where the total volume is the combined volume of all voxels containing
   at least one fibertube.
-  ``min_external_distance``: Smallest distance separating two
   fibertubes, outside their diameter.
-  ``max_voxel_anisotropic``: Diagonal vector of the largest possible
   anisotropic voxel that would not intersect two fibertubes.
-  ``max_voxel_isotropic``: Isotropic version of max_voxel_anisotropic
   made by using the smallest component. Ex: max_voxel_anisotropic: (3,
   5, 5) => max_voxel_isotropic: (3, 3, 3)
-  ``max_voxel_rotated``: Largest possible isotropic voxel obtainable with
   a different coordinate system. It is only usable if the entire tractogram
   is rotated according to [rotation_matrix]. Ex: max_voxel_anisotropic:
   (1, 0, 0) => max_voxel_rotated: (0.5774, 0.5774, 0.5774)

If the option is provided. The following matrix would be saved in a
different file:

-  ``rotation_matrix``: 4D transformation matrix containing the rotation to be
   applied on the tractogram to align max_voxel_rotated with the coordinate
   system. (see scil_tractogram_apply_transform.py).


.. image:: https://github.com/user-attachments/assets/43cebcbe-e3b1-4ca0-999e-e042db8aa937
   :alt: Metrics (without max_voxel_rotated) visualized in 3D

.. image:: https://github.com/user-attachments/assets/924ab3f9-33da-458f-a98b-b4e88b051ae8
   :alt: max_voxel_rotated visualized in 3D

Note: This information can be useful for analyzing the
reconstruction obtained through tracking, as well as for performing
track density imaging at extreme resolutions.

Performing fibertube tracking
-----------------------------

We're finally at the tracking phase! Using the script
``scil_fibertube_tracking.py``, you are able to track without relying on
a discretized grid of directions or fODFs. Instead, you will be
propagating a streamline through fibertubes and controlling the
resolution by using a ``blur_radius``. The way it works is as follows:

Seeding
~~~~~~~

A number of seeds is set randomly within the first segment of
every fibertube. We can however change the number of fibertubes that
will be tracked, as well as the amount of seeds within each. (See
Seeding options in the help menu).

Tracking
~~~~~~~~

When the tracking algorithm is about to select a new direction to
propagate the current streamline, it will build a sphere of radius
``blur_radius`` and pick randomly from all the fibertube segments
intersecting with it. The larger the intersection volume, the more
likely a fibertube segment is to be picked and used as a tracking
direction.


.. image:: https://github.com/user-attachments/assets/0308c206-c396-41c5-a0e1-bb69b692c101
   :alt: Visualization of the blurring sphere intersecting with segments


For more information and better visualization, watch the following
presentation: https://docs.google.com/presentation/d/1nRV2j_A8bHOcjGSHtNmD8MsA9n5pHvR8/edit#slide=id.p19


This makes fibertube tracking inherently probabilistic.
Theoretically, with a ``blur_radius`` of 0, any given set of coordinates
has either a single tracking direction because it is within a fibertube,
or no direction at all from being out of one. In fact, this behavior
won't change until the diameter of the sphere is larger than the
smallest distance separating two fibertubes. When this happens, more
than one fibertubes will intersect the ``blur_radius`` sphere and
introduce partial volume effect.

The interface of the script is very similar to
``scil_tracking_local_dev.py``, but simplified and with a ``blur_radius``
option. Let us do:

::

   scil_fibertube_tracking.py fibertubes.trk tracking.trk --blur_radius 0.1 --step_size 0.1 --nb_fibertubes 3 --out_config tracking_config.json --processes 4 -v -f

This should take a minute or two and will produce 15 streamlines. The loading
bar of each thread will only update every 100 streamlines. It may look
like it's frozen, but rest assured. it's still going!

Reconstruction analysis
~~~~~~~~~~~~~~~~~~~~~~~

By using the ``scil_fibertube_score_tractogram.py`` script, you are able
to obtain measures on the quality of the fibertube tracking that was
performed.

Each streamline is associated with an "Arrival fibertube segment", which is
the closest fibertube segment to its before-last coordinate. We then define
the following terms:

VC: "Valid Connection": A streamline whose arrival fibertube segment is
the final segment of the fibertube in which is was originally seeded.

IC: "Invalid Connection": A streamline whose arrival fibertube segment is
the start or final segment of a fibertube in which is was not seeded.

NC: "No Connection": A streamline whose arrival fibertube segment is
not the start or final segment of any fibertube.

.. image:: https://github.com/user-attachments/assets/ac36d847-2363-4b23-a69b-43c9d4d40b9a
   :alt: Visualization of VC, IC and NC

The "absolute error" of a coordinate is the distance in mm between that
coordinate and the closest point on its corresponding fibertube. The
average of all coordinate absolute errors of a streamline is called the
"Mean absolute error" or "mae".

Here is a visual representation of streamlines (Green) tracked along a fibertube
(Only the centerline is shown in blue) with their coordinate absolute error (Red).


.. image:: https://github.com/user-attachments/assets/62324b66-f66b-43ae-a772-086560ef713a
   :alt: Visualization of the coordinate absolute error

Computed metrics:

-  vc_ratio: Number of VC divided by the number of streamlines.
-  ic_ratio: Number of IC divided by the number of streamlines.
-  nc_ratio: Number of NC divided by the number of streamlines.
-  mae_min: Minimum MAE for the tractogram.
-  mae_max: Maximum MAE for the tractogram.
-  mae_mean: Average MAE for the tractogram.
-  mae_med: Median MAE for the tractogram.

To score the produced tractogram, we run:

::

   scil_fibertube_score_tractogram.py fibertubes.trk tracking.trk tracking_config.json reconstruction_metrics.json -f

giving us the following output in ``reconstruction_metrics.json``:

::

   {
     "vc_ratio": 0.3333333333333333,
     "ic_ratio": 0.4,
     "nc_ratio": 0.26666666666666666,
     "mae_min": 0.004093314514974615,
     "mae_max": 10.028780087103556,
     "mae_mean": 3.055598084631571,
     "mae_med": 0.9429987731800447
   }

This data tells us that 1/3 of streamlines had the end of their own fibertube as
their arrival fibertube segment (``"vc_ratio": 0.3333333333333333``).
For 40% of streamlines, their arrival fibertube segment was the start or end of
another fibertube (``"ic_ratio": 0.4``). 26% of streamlines had an arrival fibertube
segment that was not a start or end segment (``"nc_ratio": 0.26666666666666666``).
Lastly, we notice that the streamline with the "worst" trajectory was on average
~10.03mm away from its fibertube (``"mae_max": 10.028780087103556``).

This is not very good, but it's to be expected with a --blur_radius and
--step_size of 0.1. If you have a few minutes, try again with 0.01!

End of Demo
-----------
