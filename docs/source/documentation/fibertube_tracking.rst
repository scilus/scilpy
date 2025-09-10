Introduction to the Fibertube Tracking environment through an interactive demo.
===============================================================================

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

.. image:: https://github.com/user-attachments/assets/9a1974cc-452c-4bac-93e1-aaa02a7ea169
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

   scil_tractogram_resample_nb_points centerlines.trk centerlines_resampled.trk --step_size 0.2 -f

Next, we want to filter out intersecting fibertubes (collisions), to
make the data anatomically plausible and ensure that there exists a
resolution at which there is no unit of space containing partial
volume.

.. image:: https://github.com/user-attachments/assets/d9b0519b-c1e3-4de0-8529-92aa92041ce2
   :alt: Fibertube intersection visualized in 3D

This is accomplished using ``scil_tractogram_filter_collisions``.

::

   scil_tractogram_filter_collisions centerlines_resampled.trk diameters.txt fibertubes.trk --save_colliding --out_metrics metrics.json -v -f

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

``scil_tractogram_print_info fibertubes.trk``

Visualising collisions
----------------------

By calling:

::

   scil_viz_tractogram_collisions centerlines_resampled_invalid.trk --in_tractogram_obstacle centerlines_resampled_obstacle.trk --ref_tractogram centerlines.trk

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
   system. (see scil_tractogram_apply_transform).


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
``scil_fibertube_tracking``, you are able to track without relying on
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
``blur_radius`` and examine all the fibertube segments intersecting
with it. We get a list of segment directions with each a probability
of getting picked. This propability is based on the segment's volume of
intersection with the sphere. So the more a fibertube segment intersects
the sphere, the more likely it is to be picked and used as a tracking
direction.

.. image:: https://github.com/user-attachments/assets/782bb6d2-0e5c-48a5-8606-1d95d6675e0d
   :alt: Visualization of the blurring sphere intersecting with segments

This is similar to computing the Track Orientation Distribution (TOD)
but it is not yet represented as a spherical function. It is merely
an array of directions. This process is very power and provides much
better reconstructions than traditional tractography. This is because
each potential direction is an actual fibertube orientation. It is not
possible to step in between them and get lost.

To align as best as possible the performances of fibertube tracking with
traditional tractography given the same resolution, the fibertube
directions near each tracking position should be mapped on a sphere
and then approximated with spherical harmonics. This gives us a
fibertube ODF or ftODF. A ftODF is nothing short of a local, volume-weighted TODI!
It can be used to track probabilistically or deterministically through peak
extraction.

For more information and better visualization, watch the following
presentation: https://docs.google.com/presentation/d/1nRV2j_A8bHOcjGSHtNmD8MsA9n5pHvR8/edit#slide=id.p19


Theoretically, with a ``blur_radius`` of 0, any given set of coordinates
has either a single tracking direction because it is within a fibertube,
or no direction at all from being out of one. In fact, this behavior
won't change until the diameter of the sphere is larger than the
smallest distance separating two fibertubes. When this happens, more
than one fibertubes will intersect the ``blur_radius`` sphere and
introduce partial volume effect.

The interface of the script is very similar to
``scil_tracking_local_dev``, but simplified and with a ``blur_radius``
option. Let us do:

::

   scil_fibertube_tracking fibertubes.trk tracking.trk --blur_radius 0.1 --step_size 0.1 --nb_fibertubes 3 --out_config tracking_config.json --processes 4 -v -f

This should take a minute or two and will produce 15 streamlines.

Reconstruction analysis
~~~~~~~~~~~~~~~~~~~~~~~

By using the ``scil_fibertube_score_tractogram`` script, you are able
to obtain measures on the quality of the fibertube tracking that was
performed.

First, streamlines are truncated to remove their last coordinate. It
was not in range or aligned with any fibertube, and thus represents
an invalid step that should be removed. Each streamline is then
associated with an "Termination fibertube segment", which is the closest
fibertube segment to its last coordinate. We define the following terms:

VC: "Valid Connection": A streamline whose termination fibertube segment is
the final segment of the fibertube in which is was originally seeded.

IC: "Invalid Connection": A streamline whose termination fibertube segment is
the start or final segment of a fibertube in which is was not seeded.

NC: "No Connection": A streamline whose termination fibertube segment is
not the start or final segment of any fibertube.

The "absolute error" of a coordinate is the distance in mm between that
coordinate and the closest point on its corresponding fibertube. The
average of all coordinate absolute errors of a streamline is called the
"Mean absolute error" (MAE). The "endpoint distance" is the distance
between the final coordinate of a streamline and the final coordinate of
its fibertube. Typically, an IC is expected to have a high impact on MAE
and a medium impact on the endpoint distance. A NC might have a low impact
on MAE but a high impact on the endpoint distance.

In this image, green is a VC, yellow is an IC and red is a NC. The
coordinate error is represented by black lines, and the thicker one is the
endpoint distance. The white and black circles are the seeding and termination
locations respectively.

.. image:: https://github.com/user-attachments/assets/dbbeea60-54e5-4269-a387-2ea3e6b06bcc
   :alt: Visualization of all metrics

The next image features actual streamlines from this demo (Green) tracked
along a fibertube (Only the centerline is shown in blue) with their coordinate
error (Red).

.. image:: https://github.com/user-attachments/assets/62324b66-f66b-43ae-a772-086560ef713a
   :alt: Visualization of the coordinate absolute error through a real tracking

Computed metrics:

-  vc_ratio: Number of VC divided by the number of streamlines.
-  ic_ratio: Number of IC divided by the number of streamlines.
-  nc_ratio: Number of NC divided by the number of streamlines.
-  mae_min: Minimum MAE for the tractogram.
-  mae_max: Maximum MAE for the tractogram.
-  mae_mean: Average MAE for the tractogram.
-  mae_med: Median MAE for the tractogram.
-  endpoint_dist_min: Minimum endpoint distance for the tractogram.
-  endpoint_dist_max: Maximum endpoint distance for the tractogram.
-  endpoint_dist_mean: Average endpoint distance for the tractogram.
-  endpoint_dist_med: Median endpoint distance for the tractogram.

To score the produced tractogram, we run:

::

   scil_fibertube_score_tractogram fibertubes.trk tracking.trk tracking_config.json reconstruction_metrics.json -f

giving us the following output in ``reconstruction_metrics.json``:

::

   {
     "vc_ratio": 0.4,
     "ic_ratio": 0.4,
     "nc_ratio": 0.2,
     "mae_min": 0.010148868692306913,
     "mae_max": 9.507027053725844,
     "mae_mean": 2.974526457370884,
     "mae_med": 1.0589793885582628,
     "endpoint_dist_min": 0.03928468596245134,
     "endpoint_dist_max": 73.03314003616677,
     "endpoint_dist_mean": 25.675430285869695,
     "endpoint_dist_med": 34.45811150476051
   }

This data tells us that:

- 40% of streamlines had the end of their own fibertube as
  their termination fibertube segment. (``"vc_ratio": 0.3``)
- 40% of streamlines did connect their own fibertube, but instead another fibertube.
  (``"ic_ratio": 0.4``)
- 26% of streamlines had an termination fibertube segment that
  was not a start nor end segment. (``"nc_ratio": 0.2``)
- Lastly, we notice that the streamline with the "worst" trajectory was on average
  ~9.5mm away from its fibertube. (``"mae_max": 9.507027053725844``)
- Streamlines terminated on average 25.68mm away from the ending of their own
  fibertube. (``endpoint_dist_mean": 25.675430285869695``)

To make sense of these numbers, here is a visual representation of the
tracking and scoring you just performed:

Blue: fibertubes that were seeded
Red: streamlines
Yellow: coordinate absolute error (AE)
Pink: Maximum endpoint distance

.. image:: https://github.com/user-attachments/assets/552f0d64-c8f3-4859-879b-531599515ba5
   :alt: Visualization tracking and scoring

As you can see, the maximum AE is not equal to the maximum endpoint distance.
This is because AE connects each streamline coordinate with the closest fibertube
coordinate.

This reconstruction is not very good, but it is to be expected with
a --blur_radius and --step_size of 0.1. If you have a few minutes,
try again with 0.01!

End of Demo
-----------
