Introduction to the Fibertube Tracking environment through an interactive demo.
====

In this demo, you will be introduced to the main scripts of this project
as you apply them on simple data. Our main objective is better
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

To obtain the data required for this demo, open a terminal and activate your
scilpy virtual environment. Then, navigate to the scilpy repository on your
computer and enter the command: ``pytest -v``. This will pull all the files
required for testing scilpy scripts and then begin the testing sequence. As
soon as the tests start, you can abort the process and navigate to any location
outside of the scilpy repository that you see fit for this demo.

Then, execute the following command:
``cp ~/.scilpy/others/fibercup_bundles.trk ./centerlines.trk`` to bring our
data to your current location and rename it to ``centerlines.trk``.

It is a subset of the FiberCup phantom ground truth:

.. image:: https://github.com/user-attachments/assets/3be43cc9-60ec-4e97-95ef-a436c32bba83
   :alt: Fibercup subset visualized in 3D

Now that we have a tractogram to act as our set of centerlines, we will need
to create a file containing the diameters. To do this, create a text file
named `diameters.txt` and enter `0.001` on the very first line. This single
diameter will later be applied to all the centerlines to form a set of fibertubes.


The first thing to do to is resample ``centerlines.trk`` so that each
centerline is formed of segments no longer than 0.2 mm.

Note: This is because the next script will rely on a KDTree to find
all neighboring fibertube segments of any given point. Because the
search radius is set at the length of the longest fibertube segment,
the performance drops significantly if they are not shortened to
~0.2mm.

To resample a tractogram, we can use this script from scilpy:

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

   scil_tractogram_filter_collisions.py centerlines_resampled.trk diameters.txt fibertubes.trk --save_colliding --out_metrics metrics.txt -v -f

After 3-5 minutes, you should get something like:

::

   ...
   ├── centerlines_resampled_obstacle.trk
   ├── centerlines_resampled_invalid.trk
   ├── fibertubes.trk
   ├── metrics.txt
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

   scil_viz_tractogram_collisions.py centerlines_resampled_invalid.trk --obstacle centerlines_resampled_obstacle.trk --ref_tractogram centerlines.trk

You are able to see exactly which streamline has been filtered
("invalid" - In red) as well as the streamlines they collided with
("obstacle" - In green). In white and lower opacity is the original
tractogram passed as ``--ref_tractogram``.

.. image:: https://github.com/user-attachments/assets/9cb95488-227f-4c96-b88c-ead9100ac708
   :alt: Filtered intersections visualized in 3D

Fibertube metrics
-----------------

Before we get into tracking. Here is an overview of the metrics that we
saved in ``metrics.txt``. (Values expressed in mm):

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


|Metrics (without max_voxel_rotated) visualized in 3D|

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

For now, a number of seeds is set randomly within the first segment of
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

   scil_fibertube_tracking.py fibertubes.trk tracking.trk 0.01 0.01 --nb_fibertubes 3 --out_config tracking_config.txt --processes 4 -v -f

This should take around 5 minutes. The loading bar of each thread will
only update every 100 streamlines. It may look like it's frozen, but it
rest assured it's still going!

Reconstruction analysis
~~~~~~~~~~~~~~~~~~~~~~~

By using the ``scil_fibertube_score_tractogram.py`` script, you are able
to obtain measures on the quality of the fibertube tracking that was
performed. Here is a description of the computed metrics:

VC: "Valid Connection": A streamline that ended within the final segment
of the fibertube in which it was seeded.

IC: "Invalid Connection": A streamline that ended in the first or final
segment of another fibertube.

NC: "No Connection": A streamline that has not ended in the first or final
segment of any fibertube.

.. image:: https://github.com/user-attachments/assets/bc61ce87-6581-4714-83d2-9602380f2697
   :alt: Visual representation of VC, IC, and NC

Res_VC: "Resolution-wise Valid Connection": A streamline that passes
closer than [blur_darius] away from the last segment of the fibertube
in which it was seeded.

Res_IC: "Resolution-wise Invalid Connection": A streamline that passes
closer than [blur_darius] away from the first or last segment of another
fibertube.

Res_NC: "Resolution-wise No Connection": A streamlines that does not pass
closer than [blur_radius] away from the first or last segment of any
fibertube.

.. image:: https://github.com/user-attachments/assets/d8c1a376-e2b9-454c-9234-5a124bde3c02
   :alt: Visual representation of Res_VC, Res_IC, and Res_NC

The "absolute error" of a coordinate is the distance in mm between that
coordinate and the closest point on its corresponding fibertube. The
average of all coordinate absolute errors of a streamline is called the
"Mean absolute error" or "mae".

Here is a visual representation of streamlines (Green) tracked along a fibertube
(Only the centerline is shown in blue) with their coordinate absolute error (Red).


.. image:: https://github.com/user-attachments/assets/62324b66-f66b-43ae-a772-086560ef713a
   :alt: Visualization of the coordinate absolute error

Computed metrics:

-  vc_ratio Number of VC divided by the number of streamlines.
-  ic_ratio Number of IC divided by the number of streamlines.
-  nc_ratio Number of NC divided by the number of streamlines.
-  res_vc_ratio Number of Res_VC divided by the number of streamlines.
-  res_ic_ratio Number of Res_IC divided by the number of streamlines.
-  res_nc_ratio Number of Res_NC divided by the number of streamlines.
-  mae_min Minimum MAE for the tractogram.
-  mae_max Maximum MAE for the tractogram.
-  mae_mean Average MAE for the tractogram.
-  mae_med Median MAE for the tractogram.

To score the produced tractogram, we run:

::

   scil_fibertube_score_tractogram.py fibertubes.trk tracking.trk tracking_config.txt reconstruction_metrics.txt -v -f

giving us the following output in ``reconstruction_metrics.txt``:

::

   {
     "vc_ratio": 0.13333333333333333,
     "ic_ratio": 0.0,
     "nc_ratio": 0.8666666666666667,
     "res_vc_ratio": 0.8,
     "res_ic_ratio": 0.13333333333333333,
     "res_nc_ratio": 0.06666666666666667,
     "mae_min": 2.023046655518677e-06,
     "mae_max": 5.140102678615527,
     "mae_mean": 0.7342005034643644,
     "mae_med": 0.0009090212918552973
   }

This data tells us that about 13% of our streamlines managed to stay
within the fibertube in which they were seeded (``"vc_ratio": 0.13333~``).
However, 80% of streamlines ended closer than one ``blur_radius`` away from
the end of their respective fibertube (``"res_vc_ratio": 0.8``).
Lastly, we notice that the streamline with the "worst" trajectory was on average
5.14mm away from its fibertube (``"mae_max": 5.140102678615527``).

End of Demo
-----------

.. |Metrics (without max_voxel_rotated) visualized in 3D| image:: https://github.com/user-attachments/assets/43cebcbe-e3b1-4ca0-999e-e042db8aa937
