Introduction to the Fibertube Tracking project through an interactive demo.
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

.. image:: https://github.com/user-attachments/assets/2e4253b3-05ca-4881-8482-3a96db0a76c9
   :alt: Fibertube visualized in 3D

Methodology
-----------

This project can be split into 3 major steps:

-  Preparing ground-truth data We will be using the ground-truth of
   simulated phantoms of streamlines with a diameter (giving us
   fibertubes) ensuring that they are void of any collision, i.e.
   fibertubes in the simulated phantom should not intersect one another.
   This is physically impossible to respect the geometry of axons.
-  Tracking and experimentation We will perform 'Fibertube Tracking' on
   our newly formed set of fibertubes with a variety of parameter
   combinations.
-  Evaluation metrics computation By passing the resulting tractogram
   through different evaluation scripts (like Tractometer), we will
   acquire connectivity and fiber reconstruction scores for each of the
   parameter combinations.

Preparing the data
------------------

To obtain the data required for this demo, activate your scilpy virtual
environment, navigate to the scilpy repository on your computer and enter
the command: `pytest -v`. This will pull all the files required for testing
and then begin the test sequence. As soon as the tests start, you can abort
the process and navigate to any location outside of the scilpy repository that
you see fit for this demo.

Then, execute the following command:
`cp ~/.scilpy/others/fibercup_bundles.trk ./centerlines.trk` to bring the data
to your current location and rename it to `centerlines.trk`.

Here is the tractogram that you have just fetched:
.. image:: https://github.com/user-attachments/assets/3be43cc9-60ec-4e97-95ef-a436c32bba83
   :alt: Fibercup subset visualized in 3D

Now that we have a tractogram to act as our set of centerlines, we will need
to create a file containing the diameters. To do this, create a text file
named `diameters.txt` and enter `0.001` on the very first line. This single
diameter will be used in conjunction with the centerlines to form a set of
fibertubes.


The first thing to do is resample ``centerlines.trk`` so that each
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
make the data anatomically plausible and remove any partial volume
effect.

.. image:: https://github.com/user-attachments/assets/d9b0519b-c1e3-4de0-8529-92aa92041ce2
   :alt: Fibertube intersection visualized in 3D

This is accomplished using ``scil_tractogram_filter_collisions.py``.

::

   scil_tractogram_filter_collisions.py centerlines_resampled.trk diameters.txt fibertubes.trk --save_colliding --out_metrics metrics.txt -v -f

After a short wait, you should get something like:

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

.. image:: https://github.com/user-attachments/assets/d88982c3-2228-41e1-a81a-d2ce23ec8151
   :alt: Filtered intersections visualized in 3D

Fibertube metrics
~~~~~~~~~~~~~~~~~

Before we get into tracking. Here is an overview of the metrics that we
saved in ``metrics.txt``. (Values expressed in mm):

-  ``min_external_distance``: Smallest distance separating two
   fibertubes, outside their diameter.
-  ``max_voxel_anisotropic``: Diagonal vector of the largest possible
   anisotropic voxel that would not intersect two fibertubes.
-  ``max_voxel_isotropic``: Isotropic version of max_voxel_anisotropic
   made by using the smallest component. Ex: max_voxel_anisotropic: (3,
   5, 5) => max_voxel_isotropic: (3, 3, 3)
-  ``max_voxel_rotated``: Largest possible isotropic voxel obtainable if
   the tractogram is rotated. It is only usable if the entire tractogram
   is rotated according to [rotation_matrix]. Ex: max_voxel_anisotropic:
   (1, 0, 0) => max_voxel_rotated: (0.5774, 0.5774, 0.5774)

|Metrics (without max_voxel_rotated) visualized in 3D|

.. image:: https://github.com/user-attachments/assets/924ab3f9-33da-458f-a98b-b4e88b051ae8
   :alt: max_voxel_rotated visualized in 3D

Note: This information can be useful for analyzing the
      reconstruction obtained through tracking, as well as for performing
      track density imaging.

Performing fibertube tracking
-----------------------------

We're finally at the tracking phase! Using the script
``scil_fibertube_tracking.py``, you are able to track without relying on
a discretized grid of directions or fODFs. Instead, you will be
propagating a streamline through fibertubes and controlling the
resolution by using a ``blur_radius``. The way it works is as follows:

Tracking
~~~~~~~~

When the tracking algorithm is about to select a new direction to
propagate the current streamline, it will build a sphere of radius
``blur_radius`` and pick randomly from all the fibertube segments
intersecting with it. The larger the intersection volume, the more
likely a fibertube segment is to be picked and used as a tracking
direction. This makes fibertube tracking inherently probabilistic.
Theoretically, with a ``blur_radius`` of 0, any given set of coordinates
has either a single tracking direction because it is within a fibertube,
or no direction at all from being out of one. In fact, this behavior
won't change until the diameter of the sphere is larger than the
smallest distance separating two fibertubes. When this happens, more
than one fibertubes will intersect the ``blur_radius`` sphere and
introduce partial volume effect.

Seeding
~~~~~~~

For now, a number of seeds is set randomly within the first segment of
every fibertube. We can however change the number of fibertubes that
will be tracked, as well as the amount of seeds within each. (See
Seeding options in the help menu).

.. raw:: html

   <br>
   The interface of the script is very similar to `scil_tracking_local_dev.py`, but simplified and with a `blur_radius` option. Let us do:

::

   scil_fibertube_tracking.py fibertubes.trk tracking.trk 0.01 0.01 --nb_fibertubes 3 --out_config tracking_config.txt --processes 4 -v -f

This should take a few minutes at most. However, if you don't mind
waiting a little bit, feel free to play with the parameters and explore
the resulting tractogram.

Note: Given the time required for each streamline, the
      ``--processes`` parameter will be very useful.

Reconstruction analysis
~~~~~~~~~~~~~~~~~~~~~~~

By using the ``scil_fibertube_score_tractogram.py`` script, you are able
to obtain measures on the quality of the fibertube tracking that was
performed. Here is a description of the computed metrics:

VC: "Valid Connection": A streamline that passes WITHIN the final
segment of the fibertube in which it was seeded. IC: "Invalid
Connection": A streamline that ended in the final segment of another
fibertube. NC: "No Connection": A streamlines that has not ended in the
final segment of any fibertube.

.. image:: https://github.com/user-attachments/assets/4871cb09-313e-499a-b56d-a668bdb631db
   :alt: Visual representation of VC, IC, and NC

Res_VC: "Resolution-wise Valid Connection": A streamline that passes
closer than [blur_darius] away from the last segment of the fibertube in
which it was seeded. Res_IC: "Resolution-wise Invalid Connection": A
streamline that passes closer than [blur_darius] away from the first or
last segment of another fibertube. Res_NC: "Resolution-wise No
Connection": A streamlines that does not pass closer than [blur_radius]
away from the first or last segment of any fibertube.

.. image:: https://github.com/user-attachments/assets/c480f5e6-14f8-456a-b8e8-77569661c452
   :alt: Visual representation of Res_VC, Res_IC, and Res_NC

The "absolute error" of a coordinate is the distance in mm between that
coordinate and the closest point on its corresponding fibertube. The
average of all coordinate absolute errors of a streamline is called the
"Mean absolute error" or "mae".

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
     "vc_ratio": 0.0,
     "ic_ratio": 0.0,
     "nc_ratio": 1.0,
     "res_vc_ratio": 0.4,
     "res_ic_ratio": 0.0,
     "res_nc_ratio": 0.6,
     "mae_min": 0.0014691361472782293,
     "mae_max": 0.0055722481609273775,
     "mae_mean": 0.003883039143304128,
     "mae_med": 0.003927314695651083
   }

This data tells us that none of our streamline managed to stay within
the fibertube in which it was seeded (``"vc_ratio": 0.0``). However, 40%
of streamlines pass closer than one ``blur_radius`` away from the end of
their respective fibertube (``"res_vc_ratio": 0.4``). Lastly, we notice
that the streamline with the "worst" trajectory was on average 5.5um
away from its fibertube (``"mae_max": 0.0055722481609273775``).

End of Demo
-----------

.. |Metrics (without max_voxel_rotated) visualized in 3D| image:: https://github.com/user-attachments/assets/43cebcbe-e3b1-4ca0-999e-e042db8aa937
