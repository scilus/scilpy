Instructions for estimation of bundle diameter
========================================================

Script to estimate the diameter of bundle(s) along their length.
The script expects:

- bundles with coherent endpoints from scil_uniformize_streamlines_endpoints.py

- labels maps with around 5-50 points scil_compute_bundle_voxel_label_map.py
    - <5 is not enough, high risk of bad fit

    - >50 is too much, high risk of bad fit
- bundles that are close to a tube
    - without major fanning in a single axis
    
    - fanning is in 2 directions (uniform dispersion) good approximation

The scripts prints a JSON file with mean/std to be compatible with tractometry.
WARNING: STD is in fact an ERROR measure from the fit and NOT an STD.

Since the estimation and fit quality is not always intuitive for some bundles
and the tube with varying diameter is not easy to color/visualize,
the script comes with its own VTK rendering to allow exploration of the data.
(optional).

The use of the **--fitting_func** option for the least-square fitting:

- ``None``: Default, all points are weighted equally no matter their (normalized 0-1) distance to the barycenter.
- ``lin_up``': Points are weighted using their distance to the barycenter (linear, x). Farther = increased weight. Bigger envelope. 
- ``lin_down``: Points are weighted using their distance to the barycenter (linear, 1-x). Farther = decreased weight. Smaller envelope.
- ``exp``: Points are weighted using their distance to the barycenter (exponential, e^x). Farther = decreased weight. Smaller envelope.
- ``inv``: Points are weighted using their distance to the barycenter (inverse, 1/x). Farther = decreased weight. Much smaller envelope.
- ``log``: Points are weighted using their distance to the barycenter (logarithmic, ln x+1). Farther = increased weight. Bigger envelope.