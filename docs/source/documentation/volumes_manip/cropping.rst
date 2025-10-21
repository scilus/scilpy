.. _page_cropping:

Cropping volumes
================

To crop a file, you may use :ref:`scil_volume_crop`.

.. tip::
    This is used for instance at the beggining of tutorial script for `aodf </_static/bash/reconst/aodf_scripts.sh>`_ or `msmt </_static/bash/reconst/msmt_fodf.sh>`_

You first need to find the bounding box of your file. It is a json file looking like:

.. code-block:: json

    {
        "minimums": [-67.5, -96.0, -64.5],
        "maximums": [75.0, 71.5, 75.5],
        "voxel_size": [2.5, 2.5, 2.5]}
    }

You can find this information with:

.. code-block:: bash

    scil_volume_crop $my_file test.nii.gz --out bounding_box.json
    cat bounding_box.json

So let's change the bounding box. Open the json file in any text editor or code editor and change the content. For instance:

.. code-block:: json

    {
        "minimums": [-20, -30, -20],
        "maximums": [20, 30, 20],
        "voxel_size": [2.5, 2.5, 2.5]}
    }

And run:

.. code-block:: bash

    scil_volume_crop your_file out_file --input_bbox $out_dir/bounding_box.json

You can check the final input size with either mrinfo from MRtrix or :ref:`scil_header_print_info`.
