Using trk or tck files
======================

TRK files already have some metadata stored in their headers. The header knows the position of the tractogram in the space. When using TCK files however, this information is not given. You may view this file as a list of streamline coordinates, but no information on where point (0,0,0) actually is. All our scripts using tractograms offer the option ``--reference``. This is meant to be used when your input is a TCK file. We will read the information in another associated file (for instance a nifti volume) that is in the same space as your tractogram. It should be a file from the same subject, in the same space.

In our tutorials using tractograms, we make sure to use inputs as TRK files to declutter our command lines, removing the ``--reference`` at each new call. Don't hesitate to add them back if your own data is in TCK format.

Else, you can use this to convert your file to TRK before using it:

.. code-block:: bash

    scil_tractogram_convert my_file.tck my_file.trk --reference my_volume.nii.gz

You may also use :ref:`scil_header_print_info` on both types of files. The result will look like::

    **************************************
    File name:    tracto.tck
    **************************************
      Dimensions: (138, 166, 134)
      Voxel size: (1.0, 1.0, 1.0)
      Datatype: Float32LE
      Orientation: ('R', 'A', 'S')
      Afine (vox2rasmm):
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
