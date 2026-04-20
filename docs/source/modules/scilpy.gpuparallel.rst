scilpy.gpuparallel package
===============================

scilpy.gpuparallel.opencl\_utils module
------------------------------------------------------

OpenCL environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The OpenCL helper accepts the following environment variable:

- SCILPY_OPENCL_BUILD_OPTIONS: space-separated compiler flags passed to
  ``pyopencl.Program.build``. Default is ``-w`` to keep warnings quiet.
  Example:

  .. code-block:: bash

     export SCILPY_OPENCL_BUILD_OPTIONS="-cl-fast-relaxed-math -Werror"

.. automodule:: scilpy.gpuparallel.opencl_utils
   :members:
   :undoc-members:
   :show-inheritance:
