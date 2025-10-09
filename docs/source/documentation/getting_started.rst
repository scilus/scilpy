Getting ready for tutorials
===========================

Scilpy is the main library supporting research and development at the Sherbrooke Connectivity Imaging Lab (SCIL).

Scilpy mainly comprises tools and utilities to quickly work with diffusion MRI.
Most of the tools are based on or are wrappers of the DIPY library, and most of them will eventually be migrated to DIPY. Those tools implement the recommended workflows and parameters used in the lab.


Installation 
#############
⚠️ We highly suggest to install uv to speedup scilpy installation: https://docs.astral.sh/uv/getting-started/installation/


.. code-block:: bash

    # If you are using Python3.10 or Python3.11, export this variable before installing
    export SETUPTOOLS_USE_DISTUTILS=stdlib

    uv pip install scilpy # For the most recent release from PyPi


Download data
#############

In order to follow the tutorials we highly encourage you to download this archive full of data.

.. code-block:: bash

    wget -O WB-common.zip https://scil.usherbrooke.ca/scil_test_data/dvc-store/files/md5/21/0b52975dc2b84d94426166e74245a3
    unzip WB-common.zip
    rm WM-common.zip