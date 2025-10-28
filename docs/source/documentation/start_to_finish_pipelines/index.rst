-- From raw data to a complete project --
===============================

A unique aspect of scilpy is its emphasis on ensuring that each script performs a single, well-defined task.  This granularity leads to a very high number of scripts, but allows you to manipulate data exactly as you want. You may then organize processing steps in your favorite order. To automate processes, for a single subject or a larger database, options range from simply aligning command lines sequentially in a bash script, to using more complex workflow management systems such as nextflow, nipype, snakemake, `nf-neuro <https://scilus.github.io/nf-neuro/>`_ and more, which allow parallel processing over various subjects for accelerated workflows. Users interested in large cohorts processing may discover `Tractoflow <https://tractoflow-documentation.readthedocs.io/en/latest/>`_, which includes 26 scilpy scripts.

.. toctree::
   :maxdepth: 1

   using_bash
   using_nextflow

Here are examples of start-to-finish automated processing of a single subject:

.. toctree::
   :maxdepth: 1

   from_raw_data_to_tractogram
