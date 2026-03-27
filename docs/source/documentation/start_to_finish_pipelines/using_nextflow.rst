.. _using_nextflow:

Using nextflow
==============

Most of our tutorials give you examples of the command-line to execute scripts on a single subject. Once you have mastered all processing steps for a subject, (see for instance :ref:`page_from_raw_to_tractogram`), you might want to start running many subjects automatically without having to enter the command lines manually again for each new subject. The simplest way to do that is to loop on all subjects in a bash script (:ref:`using_bash`). However, this will still run all commands one after the other. For an efficient creation of pipelines, with parallel processing of subjects, and even of various processing steps for a same subject, one option is to use Nextlow.

.. Should we add more? Move stuff from the scil-doc? Or do we keep it short and sweet?

Nextflow
********

See `nextflow's official page <https://www.nextflow.io/>`_ for a description.

nf-neuro
********

`nf-neuro <https://scilus.github.io/nf-neuro/>`_ is an open-source initiative originally developed by the Sherbrooke Connectivity Imaging Lab (SCIL), which is also responsible for scilpy. It builds on nextflow to offer an easy way to create medical imaging pipelines. It will be described soon in an upcoming paper.


Tractoflow
**********

One already published nextflow-based pipeline that uses many scilpy scripts is `Tractoflow <https://tractoflow-documentation.readthedocs.io>`_
