.. _using_bash:

Using bash to loop on subjects
==============================

Most of our tutorials give you examples of the command-line to execute scripts on a single subject. They also give you the option of downloading the bash script associated to the tutorial.

Once you have mastered all processing steps to run a start-to-finish pipeline for a subject (see for instance :ref:`page_from_raw_to_tractogram`), you might want to start running pipelines on many subjects automatically without having to enter the command lines manually again for each new subject. The simplest way to do that is to loop on all subjects in a bash script, which is what we will show here.


.. code-block:: bash

    for subj in "subj1 subj2"
    do
        # Run all steps!
        ...
    done

If your database is well organized and looking like this::

    ├── root_folder
    │   ├── subj1
    │   ├── subj2

You can do:

.. code-block:: bash

    for subj_folder in $root_folder/*
    do
        subj_name=${subj_folder#$root_folder}
        echo "Processing subject $subj_name!"

        # Run all steps!
        ...
    done


Note however that this will still run all commands one after the other, which prevents scalability to large cohorts. For more computationally efficient ways to run pipelines, see :ref:`use_nextflow`.