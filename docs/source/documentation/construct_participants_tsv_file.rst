Instructions to write the tsv files "participants.tsv" for the script scil_group_comparison.py
===============================================================================================

The TSV file should follow the BIDS `specification <https://bids-specification.readthedocs.io/en/stable/03-modality-agnostic-files.html#participants-file>`_.

Header
------

First row of the tsv file should be written as follow:

participant_id categorical_var_1 categorical_var_2 ...

(ex: participant_id sex nb_children)

The categorical variable name are the "group_by" variable that can be called by scil_group_comparison.py

Specific row
------------
The other rows should be written according to the header
(ex: patient_32 F 3)

::

    participant_id sex nb_children
     patient_32 F 3
     Patient_49 M 0
     patient_44 F 1
     patient_47 M 1
     patient_28 M 2
