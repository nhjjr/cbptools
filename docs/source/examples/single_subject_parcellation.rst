.. _ExampleSingleSubjectParcellation:

===========================
Single-subject Parcellation
===========================
When the input masks are subject-specific (i.e., the masks are in the subject’s native space), group-level
parcellations cannot be computed. This is because CBPtools does not perform any transformations to bring native
data into a common reference space. As a result, only the single-subject parcellations can be computed. When
subject-specific input masks are provided, CBPtools will instead generate all figures for each individual subject,
rather than for the entire group. Note that in any case, seed and target masks must always match the same space.
CBPtools can also generate subject-specific output on top of the group-level clustering output if specified in the
configuration file. This option is turned off by default as it requires more computation time.

Single-subject parcellation data can be defined in the configuration file as follows:

.. code-block:: yaml

    modality: rsfmri

    data:
        time_series: /path/to/data_set/{participant_id}/time_series.nii.gz
        confounds:
            columns: ['constant', 'wm.linear', 'csf.linear', 'motion-*']
            delimiter: \t
            file: /path/to/data_set/{participant_id}/confounds.tsv
        masks:
            seed: /path/to/data_set/{participant_id}/seed_mask.nii.gz
            target: /path/to/data_set/{participant_id}/target_mask.nii.gz
            space: native

    ...

The important fields are `data.masks.space` which must be set to 'native', and both the `data.masks.seed` and
`data.masks.target` must be defined and contain the `{participant_id}` wildcard.

As for the parameters, the entire `parameters.grouping` field can be ignored.