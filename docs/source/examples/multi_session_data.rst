.. _ExampleMultiSessionData:

===================
Multi-session Data
===================
Multi-session subject data (i.e., a data set with multiple runs) can be processed by CBPtools in full when sessions
are defined in the configuration file. Data for each session will be processed separately until the connectivity step,
after which the connectivity matrices for each subject will be averaged across all the sessions. When multi-session
data is being used, the (optional) PCA transformation will be performed after the sessions have been averaged, whereas
the other transformations (Fisher’s Z transform and cubic transform) will be applied before averaging. The averaged
connectivity matrices are then used for the remainder of the procedure.

Multi-session data can be defined in the configuration file as follows:

.. code-block:: yaml

    modality: rsfmri

    data:
        session: [sess1, sess2, sess3]
        time_series: /path/to/data_set/{participant_id}/{session}/time_series.nii.gz
        confounds:
            columns: ['constant', 'wm.linear', 'csf.linear', 'motion-*']
            delimiter: \t
            file: /path/to/data_set/{participant_id}/{session}/confounds.tsv
        masks:
            seed: /path/to/seed_mask.nii.gz
            target: /path/to/target_mask.nii.gz
            space: standard

    ...

Note how the `{session}` wildcard is now used to reference the data set files. Assuming two subjects with
participant IDs 'sub-001' and 'sub-002', and the sessions defined as in the code-block above, the `time_series` files
will become:

* `/path/to/data_set/sub-001/sess1/time_series.nii.gz`
* `/path/to/data_set/sub-001/sess2/time_series.nii.gz`
* `/path/to/data_set/sub-001/sess3/time_series.nii.gz`
* `/path/to/data_set/sub-002/sess1/time_series.nii.gz`
* `/path/to/data_set/sub-002/sess2/time_series.nii.gz`
* `/path/to/data_set/sub-002/sess3/time_series.nii.gz`

All wildcards can be used multiple times and don't have to indicate a folder. As long as the structure of the data set
is consistent, there should be no problem. For example, the time-series can also be referenced as:

.. code-block:: yaml

    time_series: /path/to/data_set/xyz-{participant_id}a/session_{session}/sw{participant_id}_{session}.nii.gz

For participant 'sub-001' and session `sess1` this will become:
`/path/to/data_set/xyz-sub-001a/session_sess1/swsub-001_sess1.nii.gz`

.. note::
   This example is by no means an endorsement for storing your data in such a way.