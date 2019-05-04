from cbptools import tasks, taskutils
import os

participants = taskutils.get_participant_ids('participants.tsv', sep='\t', index_col='participant_id')
<cbptools['parameters:clustering:n_clusters']>


rule all:
    input:
        taskutils.expected_output(
            <cbptools['parameters:clustering:n_clusters']>,
            <cbptools['parameters:summary:figure_format']>,
            <cbptools['parameters:clustering:internal_validity_metrics']>
        )

