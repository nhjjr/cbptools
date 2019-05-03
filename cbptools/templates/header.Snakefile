from cbptools import tasks, taskutils
import os

participants = taskutils.get_participant_ids('participants.tsv', sep='\t', index_col='participant_id')
k = <cbptools['parameters:clustering:n_clusters']>


rule all:
    input:
        taskutils.expected_output(
            k=k, ext=<cbptools['parameters:summary:figure_format']>,
            metrics=<cbptools['parameters:clustering:internal_validity_metrics']>
        )

