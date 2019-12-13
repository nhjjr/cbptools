.. |br| raw:: html

    <br/>

.. _TaskReport:

======
Report
======
As described for the earlier tasks, the wildcards `{participant_id}` is a placeholder.

.. glossary::
    Configuration fields
        parameters.report.figure_format |br|
        parameters.clustering.validity.similarity |br|
        parameters.clustering.n_clusters

    Output
        `individual/internal_validity_{metric}.[png, svg, pdf, ps, eps]` |br|
        `group/{n_clusters}clusters/individual_similarity_heatmap.png` |br|
        `group/{n_clusters}clusters/individual_similarity_clustermap.png` |br|
        `group/group_similarity.[png, svg, pdf, ps, eps]` |br|
        `group/relabeling_accuracy.[png, svg, pdf, ps, eps]` |br|
        `group/cophenetic_correlation.[png, svg, pdf, ps, eps]` |br|
        `group/{n_clusters}clusters/voxel_plot_{view}.[png, svg, pdf, ps, eps]` |br|
        `individual/{participant_id}/{n_clusters}cluster_voxel_plot_{view}..[png, svg, pdf, ps, eps]` (optional) |br|
        `group/reference_similarity.[png, svg, pdf, ps, eps]`

    Benchmarking
        `benchmarks/{metric}.plot_internal_validity.log` |br|
        `benchmarks/k{n_clusters}.plot_individual_similarity.log` |br|
        `benchmarks/plot_group_similarity.log` |br|
        `benchmarks/k{n_clusters}.{view}.plot_labeled_roi.log` |br|
        `benchmarks/{participant_id}.k{n_clusters}.{view}.plot_individual_labeled_roi.log` (optional) |br|
        `benchmarks/plot_reference_similarity.log` (optional)

For all the various statistics generated during the CBPtools procedure, plots are generated for the internal validity,
individual similarity, group similarity, relabeling accuracy, cophenetic correlation, and reference similarity scores.
In addition, voxel plots are generated to visualize the cluster labels mapped onto the :term:`ROI`.

This is the last task to be executed and the procedure is now complete. Both the summary results as well as the
interim data can be viewed in the project folder.
