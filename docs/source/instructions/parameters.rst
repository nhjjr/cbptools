.. raw:: html

    <style> .green {color:green}  .black {color:black}  .red {color:red}</style>

.. role:: green
.. role:: black
.. role:: red

==========
Parameters
==========
Not all parameters are used for each `modality`. The parameter descriptions are organized by modality. The parameters
are listed under the `parameters` key in the configuration file. Each member of parameters is a task that is
executed separately. For example:

.. code-block:: yaml

    parameters:
        mask_preproc_seed:
            threshold: 0.0
            median_filter: false
        time_series_proc:
            low_variance_error:
                apply: true
                ...

In this snippet, the mask_preproc_seed and time_series_proc fields are tasks, with the parameters and their values
being threshold, median_filter, and low_variance_error, respectively.

When a parameter is not :green:`required` and not defined in the configuration file, the default value is used. This
will cause the setup procedure to issue a warning in the log file.

*******************************
Modality-independent Parameters
*******************************
The parameters listed below can be used regardless of the selected modality, hence they are always applicable.

.. jinja:: schema-general

    {% for task, param in parameters.items() %}

    {{task}}
    ----------------------------
    .. glossary::
       {% for field, val in param.items() %}

       {% if val.type %}

       :black:`{{field}}` *type = {{val.type}}, default = {{val.default}}*{% if val.required %}, :green:`required`{% endif %}
          {% if val.desc %}{{val.desc}}{% else %}No description given{% endif %}

          {% if val.allowed %}**Allowed values**: {{val.allowed}}{% endif %}

       {% else %}

       :black:`{{field}}`

       {% for sfield, sval in val.items() %}

       .. glossary::

           :black:`{{sfield}}` *type = {{sval.type}}, default = {{sval.default}}*{% if sval.required %}, :green:`required`{% endif %}
              {% if sval.desc %}{{sval.desc}}{% else %}No description given{% endif %}

              {% if sval.allowed %}**Allowed values**: {{sval.allowed}}{% endif %}

       {% endfor %}

       {% endif %}

       {% endfor %}
    {% endfor %}

*****************************
Modality-dependent Parameters
*****************************
The following parameters are used only for the respective modality. Note that the mask preprocessing (mask_preproc_seed
and mask_preproc_target) is executed during the setup step (i.e., prior to execution of the workflow). It transforms the
masks based on the parameters outlined below. The available parameters differ per modality, and this step is skipped
entirely when connectivity matrices are given as input directly (i.e., `modality = connectivity`).

Resting-State fMRI Parameters
=============================

.. jinja:: schema-rsfmri

    {% for task, param in parameters.items() %}

    {{task}}
    ----------------------------
    .. glossary::
       {% for field, val in param.items() %}

       {% if val.type %}

       :black:`{{field}}` *type = {{val.type}}, default = {{val.default}}*{% if val.required %}, :green:`required`{% endif %}
          {% if val.desc %}{{val.desc}}{% else %}No description given{% endif %}

          {% if val.allowed %}**Allowed values**: {{val.allowed}}{% endif %}

       {% else %}

       :black:`{{field}}`

       {% for sfield, sval in val.items() %}

       .. glossary::

           :black:`{{sfield}}` *type = {{sval.type}}, default = {{sval.default}}*{% if sval.required %}, :green:`required`{% endif %}
              {% if sval.desc %}{{sval.desc}}{% else %}No description given{% endif %}

              {% if sval.allowed %}**Allowed values**: {{sval.allowed}}{% endif %}

       {% endfor %}

       {% endif %}

       {% endfor %}
    {% endfor %}

Diffusion MRI Parameters
========================

.. jinja:: schema-dmri

    {% for task, param in parameters.items() %}

    {{task}}
    ----------------------------
    .. glossary::
       {% for field, val in param.items() %}

       {% if val.type %}

       :black:`{{field}}` *type = {{val.type}}, default = {{val.default}}*{% if val.required %}, :green:`required`{% endif %}
          {% if val.desc %}{{val.desc}}{% else %}No description given{% endif %}

          {% if val.allowed %}**Allowed values**: {{val.allowed}}{% endif %}

       {% else %}

       :black:`{{field}}`

       {% for sfield, sval in val.items() %}

       .. glossary::

           :black:`{{sfield}}` *type = {{sval.type}}, default = {{sval.default}}*{% if sval.required %}, :green:`required`{% endif %}
              {% if sval.desc %}{{sval.desc}}{% else %}No description given{% endif %}

              {% if sval.allowed %}**Allowed values**: {{sval.allowed}}{% endif %}

       {% endfor %}

       {% endif %}

       {% endfor %}
    {% endfor %}

Connectivity Parameters
=======================

.. jinja:: schema-connectivity

    {% if parameters %}

    {% for task, param in parameters.items() %}

    {{task}}
    ----------------------------
    .. glossary::
       {% for field, val in param.items() %}

       {% if val.type %}

       :black:`{{field}}` *type = {{val.type}}, default = {{val.default}}*{% if val.required %}, :green:`required`{% endif %}
          {% if val.desc %}{{val.desc}}{% else %}No description given{% endif %}

          {% if val.allowed %}**Allowed values**: {{val.allowed}}{% endif %}

       {% else %}

       :black:`{{field}}`

       {% for sfield, sval in val.items() %}

       .. glossary::

           :black:`{{sfield}}` *type = {{sval.type}}, default = {{sval.default}}*{% if sval.required %}, :green:`required`{% endif %}
              {% if sval.desc %}{{sval.desc}}{% else %}No description given{% endif %}

              {% if sval.allowed %}**Allowed values**: {{sval.allowed}}{% endif %}

       {% endfor %}

       {% endif %}

       {% endfor %}
    {% endfor %}

    {% else %}

    There are no parameters specific to this modality.

    {% endif %}

**********
References
**********
References to external documentation:

   * `probtrackx2 <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#PROBTRACKX_-_probabilistic_tracking_with_crossing_fibres)>`_
   * `sklearn.cluster.KMeans <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)>`_
   * `scipy.cluster.hierarchy.linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)>`_
