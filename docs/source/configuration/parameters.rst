.. |br| raw:: html

    <br/>

.. raw:: html

    <style>
        .section #parameter-fields li { list-style: none; }
        .section #parameter-fields blockquote { margin: 0px; padding: 0px; }
        .section #parameter-fields p { margin: 0px; padding: 0px; }
        .title { font-weight: bold; color: green; }
        .desc { padding: 5px 0px 15px 25px; display: block; }
        .schema { color: black; font-weight: bold }
        .modality { color: red; font-weight: bold }
    </style>

.. role:: title
.. role:: desc
.. role:: schema
.. role:: modality

.. _ConfigurationParameters:

==========
Parameters
==========
Not all parameters are used for each `modality`. The parameter descriptions are organized by modality. The parameters
are listed under the `parameters` key in the configuration file. Each member of parameters is a task that is
executed separately. For example:

.. code-block:: yaml

    parameters:
        clustering:
            method: kmeans
            n_clusters: [2, 3, 4, 5]
            cluster_options:
                algorithm: auto
                init: k-means++
                max_iter: 1000
                n_init: 256
                ...

In this snippet, the mask_preproc_seed and time_series_proc fields are tasks, with the parameters and their values
being threshold, median_filter, and low_variance_error, respectively.

When a parameter is not **required** and not defined in the configuration file, the default value is used. This
will cause the setup procedure to issue a warning in the log file.

Parameter fields
================

.. jinja:: schema

    {% for task, param in parameters.items() %}

    * :title:`{{task}}:`

    {% for k, v in param.items() %}
    {% if v.type is not defined %}

       * :title:`{{k}}:`

       {% for k1, v1 in v.items() %}
       {% if v1.type is not defined %}

           * :title:`{{k1}}:`

           {% for k2, v2 in v1.items() %}
           {% if v2.type is not defined %}

               * :title:`{{k2}}:`

               {% for k3, v3 in v2.items() %}

                   * :title:`{{k3}}:` :schema:`{{v3.type}}, default = {{v3.default}}{% if v3.required %}, required{% endif %}{% if v3.allowed %}, allowed = {{v3.allowed}}{% endif %}, modality =`
                     :modality:`{% if v3.dependency and v3.dependency[0].modality %}{{v3.dependency[0].modality}}{% else %}any{% endif %}`
                     :desc:`{% if v3.desc %}{{v3.desc}}{% else %}No description given{% endif %}`

               {% endfor %}
           {% else %}

               * :title:`{{k2}}:` :schema:`{{v2.type}}, default = {{v2.default}}{% if v2.required %}, required{% endif %}{% if v2.allowed %}, allowed = {{v2.allowed}}{% endif %}, modality =`
                 :modality:`{% if v2.dependency and v2.dependency[0].modality %}{{v2.dependency[0].modality}}{% else %}any{% endif %}`
                 :desc:`{% if v2.desc %}{{v2.desc}}{% else %}No description given{% endif %}`

           {% endif %}
           {% endfor %}
       {% else %}

           * :title:`{{k1}}:` :schema:`{{v1.type}}, default = {{v1.default}}{% if v1.required %}, required{% endif %}{% if v1.allowed %}, allowed = {{v1.allowed}}{% endif %}, modality =`
             :modality:`{% if v1.dependency and v1.dependency[0].modality %}{{v1.dependency[0].modality}}{% else %}any{% endif %}`
             :desc:`{% if v1.desc %}{{v1.desc}}{% else %}No description given{% endif %}`

       {% endif %}
       {% endfor %}
    {% else %}

       * :title:`{{k}}:` :schema:`{{v.type}}, default = {{v.default}}{% if v.required %}, required{% endif %}{% if v.allowed %}, allowed = {{v.allowed}}{% endif %}, modality =`
         :modality:`{% if v.dependency and v.dependency[0].modality %}{{v.dependency[0].modality}}{% else %}any{% endif %}`
         :desc:`{% if v.desc %}{{v.desc}}{% else %}No description given{% endif %}`

    {% endif %}
    {% endfor %}
    {% endfor %}

References
==========
References to external documentation:

   * `probtrackx2 <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#PROBTRACKX_-_probabilistic_tracking_with_crossing_fibres)>`_
   * `sklearn.cluster.KMeans <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)>`_
   * `scipy.cluster.hierarchy.linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)>`_
