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
        .green { color: green; font-weight: bold }
    </style>

.. role:: title
.. role:: desc
.. role:: schema
.. role:: modality
.. role:: green

.. _ConfigurationParameters:

==========
Parameters
==========
The parameters are listed under the :green:`parameters` key in the configuration file and passed to workflow tasks when
needed. Importantly, not all parameters are used for each `modality`. In fact, some parameters may only be used if
other parameters are defined in a certain way. When a parameter is set that is not used, *CBPtools* will warn about
this in the log file, however.

Below is an example of what parameters look like in the configuration file:

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

In this snippet the clustering parameters are defined for the k-means clustering algorithm (method). The
:green:`parameters:clustering:n_clusters` field is required regardless of modality and clustering algorithm, although
the :green:`parameters:clustering:cluster_options` differ based on the selected method.

When a parameter is not **required** and not defined in the configuration file, the default value is used. This
will cause the setup procedure to issue a warning in the log file. If no default value can be found, an error will be
logged that must first be resolved before resuming the setup.

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
