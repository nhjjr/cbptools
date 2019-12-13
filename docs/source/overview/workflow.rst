.. _workflow:

========
Workflow
========

.. image:: ../_static/workflow.png
   :align: center

The CBPtools workflow for applying the :term:`rCBP` procedure to :term:`dMRI` or :term:`rsfMRI` data. After
customizing the parameters of the procedure, input data (A) is processed through each step (B through H) of the
workflow, culminating in the output (I).

The different types of input (dMRI Input, rsfMRI Input, and Connectivity Input) are not processed in parallel. Each
carries the same section key (A) to highlight the different types of input data CBPtools supports, and at what stage
of the processing the input data is used. For more information on what kind of input data is expected, read the
:ref:`validationInputData` section, whereas if you want to know how to use your data set with CBPtools, read
:ref:`ConfigurationInputData`.

Each step is addressed in more detail in the processing section. Specifically:
(A) :ref:`Input Data <validationInputData>`, (B) :ref:`Masks <TaskMasking>`,
(C) :ref:`rsfMRI Connectivity <TaskConnectivityrsfMRI>` or :ref:`dMRI Connectivity <TaskConnectivitydMRI>`,
(D) :ref:`Clustering <TaskClustering>`, (E) :ref:`Validity <TaskValidity>`, (F) :ref:`Grouping <TaskGrouping>`,
(G) :ref:`Similarity <TaskSimilarity>`, (H) :ref:`Reports <TaskReport>`
