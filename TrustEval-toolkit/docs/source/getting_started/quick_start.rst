Getting Started
===============

Quick Start
~~~~~~~~~~~

*The following example demonstrates an **Advanced AI Risk Evaluation** workflow.*

Step 0: Set Your Project Base Directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    base_dir = os.getcwd() + '/advanced_ai_risk'

Step 1: Download Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from trusteval import download_metadata

    download_metadata(
        section='advanced_ai_risk',
        output_path=base_dir
    )

Step 2: Generate Datasets Dynamically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from trusteval.dimension.ai_risk import dynamic_dataset_generator

    dynamic_dataset_generator(
        base_dir=base_dir,
    )

Step 3: Apply Contextual Variations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from trusteval import contextual_variator_cli

    contextual_variator_cli(
        dataset_folder=base_dir
    )

Step 4: Generate Model Responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from trusteval import generate_responses

    request_type = ['llm']  # Options: 'llm', 'vlm', 't2i'
    async_list = ['your_async_model']
    sync_list = ['your_sync_model']

    await generate_responses(
        data_folder=base_dir,
        request_type=request_type,
        async_list=async_list,
        sync_list=sync_list,
    )

Step 5: Evaluate and Generate Reports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Judge the Responses
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from trusteval import judge_responses

    target_models = ['your_target_model1', 'your_target_model2']
    judge_type = 'llm'  # Options: 'llm', 'vlm', 't2i'
    judge_key = 'your_judge_key'
    async_judge_model = ['your_async_model']

    await judge_responses(
        data_folder=base_dir,
        async_judge_model=async_judge_model,
        target_models=target_models,
        judge_type=judge_type,
    )

2. Generate Evaluation Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from trusteval import lm_metric

    lm_metric(
        base_dir=base_dir,
        aspect='ai_risk',
        model_list=target_models,
    )

3. Generate Final Report
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from trusteval import report_generator

    report_generator(
        base_dir=base_dir,
        aspect='ai_risk',
        model_list=target_models,
    )

Your ``report.html`` will be saved in the ``base_dir`` folder. For additional examples, check the ``examples`` folder.