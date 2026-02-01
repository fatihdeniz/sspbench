Judgement Processor
===================

The Judgement Processor module is designed to handle the evaluation of responses using various judgement models. It includes two main components: `judge_responses` for evaluating responses and `judge_images` for evaluating images.

Quick Start
-----------

judge_responses
~~~~~~~~~~~~~~~

The ``judge_responses`` function processes all data files in a specified directory to evaluate responses using specified models.

**Definition:**

.. code-block:: python

    judge_responses(
        data_folder: str,
        async_judge_model: List[str],
        target_models: List[str],
        judge_type: str,
        response_key: List[str] = ['responses'],
        judge_key: str = 'judge',
        response_extension: str = '_responses',
        judge_extension: str = '_judge',
        reverse_choice: bool = False
    ) -> None

:Parameters:

- **data_folder** (:type:`str`) Path to the folder containing JSON files to process
- **async_judge_model** (:type:`List[str]`) List of asynchronous judge models
- **target_models** (:type:`List[str]`) Combined list of target asynchronous and synchronous models
- **judge_type** (:type:`str`) Type of judge ('llm', 'vlm', 'toxicity', etc.)
- **response_key** (:type:`List[str]`, optional) List of keys to look for in the responses
- **judge_key** (:type:`str`, optional) Key to store judge results
- **response_extension** (:type:`str`, optional) Extension for response files
- **judge_extension** (:type:`str`, optional) Extension for judge result files
- **reverse_choice** (:type:`bool`, optional) Whether to reverse choices in mappings

**Examples:**

For LLM Usage:

.. code-block:: python

    import trusteval
    await trusteval.judge_responses(
        data_folder='path/to/data',
        async_judge_model=['model1', 'model2'],
        target_models=['model3'],
        judge_type='llm',
        response_key=['responses'],
        judge_key='judge'
    )

For VLM Usage:

.. code-block:: python

    import trusteval
    await trusteval.judge_responses(
        data_folder='path/to/data',
        async_judge_model=['model1', 'model2'],
        target_models=['model3'],
        judge_type='vlm',
        response_key=['responses'],
        judge_key='judge',
    )

judge_images
~~~~~~~~~~~~

The ``judge_images`` function processes all image data files in a specified directory to evaluate images using specified models.

**Definition:**

.. code-block:: python

    judge_images(
        base_dir: str,
        aspect: str,
        handler_type: str = 'api',
        target_models: List[str] = None
    ) -> None

:Parameters:

- **base_dir** (:type:`str`) Base directory for data and output
- **aspect** (:type:`str`) Evaluation aspect ('robustness', 'fairness', etc.)
- **handler_type** (:type:`str`, optional) Type of handler ('api' or 'local')
- **target_models** (:type:`List[str]`, optional) List of model names to evaluate

**Example Usage:**

.. code-block:: python

    import trusteval
    
    trusteval.judge_images(
        base_dir='path/to/base_dir', 
        aspect='robustness_t2i', 
        handler_type='api', 
        target_models=['model1', 'model2']
    )

Classes
-------

JudgeProcessor
~~~~~~~~~~~~~~

The `JudgeProcessor` class processes responses from different models, handling both asynchronous and synchronous services.

:Parameters:

- **data_folder** (:type:`str`) Path to the folder containing JSON files to process
- **async_judge_model** (:type:`List[str]`) List of asynchronous judge models
- **response_key** (:type:`List[str]`, optional) List of keys to look for in the responses
- **judge_key** (:type:`str`, optional) Key to store judge results
- **target_models** (:type:`List[str]`) Combined list of target asynchronous and synchronous models
- **response_extension** (:type:`str`, optional) Extension for response files
- **judge_extension** (:type:`str`, optional) Extension for judge result files
- **judge_type** (:type:`str`) Type of judge ('llm', 'vlm', 'toxicity', etc.)
- **reverse_choice** (:type:`bool`, optional) Whether to reverse choices in mappings

Functions
---------

get_response
~~~~~~~~~~~~

**Definition:**

.. code-block:: python

    get_response(
        task_config: Dict[str, Any],
        data_path: str,
        max_concurrent_tasks: int = 30
    ) -> None

:Parameters:

- **task_config** (:type:`Dict[str, Any]`) Configuration for the current task
- **data_path** (:type:`str`) Path to the data file
- **max_concurrent_tasks** (:type:`int`, optional) Maximum number of concurrent tasks

toxicity
~~~~~~~~

**Definition:**

.. code-block:: python

    toxicity(
        data_path: str,
        response_key: List[str]
    ) -> None

:Parameters:

- **data_path** (:type:`str`) Path to the data file
- **response_key** (:type:`List[str]`) Key(s) to extract responses from
