Response Generator
==================

The Response Generator module is designed to handle the generation of responses and images using various models. It includes two main components: `lm_response` for language model responses and `t2i_response` for text-to-image responses.

Quick Start
-----------

generate_responses
~~~~~~~~~~~~~~~~~~

The ``generate_responses`` function processes all data files in a specified directory to generate responses using specified models.

**Definition:**

.. code-block:: python

    generate_responses(
        data_folder: str,
        request_type: List[str],
        async_list: List[str],
        sync_list: Optional[List[str]] = [],
        prompt_key: List[str] = 'prompt_key',
        result_key: List[str] = 'result_key',
        file_name_extension: str = '_responses',
        image_key: Optional[List[Optional[str]]] = None,
        **kwargs
    ) -> None

:Parameters:

- **data_folder** (:type:`str`) Path to the folder containing data files
- **request_type** (:type:`List[str]`) List of request types ('llm' or 'vlm')
- **async_list** (:type:`List[str]`) List of asynchronous model names
- **sync_list** (:type:`List[str]`, optional) List of synchronous model names
- **prompt_key** (:type:`List[str]`, optional) Keys to extract prompts from data
- **result_key** (:type:`List[str]`, optional) Keys to store responses in data
- **file_name_extension** (:type:`str`, optional) Extension for output files
- **image_key** (:type:`List[Optional[str]]`, optional) Keys for image paths in VLM tasks
- **kwargs** Additional model parameters (e.g., temperature, max_new_tokens)

**Examples:**

For LLM Usage:

.. code-block:: python

    import trusteval
    await trusteval.generate_responses(
        data_folder='path/to/data',
        request_type=['llm'],
        async_list=['model1', 'model2'],
        sync_list=['model3'],
        prompt_key=['prompt_key'],
        result_key=['result_key'],
        temperature=0.7  # Optional: model parameter
    )

For VLM Usage:

.. code-block:: python

    import trusteval
    await trusteval.generate_responses(
        data_folder='path/to/data',
        request_type=['vlm'],
        async_list=['model1', 'model2'],
        prompt_key=['prompt_key'],
        result_key=['result_key'],
        image_key=['image_path'],
        max_new_tokens=100 # Optional: model parameter
    )

generate_images
~~~~~~~~~~~~~~~~

**Definition:**

.. code-block:: python

    generate_images(
        base_dir: str,
        aspect: str,
        local_models: Optional[List[str]] = None,
        api_models: Optional[List[str]] = None
    ) -> None

:Parameters:

- **base_dir** (:type:`str`) Base directory for data and output
- **aspect** (:type:`str`) Evaluation aspect ('robustness', 'fairness', etc.)
- **local_models** (:type:`List[str]`, optional) List of local model names
- **api_models** (:type:`List[str]`, optional) List of API model names

**Example Usage:**

.. code-block:: python

    import trusteval
    
    trusteval.generate_images(
        base_dir=base_dir = 'path/to/base_dir', 
        aspect='robustness_t2i', 
        local_models=['model1', 'model2'], 
        api_models=['model3']
    )

lm_response
-----------

The `lm_response` module processes responses from different language models, handling both asynchronous and synchronous services.

Classes
~~~~~~~

ResponseProcessor
^^^^^^^^^^^^^^^^^

The `ResponseProcessor` class processes responses from different models, handling both asynchronous and synchronous services.

:Parameters:

- **request_type** (:type:`str`) The type of request ('llm' or 'vlm')
- **async_list** (:type:`List[str]`, optional) List of asynchronous model names
- **sync_list** (:type:`List[str]`, optional) List of synchronous model names
- **save_path** (:type:`str`, optional) Path to save the processed responses

Functions
~~~~~~~~~

process_async_service
^^^^^^^^^^^^^^^^^^^^^

**Definition:**

.. code-block:: python

    process_async_service(
        service: ModelService,
        model_name: str,
        prompt: str,
        image_urls: Optional[List[str]],
        **kwargs
    ) -> Dict[str, Any]

:Parameters:

- **service** (:type:`ModelService`) The ModelService instance
- **model_name** (:type:`str`) Name of the model
- **prompt** (:type:`str`) The input prompt
- **image_urls** (:type:`List[str]`, optional) List of image URLs for VLM
- **kwargs** Additional model parameters

process_sync_service
^^^^^^^^^^^^^^^^^^^^

**Definition:**

.. code-block:: python

    process_sync_service(
        service: ModelService,
        model_name: str,
        prompt: str,
        image_urls: Optional[List[str]]
    ) -> Dict[str, Any]

:Parameters:

- **service** (:type:`ModelService`) The ModelService instance
- **model_name** (:type:`str`) Name of the model
- **prompt** (:type:`str`) The input prompt
- **image_urls** (:type:`List[str]`, optional) List of image URLs for VLM

t2i_response
------------

The `t2i_response` module generates images based on text prompts using various models.

Functions
~~~~~~~~~

generate_images_local
^^^^^^^^^^^^^^^^^^^^^

**Definition:**

.. code-block:: python

    generate_images_local(
        prompts: List[str],
        output_paths: List[str],
        model_name: Optional[str] = None
    ) -> None

:Parameters:

- **prompts** (:type:`List[str]`) List of prompt strings
- **output_paths** (:type:`List[str]`) List of output file paths
- **model_name** (:type:`str`, optional) Name of the model

generate_images_api
^^^^^^^^^^^^^^^^^^^

**Definition:**

.. code-block:: python

    generate_images_api(
        prompts: List[str],
        output_paths: List[str],
        model_name: Optional[str] = None
    ) -> None

:Parameters:

- **prompts** (:type:`List[str]`) List of prompt strings
- **output_paths** (:type:`List[str]`) List of output file paths
- **model_name** (:type:`str`, optional) Name of the model

process_data
^^^^^^^^^^^^

**Definition:**

.. code-block:: python

    process_data(
        data_path: Optional[str] = None,
        model_name: Optional[str] = None,
        base_dir: Optional[str] = None,
        process_type: str = 'local',
        aspect: Optional[str] = None
    ) -> None

:Parameters:

- **data_path** (:type:`str`, optional) Path to the JSON data file
- **model_name** (:type:`str`, optional) Name of the model
- **base_dir** (:type:`str`, optional) Base directory for output
- **process_type** (:type:`str`) Type of processing ('local' or 'api')
- **aspect** (:type:`str`, optional) Evaluation aspect
