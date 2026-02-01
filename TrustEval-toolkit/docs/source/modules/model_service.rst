ModelService
============

The `ModelService` class is the central utility for managing various models, including **Large Language Models (LLMs)**, **Vision-Language Models (VLMs)**, and **Text-to-Image (T2I)** models. It abstracts the complexity of model initialization, request handling, and response generation, providing a unified interface for both text and image-based tasks.

Quick Start
-----------

The `ModelService` is included in the `trustgen` package. To get started:

.. code-block:: python

    from trusteval import ModelService

    # Initialize ModelService
    model_service = ModelService(
        request_type="llm",
        handler_type="api",
        model_name="gpt-4o",
        config_path="/path/to/config.yaml"
    )

    # Process a single prompt
    response = model_service.process("Your prompt here.")

Initialization
--------------

To create an instance of the `ModelService` class, you need to specify the following parameters:

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Parameter
     - Type
     - Description
   * - `request_type`
     - `str`
     - The type of request: "llm" (Language Model), "vlm" (Vision Model), or "t2i" (Text-to-Image).
   * - `handler_type`
     - `str`
     - The handler type: "api" (remote API-based inference) or "local" (local inference).
   * - `model_name`
     - `str`
     - The name of the model to use, mapped internally.
   * - `config_path`
     - `str`
     - Path to the YAML configuration file.
   * - `save_folder`
     - `str`
     - (Optional) Folder to save the generated images (for `t2i` requests).
   * - `file_name`
     - `str`
     - (Optional) Name of the file to save the generated image as (for `t2i` requests).
   * - `image_urls`
     - `List[str]`
     - (Optional) List of image URLs or local image paths (for `vlm` requests).
   * - `**kwargs`
     - `dict`
     - Additional parameters to customize behavior (e.g., temperature).

Supported Models
----------------

The `ModelService` supports a wide range of models for text, vision, and text-to-image tasks. Below is a comprehensive table of the supported models, categorized by request type (`llm`, `vlm`, `t2i`), model name, and whether the model uses `api` or `local` inference.

.. list-table::
   :header-rows: 1
   :widths: 20 50 20

   * - Request Type
     - Model Name
     - Handler Type
   * - **LLM**
     - `gpt-4o`
     - `api`
   * - **LLM**
     - `gpt-4o-mini`
     - `api`
   * - **LLM**
     - `gpt-3.5-turbo`
     - `api`
   * - **LLM**
     - `text-embedding-ada-002`
     - `api`
   * - **LLM**
     - `glm-4`
     - `api`
   * - **LLM**
     - `glm-4-plus`
     - `api`
   * - **LLM**
     - `llama-3-8B`
     - `api/local`
   * - **LLM**
     - `llama-3.1-70B`
     - `api/local`
   * - **LLM**
     - `llama-3.1-8B`
     - `api/local`
   * - **LLM**
     - `qwen-2.5-72B`
     - `api/local`
   * - **LLM**
     - `mistral-7B`
     - `api/local`
   * - **LLM**
     - `mistral-8x7B`
     - `api/local`
   * - **LLM**
     - `claude-3.5-sonnet`
     - `api`
   * - **LLM**
     - `claude-3-haiku`
     - `api`
   * - **LLM**
     - `gemini-1.5-pro`
     - `api`
   * - **LLM**
     - `gemini-1.5-flash`
     - `api`
   * - **LLM**
     - `command-r-plus`
     - `api`
   * - **LLM**
     - `command-r`
     - `api`
   * - **LLM**
     - `gemma-2-27B`
     - `api/local`
   * - **LLM**
     - `deepseek-chat`
     - `api/local`
   * - **LLM**
     - `yi-lightning`
     - `api`
   * - **VLM**
     - `glm-4v`
     - `api`
   * - **VLM**
     - `glm-4v-plus`
     - `api`
   * - **VLM**
     - `llama-3.2-90B-V`
     - `api/local`
   * - **VLM**
     - `llama-3.2-11B-V`
     - `api/local`
   * - **VLM**
     - `qwen-vl-max-0809`
     - `api`
   * - **VLM**
     - `qwen-2-vl-72B`
     - `api/local`
   * - **VLM**
     - `internLM-72B`
     - `api/local`
   * - **VLM**
     - `claude-3-haiku`
     - `api`
   * - **VLM**
     - `gemini-1.5-pro`
     - `api`
   * - **VLM**
     - `gemini-1.5-flash`
     - `api`
   * - **T2I**
     - `dall-e-3`
     - `api`
   * - **T2I**
     - `flux-1.1-pro`
     - `api`
   * - **T2I**
     - `flux_schnell`
     - `api`
   * - **T2I**
     - `cogview-3-plus`
     - `api`
   * - **T2I**
     - `sd-3.5-large`
     - `local`
   * - **T2I**
     - `sd-3.5-large-turbo`
     - `local`
   * - **T2I**
     - `HunyuanDiT`
     - `local`
   * - **T2I**
     - `kolors`
     - `local`
   * - **T2I**
     - `playground-v2.5`
     - `local`

Pipeline Initialization
-----------------------

The `_initialize_pipeline` method sets up the appropriate pipeline based on the provided parameters. It automatically configures the model, handler, and other runtime options.

**Example: Initialize a GPT-4o Pipeline for API Use**

.. code-block:: python

    model_service = ModelService(
        request_type="llm",
        handler_type="api",
        model_name="gpt-4o",
        config_path="/path/to/config.yaml"
    )

Methods
-------

process
~~~~~~~

**Definition:**  
``process(prompt: Union[str, List[str]], **kwargs) -> str``  

Processes a single prompt or a list of prompts synchronously. It supports both one-off interactions and multi-turn conversations.

**Parameters**:

- **prompt** (:type:`str` or :type:`List[str]`): The input prompt(s).
- **kwargs**: Additional parameters for model customization.

**Returns**:  
Model-generated responses as a string.

**Example**:

.. code-block:: python

    # Single prompt
    response = model_service.process("Your prompt here.")

    # Multi-turn interaction
    prompts = [
        "What is the capital of France?",
        "What is the population of Paris?"
    ]
    responses = model_service.process(prompts)

process_async
~~~~~~~~~~~~~

**Definition:**  
``process_async(prompt: Union[str, List[str]], **kwargs) -> str``  

Handles requests asynchronously, enabling high concurrency for demanding applications.

**Parameters**:

- **prompt** (:type:`str` or :type:`List[str]`): The input prompt(s).
- **kwargs**: Additional parameters for model customization.

**Returns**:  
Model-generated responses as a string.

**Example**:

.. code-block:: python

    # Asynchronous prompt
    response = await model_service.process_async("Your prompt here.")
