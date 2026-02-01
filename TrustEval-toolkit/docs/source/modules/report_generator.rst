Report Generator
================

The Report Generator module is designed to handle the generation of reports using various models. It includes the main component: `report_pipeline` for generating reports.

Quick Start
-----------

report_pipeline
~~~~~~~~~~~~~~~

The ``report_pipeline`` function processes all data files in a specified directory to generate reports using specified models.

**Definition:**

.. code-block:: python

    report_pipeline(
        base_dir: str,
        model_type: str,
        aspect: str
    ) -> None

:Parameters:

- **base_dir** (:type:`str`) Base directory for data and output
- **model_type** (:type:`str`) Type of model ('llm', 'vlm', etc.)
- **aspect** (:type:`str`) Evaluation aspect ('robustness', 'fairness', etc.)

**Examples:**

.. code-block:: python

    import trusteval
    trusteval.report_pipeline(
        base_dir='path/to/base_dir',
        model_type='llm', # or 'vlm', 't2i'
        aspect='robustness'
    )


Classes
-------

ReportGenerator
~~~~~~~~~~~~~~~

The `ReportGenerator` class processes data from different models to generate reports.

:Parameters:

- **base_dir** (:type:`str`) Base directory for data and output
- **model_type** (:type:`str`) Type of model ('llm', 'vlm', etc.)
- **aspect** (:type:`str`) Evaluation aspect ('robustness', 'fairness', etc.)

Functions
---------

generate_report
~~~~~~~~~~~~~~~

**Definition:**

.. code-block:: python

    generate_report(
        test_data: pd.DataFrame,
        leaderboard: pd.DataFrame,
        aspect: str,
        case_data: Dict[str, Any],
        base_dir: str
    ) -> None

:Parameters:

- **test_data** (:type:`pd.DataFrame`) DataFrame containing test data
- **leaderboard** (:type:`pd.DataFrame`) DataFrame containing leaderboard data
- **aspect** (:type:`str`) Evaluation aspect ('robustness', 'fairness', etc.)
- **case_data** (:type:`Dict[str, Any]`) Dictionary containing case data
- **base_dir** (:type:`str`) Base directory for data and output

prepare_chart_data
~~~~~~~~~~~~~~~~~~

**Definition:**

.. code-block:: python

    prepare_chart_data(
        leaderboard: pd.DataFrame,
        test_data: pd.DataFrame,
        aspect: str
    ) -> Dict[str, Any]

:Parameters:

- **leaderboard** (:type:`pd.DataFrame`) DataFrame containing leaderboard data
- **test_data** (:type:`pd.DataFrame`) DataFrame containing test data
- **aspect** (:type:`str`) Evaluation aspect ('robustness', 'fairness', etc.)

generate_summary_with_gpt
~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition:**

.. code-block:: python

    generate_summary_with_gpt(
        test_data: pd.DataFrame,
        leaderboard: pd.DataFrame
    ) -> str

:Parameters:

- **test_data** (:type:`pd.DataFrame`) DataFrame containing test data
- **leaderboard** (:type:`pd.DataFrame`) DataFrame containing leaderboard data
