Configuration
=============

⚙️ Installation
---------------

To install the **TrustEval-toolkit**, follow these steps:

1. Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/nauyisu022/TrustEval-toolkit.git
    cd TrustEval-toolkit

2. Set Up a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and activate a new environment with Python 3.10:

.. code-block:: bash

    conda create -n trusteval_env python=3.10
    conda activate trusteval_env

3. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Install the package and its dependencies:

.. code-block:: bash

    pip install .

🤖 Configure API Keys
---------------------

Run the configuration script to set up your API keys:

.. code-block:: bash

    python trusteval/src/configuration.py

.. image:: ../images/api_config.png
   :alt: API Configuration
