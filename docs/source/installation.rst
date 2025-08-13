Installation
============

Requirements
------------

Cheese3D is a Python package with a few external dependencies. Our environment is managed by `Pixi <https://pixi.sh/latest/>`__. Install Pixi using (on macOS or Windows):

.. code-block:: bash

    curl -fsSL https://pixi.sh/install.sh | sh

Setup
-----

Clone the Cheese3D repository:

.. code-block:: bash

    git clone https://github.com/HouLabCSHL/cheese3d.git

Setup the environment then activate it. Any future commands require activating the environment first.

.. code-block:: bash

    cd cheese3d
    pixi shell

Platform-specific support
-------------------------

.. list-table:: Support matrix
    :header-rows: 1

    * - Platform
      - Basic support
      - GPU acceleration
    * - Linux
      - |:white_check_mark:|
      - |:white_check_mark:|
    * - macOS (Apple Silicon)
      - |:white_check_mark:|
      - |:white_check_mark:|
    * - macOS (Intel)
      - |:white_check_mark:|
      - |:x:|
    * - Windows
      - |:construction:|
      - |:x:|
