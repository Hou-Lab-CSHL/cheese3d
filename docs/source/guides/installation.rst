Installation
============

Requirements
------------

Cheese3D is a Python package with a few external dependencies. Our environment is managed by `Pixi <https://pixi.sh/latest/>`__. Install Pixi using (on macOS or Windows):

.. code-block:: bash

    curl -fsSL https://pixi.sh/install.sh | sh

Setup
-----

.. warning::

    We are currently upstreaming changes in order to publish Cheese3D on PyPi. Until then, follow the instructions below to install Cheese3D directly from Github.

Clone the Cheese3D repository:

.. code-block:: bash

    git clone https://github.com/Hou-Lab-CSHL/cheese3d.git

Setup the environment then activate it. Any future commands require activating the environment first.

.. code-block:: bash

    cd cheese3d
    pixi shell

Installation should take a few minutes for each step to complete. If any step is taking too long, please [open an issue](https://github.com/Hou-Lab-CSHL/cheese3d/issues).

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
