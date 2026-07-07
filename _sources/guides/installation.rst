Installation
============

Requirements
------------

Cheese3D is a Python package with a few external dependencies. Our environment is managed by `Pixi <https://pixi.sh/latest/>`__. Install Pixi using (on Linux or macOS):

.. code-block:: bash

    curl -fsSL https://pixi.sh/install.sh | sh

Pixi is like Anaconda (i.e., ``conda``) but faster and more robust.

Setup
-----

Cheese3D is published on PyPi. We ship a minimal Pixi environment (``pixi.toml`` + ``pixi.lock``) that pins ``cheese3d``, ``cheese3d-annotator``, and the native dependencies (ffmpeg, CUDA, etc.) they need. You don't need to clone the repository, just download the two manifest files.

Create a directory for the environment and fetch the manifest files:

.. code-block:: bash

    mkdir cheese3d && cd cheese3d
    curl -fsSLO https://raw.githubusercontent.com/Hou-Lab-CSHL/cheese3d/main/public-env/pixi.toml
    curl -fsSLO https://raw.githubusercontent.com/Hou-Lab-CSHL/cheese3d/main/public-env/pixi.lock

Now, you should have a folder called ``cheese3d`` with two files inside it: ``pixi.toml`` and ``pixi.lock``.

Activate the environment. Pixi will install everything on first use. Any future commands require activating the environment first.

.. code-block:: bash

    pixi shell

Installation should take a few minutes for each step to complete. If any step is taking too long, please open an issue (https://github.com/Hou-Lab-CSHL/cheese3d/issues).

.. note::

    These instructions are for **using** Cheese3D. If you want to edit the source code for Cheese3D, see the `CONTRIBUTING guide <https://github.com/Hou-Lab-CSHL/cheese3d/blob/main/CONTRIBUTING.md>`__ for details.

Manual environment
^^^^^^^^^^^^^^^^^^

Pixi installs everything into a ``.pixi/`` directory inside the ``cheese3d`` folder you just created. If you'd rather have a global environment or manage dependencies yourself (e.g., with ``conda``, ``mamba``, ``venv``, or system package managers), here is what Cheese3D needs:

- **Python** 3.10
- **ffmpeg** >= 5.1.2
- **NumPy** < 2.0
- **CUDA** 11.7 and **cuDNN** 8 (Linux only, for GPU acceleration). Make sure these libraries are discoverable on ``LD_LIBRARY_PATH``; Pixi handles this automatically.
- **Cheese3D** itself, from PyPi:

  .. code-block:: bash

      pip install cheese3d cheese3d-annotator

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
