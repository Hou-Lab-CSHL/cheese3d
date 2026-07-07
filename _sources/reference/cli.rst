Command-line interface
======================

Cheese3D provides a command-line interface (CLI). All commands are accessible via the ``cheese3d`` entry point.

You can open the help by running

.. code-block:: shell

   $ cheese3d --help
   Usage: cheese3d [OPTIONS] COMMAND [ARGS]...

   ╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────╮
   │ --path                      TEXT  Path to project directory                                         │
   │                                   [default: /data/disk1/daruwal/repos/cheese3d]                     │
   │ --configs                   TEXT  Path to additional configs (relative to project)                  │
   │                                   [default: configs]                                                │
   │ --install-completion              Install completion for the current shell.                         │
   │ --show-completion                 Show completion for the current shell, to copy it or customize    │
   │                                   the installation.                                                 │
   │ --help                            Show this message and exit.                                       │
   ╰─────────────────────────────────────────────────────────────────────────────────────────────────────╯
   ╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────╮
   │ setup            Setup a new Cheese3D project called NAME under --path.                             │
   │ import           Import an existing pose model project into NAME.                                   │
   │ summarize        Summarize a Cheese3D project based on its configuration file.                      │
   │ sync             Synchronize the video (and possibly ephys) files in a Cheese3D project.            │
   │ extract          Extract frames from video data.                                                    │
   │ label            Label extracted frames for model training.                                         │
   │ train            Train 2d pose model.                                                               │
   │ calibrate        Extract camera calibration for triangulation.                                      │
   │ track            Track 2D keypoints using trained pose estimation model.                            │
   │ triangulate      Triangulate 2D pose estimation result to obtain 3D keypoints and Cheese3D          │
   │                  features.                                                                          │
   │ analyze          Run full analysis including: Camera calibration, 2D keypoint tracking,             │
   │                  3D triangulation, and feature extraction.                                          │
   │ generate-videos  Generate videos with pose estimation results overlaid.                             │
   │ visualize        Visualize the output of Cheese3D using interactive GUI.                            │
   │ interactive      Run interactive GUI.                                                               │
   │ checkpoint       Create a checkpoint of this project.                                               │
   │ restore          Restore a project from a checkpoint archive.                                       │
   ╰─────────────────────────────────────────────────────────────────────────────────────────────────────╯

Global options and arguments
----------------------------

These options are available on **every** command and are set before the subcommand.

.. list-table:: Global options
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``--path``
      - ``str``
      - Current working directory
      - Path to project directory (where ``cheese3d`` looks for your project).
    * - ``--configs``
      - ``str``
      - ``configs``
      - Path to additional configs (relative to project).

Several commands share the same positional arguments. They are documented here once for brevity.

.. list-table:: Common arguments
    :header-rows: 1

    * - Argument
      - Type
      - Default
      - Description
    * - ``NAME``
      - ``str``
      - ``"."``
      - Name of the project (or ``"."`` for the current directory).
    * - ``CONFIG_OVERRIDES``
      - ``Optional[List[str]]``
      - ``None``
      - Config overrides passed to `Hydra <https://hydra.cc/docs/intro/>`_.

Commands
--------

setup
^^^^^

.. code-block:: bash

   cheese3d setup NAME

Setup a new Cheese3D project called **NAME** under ``--path``.

.. list-table:: ``setup`` arguments
    :header-rows: 1

    * - Argument
      - Type
      - Description
    * - ``NAME``
      - ``str``
      - Name of the new project.

import
^^^^^^

.. code-block:: bash

   cheese3d import MODEL [NAME] [--model-type TYPE]

Import an existing pose model project into **NAME**.

.. list-table:: ``import`` arguments
    :header-rows: 1

    * - Argument
      - Type
      - Default
      - Description
    * - ``MODEL``
      - ``str``
      - N/A
      - Path to existing model.
    * - ``NAME``
      - ``str``
      - ``"."``
      - Name of the project.
    * - ``CONFIG_OVERRIDES``
      - ``Optional[List[str]]``
      - ``None``
      - Config overrides passed to Hydra.

.. list-table:: ``import`` options
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``--model-type``
      - ``str``
      - ``"dlc"``
      - Type of project to import (only ``"dlc"`` is valid for now).

summarize
^^^^^^^^^

.. code-block:: bash

   cheese3d summarize [NAME]

Summarize a Cheese3D project based on its configuration file.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

sync
^^^^

.. code-block:: bash

   cheese3d sync [NAME]

Synchronize the video (and possibly ephys) files in a Cheese3D project.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

extract
^^^^^^^

.. code-block:: bash

   cheese3d extract [NAME] [--manual]

Extract frames from video data.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

.. list-table:: ``extract`` options
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``--manual``
      - ``bool``
      - ``False``
      - Set to launch manual frame picking GUI. When enabled, an interactive prompt lets you select which recording to extract frames for.

label
^^^^^

.. code-block:: bash

   cheese3d label [NAME]

Label extracted frames for model training.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

train
^^^^^

.. code-block:: bash

   cheese3d train [NAME] [--gpu ID] [--iterate-dataset / --no-iterate-dataset]

Train 2D pose model.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

.. list-table:: ``train`` options
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``--gpu``
      - ``int``
      - ``0``
      - GPU ID(s) to use.
    * - ``--iterate-dataset`` / ``--no-iterate-dataset``
      - ``bool``
      - ``True``
      - Create a new dataset iteration before training.

calibrate
^^^^^^^^^

.. code-block:: bash

   cheese3d calibrate [NAME] [--session SESSION]

Extract camera calibration for triangulation.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

.. list-table:: ``calibrate`` options
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``--session``
      - ``Optional[str]``
      - ``None``
      - Name of a specific session subfolder to process.

track
^^^^^

.. code-block:: bash

   cheese3d track [NAME] [--session SESSION]

Track 2D keypoints using trained pose estimation model.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

.. list-table:: ``track`` options
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``--session``
      - ``Optional[str]``
      - ``None``
      - Name of a specific session subfolder to process.

triangulate
^^^^^^^^^^^

.. code-block:: bash

   cheese3d triangulate [NAME] [--session SESSION]

Triangulate 2D pose estimation results to obtain 3D keypoints and Cheese3D features.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

.. list-table:: ``triangulate`` options
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``--session``
      - ``Optional[str]``
      - ``None``
      - Name of a specific session subfolder to process.

analyze
^^^^^^^

.. code-block:: bash

   cheese3d analyze [NAME] [--session SESSION]

Run full analysis including: camera calibration, 2D keypoint tracking, 3D triangulation, and feature extraction.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

.. list-table:: ``analyze`` options
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``--session``
      - ``Optional[str]``
      - ``None``
      - Name of a specific session subfolder to process.

generate-videos
^^^^^^^^^^^^^^^

.. code-block:: bash

   cheese3d generate-videos [NAME]

Generate videos with pose estimation results overlaid.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

visualize
^^^^^^^^^

.. code-block:: bash

   cheese3d visualize [NAME]

Visualize the output of Cheese3D using interactive GUI. An interactive prompt lets you select which recording to visualize.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

interactive
^^^^^^^^^^^

.. code-block:: bash

   cheese3d interactive [--web]

Run interactive GUI.

.. list-table:: ``interactive`` options
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``--web``
      - ``bool``
      - ``False``
      - Set to enable web mode UI.

checkpoint
^^^^^^^^^^

.. code-block:: bash

   cheese3d checkpoint [NAME] [--skip-source] [--portable]

Create a checkpoint of this project.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

.. list-table:: ``checkpoint`` options
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``--skip-source``
      - ``bool``
      - ``False``
      - Skip recording directory when creating archive.
    * - ``--portable``
      - ``bool``
      - ``False``
      - Resolve all links or relative paths, ensuring that data is copied into the archive.

restore
^^^^^^^

.. code-block:: bash

   cheese3d restore CHECKPOINT_FILE [NAME] [--skip-source] [--portable]

Restore a project from a checkpoint archive.

This command accepts the :ref:`common arguments <reference/cli:Global options and arguments>`.

.. list-table:: ``restore`` arguments
    :header-rows: 1

    * - Argument
      - Type
      - Default
      - Description
    * - ``CHECKPOINT_FILE``
      - ``str``
      - N/A
      - Path to checkpoint archive.

.. list-table:: ``restore`` options
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``--skip-source``
      - ``bool``
      - ``False``
      - Skip recording directory when extracting archive.
    * - ``--portable``
      - ``bool``
      - ``False``
      - The archive was created with portable mode. No-op flag for API compatibility.
