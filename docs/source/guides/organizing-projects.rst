Organizing your Cheese3D projects/data
======================================

A Cheese3D project is just a directory on disk: a configuration file, some video data, optionally an ephys recording, and a pose estimation model. This basic structure offers an easy way to keep projects self-contained and organized. This guide describes the recommended layout and the conventions Cheese3D uses to locate files inside it.

.. contents::
    :local:
    :depth: 2
    :backlinks: none

Anatomy of a project folder
---------------------------

A Cheese3D project is self-contained in a single folder whose name matches the ``name`` field in :ref:`main_config_ref`. A fully populated project looks like this:

.. code-block:: text

    <project>/
    |__ config.yaml                  # All project settings (see Configuration reference)
    |__ videos/                      # Source video recordings, one sub-folder per session
    |  |__ <session-1>/
    |  |  |__ <video files>
    |  |__ <session-2>/
    |     |__ <video files>
    |__ ephys/                       # Optional; only when ephys_root is set
    |  |__ <session-1>/
    |     |__ <ephys files>
    |__ model/                       # 2D pose estimation models for this project
    |  |__ <model-name>/
    |     |__ backend/               # The underlying DLC project
    |     |__ labels/                # Frames extracted/labeled in this project
    |__ triangulation/               # Auto-created during the 3D stage
       |__ config.toml               # Anipose configuration
       |__ <session-1>/
          |__ videos-raw/            # Synchronized inputs (symlinks to videos/)
          |__ calibration/           # Camera calibration output
          |__ pose-2d/               # 2D keypoint predictions
          |__ pose-3d/               # 3D triangulated keypoints
          |__ cheese3d/              # Cheese3D feature CSV
          |__ videos-compare/        # Labeled output videos

Only ``config.yaml`` is required up front. Every other folder is either populated by you (e.g., ``videos``, ``ephys``) or created by Cheese3D as you advance through the pipeline (e.g., the ``triangulation/`` folder).

The names ``videos``, ``ephys``, ``model``, and ``triangulation`` are defaults; they can be remapped via the ``video_root``, ``ephys_root``, and ``model_root`` keys in :ref:`main_config_ref`. The discussion below uses the default names.

Organizing source video recordings
----------------------------------

Cheese3D treats each recording session as a folder of videos---typically one file per camera view---under ``videos/``. The folder name is the **session name**, which you list under ``sessions`` in :ref:`main_config_ref`:

.. code-block:: text

    videos/
    |__ 20231031_chew/
    |  |__ chewing_TL_001.avi
    |  |__ chewing_TR_001.avi
    |  |__ chewing_BC_001.avi
    |  |__ ...
    |__ 20231101_baseline/
       |__ baseline_TL_001.avi
       |__ ...

Within each session folder, Cheese3D discovers individual videos using the ``video_regex`` pattern. The regex must capture at least a ``view`` group (so each file is matched to a camera) and may include other groups like ``type`` for further grouping and filtering.

Naming video files
^^^^^^^^^^^^^^^^^^

The video filename is what Cheese3D uses to attach a camera view and other information to a recording. The more metadata you encode in the filename, the more flexibility you have later when filtering or grouping recordings via ``video_regex``.

A useful filename scheme typically encodes:

* **view** (required): which camera the video came from (e.g. ``TL``, ``TR``, ``BC``). Must match the ``view`` field of an entry in :ref:`reference/configuration:View options`.
* **type**: the kind of recording (e.g. ``cal`` for calibration, ``behavior`` for experimental trials).
* **subject**: the mouse identifier (e.g. ``mouse1``, ``M042``). Encoding this lets you keep many subjects' recordings in one folder and split them apart via configuration.
* **condition**: an experimental condition or stimulus label (e.g. ``baseline``, ``stim``).
* **session number** or **repetition**: a counter to disambiguate repeated recordings.

For example, a recording folder with two subjects across baseline and stim conditions might look like:

.. code-block:: text

    videos/
    |__ 20240115_pilot/
       |__ mouse1_cal_TL_001.avi
       |__ mouse1_cal_TR_001.avi
       |__ ...
       |__ mouse1_baseline_TL_001.avi
       |__ mouse1_baseline_TR_001.avi
       |__ ...
       |__ mouse2_stim_TL_001.avi
       |__ mouse2_stim_TR_001.avi
       |__ ...

You can describe this layout with a regex that captures each piece of metadata as a named group:

.. code-block:: yaml

    video_regex:
      _path_: "{{subject}}_{{type}}_{{view}}_{{rep}}\\.avi"
      subject: "mouse\\d+"
      type: "cal|baseline|stim"
      view: "TL|TR|L|R|TC|BC"
      rep: "\\d+"

Any group you define here becomes a key you can filter on in ``sessions`` and ``calibration``. For example, to only use specific subjects from a folder for analysis:

.. code-block:: yaml

    sessions:
      - name: 20240115_pilot
        subject: mouse1
        type: baseline
      - name: 20240115_pilot
        subject: mouse2
        type: stim
    calibration:
      type: cal

In this configuration, calibration videos are shared across subjects (since they only filter on ``type: cal``), while each session pulls only the trials belonging to one mouse. See :doc:`/howto/regex` for a complete walkthrough of how the dictionary form is translated into a Python regex with named groups.

.. tip::

    Pick a filename convention before you start recording and stick to it. The ``video_regex`` only needs to be written once if every recording follows the same scheme. Renaming files after the fact is tedious; baking subject IDs, conditions, and view codes into your acquisition pipeline pays off later.

Organizing ephys data
---------------------

Ephys is optional. To use it, set ``ephys_root`` in :ref:`main_config_ref` (commonly to ``ephys``) and add an ``ephys_param`` block for your acquisition system. Cheese3D currently supports `Allego <https://www.neuronexus.com/products/software/radiens/>`__, `Open Ephys <https://open-ephys.org/gui>`__, and `DSI <https://www.datasci.com/products/software/ponemah>`__ --- see :ref:`reference/configuration:Ephys options`.

Ephys files are organized **per session, mirroring the video layout**. Cheese3D looks for ephys recordings under ``<ephys_root>/<session-name>/``, matching the same session names listed in ``sessions``:

.. code-block:: text

    ephys/
    |__ 20231031_chew/
    |  |__ <ephys recording files>
    |__ 20231101_baseline/
       |__ <ephys recording files>

Which files Cheese3D picks up inside each session folder is controlled by the ``ephys_regex`` pattern, just like videos. After running ``cheese3d sync``, a sibling ``.align.json`` file is written next to each ephys recording with the alignment parameters (see :doc:`/howto/sync`).

Pose estimation models
----------------------

The 2D pose estimation model lives under ``model/<model-name>/``, where ``<model-name>`` is the ``model.name`` field in :ref:`main_config_ref`:

.. code-block:: text

    model/
    |__ cheese3d_demo_model/
       |__ backend/        # The DLC project (config.yaml, training-datasets/, etc.)
       |__ labels/         # Frames extracted and labeled inside Cheese3D

You can either **create** a fresh model when setting up a new project or **import** an existing DLC project that you've already trained (the interactive UI prompts for either). Importing is the right choice when:

* you have a pre-trained model that already covers the keypoints you care about, or
* multiple projects share the same keypoint schema and you'd rather not retrain.

Importing copies the DLC project into ``model/<model-name>/backend/``, so each Cheese3D project gets its own independent copy. If you want a single canonical model to be the source of truth, keep the model outside Cheese3D (e.g. in a shared ``dlc-projects/`` directory) and either:

* re-import them into each project, keeping independent copies, or
* change the ``model_root`` key in :ref:`main_config_ref` to point to the shared DLC project folder.

.. tip::

    Newly labeled frames go into ``model/<model-name>/labels/`` and are automatically kept in sync with the backend folder.

The ``triangulation/`` folder
-----------------------------

The ``triangulation`` folder contains an `Anipose <https://github.com/lambdaloop/anipose>`__ project with one sub-folder per session, plus a top-level ``config.toml`` that Cheese3D generates from your ``config.yaml``. Everything in this folder is generated.

Managing multiple projects
--------------------------

Each Cheese3D project is independent. If you run several experiments, give each one its own project folder rather than packing them together. A common workspace layout looks like:

.. code-block:: text

    cheese3d-workspace/
    |__ dlc-projects/                       # Shared DLC models, used by projects below
    |  |__ mymodel/
    |__ 20231031_chewing_study/
    |  |__ config.yaml
    |  |__ videos/
    |__ 20240115_baseline_pilot/
       |__ config.yaml
       |__ videos/

Putting all projects under a single parent directory makes it easy to back up everything at once, browse past experiments, and share models between them. Similar to sharing DLC models, you can also share video or ephys data by using a shared folder and adjusting ``video_root`` and ``ephys_root`` in the config.
