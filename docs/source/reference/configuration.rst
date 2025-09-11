Configuration file
==================

All Cheese3D projects are defined by a configuration file. For convenience, the interactive UI allows you to define a "typical" configuration file (see :ref:`create_new_project`). If you need more advanced settings, you can edit the configuration file directly (located under ``config.yaml`` in your project folder).

Main configuration options
--------------------------

Below is a description of all the options available in the configuration file.

.. _main_config_ref:
.. list-table:: Main configuration file parameters
    :header-rows: 1

    * - Key
      - Type
      - Default
      - Description
    * - ``name``
      - ``str``
      - N/A
      - The name of your project which should match the project folder name exactly.
    * - ``recording_root``
      - ``str``
      - ``videos``
      - The path relative to the project folder where video data should be found.
    * - ``ephys_root``
      - ``str``
      - ``null``
      - The path relative to the project folder where ephys data should be found. Not required if ephys is not used in the project.
    * - ``model_root``
      - ``str``
      - ``model``
      - The path relative to the project folder where models should be stored.
    * - ``video_regex``
      - :doc:`Regex options </howto/regex>`
      - ``{"_path_": ".*_{{type}}_{{view}}.*\.avi", "type": "[^_]+", "view": "TL|TR|L|R|TC|BC"}``
      - Video file matching regex. Requires groups

        - ``type``: a substring indicating the type of video (e.g., ``"cal"`` for calibration)
        - ``view``: a substring indicating the video camera view (e.g. ``"TL"`` for top left); should with align ``views`` option
    * - ``model``
      - :ref:`reference/configuration:Model options`
      - :ref:`reference/configuration:Model options`
      - Options for the 2D pose estimation model.
    * - ``ephys_regex``
      - :doc:`Regex options </howto/regex>`
      - ``null``
      - Optional ephys file matching regex.
    * - ``ephys_param``
      - :ref:`reference/configuration:Ephys options`
      - ``null``
      - Optional options for the ephys acquisition system.
    * - ``fps``
      - ``int``
      - ``100``
      - Frames per second of *all* cameras.
    * - ``sync``
      - :ref:`reference/configuration:Sync options`
      - :ref:`reference/configuration:Sync options`
      - Options for temporally synchronizing video (and ephys) data sources.
    * - ``recordings``
      - ``List[Dict[str, str]]`` (see :ref:`reference/configuration:Recording options`)
      - ``[]``
      - List of recording sessions to process. Each entry must contain a ``name`` key.
    * - ``triangulation``
      - :ref:`reference/configuration:Triangulation options`
      - :ref:`reference/configuration:Triangulation options`
      - Options for configuring 3D reconstruction stage of the pipeline.
    * - ``views``
      - :ref:`reference/configuration:View options`
      - :ref:`reference/configuration:View options`
      - Multi-camera setup configuration.
    * - ``calibration``
      - :ref:`reference/configuration:Calibration options`
      - ``{"type": "cal"}``
      - Options specifying which video type to use for calibration.
    * - ``keypoints``
      - :ref:`reference/configuration:Keypoint options`
      - :ref:`reference/configuration:Keypoint options`
      - List of anatomical keypoints to track.
    * - ``ignore_keypoint_labels``
      - ``List[str]``
      - ``["ref(head-post)"]``
      - List of keypoints to ignore when generating videos of pipeline output.

Model options
-------------

Below is a description of the sub-configuration under the ``model`` key in the :ref:`main_config_ref`.

.. _model_config_ref:
.. list-table:: Model configuration file parameters
    :header-rows: 1

    * - Key
      - Type
      - Default
      - Description
    * - ``name``
      - ``str``
      - N/A
      - Name of the model.
    * - ``backend_type``
      - ``Literal["dlc"]``
      - ``dlc``
      - Type of 2D pose estimation model framework used.
    * - ``backend_options``
      - ``Dict[str, Any]``
      - Auto-generated
      - Additional options relevant to the backend. Just ``experimenter`` and ``date`` for now with DLC. These are auto-generated when creating a project based on whether the model is created or imported.

Ephys options
-------------

Below is a description of the sub-configurations under the ``ephys_params`` key in the :ref:`main_config_ref`. Each type of ephys system has a different set of allowed configurations.

.. _allego_config_ref:
.. list-table:: Allego configuration file parameters
    :header-rows: 1

    * - Key
      - Type
      - Default
      - Description
    * - ``type``
      - ``Literal["allego"]``
      - ``allego``
      - Type of ephys system.
    * - ``sync_channel``
      - ``int``
      - ``32``
      - Channel for synchronization signal.
    * - ``sync_threshold``
      - ``float``
      - ``0.2``
      - Voltage threshold for detecting an "on" synchronization pulse.
    * - ``sample_rate``
      - ``int``
      - ``30000``
      - Sample rate of the acquisition system.

.. _openephys_config_ref:
.. list-table:: Open Ephys configuration file parameters
    :header-rows: 1

    * - Key
      - Type
      - Default
      - Description
    * - ``type``
      - ``Literal["openephys"]``
      - ``openephys``
      - Type of ephys system.
    * - ``sync_channel``
      - ``int``
      - ``32``
      - Channel for synchronization signal.
    * - ``sync_threshold``
      - ``float``
      - ``0.2``
      - Voltage threshold for detecting an "on" synchronization pulse.
    * - ``sample_rate``
      - ``int``
      - ``30000``
      - Sample rate of the acquisition system.

.. _dsi_config_ref:
.. list-table:: DSI configuration file parameters
    :header-rows: 1

    * - Key
      - Type
      - Default
      - Description
    * - ``type``
      - ``Literal["dsi"]``
      - ``dsi``
      - Type of ephys system.
    * - ``sync_threshold``
      - ``float``
      - ``0.2``
      - Voltage threshold for detecting an "on" synchronization pulse.
    * - ``sample_rate``
      - ``int``
      - ``1000``
      - Sample rate of the acquisition system.

Sync options
------------

Cheese3D will temporally synchronize video data across all cameras (and optionally ephys data). Below are the options for the ``sync`` configuration in the :ref:`main_config_ref`.

.. _sync_config_ref:
.. list-table:: Sync configuration file parameters
    :header-rows: 1

    * - Key
      - Type
      - Default
      - Description
    * - ``pipeline``
      - ``List[str]``
      - ``["crosscorr", "regression", "samplerate"]``
      - Sequential list of alignment algorithms to apply. Available options: ``"crosscorr"`` (cross-correlation), ``"regression"`` (linear regression), ``"samplerate"`` (sample rate correction).
    * - ``led_threshold``
      - ``float``
      - ``0.9``
      - Threshold for detecting LED synchronization signals in video frames (0.0-1.0).
    * - ``max_regression_rmse``
      - ``float``
      - ``0.01``
      - Maximum allowed RMSE for regression-based alignment. Higher values are more permissive.
    * - ``ref_view``
      - ``str``
      - ``"bottomcenter"``
      - Reference camera view to use for synchronization. Must match one of the view names in the ``views`` configuration.
    * - ``ref_crop``
      - ``str``
      - ``"default"``
      - Crop region to use from the reference view. Use ``"default"`` for main crop or specify an ``extra_crops`` key (see :ref:`reference/configuration:View options`.

Recording options
-----------------

The ``recordings`` section in the :ref:`main_config_ref` defines the video recordings to process. Each recording has the following options:

.. list-table:: Recordings configuration
    :header-rows: 1

    * - Key
      - Type
      - Description
    * - ``name``
      - ``str``
      - The name of the folder in ``recording_root`` that matches this video recording.
    * - ``<regex_group>``
      - ``<value>``
      - Additional identifiers corresponding to regex groups in ``video_regex`` that can further filter the list of videos in ``<recording_root>/<name>`` to the videos that match a single session.

Example:

.. code-block:: yaml

   recordings:
   - name: 20250522_B1_ephys-record_rig1
   - name: 20250523_B1_ephys-record_rig1
     type: control

In the example, the ``<recording_root>/20250523_B1_ephys-record_rig1`` folder contains multiple recordings, so we filter the potential matches to the files where the ``type`` group has a value of ``control``. See :doc:`/howto/regex` for more information on regex groups and how to use them.

Calibration options
^^^^^^^^^^^^^^^^^^^

The ``calibration`` key allows you to filter a matched list of videos for a recording into calibration recordings and normal video recordings. Its value is similar to an entry in ``recordings`` but excludes the ``name`` key (i.e. it is just a list of regex groups to filter the matched files).

Example:

.. code-block:: yaml

   calibration:
     type: cal

Triangulation options
---------------------

The ``triangulation`` section configures 3D reconstruction:

.. list-table:: Triangulation configuration file parameters
    :header-rows: 1

    * - Key
      - Type
      - Default
      - Description
    * - ``axes``
      - ``List[List[str]]``
      - ``[["z", "nose(top)", "nose(bottom)"], ["x", "eye(front)(left)", "eye(front)(right)"]]``
      - Defines coordinate system axes using keypoint pairs. Each sub-list contains [axis_name, point1, point2].
    * - ``ref_point``
      - ``str``
      - ``"ref(head-post)"``
      - Reference keypoint for coordinate system origin.
    * - ``filter2d``
      - ``bool``
      - ``false``
      - Whether to apply 2D filtering before triangulation.
    * - ``score_threshold``
      - ``float``
      - ``0.9``
      - Minimum confidence score for keypoints to be used in triangulation.

View options
------------

The ``views`` section defines multi-camera setup configuration. Each view has the following parameters:

.. list-table:: View configuration parameters
    :header-rows: 1

    * - Key
      - Type
      - Description
    * - ``path``
      - ``str``
      - Camera identifier that matches the ``view`` regex group in ``video_regex``.
    * - ``crop``
      - ``List[Optional[int]]``
      - Main bounding box crop as ``[xstart, xend, ystart, yend]``. Use ``null`` for no cropping on that dimension.
    * - ``extra_crops``
      - ``Dict[str, List[int]]``
      - Named additional crop regions, typically used for synchronization LEDs bounding boxes.
    * - ``filterspec``
      - ``Optional[Dict[str, float]]``
      - FFMPEG filter specifications for brightness, contrast, and saturation adjustments.

Example:

.. code-block:: yaml

   views:
     topleft:
       path: TL
       crop: [null, null, null, null]
       extra_crops:
         sync_led: [250, 265, 20, 35]
       filterspec: null

Keypoint options
----------------

The ``keypoints`` section defines anatomical points to track. Each keypoint has the following options:

.. list-table:: Keypoint configuration parameters
    :header-rows: 1

    * - Key
      - Type
      - Description
    * - ``label``
      - ``str``
      - Unique name for the anatomical keypoint.
    * - ``groups``
      - ``List[str]``
      - Functional groups this keypoint belongs to (e.g., "nose", "eye(left)", "whiskers(right)").
    * - ``views``
      - ``List[str]``
      - List of camera views where this keypoint should be labeled and tracked.

Example:

.. code-block:: yaml

   keypoints:
   - label: nose(tip)
     groups: [nose]
     views: [topleft, topright, left, right, topcenter, bottomcenter]
   - label: eye(front)(left)
     groups: [eye(left)]
     views: [topleft, left, topcenter]
