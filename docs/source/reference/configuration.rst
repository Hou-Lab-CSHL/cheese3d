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
        - ``view``: a substring indicating the video camera view (e.g. ``"TL"`` for top left); should with ``views`` option
    * - ``model``
      - :ref:`model_options`
      - :ref:`model_options`
      - Options for the 2D pose estimation model.
    * - ``ephys_regex``
      - :doc:`Regex options </howto/regex>`
      - ``null``
      - Optional ephys file matching regex.
    * - ``ephys_param``
      - :ref:`ephys_params`
      - ``null``
      - Optional options for the ephys acquisition system.
    * - ``fps``
      - ``int``
      - ``100``
      - Frames per second of *all* cameras.
    * - ``sync``
      - :ref:`sync_options`
      - :ref:`sync_options`
      - Options for temporally synchronizing video (and ephys) data sources.

.. _model_options:

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

.. _ephys_params:

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

.. _sync_options:

Sync options
------------

Cheese3D will temporally synchronize video data across all cameras (and optionally ephys data). Below are the options for the
