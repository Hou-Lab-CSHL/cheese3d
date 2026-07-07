Troubleshooting guide
=====================

Below are some common issues you might face and steps to resolve them. If you encounter something not explained below, then please open an issue: https://github.com/Hou-Lab-CSHL/cheese3d/issues.

Minimal Pixi version
--------------------

If you setup your environment using Pixi as described in :ref:`guides/installation:Setup`, then you might encounter issues where the ``pixi.toml`` file is has errors. This is because our environment requires Pixi v0.66.0. Check your version with:

.. code-block:: bash

    pixi --version

If this reports anything lower than v0.66.0, please install the latest version of Pixi.

Trouble launching labeling or visualization GUIs (napari)
---------------------------------------------------------

.. note::

    Napari GUIs require a system with a display to function correctly. Headless systems like servers cannot run the GUIs. This is an expected limitation.

Sometimes napari-based GUIs will fail to launch even on a system with a display. In particular, this can fail silently when running via ``cheese3d interactive``. Debug this by manually launching the GUI:

.. code-block:: bash

    cheese3d visualize [project name]

If you receive an error similar to below:

.. code-block:: text

    This application failed to start because it could not find or load the Qt platform plugin "xcb".

    Available platform plugins are: linuxfb, minimal, offscreen, xcb.

    Reinstalling the application may fix this problem. Aborted (core dumped)

This means that the graphics library underlying Napari failed to install correctly. You can try to fix this by reinstalling:

.. code-block:: bash

    pixi reinstall

You may need to reinstall multiple times before the issue is resolved. Sometimes exiting the terminal completely and restarting helps. This should be a one time issue that does not re-occur after it has been resolved.
