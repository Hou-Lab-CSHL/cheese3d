Backing up and sharing projects
===============================

Two CLI commands help you move a project between machines or archive it for long-term storage:

* ``cheese3d checkpoint``: bundles a project into a single archive
* ``cheese3d restore``: expands an archive back into a working project

This guide explains:

.. contents::
    :local:
    :depth: 1
    :backlinks: none

What is a checkpoint?
---------------------

A **checkpoint** is a single ``.tar.xz`` archive that captures the state of a Cheese3D project at a point in time. Archives are written to the project's ``checkpoints/`` folder using the naming scheme:

.. code-block:: text

    <project>/checkpoints/<name>_<YYYY-MM-DD-HH-MM-SS>.tar.xz

The timestamp is generated when the command is run, so successive checkpoints never overwrite each other.

Checkpoints are useful for:

* **Snapshotting before a destructive change.** Take a checkpoint before retraining the pose model, re-running synchronization, or editing labels. If the new run is worse, ``restore`` brings you back to the previous state.
* **Sharing a project.** Hand a single ``.tar.xz`` to a collaborator or upload it to long-term storage instead of dealing with many loose files and possibly external symlinks.
* **Moving a project between machines.** Use ``--portable`` (see :ref:`howto/backups:Portable archives` below) to ensure the archive is self-contained regardless of where its data originally lived.

What is included in a checkpoint
--------------------------------

By default, ``cheese3d checkpoint`` includes everything inside the project folder *except* the ``checkpoints/`` subfolder itself. Concretely, that means:

* ``config.yaml``: always included (and possibly rewritten; see :ref:`howto/backups:Portable archives` below)
* ``triangulation/``: always included
* ``model/`` (the default ``model_root``): included *if* the model lives inside the project folder; externalized model directories are only included when ``--portable`` is set
* ``videos/`` (the default ``video_root``): included *if* the videos live inside the project folder and ``--skip-source`` is *not* set; externalized video roots require ``--portable``
* ``ephys/`` (the default ``ephys_root``): same rules as ``videos/``---included by default when ``ephys_root`` is inside the project folder, requires ``--portable`` otherwise

See :doc:`/guides/organizing-projects` for the project layout these ``*_root`` folders refer to.

.. tip::

    Because source recordings (videos and ephys) are typically large, you can opt out of them at archive time with ``--skip-source``; the resulting archive contains the configuration, the trained model, the triangulation outputs, and any other project files, but not the raw inputs.

Creating a checkpoint
---------------------

The minimal invocation, from inside a project folder:

.. code-block:: bash

    cheese3d checkpoint

To explicitly skip source recordings (useful when you only want to share the *results* of an analysis):

.. code-block:: bash

    cheese3d checkpoint --skip-source

See :ref:`reference/cli:checkpoint` for the full set of options.

Portable archives
^^^^^^^^^^^^^^^^^

The defaults above only bundle source recordings and models that live **inside the project folder**. If your ``video_root``, ``ephys_root``, or ``model_root`` points to a shared location (e.g., a lab-wide ``dlc-projects/`` folder or an external drive), those files won't appear in a default archive---the resulting checkpoint won't be self-contained for transfer to another machine.

The ``--portable`` flag fixes this:

.. code-block:: bash

    cheese3d checkpoint --portable

With ``--portable``:

* External directories pointed to by ``video_root``, ``ephys_root``, and ``model_root`` are copied into the archive under standard subfolders (``videos/``, ``ephys/``, and ``models/``).
* Symbolic links inside source folders are resolved and the linked files are copied in directly.
* The embedded ``config.yaml`` is rewritten so ``video_root``, ``ephys_root``, and ``model_root`` are set to those standard subfolders. So on restore, the project simply works without further reconfiguration.

``--portable`` and ``--skip-source`` are independent. Combining them produces a self-contained archive of *just* the model and analysis outputs without raw recordings:

.. code-block:: bash

    cheese3d checkpoint --portable --skip-source

.. tip::

    If you're handing a project off to someone who doesn't share your filesystem layout, always use ``--portable``. The archive is then guaranteed to restore into a working project on any machine.

Restoring a checkpoint
----------------------

``cheese3d restore`` expands an archive back into a project on disk:

.. code-block:: bash

    cheese3d restore path/to/<name>_<timestamp>.tar.xz

By default, the project is restored under the ``--path`` directory using the project name baked into the archive. To restore under a different name, pass it as the second argument:

.. code-block:: bash

    cheese3d restore my_project_2026-05-13-09-30-00.tar.xz my_project_copy

You can also skip source recordings at restore time. This is useful when you have a large portable archive but only need the configuration, model, and analysis results on the target machine:

.. code-block:: bash

    cheese3d restore my_project.tar.xz --skip-source

See :ref:`reference/cli:restore` for the full set of options.

.. note::

    ``restore`` will not delete files that already exist in the target directory; it only extracts files from the archive on top of what's already there. To guarantee a clean restore, point ``--path`` at an empty folder, or remove the target project directory before running ``restore``.
