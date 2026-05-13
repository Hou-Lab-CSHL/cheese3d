Synchronization guide
=====================

Cheese3D records from multiple cameras (and optionally an electrophysiology system) that each have their own internal clocks. Even though the cameras are triggered together, small differences in start times and sample rates accumulate over a recording. The ``cheese3d sync`` command aligns each video (and ephys file) to a single reference video so that frame ``i`` from every camera corresponds to the same moment in time.

This guide explains:

.. contents::
    :local:
    :depth: 1
    :backlinks: none

How synchronization works
-------------------------

Cheese3D relies on an infrared LED that is visible to every camera (see :doc:`/guides/hardware`). The LED pulses on for ``20 ms`` every ``10 ± 0.5 s``. For ephys, the same pulse train is sent to an analog input channel on the acquisition system. Because every recording sees the same pulse train, we can use the pulses as common landmarks to align signals.

For each recording, Cheese3D:

1. Picks one camera as the **reference view** (set by ``sync.ref_view`` in :ref:`reference/configuration:Sync options`).
2. Reads the LED brightness within the ``sync_led`` crop region of the reference video and converts it to a binary on/off signal.
3. Repeats step 2 for every other video (the **targets**).
4. Runs a pipeline of alignment algorithms (``sync.pipeline``) to estimate, for each target, a time lag and a corrected sample rate relative to the reference.
5. Uses FFmpeg to shift each target video by the estimated lag so all videos start at the same instant.
6. If ephys data is configured, repeats the alignment between the reference video and the ephys sync channel, writing the lag and sample rate to disk (the ephys file itself is not modified).

After running ``cheese3d sync``, every synchronized video has a sibling ``<video>.align.json`` file alongside several quality-control (QC) plots used for debugging.

Configuring the sync pipeline
-----------------------------

The ``sync`` section of ``config.yaml`` controls the alignment behavior. See :ref:`reference/configuration:Sync options` for the full list of keys. A typical configuration looks like:

.. code-block:: yaml

   sync:
     pipeline: [crosscorr, regression, samplerate]
     led_threshold: 0.9
     led_peak_algorithm: dynamic
     max_regression_rmse: 0.01
     ref_view: bottomcenter
     ref_crop: default

The ``pipeline`` key is a list of alignment stages applied in order. Each stage refines the lag/sample-rate estimate produced by the previous stage, and potentially rejects poor results:

* ``crosscorr`` — Computes the cross-correlation between reference and target LED signals to find a coarse lag. Best at finding the initial offset between recordings. Rejects results when the peak cross correlation amplitude is smaller than the number of total peaks.
* ``regression`` — Regresses target pulse times against reference pulse times. Both the lag *and* the target sample rate are adjusted. The result is only accepted if the regression RMSE is below ``max_regression_rmse``.
* ``samplerate`` — Estimates each signal's pulse-to-pulse interval and corrects the target sample rate by the ratio of medians. Useful when clock drift is the dominant source of misalignment. Generally always accepted barring numerical issues (division by zero).

You can include or exclude stages depending on your setup. For example, if your cameras share a hardware trigger and you only care about clock drift, you can use ``pipeline: [samplerate]``.

.. tip::

    We recommend always using ``regression`` as it is most robust to differences in experimental setups.

The ``ref_view`` must match a view name from the ``views`` configuration, and the ``sync_led`` ``extra_crops`` region must be defined for every view so the LED can be located in each camera's frame:

.. code-block:: yaml

   views:
     bottomcenter:
       view: BC
       crop: [null, null, null, null]
       extra_crops:
         sync_led: [250, 265, 20, 35]
     topleft:
       view: TL
       crop: [null, null, null, null]
       extra_crops:
         sync_led: [310, 325, 40, 55]
     # ... one entry per camera

Adjust the ``[xstart, xend, ystart, yend]`` of each ``sync_led`` crop so it tightly encloses the LED in the corresponding view. Tight crops give a cleaner brightness trace and more reliable pulse detection.

Synchronizing ephys data
------------------------

To synchronize an ephys recording with the videos, fill in both the ``ephys_root`` and ``ephys_param`` keys in :ref:`main_config_ref`. Cheese3D currently supports three systems out of the box:

* `Allego <https://neuronexus.com/products/data-acquisition-systems/allego-software/>`__ (``type: allego``)
* `Open Ephys <https://open-ephys.org/gui>`__ (``type: openephys``)
* `DSI <https://www.dsiplus.com/>`__ (``type: dsi``)

See :ref:`reference/configuration:Ephys options` for the parameters available for each system. The key things to set are:

* ``sample_rate`` — Must match the acquisition system's configured sample rate.
* ``sync_channel`` — The analog input channel wired to the LED pulse train (not needed for DSI).
* ``sync_threshold`` — Voltage threshold for detecting an "on" pulse. Adjust if your pulse amplitude differs from the default ``0.2``.

When ``cheese3d sync`` runs and ephys data is configured, an additional ``.align.json`` file is written next to each ephys recording. Downstream analysis code can use the ``lag_time``, ``slope``, and ``sample_rate`` fields to map ephys samples to video frames.

Running synchronization
-----------------------

To synchronize a project, run:

.. code-block:: bash

   cheese3d sync [NAME]

where ``NAME`` is the project name (or ``.`` for the current directory). See :ref:`reference/cli:sync` for command-line details.

.. note::
    If a ``.align.json`` file already exists for a target video, it is skipped. Delete the file to re-run synchronization for that target.

After ``sync`` completes, the recording folder will contain several new files for each video:

* ``<video>.align.json`` — The estimated alignment parameters (lag, slope, sample rate).
* ``<video>-qc-brightness.png`` — The brightness trace of the ``sync_led`` crop with the detection threshold overlaid.
* ``<video>-qc-bbox.png`` — An exemplar frame with the ``sync_led`` crop drawn as a red rectangle.
* ``<video>.qc-stage-<i>.png`` — Per-stage debug plot for the ``i``-th pipeline stage.
* ``<video>.qc-final.png`` — The aligned vs. unaligned signals over time, with first/last pulse zoom-ins.

Debugging synchronization issues
--------------------------------

When alignment fails or produces obviously wrong shifts, the QC plots are the first place to look.

**The LED crop is wrong.** Open ``<video>-qc-bbox.png`` and check that the red rectangle tightly encloses the LED. If the LED is outside the box, adjust the ``sync_led`` crop in the ``extra_crops`` for that view. If the box is too large, ambient brightness changes can swamp the LED signal.

**No pulses are detected.** Open ``<video>-qc-brightness.png``. You should see a flat baseline with sharp spikes every ``~10 s``. Some potential issues:

* If you see only noise, the crop is mispositioned (see above)
* If the peaks do not cross the dashed red line or noise crosses this line, ``led_threshold`` is too high---try lowering it (e.g. ``0.7``)
* If the dashed black line is a poor "max brightness", ``led_peak_algorithm`` is misestimating the peak---try ``"max"`` instead

**The cross-correlation lag looks wrong.** Open ``<video>.qc-stage-0.png``. A correct alignment shows a single dominant peak in the cross-correlation (with height greater than the number of total peaks). Multiple peaks of similar height might mean the LED detection is unreliable---fix the brightness trace first. Occasionally, there may not be a unique lag at which all peaks across source and target are overlapping; in this case, you can trust the algorithm to reject the results and rely on ``regression`` instead.

**The regression stage is rejected.** Cheese3D prints the regression RMSE during sync. If it is above ``max_regression_rmse``, the stage is considered "bad" and the previous estimate is kept. Either:

* Raise ``max_regression_rmse`` if your cameras are known to drift significantly, or
* Inspect ``<video>.qc-stage-1.png``: the scatter of target pulse times vs. reference pulse times should lie on a near-identity line. Outliers usually indicate spurious or missed pulses. Clearly non-linear trends indicate time-varying misalignment of the sample rate which our pipeline does not currently correct.

**The final alignment looks correct, but downstream tracking is off.** Open ``<video>.qc-final.png``. The "First pulse (after)" and "Last pulse (after)" panels should show the reference and target pulses overlapping. If this isn't the case, then inspect the other QC plots to debug more.
