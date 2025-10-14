.. cheese3d documentation master file

Cheese3D
========

Cheese3D is a pipeline for tracking mouse facial movements built on top of existing tools like https://github.com/DeepLabCut/DeepLabCut and https://github.com/lambdaloop/anipose. By tracking anatomically-informed keypoints using multiple cameras registered in 3D, our pipeline produces sensitive, high-precision facial movement data that can be related internal state (e.g., electrophysiology).

.. image:: /_static/Cheese3D.gif
    :width: 59%
.. image:: /_static/Cheese3DIcon.png
    :width: 33%

|
Cheese3D output can be visualized interactively.

.. image:: /_static/Cheese3DVisualizer.gif
    :width: 49%
.. image:: /_static/Cheese3DVisualizerStatic.png
    :width: 49.2%

|
Using a combination of hardware synchronization signals and a multi-stage pipeline, we are able to precisely synchronize video and electrophysiology data. This allows us to relate spikes recorded in the brainstem to various facial movements (here, we highlight two example units correlated with ipsilateral ear movements).

.. image:: /_static/Cheese3DFlowchart.png
|
.. image:: /_static/Cheese3DSync.png

|
If you use Cheese3D, please cite our preprint:

.. code-block:: bibtex

    @article {Daruwalla2024.05.07.593051,
    	author = {Daruwalla, Kyle and Martin, Irene Nozal and Zhang, Linghua and Nagli{\v c}, Diana and Frankel, Andrew and Rasgaitis, Catherine and Zhang, Xinyan and Ahmad, Zainab and Borniger, Jeremy C. and Hou, Xun Helen},
    	title = {Cheese3D: Sensitive Detection and Analysis of Whole-Face Movement in Mice},
    	elocation-id = {2024.05.07.593051},
    	year = {2025},
    	doi = {10.1101/2024.05.07.593051},
    	publisher = {Cold Spring Harbor Laboratory},
    	URL = {https://www.biorxiv.org/content/early/2025/03/01/2024.05.07.593051},
    	eprint = {https://www.biorxiv.org/content/early/2025/03/01/2024.05.07.593051.full.pdf},
    	journal = {bioRxiv}
    }

.. toctree::
    :maxdepth: 2
    :caption: Guides

    guides/installation
    guides/hardware
    guides/quick-start

.. toctree::
    :maxdepth: 2
    :caption: How-to examples

    howto/regex

.. toctree::
    :maxdepth: 2
    :caption: Reference

    reference/configuration

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
