# Cheese3D

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://hou-lab-cshl.github.io/cheese3d/)
[![Data](https://img.shields.io/badge/download-demodata-blue)](https://labshare.cshl.edu/shares/houlab/www-data/cheese3d_paper_data/cheese3d_demo.tar.gz)
<!--[Download demo data](https://labshare.cshl.edu/shares/houlab/www-data/cheese3d_paper_data/cheese3d_demo.tar.gz)-->

Cheese3D is a pipeline for tracking mouse facial movements built on top of existing tools like [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) and [Anipose](https://github.com/lambdaloop/anipose). By tracking anatomically-informed keypoints using multiple cameras registered in 3D, our pipeline produces sensitive, high-precision facial movement data that can be related internal state (e.g., electrophysiology).

<p align="center">
   <img src="docs/source/_static/Cheese3D.gif" alt="Animation of Cheese3D pipeline", width=60%>
   <img src="docs/source/_static/Cheese3DIcon.png" alt="Animation of Cheese3D pipeline", width=34%>
</p>

Cheese3D output can be visualized interactively.

<p align="center">
   <img src="docs/source/_static/Cheese3DVisualizer.gif" alt="Animation of Cheese3D visualizer", width=49%>
   <img src="docs/source/_static/Cheese3DVisualizerStatic.png" alt="Animation of Cheese3D visualizer", width=49%>
</p>

Using a combination of hardware synchronization signals and a multi-stage pipeline, we are able to precisely synchronize video and electrophysiology data. This allows us to relate spikes recorded in the brainstem to various facial movements (here, we highlight two example units correlated with ipsilateral ear movements).

![](docs/source/_static/Cheese3DFlowchart.png)

![](docs/source/_static/Cheese3DSync.png)

 If you use Cheese3D, please cite our preprint:
```
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
```

## System Requirements

Cheese3D is supported on most Linux and macOS systems (including GPU support for CUDA and Apple Silicon). Partial support is available on Windows. For details, please refer to [our documentation](https://hou-lab-cshl.github.io/cheese3d/guides/installation.html#platform-specific-support).
