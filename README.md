# Cheese3D

Cheese3D is a pipeline for tracking mouse facial movements built on top of existing tools like [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) and [Anipose](https://github.com/lambdaloop/anipose). By tracking anatomically-informed keypoints using multiple cameras registered in 3D, our pipeline produces sensitive, high-precision facial movement data that can be related internal state (e.g., electrophysiology).

![](Cheese3D.gif)

Using a combination of hardware synchronization signals and a multi-stage pipeline, we are able to precisely synchronize video and electrophysiology data. This allows us to relate spikes recorded in the brainstem to various facial movements (here, we highlight two example units correlated with ipsilateral ear movements).

![](Cheese3DSync.png)

If you use Cheese3D, please cite our preprint:
```
@article {Daruwalla2024.05.07.593051,
    author = {Daruwalla, Kyle and Martin, Irene Nozal and Frankel, Andrew and Nagli{\v c}, Diana and Ahmad, Zainab and Hou, Xun Helen},
    title = {A 3D whole-face movement analysis system to uncover underlying physiology in mice},
    elocation-id = {2024.05.07.593051},
    year = {2024},
    doi = {10.1101/2024.05.07.593051},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2024/05/08/2024.05.07.593051},
    eprint = {https://www.biorxiv.org/content/early/2024/05/08/2024.05.07.593051.full.pdf},
    journal = {bioRxiv}
}
```

## Quick start

First, follow the [installation instructions](#installation-instructions). Next, either
- Refer to the [usage guide](USAGE.md)
- Refer to the [hardware guide](https://github.com/Hou-Lab-CSHL/cheese3d/tree/main/hardware)
- Look at example of analysis with Cheese3D output from our preprint: [paper figure notebooks](https://github.com/Hou-Lab-CSHL/cheese3d/tree/main/paper/)

## Installation instructions

1. Install [pixi](https://pixi.sh/latest/#installation)
2. Clone or download this repository
   (`git clone https://github.com/Hou-Lab-CSHL/cheese3d`)
3. Change directories to the repository folder
   (`cd <path to repo>/cheese3d`)
4. Activate the environment (`pixi shell`)
