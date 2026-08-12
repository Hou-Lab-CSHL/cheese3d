# Cheese3D

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://hou-lab-cshl.github.io/cheese3d/)
[![Data](https://img.shields.io/badge/download-demodata-blue)](https://labshare.cshl.edu/shares/houlab/www-data/cheese3d_paper_data/cheese3d_demo.tar.gz)

Cheese3D is a pipeline for tracking mouse facial movements built on top of existing tools ([DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) and [Anipose](https://github.com/lambdaloop/anipose)). By tracking anatomically-informed keypoints using multiple cameras registered in 3D, our pipeline produces sensitive, high-precision facial movement data that can be related internal state (e.g., electrophysiology).

**NEW!** Check out the [Cheese3D paper](https://www.nature.com/articles/s41593-026-02262-8). 

<p align="center">
   <img src="docs/source/_static/Cheese3D.gif" alt="Animation of Cheese3D pipeline" style="height:200px; width:auto;">
   <img src="docs/source/_static/Cheese3DIcon.png" alt="Cheese3D icon" style="height:200px; width:auto;">
</p>

Cheese3D output can be visualized interactively.

<p align="center">
   <img src="docs/source/_static/Cheese3DVisualizer.gif" alt="Animation of Cheese3D visualizer", width=49%>
   <img src="docs/source/_static/Cheese3DVisualizerStatic.png" alt="Static view of Cheese3D visualizer", width=49%>
</p>

Using a combination of hardware synchronization signals and a multi-stage pipeline, we are able to precisely synchronize video and electrophysiology data. This allows us to relate spikes recorded in the brainstem to various facial movements (here, we highlight two example units correlated with ipsilateral ear movements).

![](docs/source/_static/Cheese3DFlowchart.png)

![](docs/source/_static/Cheese3DSync.png)

If you use Cheese3D, please cite our [manuscript](https://www.nature.com/articles/s41593-026-02262-8):
```
@article{Daruwalla2026cheese3d,
  author    = {Daruwalla, Kyle and Nozal Martin, Irene and Zhang, Linghua and Nagli{\v{c}}, Diana and Frankel, Andrew and Rasgaitis, Catherine and Zhao, Rubin and Zhang, Xinyan and Ahmad, Zainab and Borniger, Jeremy C. and Hou, Xun Helen},
  title     = {Cheese3D enables sensitive detection and analysis of whole-face movement in mice},
  journal   = {Nature Neuroscience},
  year      = {2026},
  doi       = {10.1038/s41593-026-02262-8},
  publisher = {Springer Nature},
  URL       = {https://www.nature.com/articles/s41593-026-02262-8}
}
```

## Reproducing Cheese3D paper figures

The following notebooks contain the code required to reproduce the figures in our paper. They also serve as a showcase of the type of analysis enabled by Cheese3D's output. You can find the complete collection under the [`paper/`](/paper/) directory.

| Example figure panel | Notebook | Description |
|:--------------------:|:---------|:------------|
| <img src="paper/Fig1Example.png" width=200> | `paper/fig1-cheese3d-accuracy.ipynb` | Framework and validation of capturing face-wide movement as 3D geometric features in mice |
| <img src="paper/Fig2Example.png" width=200> | `paper/fig2-cheese3d-jitter-analysis.ipynb` | Reduction in keypoint tracking jitter due to 3D triangulation of data from six camera views |
| <img src="paper/Fig3-1Example.png" width=200> | `paper/fig3-part1-cheese3d-general-anesthesia-eeg.ipynb` | Distinct facial patterns track time during induction and recovery from ketamine-induced anesthesia |
| <img src="paper/Fig3-2Example.png" width=200> | `paper/fig3-part2-prediction-of-eeg-from-facial-features.ipynb` | Predicting EEG frequency band power from facial features |
| <img src="paper/Fig3-3Example.png" width=200> | `paper/fig3-part3-cheese3d-redose-facial-features.ipynb` | Detecting differences in total anesthetic dosage from facial features |
| <img src="paper/Fig4-1Example.png" width=200> | `paper/fig4-part1-chewing-whole-face-kinematics.ipynb` | Chewing kinematics in mouth and surrounding facial areas |
| <img src="paper/Fig4-2Example.png" width=200> | `paper/fig4-part2-consummatory-behavior.ipynb` | Changes in consummatory behavior measured by Cheese3D features |
| <img src="paper/Fig5-1Example.png" width=200> | `paper/fig5-part1-cheese3d-stimulation-triggered-movement.ipynb` | Stimulation triggered facial movements in anesthetized mice |
| <img src="paper/Fig5-2Example.png" width=200> | `paper/fig5-part2-cheese3d-synchronized-electrophysiology.ipynb` | Synchronized Cheese3D with electrophysiology relates motor control activity to subtle facial movements |
| <img src="paper/Fig5-3Example.png" width=200> | `paper/fig5-part3-prediction-of-neural-activity-from-cheese3d.ipynb` | Predicting neural activity of brainstem units from single facial features |

## System Requirements

Cheese3D is supported on most Linux and macOS systems (including GPU support for CUDA and Apple Silicon). Partial support is available on Windows. For details, please refer to [our documentation](https://hou-lab-cshl.github.io/cheese3d/guides/installation.html#platform-specific-support).

Software dependencies are listed in the [pixi.toml](/pixi.toml), [`cheese3d` pyproject.toml](/packages/cheese3d/pyproject.toml), and [`cheese3d-annotator` pyproject.toml](/packages/cheese3d-annotator/pyproject.toml) files. Hardware specifications can be found in [our hardware guide](https://hou-lab-cshl.github.io/cheese3d/guides/hardware.html).

## SLEAP backend

SLEAP 1.6 is isolated from DLC and Lightning Pose in its own Pixi environment:

```bash
pixi install -e sleap
pixi shell -e sleap
cheese3d --path /path/to/projects interactive
```

Choose `sleap` as the model backend when creating a project, or set
`model.backend_type: sleap` in an existing Cheese3D `config.yaml`. The initial
integration uses SLEAP-NN's single-instance pipeline, which matches Cheese3D's
one-animal-per-camera workflow. The Training tab exposes UNet, ConvNeXt, and
SwinT backbones; epoch/step controls; validation split; Adam/AdamW; checkpoint
retention; early stopping; augmentation; and multi-GPU DDP settings.

To initialize SLEAP from an existing DLC, Lightning Pose, or SLEAP project's
labeled data, add the source project directory and its format:

```yaml
model:
  name: cheese3d_sleap_model
  backend_type: sleap
  backend_options:
    source_project_path: /absolute/path/to/source/project
    source_format: dlc  # or lightning_pose, sleap
```

The same `source_project_path`/`source_format` options work for any active
backend (`dlc`, `lightning_pose`, `sleap`), so any Pixi environment can seed a
new project from a DLC, Lightning Pose, or SLEAP source's labeled frames.

SLEAP training produces selectable checkpoints and inference writes the same
DLC-compatible per-camera HDF5 pose files consumed by Cheese3D triangulation.
Importing arbitrary existing SLEAP model directories and round-tripping edits
from SLEAP's native labeling GUI are not implemented in this first version.
