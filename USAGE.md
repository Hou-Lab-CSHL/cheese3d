# Cheese3D Workflow Usage Guide

This guide provides an overview of the Cheese3D workflow including analyzing new data and training new models. We provide example projects and datasets to walk through this tutorial.

!!! note
    Before starting, make sure you have installed Cheese3D as described in the [README](README.md).

Download the example projects and datasets HERE. Unpack the downloaded archive using:
```bash
tar -xvzf cheese3d_examples.tar.gz
```

Use the flow chart below to guide you through the workflow.
![](Cheese3DFlowchart.png)

## Stage 1: Analyzing new videos

Suppose you have new video data that you wish to analyze with Cheese3D. Start by placing the videos in `video-data` in their own subfolder. For this tutorial, we already provide a dataset called `20231031_B6-B20-B21-B26-B31-B32-B33_chew_rig2` as an example.

Next, create an Anipose project configuration for your new data under `configs/anipose`. We provide `configs/anipose/example-anipose.yaml` as a reference. Let's go over the parts of this template that you should customize.

First, the following keys specify the name of your Anipose project and a path to a DLC project to use for analysis:
```yaml
name: example-anipose
dlc_model: ${paths.dlc.projects}/example-dlc-defaultuser-2023-12-14
```
These can be any name and path that you want (as long as it points to a valid DLC model). In this example, we use the name and model path for the projects provided by the example data that you previously downloaded.

Next, you should add an entry for the new data that you want to analyze. For example, we can refer to the entry for the `20231031_B6-B20-B21-B26-B31-B32-B33_chew_rig2` dataset:
```yaml
sessions:
- name: 20231031_B6-B20-B21-B26-B31-B32-B33_chew_rig2
  mouse: B6
  condition: bl
  cal_run: "001"
```
The `name` key specifies the subfolder in `video-data` that you want to analyze. Since each subfolder in `video-data` may contain recordings for many mice, conditions, and runs, we specify the `mouse` and `condition` keys to choose a specific recording. We also specify the `cal_run` key to choose a specific calibration recording.

!!! note
    You do not need to create a new Anipose configuration for each dataset. You can add as many entries to the `sessions` list as you want, and you can update it even after you have started the analysis.

Now that you have an Anipose configuration, you can create or update the Anipose project by running:
```bash
python run-anipose.py anipose=example-anipose setupproject=true
```
We specify the `anipose=` flag to indicate which configuration file should be used. This will create a new project with the name specified in the configuration file under `anipose-projects`. In our case, there is already an existing project provided in the examples download that has already been created and analyzed.

From here, we can proceed with analysis:
```bash
python run-anipose.py anipose=example-anipose run_data=true
```
If data has already been analyzed, it will be skipped.

## Stage 2: Training a new model

Sometimes, we may not have an existing  model ready to use. In this case, we can train a new DeepLabCut (DLC) model through Cheese3D. Start by placing the videos used for training in `video-data` in their own subfolder. For this tutorial, we already provide a dataset called `20231031_B6-B20-B21-B26-B31-B32-B33_chew_rig2` as an example.

Next, create a new DLC project configuration under `configs/dlc`. We provide `configs/dlc/example-dlc.yaml` as a reference. Let's go over the parts of this template that you should customize.

The `name` key specifies the name of your project:
```yaml
name: example-dlc
```
In this case, we use the name of the example project provided in the examples download.

The `sessions` list provides data from `video-data` to use for training the model:
```yaml
sessions:
# chewing rig 2
- name: 20231031_B6-B20-B21-B26-B31-B32-B33_chew_rig2
  concat_videos: true
```
The `name` key specifies the subfolder in `video-data` to use. Typically, there are many recordings in the same subfolder. Just like in [the Anipose configuration example above](#stage-1-analyzing-new-videos), you can specify keys like `mouse`, `condition`, or `run` to narrow down which data to use. Alternatively, you can specify none of these keys to use all the data, and each unique recording will correspond to a separate video in the DLC project. In this case, we use the `concat_videos: true` option to merge all the recordings into a single video. DLC operates frame by frame, so it does not matter that these videos are discontinuous in time.

Finally, notice at the top of the file that we have some defaults specified:
```yaml
defaults:
- /labutils/dlcconfig@_here_
- common
- _self_
```
The first an last entry are required for Cheese3D to function; however, the middle entry imports a set of common configurations which you can find under `configs/dlc/common.yaml`. You can override any of these settings by specifying the same key in `configs/dlc/<your-project-name>.yaml`.

Once we have a configuration, we can proceed with setting up the project, extracting frames to label, labeling frames, and training the model.

!!! note
    Since we already provided an example DLC project, running the following commands on the example project will modify the model.

Run through the following steps:
1. Create the project with `python run-dlc.py dlc=example-dlc setupproject=true`
2. Extract frames to label with `python run-dlc.py dlc=example-dlc extractframes=true`
3. Label frames with `python run-dlc.py dlc=example-dlc labelframes=true`
4. Build the dataset with `python run-dlc.py dlc=example-dlc builddataset=true`
5. Train the model with `python run-dlc.py dlc=example-dlc trainnetwork=true evaluatenetwork=true`

Now you should have a trained DLC model!

## Stage 3: Refining an existing model

After running Anipose analysis, you may wish to refine the DLC model using your new or existing data.

1. First, extract frames to label using DLC's active learning with `python run-dlc.py dlc=example-dlc extractframes=true` (Cheese3D will detect a trained model and use it to extract frames)
2. Label frames with `python run-dlc.py dlc=example-dlc labelframes=true`
3. Update the dataset with `python run-dlc.py dlc=example-dlc mergedataset=true`
4. Retrain or refine the model:
    1. Retrain the model with `python run-dlc.py dlc=example-dlc trainnetwork=true evaluatenetwork=true`
    2. Refine the model with `python run-dlc.py dlc=example-dlc initweights=true` then `python run-dlc.py dlc=example-dlc trainnetwork=true evaluatenetwork=true`
