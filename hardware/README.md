<!--
TODO: Fill in the details of the hardware protocol for Cheese3D.
- CONE Add STL file for headpost design | picture
- DONE Add ChaRuCo marker design
- DONE Add description of rig setup
    - table: parts list for Cheese3D (rig2 specifically): see updated yellow tab (https://docs.google.com/spreadsheets/d/13R9PtIZJ4nx72vV8oIzqA-8vUrAW3IE-dqtb1uiIOvg/edit?gid=0#gid=0)
    - supp table 1 (which camera sees what face parts) 
-->
# Cheese3D Hardware Protocol
This README file contains information about:
- The [part list](#cheese3d-rig-part-list) of a Cheese3D rig.
- The [spatio-temporal synchronization](#synchronization) protocol.
- The [head-fixation](#head-fixation) design.
- The [recording protocol](#recording-protocol)

## Cheese3D rig part list

Six high-speed monochrome cameras were used to record the video data at `100 fps`. See the table below for the part list of a Cheese3D camera set-up:

| Part          | Quantity      | Buyer (Part #)        | Notes                             |
|---------------|---------------|-----------------------|-----------------------------------|
| Chameleon®3 Monochrome Camera   | 6 | Edmund Optics (33-162)   | Camera locations should be selected such that each keypoint is visible by at least 2 cameras. |
| Camera Lens (8 mm)   | 4 | Thorlabs (MVL8M23) | For the LEFT, RIGHT, TOP LEFT and TOP RIGHT views. |
| Camera Lens (12 mm)  | 2 | Thorlabs (MVL12M23) | For the TOP CENTER and BOTTOM CENTER views. |
| CS to C-mount Lens Adaptor | 6 | Edmund Optics (03-618) | To mount the lens on the camera. |
| Spacer ring | 6 | Lab-custom 3D-printed | Inspired on Edmund Optics (03-633). Thickness may vary depending on camera location. |
| Blackfly S Tripod Adapter | 6 | Edmund Optics (88-210) | To mount the cameras. |
| Type-A to Micro-B Cable | 6 | Edmund Optics (86-770) | To connect the camera to the computer. |
| IR30 WideAngle IR Illuminator | 3 | Amazon (B001P2E4U4) | To light the rig. |
| Infrared Emitter LED | 1 | Mouser (SML-S15R2TT86) | To temporally synchronize the cameras. |
| Arduino MEGA Rev3 | 2 | Arduino (A000067) | One board is used to send the sync pulse to the infrared LED. The other board is used to send the signal that starts the video-recording. |
| Push button | 1 | Amazon (B01E38OS7K) | To trigger the start of the recording. |
| Breadboard | 1 | Amazon (B00Q9G8MQS) | To connect electronic componenets (e.g., push button)

Placing the cameras at the recommended locations will result in the following keypoints visible by each camera view. See **Supplementary Table 1** from manuscript below:

| Facial keypoint | Left | Right | Top Left | Top Right | Top Center | Bottom Center |
|-----------------|------|-------|----------|-----------|------------|---------------| 
| nose(bottom) | Y | Y | | | | Y |
| nose(tip) | Y | Y | Y | Y | Y | Y |
| nose(top) | Y | Y | Y | Y | Y | Y |
| pad(top)(left) | Y | | Y | | | Y |
| pad(side)(left) | Y | | | | | Y |
| pad(top)(right) | Y | | | Y | | Y |
| pad(side)(right) | Y | | | | | Y |
| pad(center) | Y | Y | | | | Y |
| lowerlip | Y | Y | |  | | Y |
| upperlip(left) | Y | | | | | Y |
| upperlip(right) | | Y | | | | Y |
| eye(front)(left) | Y | | Y | | Y |  |
| eye(top)(left) | Y | | Y | | Y |  |
| eye(back)(left) | Y | | Y | | Y |  |
| eye(bottom)(left) | Y | | Y | | Y |  |
| eye(front)(right) | | Y | | Y | Y |  |
| eye(top)(right) | | Y | | Y | Y |  |
| eye(back)(right) | | Y | | Y | Y |  |
| eye(bottom)(right) | | Y | | Y | Y |  |
| ear(base)(left) | Y | | Y | | |  |
| ear(top)(left) | Y | | Y | | |  |
| ear(tip)(left) | Y | | Y | | |  |
| ear(bottom)(left) | Y | | Y | | |  |
| ear(base)(right) | | Y | | Y | |  |
| ear(top)(right) | | Y | | Y | |  |
| ear(tip)(right) | | Y | | Y | |  |
| ear(bottom)(right) | | Y | | Y | |  | 

## Synchronization

Cameras are temporally synchronized with an infrarred LED that is visible from the six views. The LED (see [table](#cheese3d-rig-part-list)) turns on for `20 ms` every `10 ± 0.5 s`. The pulse is detected on each view post-hoc through our pipeline (refer to the Methods section **Video capture, synchronization, and 3D calibration system** in our manuscript for more information).

Spatial synchronization is achieved with a manufactured calibration board with a standard [ChArUco template](https://github.com/dogod621/OpenCVMarkerPrinter) imprinted on its surface. We used `7 × 7` ChArUco board (`4.5 mm` marker length, `6 mm` square side length, ArUco dictionary DICT_4x4_50). Before and after each experimental recording, the ChaRuCo board is placed inside the rig by the experimenter, while rotating over the camera views and making sure it stays on focus.

## Head-fixation

To acquire high-resolution facial video while maintaining comfort for natural behavior, mice are acclimated to sitting in a tunnel with the head secured using a lightweight headpost, custom-designed to allow unobstructed viewing of all facial areas.

See below the [headpost](final_headpost_design.stl) and [tunnel](mouse_tunnel_tube_mark21_v2.stl) models (note the parts are not to scale, we recommend referring to the manuscript or the .STL files in this folder for the true measurements).

![](headpost-and-tunnel.png)

## Recording protocol

The following code was optimized to run on Windows and the following programs should be installed:
- [Arduino IDE](https://www.arduino.cc/en/software) (version 2.1.0)
- [Bonsai](https://bonsai-rx.org/) (version 2.8.1)
- [Spinview](https://www.teledynevisionsolutions.com/products/spinnaker-sdk/?model=Spinnaker%20SDK&vertical=machine%20vision&segment=iis) (version 1.29.0.5)
- [FFMPEG](https://ffmpeg.org/download.html) (version 6.0)

### Organization

- [`run-behavior.ps1`](run-behavior.ps1): The main script used to start recording
- [`20230928_Bonsai_Behavior_6cam_hd.bonsai`](Bonsai-config/20230928_Bonsai_Behavior_6cam_hd.bonsai): Bonsai program that records from six cameras to Windows pipes and outputs serial port metadata from Arduino to `stdout`. Note you will need to change the serial number associated to the cameras and Arduino nodes.
- [`ffmpeg_filter.txt`](ffmpeg_filter.txt): An FFMPEG "complex filter graph" specification used by `run-behavior.ps1` to post-process pipe output from Bonsai to video file
- [`manual_recording.ino`](Arduino/manual_recording/manual_recording.ino): This code should be uploaded to the Arduino that will trigger the start of the recording. Note this board should be connected to the computer.
- [`simple_LED_blink.ino`](Arduino/simple_LED_blink/simple_LED_blink.ino): This code should be uploaded to the Arduino that will send the pulse to the synchronizing LED. Note this board does not need to be connected to the computer, but to a power source.

### Preparation

Open a Powershell prompt by going to the "Start Menu > Terminal". Change directories to the location of this folder (typically `C:\\Users\...\hardware`).

First, make sure the Arduino script `manual_recording` is uploaded, reset and running. Perform any actions to trigger recording on the Arduino. The Arduino script should be configured to send a "start" message on across the serial port after some delay or when a button is pressed.

### Run the behavior script:
```
.\run-behavior.ps1 -EXP "chewing" -EXPERIMENTER "you" -SESSION="mice" -MOUSE "1" -COND "baseline"
```
This will run the Bonsai program, which should start recording as soon as the "start" message is delivered over the serial port. It will also start FFMPEG after a 4 second delay (to ensure that the Bonsai program has finised creating the Windows pipes). As recording progresses, FFMPEG will print out the length of the video file written to disk and the speed of writing to disk (which should be around 0.9-1.0x).

To stop recording, go to the bottom right corner of the taskbar in Windows (where volume, etc. is located). Find the Bonsai logo and right-click, then select "Stop". This will close the currently running Bonsai process and FFMPEG will stop printing in the Terminal window. The script should finish by itself. To ensure that no data is lost, add some buffer time to the end of recording before stopping Bonsai.

### Behavior script flags

There are a number of flags/parameters passed to the behavior script.
- `MOUSE` (required): the name of the mouse currently on the rig
- `EXP` (required): the name of the experiment
- `COND` (required): the name of the condition (sub-variant) of the experiment
- `EXPERIMENTER` (required): the name of the person recording
- `SESSION`: the identifier all the mice in the session (defaults to `MOUSE`)
- `ID`: a unique identifier to separate repetitions of the same condition and
    mouse (defaults to "000"; _do not use this to repeat a failed trial_)
- `GRID`: set to 1 to upload the experiment folder to the grid on completion
    (defaults to 0)
- `RIG`: the rig identifier (defaults to "2")
- `NCAMS`: the number of cameras on the rig (defaults to 6)
