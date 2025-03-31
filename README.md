# Stereo Camera Fault Detection

This project utilizes a ZED stereo camera to detect potential faults by comparing the images collected from the left and right cameras. The detection algorithm is based on evaluating the Mean Absolute Difference (MAD) between the frames, fault rates, and pixel position differences. If the difference exceeds certain thresholds, the system suspects a fault and displays relevant information.

## Features

- **Stereo Camera Integration**: Uses the ZED stereo camera to collect images from both left and right cameras.
- **Fault Detection**: Detects potential faults by comparing the difference between the frames of the left and right cameras.
- **Real-Time Visualization**: Displays the difference map and other visualizations for debugging and monitoring.
- **Fault Detection Stages**: Implements multi-stage detection with configurable thresholds for fault confirmation.


## Prerequisites

Before using this project, make sure to install the following:

- Python 3.10.12
- OpenCV
- NumPy
- ZED SDK (for ZED Camera integration)

You can install OpenCV and NumPy using pip:

```bash
pip install opencv-python numpy
```
To install the ZED SDK, follow the official instructions from StereoLabs ZED SDK.


## Installation

1. Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/stereo-camera-fault-detection.git
cd stereo-camera-fault-detection
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Make sure you have your **ZED camera** set up correctly and connected to your machine.


## Usage

1. Run the Python script to start the fault detection:
```bash
python3 stereo_camera_fault_detection.py
```
2. The script will start processing frames from the ZED stereo camera and display the following:
 - Left and right images in separate windows
 - A Difference Map showing the discrepancies between the two images
 - Console output with fault detection status
3. The program will show the status of the camera:
 - **"Lens OK"**: Indicates no fault detected.
 - **"Camera Fault Detected"**: Indicates potential fault or malfunction.
4. Press **ESC** to stop


## Configuration

You can modify the following parameters in the script for custom detection behavior:

- **`MAD_THRESHOLD`**: The minimum MAD value above which potential fault is suspected.
- **`FAULT_RATE_THRESHOLD`**: The threshold for the fault pixel ratio.
- **`MAX_DIFF_THRESHOLD`**: The maximum allowable difference in pixel intensity.
- **`POSITION_DIFF_THRESHOLD`**: The threshold for position-based fault detection.
- **`RESET_TIME`**: The time duration to confirm if the camera has returned to a normal state after fault detection.

## Example

Here is an example output of the program:

```markdown
--------------------------------------------------
Mean Absolute Difference (MAD): 12.45
Fault Rate: 0.0254
Position Damage Rate: 0.0187
Status: << Camera Fault Detected !!! >>
--------------------------------------------------
```


## How It Works

The algorithm compares frames from the left and right cameras of the ZED stereo camera:

1. **Difference Map**: The difference between the two frames is calculated, and any noise is removed. If the difference is significant, it may indicate a fault.
2. **MAD (Mean Absolute Difference)**: The MAD value is computed for the frames, and a high MAD value could indicate a fault.
3. **Fault Rate**: The proportion of pixels that are considered damaged based on the difference map is computed. If it exceeds a certain threshold, a fault is suspected.
4. **Position Damage Rate**: This is the ratio of pixel differences in positions between the frames. If it exceeds the threshold, a fault is confirmed.
5. **Multi-stage Detection**: The system goes through two stages:
   - **Stage 1**: Initial suspicion of a fault based on MAD and fault rate.
   - **Stage 2**: Confirmation of a fault if the changes in MAD, fault rate, and position damage rate are significant.














