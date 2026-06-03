# Hand Tracking
Python wrapper around Google Mediapipe's [Hand Landmark Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) that organizes raw landmark data into usable hand objects with per-finger access, angle calculation, and raised finger detection.
 
## Install
 
``` bash
pip install mediapipe opencv-contrib-python numpy
git clone https://github.com/DwnNyxDev/hand_tracking.git
```

The `hand_landmarker.task` model file is bundled — no separate download needed.

## Usage

Example usage of this package is provided within `example.py` which can be run with:

``` bash
python example.py
```

## Used In
 
- [hand_controller](https://github.com/hcr-vvatel/hand_controller) — ROS 2 package that wraps this library into a node for robot control pipelines
