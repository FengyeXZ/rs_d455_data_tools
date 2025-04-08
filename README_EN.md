📘 This README is also available in [中文](README.md).

# RS D455 Data Collection Tools

This repository contains data collection and processing tools for the Intel RealSense D455 depth camera. These tools help you synchronously capture RGB images, depth images, and dual infrared stereo images, while providing depth image post-processing and alignment capabilities.

The program uses fully asynchronous methods to process data streams to maximize data collection quality. Therefore, this program has certain requirements for host computer real-time processing performance and camera connection bandwidth.

Please note that this program is currently experimental and cannot guarantee the stability of the acquisition process. If crashes occur, they may be triggered by the underlying C++ interface. Multiple attempts to run may be needed.

## Main Features

- **Multi-modal Data Collection**: Synchronously capture RGB, depth, and dual infrared images
- **Depth Image Post-processing**: Includes depth filtering and noise reduction
- **Depth Image Alignment**: Align depth images to RGB image coordinate system
- **Real-time Visualization**: Real-time visualization of data streams
- **Data Storage**: Save captured data as formatted datasets
- **Camera Parameter Management**: Save camera intrinsic and extrinsic parameters

## Installation

### Dependencies

```
numba
numpy
opencv_python
pyrealsense2
tqdm
```

### Installation Steps

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/rs_d455_data_tools.git
   cd rs_d455_data_tools
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Collection Tool

```
python d455_data_collection_tools/d455_collector.py
```

The collection tool creates a directory named `rs_d455_data` (which can be changed by modifying the `OUTPUT_DIR` variable in the script), containing the following subdirectories:
- `rgb/`: Stores RGB images
- `depth/`: Stores depth images
- `left/`: Stores left infrared images
- `right/`: Stores right infrared images
- `depth_aligned/`: Stores depth images aligned to RGB (if `ALIGN_DEPTH` is enabled)

Additionally, it generates the following files:
- `rgb.txt`: List of RGB image files and timestamps
- `depth.txt`: List of depth image files and timestamps
- `associations.txt`: Association information for RGB and depth images
- `times.txt`: Timestamps for stereo images
- `RS_D455.yaml`: Camera parameter configuration file
- `calib.txt`: Camera calibration information (KITTI format)

### Depth Image Alignment Tool

```
python d455_data_collection_tools/depth_align.py --depth-dir INPUT_DEPTH_DIR --output-dir OUTPUT_ALIGNED_DIR [--use-device]
```

Parameter description:
- `--depth-dir`: Input depth image directory
- `--output-dir`: Output aligned depth image directory
- `--depth-scale`: Depth scale factor, default is 0.001 (millimeters to meters)
- `--use-device`: Use connected RealSense device to get camera parameters
- `--rgb-width`: RGB image width, default is 848
- `--rgb-height`: RGB image height, default is 480

## Configuration Options

In `d455_collector.py`, you can adjust the following parameters:

```python
FPS_DEPTH       = 90                # Depth camera frame rate
FPS_RGB         = 30                # Color camera frame rate
WIDTH           = 848               # Image width
HEIGHT          = 480               # Image height
LASER_POWER     = 100               # Laser power 0% - 100%
POST_DEPTH      = True              # Whether to post-process depth
ALIGN_DEPTH     = True              # Whether to align depth to color
OUTPUT_DIR      = "rs_d455_data"    # Output directory
CLEAN_LAST      = True              # Whether to clean last output directory
RECORD          = True              # Whether to save data
VISUALIZE       = True              # Whether to visualize
PRESET          = 3                 # Preset in high accuracy mode
```

## Notes

1. Ensure that the RealSense D455 camera is properly connected to your computer
2. The camera requires sufficient USB bandwidth; it is recommended to use USB 3.0 or higher interfaces
3. Avoid drastic camera movements during capture to obtain better data quality
4. Depth post-processing may increase CPU load; if your computer has limited performance, consider disabling this feature

## Author

**Shang Xu**

- Project creator and main maintainer
- If you use this code in your project, please credit the original author
- Feel free to submit issues for feedback or feature suggestions
- For further technical support or collaboration, please contact the author through GitHub

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.