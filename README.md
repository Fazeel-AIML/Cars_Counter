# Vehicle Counter using YOLO and OpenCV

This repository contains a project for counting vehicles in a video using YOLO (You Only Look Once) object detection model and OpenCV. The project overlays a graphical image on the video and tracks vehicles as they pass through a designated line.

## Project Overview

The goal of this project is to detect and count vehicles (cars, trucks, buses) in a video using YOLOv8. The system processes video frames to detect vehicles, tracks their movement, and counts the number of vehicles that cross a specific line.

## Features

- **Vehicle Detection**: Uses YOLOv8 for detecting vehicles in video frames.
- **Vehicle Tracking**: Tracks the detected vehicles across frames and maintains a count.
- **Overlay Graphics**: Displays an overlay image and a mask on the video.
- **Real-time Visualization**: Shows the live video feed with detected and counted vehicles.

## Requirements

- `opencv-python`
- `cvzone`
- `ultralytics`
- `numpy`
- `sort`

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Fazeel-AIML/Cars_Counter.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd Cars_Counter
    ```

3. **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

4. **Install the required packages**:
    ```bash
    pip install opencv-python cvzone ultralytics numpy sort
    ```

## Usage

1. **Set up YOLO Model**:
   - Ensure you have the YOLOv8 model file (`yolov8n.pt`). You can download it from [Ultralytics YOLO Model Zoo](https://github.com/ultralytics/yolov5/releases).

2. **Prepare the Input Video and Images**:
   - Place your video file (e.g., `cars.mp4`), overlay image (e.g., `graphics.png`), and mask image (e.g., `mask-950x480.png`) in the `Things` directory.

3. **Run the Script**:
   - Execute the script to start the vehicle counting process:
     ```bash
     python vehicle_counter.py
     ```

4. **View the Results**:
   - The script will open a window displaying the video with vehicle counts. Press `q` to exit the window.

## Script Details

- **Vehicle Detection**: The script uses YOLOv8 to detect vehicles in each frame of the video.
- **Overlay Graphics**: An overlay image is applied to the video frames to enhance visual information.
- **Vehicle Tracking**: The `Sort` algorithm is used for tracking the detected vehicles and counting them as they cross a specified line.
- **Visualization**: Results are displayed in real-time with vehicle IDs and count updates.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Demo

![Demo](path/to/your/demo.gif)  # Add your GIF file path here
