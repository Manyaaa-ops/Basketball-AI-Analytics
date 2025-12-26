# Basketball-AI-Analytics
This project implements a real-time computer vision pipeline designed to analyze basketball shooting mechanics. By combining deep learning-based object detection with classical physics modeling, the system tracks the ball's flight path and predicts the likelihood of a successful shot before the ball reaches the hoop.


## Features
- **Object Detection**: Uses YOLOv8 to track the ball.
- **Predictive Modeling**: Uses 2nd-degree polynomial regression to forecast ball trajectory.
- **Dynamic Feedback**: Real-time probability bar based on spatial intersection with the hoop zone.

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your video as `test_video.mp4`
4. Run: `python ball_tracker.py`
