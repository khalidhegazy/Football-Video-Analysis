Football Match Analytics Using Object Detection and Tracking
Project Overview
This project analyzes football match videos to extract valuable insights such as player tracking, possession statistics, and goal detection. Using advanced computer vision techniques, the system processes video footage to identify players, track their movements, detect ball position, and determine key events like goals and possession changes.

Features
Player detection and tracking using YOLOv3

Team separation based on jersey color

Ball detection and tracking

Field localization via homography

Possession calculation for each team

Goal event detection and scoring team identification

Output of match statistics and visualizations

Tools and Technologies
YOLOv3: Object detection framework used for detecting players and the ball.

OpenCV: Image processing and computer vision tasks including video processing, homography, and tracking.

Python: Main programming language for implementing the solution.

NumPy & Pandas: Data manipulation and analysis.

Matplotlib / Seaborn: Visualization of results and statistics.

How It Works
Video Input: The system accepts raw football match videos.

Object Detection: Players and ball are detected in each frame using YOLOv3.

Team Classification: Players are grouped by team based on jersey colors.

Tracking: Player and ball positions are tracked across frames.

Field Localization: A homography transformation maps the video frame to a bird’s-eye view of the field.

Event Detection:

Possession is calculated by determining which team has control of the ball over time.

Goal events are detected by tracking the ball crossing the goal line.

Output: Statistical summaries and visualizations are generated to analyze the match.

Possession Calculation
Possession time for each team is computed based on the team controlling the ball in consecutive frames. The control is assigned by proximity of the ball to players of a given team.

Goal Detection
Goals are detected when the ball crosses the goal line coordinates determined by homography. The scoring team is identified based on the attacking direction and the last player in possession.

Usage
Clone the repository.

Install the required dependencies listed in requirements.txt.

Prepare your match video file.

Run the main script:

bash
Copy
Edit
python main.py --video path_to_video.mp4
View the output statistics and visualizations.

Future Work
Improve accuracy with advanced player re-identification models.

Integrate real-time event detection and live streaming capabilities.

Add support for multi-camera video analysis.
