# RealTime-Color-Detection
A Python-based real-time color detection project using OpenCV and MediaPipe that identifies the color of an object held between the thumb and index finger."
# Real-Time Color Detection Using OpenCV and MediaPipe

This project uses Python, OpenCV, and MediaPipe to detect the color of an object held between your **thumb and index finger** in real time via webcam.

## 🔍 Features

- Detects objects held between thumb and index finger.
- Determines the closest color name using Euclidean distance in RGB space.
- Displays the detected color name, a color preview, and confidence score.
- Real-time webcam interface.

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/BiswasApurbo/RealTime-Color-Detection.git
cd Realme-Color-Detection
pip install opencv-python mediapipe numpy
python color.py

📸 Output Example
A webcam window will show:

A box displaying the dominant color.

The name of the color with a confidence percentage.

ROI rectangle between your thumb and index finger.

🧠 How It Works
Detects hand landmarks using MediaPipe.

Finds the midpoint between thumb tip and index fingertip.

Averages the pixel colors in a region around the midpoint.

Calculates the closest named color based on RGB distance.

📃 License
This project is open-source and available under the MIT License.



