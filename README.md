
# Study of Intelligent CCTV Real-time Object Zoom Engine

## Project Overview
This project implements an intelligent CCTV system featuring real-time object detection and zooming, enhancing surveillance precision. It uses an 80-20 train-test data split for model training with the YOLOv11n algorithm. The system connects to CCTV streams using the RTSP protocol and ONVIF PTZ service for dynamic zoom in/out control. A Streamlit-based web interface is provided for video upload and real-time output visualization. Additionally, ESRGAN is applied to enhance image quality.

## Features
- Object detection with YOLOv11n model using 80-20 data split
- Real-time video input from CCTV via RTSP protocol
- PTZ (Pan-Tilt-Zoom) control using ONVIF service for zooming features
- Streamlit web app for uploading videos and viewing live results
- ESRGAN model to improve image resolution and clarity

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/pramanikprapti/Study-of-Intelligent-CCTV-real-time-Object-Zoom-Engine.git
   ```
2. Create and activate a Python virtual environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Train the YOLOv11n model using an 80-20 data split with provided scripts.
2. Connect the system to a CCTV feed via RTSP.
3. Run the Streamlit app for video uploading and output viewing:
   ```
   streamlit run app.py
   ```
4. Use the ONVIF PTZ services integrated for zoom-in and zoom-out control during live feed processing.
5. ESRGAN enhances the output image quality for better visual clarity.

## Code Structure

| Folder/File        | Description                        |
|--------------------|----------------------------------|
| `data/`            | Training and testing datasets     |
| `models/`          | YOLOv11n and ESRGAN models        |
| `app.py`           | Streamlit web application script  |
| `utils/`           | ONVIF PTZ service integration     |
| `requirements.txt` | Python dependencies               |

## License
This project is licensed under the MIT License. See the LICENSE file for details.



