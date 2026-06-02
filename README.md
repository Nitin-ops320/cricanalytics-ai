# cricanalytics-ai
An AI-powered cricket video analysis tool using Computer Vision (MediaPipe/YOLOv8) and LLMs to provide automated biomechanical coaching feedback.
# CricAnalytics AI 🏏
An end-to-end production Computer Vision & Generative AI pipeline designed to automate biomechanical tracking and technical analysis for cricket coaching.

## 🚀 Architectural Blueprint
- **Inference Layer:** Leverages Custom-tuned MediaPipe Pose Estimation to track 33 kinematic keypoints across individual human bodies frame-by-frame.
- **Biomechanical Logic Engine:** Computes real-time vector angular mechanics (trigonometric dot-product arrays) of the lead arm swing and knee flexion to capture technical integrity.
- **Telemetry System:** Overlays live data visualizations onto video assets via OpenCV streams.
- **AI Automation Layer:** Distills unstructured visual data into a strict JSON payload, mapping anomalies directly to an orchestrated Google Gemini LLM workflow to generate actionable training drills.

## 💻 Technical Stack
- **Languages:** Python (3.10+)
- **Computer Vision:** OpenCV, MediaPipe
- **Mathematics:** NumPy
- **Generative AI:** Google GenAI SDK (Gemini Core Ensembles)
- **Deployment & UI:** Streamlit

## 📦 Local Installation & Setup
1. Clone this repository to your machine:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/cricanalytics-ai.git](https://github.com/YOUR_USERNAME/cricanalytics-ai.git)
   cd cricanalytics-ai
   pip install opencv-python mediapipe==0.10.21 numpy google-generativeai streamlit
   streamlit run app.py
