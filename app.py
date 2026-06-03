import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import google.generativeai as genai
import os

# 1. Helper function for calculation
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

# Streamlit User Interface Setup
st.set_page_config(page_title="CricAnalytics AI", layout="wide")
st.title("🏏 CricAnalytics AI: Biomechanical Coaching Assistant")
st.markdown("An end-to-end Computer Vision & Generative AI pipeline tracking cricket performance metrics.")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration Panel")
user_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload Cricket Session Video", type=['mp4', 'mov', 'avi'])

if uploaded_file and user_api_key:
    # Configure GenAI
    genai.configure(api_key=user_api_key)
    
    # Save uploaded video to local temp file
    input_path = "temp_input.mp4"
    output_path = "telemetry_output.mp4"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
        
    if st.sidebar.button("🚀 Run Biomechanical Pipeline"):
        # Setup Video Stream & MediaPipe
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
        mp_drawing = mp.solutions.drawing_utils
        
        elbow_angles, knee_angles = [], []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extract Left side landmarks
                l_shoulder = [landmarks[11].x, landmarks[11].y]
                l_elbow    = [landmarks[13].x, landmarks[13].y]
                l_wrist    = [landmarks[15].x, landmarks[15].y]
                l_hip      = [landmarks[23].x, landmarks[23].y]
                l_knee     = [landmarks[25].x, landmarks[25].y]
                l_ankle    = [landmarks[27].x, landmarks[27].y]
                
                # Process Angles
                e_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                k_angle = calculate_angle(l_hip, l_knee, l_ankle)
                elbow_angles.append(e_angle)
                knee_angles.append(k_angle)
                
                # Overlays
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, f"{e_angle} Deg", (int(l_elbow[0]*width)+15, int(l_elbow[1]*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"{k_angle} Deg", (int(l_knee[0]*width)+15, int(l_knee[1]*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
          
            out.write(frame)
            progress_bar.progress(int((frame_idx / total_frames) * 100))
            status_text.text(f"Processing frame {frame_idx}/{total_frames}...")

        cap.release()
        out.release()
        pose.close()
        
        # Build Summary JSON
         cap.release()
        out.release()
        pose.close()
        
        # FIX: Check if we actually collected any data before running min()
        if not elbow_angles or not knee_angles:
            status_text.empty()
            progress_bar.empty()
            st.error("❌ **Biomechanical Tracking Failed:** MediaPipe could not detect a distinct human pose in this video. Please ensure the player's full body is visible, well-lit, and facing the camera sidebar profile.")
        else:
            # Build Summary JSON safely now that lists are verified
            coaching_flags = []
            min_elbow, min_knee = min(elbow_angles), min(knee_angles)
            
            if min_elbow < 110: 
                coaching_flags.append("Collapsed Front Elbow: Loss of control/power.")
            else: 
                coaching_flags.append("Good High Elbow stable position.")
                
            if min_knee > 145: 
                coaching_flags.append("Stiff Front Leg: Insufficient weight transfer.")
            else: 
                coaching_flags.append("Solid Front-Foot Stride.")
            
            session_json = {
                "metrics_summary": {"elbow_min": min_elbow, "knee_min": min_knee},
                "technical_coaching_insights": coaching_flags
            }
            
            # Generate LLM Feedback
            status_text.text("🤖 Generating Professional AI Coaching Analysis...")
            prompt = f"You are an elite cricket coach. Review this tracking data and write an action-oriented coaching report with 1 technical breakdown and 1 specific drill:\n{json.dumps(session_json)}"
            model = genai.GenerativeModel('gemini-3.5-flash')
            response = model.generate_content(prompt)
            
            # Render Results to dashboard splits
            status_text.text("✅ Analysis Complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Tracked Telemetry Video")
                st.video(output_path)
            with col2:
                st.subheader("📋 Official AI Coaching Report")
                st.write(response.text)
        # Generate LLM Feedback
        status_text.text("🤖 Generating Professional AI Coaching Analysis...")
        prompt = f"You are an elite cricket coach. Review this tracking data and write an action-oriented coaching report with 1 technical breakdown and 1 specific drill:\n{json.dumps(session_json)}"
        model = genai.GenerativeModel('gemini-3.5-flash')
        response = model.generate_content(prompt)
        
        # Render Results to dashboard splits
        status_text.text("✅ Analysis Complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Tracked Telemetry Video")
            st.video(output_path)
        with col2:
            st.subheader("📋 Official AI Coaching Report")
            st.write(response.text)
else:
    st.info("💡 Please input your Gemini API Key and upload an operational cricket video clip in the sidebar to begin processing.")
