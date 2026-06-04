import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import google.generativeai as genai
import tensorflow as tf
import os

# Page Configuration Setup
st.set_page_config(page_title="CricAnalytics AI - Deep Learning", layout="wide")
st.title("🏏 CricAnalytics AI: Sequence-Based Deep Learning Analytics")

user_api_key = st.sidebar.text_input("Gemini API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload Cricket Clip", type=['mp4', 'mov', 'avi'])

# Check for our Deep Learning Model file in GitHub environment
dl_model_path = 'cricket_dl_model.keras'
if not os.path.exists(dl_model_path):
    st.error(f"❌ Core Component Missing: Ensure '{dl_model_path}' is committed into your GitHub repo.")
    st.stop()

# Load the compiled Deep Learning model
dl_network = tf.keras.models.load_model(dl_model_path)

if uploaded_file and user_api_key:
    genai.configure(api_key=user_api_key)
    
    input_path = "temp_input.mp4"
    output_path = "telemetry_output.mp4"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
        
    if st.sidebar.button("🚀 Execute Sequential Diagnostics"):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
        mp_drawing = mp.solutions.drawing_utils
        
        # This frame window list will build our Deep Learning sequence matrix over time
        frame_window_sequence = []
        full_video_telemetry = [] # Track structural metrics to send to Gemini
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def calculate_angle(a, b, c):
            a, b, c = np.array(a), np.array(b), np.array(c)
            ba, bc = a - b, c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                def get_pt(idx): return [landmarks[idx].x, landmarks[idx].y]
                
                ls, le, lw = get_pt(11), get_pt(13), get_pt(15)
                rs, re, rw = get_pt(12), get_pt(14), get_pt(16)
                lh, lk, la = get_pt(23), get_pt(25), get_pt(27)
                rh, rk, ra = get_pt(24), get_pt(26), get_pt(28)
                
                le_ang = calculate_angle(ls, le, lw)
                re_ang = calculate_angle(rs, re, rw)
                lk_ang = calculate_angle(lh, lk, la)
                rk_ang = calculate_angle(rh, rk, ra)
                
                # Single frame feature vector matching the model's exact shape requirements
                current_frame_features = [le_ang, re_ang, lk_ang, rk_ang, lw[1], ls[1]]
                frame_window_sequence.append(current_frame_features)
                
                # Maintain the sliding window buffer size matching SEQUENCE_LENGTH=30
                if len(frame_window_sequence) > 30:
                    frame_window_sequence.pop(0)
                    
                full_video_telemetry.append(current_frame_features)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
            out.write(frame)
            progress_bar.progress(int((frame_idx / total_frames) * 100))
            status_text.text(f"Extracting frame timeline matrix: {frame_idx}/{total_frames}...")

        cap.release()
        out.release()
        pose.close()
        
        if len(frame_window_sequence) < 30:
            st.error("❌ Video duration too brief. Could not populate a full 30-frame timeline vector window.")
        else:
            status_text.text("🧠 Deploying Sequence-Window to Deep Learning Model Layer...")
            
            # Reconstruct buffer list into a 3D Tensor dimension block: (1 sample, 30 timesteps, 6 features)
            input_tensor = np.array([frame_window_sequence])
            
            # Run inference through our Deep Learning Network
            predictions = dl_network.predict(input_tensor)[0]
            class_idx = np.argmax(predictions)
            
            classes_map = {0: "Grounded Cover Drive", 1: "Lofted Power Hit (Six)", 2: "Fast Bowling Action"}
            predicted_action_label = classes_map.get(class_idx, "Unknown Shot Type")
            
            status_text.text("✍️ Generating Movement-Based Problem Scorecard...")
            
            # Map analytical trends over time to guide the AI context
            all_metrics = np.array(full_video_telemetry)
            payload = {
                "deep_learning_verified_shot": predicted_action_label,
                "confidence_score": float(np.max(predictions)),
                "movement_time_series_summary": {
                    "minimum_front_knee_flexion": int(np.min(all_metrics[:, 2])),
                    "maximum_arm_extension_reach": int(np.max(all_metrics[:, 1])),
                    "net_hand_vertical_lift": float(np.min(all_metrics[:, 4]) - np.mean(all_metrics[:, 5])) # Lower means higher reach
                }
            }
            
            # Core movement-based prompt framing
            targeted_prompt = f"""
            You are a senior computer vision engineer and world-class high-performance cricket coach.
            Review this telemetry analysis payload generated from our Deep Learning system:
            {json.dumps(payload, indent=2)}

            Our custom LSTM Network verified with high confidence that the shot being played is a: "{predicted_action_label}".
            
            Deliver a movement-based problem analysis scorecard using simple words and short bullet points. 
            Do NOT output code-based variables, write a code analysis, or include long text blocks.

            Structure your reply using these exact bold Markdown headers:

            🎯 **SHOT PLAYED BY PLAYER**
            {predicted_action_label} (Verified via Sequence Analysis)

            ⚠️ **THE ACTUAL MOVEMENT PROBLEM**
            - [Based on the shot and the joint numbers, explain the exact biological movement error. For example, if it's a cover drive and knee angle is above 140, they didn't bend their front knee forward enough to commit weight.]

            ⚙️ **BODY TIMELINE MOVEMENT EVALUATION**
            - [Explain simply what went wrong with their arm extension or hand heights during the timeline follow-through]

            🛠️ **MOVEMENT FIXING DRILL**
            - [Provide exactly one short 2-sentence physical on-field drill to fix this specific tactical movement mistake]
            """
            
            try:
                model = genai.GenerativeModel('gemini-3.5-flash')
                response = model.generate_content(targeted_prompt)
                
                status_text.empty()
                progress_bar.empty()
                st.success("✅ Architecture Executed Successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Pose Tracked Clip")
                    st.video(output_path)
                with col2:
                    st.subheader("📋 Targeted Biomechanical Scorecard")
                    st.markdown(response.text)
                    
            except Exception as e:
                st.error(f"❌ AI Core Exception Error: {e}")
