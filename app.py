
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import google.generativeai as genai
import os

# Safe vector angle calculator (2D Projection)
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

# Page Layout configuration
st.set_page_config(page_title="CricAnalytics AI - Pro Engine", layout="wide", initial_sidebar_state="expanded")
st.title("🏏 CricAnalytics AI: Universal Movement Performance Engine")
st.markdown("Advanced computer vision and generative biomechanical analysis scaling across all cricket motions.")

# Sidebar System Configuration
st.sidebar.header("⚙️ Core Configuration")
user_api_key = st.sidebar.text_input("Gemini API Key", type="password")
analysis_mode = st.sidebar.selectbox("Select Analysis Discipline", [
    "Batting Biomechanics & Shot Mechanics",
    "Bowling Action, Release & Stride Dynamics",
    "Fielding, Catching, Throwing & Agility"
])
uploaded_file = st.sidebar.file_uploader("Upload Session Video Clip", type=['mp4', 'mov', 'avi', 'mkv'])

if uploaded_file and user_api_key:
    genai.configure(api_key=user_api_key)
    
    # Clean staging setup for processing
    input_path = "temp_input.mp4"
    output_path = "telemetry_output.mp4"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
        
    if st.sidebar.button("🚀 Execute Biomechanical Pipeline"):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Guard against zero/broken video values
        if total_frames <= 0 or fps <= 0:
            st.error("❌ Invalid or corrupted video file format. Try uploading a different clip.")
            st.stop()
            
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # Init Tracking Architecture
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        
        # Full-body universal telemetry arrays
        telemetry_data = {
            "left_elbow": [], "right_elbow": [],
            "left_shoulder": [], "right_shoulder": [],
            "left_knee": [], "right_knee": [],
            "left_hip": [], "right_hip": [],
            "center_mass_displacement": [] # Tracking global movement velocity
        }
        
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
                
                # Extract positional mappings
                def get_pt(idx): return [landmarks[idx].x, landmarks[idx].y]
                
                # Joint extraction mappings (MediaPipe indices)
                p = {
                    "ls": get_pt(11), "le": get_pt(13), "lw": get_pt(15),
                    "rs": get_pt(12), "re": get_pt(14), "rw": get_pt(16),
                    "lh": get_pt(23), "lk": get_pt(25), "la": get_pt(27),
                    "rh": get_pt(24), "rk": get_pt(26), "ra": get_pt(28)
                }
                
                # Calculate all structural kinematics
                ang = {
                    "le": calculate_angle(p["ls"], p["le"], p["lw"]),
                    "re": calculate_angle(p["rs"], p["re"], p["rw"]),
                    "ls": calculate_angle(p["lh"], p["ls"], p["le"]),
                    "rs": calculate_angle(p["rh"], p["rs"], p["re"]),
                    "lk": calculate_angle(p["lh"], p["lk"], p["la"]),
                    "rk": calculate_angle(p["rh"], p["rk"], p["ra"]),
                    "lh": calculate_angle(p["ls"], p["lh"], p["lk"]),
                    "rh": calculate_angle(p["rs"], p["rh"], p["rk"])
                }
                
                # Append to datasets
                telemetry_data["left_elbow"].append(ang["le"])
                telemetry_data["right_elbow"].append(ang["re"])
                telemetry_data["left_shoulder"].append(ang["ls"])
                telemetry_data["right_shoulder"].append(ang["rs"])
                telemetry_data["left_knee"].append(ang["lk"])
                telemetry_data["right_knee"].append(ang["rk"])
                telemetry_data["left_hip"].append(ang["lh"])
                telemetry_data["right_hip"].append(ang["rh"])
                
                # Track position of midpoint hip to compute translational velocity vectors
                mid_hip_x = (p["lh"][0] + p["rh"][0]) / 2.0
                telemetry_data["center_mass_displacement"].append(float(mid_hip_x))
                
                # Render complete overlay graphics
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Draw minimal key contextual telemetry text onto screen graphics
                cv2.putText(frame, f"L_Elbow: {ang['le']}deg", (int(p['le'][0]*width)+10, int(p['le'][1]*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.putText(frame, f"R_Elbow: {ang['re']}deg", (int(p['re'][0]*width)+10, int(p['re'][1]*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.putText(frame, f"L_Knee: {ang['lk']}deg", (int(p['lk'][0]*width)+10, int(p['lk'][1]*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
                cv2.putText(frame, f"R_Knee: {ang['rk']}deg", (int(p['rk'][0]*width)+10, int(p['rk'][1]*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

            out.write(frame)
            progress_bar.progress(int((frame_idx / total_frames) * 100))
            status_text.text(f"Processing motion telemetry frame {frame_idx}/{total_frames}...")

        cap.release()
        out.release()
        pose.close()
        
        # Robust evaluation check to completely prevent ValueError on empty datasets
        if not telemetry_data["left_elbow"]:
            status_text.empty()
            progress_bar.empty()
            st.error("❌ **Kinematic Tracking Error:** The software could not reliably isolate human skeletal metrics from this video. Ensure background clarity and clear full-body visibility.")
        else:
            status_text.text("🤖 Constructing Unified Kinematic Payload...")
            
            # Formulate structured summary dataset metrics mapping max, min, ranges
            def compile_stats(arr):
                return {"min": int(np.min(arr)), "max": int(np.max(arr)), "range": int(np.max(arr) - np.min(arr))}
                
            packaged_payload = {
                "discipline_context": analysis_mode,
                "kinematic_metrics": {
                    "upper_body_joints": {
                        "left_elbow": compile_stats(telemetry_data["left_elbow"]),
                        "right_elbow": compile_stats(telemetry_data["right_elbow"]),
                        "left_shoulder": compile_stats(telemetry_data["left_shoulder"]),
                        "right_shoulder": compile_stats(telemetry_data["right_shoulder"])
                    },
                    "lower_body_joints": {
                        "left_knee": compile_stats(telemetry_data["left_knee"]),
                        "right_knee": compile_stats(telemetry_data["right_knee"]),
                        "left_hip": compile_stats(telemetry_data["left_hip"]),
                        "right_hip": compile_stats(telemetry_data["right_hip"])
                    }
                },
                "locomotive_dynamics": {
                    "total_frames_processed": frame_idx,
                    "horizontal_displacement_range": float(np.max(telemetry_data["center_mass_displacement"]) - np.min(telemetry_data["center_mass_displacement"]))
                }
            }
            
            # Orchestrate specialized targeted system prompt logic based on disciplinary track
            system_instruction = f"""
            You are a world-class high-performance cricket sports scientist and national team bio-mechanics coach. 
            You are analyzing the kinematic tracking output of a movement sequence categorized under the discipline: "{analysis_mode}".

            Review the raw physical data metrics compiled below:
            {json.dumps(packaged_payload, indent=2)}

            Provide a comprehensive, elite coaching report. Your analysis must adapt to whatever action is in the video based on the data:
            1. If it's batting, interpret what the data indicates about their posture, backlift, stance, or extension.
            2. If it's bowling, evaluate what the joint ranges mean for their loading, alignment, and follow-through.
            3. If it's fielding, interpret what the displacement and flexibility angles suggest about their speed, throwing mechanics, or core agility.

            Structure your report cleanly with sections for: Technical Biomechanical Evaluation, Core Vulnerabilities/Strengths Identified, and 1 Specialized Elite Training Drill. 
            Maintain professional, actionable sports terminology. Do not output any markdown code blocks, system strings, or raw script syntax.
            """
            
            status_text.text("🧠 Deploying Generative Large Language Model Evaluation Layer...")
            try:
                model = genai.GenerativeModel('gemini-3.5-flash')
                response = model.generate_content(system_instruction)
                
                status_text.empty()
                progress_bar.empty()
                st.success("✅ Analysis Successfully Concluded!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 CV Overlaid Telemetry Output")
                    st.video(output_path)
                with col2:
                    st.subheader("📋 Professional Biomechanical Analysis")
                    st.write(response.text)
                    
            except Exception as e:
                st.error(f"❌ Verification Layer Error: Unable to extract data from Gemini API. Details: {e}")
else:
    st.info("💡 Configuration Status: Open the sidebar, insert your Gemini API Key, select your target cricket discipline, and upload your movement video to initialize processing.")
