
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import google.generativeai as genai
import os

# Safe vector angle calculator
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

# Streamlit Page Design
st.set_page_config(page_title="CricAnalytics AI", layout="wide")
st.title("🏏 CricAnalytics AI: Universal Cricket Analyzer")
st.markdown("Upload any cricket movement (Batting, Bowling, or Fielding) for an instant biomechanical check.")

# Sidebar Controls
st.sidebar.header("📋 Setup Panel")
user_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
analysis_mode = st.sidebar.selectbox("Select Discipline", [
    "Batting Performance",
    "Bowling Action",
    "Fielding & Agility"
])
uploaded_file = st.sidebar.file_uploader("Upload Cricket Clip", type=['mp4', 'mov', 'avi', 'mkv'])

if uploaded_file and user_api_key:
    genai.configure(api_key=user_api_key)
    
    # Define file paths
    input_path = "temp_input.mp4"
    raw_output_path = "raw_output.mp4"
    final_web_path = "telemetry_output.mp4"
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
        
    if st.sidebar.button("🚀 Analyze Movement"):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0 or fps <= 0:
            st.error("❌ Unreadable video file. Please try another clip.")
            st.stop()
            
        # Write to a temporary raw file first
        out = cv2.VideoWriter(raw_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
        mp_drawing = mp.solutions.drawing_utils
        
        telemetry_data = {
            "left_elbow": [], "right_elbow": [],
            "left_knee": [], "right_knee": [],
            "left_hip": [], "right_hip": [],
            "movement_speed": []
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
                
                # Get points
                l_shoulder = [landmarks[11].x, landmarks[11].y]
                l_elbow    = [landmarks[13].x, landmarks[13].y]
                l_wrist    = [landmarks[15].x, landmarks[15].y]
                l_hip      = [landmarks[23].x, landmarks[23].y]
                l_knee     = [landmarks[25].x, landmarks[25].y]
                l_ankle    = [landmarks[27].x, landmarks[27].y]
                
                r_shoulder = [landmarks[12].x, landmarks[12].y]
                r_elbow    = [landmarks[14].x, landmarks[14].y]
                r_wrist    = [landmarks[16].x, landmarks[16].y]
                r_hip      = [landmarks[24].x, landmarks[24].y]
                r_knee     = [landmarks[26].x, landmarks[26].y]
                r_ankle    = [landmarks[28].x, landmarks[28].y]
                
                # Calculate angles
                le = calculate_angle(l_shoulder, l_elbow, l_wrist)
                re = calculate_angle(r_shoulder, r_elbow, r_wrist)
                lk = calculate_angle(l_hip, l_knee, l_ankle)
                rk = calculate_angle(r_hip, r_knee, r_ankle)
                lh = calculate_angle(l_shoulder, l_hip, l_knee)
                rh = calculate_angle(r_shoulder, r_hip, r_knee)
                
                telemetry_data["left_elbow"].append(le)
                telemetry_data["right_elbow"].append(re)
                telemetry_data["left_knee"].append(lk)
                telemetry_data["right_knee"].append(rk)
                telemetry_data["left_hip"].append(lh)
                telemetry_data["right_hip"].append(rh)
                telemetry_data["movement_speed"].append(float((l_hip[0] + r_hip[0]) / 2.0))
                
                # Render tracking lines
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
            out.write(frame)
            progress_bar.progress(int((frame_idx / total_frames) * 100))
            status_text.text(f"Processing video framework: Frame {frame_idx}/{total_frames}")

        cap.release()
        out.release()
        pose.close()
        
        if not telemetry_data["left_elbow"]:
            status_text.empty()
            progress_bar.empty()
            st.error("❌ No player detected in the video frames. Make sure the full body is visible.")
        else:
            status_text.text("🎬 Converting video for web playback...")
            
            # CRITICAL FIX: Use ffmpeg to convert video format to H.264 so it's not a 0-second broken file
            if os.path.exists(final_web_path):
                os.remove(final_web_path)
            os.system(f"ffmpeg -y -i {raw_output_path} -vcodec libx264 -pix_fmt yuv420p {final_web_path}")
            
            # Build data packet
            def get_stats(arr):
                return {"min_deg": int(np.min(arr)), "max_deg": int(np.max(arr))}
                
            summary_metrics = {
                "discipline": analysis_mode,
                "left_elbow": get_stats(telemetry_data["left_elbow"]),
                "right_elbow": get_stats(telemetry_data["right_elbow"]),
                "left_knee": get_stats(telemetry_data["left_knee"]),
                "right_knee": get_stats(telemetry_data["right_knee"])
            }
            
            # Simple, plain layout instructions for the AI
            system_instruction = f"""
            You are an elite sports science cricket coach. Review this tracking data:
            {json.dumps(summary_metrics, indent=2)}

            Write a short, highly simplified bullet-point report. Avoid long essays. 
            Use exactly these headings and provide clear, simple feedback under each:

            ### 🎯 Key Points
            (List 2 simple observations about the movement speed or joint extensions)

            ### ❌ Mistakes
            (Identify 1 or 2 specific technical faults shown by the minimum or maximum angles)

            ### 💪 Strengths
            (Identify what part of the posture or leg bend looks solid)

            ### 🌱 Wellness & Safety
            (Provide a quick tip on body alignment to avoid injury or strain)

            ### 🦴 Body Mechanics Score
            (Give a quick 1-sentence wrap-up of their body mechanics)
            """
            
            status_text.text("🧠 Generating your simple coaching card...")
            try:
                model = genai.GenerativeModel('gemini-3.5-flash')
                response = model.generate_content(system_instruction)
                
                status_text.empty()
                progress_bar.empty()
                st.success("✅ Analysis Completed!")
                
                # Display Layout
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📹 AI Telemetry Stream")
                    st.video(final_web_path) # Will play smoothly now!
                with col2:
                    st.subheader("📊 Performance Scorecard")
                    st.markdown(response.text)
                    
            except Exception as e:
                st.error(f"AI Connection Error: {e}")
else:
    st.info("💡 Open the sidebar panel, insert your Gemini API Key, and upload your movement video to begin processing.")
