
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import google.generativeai as genai

# Safe vector angle calculator
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

# Page UI Config
st.set_page_config(page_title="CricAnalytics AI - Simplified", layout="wide")
st.title("🏏 CricAnalytics AI: Simple Performance Scorecard")
st.markdown("Instantly identify actions and get clean, simple biomechanical feedback.")

# Sidebar Configuration
st.sidebar.header("⚙️ Core Setup")
user_api_key = st.sidebar.text_input("Gemini API Key", type="password")
analysis_mode = st.sidebar.selectbox("Select Track", ["Batting", "Bowling", "Fielding"])
uploaded_file = st.sidebar.file_uploader("Upload Cricket Clip", type=['mp4', 'mov', 'avi'])

if uploaded_file and user_api_key:
    genai.configure(api_key=user_api_key)
    
    input_path = "temp_input.mp4"
    output_path = "telemetry_output.mp4"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
        
    if st.sidebar.button("🚀 Run Simple Analysis"):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0 or fps <= 0:
            st.error("❌ Broken video file. Please upload another video clip.")
            st.stop()
            
        # FIX: Changed codec to 'avc1' (H.264) so it plays perfectly in HTML5 web browsers instead of showing 0 seconds
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        
        # Init MediaPipe
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        
        # Telemetry storage
        telemetry_data = {
            "left_elbow": [], "right_elbow": [],
            "left_knee": [], "right_knee": [],
            "left_hip": [], "right_hip": []
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
                
                # Simple point picker
                def get_pt(idx): return [landmarks[idx].x, landmarks[idx].y]
                
                # Pull raw coordinates
                ls, le, lw = get_pt(11), get_pt(13), get_pt(15)
                rs, re, rw = get_pt(12), get_pt(14), get_pt(16)
                lh, lk, la = get_pt(23), get_pt(25), get_pt(27)
                rh, rk, ra = get_pt(24), get_pt(26), get_pt(28)
                
                # Compute angles
                le_ang = calculate_angle(ls, le, lw)
                re_ang = calculate_angle(rs, re, rw)
                lk_ang = calculate_angle(lh, lk, la)
                rk_ang = calculate_angle(rh, rk, ra)
                
                telemetry_data["left_elbow"].append(le_ang)
                telemetry_data["right_elbow"].append(re_ang)
                telemetry_data["left_knee"].append(lk_ang)
                telemetry_data["right_knee"].append(rk_ang)
                telemetry_data["left_hip"].append(calculate_angle(ls, lh, lk))
                telemetry_data["right_hip"].append(calculate_angle(rs, rh, rk))
                
                # Draw simple tracking skeleton
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
            out.write(frame)
            progress_bar.progress(int((frame_idx / total_frames) * 100))
            status_text.text(f"Reading tracking data: {frame_idx}/{total_frames} frames...")

        cap.release()
        out.release()
        pose.close()
        
        # Verify data exists to prevent code crashes
        if not telemetry_data["left_elbow"]:
            status_text.empty()
            progress_bar.empty()
            st.error("❌ **Tracking Failure:** Could not find a clear human body shape in the video. Please verify lighting and try again.")
        else:
            status_text.text("🧠 AI is identifying the movement and building scorecard...")
            
            # Pack statistical limits
            def get_stats(arr): return {"min": int(np.min(arr)), "max": int(np.max(arr))}
            
            payload = {
                "user_selected_track": analysis_mode,
                "data_points": {
                    "left_elbow": get_stats(telemetry_data["left_elbow"]),
                    "right_elbow": get_stats(telemetry_data["right_elbow"]),
                    "left_knee": get_stats(telemetry_data["left_knee"]),
                    "right_knee": get_stats(telemetry_data["right_knee"]),
                    "left_hip": get_stats(telemetry_data["left_hip"]),
                    "right_hip": get_stats(telemetry_data["right_hip"])
                }
            }
            
            # STRICT, SIMPLIFIED PROMPT FOR CLEAN DISPLAY
            simple_instruction = f"""
            You are an expert cricket performance coach. Review the video context tracking payload here:
            {json.dumps(payload, indent=2)}

            Your task is to provide a short, clean, and punchy evaluation scorecard. 
            Do NOT write long paragraphs. Keep descriptions short, using simple, clear words.

            Format your exact response using these specific headers:
            
            🎯 **ACTION IDENTIFIED**
            [Look at the upper/lower body data distributions under the '{analysis_mode}' track and state the exact specific movement, shot played, or delivery type you deduce from the physics data].

            💪 **MAIN STRENGTHS**
            - [Bullet point showing what went well according to the numbers]

            ⚠️ **KEY MISTAKES**
            - [Bullet point flagging technical errors found in the range of motion]

            ⚙️ **BODY MECHANISM RATING**
            - [Explain simply what the elbow and knee extensions show about their body shape during execution]

            🛠️ **SIMPLE TRAINING DRILL**
            - [Provide exactly one short 2-sentence drill to fix the biggest mistake noted]
            """
            
            try:
                model = genai.GenerativeModel('gemini-3.5-flash')
                response = model.generate_content(simple_instruction)
                
                status_text.empty()
                progress_bar.empty()
                st.success("✅ Analysis Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Tracked Telemetry Video")
                    # Will now load seamlessly in web view without the 0-second glitch
                    st.video(output_path)
                with col2:
                    st.subheader("📋 Simple Coaching Scorecard")
                    st.markdown(response.text)
                    
            except Exception as e:
                st.error(f"❌ Gemini Error: {e}")
else:
    st.info("💡 Open the sidebar, insert your Gemini API Key, select your track, and upload a video clip to begin.")
