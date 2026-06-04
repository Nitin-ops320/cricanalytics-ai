import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import google.generativeai as genai
import tensorflow as tf
import os

st.set_page_config(page_title="CricAnalytics AI - Frame Analysis", layout="wide")
st.title("🏏 CricAnalytics AI: Biomechanical Mistake Highlight Engine")

user_api_key = st.sidebar.text_input("Gemini API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload Cricket Clip", type=['mp4', 'mov', 'avi'])

dl_model_path = 'cricket_dl_model.keras'
if not os.path.exists(dl_model_path):
    st.error(f"❌ Core Component Missing: Ensure '{dl_model_path}' is committed into your GitHub repo.")
    st.stop()

dl_network = tf.keras.models.load_model(dl_model_path)

if uploaded_file and user_api_key:
    genai.configure(api_key=user_api_key)
    
    input_path = "temp_input.mp4"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
        
    if st.sidebar.button("🚀 Analyze Player Movements"):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
        
        frame_window_sequence = []
        full_video_telemetry = []
        
        # Array list to store images of mistakes we find
        detected_mistakes_gallery = []
        
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
                
                # Helper to get normalized coordinates
                def get_pt(idx): return [landmarks[idx].x, landmarks[idx].y]
                
                # Convert normalized coordinates to actual frame pixels for drawing circles
                def get_pixel_coords(idx): 
                    return int(landmarks[idx].x * width), int(landmarks[idx].y * height)
                
                ls, le, lw = get_pt(11), get_pt(13), get_pt(15)
                rs, re, rw = get_pt(12), get_pt(14), get_pt(16)
                lh, lk, la = get_pt(23), get_pt(25), get_pt(27)
                rh, rk, ra = get_pt(24), get_pt(26), get_pt(28)
                
                le_ang = calculate_angle(ls, le, lw)
                re_ang = calculate_angle(rs, re, rw)
                lk_ang = calculate_angle(lh, lk, la)
                rk_ang = calculate_angle(rh, rk, ra)
                
                current_frame_features = [le_ang, re_ang, lk_ang, rk_ang, lw[1], ls[1]]
                frame_window_sequence.append(current_frame_features)
                
                if len(frame_window_sequence) > 30:
                    frame_window_sequence.pop(0)
                    
                full_video_telemetry.append(current_frame_features)
                
                # --- BIOMECHANICAL MISTAKE DETECTION RULES ---
                # Rule 1: Bowling Arm/Wrist Drop (If wrist falls below shoulder baseline too early)
                if rw[1] > rs[1] and len(detected_mistakes_gallery) < 3:
                    # Create a clean copy of the frame to draw on
                    annotated_frame = frame.copy()
                    pixel_x, pixel_y = get_pixel_coords(16) # Right Wrist index
                    
                    # Draw a bright red circle on the right wrist mistake area
                    cv2.circle(annotated_frame, (pixel_x, pixel_y), 25, (0, 0, 255), 4)
                    
                    # Save the frame image array and context
                    detected_mistakes_gallery.append({
                        "type": "Wrist Drop Error",
                        "frame_rgb": cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        "description": f"At frame {frame_idx}, the bowler's wrist dropped lower than the shoulder plane prematurely, destroying rotation leverage."
                    })
                
                # Rule 2: Unbent Front Knee during Drive (Knee angle too straight)
                if lk_ang > 150 and len(detected_mistakes_gallery) < 3:
                    annotated_frame = frame.copy()
                    pixel_x, pixel_y = get_pixel_coords(25) # Left Knee index
                    
                    # Draw a bright yellow circle on the knee mistake area
                    cv2.circle(annotated_frame, (pixel_x, pixel_y), 25, (0, 255, 255), 4)
                    
                    detected_mistakes_gallery.append({
                        "type": "Poor Knee Flexion",
                        "frame_rgb": cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        "description": f"At frame {frame_idx}, the batsman's front knee flexed at {lk_ang}°. The leg is too straight, restricting forward body weight transfer."
                    })
                    
            progress_bar.progress(int((frame_idx / total_frames) * 100))
            status_text.text(f"Scanning frame sequences: {frame_idx}/{total_frames}...")

        cap.release()
        pose.close()
        
        if len(frame_window_sequence) < 30:
            st.error("❌ Video sequence too short to process.")
        else:
            status_text.text("🧠 Calculating Deep Learning Motion Metrics...")
            input_tensor = np.array([frame_window_sequence])
            predictions = dl_network.predict(input_tensor)[0]
            class_idx = np.argmax(predictions)
            
            classes_map = {0: "Grounded Cover Drive", 1: "Lofted Power Hit (Six)", 2: "Fast Bowling Action"}
            predicted_action_label = classes_map.get(class_idx, "Unknown Shot Type")
            
            status_text.text("✍️ Assembling Visual Bio-Report...")
            all_metrics = np.array(full_video_telemetry)
            
            payload = {
                "deep_learning_verified_shot": predicted_action_label,
                "confidence_score": float(np.max(predictions)),
                "movement_time_series_summary": {
                    "minimum_front_knee_flexion": int(np.min(all_metrics[:, 2])),
                    "maximum_arm_extension_reach": int(np.max(all_metrics[:, 1]))
                }
            }
            
            targeted_prompt = f"""
            You are an elite sports science cricket coach. Review this telemetry payload:
            {json.dumps(payload, indent=2)}
            
            The system verified the movement as a: {predicted_action_label}.
            Provide a short summary detailing what movement errors occurred based on these metrics. 
            Keep it strictly using these bold headers:
            🎯 **SHOT PLAYED**
            ⚠️ **THE ACTUAL MOVEMENT PROBLEM**
            🛠️ **MOVEMENT FIXING DRILL**
            """
            
            try:
                model = genai.GenerativeModel('gemini-3.5-flash')
                response = model.generate_content(targeted_prompt)
                
                status_text.empty()
                progress_bar.empty()
                st.success("✅ Analysis Completed Automatically!")
                
                # LAYOUT STRUCTURE
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📋 Core Biomechanical Assessment")
                    st.markdown(response.text)
                
                with col2:
                    st.subheader("📸 Visual Keyframe Mistake Highlights")
                    if not detected_mistakes_gallery:
                        st.write("🌿 No major biomechanical threshold violations detected in this movement cycle.")
                    else:
                        # Display each captured image mistake sequentially on the screen
                        for item in detected_mistakes_gallery[:2]: # Show up to 2 distinct mistakes
                            st.image(item["frame_rgb"], caption=item["type"], use_container_width=True)
                            st.caption(f"🔍 **Technical Mistake Detailing:** {item['description']}")
                            st.markdown("---")
                            
            except Exception as e:
                st.error(f"❌ AI Engine Error: {e}")
