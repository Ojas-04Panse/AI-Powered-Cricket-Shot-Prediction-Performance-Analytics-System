<<<<<<< HEAD
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import scipy.signal
import io
from scipy.spatial.distance import euclidean
import logging
from collections import deque
from predictor import CricketShotPredictor
from config import config

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Sports Analytics Dashboard")

# --- UI Styling and Parameters ---
st.title("🏏 Sports Movement Analytics Dashboard")
st.info("Upload a video to analyze player, bat, and ball movements. For the best experience, use Streamlit's dark theme (Menu > Settings > Theme).")

with st.expander("⚙️ Advanced Detection & Physics Tuning"):
    st.header("Player Settings")
    PLAYER_MIN_AREA = st.slider("Player Min Area", 500, 5000, 900, key="p_min")
    PLAYER_MAX_AREA = st.slider("Player Max Area", 5000, 50000, 20000, key="p_max")

    st.header("Bat Settings")
    BAT_MIN_AREA = st.slider("Bat Min Area", 50, 1000, 150, key="b_min")
    BAT_MAX_AREA = st.slider("Bat Max Area", 200, 2000, 800, key="b_max")

    st.header("Ball Settings")
    BALL_MIN_AREA = st.slider("Ball Min Area", 10, 200, 35, key="ball_min")
    BALL_MAX_AREA = st.slider("Ball Max Area", 50, 500, 150, key="ball_max")
    BALL_ASPECT_RATIO_MIN = st.slider("Ball Min Aspect Ratio", 0.1, 1.0, 0.7, key="ball_ar_min")
    BALL_ASPECT_RATIO_MAX = st.slider("Ball Max Aspect Ratio", 1.0, 2.0, 1.3, key="ball_ar_max")

    st.header("Physics & Event Settings")
    IMPACT_THRESHOLD_PIXELS = st.slider("Impact Proximity (pixels)", 10, 200, 75, key="impact")
    PIXELS_PER_METER = st.slider("Pixels-to-Meters Ratio", 10, 200, 40,
                                 help="Estimate how many pixels in the video correspond to 1 real-world meter.")


# --- ENHANCED SPEED CALCULATION MODULE ---
class EnhancedSpeedCalculator:
    """Enhanced speed calculation with smoothing and validation"""
    
    def __init__(self, pixels_per_meter=40, speed_window_size=5):
        self.pixels_per_meter = pixels_per_meter
        self.speed_window_size = speed_window_size
        self.speed_history = deque(maxlen=speed_window_size)
        
        # Calibration settings
        self.calibration = type('obj', (object,), {
            'homography_matrix': None,
            'pixel_to_meter_ratio': pixels_per_meter
        })()
    
    def calculate_enhanced_speed(self, ball_positions, fps):
        """Enhanced speed calculation with smoothing and validation"""
        if len(ball_positions) < 2 or fps == 0:
            return np.array([]), np.array([])
        
        enhanced_speeds = []
        frame_numbers = []
        
        for i in range(1, len(ball_positions)):
            try:
                # Get current and previous positions
                current_pos = ball_positions[i]
                prev_pos = ball_positions[i-1]
                
                # Skip if either position is None
                if current_pos is None or prev_pos is None:
                    # Use last known speed or 0
                    last_speed = enhanced_speeds[-1] if enhanced_speeds else 0.0
                    enhanced_speeds.append(last_speed)
                    frame_numbers.append(i + 1)
                    continue
                
                # Extract positions (handle tuple format from original code)
                if isinstance(current_pos, tuple) and len(current_pos) >= 2:
                    curr_xy = (current_pos[0], current_pos[1])
                    prev_xy = (prev_pos[0], prev_pos[1])
                else:
                    curr_xy = current_pos
                    prev_xy = prev_pos
                
                # Calculate distance and speed
                speed_kmh = self._calculate_speed_between_points(prev_xy, curr_xy, fps)
                
                # Apply smoothing and validation
                validated_speed = self._validate_and_smooth_speed(speed_kmh)
                enhanced_speeds.append(validated_speed)
                frame_numbers.append(i + 1)
                
            except Exception as e:
                logger.warning(f"Speed calculation error at frame {i}: {e}")
                # Fallback to last speed or 0
                last_speed = enhanced_speeds[-1] if enhanced_speeds else 0.0
                enhanced_speeds.append(last_speed)
                frame_numbers.append(i + 1)
        
        return np.array(enhanced_speeds), np.array(frame_numbers)
    
    def _calculate_speed_between_points(self, pos1, pos2, fps):
        """Calculate speed between two points"""
        # Calculate distance in pixels
        distance_px = euclidean(pos1, pos2)
        
        # Convert to real-world distance
        if self.calibration.homography_matrix is not None:
            try:
                w1 = self._pixel_to_world(pos1)
                w2 = self._pixel_to_world(pos2)
                distance_m = euclidean(w1, w2)
            except:
                distance_m = distance_px / self.pixels_per_meter
        else:
            distance_m = distance_px / self.pixels_per_meter
        
        # Calculate time difference (assuming consecutive frames)
        dt = 1.0 / fps
        
        if dt > 0:
            speed_ms = distance_m / dt
            speed_kmh = speed_ms * 3.6
            return speed_kmh
        
        return 0.0
    
    def _validate_and_smooth_speed(self, speed_kmh):
        """Validate and smooth speed measurement"""
        # Sanity check for cricket ball speeds (5-200 km/h)
        if 5 <= speed_kmh <= 200:
            self.speed_history.append(speed_kmh)
            
            # Return smoothed speed (moving average)
            smoothed_speed = np.mean(list(self.speed_history))
            return smoothed_speed
        else:
            # Invalid speed, use last known speed or 0
            if self.speed_history:
                return list(self.speed_history)[-1]
            else:
                return 0.0
    
    def _pixel_to_world(self, pixel_point):
        """Convert pixel coordinates to world coordinates (placeholder)"""
        # This would use homography matrix if available
        # For now, just return scaled coordinates
        return (pixel_point[0] / self.pixels_per_meter, pixel_point[1] / self.pixels_per_meter)


def calculate_and_smooth_speeds(ball_pos, fps, pixels_per_meter, speed_window_size=5):
    """Enhanced version of the original speed calculation function"""
    
    # Initialize enhanced speed calculator
    speed_calc = EnhancedSpeedCalculator(pixels_per_meter, speed_window_size)
    
    # Use enhanced calculation
    enhanced_speeds, frame_array = speed_calc.calculate_enhanced_speed(ball_pos, fps)
    
    # Return in same format as original function
    return enhanced_speeds, frame_array


# --- Shot Prediction Function ---
@st.cache_data
def predict_cricket_shot(video_path):
    """Predict cricket shot type from video"""
    try:
        # Initialize predictor
        predictor = CricketShotPredictor()
        predictor.load_model(config.OUTPUT_DIR)
        
        # Make prediction
        result = predictor.predict_video(video_path)
        
        if result['error']:
            return None, result['message']
        else:
            return result, None
            
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

# --- Core Processing and Plotting Functions (UNCHANGED) ---

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return None, None, None, None, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        st.warning("Could not determine video FPS. Defaulting to 30.")
        fps = 30
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_dir = tempfile.gettempdir()
    output_video_path = os.path.join(output_dir, "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        st.error("Error: Failed to create the video writer."); cap.release(); return None, None, None, None, 0
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    player_positions, bat_positions, ball_positions = [], [], []
    frame_count, total_frames = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="Analyzing video...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        fgmask = fgbg.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0 or w == 0: continue
            aspect_ratio = w / float(h)
            center = (x + w // 2, y + h // 2)
            if PLAYER_MIN_AREA < area < PLAYER_MAX_AREA:
                player_positions.append((center[0], center[1], frame_count)); cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            elif BAT_MIN_AREA < area < BAT_MAX_AREA and (aspect_ratio > 2.5 or aspect_ratio < 0.4):
                bat_positions.append((center[0], center[1], frame_count)); cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif BALL_MIN_AREA < area < BALL_MAX_AREA and BALL_ASPECT_RATIO_MIN < aspect_ratio < BALL_ASPECT_RATIO_MAX:
                ball_positions.append((center[0], center[1], frame_count)); cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        out.write(frame)
        progress_bar.progress(frame_count / total_frames, text=f"Analyzing Frame {frame_count}/{total_frames}")
    cap.release(); out.release(); progress_bar.empty()
    if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
        st.error("Output video file was not created or is empty."); return None, None, None, None, 0
    return player_positions, bat_positions, ball_positions, output_video_path, fps

def find_impact_frame(ball_pos, bat_pos, threshold):
    if not ball_pos or not bat_pos: return None
    bat_frames = {frame: (x, y) for x, y, frame in bat_pos}
    for ball_x, ball_y, frame_num in ball_pos:
        if frame_num in bat_frames:
            bat_x, bat_y = bat_frames[frame_num]
            distance = np.sqrt((ball_x - bat_x)**2 + (ball_y - bat_y)**2)
            if distance < threshold: return frame_num
    return None

def plot_bat_swing_arc(bat_pos, background_frame):
    if len(bat_pos) < 10: return background_frame
    output_image = background_frame.copy()
    positions = np.array([(p[0], p[1]) for p in bat_pos])
    x, y = positions[:, 0], positions[:, 1]
    window_length = min(11, len(x) - 1 if len(x) % 2 == 0 else len(x))
    if window_length < 3: return background_frame
    smooth_x = scipy.signal.savgol_filter(x, window_length, 2)
    smooth_y = scipy.signal.savgol_filter(y, window_length, 2)
    points = np.column_stack((smooth_x, smooth_y)).astype(np.int32)
    cv2.polylines(output_image, [points], isClosed=False, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.circle(output_image, tuple(points[0]), 8, (0, 255, 0), -1)
    cv2.circle(output_image, tuple(points[-1]), 8, (0, 0, 255), -1)
    return output_image

def plot_ball_trajectory(ball_pos):
    if len(ball_pos) < 25: return None
    plt.style.use('dark_background')
    try:
        positions = np.array([(p[0], p[1]) for p in ball_pos])
        x, y = positions[:, 0], positions[:, 1]
        if len(x) < 25: return None
        window_length, polyorder = 21, 2
        if len(x) <= window_length: return None
        smooth_x = scipy.signal.savgol_filter(x, window_length, polyorder)
        smooth_y = scipy.signal.savgol_filter(y, window_length, polyorder)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(smooth_x, smooth_y, marker='o', color='#00A676', label="Smoothed Ball Trajectory")
        ax.invert_yaxis(); ax.set_title("2D Ball Trajectory Curve (Pre-Impact)"); ax.set_xlabel("X Position (pixels)"); ax.set_ylabel("Y Position (pixels)")
        ax.grid(True, linestyle='--', alpha=0.5); ax.legend()
        return fig
    except Exception as e:
        st.warning(f"Could not plot trajectory: {e}"); return None

def plot_ball_speed(speeds_kmh, frame_array):
    if len(speeds_kmh) == 0: return None
    plt.style.use('dark_background')
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(frame_array, speeds_kmh, color='#FF6F61', label="Enhanced Smoothed Ball Speed (km/h)")
        ax.fill_between(frame_array, speeds_kmh, color='#FF6F61', alpha=0.2)
        ax.set_title("Enhanced Ball Speed Over Frames (Pre-Impact)"); ax.set_xlabel("Frame Number"); ax.set_ylabel("Speed (km/h)")
        ax.grid(True, linestyle='--', alpha=0.5); ax.legend()
        
        # Add peak speed annotation
        if len(speeds_kmh) > 0:
            peak_speed = np.max(speeds_kmh)
            peak_frame = frame_array[np.argmax(speeds_kmh)]
            ax.annotate(f'Peak: {peak_speed:.1f} km/h', 
                       xy=(peak_frame, peak_speed), 
                       xytext=(peak_frame + len(frame_array)*0.1, peak_speed + np.max(speeds_kmh)*0.1),
                       arrowprops=dict(arrowstyle='->', color='yellow'),
                       color='yellow', fontweight='bold')
        
        return fig
    except Exception as e:
        st.warning(f"Could not plot speed: {e}"); return None

# --- Streamlit App UI (UNCHANGED) ---
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_video_path, "wb") as tfile:
        tfile.write(uploaded_file.getbuffer())

    if st.button("Analyze Video", key="analyze", use_container_width=True):
        # Show progress for video analysis
        progress_container = st.container()
        with progress_container:
            st.info("🔍 Analyzing video for movement tracking...")
        
        player_pos, bat_pos, ball_pos, out_video_path, fps = process_video(temp_video_path)

        if out_video_path:
            # Show progress for shot prediction
            with progress_container:
                st.info("🏏 Predicting cricket shot type...")
            
            # Perform shot prediction
            prediction_result, prediction_error = predict_cricket_shot(temp_video_path)
            
            progress_container.empty()
            st.success("Processing Complete!")
            
            impact_frame = find_impact_frame(ball_pos, bat_pos, IMPACT_THRESHOLD_PIXELS)
            ball_pos_pre_impact = [p for p in ball_pos if p[2] <= impact_frame] if impact_frame else ball_pos
            
            # --- KPI Metrics Row ---
            st.divider()
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            
            # Use enhanced speed calculation
            smoothed_speeds_kmh, _ = calculate_and_smooth_speeds(ball_pos_pre_impact, fps, PIXELS_PER_METER)
            peak_speed_kmh_str = f"{max(smoothed_speeds_kmh):.1f}" if len(smoothed_speeds_kmh) > 0 else "N/A"

            mcol1.metric("Impact Frame", str(impact_frame) if impact_frame else "N/A")
            mcol2.metric("Peak Ball Speed (km/h)", peak_speed_kmh_str)
            mcol3.metric("Bat Detections", str(len(bat_pos)))
            mcol4.metric("Ball Detections", str(len(ball_pos)))
            
            # --- Shot Prediction Results ---
            st.divider()
            st.header("🏏 Cricket Shot Prediction")
            
            if prediction_error:
                st.error(f"❌ Shot prediction failed: {prediction_error}")
            elif prediction_result:
                # Main prediction result
                pred_col1, pred_col2 = st.columns([2, 1])
                
                with pred_col1:
                    predicted_shot = prediction_result['predicted_class'].replace('_', ' ').title()
                    confidence = prediction_result['confidence']
                    description = prediction_result['description']
                    
                    st.subheader(f"🎯 Predicted Shot: **{predicted_shot}**")
                    st.write(f"📝 **Description:** {description}")
                    
                    # Confidence bar
                    st.write("🎯 **Confidence Level:**")
                    st.progress(confidence, text=f"{confidence:.1%}")
                
                with pred_col2:
                    # Top 3 predictions
                    st.subheader("🏆 Top 3 Predictions")
                    for i, pred in enumerate(prediction_result['top_3_predictions'], 1):
                        shot_name = pred['class'].replace('_', ' ').title()
                        prob = pred['probability']
                        
                        # Color coding for confidence
                        if i == 1:
                            st.success(f"🥇 **{shot_name}**\n{prob:.1%}")
                        elif i == 2:
                            st.info(f"🥈 **{shot_name}**\n{prob:.1%}")
                        else:
                            st.warning(f"🥉 **{shot_name}**\n{prob:.1%}")
                
                # Detailed probability breakdown
                with st.expander("📊 Detailed Probability Breakdown"):
                    st.subheader("All Shot Probabilities")
                    
                    # Create a dataframe for better visualization
                    import pandas as pd
                    prob_data = []
                    for class_name, prob in prediction_result['all_probabilities'].items():
                        shot_name = class_name.replace('_', ' ').title()
                        prob_data.append({
                            'Shot Type': shot_name,
                            'Probability': prob,
                            'Percentage': f"{prob:.1%}"
                        })
                    
                    prob_df = pd.DataFrame(prob_data)
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Display as a nice table
                    for idx, row in prob_df.iterrows():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"**{row['Shot Type']}**")
                        with col2:
                            st.progress(row['Probability'])
                        with col3:
                            st.write(row['Percentage'])
            
            st.divider()

            # --- Main Visuals Layout ---
            col1, col2 = st.columns(2)
            with col1:
                st.header("Video Analysis")
                with open(out_video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
            
            with col2:
                st.header("Bat Swing Arc")
                cap = cv2.VideoCapture(temp_video_path)
                ret, first_frame = cap.read()
                cap.release()
                if ret:
                    bat_swing_pos = [p for p in bat_pos if p[2] > impact_frame - 30 and p[2] <= impact_frame] if impact_frame else bat_pos[-30:]
                    swing_arc_img = plot_bat_swing_arc(bat_swing_pos, first_frame)
                    st.image(swing_arc_img, channels="BGR", caption="Path of the bat just before impact (Start: Green, End: Red).")
            
            st.divider()

            col3, col4 = st.columns(2)
            with col3:
                st.header("Ball Trajectory")
                trajectory_fig = plot_ball_trajectory(ball_pos_pre_impact)
                if trajectory_fig: st.pyplot(trajectory_fig)
                else: st.warning("Could not generate a clean ball trajectory.")
            
            with col4:
                st.header("Ball Speed")
                # Use enhanced speed calculation
                smoothed_speeds_kmh, frame_array = calculate_and_smooth_speeds(ball_pos_pre_impact, fps, PIXELS_PER_METER)
                speed_fig = plot_ball_speed(smoothed_speeds_kmh, frame_array)
                if speed_fig: st.pyplot(speed_fig)
                else: st.warning("Not enough data to plot ball speed.")

            # --- Enhanced Speed Statistics (NEW SECTION) ---
            if len(smoothed_speeds_kmh) > 0:
                st.divider()
                st.subheader("📊 Enhanced Speed Analytics")
                
                speed_col1, speed_col2, speed_col3 = st.columns(3)
                
                with speed_col1:
                    st.metric("Average Speed", f"{np.mean(smoothed_speeds_kmh):.1f} km/h")
                    st.metric("Speed Std Dev", f"{np.std(smoothed_speeds_kmh):.1f} km/h")
                
                with speed_col2:
                    st.metric("Minimum Speed", f"{np.min(smoothed_speeds_kmh):.1f} km/h")
                    st.metric("Maximum Speed", f"{np.max(smoothed_speeds_kmh):.1f} km/h")
                
                with speed_col3:
                    st.metric("Median Speed", f"{np.median(smoothed_speeds_kmh):.1f} km/h")
                    st.metric("Speed Range", f"{np.max(smoothed_speeds_kmh) - np.min(smoothed_speeds_kmh):.1f} km/h")

            os.remove(temp_video_path)
=======
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import scipy.signal
import io
from scipy.spatial.distance import euclidean
import logging
from collections import deque
from predictor import CricketShotPredictor
from config import config

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Sports Analytics Dashboard")

# --- UI Styling and Parameters ---
st.title("🏏 Sports Movement Analytics Dashboard")
st.info("Upload a video to analyze player, bat, and ball movements. For the best experience, use Streamlit's dark theme (Menu > Settings > Theme).")

with st.expander("⚙️ Advanced Detection & Physics Tuning"):
    st.header("Player Settings")
    PLAYER_MIN_AREA = st.slider("Player Min Area", 500, 5000, 900, key="p_min")
    PLAYER_MAX_AREA = st.slider("Player Max Area", 5000, 50000, 20000, key="p_max")

    st.header("Bat Settings")
    BAT_MIN_AREA = st.slider("Bat Min Area", 50, 1000, 150, key="b_min")
    BAT_MAX_AREA = st.slider("Bat Max Area", 200, 2000, 800, key="b_max")

    st.header("Ball Settings")
    BALL_MIN_AREA = st.slider("Ball Min Area", 10, 200, 35, key="ball_min")
    BALL_MAX_AREA = st.slider("Ball Max Area", 50, 500, 150, key="ball_max")
    BALL_ASPECT_RATIO_MIN = st.slider("Ball Min Aspect Ratio", 0.1, 1.0, 0.7, key="ball_ar_min")
    BALL_ASPECT_RATIO_MAX = st.slider("Ball Max Aspect Ratio", 1.0, 2.0, 1.3, key="ball_ar_max")

    st.header("Physics & Event Settings")
    IMPACT_THRESHOLD_PIXELS = st.slider("Impact Proximity (pixels)", 10, 200, 75, key="impact")
    PIXELS_PER_METER = st.slider("Pixels-to-Meters Ratio", 10, 200, 40,
                                 help="Estimate how many pixels in the video correspond to 1 real-world meter.")


# --- ENHANCED SPEED CALCULATION MODULE ---
class EnhancedSpeedCalculator:
    """Enhanced speed calculation with smoothing and validation"""
    
    def __init__(self, pixels_per_meter=40, speed_window_size=5):
        self.pixels_per_meter = pixels_per_meter
        self.speed_window_size = speed_window_size
        self.speed_history = deque(maxlen=speed_window_size)
        
        # Calibration settings
        self.calibration = type('obj', (object,), {
            'homography_matrix': None,
            'pixel_to_meter_ratio': pixels_per_meter
        })()
    
    def calculate_enhanced_speed(self, ball_positions, fps):
        """Enhanced speed calculation with smoothing and validation"""
        if len(ball_positions) < 2 or fps == 0:
            return np.array([]), np.array([])
        
        enhanced_speeds = []
        frame_numbers = []
        
        for i in range(1, len(ball_positions)):
            try:
                # Get current and previous positions
                current_pos = ball_positions[i]
                prev_pos = ball_positions[i-1]
                
                # Skip if either position is None
                if current_pos is None or prev_pos is None:
                    # Use last known speed or 0
                    last_speed = enhanced_speeds[-1] if enhanced_speeds else 0.0
                    enhanced_speeds.append(last_speed)
                    frame_numbers.append(i + 1)
                    continue
                
                # Extract positions (handle tuple format from original code)
                if isinstance(current_pos, tuple) and len(current_pos) >= 2:
                    curr_xy = (current_pos[0], current_pos[1])
                    prev_xy = (prev_pos[0], prev_pos[1])
                else:
                    curr_xy = current_pos
                    prev_xy = prev_pos
                
                # Calculate distance and speed
                speed_kmh = self._calculate_speed_between_points(prev_xy, curr_xy, fps)
                
                # Apply smoothing and validation
                validated_speed = self._validate_and_smooth_speed(speed_kmh)
                enhanced_speeds.append(validated_speed)
                frame_numbers.append(i + 1)
                
            except Exception as e:
                logger.warning(f"Speed calculation error at frame {i}: {e}")
                # Fallback to last speed or 0
                last_speed = enhanced_speeds[-1] if enhanced_speeds else 0.0
                enhanced_speeds.append(last_speed)
                frame_numbers.append(i + 1)
        
        return np.array(enhanced_speeds), np.array(frame_numbers)
    
    def _calculate_speed_between_points(self, pos1, pos2, fps):
        """Calculate speed between two points"""
        # Calculate distance in pixels
        distance_px = euclidean(pos1, pos2)
        
        # Convert to real-world distance
        if self.calibration.homography_matrix is not None:
            try:
                w1 = self._pixel_to_world(pos1)
                w2 = self._pixel_to_world(pos2)
                distance_m = euclidean(w1, w2)
            except:
                distance_m = distance_px / self.pixels_per_meter
        else:
            distance_m = distance_px / self.pixels_per_meter
        
        # Calculate time difference (assuming consecutive frames)
        dt = 1.0 / fps
        
        if dt > 0:
            speed_ms = distance_m / dt
            speed_kmh = speed_ms * 3.6
            return speed_kmh
        
        return 0.0
    
    def _validate_and_smooth_speed(self, speed_kmh):
        """Validate and smooth speed measurement"""
        # Sanity check for cricket ball speeds (5-200 km/h)
        if 5 <= speed_kmh <= 200:
            self.speed_history.append(speed_kmh)
            
            # Return smoothed speed (moving average)
            smoothed_speed = np.mean(list(self.speed_history))
            return smoothed_speed
        else:
            # Invalid speed, use last known speed or 0
            if self.speed_history:
                return list(self.speed_history)[-1]
            else:
                return 0.0
    
    def _pixel_to_world(self, pixel_point):
        """Convert pixel coordinates to world coordinates (placeholder)"""
        # This would use homography matrix if available
        # For now, just return scaled coordinates
        return (pixel_point[0] / self.pixels_per_meter, pixel_point[1] / self.pixels_per_meter)


def calculate_and_smooth_speeds(ball_pos, fps, pixels_per_meter, speed_window_size=5):
    """Enhanced version of the original speed calculation function"""
    
    # Initialize enhanced speed calculator
    speed_calc = EnhancedSpeedCalculator(pixels_per_meter, speed_window_size)
    
    # Use enhanced calculation
    enhanced_speeds, frame_array = speed_calc.calculate_enhanced_speed(ball_pos, fps)
    
    # Return in same format as original function
    return enhanced_speeds, frame_array


# --- Shot Prediction Function ---
@st.cache_data
def predict_cricket_shot(video_path):
    """Predict cricket shot type from video"""
    try:
        # Initialize predictor
        predictor = CricketShotPredictor()
        predictor.load_model(config.OUTPUT_DIR)
        
        # Make prediction
        result = predictor.predict_video(video_path)
        
        if result['error']:
            return None, result['message']
        else:
            return result, None
            
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

# --- Core Processing and Plotting Functions (UNCHANGED) ---

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return None, None, None, None, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        st.warning("Could not determine video FPS. Defaulting to 30.")
        fps = 30
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_dir = tempfile.gettempdir()
    output_video_path = os.path.join(output_dir, "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        st.error("Error: Failed to create the video writer."); cap.release(); return None, None, None, None, 0
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    player_positions, bat_positions, ball_positions = [], [], []
    frame_count, total_frames = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="Analyzing video...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        fgmask = fgbg.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0 or w == 0: continue
            aspect_ratio = w / float(h)
            center = (x + w // 2, y + h // 2)
            if PLAYER_MIN_AREA < area < PLAYER_MAX_AREA:
                player_positions.append((center[0], center[1], frame_count)); cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            elif BAT_MIN_AREA < area < BAT_MAX_AREA and (aspect_ratio > 2.5 or aspect_ratio < 0.4):
                bat_positions.append((center[0], center[1], frame_count)); cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif BALL_MIN_AREA < area < BALL_MAX_AREA and BALL_ASPECT_RATIO_MIN < aspect_ratio < BALL_ASPECT_RATIO_MAX:
                ball_positions.append((center[0], center[1], frame_count)); cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        out.write(frame)
        progress_bar.progress(frame_count / total_frames, text=f"Analyzing Frame {frame_count}/{total_frames}")
    cap.release(); out.release(); progress_bar.empty()
    if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
        st.error("Output video file was not created or is empty."); return None, None, None, None, 0
    return player_positions, bat_positions, ball_positions, output_video_path, fps

def find_impact_frame(ball_pos, bat_pos, threshold):
    if not ball_pos or not bat_pos: return None
    bat_frames = {frame: (x, y) for x, y, frame in bat_pos}
    for ball_x, ball_y, frame_num in ball_pos:
        if frame_num in bat_frames:
            bat_x, bat_y = bat_frames[frame_num]
            distance = np.sqrt((ball_x - bat_x)**2 + (ball_y - bat_y)**2)
            if distance < threshold: return frame_num
    return None

def plot_bat_swing_arc(bat_pos, background_frame):
    if len(bat_pos) < 10: return background_frame
    output_image = background_frame.copy()
    positions = np.array([(p[0], p[1]) for p in bat_pos])
    x, y = positions[:, 0], positions[:, 1]
    window_length = min(11, len(x) - 1 if len(x) % 2 == 0 else len(x))
    if window_length < 3: return background_frame
    smooth_x = scipy.signal.savgol_filter(x, window_length, 2)
    smooth_y = scipy.signal.savgol_filter(y, window_length, 2)
    points = np.column_stack((smooth_x, smooth_y)).astype(np.int32)
    cv2.polylines(output_image, [points], isClosed=False, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.circle(output_image, tuple(points[0]), 8, (0, 255, 0), -1)
    cv2.circle(output_image, tuple(points[-1]), 8, (0, 0, 255), -1)
    return output_image

def plot_ball_trajectory(ball_pos):
    if len(ball_pos) < 25: return None
    plt.style.use('dark_background')
    try:
        positions = np.array([(p[0], p[1]) for p in ball_pos])
        x, y = positions[:, 0], positions[:, 1]
        if len(x) < 25: return None
        window_length, polyorder = 21, 2
        if len(x) <= window_length: return None
        smooth_x = scipy.signal.savgol_filter(x, window_length, polyorder)
        smooth_y = scipy.signal.savgol_filter(y, window_length, polyorder)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(smooth_x, smooth_y, marker='o', color='#00A676', label="Smoothed Ball Trajectory")
        ax.invert_yaxis(); ax.set_title("2D Ball Trajectory Curve (Pre-Impact)"); ax.set_xlabel("X Position (pixels)"); ax.set_ylabel("Y Position (pixels)")
        ax.grid(True, linestyle='--', alpha=0.5); ax.legend()
        return fig
    except Exception as e:
        st.warning(f"Could not plot trajectory: {e}"); return None

def plot_ball_speed(speeds_kmh, frame_array):
    if len(speeds_kmh) == 0: return None
    plt.style.use('dark_background')
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(frame_array, speeds_kmh, color='#FF6F61', label="Enhanced Smoothed Ball Speed (km/h)")
        ax.fill_between(frame_array, speeds_kmh, color='#FF6F61', alpha=0.2)
        ax.set_title("Enhanced Ball Speed Over Frames (Pre-Impact)"); ax.set_xlabel("Frame Number"); ax.set_ylabel("Speed (km/h)")
        ax.grid(True, linestyle='--', alpha=0.5); ax.legend()
        
        # Add peak speed annotation
        if len(speeds_kmh) > 0:
            peak_speed = np.max(speeds_kmh)
            peak_frame = frame_array[np.argmax(speeds_kmh)]
            ax.annotate(f'Peak: {peak_speed:.1f} km/h', 
                       xy=(peak_frame, peak_speed), 
                       xytext=(peak_frame + len(frame_array)*0.1, peak_speed + np.max(speeds_kmh)*0.1),
                       arrowprops=dict(arrowstyle='->', color='yellow'),
                       color='yellow', fontweight='bold')
        
        return fig
    except Exception as e:
        st.warning(f"Could not plot speed: {e}"); return None

# --- Streamlit App UI (UNCHANGED) ---
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_video_path, "wb") as tfile:
        tfile.write(uploaded_file.getbuffer())

    if st.button("Analyze Video", key="analyze", use_container_width=True):
        # Show progress for video analysis
        progress_container = st.container()
        with progress_container:
            st.info("🔍 Analyzing video for movement tracking...")
        
        player_pos, bat_pos, ball_pos, out_video_path, fps = process_video(temp_video_path)

        if out_video_path:
            # Show progress for shot prediction
            with progress_container:
                st.info("🏏 Predicting cricket shot type...")
            
            # Perform shot prediction
            prediction_result, prediction_error = predict_cricket_shot(temp_video_path)
            
            progress_container.empty()
            st.success("Processing Complete!")
            
            impact_frame = find_impact_frame(ball_pos, bat_pos, IMPACT_THRESHOLD_PIXELS)
            ball_pos_pre_impact = [p for p in ball_pos if p[2] <= impact_frame] if impact_frame else ball_pos
            
            # --- KPI Metrics Row ---
            st.divider()
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            
            # Use enhanced speed calculation
            smoothed_speeds_kmh, _ = calculate_and_smooth_speeds(ball_pos_pre_impact, fps, PIXELS_PER_METER)
            peak_speed_kmh_str = f"{max(smoothed_speeds_kmh):.1f}" if len(smoothed_speeds_kmh) > 0 else "N/A"

            mcol1.metric("Impact Frame", str(impact_frame) if impact_frame else "N/A")
            mcol2.metric("Peak Ball Speed (km/h)", peak_speed_kmh_str)
            mcol3.metric("Bat Detections", str(len(bat_pos)))
            mcol4.metric("Ball Detections", str(len(ball_pos)))
            
            # --- Shot Prediction Results ---
            st.divider()
            st.header("🏏 Cricket Shot Prediction")
            
            if prediction_error:
                st.error(f"❌ Shot prediction failed: {prediction_error}")
            elif prediction_result:
                # Main prediction result
                pred_col1, pred_col2 = st.columns([2, 1])
                
                with pred_col1:
                    predicted_shot = prediction_result['predicted_class'].replace('_', ' ').title()
                    confidence = prediction_result['confidence']
                    description = prediction_result['description']
                    
                    st.subheader(f"🎯 Predicted Shot: **{predicted_shot}**")
                    st.write(f"📝 **Description:** {description}")
                    
                    # Confidence bar
                    st.write("🎯 **Confidence Level:**")
                    st.progress(confidence, text=f"{confidence:.1%}")
                
                with pred_col2:
                    # Top 3 predictions
                    st.subheader("🏆 Top 3 Predictions")
                    for i, pred in enumerate(prediction_result['top_3_predictions'], 1):
                        shot_name = pred['class'].replace('_', ' ').title()
                        prob = pred['probability']
                        
                        # Color coding for confidence
                        if i == 1:
                            st.success(f"🥇 **{shot_name}**\n{prob:.1%}")
                        elif i == 2:
                            st.info(f"🥈 **{shot_name}**\n{prob:.1%}")
                        else:
                            st.warning(f"🥉 **{shot_name}**\n{prob:.1%}")
                
                # Detailed probability breakdown
                with st.expander("📊 Detailed Probability Breakdown"):
                    st.subheader("All Shot Probabilities")
                    
                    # Create a dataframe for better visualization
                    import pandas as pd
                    prob_data = []
                    for class_name, prob in prediction_result['all_probabilities'].items():
                        shot_name = class_name.replace('_', ' ').title()
                        prob_data.append({
                            'Shot Type': shot_name,
                            'Probability': prob,
                            'Percentage': f"{prob:.1%}"
                        })
                    
                    prob_df = pd.DataFrame(prob_data)
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Display as a nice table
                    for idx, row in prob_df.iterrows():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"**{row['Shot Type']}**")
                        with col2:
                            st.progress(row['Probability'])
                        with col3:
                            st.write(row['Percentage'])
            
            st.divider()

            # --- Main Visuals Layout ---
            col1, col2 = st.columns(2)
            with col1:
                st.header("Video Analysis")
                with open(out_video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
            
            with col2:
                st.header("Bat Swing Arc")
                cap = cv2.VideoCapture(temp_video_path)
                ret, first_frame = cap.read()
                cap.release()
                if ret:
                    bat_swing_pos = [p for p in bat_pos if p[2] > impact_frame - 30 and p[2] <= impact_frame] if impact_frame else bat_pos[-30:]
                    swing_arc_img = plot_bat_swing_arc(bat_swing_pos, first_frame)
                    st.image(swing_arc_img, channels="BGR", caption="Path of the bat just before impact (Start: Green, End: Red).")
            
            st.divider()

            col3, col4 = st.columns(2)
            with col3:
                st.header("Ball Trajectory")
                trajectory_fig = plot_ball_trajectory(ball_pos_pre_impact)
                if trajectory_fig: st.pyplot(trajectory_fig)
                else: st.warning("Could not generate a clean ball trajectory.")
            
            with col4:
                st.header("Ball Speed")
                # Use enhanced speed calculation
                smoothed_speeds_kmh, frame_array = calculate_and_smooth_speeds(ball_pos_pre_impact, fps, PIXELS_PER_METER)
                speed_fig = plot_ball_speed(smoothed_speeds_kmh, frame_array)
                if speed_fig: st.pyplot(speed_fig)
                else: st.warning("Not enough data to plot ball speed.")

            # --- Enhanced Speed Statistics (NEW SECTION) ---
            if len(smoothed_speeds_kmh) > 0:
                st.divider()
                st.subheader("📊 Enhanced Speed Analytics")
                
                speed_col1, speed_col2, speed_col3 = st.columns(3)
                
                with speed_col1:
                    st.metric("Average Speed", f"{np.mean(smoothed_speeds_kmh):.1f} km/h")
                    st.metric("Speed Std Dev", f"{np.std(smoothed_speeds_kmh):.1f} km/h")
                
                with speed_col2:
                    st.metric("Minimum Speed", f"{np.min(smoothed_speeds_kmh):.1f} km/h")
                    st.metric("Maximum Speed", f"{np.max(smoothed_speeds_kmh):.1f} km/h")
                
                with speed_col3:
                    st.metric("Median Speed", f"{np.median(smoothed_speeds_kmh):.1f} km/h")
                    st.metric("Speed Range", f"{np.max(smoothed_speeds_kmh) - np.min(smoothed_speeds_kmh):.1f} km/h")

            os.remove(temp_video_path)
>>>>>>> 55cf882ae12d7b7a383dc56a61ff28ccc63d8322
            os.remove(out_video_path)