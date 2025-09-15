import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Set page configuration
st.set_page_config(
    page_title="Real-Time Hand Gesture Recognition",
    page_icon="✋",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
        color: #FF4B4B;
    }
    .confidence-text {
        font-size: 20px !important;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.75
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=15)
if 'current_gesture' not in st.session_state:
    st.session_state.current_gesture = "None"
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 0.0
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = time.time()
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'hands' not in st.session_state:
    st.session_state.hands = None

# App title
st.title("Real-Time Hand Gesture Recognition")

# Sidebar
with st.sidebar:
    st.header("Controls")

    # Camera on/off toggle
    st.session_state.camera_on = st.toggle("Start Camera", value=st.session_state.camera_on)

    # Debug mode toggle
    st.session_state.debug_mode = st.toggle("Debug Mode", value=st.session_state.debug_mode)

    # Confidence threshold slider
    st.session_state.confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_threshold,
        step=0.05
    )

    # Model loading status
    st.subheader("Model Status")
    if st.session_state.model is not None:
        st.success("Model loaded successfully")
    else:
        st.error("Model not loaded")

    if st.session_state.class_names is not None:
        st.write(f"Classes: {', '.join(st.session_state.class_names)}")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Live Camera Feed")
    frame_placeholder = st.empty()
    st.session_state.frame_placeholder = frame_placeholder

    # Debug view if enabled
    if st.session_state.debug_mode:
        st.subheader("Debug View")
        debug_placeholder = st.empty()

# with col2:
#     st.subheader("Recognition Results")
#     st.markdown(f'<p class="big-font">Gesture: {st.session_state.current_gesture}</p>',
#                 unsafe_allow_html=True)
#     st.markdown(f'<p class="confidence-text">Confidence: {st.session_state.current_confidence:.2f}</p>',
#                 unsafe_allow_html=True)
#
#     # Display prediction history
#     st.subheader("Prediction History")
#     if st.session_state.prediction_history:
#         history_text = ""
#         for gesture, confidence in list(st.session_state.prediction_history)[-5:]:
#             history_text += f"{gesture} ({confidence:.2f})\n"
#         st.text(history_text)
#     else:
#         st.text("No predictions yet")


# Initialize MediaPipe and model
def initialize_models():
    try:
        # Load the trained model
        if st.session_state.model is None:
            st.session_state.model = tf.keras.models.load_model('model')
            st.success("Model loaded successfully")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.session_state.camera_on = False
        return False

    try:
        # Load class names
        if st.session_state.class_names is None:
            st.session_state.class_names = np.load('class_names.npy', allow_pickle=True).tolist()
            st.sidebar.write(f"Classes: {', '.join(st.session_state.class_names)}")
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        st.session_state.camera_on = False
        return False

    # Initialize MediaPipe Hands
    if st.session_state.hands is None:
        mp_hands = mp.solutions.hands
        st.session_state.hands = mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )

    return True


# Image processing functions
def extract_hand_roi(landmarks, frame_shape):
    """Extract bounding box around hand landmarks with better padding"""
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]

    x_min = int(min(x_coords) * frame_shape[1])
    y_min = int(min(y_coords) * frame_shape[0])
    x_max = int(max(x_coords) * frame_shape[1])
    y_max = int(max(y_coords) * frame_shape[0])

    # Add proportional padding
    width = x_max - x_min
    height = y_max - y_min
    padding_x = int(width * 0.2)  # 20% padding
    padding_y = int(height * 0.2)

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(frame_shape[1], x_max + padding_x)
    y_max = min(frame_shape[0], y_max + padding_y)

    # Ensure minimum size
    min_size = 50
    if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
        # Expand to minimum size while keeping center
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        x_min = max(0, center_x - min_size // 2)
        x_max = min(frame_shape[1], center_x + min_size // 2)
        y_min = max(0, center_y - min_size // 2)
        y_max = min(frame_shape[0], center_y + min_size // 2)

    return (x_min, y_min, x_max, y_max)


def preprocess_hand_roi(roi, img_size=(224, 224)):
    """Preprocess the hand ROI for model prediction"""
    if roi.size == 0:
        return None

    # Resize to match model input size
    resized = cv2.resize(roi, img_size)

    # Apply the same preprocessing as during training
    processed = tf.keras.applications.efficientnet.preprocess_input(resized)

    # Add batch dimension
    return np.expand_dims(processed, axis=0)


def debug_display_roi(roi, gesture_label, confidence):
    """Display the processed ROI for debugging"""
    if st.session_state.debug_mode and roi is not None and roi.size > 0:
        debug_img = roi.copy()
        debug_img = cv2.resize(debug_img, (200, 200))
        cv2.putText(debug_img, f"{gesture_label} ({confidence:.2f})",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return debug_img
    return None


# Process each frame
def process_frame(frame):
    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    # Process the frame with MediaPipe Hands
    results = st.session_state.hands.process(rgb_frame)

    # Convert back to BGR for OpenCV
    rgb_frame.flags.writeable = True
    output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    gesture_label = "No hand detected"
    gesture_confidence = 0.0
    hand_detected = False

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                output_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract hand ROI
            x_min, y_min, x_max, y_max = extract_hand_roi(hand_landmarks, frame.shape)

            # Skip invalid ROI
            if x_max <= x_min or y_max <= y_min:
                continue

            # Draw bounding box around hand
            cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Extract and preprocess the hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]

            if hand_roi.size > 0 and hand_roi.shape[0] > 50 and hand_roi.shape[1] > 50:
                input_tensor = preprocess_hand_roi(hand_roi)

                if input_tensor is not None:
                    # Make prediction using the trained model
                    try:
                        predictions = st.session_state.model.predict(input_tensor, verbose=0)
                        predicted_class = int(np.argmax(predictions, axis=1)[0])
                        gesture_confidence = float(np.max(predictions))

                        # Only accept predictions with sufficient confidence
                        if gesture_confidence >= st.session_state.confidence_threshold:
                            gesture_label = st.session_state.class_names[predicted_class]
                            hand_detected = True

                            # Display debug ROI if enabled
                            if st.session_state.debug_mode:
                                debug_img = debug_display_roi(hand_roi, gesture_label, gesture_confidence)
                                if debug_img is not None:
                                    debug_placeholder.image(debug_img, channels="BGR")
                        else:
                            gesture_label = "Low confidence"

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        gesture_label = "Prediction error"

    # Stabilize predictions using a history buffer
    if hand_detected:
        st.session_state.prediction_history.append((gesture_label, gesture_confidence))
        st.session_state.last_prediction_time = time.time()
    else:
        # Clear history if no hand detected for a while
        if time.time() - st.session_state.last_prediction_time > 2.0:  # 2 seconds
            st.session_state.prediction_history.clear()
            st.session_state.current_gesture = "None"
            st.session_state.current_confidence = 0.0

    # Get the most frequent gesture with sufficient confidence
    if st.session_state.prediction_history:
        # Count occurrences of each gesture
        gesture_counts = {}
        confidence_sums = {}

        for gesture, confidence in st.session_state.prediction_history:
            if gesture in gesture_counts:
                gesture_counts[gesture] += 1
                confidence_sums[gesture] += confidence
            else:
                gesture_counts[gesture] = 1
                confidence_sums[gesture] = confidence

        # Find the most frequent gesture
        if gesture_counts:
            most_frequent = max(gesture_counts.items(), key=lambda x: x[1])

            # Only update if we have a clear majority
            if most_frequent[1] >= len(st.session_state.prediction_history) * 0.6:  # 60% majority
                st.session_state.current_gesture = most_frequent[0]
                st.session_state.current_confidence = confidence_sums[st.session_state.current_gesture] / most_frequent[
                    1]

    # Display the recognized gesture on the frame
    cv2.putText(output_frame, f"Gesture: {st.session_state.current_gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output_frame, f"Confidence: {st.session_state.current_confidence:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display hand detection status
    status_text = "Hand detected" if results.multi_hand_landmarks else "No hand detected"
    cv2.putText(output_frame, status_text, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return output_frame


# Main application logic
def main():
    # Initialize models if not already loaded
    if st.session_state.model is None or st.session_state.class_names is None:
        if not initialize_models():
            st.error("Failed to initialize models. Please check if the model files exist.")
            st.session_state.camera_on = False
            return

    # Start camera if toggled on
    if st.session_state.camera_on:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam")
            st.session_state.camera_on = False
            return

        # Set camera resolution for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Process frames from webcam
        while st.session_state.camera_on:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            # Process the frame
            processed_frame = process_frame(frame)

            # Display the processed frame
            st.session_state.frame_placeholder.image(processed_frame, channels="BGR")

            # Add a small delay to prevent high CPU usage
            time.sleep(0.01)

        # Release the camera when stopped
        cap.release()
    else:
        # Display placeholder when camera is off
        st.session_state.frame_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")


# Run the main function
if __name__ == "__main__":
    main()
