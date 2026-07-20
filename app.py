import math
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import random
import string

st.set_page_config(page_title="MindBuzz Tracing AI", layout="wide")


class MindBuzzCNN(nn.Module):
    def __init__(self):
        super(MindBuzzCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 27)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@st.cache_resource
def load_model():
    model = MindBuzzCNN()
    model.load_state_dict(torch.load("robust_cnn.pth", map_location="cpu"))
    model.eval()
    return model


model = load_model()


def predict_image(img_array):
    tensor_img = torch.tensor(img_array, dtype=torch.float32) / 255.0
    tensor_img = (tensor_img - 0.1307) / 0.3081
    tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor_img)
        prediction = torch.argmax(output, dim=1).item()
        letter = chr(prediction + 96).upper()

        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = probabilities[0][prediction].item() * 100

    return letter, confidence


def analyze_error_location(high_res_img, predicted_letter):
    """High-resolution shape feedback for the tracing app."""
    hierarchy_check, hierarchy = cv2.findContours(high_res_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    has_internal_hole = False

    if hierarchy is not None:
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][3] != -1:
                has_internal_hole = True
                break

    letters_with_holes = ["A", "B", "D", "O", "P", "Q", "R"]
    if predicted_letter in letters_with_holes and not has_internal_hole:
        return high_res_img, None, f"Make sure you close the lines to form a complete {predicted_letter}!"

    template = np.zeros((280, 280), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(predicted_letter, font, 8, 20)[0]
    text_x = (280 - text_size[0]) // 2
    text_y = (280 + text_size[1]) // 2
    cv2.putText(template, predicted_letter, (text_x, text_y), font, 8, 255, 20)

    user_contours, _ = cv2.findContours(high_res_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not user_contours:
        return None, None, "Draw something!"

    largest_user_contour = max(user_contours, key=cv2.contourArea)
    ux, uy, uw, uh = cv2.boundingRect(largest_user_contour)

    centered_user = np.zeros((280, 280), dtype=np.uint8)
    cx, cy = (280 - uw) // 2, (280 - uh) // 2
    centered_user[cy:cy + uh, cx:cx + uw] = high_res_img[uy:uy + uh, ux:ux + uw]

    kernel = np.ones((70, 70), np.uint8)
    safe_zone = cv2.dilate(template, kernel, iterations=1)
    inverse_safe_zone = cv2.bitwise_not(safe_zone)
    errors = cv2.bitwise_and(centered_user, centered_user, mask=inverse_safe_zone)

    err_contours, _ = cv2.findContours(errors, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if err_contours:
        largest_error = max(err_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_error) > 1200:
            moments = cv2.moments(largest_error)
            if moments["m00"] != 0:
                err_cx = int(moments["m10"] / moments["m00"])
                if err_cx < 100:
                    position = "on the left"
                elif err_cx > 180:
                    position = "on the right"
                else:
                    position = "in the middle"
                return errors, largest_error, f"Your shape is bleeding out {position}. Try following the structure closer!"

    return None, None, "Perfect shape! Great job!"


def analyze_stroke_direction(json_data):
    """Checks whether the stroke appears to start from the bottom up."""
    if json_data is not None and "objects" in json_data and len(json_data["objects"]) > 0:
        first_stroke = json_data["objects"][0]
        if "path" in first_stroke:
            path = first_stroke["path"]
            if len(path) > 1 and len(path[0]) >= 3:
                start_y = path[0][2]
                all_y = [pt[2] for pt in path if len(pt) >= 3 and isinstance(pt[2], (int, float))]
                if all_y:
                    min_y = min(all_y)
                    max_y = max(all_y)
                    if start_y > (min_y + max_y) / 2 + 10:
                        return "I noticed you drew from the bottom up! Try starting from the top next time."
    return None


st.title("MindBuzz Tracing AI")
st.write("Blend digital and physical learning with tracing, physics, and memory games.")


menu = [
    "Digital Canvas",
    "Physical Paper (Webcam)",
    "🌊 Leaky Pipe Physics",
    "🏎️ Velocity Tracker",
    "🧠 Memory Mode"
]

choice = st.sidebar.radio("Select a Game Mode:", menu)

if choice == "Digital Canvas":
    st.subheader("Draw on the screen")
    st.write("Trace a letter and let the AI give feedback on shape and confidence.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_digital",
    )

    if st.button("Evaluate Digital Drawing", key="btn_digital"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            high_res_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            resized = cv2.resize(high_res_gray, (28, 28))

            letter, confidence = predict_image(resized)
            telemetry_feedback = analyze_stroke_direction(canvas_result.json_data)
            error_mask, error_contour, text_feedback = analyze_error_location(high_res_gray, letter)

            st.success(f"**Predicted Letter:** {letter}")
            st.metric("AI Confidence Score", f"{confidence:.1f}%")

            if telemetry_feedback:
                st.warning(f"✍️ **Telemetry:** {telemetry_feedback}")

            if error_mask is not None:
                st.warning(f"🤖 **AI Coach:** {text_feedback}")
            else:
                st.balloons()
                st.info(f"🤖 **AI Coach:** {text_feedback}")
        else:
            st.warning("Please draw something first!")

elif choice == "Physical Paper (Webcam)":
    st.subheader("Hold your paper up to the camera")
    st.write("Draw a letter clearly on paper with a dark marker and let the camera-based mode recognize it.")
    camera_image = st.camera_input("Capture Paper")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            15,
        )

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_thresh = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            h_img, w_img = cleaned_thresh.shape
            valid_contours = []

            for c in contours:
                area = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                margin = 15
                if area > 500 and x > margin and y > margin and (x + w) < (w_img - margin) and (y + h) < (h_img - margin):
                    valid_contours.append(c)

            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                padding = 20
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(cleaned_thresh.shape[1] - x_pad, w + 2 * padding)
                h_pad = min(cleaned_thresh.shape[0] - y_pad, h + 2 * padding)
                roi = cleaned_thresh[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]

                kernel_thicken = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                roi = cv2.dilate(roi, kernel_thicken, iterations=1)

                height, width = roi.shape
                if height > width:
                    pad = (height - width) // 2
                    roi = cv2.copyMakeBorder(roi, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
                elif width > height:
                    pad = (width - height) // 2
                    roi = cv2.copyMakeBorder(roi, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

                resized_roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                st.image(resized_roi, caption="What the AI sees (preprocessed image)", width=150)

                letter, confidence = predict_image(resized_roi)
                st.success(f"**Predicted Letter from Paper:** {letter}")
                st.metric("AI Confidence Score", f"{confidence:.1f}%")
            else:
                st.warning("Found shapes, but they were too close to the edge. Center the letter!")
        else:
            st.warning("Could not detect a clear letter. Try bringing it closer to the camera!")


elif choice == "🌊 Leaky Pipe Physics":
    st.subheader("🌊 The Leaky Pipe Challenge")
    st.write("Draw a letter like O or D to build a pipe. If the shape has a gap, the water will leak out.")

    canvas_pipe = st_canvas(
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_pipe",
    )

    if st.button("Pour Water!", key="btn_pipe"):
        if canvas_pipe.image_data is not None:
            img = canvas_pipe.image_data
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            letter, _ = predict_image(cv2.resize(gray, (28, 28)))

            water_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            has_hole = False
            leak_mark = False
            if hierarchy is not None:
                for i in range(len(contours)):
                    if hierarchy[0][i][3] != -1:
                        has_hole = True
                        cv2.drawContours(water_img, contours, i, (255, 150, 0), cv2.FILLED)
                    else:
                        cv2.drawContours(water_img, contours, i, (180, 180, 180), 2)

            letters_with_holes = ["A", "B", "D", "O", "P", "Q", "R"]
            if letter in letters_with_holes and not has_hole:
                leak_mark = True

            col1, col2 = st.columns(2)
            with col1:
                st.image(water_img, caption="Physics Simulation", width=200)
            with col2:
                if leak_mark:
                    st.error("Oh no! The water leaked out! Close the shape to keep the pipe sealed. 🌊")
                elif letter in letters_with_holes and has_hole:
                    st.success("Great job! The pipe held the water perfectly! 💧")
                    st.balloons()
                else:
                    st.info(f"You drew a '{letter}'. Try drawing a letter with a loop, like an 'O' or 'D'!")
        else:
            st.warning("Please draw something first!")


elif choice == "🏎️ Velocity Tracker":
    st.subheader("🏎️ The Velocity Track")
    st.write("Draw a line and the app will turn it into a speed test. Smooth strokes give a boost, shaky ones become a speed bump.")
    canvas_speed = st_canvas(
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_speed",
    )

    if st.button("Run Race!", key="btn_speed"):
        if canvas_speed.json_data is not None and "objects" in canvas_speed.json_data:
            objects = canvas_speed.json_data["objects"]
            if len(objects) > 0 and "path" in objects[0]:
                path = objects[0]["path"]
                x_coords, y_coords, speeds = [], [], []

                for i in range(1, len(path)):
                    if len(path[i]) >= 3 and len(path[i - 1]) >= 3:
                        x1, y1 = path[i - 1][1], path[i - 1][2]
                        x2, y2 = path[i][1], path[i][2]
                        if isinstance(x1, (int, float)) and isinstance(y1, (int, float)):
                            x_coords.append(x1)
                            y_coords.append(-y1)
                            dist = math.hypot(x2 - x1, y2 - y1)
                            speeds.append(dist)

                if len(speeds) > 0:
                    avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0
                    slow_segments = sum(1 for s in speeds if s < max(avg_speed * 0.6, 3))
                    fast_segments = sum(1 for s in speeds if s > avg_speed * 1.2)

                    if fast_segments > slow_segments:
                        feedback = "Boost! Your stroke stayed smooth and steady."
                    elif slow_segments > fast_segments:
                        feedback = "Speed bump! Try smoothing out the line and keeping it steady."
                    else:
                        feedback = "Nice pace. A little more consistency would make the ride smoother."

                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.plot(x_coords, y_coords, color="white", linewidth=2)
                    ax.scatter(x_coords, y_coords, c=speeds, cmap="RdYlGn", s=50)
                    ax.scatter([x_coords[-1]], [y_coords[-1]], color="yellow", s=120, marker="s")
                    ax.text(x_coords[-1], y_coords[-1], "🚗", fontsize=18, ha="center", va="center")
                    ax.set_facecolor("black")
                    ax.axis("off")

                    st.image(canvas_speed.image_data, width=150, caption="Your Drawing")
                    st.pyplot(fig)
                    st.info(feedback)
                else:
                    st.warning("Please draw a longer line to race along!")
            else:
                st.warning("Please draw something first!")
        else:
            st.warning("Please draw a long, continuous line first!")



elif choice == "🧠 Memory Mode":
    st.subheader("🧠 Memory Mode")
    st.write("The letter flashes for 3 seconds, then disappears. Try drawing it from memory.")

    # 1. Initialize variables (Removed 'has_flashed', using three clean phases instead)
    if "memory_phase" not in st.session_state:
        st.session_state.memory_phase = "idle"
    if "current_target" not in st.session_state:
        st.session_state.current_target = "A"

    # ---------------------------------------------------------
    # PHASE 1: IDLE (Generate Random Letter)
    # ---------------------------------------------------------
    if st.session_state.memory_phase == "idle":
        st.write("Are you ready? Click below, and the AI will challenge you with a random letter!")
        
        if st.button("Start Random Memory Round", use_container_width=True):
            import string
            import random
            
            # Make the app choose randomly from A-Z
            all_letters = list(string.ascii_uppercase) 
            st.session_state.current_target = random.choice(all_letters) 
            
            # Move to the FLASH phase
            st.session_state.memory_phase = "flash"
            st.rerun() 

    # ---------------------------------------------------------
    # PHASE 2: FLASH (Show letter, wait 3s, then force a reload)
    # ---------------------------------------------------------
    elif st.session_state.memory_phase == "flash":
        st.info("Memorize the letter before it disappears...")
        
        # THE FIX: Changed color to #FF4B4B (Red) and added font-weight: bold
        st.markdown(
            f"<div style='font-size: 180px; text-align: center; color: #FF4B4B; font-weight: bold;'>{st.session_state.current_target}</div>",
            unsafe_allow_html=True,
        )
        
        # Pause the backend for 3 seconds while the browser shows the letter
        time.sleep(3)
        
        # Move to the DRAW phase and force a hard reload so the letter vanishes
        st.session_state.memory_phase = "draw"
        st.rerun()

    # ---------------------------------------------------------
    # PHASE 3: DRAW (Evaluate the drawing)
    # ---------------------------------------------------------
    elif st.session_state.memory_phase == "draw":
        st.warning("Target hidden! Draw the letter from memory without guides.")
        
        canvas_memory = st_canvas(
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas_memory",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Evaluate Memory Test"):
                if canvas_memory.image_data is not None:
                    img = canvas_memory.image_data
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                    resized = cv2.resize(gray, (28, 28))
                    
                    predicted_letter, _ = predict_image(resized)

                    if predicted_letter == st.session_state.current_target:
                        st.success(f"**Incredible Memory!** You perfectly drew the '{st.session_state.current_target}'! 🏆")
                        st.balloons()
                    else:
                        st.error(f"Almost! You aimed for '{st.session_state.current_target}', but the AI thought it looked more like a '{predicted_letter}'. Try again!")
                else:
                    st.warning("Please draw something first!")

        with col2:
            if st.button("Play Again"):
                # Reset back to phase 1
                st.session_state.memory_phase = "idle"
                st.rerun()
    
    

    
