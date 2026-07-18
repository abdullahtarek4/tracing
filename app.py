import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

# 1. Re-initialize the Model Structure
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

# 2. Load the Weights
@st.cache_resource
def load_model():
    model = MindBuzzCNN()
    model.load_state_dict(torch.load('robust_cnn.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# --- Helper Function for Prediction ---
def predict_image(img_array):
    # Normalize to PyTorch tensor format
    tensor_img = torch.tensor(img_array, dtype=torch.float32) / 255.0
    tensor_img = (tensor_img - 0.1307) / 0.3081
    tensor_img = tensor_img.unsqueeze(0).unsqueeze(0) # Shape: [1, 1, 28, 28]
    
    with torch.no_grad():
        output = model(tensor_img)
        prediction = torch.argmax(output, dim=1).item()
        letter = chr(prediction + 96).upper()
        
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = probabilities[0][prediction].item() * 100
        
    return letter, confidence

def analyze_error_location(high_res_img, predicted_letter):
    """High-Res Geometric and Topological Error Engine"""
    
    # --- 1. TOPOLOGY CHECK (The Gap Finder) ---
    # We check if the drawing has internal closed loops (holes)
    hierarchy_check, hierarchy = cv2.findContours(high_res_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    has_internal_hole = False
    
    if hierarchy is not None:
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][3] != -1: # A contour with a parent is an internal hole!
                has_internal_hole = True
                break

    letters_with_holes = ['A', 'B', 'D', 'O', 'P', 'Q', 'R']
    if predicted_letter in letters_with_holes and not has_internal_hole:
        return high_res_img, None, f"Make sure you close the lines to form a complete {predicted_letter}!"

    # --- 2. HIGH-RES GEOMETRIC CHECK (The Flip & Shape Finder) ---
    # Create a perfect 280x280 template
    template = np.zeros((280, 280), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(predicted_letter, font, 8, 20)[0]
    text_x = (280 - text_size[0]) // 2
    text_y = (280 + text_size[1]) // 2
    cv2.putText(template, predicted_letter, (text_x, text_y), font, 8, 255, 20)
    
    # Center the user's drawing so it perfectly aligns with the template
    user_contours, _ = cv2.findContours(high_res_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not user_contours: return None, None, "Draw something!"
    
    largest_user_contour = max(user_contours, key=cv2.contourArea)
    ux, uy, uw, uh = cv2.boundingRect(largest_user_contour)
    
    centered_user = np.zeros((280, 280), dtype=np.uint8)
    cx, cy = (280 - uw) // 2, (280 - uh) // 2
    centered_user[cy:cy+uh, cx:cx+uw] = high_res_img[uy:uy+uh, ux:ux+uw]

    # Create a strict Safe Zone
    kernel = np.ones((70, 70), np.uint8) 
    safe_zone = cv2.dilate(template, kernel, iterations=1)
    
    # Find errors bleeding outside the safe zone
    inverse_safe_zone = cv2.bitwise_not(safe_zone)
    errors = cv2.bitwise_and(centered_user, centered_user, mask=inverse_safe_zone)
    
    err_contours, _ = cv2.findContours(errors, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if err_contours:
        largest_error = max(err_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_error) > 1200: 
            M = cv2.moments(largest_error)
            if M["m00"] != 0:
                err_cx = int(M["m10"] / M["m00"])
                if err_cx < 100: position = "on the left"
                elif err_cx > 180: position = "on the right"
                else: position = "in the middle"
                
                return errors, largest_error, f"Your shape is bleeding out {position}. Try following the structure closer!"
                
    return None, None, "Perfect shape! Great job!"

def analyze_stroke_direction(json_data):
    """Parses the raw canvas JSON to determine if the child drew bottom-to-top."""
    if json_data is not None and "objects" in json_data and len(json_data["objects"]) > 0:
        first_stroke = json_data["objects"][0]
        if "path" in first_stroke:
            path = first_stroke["path"]
            # Fabric.js paths usually start with ['M', x, y] (Move To)
            if len(path) > 1 and len(path[0]) >= 3:
                start_y = path[0][2] 
                
                # Find all Y coordinates in the stroke path
                all_y = [pt[2] for pt in path if len(pt) >= 3 and isinstance(pt[2], (int, float))]
                if all_y:
                    min_y = min(all_y) # Highest point on screen (y=0 is top)
                    max_y = max(all_y) # Lowest point on screen
                    
                    # If the starting point is in the bottom half of the drawing
                    if start_y > (min_y + max_y) / 2 + 10:
                        return "I noticed you drew from the bottom up! Try starting from the top next time."
    return None
# 3. Build the UI
st.title("MindBuzz Tracing AI")
st.write("Blend digital and physical learning!")

# Create Tabs for the two modes
tab1, tab2 = st.tabs(["Digital Canvas", "Physical Paper (Webcam)"])

with tab1:
    st.subheader("Draw on the Screen")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Evaluate Digital Drawing"):
        if canvas_result.image_data is not None:
            # 1. Image Prep
            img = canvas_result.image_data
            high_res_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY) # Keep 280x280 for error engine
            resized = cv2.resize(high_res_gray, (28, 28))          # Shrink for CNN prediction
            
            # 2. Get CNN Prediction
            letter, confidence = predict_image(resized)
            
            # 3. Run the Telemetry & Error Analysis Engines
            telemetry_feedback = analyze_stroke_direction(canvas_result.json_data)
            error_mask, error_contour, text_feedback = analyze_error_location(high_res_gray, letter)
            
            # --- UI OUTPUT ---
            st.success(f"**Predicted Letter:** {letter}")
            st.metric(label="AI Confidence Score", value=f"{confidence:.1f}%")
            
            # Show Stroke Order Warning if detected
            if telemetry_feedback:
                st.warning(f"✍️ **Telemetry:** {telemetry_feedback}")
            
            # Display Visual Error Feedback
            if error_mask is not None:
                st.warning(f"🤖 **AI Coach:** {text_feedback}")
            else:
                st.balloons() 
                st.info(f"🤖 **AI Coach:** {text_feedback}")

with tab2:
    st.subheader("Hold your paper up to the camera!")
    st.write("Draw a letter clearly on a piece of paper using a dark marker.")
    
    camera_image = st.camera_input("Capture Paper")
    
    if camera_image is not None:
        # Convert the uploaded image buffer to an OpenCV image
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
# --- THE TUNED PREPROCESSING PIPELINE ---
        
        # 1. Grayscale
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        # 2. Noise Removal 
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # 3. Tuned Adaptive Thresholding 
        # Increasing block size to 21 and the 'C' constant to 15 (up from 5).
        # This tells the AI: "Only keep pixels that are SIGNIFICANTLY darker than their surroundings."
        # This aggressively filters out faint notebook lines while keeping the thick marker.
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 15
        )
        
        # 4. Morphological Operations
        # Step A: CLOSE (Dilate then Erode) with a larger kernel to stitch broken parts of the letter together
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        
        # Step B: OPEN (Erode then Dilate) with a smaller kernel to wipe away tiny isolated noise specks
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_thresh = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
# 5. Edge Detection / Contour Finding
        contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            h_img, w_img = cleaned_thresh.shape
            valid_contours = []
            
            for c in contours:
                area = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                
                # Rule 1: Must be large enough (removes tiny specks)
                if area > 500:
                    # Rule 2: EDGE REJECTION (Fixes the spiral binding issue!)
                    # Ignore any shape that touches the outer 15 pixels of the camera frame
                    margin = 15
                    if x > margin and y > margin and (x+w) < (w_img - margin) and (y+h) < (h_img - margin):
                        valid_contours.append(c)
            
            if valid_contours:
                # Now find the largest valid contour inside the safe zone
                largest_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Extract Region of Interest (ROI) with padding
                padding = 20
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(cleaned_thresh.shape[1] - x_pad, w + 2*padding)
                h_pad = min(cleaned_thresh.shape[0] - y_pad, h + 2*padding)
                roi = cleaned_thresh[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                
                # FIX FOR LOW CONFIDENCE: Digitally thicken the pen strokes to match EMNIST!
                kernel_thicken = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                roi = cv2.dilate(roi, kernel_thicken, iterations=1)
                
                # Make it square to preserve aspect ratio
                height, width = roi.shape
                if height > width:
                    pad = (height - width) // 2
                    roi = cv2.copyMakeBorder(roi, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
                elif width > height:
                    pad = (width - height) // 2
                    roi = cv2.copyMakeBorder(roi, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
                    
                resized_roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                
                st.image(resized_roi, caption="What the AI sees (Preprocessed Image)", width=150)
                
                letter, confidence = predict_image(resized_roi)
                
                st.success(f"**Predicted Letter from Paper:** {letter}")
                st.metric(label="AI Confidence Score", value=f"{confidence:.1f}%")
            else:
                st.warning("Found shapes, but they were too close to the edge. Center the letter!")
        else:
            st.warning("Could not detect a clear letter. Try bringing it closer to the camera!")