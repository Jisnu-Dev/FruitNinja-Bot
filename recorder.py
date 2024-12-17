import torch
import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO


# Function to check if the system is using GPU or CPU
def check_device():
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")


# Load the YOLO model and set the device
device = check_device()  # This will automatically select GPU or CPU
model = YOLO('best.pt').to(device)  # Ensure the model is moved to the selected device

# Set screen capture dimensions
screen_width, screen_height = pyautogui.size()
capture_region = (0, 0, screen_width, screen_height)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (screen_width, screen_height))


# Define frame processing function
def process_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 640))  # Resize to YOLOv8's input size
    img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
    img_tensor = torch.tensor(img_resized).float() / 255.0  # Normalize
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to GPU/CPU

    # Perform inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Post-process predictions
    fruits = []
    bombs = []
    for pred in predictions[0].boxes:
        bbox = pred.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = bbox
        label = int(pred.cls[0].cpu().numpy())

        # Scale bbox to screen size
        x1 = int(x1 * screen_width / 640)
        y1 = int(y1 * screen_height / 640)
        x2 = int(x2 * screen_width / 640)
        y2 = int(y2 * screen_height / 640)

        if label == 0:  # Assuming 0 is the label for fruits
            fruits.append((x1, y1, x2, y2))
        elif label == 1:  # Assuming 1 is the label for bombs
            bombs.append((x1, y1, x2, y2))

    return fruits, bombs


# Function to simulate slicing by dragging the cursor
def slice_fruits(fruits):
    if not fruits:
        return

    # Sort fruits by y-coordinate to group those that are closer together
    fruits.sort(key=lambda x: (x[1] + x[3]) // 2)  # Sort by vertical center of the fruit

    # Grouping fruits that are close enough
    slices = []
    current_slice = [fruits[0]]

    for i in range(1, len(fruits)):
        # Calculate the distance between current fruit and the previous one
        prev_x1, prev_y1, prev_x2, prev_y2 = current_slice[-1]
        curr_x1, curr_y1, curr_x2, curr_y2 = fruits[i]

        # Check if the fruits are close horizontally
        if abs(curr_y1 - prev_y1) < 100:  # You can adjust this threshold
            current_slice.append(fruits[i])
        else:
            slices.append(current_slice)
            current_slice = [fruits[i]]

    # Append the last group of fruits
    if current_slice:
        slices.append(current_slice)

    # Perform slicing on each group of fruits
    for slice_group in slices:
        # Find the center of the slice group
        slice_center_x = sum([(x1 + x2) // 2 for x1, y1, x2, y2 in slice_group]) // len(slice_group)
        slice_center_y = sum([(y1 + y2) // 2 for x1, y1, x2, y2 in slice_group]) // len(slice_group)

        # Start and end points for longer cuts
        slice_start_x = max(slice_center_x - 150, 0)
        slice_start_y = max(slice_center_y - 200, 0)
        slice_end_x = min(slice_center_x + 150, screen_width)
        slice_end_y = min(slice_center_y + 200, screen_height)

        # Simulate dragging for slicing
        pyautogui.moveTo(slice_start_x, slice_start_y, duration=0.05)  # Move to starting point
        pyautogui.mouseDown(button='left')  # Simulate pressing the mouse
        pyautogui.moveTo(slice_end_x, slice_end_y, duration=0.1)  # Drag to the endpoint
        pyautogui.mouseUp(button='left')  # Release the mouse


# Start screen capture and processing
while True:
    # Capture screen
    screen = pyautogui.screenshot(region=capture_region)
    frame = np.array(screen)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Process the frame
    fruits, bombs = process_frame(frame)

    # Simulate slicing by dragging the cursor
    slice_fruits(fruits)

    # Save the frame to video
    out.write(frame)

    # Exit logic
    if pyautogui.position() == (0, 0):  # Move cursor to (0, 0) to stop
        break

# Release video writer resources
out.release()
