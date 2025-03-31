import time
import numpy as np
import cv2
import pyzed.sl as sl

# ====================== Configuration ======================
THRESHOLD_MAD = 10  # Ignore Difference Map if MAD is below this value (Not actually used)
FAULT_RATE_THRESHOLD = 0.1  # If the damaged pixel ratio exceeds this value, start suspicion
MAD_THRESHOLD = 10  # MAD value above this threshold may indicate damage
MAD_PERSISTENT_THRESHOLD = 15  # If MAD exceeds this threshold, damage is likely confirmed
RESET_TIME = 0.5  # Time after damage detection, reset to normal state if no issues for this period
ZERO_THRESHOLD = 10  # If difference is below this value, it is not considered damage (Not actually used)
MAX_DIFF_THRESHOLD = 85  # Ignore differences exceeding this value in the Difference Map (Sensor malfunction or external disturbance)
POSITION_DIFF_THRESHOLD = 15  # If the position difference between left and right frames exceeds this value, consider damage
NOISE_THRESHOLD = 10  # Lower bound for noise removal in the Difference Map

# ====================== Visualization Settings ======================
DISPLAY_WIDTH = 480
DISPLAY_HEIGHT = 270

# ====================== Previous Frame Comparison Variables ======================
prev_mad = None
prev_fault_rate = None
prev_position_damage_rate = None

# ====================== Lens Damage Detection Function ======================
def detect_lens_damage(left_frame, right_frame):
    # Calculate difference between left and right frames and remove noise
    diff_map = cv2.absdiff(left_frame, right_frame)
    _, diff_map = cv2.threshold(diff_map, NOISE_THRESHOLD, 255, cv2.THRESH_TOZERO)
    diff_map = cv2.GaussianBlur(diff_map, (3, 3), 0)
    diff_map = np.clip(diff_map, 0, MAX_DIFF_THRESHOLD).astype(np.uint8)
    
    # Calculate Mean Absolute Difference (MAD)
    mad_value = np.mean(diff_map)

    # Calculate the damaged pixel ratio
    damaged_pixels = (diff_map > MAD_THRESHOLD) & (diff_map < MAX_DIFF_THRESHOLD)
    fault_rate = np.sum(damaged_pixels) / (diff_map.shape[0] * diff_map.shape[1])

    # Calculate position difference-based damage ratio
    position_diff = np.abs(left_frame - right_frame)
    position_damage_pixels = (position_diff > POSITION_DIFF_THRESHOLD)
    position_damage_rate = np.sum(position_damage_pixels) / (position_diff.shape[0] * position_diff.shape[1])

    return diff_map, mad_value, fault_rate, position_damage_rate  

# ====================== ZED Camera Initial Setup ======================
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Set resolution to 720p
init_params.camera_fps = 30  # Set frame rate

runtime_params = sl.RuntimeParameters()
runtime_params.confidence_threshold = 50  # Confidence threshold for depth map

# Try to open the camera
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"Unable to open ZED camera. Error code: {status}")
    exit()

# ====================== Detection State Variables ======================
damage_confirmed = False  
damage_suspected = False  
detect_stage = 0  
damage_reset_time = time.time()  

# ====================== Main Loop ======================
while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        left_image = sl.Mat()
        right_image = sl.Mat()

        zed.retrieve_image(left_image, sl.VIEW.LEFT)  
        zed.retrieve_image(right_image, sl.VIEW.RIGHT)  

        frame_left = left_image.get_data()[:, :, :3]  # RGB
        frame_right = right_image.get_data()[:, :, :3]

        # Convert to grayscale
        left_gray = cv2.cvtColor(frame_left, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(frame_right, cv2.COLOR_RGB2GRAY)

        # Resize images for visualization
        left_image_resized = cv2.resize(frame_left, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        right_image_resized = cv2.resize(frame_right, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        cv2.imshow("Left Image", left_image_resized)
        cv2.imshow("Right Image", right_image_resized)

        # Lens damage detection
        diff_map, mad_value, fault_rate, position_damage_rate = detect_lens_damage(left_gray, right_gray)

        print("-" * 50)
        print(f"Mean Absolute Difference (MAD): {mad_value:.2f}")
        print(f"Fault Rate: {fault_rate:.4f}")
        print(f"Position Damage Rate: {position_damage_rate:.4f}")

        # ---------------- Stage 1 Detection ----------------
        if detect_stage == 0:
            if diff_map is not None and mad_value > MAD_THRESHOLD and fault_rate > FAULT_RATE_THRESHOLD:
                damage_suspected = True
                detect_stage = 1
                prev_mad, prev_fault_rate, prev_position_damage_rate = mad_value, fault_rate, position_damage_rate

        # ---------------- Stage 2 Detection ----------------
        elif detect_stage == 1:
            if damage_suspected:
                mad_change = (mad_value - prev_mad) / prev_mad if prev_mad > 0 else 0
                fault_rate_change = (fault_rate - prev_fault_rate) / prev_fault_rate if prev_fault_rate > 0 else 0
                position_damage_change = (position_damage_rate - prev_position_damage_rate) / prev_position_damage_rate if prev_position_damage_rate > 0 else 0

                # Conditions for damage confirmation
                if (mad_value > MAD_PERSISTENT_THRESHOLD or mad_change > 0.5) and \
                   (fault_rate > FAULT_RATE_THRESHOLD or fault_rate_change > 0.5) and \
                   (position_damage_rate > 0.6 or position_damage_change > 0.2):
                    damage_confirmed = True
                    detect_stage = 0  
                    damage_reset_time = time.time()  

        # ---------------- Damage Reset Condition ----------------
        if damage_confirmed:
            if time.time() - damage_reset_time > RESET_TIME and \
               (mad_value < MAD_THRESHOLD or fault_rate < FAULT_RATE_THRESHOLD or position_damage_rate < 0.6):
                damage_confirmed = False  
                damage_suspected = False  
                detect_stage = 0  # Reset for possible re-detection

        # ---------------- Status Output ----------------
        if damage_confirmed:
            print("Status: << Camera Fault Detected !!!>>")
        else:
            print("Status: Lens OK")

        # ---------------- Difference Map Visualization ----------------
        if diff_map is not None:
            diff_map_normalized = cv2.normalize(diff_map, None, 50, 255, cv2.NORM_MINMAX)
            diff_map_colored = cv2.applyColorMap(diff_map_normalized, cv2.COLORMAP_TURBO)
            diff_map_resized = cv2.resize(diff_map_colored, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow("Difference Map", diff_map_resized)

        # Exit program on pressing ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            print("Exiting program.")
            break

# ====================== Shutdown ======================
zed.close()
cv2.destroyAllWindows()

