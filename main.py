"""
Gesture-Zoom-cv
Real-Time Gesture-Controlled Image Manipulation Engine
Author: Cholan Kinnera
Powered by MediaPipe Tasks API and OpenCV.
Model: hand_landmarker.task
"""
import argparse
import math
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand Landmark connections for visual skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


def get_hand_center(landmarks_px):
    """
    Computes a stable, palm/wrist-independent hand center
    using the average of Index MCP (landmark 5) and Pinky MCP (landmark 17).
    """
    cx = (landmarks_px[5][0] + landmarks_px[17][0]) // 2
    cy = (landmarks_px[5][1] + landmarks_px[17][1]) // 2
    return (cx, cy)


def check_hand_gesture(landmarks_px, handedness="Hand"):
    """
    Checks if a hand satisfies the zoom gesture:
    - Thumb: EXTENDED outward
    - Index finger: EXTENDED outward
    - Middle, Ring, Pinky fingers: FOLDED into palm
    
    Uses landmark geometry relative to palm center and joints,
    independent of rotation, camera orientation, or tilt.
    """
    wrist = landmarks_px[0]
    p_center = get_hand_center(landmarks_px)

    def dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    palm_span = max(10.0, dist(landmarks_px[5], landmarks_px[17]))

    # --- 1. Index Finger (Tip: 8, PIP: 6, MCP: 5) ---
    d_tip_center_idx = dist(landmarks_px[8], p_center)
    d_pip_center_idx = dist(landmarks_px[6], p_center)
    d_tip_wrist_idx = dist(landmarks_px[8], wrist)
    d_pip_wrist_idx = dist(landmarks_px[6], wrist)
    index_extended = (d_tip_center_idx > d_pip_center_idx * 1.1) and (d_tip_wrist_idx > d_pip_wrist_idx)

    # --- 2. Middle, Ring, Pinky Fingers (Must be curled / folded) ---
    # Middle: Tip 12, PIP 10
    d_tip_center_mid = dist(landmarks_px[12], p_center)
    d_pip_center_mid = dist(landmarks_px[10], p_center)
    d_tip_wrist_mid = dist(landmarks_px[12], wrist)
    d_pip_wrist_mid = dist(landmarks_px[10], wrist)
    middle_folded = (d_tip_center_mid < d_pip_center_mid * 1.15) or (d_tip_wrist_mid < d_pip_wrist_mid * 1.05)

    # Ring: Tip 16, PIP 14
    d_tip_center_ring = dist(landmarks_px[16], p_center)
    d_pip_center_ring = dist(landmarks_px[14], p_center)
    d_tip_wrist_ring = dist(landmarks_px[16], wrist)
    d_pip_wrist_ring = dist(landmarks_px[14], wrist)
    ring_folded = (d_tip_center_ring < d_pip_center_ring * 1.15) or (d_tip_wrist_ring < d_pip_wrist_ring * 1.05)

    # Pinky: Tip 20, PIP 18
    d_tip_center_pinky = dist(landmarks_px[20], p_center)
    d_pip_center_pinky = dist(landmarks_px[18], p_center)
    d_tip_wrist_pinky = dist(landmarks_px[20], wrist)
    d_pip_wrist_pinky = dist(landmarks_px[18], wrist)
    pinky_folded = (d_tip_center_pinky < d_pip_center_pinky * 1.15) or (d_tip_wrist_pinky < d_pip_wrist_pinky * 1.05)

    # --- 3. Thumb (Tip: 4, IP: 3, MCP: 2) ---
    t_tip = landmarks_px[4]
    t_ip = landmarks_px[3]
    t_mcp = landmarks_px[2]
    p_mcp = landmarks_px[17]

    d_tip_tmcp = dist(t_tip, t_mcp)
    d_ip_tmcp = dist(t_ip, t_mcp)
    d_tip_pmcp = dist(t_tip, p_mcp)
    d_ip_pmcp = dist(t_ip, p_mcp)
    d_tip_pcenter = dist(t_tip, p_center)

    thumb_straight = d_tip_tmcp > d_ip_tmcp * 1.05
    thumb_away = (d_tip_pmcp > d_ip_pmcp * 1.02) or (d_tip_pcenter > palm_span * 0.65)
    thumb_extended = thumb_straight and thumb_away

    is_ready = index_extended and thumb_extended and middle_folded and ring_folded and pinky_folded

    status = {
        "handedness": handedness,
        "is_ready": is_ready,
        "thumb": thumb_extended,
        "index": index_extended,
        "middle": not middle_folded,
        "ring": not ring_folded,
        "pinky": not pinky_folded
    }
    return is_ready, status


def overlay_image_safe(frame, overlay, center_x, center_y):
    """
    Overlays 'overlay' onto 'frame' centered at (center_x, center_y).
    Safely clips edges so image stays visible even when partially off-screen or
    larger than the webcam frame itself.
    """
    h_ov, w_ov = overlay.shape[:2]
    h_frame, w_frame = frame.shape[:2]

    x1 = int(round(center_x - w_ov / 2.0))
    y1 = int(round(center_y - h_ov / 2.0))
    x2 = x1 + w_ov
    y2 = y1 + h_ov

    fx1 = max(0, x1)
    fy1 = max(0, y1)
    fx2 = min(w_frame, x2)
    fy2 = min(h_frame, y2)

    if fx2 <= fx1 or fy2 <= fy1:
        return frame

    ox1 = fx1 - x1
    oy1 = fy1 - y1
    ox2 = ox1 + (fx2 - fx1)
    oy2 = oy1 + (fy2 - fy1)

    frame[fy1:fy2, fx1:fx2] = overlay[oy1:oy2, ox1:ox2]
    return frame


def render_hud(frame, is_active, zoom_factor, pan_pos, num_hands, current_dist, initial_dist):
    """
    Renders a compact, presentation-quality HUD banner with translucent styling.
    """
    hud_w, hud_h = 490, 102
    x0, y0 = 15, 15

    # Translucent card background
    sub_img = frame[y0:y0 + hud_h, x0:x0 + hud_w]
    dark_rect = sub_img.copy()
    cv2.rectangle(dark_rect, (0, 0), (hud_w, hud_h), (25, 25, 25), -1)
    cv2.addWeighted(dark_rect, 0.78, sub_img, 0.22, 0, sub_img)
    frame[y0:y0 + hud_h, x0:x0 + hud_w] = sub_img

    # Rounded-feel border
    cv2.rectangle(frame, (x0, y0), (x0 + hud_w, y0 + hud_h), (70, 70, 70), 1)

    # Row 1: Status indicator pill
    if is_active:
        badge_color = (40, 220, 80)    # Green
        status_text = "ZOOM & PAN ACTIVE"
    elif num_hands >= 2:
        badge_color = (0, 190, 255)    # Orange
        status_text = "READY - MAKE GESTURE (THUMB+INDEX)"
    elif num_hands == 1:
        badge_color = (0, 190, 255)
        status_text = "1 HAND DETECTED - SHOW 2ND HAND"
    else:
        badge_color = (160, 160, 160)  # Gray
        status_text = "IDLE - SHOW 2 HANDS TO INTERACT"

    # Status circle & text
    cv2.circle(frame, (x0 + 16, y0 + 22), 6, badge_color, -1)
    cv2.putText(frame, status_text, (x0 + 30, y0 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    # Row 2: Metrics (Zoom percentage, Pan position, Hand distance)
    zoom_pct = int(round(zoom_factor * 100))
    dist_str = f"{int(current_dist)}px" if current_dist is not None else "--"
    pan_x, pan_y = int(pan_pos[0]), int(pan_pos[1])

    metrics_text = f"Zoom: {zoom_pct}%   Dist: {dist_str}   Pan: ({pan_x}, {pan_y})"
    cv2.putText(frame, metrics_text, (x0 + 15, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)

    # Row 3: Quick Controls Guide
    help_text = "[r] Reset   [+/-] Zoom   [f] Fullscreen   [q] Quit"
    cv2.putText(frame, help_text, (x0 + 15, y0 + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 200, 160), 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gesture-Zoom-cv: Real-Time Hand Gesture Controlled Image Manipulation Engine"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="one.jpg",
        help="Path to the target image to be manipulated (default: one.jpg)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="hand_landmarker.task",
        help="Path to the MediaPipe HandLandmarker task model (default: hand_landmarker.task)"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Webcam stream capture width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Webcam stream capture height (default: 720)"
    )
    return parser.parse_args()


def main(cli_args=None):
    if cli_args is None:
        cli_args = parse_args()

    # 1. Load image once
    img_path = cli_args.image
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Target image '{img_path}' not found.")
    img_base = cv2.imread(img_path)
    if img_base is None:
        raise ValueError(f"Failed to read image '{img_path}'.")
    h_orig, w_orig = img_base.shape[:2]
    aspect_ratio = w_orig / float(h_orig)

    # 2. Initialize MediaPipe Tasks HandLandmarker
    model_path = cli_args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model '{model_path}' not found. Download from: "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 3. Setup Webcam
    cap = cv2.VideoCapture(cli_args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cli_args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cli_args.height)

    # Window configuration
    window_name = "Gesture-Zoom-cv [Cholan Kinnera]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    is_fullscreen = False

    # State variables
    current_zoom = 1.0       # 1.0 = 100%
    target_zoom = 1.0
    MIN_ZOOM = 0.35          # 35%
    MAX_ZOOM = 3.5           # 350%
    ZOOM_SMOOTH_ALPHA = 0.30 # Exponential smoothing factor for zoom

    # Pan / position state
    pan_x = 640.0
    pan_y = 360.0
    target_pan_x = 640.0
    target_pan_y = 360.0
    PAN_SMOOTH_ALPHA = 0.32  # Exponential smoothing factor for pan

    # Gesture reference anchors (captured when gesture activates)
    ref_dist = None
    ref_zoom = 1.0
    ref_midpoint = None
    ref_pan = (pan_x, pan_y)

    # Hysteresis confirmation counters
    active_confirm_count = 0
    inactive_confirm_count = 0
    gesture_active = False

    CONFIRM_FRAMES_ON = 2   # Require 2 consecutive frames to activate
    CONFIRM_FRAMES_OFF = 3  # Require 3 consecutive frames to deactivate (graceful drop)

    print("=======================================================")
    print("Gesture-Zoom-cv | Developed by Cholan Kinnera")
    print("=======================================================")
    print("Controls:")
    print("  Two-Hand Gesture (Thumb + Index): Zoom & Pan")
    print("  [r]: Reset zoom and position to default")
    print("  [+ / =]: Keyboard zoom in (+15%)")
    print("  [-]: Keyboard zoom out (-15%)")
    print("  [f]: Toggle fullscreen")
    print("  [q]: Quit cleanly")
    print("=======================================================")

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                continue

            h_frame, w_frame = frame.shape[:2]

            # Detect hands
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            hands_detected = detection_result.hand_landmarks
            handedness_list = detection_result.handedness
            num_hands = len(hands_detected)

            hand_centers = []
            hand_statuses = []
            current_dist = None
            current_midpoint = None

            # Process detected hands
            for i in range(num_hands):
                hand_lms = hands_detected[i]
                label = handedness_list[i][0].category_name if i < len(handedness_list) else f"Hand_{i+1}"
                h_px = [(int(lm.x * w_frame), int(lm.y * h_frame)) for lm in hand_lms]

                # Stable palm/MCP center
                hc = get_hand_center(h_px)
                hand_centers.append(hc)

                # Check gesture
                is_ready, status = check_hand_gesture(h_px, label)
                hand_statuses.append((h_px, is_ready, status))

            # Raw detection condition: 2 hands both matching gesture
            raw_detected = (
                num_hands == 2 and
                hand_statuses[0][1] and
                hand_statuses[1][1]
            )

            # Hysteresis filtering for robust activation
            if raw_detected:
                active_confirm_count += 1
                inactive_confirm_count = 0
                if active_confirm_count >= CONFIRM_FRAMES_ON:
                    gesture_active = True
            else:
                inactive_confirm_count += 1
                active_confirm_count = 0
                if inactive_confirm_count >= CONFIRM_FRAMES_OFF:
                    gesture_active = False

            # Update Zoom & Pan when gesture is active
            if gesture_active and len(hand_centers) == 2:
                c1, c2 = hand_centers[0], hand_centers[1]
                current_dist = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
                current_midpoint = ((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0)

                # First frame of active session: capture stable reference anchors
                if ref_dist is None or ref_midpoint is None:
                    ref_dist = max(20.0, current_dist)
                    ref_zoom = current_zoom
                    ref_midpoint = current_midpoint
                    ref_pan = (pan_x, pan_y)

                # 1. Smooth Relative Zoom calculation
                dist_ratio = current_dist / max(10.0, ref_dist)
                target_zoom = max(MIN_ZOOM, min(MAX_ZOOM, ref_zoom * dist_ratio))

                # 2. Smooth Decoupled Pan calculation
                delta_x = current_midpoint[0] - ref_midpoint[0]
                delta_y = current_midpoint[1] - ref_midpoint[1]
                target_pan_x = ref_pan[0] + delta_x
                target_pan_y = ref_pan[1] + delta_y
            else:
                # Gesture inactive: clear anchors so reactivation doesn't cause jumps
                ref_dist = None
                ref_midpoint = None

            # Exponential smoothing for jitter-free zoom & pan
            current_zoom += ZOOM_SMOOTH_ALPHA * (target_zoom - current_zoom)
            pan_x += PAN_SMOOTH_ALPHA * (target_pan_x - pan_x)
            pan_y += PAN_SMOOTH_ALPHA * (target_pan_y - pan_y)

            # Calculate new image dimensions preserving aspect ratio
            new_h = max(40, int(round(h_orig * current_zoom)))
            new_w = max(40, int(round(new_h * aspect_ratio)))

            # Boundary clamping: ensure image cannot drift completely outside frame
            min_margin = 60
            pan_x = max(-new_w // 2 + min_margin, min(w_frame + new_w // 2 - min_margin, pan_x))
            pan_y = max(-new_h // 2 + min_margin, min(h_frame + new_h // 2 - min_margin, pan_y))

            # Resize overlay with optimal interpolation for sharpness
            interp = cv2.INTER_AREA if current_zoom < 1.0 else cv2.INTER_LINEAR
            overlay_resized = cv2.resize(img_base, (new_w, new_h), interpolation=interp)

            # Safely overlay image at (pan_x, pan_y)
            frame = overlay_image_safe(frame, overlay_resized, pan_x, pan_y)

            # Draw visual landmarks and hand skeletons
            for h_px, is_ready, status in hand_statuses:
                line_color = (50, 220, 90) if is_ready else (0, 165, 255)
                for p1_idx, p2_idx in HAND_CONNECTIONS:
                    cv2.line(frame, h_px[p1_idx], h_px[p2_idx], line_color, 2)
                for pt in h_px:
                    cv2.circle(frame, pt, 3, (0, 0, 255), -1)

                # Palm center circle
                c_palm = get_hand_center(h_px)
                cv2.circle(frame, c_palm, 6, (255, 0, 255), -1)

            # Draw connecting gesture line when active
            if gesture_active and len(hand_centers) == 2:
                c1, c2 = hand_centers[0], hand_centers[1]
                cv2.circle(frame, c1, 9, (255, 0, 255), -1)
                cv2.circle(frame, c2, 9, (255, 0, 255), -1)
                cv2.line(frame, c1, c2, (40, 220, 80), 3)

                # Midpoint indicator
                if current_midpoint is not None:
                    mid_pt = (int(current_midpoint[0]), int(current_midpoint[1]))
                    cv2.circle(frame, mid_pt, 7, (0, 255, 255), -1)

            # Draw center crosshair on the controlled image
            img_center = (int(pan_x), int(pan_y))
            if 0 <= img_center[0] < w_frame and 0 <= img_center[1] < h_frame:
                cv2.circle(frame, img_center, 4, (0, 255, 255), -1)

            # Render presentation HUD
            render_hud(frame, gesture_active, current_zoom, (pan_x, pan_y), num_hands, current_dist, ref_dist)

            # Show frame
            cv2.imshow(window_name, frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset zoom and position
                target_zoom = 1.0
                current_zoom = 1.0
                target_pan_x = w_frame / 2.0
                target_pan_y = h_frame / 2.0
                pan_x = target_pan_x
                pan_y = target_pan_y
                ref_dist = None
                ref_midpoint = None
            elif key in [ord('+'), ord('=')]:
                # Keyboard zoom in
                target_zoom = min(MAX_ZOOM, current_zoom + 0.15)
                current_zoom = target_zoom
                ref_dist = None
            elif key in [ord('-'), ord('_')]:
                # Keyboard zoom out
                target_zoom = max(MIN_ZOOM, current_zoom - 0.15)
                current_zoom = target_zoom
                ref_dist = None
            elif key == ord('f'):
                # Toggle fullscreen
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            detector.close()
        except Exception:
            pass
        print("Application cleanly exited and camera released.")


if __name__ == "__main__":
    main()