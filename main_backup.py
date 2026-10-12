"""
Zoom Any Picture Using Hand Gestures
Powered by MediaPipe Tasks API and OpenCV.
Model: hand_landmarker.task
"""
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
    independent of rotation, mirroring, or tilt.
    """
    wrist = landmarks_px[0]
    p_center = get_hand_center(landmarks_px)

    def dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    palm_span = max(10.0, dist(landmarks_px[5], landmarks_px[17]))

    # --- 1. Index Finger (Tip: 8, DIP: 7, PIP: 6, MCP: 5) ---
    # Extended if tip is clearly further from palm center & wrist than PIP
    d_tip_center_idx = dist(landmarks_px[8], p_center)
    d_pip_center_idx = dist(landmarks_px[6], p_center)
    d_tip_wrist_idx = dist(landmarks_px[8], wrist)
    d_pip_wrist_idx = dist(landmarks_px[6], wrist)
    index_extended = (d_tip_center_idx > d_pip_center_idx * 1.1) and (d_tip_wrist_idx > d_pip_wrist_idx)

    # --- 2. Middle, Ring, Pinky Fingers (Must be curled / folded) ---
    # For curled fingers, the tip folds into palm, making tip closer to palm center or wrist
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

    # --- 3. Thumb (Tip: 4, IP: 3, MCP: 2, CMC: 1) ---
    # Extended if thumb tip is straightened away from thumb MCP and away from palm center
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
    Safely clips edges so image stays visible even when touching or crossing frame borders.
    """
    h_ov, w_ov = overlay.shape[:2]
    h_frame, w_frame = frame.shape[:2]

    x1 = int(center_x - w_ov // 2)
    y1 = int(center_y - h_ov // 2)
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


def main():
    # 1. Load overlay image once
    img_path = "one.jpg"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Target image '{img_path}' not found.")
    img_base = cv2.imread(img_path)
    if img_base is None:
        raise ValueError(f"Failed to read image '{img_path}'.")
    h_orig, w_orig = img_base.shape[:2]
    aspect_ratio = w_orig / h_orig

    # 2. Initialize MediaPipe HandLandmarker Task
    model_path = "hand_landmarker.task"
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
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Zoom & Position State
    initial_dist = None     # Reference distance when gesture starts
    initial_scale = 0       # Scale snapshot at gesture start
    current_scale = 0       # Persistent scale offset in pixels
    cx, cy = 640, 360       # Persistent center position

    # Minimum and Maximum scale limits
    MIN_SCALE = -180        # Image shrinks to 70px minimum
    MAX_SCALE = 650         # Image zooms to 900px maximum
    ZOOM_SENSITIVITY = 1.2  # Multiplier for hand distance change

    print("Application started. Showing webcam stream. Press 'q' to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                continue

            h_frame, w_frame = frame.shape[:2]

            # Detect hands using MediaPipe Tasks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            hands_detected = detection_result.hand_landmarks
            handedness_list = detection_result.handedness
            num_hands = len(hands_detected)

            hand_centers = []
            hand_statuses = []
            gesture_active = False
            current_dist = None

            # Process detected hands
            for i in range(num_hands):
                hand_lms = hands_detected[i]
                label = handedness_list[i][0].category_name if i < len(handedness_list) else f"Hand_{i+1}"
                h_px = [(int(lm.x * w_frame), int(lm.y * h_frame)) for lm in hand_lms]

                # Hand center (Index MCP 5 + Pinky MCP 17 average)
                hc = get_hand_center(h_px)
                hand_centers.append(hc)

                # Check gesture
                is_ready, status = check_hand_gesture(h_px, label)
                hand_statuses.append((h_px, is_ready, status))

            # When exactly two hands satisfy the gesture
            if num_hands == 2 and hand_statuses[0][1] and hand_statuses[1][1]:
                c1 = hand_centers[0]
                c2 = hand_centers[1]
                current_dist = math.hypot(c2[0] - c1[0], c2[1] - c1[1])

                # Record reference distance on initial gesture activation
                if initial_dist is None:
                    initial_dist = current_dist
                    initial_scale = current_scale

                # Relative zoom calculation
                delta_dist = (current_dist - initial_dist) * ZOOM_SENSITIVITY
                current_scale = int(initial_scale + delta_dist)
                current_scale = max(MIN_SCALE, min(MAX_SCALE, current_scale))

                # Update position to midpoint between hands
                cx = (c1[0] + c2[0]) // 2
                cy = (c1[1] + c2[1]) // 2
                gesture_active = True
            else:
                # Inactive: Reset reference distance, but keep current_scale persistent!
                initial_dist = None

            # Render hand skeletons and landmark debug labels
            for h_px, is_ready, status in hand_statuses:
                # Skeleton lines
                line_color = (0, 255, 0) if is_ready else (0, 165, 255)
                for p1_idx, p2_idx in HAND_CONNECTIONS:
                    cv2.line(frame, h_px[p1_idx], h_px[p2_idx], line_color, 2)
                for pt in h_px:
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)

                # Palm center circle
                c_palm = get_hand_center(h_px)
                cv2.circle(frame, c_palm, 7, (255, 0, 255), -1)

                # Hand status badge above wrist
                badge_text = f"[{status['handedness']}] READY" if is_ready else f"[{status['handedness']}] T:{int(status['thumb'])} I:{int(status['index'])}"
                badge_col = (0, 255, 0) if is_ready else (0, 140, 255)
                cv2.putText(frame, badge_text, (h_px[0][0] - 40, h_px[0][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, badge_col, 2)

            # Draw zoom line and centers when gesture is active
            if gesture_active and len(hand_centers) == 2:
                cv2.circle(frame, hand_centers[0], 10, (255, 0, 255), -1)
                cv2.circle(frame, hand_centers[1], 10, (255, 0, 255), -1)
                cv2.line(frame, hand_centers[0], hand_centers[1], (0, 255, 0), 3)
                cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                # Live distance on the line
                mid_line_x = (hand_centers[0][0] + hand_centers[1][0]) // 2
                mid_line_y = (hand_centers[0][1] + hand_centers[1][1]) // 2 - 15
                cv2.putText(frame, f"{int(current_dist)}px", (mid_line_x - 30, mid_line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Compute image dimensions from persistent current_scale
            new_h = max(50, min(1200, int(h_orig + current_scale)))
            new_w = max(50, int(new_h * aspect_ratio))
            overlay_resized = cv2.resize(img_base, (new_w, new_h))

            # Overlay image at (cx, cy)
            frame = overlay_image_safe(frame, overlay_resized, cx, cy)

            # --- HUD & Debug Panel ---
            # Translucent top bar
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (10, 10), (700, 145), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0, frame)

            # Row 1: Status
            status_text = "ZOOM ACTIVE" if gesture_active else "WAITING FOR 2-HAND GESTURE"
            status_color = (0, 255, 0) if gesture_active else (0, 165, 255)
            cv2.putText(frame, f"Status: {status_text}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

            # Row 2: Hands detected & Gesture requirement
            cv2.putText(
                frame,
                f"Hands: {num_hands}/2 | Gesture: Thumb+Index Extended, 3 Folded",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1
            )

            # Row 3: Per-hand status
            if num_hands == 0:
                h_debug = "No hands detected in webcam view"
            elif num_hands == 1:
                st1 = hand_statuses[0][2]
                r1 = "READY" if st1['is_ready'] else "WAITING"
                h_debug = f"H1 [{st1['handedness']}]: {r1} (Thumb:{int(st1['thumb'])}, Index:{int(st1['index'])}) | Need 2nd hand"
            else:
                st1, st2 = hand_statuses[0][2], hand_statuses[1][2]
                r1 = "OK" if st1['is_ready'] else "NO"
                r2 = "OK" if st2['is_ready'] else "NO"
                h_debug = f"H1 [{st1['handedness']}]: {r1} (T:{int(st1['thumb'])} I:{int(st1['index'])}) | H2 [{st2['handedness']}]: {r2} (T:{int(st2['thumb'])} I:{int(st2['index'])})"
            cv2.putText(frame, h_debug, (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200) if gesture_active else (180, 180, 180), 1)

            # Row 4: Metrics (Distance, Reference, Scale, Dimensions)
            dist_disp = f"{int(current_dist)}px" if current_dist is not None else "--"
            ref_disp = f"{int(initial_dist)}px" if initial_dist is not None else "--"
            metrics_str = f"Dist: {dist_disp} (Ref: {ref_disp}) | Scale: {current_scale:+d}px | Size: {new_w}x{new_h}"
            cv2.putText(frame, metrics_str, (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Display window
            cv2.imshow("Zoom Any Picture Using Hand Gestures", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("Application cleanly exited and camera released.")


if __name__ == "__main__":
    main()