"""Facial landmark extraction and pixel-to-mm conversion using MediaPipe FaceLandmarker."""

import numpy as np
import cv2
from mediapipe import Image, ImageFormat
from typing import Optional


# MediaPipe FaceMesh landmark indices (468-point mesh)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark.tflite
# All indices are from the 468-point face mesh topology.
LANDMARK_INDICES = {
    "chin_tip": 152,
    "nose_tip": 4,
    "nose_base_left": 36,
    "nose_base_right": 206,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_outer": 362,
    "right_eye_inner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
    "left_cheekbone": 234,
    "right_cheekbone": 454,
    "brow_left": 70,
    "brow_right": 300,
    "glabella": 10,
    "gonion_left": 172,
    "gonion_right": 397,
    "upper_lip_top": 13,
    "lower_lip_bottom": 14,
    "stomion": 0,
    "left_ear": 200,
    "right_ear": 420,
    "subnasale": 2,
}

NUM_LANDMARKS = 478
AVERAGE_IPD_MM = 63.0


def extract_landmarks(
    frame: np.ndarray, landmarker
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Extract 478 facial landmarks using MediaPipe FaceLandmarker.

    Args:
        frame: BGR frame from OpenCV.
        landmarker: FaceLandmarker instance.

    Returns:
        Tuple of (landmark_coords array (N,2), average_confidence) or (None, None).
    """
    h, w, _ = frame.shape
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame)
    timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.face_landmarks:
        return None, None

    face = result.face_landmarks[0]
    coords = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
    confidences = []
    for i, lm in enumerate(face):
        if i < NUM_LANDMARKS:
            coords[i] = (lm.x * w, lm.y * h)
            confidences.append(lm.visibility if hasattr(lm, 'visibility') and lm.visibility else 1.0)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return coords, avg_conf


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draw key landmarks and connection lines on the frame."""
    out = frame.copy()
    for idx in range(468):
        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
        cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
    connections = [
        (33, 133),
        (362, 263),
        (61, 291),
        (234, 454),
        (4, 152),
        (70, 300),
        (172, 397),
    ]
    for a, b in connections:
        if a < 468 and b < 468:
            ax, ay = int(landmarks[a][0]), int(landmarks[a][1])
            bx, by = int(landmarks[b][0]), int(landmarks[b][1])
            cv2.line(out, (ax, ay), (bx, by), (255, 0, 0), 2)
    return out


def detect_hairline(
    frame: np.ndarray, landmarks: np.ndarray
) -> tuple[np.ndarray, float]:
    """Detect the hairline using HSV skin/hair segmentation.

    Scans upward from the brow landmarks through a region of interest,
    identifying the skin-to-hair transition using color analysis.

    Args:
        frame: BGR frame from webcam.
        landmarks: 478-point facial landmarks array.

    Returns:
        Tuple of:
        - hairline_points: (N, 2) array of detected hairline points across columns
        - confidence: 0.0-1.0 detection confidence
    """
    h, w = frame.shape[:2]

    brow_left = landmarks[70].astype(int)
    brow_right = landmarks[300].astype(int)
    glabella = landmarks[10].astype(int)

    x_left = max(0, int(brow_left[0]) - 15)
    x_right = min(w, int(brow_right[0]) + 15)

    brow_y = min(int(brow_left[1]), int(brow_right[1]))
    search_top = max(0, brow_y - int(brow_y * 0.8))
    search_bottom = max(5, brow_y - 10)
    roi_height = search_bottom - search_top
    roi_width = x_right - x_left

    if roi_height < 20 or roi_width < 20:
        center_x = glabella[0]
        est_y = int(brow_left[1] * 0.75)
        return np.array([[center_x, est_y]]), 0.0

    roi = frame[search_top:search_bottom, x_left:x_right].copy()
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    adaptive_skin_lo, adaptive_skin_hi = _adaptive_skin_bounds(hsv)

    skin_mask = cv2.inRange(hsv, adaptive_skin_lo, adaptive_skin_hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    hairline_points = []
    confidences = []

    for col in range(roi_width):
        column_mask = skin_mask[:, col]
        scan_from = roi_height - 1

        for row in range(scan_from, -1, -1):
            if column_mask[row] == 0:
                non_skin_count = 0
                for check_row in range(max(0, row - 8), row + 1):
                    if column_mask[check_row] == 0:
                        non_skin_count += 1

                if non_skin_count >= 4:
                    hair_y = search_top + row
                    hairline_points.append([x_left + col, hair_y])
                    confidences.append(min(1.0, non_skin_count / 8.0))
                    break
        else:
            hairline_points.append([x_left + col, search_top])
            confidences.append(0.0)

    if not hairline_points:
        center_x = (x_left + x_right) // 2
        est_y = int(brow_left[1] * 0.7)
        return np.array([[center_x, est_y]], dtype=np.float32), 0.0

    hairline_array = np.array(hairline_points, dtype=np.float32)
    confidence = float(np.mean(confidences)) if confidences else 0.0

    outliers = _remove_hairline_outliers(hairline_array)
    hairline_array = hairline_array[outliers]
    confidence *= 0.9

    if len(hairline_array) < 3:
        center_x = int(np.mean(hairline_array[:, 0]))
        center_y = int(np.mean(hairline_array[:, 1]))
        return np.array([[center_x, center_y]], dtype=np.float32), max(0.0, confidence)

    return hairline_array, confidence


def _adaptive_skin_bounds(hsv_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute adaptive HSV skin bounds from the ROI.

    Uses the lower half of the forehead ROI (closest to brows, most likely skin)
    to calibrate the skin color range.
    """
    h, w = hsv_roi.shape[:2]
    sample_region = hsv_roi[int(h * 0.6):, :]

    if sample_region.size == 0:
        return np.array([0, 20, 70]), np.array([25, 255, 255])

    h_vals = sample_region[:, :, 0].flatten()
    s_vals = sample_region[:, :, 1].flatten()
    v_vals = sample_region[:, :, 2].flatten()

    h_median = float(np.median(h_vals))
    s_median = float(np.median(s_vals))
    v_median = float(np.median(v_vals))

    h_std = float(np.std(h_vals))
    s_std = float(np.std(s_vals))

    h_lo = max(0, h_median - max(h_std * 2.5, 12))
    h_hi = min(179, h_median + max(h_std * 2.5, 12))

    s_lo = max(0, s_median - max(s_std * 2, 30))
    v_lo = max(0, min(80, v_median * 0.4))

    return (
        np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
        np.array([h_hi, min(255, s_median + max(s_std * 2, 80)), 255], dtype=np.uint8),
    )


def _remove_hairline_outliers(hairline_array: np.ndarray) -> np.ndarray:
    """Remove outlier hairline points using median absolute deviation."""
    if len(hairline_array) < 5:
        return np.ones(len(hairline_array), dtype=bool)

    y_vals = hairline_array[:, 1]
    median_y = np.median(y_vals)
    mad = np.median(np.abs(y_vals - median_y))

    if mad < 1:
        return np.ones(len(hairline_array), dtype=bool)

    threshold = max(15, 3 * mad)
    mask = np.abs(y_vals - median_y) < threshold
    return mask


def draw_face_guide(frame: np.ndarray) -> np.ndarray:
    """Draw an oval guide for face alignment."""
    out = frame.copy()
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    rx, ry = int(w * 0.30), int(h * 0.38)
    cv2.ellipse(out, (cx, cy), (rx, ry), 0, 0, 360, (0, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, "Align your face inside the oval", (cx - 180, h - 60), font, 0.7, (0, 255, 255), 2)
    cv2.putText(out, "Look straight ahead - neutral expression", (cx - 200, h - 30), font, 0.6, (0, 255, 255), 2)
    return out


def draw_profile_guide(frame: np.ndarray) -> np.ndarray:
    """Draw a center line guide for profile alignment."""
    out = frame.copy()
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(out, (cx, 0), (cx, h), (0, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, "Turn your head 90 degrees to the right", (30, h - 60), font, 0.7, (0, 255, 255), 2)
    cv2.putText(out, "Align your profile with the center line", (30, h - 30), font, 0.6, (0, 255, 255), 2)
    return out


def compute_pixel_to_mm(landmarks: np.ndarray) -> float:
    """Convert pixel distances to millimeters using average IPD (63mm) as reference."""
    left_eye = landmarks[33]
    right_eye = landmarks[362]
    ipd_pixels = np.linalg.norm(right_eye - left_eye)
    if ipd_pixels <= 0:
        return 1.0
    return AVERAGE_IPD_MM / ipd_pixels


def get_landmark(landmarks: np.ndarray, name: str) -> np.ndarray:
    """Get landmark coordinates by name."""
    idx = LANDMARK_INDICES[name]
    if idx >= NUM_LANDMARKS:
        return np.array([0.0, 0.0])
    return landmarks[idx]


def dist(landmarks: np.ndarray, a_name: str, b_name: str) -> float:
    """Distance between two landmarks by name."""
    a = get_landmark(landmarks, a_name)
    b = get_landmark(landmarks, b_name)
    return float(np.linalg.norm(b - a))


def dist_px(landmarks: np.ndarray, a_idx: int, b_idx: int) -> float:
    """Distance between two landmarks by index."""
    a = landmarks[a_idx]
    b = landmarks[b_idx]
    return float(np.linalg.norm(b - a))


def angle_between(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle at p2 formed by p1-p2-p3 in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def compute_head_tilt(landmarks: np.ndarray) -> float:
    """Compute head tilt angle from eye line horizontal."""
    left_eye = landmarks[33]
    right_eye = landmarks[362]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    tilt = np.degrees(np.arctan2(dy, dx))
    return float(tilt)


def check_lighting(frame: np.ndarray) -> float:
    """Estimate lighting quality from frame histogram (0-1)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    brightness = np.sum(hist * np.arange(256))
    contrast = np.sqrt(np.sum(hist * (np.arange(256) - brightness) ** 2))
    score = min(1.0, contrast / 60.0) * min(1.0, brightness / 100.0)
    return float(score)


def check_face_centered(landmarks: np.ndarray, frame_shape: tuple) -> tuple[bool, float]:
    """Check if face is centered in the frame."""
    h, w = frame_shape[:2]
    nose = landmarks[4]
    cx, cy = w / 2, h / 2
    offset = np.sqrt((nose[0] - cx) ** 2 + (nose[1] - cy) ** 2)
    max_offset = min(w, h) * 0.15
    return offset < max_offset, offset / max_offset


def is_profile_view(landmarks: np.ndarray) -> bool:
    """Detect if the face is in profile (side) view."""
    nose = landmarks[4]
    left_eye = landmarks[33]
    right_eye = landmarks[362]
    nose_x = nose[0]
    left_x = left_eye[0]
    right_x = right_eye[0]
    if nose_x < left_x or nose_x > right_x:
        return False
    nose_eye_dist_l = abs(nose_x - left_x)
    nose_eye_dist_r = abs(nose_x - right_x)
    ratio = min(nose_eye_dist_l, nose_eye_dist_r) / (max(nose_eye_dist_l, nose_eye_dist_r) + 1e-10)
    return ratio < 0.3


def validate_landmark_distances(landmarks: np.ndarray) -> tuple[bool, list[str]]:
    """Check if landmark measurements are physically plausible.

    Returns (is_valid, list_of_warnings).
    """
    warnings = []

    eye_w = dist_px(landmarks, 33, 133)
    if eye_w < 20:
        warnings.append(f"Left eye width too small ({eye_w:.0f}px) — move closer to camera")

    nose_w = dist_px(landmarks, 36, 206)
    if nose_w < 10:
        warnings.append(f"Nose width too small ({nose_w:.0f}px) — detection may be inaccurate")

    face_h = dist(landmarks, "brow_left", "chin_tip")
    if face_h < 100:
        warnings.append(f"Face too small ({face_h:.0f}px) — move closer to camera")

    mouth_w = dist(landmarks, "left_mouth_corner", "right_mouth_corner")
    if mouth_w < 15:
        warnings.append(f"Mouth width too small ({mouth_w:.0f}px) — relax your expression")

    lip_lower = dist(landmarks, "stomion", "chin_tip")
    if lip_lower < 2:
        warnings.append("Lower lip detection near zero — slightly part your lips for accurate ratio")

    return len(warnings) == 0, warnings


def draw_hairline_overlay(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draw the detected hairline on the frame for visual feedback."""
    out = frame.copy()
    hairline_pts, confidence = detect_hairline(frame, landmarks)

    if confidence < 0.3 or len(hairline_pts) < 2:
        return out

    color = (0, 200, 255) if confidence > 0.6 else (0, 100, 255)
    for i in range(len(hairline_pts) - 1):
        p1 = hairline_pts[i].astype(int)
        p2 = hairline_pts[i + 1].astype(int)
        cv2.line(out, tuple(p1), tuple(p2), color, 2)

    for pt in hairline_pts:
        cv2.circle(out, tuple(pt.astype(int)), 3, color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"Hairline ({confidence:.0%})"
    cv2.putText(out, label, (10, 25), font, 0.5, color, 1)

    return out
