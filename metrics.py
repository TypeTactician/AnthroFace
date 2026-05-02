"""All facial measurement functions with scientific citations.

References:
    Farkas, L.G. (1994). Anthropometry of the Head and Face.
    Powell, N. & Humphreys, B. (1984). Proportions of the Aesthetic Face.
    Goode, R.L. (1981). Nasal projection measurement.
    Ricketts, R.M. (1981). The aesthetic environment.
"""

import math
import numpy as np
from typing import Optional
from landmarks import (
    get_landmark, dist_px, dist, angle_between,
    compute_pixel_to_mm, validate_landmark_distances,
)


PHI = 1.618


def _get_hairline_center(landmarks: np.ndarray, frame) -> tuple[np.ndarray, bool]:
    """Get the center hairline point from detection or estimation.

    Returns (point, was_detected).
    """
    from landmarks import detect_hairline

    hairline_pts, confidence = detect_hairline(frame, landmarks)

    if confidence > 0.3 and len(hairline_pts) >= 2:
        center_x = int(np.mean(hairline_pts[:, 0]))
        center_y = int(np.mean(hairline_pts[:, 1]))
        return np.array([center_x, center_y], dtype=np.float32), True

    brow_left = get_landmark(landmarks, "brow_left")
    brow_right = get_landmark(landmarks, "brow_right")
    brow = (brow_left + brow_right) / 2
    subnasale = get_landmark(landmarks, "subnasale")
    middle = float(np.linalg.norm(subnasale - brow))
    direction = brow - subnasale
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    estimated = brow + direction * middle
    return estimated, False


def measure_facial_thirds(landmarks: np.ndarray, frame) -> dict:
    """Measure the three horizontal facial thirds.

    Uses HSV-based hairline detection from the captured frame.
    Falls back to neoclassical canon estimation if detection fails.

    Upper: hairline to brow (glabella)
    Middle: brow to subnasale (nose base)
    Lower: subnasale to menton (chin tip)

    Reference: Farkas 1994, Neoclassical canons.
    """
    brow_left = get_landmark(landmarks, "brow_left")
    brow_right = get_landmark(landmarks, "brow_right")
    brow = (brow_left + brow_right) / 2
    subnasale = get_landmark(landmarks, "subnasale")
    chin = get_landmark(landmarks, "chin_tip")

    hairline, was_detected = _get_hairline_center(landmarks, frame)

    middle = float(np.linalg.norm(subnasale - brow))
    lower = float(np.linalg.norm(chin - subnasale))
    upper = float(np.linalg.norm(brow - hairline))
    total = upper + middle + lower

    if total == 0:
        return {"upper_ratio": 0.33, "middle_ratio": 0.33, "lower_ratio": 0.33,
                "upper_mm": 0, "middle_mm": 0, "lower_mm": 0,
                "hairline_detected": False}

    p2m = compute_pixel_to_mm(landmarks)
    return {
        "upper_ratio": round(upper / total, 3),
        "middle_ratio": round(middle / total, 3),
        "lower_ratio": round(lower / total, 3),
        "upper_mm": round(upper * p2m, 1),
        "middle_mm": round(middle * p2m, 1),
        "lower_mm": round(lower * p2m, 1),
        "hairline_detected": was_detected,
    }


def measure_facial_fifths(landmarks: np.ndarray) -> dict:
    """Measure the five vertical fifths of the face.

    Each fifth should equal one eye width.

    Reference: Farkas 1994, Neoclassical canons.
    """
    left_eye_outer = get_landmark(landmarks, "left_eye_outer")
    left_eye_inner = get_landmark(landmarks, "left_eye_inner")
    right_eye_inner = get_landmark(landmarks, "right_eye_inner")
    right_eye_outer = get_landmark(landmarks, "right_eye_outer")

    left_eye_w = float(np.linalg.norm(left_eye_inner - left_eye_outer))
    right_eye_w = float(np.linalg.norm(right_eye_inner - right_eye_outer))
    avg_eye_w = (left_eye_w + right_eye_w) / 2

    intercanthal = float(np.linalg.norm(right_eye_inner - left_eye_inner))

    nose_base_l = get_landmark(landmarks, "nose_base_left")
    nose_base_r = get_landmark(landmarks, "nose_base_right")
    nose_w = float(np.linalg.norm(nose_base_r - nose_base_l))

    face_l = get_landmark(landmarks, "left_cheekbone")
    face_r = get_landmark(landmarks, "right_cheekbone")
    bizygomatic = float(np.linalg.norm(face_r - face_l))

    outer_left = float(np.linalg.norm(left_eye_outer - face_l))
    outer_right = float(np.linalg.norm(right_eye_outer - face_r))

    p2m = compute_pixel_to_mm(landmarks)

    return {
        "left_eye_width": round(left_eye_w, 1),
        "right_eye_width": round(right_eye_w, 1),
        "avg_eye_width": round(avg_eye_w, 1),
        "intercanthal_distance": round(intercanthal, 1),
        "nose_width": round(nose_w, 1),
        "bizygomatic_width": round(bizygomatic, 1),
        "left_outer_fifth": round(outer_left, 1),
        "right_outer_fifth": round(outer_right, 1),
        "avg_eye_width_mm": round(avg_eye_w * p2m, 1),
        "intercanthal_mm": round(intercanthal * p2m, 1),
        "nose_width_mm": round(nose_w * p2m, 1),
    }


def measure_golden_ratio_proportions(landmarks: np.ndarray, frame) -> dict:
    """Measure golden ratio relationships in the face.

    Uses detected hairline when available, otherwise falls back to estimation.

    Reference: Powell & Humphreys 1984; historical phi applications.
    """
    face_l = get_landmark(landmarks, "left_cheekbone")
    face_r = get_landmark(landmarks, "right_cheekbone")
    face_width = float(np.linalg.norm(face_r - face_l))

    brow_left = get_landmark(landmarks, "brow_left")
    brow_right = get_landmark(landmarks, "brow_right")
    brow = (brow_left + brow_right) / 2
    subnasale = get_landmark(landmarks, "subnasale")
    chin = get_landmark(landmarks, "chin_tip")

    hairline, _ = _get_hairline_center(landmarks, frame)
    face_height = float(np.linalg.norm(chin - hairline))

    nose_tip = get_landmark(landmarks, "nose_tip")
    nose_base_l = get_landmark(landmarks, "nose_base_left")
    nose_base_r = get_landmark(landmarks, "nose_base_right")
    nose_width = float(np.linalg.norm(nose_base_r - nose_base_l))
    nose_length = float(np.linalg.norm(nose_tip - brow))

    mouth_l = get_landmark(landmarks, "left_mouth_corner")
    mouth_r = get_landmark(landmarks, "right_mouth_corner")
    mouth_width = float(np.linalg.norm(mouth_r - mouth_l))

    chin_to_nose = float(np.linalg.norm(chin - subnasale))
    nose_to_brow = float(np.linalg.norm(nose_tip - brow))

    ratios = {}
    if face_height > 0:
        ratios["face_width_height"] = face_width / face_height
    if nose_length > 0:
        ratios["nose_width_length"] = nose_width / nose_length
    if nose_width > 0:
        ratios["mouth_nose_width"] = mouth_width / nose_width
    if chin_to_nose > 0 and nose_to_brow > 0:
        ratios["chin_nose_to_nose_brow"] = chin_to_nose / nose_to_brow

    p2m = compute_pixel_to_mm(landmarks)
    return {
        **ratios,
        "face_width_mm": round(face_width * p2m, 1),
        "face_height_mm": round(face_height * p2m, 1),
        "nose_width_mm": round(nose_width * p2m, 1),
        "nose_length_mm": round(nose_length * p2m, 1),
        "mouth_width_mm": round(mouth_width * p2m, 1),
    }


def measure_symmetry(landmarks: np.ndarray) -> dict:
    """Compute facial symmetry index by comparing left/right landmark distances.

    Reference: Farkas 1994; Peck et al. (2008) facial asymmetry norms.
    """
    pairs = [
        ("brow_left", "brow_right"),
        ("left_cheekbone", "right_cheekbone"),
        ("left_eye_outer", "right_eye_outer"),
        ("left_eye_inner", "right_eye_inner"),
        ("left_mouth_corner", "right_mouth_corner"),
        ("gonion_left", "gonion_right"),
    ]

    mid_x = np.mean([landmarks[33][0], landmarks[362][0]])
    asymmetries = []

    for left_name, right_name in pairs:
        left = get_landmark(landmarks, left_name)
        right = get_landmark(landmarks, right_name)
        left_dist = abs(left[0] - mid_x)
        right_dist = abs(right[0] - mid_x)
        if max(left_dist, right_dist) > 0:
            asym = abs(left_dist - right_dist) / max(left_dist, right_dist)
            asymmetries.append(asym)

    mean_asymmetry = float(np.mean(asymmetries)) if asymmetries else 0.0
    symmetry_index = max(0, 1.0 - mean_asymmetry)

    return {
        "symmetry_index": round(symmetry_index, 3),
        "mean_asymmetry": round(mean_asymmetry, 3),
        "individual_asymmetries": [round(a, 3) for a in asymmetries],
    }


def measure_eye_spacing(landmarks: np.ndarray) -> dict:
    """Measure eye spacing ratio.

    Ideal intercanthal / biocular ratio ~ 0.46.

    Reference: Farkas 1994.
    """
    left_eye_inner = get_landmark(landmarks, "left_eye_inner")
    right_eye_inner = get_landmark(landmarks, "right_eye_inner")
    left_eye_outer = get_landmark(landmarks, "left_eye_outer")
    right_eye_outer = get_landmark(landmarks, "right_eye_outer")

    intercanthal = float(np.linalg.norm(right_eye_inner - left_eye_inner))
    biocular = float(np.linalg.norm(right_eye_outer - left_eye_outer))

    ratio = intercanthal / biocular if biocular > 0 else 0

    p2m = compute_pixel_to_mm(landmarks)
    ipd = dist(landmarks, "left_eye_outer", "right_eye_outer")

    return {
        "intercanthal_distance": round(intercanthal, 1),
        "biocular_width": round(biocular, 1),
        "eye_spacing_ratio": round(ratio, 3),
        "ipd_mm": round(ipd * p2m, 1),
    }


def measure_lip_ratio(landmarks: np.ndarray) -> dict:
    """Measure upper lip to lower lip height ratio.

    Upper lip: subnasale (base of nose) to stomion (where lips meet)
    Lower lip: stomion to menton (chin tip)

    Ideal ratio ~ 1:1.6 (upper:lower, i.e. upper/lower = 0.625).

    Reference: Powell & Humphreys 1984.
    """
    subnasale = get_landmark(landmarks, "subnasale")
    stomion = get_landmark(landmarks, "stomion")
    chin = get_landmark(landmarks, "chin_tip")

    upper_lip_height = float(np.linalg.norm(stomion - subnasale))
    lower_lip_height = float(np.linalg.norm(chin - stomion))

    if lower_lip_height < 1:
        ratio = 0
    else:
        ratio = upper_lip_height / lower_lip_height

    p2m = compute_pixel_to_mm(landmarks)
    return {
        "upper_lip_height": round(upper_lip_height, 1),
        "lower_lip_height": round(lower_lip_height, 1),
        "lip_ratio": round(ratio, 3),
        "upper_lip_mm": round(upper_lip_height * p2m, 1),
        "lower_lip_mm": round(lower_lip_height * p2m, 1),
    }


def measure_nose_projection(landmarks: np.ndarray) -> dict:
    """Measure nasal projection using Goode's method.

    Goode's ratio = nasal projection / nasal length, ideal 0.55-0.60.

    Reference: Goode, R.L. (1981).
    """
    nose_tip = get_landmark(landmarks, "nose_tip")
    subnasale = get_landmark(landmarks, "subnasale")
    brow_l = get_landmark(landmarks, "brow_left")
    brow_r = get_landmark(landmarks, "brow_right")
    brow = (brow_l + brow_r) / 2

    nasal_projection = float(np.linalg.norm(nose_tip - brow))
    nasal_length = float(np.linalg.norm(nose_tip - subnasale))

    goode_ratio = nasal_length / nasal_projection if nasal_projection > 0 else 0

    p2m = compute_pixel_to_mm(landmarks)
    return {
        "nasal_projection": round(nasal_projection, 1),
        "nasal_length": round(nasal_length, 1),
        "goode_ratio": round(goode_ratio, 3),
        "nasal_projection_mm": round(nasal_projection * p2m, 1),
        "nasal_length_mm": round(nasal_length * p2m, 1),
    }


def measure_gonial_angle(landmarks: np.ndarray) -> dict:
    """Estimate the gonial (jaw) angle.

    Ideal: ~120 degrees for males, ~126 degrees for females.

    Reference: Argyropoulos & Sassouni (1989).
    """
    gonion_l = get_landmark(landmarks, "gonion_left")
    gonion_r = get_landmark(landmarks, "gonion_right")
    chin = get_landmark(landmarks, "chin_tip")
    ear_l = get_landmark(landmarks, "left_ear")
    ear_r = get_landmark(landmarks, "right_ear")

    angle_l = angle_between(ear_l, gonion_l, chin)
    angle_r = angle_between(ear_r, gonion_r, chin)
    avg_angle = (angle_l + angle_r) / 2

    return {
        "left_gonial_angle": round(angle_l, 1),
        "right_gonial_angle": round(angle_r, 1),
        "avg_gonial_angle": round(avg_angle, 1),
    }


def measure_cheekbone_jaw_ratio(landmarks: np.ndarray) -> dict:
    """Cheekbone width / jaw width ratio.

    Ideal ~1.3 for males.

    Reference: Farkas 1994.
    """
    cheek_l = get_landmark(landmarks, "left_cheekbone")
    cheek_r = get_landmark(landmarks, "right_cheekbone")
    gonion_l = get_landmark(landmarks, "gonion_left")
    gonion_r = get_landmark(landmarks, "gonion_right")

    cheekbone_width = float(np.linalg.norm(cheek_r - cheek_l))
    jaw_width = float(np.linalg.norm(gonion_r - gonion_l))

    ratio = cheekbone_width / jaw_width if jaw_width > 0 else 0

    p2m = compute_pixel_to_mm(landmarks)
    return {
        "cheekbone_width": round(cheekbone_width, 1),
        "jaw_width": round(jaw_width, 1),
        "cheekbone_jaw_ratio": round(ratio, 3),
        "cheekbone_width_mm": round(cheekbone_width * p2m, 1),
        "jaw_width_mm": round(jaw_width * p2m, 1),
    }


def measure_profile(landmarks: np.ndarray) -> dict:
    """Measure profile angles from a side view capture.

    Uses only reliably detected landmarks: nose, chin, ear, subnasale.
    Forehead angle omitted — MediaPipe does not detect forehead reliably.

    Reference: Ricketts 1981 (E-line); Arnett & Bergman 1993.
    """
    nose_tip = get_landmark(landmarks, "nose_tip")
    chin = get_landmark(landmarks, "chin_tip")
    brow = get_landmark(landmarks, "brow_left")
    subnasale = get_landmark(landmarks, "subnasale")
    ear = get_landmark(landmarks, "left_ear")

    nasal_angle = angle_between(brow, nose_tip, subnasale)
    chin_projection = angle_between(ear, chin, subnasale)
    neck_angle = angle_between(ear, chin, get_landmark(landmarks, "gonion_left"))
    convexity = angle_between(brow, nose_tip, chin)

    p2m = compute_pixel_to_mm(landmarks)
    return {
        "nasal_angle": round(nasal_angle, 1),
        "chin_projection_angle": round(chin_projection, 1),
        "neck_chin_angle": round(neck_angle, 1),
        "profile_convexity": round(convexity, 1),
        "e_line_distance": round(dist_px(landmarks, 4, 152) * p2m, 1),
    }


def measure_skin_uniformity(frame) -> dict:
    """Analyze skin tone uniformity from captured image.

    Reference: Basic image statistics for dermatological assessment.
    """
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask_lower = np.array([0, 20, 70])
    skin_mask_upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, skin_mask_lower, skin_mask_upper)

    masked = cv2.bitwise_and(hsv, hsv, mask=mask)
    skin_pixels = masked[mask > 0]

    if len(skin_pixels) < 100:
        return {"uniformity_score": 0.5, "mean_hue": 0, "std_saturation": 0}

    std_h = float(np.std(skin_pixels[:, 0]))
    std_s = float(np.std(skin_pixels[:, 1]))
    mean_h = float(np.mean(skin_pixels[:, 0]))

    uniformity = max(0, 1.0 - std_h / 30.0)
    return {
        "uniformity_score": round(uniformity, 3),
        "mean_hue": round(mean_h, 1),
        "std_saturation": round(std_s, 1),
    }


def compute_metric_score(deviation_pct: float) -> float:
    """Convert percentage deviation to 0-100 score using gentle exponential decay.

    Natural faces have variation from statistical ideals.
    This curve is calibrated so that typical variation still scores well:

    0% deviation = 100
    10% deviation = 90
    20% deviation = 82
    30% deviation = 74
    50% deviation = 61
    75% deviation = 47
    100% deviation = 37
    150% deviation = 22
    200% deviation = 13
    """
    score = 100 * math.exp(-deviation_pct / 100.0)
    return round(max(0, min(100, score)), 1)


def compute_all_metrics(
    landmarks: np.ndarray, frame, sex: str = "male"
) -> list[dict]:
    """Run all measurements and return structured metric dicts.

    Each metric: {name, measured_value, ideal_value, deviation, score,
                   reference, category}
    """
    data_valid, validation_warnings = validate_landmark_distances(landmarks)

    metrics = []
    p2m = compute_pixel_to_mm(landmarks)

    thirds = measure_facial_thirds(landmarks, frame)
    hairline_info = " (detected)" if thirds.get("hairline_detected") else " (estimated)"
    for third_name, ideal_ratio in [("upper", 0.33), ("middle", 0.33), ("lower", 0.33)]:
        measured = thirds[f"{third_name}_ratio"]
        deviation = abs(measured - ideal_ratio) / ideal_ratio * 100
        ref = "Farkas 1994"
        if third_name == "upper" and not thirds.get("hairline_detected"):
            ref += " — hairline estimated"
        metrics.append({
            "name": f"Facial {third_name.capitalize()} Third",
            "measured_value": round(measured, 3),
            "ideal_value": ideal_ratio,
            "deviation": round(deviation, 1),
            "score": compute_metric_score(deviation),
            "reference": ref,
            "category": "Proportions",
        })

    fifths = measure_facial_fifths(landmarks)
    eye_width = fifths["avg_eye_width"]
    intercanthal_ratio = fifths["intercanthal_distance"] / (eye_width + 1e-10)
    intercanthal_dev = abs(intercanthal_ratio - 1.0) / 1.0 * 100
    metrics.append({
        "name": "Interpupillary / Intercanthal Fifth",
        "measured_value": round(intercanthal_ratio, 3),
        "ideal_value": 1.0,
        "deviation": round(intercanthal_dev, 1),
        "score": compute_metric_score(intercanthal_dev),
        "reference": "Farkas 1994",
        "category": "Proportions",
    })

    nose_width_ratio = fifths["nose_width"] / (eye_width + 1e-10)
    nose_width_dev = abs(nose_width_ratio - 1.0) / 1.0 * 100
    metrics.append({
        "name": "Nose Width (vs Eye Width)",
        "measured_value": round(nose_width_ratio, 3),
        "ideal_value": 1.0,
        "deviation": round(nose_width_dev, 1),
        "score": compute_metric_score(nose_width_dev),
        "reference": "Farkas 1994",
        "category": "Proportions",
    })

    gr = measure_golden_ratio_proportions(landmarks, frame)
    if "face_width_height" in gr:
        fwh = gr["face_width_height"]
        fwh_dev = abs(fwh - (1 / PHI)) / (1 / PHI) * 100
        metrics.append({
            "name": "Face Width/Height (Golden Ratio)",
            "measured_value": round(fwh, 3),
            "ideal_value": round(1 / PHI, 3),
            "deviation": round(fwh_dev, 1),
            "score": compute_metric_score(fwh_dev),
            "reference": "Powell & Humphreys 1984",
            "category": "Golden Ratio",
        })

    if "nose_width_length" in gr:
        nwl = gr["nose_width_length"]
        nwl_dev = abs(nwl - (1 / PHI)) / (1 / PHI) * 100
        metrics.append({
            "name": "Nose Width/Length (Golden Ratio)",
            "measured_value": round(nwl, 3),
            "ideal_value": round(1 / PHI, 3),
            "deviation": round(nwl_dev, 1),
            "score": compute_metric_score(nwl_dev),
            "reference": "Powell & Humphreys 1984",
            "category": "Golden Ratio",
        })

    if "mouth_nose_width" in gr:
        mnw = gr["mouth_nose_width"]
        mnw_dev = abs(mnw - PHI) / PHI * 100
        metrics.append({
            "name": "Mouth/Nose Width Ratio",
            "measured_value": round(mnw, 3),
            "ideal_value": round(PHI, 3),
            "deviation": round(mnw_dev, 1),
            "score": compute_metric_score(mnw_dev),
            "reference": "Powell & Humphreys 1984",
            "category": "Golden Ratio",
        })

    sym = measure_symmetry(landmarks)
    sym_dev = (1 - sym["symmetry_index"]) * 100
    metrics.append({
        "name": "Facial Symmetry Index",
        "measured_value": sym["symmetry_index"],
        "ideal_value": 1.0,
        "deviation": round(sym_dev, 1),
        "score": compute_metric_score(sym_dev),
        "reference": "Farkas 1994; Peck et al. 2008",
        "category": "Symmetry",
    })

    eye_sp = measure_eye_spacing(landmarks)
    esp_dev = abs(eye_sp["eye_spacing_ratio"] - 0.46) / 0.46 * 100
    metrics.append({
        "name": "Eye Spacing Ratio",
        "measured_value": eye_sp["eye_spacing_ratio"],
        "ideal_value": 0.46,
        "deviation": round(esp_dev, 1),
        "score": compute_metric_score(esp_dev),
        "reference": "Farkas 1994",
        "category": "Proportions",
    })

    lip = measure_lip_ratio(landmarks)
    if lip["lip_ratio"] > 0 and lip["lip_ratio"] < 5:
        lip_dev = abs(lip["lip_ratio"] - 0.625) / 0.625 * 100
        metrics.append({
            "name": "Lip Ratio (Upper/Lower)",
            "measured_value": lip["lip_ratio"],
            "ideal_value": 0.625,
            "deviation": round(lip_dev, 1),
            "score": compute_metric_score(lip_dev),
            "reference": "Powell & Humphreys 1984",
            "category": "Proportions",
        })
    else:
        metrics.append({
            "name": "Lip Ratio (Upper/Lower)",
            "measured_value": lip["lip_ratio"],
            "ideal_value": 0.625,
            "deviation": 100.0,
            "score": 15.0,
            "reference": "Powell & Humphreys 1984",
            "category": "Proportions",
            "note": "Measurement may be inaccurate — slightly part lips for better detection",
        })

    nose_proj = measure_nose_projection(landmarks)
    goode_ideal = 0.575
    goode_dev = abs(nose_proj["goode_ratio"] - goode_ideal) / goode_ideal * 100
    metrics.append({
        "name": "Nasal Projection (Goode's Ratio)",
        "measured_value": nose_proj["goode_ratio"],
        "ideal_value": goode_ideal,
        "deviation": round(goode_dev, 1),
        "score": compute_metric_score(goode_dev),
        "reference": "Goode 1981",
        "category": "Proportions",
    })

    gonial = measure_gonial_angle(landmarks)
    ideal_gonial = 120.0 if sex == "male" else 126.0
    gonial_dev = abs(gonial["avg_gonial_angle"] - ideal_gonial) / ideal_gonial * 100
    metrics.append({
        "name": "Gonial Angle (Jawline)",
        "measured_value": gonial["avg_gonial_angle"],
        "ideal_value": ideal_gonial,
        "deviation": round(gonial_dev, 1),
        "score": compute_metric_score(gonial_dev),
        "reference": "Argyropoulos & Sassouni 1989",
        "category": "Profile",
    })

    cj = measure_cheekbone_jaw_ratio(landmarks)
    ideal_cj = 1.3 if sex == "male" else 1.2
    cj_dev = abs(cj["cheekbone_jaw_ratio"] - ideal_cj) / ideal_cj * 100
    metrics.append({
        "name": "Cheekbone/Jaw Width Ratio",
        "measured_value": cj["cheekbone_jaw_ratio"],
        "ideal_value": ideal_cj,
        "deviation": round(cj_dev, 1),
        "score": compute_metric_score(cj_dev),
        "reference": "Farkas 1994",
        "category": "Proportions",
    })

    skin = measure_skin_uniformity(frame)
    skin_dev = (1 - skin["uniformity_score"]) * 100
    metrics.append({
        "name": "Skin Tone Uniformity",
        "measured_value": skin["uniformity_score"],
        "ideal_value": 1.0,
        "deviation": round(skin_dev, 1),
        "score": compute_metric_score(skin_dev),
        "reference": "Image analysis - skin tone statistics",
        "category": "Skin",
    })

    return metrics
