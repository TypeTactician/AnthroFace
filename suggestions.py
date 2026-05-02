"""Improvement database: maps metric scores to evidence-based suggestions."""


def get_suggestions(metrics: list[dict], category_scores: dict) -> list[dict]:
    """Generate improvement suggestions for metrics scoring below 80.

    Returns list of: {priority, metric_name, score, suggestion, evidence}
    """
    suggestions = []
    for m in metrics:
        if m["score"] >= 80:
            continue
        s = _lookup_suggestion(m)
        if s:
            suggestions.append({
                "priority": len(suggestions) + 1,
                "metric_name": m["name"],
                "score": m["score"],
                "suggestion": s["text"],
                "evidence": s["evidence"],
                "category": m["category"],
            })

    suggestions.sort(key=lambda x: x["score"])
    for i, s in enumerate(suggestions):
        s["priority"] = i + 1
    return suggestions[:10]


def _lookup_suggestion(metric: dict) -> dict | None:
    """Match metrics to suggestions using exact name patterns, not substrings."""
    name = metric["name"]
    score = metric["score"]

    if name == "Facial Symmetry Index":
        return _symmetry_suggestions(score)
    if name in ("Facial Upper Third", "Facial Middle Third", "Facial Lower Third"):
        return _thirds_suggestions(score, metric)
    if name == "Interpupillary / Intercanthal Fifth":
        return _fifths_suggestions(score, metric)
    if name == "Nose Width (vs Eye Width)":
        return _nose_width_suggestions(score, metric)
    if "Golden Ratio" in name:
        return _golden_ratio_suggestions(score, name)
    if name == "Eye Spacing Ratio":
        return _eye_spacing_suggestions(score, metric)
    if name == "Lip Ratio (Upper/Lower)":
        return _lip_suggestions(score, metric)
    if name == "Nasal Projection (Goode's Ratio)":
        return _nasal_projection_suggestions(score, metric)
    if name == "Gonial Angle (Jawline)":
        return _gonial_suggestions(score, metric)
    if name == "Cheekbone/Jaw Width Ratio":
        return _cheekbone_suggestions(score, metric)
    if name == "Skin Tone Uniformity":
        return _skin_suggestions(score, metric)
    return None


def _symmetry_suggestions(score: float) -> dict:
    if score < 70:
        return {
            "text": "Consider consulting an orthodontist. Dental midline asymmetry is the most common correctable cause of facial asymmetry. Orthodontic treatment or surgical correction can significantly improve symmetry.",
            "evidence": "Peck et al. (2008). Facial asymmetry: prevalence and causes. American Journal of Orthodontics.",
        }
    return {
        "text": "Unilateral masseter hypertrophy may cause jaw asymmetry. Chewing on both sides equally can help over time. Consider alternating chewing sides and avoiding gum on one side only.",
        "evidence": "Raustia & Salonen (1996). Masseter muscle asymmetry in unilateral chewing.",
    }


def _thirds_suggestions(score: float, metric: dict) -> dict:
    measured = metric.get("measured_value", 0.33)
    name = metric.get("name", "")

    if "Upper" in name:
        return {
            "text": "Upper third (hairline to brow) relies on automated hairline detection. "
                    "Accuracy depends on lighting and hair color contrast. "
                    "Ensure your hairline is clearly visible and not obscured by bangs or shadows.",
            "evidence": "Farkas (1994). Anthropometry of the Head and Face.",
        }
    if "Lower" in name and measured > 0.36:
        return {
            "text": "Your lower third is proportionally longer than ideal. This is often related to mouth breathing. Nasal breathing exercises and myofunctional therapy can help improve facial posture over time.",
            "evidence": "Jefferson (2010). Mouth breathing and facial development. Journal of Orthodontics.",
        }
    if "Lower" in name and measured < 0.30:
        return {
            "text": "Your lower third is shorter than ideal. Check for vertical dimension of occlusion. A dentist can evaluate if dental wear is reducing face height. Orthodontic or restorative options exist.",
            "evidence": "Farkas (1994). Anthropometry of the Head and Face.",
        }
    if "Middle" in name and measured > 0.36:
        return {
            "text": "Your middle third is proportionally longer. Non-surgical options are limited for mid-face length. Consult a craniofacial specialist for options.",
            "evidence": "Farkas (1994). Anthropometry of the Head and Face.",
        }
    return {
        "text": "Facial thirds are reasonably balanced. Maintain good posture and nasal breathing.",
        "evidence": "Farkas (1994).",
    }


def _fifths_suggestions(score: float, metric: dict) -> dict:
    return {
        "text": "Facial fifths are within acceptable range. Good overall horizontal proportion.",
        "evidence": "Farkas (1994). Anthropometry of the Head and Face.",
    }


def _nose_width_suggestions(score: float, metric: dict) -> dict:
    if score < 60:
        return {
            "text": "Nose width is wider relative to eye width. Non-surgical rhinoplasty (filler) can adjust perceived width by improving nasal bridge definition. Contouring makeup techniques provide immediate visual effect.",
            "evidence": "Barton et al. (2017). Liquid rhinoplasty techniques. Aesthetic Surgery Journal.",
        }
    return {
        "text": "Nose width is slightly off from ideal. Contouring techniques with makeup can create the illusion of narrower nasal width.",
        "evidence": "Makeup artistry principles based on facial proportion theory.",
    }


def _golden_ratio_suggestions(score: float, name: str) -> dict:
    return {
        "text": f"Golden ratio proportion in '{name}' deviates from ideal. Note that golden ratio adherence is a mathematical ideal, not a beauty standard. Natural faces rarely achieve perfect phi ratios.",
        "evidence": "Powell & Humphreys (1984). Proportions of the Aesthetic Face.",
    }


def _eye_spacing_suggestions(score: float, metric: dict) -> dict:
    ratio = metric.get("measured_value", 0.46)
    if ratio > 0.50:
        return {
            "text": "Eyes appear wider-set than average (hypertelorism tendency). Cannot be changed non-surgically. Makeup tip: darker inner corner eyeshadow reduces perceived spacing. Avoid highlighter on inner corners.",
            "evidence": "Tebble (1988). Applied Facial Anatomy for Artists.",
        }
    if ratio < 0.42:
        return {
            "text": "Eyes appear closer-set than average. Highlight inner corners and use lighter eyeshadow on inner lids to create the illusion of wider spacing.",
            "evidence": "Tebble (1988). Applied Facial Anatomy for Artists.",
        }
    return {
        "text": "Eye spacing is within normal range. No specific interventions needed.",
        "evidence": "Farkas (1994).",
    }


def _lip_suggestions(score: float, metric: dict) -> dict:
    ratio = metric.get("measured_value", 0.625)
    if ratio > 3:
        return {
            "text": "Lip ratio measurement may be inaccurate. Try slightly parting your lips during capture for more accurate detection. If confirmed, dermal fillers can address lip proportion.",
            "evidence": "Cohen et al. (2018). Lip augmentation: anatomy and technique. Plastic and Reconstructive Surgery.",
        }
    return {
        "text": "Lip ratio deviates from the ideal 1:1.6. Dermal fillers can address lip proportion asymmetry. Hydration and lip care improve overall lip appearance significantly.",
        "evidence": "Cohen et al. (2018). Lip augmentation: anatomy and technique. Plastic and Reconstructive Surgery.",
    }


def _nasal_projection_suggestions(score: float, metric: dict) -> dict:
    goode = metric.get("measured_value", 0.575)
    if goode < 0.55:
        return {
            "text": "Nasal projection is below the ideal range (Goode's ratio < 0.55). Tip exercises have no proven effect. Consult a rhinoplasty specialist if this is a concern. Tip rhinoplasty is minimally invasive with high satisfaction rates.",
            "evidence": "Goode (1981). Nasal projection measurement. Archives of Otolaryngology.",
        }
    if goode > 0.60:
        return {
            "text": "Nasal projection exceeds the ideal range. This creates a prominent nasal profile. Reduction rhinoplasty can address this if desired.",
            "evidence": "Goode (1981). Nasal projection measurement.",
        }
    return {
        "text": "Nasal projection is within the ideal range.",
        "evidence": "Goode (1981).",
    }


def _gonial_suggestions(score: float, metric: dict) -> dict:
    angle = metric.get("measured_value", 120)
    if angle > 130:
        return {
            "text": "Gonial angle is too obtuse (weak jaw definition). Mewing (correct tongue posture) has growing anecdotal support. Masseter exercises: chew hard foods (mastic gum) consistently. Sleep position: avoid stomach sleeping which pushes jaw back. Posture correction improves jaw appearance.",
            "evidence": "Argyropoulos & Sassouni (1989). Comparison of craniofacial morphology. American Journal of Orthodontics.",
        }
    if angle < 110:
        return {
            "text": "Gonial angle is too acute. This creates a very square jaw. Botox to the masseter muscles can soften the jaw angle if desired.",
            "evidence": "Ahn & Kim (2013). Botulinum toxin for masseter reduction. Journal of Cosmetic Dermatology.",
        }
    return {
        "text": "Jawline angle is within acceptable range. Maintain good posture and jaw exercises.",
        "evidence": "Argyropoulos & Sassouni (1989).",
    }


def _cheekbone_suggestions(score: float, metric: dict) -> dict:
    return {
        "text": "Cheekbone to jaw width ratio deviates from ideal. Dermal fillers to the cheekbones can improve this ratio non-surgically. Facial exercises have limited evidence for changing bone structure.",
        "evidence": "Farkas (1994). Anthropometry of the Head and Face.",
    }


def _skin_suggestions(score: float, metric: dict) -> dict:
    if score < 60:
        return {
            "text": "Skin tone uniformity is below average. Improving skin texture and tone has the highest ROI of any facial improvement. Daily routine: SPF 30+ sunscreen, retinol at night, consistent hydration, gentle cleanser. Consider consulting a dermatologist for targeted treatments.",
            "evidence": "Kafi et al. (2007). Improvement of naturally aged skin with retinol. Archives of Dermatology.",
        }
    return {
        "text": "Skin tone uniformity can be improved. Consider a consistent skincare routine: SPF daily, retinol, vitamin C serum, and adequate hydration.",
        "evidence": "Kafi et al. (2007). Improvement of naturally aged skin with retinol.",
    }


def get_posture_warnings(head_tilt: float, is_profile: bool = False) -> list[str]:
    """Generate posture warnings based on head position data."""
    warnings = []
    if abs(head_tilt) > 10:
        warnings.append(
            f"Head tilt detected: {abs(head_tilt):.1f} degrees. "
            "Forward head posture reduces perceived jaw definition significantly. "
            "Wall angels, chin tucks: 3 sets of 10 daily. "
            "Reference: Ruivo et al. (2017). Cervical posture and muscle activation."
        )
    if is_profile:
        warnings.append(
            "Profile assessment suggests monitoring forward head posture. "
            "Maintain ear alignment over shoulders during daily activities."
        )
    return warnings
