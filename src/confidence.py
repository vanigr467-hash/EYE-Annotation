from typing import Dict, Any
from src.pupil_detection import PupilResult
from src.iris_detection import IrisResult

def compute_overall_confidence(
    pupil_res: PupilResult,
    iris_res: IrisResult
) -> Dict[str, Any]:
    """
    Evaluates multi-factor confidence rating for both pupil and iris detections.
    Returns composite score, confidence category (High, Medium, Low), and warning text.
    """
    if not pupil_res.detected:
        return {
            "pupil_confidence": 0.0,
            "pupil_level": "Low",
            "iris_confidence": 0.0,
            "iris_level": "Low",
            "overall_confidence": 0.0,
            "overall_level": "Low",
            "warning": "CRITICAL: Pupil detection failed. Annotation is invalid."
        }

    # Pupil Confidence Level
    p_conf = pupil_res.confidence
    if p_conf >= 75.0:
        p_level = "High"
    elif p_conf >= 50.0:
        p_level = "Medium"
    else:
        p_level = "Low"

    # Iris Confidence Level
    if not iris_res.detected:
        i_conf = 20.0
        i_level = "Low"
    else:
        i_conf = iris_res.confidence
        if i_conf >= 75.0:
            i_level = "High"
        elif i_conf >= 50.0:
            i_level = "Medium"
        else:
            i_level = "Low"

    # Overall Combined Confidence
    overall_conf = (p_conf * 0.6) + (i_conf * 0.4)
    if overall_conf >= 75.0:
        overall_level = "High"
    elif overall_conf >= 50.0:
        overall_level = "Medium"
    else:
        overall_level = "Low"

    # Generate user-friendly advice/warning
    warnings = []
    if p_level == "Low":
        warnings.append("Pupil boundary confidence is low. Ensure the eye is open and lighting is even.")
    if i_level == "Low":
        warnings.append("Iris boundary confidence is low. Shadows or eyelid occlusion may affect detection.")

    warning_msg = " ".join(warnings) if warnings else "High confidence detection. Boundary annotations verified."

    return {
        "pupil_confidence": round(p_conf, 1),
        "pupil_level": p_level,
        "iris_confidence": round(i_conf, 1),
        "iris_level": i_level,
        "overall_confidence": round(overall_conf, 1),
        "overall_level": overall_level,
        "warning": warning_msg
    }
