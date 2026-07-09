import numpy as np


def rectangular_zone(mask_shape, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0):
    """Create a binary zone mask from relative coordinates."""
    height, width = mask_shape[:2]
    x0 = int(width * x_min)
    x1 = int(width * x_max)
    y0 = int(height * y_min)
    y1 = int(height * y_max)

    zone = np.zeros((height, width), dtype=bool)
    zone[y0:y1, x0:x1] = True
    return zone


def water_ratio_in_zone(water_mask, zone_mask):
    """Return the ratio of zone pixels predicted as water."""
    water_mask = np.asarray(water_mask).squeeze() > 0
    zone_mask = np.asarray(zone_mask).squeeze().astype(bool)
    zone_size = zone_mask.sum()
    if zone_size == 0:
        return 0.0
    return float(np.logical_and(water_mask, zone_mask).sum() / zone_size)


def overflow_confidence(water_mask, zone_mask, alert_ratio=0.15):
    """Estimate overflow confidence from water presence in a critical zone."""
    ratio = water_ratio_in_zone(water_mask, zone_mask)
    confidence = min(1.0, ratio / alert_ratio) if alert_ratio > 0 else 0.0
    return {
        "zone_water_ratio": ratio,
        "overflow_confidence": confidence,
        "overflow_confidence_percent": round(confidence * 100, 2),
    }


def overflow_label(confidence, low=0.3, high=0.7):
    """Convert a confidence score to a readable decision."""
    if confidence < low:
        return "normal"
    if confidence < high:
        return "surveillance"
    return "debordement_probable"

