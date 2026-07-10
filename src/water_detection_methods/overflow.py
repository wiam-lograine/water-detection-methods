import numpy as np


def rectangular_zone(mask_shape, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0):
    """Create a binary zone mask from relative coordinates.

    Coordinates are relative values in [0, 1], not pixels. This makes the zone
    definition independent from image resolution. For example, y_min=0.75 and
    y_max=1.0 selects the bottom quarter of the image.

    This is useful for defining a critical area such as the border of a basin
    or a zone where water should not appear.
    """
    height, width = mask_shape[:2]
    x0 = int(width * x_min)
    x1 = int(width * x_max)
    y0 = int(height * y_min)
    y1 = int(height * y_max)

    zone = np.zeros((height, width), dtype=bool)
    zone[y0:y1, x0:x1] = True
    return zone


def water_ratio_in_zone(water_mask, zone_mask):
    """Return the ratio of zone pixels predicted as water.

    A value of 0.0 means no detected water in the critical zone. A value of 1.0
    means the whole critical zone is detected as water.
    """
    water_mask = np.asarray(water_mask).squeeze() > 0
    zone_mask = np.asarray(zone_mask).squeeze().astype(bool)
    zone_size = zone_mask.sum()
    if zone_size == 0:
        # Empty zones can happen with wrong coordinates. Returning 0 avoids a
        # crash and means "no evidence of overflow from this zone".
        return 0.0
    return float(np.logical_and(water_mask, zone_mask).sum() / zone_size)


def overflow_confidence(water_mask, zone_mask, alert_ratio=0.15):
    """Estimate overflow confidence from water presence in a critical zone.

    `alert_ratio=0.15` means that if at least 15% of the critical zone is
    detected as water, the confidence reaches 100%. This is intentionally
    conservative for overflow detection: water appearing in a forbidden or
    critical area should quickly raise an alert.

    The confidence is not a calibrated probability. It is a rule-based score
    derived from the segmentation mask and should be validated on real images.
    """
    ratio = water_ratio_in_zone(water_mask, zone_mask)
    confidence = min(1.0, ratio / alert_ratio) if alert_ratio > 0 else 0.0
    return {
        "zone_water_ratio": ratio,
        "overflow_confidence": confidence,
        "overflow_confidence_percent": round(confidence * 100, 2),
    }


def overflow_label(confidence, low=0.3, high=0.7):
    """Convert a confidence score to a readable decision.

    Default thresholds:
    - low=0.3: below 30%, the system considers the situation normal because
      only a small part of the critical zone is detected as water.
    - high=0.7: above 70%, the system raises a probable overflow decision.
    - between both values, the image is marked as surveillance.

    These thresholds create three interpretable levels for the prototype. They
    should be adjusted with the project supervisor after observing real cases.
    """
    if confidence < low:
        return "normal"
    if confidence < high:
        return "surveillance"
    return "debordement_probable"
