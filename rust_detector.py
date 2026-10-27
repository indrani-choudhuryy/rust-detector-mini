"""Core logic: estimate the percentage of rust (corrosion) in an image using HSV color thresholding."""

import cv2
import numpy as np

# OpenCV HSV ranges: H = 0-179, S = 0-255, V = 0-255
# Rust is mostly orange-brown or reddish-brown.
RUST_HUE_RANGES = [
    (5, 25),   # orange-brown
    (0, 5),    # reddish-brown
]

LOW_LIMIT = 5.0      # < 5%  -> Low
MEDIUM_LIMIT = 20.0  # < 20% -> Medium, anything above is High


def classify(rust_percent: float) -> str:
    """Convert a rust percentage into a Low / Medium / High corrosion level."""
    if rust_percent < LOW_LIMIT:
        return "Low"
    if rust_percent < MEDIUM_LIMIT:
        return "Medium"
    return "High"


def build_rust_mask(image_bgr: np.ndarray, min_saturation: int = 80, min_value: int = 40) -> np.ndarray:
    """Return a binary mask where 255 marks pixels that look like rust."""
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)  # reduce noise
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_low, h_high in RUST_HUE_RANGES:
        lower = np.array([h_low, min_saturation, min_value])
        upper = np.array([h_high, 255, 255])
        mask |= cv2.inRange(hsv, lower, upper)

    # Remove small isolated specks (open) and fill small gaps (close)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def analyze_rust(image_bgr: np.ndarray, min_saturation: int = 80, min_value: int = 40):
    """
    Analyze an image for rust.

    Returns:
        rust_percent (float), level (str), mask (np.ndarray), overlay (np.ndarray, BGR)
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Image is empty or could not be read.")

    # Downscale very large images so processing stays fast
    h, w = image_bgr.shape[:2]
    max_side = 800
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

    mask = build_rust_mask(image_bgr, min_saturation, min_value)
    rust_percent = float(np.count_nonzero(mask)) / mask.size * 100.0
    level = classify(rust_percent)

    # Highlight rust regions in red
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(image_bgr, 0.5, overlay, 0.5, 0)

    return rust_percent, level, mask, overlay
