"""
Ruled Sheet Line & Margin Detector for Homework-Bud
Detects ruled notebook lines, margins, and page tilt from phone camera photos.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class PageLayout:
    width: int
    height: int
    baselines: List[float]       # Y-coordinates of ruled lines across page
    line_spacing: float          # Average spacing between lines in pixels
    left_margin: float           # X-coordinate where text starts
    right_margin: float          # X-coordinate where text ends
    tilt_angle_deg: float        # Rotation angle of ruled lines in degrees
    has_detected_lines: bool     # True if real ruled lines were found


class RuledSheetDetector:
    def __init__(self, min_line_spacing: int = 15, max_lines: int = 60):
        self.min_line_spacing = min_line_spacing
        self.max_lines = max_lines

    def analyze_image(self, image_path: str) -> PageLayout:
        """
        Loads an image from image_path and extracts page layout parameters.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Detect ruled lines and angle
        baselines, angle = self._detect_horizontal_lines(gray)

        # 2. Detect left margin (red/pink margin line or vertical boundary)
        left_margin = self._detect_left_margin(img, gray)
        right_margin = w * 0.94  # standard right margin boundary

        # Fallback if no lines found (e.g. unruled paper photo)
        if len(baselines) < 3:
            default_spacing = h / 28.0  # standard ~28 lines per sheet
            baselines = [float(y) for y in np.arange(h * 0.12, h * 0.92, default_spacing)]
            line_spacing = default_spacing
            has_detected_lines = False
            angle = 0.0
        else:
            diffs = np.diff(baselines)
            valid_diffs = diffs[(diffs >= self.min_line_spacing) & (diffs <= np.percentile(diffs, 90) * 1.5)]
            line_spacing = float(np.median(valid_diffs)) if len(valid_diffs) > 0 else float(np.median(diffs))
            has_detected_lines = True

        return PageLayout(
            width=w,
            height=h,
            baselines=baselines,
            line_spacing=line_spacing,
            left_margin=left_margin,
            right_margin=right_margin,
            tilt_angle_deg=angle,
            has_detected_lines=has_detected_lines
        )

    def _detect_horizontal_lines(self, gray: np.ndarray) -> Tuple[List[float], float]:
        h, w = gray.shape

        # Blur to reduce paper texture / grain
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding to pick up faint blue/grey notebook lines
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 9
        )

        # Horizontal morphological kernel
        kernel_len = max(20, int(w * 0.06))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Detect lines using Probabilistic Hough Transform
        min_line_length = int(w * 0.25)
        max_line_gap = int(w * 0.05)
        lines = cv2.HoughLinesP(
            horizontal_lines, 1, np.pi / 180,
            threshold=50, minLineLength=min_line_length, maxLineGap=max_line_gap
        )

        if lines is None or len(lines) == 0:
            return [], 0.0

        # Collect Y-positions and angles
        y_centers = []
        angles = []
        for line in lines:
            pts = np.asarray(line).ravel()
            if len(pts) < 4:
                continue
            x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < 1e-3:
                continue
            deg = float(np.degrees(np.arctan2(dy, dx)))
            # Keep near-horizontal lines (-10 to +10 degrees)
            if abs(deg) <= 10.0:
                y_center = float(y1 + y2) / 2.0
                y_centers.append(y_center)
                angles.append(deg)

        if not y_centers:
            return [], 0.0

        median_angle = float(np.median(angles))
        y_centers.sort()

        # Cluster lines that belong to the same ruled stripe (within min_spacing / 2)
        clustered_ys = []
        cluster = [y_centers[0]]
        cluster_threshold = max(8.0, self.min_line_spacing * 0.4)

        for y in y_centers[1:]:
            if y - cluster[-1] < cluster_threshold:
                cluster.append(y)
            else:
                clustered_ys.append(float(np.mean(cluster)))
                cluster = [y]
        if cluster:
            clustered_ys.append(float(np.mean(cluster)))

        return clustered_ys, median_angle

    def _detect_left_margin(self, img: np.ndarray, gray: np.ndarray) -> float:
        h, w = gray.shape
        left_bound = int(w * 0.35)
        left_roi = img[:, :left_bound]

        # 1. Try color-based margin line detection (red/pink vertical margin line)
        hsv = cv2.cvtColor(left_roi, cv2.COLOR_BGR2HSV)
        # Red hue wraps around 0 and 180 in OpenCV
        mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 40, 40]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.15)))
        red_vert = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, vertical_kernel)

        col_sums = np.sum(red_vert, axis=0)
        if np.max(col_sums) > h * 255 * 0.2:  # found prominent vertical red line
            margin_x = float(np.argmax(col_sums))
            # Text starts slightly to the right of the margin line
            return margin_x + max(12.0, w * 0.015)

        # 2. Try vertical morphological opening on grayscale
        binary = cv2.adaptiveThreshold(
            gray[:, :left_bound], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 9
        )
        vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        col_sums_bw = np.sum(vert, axis=0)
        if np.max(col_sums_bw) > h * 255 * 0.25:
            margin_x = float(np.argmax(col_sums_bw))
            return margin_x + max(12.0, w * 0.015)

        # Default standard left margin (8% of page width)
        return float(w * 0.08)
