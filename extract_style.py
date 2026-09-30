"""
Extracts training style strokes and character text from user handwritten notebook photos.
"""

import os
import glob
import cv2
import numpy as np
import drawing
from paper_stroke_extractor import skeletonize, trace_skeleton_strokes


def extract_user_style_from_page(
    image_path: str,
    output_prefix: str = "styles/style-user"
):
    """
    Extracts stroke sequence from the clearest writing line on the page to prime the RNN.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    h, w = img.shape[:2]

    # Mask royal blue ink specifically
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 138) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] <= 215)
    ink_bin = (blue_mask * 255).astype(np.uint8)

    # Clean isolated noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ink_bin = cv2.morphologyEx(ink_bin, cv2.MORPH_OPEN, kernel)

    # Find a clean, continuous line of text around the middle/top
    # Sum ink across horizontal rows
    row_sums = np.sum(ink_bin, axis=1)
    # Smooth row sums
    smoothed = np.convolve(row_sums, np.ones(25) / 25.0, mode='same')
    peaks = []
    min_dist = 35
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1] and smoothed[i] > 255 * 10:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)

    if not peaks:
        # Fallback to center region
        y_start, y_end = int(h * 0.25), int(h * 0.35)
    else:
        # Pick the most prominent line
        best_peak = max(peaks, key=lambda p: smoothed[p])
        y_start = max(0, best_peak - 25)
        y_end = min(h, best_peak + 25)

    line_crop = ink_bin[y_start:y_end, int(w * 0.08):int(w * 0.92)]

    # Skeletonize
    skel = skeletonize(line_crop)
    stroke_paths = trace_skeleton_strokes(skel)

    raw_coords = []
    for stroke in stroke_paths:
        for i, pt in enumerate(stroke):
            eos = 1.0 if (i == len(stroke) - 1) else 0.0
            raw_coords.append([float(pt[0]), -1.0 * float(pt[1]), eos])

    if len(raw_coords) == 0:
        raise ValueError("Could not extract enough stroke coordinates from the page.")

    coords = np.array(raw_coords, dtype=np.float32)
    coords = drawing.align(coords)
    coords = drawing.denoise(coords)
    offsets = drawing.coords_to_offsets(coords)
    offsets = offsets[:drawing.MAX_STROKE_LEN]
    offsets = drawing.normalize(offsets)

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    np.save(f"{output_prefix}-strokes.npy", offsets)
    np.save(f"{output_prefix}-chars.npy", np.array("sample handwriting"))

    print(f"[OK] Extracted {len(offsets)} stroke points from {os.path.basename(image_path)}")
    print(f"[OK] Saved custom style to {output_prefix}-strokes.npy")
    return offsets


if __name__ == '__main__':
    train_files = sorted(glob.glob('Training_Images/*.jpeg'))
    if train_files:
        extract_user_style_from_page(train_files[0], output_prefix='styles/style-user')
