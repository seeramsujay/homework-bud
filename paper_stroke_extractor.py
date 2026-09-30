"""
Paper Stroke Extractor for Homework-Bud
Extracts stroke sequences (x, -y, eos) from photos of handwritten paper pages
and formats them for Graves RNN style conditioning.
"""

import os
import json
import cv2
import numpy as np
from typing import Tuple, List, Optional
import drawing


def skeletonize(img_bin: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization using OpenCV.
    Input img_bin: binary image with 255 for stroke, 0 for background.
    """
    size = np.size(img_bin)
    skel = np.zeros(img_bin.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = np.copy(img_bin)

    done = False
    while not done:
        eroded = cv2.erode(temp, element)
        temp_open = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        subset = cv2.subtract(eroded, temp_open)
        skel = cv2.bitwise_or(skel, subset)
        temp = eroded.copy()

        zeros = size - cv2.countNonZero(temp)
        if zeros == size:
            done = True

    return skel


def trace_skeleton_strokes(skel: np.ndarray) -> List[np.ndarray]:
    """
    Traces skeleton connected components into ordered (x, y) stroke paths
    ordered generally from left to right.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel, connectivity=8)
    # Sort components by their leftmost x coordinate (reading order)
    comp_indices = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_LEFT])

    strokes = []
    for idx in comp_indices:
        pts = np.argwhere(labels == idx)
        if len(pts) < 4:
            continue
        # pts has (row, col) = (y, x)
        ys = pts[:, 0]
        xs = pts[:, 1]

        # Order points: start at the leftmost point
        start_idx = np.argmin(xs)
        ordered = [start_idx]
        visited = set(ordered)

        current = start_idx
        while len(visited) < len(pts):
            dists = np.hypot(xs - xs[current], ys - ys[current])
            dists[list(visited)] = np.inf
            nearest = np.argmin(dists)
            if dists[nearest] > 10.0:  # gap too large, break stroke
                break
            visited.add(nearest)
            ordered.append(nearest)
            current = nearest

        stroke_pts = np.stack([xs[ordered], ys[ordered]], axis=1)
        if len(stroke_pts) >= 4:
            strokes.append(stroke_pts)

    return strokes


def process_paper_sample(
    image_path: str,
    transcription_text: str,
    output_prefix: str = "styles/style-custom"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts strokes from a paper photo and exports style .npy files for demo.py.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to extract pen strokes from paper
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # Thin to 1-pixel skeleton
    skel = skeletonize(binary)

    # Trace strokes into continuous paths
    stroke_paths = trace_skeleton_strokes(skel)

    raw_coords = []
    for stroke in stroke_paths:
        for i, pt in enumerate(stroke):
            eos = 1.0 if (i == len(stroke) - 1) else 0.0
            raw_coords.append([float(pt[0]), -1.0 * float(pt[1]), eos])

    if len(raw_coords) == 0:
        raise ValueError("No strokes could be extracted from image!")

    coords = np.array(raw_coords, dtype=np.float32)

    # Normalize using standard drawing.py pipeline
    coords = drawing.align(coords)
    coords = drawing.denoise(coords)
    offsets = drawing.coords_to_offsets(coords)
    offsets = offsets[:drawing.MAX_STROKE_LEN]
    offsets = drawing.normalize(offsets)

    # Prepare characters
    clean_chars = transcription_text.strip()
    encoded_chars = drawing.encode_ascii(clean_chars)[:drawing.MAX_CHAR_LEN]

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    np.save(f"{output_prefix}-strokes.npy", offsets)
    np.save(f"{output_prefix}-chars.npy", np.array(clean_chars))

    print(f"[OK] Extracted {len(offsets)} stroke points from {image_path}")
    print(f"[OK] Saved style to {output_prefix}-strokes.npy and {output_prefix}-chars.npy")
    return offsets, np.array(clean_chars)


def process_json_style(
    json_path: str,
    output_prefix: str = "styles/style-custom"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts style exported from touch_recorder.html into model style files.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    text = data['text']
    coords = np.array(data['coords'], dtype=np.float32)

    coords = drawing.align(coords)
    coords = drawing.denoise(coords)
    offsets = drawing.coords_to_offsets(coords)
    offsets = offsets[:drawing.MAX_STROKE_LEN]
    offsets = drawing.normalize(offsets)

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    np.save(f"{output_prefix}-strokes.npy", offsets)
    np.save(f"{output_prefix}-chars.npy", np.array(text))

    print(f"[OK] Converted {len(offsets)} touch points from {json_path}")
    print(f"[OK] Saved style to {output_prefix}-strokes.npy and {output_prefix}-chars.npy")
    return offsets, np.array(text)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python paper_stroke_extractor.py <image_path> <transcription_text>")
        print("  python paper_stroke_extractor.py --json <touch_style.json>")
    else:
        if sys.argv[1] == '--json':
            process_json_style(sys.argv[2])
        else:
            process_paper_sample(sys.argv[1], sys.argv[2])
