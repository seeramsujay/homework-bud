"""
Dataset Builder from Verified Lines CSV and Cropped Images.
Pairs line images with user-verified text transcriptions and extracts
stroke sequences for Graves RNN fine-tuning.
"""

import os
import csv
import cv2
import numpy as np
import drawing
from paper_stroke_extractor import skeletonize, trace_skeleton_strokes


def build_dataset_from_verified_csv(
    csv_path: str = "lines_transcription.csv",
    lines_dir: str = "segmented_lines",
    output_dir: str = "data/processed"
):
    """
    Reads the user-reviewed CSV and corresponding line PNGs,
    extracts sequential strokes (dx, dy, eos), and saves x.npy, c.npy for fine-tuning.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Review CSV not found: {csv_path}")

    valid_charset = set(drawing.alphabet)
    rows = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use user_verified_transcription if filled, else fallback to ocr_transcription
            text = row.get('user_verified_transcription', '').strip()
            if not text:
                text = row.get('ocr_transcription', '').strip()

            clean_text = "".join([c for c in text if c in valid_charset])
            clean_text = " ".join(clean_text.split())

            if len(clean_text) >= 3:
                rows.append((row['filename'], clean_text))

    print(f"[INFO] Found {len(rows)} verified line entries in CSV.")

    all_strokes = []
    all_transcriptions = []

    for filename, text in rows:
        img_path = os.path.join(lines_dir, filename)
        if not os.path.exists(img_path):
            continue

        crop = cv2.imread(img_path)
        if crop is None:
            continue

        # Mask royal blue ink
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        blue_mask = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 1] >= 40)
        ink_bin = (blue_mask * 255).astype(np.uint8)

        if cv2.countNonZero(ink_bin) < 40:
            continue

        skel = skeletonize(ink_bin)
        stroke_paths = trace_skeleton_strokes(skel)

        raw_coords = []
        for stroke in stroke_paths:
            for i, pt in enumerate(stroke):
                eos = 1.0 if (i == len(stroke) - 1) else 0.0
                raw_coords.append([float(pt[0]), -1.0 * float(pt[1]), eos])

        if len(raw_coords) < 15:
            continue

        coords = np.array(raw_coords, dtype=np.float32)
        coords = drawing.align(coords)
        coords = drawing.denoise(coords)
        offsets = drawing.coords_to_offsets(coords)
        offsets = offsets[:drawing.MAX_STROKE_LEN]
        offsets = drawing.normalize(offsets)

        all_strokes.append(offsets)
        all_transcriptions.append(text[:drawing.MAX_CHAR_LEN])

    if not all_strokes:
        raise ValueError("Could not extract valid strokes from line crops.")

    print(f"[OK] Successfully built {len(all_strokes)} training pairs from verified lines!")

    num_samples = len(all_strokes)
    x = np.zeros([num_samples, drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([num_samples], dtype=np.int16)
    c = np.zeros([num_samples, drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([num_samples], dtype=np.int8)

    for i, (stroke, text) in enumerate(zip(all_strokes, all_transcriptions)):
        x[i, :len(stroke), :] = stroke
        x_len[i] = len(stroke)
        encoded = drawing.encode_ascii(text)
        c[i, :len(encoded)] = encoded
        c_len[i] = len(encoded)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'x.npy'), x)
    np.save(os.path.join(output_dir, 'x_len.npy'), x_len)
    np.save(os.path.join(output_dir, 'c.npy'), c)
    np.save(os.path.join(output_dir, 'c_len.npy'), c_len)

    print(f"[OK] Saved processed dataset in {output_dir}/")
    print(f"  x.npy: {x.shape}")
    print(f"  c.npy: {c.shape}")


if __name__ == '__main__':
    build_dataset_from_verified_csv()
