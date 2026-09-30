"""
Ruled Line Segmenter for Handwriting Dataset Preparation.
Segments individual text lines from notebook pages, crops them into numbered PNG images
(line_0001.png, line_0002.png, ...), and runs neural OCR on each crop to produce
a review CSV (lines_transcription.csv).
"""

import os
import glob
import csv
import cv2
import numpy as np
import drawing


def segment_page_into_lines(
    image_path: str,
    output_dir: str,
    start_line_idx: int = 1,
    line_height_estimate: int = 44
):
    """
    Slices each written line using local ink density and row projection profiles.
    Extracts individual line crops with padding, writes line_XXXX.png, and returns crop metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    h, w = img.shape[:2]

    # Blue ink mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 1] >= 40)
    ink_bin = (blue_mask * 255).astype(np.uint8)

    # Clean isolated noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ink_bin = cv2.morphologyEx(ink_bin, cv2.MORPH_OPEN, kernel)

    # Horizontal projection profile across notebook text width
    x_start = int(w * 0.05)
    x_end = int(w * 0.95)
    ink_middle = ink_bin[:, x_start:x_end]

    row_sums = np.sum(ink_middle, axis=1) / 255.0
    # Kernel size tuned to handwriting line frequency (~line_height_estimate / 2)
    k_size = max(5, int(line_height_estimate * 0.4))
    if k_size % 2 == 0:
        k_size += 1
    smoothed = np.convolve(row_sums, np.ones(k_size) / float(k_size), mode='same')

    # Find peaks in smoothed profile (representing line centers)
    min_distance = int(line_height_estimate * 0.7)
    peaks = []
    threshold = np.max(smoothed) * 0.08

    for y in range(1, h - 1):
        if smoothed[y] > threshold and smoothed[y] >= smoothed[y - 1] and smoothed[y] >= smoothed[y + 1]:
            if not peaks or (y - peaks[-1]) >= min_distance:
                peaks.append(y)

    # Fallback to regular spacing if projection is diffuse
    if len(peaks) < 5:
        peaks = list(range(int(h * 0.08), int(h * 0.92), line_height_estimate))

    crops = []
    current_idx = start_line_idx
    half_box = int(line_height_estimate * 0.65)

    for peak_y in peaks:
        y1 = max(0, peak_y - half_box)
        y2 = min(h, peak_y + half_box)

        line_crop = img[y1:y2, x_start:x_end]

        # Verify crop has enough handwriting ink
        crop_hsv = cv2.cvtColor(line_crop, cv2.COLOR_BGR2HSV)
        crop_ink = (crop_hsv[:, :, 0] >= 85) & (crop_hsv[:, :, 0] <= 140) & (crop_hsv[:, :, 1] >= 40)
        if np.sum(crop_ink) < 60:
            continue

        filename = f"line_{current_idx:04d}.png"
        crop_path = os.path.join(output_dir, filename)
        cv2.imwrite(crop_path, line_crop)

        crops.append({
            'line_id': current_idx,
            'filename': filename,
            'filepath': crop_path,
            'source_image': os.path.basename(image_path),
            'y_range': (y1, y2)
        })
        current_idx += 1

    return crops, current_idx


def process_all_training_pages_to_lines(
    training_images_dir: str = "Training_Images",
    output_lines_dir: str = "segmented_lines",
    csv_path: str = "lines_transcription.csv"
):
    """
    Segments all training pages into individual numbered line PNGs,
    runs OCR on each line crop, and writes lines_transcription.csv for review.
    """
    image_paths = sorted(
        glob.glob(os.path.join(training_images_dir, "*.jpeg")) +
        glob.glob(os.path.join(training_images_dir, "*.jpg")) +
        glob.glob(os.path.join(training_images_dir, "*.png"))
    )
    image_paths = [p for p in image_paths if "blank" not in os.path.basename(p).lower()]

    if not image_paths:
        raise FileNotFoundError(f"No training images found in {training_images_dir}")

    print(f"[INFO] Segmenting {len(image_paths)} pages into line crops...")
    all_crops = []
    next_idx = 1

    for p in image_paths:
        crops, next_idx = segment_page_into_lines(p, output_lines_dir, start_line_idx=next_idx)
        print(f"  {os.path.basename(p)} -> {len(crops)} lines")
        all_crops.extend(crops)

    print(f"\n[INFO] Cropped {len(all_crops)} line images. Running neural OCR on each crop...")

    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)
        has_easyocr = True
    except Exception:
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            has_easyocr = True
        except Exception:
            import pytesseract
            has_easyocr = False

    csv_rows = []
    valid_charset = set(drawing.alphabet)

    for item in all_crops:
        crop_path = item['filepath']
        ocr_text = ""

        try:
            if has_easyocr:
                res = reader.readtext(crop_path, detail=0)
                ocr_text = " ".join(res).strip()
            else:
                crop = cv2.imread(crop_path)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                ocr_text = pytesseract.image_to_string(gray, config='--psm 7').strip()
        except Exception:
            ocr_text = ""

        # Normalize quotes and filter to valid English characters
        text = (
            ocr_text.replace('“', '"')
            .replace('”', '"')
            .replace('‘', "'")
            .replace('’', "'")
            .replace('—', '-')
            .replace('–', '-')
        )
        cleaned_ocr = "".join([c for c in text if c in valid_charset])
        cleaned_ocr = " ".join(cleaned_ocr.split())

        csv_rows.append({
            'line_id': item['line_id'],
            'filename': item['filename'],
            'source_page': item['source_image'],
            'ocr_transcription': cleaned_ocr,
            'user_verified_transcription': cleaned_ocr
        })

    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['line_id', 'filename', 'source_page', 'ocr_transcription', 'user_verified_transcription']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[OK] Line segmentation & initial OCR complete!")
    print(f"  Cropped images: {output_lines_dir}/")
    print(f"  Review CSV: {csv_path}")
    print(f"  Total lines ready for review: {len(csv_rows)}")
    return csv_path
