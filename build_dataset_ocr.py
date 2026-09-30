"""
Deep Learning Handwriting OCR and Dataset Builder for Homework-Bud
Uses modern neural OCR (EasyOCR English model) on Kaggle GPU
to accurately transcribe cursive/fast handwriting from notebook photos
and build training pairs (x.npy, c.npy) for fine-tuning.
Strictly configured for English alphabet and ASCII characters.
"""

import os
import glob
import string
import cv2
import numpy as np
import drawing
from paper_stroke_extractor import skeletonize, trace_skeleton_strokes


# Standard English printable character set
ENGLISH_CHARSET = set(drawing.alphabet)


def clean_english_text(raw_text: str) -> str:
    """
    Sanitizes raw OCR text to strictly keep valid English characters,
    digits, and punctuation supported by the handwriting synthesis model.
    """
    # Replace non-standard unicode quotes/dashes
    text = (
        raw_text.replace('“', '"')
        .replace('”', '"')
        .replace('‘', "'")
        .replace('’', "'")
        .replace('—', '-')
        .replace('–', '-')
    )
    # Strictly filter to English alphabet
    cleaned = ''.join([ch for ch in text if ch in ENGLISH_CHARSET])
    # Collapse multiple spaces
    cleaned = ' '.join(cleaned.split())
    return cleaned


class HandwritingDatasetBuilder:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.reader = None

    def load_neural_ocr(self):
        """Initializes EasyOCR neural recognizer strictly for English."""
        if self.reader is None:
            import easyocr
            print("[INFO] Initializing Neural OCR Reader on GPU (English only)...")
            self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
            print("[OK] English Neural OCR loaded.")

    def transcribe_page_lines(self, image_path: str):
        """
        Extracts individual lines of English text, crops them, and runs neural OCR.
        Returns a list of dicts: [{'crop': img, 'bbox': (x,y,w,h), 'text': '...'}]
        """
        self.load_neural_ocr()
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load {image_path}")

        # English paragraph recognition
        results = self.reader.readtext(img, paragraph=True, detail=1)

        extracted_lines = []
        for bbox, text in results:
            clean_text = clean_english_text(text)
            # Must have at least 3 valid English characters
            if len(clean_text) < 3:
                continue

            pts = np.array(bbox, dtype=np.int32)
            x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
            x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])

            # Pad bounding box slightly to avoid cutting off ascenders/descenders
            h_pad = int((y_max - y_min) * 0.15)
            w_pad = int((x_max - x_min) * 0.05)
            y1 = max(0, y_min - h_pad)
            y2 = min(img.shape[0], y_max + h_pad)
            x1 = max(0, x_min - w_pad)
            x2 = min(img.shape[1], x_max + w_pad)

            line_crop = img[y1:y2, x1:x2]
            extracted_lines.append({
                'crop': line_crop,
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'text': clean_text
            })

        print(f"[OK] Neural OCR extracted {len(extracted_lines)} English lines from {os.path.basename(image_path)}")
        return extracted_lines

    def build_dataset_from_images(
        self,
        image_paths: list,
        output_dir: str = "data/processed"
    ):
        """
        Extracts stroke sequences and paired English OCR transcriptions from all training images,
        and saves x.npy, x_len.npy, c.npy, c_len.npy ready for rnn.py fine-tuning.
        """
        all_strokes = []
        all_transcriptions = []

        for img_path in image_paths:
            print(f"\n[PROCESSING] {img_path}...")
            lines = self.transcribe_page_lines(img_path)

            for line_data in lines:
                crop = line_data['crop']
                clean_text = line_data['text']

                if len(clean_text) < 3:
                    continue

                # Filter blue ink
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                blue_mask = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 1] >= 40)
                ink_bin = (blue_mask * 255).astype(np.uint8)

                if cv2.countNonZero(ink_bin) < 50:
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
                all_transcriptions.append(clean_text[:drawing.MAX_CHAR_LEN])

        if not all_strokes:
            raise ValueError("No valid English line pairs extracted. Check image quality.")

        print(f"\n[DONE] Successfully processed {len(all_strokes)} English training pairs!")

        # Format arrays for Graves RNN
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

        print(f"[OK] Saved processed dataset to {output_dir}/")
        print(f"  x.npy: {x.shape}")
        print(f"  c.npy: {c.shape}")


if __name__ == '__main__':
    builder = HandwritingDatasetBuilder(use_gpu=False)
    images = sorted(glob.glob('Training_Images/*.jpeg'))
    if images:
        builder.build_dataset_from_images(images)
