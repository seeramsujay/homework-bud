"""
Script to generate the complete homework_bud_kaggle.ipynb notebook.
"""

import json

def create_notebook():
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    add_md("""
# 📝 Homework-Bud: Automated Line Segmentation, OCR Review & Fine-Tuning
### Slices notebook pages into numbered line PNGs, generates a review CSV with OCR transcriptions on Kaggle, fine-tunes on your verified handwriting, and synthesizes ruled homework pages in Royal Blue ink.

---
### 🌟 Workflow:
1. **Segment Pages into Lines**: Crops each handwritten line into `segmented_lines/line_XXXX.png`.
2. **Neural OCR & CSV Export**: Runs OCR on each cropped line and saves `lines_transcription.csv` for easy review.
3. **Fine-Tuning on Kaggle GPU**: Trains on line pairs to adapt Alex Graves' RNN weights to your personal handwriting style.
4. **Ruled Sheet Detection & Alignment**: Snaps synthesized handwriting onto `Blank_Page.jpeg`.
5. **High-Contrast Scan Export**: Renders Royal Blue ink (`#3057a3`) with document-scanner contrast into a multi-page PDF.
    """)

    # Cell 1: Setup
    add_md("## 1. Setup & Environment")
    add_code("""
import os
import sys

# Clone project repository if running in a fresh Kaggle container
if not os.path.exists("checkpoints"):
    !git clone --depth 1 https://github.com/seeramsujay/homework-bud.git repo_code
    !cp -rn repo_code/* .
    !cp -rn repo_code/.* . 2>/dev/null || true

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Install required dependencies
!pip install --quiet easyocr svgwrite opencv-python-headless pillow reportlab scipy matplotlib

import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("Kaggle GPU environment initialized!")
    """)

    # Cell 2: Imports
    add_md("## 2. Load Pipeline Modules")
    add_code("""
from line_segmenter import process_all_training_pages_to_lines
from build_dataset_verified import build_dataset_from_verified_csv
from ruled_sheet_detector import RuledSheetDetector
from scan_renderer import ScanRenderer
from homework_engine import HomeworkEngine
from finetune_user import finetune_user_handwriting

print("All pipeline modules loaded successfully!")
    """)

    # Cell 3: Segment Pages into Numbered Line Crops + Run OCR to CSV
    add_md("""
## 3. Step 1: Crop Lines & Generate Review CSV (`lines_transcription.csv`)
Automatically detects line boundaries across all uploaded pages in `Training_Images/`,
saves individual numbered crops (`segmented_lines/line_0001.png`, ...),
runs neural OCR on each line, and writes `lines_transcription.csv`.
    """)
    add_code("""
# Locate training pages
train_dir = "Training_Images" if os.path.exists("Training_Images") else "/kaggle/input"
output_lines_dir = "segmented_lines"
csv_path = "lines_transcription.csv"

# Run line segmentation and line-level OCR
process_all_training_pages_to_lines(
    training_images_dir=train_dir,
    output_lines_dir=output_lines_dir,
    csv_path=csv_path
)

# Display first 15 entries for quick review
df = pd.read_csv(csv_path)
print(f"\\nGenerated CSV with {len(df)} lines! First 15 lines:")
display(df.head(15))
    """)

    # Cell 4: Preview Sample Line Crops
    add_md("## 4. Step 2: Visual Inspection of Cropped Lines")
    add_code("""
df = pd.read_csv("lines_transcription.csv")
sample_rows = df.head(6)

plt.figure(figsize=(14, 8))
for i, (_, row) in enumerate(sample_rows.iterrows()):
    img_p = os.path.join("segmented_lines", row['filename'])
    if os.path.exists(img_p):
        img = Image.open(img_p)
        plt.subplot(6, 1, i + 1)
        plt.imshow(img)
        plt.title(f"[{row['filename']}] OCR: \"{row['ocr_transcription']}\"", fontsize=10, loc='left')
        plt.axis("off")
plt.tight_layout()
plt.show()
    """)

    # Cell 5: Build Training Pairs from CSV & Fine-Tune
    add_md("""
## 5. Step 3: Build Dataset & Fine-Tune RNN on Kaggle GPU
Extracts stroke coordinates paired with verified text from the CSV and fine-tunes `model-17900`.
*(Tip: If you edited any text in `lines_transcription.csv`, it will automatically use your corrections!)*
    """)
    add_code("""
# Build paired dataset (x.npy, c.npy)
build_dataset_from_verified_csv(
    csv_path="lines_transcription.csv",
    lines_dir="segmented_lines",
    output_dir="data/processed"
)

# Run fine-tuning on Kaggle GPU
finetune_user_handwriting(
    data_dir="data/processed/",
    checkpoint_dir="checkpoints",
    warm_start_step=17900,
    finetune_steps=2000,
    learning_rate=0.00005,
    batch_size=16
)
    """)

    # Cell 6: Analyze Blank Ruled Page
    add_md("## 6. Step 4: Analyze Blank Ruled Notebook Sheet")
    add_code("""
blank_candidates = glob.glob("*blank*.jpeg") + glob.glob("*Blank*.jpeg") + glob.glob("/kaggle/input/**/*blank*.jpeg")
blank_sheet = blank_candidates[0] if blank_candidates else "Blank_Page.jpeg"

detector = RuledSheetDetector()
layout = detector.analyze_image(blank_sheet)
print(f"Detected {len(layout.baselines)} ruled lines.")
print(f"Line spacing: {layout.line_spacing:.1f}px | Left margin: {layout.left_margin:.1f}px | Tilt: {layout.tilt_angle_deg:.2f}°")

# Preview detection overlay
preview = cv2.imread(blank_sheet).copy()
for y in layout.baselines:
    cv2.line(preview, (int(layout.left_margin), int(y)), (int(layout.right_margin), int(y)), (0, 255, 0), 1)
cv2.line(preview, (int(layout.left_margin), 0), (int(layout.left_margin), layout.height), (255, 0, 0), 2)

plt.figure(figsize=(8, 11))
plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
plt.title("Detected Ruled Notebook Lines (Green) and Margin (Blue)")
plt.axis("off")
plt.show()
    """)

    # Cell 7: Synthesize Homework on Ruled Sheet
    add_md("## 7. Step 5: Synthesize Homework in Royal Blue Ink")
    add_code("""
# Royal Blue ink extracted from your pen (#3057a3)
ROYAL_BLUE = (48, 87, 163)

# Neatness bias (0.85 = neat, uniform handwriting)
NEATNESS_BIAS = 0.85

HOMEWORK_TEXT = \"\"\"
Homework Assignment: Sequential Deep Learning

1. How does the Attention Mechanism guide handwriting generation?
The attention mechanism acts as a soft window that slides across the character sequence as strokes are drawn. It determines which character the network is currently focusing on and synchronizes the pen's physical movement with the text.

2. Explain the mixture density output layer.
Instead of predicting deterministic coordinates, the mixture density network outputs the means, variances, correlations, and mixture weights of a bivariate Gaussian mixture model, along with a Bernoulli end-of-stroke probability.

3. Output Rendering:
The synthesized strokes are aligned directly with the detected ruled baselines and rendered in solid royal blue ink with scanner contrast.
\"\"\"

engine = HomeworkEngine(
    checkpoint_dir="checkpoints",
    ink_rgb=ROYAL_BLUE,
    bias=NEATNESS_BIAS
)

generated_images, pdf_path = engine.generate_homework(
    text=HOMEWORK_TEXT,
    blank_sheet_paths=[blank_sheet],
    output_dir="output_homework",
    max_chars_per_line=50
)

print(f"Generated {len(generated_images)} homework pages!")
    """)

    # Cell 8: Preview Final High-Contrast Pages & PDF
    add_md("## 8. Step 6: Preview Output Pages & Download Submission PDF")
    add_code("""
for idx, img_path in enumerate(generated_images):
    img = Image.open(img_path)
    plt.figure(figsize=(10, 14))
    plt.imshow(img)
    plt.title(f"Generated Homework - Page {idx + 1}")
    plt.axis("off")
    plt.show()

print(f"\\nDownload your submission PDF from: {pdf_path}")
    """)

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("homework_bud_kaggle.ipynb", "w") as f:
        json.dump(notebook_data, f, indent=2)
    print("Successfully built homework_bud_kaggle.ipynb!")

if __name__ == "__main__":
    create_notebook()
