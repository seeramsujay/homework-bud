"""
Script to generate the Kaggle notebook for Stage 1: Line Segmentation & GPU Neural OCR.
Extracts individual text lines from notebook pages, runs EasyOCR on Kaggle GPU,
and exports lines_transcription.csv + segmented_lines.zip for manual verification.
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
# 📝 Homework-Bud: Stage 1 — Line Segmentation & Neural OCR
### Automatically segments notebook pages into numbered line PNGs, runs GPU-accelerated EasyOCR on each line, and exports `lines_transcription.csv` & `segmented_lines.zip` for user verification.

---
### 🌟 Stage 1 Goals:
1. **Detect & Slice Lines**: Automatically detects ink line baselines and crops each handwritten line into `segmented_lines/line_XXXX.png`.
2. **GPU Neural OCR**: Runs EasyOCR with GPU acceleration to transcribe each line crop.
3. **Export for Review**: Saves `lines_transcription.csv` (with `ocr_transcription` and `user_verified_transcription`) and `segmented_lines.zip` in `/kaggle/working/`.
4. **Visual Gallery**: Renders line crops with their recognized text directly in the notebook for immediate inspection.
    """)

    # Cell 1: Setup & Dependencies
    add_md("## 1. Setup & Dependencies")
    add_code("""
import os
import sys
import glob
import shutil
import subprocess

# Install high-performance neural OCR and imaging libraries
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "easyocr", "opencv-python-headless", "pillow", "pandas", "matplotlib"], check=True)

# Clone project repository if helper files are not present
if not os.path.exists("line_segmenter.py"):
    os.system("git clone --depth 1 https://github.com/seeramsujay/homework-bud.git repo_code")
    os.system("cp -rn repo_code/* . 2>/dev/null || true")

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
print("Environment initialized successfully!")
    """)

    # Cell 2: Locate Training Pages
    add_md("## 2. Locate Training Pages")
    add_code("""
# Locate training images from attached dataset or workspace
candidate_dirs = [
    "/kaggle/input/homework-bud-data/Training_Images",
    "/kaggle/input/homework-bud-data",
    "/kaggle/input",
    "Training_Images",
    "kaggle_dataset_bundle/Training_Images"
]

train_dir = None
for d in candidate_dirs:
    if os.path.exists(d):
        found = glob.glob(os.path.join(d, "**", "*.jp*g"), recursive=True) + glob.glob(os.path.join(d, "**", "*.png"), recursive=True)
        found = [p for p in found if "blank" not in os.path.basename(p).lower()]
        if found:
            train_dir = d
            print(f"[OK] Found {len(found)} training pages in: {train_dir}")
            for p in sorted(found):
                print(f"  - {os.path.basename(p)}")
            break

if train_dir is None:
    raise FileNotFoundError("Could not find any training pages in candidate directories!")
    """)

    # Cell 3: Run Line Segmentation & Neural OCR
    add_md("## 3. Run Line Segmentation & GPU Neural OCR")
    add_code("""
from line_segmenter import process_all_training_pages_to_lines

output_lines_dir = "segmented_lines"
csv_path = "lines_transcription.csv"

# Process all pages: crop lines + run neural OCR + create zip archive
process_all_training_pages_to_lines(
    training_images_dir=train_dir,
    output_lines_dir=output_lines_dir,
    csv_path=csv_path
)
    """)

    # Cell 4: Review Extracted Transcriptions
    add_md("## 4. Review OCR Transcriptions Table")
    add_code("""
import pandas as pd

df = pd.read_csv("lines_transcription.csv")
print(f"Extracted {len(df)} lines from {df['source_page'].nunique()} training pages!")
print(f"Columns: {list(df.columns)}")

# Display preview table
display(df.head(25))
    """)

    # Cell 5: Visual Inspection Gallery
    add_md("## 5. Visual Inspection of Cropped Lines")
    add_code("""
import matplotlib.pyplot as plt
from PIL import Image

sample_count = min(10, len(df))
sample_df = df.head(sample_count)

plt.figure(figsize=(15, 2.2 * sample_count))
for i, (_, row) in enumerate(sample_df.iterrows()):
    img_path = os.path.join("segmented_lines", row['filename'])
    if os.path.exists(img_path):
        img = Image.open(img_path)
        plt.subplot(sample_count, 1, i + 1)
        plt.imshow(img)
        title_text = f"[{row['filename']}] OCR: " + str(row['ocr_transcription'])
        plt.title(title_text, fontsize=10, loc='left')
        plt.axis("off")
plt.tight_layout()
plt.show()
    """)

    # Cell 6: Export Artifacts Summary
    add_md("## 6. Generated Output Artifacts")
    add_code("""
print("=" * 60)
print("STAGE 1 ARTIFACTS READY FOR DOWNLOAD:")
print("=" * 60)

for f in ["lines_transcription.csv", "segmented_lines.zip"]:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  ✓ {f:<26} ({size_mb:.2f} MB)")

print("\\nInstructions:")
print("1. Download 'lines_transcription.csv' and 'segmented_lines.zip' from Kaggle Output.")
print("2. Open 'lines_transcription.csv' to review and verify the 'user_verified_transcription' column.")
print("3. When verified, upload the CSV back for Stage 2 (RNN Fine-Tuning & Homework Synthesis)!")
print("=" * 60)
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
