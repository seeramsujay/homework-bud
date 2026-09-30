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
# 📝 Homework-Bud: Automated Neural OCR, Fine-Tuning & Ruled Paper Synthesis
### Fine-tunes Alex Graves' Handwriting RNN on your actual handwritten notebook notes using Deep Learning OCR on Kaggle GPU, aligns onto your blank ruled sheet photo, and renders crisp Royal Blue ink scans.

---
### 🚀 End-to-End Pipeline:
1. **Deep Learning OCR (EasyOCR / TrOCR)**: Automatically detects lines and transcribes cursive handwriting from your `Training_Images/`.
2. **Fine-Tuning on Kaggle GPU**: Runs gradient descent from `model-17900` on your handwriting pairs $(x, c)$ to adapt the neural weights to your exact letter shapes and flow.
3. **Ruled Sheet Detection**: Detects notebook lines and margins on `Blank_Page.jpeg`.
4. **Snapping & Synthesis**: Synthesizes homework text snapped to ruled lines with user-tunable neatness bias.
5. **High-Contrast Scan Overlay**: Solid Royal Blue ink (`#3057a3`) Multiply-blended with high scanner gamma/contrast, compiled to multi-page PDF.
    """)

    # Cell 1: Setup & Dependencies
    add_md("## 1. Install GPU Dependencies")
    add_code("""
!pip install --quiet easyocr svgwrite opencv-python-headless pillow reportlab scipy matplotlib
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("GPU environment initialized!")
    """)

    # Cell 2: Import Modules
    add_md("## 2. Load Pipeline Modules")
    add_code("""
from ruled_sheet_detector import RuledSheetDetector
from scan_renderer import ScanRenderer
from homework_engine import HomeworkEngine
from build_dataset_ocr import HandwritingDatasetBuilder
from finetune_user import finetune_user_handwriting

print("Pipeline modules loaded successfully!")
    """)

    # Cell 3: Neural OCR & Dataset Creation
    add_md("""
## 3. Step 1: Deep Learning OCR on Your Handwritten Pages
Automatically reads and crops lines from `Training_Images/` and pairs pen strokes with text.
    """)
    add_code("""
train_images = sorted(glob.glob("Training_Images/*.jpeg") + glob.glob("Training_Images/*.jpg"))
print(f"Found {len(train_images)} training pages.")

# Initialize Neural OCR on GPU
builder = HandwritingDatasetBuilder(use_gpu=True)

# Build paired dataset (x.npy, c.npy)
builder.build_dataset_from_images(train_images, output_dir="data/processed")
    """)

    # Cell 4: Fine-Tuning the RNN
    add_md("""
## 4. Step 2: Fine-Tune the Handwriting RNN on Kaggle GPU
Adapts the model weights directly to your personal handwriting style.
    """)
    add_code("""
# Fine-tune starting from checkpoint 17900
finetune_user_handwriting(
    data_dir="data/processed/",
    checkpoint_dir="checkpoints",
    warm_start_step=17900,
    finetune_steps=2000,
    learning_rate=0.00005,
    batch_size=16
)
    """)

    # Cell 5: Analyze Blank Ruled Page
    add_md("## 5. Step 3: Analyze Blank Ruled Notebook Sheet")
    add_code("""
blank_sheet = "Blank_Page.jpeg"
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
plt.title("Detected Ruled Lines (Green) and Left Margin (Blue)")
plt.axis("off")
plt.show()
    """)

    # Cell 6: Homework Text & Synthesis
    add_md("## 6. Step 4: Generate Your Homework in Royal Blue Ink")
    add_code("""
# Royal Blue ink (#3057a3)
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
The synthesized strokes are aligned directly with the detected ruled baselines and rendered in solid royal blue ink.
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

    # Cell 7: Preview High-Contrast Scanned Pages
    add_md("## 7. Step 5: Preview Scanned Assignment Pages & Download PDF")
    add_code("""
for idx, img_path in enumerate(generated_images):
    img = Image.open(img_path)
    plt.figure(figsize=(10, 14))
    plt.imshow(img)
    plt.title(f"Homework Page {idx + 1}")
    plt.axis("off")
    plt.show()

print(f"Submission PDF ready at: {pdf_path}")
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
