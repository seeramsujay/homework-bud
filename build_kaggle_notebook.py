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
# 📝 Homework-Bud: Personalized Handwriting on Ruled Sheets (Kaggle Cloud)
### Generates realistic handwritten assignments from your own handwriting, snapped onto your real ruled notebook photo, rendered with Royal Blue ink and high-contrast phone-scan quality.

---
### 🌟 Key Pipeline Features:
1. **Style Priming**: Extracts stroke trajectories directly from your uploaded training images to condition the RNN on your exact handwriting slant, curve roundness, and spacing.
2. **Ruled Line & Margin Detection**: Automatically scans your `Blank_Page.jpeg` to detect horizontal notebook lines and margins.
3. **Natural Baseline Alignment**: Aligns and scales words so letters sit naturally on each notebook line.
4. **Solid Royal Blue Ink & High Contrast**: Uses solid pigment multiply blending and elevated gamma/contrast so the output looks like a high-contrast document scan.
5. **Multi-Page Submission PDF**: Automatically compiles multi-page homework assignments into a single PDF.
    """)

    # Cell 1: Setup & Dependencies
    add_md("## 1. Setup & Environment")
    add_code("""
!pip install --quiet svgwrite opencv-python-headless pillow reportlab scipy matplotlib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Enable TF 1.x compatibility mode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("Kaggle environment initialized successfully!")
    """)

    # Cell 2: Import Modules
    add_md("## 2. Load Core Engine & Vision Modules")
    add_code("""
from ruled_sheet_detector import RuledSheetDetector
from scan_renderer import ScanRenderer
from paper_stroke_extractor import process_paper_sample, process_json_style
from homework_engine import HomeworkEngine
from extract_style import extract_user_style_from_page

print("All Homework-Bud modules loaded!")
    """)

    # Cell 3: Analyze Blank Ruled Page
    add_md("""
## 3. Step 1: Detect Notebook Lines & Margins
Detects the lines on your uploaded blank notebook photo (`Blank_Page.jpeg`).
    """)
    add_code("""
blank_sheet_path = "Blank_Page.jpeg"

if not os.path.exists(blank_sheet_path):
    print(f"Warning: {blank_sheet_path} not found in current folder. Using default layout.")
    blank_sheets = ["test_blank_ruled.jpg"]
else:
    blank_sheets = [blank_sheet_path]

detector = RuledSheetDetector()
layout = detector.analyze_image(blank_sheets[0])
print(f"Detected {len(layout.baselines)} ruled lines.")
print(f"Line spacing: {layout.line_spacing:.1f}px | Left margin: {layout.left_margin:.1f}px | Tilt: {layout.tilt_angle_deg:.2f}°")

# Preview detection overlay
preview = cv2.imread(blank_sheets[0]).copy()
for y in layout.baselines:
    cv2.line(preview, (int(layout.left_margin), int(y)), (int(layout.right_margin), int(y)), (0, 255, 0), 1)
cv2.line(preview, (int(layout.left_margin), 0), (int(layout.left_margin), layout.height), (255, 0, 0), 2)

plt.figure(figsize=(8, 11))
plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
plt.title("Detected Ruled Notebook Lines (Green) and Margin (Blue)")
plt.axis("off")
plt.show()
    """)

    # Cell 4: Handwriting Style Extraction / Priming
    add_md("""
## 4. Step 2: Extract Your Handwriting Style
Pulls the stroke sequence from your uploaded `Training_Images/` to condition the RNN on your exact penmanship.
    """)
    add_code("""
import glob
train_images = sorted(glob.glob("Training_Images/*.jpeg") + glob.glob("Training_Images/*.jpg"))

if train_images:
    print(f"Extracting handwriting style from {train_images[0]}...")
    extract_user_style_from_page(train_images[0], output_prefix="styles/style-user")
    custom_style_prefix = "styles/style-user"
    print("User handwriting style ready!")
elif os.path.exists("styles/style-user-strokes.npy"):
    custom_style_prefix = "styles/style-user"
    print("Using cached user style from styles/style-user")
else:
    custom_style_prefix = None
    selected_style_id = 9
    print(f"No training images found; falling back to clean built-in style {selected_style_id}")
    """)

    # Cell 5: Homework Text & Ink Configuration
    add_md("""
## 5. Step 3: Homework Text & Configuration
Input your assignment text and configure the Royal Blue color.
    """)
    add_code("""
# Exact Royal Blue ink extracted from your handwritten notes (#3057a3)
ROYAL_BLUE = (48, 87, 163)

# Neatness bias: 0.85 gives clean, legible handwriting
NEATNESS_BIAS = 0.85

HOMEWORK_TEXT = \"\"\"
Homework Assignment: Sequence Modeling

1. What is an autoregressive recurrent model?
An autoregressive recurrent model predicts future values in a sequence by feeding its own previous outputs back as inputs at subsequent timesteps. This allows generating long, coherent sequential data such as handwriting and speech.

2. How does mixture density modeling capture pen strokes?
Instead of outputting single deterministic delta coordinates, the network predicts the parameters of a bivariate Gaussian Mixture Model along with a Bernoulli distribution for pen lifts. This models multi-modal possibilities for natural human variations.

3. Resulting Output:
The generated pen coordinates are snapped onto the ruled lines of the notebook page and rendered with royal blue ink.
\"\"\"

print("Homework text loaded!")
    """)

    # Cell 6: Run Synthesis & Overlay
    add_md("## 6. Step 4: Synthesize Handwriting & Render High-Contrast Scan")
    add_code("""
engine = HomeworkEngine(
    ink_rgb=ROYAL_BLUE,
    bias=NEATNESS_BIAS
)

generated_images, pdf_path = engine.generate_homework(
    text=HOMEWORK_TEXT,
    blank_sheet_paths=blank_sheets,
    output_dir="output_homework",
    style_id=None if custom_style_prefix else 9,
    custom_style_prefix=custom_style_prefix,
    max_chars_per_line=50
)

print(f"Successfully generated {len(generated_images)} page(s)!")
    """)

    # Cell 7: Preview Pages & PDF Download
    add_md("## 7. Step 5: Preview Scanned Homework Pages")
    add_code("""
for idx, img_path in enumerate(generated_images):
    img = Image.open(img_path)
    plt.figure(figsize=(10, 14))
    plt.imshow(img)
    plt.title(f"Homework Page {idx + 1}")
    plt.axis("off")
    plt.show()

print(f"Multi-page PDF saved at: {pdf_path}")
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
