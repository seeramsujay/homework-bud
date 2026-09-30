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
# 📝 Homework-Bud: Personalized Handwriting on Ruled Sheets
### Generates realistic handwritten assignments from your own handwriting, snapped onto real ruled notebook photos, rendered with Royal Blue ink and phone-camera scan effects.

---
### 🌟 What this Notebook Does:
1. **Learns / Clones Your Handwriting**: Primed with your personal handwriting sample (either a photo of your paper writing + transcription, or a touch-recorded sample).
2. **Auto-Detects Ruled Lines & Margins**: Uses computer vision to find the horizontal lines and margins on your blank notebook photo.
3. **Baseline Alignment**: Snaps and scales each line of synthesized text so letters sit naturally on the notebook's ruled lines.
4. **Photorealistic Phone Scan**: Uses physical Multiply ink blending, pressure dynamics, and royal blue shading so it looks like a photo taken on your phone.
5. **Multi-Page PDF**: Automatically compiles multi-page homework assignments into a submission-ready PDF.
    """)

    # Cell 1: Install Dependencies
    add_md("## 1. Setup & Dependencies")
    add_code("""
!pip install --quiet svgwrite opencv-python-headless pillow reportlab scipy matplotlib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("Environment ready!")
    """)

    # Cell 2: TF Compatibility & Code Setup
    add_md("## 2. Load Core Handwriting Synthesis Engine")
    add_code("""
# Ensure TensorFlow 1.x compatibility under TF 2.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Verify project files
assert os.path.exists("checkpoints/model-17900.meta"), "Checkpoints directory missing!"
assert os.path.exists("styles/style-0-strokes.npy"), "Styles directory missing!"

from ruled_sheet_detector import RuledSheetDetector
from scan_renderer import ScanRenderer
from paper_stroke_extractor import process_paper_sample, process_json_style
from homework_engine import HomeworkEngine

print("Homework-Bud engine successfully initialized!")
    """)

    # Cell 3: Ruled Paper Photo Setup
    add_md("""
## 3. Step 1: Your Blank Ruled Paper Photo
Take a photo of a blank page of your notebook using your phone camera and upload it here (e.g. `blank_page_1.jpg`).
You can upload multiple photos to have different pages for multi-page assignments!
    """)
    add_code("""
# List of blank ruled paper photos (upload your own to the Kaggle notebook files)
# If you don't have one uploaded yet, set USE_SYNTHETIC = True to test with a simulated notebook page!
USE_SYNTHETIC = True

blank_sheets = []

if USE_SYNTHETIC:
    # Generate a realistic blank notebook sheet photo for demonstration
    w, h = 1200, 1600
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    gradient = 250 - 25 * (x_coords / w + y_coords / h) / 2.0
    paper = np.stack([gradient, gradient + 2, gradient + 4], axis=-1).astype(np.uint8)
    noise = np.random.normal(0, 3, (h, w, 3))
    paper = np.clip(paper + noise, 0, 255).astype(np.uint8)
    # Draw faint ruled lines
    for y in range(150, h - 100, 52):
        cv2.line(paper, (60, y), (w - 60, y), (210, 190, 170), 1)
    # Draw pink/red left margin
    cv2.line(paper, (160, 80), (160, h - 60), (160, 150, 230), 2)
    demo_blank = "blank_ruled_sample.jpg"
    cv2.imwrite(demo_blank, paper)
    blank_sheets.append(demo_blank)
else:
    # Put your uploaded image filenames here:
    blank_sheets = ["my_blank_page_1.jpg"]

# Inspect detected ruled lines and margins
detector = RuledSheetDetector()
layout = detector.analyze_image(blank_sheets[0])
print(f"Detected {len(layout.baselines)} ruled lines on sheet.")
print(f"Line spacing: {layout.line_spacing:.1f}px | Left margin: {layout.left_margin:.1f}px")

# Preview detection overlay
preview = cv2.imread(blank_sheets[0]).copy()
for y in layout.baselines:
    cv2.line(preview, (int(layout.left_margin), int(y)), (int(layout.right_margin), int(y)), (0, 255, 0), 1)
cv2.line(preview, (int(layout.left_margin), 0), (int(layout.left_margin), layout.height), (255, 0, 0), 2)

plt.figure(figsize=(8, 10))
plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
plt.title("Detected Ruled Lines (Green) and Left Margin (Blue)")
plt.axis("off")
plt.show()
    """)

    # Cell 4: Handwriting Style Input
    add_md("""
## 4. Step 2: Personalize with Your Handwriting
Choose how you want to provide your style:
- **Option 1**: Upload a photo of your paper handwriting + transcription (`process_paper_sample`)
- **Option 2**: Upload `user_style.json` recorded via `touch_recorder.html` (`process_json_style`)
- **Option 3**: Select one of the 13 built-in styles (`0` to `12`)
    """)
    add_code("""
# Select style source: 'builtin', 'paper_photo', or 'touch_json'
STYLE_SOURCE = 'builtin'

custom_prefix = None
selected_style_id = 9  # Style 9 is a clean, natural handwriting

if STYLE_SOURCE == 'paper_photo':
    # Example: you uploaded a photo of one sentence written on paper
    photo_file = "my_handwriting_snippet.jpg"
    text_transcription = "The quick brown fox jumps over the lazy dog"
    process_paper_sample(photo_file, text_transcription, output_prefix="styles/style-custom")
    custom_prefix = "styles/style-custom"

elif STYLE_SOURCE == 'touch_json':
    # Example: uploaded user_style.json exported from touch_recorder.html
    process_json_style("user_style.json", output_prefix="styles/style-custom")
    custom_prefix = "styles/style-custom"

elif STYLE_SOURCE == 'builtin':
    print(f"Using built-in handwriting style {selected_style_id}")
    """)

    # Cell 5: Homework Assignment Text & Ink Color
    add_md("""
## 5. Step 3: Input Homework Text & Ink Configuration
Set your Royal Blue ink color and paste your homework questions / answers below.
    """)
    add_code("""
# Ink Color (Royal Blue RGB)
ROYAL_BLUE = (26, 62, 175)

# Neatness control: 0.70 (natural) to 0.95 (super neat and uniform)
NEATNESS_BIAS = 0.85

# Enter your homework text below:
HOMEWORK_TEXT = \"\"\"
Homework Assignment: History of Computing

Question 1: What is the Church-Turing Thesis?
The Church-Turing thesis states that a function on the natural numbers can be calculated by an effective method if and only if it is computable by a Turing machine. It connects the intuitive concept of algorithm with formal computation.

Question 2: Explain the significance of Recurrent Neural Networks.
Recurrent Neural Networks (RNNs) process sequential data by maintaining an internal hidden state. This state acts as a memory capable of retaining information across arbitrary sequence lengths, making RNNs suited for handwriting and language modeling.

Conclusion:
Handwriting synthesis combines sequence learning with continuous mixture density outputs to create smooth, natural pen strokes.
\"\"\"

print("Homework text loaded!")
    """)

    # Cell 6: Run Synthesis & Render Pages
    add_md("## 6. Step 4: Run Handwriting Synthesis & Realistic Overlay")
    add_code("""
engine = HomeworkEngine(
    ink_rgb=ROYAL_BLUE,
    bias=NEATNESS_BIAS
)

generated_images, pdf_path = engine.generate_homework(
    text=HOMEWORK_TEXT,
    blank_sheet_paths=blank_sheets,
    output_dir="output_homework",
    style_id=selected_style_id if custom_prefix is None else None,
    custom_style_prefix=custom_prefix,
    max_chars_per_line=55
)

print(f"Generated {len(generated_images)} page(s) successfully!")
    """)

    # Cell 7: Preview Generated Homework
    add_md("## 7. Step 5: Preview Your Generated Homework Pages")
    add_code("""
for idx, img_path in enumerate(generated_images):
    img = Image.open(img_path)
    plt.figure(figsize=(10, 14))
    plt.imshow(img)
    plt.title(f"Generated Homework - Page {idx + 1}")
    plt.axis("off")
    plt.show()

print(f"Final multi-page PDF ready at: {pdf_path}")
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
