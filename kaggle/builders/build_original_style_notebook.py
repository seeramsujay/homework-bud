"""
Script to create the Kaggle Notebook for generating homework using the original Graves RNN
with Style 6, maximum legibility (bias=1.0), and the user's Royal Blue pen ink on Blank_Page.jpeg.
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "kernels", "style6_generator", "homework_original_style_generator.ipynb"))

def create_original_style_notebook(output_path=DEFAULT_OUTPUT):
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
# 📝 Homework-Bud: Original RNN Style 6 Generator
### Synthesizes handwriting using the original Graves RNN pre-trained model with **Style 6**, **Maximum Legibility (Bias = 1.0)**, and **User's Royal Blue Pen Ink**, rendered directly onto ruled sheets ([`Blank_Page.jpeg`](Blank_Page.jpeg)).
    """)

    # Cell 1: Environment & Setup
    add_md("## 1. Environment & Setup")
    add_code("""
import os
import sys
import glob
import shutil
import zipfile
import subprocess

# Install lightweight dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "opencv-python-headless", "pillow", "pandas", "matplotlib", "reportlab", "svgwrite"], check=True)

# Clone or pull latest project code
if not os.path.exists("repo_code"):
    os.system("git clone --depth 1 https://github.com/seeramsujay/homework-bud.git repo_code")
    os.system("cp -rf repo_code/* . 2>/dev/null || true")
else:
    os.system("cd repo_code && git pull && cp -rf * .. 2>/dev/null || true")

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print("Environment ready for Style 6 generation!")
    """)

    # Cell 2: Load Original Checkpoints & Ruled Paper
    add_md("## 2. Load Original Base Model (model-17900) & Blank Paper")
    add_code("""
# Locate original checkpoints (step 17900) and styles
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("styles", exist_ok=True)

# 1. Base checkpoints from dataset or repo
base_copied = False
for candidate in ["/kaggle/input/homework-bud-data/checkpoints", "checkpoints", "repo_code/checkpoints"]:
    if os.path.exists(candidate) and glob.glob(os.path.join(candidate, "model-17900*")):
        print(f"[OK] Loading original base checkpoint from {candidate}...")
        os.system(f"cp -rf {candidate}/* checkpoints/ 2>/dev/null || true")
        base_copied = True
        break

if not base_copied:
    print("[INFO] Checking recursive input for base checkpoints...")
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if "model-17900" in f:
                shutil.copy(os.path.join(root, f), os.path.join("checkpoints", f))
                base_copied = True

# 2. Copy styles directory
for s_cand in ["/kaggle/input/homework-bud-data/styles", "styles", "repo_code/styles"]:
    if os.path.exists(s_cand) and os.path.exists(os.path.join(s_cand, "style-6-strokes.npy")):
        print(f"[OK] Loading original styles from {s_cand}...")
        os.system(f"cp -rf {s_cand}/* styles/ 2>/dev/null || true")
        break

# 3. Locate Blank_Page.jpeg
blank_sheets = []
for root, dirs, files in os.walk("."):
    for f in files:
        if "blank_page" in f.lower() and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(root, f)
            dest = os.path.basename(f)
            if not os.path.exists(dest):
                shutil.copy(full_path, dest)
            if dest not in blank_sheets:
                blank_sheets.append(dest)

if not blank_sheets:
    for fallback in ["Blank_Page.jpeg", "repo_code/Blank_Page.jpeg", "/kaggle/input/homework-bud-data/Blank_Page.jpeg"]:
        if os.path.exists(fallback):
            shutil.copy(fallback, "Blank_Page.jpeg")
            blank_sheets = ["Blank_Page.jpeg"]
            break

print(f"\\n[SETUP READY]")
print(f"  - Original Checkpoint: model-17900 ready in checkpoints/")
print(f"  - Style 6: {'Ready' if os.path.exists('styles/style-6-strokes.npy') else 'Missing'}")
print(f"  - Blank Ruled Paper: {blank_sheets}")
    """)

    # Cell 3: User Essay Input & Parameters
    add_md("""
## 3. Essay Text & Style 6 Configuration
- **Model**: Original Graves RNN (`model-17900`)
- **Style**: **Style 6** (Byron style)
- **Legibility**: **Max** (`bias = 1.0`)
- **Ink Color**: Royal Blue (`rgb(26, 62, 175)`)
    """)
    add_code("""
# ==============================================================================
# 📝 PASTE YOUR ESSAY OR HOMEWORK TEXT HERE:
# ==============================================================================
essay_text = \"\"\"
Reading “Yajnaseni,” what touched me most wasn’t the invincible queen of an ancient epic, but the painful vulnerability of a woman standing completely alone while her world collapsed. We often remember Draupadi as born of fire, yet I kept thinking about how fragile that moment in the dice hall must have felt. Looking around at her husbands and revered elders—people who loved her, yet stared silently at the floor—must have broken something deep inside her. To be wagered away like property is an ache that feels almost too intimate and painful to read.

Yet, in that hollow silence, her story shifted. Her courage—what the chapter calls *vipatti-dhairya*—did not begin with fierce pride, but with trembling tears and questions that shook the courtroom. Even when every pillar of security deserted her, she refused to erase herself. In daily life, too, managing the kingdom’s sprawling affairs while carrying the complex emotions of five husbands, she bore an exhausting, quiet grace.

Amma’s parable of the eagle raised among chicks lingered with me. It made me realize how easily we accept our own helplessness, mistaking our gentleness and pain for weakness. Draupadi closing her eyes in final surrender to Krishna wasn’t helplessness; it was the poignant understanding that when all human shelter fails, faith is the only home left.

Her reflection leaves a quiet ache. It showed me that fragility does not mean brokenness. Even when life burns us down to cold ashes, the quietest spark within us still remembers how to breathe.
\"\"\"

# ==============================================================================
# 🎨 STYLE 6 CONFIGURATION (MAX LEGIBILITY & USER PEN COLOR):
# ==============================================================================
style_id = 6                 # Original Style 6
legibility_bias = 1.0        # Max legibility (neat, sharp, uniform strokes)
ink_color = (26, 62, 175)    # User's Royal Blue pen color
max_chars_per_line = 58      # Ruled line fit
    """)

    # Cell 4: Synthesize & Render on Ruled Sheet
    add_md("## 4. Synthesize with Original RNN & Render on Ruled Paper")
    add_code("""
from homework_engine import HomeworkEngine

engine = HomeworkEngine(
    checkpoint_dir="checkpoints",
    styles_dir="styles",
    ink_rgb=ink_color,
    bias=legibility_bias
)

output_pages, pdf_path = engine.generate_homework(
    text=essay_text,
    blank_sheet_paths=blank_sheets,
    output_dir="output",
    style_id=style_id,
    max_chars_per_line=max_chars_per_line
)

print(f"\\n{'=' * 65}")
print(f"GENERATION COMPLETE (ORIGINAL RNN + STYLE 6 + MAX LEGIBILITY)")
print(f"{'=' * 65}")
for i, p in enumerate(output_pages, 1):
    size_kb = os.path.getsize(p) / 1024
    print(f"  Page {i:02d}: {p} ({size_kb:.1f} KB)")
print(f"\\nCompiled PDF: {pdf_path} ({os.path.getsize(pdf_path) / (1024 * 1024):.2f} MB)")
    """)

    # Cell 5: Visual Gallery
    add_md("## 5. Visual Inspection of Style 6 Ruled Homework")
    add_code("""
import matplotlib.pyplot as plt
from PIL import Image

for i, page_path in enumerate(output_pages, 1):
    img = Image.open(page_path)
    plt.figure(figsize=(14, 19))
    plt.imshow(img)
    plt.title(f"Page {i} of {len(output_pages)} — Original RNN (Style 6, Max Legibility, Royal Blue Ink)", fontsize=15, pad=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    """)

    # Cell 6: Package & Download
    add_md("## 6. Package Submission Archive")
    add_code("""
shutil.make_archive("style6_homework_submission", "zip", "output")
zip_size_mb = os.path.getsize("style6_homework_submission.zip") / (1024 * 1024)
print(f"[OK] Download ready: style6_homework_submission.zip ({zip_size_mb:.2f} MB)")
print("Download style6_homework_submission.zip or output/homework_assignment.pdf from the right panel!")
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(notebook_data, f, indent=2)
    print(f"[OK] Successfully built {output_path}!")

if __name__ == "__main__":
    create_original_style_notebook()
