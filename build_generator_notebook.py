"""
Script to create the dedicated Kaggle Notebook for Essay & Homework Generation.
Designed for 1,000+ word essays, multi-page assignments, and cycling blank pages.
"""

import json

def create_generator_notebook():
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
# 📝 Homework-Bud: 1,000+ Word Essay & Homework Generator
### Synthesizes photorealistic multi-page handwritten assignments from arbitrary text in your personal handwriting (fine-tuned style from `7.jpeg`), rendered onto cycled blank ruled pages in Royal Blue ink.
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
print("Environment ready for handwriting generation!")
    """)

    # Cell 2: Locate Checkpoints & Blank Paper Scans
    add_md("## 2. Load Fine-Tuned Weights & Blank Ruled Pages")
    add_code("""
# Check possible locations for checkpoints and blank pages
candidate_dirs = [
    "/kaggle/input/homework-bud-handwriting-synthesis",
    "/kaggle/input/homework-bud-data",
    "."
]

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("styles", exist_ok=True)

# 1. Search for fine-tuned checkpoints anywhere in /kaggle/input or local
checkpoints_found = False
for search_root in ["/kaggle/input", "."]:
    if not os.path.exists(search_root):
        continue
    for root, dirs, files in os.walk(search_root):
        for f in files:
            if "finetuned_checkpoints" in f and f.endswith(".zip"):
                zip_path = os.path.join(root, f)
                print(f"[OK] Extracting fine-tuned checkpoints from {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("checkpoints")
                checkpoints_found = True
                break
        if checkpoints_found:
            break
    if checkpoints_found:
        break

if not checkpoints_found:
    print("[WARNING] Could not find finetuned_checkpoints.zip, searching for existing model checkpoints...")
    for search_root in ["/kaggle/input", "."]:
        if not os.path.exists(search_root):
            continue
        for root, dirs, files in os.walk(search_root):
            for f in files:
                if f.startswith("model-") and not f.endswith(".zip"):
                    shutil.copy(os.path.join(root, f), os.path.join("checkpoints", f))
                    checkpoints_found = True

# 2. Extract styles
styles_found = False
for search_root in ["/kaggle/input", "."]:
    if not os.path.exists(search_root):
        continue
    for root, dirs, files in os.walk(search_root):
        for f in files:
            if "user_style" in f and f.endswith(".zip"):
                zip_path = os.path.join(root, f)
                print(f"[OK] Extracting user style conditioning from {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("styles")
                styles_found = True
                break
            elif "style-user-current" in f and f.endswith(".npy"):
                shutil.copy(os.path.join(root, f), os.path.join("styles", f))
                styles_found = True
        if styles_found:
            break
    if styles_found:
        break

# 3. Discover all blank ruled sheet candidates to cycle through
blank_sheets = []
for search_root in ["/kaggle/input", "."]:
    if not os.path.exists(search_root):
        continue
    for root, dirs, files in os.walk(search_root):
        for f in files:
            if "blank" in f.lower() and f.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, f)
                dest = os.path.basename(f)
                if not os.path.exists(dest):
                    shutil.copy(full_path, dest)
                if dest not in blank_sheets:
                    blank_sheets.append(dest)

if not blank_sheets:
    for fallback in ["Blank_Page.jpeg", "repo_code/Blank_Page.jpeg"]:
        if os.path.exists(fallback):
            if fallback != "Blank_Page.jpeg":
                shutil.copy(fallback, "Blank_Page.jpeg")
            blank_sheets = ["Blank_Page.jpeg"]
            break

print(f"\\n[ASSETS READY]")
print(f"  - Checkpoints: {len(glob.glob('checkpoints/*'))} files in checkpoints/")
print(f"  - Blank Ruled Sheets to Cycle: {len(blank_sheets)} pages -> {blank_sheets}")
print(f"  - User Style: {'style-user-current' if os.path.exists('styles/style-user-current-strokes.npy') else 'Default'}")
    """)

    # Cell 3: User Essay Input
    add_md("""
## 3. Enter Your Essay or Homework Text
Paste your 1,000+ word essay, questions, answers, or lab report below.
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
# 🎨 STYLE & INK CONFIGURATION:
# ==============================================================================
ink_color = (26, 62, 175)    # Royal Blue (default). Try (20, 45, 130) for Navy, (35, 35, 35) for Black
bias = 0.88                  # 0.88 = neat, legible handwriting; 0.75 = more casual variation
max_chars_per_line = 58      # Characters per ruled line (~55-60 fits ruled margins perfectly)
    """)

    # Cell 4: Generate Multi-Page Homework
    add_md("## 4. Synthesize Handwritten Ruled Pages & Compile PDF")
    add_code("""
from homework_engine import HomeworkEngine

engine = HomeworkEngine(
    checkpoint_dir="checkpoints",
    styles_dir="styles",
    ink_rgb=ink_color,
    bias=bias
)

output_pages, pdf_path = engine.generate_homework(
    text=essay_text,
    blank_sheet_paths=blank_sheets,
    output_dir="output",
    custom_style_prefix="styles/style-user-current" if os.path.exists("styles/style-user-current-strokes.npy") else None,
    max_chars_per_line=max_chars_per_line
)

print(f"\\n{'=' * 65}")
print(f"GENERATION COMPLETE: {len(output_pages)} PAGES CREATED")
print(f"{'=' * 65}")
for i, p in enumerate(output_pages, 1):
    size_kb = os.path.getsize(p) / 1024
    print(f"  Page {i:02d}: {p} ({size_kb:.1f} KB)")
print(f"\\nCompiled PDF: {pdf_path} ({os.path.getsize(pdf_path) / (1024 * 1024):.2f} MB)")
    """)

    # Cell 5: Visual Gallery of All Generated Pages
    add_md("## 5. Visual Inspection of All Generated Pages")
    add_code("""
import matplotlib.pyplot as plt
from PIL import Image

for i, page_path in enumerate(output_pages, 1):
    img = Image.open(page_path)
    plt.figure(figsize=(14, 19))
    plt.imshow(img)
    plt.title(f"Page {i} of {len(output_pages)} — Ruled Sheet Handwriting in Royal Blue", fontsize=15, pad=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    """)

    # Cell 6: Package All Artifacts for Download
    add_md("## 6. Download Homework Submission Archive")
    add_code("""
# Package output folder containing all PNGs + compiled PDF
shutil.make_archive("homework_submission", "zip", "output")

zip_size_mb = os.path.getsize("homework_submission.zip") / (1024 * 1024)
print(f"[OK] Download ready: homework_submission.zip ({zip_size_mb:.2f} MB)")
print("You can download homework_submission.zip from the right panel under 'Output'!")
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

    with open("homework_generator_kaggle.ipynb", "w") as f:
        json.dump(notebook_data, f, indent=2)
    print("[OK] Successfully built homework_generator_kaggle.ipynb!")

if __name__ == "__main__":
    create_generator_notebook()
