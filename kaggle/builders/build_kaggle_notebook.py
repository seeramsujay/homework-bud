"""
Script to generate the Kaggle notebook for Stage 2:
Fine-Tuning Alex Graves Handwriting Synthesis RNN on User's Current Handwriting
and Synthesizing Ruled Homework Sheets.
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "kernels", "finetune", "homework_bud_kaggle.ipynb"))

def create_notebook(output_path=DEFAULT_OUTPUT):
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
# 📝 Homework-Bud: Fine-Tuning & Ruled Homework Synthesis
### Trains Alex Graves' Handwriting Synthesis RNN on the user's personal handwriting (prioritizing current style from `7.jpeg`) and synthesizes multi-page ruled homework assignments in Royal Blue ink.
    """)

    # Cell 1: Environment & Setup
    add_md("## 1. Environment & Setup")
    add_code("""
import os
import sys
import glob
import shutil
import subprocess

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "opencv-python-headless", "pillow", "pandas", "matplotlib", "reportlab", "svgwrite"], check=True)

# Clone or pull latest project repository
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
    print(f"Device: {torch.cuda.get_device_name(0)}")
print("Environment initialized successfully!")
    """)

    # Cell 2: Locate Dataset & Training Pages
    add_md("## 2. Locate Dataset Assets & Training Pages")
    add_code("""
dataset_dir = "/kaggle/input/homework-bud-data"

# Copy base checkpoints
if os.path.exists(os.path.join(dataset_dir, "checkpoints")):
    os.makedirs("checkpoints", exist_ok=True)
    os.system(f"cp -rf {dataset_dir}/checkpoints/* checkpoints/")
    print(f"[OK] Copied base checkpoints to checkpoints/")

# Copy styles
if os.path.exists(os.path.join(dataset_dir, "styles")):
    os.makedirs("styles", exist_ok=True)
    os.system(f"cp -rf {dataset_dir}/styles/* styles/")
    print(f"[OK] Copied styles to styles/")

# Copy Blank_Page.jpeg
for blank_candidate in [os.path.join(dataset_dir, "Blank_Page.jpeg"), "Blank_Page.jpeg"]:
    if os.path.exists(blank_candidate):
        shutil.copy(blank_candidate, "Blank_Page.jpeg")
        print(f"[OK] Blank ruled page ready: Blank_Page.jpeg")
        break

# Locate training images directory
train_dir = os.path.join(dataset_dir, "Training_Images")
if not os.path.exists(train_dir):
    train_dir = "Training_Images"

train_images = sorted(glob.glob(os.path.join(train_dir, "*.jp*g")) + glob.glob(os.path.join(train_dir, "*.png")))
train_images = [p for p in train_images if "blank" not in os.path.basename(p).lower()]
print(f"[OK] Found {len(train_images)} training pages in {train_dir}:")
for p in train_images:
    print(f"  - {os.path.basename(p)}")

# Copy lines_transcription.csv
if os.path.exists(os.path.join(dataset_dir, "lines_transcription.csv")):
    shutil.copy(os.path.join(dataset_dir, "lines_transcription.csv"), "lines_transcription.csv")
    print(f"[OK] Verified lines_transcription.csv ready from dataset.")
    """)

    # Cell 3: Segment Lines
    add_md("## 3. Segment Training Pages into Lines")
    add_code("""
from line_segmenter import segment_page_into_lines
import pandas as pd

output_lines_dir = "segmented_lines"
os.makedirs(output_lines_dir, exist_ok=True)

next_line_idx = 1
total_cropped = 0
for p in train_images:
    crops, next_line_idx = segment_page_into_lines(p, output_lines_dir, start_line_idx=next_line_idx)
    print(f"  {os.path.basename(p):<10} -> {len(crops)} line crops (up to line_{next_line_idx - 1:04d})")
    total_cropped += len(crops)

print(f"\\n[OK] Sliced {total_cropped} total lines across all {len(train_images)} pages!")

df_check = pd.read_csv("lines_transcription.csv")
print(f"[OK] lines_transcription.csv contains {len(df_check)} verified entries!")
display(df_check.tail(10))
    """)

    # Cell 4: Build Fine-Tuning Dataset (Prioritizing 7.jpeg)
    add_md("## 4. Extract Sequential Strokes & Weight Current Handwriting (7.jpeg)")
    add_code("""
from build_dataset_verified import build_dataset_from_verified_csv

build_dataset_from_verified_csv(
    csv_path="lines_transcription.csv",
    lines_dir="segmented_lines",
    output_dir="data/processed",
    current_page="7.jpeg",
    current_page_multiplier=12,
    style_output_prefix="styles/style-user-current"
)
    """)

    # Cell 5: Fine-Tune Graves RNN on Kaggle GPU
    add_md("## 5. Fine-Tune Graves RNN on GPU")
    add_code("""
from finetune_user import finetune_user_handwriting

# Fine-tune starting from warm_start_step 17900 for 150 steps (~10 full epochs)
finetune_user_handwriting(
    data_dir="data/processed/",
    checkpoint_dir="checkpoints",
    warm_start_step=17900,
    finetune_steps=150,
    learning_rate=0.00005,
    batch_size=16
)
    """)

    # Cell 6: End-to-End Ruled Homework Synthesis
    add_md("## 6. Synthesize Ruled Homework in Royal Blue Ink")
    add_code("""
from homework_engine import HomeworkEngine

homework_text = \"\"\"Assignment: Principles of Modern Physics & Mechanics
Question 1: Explain the photoelectric effect and how Planck's hypothesis contributed to Einstein's discovery.
Answer: The photoelectric effect demonstrates that light consists of discrete energy packets called quanta or photons. In 1905, Albert Einstein proposed that each photon carries energy proportional to its frequency, given by the relation E = hf. When incident photons strike a metallic surface, their entire quantum of energy is transferred instantaneously to conduction electrons. If the photon energy exceeds the work function of the material, electrons are emitted with maximum kinetic energy.

Question 2: State Newton's second law and describe why momentum is conserved in an isolated system.
Answer: Newton's second law establishes that the net applied external force equals the rate of change of linear momentum. In the absence of external forces, the total momentum of interacting bodies remains strictly constant across all collisions. This conservation law underlies both classical dynamics and relativistic interactions.
\"\"\"

engine = HomeworkEngine(
    checkpoint_dir="checkpoints",
    styles_dir="styles",
    ink_rgb=(26, 62, 175), # Royal Blue
    bias=0.88               # Clean, consistent handwriting
)

output_pages, pdf_path = engine.generate_homework(
    text=homework_text,
    blank_sheet_paths=["Blank_Page.jpeg"],
    output_dir="output",
    custom_style_prefix="styles/style-user-current" if os.path.exists("styles/style-user-current-strokes.npy") else None,
    max_chars_per_line=58
)

print(f"\\n[DONE] Generated {len(output_pages)} pages:")
for p in output_pages:
    print(f"  - {p}")
print(f"Assignment PDF: {pdf_path}")
    """)

    # Cell 7: Display Synthesized Homework Image
    add_md("## 7. Visual Inspection of Synthesized Ruled Homework")
    add_code("""
import matplotlib.pyplot as plt
from PIL import Image

if output_pages:
    first_page = Image.open(output_pages[0])
    plt.figure(figsize=(14, 18))
    plt.imshow(first_page)
    plt.title("Synthesized Homework Assignment (Ruled Sheet + Royal Blue Ink)", fontsize=14)
    plt.axis("off")
    plt.show()
    """)

    # Cell 8: Package Artifacts
    add_md("## 8. Package & Export Artifacts for Download")
    add_code("""
import shutil

# 1. Zip fine-tuned checkpoints
shutil.make_archive("finetuned_checkpoints", "zip", "checkpoints")

# 2. Zip output pages & PDF
shutil.make_archive("homework_output", "zip", "output")

# 3. Zip custom user style
shutil.make_archive("user_style", "zip", "styles")

print("=" * 65)
print("FINETUNING & SYNTHESIS ARTIFACTS READY:")
print("=" * 65)
for f in ["finetuned_checkpoints.zip", "homework_output.zip", "user_style.zip", "output/homework_assignment.pdf"]:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  ✓ {f:<35} ({size_mb:.2f} MB)")
print("=" * 65)
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
    create_notebook()
