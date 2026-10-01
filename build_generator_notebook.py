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

# 1. Extract checkpoints
checkpoints_found = False
for c_dir in candidate_dirs:
    zip_path = os.path.join(c_dir, "finetuned_checkpoints.zip")
    if os.path.exists(zip_path):
        print(f"[OK] Extracting fine-tuned checkpoints from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("checkpoints")
        checkpoints_found = True
        break
    elif os.path.exists(os.path.join(c_dir, "checkpoints")):
        files = glob.glob(os.path.join(c_dir, "checkpoints", "model-*"))
        if files:
            print(f"[OK] Copying checkpoints from {c_dir}/checkpoints...")
            os.system(f"cp -rf {c_dir}/checkpoints/* checkpoints/")
            checkpoints_found = True
            break

if not checkpoints_found:
    print("[WARNING] Could not locate fine-tuned checkpoints archive; checking local directory...")

# 2. Extract styles
styles_found = False
for c_dir in candidate_dirs:
    zip_path = os.path.join(c_dir, "user_style.zip")
    if os.path.exists(zip_path):
        print(f"[OK] Extracting user style conditioning from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("styles")
        styles_found = True
        break
    elif os.path.exists(os.path.join(c_dir, "styles")):
        files = glob.glob(os.path.join(c_dir, "styles", "*.npy"))
        if files:
            os.system(f"cp -rf {c_dir}/styles/* styles/")
            styles_found = True
            break

# 3. Discover all blank ruled sheet candidates to cycle through
blank_sheets = []
for c_dir in candidate_dirs:
    found = sorted(glob.glob(os.path.join(c_dir, "Blank_Page*.jp*g")) + glob.glob(os.path.join(c_dir, "*blank*.jp*g")) + glob.glob(os.path.join(c_dir, "Blank_Pages", "*.jp*g")))
    for b in found:
        dest = os.path.basename(b)
        if not os.path.exists(dest):
            shutil.copy(b, dest)
        if dest not in blank_sheets:
            blank_sheets.append(dest)

if not blank_sheets and os.path.exists("Blank_Page.jpeg"):
    blank_sheets = ["Blank_Page.jpeg"]

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
The Renaissance and the Scientific Revolution: Foundations of Modern Inquiry

The period transitioning from the late Middle Ages to the early modern era witnessed an unprecedented transformation in human intellect, artistic expression, and empirical methodology. Beginning in fourteenth-century Italy, the Renaissance rekindled a passionate fascination with classical Greco-Roman philosophy, literature, and humanist ethics. Scholars such as Petrarch and Erasmus challenged scholastic dogmatism, championing the capability of the human intellect to observe, interpret, and reshape the natural world. This cultural flourishing gradually dismantled medieval intellectual hierarchies and paved the way for the profound transformation known as the Scientific Revolution.

Central to this intellectual upheaval was the radical reconstitution of cosmology and astronomical observation. For centuries, Western thought had adhered rigidly to the Aristotelian-Ptolemaic geocentric model, which posited an immovable Earth situated at the center of concentric crystalline celestial spheres. In 1543, Nicolaus Copernicus published De revolutionibus orbium coelestium, proposing a heliocentric hypothesis that positioned the Sun at the center of orbital celestial motions. Although initially met with skepticism and resistance, the Copernican model found rigorous empirical support through Johannes Kepler's discovery of elliptical planetary orbits and Galileo Galilei's telescopic observations of lunar craters, sunspots, and the four Galilean moons orbiting Jupiter.

Concurrently, the philosophical framework of natural philosophy underwent a foundational shift toward experimental empiricism and mathematical formulation. Francis Bacon articulated the inductive method in his Novum Organum, arguing that genuine scientific comprehension must emerge from systematic observation, controlled experimentation, and the cautious induction of general axioms. Simultaneously, Rene Descartes introduced Cartesian doubt and analytical coordinate geometry, asserting that the physical universe could be understood as a vast mechanical system governed by precise mathematical laws.

This synthesis of mathematical description and empirical verification culminated in the monumental work of Sir Isaac Newton. In Philosophiae Naturalis Principia Mathematica published in 1687, Newton established the three universal laws of motion and formulated the law of universal gravitation. By demonstrating that the force governing an apple falling to the Earth is identical to the centripetal force retaining the Moon in its orbit, Newton unified terrestrial and celestial mechanics into a cohesive, deterministic mathematical continuum.

The implications of this scientific awakening reverberated far beyond the confines of astronomy and physics. In medicine, Andreas Vesalius published De humani corporis fabrica, correcting centuries of Galenic anatomical misconceptions through direct dissection of human cadavers. William Harvey subsequently elucidated the systemic circulation of blood pumped by the muscular contractions of the heart, replacing ancient humoral theories with mechanical physiology. Robert Boyle applied corpuscular theory to chemistry, distinguishing between mixtures and compounds while establishing the inverse relationship between gas pressure and volume.

Ultimately, the confluence of the Renaissance and the Scientific Revolution reshaped the epistemology of civilization. It established that nature operates in accordance with decipherable, immutable laws accessible to rational scrutiny. This paradigm of empirical skepticism, rigorous quantification, and peer-reviewed replication became the enduring engine of technological innovation, democratic governance, and contemporary scientific advancement.
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
