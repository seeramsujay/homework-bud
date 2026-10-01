#!/usr/bin/env python3
"""
Master builder script to build all 3 Kaggle notebooks:
1. Stage 2 Fine-Tuning: kaggle/kernels/finetune/homework_bud_kaggle.ipynb
2. 1000-Word Essay Generator: kaggle/kernels/essay_generator/homework_generator_kaggle.ipynb
3. Original RNN Style 6 Generator: kaggle/kernels/style6_generator/homework_original_style_generator.ipynb
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from build_kaggle_notebook import create_notebook as build_finetune
from build_generator_notebook import create_generator_notebook as build_generator
from build_original_style_notebook import create_original_style_notebook as build_style6

def build_all():
    print("=" * 60)
    print("Building all Kaggle Notebooks for Homework-Bud...")
    print("=" * 60)

    print("\n[1/3] Building Fine-Tuning Notebook...")
    build_finetune()

    print("\n[2/3] Building Essay Generator Notebook...")
    build_generator()

    print("\n[3/3] Building Original RNN Style 6 Generator Notebook...")
    build_style6()

    print("\n" + "=" * 60)
    print("[SUCCESS] All Kaggle notebooks generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    build_all()
