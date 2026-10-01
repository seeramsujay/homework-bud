#!/usr/bin/env python3
"""
Stages the homework-bud Kaggle dataset bundle for upload to Kaggle.
Collects files from the repository into kaggle/dataset/bundle/ without duplicating
them in Git tracking.

Usage:
    python kaggle/dataset/stage_dataset.py [--symlink | --copy]
    kaggle datasets version -p kaggle/dataset/bundle -m "Update dataset" --dir-mode zip
"""

import os
import sys
import shutil
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BUNDLE_DIR = os.path.join(SCRIPT_DIR, "bundle")

def stage_dataset(use_symlinks: bool = False):
    print(f"[INFO] Staging dataset bundle at: {BUNDLE_DIR}")
    if os.path.exists(BUNDLE_DIR):
        shutil.rmtree(BUNDLE_DIR)
    os.makedirs(BUNDLE_DIR, exist_ok=True)

    # 1. Metadata
    meta_src = os.path.join(SCRIPT_DIR, "dataset-metadata.json")
    if os.path.exists(meta_src):
        shutil.copy(meta_src, os.path.join(BUNDLE_DIR, "dataset-metadata.json"))
        print("  ✓ Staged dataset-metadata.json")

    # 2. Blank ruled page
    blank_src = os.path.join(REPO_ROOT, "Blank_Page.jpeg")
    if os.path.exists(blank_src):
        _link_or_copy(blank_src, os.path.join(BUNDLE_DIR, "Blank_Page.jpeg"), use_symlinks)
        print("  ✓ Staged Blank_Page.jpeg")

    # 3. Transcriptions
    csv_src = os.path.join(REPO_ROOT, "lines_transcription.csv")
    if os.path.exists(csv_src):
        _link_or_copy(csv_src, os.path.join(BUNDLE_DIR, "lines_transcription.csv"), use_symlinks)
        print("  ✓ Staged lines_transcription.csv")

    # 4. Checkpoints
    ckpt_src = os.path.join(REPO_ROOT, "checkpoints")
    if os.path.exists(ckpt_src):
        dest_ckpt = os.path.join(BUNDLE_DIR, "checkpoints")
        _copy_tree(ckpt_src, dest_ckpt, use_symlinks)
        print("  ✓ Staged checkpoints/")

    # 5. Styles
    styles_src = os.path.join(REPO_ROOT, "styles")
    if os.path.exists(styles_src):
        dest_styles = os.path.join(BUNDLE_DIR, "styles")
        _copy_tree(styles_src, dest_styles, use_symlinks)
        print("  ✓ Staged styles/")

    # 6. Training Images (if present locally)
    train_src = os.path.join(REPO_ROOT, "Training_Images")
    if os.path.exists(train_src):
        dest_train = os.path.join(BUNDLE_DIR, "Training_Images")
        _copy_tree(train_src, dest_train, use_symlinks)
        print("  ✓ Staged Training_Images/")

    print("\n[OK] Dataset bundle staged successfully!")
    print(f"To upload or update dataset on Kaggle, run:")
    print(f"  kaggle datasets version -p {os.path.relpath(BUNDLE_DIR, REPO_ROOT)} -m \"Update dataset\" --dir-mode zip")

def _link_or_copy(src, dst, use_symlink):
    if use_symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)

def _copy_tree(src, dst, use_symlink):
    if use_symlink:
        os.symlink(src, dst)
    else:
        shutil.copytree(src, dst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage Kaggle dataset bundle")
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying")
    args = parser.parse_args()
    stage_dataset(use_symlinks=args.symlink)
