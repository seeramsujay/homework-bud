# Kaggle Pipelines & Notebooks

This directory manages all Kaggle integrations for **Homework-Bud**:

```
kaggle/
├── builders/                       # Generators for Kaggle Jupyter notebooks
│   ├── build_kaggle_notebook.py    # Fine-tuning notebook builder
│   ├── build_generator_notebook.py # 1,000-word essay generator builder
│   ├── build_original_style_notebook.py # Style 6 generator builder
│   └── build_all.py                # Master builder to regenerate all notebooks
│
├── kernels/                        # Self-contained Kaggle kernels
│   ├── finetune/                   # Stage 2 RNN Fine-Tuning
│   │   ├── kernel-metadata.json    # sujaykid/homework-bud-handwriting-synthesis
│   │   └── homework_bud_kaggle.ipynb
│   ├── essay_generator/            # 1,000+ Word Essay Generator (Custom Style)
│   │   ├── kernel-metadata.json    # sujaykid/homework-bud-essay-assignment-generator
│   │   └── homework_generator_kaggle.ipynb
│   └── style6_generator/           # Style 6 Maximum Legibility Generator
│       ├── kernel-metadata.json    # sujaykid/homework-bud-original-rnn-style-6-generator
│       └── homework_original_style_generator.ipynb
│
└── dataset/                        # Kaggle Dataset bundle management
    ├── dataset-metadata.json       # sujaykid/homework-bud-data
    └── stage_dataset.py            # Stages blank pages, checkpoints, and styles
```

---

## 🚀 Workflows

### 1. Rebuild Notebooks
To regenerate all 3 Kaggle notebooks from the builder scripts:
```bash
python kaggle/builders/build_all.py
```

### 2. Push Kernels to Kaggle
Push any kernel directly with the Kaggle CLI:

- **Stage 2 Fine-Tuning**:
  ```bash
  kaggle kernels push -p kaggle/kernels/finetune
  ```
- **Essay Generator (Custom Style)**:
  ```bash
  kaggle kernels push -p kaggle/kernels/essay_generator
  ```
- **Original Style 6 Generator**:
  ```bash
  kaggle kernels push -p kaggle/kernels/style6_generator
  ```

### 3. Check Kernel Status
```bash
kaggle kernels status sujaykid/homework-bud-original-rnn-style-6-generator
kaggle kernels status sujaykid/homework-bud-essay-assignment-generator
kaggle kernels status sujaykid/homework-bud-handwriting-synthesis
```

### 4. Stage & Update Dataset
To update `sujaykid/homework-bud-data` with new checkpoints, styles, or handwriting images without cluttering git:
```bash
python kaggle/dataset/stage_dataset.py
kaggle datasets version -p kaggle/dataset/bundle -m "Updated dataset" --dir-mode zip
```
