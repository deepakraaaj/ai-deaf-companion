# AI Deaf Companion System Using Machine Learning

A machine learning system that recognises American Sign Language (ASL) hand gestures and translates them into text, to help deaf and hard-of-hearing users communicate with people who don't know sign language.

**Tech stack:** Python · Scikit-learn · Pandas · NumPy · Matplotlib/Seaborn · Jupyter Notebook

![Translation demo](reports/translation_demo.png)

## Dataset

[Sign Language MNIST](https://www.openml.org/d/45082) (downloaded automatically from OpenML): 34,627 grayscale 28×28 images of ASL hand gestures for 24 letters, 27,455 for training and 7,172 for testing. **J** and **Z** are excluded because they require motion.

## Pipeline

1. **Data cleaning** (`src/data.py`): checks for missing pixel values (filled with the column median), out-of-range intensities (clipped to 0–255) and duplicate rows, then scales pixels to 0–1.
2. **Class balancing:** training classes range from 957 to 1,294 samples, so every class is upsampled to the size of the largest class.
3. **Exploratory data analysis** (`notebooks/eda.ipynb`): class distributions before/after balancing, sample and average gestures per letter, pixel intensity distribution.
4. **Model training** (`src/train.py`): three multi-class supervised classifiers compared on the held-out test set:
   - PCA + Logistic Regression
   - Random Forest
   - PCA + SVM (RBF kernel)
5. **Evaluation:** accuracy, macro precision, recall and F1-score; per-letter classification report; confusion matrix; learning curve.
6. **Translation** (`src/translate.py`): classifies a sequence of gesture images and outputs the text.

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|-------|---------:|----------:|-------:|---------:|
| Logistic Regression (PCA 100) | 0.684 | 0.674 | 0.672 | 0.662 |
| Random Forest (200 trees) | 0.824 | 0.817 | 0.815 | 0.808 |
| **SVM RBF (PCA 60)** | **0.869** | **0.858** | **0.862** | **0.855** |

The best model (SVM) recognises letters like B, E, P and Q almost perfectly. Visually similar hand shapes such as **R / U / V / W** and **M / N** are the most common confusions; see the confusion matrix. Note: cross-validation F1 on the training split reaches ~0.99 (see learning curve) because the training set contains augmented variants of the same source photos, so near-duplicates land in both folds. The separate test set is the honest measure of real-world performance, and closing that gap (e.g. with more signers or hand-landmark features) is the main next step.

| Model comparison | Confusion matrix | Learning curve |
|---|---|---|
| ![](reports/model_comparison.png) | ![](reports/confusion_matrix.png) | ![](reports/learning_curve.png) |

All metrics are written to `reports/`:
- `metrics.csv`: model comparison table
- `classification_report.txt`: per-letter precision/recall/F1 for the best model
- `metrics_log.jsonl`: one timestamped entry per training run (cleaning stats + all model scores) to track progress over time

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.train                 # downloads data, trains & compares models, writes reports/ and models/
python -m src.translate HELLO       # translates sample gestures for a word into text
jupyter notebook notebooks/eda.ipynb
```

## Project structure

```
├── src/
│   ├── data.py          # loading, cleaning, balancing
│   ├── train.py         # model training, evaluation, charts, metric logs
│   └── translate.py     # gesture sequence -> text
├── notebooks/eda.ipynb  # exploratory data analysis + results
├── reports/             # charts and metrics
└── requirements.txt
```

## Future work

- Real-time recognition from a webcam using hand landmarks (e.g. MediaPipe) instead of static images
- Support for motion letters (J, Z) and whole-word signs with sequence models
- Text-to-speech output so the translated text can be spoken aloud
