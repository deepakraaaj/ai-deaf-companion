"""Train and compare multi-class classifiers, then save the best model, metrics and charts.

Usage: python -m src.train
"""
import json
import time
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from src.data import LETTERS, ROOT, balance, load_splits, to_xy

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def candidate_models():
    return {
        "logistic_regression": make_pipeline(PCA(n_components=100, random_state=42),
                                             LogisticRegression(max_iter=2000, C=1.0)),
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "svm_rbf": make_pipeline(PCA(n_components=60, random_state=42),
                                 SVC(kernel="rbf", C=10, gamma="scale")),
    }


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_test, pred), "precision_macro": precision,
            "recall_macro": recall, "f1_macro": f1}


def plot_comparison(results: pd.DataFrame):
    ax = results.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot.bar(
        figsize=(9, 5), rot=0, ylim=(0, 1.12))
    ax.set_title("Model comparison on the test set")
    ax.set_ylabel("score")
    ax.legend(loc="upper center", ncol=4)
    plt.tight_layout()
    plt.savefig(REPORTS / "model_comparison.png", dpi=120)
    plt.close()


def plot_confusion(model, X_test, y_test, name):
    labels = sorted(LETTERS)
    fig, ax = plt.subplots(figsize=(11, 10))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, labels=labels,
                                          display_labels=[LETTERS[i] for i in labels],
                                          normalize="true", values_format=".1f", cmap="Blues",
                                          colorbar=False, ax=ax)
    ax.set_title(f"Normalised confusion matrix ({name})")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=120)
    plt.close()


def plot_learning_curve(model, X, y, name):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(8000, len(X)), replace=False)  # subsample to keep runtime reasonable
    sizes, train_scores, val_scores = learning_curve(model, X[idx], y[idx], cv=3, n_jobs=-1,
                                                     train_sizes=np.linspace(0.1, 1.0, 5), scoring="f1_macro")
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_scores.mean(axis=1), "o-", label="training F1")
    plt.plot(sizes, val_scores.mean(axis=1), "o-", label="cross-validation F1")
    plt.xlabel("training samples")
    plt.ylabel("macro F1")
    plt.title(f"Learning curve ({name})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / "learning_curve.png", dpi=120)
    plt.close()


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    raw_train, test, clean_stats = load_splits(balanced=False)
    X_train, y_train = to_xy(balance(raw_train))
    X_test, y_test = to_xy(test)
    print(f"Cleaning: {clean_stats}")
    print(f"Train (balanced): {X_train.shape}, test: {X_test.shape}")

    rows, fitted = [], {}
    for name, model in candidate_models().items():
        start = time.time()
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        metrics.update(model=name, train_seconds=round(time.time() - start, 1))
        rows.append(metrics)
        fitted[name] = model
        print(f"{name:20s} acc={metrics['accuracy']:.3f} f1={metrics['f1_macro']:.3f} ({metrics['train_seconds']}s)")

    results = pd.DataFrame(rows)[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "train_seconds"]]
    results.round(4).to_csv(REPORTS / "metrics.csv", index=False)
    best_name = results.sort_values("f1_macro", ascending=False).iloc[0]["model"]
    best = fitted[best_name]

    report = classification_report(y_test, best.predict(X_test), labels=sorted(LETTERS),
                                   target_names=[LETTERS[i] for i in sorted(LETTERS)], digits=3)
    (REPORTS / "classification_report.txt").write_text(f"Best model: {best_name}\n\n{report}")
    plot_comparison(results)
    plot_confusion(best, X_test, y_test, best_name)
    # Learning curve uses the unbalanced data: upsampled duplicates would leak across CV folds.
    plot_learning_curve(candidate_models()[best_name], *to_xy(raw_train), best_name)
    joblib.dump({"model": best, "letters": LETTERS, "name": best_name}, MODELS / "sign_classifier.joblib")

    with open(REPORTS / "metrics_log.jsonl", "a") as log:
        log.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "cleaning": clean_stats,
                              "best_model": best_name, "results": results.round(4).to_dict("records")}) + "\n")
    print(f"\nBest model: {best_name}\n{report}")


if __name__ == "__main__":
    main()
