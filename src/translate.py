"""Translate a sequence of sign-language gesture images into text.

Demo: picks one test image per letter of a word, classifies each and prints the translated text.
Usage: python -m src.translate HELLO
"""
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import IMG_SIZE, LETTERS, ROOT, clean, load_raw, split, to_xy

MODEL_PATH = ROOT / "models" / "sign_classifier.joblib"


def translate(images: np.ndarray) -> str:
    """images: (n, 784) array scaled 0-1, one gesture per letter. Returns the predicted text."""
    bundle = joblib.load(MODEL_PATH)
    return "".join(bundle["letters"][int(p)] for p in bundle["model"].predict(images))


def main(word: str):
    word = word.upper()
    codes = {v: k for k, v in LETTERS.items()}
    unsupported = sorted({c for c in word if c not in codes and c != " "})
    if unsupported:
        sys.exit(f"Unsupported letters {unsupported}: J and Z are motion gestures and not in the dataset.")

    _, test = split(clean(load_raw())[0])
    X, y = to_xy(test)
    rng = np.random.default_rng(0)
    letters = [c for c in word if c != " "]
    picks = [rng.choice(np.flatnonzero(y == codes[c])) for c in letters]
    predicted = iter(translate(X[picks]))
    text = "".join(" " if c == " " else next(predicted) for c in word)
    print(f"Signed : {word}\nText   : {text}")

    fig, axes = plt.subplots(1, len(picks), figsize=(1.6 * len(picks), 2.2), squeeze=False)
    for ax, i, true, pred in zip(axes[0], picks, letters, text.replace(" ", "")):
        ax.imshow(X[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        ax.set_title(f"{pred}", color="green" if pred == true else "red")
        ax.axis("off")
    fig.suptitle(f"Translated: {text}")
    plt.tight_layout()
    out = ROOT / "reports" / "translation_demo.png"
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "HELLO")
