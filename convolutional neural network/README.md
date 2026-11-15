# MNIST CNN Digit Recognizer

A complete educational machine-learning application: train a convolutional neural network on real handwritten digits, evaluate it on a held-out test set, save the selected model, and recognize a single digit from an uploaded or drawn image.

**Honest project status:** training has not been run in the preparation environment because TensorFlow is unavailable there. No pretrained weights, measured accuracy, or generated training charts are bundled. Run training before inference. The architecture is intentionally small enough for CPU training; runtime depends on your machine.

## 1. What this application does

- Downloads the real MNIST dataset through `keras.datasets.mnist`.
- Uses 50,000 training, 10,000 validation, and 10,000 independent test images.
- Trains a CNN to classify digits **0 through 9**.
- Saves the best model by validation loss and reports final test metrics.
- Produces learning curves, a confusion matrix, and examples of mistakes.
- Provides command-line inference and a local Streamlit web interface.
- Shows the actual preprocessed input so preprocessing failures are visible.

It is **not** general OCR: it does not recognize words, Persian numerals, multiple digits, mathematical expressions, or arbitrary photos. MNIST is real data but also a simplified benchmark, not proof of production readiness.

## 2. Dataset and data separation

MNIST contains 70,000 grayscale 28×28 images of handwritten digits. Keras exposes an official 60,000/10,000 train/test split. This project stratifies the original 60,000 examples into 50,000 training and 10,000 validation examples. The official 10,000-example test set stays separate.

- Training data updates weights.
- Validation data selects the checkpoint and stopping epoch.
- Test data estimates final performance; do not tune using test outcomes.

Images are converted to float32, divided by 255, and reshaped from `(N, 28, 28)` to `(N, 28, 28, 1)`. Integer labels are retained, so the loss is **sparse categorical cross-entropy**, not binary cross-entropy. Classes are mutually exclusive, so the output uses **softmax**, not independent sigmoids.

References: [MNIST dataset background](https://yann.lecun.com/exdb/mnist/) · [Keras dataset API](https://keras.io/api/datasets/mnist/). Consult original dataset terms before redistribution. This repository does not bundle the dataset.

## 3. Files

```text
mnist-cnn-digit-recognizer/
├── README.md
├── requirements.txt
├── .gitignore
├── model.py                 # Architecture and compilation
├── train.py                 # Training, checkpoint selection, evaluation, plots
├── preprocess.py            # External image preparation
├── predict.py               # Command-line prediction
├── app.py                   # Local upload/drawing web interface
├── MNIST_CNN.ipynb          # Notebook runner for the same project
└── tests/test_preprocess.py # Tests that do not require TensorFlow
```

All paths in the examples are relative to the repository root. Run commands there.

## 4. Installation

Use **Python 3.11** for the recommended starting environment. Dependencies are bounded ranges, not a verified cross-platform lockfile. TensorFlow wheel availability depends on Python, OS, and architecture. A GPU is optional; no separate GPU or Metal package is required for the basic CPU workflow.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: use this instead
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After a successful local run, optionally capture the environment:

```bash
python -m pip freeze > requirements-lock.txt
```

## 5. Train and evaluate

```bash
python train.py
# Explicit settings:
python train.py --epochs 15 --batch-size 128 --seed 42 --output artifacts
# Pipeline smoke run; this still uses the full dataset:
python train.py --epochs 1
```

The first execution needs internet to download MNIST into the Keras cache (usually `~/.keras/datasets`). Later runs can use the cache. Early stopping may finish before the requested epoch limit. The best checkpoint is chosen by minimum validation loss, not by maximum test accuracy. Existing output files in the same directory are overwritten; use distinct `--output` directories to preserve experiments. An old `misclassified.png` can remain if a later run has no mistakes, so use a fresh directory when comparing runs.

### Generated artifacts

| File | Meaning |
|---|---|
| `best_model.keras` | Full selected Keras model, including architecture and weights |
| `metrics.json` | Test loss, accuracy, per-class precision/recall/F1, settings and versions |
| `history.json` | Training and validation metrics for every epoch |
| `training_log.csv` | Tabular training log |
| `learning_curves.png` | Loss and accuracy over epochs |
| `confusion_matrix.png` | True labels on rows and predicted labels on columns |
| `misclassified.png` | Up to the first 16 incorrectly classified test images, if any |
| `examples/digit_0.png` … `digit_9.png` | One raw test example per class for interface trials |

Do not invent an accuracy value for the README: publish your actual `metrics.json` result after running. A single seeded run is not a complete statistical comparison. Seeds improve repeatability but do not guarantee bitwise-identical results across hardware or library versions.

## 6. Recognize a local image

Train first, then run:

```bash
python predict.py artifacts/examples/digit_7.png
python predict.py my-digit.png --preview artifacts/processed.png
python predict.py my-digit.png --model artifacts/best_model.keras --invert auto
```

The tool prints the three most likely classes and their model probabilities. `--invert yes` forces inversion, `--invert no` disables inversion, and `auto` estimates the background from border pixels. Use the preview to diagnose problems.

For best results: use a single handwritten digit, a plain background, adequate margins, good contrast, and no surrounding text. A tightly cropped digit touching the border can confuse the background heuristic. Camera shadows, colored paper, unusual handwriting, and printed fonts may be outside the training distribution.

## 7. Browser interface: upload or draw

```bash
streamlit run app.py
```

Open the local address printed by Streamlit. Either upload your own image or draw a digit on the provided canvas, click **Download drawing**, and upload the resulting PNG using the file picker. This deliberate two-step interaction avoids an additional drawing-component dependency; the drawing is **not** classified automatically while drawing.

The interface displays the original image, the 28×28 processed image, the predicted class, and all ten probabilities. Local inference needs the saved model but no external inference API. Uploaded images are processed in application memory and are not deliberately saved by the app. Hosting the app publicly requires your own deployment security, resource limits, and privacy review. Do not load untrusted model files.

## 8. CNN architecture explained

| Layer | Output excluding batch | Trainable parameters | Purpose |
|---|---|---:|---|
| Input | 28×28×1 | 0 | One grayscale image |
| Conv2D, 32 filters, 3×3, same, ReLU | 28×28×32 | 320 | Learn local stroke patterns |
| MaxPooling2D, 2×2 | 14×14×32 | 0 | Reduce spatial dimensions |
| Conv2D, 64 filters, 3×3, same, ReLU | 14×14×64 | 18,496 | Combine lower-level features |
| MaxPooling2D, 2×2 | 7×7×64 | 0 | Further spatial reduction |
| Flatten | 3136 | 0 | Convert feature maps to a vector |
| Dense, 128, ReLU | 128 | 401,536 | Combine features for classification |
| Dropout, 0.3 | 128 | 0 | Drop units with probability 30% during training |
| Dense, 10, softmax | 10 | 1290 | Distribution over digits 0–9 |
| **Total** | | **421,642** | |

A convolution filter shares weights across image locations. Its parameter count is `(kernel_height × kernel_width × input_channels + 1) × filters`, including biases. A dense layer uses `(inputs + 1) × outputs`. Pooling and flattening have no learned parameters.

ReLU computes `max(0, x)`. Max pooling keeps the largest activation in each local window. Softmax converts ten logits into nonnegative values summing to one. For a correct class y, sparse cross-entropy is `-log(p[y])`, averaged over examples. Adam updates parameters using gradients computed by backpropagation; learning rate is 0.001. With 50,000 training images and batch size 128 there are 391 optimizer updates per epoch (the last batch is smaller).

Dropout is active only during training. Thus training metrics and inference-mode validation metrics need not match, even without data leakage. This model has no explicit L1/L2 weight penalty, so its loss has no added weight-regularization term.

## 9. Why external preprocessing matters

`preprocess.py` performs the following steps:

1. Composite transparency over white and convert to grayscale.
2. Limit oversized images to a 2048×2048 bounding box while preserving aspect ratio.
3. Estimate whether the background is light from border pixels; invert if needed.
4. Subtract the estimated background and normalize contrast.
5. Reject blank/low-contrast images.
6. Locate foreground pixels above a threshold and crop their bounding box.
7. Fit the crop inside 20×20 without stretching its aspect ratio.
8. Place it into a black 28×28 field.
9. Shift its center of mass toward the center with zero-filled translation.
10. Return a float32 array of shape `(1, 28, 28, 1)` in `[0, 1]`.

This is a heuristic approximation to MNIST formatting, not a guaranteed reproduction. Official test evaluation uses the already formatted MNIST inputs directly; the prediction UI applies this extra pipeline. Consequently interface accuracy is not identical to the reported benchmark accuracy. A high softmax probability does not prove the input is a digit or the prediction is correct; there is no calibrated rejection or out-of-distribution detector.

## 10. Training and evaluation walkthrough

`train.py` parses CLI settings, validates positive epochs/batch size, creates the output directory, and sets the random seed. It loads and normalizes MNIST, performs the stratified validation split, and builds a fresh CNN. `model.summary()` displays shapes and parameter counts.

Three callbacks handle training: `ModelCheckpoint` saves the lowest-validation-loss model, `EarlyStopping` waits three unimproved epochs and restores the best in-memory weights, and `CSVLogger` records epoch metrics. The code explicitly reloads the checkpoint before evaluating test loss and accuracy. It predicts all test labels once more for per-class metrics and visualization, exports JSON and PNG files, and saves ten demonstration images. These examples are not a substitute for the full test evaluation.

`predict.py` loads a saved model rather than retraining it, prepares one external image, optionally saves its actual network input, predicts ten probabilities, and prints the top three. `app.py` wraps the same preprocessing/model behavior in a local UI and caches the loaded model. Its cache key includes file modification time to refresh after retraining.

## 11. Notebook and Colab

```bash
jupyter lab MNIST_CNN.ipynb
```

The notebook is a runner for the same modules, **not a second, potentially divergent copy of the training code**. Keep it with the Python files and run all cells in order. It invokes training, displays generated plots/metrics, and demonstrates prediction.

In Colab, upload/extract the whole repository into a directory, change into it with `%cd`, install `requirements.txt` if needed, and run the notebook there. GPU acceleration is optional. The local Streamlit interface is intended for a normal local terminal; this project does not include public tunnels for Colab.

## 12. Tests and verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q model.py train.py preprocess.py predict.py app.py
```

The unit tests require only NumPy and Pillow and cover output shape/range/centering, auto inversion, alpha transparency, blank input rejection, and invalid options. These tests do **not** validate CNN accuracy, training, checkpoint reload, or browser behavior. Syntax checks and notebook schema validation are distinct from successful model execution.

Before publishing measured results, install the complete environment, run training end-to-end, run CLI prediction on your own images, and manually exercise the browser drawing/upload workflow. Check actual parameter counts using `model.summary()`.

## 13. Publish on GitHub

Create an empty GitHub repository, then run:

```bash
git init
git add .
git commit -m "Add MNIST CNN training and digit recognition application"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mnist-cnn-digit-recognizer.git
git push -u origin main
```

Replace the username. Generated artifacts and virtual environments are ignored. If you choose to publish a trained model, document its actual metrics/environment and distribute it intentionally (for example via a release). Never commit secrets. No software license has been chosen on your behalf; add one only if you have the rights to do so. Public visibility alone does not grant an open-source license.

## 14. Suggested improvements

Add independent handwritten user examples, realistic augmentations, probability calibration, repeated-seed experiments, a lightweight convolution-only head, model export, automated integration tests, and an explicitly validated input rejection strategy. Do not claim these features are implemented here.

## 15. Annotated source reference

The following sections reproduce every Python source module with stable line numbers. Comments and docstrings describe intent, while the walkthroughs above explain the full pipeline. Consult source files, not these line numbers, when making edits.

### `model.py`

```text
  1: """CNN architecture shared by training and documentation."""
  2: from tensorflow import keras
  3: from tensorflow.keras import layers
  4: 
  5: 
  6: def build_model():
  7:     # Inputs have already been scaled to [0, 1]. Do not normalize twice.
  8:     model = keras.Sequential([
  9:         keras.Input(shape=(28, 28, 1)),
 10:         layers.Conv2D(32, 3, padding="same", activation="relu"),
 11:         layers.MaxPooling2D(2),
 12:         layers.Conv2D(64, 3, padding="same", activation="relu"),
 13:         layers.MaxPooling2D(2),
 14:         layers.Flatten(),
 15:         layers.Dense(128, activation="relu"),
 16:         layers.Dropout(0.3),
 17:         layers.Dense(10, activation="softmax"),
 18:     ], name="mnist_cnn")
 19:     model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
 20:                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
 21:     return model
```

### `preprocess.py`

```text
  1: """Convert a single, tightly related handwritten digit image into MNIST format."""
  2: import numpy as np
  3: from PIL import Image, ImageOps
  4: 
  5: 
  6: def prepare_image(image, invert="auto"):
  7:     """Return a (1, 28, 28, 1) float32 array; input must contain one digit.
  8: 
  9:     Auto inversion estimates the background from border pixels. This is a
 10:     heuristic, not a document segmentation or general OCR algorithm.
 11:     """
 12:     if invert not in ("auto", "yes", "no"):
 13:         raise ValueError("invert must be auto, yes, or no")
 14:     # Composite transparent images on white instead of treating alpha as black.
 15:     rgba = image.convert("RGBA")
 16:     background = Image.new("RGBA", rgba.size, "white")
 17:     image = Image.alpha_composite(background, rgba).convert("L")
 18:     image.thumbnail((2048, 2048))
 19:     pixels = np.asarray(image, dtype=np.float32)
 20:     border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
 21:     if invert == "yes" or (invert == "auto" and np.median(border) > 127):
 22:         pixels = 255.0 - pixels
 23:     # Remove an estimated background offset and stretch foreground contrast.
 24:     pixels = np.maximum(pixels - np.median(np.concatenate(
 25:         (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))), 0)
 26:     if pixels.max() < 10:
 27:         raise ValueError("The image appears blank or has insufficient contrast.")
 28:     pixels = pixels / pixels.max()
 29:     ys, xs = np.where(pixels > 0.15)
 30:     if not len(xs):
 31:         raise ValueError("No foreground digit was found.")
 32:     crop = Image.fromarray(np.uint8(pixels[ys.min():ys.max()+1, xs.min():xs.max()+1] * 255))
 33:     # Preserve aspect ratio, fit within 20x20, then center in a 28x28 field.
 34:     crop.thumbnail((20, 20), Image.Resampling.LANCZOS)
 35:     field = Image.new("L", (28, 28))
 36:     field.paste(crop, ((28-crop.width)//2, (28-crop.height)//2))
 37:     array = np.asarray(field, dtype=np.float32) / 255.0
 38:     yy, xx = np.indices(array.shape)
 39:     mass = array.sum()
 40:     dx = int(round(13.5 - (array * xx).sum() / mass))
 41:     dy = int(round(13.5 - (array * yy).sum() / mass))
 42:     # PIL translation fills exposed pixels with zero, unlike a wrapping roll.
 43:     field = field.transform((28, 28), Image.Transform.AFFINE,
 44:                             (1, 0, -dx, 0, 1, -dy), fillcolor=0)
 45:     return (np.asarray(field, dtype=np.float32) / 255.0)[None, ..., None]
```

### `train.py`

```text
  1: """Train once, select by validation loss, and evaluate on untouched MNIST test data."""
  2: import argparse
  3: import json
  4: from pathlib import Path
  5: import numpy as np
  6: import matplotlib
  7: matplotlib.use("Agg")
  8: import matplotlib.pyplot as plt
  9: from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
 10: from sklearn.model_selection import train_test_split
 11: import tensorflow as tf
 12: from tensorflow import keras
 13: from model import build_model
 14: 
 15: 
 16: def main():
 17:     parser = argparse.ArgumentParser(description=__doc__)
 18:     parser.add_argument("--epochs", type=int, default=15)
 19:     parser.add_argument("--batch-size", type=int, default=128)
 20:     parser.add_argument("--seed", type=int, default=42)
 21:     parser.add_argument("--output", type=Path, default=Path("artifacts"))
 22:     args = parser.parse_args()
 23:     if args.epochs < 1 or args.batch_size < 1:
 24:         parser.error("epochs and batch-size must be positive")
 25:     args.output.mkdir(parents=True, exist_ok=True)
 26:     keras.utils.set_random_seed(args.seed)
 27:     # Download the official split through Keras; subsequent runs use its cache.
 28:     (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
 29:     x = x[..., None].astype("float32") / 255.0
 30:     x_test = x_test[..., None].astype("float32") / 255.0
 31:     x_train, x_val, y_train, y_val = train_test_split(
 32:         x, y, test_size=10000, random_state=args.seed, stratify=y)
 33:     model = build_model()
 34:     model.summary()
 35:     callbacks = [
 36:         keras.callbacks.ModelCheckpoint(str(args.output / "best_model.keras"),
 37:             monitor="val_loss", save_best_only=True),
 38:         keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
 39:             restore_best_weights=True),
 40:         keras.callbacks.CSVLogger(str(args.output / "training_log.csv")),
 41:     ]
 42:     history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
 43:         epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
 44:     # Reload the selected checkpoint explicitly before the final test evaluation.
 45:     model = keras.models.load_model(args.output / "best_model.keras")
 46:     loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
 47:     probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=0)
 48:     predicted = probabilities.argmax(axis=1)
 49:     metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy),
 50:         "epochs_run": len(history.epoch),
 51:         "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
 52:         "split_sizes": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
 53:         "config": {**vars(args), "output": str(args.output)},
 54:         "versions": {"tensorflow": tf.__version__, "numpy": np.__version__},
 55:         "classification_report": classification_report(y_test, predicted,
 56:             labels=list(range(10)), output_dict=True, zero_division=0)}
 57:     (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
 58:     (args.output / "history.json").write_text(json.dumps(history.history, indent=2))
 59:     print(json.dumps({"test_loss": loss, "test_accuracy": accuracy}, indent=2))
 60:     fig, axes = plt.subplots(1, 2, figsize=(10, 4))
 61:     for ax, metric in zip(axes, ("loss", "accuracy")):
 62:         ax.plot(np.arange(1, len(history.epoch)+1), history.history[metric], label="Train")
 63:         ax.plot(np.arange(1, len(history.epoch)+1), history.history["val_"+metric], label="Validation")
 64:         ax.set(xlabel="Epoch", ylabel=metric, title=metric.title())
 65:         ax.legend()
 66:     fig.tight_layout()
 67:     fig.savefig(args.output / "learning_curves.png", dpi=150)
 68:     plt.close(fig)
 69:     fig, ax = plt.subplots(figsize=(8, 8))
 70:     ConfusionMatrixDisplay(confusion_matrix(y_test, predicted), display_labels=range(10)).plot(ax=ax, colorbar=False)
 71:     fig.tight_layout()
 72:     fig.savefig(args.output / "confusion_matrix.png", dpi=150)
 73:     plt.close(fig)
 74:     # The first mistakes are selected deterministically, rather than cherry-picked.
 75:     wrong = np.flatnonzero(predicted != y_test)[:16]
 76:     if len(wrong):
 77:         fig, axes = plt.subplots(4, 4, figsize=(9, 9))
 78:         for ax in axes.flat:
 79:             ax.axis("off")
 80:         for ax, index in zip(axes.flat, wrong):
 81:             ax.imshow(x_test[index, ..., 0], cmap="gray")
 82:             ax.set_title(f"True: {y_test[index]} | Pred: {predicted[index]}")
 83:         fig.tight_layout()
 84:         fig.savefig(args.output / "misclassified.png", dpi=150)
 85:         plt.close(fig)
 86:     # Save raw test digits as convenient examples for the prediction interface.
 87:     from PIL import Image
 88:     example_dir = args.output / "examples"
 89:     example_dir.mkdir(exist_ok=True)
 90:     for digit in range(10):
 91:         index = np.flatnonzero(y_test == digit)[0]
 92:         Image.fromarray(np.uint8(x_test[index, ..., 0] * 255)).save(example_dir / f"digit_{digit}.png")
 93: 
 94: 
 95: if __name__ == "__main__":
 96:     main()
```

### `predict.py`

```text
  1: """Recognize one handwritten digit from a local image."""
  2: import argparse
  3: from pathlib import Path
  4: from PIL import Image
  5: from tensorflow import keras
  6: from preprocess import prepare_image
  7: 
  8: 
  9: def main():
 10:     parser = argparse.ArgumentParser(description=__doc__)
 11:     parser.add_argument("image", type=Path)
 12:     parser.add_argument("--model", type=Path, default=Path("artifacts/best_model.keras"))
 13:     parser.add_argument("--invert", choices=["auto", "yes", "no"], default="auto")
 14:     parser.add_argument("--preview", type=Path, help="Optional path to save the processed 28x28 image")
 15:     args = parser.parse_args()
 16:     if not args.model.is_file():
 17:         parser.error("Model not found. Run python train.py first or supply --model.")
 18:     with Image.open(args.image) as image:
 19:         batch = prepare_image(image, invert=args.invert)
 20:     if args.preview:
 21:         args.preview.parent.mkdir(parents=True, exist_ok=True)
 22:         Image.fromarray((batch[0, ..., 0]*255).astype("uint8")).save(args.preview)
 23:     model = keras.models.load_model(args.model)
 24:     probabilities = model.predict(batch, verbose=0)[0]
 25:     for rank, digit in enumerate(probabilities.argsort()[::-1][:3], start=1):
 26:         print(f"{rank}. Digit {digit}: model probability {probabilities[digit]:.2%}")
 27:     print("Softmax probabilities are not calibrated guarantees of correctness.")
 28: 
 29: 
 30: if __name__ == "__main__":
 31:     main()
```

### `app.py`

```text
  1: """Local browser UI: upload a digit or draw one with a mouse/touchscreen."""
  2: from io import BytesIO
  3: from pathlib import Path
  4: import base64
  5: import streamlit as st
  6: import streamlit.components.v1 as components
  7: from PIL import Image
  8: from tensorflow import keras
  9: from preprocess import prepare_image
 10: 
 11: st.set_page_config(page_title="MNIST CNN Digit Recognizer", page_icon="✍️")
 12: st.title("Handwritten Digit Recognizer")
 13: st.write("Upload one handwritten digit, or draw and download a PNG below, then upload it.")
 14: st.caption("Runs locally. This model recognizes digits 0–9, not words or multi-digit numbers.")
 15: # Use a standard HTML canvas, avoiding a version-sensitive drawing extension.
 16: components.html("""
 17: <canvas id="pad" width="280" height="280" style="border:1px solid #888;touch-action:none"></canvas><br>
 18: <button id="clear">Clear</button> <button id="save">Download drawing</button>
 19: <script>
 20: const c=document.getElementById('pad'), x=c.getContext('2d');
 21: function clear(){x.fillStyle='white';x.fillRect(0,0,280,280);}
 22: clear(); x.lineWidth=20; x.lineCap='round'; x.strokeStyle='black';
 23: let down=false;
 24: function point(e){const r=c.getBoundingClientRect();return [
 25: (e.clientX-r.left)*c.width/r.width,(e.clientY-r.top)*c.height/r.height];}
 26: c.onpointerdown=e=>{down=true;c.setPointerCapture(e.pointerId);x.beginPath();x.moveTo(...point(e));};
 27: c.onpointermove=e=>{if(down){x.lineTo(...point(e));x.stroke();}};
 28: c.onpointerup=()=>down=false;c.onpointercancel=()=>down=false;
 29: document.getElementById('clear').onclick=clear;
 30: document.getElementById('save').onclick=()=>{const a=document.createElement('a');
 31: a.download='my-digit.png';a.href=c.toDataURL('image/png');a.click();};
 32: </script>
 33: """, height=340)
 34: model_path = Path(st.text_input("Model path", "artifacts/best_model.keras"))
 35: invert = st.selectbox("Invert colors", ["auto", "yes", "no"])
 36: upload = st.file_uploader("Upload a PNG, JPG, or WEBP containing one digit", type=["png", "jpg", "jpeg", "webp"])
 37: 
 38: @st.cache_resource
 39: def load_model(path, modification_time):
 40:     # The timestamp invalidates the cache after retraining at the same path.
 41:     return keras.models.load_model(path)
 42: 
 43: if upload:
 44:     try:
 45:         with Image.open(BytesIO(upload.getvalue())) as image:
 46:             st.image(image, caption="Original image", width=220)
 47:             batch = prepare_image(image, invert=invert)
 48:         st.image(batch[0, ..., 0], caption="Actual 28×28 model input", width=220, clamp=True)
 49:         if not model_path.is_file():
 50:             st.warning("Train a model first: python train.py")
 51:         else:
 52:             model = load_model(str(model_path), model_path.stat().st_mtime_ns)
 53:             probabilities = model.predict(batch, verbose=0)[0]
 54:             digit = int(probabilities.argmax())
 55:             st.success(f"Prediction: {digit} — model probability: {probabilities[digit]:.1%}")
 56:             st.bar_chart({"Probability": probabilities})
 57:             st.caption("Probabilities are not calibrated certainty. Non-digit images may still receive confident predictions.")
 58:     except (ValueError, OSError) as exc:
 59:         st.error(str(exc))
```

### `tests/test_preprocess.py`

```text
  1: """Dependency-light preprocessing tests; TensorFlow is not required."""
  2: import unittest
  3: import numpy as np
  4: from PIL import Image, ImageDraw, ImageOps
  5: from preprocess import prepare_image
  6: 
  7: 
  8: class PreprocessingTests(unittest.TestCase):
  9:     def digit(self):
 10:         image = Image.new("L", (120, 160), 255)
 11:         ImageDraw.Draw(image).line([(55, 30), (55, 130)], fill=0, width=12)
 12:         return image
 13: 
 14:     def test_shape_range_and_center(self):
 15:         array = prepare_image(self.digit())
 16:         self.assertEqual(array.shape, (1, 28, 28, 1))
 17:         self.assertEqual(array.dtype, np.float32)
 18:         self.assertGreater(array.max(), 0.9)
 19:         self.assertGreaterEqual(array.min(), 0)
 20:         self.assertLessEqual(array.max(), 1)
 21:         y, x = np.indices((28, 28))
 22:         a = array[0, ..., 0]
 23:         self.assertLess(abs((a*x).sum()/a.sum()-13.5), 0.6)
 24:         self.assertLess(abs((a*y).sum()/a.sum()-13.5), 0.6)
 25: 
 26:     def test_auto_inversion(self):
 27:         np.testing.assert_allclose(prepare_image(self.digit()),
 28:                                   prepare_image(ImageOps.invert(self.digit())))
 29: 
 30:     def test_transparency(self):
 31:         image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
 32:         ImageDraw.Draw(image).line([(50, 10), (50, 90)], fill="black", width=10)
 33:         self.assertEqual(prepare_image(image).shape, (1, 28, 28, 1))
 34: 
 35:     def test_blank_rejected(self):
 36:         for color in (0, 255):
 37:             with self.assertRaises(ValueError):
 38:                 prepare_image(Image.new("L", (32, 32), color))
 39: 
 40:     def test_invalid_option(self):
 41:         with self.assertRaises(ValueError):
 42:             prepare_image(self.digit(), invert="invalid")
 43: 
 44: 
 45: if __name__ == "__main__":
 46:     unittest.main()
```

