"""Train once, select by validation loss, and evaluate on untouched MNIST test data."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(args.seed)
    # Download the official split through Keras; subsequent runs use its cache.
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
    x = x[..., None].astype("float32") / 255.0
    x_test = x_test[..., None].astype("float32") / 255.0
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=10000, random_state=args.seed, stratify=y)
    model = build_model()
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(args.output / "best_model.keras"),
            monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
            restore_best_weights=True),
        keras.callbacks.CSVLogger(str(args.output / "training_log.csv")),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    # Reload the selected checkpoint explicitly before the final test evaluation.
    model = keras.models.load_model(args.output / "best_model.keras")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    predicted = probabilities.argmax(axis=1)
    metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy),
        "epochs_run": len(history.epoch),
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "split_sizes": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "config": {**vars(args), "output": str(args.output)},
        "versions": {"tensorflow": tf.__version__, "numpy": np.__version__},
        "classification_report": classification_report(y_test, predicted,
            labels=list(range(10)), output_dict=True, zero_division=0)}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "history.json").write_text(json.dumps(history.history, indent=2))
    print(json.dumps({"test_loss": loss, "test_accuracy": accuracy}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(np.arange(1, len(history.epoch)+1), history.history[metric], label="Train")
        ax.plot(np.arange(1, len(history.epoch)+1), history.history["val_"+metric], label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric, title=metric.title())
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "learning_curves.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix(y_test, predicted), display_labels=range(10)).plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    # The first mistakes are selected deterministically, rather than cherry-picked.
    wrong = np.flatnonzero(predicted != y_test)[:16]
    if len(wrong):
        fig, axes = plt.subplots(4, 4, figsize=(9, 9))
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, wrong):
            ax.imshow(x_test[index, ..., 0], cmap="gray")
            ax.set_title(f"True: {y_test[index]} | Pred: {predicted[index]}")
        fig.tight_layout()
        fig.savefig(args.output / "misclassified.png", dpi=150)
        plt.close(fig)
    # Save raw test digits as convenient examples for the prediction interface.
    from PIL import Image
    example_dir = args.output / "examples"
    example_dir.mkdir(exist_ok=True)
    for digit in range(10):
        index = np.flatnonzero(y_test == digit)[0]
        Image.fromarray(np.uint8(x_test[index, ..., 0] * 255)).save(example_dir / f"digit_{digit}.png")


if __name__ == "__main__":
    main()
