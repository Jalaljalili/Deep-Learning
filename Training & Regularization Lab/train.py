# -*- coding: utf-8 -*-
"""Educational two-moons experiments. Run from the repository root."""

# %% 1. Imports and environment setup
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers

keras.utils.set_random_seed(42)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% 2. Generate and split the dataset
X, y = make_moons(n_samples=600, noise=0.25, random_state=42)


X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=200, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

print(f"Train: {len(X_train)}  |  Validation: {len(X_val)}  |  Test: {len(X_test)}\n")

# %% 3. Model-building function
def build_model(hidden=(), dropout=0.0, l1=0.0, l2=0.0):
    """hidden: number of units in each hidden layer, e.g. (32, 32)"""
    model = keras.Sequential([keras.Input(shape=(2,))])

    for units in hidden:
        # L1/L2 penalties on hidden-layer kernels are added to the loss.
        reg = regularizers.L1L2(l1=l1, l2=l2) if (l1 or l2) else None
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=reg))

        # Dropout is active during training and disabled during inference.
        if dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation="sigmoid"))  # Output: probability of class 1.

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),  # Learning rate for Adam.
        loss="binary_crossentropy",                           # Loss for binary classification.
        metrics=["accuracy"],
    )
    return model

# %% 4. Training function
def train(model, epochs=200, batch_size=32, callbacks=None):
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),  # Evaluate on validation data after each epoch.
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

# %% 5. Data loss without regularization penalties
def data_loss(model, X_, y_):
    """Return prediction loss (L_data), excluding L1/L2 penalties, for comparison."""
    p = model.predict(X_, verbose=0).ravel()
    return float(keras.losses.BinaryCrossentropy()(y_, p))

# %% 6. Collect comparison results
results = []
def report(name, model):
    tr = data_loss(model, X_train, y_train)
    va = data_loss(model, X_val, y_val)
    results.append((name, tr, va))
    print(f"{name:<22} train_loss={tr:.3f}   val_loss={va:.3f}   gap={va - tr:+.3f}")

# %% 7. Plot training history
def plot_history(history, title, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=110)
    plt.show()
    plt.close()

# %% 8. Define the large architecture
BIG = (128, 128, 128)

# %% 9. Linear baseline
m_under = build_model(hidden=())
h = train(m_under)
report("1) Underfitting", m_under)
plot_history(h, "1) Underfitting: linear baseline", "1_underfitting.png")

# %% 10. High-capacity model
m_over = build_model(hidden=BIG)
h = train(m_over)
report("2) Overfitting", m_over)
plot_history(h, "2) Overfitting: high-capacity model", "2_overfitting.png")

# %% 11. Compare batch sizes
plt.figure(figsize=(6, 4))
for bs in [4, 32, 200]:  # 200 uses the full training set per update; the optimizer is still Adam.
    m = build_model(hidden=(32, 32))
    h = train(m, epochs=100, batch_size=bs)
    updates = int(np.ceil(len(X_train) / bs))
    plt.plot(h.history["loss"], label=f"batch_size={bs}  ({updates} updates/epoch)")
plt.title("3) Batch Size: training-loss comparison")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "3_batch_size.png", dpi=110)
plt.show()
plt.close()

# %% 12. L2 and L1 experiments
m_l2 = build_model(hidden=BIG, l2=0.01)  # Loss + λ Σ w²
h = train(m_l2)
report("4a) L2 (λ=0.01)", m_l2)
plot_history(h, "4a) L2 Regularization", "4a_l2.png")

m_l1 = build_model(hidden=BIG, l1=0.001)  # Loss + λ Σ |w|
h = train(m_l1)
report("4b) L1 (λ=0.001)", m_l1)
plot_history(h, "4b) L1 Regularization", "4b_l1.png")

# %% 13. Dropout experiment
m_drop = build_model(hidden=BIG, dropout=0.3)  # Each hidden unit is dropped with probability 0.3 during training.
h = train(m_drop)
report("5) Dropout (0.3)", m_drop)
plot_history(h, "5) Dropout: dropout rate = 0.3", "5_dropout.png")

# %% 14. Early stopping
early = keras.callbacks.EarlyStopping(
    monitor="val_loss",          # Monitor validation loss, not training loss.
    patience=20,                 # Allow 20 epochs without improvement.
    restore_best_weights=True,   # Restore the best weights in memory; no checkpoint file is written.
)
m_es = build_model(hidden=BIG)
h = train(m_es, callbacks=[early])
best_epoch = int(np.argmin(h.history["val_loss"])) + 1
print(f"Epochs run: {len(h.history['loss'])}; best validation epoch: {best_epoch}")
report("6) Early Stopping", m_es)
plot_history(h, f"6) Early Stopping (best epoch = {best_epoch})", "6_early_stopping.png")

# %% 15. Candidate final model
m_good = build_model(hidden=(32, 32), dropout=0.2, l2=0.001)  # Moderate model capacity.
early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
h = train(m_good, epochs=300, batch_size=32, callbacks=[early])
report("7) Good model", m_good)
plot_history(h, "7) Good Fit: candidate final model", "7_good_model.png")

# %% 16. Summary and test evaluation
print("\n" + "=" * 62)
print(f"{'model':<22}{'train_loss':>12}{'val_loss':>12}{'gap':>10}")
print("-" * 62)
for name, tr, va in results:
    print(f"{name:<22}{tr:>12.3f}{va:>12.3f}{va - tr:>+10.3f}")
print("=" * 62)
print("Large gap: possible overfitting | Both high: possible underfitting | Both low and close: possible good fit")

test_total_loss, test_acc = m_good.evaluate(X_test, y_test, verbose=0)
test_data_loss = data_loss(m_good, X_test, y_test)
print(f"\nTest: data_loss={test_data_loss:.3f}  total_loss={test_total_loss:.3f}  accuracy={test_acc:.1%}")

# %% 17. Decision boundaries
xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.7, 2.2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (model, title) in zip(axes, [(m_under, "Underfitting"), (m_over, "Overfitting"), (m_good, "Good Fit")]):
    zz = model.predict(grid, verbose=0).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=["#92c5de", "#f4a582"], alpha=0.6)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", s=12, edgecolors="k", linewidths=0.3)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "8_decision_boundaries.png", dpi=110)
plt.show()
plt.close(fig)

