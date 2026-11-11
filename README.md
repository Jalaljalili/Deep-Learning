# Deep Learning: Training & Regularization Lab

An educational TensorFlow/Keras project exploring model capacity, batch size, L1/L2 regularization, dropout, and early stopping on a synthetic two-moons dataset.

Both the notebook and the Python script contain the same experiments in the same order. All documentation, comments, docstrings, and explanatory messages are in English.

## Repository structure

```text
.
├── HW.ipynb          # Clean notebook with section headings and no saved outputs
├── train.py          # Equivalent experiments as a Python script
├── README.md         # Setup, interpretation, and line-by-line explanation
├── requirements.txt  # Dependency version ranges
```

## Installation

Python 3.11 or 3.12 is recommended. Run the following commands from the repository directory:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: use this instead of the command above
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Select a TensorFlow release compatible with your Python version, operating system, and hardware. A GPU is not required. The dependency file specifies version ranges, not a tested lockfile. After a successful run, record your exact environment if needed:

```bash
python -m pip freeze > requirements-lock.txt
```

## Running the project

### Python script

```bash
python train.py
```

With an interactive plotting backend, `plt.show()` may wait until you close each figure window. For a headless environment on macOS/Linux:

```bash
MPLBACKEND=Agg python train.py
```

In PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
python train.py
```

### Jupyter Notebook

```bash
jupyter lab HW.ipynb
```

Restart the kernel and run all cells from top to bottom. The cells depend on variables created earlier; rerunning only selected cells can produce stale results or duplicate entries in the results list.

### Google Colab

Upload `HW.ipynb` to Colab and run all cells in order. Install missing dependencies if the runtime does not already provide them. Files in `outputs/` are stored in the runtime filesystem and must be downloaded before that runtime is discarded.

## Experimental design

- Generate 600 samples with two features and binary labels using `make_moons`.
- Set the noise level to 0.25 and use a fixed seed of 42.
- Split the samples into 200 training, 200 validation, and 200 test examples.
- Preserve the original non-stratified split; class counts are not explicitly balanced in each subset.
- Train a linear baseline without hidden layers.
- Train a high-capacity network with three hidden layers of 128 units each.
- Compare batch sizes of 4, 32, and 200 using two 32-unit hidden layers.
- Test L2 with coefficient 0.01, L1 with coefficient 0.001, and dropout with rate 0.3 separately.
- Apply early stopping based on validation loss.
- Train a predefined final candidate with hidden layers `(32, 32)`, dropout 0.2, and L2 coefficient 0.001, for up to 300 epochs with patience 30.

All models use Adam with learning rate 0.01 and binary cross-entropy. The final candidate is predefined: the code does not automatically choose the best model from the comparison table. Test data is not used to compute training gradients or select the stopping epoch.

## Understanding the losses

For binary labels and sigmoid probabilities, the prediction loss is:

$$L_{data}=-\frac{1}{N}\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]$$

With regularization, the objective is:

$$L_{total}=L_{data}+\lambda_1\sum|w|+\lambda_2\sum w^2$$

Here, penalties apply only to hidden-layer kernels. Biases and the output-layer kernel are not regularized.

- `history.history["loss"]` and `history.history["val_loss"]` include applicable regularization penalties.
- `data_loss()` computes only binary cross-entropy and excludes the model's regularization losses.
- `report()` uses `data_loss()` for both training and validation data, making the reported loss definition consistent across models.
- Training loss recorded by `fit()` is accumulated during weight updates, with dropout active when present.
- Post-training loss reported by `report()` is computed with fixed final weights and dropout disabled. It need not equal the last recorded training loss.
- The final test report includes both `test_data_loss` and `test_total_loss`, along with accuracy.

## Interpreting the experiments responsibly

### Underfitting and overfitting

A large positive validation-minus-training loss gap can indicate overfitting. A small gap alone is not enough: both losses may be high. Consider the loss magnitudes, their trends, and the decision boundary together.

The experiment names describe their teaching purpose, not guaranteed outcomes. A linear classifier has limited capacity for the two-moons geometry, but whether a particular loss is considered high is contextual. A large network does not necessarily overfit on every run.

### Dropout

A dropout rate of 0.3 means each hidden unit is independently dropped with probability 30% during training, not that half the units are removed. Surviving activations are rescaled during training. Dropout is disabled for prediction and evaluation.

### Early stopping

Early stopping monitors validation loss and restores the best weights in memory. It does not write a model checkpoint file. Training can also finish because it reaches the epoch limit; the printed epoch count alone does not prove that early stopping triggered.

### Batch-size comparison

With 200 training samples, batch sizes 4, 32, and 200 produce 50, 7, and 1 updates per epoch, respectively. The last batch may be smaller than the configured batch size.

Equal epoch counts do not imply equal update counts. The models also have different initial weights because the global random state advances between constructions. Therefore, this is an illustrative comparison, not a rigorously controlled single-variable study. The plot shows epoch-averaged loss, not per-update gradient noise. Full-batch updates still use Adam; they do not turn the optimizer into plain gradient descent.

### Reproducibility and evaluation

Setting the seed improves reproducibility, but does not guarantee identical results across dependency versions, devices, and backends. For a stronger study, repeat with multiple seeds and report means and variability. Repeatedly using validation data for model development can also make its performance optimistic; keep the test set independent and avoid tuning based on test results.

## Generated outputs

The script prints a comparison table and final test metrics. It saves nine figures under `outputs/`:

| File | Contents |
|---|---|
| `1_underfitting.png` | Linear baseline loss curves |
| `2_overfitting.png` | High-capacity model loss curves |
| `3_batch_size.png` | Training losses for different batch sizes |
| `4a_l2.png` | L2 experiment loss curves |
| `4b_l1.png` | L1 experiment loss curves |
| `5_dropout.png` | Dropout experiment loss curves |
| `6_early_stopping.png` | Early-stopping experiment loss curves |
| `7_good_model.png` | Final candidate loss curves |
| `8_decision_boundaries.png` | Decision regions for three models |

The project does not save trained weights or model files. Old notebook outputs were intentionally removed; no newly measured numerical results are claimed in this repository. Generated figures are ignored by Git by default.

## Changes from the supplied notebook

1. Removed saved outputs, execution counts, and Colab-specific metadata.
2. Added notebook section headings and an equivalent Python script.
3. Corrected the dropout comment and removed guaranteed-outcome claims from plot titles.
4. Centralized figure output in `outputs/` and closed figures after display.
5. Reported test prediction loss separately from total loss including L2.
6. Preserved the original dataset, seed, architectures, hyperparameters, and experiment order.
7. Converted documentation, code comments, docstrings, section headings, and explanatory console messages to English without changing the numerical operations.

## Publishing to GitHub

Create an empty GitHub repository, then run these commands from this project directory:

```bash
git init
git add .
git commit -m "Add deep learning regularization experiments and documentation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the username and repository placeholders. Do not commit credentials, access tokens, or virtual environments. No license has been selected on your behalf. A public repository is not automatically open-source licensed; add a suitable license only if you hold the necessary rights.

## Validation status

The Python script has been checked for valid syntax, the notebook has been validated against its format schema, and its code cells have been checked against the script. The English revision also preserves the computational AST of the previous script, apart from translated docstrings and explanatory text.

TensorFlow was not installed in the preparation environment. Full training, fresh plots, and numerical results have therefore not been executed or verified there. Run the complete notebook or script in your own compatible environment before publishing results.

## Line-by-line code explanation

Line numbers below refer to the included `train.py`. Each code block corresponds to one notebook code cell. Every line inside each block, including blank lines, is explained. The `# %%` lines are section labels and are not executable. The script header consists of line 1 (UTF-8 encoding declaration), line 2 (module docstring), and line 3 (a blank separator). Blank lines between sections have no computational effect.

### Imports and environment setup

```python
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
```

- **Line 5:** Import the path-handling class used to construct portable output paths.
- **Line 6:** Blank line for readability; it has no effect on execution.
- **Line 7:** Import NumPy as np for array operations, coordinate grids, and numerical utilities.
- **Line 8:** Import Matplotlib plotting functions as plt.
- **Line 9:** Import the synthetic two-moons dataset generator.
- **Line 10:** Import the utility for randomly splitting features and labels together.
- **Line 11:** Import the Keras interface provided by TensorFlow.
- **Line 12:** Import neural-network layers and weight-regularization utilities.
- **Line 13:** Blank line for readability; it has no effect on execution.
- **Line 14:** Set the random seed for Python, NumPy, and the backend to improve reproducibility.
- **Line 15:** Blank line for readability; it has no effect on execution.
- **Line 16:** Define the output directory relative to the current working directory.
- **Line 17:** Create the output directory and any missing parents; an existing directory is allowed.

### Generate and split the dataset

```python
X, y = make_moons(n_samples=600, noise=0.25, random_state=42)


X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=200, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

print(f"Train: {len(X_train)}  |  Validation: {len(X_val)}  |  Test: {len(X_test)}\n")
```

- **Line 20:** Generate 600 two-feature samples and binary labels, with noise 0.25 and dataset seed 42.
- **Line 21:** Blank line for readability; it has no effect on execution.
- **Line 22:** Blank line for readability; it has no effect on execution.
- **Line 23:** Keep 200 aligned feature/label pairs for training and reserve the remaining 400 for the next split; no stratification is requested.
- **Line 24:** Split the remaining 400 samples equally into validation and test subsets.
- **Line 25:** Blank line for readability; it has no effect on execution.
- **Line 26:** Print the specified text or metrics. Where used, .3f gives three decimal places, .1% formats a percentage, and + explicitly shows the sign of the gap.

### Model-building function

```python
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
```

- **Line 29:** Define the model factory. hidden specifies hidden-layer widths; the other arguments control dropout and regularization. The default has no hidden layers.
- **Line 30:** Docstring documenting the function purpose or arguments; it does not perform a numerical calculation.
- **Line 31:** Create a sequential model accepting two features per example; the batch dimension is implicit.
- **Line 32:** Blank line for readability; it has no effect on execution.
- **Line 33:** Iterate through the requested hidden-layer widths, creating one layer per value.
- **Line 34:** Comment only; Python does not execute it. L1/L2 penalties on hidden-layer kernels are added to the loss.
- **Line 35:** Create an L1/L2 regularizer when either coefficient is nonzero; otherwise use no regularizer.
- **Line 36:** Append a fully connected ReLU layer and apply the optional penalty to its kernel, not its bias.
- **Line 37:** Blank line for readability; it has no effect on execution.
- **Line 38:** Comment only; Python does not execute it. Dropout is active during training and disabled during inference.
- **Line 39:** Add a dropout layer only when the requested rate is positive.
- **Line 40:** Append dropout with the specified drop probability; it is active only during training.
- **Line 41:** Blank line for readability; it has no effect on execution.
- **Line 42:** Append a single sigmoid output unit representing the probability of class 1; its kernel has no regularizer.
- **Line 43:** Blank line for readability; it has no effect on execution.
- **Line 44:** Begin configuring the optimizer, objective, and metrics; compilation does not train the model.
- **Line 45:** Use Adam with learning rate 0.01; this is an adaptive optimizer, not plain gradient descent.
- **Line 46:** Select binary cross-entropy as the training objective for the two-class problem.
- **Line 47:** Track classification accuracy in addition to the loss.
- **Line 48:** Close the preceding multi-line function call.
- **Line 49:** Return the configured model.

### Training function

```python
def train(model, epochs=200, batch_size=32, callbacks=None):
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),  # Evaluate on validation data after each epoch.
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
```

- **Line 52:** Define a training wrapper with defaults of 200 epochs and batch size 32; it reads the dataset variables from the surrounding module.
- **Line 53:** Run training and return the History object containing epoch-level metrics.
- **Line 54:** Supply training features and labels as positional arguments to fit.
- **Line 55:** Evaluate this held-out validation subset after each epoch without using it to compute training updates.
- **Line 56:** Set the maximum number of complete passes through the training dataset.
- **Line 57:** Set the number of samples per update; the final batch may be smaller.
- **Line 58:** Attach optional training callbacks such as EarlyStopping.
- **Line 59:** Suppress progress output from fit.
- **Line 60:** Close the preceding multi-line function call.

### Data loss without regularization penalties

```python
def data_loss(model, X_, y_):
    """Return prediction loss (L_data), excluding L1/L2 penalties, for comparison."""
    p = model.predict(X_, verbose=0).ravel()
    return float(keras.losses.BinaryCrossentropy()(y_, p))
```

- **Line 63:** Define a helper for computing prediction-only binary cross-entropy on a supplied dataset.
- **Line 64:** Docstring documenting the function purpose or arguments; it does not perform a numerical calculation.
- **Line 65:** Predict probabilities in inference mode and flatten the (N, 1) output to shape (N,).
- **Line 66:** Compute mean binary cross-entropy and return a Python float, without adding model regularization losses.

### Collect comparison results

```python
results = []
def report(name, model):
    tr = data_loss(model, X_train, y_train)
    va = data_loss(model, X_val, y_val)
    results.append((name, tr, va))
    print(f"{name:<22} train_loss={tr:.3f}   val_loss={va:.3f}   gap={va - tr:+.3f}")
```

- **Line 69:** Initialize the results list; rerunning this line clears previously recorded entries.
- **Line 70:** Define a reporting helper for one named model.
- **Line 71:** Compute prediction-only loss on the training set using final weights and inference mode.
- **Line 72:** Compute the same prediction-only loss on the validation set.
- **Line 73:** Append the model name and its two losses to the comparison list.
- **Line 74:** Print the specified text or metrics. Where used, .3f gives three decimal places, .1% formats a percentage, and + explicitly shows the sign of the gap.

### Plot training history

```python
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
```

- **Line 77:** Define a helper that plots and saves the training history.
- **Line 78:** Create a new figure measuring 6 by 4 inches.
- **Line 79:** Plot recorded training loss, including applicable regularization penalties and training-mode dropout effects.
- **Line 80:** Plot end-of-epoch validation loss, including applicable penalties but with dropout disabled.
- **Line 81:** Set the figure title; inspect the observed curves rather than assuming a particular outcome.
- **Line 82:** Label the horizontal axis; history curves use zero-based plotted indices although the label says Epoch.
- **Line 83:** Label the vertical axis with the displayed loss quantity.
- **Line 84:** Display the legend using the labels assigned to plotted curves.
- **Line 85:** Show reference grid lines with opacity 0.3.
- **Line 86:** Adjust spacing to reduce overlapping labels and panels.
- **Line 87:** Save the figure under outputs/ at a resolution of 110 dots per inch.
- **Line 88:** Display the figure; interactive backends may pause the script until its window is closed.
- **Line 89:** Close the current figure to release plotting resources.

### Define the large architecture

```python
BIG = (128, 128, 128)
```

- **Line 92:** Define the large model as three hidden layers with 128 units each.

### Linear baseline

```python
m_under = build_model(hidden=())
h = train(m_under)
report("1) Underfitting", m_under)
plot_history(h, "1) Underfitting: linear baseline", "1_underfitting.png")
```

- **Line 95:** Build a fresh independent model with the architecture and hyperparameters shown; its weights are initialized from scratch.
- **Line 96:** Train the indicated model with the supplied arguments and store its History in h, replacing the previous history reference.
- **Line 97:** Compute, print, and record this model's training and validation prediction losses.
- **Line 98:** Plot this experiment's history using the given title and save the named PNG file.

### High-capacity model

```python
m_over = build_model(hidden=BIG)
h = train(m_over)
report("2) Overfitting", m_over)
plot_history(h, "2) Overfitting: high-capacity model", "2_overfitting.png")
```

- **Line 101:** Build a fresh independent model with the architecture and hyperparameters shown; its weights are initialized from scratch.
- **Line 102:** Train the indicated model with the supplied arguments and store its History in h, replacing the previous history reference.
- **Line 103:** Compute, print, and record this model's training and validation prediction losses.
- **Line 104:** Plot this experiment's history using the given title and save the named PNG file.

### Compare batch sizes

```python
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
```

- **Line 107:** Create a new figure measuring 6 by 4 inches.
- **Line 108:** Repeat the experiment for batch sizes 4, 32, and 200; 200 is the full training set, but optimization still uses Adam.
- **Line 109:** Build a fresh independent model with the architecture and hyperparameters shown; its weights are initialized from scratch.
- **Line 110:** Train the indicated model with the supplied arguments and store its History in h, replacing the previous history reference.
- **Line 111:** Compute updates per epoch as the ceiling of training-set size divided by batch size.
- **Line 112:** Plot this batch-size run's epoch-level training loss and label its batch size and update count.
- **Line 113:** Set the figure title; inspect the observed curves rather than assuming a particular outcome.
- **Line 114:** Label the horizontal axis; history curves use zero-based plotted indices although the label says Epoch.
- **Line 115:** Label the vertical axis with the displayed loss quantity.
- **Line 116:** Display the legend using the labels assigned to plotted curves.
- **Line 117:** Show reference grid lines with opacity 0.3.
- **Line 118:** Adjust spacing to reduce overlapping labels and panels.
- **Line 119:** Save the figure under outputs/ at a resolution of 110 dots per inch.
- **Line 120:** Display the figure; interactive backends may pause the script until its window is closed.
- **Line 121:** Close the current figure to release plotting resources.

### L2 and L1 experiments

```python
m_l2 = build_model(hidden=BIG, l2=0.01)  # Loss + λ Σ w²
h = train(m_l2)
report("4a) L2 (λ=0.01)", m_l2)
plot_history(h, "4a) L2 Regularization", "4a_l2.png")

m_l1 = build_model(hidden=BIG, l1=0.001)  # Loss + λ Σ |w|
h = train(m_l1)
report("4b) L1 (λ=0.001)", m_l1)
plot_history(h, "4b) L1 Regularization", "4b_l1.png")
```

- **Line 124:** Build a fresh independent model with the architecture and hyperparameters shown; its weights are initialized from scratch.
- **Line 125:** Train the indicated model with the supplied arguments and store its History in h, replacing the previous history reference.
- **Line 126:** Compute, print, and record this model's training and validation prediction losses.
- **Line 127:** Plot this experiment's history using the given title and save the named PNG file.
- **Line 128:** Blank line for readability; it has no effect on execution.
- **Line 129:** Build a fresh independent model with the architecture and hyperparameters shown; its weights are initialized from scratch.
- **Line 130:** Train the indicated model with the supplied arguments and store its History in h, replacing the previous history reference.
- **Line 131:** Compute, print, and record this model's training and validation prediction losses.
- **Line 132:** Plot this experiment's history using the given title and save the named PNG file.

### Dropout experiment

```python
m_drop = build_model(hidden=BIG, dropout=0.3)  # Each hidden unit is dropped with probability 0.3 during training.
h = train(m_drop)
report("5) Dropout (0.3)", m_drop)
plot_history(h, "5) Dropout: dropout rate = 0.3", "5_dropout.png")
```

- **Line 135:** Build a fresh independent model with the architecture and hyperparameters shown; its weights are initialized from scratch.
- **Line 136:** Train the indicated model with the supplied arguments and store its History in h, replacing the previous history reference.
- **Line 137:** Compute, print, and record this model's training and validation prediction losses.
- **Line 138:** Plot this experiment's history using the given title and save the named PNG file.

### Early stopping

```python
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
```

- **Line 141:** Begin constructing the early-stopping callback.
- **Line 142:** Monitor validation loss to determine whether performance has improved.
- **Line 143:** Allow 20 consecutive epochs without improvement before stopping.
- **Line 144:** Restore the best observed weights in memory rather than leaving the last epoch's weights; no file is saved.
- **Line 145:** Close the preceding multi-line function call.
- **Line 146:** Build a fresh independent model with the architecture and hyperparameters shown; its weights are initialized from scratch.
- **Line 147:** Train the indicated model with the supplied arguments and store its History in h, replacing the previous history reference.
- **Line 148:** Find the minimum validation-loss position and convert its zero-based index to a one-based epoch number.
- **Line 149:** Print the specified text or metrics. Where used, .3f gives three decimal places, .1% formats a percentage, and + explicitly shows the sign of the gap.
- **Line 150:** Compute, print, and record this model's training and validation prediction losses.
- **Line 151:** Plot this experiment's history using the given title and save the named PNG file.

### Candidate final model

```python
m_good = build_model(hidden=(32, 32), dropout=0.2, l2=0.001)  # Moderate model capacity.
early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
h = train(m_good, epochs=300, batch_size=32, callbacks=[early])
report("7) Good model", m_good)
plot_history(h, "7) Good Fit: candidate final model", "7_good_model.png")
```

- **Line 154:** Build a fresh independent model with the architecture and hyperparameters shown; its weights are initialized from scratch.
- **Line 155:** Create early stopping for the final candidate with validation monitoring, patience 30, and restoration of the best weights.
- **Line 156:** Train the indicated model with the supplied arguments and store its History in h, replacing the previous history reference.
- **Line 157:** Compute, print, and record this model's training and validation prediction losses.
- **Line 158:** Plot this experiment's history using the given title and save the named PNG file.

### Summary and test evaluation

```python
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
```

- **Line 161:** Print the specified text or metrics. Where used, .3f gives three decimal places, .1% formats a percentage, and + explicitly shows the sign of the gap.
- **Line 162:** Print the specified text or metrics. Where used, .3f gives three decimal places, .1% formats a percentage, and + explicitly shows the sign of the gap.
- **Line 163:** Print the specified text or metrics. Where used, .3f gives three decimal places, .1% formats a percentage, and + explicitly shows the sign of the gap.
- **Line 164:** Iterate through saved comparison records and unpack the name and two losses.
- **Line 165:** Print the specified text or metrics. Where used, .3f gives three decimal places, .1% formats a percentage, and + explicitly shows the sign of the gap.
- **Line 166:** Print the specified text or metrics. Where used, .3f gives three decimal places, .1% formats a percentage, and + explicitly shows the sign of the gap.
- **Line 167:** Print qualitative interpretation hints; these are possible diagnoses, not definitive tests.
- **Line 168:** Blank line for readability; it has no effect on execution.
- **Line 169:** Evaluate the final candidate on the test set; total loss includes L2, and accuracy uses the binary decision threshold.
- **Line 170:** Compute test binary cross-entropy without L2 so its definition matches the comparison table.
- **Line 171:** Print the specified text or metrics. Where used, .3f gives three decimal places, .1% formats a percentage, and + explicitly shows the sign of the gap.

### Decision boundaries

```python
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
```

- **Line 174:** Create a 200-by-200 coordinate grid over the specified horizontal and vertical ranges.
- **Line 175:** Flatten and combine grid coordinates into 40,000 two-feature input points.
- **Line 176:** Blank line for readability; it has no effect on execution.
- **Line 177:** Create a figure with three side-by-side axes for comparing decision regions.
- **Line 178:** Pair each subplot with one of the three trained models and its display title.
- **Line 179:** Predict class-1 probabilities over the grid and reshape them to the coordinate-grid dimensions.
- **Line 180:** Color the regions below and above probability 0.5 using two fills, revealing the classification boundary.
- **Line 181:** Overlay training points using their two features as coordinates and their true labels as colors.
- **Line 182:** Assign the corresponding model title to this subplot.
- **Line 183:** Adjust spacing to reduce overlapping labels and panels.
- **Line 184:** Save the figure under outputs/ at a resolution of 110 dots per inch.
- **Line 185:** Display the figure; interactive backends may pause the script until its window is closed.
- **Line 186:** Close the explicitly referenced multi-panel figure.

## Suggested extensions

For a more rigorous study, record exact dependency versions, repeat experiments with independent seeds, share initial weights when comparing batch sizes, choose an explicit computational budget, export metrics to CSV, save trained models, and select a final model using validation data before the final independent test evaluation. These extensions are not implemented in this version.
