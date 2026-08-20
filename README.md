# Eye State & Driver Drowsiness Detection (EEG)

Two ML projects built end to end from raw EEG data: one using classical ML, one
using a neural network built from scratch in PyTorch. The focus was the full
workflow — loading real data, cleaning and preprocessing it properly, splitting it
without leaking information between sets, training and comparing multiple model
types, and evaluating them honestly.

## What's here

1. **`eye_state_detection/`** — classify eyes open (1) vs. closed (2) from 14-channel
   EEG headset data, using classical ML (Logistic Regression, Random Forest).
2. **`sleepy_driver.ipynb`** — classify a driver as drowsy vs. alert from a separate
   EEG-derived dataset (attention/meditation + frequency-band power), using a custom
   PyTorch MLP trained on 5-timestep sliding windows.

## Results

### Eye state — classical ML (`classical-ML.py` → `results.json`)

| Model | Validation | Test |
|---|---|---|
| Logistic Regression | 74.8% | 66.8% |
| Random Forest | 73.3% | 67.2% |

### Driver drowsiness — PyTorch MLP (`sleepy_driver.ipynb`)

~80–82% test accuracy (varies slightly run to run — see notes below).

## Project structure

- **`classical-ML.py`** — the eye_state pipeline: loads the 5 subject CSVs, cleans
  outliers, standard-scales, plots before/after, trains + evaluates Logistic
  Regression and Random Forest, writes `results.json`.
- **`eye_state_detection/data/`** — the 5 raw EEG CSVs (14 electrode channels +
  an `eye_state` label).
- **`sleepy_driver.ipynb`** — the PyTorch project: windows the driver-state dataset
  into 5-timestep sequences, then trains a small MLP (BatchNorm + Dropout) to
  classify drowsy vs. alert, with a custom `Dataset`/`DataLoader`, training loop,
  and evaluation with `classification_report`.
- **`sleepy_driver/data/acquiredDataset.csv`** — the driver-attention dataset.
- **`setup_test.py`, `pyproject.toml`, `uv.lock`, `.python-version`** — environment
  setup, managed with `uv`.
- **`results.json`, `*.png`** — generated artifacts from running `classical-ML.py`.

## Running it

```bash
uv sync
uv run -- python setup_test.py    # sanity check: should print "Congratulations!"
uv run -- python classical-ML.py  # eye_state pipeline -> results.json + plots
```

Then open `sleepy_driver.ipynb` in Jupyter/VS Code and run all cells for the
neural net project.

## Strengths

- The full pipeline is implemented and verified end to end: raw CSV → cleaned and
  scaled features → multiple model types → real evaluation metrics, for both a
  classical ML approach and a from-scratch neural network.
- Correct ML hygiene in the places that matter most: the scaler's mean/std are fit
  on train only, never on validation/test, and splits use `stratify` (or a
  deliberately chosen subject) to keep class balance reasonable rather than relying
  on luck.
- The PyTorch model isn't a black box built by trial and error. Every piece —
  `Dataset`/`DataLoader`, `Flatten`/`BatchNorm1d`/`Linear`/`ReLU`/`Dropout`, the
  training loop, `BCEWithLogitsLoss`, the Adam optimizer — was gone through line by
  line, so the reasoning behind each design choice is understood, not just copied.
- Both projects hit accuracy well above chance (50% for a binary problem), with the
  neural network reaching ~82% on genuinely held-out test data.

## Limitations and honest caveats

- **Small, single-source datasets.** `eye_state_detection` has only 5 subjects, and
  `sleepy_driver`'s windows slide by 1 timestep, so adjacent windows overlap
  heavily. Reported accuracy is a fair measure on this data, but likely optimistic
  as an estimate of how well this generalizes to a genuinely new subject or session.
- **Subject 1 is split across validation and test in `eye_state_detection`**,
  rather than held out entirely from the training subjects — a deliberate choice to
  get a more even class balance in those sets, but it does mean that subject's
  recording session contributes to both a fitted model and its evaluation.
- **Neural net results aren't seeded**, so `sleepy_driver.ipynb`'s accuracy shifts a
  few points between runs — the sklearn data splits are reproducible
  (`random_state=42`), but model initialization and training are not.
- **Hyperparameters were chosen once, not tuned.** Random Forest and the MLP both
  use reasonable first-guess settings rather than a systematic search.
- **Architecture is intentionally simple** — a 2-layer MLP, not a CNN/LSTM — even
  though the sequential, time-windowed nature of the driver-state data would be a
  good fit for one.
- **No cross-validation.** Results come from a single train/valid/test split rather
  than an average across multiple splits.

## What I'd do next

- Seed `torch` for reproducible neural net results.
- Use a genuine subject-holdout for eye_state evaluation.
- Try a CNN or LSTM on the windowed `sleepy_driver` data.
- Run an actual hyperparameter search instead of one-shot settings.
