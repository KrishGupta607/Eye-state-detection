# Eye State & Driver Drowsiness Detection (EEG)

Two small ML projects built from raw EEG data, going from classical ML through to a
from-scratch PyTorch neural network. Built while working through Purdue IEEE SMC's
software team ML learning guide, using it as material to practice the full workflow:
load real data, clean it, split it properly, train multiple model types, and actually
understand every line well enough to explain it.

The goal was never a state-of-the-art model. It was learning the workflow end to end
and being able to defend every design choice.

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

~80–82% test accuracy. Not a fixed number — see Limitations, training isn't seeded.

## Project structure

- **`classical-ML.py`** — the eye_state pipeline: loads the 5 subject CSVs, cleans
  outliers, standard-scales, plots before/after, trains + evaluates Logistic
  Regression and Random Forest, writes `results.json`.
- **`eye_state_detection/data/`** — the 5 raw EEG CSVs (14 electrode channels +
  an `eye_state` label).
- **`eye_state_detection/main.py`** — an early, unfinished scaffold from before
  `classical-ML.py` existed. Superseded and unused; kept only as history.
- **`eye_state_detection/ML-proces.ipynb`** — an abandoned exploratory notebook
  (a partial PyTorch `Dataset` + an empty model on the eye_state data). Incomplete,
  not the real neural net — that's `sleepy_driver.ipynb`.
- **`sleepy_driver.ipynb`** — the actual finished PyTorch project: windows the
  driver-state dataset into 5-timestep sequences, trains a small MLP (BatchNorm +
  Dropout) to classify drowsy vs. alert.
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

- The full pipeline is actually implemented and verified end to end: raw CSV →
  cleaned/scaled features → multiple model types → real evaluation metrics.
- Correct ML hygiene in the places that matter: the scaler's mean/std are fit on
  train only, never on validation/test; splits use `stratify` (or a deliberately
  chosen subject) to keep class balance reasonable.
- The PyTorch model isn't a black box. Every piece — `Dataset`/`DataLoader`,
  `Flatten`/`BatchNorm1d`/`Linear`/`ReLU`/`Dropout`, the training loop,
  `BCEWithLogitsLoss`, the Adam optimizer — was reviewed line by line, and the
  reasoning behind each is understood, not just copied.

## Weaknesses & limitations (being honest)

- **Small, single-source datasets.** `eye_state_detection` has only 5 subjects.
  `sleepy_driver`'s windows slide by 1 timestep (stride 1), so adjacent windows are
  highly overlapping/near-duplicates of each other — reported accuracy likely
  overstates how well this would generalize to a genuinely new subject or session.
- **No true subject-holdout for eye_state's evaluation.** Per the source material's
  own suggestion, subject 1 is deliberately split across validation *and* test
  (rather than kept fully separate from the training subjects), specifically to get
  a more even class balance. In a strict "does this generalize to an unseen person"
  sense, that's a form of leakage — the same recording session contributes to both
  a fitted model and its evaluation.
- **Not reproducible run-to-run.** The sklearn splits are seeded (`random_state=42`),
  but `sleepy_driver.ipynb` never calls `torch.manual_seed(...)`, so model
  initialization and training are non-deterministic — the ~82% test accuracy will
  drift a bit each time the notebook is rerun.
- **No real hyperparameter tuning.** Random Forest and the MLP both use mostly
  first-guess settings (`hidden_size=16`, `lr=0.01`, `epochs=100`, `dropout=0.5`,
  default Random Forest params) chosen once, not searched.
- **The outlier threshold (250) in `classical-ML.py`** was picked by eyeballing
  plots, not any principled method. Fine for this exercise, not something to trust
  blindly on new data.
- **Loose ends / dead code, left in deliberately as an honest record:**
  - `eye_state_detection/main.py` — unfinished early scaffold, unused.
  - `eye_state_detection/ML-proces.ipynb` — abandoned, incomplete experiment,
    replaced by `sleepy_driver.ipynb`.
  - `sleepy_driver.ipynb` computes `test_results`/`report` at the end but never
    prints them — you have to add `print()` yourself to see the final numbers.
  - The `'loss'` returned by `evaluate_model()` is an un-averaged sum across
    batches, not directly comparable to the per-epoch *averaged* validation loss
    printed during training.
- **Simple architecture.** The MLP is 2 linear layers; no CNN/LSTM/attention was
  attempted, even though the sequential, time-windowed nature of the data is a
  natural fit for one.
- **No cross-validation.** A single train/valid/test split — one point estimate,
  not backed by variance across multiple splits.

## What I'd do differently

- Seed `torch` for reproducible neural net results.
- Use a genuine subject-holdout for eye_state evaluation, accepting imperfect class
  balance instead of splitting one subject across sets.
- Try a CNN or LSTM on the windowed `sleepy_driver` data, since it's real
  sequential data.
- Run an actual hyperparameter search instead of one-shot guesses.
- Clean up the superseded files (`main.py`, `ML-proces.ipynb`) instead of leaving
  them as history.
