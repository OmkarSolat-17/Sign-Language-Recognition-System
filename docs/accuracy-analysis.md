# Accuracy analysis — does this system actually work?

**Scope:** an evidence-based audit of the recognition pipeline (model, training,
inference, data handling) and of the datasets available for it. Every number
below was produced by running the code in this repository; the scripts that
produce them are in [`scripts/analysis/`](../scripts/analysis/).

**Short answer:** the *engineering* is sound — it runs, it fails gracefully, and
it is tested. The *accuracy* is not what the tooling reports, and the live
webcam path is close to broken for the way most people will train it. Three
specific, fixable defects explain nearly all of the gap. There is also no
dataset in this repository, so "is the dataset perfect?" has a blunt answer:
there isn't one to be perfect yet.

---

## 1. Verdict at a glance

| Question | Verdict | Evidence |
|---|---|---|
| Does the code run correctly? | ✅ Yes — clean CLI, tested, graceful failures | 60-test suite, CI green |
| Is the printed test accuracy trustworthy? | ⚠️ No — it is tuned on itself | `exp1`: **+5.65 points** optimistic on small data |
| Does the live webcam path match training? | ❌ No — different preprocessing | `exp2`: **9.56%** live vs **93.56%** consistent |
| Is `--augment` safe for all sign sets? | ❌ No — horizontal flip destroys mirror-sensitive classes | `exp3`: **100% → 50%** (chance) |
| Is the bundled dataset good? | ⚠️ None is bundled | `dataset/` holds a README and `.gitkeep` |
| Is "perfect accuracy" achievable? | ❌ No — no single-frame model can do it | See §5 |

**The one-sentence version:** this is a good *program* wrapped around a model
that is being measured wrong, fed differently at inference than at training,
and given no mechanism to generalise beyond the webcam it was trained on.

---

## 2. How this was measured (and the honest limits)

No dataset ships with this repository, so the experiments train the repo's real
model — `sign_language.model.build_model`, unmodified — on a **stand-in gesture
alphabet**: scikit-learn's 10-class handwritten-symbol set, upscaled to the
repo's default 64×64 and replicated to 3 channels.

That is a proxy, and proxies prove *mechanism*, not final figures. What carries
over from the proxy to your real data:

* the split logic, the early-stopping logic and the augmentation layer are the
  repo's own code, not reimplementations;
* the failure modes measured are structural (a resize done in one place and not
  the other; an accuracy measured on the set that selected the model; a flip
  that maps one class onto another), so they appear for any image dataset;
* the *magnitudes* will differ with your data — the direction will not.

What the proxy cannot tell you is the accuracy you will get on real signs. For
that, §5 lists published results on real corpora.

---

## 3. The three defects, with measured numbers

### 3.1 The reported accuracy is tuned on the test set

`sign_language/training.py` makes one split and uses it twice:

```python
X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=config.test_size, ...)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),   # (1) test set guides early stopping
    callbacks=callbacks,
)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)  # (2) measured on the same set
```

`EarlyStopping(monitor="val_loss", restore_best_weights=True)` keeps the epoch
that scored best **on the test set**, and the accuracy printed afterwards is
computed on those same images. The metric is optimised and then reported, so it
is biased upwards — by how much depends on how hard the model is struggling.

Measured with `exp1_split_leakage.py` (3 seeds each):

| Training set size | Reported by repo | Honest held-out | Bias |
|---|---|---|---|
| ~180 images/class | 98.52% | 98.40% | +0.12 pts |
| **25 images/class** | **96.00%** | **90.35%** | **+5.65 pts** |

Individual small-data runs were worse than the average: one seed printed
**100.00%** where the honest number was **97.37%**, another **96.00%** vs
**86.84%**.

This matters because 25 images per class is *normal* for a hand-collected sign
set. When you see `Test accuracy: 100.00%` from a few hundred photos, part of
that number is the pipeline grading its own homework.

### 3.2 Training and inference preprocess images differently — this is the big one

Two different preprocessing paths exist for the same model:

| Stage | Code | Operation |
|---|---|---|
| Training | `data.py::load_dataset` | `cv2.resize(whole_image, (64, 64))` |
| Inference | `inference.py::run_stream` | crop MediaPipe hand box + padding, *then* resize to 64×64 |

Training never uses MediaPipe. So the model learns whatever framing your
dataset images happen to have, and at inference it is handed a tight hand crop
— a different zoom level, different background, different aspect ratio. The
classes are the same; the *input distribution* is not.

Measured with `exp2_crop_mismatch.py` (symbol placed in a 160×160 frame, ~9% of
the canvas, MediaPipe-style tight crop at inference):

| Trained on | Evaluated on | Accuracy |
|---|---|---|
| full frames | full frames | 58.44% — *the number `train.py` prints* |
| **full frames** | **hand crops** | **9.56%** — ***the live webcam path*** |
| hand crops | hand crops | 93.56% — consistent preprocessing |

Chance for 10 classes is 10%. **The live path scored below chance.** Training
on frames and cropping at inference does not degrade accuracy, it destroys it:
48.89 points, the difference between a demo and a random-number generator.

The asymmetry explains a lot of the confusion people hit with projects like
this: the offline evaluation looks fine, the webcam looks broken.

### 3.3 `--augment` includes a horizontal flip, which can be fatal

`model.py` adds three augmentation layers under `--augment`:

```python
layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomZoom(0.1)
```

Rotation and zoom are safe. Horizontal flip is not, for a hand-shape alphabet:
it converts a right hand into a left hand and a leftward sign into a rightward
one. If two of your classes differ by handedness or direction, the flip makes
them the same image for half the training samples.

Measured with `exp3_mirror_flip.py` on a 2-class set — a symbol and its exact
mirror, i.e. the `p`/`q` situation:

| Setting | Accuracy |
|---|---|
| no flip | **100.00%** |
| `RandomFlip("horizontal")` | **50.00%** — exactly chance |

A flip-augmented alphabet containing a mirrored pair has a hard ceiling of 50%
on that pair, and the damage spreads to neighbouring classes through the shared
feature layers.

---

## 4. Root causes, ranked by how much accuracy they cost

| # | Issue | Where | Impact | Fix difficulty |
|---|---|---|---|---|
| 1 | Train/inference preprocessing mismatch | `data.py`, `inference.py` | **~49 pts** | Medium |
| 2 | Test set doubles as validation set | `training.py:62,71` | +6 pts of fiction | **Trivial** |
| 3 | No transfer learning — 3-conv CNN trained from scratch | `model.py` | Large on small data | Small |
| 4 | Pixel classification instead of hand landmarks | `model.py`, `data.py` | Large on unseen users/rooms | Medium–large |
| 5 | Horizontal flip in `--augment` | `model.py:70` | 0–50 pts depending on alphabet | **Trivial** |
| 6 | Single frame only — no time dimension | whole pipeline | Cannot do J, Z, or any word sign | Large |
| 7 | Square resize ignores aspect ratio | `data.py:220`, `predictor.py:55` | Small but free to fix | Trivial |
| 8 | No per-class metrics, confusion matrix or class weights | `training.py` | 0 directly — but you cannot *see* the problem | Small |
| 9 | Always answers with a known class (no "unknown") | `predictor.py` | Wrong labels shown confidently | Small |

Two of these (2 and 5) are one-line fixes. Two more (1 and 7) are one shared
preprocessing function used by both paths. None require rewriting the project —
but item 4 does, and item 6 is a different project.

---

## 5. Is the dataset good? Is there a better one?

### 5.1 There is no dataset in this repository

`dataset/` contains a `README.md` and a `.gitkeep`. The original scripts
pointed at a hard-coded `C:\Users\omkar\...` path, so the data lived outside the
repo and is not recoverable from it. **"Is the dataset used perfect?" — there is
no dataset in this codebase, so there is currently nothing to measure.**

Whatever you train on is the single largest determinant of accuracy: more than
the architecture, more than the optimiser. Here is the honest landscape.

### 5.2 The famous "perfect accuracy" dataset is a trap

The most-copied choice is the Kaggle **ASL Alphabet** set (87,000 images,
29 classes — A–Z plus `space`, `del`, `nothing`). Projects routinely report
99–100% on it. That number is an artefact:

* the 87,000 images are **consecutive frames of video** — frame 1041 lands in
  train and frame 1042 in test, differing by a few pixels of hand tremor;
* an independent signer-honest audit found **99.14% of the test set had a
  near-duplicate in train**, and a random split scores **1.0000**;
* keeping whole capture sessions together instead (**group split**) drops it to
  **63.18%**, and testing on a corpus shot by *different people* drops it to
  **60.57%** [1](https://github.com/Vijaycode2007/signer-honest-asl).

So: 100% on paper, ~61% on a new person's webcam. That gap is the entire
difference between a portfolio screenshot and a working system. If you use this
set, split by session, never at random, and expect ~60–65% across signers —
which, for reference, is still far better than the 9.56% the current pipeline
delivers live.

It also contains **J and Z**, which in ASL fingerspelling are *motion* letters.
Static images of them are ill-defined, so those two classes are intrinsically
noisy and will be confused with their neighbours no matter what you do.

### 5.3 Datasets worth your time

| Dataset | Language / task | Size | Fits this codebase? | Notes |
|---|---|---|---|---|
| **ASL Alphabet** (Kaggle `grassknoted/asl-alphabet`) | ASL static letters | 87k images, 29 classes, 200×200 | ✅ Yes, directly | Beware the leakage (§5.2). J/Z are noise. Best for a *static letters* demo. |
| **Sign Language MNIST** (Kaggle) | ASL letters, no J/Z | 34,627 train / 7,172 test, 28×28 gray | ✅ Yes | Easy sanity baseline, but 28×28 grey is unrealistic for webcam use. |
| **INCLUDE** (AI4Bharat, IIT Madras) | **Indian** SL, words | 4,287 videos, 263 signs, 15 categories; `INCLUDE-50` subset | ❌ Needs a temporal model | Published baselines: **85.6%** on 263 classes, **94.5%** on INCLUDE-50. The credible ISL choice. |
| **ISL-CSLTR** | **Indian** SL, letters + digits | ~1k+ videos | ❌ Temporal | ISL alphabet, video-based. |
| **WLASL** | ASL, 2000 word signs | ~21k videos, 100+ signers | ❌ Temporal | Signer-disjoint splits — honest evaluation by design. |
| **AUTSL** | Turkish SL, isolated signs | 226 signs, ~36k videos | ❌ Temporal | Clean isolated-sign benchmark. |
| **PHOENIX-2014T** | German SL, continuous + translation | ~8k sentences | ❌ Sequence-to-sequence | Standard for continuous SLR/translation. |
| **CSL-Daily / How2Sign** | Chinese / ASL, continuous | 20k+ / 35h | ❌ Sequence-to-sequence | Research-grade continuous translation. |

**If your goal is Indian Sign Language** — likely, given the context — then
**INCLUDE** is the dataset to build toward, and it forces a design decision
this repository cannot currently express: it is *video*, so a single-frame
classifier is the wrong model. The realistic path is MediaPipe Holistic
landmarks (hands + pose + face) → LSTM or Transformer over ~30 frames. That
architecture also generalises far better across signers, because landmarks are
invariant to room, lighting and skin tone in a way pixels are not.

**If your goal is a defensible static-letters demo**, then ASL Alphabet *with a
session-grouped split* plus transfer learning is the honest, achievable option.

**If your goal is your own vocabulary**, then recording it yourself is the best
data you can get — see §6.

### 5.4 What "good data" means concretely

The fastest way to a high number is not a bigger model, it is better data. For
each sign, record:

* **200–300 images**, not 25 — the leakage experiment shows small data is where
  measurement and reality diverge fastest;
* **5+ separate sessions**, different day/position/lighting — this is what makes
  a *group split* meaningful and what prevents the "99% offline, 60% live" gap;
* **2+ different people**, if you can — signer diversity is the single strongest
  predictor of cross-user accuracy;
* **the same framing the webcam will produce** — record at arm's length, full
  frame, then let the *pipeline* crop. Never mix hand-cropped training images
  with full-frame inference images, or vice versa;
* **an "unknown" class** — 100–200 images of random gestures and empty frames,
  so the model has somewhere to put a hand it does not recognise instead of
  confidently naming the nearest known sign;
* **consistent handedness policy** — decide left, right, or both, and document
  it. A model trained on right hands only will mislabel left hands.

### 5.5 Realistic accuracy targets

"Perfect accuracy" is not a reachable target; treat any 100% as a measured bug
until proven otherwise. Realistic figures:

| Setting | Realistic accuracy |
|---|---|
| Held-out images, same sessions as training, random split | 98–100% — **and meaningless** |
| Held-out images, session-grouped split | 90–97% (static letters, transfer learning) |
| Held-out **signers**, static letters, full pipeline | 85–95% |
| Cross-corpus (different people, cameras, rooms) | 60–75% for pixel CNNs; notably higher for landmark models |
| Published, INCLUDE-50 word signs (263 classes: 85.6%) | 94.5% on the 50-class subset |
| Dynamic signs (J, Z, any word) with this static model | **Not achievable — by construction** |

---

## 6. What to fix, in order

### P0 — correctness (hours, not days)

1. **Three-way split.** Add a `validation_size` and split
   train/validation/test; early-stop on validation, report test. Kills the
   +5.65-point fiction and costs ~10 lines.
2. **One preprocessing function, used by both paths.** A single
   `preprocess_for_model(image_or_path, img_size)` that every caller (dataset
   loader, `--image`, webcam crop) goes through, plus aspect-ratio-preserving
   letterboxing instead of a square stretch. This is defect #1, the 49-point one.
3. **Make horizontal flip opt-in** (`--flip`, default off) and document that it
   is invalid for handedness-sensitive alphabets. One line.
4. **Report per-class precision/recall and a confusion matrix** to
   `training_metrics.json`, plus macro-F1. You cannot improve what you cannot
   see, and the confused pairs tell you exactly which signs need more data.

### P1 — accuracy (days)

5. **Transfer learning.** Swap the from-scratch CNN for a frozen-then-fine-tuned
   `MobileNetV2` / `EfficientNetB0` at 128–224px. On a few hundred images per
   class this is typically worth **10–25 points** over a 3-conv scratch net, for
   roughly the same training time.
6. **Train through the hand detector.** Run MediaPipe over the dataset once,
   cache the crops, and train on exactly what inference will see. This makes
   defect #1 structurally impossible rather than merely fixed.
7. **Class weights + an `unknown` class**, so imbalance and out-of-vocabulary
   gestures do not silently become confident wrong answers.
8. **Calibrate the confidence threshold** on the validation set instead of
   leaving the `(?)` marker at an arbitrary 0.5.

### P2 — only if you need word signs or motion letters

9. **Landmark + temporal model.** MediaPipe Holistic keypoints → LSTM/Transformer
   over a frame window. Required for J, Z, and every word sign; also the
   architecture that generalises across signers.
10. **Video dataset** (INCLUDE-50 for ISL, WLASL for ASL) with signer-disjoint
    splits.

---

## 7. Bottom line

The repository is a well-built shell around a model that has never been given a
fair test. It does not have "perfect accuracy" — it has a measurement that
hides its own error, a training path that mismatches its inference path, and no
data. None of those are mysterious or expensive to fix: items 1–4 in §6 are
hours of work and recover most of the lost accuracy, and they turn the printed
number from a fiction into something you can defend in an interview or a demo.

The honest framing for a portfolio or report: *"95% on session-held-out data,
~85% on new signers"* is a far stronger claim than *"100% accuracy"* — because
the first one is true, and anyone who has built one of these will know the
second one is not.

---

## Reproducing

```bash
pip install -r requirements.txt
python scripts/analysis/exp1_split_leakage.py            # split leakage
python scripts/analysis/exp1_split_leakage.py --small    # realistic small dataset
python scripts/analysis/exp2_crop_mismatch.py            # train vs inference mismatch
python scripts/analysis/exp3_mirror_flip.py              # flip augmentation hazard
```

Reference: [1] signer-honest ASL audit — https://github.com/Vijaycode2007/signer-honest-asl
