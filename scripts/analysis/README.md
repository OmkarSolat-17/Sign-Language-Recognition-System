# Accuracy analysis scripts

Three small, self-contained experiments that measure how accurate this system
actually is — and where the accuracy leaks away. They train the repo's own
model (`sign_language.model.build_model`) on a stand-in gesture alphabet, so
they run offline, in a couple of minutes each, on CPU.

```bash
pip install -r requirements.txt        # TensorFlow + scikit-learn are needed
python scripts/analysis/exp1_split_leakage.py
python scripts/analysis/exp1_split_leakage.py --small   # realistic small dataset
python scripts/analysis/exp2_crop_mismatch.py
python scripts/analysis/exp3_mirror_flip.py
```

| Script | Question | Headline result |
|---|---|---|
| `exp1_split_leakage.py` | Is the printed test accuracy trustworthy? | 100 images/class: honest. **25/class: +5.65 points too optimistic** |
| `exp2_crop_mismatch.py` | Does training match what the webcam feeds the model? | **9.56% live vs 93.56% when preprocessing is consistent** |
| `exp3_mirror_flip.py` | Is `--augment`'s horizontal flip safe? | **100% → 50% (chance)** on mirror-pair classes |

Full discussion, root-cause ranking and the dataset comparison:
[`docs/accuracy-analysis.md`](../../docs/accuracy-analysis.md).

These are diagnostics, not tests — they are deliberately excluded from the
pytest suite because they need TensorFlow and take minutes to run.
