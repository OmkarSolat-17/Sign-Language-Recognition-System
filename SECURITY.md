# Security policy

## Supported versions

Fixes are applied to the `main` branch; please use the latest code.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Use
GitHub's private vulnerability reporting ("Report a vulnerability"
under the repository's Security tab), or contact the maintainer through
GitHub. You can expect an acknowledgement within a few days.

## Scope notes for this project

- **Model files are code-like artefacts.** Keras/TensorFlow models
  (`.keras`, `.h5`) can execute arbitrary code when loaded. Only load
  models you trained yourself or obtained from a source you trust —
  the same applies to any `labels.json` you pair with them.
- **Camera access** is requested only while `predict.py` is running;
  frames are processed locally and never uploaded anywhere.
- **Datasets** downloaded from third parties should be treated as
  untrusted input — the loader already skips undecodable files, but
  check licences before redistributing images.
