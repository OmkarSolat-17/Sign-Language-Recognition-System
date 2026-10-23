#!/usr/bin/env python3
"""Recognise sign language from a webcam, video file, or single image.

This is a thin, backwards-compatible entry point — the implementation
lives in :mod:`sign_language.inference`.  Run ``python predict.py --help``
for all options.

Examples::

    python predict.py                        # webcam (q / Esc to quit)
    python predict.py --source clip.mp4      # video file
    python predict.py --image photo.jpg      # single image, no camera

Expected failures (missing model/labels, unavailable camera, corrupt
image, ...) print a single actionable ``ERROR: ...`` line and exit with
status 1 instead of a traceback.
"""

import sys

from sign_language.inference import main

if __name__ == "__main__":
    sys.exit(main())
