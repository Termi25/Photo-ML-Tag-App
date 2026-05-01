"""Module-level constants shared by all ml/ sub-modules.

Kept in a leaf module (no imports from the ml package) to prevent circular
imports when ml/__init__.py imports from sub-modules that need these constants.
"""

import platform
import warnings

import torch
from PIL import Image

# ── AMP ──────────────────────────────────────────────────────────────────────
# torch.amp.autocast / GradScaler available since PyTorch 2.0.
# Falls back gracefully on older versions or CPU-only builds.
AMP_AVAILABLE: bool  = hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast')
_AMP_AVAILABLE       = AMP_AVAILABLE   # legacy alias

# ── DataLoader workers ────────────────────────────────────────────────────────
# Windows uses spawn for worker processes which deadlocks inside Tkinter threads.
SAFE_WORKERS: int = 0 if platform.system() == 'Windows' else 2
_SAFE_WORKERS     = SAFE_WORKERS       # legacy alias

# ── Matplotlib (optional) ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')              # non-interactive, safe in background threads
    import matplotlib.pyplot as _plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    _plt = None                        # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False

_MATPLOTLIB_AVAILABLE = MATPLOTLIB_AVAILABLE  # legacy alias

# ── PIL global settings ───────────────────────────────────────────────────────
Image.MAX_IMAGE_PIXELS = 200_000_000
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
warnings.filterwarnings('ignore', message='Palette images with Transparency')
