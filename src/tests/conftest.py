"""Pytest configuration for src/tests.

Adds the ``src/`` directory to ``sys.path`` so that ``proxy_pointer_rag``
is importable without being an installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert src/ at the front of sys.path so proxy_pointer_rag is importable
_SRC = str(Path(__file__).parent.parent.resolve())
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
