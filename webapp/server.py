#!/usr/bin/env python3
"""Entry point kept at the historical path.

The application itself lives in the ``fermi_llm`` package; this file only
starts it, so ``python webapp/server.py`` and every existing deployment
script keep working.
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from fermi_llm.app import main  # noqa: E402

if __name__ == '__main__':
    main()
