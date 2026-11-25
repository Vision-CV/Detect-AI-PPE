#!/usr/bin/env python3
"""Запуск без установки пакета: `python main.py --source test/test2.mp4`.

После `pip install -e .` то же самое доступно как `ppe-monitor` или `python -m ppe_monitor`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ppe_monitor.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
