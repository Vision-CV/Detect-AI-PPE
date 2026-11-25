"""python -m ppe_receiver --port 8000 --data data/receiver"""

import argparse
import logging
from pathlib import Path

import uvicorn

from ppe_receiver.app import create_app


def main() -> None:
    p = argparse.ArgumentParser(prog="ppe-receiver", description="Приёмник пакетов инцидентов СИЗ")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data", type=Path, default=Path("data/receiver"), help="папка для БД и файлов")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args.data.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(args.data), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
