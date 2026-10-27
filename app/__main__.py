"""Punto de entrada para `python -m app [collect|train|run]`."""

from app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

