"""Punto de entrada para `python -m app [collect|train|run|avatar]`."""

from app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

