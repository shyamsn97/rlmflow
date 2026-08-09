"""Allow ``python -m rlmflow`` alongside the installed ``rlmflow`` script."""

from rlmflow.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
