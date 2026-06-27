from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .campaign import run_campaign


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--all" not in argv:
        argv.append("--all")
    return run_campaign(argv)


if __name__ == "__main__":
    raise SystemExit(main())
