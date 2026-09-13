"""YAML config loader.

Returns a flat dict suitable for shell expansion into ``--flag value`` pairs
consumed by the argparse in ``train.py`` / eval entrypoints.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def _coerce_numeric(v: Any) -> Any:
    """Best-effort: convert numeric-looking strings to int/float.

    PyYAML 1.1 treats "3e-4" (no decimal point) as a string, which trips users.
    We try int first, then float; on failure return the value unchanged.
    """
    if not isinstance(v, str):
        return v
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file. Returns a flat dict (no nesting allowed).

    Numeric-looking strings (e.g. ``"3e-4"``) are coerced to int/float so users
    don't have to remember PyYAML 1.1's strict scientific-notation grammar.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping at top level: {p}")
    for k, v in cfg.items():
        if isinstance(v, (dict, list)):
            raise ValueError(
                f"Config must be flat (no nested dicts/lists). Offending key: {k!r}"
            )
    return {k: _coerce_numeric(v) for k, v in cfg.items()}


def to_cli_args(cfg: dict[str, Any]) -> list[str]:
    """Convert flat config dict to a list of CLI args ``['--key', 'value', ...]``.

    Behavior:
    - ``True`` → just ``--key`` (assumes argparse uses ``store_true``).
    - ``False`` → omitted entirely.
    - Anything else → ``--key str(value)``.
    - Strings with spaces (e.g. ``"A B C D"``) are passed as a single arg.
      The caller is responsible for any further shell quoting.
    """
    args: list[str] = []
    for k, v in cfg.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(flag)
        else:
            args.extend([flag, str(v)])
    return args
