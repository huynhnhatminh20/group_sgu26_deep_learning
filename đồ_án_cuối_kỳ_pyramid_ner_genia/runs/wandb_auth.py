"""Automatic W&B credential bootstrap for local runs.

The plaintext key lives outside the repository at
``../.pyramid_private/wandb_api.md`` and stays git-ignored. This module makes
local notebooks/scripts pick it up without ``wandb login``:

* ``ensure_wandb_credentials()`` exports ``WANDB_API_KEY`` for the current
  process and, by default, refreshes the ``~/.netrc``/``~/_netrc`` entry so
  later processes authenticate on their own.
* ``read_api_key()`` only parses and returns the key.

On Kaggle the private file does not exist, so ``runs/wandb_tracking.py`` reads
the ``Wandb`` Kaggle Secret into ``WANDB_API_KEY``. The key itself is never printed or
logged, and nothing is written inside the repository.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

WANDB_HOST = "https://api.wandb.ai"
_NETRC_MACHINE = "api.wandb.ai"

_KEY_PATTERN = re.compile(r"wandb_v1_[A-Za-z0-9_-]+")
_FALLBACK_PATTERN = re.compile(r"[A-Za-z0-9_-]{40,}")
_KEY_FILENAMES = (
    Path(".pyramid_private") / "wandb_api.md",
    Path(".pyramid_private") / "wandb_api_key",
    Path(".pyramid_private") / "wandb_api.txt",
    Path(".wandb_api"),
    Path(".wandb_api_key"),
)


def _candidate_paths():
    """Yield plausible private key files, nearest locations first."""
    override = os.environ.get("PYRAMID_WANDB_KEY_FILE")
    if override:
        yield Path(override).expanduser()

    seen = set()
    roots = (Path.cwd(), Path(__file__).resolve().parents[1], Path.home())
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        for directory in (root, *root.parents):
            for name in _KEY_FILENAMES:
                path = directory / name
                if path not in seen:
                    seen.add(path)
                    yield path


def read_api_key(path=None):
    """Return the W&B API key from ``path`` or the first discoverable key file."""
    candidates = [Path(path).expanduser()] if path else list(_candidate_paths())
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = _KEY_PATTERN.search(text) or _FALLBACK_PATTERN.search(text)
        if match:
            return match.group(0)
    return None


def _netrc_path():
    override = os.environ.get("NETRC")
    if override:
        return Path(override).expanduser()
    for name in (".netrc", "_netrc"):
        candidate = Path.home() / name
        if candidate.exists():
            return candidate
    return Path.home() / (".netrc" if os.name != "nt" else "_netrc")


def _split(line):
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def read_netrc_key(path=None):
    """Return the API key already stored for ``api.wandb.ai`` in the netrc file."""
    path = Path(path) if path else _netrc_path()
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    in_entry = False
    for line in lines:
        tokens = _split(line)
        if tokens[:2] == ["machine", _NETRC_MACHINE]:
            in_entry = True
        elif tokens and tokens[0] in ("machine", "default"):
            in_entry = False
        elif in_entry and tokens[:1] == ["password"] and len(tokens) > 1:
            return tokens[1]
    return None


def write_netrc_key(api_key, path=None):
    """Insert or replace the ``api.wandb.ai`` entry, keeping other entries."""
    path = Path(path) if path else _netrc_path()
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        lines = []

    kept = []
    skipping = False
    for line in lines:
        tokens = _split(line)
        if tokens[:2] == ["machine", _NETRC_MACHINE]:
            skipping = True
        elif tokens and tokens[0] in ("machine", "default"):
            skipping = False
        if not skipping and line.strip():
            kept.append(line)
    kept += [f"machine {_NETRC_MACHINE}", "  login user", f"  password {api_key}"]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ensure_wandb_credentials(persist=True, path=None):
    """Make the local key available to ``wandb``; return it, or ``None``.

    ``persist=True`` refreshes the netrc entry only when it differs, so later
    processes (including ``python train_ner.py``) authenticate by themselves.
    """
    key = os.getenv("WANDB_API_KEY")
    if key:
        return key
    key = read_api_key(path)
    if not key:
        return None
    os.environ["WANDB_API_KEY"] = key
    if persist:
        try:
            if read_netrc_key() != key:
                write_netrc_key(key)
        except OSError:
            pass
    return key
