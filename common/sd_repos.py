"""
Locate + import the upstream speculative-decoding repos (EAGLE, Medusa).

The two repos live as git SUBMODULES under ``sd_repos/`` (see ``.gitmodules``),
pinned to the exact commits the collectors were written against:

    sd_repos/EAGLE   @ cb7e084   -> ``import eagle...``   (CAPIM draft stack)
    sd_repos/Medusa  @ e2a5d20   -> ``import medusa...``  (LP-Spec baseline stack)

They are NOT vendored into this tree: a fresh clone fetches them via
``git clone --recurse-submodules`` (or ``git submodule update --init``).  This
module resolves their on-disk location and puts them on ``sys.path`` so the
collectors can ``from eagle.model.ea_model import EaModel`` /
``from medusa.model.medusa_model import MedusaModel`` unchanged.

Design notes
------------
* GPU-free and torch-free: importing THIS module never imports the repos.  Path
  resolution + the "did you run --recurse-submodules?" guard are pure filesystem
  checks, so unit tests (e.g. the topK parity guard) can decide to SKIP without a
  GPU or the ML deps installed.
* The presence check keys on a *specific source file* inside each submodule
  (``eagle/model/cnets1.py`` / ``medusa/model/utils.py``), not just the directory:
  an un-initialised submodule leaves an empty directory, which the dir check alone
  would pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

# repo root = the directory that contains ``common/`` and ``sd_repos/``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SD_REPOS = _REPO_ROOT / "sd_repos"

# (submodule dir, sentinel file that must exist once initialised, import package)
_EAGLE = (_SD_REPOS / "EAGLE", "eagle/model/cnets1.py", "eagle")
_MEDUSA = (_SD_REPOS / "Medusa", "medusa/model/utils.py", "medusa")

_MISSING_MSG = (
    "speculative-decoding submodule not found at {path!s}.\n"
    "The EAGLE/Medusa repos are git submodules under sd_repos/ -- fetch them with:\n"
    "    git submodule update --init sd_repos/EAGLE sd_repos/Medusa\n"
    "(or clone the project with `git clone --recurse-submodules`)."
)


def _resolve(spec) -> Path:
    """Return the submodule root, raising a clear message if not initialised."""
    root, sentinel, _pkg = spec
    if not (root / sentinel).is_file():
        raise FileNotFoundError(_MISSING_MSG.format(path=root))
    return root


def eagle_path() -> Path:
    """Filesystem root of the EAGLE submodule (raises if uninitialised)."""
    return _resolve(_EAGLE)


def medusa_path() -> Path:
    """Filesystem root of the Medusa submodule (raises if uninitialised)."""
    return _resolve(_MEDUSA)


def _prepend(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def add_eagle_to_path() -> Path:
    """Put the EAGLE submodule on ``sys.path`` (idempotent); return its root."""
    root = eagle_path()
    _prepend(root)
    return root


def add_medusa_to_path() -> Path:
    """Put the Medusa submodule on ``sys.path`` (idempotent); return its root."""
    root = medusa_path()
    _prepend(root)
    return root
