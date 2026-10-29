# BY MATEN/T | M/ License v1.0
"""Local attribution consistency check for MATEN/T MiniAI.

This is a source-level guard, NOT cryptographic copy protection: users with
access to the source code can remove or change these checks. It never deletes
files, changes models, alters checkpoints, or accesses the network.
"""
from __future__ import annotations
from pathlib import Path
import sys

OWNER = 'MATEN/T'
RULES = {
    'brand_guard.py': ('# BY MATEN/T', 'M/ License v1.0'),
    'miniai.py': ('# BY MATEN/T', 'MATEN/T MiniAI v1'),
    'train_full.py': ('# BY MATEN/T', 'MATEN/T MiniAI v1'),
    'model.py': ('# BY MATEN/T',),
    'build_datasets.py': ('# BY MATEN/T',),
    'catalog.py': ('# BY MATEN/T',),
    'facts.py': ('# BY MATEN/T',),
    'conversation_human.py': ('# BY MATEN/T',),
    'README.md': ('MATEN/T MiniAI v1', 'BY MATEN/T'),
    'LICENSE.md': ('M/ License', 'Maten Faesal (MATEN/T)'),
    'ui/index.html': ('<!-- BY MATEN/T', 'MATEN/T MiniAI'),
}

class AttributionError(RuntimeError):
    """A required source attribution notice is missing."""


def verify_attribution(root: Path | None = None) -> bool:
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    for name, notices in RULES.items():
        file = root / name
        try:
            contents = file.read_text(encoding='utf-8')
        except (OSError, UnicodeError) as ex:
            raise AttributionError(f'M/ License: required file unavailable: {name}') from ex
        if not all(notice in contents for notice in notices):
            raise AttributionError(
                f'M/ License: required MATEN/T attribution missing or changed in {name}. '
                'Removing the MATEN/T credit is not permitted for redistributed copies; restore the original notice to continue.'
            )
    return True


def enforce_attribution(root: Path | None = None) -> None:
    try:
        verify_attribution(root)
    except AttributionError as ex:
        raise SystemExit(str(ex)) from None


if __name__ == '__main__':
    enforce_attribution()
    print('BY MATEN/T | attribution verified')
