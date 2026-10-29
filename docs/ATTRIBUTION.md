# MATEN/T Attribution Guard — v1

**BY MATEN/T · M/ License v1.0**

This optional source-level product protection checks the notices in the main
Python files, README files, LICENSE.md and UI before running the main CLI or
dataset generator. It stops cleanly if a required notice is missing.

## Try it

```shell
python brand_guard.py
python miniai.py info
python -m unittest discover -s tests -v
```

## Demonstrate a failure without altering your project

The test `tests/test_brand_guard.py` creates a temporary copy of only the
required text files, removes the name from a copied file, and confirms the
check raises an AttributionError. It does not modify the real training data.

## Boundaries

- Source-access users can remove this check; no local Python check is
  tamper-proof. It is a visible, non-obfuscated attribution guard.
- The guard never deletes or changes files, contacts a server, or reads
  private user data.
- The custom M/ License is not OSI-approved or legally vetted.
- Existing licenses of external packages remain in force.
