# BY MATEN/T | M/ License v1.0
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brand_guard import RULES, AttributionError, verify_attribution

class GuardTests(unittest.TestCase):
    def test_real_project_is_valid(self):
        self.assertTrue(verify_attribution(ROOT))

    def test_detects_removed_attribution(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in RULES:
                destination = folder / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
            target = folder / 'miniai.py'
            target.write_text(target.read_text(encoding='utf-8').replace('MATEN/T', 'REMOVED'), encoding='utf-8')
            with self.assertRaises(AttributionError):
                verify_attribution(folder)

    def test_detects_missing_license_file(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in RULES:
                dest = folder / name
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            (folder / 'LICENSE.md').unlink()
            with self.assertRaises(AttributionError):
                verify_attribution(folder)

if __name__ == '__main__': unittest.main()
