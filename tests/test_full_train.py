# BY MATEN/T | M/ License v1.0
"""Lightweight checks for full-training launcher defaults."""
import sys
import tempfile
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from train_full import complete_data,parser

class FullTrainTests(unittest.TestCase):
    def test_defaults_request_full_epoch(self):
        args=parser().parse_args([])
        self.assertEqual(args.epochs,1)
        self.assertEqual(args.sequence_length,1024)
        self.assertEqual(args.batch_size,1)
        self.assertFalse(hasattr(args,'max_steps'))

    def test_no_incomplete_corpus_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(complete_data(Path(temp)))

if __name__=='__main__':
    unittest.main()
