# BY MATEN/T | MATEN/T MiniAI v1.1 | M/ License v1.0
"""Sanity tests for the one-command causal-pretraining launcher."""
import sys
import tempfile
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_full import full_dataset
from pretrain import parser as pretrain_parser
from model import ModelConfig

class FullTrainTests(unittest.TestCase):
    def test_uncapped_pretraining_default(self):
        args=pretrain_parser().parse_args([])
        self.assertEqual(args.epochs,1)
        self.assertEqual(args.max_steps,0)
        self.assertEqual(args.sequence_length,512)
        self.assertEqual(ModelConfig().context_length,6144)
    def test_partial_data_not_accepted_as_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(full_dataset(Path(folder)))

if __name__=='__main__':unittest.main()
