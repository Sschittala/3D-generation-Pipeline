"""Dependency-free checks; real tensor numerical checks live in sdpa_compat.py."""
import os
import sys
import unittest
from unittest.mock import patch
from experiments.sketch_adaptation import sdpa_compat as adapter


class Offsets:
    def __init__(self,values): self.values=values
    def tolist(self): return self.values


class AdapterTests(unittest.TestCase):
    def test_valid_offsets(self):
        self.assertEqual(adapter.offsets(Offsets([0,3,3,5]),5,3),[0,3,3,5])

    def test_invalid_offsets(self):
        for offsets,total,maximum in [([1,3],3,3),([0,3,2],2,3),([0,5],5,3),([0,2],3,3),([0.,2.],2,3)]:
            with self.assertRaises(ValueError):adapter.offsets(Offsets(offsets),total,maximum)

    def test_install_is_explicit_and_does_not_require_torch(self):
        saved=sys.modules.get('flash_attn')
        try:
            sys.modules.pop('flash_attn',None)
            with patch.dict(os.environ,{},clear=False):
                adapter.install()
                self.assertEqual(os.environ['ATTN_BACKEND'],'sdpa')
                self.assertEqual(os.environ['SPARSE_ATTN_BACKEND'],'flash_attn')
                module=sys.modules['flash_attn']
                self.assertEqual(len([name for name in vars(module) if name.startswith('flash_attn_')]),6)
                with self.assertRaises(RuntimeError):adapter.install()
        finally:
            sys.modules.pop('flash_attn',None)
            if saved is not None:sys.modules['flash_attn']=saved


if __name__=='__main__':unittest.main()
