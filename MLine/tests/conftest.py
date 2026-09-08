"""Pytest bootstrap for MLine.

Ensures `MLine/` is on sys.path so `from src...` works
regardless of where pytest is invoked from (repo root or MLine/).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
