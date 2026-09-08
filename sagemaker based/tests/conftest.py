"""Pytest bootstrap for AutoML Studio (sagemaker based).

Ensures `sagemaker based/` is on sys.path so `from config ...`
and `from src...` work from the repo root.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
