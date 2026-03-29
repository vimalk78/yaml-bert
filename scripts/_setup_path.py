"""Add project root to sys.path. Import this before any yaml_bert imports."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
