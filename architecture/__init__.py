import os
import sys

# 方便从根目录或子目录导入本模块
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from .rrdb_variants import build_rrdb

