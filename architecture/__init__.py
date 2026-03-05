import os
import sys

# 方便从根目录或子目录导入本模块
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from .rrdb_variants import build_rrdb

# ViT-5 系列（依赖 timm、rope 等，按需使用）
try:
    from .models_vit5 import (
        vit_models,
        vit5_small,
        vit5_base,
        vit5_large,
        vit5_xlarge,
    )
except ImportError as e:
    vit_models = vit5_small = vit5_base = vit5_large = vit5_xlarge = None
    _vit5_import_error = e

