"""
轻量封装的 RRDB 架构，用于统一 6-block / 23-block RRDB 以及 APISR 预训练模型的构建。

实际网络实现仍然复用 `APISR_tools/architecture/rrdb.RRDBNet`，本文件主要做两件事：
1. 提供一个统一的 `build_rrdb(scale, num_block)` 接口，便于在评估 notebook 中切换不同配置；
2. 避免在评估脚本里到处硬编码 APISR_tools 的路径。
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APISR_DIR = os.path.join(ROOT_DIR, "APISR_tools")

if APISR_DIR not in sys.path:
    sys.path.append(APISR_DIR)

from architecture.rrdb import RRDBNet  # 来自 APISR_tools/architecture/rrdb.py


def build_rrdb(scale: int = 4, num_block: int = 23, num_feat: int = 64):
    """
    构建 RRDBNet 模型。

    参数说明：
    - scale: 放大倍数（2 或 4），必须与训练/预训练模型一致；
    - num_block: RRDB block 数量：
        * 6  对应 APISR 默认 anime RRDB（轻量版）；
        * 23 对应 ESRGAN/APISR 标准 RRDB 架构；
    - num_feat: 中间通道数（默认 64，与 APISR 一致）。
    """
    if scale not in (1, 2, 4):
        raise ValueError(f"Unsupported scale={scale}, expected 1/2/4.")
    if num_block <= 0:
        raise ValueError(f"num_block 必须为正整数，当前为 {num_block}")

    # APISR 的 RRDBNet 构造函数为 RRDBNet(num_in_ch, num_out_ch, scale, num_feat=64, num_block=6, ...)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=num_feat, num_block=num_block)
    return model

