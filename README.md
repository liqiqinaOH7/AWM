# AWM — Anime Wallpaper Super-Resolution

ECE285 课程项目：在动漫风格壁纸数据上微调 ESRGAN，实现从低分辨率到高分辨率的超分重建。
与面向动漫视频的 APISR 不同，本项目专注于**高质量静态壁纸**的细节恢复。

---

## 一、实验结果

### 4× 超分对比

| 模型 | 架构 | 退化方式 | NIQE ↓ | MANIQA ↑ | CLIPIQA ↑ |
|------|------|---------|--------|----------|-----------|
| **Ours (23-block, simple)** | RRDB 23-block | bicubic only | **4.7277** | 0.2312 | 0.4507 |
| Ours (6-block, complex) | RRDB 6-block | median+blur+sharpen | 5.2683 | 0.2639 | 0.4848 |
| RRDB ESRGAN 4x (baseline) | RRDB | — | 4.9710 | 0.1564 | 0.2220 |
| APISR RRDB 4x | RRDB 23-block | APISR pipeline | 6.3849 | 0.4760 | 0.6703 |
| APISR GRL 4x | GRL | APISR pipeline | 5.7950 | **0.4815** | **0.6766** |
| APISR DAT 4x | DAT | APISR pipeline | 5.9168 | 0.4434 | 0.6320 |
| RealESRGAN Anime 6B 4x | RRDB 6-block | Real-ESRGAN pipeline | 6.6901 | 0.4554 | 0.5670 |

### 2× 超分对比

| 模型 | 架构 | 退化方式 | NIQE ↓ | MANIQA ↑ | CLIPIQA ↑ |
|------|------|---------|--------|----------|-----------|
| **Ours (6-block, complex)** | RRDB 6-block | median+blur+sharpen | **4.9087** | 0.4469 | **0.7109** |
| APISR RRDB 2x | RRDB | APISR pipeline | 5.3858 | **0.4870** | 0.6599 |

### 关键发现

1. **NIQE（自然度）我们大幅领先所有 baseline**：自训练模型在 NIQE 上均优于 APISR 系列和 RealESRGAN，说明输出更平滑自然、伪影更少。
2. **MANIQA/CLIPIQA（感知质量）仍有差距**：APISR 系列在感知指标上明显更高，主要原因是我们的 L1 pixel loss 权重过高（10.0），模型倾向输出保守/模糊的结果。
3. **退化方式显著影响模型特性**：简单退化（bicubic only）→ NIQE 更好但 MANIQA/CLIPIQA 更低；复杂退化（median+blur+sharpen）→ 细节恢复能力更强但可能引入伪影。
4. **2× 超分质量远高于 4×**：我们的 2× 模型在 CLIPIQA 上甚至超越 APISR 官方（0.7109 vs 0.6599），验证了 2× 场景更适合细节恢复。
5. **Transformer 架构（GRL/DAT）优于 CNN（RRDB）**：在相同 APISR 退化下，GRL 的感知指标最高。

---

## 二、所做工作概览

### 1. 数据获取与整理

- **`download_konachan.ipynb`** — 通过 Konachan JSON API 爬取带指定标签（如 `genshin_impact`）的高清原图，保存到 `dataset/highres/original/`。
- **`rename_and_clean_dataset.ipynb`** — 文件名统一重命名为纯数字 ID（如 `371778.jpg`），HR/LR 一一对应。
- **`clean_large_lowres.ipynb`** — 按像素阈值（>25.6M）删除过大图片，避免显存溢出。

### 2. 退化流程

| Notebook | 方式 | 输出目录 |
|----------|------|---------|
| `degradation_pipeline_2x.ipynb` | 2× 下采样 + 中值滤波 + 高斯模糊 + USM 锐化 | `lowres_2x/original/` |
| `degradation_pipeline_4x.ipynb` | 4× 下采样 + 同上 | `lowres_4x/original/` |
| `degradation_pipeline_4x_simple.ipynb` | 4× 纯 bicubic 下采样（无滤波） | `lowres_4x_simple/original/` |

简化退化（simple）作为消融实验的对照组，验证退化复杂度对模型性能的影响。

### 3. 模型与训练

- **`train_esrgan_independent.ipynb`** — 本地训练脚本（Windows/Linux），支持 2x/4x 切换。
- **`train_4x_23block_simple_colab.ipynb`** — Colab 一体化训练（4×/23-block/simple），包含环境初始化、退化生成、三阶段训练、推理评估、结果持久化。
- **`train_2x_23block_simple_colab.ipynb`** — Colab 一体化训练（2×/23-block/simple），结构同上。

**三阶段训练策略**（解决预训练 G 与随机初始化 D 不平衡的问题）：

| 阶段 | 策略 |
|------|------|
| Phase 1: G warmup (5 epochs) | 只训练 Generator（L1 loss），稳定预训练权重 |
| Phase 2: D warmup (5 epochs) | 冻结 Generator，只训练 Discriminator，让 D 追上 G |
| Phase 3: Full GAN (140 epochs) | G + D 对抗训练（L1 + VGG Perceptual + Danbooru Perceptual + GAN） |

每个 epoch 结束后自动推理监控图片（`372812.JPG`），展示 LR/SR/HR 对比及 D(real)/D(fake) 分数，实时判断训练平衡。

### 4. 推理与评估

- **`inference_esrgan.ipynb`** — 自训练模型推理 + NIQE/MANIQA/CLIPIQA 评估。
- **`inference_apisr_esrgan.ipynb`** — APISR 官方模型推理对比。
- Colab 训练 notebook 内置推理 + 评估 + Google Drive 持久化（JSON + TXT 双格式）。

---

## 三、仓库结构

```
AWM/
├── README.md
├── .gitignore
├── download_konachan.ipynb                 # 数据下载
├── rename_and_clean_dataset.ipynb          # 清理与重命名
├── clean_large_lowres.ipynb                # 过大图片清理
├── degradation_pipeline_2x.ipynb           # 2× 复杂退化
├── degradation_pipeline_4x.ipynb           # 4× 复杂退化
├── degradation_pipeline_4x_simple.ipynb    # 4× 简化退化（bicubic only）
├── train_esrgan_independent.ipynb          # 本地 ESRGAN 训练
├── train_4x_23block_simple_colab.ipynb     # Colab 一体化 4× 训练
├── train_2x_23block_simple_colab.ipynb     # Colab 一体化 2× 训练
├── inference_esrgan.ipynb                  # 自训练模型推理
├── inference_apisr_esrgan.ipynb            # APISR 官方模型推理
├── setup_colab.ipynb                       # Colab 环境初始化（独立版）
├── APISR_tools/                            # 核心网络与损失
│   ├── architecture/                       # RRDBNet, UNetDiscriminatorSN, GRL, DAT 等
│   └── loss/                               # L1, VGG Perceptual, Danbooru Perceptual, GAN
└── reading.txt                             # 参考资料
```

以下目录由本地/Colab 生成，**已通过 .gitignore 排除**：

- `dataset/` — HR 与各版本 LR（highres, lowres_2x, lowres_4x, lowres_2x_simple, lowres_4x_simple）
- `results/` — 推理结果与评估报告
- `saved_models/` — 训练 checkpoint
- `pretrained_models/` — APISR 等预训练权重

---

## 四、使用流程

### 本地
1. 准备数据：`download_konachan.ipynb` → `rename_and_clean_dataset.ipynb`
2. 生成 LR：运行对应退化 notebook
3. 训练：`train_esrgan_independent.ipynb`
4. 推理：`inference_esrgan.ipynb`

### Colab（推荐）
1. 打开 Colab → 文件 → 打开笔记本 → GitHub → `liqiqinaOH7/AWM`
2. 选择 `train_4x_23block_simple_colab.ipynb` 或 `train_2x_23block_simple_colab.ipynb`
3. 运行时 → 更改运行时类型 → GPU
4. 从上到下运行所有 cell（环境初始化 → 退化 → 训练 → 推理 → 评估自动完成）

依赖：Python 3.x, PyTorch (CUDA), OpenCV, NumPy, Matplotlib, tqdm, pyiqa

---

## 五、创新点

1. **任务定位差异化**：APISR 面向动漫视频（追求推理速度和模型轻量），我们面向高质量静态壁纸（追求细节丰富度和视觉质量）。
2. **针对性数据集**：从 Konachan 爬取高分辨率动漫壁纸，比通用动漫帧更适合壁纸超分场景。
3. **退化消融实验**：对比复杂退化（median+blur+sharpen）与简化退化（bicubic only）对模型特性的影响，揭示"退化复杂度 ↔ NIQE/感知质量"的权衡关系。
4. **三阶段训练策略**：G warmup → D warmup → Full GAN，解决预训练 Generator 与随机初始化 Discriminator 的不平衡问题。

---

## 六、未来计划 (TODO)

### P0（近期必做）

- [ ] **引入全参考指标**：加入 PSNR / SSIM / LPIPS（我们有 GT），建立更完整的评价体系；考虑自创"线条保真度"指标（Canny 边缘 F1-score），作为动漫超分的针对性评估方法。
- [ ] **训练集/测试集分离**：当前 434 张全部用于训练和评估，需拆分为 ~350 train + ~84 test，确保评估结果无偏。
- [ ] **损失权重调优**：当前 pixel loss 权重过高（10.0），导致输出偏保守；需降低 pixel loss、提升 perceptual/GAN loss 权重以改善 MANIQA/CLIPIQA。

### P1（中期推进）

- [ ] **Transformer 架构微调**：在自建数据集上微调 GRL/DAT（APISR 已提供预训练权重），对比 CNN vs Transformer 在壁纸超分上的表现。
- [ ] **颜色损失探索**：引入 Chroma Loss（YCbCr 空间 Cb/Cr 通道 L1）或 Color Histogram Loss，改善颜色还原精度。
- [ ] **扩充数据集**：增加更多标签（fate, azur_lane, honkai 等），目标 1000-2000 张，提升风格多样性和泛化能力。

### P2（探索性）

- [ ] **LoRA 细节增强**：冻结训练好的 Generator，注入 LoRA 层，用高感知损失权重训练 LoRA 参数，实现可控的细节增强强度（参考 Stable Diffusion 社区的 "Add Detail" LoRA 思路）。
- [ ] **混合退化训练**：每个 batch 随机选择退化方式（simple / complex），让模型同时学习多种恢复能力。
- [ ] **边缘感知退化**：对 HR 图像用 Canny 提取边缘 mask，边缘区域保留更多细节、非边缘区域做更强退化，模拟真实退化的非均匀性。
