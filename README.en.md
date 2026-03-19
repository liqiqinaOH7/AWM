# AWM / AWMSR — Anime Wallpaper Master Super-Resolution

**Language**: [简体中文](README.md) | **English**

[![HF Models](https://img.shields.io/static/v1?label=Models&message=HuggingFace&color=orange)](https://huggingface.co/liqiqinaOH7/AWMSR/tree/main)
&ensp;
[![HF Dataset](https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=orange)](https://huggingface.co/datasets/liqiqinaOH7/AWM)

🔥 [Update](#update) **|** 👀 [Visualization](#visualization) **|** 🔧 [Installation](#installation) **|** 🏰 [Model Zoo](#model-zoo) **|** ⚡ [Inference](#inference) **|** 🧩 [Dataset Curation](#dataset-curation) **|** 💻 [Train](#train) **|** 📊 [Results](#results)

---

## Overview

**AWMSR (Anime Wallpaper Master Super-Resolution)** studies **single-image super-resolution for high-quality static anime wallpapers**. Compared to animation-frame restoration, wallpapers typically contain richer high-frequency details (e.g., accessories, highlights, particle-like effects) and are often degraded by **soft real-world processes** such as resizing, blur, and web compression.

This repository provides a wallpaper-oriented SR pipeline:

- **Dataset**: a Konachan-based high-resolution wallpaper set (1,481 images) with a fixed train/test split
- **Degradation**: a wallpaper-oriented synthetic degradation recipe (median → blur → downsample → USM → optional JPEG)
- **Models**: ESRGAN-style training with two generator families
  - **AWMSR RRDB** (RRDBNet / ESRGAN family)
  - **AWM GRL** (transformer-based GRL restoration backbone)
- **Evaluation**:
  - **No-reference**: NIQE (↓), MANIQA (↑), CLIP-IQA (↑)
  - **Full-reference**: PSNR (↑), SSIM (↑), LPIPS (↓), DISTS (↓), Edge F1 (↑)

---

## Update

- **2026-03**: Uploaded pretrained checkpoints and dataset to Hugging Face.
  - Models: `AWM_RRDB_2x.pth`, `AWM_GRL_2x.pth`, `AWM_RRDB_4x.pth`
  - Dataset: `AWM` (1.48k images; train 1.05k / test 434)

---

## Visualization

Motivation (wallpapers vs. animation frames):

![Wallpaper vs Animation Frames](figures/pip_comparison.jpg)

Wallpaper-oriented degradation pipeline:

![Degradation pipeline](figures/degration.png)

RRDB generator architecture used in this project:

![RRDB architecture](figures/RRDB.png)

Qualitative comparisons in the \(2\times\) setting:

![Qualitative multi](figures/qualitative_multi.png)

---

## Installation

This repo is notebook-centric. A typical environment includes:

- Python 3.x
- PyTorch (CUDA recommended)
- NumPy / OpenCV / Pillow / tqdm
- IQA libraries for NIQE / MANIQA / CLIP-IQA (e.g., `pyiqa`)

Notes:

- If you run the evaluation notebooks, ensure the corresponding IQA and full-reference metric dependencies are installed.
- Exact requirements may vary by notebook; follow the import cells in each notebook.

---

## Model Zoo

### Pretrained models (Hugging Face)

- **Download page**: https://huggingface.co/liqiqinaOH7/AWMSR/tree/main
- **Files**:
  - `AWM_RRDB_2x.pth`
  - `AWM_GRL_2x.pth`
  - `AWM_RRDB_4x.pth`
- **Recommended placement**: `pretrained_models/` (or your configured checkpoint directory in notebooks)

### Model naming in this repo

- **AWMSR RRDB**: RRDBNet-based ESRGAN-style generator (our main model family)
- **AWM GRL**: GRL-based transformer generator (alternative backbone under the same pipeline)

---

## Inference

Typical workflow:

- Run `inference_esrgan.ipynb` to perform inference and no-reference evaluation.
- Use the output directories under `results/` to locate generated images.

If you want to compare multiple baselines, also check:

- `inference_apisr_esrgan.ipynb`

---

## Dataset Curation

### Hugging Face dataset

- **Dataset page**: https://huggingface.co/datasets/liqiqinaOH7/AWM
- **Scale**: 1,481 images
- **Split**:
  - train: 1,047
  - test: 434

### Local dataset structure (recommended)

This project expects (or generates) a structure similar to:

- `dataset/highres/original/`
- `dataset/lowres_2x/original/`
- `dataset/lowres_4x/original/`
- `dataset/lowres_4x_simple/original/` (bicubic-only ablation)
- `dataset/train_list.txt`
- `dataset/test_list.txt`

If you are building the dataset from scratch (Konachan crawling), the following notebooks are relevant:

- `download_konachan.ipynb`
- `rename_and_clean_dataset.ipynb`
- `split_train_test_lists.ipynb`

---

## Train

Training is organized in notebooks and follows an ESRGAN-style progressive schedule:

- generator warmup (L1)
- discriminator warmup
- full GAN training with multi-loss objective

Key notebooks:

- `train_esrgan_independent.ipynb`
- `train_2x_23block_simple_colab.ipynb`
- `train_4x_23block_simple_colab.ipynb`

---

## Results

All results below are from the fixed test split (434 images).

### \(2\times\) no-reference (NIQE / MANIQA / CLIP-IQA)

| Model | NIQE ↓ | MANIQA ↑ | CLIP-IQA ↑ |
|---|---:|---:|---:|
| **AWM RRDB 2×** | **4.9087** | 0.4469 | **0.7109** |
| AWM GRL 2× | 5.3640 | 0.3243 | 0.5541 |
| APISR RRDB 2× | 5.3858 | 0.4870 | 0.6599 |
| Waifu2x 2× | 5.5370 | 0.4882 | 0.6614 |
| RealCUGAN 2× | 5.5862 | **0.4935** | 0.6856 |

### \(2\times\) full-reference (PSNR / SSIM / LPIPS / DISTS / Edge F1)

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | Edge F1 ↑ |
|---|---:|---:|---:|---:|---:|
| **AWM RRDB 2×** | **28.8250** | **0.8544** | **0.2186** | **0.1538** | **0.5619** |
| AWM GRL 2× | 28.1727 | 0.8290 | 0.2763 | 0.1910 | 0.5111 |
| APISR RRDB 2× | 23.7874 | 0.8179 | 0.2552 | 0.1948 | 0.5268 |
| Waifu2x 2× | 27.5192 | 0.8501 | 0.2402 | 0.1743 | 0.4970 |
| RealCUGAN 2× | 27.5141 | 0.8462 | 0.2480 | 0.1779 | 0.4732 |

### \(4\times\) no-reference

| Model | NIQE ↓ | MANIQA ↑ | CLIP-IQA ↑ |
|---|---:|---:|---:|
| AWMSR RRDB 4× | 5.2683 | 0.2639 | 0.4848 |
| **RRDB ESRGAN baseline 4×** | **4.9710** | 0.1564 | 0.2220 |
| APISR RRDB 4× | 6.3849 | 0.4760 | 0.6703 |
| APISR GRL 4× | 5.7950 | **0.4815** | **0.6766** |
| APISR DAT 4× | 5.9168 | 0.4434 | 0.6320 |
| RealESRGAN Anime 6B 4× | 6.6901 | 0.4554 | 0.5670 |

### \(4\times\) full-reference

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | Edge F1 ↑ |
|---|---:|---:|---:|---:|---:|
| AWMSR RRDB 4× | 24.0451 | 0.7251 | 0.3828 | 0.2545 | 0.2618 |
| **RealESRGAN Anime 6B 4×** | **24.4343** | **0.7799** | 0.3270 | 0.2355 | 0.2643 |
| RRDB ESRGAN baseline 4× | 24.2179 | 0.7305 | 0.4360 | 0.3038 | 0.0833 |
| **APISR RRDB 4×** | 22.0093 | 0.7616 | **0.3155** | **0.2281** | **0.3547** |
| APISR GRL 4× | 19.9061 | 0.6819 | 0.3813 | 0.2521 | 0.1436 |
| APISR DAT 4× | 20.5051 | 0.6934 | 0.3660 | 0.2379 | 0.1496 |

---

## License & Disclaimer

- This repository is intended for research and educational purposes.
- The Konachan-sourced data and third-party baselines (APISR / Real-ESRGAN / Waifu2x / RealCUGAN, etc.) are subject to their own licenses and terms. Please verify compliance before redistribution or commercial usage.

