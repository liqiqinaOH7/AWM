# AWM — 动漫壁纸超分辨率 (Anime Wallpaper Super-Resolution)

ECE285 课程项目：在动漫风格壁纸数据上微调 ESRGAN，实现从低分辨率（模拟 Diffusion 粗图）到高分辨率的超分重建。

---

## 一、所做工作概览

### 1. 数据获取与整理

- **`download_konachan.ipynb`**  
  通过 Konachan JSON API 爬取带指定标签（如 `genshin_impact`）的高清原图，保存到 `dataset/highres/original/`。支持过滤 NSFW、变体图，按 `file_url` 下载无损大图。

- **`rename_and_clean_dataset.ipynb`**  
  对 HR 目录做后缀合法性清理（仅保留 `.png` / `.jpg` / `.jpeg`），并将 HR/LR 中的文件名统一重命名为纯数字 ID（如 `371778.jpg`），便于与后续退化、训练一一对应。

- **`clean_large_lowres.ipynb`**  
  按 **highres 总像素 > 25.6M** 筛选并删除过大的图片及其对应的 lowres，避免训练/推理时显存溢出；配对规则与退化、训练一致（同名文件一一对应）。

### 2. 退化流程（模拟 LR）

在 HR 上做可控退化，得到与 highres 一一对应的 lowres，用于训练与测试。

- **`degradation_pipeline_2x.ipynb`**  
  **2× 下采样** + 轻量中值滤波（3×3）+ 轻量高斯模糊与 USM 锐化，尽量模仿 Diffusion 生成的壁纸粗图。  
  输出目录：`dataset/lowres_2x/original/`。

- **`degradation_pipeline_4x.ipynb`**  
  **4× 下采样** + 与 2× 版相同的轻量模糊/中值参数（弱于原始 4× 强退化），在保留更多细节的前提下做 4 倍缩小。  
  输出目录：`dataset/lowres_4x/original/`。

两套 lowres 与 highres 通过**相同文件名**对应；若 HR 为 `.png`，LR 仍以 `.jpg` 保存（质量 85）。

### 3. 模型与训练

- **`train_esrgan_independent.ipynb`**  
  独立训练脚本：使用本仓库内 **APISR_tools** 的 RRDBNet、UNetDiscriminatorSN、平衡双感知损失与 GAN 损失，在自定义动漫数据上微调 ESRGAN。  
  - 数据路径可配置（如 `dataset/lowres/original` + `dataset/highres/original`）。  
  - 支持两阶段（warmup 仅 L1）、混合精度、学习率每 30 epoch 衰减、best 与定期 checkpoint 保存到 `saved_models/`。  
  - 针对 16GB 显存给出了 batch_size、patch_size 等建议。

### 4. 推理与评估

- **`inference_esrgan.ipynb`**  
  加载微调后的 ESRGAN（如 `saved_models/esrgan_finetune_best.pth`），对指定 LR 目录做 4× 超分，结果保存到 `results/ESRGAN_inference/`，可与 HR 对比。

- **`inference_apisr_esrgan.ipynb`**  
  使用 APISR 官方预训练 ESRGAN（如 `4x_APISR_RRDB_GAN_generator.pth`）在同一批 LR 上推理，结果保存到 `results/APISR_ESRGAN_inference/`，便于与自训练模型对比。

- 评估指标（见 `results/results.txt` 等）：**NIQE**（越低越好）、**MANIQA**、**CLIPIQA**（越高越好），用于对比 AWMSR 与 APISR 在无参考质量上的表现。

---

## 二、仓库结构（不含数据与权重）

```
AWM/
├── README.md
├── .gitignore
├── download_konachan.ipynb          # 数据下载
├── rename_and_clean_dataset.ipynb   # 清理与重命名
├── clean_large_lowres.ipynb         # 按 highres 像素清理
├── degradation_pipeline_2x.ipynb   # 2× 轻量退化 → lowres_2x/original
├── degradation_pipeline_4x.ipynb   # 4× 轻量退化 → lowres_4x/original
├── train_esrgan_independent.ipynb   # ESRGAN 微调训练
├── inference_esrgan.ipynb           # 自训练模型推理
├── inference_apisr_esrgan.ipynb     # APISR 官方模型推理对比
├── APISR_tools/                     # 网络与损失（RRDBNet、判别器、感知/GAN 损失等）
│   ├── architecture/
│   └── loss/
└── reading.txt                      # 参考资料等（如有）
```

以下目录由本地生成，**已通过 .gitignore 排除，不会上传**：

- `dataset/` — 高清图与各类 lowres（highres/original, lowres_2x/original, lowres_4x/original 等）
- `results/` — 推理结果与评估输出（如 results.txt）
- `saved_models/` — 训练得到的 .pth
- `pretrained_models/` — 从 APISR 等下载的预训练权重

---

## 三、使用流程建议

1. **准备数据**：运行 `download_konachan.ipynb` → `rename_and_clean_dataset.ipynb` →（可选）`clean_large_lowres.ipynb`。  
2. **生成 LR**：按需运行 `degradation_pipeline_2x.ipynb` 和/或 `degradation_pipeline_4x.ipynb`，得到 `lowres_2x/original`、`lowres_4x/original`。  
3. **训练**：在 `train_esrgan_independent.ipynb` 中设置 `LR_DIR`/`HR_DIR`（如指向 `lowres_4x` 或 `lowres`），运行训练；权重会写入 `saved_models/`。  
4. **推理与对比**：用 `inference_esrgan.ipynb` 跑自训练模型，用 `inference_apisr_esrgan.ipynb` 跑 APISR 官方模型，对比 `results/` 下的输出与指标。

依赖：Python 3.x、PyTorch（建议带 CUDA）、OpenCV、NumPy、Matplotlib、tqdm 等；评估部分可选 `pyiqa`。预训练权重需自行下载并放入 `pretrained_models/`（或按各 notebook 内说明配置路径）。

---

## 四、未来计划 (TODO)

- [ ] **模型架构升级（优先级最高）**：当前 RRDB 表达能力有限，下一步优先尝试 Transformer 系列超分模型；近期先复现/对比 APISR 中提到的 **GRL / DAT / HAT**，并继续调研更新的 SOTA 架构。  
- [ ] **损失函数重构与排查**：重点检查当前 Danbooru 感知损失的稳定性与权重设置，针对“噪点多、伪影多、细节不足、局部碎片化”问题，重新做损失组合和权重扫描（L1 / Perceptual / GAN）。  
- [ ] **任务重心转向 2× 超分**：结合当前实验经验，2× 更符合“细节恢复”目标；4× 场景下信息缺失过大，学习难度和不确定性显著上升。后续优先做 2× 方案优化，4× 作为对照与补充。  
- [ ] **扩充并多样化数据集**：目前数据以原神主题为主且规模约 434 张，可能限制泛化能力；计划增加图片数量与风格多样性，并评估“数据规模收益 vs 训练时间成本”的平衡。  
- [ ] **建立更完整的实验评估闭环**：持续记录 NIQE/MANIQA/CLIPIQA 与可视化对比，形成结构化实验表（架构、损失、2×/4×、数据规模），支持后续迭代决策。
- [ ] **退化过程**：对边缘进行mask，对其余部分做中值滤波。

如你后续有新的实验或脚本，可在此 README 中继续补充「所做工作」与「TODO」列表。

1. 跑完2x的APISR的和我们的推理结果
2. 退化过程改进，然后再跑APISR和我们的RRDB的结果
3. 跑GRL模型


创新点的总结：
1. 别人针对的是动漫的视频，我们针对的是壁纸图片。（他们需要的是推理快、模型小、底子平滑+边缘尖锐，我们要的是细节多）
2. dataset
3. 退化
4. 自创一个评估方法


最终的成绩的衡量：
构建一个测试集里面需要是diffusion的出图，然后跑出来的SR的比分比其他人高？X
估计是打不过RealESRGAN-anime 6B的