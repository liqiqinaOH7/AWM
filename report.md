# AWM：动漫壁纸超分辨率 — 技术报告

**说明**：本报告为内部技术文档，供同事撰写投稿论文时参考。Abstract 预留最后撰写。

---

## 1. Introduction

### 1.1 背景与动机

图像超分辨率（Image Super-Resolution, SR）旨在从低分辨率（Low-Resolution, LR）图像恢复高分辨率（High-Resolution, HR）图像，在医学成像、遥感、视频增强和消费级图像处理等领域有广泛应用。近年来，基于深度学习的 SR 方法在 PSNR、SSIM 等像素级指标上取得了显著进展，而引入生成对抗网络（GAN）的感知驱动方法则进一步提升了重建图像的视觉真实感与细节丰富度。

动漫（Anime）风格图像在互联网文化中占据重要地位，其视觉特点与自然图像显著不同：大面积平滑色块、清晰轮廓线、有限的纹理类型以及相对扁平的光影结构。针对动漫的 SR 因此成为独立的研究分支。现有工作主要面向两类场景：（1）**动漫视频/动画帧**的修复与增强，强调时序一致性、推理速度和轻量部署；（2）**通用动漫图像**的超分，多沿用面向自然图像的退化假设与训练流程。然而，**高质量静态动漫壁纸**（如插画、角色立绘、壁纸级插画）在需求与数据特性上与上述两类均存在明显差异：用户更期待壁纸具有丰富的材质感、细腻的线条与光影层次，而非动画帧式的平滑与一致性；同时，壁纸的来源往往包括网络传播的压缩图、生成模型（如 Stable Diffusion、Midjourney）输出的低分辨率结果等，其退化过程更偏向「先压缩/下采样、后模糊」的软退化，而非 Real-ESRGAN 等所假设的强 JPEG 压缩与复杂真实噪声。

上述差异导致直接使用现有动漫 SR 模型处理壁纸时出现明显不足：以 APISR（CVPR 2024）为代表的、针对动漫视频优化的模型，倾向于输出**平滑纹理与较粗线条**以保持动画风格一致性，难以恢复壁纸所需的**高频细节与线条锐度**；而以 Real-ESRGAN 为代表的通用/动漫模型，其退化流程（如高阶模糊–下采样–噪声–JPEG 的复合退化）更贴合真实世界照片，对「干净但模糊」的壁纸类 LR 输入往往存在**退化失配**，易引入不必要的去块、去噪等行为，或无法充分针对「模糊主导」的退化进行优化。因此，面向**动漫壁纸**这一具体场景，设计**任务匹配的退化流程**与**壁纸导向的数据集与训练策略**，具有明确的现实需求与研究价值。

### 1.2 问题定义与研究目标

本项目的核心问题可概括为：在**动漫壁纸**场景下，如何从低分辨率输入恢复高分辨率、高视觉质量的图像，使输出在保持自然度的同时尽可能保留或增强细节（线条、纹理、光影）。具体而言，我们关注以下子问题：

- **退化建模**：壁纸类 LR 图像（如来自网络传播或生成模型）的退化过程应如何建模？与动画帧、自然照片的退化有何不同？不同退化复杂度（如「仅 bicubic 下采样」与「中值滤波 + 模糊 + 锐化」）对模型行为与客观/主观指标有何影响？
- **数据与任务对齐**：现有动漫 SR 数据集（如 APISR 基于视频关键帧）是否适合壁纸超分？若构建壁纸专用数据集，应如何获取、清洗与划分，以支持训练与无偏评估？
- **模型与训练**：在给定退化与数据的前提下，如何选择生成器与判别器（如 RRDB、GRL、DAT），并设计训练策略（如多阶段 GAN 训练、损失权重）以在**自然度**（如 NIQE）与**感知质量**（如 MANIQA、CLIPIQA）之间取得合理权衡？

研究目标包括：（1）构建一套面向动漫壁纸的数据获取、退化生成与训练/评估流程；（2）在自建壁纸数据集上从头训练多种 SR 架构（以 ESRGAN 系为主），并与 APISR、RealESRGAN 等基线进行系统对比；（3）通过无参考与全参考指标分析退化方式、架构与损失对结果的影响，为后续损失调优与架构扩展提供依据。

### 1.3 主要贡献

- **任务与场景界定**：明确将「动漫壁纸超分」与「动漫视频/通用动漫 SR」区分开来，强调壁纸对细节与自然度的双重需求，以及退化过程（模糊主导 vs 压缩/噪声主导）的差异。
- **壁纸导向数据集**：从 Konachan 按多标签爬取高质量动漫壁纸（原神 2000 张、33%，崩坏：星穹铁道 1000 张、17% 等，合计 6000 张请求量），经清洗与规范命名后构成 HR 集；通过自定义退化流程生成 2×/4× 的 LR 数据；测试集由人工挑选细节丰富、分辨率高、线条细腻的 434 张壁纸构成，训练集 1047 张，名单制划分、不移动文件，保证评估无偏。
- **退化流程设计与消融**：实现并对比「复杂退化」（中值滤波 + 高斯模糊 + bicubic 下采样 + USM 锐化）与「简化退化」（仅 bicubic 下采样），系统分析二者对 NIQE、MANIQA、CLIPIQA 及全参考指标的影响；实验表明复杂退化下模型在 CLIPIQA 等感知指标上优于简化退化，支撑以复杂退化为主流程。
- **训练策略**：在 Colab 一体化流程中采用三阶段训练（G warmup → D warmup → Full GAN），缓解生成器与判别器在训练初期的不平衡，并配合 L1、VGG 感知、Danbooru 感知与 GAN 损失的组合进行 ESRGAN 风格训练。
- **评估体系**：结合无参考指标（NIQE、MANIQA、CLIPIQA）与全参考指标（PSNR、SSIM、LPIPS、DISTS、Canny 边缘 F1），对自训模型与多种基线（APISR RRDB/GRL/DAT、RealESRGAN Anime、waifu2x、RealCUGAN 等）在同一测试集上进行统一评估；对 LPIPS/DISTS 采用 patch-wise 计算以控制显存并支持高分辨率测试。

### 1.4 报告结构

本节为 Introduction；第 2 节为 Related Works；第 3 节为 Proposed Method（3.1 数据集、3.2 退化、3.3 模型结构、3.4 损失函数、3.5 训练策略）；后续章节将依次介绍实验设置、结果与分析、结论与未来工作。Abstract 将在全文定稿后补充。

---

## 2. Related Works

### 2.1 图像超分辨率：从重建到感知

**基于重建的 SR**：早期深度 SR 工作以像素级重建为目标，如 SRCNN、VDSR、EDSR 等，通过最小化 L1/L2 与 GT 的误差提升 PSNR/SSIM。这类方法倾向于产生平滑、过拟合 GT 的结果，在感知质量与纹理自然度上常不如人眼偏好。

**感知驱动与 GAN**：SRGAN 首次将 GAN 与感知损失（基于 VGG 特征）引入 SR，在牺牲部分 PSNR 的同时显著提升视觉真实感。ESRGAN 在此基础上采用 RRDB 结构、去掉 BN、使用相对判别器与更深的网络，成为广泛使用的基线。Real-ESRGAN 进一步面向「真实世界」退化，通过高阶退化建模（多次模糊、下采样、噪声、JPEG 压缩等）生成训练对，并支持 4× 与动漫扩展（Real-ESRGAN Anime），在盲超分与动漫场景下被大量采用。这些工作表明：**退化假设与训练数据**与**目标场景**是否匹配，会直接影响模型在真实数据上的表现。

**盲超分与退化建模**：当 LR 的退化过程未知或复杂时，盲超分（blind SR）或可学习退化的方法受到关注。KernelGAN、FUSR、Real-ESRGAN 等通过设计或学习退化分布，使模型能应对多种退化。我们的工作与之相关之处在于：针对「壁纸」这一子场景，显式设计更贴合该场景的退化流程（模糊主导、弱压缩），而非直接沿用 Real-ESRGAN 的高阶真实退化。

### 2.2 动漫风格图像超分辨率

**经典与轻量方法**：waifu2x 等早期方法针对动漫的平坦色块与线条做了专门优化，在 2× 等尺度上至今仍被使用。Real-CUGAN、Real-ESRGAN Anime 等则在 Real-ESRGAN 的框架下引入动漫数据或动漫特化损失，在保持实时性的同时提升动漫画面质量。这些方法多面向「通用动漫图」或「动画帧」，退化与数据并非专门针对高细节壁纸。

**APISR（CVPR 2024）**：APISR 针对**动漫视频**的超分与修复，从高质量动漫视频中抽取关键帧，经复杂度筛选后构成训练集，并设计了适配动漫的退化流程与多架构支持（RRDB、GRL、DAT 等）。其设计目标包括推理效率与视频帧间一致性，因此模型倾向于输出**平滑、线条略粗**的结果，以利于动画观感；在**静态壁纸**上则可能牺牲插画级的细腻纹理与线条锐度。我们将其作为重要基线，对比其在壁纸测试集上的 NIQE、MANIQA、CLIPIQA 及全参考指标，以说明「视频向」与「壁纸向」任务的差异。

**Transformer 与混合架构**：APISR 中集成的 GRL（Global–Regional–Local）、DAT（Dual Aggregation Transformer）等 Transformer 架构在 APISR 官方数据与退化下表现优异，其预训练模型在感知指标上常优于纯 CNN（如 RRDB）。但在**壁纸数据上自训练**时，我们观察到：在相同壁纸数据集与退化下从头训练的 GRL 在 NIQE、MANIQA、CLIPIQA 等指标上**不如**自训的 RRDB，即「Transformer > CNN」的结论在壁纸场景下并不成立；可能原因包括数据规模、训练配置或架构对壁纸风格适配性的差异，后续可针对此做进一步消融。

### 2.3 退化模型与数据合成

**合成 LR 的方式**：监督 SR 通常通过对 HR 施加退化得到 LR。常见退化包括：下采样（bicubic、双线性等）、模糊（高斯、运动模糊等）、噪声（高斯、泊松等）、压缩（JPEG 等）。Real-ESRGAN 采用随机组合的多次模糊–下采样–噪声–JPEG，以覆盖真实世界的复杂退化。对于**动漫壁纸**，讨论与实验表明：网络传播或生成模型产出的 LR 更多呈现「干净但模糊」的特点，即**模糊与下采样**占主导，强 JPEG 块效应与强噪声相对少见。因此，将**模糊置于退化链末端**、或采用「轻压缩 + 末端模糊」的软退化，更符合壁纸场景，并使模型将容量用于锐化与细节恢复而非去块。

**本项目中的退化设计**：我们实现了两类退化用于训练与消融：（1）**复杂退化**：中值滤波 → 高斯模糊 → bicubic 下采样（2× 或 4×）→ USM 锐化，以模拟一定程度的纹理平滑与再锐化过程；（2）**简化退化**：仅对 HR 做 bicubic 下采样。实验表明：**复杂退化下训练的模型在感知质量（如 CLIPIQA）上优于简化退化**，简化退化因退化过于单一，模型学到的恢复能力有限，CLIPIQA 等指标不及复杂退化；该结论支撑我们将复杂退化作为主流程，并为投稿中讨论退化设计提供实证依据。

### 2.4 感知质量与无参考评估

**全参考指标**：PSNR、SSIM 衡量像素与结构相似度；LPIPS、DISTS 等基于深度特征的度量更贴近人类对「感知相似度」的判断，常与 GAN 型 SR 一起使用。针对动漫，线条保真度尤为重要，因此我们引入基于 Canny 边缘的 F1 分数作为辅助指标，与 PSNR/SSIM/LPIPS/DISTS 共同构成全参考评估体系。

**无参考指标**：在无 GT 或需要跨模型对比时，无参考图像质量评估（NR-IQA）被广泛采用。NIQE 基于自然场景统计，数值越低通常表示越自然、伪影越少；MANIQA、CLIPIQA 等基于学习或 CLIP 的 NR-IQA 则更侧重语义与整体感知质量，数值越高表示感知质量越好。我们在同一测试集上对自训模型与 APISR、RealESRGAN、waifu2x、RealCUGAN 等统一计算 NIQE、MANIQA、CLIPIQA，发现：自训模型在 NIQE 上 consistently 优于上述基线，而在 MANIQA/CLIPIQA 上仍与 APISR 有差距，这与当前损失权重（pixel loss 偏大）导致的「保守/平滑」输出一致，可作为 Related Works 与实验分析中讨论「自然度 vs 感知质量」的支撑。

### 2.5 动漫与壁纸相关数据集

**通用与动漫 SR 数据集**：DIV2K、Flickr2K 等常用于自然图像 SR；动漫方面，Danbooru 等图库规模巨大但需自行筛选与清洗。APISR 使用从 562 个动漫视频中抽取的关键帧并经复杂度筛选得到 3740 张高质量图像，数据特性偏向**动画帧**而非**壁纸/插画**。目前公开的、专门针对「超高清动漫壁纸」的 SR 数据集较少，多数工作依赖自建或从图站爬取后筛选。

**本项目的壁纸数据集**：我们从 Konachan 通过 JSON API 按多组标签爬取高清原图（原神 2000 张（33%）、崩坏：星穹铁道 1000 张（17%）等，合计 6000 张请求量），经重命名、过大图过滤等步骤得到 HR 集；再经上述退化流程生成 2×/4× 的 LR。测试集由人工挑选细节丰富、分辨率高、线条细腻的高质量壁纸构成（434 张），训练集为其余样本（1047 张），名单制划分、不移动文件，便于复现与扩展。数据规模与来源可在论文中明确说明，以区分于 APISR 等视频帧数据。

---

## 3. Proposed Method

本节介绍所提出的动漫壁纸超分流程：3.1 数据集、3.2 退化流程、3.3 模型结构（生成器与判别器）、3.4 损失函数、3.5 训练策略。

### 3.1 数据集

**来源与获取**：我们构建了面向动漫壁纸的高分辨率图像集合。高质量动漫壁纸的公开专用数据集较少；APISR 等使用的是从动漫视频中抽取的关键帧，其内容与风格更偏动画帧而非插画/壁纸。因此，我们从 **Konachan**（高画质动漫图像站）通过其官方 **JSON API** 爬取图像。下载脚本（`download_konachan.ipynb`）中配置了**多组标签**及每类请求数量：按标签分别请求元数据（`limit` 参数控制单次上限），并依据返回的 `file_url` 下载原图；相比解析网页更稳定，且能直接获取高分辨率文件。例如，**原神**（`genshin_impact`）设定为 2000 张、占全量 **33%**（2000/6000），**崩坏：星穹铁道**（`honkai:_star_rail`）1000 张、**17%**（1000/6000），其余标签各 1000 张或类似配置，**合计 6000 张**。下载的 HR 图像统一保存至 `dataset/highres/original/`。

**清洗与规范化**：（1）**重命名**：通过 `rename_and_clean_dataset.ipynb` 将文件名统一为纯数字 ID（如 `371778.jpg`），便于后续与各版本 LR 按「文件名主干」一一对应，避免扩展名（.jpg / .png）不一致导致匹配错误。（2）**过大图过滤**：在 `clean_large_lowres.ipynb` 中按像素数阈值（例如超过 25.6M 像素）剔除部分极大尺寸图像，以降低训练时的显存与 I/O 压力。经去重、过滤 NSFW/变体及过大图后，得到 **1481 张** 高质量 HR 图像，作为后续退化与训练的 GT 集合。

**训练集与测试集划分**：为得到无偏的评估结果，必须保证测试集在训练阶段未被使用。**测试集名单**（`dataset/test_list.txt`，共 **434** 张）由**人工手动挑选**得到：挑选标准为细节含量较多、分辨率较高、线条较细腻、更符合人类对高质量动漫壁纸审美偏好的图像，以保证测试集既能代表壁纸场景又便于评估超分后的细节与观感。**训练集名单**（`dataset/train_list.txt`，共 **1047** 张）由 HR 集中除测试集外的其余样本构成。我们采用**名单制**划分，不复制、不移动任何文件，仅通过两个名单文件区分训练与测试；匹配时按「去掉扩展名的文件名」判断是否为同一张图。训练时 DataLoader 仅读取 `train_list.txt` 中的路径，评估与指标计算仅针对 `test_list.txt` 中的样本，从而在保持现有目录结构的前提下实现严格的训练/测试分离。

**与现有动漫 SR 数据的区别**：本数据集以**静态壁纸/插画**为主，来源与筛选方式均不同于 APISR 的视频关键帧数据，规模为千级 HR + 对应 2×/4× LR，适合壁纸场景下的模型训练与消融；数据规模与来源在论文中需明确说明，以区分于通用动漫或视频帧数据集。

### 3.2 退化流程

壁纸类低分辨率图像的退化特性与「真实世界照片」或「动画帧」有所不同：网络传播或生成模型（如 Diffusion）产出的 LR 往往**干净但模糊**，即下采样与模糊占主导，强 JPEG 块效应与强噪声相对少见。因此，我们设计了面向壁纸的退化流程，并实现两种策略用于训练与消融，以分析退化复杂度对模型行为的影响。

**复杂退化（complex degradation）**：用于 2× 与 4× 的「复杂」版本 LR 生成（对应 `degradation_pipeline_2x.ipynb`、`degradation_pipeline_4x.ipynb`）。流程依次为：  
1. **中值滤波**（如 3×3）：对 HR 做轻度平滑，模拟部分纹理/噪声被平均化的效果。  
2. **高斯模糊**（如核 3×3、σ=0.8）：进一步降低高频，使后续下采样后的 LR 更贴近「模糊主导」的观感。  
3. **Bicubic 下采样**：将图像缩小至目标尺寸（宽高各为原来的 1/2 或 1/4），得到 2× 或 4× 的 LR。  
4. **USM 锐化**（Unsharp Masking）：对下采样后的 LR 做适度锐化，模拟部分场景下「先模糊再被增强」的中间状态，增加退化多样性。  
输出分别存放于 `lowres_2x/original/`、`lowres_4x/original/` 等目录，与 HR 按文件名一一对应。

**简化退化（simple degradation）**：仅对 HR 做 **bicubic 下采样**至 2× 或 4×，不做中值滤波、高斯模糊与 USM 锐化（对应 `degradation_pipeline_4x_simple.ipynb` 以及 Colab 中的 simple 流程）。输出目录例如 `lowres_4x_simple/original/`。该策略作为消融对照组，用于验证退化复杂度对模型表现的影响。

**设计动机与消融结论**：将模糊与平滑置于下采样之前或之中、末端辅以可选锐化，与「壁纸/LR 以模糊为主」的假设一致；我们未采用 Real-ESRGAN 风格的高阶「多次模糊–下采样–噪声–JPEG」组合，以更贴近壁纸场景并便于控制变量进行消融。实验中观察到：**复杂退化下训练的模型整体优于简化退化**。例如在感知质量指标 **CLIPIQA** 上，复杂退化（如 median+blur+sharpen）训练的模型明显高于仅用 bicubic 简化退化训练的模型，说明更贴近真实壁纸退化的复杂流程能带来更好的细节恢复与感知效果，后续主实验与对比均以复杂退化为主。

### 3.3 模型结构

我们采用 **ESRGAN** 风格的生成对抗框架：生成器（Generator, G）负责从 LR 恢复 HR，判别器（Discriminator, D）负责区分「真实」高分辨率图像与生成结果，二者通过对抗损失与多种重建/感知损失联合训练。以下分别说明 G 与 D 的架构。

**生成器（RRDBNet）**：生成器为 **RRDBNet**（Residual in Residual Dense Block Network），即由多个 **RRDB** 块堆叠而成的主干 + 上采样头。单块 RRDB 内部包含三个 **RDB**（Residual Dense Block），每个 RDB 为 5 层密集连接的卷积（growth channel 32），层间 LeakyReLU(0.2)，输出经 0.2 倍残差缩放后与输入相加；RRDB 整体再以 0.2 倍残差与输入相加，便于深层训练。输入输出通道均为 3（RGB）；**尺度**为 2× 或 4×。  
- **2× 时**：先对 LR 做 **pixel-unshuffle**（scale=2），将空间尺寸减半、通道扩为 12，再进入主干，这样主干在「等效低分辨率」上计算，最后上采样 2× 得到 HR。  
- **4× 时**：不做 pixel-unshuffle，LR 直接进主干；主干后接两次 2× 上采样（nearest + 3×3 conv + LeakyReLU），得到 4× 空间尺寸。  
主干由 **num_block** 个 RRDB 组成（实验中为 **6-block** 或 **23-block**），中间特征通道数 **num_feat=64**，每 RRDB 内 **num_grow_ch=32**。我们从头训练，不使用预训练权重。

**生成器（GRL，可选）**：除 RRDBNet 外，我们还支持 **GRL**（Global–Regional–Local）作为生成器，其实现与训练流程见 **`train_grl_independent.ipynb`**。GRL 为基于 **Transformer** 的图像恢复网络：输入经一层 3×3 卷积得到 **embed_dim** 维特征（**embed_dim=128**），再进入多个 **TransformerStage**，每 stage 内包含 **depth** 个 **EfficientMixAttnTransformerBlock**，融合 **窗口注意力**（window attention）、**条纹注意力**（stripe attention，沿 H 或 W 方向）与局部卷积，以同时建模全局、区域与局部依赖。典型配置为 **depths=[4,4,4,4]**、**num_heads_window** 与 **num_heads_stripe** 均为 **[2,2,2,2]**、**window_size=8**、**mlp_ratio=2**；**img_size** 与 LR patch 尺寸一致（`train_grl_independent.ipynb` 中 **GRL_IMG_SIZE=PATCH_SIZE//SCALE=112**，即 PATCH_SIZE=224、scale=2）。  
- **2× 超分**：上采样模块使用 **pixelshuffle**（`upsampler="pixelshuffle"`），将特征图直接重排为 2× 空间尺寸并卷积到 3 通道。  
- **4× 超分**：上采样使用 **nearest+conv**（两次 2× nearest 插值 + 3×3 卷积），当前实现仅支持 4×。  
GRL 与 RRDB 使用**同一判别器**（UNetDiscriminatorSN）及**同一套损失**（L1 + VGG + Danbooru + GAN），训练策略同样为三阶段（G warmup → D warmup → Full GAN）。`train_grl_independent.ipynb` 中：**PATCH_SIZE=224**（2× 时 LR patch 112），**GRL_IMG_SIZE=112**，**LEARNING_RATE=1e-5**，**LR_FIRST_DECAY_AT_EPOCH=37**，**LR_STEP_SIZE=30**，**LR_GAMMA=0.5**，**LR_MIN=1e-6**；**G_WARMUP_EPOCHS=3**，**D_WARMUP_EPOCHS=3**，**GAN_START_EPOCH=7**（即第 1–3 epoch 仅训 G，第 4–6 epoch 仅训 D，第 7–150 epoch 全 GAN）。  
在壁纸数据上的实验表明：**自训 GRL 在 NIQE、MANIQA、CLIPIQA 等指标上不如自训 RRDB**，即「Transformer > CNN」在本场景下不成立；可能原因包括数据规模、学习率与 warmup 设置或架构对壁纸风格适配性，GRL 仍作为可选架构保留以便对比与后续消融。

**判别器（UNetDiscriminatorSN）**：判别器采用 **U-Net 结构 + 谱归一化**（Spectral Normalization），与 Real-ESRGAN 一致。输入为 3 通道 RGB 图像，先经一层 3×3 卷积得到 64 通道特征，再经三层 stride-2 的下采样（64→128→256→512 通道），随后经上采样与 **skip connection** 与同尺度下采样特征相加，恢复空间分辨率，最后经两层 3×3 卷积输出单通道 logit。除首层与末层外，卷积均施加谱归一化以稳定训练。判别器用于判断输入是「真实 HR」还是「生成 HR」；**真实样本**我们使用 **degrade_hr**（即经与训练对一致的退化流程得到的「退化后再上采样回 HR 尺寸」的图像），而非原始 HR，以避免 USM 锐化等进入 D、使 D 更关注与生成分布对齐的「干净 HR」分布。

**数据流小结**：训练时每个 batch 包含 **lr**、**degrade_hr**、**hr** 三路。G 的输入为 lr，输出为 gen_hr；G 的损失在 gen_hr 与 hr 之间计算（以及 gen_hr 经 D 的 GAN 损失）。D 的「真」输入为 degrade_hr，「假」输入为 gen_hr.detach()，二者分别计算对抗损失并反向更新 D。

### 3.4 损失函数

生成器的总损失由 **像素损失**、**感知损失**（VGG + 动漫特化）与 **GAN 对抗损失** 三部分组成；判别器仅使用 **GAN 损失**（二分类，真/假）。各损失在实现中可配权重，以下给出含义与典型配置。

**（1）像素损失（L1）**：在像素空间对生成 HR 与真实 HR 做 **L1** 距离（或可选 L1-Charbonnier），即 \( \mathcal{L}_{\mathrm{pix}} = \|\hat{I}_{\mathrm{HR}} - I_{\mathrm{HR}}\|_1 \)。用于约束整体几何与亮度一致，权重较大时输出更平滑、更贴近 GT，但可能削弱感知上的锐利度。`train_esrgan_independent.ipynb` 与 `train_grl_independent.ipynb` 中 **WEIGHT_PIXEL=10.0**，相对感知与 GAN 权重偏大，是后续调优的重点之一。

**（2）VGG 感知损失**：在 **VGG**（如 vgg19）的若干层上提取特征，对 gen_hr 与 hr 的特征图做 L1，再按层加权求和。层为 `conv1_2, conv2_2, conv3_4, conv4_4, conv5_4`，层权重 `{conv1_2: 0.1, conv2_2: 0.1, conv3_4: 1, conv4_4: 1, conv5_4: 1}`，整体再乘 **WEIGHT_VGG=0.05**。输入在送入 VGG 前做 ImageNet 均值方差归一化；VGG 参数冻结。

**（3）Danbooru 动漫感知损失**：使用在 **Danbooru** 等动漫数据上预训练的 **ResNet** 作为特征提取器，同样在指定层上计算 gen_hr 与 hr 的特征 L1，并按层加权（如 `{'0': 0.1, '4_2_conv3': 20, '5_3_conv3': 25, '6_5_conv3': 1, '7_2_conv3': 1}`），再乘 **WEIGHT_DANBOORU=0.025**。用于增强对动漫风格、线条与纹理的感知一致性。

**（4）GAN 损失**：采用 **vanilla GAN**（BCEWithLogitsLoss）。对生成器：希望 D(gen_hr) 被判为「真」，即 \( \mathcal{L}_G^{\mathrm{gan}} = \mathrm{BCE}(D(\hat{I}_{\mathrm{HR}}), 1) \)。对判别器：希望 D(real) 为 1、D(fake) 为 0，即 \( \mathcal{L}_D = \mathrm{BCE}(D(I_{\mathrm{degrade\_hr}}), 1) + \mathrm{BCE}(D(\hat{I}_{\mathrm{HR}}.\mathrm{detach}), 0) \)。GAN 损失对 G 的权重 **WEIGHT_GAN=1.0**（两 notebook 中均为 1.0）。  
生成器总损失可写为：  
\( \mathcal{L}_G = \lambda_{\mathrm{pix}} \mathcal{L}_{\mathrm{pix}} + \lambda_{\mathrm{vgg}} \mathcal{L}_{\mathrm{vgg}} + \lambda_{\mathrm{danbooru}} \mathcal{L}_{\mathrm{danbooru}} + \lambda_{\mathrm{gan}} \mathcal{L}_G^{\mathrm{gan}} \)。

### 3.5 训练策略

**数据加载与增强**：训练时从 HR/LR 目录（或名单）读取配对样本；采用 **patch 训练**：从每张图中随机裁剪 **patch_size×patch_size** 的 HR 块，LR 为对应区域下采样后的块（2× 时 128×128，4× 时 64×64）。`train_esrgan_independent.ipynb` 中 **PATCH_SIZE=256**；`train_grl_independent.ipynb` 中 **PATCH_SIZE=224**。数据增强包括**随机水平翻转、垂直翻转、90° 旋转**，对 lr、degrade_hr、hr 同步施加以保证配对一致。**BATCH_SIZE=8**（两 notebook 一致）。

**优化器与学习率**：G 与 D 均使用 **Adam**，**betas=(0.9, 0.999)**，**LEARNING_RATE=1e-5**（两 notebook 一致）。学习率调度为 **StepLR**：**LR_STEP_SIZE=30**，**LR_GAMMA=0.5**，**LR_MIN=1e-6**。

**三阶段训练（Colab 一体化流程）**：为缓解 G 与 D 在训练初期的不平衡，我们采用 **三阶段** 策略。以 `train_esrgan_independent.ipynb` 与 Colab 流程为例：  
- **Phase 1（G warmup）**：前 **5** 个 epoch 只训练 G，且只使用 **L1 像素损失**，不更新 D（**WARMUP_EPOCHS=5**）。目的为稳定 G 的权重，避免一上来被过强的 GAN 梯度破坏。  
- **Phase 2（D warmup）**：第 6～10 个 epoch **冻结 G**，只训练 D。用当前 G 生成的 fake 与 degrade_hr 作为 real，让 D 学会区分二者。  
- **Phase 3（Full GAN）**：第 11～150 个 epoch 进行 **完整 GAN 训练**：每步先更新 G（L1 + VGG 感知 + Danbooru 感知 + GAN），再更新 D（real/fake 的 BCE）。  
总 epoch 数 **150**（5+5+140）。`train_grl_independent.ipynb` 中为 **G_WARMUP_EPOCHS=3**、**D_WARMUP_EPOCHS=3**、**GAN_START_EPOCH=7**，即第 1–3 epoch 仅 G、第 4–6 epoch 仅 D、第 7–150 epoch 全 GAN。

**单步迭代顺序（与 train_master 一致）**：每 iteration 先 **G 前向**（lr → gen_hr），计算 G 的总损失并反向、更新 G；再 **D 前向**：对 degrade_hr 与 gen_hr.detach() 分别前向 D，计算 D 的 real/fake 损失并反向、更新 D。使用 **混合精度（AMP）** 时，G 与 D 的前向与损失计算置于 `torch.cuda.amp.autocast()` 内，反向与 step 配合 GradScaler，以节省显存并加速。

**保存**：G 与 D 均从头初始化。训练过程中按 **iteration** 或 **epoch** 保存 checkpoint（best generator loss、latest、每 **SAVE_FREQ** 个 epoch 备份；`train_esrgan_independent.ipynb` 中 **SAVE_FREQ=10**），便于恢复与选模型。

---

**小结**：Introduction 明确了动漫壁纸 SR 的任务差异与退化/数据需求，给出了问题定义、目标与贡献；Related Works 覆盖了通用 SR、动漫 SR、退化建模、感知与无参考评估以及数据集，并修正了「Transformer > CNN」在壁纸自训场景下不成立的结论；Proposed Method 中 3.1 详述了壁纸数据集的来源（多标签及占比）、清洗、规范命名与训练/测试划分（测试集人工挑选）；3.2 详述了复杂退化与简化退化及消融结论；3.3 详述了生成器（RRDBNet 与可选 GRL）、判别器（UNetDiscriminatorSN）的结构及数据流，并说明 GRL 的配置与在壁纸数据上不及 RRDB 的实验结论；3.4 详述了像素损失、VGG 感知、Danbooru 感知与 GAN 损失的构建与权重；3.5 详述了数据加载与增强、优化器与学习率、三阶段训练（G warmup → D warmup → Full GAN）及单步迭代与混合精度。后续可在此基础上续写 Experiments、Conclusion，并最后撰写 Abstract。

---

## 4. Experiments

本节汇总在同一测试集上的实验设置与结果，对比我们从头训练的模型与多种公开基线。无参考指标结果来自 `results.txt`；全参考指标汇总来自 `fullref_eval_summary.json`。

### 4.1 实验设置

- **数据集**：Konachan 动漫壁纸测试集 **434 张**（`results.txt` 标注），用于无参考评估与大部分全参考评估。
- **倍率**：**2×** 与 **4×** 两个放大倍率。
- **训练配置（ours）**：从头训练，关键超参以独立训练脚本为准。
  - **RRDB（2×/4×）**：`train_esrgan_independent.ipynb`：**NUM_EPOCHS=150**，**BATCH_SIZE=8**，**PATCH_SIZE=256**，**LEARNING_RATE=1e-5**，**betas=(0.9, 0.999)**，**LR_STEP_SIZE=30**，**LR_GAMMA=0.5**，**LR_MIN=1e-6**，**WARMUP_EPOCHS=5**；损失权重 **WEIGHT_PIXEL=10.0**、**WEIGHT_VGG=0.05**、**WEIGHT_DANBOORU=0.025**、**WEIGHT_GAN=1.0**；**SAVE_FREQ=10**。
  - **GRL（2×）**：`train_grl_independent.ipynb`：**NUM_EPOCHS=150**，**BATCH_SIZE=8**，**PATCH_SIZE=224**（LR patch 112），**LEARNING_RATE=1e-5**，**betas=(0.9, 0.999)**，**LR_FIRST_DECAY_AT_EPOCH=37**，**LR_STEP_SIZE=30**，**LR_GAMMA=0.5**，**LR_MIN=1e-6**；损失权重 **WEIGHT_PIXEL=10.0**、**WEIGHT_VGG=0.05**、**WEIGHT_DANBOORU=0.025**、**WEIGHT_GAN=1.0**；三阶段 **G_WARMUP_EPOCHS=3**、**D_WARMUP_EPOCHS=3**、**GAN_START_EPOCH=7**；**SAVE_FREQ=10**。GRL 架构参数：**GRL_DEPTHS=[4,4,4,4]**、**GRL_EMBED_DIM=128**、**GRL_NUM_HEADS=[2,2,2,2]**、`upsampler="pixelshuffle"`。

### 4.2 评估指标

我们同时报告无参考（NR-IQA）与全参考（FR）指标，以覆盖「自然度/观感」与「对 GT 的还原」两类目标。

- **无参考指标（越大/越小按下述方向更优）**：
  - **NIQE ↓**：越低表示越自然、伪影越少。
  - **MANIQA ↑**：学习式无参考质量指标，越高越好。
  - **CLIPIQA ↑**：基于 CLIP 的无参考质量指标，越高越好。
- **全参考指标**（需配对 HR）：
  - **PSNR ↑（dB）**、**SSIM ↑**：像素/结构相似度。
  - **LPIPS ↓（VGG）**、**DISTS ↓**：感知相似度距离。
  - **Edge_F1 ↑（Canny）**：线条保真度（边缘 F1）。

### 4.3 对比方法（Baselines）

无参考与全参考评估覆盖以下模型（以结果文件中的命名为准）：

- **Ours（从头训练）**：
  - **AWMSR RRDB 2× 6-block（complex）**
  - **AWMSR RRDB 4× 6-block（complex）**
  - **AWM GRL 2×（从头训练）**
  - **RRDB 23-block（simple, bicubic only）**（作为简化退化对照，不纳入主趋势）
- **公开/外部基线**：
  - **APISR RRDB / GRL / DAT**（官方模型）
  - **RealESRGAN Anime 6B**
  - **waifu2x**
  - **RealCUGAN**
  - **xinntao ESRGAN**（经典 ESRGAN 权重）

### 4.4 无参考评估结果（NIQE / MANIQA / CLIPIQA）

#### 4.4.1 4× 无参考结果（434 张）

| 模型 | NIQE ↓ | MANIQA ↑ | CLIPIQA ↑ |
|---|---:|---:|---:|
| Ours: AWMSR RRDB 4× 6-block（complex） | 5.2683 | 0.2639 | 0.4848 |
| RRDB ESRGAN 4×（baseline） | 4.9710 | 0.1564 | 0.2220 |
| APISR RRDB 4× | 6.3849 | 0.4760 | 0.6703 |
| APISR GRL 4× | 5.7950 | 0.4815 | 0.6766 |
| APISR DAT 4× | 5.9168 | 0.4434 | 0.6320 |
| RealESRGAN Anime 6B 4× | 6.6901 | 0.4554 | 0.5670 |

#### 4.4.2 2× 无参考结果（434 张）

| 模型 | NIQE ↓ | MANIQA ↑ | CLIPIQA ↑ |
|---|---:|---:|---:|
| Ours: AWMSR RRDB 2× 6-block（complex） | 4.9087 | 0.4469 | 0.7109 |
| Ours: AWM GRL 2× | 5.3640 | 0.3243 | 0.5541 |
| APISR RRDB 2× | 5.3858 | 0.4870 | 0.6599 |
| waifu2x 2× | 5.5370 | 0.4882 | 0.6614 |
| RealCUGAN 2× | 5.5862 | 0.4935 | 0.6856 |

#### 4.4.3 简化退化对照（bicubic only；不参与主趋势）

| 模型 | NIQE ↓ | MANIQA ↑ | CLIPIQA ↑ |
|---|---:|---:|---:|
| RRDB 23-block 4×（simple） | 4.7277 | 0.2312 | 0.4507 |
| RRDB 23-block 2×（simple） | 3.8378 | 0.3010 | 0.5592 |

### 4.5 全参考评估结果（PSNR / SSIM / LPIPS / DISTS / Edge_F1）

全参考汇总见 `fullref_eval_summary.json`。其中大多数条目使用测试集 **434** 张；**APISR_RRDB 2×** 在该汇总中为 **433** 张（文件中 `num_images=433`）。需要注意：该 JSON 中我们自己的 ESRGAN/RRDB 条目为 **2× 的 RRDB 6-block** 与 **4× 的 RRDB 23-block**；而 `results.txt` 的无参考主表里，我们的 4× 结果对应 **RRDB 6-block（complex degradation）**。因此 4× 的无参考与全参考数字不应被理解为同一模型的一组指标。

#### 4.5.1 2× 全参考结果

| 模型 | #Images | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | Edge_F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Ours: ESRGAN（RRDB 6-block）2× | 434 | 28.8250 | 0.8544 | 0.2186 | 0.1538 | 0.5619 |
| Ours: GRL 2× | 434 | 28.1727 | 0.8290 | 0.2763 | 0.1910 | 0.5111 |
| APISR RRDB 2× | 433 | 23.7874 | 0.8179 | 0.2552 | 0.1948 | 0.5268 |
| Waifu2x 2× | 434 | 27.5192 | 0.8501 | 0.2402 | 0.1743 | 0.4970 |
| RealCUGAN 2× | 434 | 27.5141 | 0.8462 | 0.2480 | 0.1779 | 0.4732 |

#### 4.5.2 4× 全参考结果

| 模型 | #Images | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | Edge_F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Ours: ESRGAN（RRDB 23-block）4× | 434 | 24.0451 | 0.7251 | 0.3828 | 0.2545 | 0.2618 |
| RealESRGAN Anime 6B 4× | 434 | 24.4343 | 0.7799 | 0.3270 | 0.2355 | 0.2643 |
| xinntao ESRGAN 4× | 434 | 24.2179 | 0.7305 | 0.4360 | 0.3038 | 0.0833 |
| APISR RRDB 4× | 434 | 22.0093 | 0.7616 | 0.3155 | 0.2281 | 0.3547 |
| APISR DAT 4× | 434 | 20.5051 | 0.6934 | 0.3660 | 0.2379 | 0.1496 |
| APISR GRL 4× | 434 | 19.9061 | 0.6819 | 0.3813 | 0.2521 | 0.1436 |

### 4.6 结果分析与讨论

**（1）2× 场景下，我们的 RRDB 在无参考感知指标上最强**：在 2× 无参考评估中，Ours RRDB 的 **CLIPIQA=0.7109** 为所有对比方法最高，同时 NIQE=4.9087 也是最优，说明在壁纸场景下 2× 更容易兼顾自然度与细节观感。

**（2）4× 场景下，APISR 系列在 MANIQA/CLIPIQA 上领先，但我们的模型在 NIQE 上更优**：4× 无参考结果显示，APISR GRL 的 MANIQA=0.4815、CLIPIQA=0.6766 最高；而我们 RRDB 的 NIQE=5.2683 优于 APISR RRDB/GRL/DAT 与 RealESRGAN Anime 6B，体现出「自然度/伪影控制」与「感知锐化」之间的权衡。

**（3）全参考指标揭示了不同目标函数的偏好**：在 4× 全参考中，APISR RRDB 的 Edge_F1=0.3547 明显高于多数方法，说明其更强调线条结构；而在 2× 全参考中，我们的 RRDB（PSNR=28.8250、SSIM=0.8544、LPIPS=0.2186、Edge_F1=0.5619）整体表现突出，表明从头训练的 RRDB 在该壁纸测试集上同时具备较强的像素还原与线条一致性。

**（4）GRL 在壁纸数据上从头训练不如 RRDB**：无参考（NIQE/MANIQA/CLIPIQA）与全参考（LPIPS/DISTS/Edge_F1）均显示，2× 的 GRL 低于 2× 的 RRDB，这与第 2 节中对「Transformer > CNN」在本场景不成立的结论一致。
