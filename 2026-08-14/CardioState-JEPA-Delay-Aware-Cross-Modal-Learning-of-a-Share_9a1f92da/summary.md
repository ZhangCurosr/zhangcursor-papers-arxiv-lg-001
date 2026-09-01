---
title: "CardioState-JEPA-Delay-Aware-Cross-Modal-Learning-of-a-Share"
source: https://arxiv.org/pdf/2608.12944v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:27:45"
field: "多模态生理信号表征学习"
keywords: ["cardiac foundation model", "ECG", "PPG", "PCG", "joint-embedding predictive architecture", "cross-modal learning", "self-supervised learning", "physiological delay alignment"]
innovations: ["提出生理感知的联合嵌入预测架构，在 ECG/PPG/PCG 三模态上学习统一心脏潜在表征", "引入可学习延迟对齐器+生理锚定监督，实现跨模态时序对齐", "两阶段课程学习策略：先单模态掩码潜在预测建立模态内结构，再用配对数据进行延迟感知跨模态对齐"]
benchmarks: ["PTB-XL", "CPSC 2018", "CSN", "WESAD", "DaLiA", "MIMIC AF", "CirCor", "CinC2016", "PPG-EXT", "VitalDB", "EPHNOGRAM"]
---

# 论文速读：CardioState-JEPA-Delay-Aware-Cross-Modal-Learning-of-a-Share

## 一句话总结
本文提出了 **CardioState-JEPA**，一个面向心电图（ECG）、光电容积脉搏波（PPG）和心音图（PCG）的统一心脏表征基础模型，通过掩码潜在预测与生理延迟感知跨模态对齐，使三种异构信号在共享潜在空间中相互监督。实验表明该模型在 25 个下游任务上显著优于各模态独立自监督基线，PPG 分类平均提升 8.2 AUROC 点、PCG 杂音检测提升 18.8 AUROC 点、ECG 分类提升 15.5 AUROC 点。

## 研究问题与动机
- **单模态预训练的割裂问题**：现有心脏基础模型（如 ECGFounder、PaPaGei、StethoLM）均以单一传感器模态训练，无法利用跨模态共享的生理结构信息。
- **时间偏移的生理耦合**：ECG（电活动）、PCG（机械/声学活动）与 PPG（血流动力学活动）观测同一心动周期，但存在电-机械耦合延迟和脉搏 Transit 延迟，简单时间戳对齐会导致不同心动相位被错误匹配。
- **波形差异带来的表征困难**：三种模态在采样率、通道数、形态上差异显著，原始重建目标会鼓励模型学习传感器特有特征而非共享生理结构；同时负样本对比在生理数据中可能因不同受试者共享相同心律/病理而产生歧义。
- **配对数据稀缺的现实约束**：大规模单模态记录日益丰富，但同步多传感器记录相对稀少，需要一种课程学习策略充分利用异构数据分布。

## 核心贡献（创新点）
1. **统一潜在心脏状态建模框架**：将 ECG/PPG/PCG 统一建模为同一隐藏生理过程 $\mathbf{c}(t)$ 的三种时间偏移观测，提出从异构信号恢复共享潜在心脏状态的表征学习目标，与现有单模态方法有本质区别。
2. **生理学感知联合嵌入预测架构（JEPA）**：采用轻量级模态专用 Tokenizer + 单一共享 Transformer Encoder，在潜在空间进行掩码预测而非原始波形重建，避免学习传感器特有噪声，且无需负样本对。
3. **延迟感知跨模态对齐机制**：引入可学习的延迟头 $h_\delta$，通过 tanh 有界输出与高斯核软采样实现跨模态 token 对齐；以 R 峰-第一心音/脉搏到达时间为生理锚点对延迟进行回归监督，使对齐反映真实生理时序而非任意偏移。
4. **两阶段课程学习策略**：Stage I 从海量单模态数据中学习模态内结构（掩码潜在预测），Stage II 用稀缺配对数据通过延迟感知跨模态预测对齐模态，同时继续采样单模态数据防止表征漂移。
5. **跨模态统一表征的广泛验证**：在 25 个下游任务（ECG/PPG/PCG 各模态共覆盖分类与回归）上进行冻结编码器线性探测评估，证明单一共享编码器的强跨模态迁移能力。

## 方法详解
**整体架构**：每个模态通过专用轻量化 Tokenizer $f_m$（stride conv + multiscale depthwise block + linear projection + 正弦位置/模态嵌入）映射到统一 token 空间 $\mathbf{h}_m \in \mathbb{R}^{N \times d}$，再由单一共享 Transformer Encoder $g_\theta$（ViT-B，12 层，12 头，dim=768）处理， projector $\pi$ 映射到心脏码 $\mathbf{Z}_m$。动量编码器 $\bar{g}_{\bar{\theta}}$（EMA 更新）生成 stop-gradient 目标。

**Stage I — 模态内 JEPA（掩码潜在预测）**：
对可见上下文 token 进行编码，预测被掩码块的潜在码，与动量编码器对未掩码信号生成的目标对齐：
$$\mathcal{L}_{\text{intra}} = \frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} \mathcal{H}\big(\text{LN}(\hat{\mathbf{z}}^i), \text{LN}(\bar{\mathbf{z}}^i)\big)$$
其中 $\mathcal{H}$ 为 Smooth-L1 loss。掩码块通常跨越完整心动周期，迫使模型从相邻心动周期推断缺失状态，从而捕捉跨模态共享的心律与相位结构。

**Stage II — 延迟感知跨模态 JEPA**：
从配对记录中，源模态编码 $\mathbf{z}_m^i$ 与目标模态嵌入 $\mathbf{e}_n$ 输入延迟头，输出有界逐 token 偏移：
$$\tau_{m \to n}^i = \tau_{\max} \cdot \text{tanh}\big(h_\delta([\mathbf{z}_m^i; \mathbf{e}_n])\big)$$
目标通过以 $t_i + \tau^i$ 为中心的高斯核软采样 Gather：
$$a_{ij} = \text{softmax}_j\!\left(-\frac{1}{2}\Big(\frac{t_j - t_i - \tau^i}{\sigma}\Big)^2\right), \quad \tilde{\mathbf{z}}_n^i = \sum_j a_{ij} \bar{\mathbf{z}}_n^j$$
软核保证 $\tau^i$ 可微分。每位置的权重由对齐熵决定（峰值对齐权近 1，模糊对齐权重趋 0）。跨模态预测损失：
$$\mathcal{L}_{\text{cross}} = \frac{\sum_i w_i \mathcal{H}(\text{LN}(\hat{\mathbf{z}}_{mn}^i), \text{LN}(\tilde{\mathbf{z}}_n^i))}{\sum_i w_i + \epsilon}$$
延迟生理锚定监督（R 峰→第一心音用于 ECG-PCG，脉搏到达时间用于 ECG-PPG）：
$$\mathcal{L}_{\text{delay-sup}} = \frac{1}{|\mathcal{T}|} \sum_i \mathcal{H}(\tau^i, \tau_{\text{anchor}}^i)$$

**辅助目标与总损失**：
- 跨模态 VICReg $\mathcal{L}_{\text{state}}$：拉配对模态入同空间并防坍缩
- 相位预测 $\mathcal{L}_{\text{phase}}$：从锚点预测心动周期内相位
- 总损失：$\mathcal{L} = \lambda_{\text{intra}}\mathcal{L}_{\text{intra}} + \lambda_{\text{cross}}\mathcal{L}_{\text{cross}} + \lambda_{\text{delay}}\mathcal{L}_{\text{delay-sup}} + \lambda_{\text{state}}\mathcal{L}_{\text{state}} + \lambda_{\text{phase}}\mathcal{L}_{\text{phase}}$

**训练流程**：Stage I（300K steps，batch=196，lr=1.2e-4）→ Stage II（200K steps，batch=64，lr=6e-5），Stage II 中以概率 $p$ 采样配对数据，其余继续采样单模态数据。动量系数从 0.998 线性增至 0.9999。

## 实验与结果
**数据集**：
- 预训练单模态：MIMIC-IV-ECG（800K 条，12 导联，500Hz）、PPG-EXT（4.6M 条，125Hz）、BMD-HS（3,436 条，4kHz）
- 预训练配对：PPG-EXT + VitalDB（ECG-PPG 同步）、EPHNOGRAM（ECG-PCG 同步）、SensSmartTech（三模态）

**评估协议**：冻结编码器 + 线性探针，患者隔离划分（patient-disjoint），涵盖 25 个下游任务（ECG 6 数据集×3 标签比例=18 设置；PPG 6 分类+11 回归；PCG 2 任务）。

**主要结果**：
- **PPG**：分类平均 AUROC 80.4（最强基线 AnyPPG 72.2，**+8.2**）；回归平均 MAE 9.1（AnyPPG 10.9，**-1.8**）。DaLiA Activity 达 90.3（AnyPPG 79.0），MIMIC AF 达 97.7（AnyPPG 93.3）。
- **ECG**：平均 AUROC 90.9（最强自监督 MoCo-v3 68.5，**+15.5**），在所有 18 个设置中接近或超越依赖临床文本/监督标签的模型（如 MERL 78.1、D-BETA 85.9）。
- **PCG**：CirCor 杂音检测 97.9±0.1（AudioMAE 79.1，**+18.8**）；CinC2016 异常心音 66.8±0.4（AudioMAE 62.4）。

**可视化证据**：t-SNE 显示 Stage I 后模态间形成分离簇（模态 silhouette=0.121），Stage II 后完全交织（silhouette=−0.006），证实跨模态对齐有效。注意力图显示共享编码器在各模态上聚焦生理有意义区域（ECG 的 QRS 波群、PPG 的上升支、PCG 的心音 burst）。

## 相关工作脉络
- **TS2Vec / CoST**：通用时间序列自监督对比方法，但未显式建模多模态观测同一物理过程的结构性对应关系。
- **ST-MEM**：ECG 掩码预测自监督模型，仅针对单一模态；本文在其基础上扩展为三模态共享表征。
- **ECGFounder / ECG-FM / HeartLang / D-BETA**：利用大规模监督标签或弱临床报告的多模态 ECG 基础模型，需特权标注；CardioState-JEPA 仅用信号自监督即达到可比水平。
- **PaPaGei / AnyPPG**：PPG 对比预训练基础模型，仅处理光学信号；本文通过跨模态对齐使其从 ECG/PCG 信号中获益。
- **CLAP / AudioMAE / StethoLM**：通用音频模型适配 PCG；本文证明共享心脏表征可超越纯音频预训练方法。
- **JEPA（LeCun et al.）**：联合嵌入预测架构的原始视觉版本；本文将其迁移至多模态生理信号领域，并引入延迟对齐机制适配心跳时序偏移。

## 局限性与未来方向
- **配对数据量有限**：驱动跨模态对齐的同步多传感器记录远少于单模态数据，延迟对齐器训练于相对较少的干净心跳上。
- **PCG 预训练数据最小**：BMD-HS 仅 3,436 条，声学表征较依赖跨模态迁移，限制了纯 PCG 方向的上限。
- **ECG 的第三模态增益有限**：三模态模型与最强双模态变体在 ECG 上差距在噪声范围内，额外收益主要体现在 PPG/PCG。
- **延迟对齐依赖可靠事件检测**：极噪声记录中无法定位心跳时，退化为无监督对齐，可能降低精度。
- **仅评估冻结编码器线性探测**：未探索全量微调与更大编码器规模，下游性能上限可能更高。

## 研究启发与可借鉴点
1. **JEPA 范式迁移至多模态生理信号**：将联合嵌入预测从视觉扩展到时序生理信号，结合延迟对齐机制处理生理时序偏移，为其他多传感器领域（如呼吸-心率联合监测）提供了可复用的方法论框架。
2. **两阶段课程学习应对数据非对称**：先利用海量单模态数据建立稳固的模态内结构，再用稀缺配对数据进行跨模态对齐，同时保留单模态采样防止表征漂移——此策略可推广至其他"单模态数据多、配对数据少"的领域。
3. **生理锚定的延迟监督设计**：用可解释的生理参考（R 峰→第一心音、脉搏到达时间）对可学习延迟进行回归监督，既保证对齐的生理合理性又保留端到端学习能力，值得在时序对齐任务中借鉴。
4. **模态 silhouette 作为表征质量度量**：用 t-SNE + silhouette 定量分析跨模态对齐前后表征空间的变化，为多模态预训练提供直观的评估工具。
5. **跨模态对齐同时增强类可分性**：实验显示跨模态预训练不仅消除模态特有偏差（silhouette 从 0.121→−0.006），还提升了 PPG 上心颤分类的类可分性（silhouette 从 0.03→0.12），说明共享表征学习可同时改善不变性与判别性。

## 关键术语表
**CardioState-JEPA**：本文提出的心脏基础模型，通过生理感知的联合嵌入预测架构在 ECG/PPG/PCG 三模态上学习统一的心脏潜在表征。
**Joint-Embedding Predictive Architecture (JEPA)**：一种自监督学习范式，通过编码器分别处理上下文和目标区域并在共享潜在空间中进行预测匹配，避免原始空间重建和负样本对比。
**Modality Silhouette**：衡量多模态表征空间中不同模态样本可分离程度的指标，值越低表示模态间对齐越好；本文从 0.121 降至 −0.006。
**Delay Aligner**：可学习的跨模态时间偏移估计模块，通过 tanh 有界输出和高斯核软采样实现可微分的时间对齐。
**Intra-modal Masked Latent Prediction**：Stage I 的核心目标，对单模态序列的大块 token 进行掩码并预测其潜在编码，迫使模型学习心动周期结构。
**Delay-Supervised Cross-Modal Prediction**：Stage II 的跨模态预测目标，以 R 峰-心音/脉搏到达时间为锚点对可学习延迟进行回归监督。
**Linear Probing**：冻结预训练编码器，仅训练轻量线性分类头评估下游任务性能的标准协议。
**Momentum Encoder**：对主干网络的指数移动平均（EMA）副本，生成 stop-gradient 的预测目标以稳定训练。

## 可复现要素
- **预训练数据集**：MIMIC-IV-ECG、PPG-EXT、BMD-HS、VitalDB、EPHNOGRAM、SensSmartTech（均在 PhysioNet 公开）
- **下游数据集**：PTB-XL、CPSC 2018、CSN、WESAD、DaLiA、MIMIC AF、PPG Arrhythmia、BIDMC、UQVital、WildPPG、Sensors、UCI、BCG、CirCor、CinC2016（多数公开）
- **代码/权重**：论文未明确声明代码开源情况
- **关键超参**：ViT-B（12 层，12 头，dim=768）；Stage I：300K steps，batch=196，lr=1.2e-4（cosine decay to 1e-6）；Stage II：200K steps，batch=64，lr=6e-5（cosine decay to 1e-6）；EMA momentum 0.998→0.9999；AdamW，weight decay=0.05；默认损失权重 $\lambda_{\text{cross}}=1, \lambda_{\text{delay}}=1, \lambda_{\text{state}}=0.05$；训练设备：单卡 NVIDIA H100
- **下游评估**：冻结编码器 + 线性探针，batch=16，lr=1e-3，AdamW，100 epochs
