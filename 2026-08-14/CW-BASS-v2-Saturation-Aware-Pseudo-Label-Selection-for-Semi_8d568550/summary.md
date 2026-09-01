---
title: "CW-BASS-v2-Saturation-Aware-Pseudo-Label-Selection-for-Semi"
source: https://arxiv.org/pdf/2608.12773v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:27:30"
---

# 论文速读：CW-BASS-v2-Saturation-Aware-Pseudo-Label-Selection-for-Semi

## 一句话总结
本文提出CW-BASS v2，一种面向Foundation Model教师模型（DINOv2）的饱和感知伪标签选择方法，通过一次前向传播在校验集上测量教师模型的置信集可靠性$\pi_{\text{kept}}$，以此门控切换严格阈值（教师可靠时）或自适应置信度下界（教师可靠但置信度不可靠时），在Pascal VOC上复现UniMatch V2的SOTA，并在ADE20K上提升1.5 mIoU。

## 研究问题与动机
1. **Foundation Model改变了伪标签选择的运行工况**：ResNet时代的教师模型置信度分散于$[0.3, 1]$，自适应阈值能有效筛选噪声；DINOv2教师模型置信度饱和（98%像素$\geq 0.95$），导致动态阈值的区分能力崩溃。
2. **置信度自适应阈值在饱和教师下失效**：原CW-BASS的动态阈值$\tau^{\text{dyn}}$被上界$\approx 0.34$限制，当教师置信度接近1时，几乎所有像素都超过该阈值，保留率$\rho_t \to 1$，训练被噪声淹没（确认偏差）。
3. **In-batch噪声估计存在向下偏差**：在训练集上估计伪标签噪声会低估真实误差（因学生已拟合这些像素），导致自适应阈值无依据地下调，形成正反馈循环。
4. **"自适应更好"的直觉在Foundation Model时代翻转**：传统方法追求覆盖更多像素，但在饱和教师下，被过滤的低置信度区域恰恰是错误富集区，放宽阈值只会引入噪声。

## 核心贡献（创新点）
1. **饱和感知伪标签选择门控**：提出$\pi_{\text{kept}} = \text{Pr}[\text{correct} | c \geq \tau]$作为教师置信集可靠性的直接度量，当$\pi_{\text{kept}} \geq \tau$时采用严格阈值$\tau=0.95$，否则启用自适应下界，无需人工调参即可在6个DINOv2教师上盲选正确策略。
2. **无偏校准噪声估计**：将标注集划分为训练子集$\mathcal{L}_{\text{tr}}$和校准子集$\mathcal{L}_{\text{cal}}$（$\alpha=5\%$），仅在$\mathcal{L}_{\text{cal}}$上估计类别噪声率$\hat{\varepsilon}_k$，证明了其条件无偏性（Proposition 1），打破了in-batch估计的负反馈循环。
3. **保稳定的自适应置信度下界**：提出$\tau_k^{\text{floor}} = s \cdot \bar{c}_t \cdot \frac{\mu_k}{\max_j \mu_j}$，证明其在尺度族假设下将保留率固定在小于1的分位数上（Theorem 1），防止动态阈值塌陷至全保留。
4. **机制级审计与因果链分析**：首次在批次匹配条件下分解自适应阈值在Foundation Model下的失效链路：置信度饱和→动态范围崩溃→掩码洪水→早期峰值后下降，揭示了"可靠性门控"的必要性而非经验性权衡。

## 方法详解
**整体架构**：保留CW-BASS的EMA教师-学生框架，结合UniMatch V2的双强视图（图像空间CutMix + 特征通道Dropout），总损失为：
$$\mathcal{L} = \frac{1}{2}\left(\mathcal{L}_x + \frac{1}{2}(\mathcal{L}_s + \mathcal{L}_{\text{fp}})\right)$$

**关键组件**：

1. **置信度加权交叉熵**（保留自CW-BASS）：
$$\mathcal{L}_{\text{cw}} = \frac{1}{|\mathcal{M}|} \sum_{(h,w)\in\mathcal{M}} c_{h,w}^{\gamma} \text{CE}(f_\theta(x)_{h,w}, \hat{y}_{h,w})$$
对保留像素按教师置信度的$\gamma$次方加权，此处$\gamma=1$。

2. **边界感知辅助项**（保留自CW-BASS）：
$$\mathcal{L}_s = \mathcal{L}_{\text{cw}} + \beta_b \mathbb{E}_{(h,w)\in B}[\text{CE}]$$
其中$B$为Sobel检测的伪标签边界掩码，$\beta_b=0.5$。

3. **无偏校准估计**（新增）：
每epoch在$\mathcal{L}_{\text{cal}}$上运行EMA教师，统计类别$k$在截止阈值$\tau$下的噪声率：
$$\hat{\varepsilon}_k(\tau) = 1 - \frac{n_k^c(\tau)}{n_k(\tau)}$$
其中$n_k(\tau)$为预测为$k$且置信度$\geq\tau$的像素数，$n_k^c(\tau)$为其与真值匹配的像素数。

4. **自适应置信度下界**（新增）：
$$\tau_k^{\text{floor}} = s \cdot \bar{c}_t \cdot \frac{\mu_k}{\max_j \mu_j}$$
其中$\bar{c}_t$为教师平均置信度的EMA，$\mu_k$为类别$k$的均值置信度，$s=0.95$。最终阈值为$\tau_k^{\text{final}} = \max(\tau^{\text{dyn}}, \tau_k^{\text{floor}})$。

5. **饱和感知门控**（核心）：
$$\text{rule} = \begin{cases} \text{strict } \tau=0.95, & \pi_{\text{kept}} \geq \tau \\ \text{self-adaptive floor}, & \pi_{\text{kept}} < \tau \end{cases}$$
其中$\pi_{\text{kept}} = \text{Pr}[\hat{y}=y | c \geq 0.95]$在校验集上测量，仅一次前向传播。

## 实验与结果
**数据集与协议**：
- Pascal VOC 2012（21类）：1/8 split（183张标注），匹配UniMatch V2协议
- Cityscapes：标准protocol，crop 686
- ADE20K（150类）：长尾场景，crop 518
- 骨干：DINOv2-Base（ViT-B/14）+ DPT-lite解码器

**主要结果**：
| 数据集 | 严格阈值 | CW-BASS v2（门控） | 提升 |
|--------|----------|-------------------|------|
| Pascal VOC 1/8 | 87.40 | 87.40（门控→strict） | 复现SOTA |
| Cityscapes 1/8 | 83.96 | 83.96（门控→strict） | 复现SOTA |
| ADE20K 1/8 | 49.10 | 50.58（门控→floor） | +1.5 |

**关键发现**：
- 在Pascal VOC上，所有自适应变体（Dynamic: 84.07, Floor: 82.32, Per-class: 84.07）均低于严格阈值，差异3-5 mIoU
- 自适应规则呈现"早期峰值后下降"模式（epoch 4-20达峰值），严格阈值持续上升至epoch 42
- 6个DINOv2教师（S/B/L × Pascal/Cityscapes/ADE）的$\pi_{\text{kept}}$测量显示：Pascal/Cityscapes约98%，ADE约89%，门控在此边界上正确选择策略
- In-batch估计在ADE教师上给出$\pi_{\text{kept}}^{\text{in}}=98.4\%$（误导性高），held-out估计为89.3%（真实值），差距9个百分点

## 相关工作脉络
1. **CW-BASS [12]**：原作者早期工作，在ResNet-50教师下提出置信度加权CE和边界辅助，动态阈值基于教师均值置信度；CW-BASS v2继承其框架但重写选择机制以适配Foundation Model。
2. **UniMatch V2 [15]**：当前SSSS SOTA，使用DINOv2骨干+固定严格阈值$\tau=0.95$+双强视图；本文在相同 backbone 下复现其结果，并证明严格阈值是饱和教师的正确选择。
3. **FlexMatch [6] / FreeMatch [7]**：课程伪标签与自适应阈值方法，依赖类间置信度差异调整阈值；本文指出Foundation Model下类间差异消失，此类机制失效。
4. **CAFS [26]**：基于held-out校准的类自适应阈值；本文借鉴其校准思路但改变用途——从"覆盖最大化"转为"可靠性诊断"。
5. **ENCORE [16]**：反馈驱动类阈值，基于in-batch估计；本文证明其向下偏差会导致阈值无依据下降。
6. **SemiVL [30]**：CLIP-Vision语言引导的SSSS方法，代表多模态Foundation Model路线；本文未涉及但该方向可能提供更鲁棒的教师校准。

## 局限性与未来方向
1. **主干仅限DINOv2家族**：$\pi_{\text{kept}}$门控在CLIP/SAM等其他Foundation Model上尚未验证。
2. **严格臂与自适应臂的 loss 不对等**：严格臂使用UniMatch V2的双强视图loss，而自适应臂使用CW-BASS v2的单强视图+特征扰动loss，差异约1.2 mIoU，剩余差距未完全归因于阈值规则本身。
3. **ADE20K提升为单种子结果**：+1.5 mIoU的提升在种子方差范围内，需多种子验证。
4. **校准切片成本**：$\alpha=5\%$的标注 held-out 在极低标注预算下可能不经济，$\alpha$未优化。
5. **未来方向**：在CW-BASS v2 loop内实现严格臂的对照实验；探索embedding-space auxiliary（如PixCon [45]）而非阈值工程；扩展门控到其他Foundation Model骨干。

## 研究启发与可借鉴点
1. **"测量先于自适应"**：本文展示的机制审计范式——先量化信号分布（置信度饱和），再决定策略，而非直接套用历史方法——可迁移至其他SSL方法在Foundation Model上的适配研究。
2. **可靠性度量$\pi_{\text{kept}}$的可迁移性**：该指标（条件准确率）简单、一次前向传播即可计算，可作为多种自适应选择策略的统一诊断工具。
3. **失效链分析的价值**：将"自适应失效"分解为因果链（饱和→崩溃→洪水→偏差），比单纯报告"方法A比方法B好"更具可解释性，适用于消融分析的深化。
4. **校准子集的最小化使用**：5%标注用于无偏噪声估计，成本极低但关键，该设计可在资源受限的SSL场景中推广。
5. **早期峰值作为过拟合信号**：报告EMA轨迹的best-vs-final差距（本文per-class规则EMA下降6.14 mIoU）可作为监控确认偏差的实用指标。

## 关键术语表
**Semi-Supervised Semantic Segmentation (SSSS)**：仅用少量标注图像训练分割模型，通过伪标签扩展至大量无标注数据的半监督学习范式。
**Confidence Saturation**：Foundation Model教师模型的softmax置信度分布高度集中于接近1的值（如98%像素≥0.95），导致动态阈值失去区分能力。
**$\pi_{\text{kept}}$（Confident-Set Reliability）**：给定置信度阈值$\tau$，教师模型在满足$c\geq\tau$的像素上的条件准确率，用于判定教师是否"可靠"。
**Held-Out Calibration**：将标注集划分为训练子集和校准子集，仅在后者上估计伪标签噪声，避免in-batch估计的向下偏差。
**Self-Adaptive Confidence Floor**：随教师平均置信度$\bar{c}_t$缩放的学习下界，保证保留率有上界小于1，防止动态阈值塌陷。
**Confirmation Bias**：自训练中学生反复学习自身错误预测导致的误差累积，表现为训练后期性能下降。
**Dynamic-Range Collapse**：置信度分布坍缩为单峰后，任何基于置信度分布形状的自适应阈值都无法有效筛选。

## 可复现要素
- **数据集**：Pascal VOC 2012、Cityscapes、ADE20K，均为公开数据集
- **代码与权重**：完整代码、配置、checkpoints及图表生成脚本已开源：https://github.com/psychofict/CW-BASS-v2
- **关键超参**：
  - 校准比例$\alpha=0.05$
  - 严格阈值$\tau=0.95$
  - 下界比例$s=0.95$
  - 下界动量$m=0.99$
  - 置信度加权指数$\gamma=1$
  - 边界权重$\beta_b=0.5$
  - EMA衰减：$\min(1-1/(t+1), 0.996)$
  - Backbone LR：$5\times10^{-6}$，Decoder LR：$2\times10^{-4}$
  - Crop：518（Pascal/ADE），686（Cityscapes）
  - Batch size：16
  - Epochs：60（Pascal/ADE），120（Cityscapes）
