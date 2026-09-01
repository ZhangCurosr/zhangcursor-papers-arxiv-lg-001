---
title: "Discovering-Persistent-Behavioural-Patterns-for-Interpretabl"
source: https://arxiv.org/pdf/2608.12864v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:32:03"
field: "区块链行为分析与可解释取证"
keywords: ["blockchain forensics", "behavioral pattern discovery", "DeFi", "sentence embedding", "sequence model", "unsupervised clustering", "persistent behavior"]
innovations: ["两步嵌入（句级+序列级）结合可解释行为分析器的无监督用户行为发现框架", "通过长度泄漏审计与跨窗口持久性双重验证提升聚类可信度", "事后引入预标记与风险暴露证据的行为画像，实现可疑模式的可解释定位"]
benchmarks: ["XBlock-ETH (Ethereum blocks 16250000-16749999)", "De.Fi REKT / DefiLlama / DeFiHackLabs / ImmuneBytes 外部标签集", "BUBA / TF-IDF+SVD / Doc2Vec / Feature baseline"]
---

# 论文速读：Discovering-Persistent-Behavioural-Patterns-for-Interpretabl

## 一句话总结
本文提出了一种可扩展、应用无关的以太坊用户**持久性行为模式发现框架**：将海量交易转化为行为句子，经两步嵌入（句级→序列级）后无监督聚类，再借助可解释的行为分析器揭示普通与可疑行为的长期稳定性模式。

## 研究问题与动机
- **核心问题**：链上数据规模大、异构性强、本质上是时序的，如何将海量交易活动转化为支持**人口级行为分析**的表征，而非局限于单笔交易或个案？
- 现有方法多面向特定攻击/协议，依赖标签，或以事务/资金流为中心，无法同时支持**大规模无监督行为聚类 + 事后解释性画像 + 跨观测窗口的持久性验证**。
- 可疑证据往往稀疏、与日常活动交织、且随时间演化，现有工作缺乏发现长期、稳定、可解释行为结构的能力。
- 需要一种不依赖恶意/良性标签的框架，揭示大规模用户中普通行为与可疑行为的组织方式及其跨时间稳定性。

## 核心贡献（创新点）
- **两步嵌入框架**：先将交易/日志抽象为包含动作顺序、实体类型与 Market 信号的行为句子，再分别用句级嵌入与序列级聚合建模用户行为；与先前直接提取图/特征向量的方法本质不同。
- **系统性多阶段评估**：对句级嵌入模型（MiniLM/BGE/GTE/E5）、序列聚合（Mean/CNN/GRU/Mamba2）、训练目标（masked/nextstep）、序列长度策略、聚类算法（K-means/Leiden等）进行了全方位消融选择，平衡表示质量、聚类结构、可扩展性与长度泄漏。
- **可解释行为分析器（Behavioural Profiler）**：通过行为动因（motif）、有序例行、时空动态、实体暴露、以及分离的直接标签与暴露型可疑证据对聚类进行多层画像；与仅输出嵌入或分类结果的方法不同，本文提供人类可理解的行为语义报告。
- **跨窗口持久性分析**：在独立月度与累积多个月度窗口（1–6月）验证行为标签的稳定性，区分早期持续存在的标签（钓鱼、机器人、漏洞利用）与仅在长窗口清晰的标签（预言机操纵、rug pull）。
- **大规模实证（3000万+交易）**：在以太坊上识别出稳定币转账、DEX/AMM交易、NFT流通、铸造/空投、ENS/桥接等五大行为族，同时定位到钓鱼、机器人、漏洞利用、预言机操纵等多种可疑模式。

## 方法详解
- **数据收集与预处理**：基于 XBlock-ETH（区块 16250000–16749999），提取交易哈希、发送/接收地址、时间戳、gas、解码事件日志；通过 Infura API 解码事件，结合 CoinGecko/OpenSea 补充代币元数据；利用 heuristics 与 eth_getTransactionCount 区分合约与 EOA 用户，首轮得到 ~604 万用户。
- **行为句子构建**：将每笔交易转化为紧凑文本，保留事件别名（如 `transfer1`）、地址类型（ERC-20/ERC-721/合约/用户）、value、gas、时间戳与 Market 信号；重复同构事件进行压缩，合约/用户引用替换为紧凑 ID。用户历史构成有序句子序列 $\mathcal{T}(\mathbf{u}_r)$。
- **两步嵌入**：
  - 第一步：预训练句子模型 $g_\alpha$ 将每句映射为 $d_s$ 维向量 $\mathbf{z}^{\text{sent}}$。
  - 第二步：序列聚合模型 $h_\beta$（Mean/GRU/Mamba2/CNN）将有序句子序列聚合成单用户向量 $\mathbf{z}_r$；超过最大长度 $L$ 按策略裁剪/采样，不足则 pad+mask。
- **无监督聚类**：对用户嵌入集 $\mathcal{E}$ 做完全无监督聚类 $f:\mathbf{U}^+ \to \{C_1,\ldots,C_n\}$，不依赖任何恶意/良性标签。
- **行为分析器**：
  - **行为基础**：收集中间/最终句子序列、合约与代币集合。
  - **动因提取**：$\mu: D_{\text{sent}} \to \mathcal{M}^*$ 提取动作单元（transfer/swap/approval/mint 等），串联为有序动因序列 $M(\mathbf{u}_r)$。
  - **用户级证据**：活动画像 $\pi^{\text{act}}$、实体暴露画像 $\pi^{\text{ent}}$（对比风险实体参考）、直接标签画像 $\pi^{\text{lab}}$（仅事后引入）。
  - **聚类级证据**：RiskProf 聚合直接标签与实体暴露；TxProf 匹配 exploit 交易；MotifRank/SeqRank 对比簇内/簇外动因排序；TDProf 汇总时空与多样性统计。
  - **多标签分类**：CatProf 将证据与候选行为/风险标签比对，生成带证据强度 $\sigma$、证据 $E$ 与论证 $J$ 的标签集合 $\kappa_j$；最终渲染为可读报告 $F_j$。

## 实验与结果
- **数据集**：以太坊 XBlock-ETH（区块 16250000–16749999，约 30M+ 交易）；外部标签来自 De.Fi REKT、DefiLlama、DeFiHackLabs、ImmuneBytes 及 bot-address 数据集，共 358 个预标记地址（钓鱼 68+50、恶意合约 94、exploiter 96、受害合约 58）。
- **RQ1 选择结果**：
  - 句级最佳：**MiniLM-L6-v2**（num_bucket、no prompt、post_norm），全局相似度均值最低、Spread 最大。
  - 序列级最佳：**Mamba2 × MiniLM-L6-v2 × nextstep × L2**，max_len=32、prefix裁剪、uniform采样；相比 GRU masked 几何分数稍低但**长度泄漏最小、聚类最均衡**，钓鱼标签在 ≥2 交易过滤后集中率达 **46.9%（15/32）**。
  - 最终配置：PCA-128、cosine Full K-means、$k=22$（过滤后）。
- **RQ2 对比**：与 Feature baseline、BUBA（先前图方法）、TF-IDF+SVD、Doc2Vec 对比。本框架虽几何指标（SC/DBI/CHI）不及 BUBA，但在**长度/活动泄漏控制（$\overline{\eta_\ell^2}=0.248$ 最低）、Sep-NMI（0.790 最高）、Purity（0.260 最高）、OneCluster 恶意集中度最小**上全面领先，是**唯一在三类指标上均强**的方法。
- **行为族**（首月）：稳定币/转账/授权（~103.6 万用户）、DEX/AMM交互（~80.9万）、NFT流转（~52.8万）、铸造/领取/奖励（~46.0万）、ENS/桥接（~20.0万）。
- **可疑分级**："实质性可疑" 95.3万用户（簇 3/5/8/13/16）、"监控名单" 40.2万、"普通/暴露为主" 167.7万。
- **持久性**：授权关联转账/swap、DEX交易、流动性提供、NFT市场、铸造/领取、重复合约交互、稳定币流通等行为族在独立月与累积窗口（1–6月）反复出现；钓鱼/机器人/漏洞利用/bug exploit/前端运行（front-run）早期即稳定；预言机操纵/rug pull 在累积窗口更清晰；flash-loan、治理攻击、MEV、重放、价格操纵等仅在长窗口显现。

## 相关工作脉络
- **Forensic tracing（如 RiskTagger、ABCTRACER）**：以 LLM 或语义抽取追溯可疑资金流，但不追求大规模用户群的稳定行为组织与跨窗口持久性。
- **Chainlet Orbits（Azad et al., 2025b）**：学习有向图结构角色嵌入用于地址分类，但缺乏 post hoc 解释性画像与持久性分析。
- **Bitcoin taint-flow fingerprints（Tovanich & Cazabet, 2023）**：面向比特币资金流指纹，侧重分类与聚类，未扩展至以太坊多协议交互行为与可疑上下文解释。
- **DeFiGuard / LLM-based price manipulation detection**：针对特定攻击（价格操纵），依赖专项定义与标签，不适合无偏见的广泛行为发现。
- **BUBA（Zelenyanszki et al., 2026，作者先前工作）**：基于图神经网络的 DEX 用户行为聚类，几何质量更强但活动规模泄漏更重，且缺少行为分析器的多层解释与跨窗口持久性验证。
- **治理/生态层面研究（DAO治理、rug pull 综述）**：关注机制与数据集覆盖，而非面向百万级用户的无监督行为表征与解释框架。

## 局限性与未来方向
- 预标记集合规模有限（358 地址），仅能作为高置信度验证证据，非完整 ground truth。
- 可疑标签为**簇级解释信号**，不对单个用户做恶意/良性二分判定。
- 框架依赖事件解码、地址分类、代币元数据与 API  enrichment，跨链/跨协议泛化尚需验证。
- 未来方向：扩展至多链与多应用、与分类式/攻击特异性基线对比、在下游风险分类与预测任务上评估表征。

## 研究启发与可借鉴点
- **行为句子抽象设计**：将交易/解码事件压缩为含动作别名、实体类型与 Market 信号的紧凑文本，为区块链序列建模提供了可复用的"领域特化语言"范式。
- **双重诊断（几何质量 + 泄漏审计）**：用 $\eta_\ell^2$ 衡量序列长度/活动规模对聚类的伪相关，是防止"看起来好但实际靠 trivial signal"的有效实践，可直接迁移到其他用户序列聚类任务。
- **事后引入风险证据而非监督训练**：预标记仅在 profiling 阶段作为高置信度外部证据，保证聚类完全无偏；这一"先发现后解释"范式适用于任何"可疑样本稀缺且可能带噪声"的场景。
- **持久性作为模式可信度的验证信号**：用跨窗口的标签复现率而非单次分析结果来判定模式稳定性，可推广到社交网络、金融交易等领域的时序行为研究。
- **可借鉴的组合**：Mamba2 长序列建模 + 最小长度过滤 + PCA+K-means 的高效管线，为百万级用户时序行为聚类提供了工程参考。

## 关键术语表
- **Behavioural Sentence（行为句子）**：将单笔交易的元数据与解码事件映射为紧凑有序文本单元，保留动作顺序、实体类型与市场信号。
- **Two-Step Embedding（两步嵌入）**：先用句子模型编码单句，再用序列聚合模型（Mean/GRU/Mamba2 等）将有序句子序列合并为用户级向量。
- **Motif（行为动因）**：从行为句子中提取的紧凑行为单元，如 transfer、swap、approval、mint 等带实体信息的动作原子。
- **Behavioural Profiler（行为分析器）**：事后对聚类施加的解读模块，综合动因、时序、实体暴露与风险证据生成可读的簇级报告与多标签。
- **Persistent Pattern（持久模式）**：在独立月度或累积多个月度窗口中重复出现的行为/可疑标签及其支撑证据。
- **Length Leakage（长度泄漏）**：聚类结果被用户活动量/序列长度等 trivial 信号主导的程度，用 $\eta_\ell^2$ 等度量。
- **Sep-NMI / Purity（分离 NMI / 纯度）**：预标记类别在聚类中的分离度与聚类内正确类别占比，用于无标签设置下的弱监督诊断。
- **Exposure-based Suspiciousness（暴露型可疑）**：用户与风险实体/exploit交易的接触作为上下文证据，不同于基于预标记的直接标签判定。

## 可复现要素
- **数据集**：XBlock-ETH（区块 16250000–16749999），外部标签来自 De.Fi REKT、DefiLlama、DeFiHackLabs、ImmuneBytes、Niedermayer 等人 bot-address 数据集。**代码与数据集可应要求提供（Data availability: upon request）**，未声明公开仓库。
- **关键超参**：min\_len 过滤（all vs ≥2 transaction）、max\_len ∈ {32, 128}、裁剪策略（prefix/stratified\_window）、采样（uniform/sqrt\_len）、PCA 维度 128、k ∈ {10, 20, …, 650}、5 seeds。
- **训练设置**：batch=256，2000 GPU steps，AMP，bf16 precision。
