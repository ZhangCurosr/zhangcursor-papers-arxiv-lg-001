---
title: "Equivariant-learning-of-a-transferable-three-dimensional-cla"
source: https://arxiv.org/pdf/2608.13506v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:34:39"
---

# 论文速读：Equivariant-learning-of-a-transferable-three-dimensional-classical-density-functional

## 一句话总结
本文提出 Equi-cDFT，一种三维等变学习框架，直接从正则系综平衡密度场中学习过剩自由能泛函 $F_{\mathrm{exc}}[\rho, T]$，无需自由能或化学势标签。学习到的泛函在温度、系统尺寸与统计系综间实现零样本迁移，并能无监督地涌现结构因子、状态方程、气液共存与界面行为等热力学响应。

## 研究问题与动机
- **核心问题**：经典密度泛函理论（cDFT）提供可复用的变分框架，但 realistic 三维流体的相互作用过剩自由能泛函 $F_{\mathrm{exc}}[\rho]$ 未知，构造同时保持高准确度与变分一致性的泛函是核心挑战。
- **现有方法局限**：早期 ML 近似多局限于平面或二维非均匀体系，难以直接应用于三维溶剂化、成核与非平面限域问题。
- **标签依赖困境**：先前神经 cDFT 通常需显式自由能标签、化学势标注或 pair-correlation 匹配，而正则系综 MD 数据仅提供密度场，缺乏热力学势能面信息。
- **对称性挑战**：三维笛卡尔网格下的标量泛函及其泛函导数必须在平移、旋转与反射下保持正确的不变/等变行为，传统卷积或逐点 MLP 难以严格保证。

## 核心贡献（创新点）
1. **直接学习显式标量过剩自由能泛函**：以广延标量形式学习 $F_{\mathrm{exc}}[\rho, T]$，而非独立逼近 $c^{(1)}$ 或势-密度映射；本质区别在于一阶与二阶泛函导数由同一标量自动微分获得，天然保证热力学响应的一致性。
2. **基于 CACE 的立方群 $O_h$ 不变特征编码**：将笛卡尔原子簇展开适配至三维密度网格，构造旋转/反射不变的局部环境特征；区别于以往 CNN 或平面假设方法，严格保留了三维晶格的 48 阶点群对称性。
3. **局域化学势平衡驱动的无监督训练**：利用平衡态下 $\mu^{\mathrm{loc}}(\mathbf{r})$ 为空间常数的物理约束构造损失 $\mathcal{L}^{\Delta\mu}$，无需自由能或化学势标签；区别于需外部标注或逐样本隐式拟合化学势的先前工作。
4. **跨温度/尺寸/系综的零样本迁移与涌现预测**：单一泛函在未见温度、更大系统体积及正则→巨正则切换下均保持准确；区别于仅拟合给定外场下密度分布的回归模型，结构因子、状态方程、气液共存与界面展宽均为无监督涌现。

## 方法详解
- **泛函广延分解**：$F_{\mathrm{exc}}[\rho, T] = \Delta V \sum_{g} \rho_{g} \, a_{\mathrm{exc}}(\chi_{g}, T)$，共享局部自由能映射 $a_{\mathrm{exc}}$ 保证广延性并支持跨尺寸迁移。
- **对称不变特征构造**：以网格点 $g$ 为中心、截断半径 $q_{\mathrm{cut}}=3$ 的球邻域内（122 个非中心体素）计算笛卡尔矩 $A_{\ell}(\mathbf{r}_{g}) = \sum_{|\mathbf{q}|\le q_{\mathrm{cut}}} \rho_{g+\mathbf{q}} \, q_{x}^{\ell_{x}} q_{y}^{\ell_{y}} q_{z}^{\ell_{z}}$（总次数 $|\boldsymbol{\ell}|\le 3$）。对 $O_h$ 群 48 个操作求和得到不变量 $B_{K}^{(\nu)}$，保留至关联阶 $\nu=2$，共 15 个不变特征；与中心密度、温度组成 17 维输入。
- **读出头（Readout）网络**：基线分支 MLP（$2 \to 32 \to 16 \to 1$）处理局部密度与温度，CACE 分支 MLP（$17 \to 32 \to 16 \to 1$）处理不变特征，均使用 SiLU 激活，总可训练参数 1,762。
- **训练损失**：定义无量纲局域化学势 $\mu_{g}^{\mathrm{loc}} = \ln(\Lambda^3 \rho_g) + \beta V_{\mathrm{ext},g} - c_g^{(1)}$，其中 $c_g^{(1)} = -\frac{\beta}{\Delta V}\frac{\partial F_{\mathrm{exc}}}{\partial \rho_g}$ 通过自动微分获得。未知常数化学势被解析消去为空间均值，单场损失为 $\mathcal{L}^{\Delta\mu} = \sum_{g} \left| \mu_{g}^{\mathrm{loc}} - \overline{\mu^{\mathrm{loc}}} \right|^2$（仅对 $\rho_g > 10^{-3}\sigma^{-3}$ 的可靠采样体素求和）。
- **推理与求解**：给定 $V_{\mathrm{ext}}$ 与 $T$，在固定 $N$ 或固定 $\mu$ 下对网格密度最小化巨势 $\Omega[\rho]$；$c^{(2)}$ 由 Hessian 自动获得，自动满足 $g \leftrightarrow g'$ 互易性。

## 实验与结果
- **数据集**：截断位移 Lennard-Jones (LJTS) 流体，正则系综 MD 生成，温度覆盖 $T=0.625$–$1.8$；$16^3$ 网格（间距 $0.5\sigma$），共 10,957 个完整密度场，其中 8,427 训练、938 验证、1,592 测试（$T=0.7,1.1,1.5$  Held-out）。
- **评估基线与参考**：直接 NVT/NVE MD、GCMC 参考、文献 EOS（Thol et al. [25]）与共沸数据（Vrabec et al. [26]）。
- **核心结果**：
  - **前向/反向测试**：Held-out 温度下外场推断与固定 $N$ 密度求解均与 MD 高度一致（Fig 5a/b）。
  - **尺寸迁移**：训练于 $L=8\sigma$，零样本应用于 $L=12\sigma$（体积放大 3.375 倍）的 90 个测试场，密度预测保持准确（Fig 5c）。
  - **系综迁移**：正则训练模型成功预测 GCMC 场的化学势差，仅需每温度一个加性校准（Fig 5d）。
  - **热力学涌现**：无监督复现结构因子 $S(k)$（Fig 2a）、压缩率路径状态方程与 van der Waals 回环（Fig 2b）、气液共存曲线与临界点 $T_c^{\mathrm{cDFT}}=1.09$、$\rho_c^{\mathrm{cDFT}}=0.31\sigma^{-3}$（Fig 2c）、液-气界面渐进展宽（Fig 2d）。
  - **三维应用**：两胶体间溶剂桥形成/破裂的非单调媒介力与 MD 一致（Fig 3）；gy
