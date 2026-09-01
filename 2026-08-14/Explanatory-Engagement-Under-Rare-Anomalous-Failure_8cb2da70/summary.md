---
title: "Explanatory-Engagement-Under-Rare-Anomalous-Failure"
source: https://arxiv.org/pdf/2608.13063v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:35:31"
---

# 论文速读：Explanatory-Engagement-Under-Rare-Anomalous-Failure

## 一句话总结
本文构建全本地零成本实验框架，让三个开源小模型在受控工具调用失败率（p 从 0.2 降至 0.0001）下重复执行任务，系统考察模型的“解释性参与”（回复长度、自评置信度、异常识别）如何随故障渐近稀有化演变；发现**提示/ elicitation 结构**是决定检测阈值效应是否可见的首要调节变量，聚合分析会系统性掩盖真实行为形态。

## 研究问题与动机
- **核心问题**：现有研究多问“模型是否注意到异常”（二分类），本文聚焦更精细问题：当模型被嵌入低故障率工作流后，其对真实 ground-truthed 异常的解释性参与（长度、具体性、自评置信度）如何随失败概率趋近于零而演变？
- **现有方法不足**：
  1. 多数评估将提示结构视为混杂噪声直接平均，导致非单调行为被平滑为虚假的单调衰减曲线。
  2. 稀有事件研究受限于试验预算，极低发生率下常出现“零观测值空尾”，易被误判为模型行为塌陷。
  3. 现有 agentic misalignment 研究多依赖目标冲突或对抗性 prompt 强迫模型越界，缺乏对“无强迫、仅违反前提”情境下的自发解释行为的系统度量。
- **动机**：为 AI agent 在长周期自动化流程中面对罕见故障时的行为鲁棒性、可解释性及自监控动力学提供可复现的实证基准，并形式化描述该过程中的关键统计假象与行为解耦现象。

## 核心贡献（创新点）
1. **揭示 elicitation 结构是检测阈值效应的首要调节变量**：仅当采用 `immediate_forced` 条件时，原假设预测的“先升后塌”形态才显现；简单聚合五种条件会得出错误的单调下降结论，表明提示时序/结构不可作为 nuisance variable 消除。
2. **形式化定义 Empty-Tail Artifact（空尾效应）**：证明稀有事件研究中零观测值可纯粹由样本量不足导致（$P=(1-p)^n$），视觉上与真实行为塌陷完全不可区分，提出必须逐单元格核查事件计数作为标准校验流程。
3. **形式化定义 Recognition-Engagement Dissociation（识别-参与解耦）**：证明“模型是否将事件标记为异常”与“模型解释该事件的篇幅/置信度”是两个独立测量轴，同一模型在不同条件下方差极大。
4. **首次发现 llama3.1:8b 的无提示志愿自监控行为**：在 `passive_unprompted` 条件下，llama 大量自发输出结构化置信度报告并呈阶梯式衰减（100→…→0），而 qwen3:8b 与 mistral:7b 仅在固定位置输出一次模板化声明。
5. **提出 Recovery Time Equation（恢复时间方程）**：用公式 $R$ 量化模型自评置信度在遭遇偏离后回归基线的速度或失败程度，为 agent 恢复动力学提供可计算指标。

## 方法详解
- **实验 Harness**：基于 Ollama 本地部署 qwen3:8b、llama3.1:8b、mistral:7b（M1 Pro Mac），运行简单重复工具调用任务。核心自变量为故障概率 `p`（8 档：0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001）。
- **因变量**：解释长度（词数，响应幅度代理）、结构化自评置信度（0-100，提取自 `CONFIDENCE / JUSTIFICATION / EXPLANATION` 格式）、异常识别分类（manual reading: flag / normalize / mixed）。
- **5 种 Elicitation Conditions**：`immediate_forced`（失败后立即强制解释）、`grouped_runs`（批次末尾汇总）、`delayed_n_trials`（固定延迟）、`post_streak_contextual`（结合前置成功 streak 长度）、`passive_unprompted`（完全不提示）。构成从强结构压力到无压力的故意梯度。
- **Phase A（主实验）**：真随机伯努利抽样，共享确定性种子。试验上限随稀有度缩放（p≥0.05: 50次，p=0.1: 30次，p=0.2: 15次）。共 240 单元格、10,350 次试验。
- **Phase A.1（保障失败恢复设计）**：针对最稀有三档（0.001/0.0005/0.0001）主实验出现零真实故障的空尾，采用混合策略：①插入一次性合成上下文说明极长真实 streak（~1/p）；②失败前插入 8 次真实连续成功 token 块（解决 next-token predictor 对字面 token 的敏感性）；③失败后保留 15 次真实窗口以捕捉恢复动力学；其余用模板文本填充并标记 `synthetic=true`。
- **关键公式**：
  - 空尾概率：$P(\text{zero failures}) = (1-p)^n$（在 n=50 时预测出 ≈28.5/29.3/29.9 个空单元格，与实际 30/30/30 高度吻合）
  - 恢复时间：$R = \min\{t \ge 1 : |C(t_{\text{fail}}+t) - \bar{C}_{\text{pre}}| \le 5\}$，截断于 $T - t_{\text{fail}}$
  - 二次稀有度回归：$y_i = \beta_0 + \beta_1 \log_{10}(p_i) + \beta_2 [\log_{10}(p_i)]^2 + \beta_3 H_i + \varepsilon_i$，用于检验升-平台形态是否显著区别于平坦趋势（H 为 harness 指示变量）。
- **上下文与可恢复性**：CONTEXT_WINDOW_MESSAGES=80，超限处替换为合成摘要；支持 RNG fast-forward 中断精确恢复，主实验跨多次会话完成。

## 实验与结果
- **数据集与基线**：无传统 ML 基线；核心对比为不同 elicitation condition 下的行为分化，以及与原 detectability-threshold 假说的对照。
- **主要数值结果**：
  - **聚合数据**：解释长度呈近似单调下降，掩盖真实效应。
  - **immediate_forced**：解释长度在 p=0.05 达峰（28.4 词），稀有档 plateau 于 17.4-19.0 词（非假设预测的锐减）；置信度从 ~53%（p=0.2）上升至 90.0%（p=0.001）→ 72.9%（p=0.0005）→ 82.8%（p=0.0001），呈非单调爬升而非饱和。
  - **grouped_runs**
