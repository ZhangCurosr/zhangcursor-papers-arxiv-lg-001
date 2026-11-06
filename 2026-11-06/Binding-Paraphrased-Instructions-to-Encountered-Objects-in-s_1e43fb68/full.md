# Binding Paraphrased Instructions to Encountered Objects in sLLM Agents under Partial Observability

Chanyoung Kim<sup>1∗</sup>, Harim Song<sup>2∗</sup>, Byoungchan Ko<sup>2</sup>, Keonyoung Lee<sup>2</sup>, Myungwon Lee<sup>2</sup>, Dahuin Jung<sup>2†</sup>

<sup>1</sup>School of Computer Science and Engineering, Soongsil University, Seoul, Korea <sup>2</sup>Department of Artificial Intelligence, Chung-Ang University, Seoul, Korea dahuinjung@cau.ac.kr

## Abstract

Reliable deployment of LLM agents requires grounding diverse user expressions in their environments. Although RL-finetuned smaller large language model (sLLM) agents rival larger prompted LLMs on embodied tasks, their robustness to instruction paraphrases remains unclear. We introduce ALFWorld-Para to evaluate this robustness under controlled paraphrases, revealing 21–52% declines in success rates (SRs), predominantly driven by object synonyms. Despite retaining knowledge of synonym referents, agents often fail to take the corresponding objects even after observing them, exposing a gap between referential knowledge and object binding. Partial observability further complicates binding, as referents may remain unseen or have already been encountered under different labels. To address this challenge, we propose BINDIntent, a trainingfree framework that probabilistically binds referring expressions to observed labels using semantic proximity and estimated persistence against unseen candidates. Without additional training or LLM calls, BINDIntent improves synonym-paraphrase success of RL-finetuned sLLMs by 17.5–49.1 percentage points (pp) with only 0.4 ms of CPU overhead per step. It also raises GiGPO-7B’s overall SR from 61.8% to 77.4%, surpassing a prompted 24B LLM. Code and benchmark are available at https:// github.com/cau-hai-lab/ALFWorld-Para.

## 1 Introduction

Smaller large language models (sLLMs; fewer than 10B parameters) (Belcak et al., 2025) have gained traction as cost-efficient backbones for LLM agents. In partially observable embodied environments, sLLM agents fine-tuned with reinforcement learning (RL) can rival or outperform larger prompted LLMs (Feng et al., 2025; Xue et al., 2026), positioning them as viable deployment candidates. Yet deployment exposes agents to linguistic variability: as illustrated in Figure 1, users articulate the same intent in different ways and refer to identical objects using different terms (Ervin-Tripp, 1976; Furnas et al., 1987). Robustness to instruction paraphrases is therefore important for reliable deployment. Despite this need, existing agents are trained and evaluated on instructions drawn from a set of fixed templates (Shridhar et al., 2021). As summarized in Table 1, prior robustness studies likewise focus predominantly on system-side perturbations, including prompt templates and action interfaces. Consequently, the robustness of sLLM agents, particularly those fine-tuned with RL, to paraphrased user instructions remains largely unexplored.

![](images/ef1e159c8122097c5762fdb95d61e81a79fcdca140fda186cbec7ee7ab7c68a6.jpg)  
Figure 1: Under partial observability, an sLLM agent that succeeds on a templated instruction fails on its paraphrased counterpart.

To evaluate this robustness, we introduce ALFWorld-Para, which varies only goal instruction phrasing while preserving the underlying task. It independently paraphrases action and object expressions to isolate their contributions to task failure, following LIBERO-Para (Kim et al., 2026). RL-finetuned sLLM agents and promptingbased LLMs exhibit 21–52% declines in SR on ALFWorld-Para, largely eroding RL gains. This degradation stems predominantly from object synonym substitutions. Yet 7B models identify the corresponding environment labels with nearly 80% accuracy, with negligible changes after RL finetuning. Despite this knowledge, fine-tuned agents take observed targets in over 90% of cases under original labels, compared with only 20–56% under synonyms. These findings implicate failures in binding paraphrased expressions to encountered objects. Under partial observability, agents must determine whether an exact-matching label remains unseen or its referent has already been observed under another label. Exhaustive exploration resolves this ambiguity but incurs substantial step costs and requires deviations from learned behavior.

To address this binding challenge without altering the agent’s policy, we propose BINDIntent: Binding Instruction Objects in Partially Observable Environments, a training-free framework that probabilistically commits to binding as observational evidence accumulates. BINDIntent extracts object referring expressions and identifies the closest observed label using word embedding distances (candidate distances). To account for unseen candidates, it models observed distances with a Weibull distribution and estimates persistence: the probability that the best match remains superior across unexplored units (e.g., unopened drawers). As persistence increases, BINDIntent accumulates binding probability and, upon binding, substitutes the expression with the selected label before forwarding it to the frozen agent. Requiring neither additional training nor LLM calls, BINDIntent incurs about 0.4 ms of CPU overhead per step. On RL-finetuned sLLMs, it improves SRs by 17.5–49.1 pp under synonym paraphrases and 7.0–20.6 pp overall.

## 2 Related Work

## 2.1 From Prompted LLMs to RL-Finetuned sLLM Agents

LLMs have increasingly been employed as agents in long-horizon environments characterized by partial observability (Kaelbling et al., 1998) and sparse rewards (Wang et al., 2024; Shridhar et al., 2021; Wang et al., 2022). To tackle these challenges, agent development has progressed from prompting (Yao et al., 2023; Shinn et al., 2023) and supervised fine-tuning (SFT) (Zeng et al., 2024; Chen et al., 2023) to reinforcement learning (RL) (Zhang et al., 2025). Within RL, algorithms have evolved from PPO and GRPO (Schulman et al., 2017; Shao et al., 2024) toward finer-grained credit assignment (GiGPO) (Feng et al., 2025) and strategy abstraction (StraTA) (Xue et al., 2026). These advances have enabled RL-finetuned sLLM agents to outperform larger prompted LLMs, highlighting their deployment potential (Belcak et al., 2025). However, the agents are evaluated on the same instruction templates used during training (Shridhar et al., 2021), leaving their robustness to diverse user expressions (Ervin-Tripp, 1976; Micheli and Fleuret, 2021) insufficiently examined.

<table><tr><td>Prior Work</td><td>Perturbed Component</td><td>Instruction Para. Types (#)</td><td>RL Algorithms</td><td>Mitigation</td></tr><tr><td>Yang et al., 2024</td><td>Instruction</td><td>1</td><td></td><td>Training</td></tr><tr><td>Aissi et al., 2025</td><td>System</td><td></td><td>PPO</td><td>Training</td></tr><tr><td>Fu et al., 2025</td><td>System</td><td></td><td>1</td><td>Training</td></tr><tr><td>Xia et al., 2025</td><td>System</td><td></td><td>一</td><td>Training</td></tr><tr><td>Gu et al., 2026</td><td>System</td><td></td><td></td><td></td></tr><tr><td>Ye et al., 2026</td><td>System</td><td></td><td>GRPO</td><td>Training</td></tr><tr><td>Wang et al., 2026a</td><td>System</td><td>1</td><td>GiGPO</td><td>Training</td></tr><tr><td>Kim et al., 2026†</td><td>Instruction</td><td>43</td><td></td><td></td></tr><tr><td>Ours</td><td>Instruction</td><td>29</td><td>PPO, GRPO, GiGPÓ, StraTA</td><td>Training-free</td></tr></table>

Table 1: Comparison with prior work on LLM agent robustness in terms of perturbed component, instruction paraphrase types, RL algorithms, and whether mitigation requires training. <sup>†</sup>Evaluated on VLA models.

## 2.2 Linguistic Robustness of LLM Agents

LLMs are sensitive to variations in instruction phrasing (Sclar et al., 2024; Mizrahi et al., 2024), yet this sensitivity has primarily been studied in single-turn tasks, such as question answering and mathematical reasoning (Sun et al., 2024). In multi-turn agents, instruction misinterpretations can propagate through subsequent actions, amplifying their impact. Nevertheless, as Table 1 shows, existing robustness studies focus predominantly on system-side perturbations, including prompt templates (Aissi et al., 2025), action interfaces (Fu et al., 2025; Xia et al., 2025; Gu et al., 2026), environment configurations (Ye et al., 2026), and model parameters (Wang et al., 2026a). Studies on instruction variability have examined either free-form instructions without distinguishing paraphrase types (Yang et al., 2024) or systematic action and object paraphrases limited to visionlanguage-action (VLA) models (Kim et al., 2026). Furthermore, prior studies on RL-finetuned agents each evaluated a single RL algorithm, while proposed mitigation strategies require additional training. To address these gaps, we evaluate 29 paraphrase types across five prompted LLMs and agents independently fine-tuned with four RL algorithms, and introduce a training-free mitigation framework.

## 2.3 Grounding Instruction Vocabulary to Environment Entities

Existing grounding approaches for LLM agents fall into three categories. Some map free-form actions to admissible actions through semantic similarity (Huang et al., 2022) or select executable skills based on affordances (Ahn et al., 2022), assuming all candidate actions are available upfront. Others replan using observed objects (Song et al., 2023) or reconcile object names between planning and perception modules (Kim et al., 2025), addressing vocabulary discrepancies within the agent. A third line requests assistance under uncertainty (Ren et al., 2023; Suri et al., 2026) or learns policies for both acting and querying through RL (Ramrakhya et al., 2025). These approaches share a common reliance on interactions with responsive users to resolve ambiguous instructions. Our work focuses on binding paraphrased object expressions to incrementally revealed entities in partially observable multi-turn environments without access to a global object vocabulary or additional user responses.

<table><tr><td>Type</td><td>Instruction</td></tr><tr><td>Original</td><td>put a book in sidetable</td></tr><tr><td>ACTION · Lexical Addition Same-Polarity</td><td>directly put a book in the sidetable set a book in the sidetable</td></tr><tr><td>ACTION · Structural Coordination</td><td>grasp a book and put it in the sidetable</td></tr><tr><td>Subordination</td><td>having found a book, put it in the sidetable</td></tr><tr><td>ACTION · Pragmatic</td><td>I want a book inside the sidetable</td></tr><tr><td>Need Embedded</td><td>could you put a book in the sidetable?</td></tr><tr><td>Permission</td><td>may I get a book moved into the sidetable?</td></tr><tr><td>Question</td><td>is the book in the sidetable yet?</td></tr><tr><td>Hint</td><td>the sidetable is currently missing a book</td></tr><tr><td></td><td></td></tr><tr><td>OBJECT · Lexical</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Addition</td><td>put a hardcover book in the accent side table</td></tr><tr><td>Same-Polarity</td><td>put a novel in the end table</td></tr></table>

Table 2: Paraphrase taxonomy in ALFWorld-Para. Colored italic text marks the span controlled by each axis; upright text is fixed.

## 3 ALFWorld-Para

## 3.1 Benchmark Generation

Goal instructions in ALFWorld (Shridhar et al., 2021) are instantiated from a limited set of predefined task templates with entity-specific slot fillers (e.g., put a [ITEM] in [RECEP]). Consequently, training and evaluation instructions follow identical linguistic structures, differing only in entities. This template-based structure makes ALFWorld particularly suitable for controlled paraphrase evaluation. Environments with free-form instructions, such as WebShop (Yao et al., 2022), do not permit such evaluation. We construct ALFWorld-Para by adapting the linguistically grounded paraphrase taxonomy of LIBERO-Para (Kim et al., 2026) to ALFWorld and applying it to 226 unique goal instructions from the seen split. This setting isolates instruction phrasing as the sole source of variation relative to training and aligns with the standard evaluation protocol for RL-finetuned agents (Feng et al., 2025; Xue et al., 2026). As summarized in Table 2, the taxonomy varies action and object expressions along distinct linguistic categories. Because these transformations target separate instruction spans, they can be combined to yield 29 paraphrase types and 6,554 paraphrased instructions, as detailed in Appendix A. All paraphrases are designed to preserve the original task semantics and uniquely identify their intended referents in the environment. However, under partial observability, the agent knows only the object labels it has encountered, and must therefore locate each referent through exploration. We assume that each instruction specifies the user’s intent without requiring subsequent clarification.

<table><tr><td>Model</td><td>ALFWorld</td><td>ALFWorld-Para</td><td>SR Drop (%)</td></tr><tr><td colspan="4">Pretrained (Prompting method: ReAct)</td></tr><tr><td>Mistral-S-24B</td><td>80.7</td><td>63.8</td><td>-20.9</td></tr><tr><td>Qwen3-14B</td><td>82.1</td><td>60.1</td><td>-26.8</td></tr><tr><td>Phi-4-14B</td><td>63.9</td><td>49.0</td><td>-23.3</td></tr><tr><td>Llama3.1-8B</td><td>31.4</td><td>17.8</td><td>-43.3</td></tr><tr><td>Qwen2.5-7B</td><td>54.0</td><td>29.0</td><td>-46.3</td></tr><tr><td colspan="4">RL-finetuned (Base model: Qwen2.5)</td></tr><tr><td>Base-7B</td><td>24.2</td><td>15.3</td><td>-36.8</td></tr><tr><td>PPO-7B</td><td>86.0</td><td>49.8</td><td>-42.1</td></tr><tr><td>GRPO-7B</td><td>75.8</td><td>50.9</td><td>-32.8</td></tr><tr><td>GiGPO-7B</td><td>95.8</td><td>61.8</td><td>-35.5</td></tr><tr><td>StraTA-7B</td><td>92.1</td><td>62.5</td><td>-32.1</td></tr><tr><td>Base-1.5B</td><td>5.6</td><td>3.1</td><td>-44.6</td></tr><tr><td>PPO-1.5B</td><td>56.5</td><td>27.1</td><td>-52.0</td></tr><tr><td>GRPO-1.5B</td><td>73.4</td><td>46.7</td><td>-36.4</td></tr><tr><td>GiGPO-1.5B</td><td></td><td>53.8</td><td>-36.3</td></tr><tr><td></td><td>84.4</td><td></td><td></td></tr><tr><td>StraTA-1.5B</td><td>91.6</td><td>53.2</td><td>-41.9</td></tr></table>

Table 3: Success rate (SR) (%) on ALFWorld and ALFWorld-Para. SR Drop: relative change.

## 3.2 Experimental Results

We evaluate sLLMs RL-finetuned with PPO (Schulman et al., 2017), GRPO (Shao et al., 2024), GiGPO (Feng et al., 2025), and StraTA (Xue et al., 2026) on shared Qwen2.5-1.5B/7B backbones, on ALFWorld and ALFWorld-Para. For reference, we evaluate pretrained LLMs with ReAct (Yao et al., 2023): Qwen2.5-7B (Yang et al., 2025b), Llama3.1- 8B (Grattafiori et al., 2024), Qwen3-14B (Yang et al., 2025a), Phi-4-14B (Abdin et al., 2024), and Mistral-Small-3-24B (Mistral AI, 2025). Details are in Appendix A.2.1. All models degrade substantially on ALFWorld-Para, with relative SR drops of 21–52% (Table 3). Although RL-finetuned sLLMs outperform much larger pretrained LLMs on the original ALFWorld, this advantage largely vanishes under paraphrased instructions. For instance, GiGPO-7B surpasses Qwen3-14B by 13.7 pp on ALFWorld (95.8 vs. 82.1), but the margin shrinks to 1.7 pp on ALFWorld-Para (61.8 vs. 60.1). Degradation also varies across RL algorithms, reversing their rankings. StraTA-1.5B has the highest original SR among 1.5B models but suffers a 41.9% relative drop, falling behind GiGPO-1.5B. Similarly, PPO-7B leads GRPO-7B by 10.2 pp on ALF-World but trails it on ALFWorld-Para. Thus, gains on template-based benchmarks do not necessarily transfer to paraphrased instructions. Scale does not prevent this degradation. Even the largest model, Mistral-Small-3-24B, shows a 20.9% relative drop. Pretrained 7–8B LLMs suffer larger drops of 43– 46%, although their lower original SRs make it hard to disentangle paraphrase sensitivity from baseline instability. Section 4 examines action and object paraphrases separately to identify which transformations drive these losses.

![](images/e03625f057b9786e94bf3a385c6a2486db63cb2fea40fb1ac1abd000db5fb45a.jpg)  
Figure 2: Success rate (%) of pretrained (left) and RL-finetuned (right) LLMs on original ALFWorld (None) and on ALFWorld-Para, grouped by paraphrase axis and type. Each marker is one model, with marker size indicating parameter count; horizontal bars and diamonds denote the median and mean across models.

## 4 Analysis

## 4.1 Analysis 1: Object Synonym Substitution Is the Dominant Failure Factor

Figure 2 presents SRs across paraphrase types for pretrained LLMs and RL-finetuned sLLMs. As defined in §3, Addition preserves the original object word, whereas Same-Polarity (SP) replaces it with a co-referring name. Pretrained LLMs exhibit moderate, broadly distributed degradation across types. In contrast, RL-finetuned sLLMs maintain high SRs under Others (action axis) and Addition, but collapse sharply in columns involving SP.

Along the Object axis, Addition causes only minor SR drops in both groups, with RL-finetuned sLLMs retaining their advantage over pretrained LLMs. Under SP, however, their median SR falls below that of pretrained LLMs, whose performance varies widely across models, while RL-finetuned sLLMs consistently cluster at low SRs regardless of model size or RL algorithm. This indicates that RL gains diminish when the original object word is replaced, revealing strong lexical dependence.

Along the Action axis, Others, comprising seven types that retain an action verb or explicit request, achieves SRs comparable to None in both groups. Question and Hint depart from imperative phrasing and require intent inference (Table 2), causing the largest median SR declines. RL-finetuned sLLMs retain higher median SRs but exhibit substantially greater inter-model variation than under Others. Non-imperative phrasing leads to varying degrees of degradation across RL-finetuned sLLMs, while object synonym substitution results in consistently low SRs. Combining both axes (Action×Object) compounds these effects, reducing overall SRs to or below the SP level.

Why, then, do RL-finetuned sLLMs consistently fail under SP? One hypothesis is that fine-tuning on a narrow, template-based distribution erodes their knowledge that paraphrased and original object expressions refer to the same entity. If this holds, RLfinetuned sLLMs should exhibit weaker referential knowledge than their pretrained backbones.

## 4.2 Analysis 2: sLLMs Still Know What the Paraphrased Object Refers to

To test this hypothesis, we conduct a referential knowledge probe that asks each model to match an SP-substituted expression to its original ALFWorld label from the complete label list. We define paraphrase knowledge as the proportion of correct selections (chance level: 1.2%; Appendix B.2). Figure 3 shows pretrained LLMs exhibit high paraphrase knowledge, with even Qwen2.5-1.5B substantially exceeding chance. Comparing Qwen2.5-7B and 1.5B with their RL-finetuned counterparts reveals negligible differences across RL algorithms. Thus, paraphrase knowledge is largely preserved despite fine-tuning on narrow templates and fixed action formats, contradicting the hypothesis.

![](images/76b84ddee9d9b2b4e99c60752a32d2a5ade84f03092242e88a996b359c4a0d0e.jpg)  
Figure 3: Paraphrase knowledge (%) of reference pretrained LLMs (top) and RL-finetuned sLLMs (bottom). Each RL row shows four algorithms on one backbone.

This knowledge, however, diverges from task performance in §4.1. RL-finetuned 7B models achieve nearly 80% paraphrase knowledge but typically below 20% SR under SP-paraphrased instructions. They match or outperform pretrained LLMs with comparable knowledge on original ALFWorld, yet lose this advantage under SP. Despite substantially higher paraphrase knowledge in 7B than 1.5B models, both exhibit similarly low SRs under SP. These results suggest that the degradation is not attributable to knowledge loss during RL fine-tuning. We therefore examine agent trajectories to understand why failures occur.

## 4.3 Analysis 3: Agents Encounter the Target Object but Do Not Take It

An agent must observe the target object before taking it and proceeding with the task. Figure 4 reports the taken rate: across all episodes of each object-axis paraphrase type, the fraction of steps with the target observed and not yet taken, up to and including its take, at which the agent takes it (Appendix B.3). Both groups generally take the target under None and Addition, whereas SP markedly reduces taken rates despite target observation. RL fine-tuning consistently improves taken rates over pretrained backbones under None and Addition, but not SP; some algorithms at 7B even fall below the backbone (41.3%). Under SP, taken rates also vary widely across RL algorithms despite identical backbones, with rankings reversing across model sizes. Thus, agents are less consistent in taking objects referred to by synonyms than those specified by their original labels.

![](images/d3f665bb0fb78f11f38a286f7e1347764d5c6da674998ddacd5359f2a14d861e.jpg)  
Figure 4: Taken rate (%): among steps in which the target object is observed, up to and including its first take, the proportion of those at which the agent takes it.

Agent performance involves three stages: identifying the referent of an instruction expression, observing it during exploration, and binding the expression to the observed object for execution. §4.1 identifies object synonyms as the dominant failure factor; §4.2 confirms preserved referential knowledge; and §4.3 shows agents encounter target objects and readily take them under original labels. These findings point to a difficulty in binding paraphrased expressions to encountered objects. RL fine-tuning preserves referential knowledge and taking behavior for original labels, but its gains do not generalize to this binding process.

## 5 Proposed Method

## 5.1 Motivating Observation

The findings in §4 suggest that binding failures stem from partial observability rather than a lack of referential knowledge. When all environment object labels are available, models correctly identify the intended referents (§4.2). During execution, however, agents encounter objects incrementally and must determine whether a matching label remains unseen or an observed object corresponds to the instruction (§4.3). Given our focus on instructions that sufficiently specify the user’s intent (§3), this uncertainty lies in what the agent has observed rather than in the instruction. Exhaustive exploration could remove the uncertainty but only at substantial interaction cost. Motivated by these observations, we target object synonym substitutions, the dominant source of performance degradation across model sizes and RL algorithms (§4.1), and delegate referent binding to an external framework, allowing agents to focus on task execution.

![](images/b1af184ac69f2c75a0df5550d6ade117dff363dae95fe1c1e74886efbad113f2.jpg)  
Figure 5: Overview of BINDIntent. The framework binds paraphrased object expressions to observed environment labels and forwards the rewritten instruction to a frozen agent.

## 5.2 BINDIntent: Binding Instruction Objects in Partially Observable Environments

## 5.2.1 Problem Setup and Notation

We extract object referring expressions from instruction x as a set of slots $\mathcal { S } ( x ) = \{ s ^ { ( 1 ) } , \ldots , s ^ { ( I ) } \}$ For example, “Could you set a novel in the end $\underline { { \mathrm { t a b l e ? } } } ^ { \flat }$ yields $\begin{array} { r c l } { s ^ { ( 1 ) } } & { = } & { n o \nu e l } \end{array}$ and $s ^ { ( 2 ) } \ =$ end table. Let $V _ { t }$ be the object labels observed by step $t ,$ and define a unit as an object that can contain others (e.g., a drawer or cabinet). Let $k _ { t }$ count unexplored units that may reveal new objects. For each slot, BINDIntent identifies the closest object in $V _ { t } .$ , termed the best match $v _ { t } ^ { \star }$ , as the most likely referent of the user’s expression. At each step, it decides whether to bind the slot to $v _ { t } ^ { \star }$ . Once bound, the slot is replaced with $v _ { t } ^ { \star }$ before the instruction is passed to the frozen agent. This decision depends on the confidence in $v _ { t } ^ { \star }$ given the instruction, accumulated observations, and unexplored portions of the environment. We formalize this process in three stages: candidate distance (§5.2.2), best-match persistence (§5.2.3), and persistence-driven binding (§5.2.4), as shown in Figure 5. Since slots are processed independently, we omit the slot index (i).

## 5.2.2 Candidate Distance

Given a word embedding $E ,$ we define the cosine distance between two words $w _ { 1 }$ and $w _ { 2 }$ as

$$
\begin{array} { r } { \delta ( w _ { 1 } , w _ { 2 } ) = 1 - \cos \big ( E ( w _ { 1 } ) , E ( w _ { 2 } ) \big ) . } \end{array}\tag{1}
$$

We use ConceptNet Numberbatch (Speer et al., 2017) as the default embedding, selecting the object in $V _ { t }$ with the smallest distance as the best match $v _ { t } ^ { \star }$ . Before selecting $v _ { t } ^ { \star }$ , we subtract the mean embedding of $V _ { t }$ to remove scene-specific common components (Appendix C.2). We consider only objects within a lexical allowance $\tau _ { t }$ of the slot as candidate referents, collecting their distances as

$$
D _ { t } = \big ( \delta ( s , v ) : v \in V _ { t } , \delta ( s , v ) \leq \tau _ { t } \big ) .\tag{2}
$$

No binding occurs if no candidate referent exists. Early in an episode, an object whose label exactly matches the user’s expression may remain in an unexplored unit. We thus initially permit only exact matches $( \tau _ { 0 } = 0 )$ , avoiding premature binding to differently named objects. The lexical allowance then increases by a fixed step size $\Delta \tau , \tau _ { t } = t \Delta \tau$ gradually admitting objects with different labels.

## 5.2.3 Best-Match Persistence under Partial Observability

The current best match $v _ { t } ^ { \star }$ is optimal among observed objects, yet a closer candidate may reside in an unexplored unit. To quantify this uncertainty, we define the best-match distance as $d _ { t } ^ { \star } = \delta ( s , v _ { t } ^ { \star } )$ and estimate $p _ { t } = \operatorname* { P r } [ d > d _ { t } ^ { \star } \mid D _ { t } ]$ , the probability that an unseen candidate has a greater distance. Since distances are bounded below by zero, the likelihood of encountering a closer candidate is governed by the lower tail of the distance distribution (Coles, 2001). We therefore model candidate distances with a parsimonious Weibull distribution (Weibull, 1951):

$$
\operatorname* { P r } [ d > z \mid \beta , \eta ] = \exp \left[ - ( z / \eta ) ^ { \beta } \right] ,\tag{3}
$$

where $\eta$ and $\beta$ denote the scale and shape parameters, respectively. Given observed candidate distances $D _ { t }$ and the prior $\pi ( \beta , \eta ) \propto 1 / ( \beta \eta )$ , we infer the posterior $\pi ( \beta , \eta \mid D _ { t } )$ . The posterior predictive probability that an unseen candidate has a greater distance than the current best match is

$$
p _ { t } = \iint e ^ { - ( d _ { t } ^ { \star } / \eta ) ^ { \beta } } \pi ( \beta , \eta \mid D _ { t } ) d \beta d \eta ,\tag{4}
$$

<table><tr><td rowspan="2">Model</td><td colspan="4">Baseline</td><td colspan="4">Greedy</td><td colspan="4">Self-clarification</td><td colspan="4">BINDIntentsBERT</td><td colspan="4">BINDIntent (ours)</td></tr><tr><td>Split = None</td><td>Add</td><td>Same-P</td><td>All</td><td>None</td><td>Add</td><td>Same-P</td><td>All</td><td>None Add</td><td>Same-P</td><td></td><td>All</td><td>None</td><td>Add</td><td>Same-P</td><td>All</td><td>None Add</td><td>Same-P</td><td>All</td></tr><tr><td colspan="10">Pretrained (Prompting method: ReAct)</td><td colspan="10"></td></tr><tr><td>Mistral-S-24B</td><td></td><td></td><td></td><td></td><td></td><td></td><td>34.9</td><td>70.2</td><td>68.8</td><td>55.0</td><td>64.5</td><td>71.6</td><td>70.4</td><td>44.6</td><td></td><td></td><td></td><td>70.9</td><td></td><td>66.8</td></tr><tr><td>Qwen3-14B</td><td>76.5 77.5</td><td>72.0 72.9</td><td>44.2 31.7</td><td>63.8 60.1</td><td>38.1 39.0</td><td>38.2 37.9</td><td>28.7 27.8</td><td>34.7 66.6</td><td>63.4</td><td>49.2</td><td></td><td>59.5</td><td>72.2</td><td>68.6</td><td>37.6</td><td>61.9 59.1</td><td>73.7 73.3</td><td>69.1</td><td>56.5 52.8</td><td>64.8</td></tr><tr><td>Phi-4-14B</td><td>59.9</td><td>58.0</td><td>30.1</td><td>49.0</td><td>30.5</td><td>30.1</td><td>22.0</td><td>27.4</td><td>60.3</td><td>59.1 40.8</td><td></td><td>53.2</td><td>55.3</td><td>56.5</td><td>32.5</td><td>47.8</td><td>56.6</td><td>55.9</td><td>44.9</td><td>52.3</td></tr><tr><td>Llama3.1-8B</td><td>30.2</td><td>21.6</td><td>2.8</td><td>17.8</td><td>15.2</td><td>14.5</td><td>11.7</td><td>13.7</td><td>25.1</td><td>28.0</td><td>15.8</td><td>22.9</td><td>24.5</td><td>23.0</td><td>8.7</td><td>18.5</td><td>29.2</td><td>25.7</td><td>12.4</td><td>22.2</td></tr><tr><td>Qwen2.5-7B</td><td>46.4</td><td>36.3</td><td>5.9</td><td>29.0</td><td>24.9</td><td>28.2</td><td>19.8</td><td>24.3</td><td>50.3</td><td>48.8</td><td>21.0</td><td>39.7</td><td>50.9</td><td>48.6</td><td>19.6</td><td>39.3</td><td>50.1</td><td>46.7</td><td>25.4</td><td>40.4</td></tr><tr><td colspan="10">RL-finetuned (Base model: Qwen2.5)</td><td colspan="7"></td><td colspan="3"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>10.7</td><td>20.9</td><td></td><td>9.6</td><td>17.5</td><td>22.8</td><td>22.0</td><td>8.8</td><td>17.7</td><td>22.0</td><td>21.2</td><td>12.0</td><td>18.3</td></tr><tr><td>Base-7B PPO-7B</td><td>22.5</td><td>19.7</td><td>4.4</td><td>15.3</td><td>12.6</td><td>11.5</td><td>8.3</td><td></td><td>22.3 70.3</td><td>21.1</td><td></td><td></td><td>68.9</td><td></td><td>28.4</td><td>55.6</td><td>70.6</td><td>69.2</td><td></td><td>61.7</td></tr><tr><td>GRPO-7B</td><td>73.5 68.4</td><td>67.5 66.2</td><td>10.6 19.7</td><td>49.8 50.9</td><td>41.0 41.9</td><td>37.1 40.2</td><td>29.6 28.5</td><td>35.7 36.7</td><td>68.5 63.3</td><td>65.1</td><td>31.3</td><td>52.8 52.9</td><td>71.1 67.4</td><td>67.8</td><td>30.8</td><td>54.9</td><td>67.4</td><td>67.9</td><td>46.3 43.8</td><td>59.4</td></tr><tr><td>GiGPO-7B</td><td>89.2</td><td>86.8</td><td>12.1</td><td>61.8</td><td>62.5</td><td>58.7</td><td>41.9</td><td>54.1</td><td>86.7</td><td>87.6</td><td>23.9</td><td>65.3</td><td>88.2</td><td>88.3</td><td>35.1</td><td>69.9</td><td>88.9</td><td>86.8</td><td>57.6</td><td>77.4</td></tr><tr><td>StraTA-7B</td><td>89.6</td><td>85.5</td><td>15.0</td><td>62.5</td><td>67.2</td><td>63.3</td><td>51.4</td><td>60.4</td><td>85.0</td><td>84.2</td><td>41.9</td><td>69.9</td><td>90.0</td><td>87.3</td><td>29.0</td><td>68.0</td><td>89.2</td><td>86.5</td><td>44.3</td><td>72.8</td></tr><tr><td>Base-1.5B</td><td>4.7</td><td>3.5</td><td>1.2</td><td>3.1</td><td>4.9</td><td>3.5</td><td>2.9</td><td>3.7</td><td>2.9</td><td>3.1</td><td>1.1</td><td>2.4</td><td>4.7</td><td>4.5</td><td>2.6</td><td>3.9</td><td>5.3</td><td>4.1</td><td>2.0</td><td>3.8</td></tr><tr><td>PPO-1.5B</td><td>45.7</td><td>34.6</td><td>2.9</td><td>27.1</td><td>27.2</td><td>28.0</td><td>20.2</td><td>25.1</td><td>35.4</td><td>32.8</td><td>6.1</td><td>24.4</td><td>44.8</td><td>42.6</td><td>15.2</td><td>33.8</td><td>44.0</td><td>39.0</td><td>20.4</td><td>34.1</td></tr><tr><td>GRPO-1.5B</td><td>68.6</td><td>59.0</td><td>14.7</td><td>46.7</td><td>49.5</td><td>47.0</td><td>37.0</td><td>44.3</td><td>55.5</td><td>52.8 13.5</td><td></td><td>40.1</td><td>68.6</td><td>66.4</td><td>26.7</td><td>53.4</td><td>68.2</td><td>64.5</td><td>39.1</td><td>56.9</td></tr><tr><td>GiGPO-1.5B</td><td>78.3</td><td>67.2</td><td>18.2</td><td>53.8</td><td>68.9</td><td>65.0</td><td>54.2</td><td>62.5</td><td>63.3</td><td>58.9</td><td>17.6</td><td>46.0</td><td>77.4</td><td>76.9</td><td>33.4</td><td>62.1</td><td>78.2</td><td>73.3</td><td>49.6</td><td>66.6</td></tr><tr><td>StraTA-1.5B</td><td>84.4</td><td>71.3</td><td>7.0</td><td>53.2</td><td>81.9</td><td>76.4</td><td>63.7</td><td>73.7</td><td>75.6</td><td>67.8 11.5</td><td></td><td>50.8</td><td>85.7 85.9</td><td></td><td>33.9</td><td>67.9</td><td>85.2</td><td>81.2</td><td>56.1</td><td>73.8</td></tr></table>

Table 4: Success rate (%) on ALFWorld-Para by object-axis paraphrase type: None (object unchanged), Add (Addition), Same-P (Same-Polarity), and All. Best All score per row in bold. Averaged over three seeds.

where $e ^ { - ( d _ { t } ^ { \star } / \eta ) ^ { \beta } }$ denotes the Weibull survival function at $d _ { t } ^ { \star }$ . We analytically marginalize η and numerically integrate over $\beta .$ . When fewer than c candidates are available, we adopt an exchangeabilitybased fallback to ensure stable estimation. Detailed derivations are provided in Appendix C.3.1.

## 5.2.4 Persistence-Driven Binding

The probability $p _ { t }$ represents the likelihood that the current best match remains superior to a single unseen candidate, while $1 - p _ { t }$ denotes the probability of encountering a closer candidate. For the current best match to persist, it must remain superior to all future candidates. Since the number of unseen objects is unknown, we treat each of the $k _ { t }$ unexplored units as one candidate opportunity, yielding the persistence probability $P _ { t } = p _ { t } ^ { k _ { t } }$ $P _ { t }$ increases as the best match becomes more distinct or fewer units remain unexplored. BINDIntent progressively commits to binding as this confidence increases. Whenever $P _ { t }$ exceeds its previous maximum, the excess is added to the cumulative binding probability. Thus, letting $t _ { s }$ denote the step at which slot s is bound, the cumulative binding probability equals the running maximum of $P _ { t }$ :

$$
\operatorname* { P r } [ t _ { s } \leq t ] = \operatorname* { m a x } _ { u \leq t } P _ { u } .\tag{5}
$$

Here, u ranges over the steps up to t. Binding determines whether a slot is replaced with an observed object label. For each unbound slot, a Bernoulli trial determines whether binding occurs at each step, with success probability given by the increment in cumulative binding probability divided by the remaining unbound probability mass, as detailed in Appendix C.4. Once bound, the slot uses the current best match as its replacement label at each step, as specified in Algorithm 1.

## 6 Experiments

We evaluate BINDIntent against three baselines and one embedding variant. All methods share parsed slots and observed objects, differing only in referent selection and instruction modification. Baseline retains the paraphrased instruction, leaving grounding to the LLM. Greedy uses BINDIntent’s embedding distance but immediately binds each slot to its closest observed object. Self-clarification adapts query-based disambiguation (Ren et al., 2023; Ramrakhya et al., 2025) without user interaction. At each step, the agent LLM selects a referent from observed objects or defers, replacing the expression upon selection. BINDIntent<sub>SBERT</sub> replaces relation-aware Numberbatch (Speer et al., 2017) with general-purpose SBERT embeddings (Reimers and Gurevych, 2019) to assess the contribution of relation-aware representations. We use the same LLMs as in §3 (prompts and hyperparameters in Appendix D.1.1). We report SRs by object-axis type (None, Addition, Same-Polarity) and overall (All).

## 6.1 Main Results

Table 4 shows that BINDIntent improves SP by 17.5–49.1 pp over Baseline for RL-finetuned sLLMs, while None varies by at most 3 pp and Addition is maintained or improved. Overall SR (All) increases by 7.0–20.6 pp. GiGPO-7B (77.4) and StraTA-1.5B (73.8) surpass Mistral-Small-3- 24B with and without BINDIntent (66.8 and 63.8, respectively; Appendix D.1). Base models show limited gains (+3.0, +0.7), attributable to low SR even on original instructions (24.2, 5.6; Table 3), indicating that task-solving capability remains the primary limitation. Greedy highlights the importance of accounting for unseen objects. Despite achieving higher SP on some models (StraTA-7B, GiGPO-1.5B, StraTA-1.5B), premature substitutions degrade None and Addition (e.g., GiGPO-7B None: 89.2 → 62.5). Consequently, Greedy underperforms BINDIntent in All across RL-finetuned sLLMs and falls below Baseline for every pretrained LLM. The gap is marginal for StraTA-1.5B (73.7 vs. 73.8), whose vague initial strategies sustain exploration despite premature substitutions (Appendix D.1). BINDIntent mitigates premature binding by explicitly modeling the likelihood of superior unseen candidates.

![](images/64c664c6f6ec792f9d53a41ae6cb18271f7f617b33892a6940fe5cd0e2706df3.jpg)  
Figure 6: Per-step latency introduced by each method across agent models. Note the discontinuous y-axis.

Self-clarification depends on the agent’s referential knowledge. It improves All for 7B RLfinetuned sLLMs and pretrained LLMs except Qwen3-14B (0.6 pp), but underperforms Baseline for all 1.5B models, consistent with knowledge probes (§4.2: approximately 50% for 1.5B vs. 78% for 7B). Incorrect selections degrade even None (e.g., GRPO-1.5B: 68.6 → 55.5), while each decision incurs an additional LLM call. BINDIntent<sub>SBERT</sub> improves SP, but achieves roughly half the gains of Numberbatch (e.g., GiGPO-7B: +23.0 vs. +45.5 pp) and occasionally underperforms Baseline on larger pretrained LLMs. Numberbatch, built on ConceptNet relations, appears better at separating synonyms from merely related words. BINDIntent also improves All (+3.0– 11.4 pp) and SP (+9.6–21.1 pp) across all reference pretrained LLMs. However, None declines by 2.8– 4.2 pp for models with at least 14B parameters, showing some substitution cost for models already strong on the original words.

All compared methods are executed before agent inference at each step, directly contributing to userperceived response latency. Figure 6 presents the per-step latency of each method measured independently of agent inference. Greedy, BINDIntent, and BINDIntent<sub>SBERT</sub> require no agent LLM calls, rendering their latency independent of backbone size. BINDIntent incurs only 0.38 ms per step, comparable to Greedy (0.25 ms). Substituting SBERT increases latency to 2.9 ms, primarily attributable to encoder forward passes for newly observed object names. By comparison, Self-clarification requires an agent LLM call per decision, with latency increasing from approximately 30 ms at 1.5B to 107 ms at 24B, representing 80–281× the latency of BINDIntent (details in Appendix D.2).

## 6.2 Cross-Benchmark Validation

To assess generalization beyond ALFWorld, we construct ScienceWorld-Para from Science-World (Wang et al., 2022) using the same paraphrase taxonomy and evaluate four independently RL-finetuned Qwen2.5-7B agents, with detailed analyses in Appendix E. Consistent with ALFWorld-Para, paraphrasing reduces scores by 10.1–20.3%, primarily due to object synonym substitutions (§4.1). Action-axis paraphrases yield scores of 50.4–52.2, comparable to 50.9 for original instructions, whereas SP drops to 29.9. Referential knowledge is likewise preserved after RL finetuning (69.8–73.5% vs. 71.2% pretrained; §4.2). Similarly, RL fine-tuning improves taken rates under None (19.7% → 40.3–87.7%) and Addition (16.9% → 31.5–84.5%), but not consistently under SP (16.4% → 4.2–65.9%; §4.3). Without specific tuning, our BINDIntent improves SP by 7.0– 10.2 pp and All by 2.1–3.4 pp over Baseline. Notably, its per-step overhead is only 0.78 ms, two orders of magnitude below Self-clarification, which requires an LLM call per decision.

## 7 Conclusion

This study reveals that paraphrase failures arise from referent binding under partial observability rather than knowledge loss. Such failures evade fully observable probes but emerge during execution, when agents must distinguish unseen referents from previously encountered objects. These observations suggest that template-based benchmarks and knowledge probes alone may not adequately capture this vulnerability. Robustness therefore calls for principled reasoning about unobserved referents without compromising sLLM efficiency, rather than relying solely on fine-tuning or model scaling. BINDIntent substantiates this direction by recovering substantial performance through training-free probabilistic binding.

## Limitations

This work primarily focuses on binding failures induced by object synonym substitutions, which exhibit the most pronounced performance degradation (§4.1). While BINDIntent addresses the associated binding failures, its applicability to other paraphrase types, particularly action-axis variations such as Question and Hint, remains to be investigated. Furthermore, BINDIntent operates on observed objects without modifying the agent’s exploration policy. As exploration behaviors may vary across RL-finetuned agents, their influence on binding performance has not been systematically isolated. Future research could extend BINDIntent by integrating binding with exploration strategies to facilitate the discovery of relevant objects.

## References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. 2024. Phi-4 technical report. arXiv preprint arXiv:2412.08905.

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. 2022. Do as I can, not as I say: Grounding language in robotic affordances. In Proceedings of the 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 287–318.

Mohamed Salim Aissi, Clément Romac, Thomas Carta, Sylvain Lamprier, Pierre-Yves Oudeyer, Olivier Sigaud, Laure Soulier, and Nicolas Thome. 2025. Reinforcement learning for aligning large language models agents with interactive environments: Quantifying and mitigating prompt overfitting. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 7045–7061.

Anthropic. 2026. System card: Claude Opus 5.5. https://www-cdn.anthropic.com/ fc1b44717c85dc068bc6ba5024219938094694bd/ Claude%20Opus%205.5%20System%20Card.pdf. Accessed: 2026-10-01.

Peter Belcak, Greg Heinrich, Shizhe Diao, Yonggan Fu, Xin Dong, Saurav Muralidharan, Yingyan Celine Lin, and Pavlo Molchanov. 2025. Small language models are the future of agentic ai. arXiv preprint arXiv:2506.02153.

Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. 2023. Fireact: Toward language agent fine-tuning. arXiv preprint arXiv:2310.05915.

Stuart Coles. 2001. An Introduction to Statistical Modeling of Extreme Values. Springer Series in Statistics. Springer, London.

Susan Ervin-Tripp. 1976. Is Sybil there? The structure of some American English directives. Language in Society, 5(1):25–66.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. 2025. Group-in-group policy optimization for LLM agent training. In Advances in Neural Information Processing Systems, volume 38, pages 46375–46408. Curran Associates, Inc.

Dayuan Fu, Keqing He, Yejie Wang, Wentao Hong, Zhuoma Gongque, Weihao Zeng, Wei Wang, Jingang Wang, Xunliang Cai, and Weiran Xu. 2025. Agentrefine: Enhancing agent generalization through refinement tuning. In International Conference on Learning Representations, volume 2025, pages 65185– 65204.

George W. Furnas, Thomas K. Landauer, Louis M. Gomez, and Susan T. Dumais. 1987. The vocabulary problem in human-system communication. Communications ofthe ACM, 30(11):964–971.

Google DeepMind. 2025. Gemini 3 Flash model card. https://storage.googleapis. com/deepmind-media/Model-Cards/ Gemini-3-Flash-Model-Card.pdf. Accessed: 2026-07-31.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Weizheng Gu, Chengze Li, Zhuohao Yu, Mengyuan Sun, Zhibang Yang, Wei Wang, Hongrui Jia, Shikun Zhang, and Wei Ye. 2026. What do agents learn from trajectory-SFT: Semantics or interfaces? In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 37073–37109. PMLR.

Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. spaCy: Industrialstrength natural language processing in Python. Zenodo.

Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. 2022. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 9118–9147.

Zhengbao Jiang, Frank F Xu, Jun Araki, and Graham Neubig. 2020. How can we know what language models know? Transactions of the Association for Computational Linguistics, 8:423–438.

Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra. 1998. Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2):99–134.

Chanyoung Kim, Minwoo Kim, Minseok Kang, Hyunwoo Kim, and Dahuin Jung. 2026. LIBERO-Para: A diagnostic benchmark and metrics for paraphrase robustness in VLA models. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.

Taewoong Kim, Byeonghwi Kim, and Jonghyun Choi. 2025. Multi-modal grounded planning and efficient replanning for learning embodied agents with a few examples. In Proceedings of the AAAI conference on Artificial Intelligence, volume 39, pages 4329–4337.

Venelin Kovatchev, M. Antònia Martí, and Maria Salamó. 2018. ETPC - a paraphrase identification corpus annotated with extended paraphrase typology and negation. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

Eldar Kurtic, Alexandre Noll Marques, Shubhra Pandit, Mark Kurtz, and Dan Alistarh. 2025. “give me bf16 or give me death”? accuracy-performance trade-offs in llm quantization. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 26872– 26886.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe ACM SIGOPS 29th Symposium on Operating Systems Principles.

Vincent Micheli and François Fleuret. 2021. Language models are few-shot butlers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 9312–9318.

Mistral AI. 2025. Mistral Small 3: A 24B-Parameter Latency-Optimized Model. https://mistral.ai/ news/mistral-small-3/. Accessed: 2026-10-04.

Jeff Mitchell and Mirella Lapata. 2010. Composition in distributional models of semantics. Cognitive Science, 34(8):1388–1429.

Moran Mizrahi, Guy Kaplan, Dan Malkin, Rotem Dror, Dafna Shahaf, and Gabriel Stanovsky. 2024. State of what art? a call for multi-prompt llm evaluation. Transactions of the Association for Computational Linguistics, 12:933–949.

Joakim Nivre, Marie-Catherine de Marneffe, Filip Ginter, Jan Hajic, Christopher D. Manning, Sampoˇ Pyysalo, Sebastian Schuster, Francis Tyers, and Daniel Zeman. 2020. Universal Dependencies v2:

An evergrowing multilingual treebank collection. In Proceedings ofthe Twelfth Language Resources and Evaluation Conference, pages 4034–4043, Marseille, France. European Language Resources Association.

OpenAI. 2026. GPT-5.5 system card. https: //deploymentsafety.openai.com/gpt-5-5. Accessed: 2026-10-01.

Ram Ramrakhya, Matthew Chang, Xavier Puig, Ruta Desai, Zsolt Kira, and Roozbeh Mottaghi. 2025. Grounding multimodal LLMs to embodied agents that ask for help with reinforcement learning. arXiv preprint arXiv:2504.00907.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 3982–3992.

Allen Z. Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh, Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake Varley, Zhenjia Xu, Dorsa Sadigh, Andy Zeng, and Anirudha Majumdar. 2023. Robots that ask for help: Uncertainty alignment for large language model planners. In Proceedings of the 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pages 661–682.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2024. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In International Conference on Learning Representations, volume 2024, pages 25055–25083.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations.

Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M. Sadler, Wei-Lun Chao, and Yu Su. 2023.

LLM-Planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 2998–3009.

Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. Conceptnet 5.5: An open multilingual graph of general knowledge. In Proceedings ofthe AAAI conference on artificial intelligence, volume 31.

Jiuding Sun, Chantal Shaib, and Byron Wallace. 2024. Evaluating the zero-shot robustness of instructiontuned language models. In International conference on learning representations, volume 2024, pages 48103–48141.

Manan Suri, Puneet Mathur, Nedim Lipka, Franck Dernoncourt, Ryan A. Rossi, and Dinesh Manocha. 2026. Structured uncertainty guided clarification for LLM agents. In Findings ofthe Associationfor Computational Linguistics: ACL 2026.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2024. A survey on large language model based autonomous agents. Frontiers ofcomputer science, 18(6):186345.

Pengxin Wang, Yuanzhe Li, Yuxin Ren, Huanrui Yang, and Jingdi Chen. 2026a. Learning perturbation robust policies for llm agents with stable optimization. arXiv preprint arXiv:2609.34064.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. 2022. Scienceworld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11279–11298.

Zixuan Wang, Yuchen Yan, Hongxing Li, Teng Pan, Dingming Li, Ruiqing Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. 2026b. Milestone-guided policy learning for long-horizon language agents. In Proceedings ofthe 43rd International Conference on Machine Learning (ICML).

Waloddi Weibull. 1951. A statistical distribution function of wide applicability. Journal of Applied Mechanics, 18(3):293–297.

John Wieting, Mohit Bansal, Kevin Gimpel, and Karen Livescu. 2016. Towards universal paraphrastic sentence embeddings. In International Conference on Learning Representations.

Yu Xia, Jingru Fan, Weize Chen, Siyu Yan, Xin Cong, Zhong Zhang, Yaxi Lu, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2025. Agentrm: Enhancing agent generalization with reward modeling. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19277–19290.

Xiangyuan Xue, Yifan Zhou, Zidong Wang, Shengji Tang, Philip Torr, Wanli Ouyang, Lei Bai, and Zhenfei Yin. 2026. StraTA: Incentivizing agentic reinforcement learning with strategic trajectory abstraction. In Advances in Neural Information Processing Systems.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025a. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 23 others. 2025b. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Yijun Yang, Tianyi Zhou, Kanxue Li, Dapeng Tao, Lusong Li, Li Shen, Xiaodong He, Jing Jiang, and Yuhui Shi. 2024. Embodied multi-modal agent trained by an llm from a parallel textworld. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26265–26275. IEEE.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop: Towards scalable realworld web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR).

Ziang Ye, Wentao Shi, Yuxin Liu, Yu Wang, Zhengzhou Cai, Yaorui Shi, Qi Gu, Xunliang Cai, and Fuli Feng. 2026. Look before you leap: Autonomous exploration for llm agents. arXiv preprint arXiv:2605.16143.

Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. 2024. Agenttuning: Enabling generalized agent abilities for llms. In Findings of the Association for Computational Linguistics: ACL 2024, pages 3053–3077.

Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, et al. 2025. The landscape of agentic reinforcement learning for llms: A survey. arXiv preprint arXiv:2509.02547.

A ALFWorld-Para 13   
A.1 Benchmark Generation . 13   
A.1.1 Paraphrase Dataset Generation 13   
A.1.2 Statistics of ALFWorld-Para 14   
A.1.3 Parser Stability 14   
A.1.4 Human Evaluation 14   
A.2 Experimental Results 15   
A.2.1 Setup 15   
A.2.2 Extended Results 17   
B Analysis 17   
B.1 Analysis 1: Object Synonym Substitution Is the Dominant Failure Factor . 17   
B.2 Analysis 2: sLLMs Still Know What the Paraphrased Object Refers to 17   
B.3 Analysis 3: Agents Encounter the Target Object but Do Not Take It . 18   
C Proposed Method 18   
C.1 Problem Setup and Notation 18   
C.2 Candidate Distance 20   
C.3 Best-Match Persistence under Partial Observability 20   
C.3.1 Derivation of the Weibull Posterior and Posterior Predictive Probability 20   
C.4 Persistence-Driven Binding . 22   
D Experiments 22   
D.1 Main Results 22   
D.1.1 Setup 22   
D.1.2 Ablation Study 24   
D.2 Latency 25   
E ScienceWorld-Para 27   
E.1 Benchmark Generation 27   
E.1.1 Parser Stability 27   
E.1.2 Human Evaluation 27   
E.2 Experimental Results . 28   
E.2.1 Analysis 1: Object Synonym Substitution Is the Dominant Failure Factor 29   
E.2.2 Analysis 2: sLLMs Still Know What the Paraphrased Object Refers to . 29   
E.2.3 Analysis 3: Agents Encounter the Target Object but Do Not Take It 29   
E.2.4 Main Results . 29   
F AI Assistants 30

<table><tr><td>Task Type</td><td>Goal Instruction Templates</td></tr><tr><td>Pick &amp; Place</td><td>put a {item} in {recep} put some {item} on {recep}</td></tr><tr><td>Examine in Light</td><td>look at {item} under the {lamp} examine the {item} with the {lamp}</td></tr><tr><td>Clean &amp; Place</td><td>put a clean {item} in {recep} clean some {item} and put it in {recep}</td></tr><tr><td>Heat &amp; Place</td><td>put a hot {item} in {recep} heat some {item} and put it in {recep}</td></tr><tr><td>Cool &amp; Place</td><td>put a cool {item} in {recep} cool some {item} and put it in {recep}</td></tr><tr><td>Pick Two &amp; Place</td><td>put two {item} in {recep} find two {item} and put them in {recep}</td></tr></table>

Table 5: The 6 task types in ALFWorld (Shridhar et al., 2021) and their goal instruction templates. Each task type has 2 template variants.

## A ALFWorld-Para

This section describes the paraphrase taxonomy, dataset generation process, dataset statistics, and human evaluation of ALFWorld-Para.

## A.1 Benchmark Generation

Figure 10 illustrates an example interaction trajectory in ALFWorld: at each step, the agent receives a textual observation together with the admissible actions and issues a single action toward the goal instruction. LIBERO-Para (Kim et al., 2026) partitions Same-Polarity within the lexical type into two sub-types, contextual and habitual. Contextual denotes synonyms contingent on the utterance context, whereas habitual denotes synonyms fixed by linguistic convention (Kovatchev et al., 2018). This distinction is pertinent to environments with a rich entity vocabulary, wherein an entity admits varied references depending on context. The entity vocabulary of ALFWorld, however, consists of simple common nouns such as book, mug, and knife; consequently, the synonym candidates available to the two sub-types largely overlap. We therefore consolidate the two sub-types into a single Same-Polarity type on the object axis and, for consistency, apply the same consolidation to lexical Same-Polarity on the action axis.

## A.1.1 Paraphrase Dataset Generation

Paraphrases in ALFWorld-Para are produced by factored generation, generating the object axis and the action axis independently before composing them. We use Gemini 3 Flash (Google DeepMind, 2025) as the generation model.

Action Axis. ALFWorld goal instructions are built from 12 templates, two for each of the six task types (Table 5). We replace the object and receptacle slots in each template with placeholders and define the remainder as the action frame. To the action axis prefix (Figure 11), we append a type-specific suffix to generate paraphrased frames (lexical, Figure 12; structural, Figure 13; pragmatic, Figure 14).

<table><tr><td>Axis</td><td>Category</td><td>Type</td><td>Count</td></tr><tr><td>Object</td><td>Lexical</td><td>Same-Polarity Addition</td><td>226 226</td></tr><tr><td rowspan="5">Action</td><td>Lexical</td><td>Same-Polarity Addition</td><td>226 226</td></tr><tr><td>Structural</td><td>Coordination Subordination</td><td>226 226</td></tr><tr><td rowspan="3">Pragmatic</td><td>Need Embedded</td><td>226 226</td></tr><tr><td>Permission</td><td>226</td></tr><tr><td>Question Hint</td><td>226</td></tr><tr><td colspan="2">Compositional (2 × 9 combinations)</td><td>226 226 × 18</td></tr><tr><td colspan="2">Total</td><td>6,554</td></tr></table>

Table 6: ALFWorld-Para dataset breakdown by axis, category, and type. Each of the 226 base instructions is perturbed into 11 single-perturbation types (2 object + 9 action) and 18 compositional ones (every object × action pair), yielding 226 × 29 = 6,554 instructions.

Object Axis. Given that both the item and the receptacle in an ALFWorld goal instruction constitute noun phrases, we treat both slots as targets of object axis paraphrase. For each entity, we append a type-specific suffix (Figure 15) to the object axis prefix (Figure 16) and generate multiple paraphrase candidates.

Verification. Generated candidates are filtered through an LLM-based verifier. We apply a separate verifier prompt (Figure 17) to the object axis and the action axis, accepting only candidates that satisfy criteria such as semantic preservation, cell compliance, and naturalness.

Composition. We compose verified action frames with object paraphrases to form the final instructions. Since the two axes are generated independently, composition can introduce article disagreement (e.g., “put a apple”) or number disagreement (e.g., “find two spray bottle”). Repairing these post hoc with hand-written rules is brittle across irregular forms (a/an, knife→knives, potato→potatoes). We therefore generate four inflected forms of each object paraphrase at generation time (bare, indefinite, definite, plural), and adopt a slot-filling scheme that inserts verbatim whichever form the action frame requires.

## A.1.2 Statistics of ALFWorld-Para

The seen split of ALFWorld contains 240 episodes. Since multiple episodes may share the same goal instruction under different scenes, these episodes correspond to 226 unique instructions. To establish a one-to-one correspondence between each original instruction and its paraphrases, and to prevent instructions with multiple scenes from being counted more than once, we fix a single scene per unique instruction (the first episode in the split order), yielding 226 evaluation episodes. Since there are 226 original instructions, we generate one paraphrase per instruction for each type, yielding 226 paraphrases per type and 226 × 29 in total. Table 6 presents the breakdown by axis and type.

## A.1.3 Parser Stability

Candidate slot spans are extracted from each instruction by a lexicon-free procedure. Using the part-of-speech tags of the spaCy en\_core\_web\_sm pipeline (Honnibal et al., 2020), contiguous spans are grouped into regions and trimmed of closedclass words at both edges, following the Universal Dependencies tagset (Nivre et al., 2020); no entity, verb, or task-specific word list is employed. We evaluate the extraction against the gold substitutions recorded by the paraphrase generator. Across all 6,554 instructions of ALFWorld-Para, 99.8% of the 13,108 gold slots are recovered. We further conducted a manual verification on 300 instructions sampled with stratification by paraphrase type, assessing whether each extracted region denotes the same entity as the corresponding gold slot; all 300 instructions were judged correct. Extraction is performed once per episode, requires 2.2 ms, and is shared by all binding methods; it therefore does not affect their relative cost.

## A.1.4 Human Evaluation

Annotation setup. We sampled 328 paraphrase– original pairs (5% of the full 6,554-item dataset) via stratified sampling across the 29 paraphrase types. Three annotators independently judged each pair with a binary label indicating whether the paraphrase instructs a task semantically identical to the original. Each pair was accompanied by the environment’s object reference, and annotators did not consult one another. Annotators were provided solely with the binary equivalence criterion and were not informed which label would favor the benchmark: the instructions state no expected agreement rate and give no indication that either outcome supports or undermines the dataset. Annotation was completed before any agent evaluation was conducted; hence, no annotator had seen model results at judgment time. All annotators are researchers who routinely read and write Englishlanguage publications. The full annotation prompt is shown in Figure 18.

Experimental Results. Under majority vote, 324 of 328 pairs (98.8%) were judged meaningpreserving, and 289 pairs (88.1%) received unanimous approval. We adjudicated all 39 split-vote items against the environment object reference: 30 were judged equivalent, 8 referentially ambiguous, and 1 a borderline part-for-whole substitution (fork → tine), which we conservatively counted as a failure. Combined with the unanimously approved pairs, 319 of 328 (97.3%) preserve the original task semantics. All residual errors arise from lexical substitution, such as cloth → wipe, tomato → love apple, pot → saucepan, dishsponge → scouring pad, and egg → cackleberry. We observed no cases in which the action or goal state itself was altered, indicating that the pragmatic and structural transformations in our generation pipeline preserve task semantics reliably; the residual risk is confined to the lexical axis.

Retention of flagged items. The nine flagged samples were retained in the benchmark by design. Terms such as love apple and cackleberry are attested synonyms that refer to the correct object; since they are archaic and rarely used in contemporary English, they sit at the extreme end of the lexical axis. We retain them to probe whether instruction understanding holds even at the lowfrequency tail of valid synonymy rather than only for common alternatives.

Ethics. Annotators were unpaid volunteers from our research group who agreed to have their judgments used and reported in this work; given this affiliation, they were blind to which outcome would favor the benchmark (see Annotation setup). The judged text was public and non-sensitive, and no personally identifiable information was collected. We judged this not to constitute human subjects research under our institution’s guidelines and did not seek IRB review.

<table><tr><td>Model</td><td>Codebase</td><td>Weights</td><td>Fine-tuning code</td></tr><tr><td colspan="4">ALFWorld-Para</td></tr><tr><td colspan="4">Pretrained (Prompting method: ReAct)</td></tr><tr><td>Mistral-S-24B</td><td>xxyQwQ/StraTA</td><td>RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic</td><td></td></tr><tr><td>Qwen3-14B</td><td>xxyQwQ/StraTA</td><td>Qwen/Qwen3-14B-FP8</td><td></td></tr><tr><td>Phi-4-14B</td><td>xxyQwQ/StraTA</td><td>RedHatAI/phi-4-quantized.w8a8</td><td></td></tr><tr><td>Llama3.1-8B</td><td>xxyQwQ/StraTA</td><td>RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8</td><td></td></tr><tr><td>Qwen2.5-7B</td><td>xxyQwQ/StraTA</td><td>Qwen/Qwen2.5-7B-Instruct</td><td></td></tr><tr><td colspan="4">RL-finetuned (Base model: Qwen2.5)</td></tr><tr><td>Base-7B</td><td>xxyQwQ/StraTA</td><td>Qwen/Qwen2.5-7B-Instruct</td><td></td></tr><tr><td>PPO-7B</td><td>xxyQwQ/StraTA</td><td>xxyQwQ/train-ppo-alfworld-text-qwen2.5-7b</td><td></td></tr><tr><td>GRPO-7B</td><td>xxyQwQ/StraTA</td><td>xxyQwQ/train-grpo-alfworld-text-qwen2.5-7b</td><td></td></tr><tr><td>GiGPO-7B</td><td>langfengQ/verl-agent</td><td>langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld</td><td></td></tr><tr><td>StraTA-7B</td><td>xxyQwQ/StraTA</td><td>xxyQwQ/train-strata-alfworld-text-qwen2.5-7b</td><td></td></tr><tr><td>Base-1.5B</td><td>xxyQwQ/StraTA</td><td>Qwen/Qwen2.5-1.5B-Instruct</td><td></td></tr><tr><td>PPO-1.5B</td><td>langfengQ/verl-agent</td><td>Ours (to be released after acceptance)</td><td>langfengQ/verl-agent</td></tr><tr><td>GRPO-1.5B</td><td>langfengQ/verl-agent</td><td>Ours (to be released after acceptance)</td><td>langfengQ/verl-agent</td></tr><tr><td>GiGPO-1.5B</td><td>langfengQ/verl-agent</td><td>Ours (to be released after acceptance)</td><td>langfengQ/verl-agent</td></tr><tr><td>StraTA-1.5B</td><td>xxyQwQ/StraTA</td><td>Ours (to be released after acceptance)</td><td>xxyQwQ/StraTA</td></tr><tr><td colspan="4">ScienceWorld-Para</td></tr><tr><td colspan="4">RL-finetuned (Base model: Qwen2.5)</td></tr><tr><td>PPO-7B</td><td>xxyQwQ/StraTA</td><td>xxyQwQ/train-ppo-sciworld-text-qwen2.5-7b</td><td></td></tr><tr><td>GRPO-7B</td><td>xxyQwQ/StraTA</td><td>xxyQwQ/train-grpo-sciworld-text-qwen2.5-7b</td><td></td></tr><tr><td>GiGPO-7B</td><td>ZJU-REAL/BEACON</td><td>Ours (to be released after acceptance)</td><td>ZJU-REAL/BEACON†</td></tr><tr><td>StraTA-7B</td><td>xxyQwQ/StraTA</td><td>xxyQwQ/train-strata-sciworld-text-qwen2.5-7b</td><td></td></tr></table>

Table 7: Codebase and weights per model. Each model is evaluated in the codebase it was trained with; our checkpoints follow that codebase’s default training configuration. Codebases: StraTA, verl-agent, and BEACON (Wang et al., 2026b). <sup>†</sup>The original GiGPO release does not include a ScienceWorld checkpoint.

## A.2 Experimental Results

## A.2.1 Setup

Codebase-agnostic design. Both the benchmark and the method are designed to be independent of any particular agent codebase. ALFWorld-Para is defined at the level of the environment: each episode is identified by its original ALFWorld task identifier, and the paraphrased instruction replaces the original one upon environment reset. Because the game, the admissible actions, and the success condition remain untouched, any agent that runs on ALFWorld can be evaluated on ALFWorld-Para without modification. BINDIntent likewise operates solely on the observations and the instruction, outside the policy and its prompting interface, and is applied unchanged in every codebase. Neither component therefore depends on how a codebase formats prompts, parses actions, or trains its policy.

Codebases and weights. Table 7 lists the codebase and weights employed for each model. Each model is evaluated within the codebase in which it was trained, as prompt formats, action parsers, and success checks vary across codebases. Evaluation inputs are nonetheless held identical across codebases: every run draws on the same task identifiers and the same paraphrased instructions, ensuring that a policy receives exactly the same text irrespective of its codebase. All results, including those on the original ALFWorld instructions, are obtained by running each model ourselves under this setup. For a few models, these results deviate slightly from the originally reported figures owing to differences in software versions; since every drop and every method comparison is computed between our own runs of the same model within the same codebase, such deviations leave the comparisons unaffected. We do not compare absolute SRs across codebases. Results are averaged over three random seeds, amounting to 3×6,554 episodes per configuration on ALFWorld-Para and 3 × 2,378 on ScienceWorld-Para.

Pretrained models. For the larger pretrained models we use 8-bit (FP8 or INT8 weight-andactivation) checkpoints, reflecting a practical deployment setting and our hardware budget; 8-bit quantization has been shown to preserve accuracy closely across a broad range of tasks (Kurtic et al., 2025). Qwen2.5-7B is used at its original precision. Since every comparison we report is made between runs of the same model, the choice of precision does not affect the measured drop or the effect of any binding method.

Fine-tuned models. Checkpoints that the original works did not release, namely the 1.5B models on ALFWorld and GiGPO-7B on ScienceWorld, were trained by us. At 1.5B, we train PPO, GRPO, and GiGPO with verl-agent (Feng et al., 2025), which reports reference results at this scale, and

StraTA with its official codebase (Xue et al., 2026). Since verl-agent provides no evaluated Science-World setup for GiGPO, we use BEACON (Wang et al., 2026b), a codebase built on verl-agent that includes a ScienceWorld implementation of GiGPO, solely to train and evaluate GiGPO-7B. Every model is trained with all settings identical to the default configuration of its source codebase, and the resulting checkpoints reproduce the originally reported performance where reference results exist.

Episode budget and decoding. On ALFWorld-Para, every codebase adopts the standard budget of 50 steps per episode, which likewise applies to the pretrained models run in the StraTA codebase. On ScienceWorld-Para, the budget follows the setting under which each policy was trained: 20 steps for the StraTA codebase and 30 for BEACON. Although the budgets differ, we prioritize matching the conditions under which each policy was trained, since a budget that deviates from training alters the episode distribution the policy anticipates; as all comparisons are made within a codebase, every compared run shares the same budget. For the same reason, decoding follows the default settings of each codebase. In conjunction with the identical instructions, this practice keeps both the inputs and the environment of every RL-finetuned policy as close as possible to those encountered during training, allowing any change in behavior to be attributed to the paraphrase rather than to the evaluation protocol.

Prompting of pretrained models. Pretrained models are run as ReAct agents (Yao et al., 2023) with the full interaction history included in the prompt. These models frequently produce actions that do not conform to the format required by ALF-World; in such cases, the model is re-queried, up to three times, to provide an action in the required format. These re-queries neither consume an environment step nor are appended to the interaction history. This fallback is applied uniformly to every pretrained-model run, including all binding methods and the baseline; consequently, all comparisons among pretrained models are conducted under the same condition.

Hardware and compute. RL training is run on NVIDIA A100 GPUs, two per run. On ALFWorld-Para, models with up to 7B parameters are evaluated on four NVIDIA RTX 4090 GPUs and the larger pretrained models on a single NVIDIA H100 or A100; all ScienceWorld-Para evaluations run on a single A100. Every model is served without tensor parallelism, with its weights on a single GPU, and all latency measurements are taken on an A100 (Appendix D.2). Table 8 reports the compute used: training took 427 A100 hours, and a single evaluation pass took 248.8 RTX 4090 hours and 100.2 A100 hours on ALFWorld-Para and 12.3 A100 hours on ScienceWorld-Para.

<table><tr><td>Model</td><td>RTX 4090 h A100 h</td></tr><tr><td>RL training</td><td></td></tr><tr><td>PPO-1.5B (ALFWorld)</td><td>94</td></tr><tr><td>GRPO-1.5B (ALFWorld)</td><td>50</td></tr><tr><td>GiGPO-1.5B (ALFWorld)</td><td>53</td></tr><tr><td>StraTA-1.5B (ALFWorld) GiGPO-7B (ScienceWorld)</td><td>70</td></tr><tr><td>Evaluation on ALFWorld-Para (one pass)</td><td>160</td></tr><tr><td></td><td></td></tr><tr><td>Mistral-S-24B</td><td>22.7</td></tr><tr><td>Qwen3-14B</td><td>37.1</td></tr><tr><td>Phi-4-14B</td><td>26.0</td></tr><tr><td>Llama3.1-8B</td><td></td></tr><tr><td>Qwen2.5-7B 73.1</td><td>14.4</td></tr><tr><td>Qwen2.5-1.5B 30.0</td><td></td></tr><tr><td>Base-7B</td><td></td></tr><tr><td>22.7 PPO-7B 17.3</td><td></td></tr><tr><td>GRPO-7B 20.8</td><td></td></tr><tr><td>GiGPO-7B 13.3</td><td></td></tr><tr><td></td><td></td></tr><tr><td>StraTA-7B</td><td>14.7</td></tr><tr><td>Base-1.5B 13.8</td><td></td></tr><tr><td>PPO-1.5B</td><td></td></tr><tr><td>GRPO-1.5B</td><td>11.7 8.2</td></tr><tr><td>GiGPO-1.5B</td><td>10.2</td></tr><tr><td>StraTA-1.5B</td><td>13.0</td></tr><tr><td>Evaluation on ScienceWorld-Para (one pass)</td><td></td></tr><tr><td>Qwen2.5-7B</td><td></td></tr><tr><td>Base-7B</td><td>3.0</td></tr><tr><td></td><td>2.9</td></tr><tr><td>PPO-7B</td><td>2.3</td></tr><tr><td>GRPO-7B</td><td>1.7</td></tr><tr><td>GiGPO-7B</td><td>1.7</td></tr><tr><td>StraTA-7B</td><td>0.7</td></tr></table>

Table 8: Compute in GPU hours. Models with up to 7B parameters on ALFWorld-Para are reported in RTX 4090 hours and all other rows in A100 hours, with H100 times converted at 2 A100 hours per H100 hour. Evaluation rows report a single pass (one seed) over each benchmark; some times are estimated from measured throughput.

Consistency across codebases. Because the three codebases differ in prompting, action parsing, training algorithm, and decoding, they provide independent tests of our findings. The degradation under object paraphrase and the gain of BINDIntent over the baseline appear in every model, across all three codebases and both environments. The findings and the effectiveness of BINDIntent are therefore not artifacts of a particular implementation, but properties of how RL-finetuned agents ground paraphrased instructions.

Artifacts and licenses. ALFWorld (MIT) and ScienceWorld (Apache 2.0) serve as evaluation environments; ConceptNet Numberbatch (CC BY-SA 4.0), all-mpnet-base-v2 (Apache 2.0), and spaCy en\_core\_web\_sm (MIT) are used for slot extraction and binding. Agents are trained and evaluated with verl-agent, StraTA, and BEACON (all Apache 2.0). Model weights are used under their respective licenses: Qwen2.5, Qwen3, and Mistral-Small-3 (Apache 2.0), Phi-4 (MIT), and Llama 3.1 (Llama 3.1 Community License); quantized checkpoints follow the license of their base model. Paraphrases were generated with Gemini 3 Flash under the Gemini API Terms of Service, under which Google claims no ownership of generated content. All artifacts are used solely for research evaluation, consistent with their intended use. Our benchmarks, code, and checkpoints will be released under Apache 2.0, which is compatible with the licenses of the environments, codebases, and models they build on.

Release. Upon acceptance, we will release ALFWorld-Para, ScienceWorld-Para, the evaluation harness, BINDIntent, the training scripts, and all trained checkpoints in a single repository. Each benchmark is distributed as paraphrased instructions keyed by the original task identifiers, together with their paraphrase-type labels and gold slot annotations; it runs on an unmodified installation of ALFWorld or ScienceWorld and redistributes none of their game files. The generation and verification prompts (Appendix A.1.1) are included for reproduction, the license notices of the source environments and codebases are retained, and Numberbatch vectors are obtained from their original source rather than redistributed.

## A.2.2 Extended Results

Figure 22 reports Baseline SRs on ALFWorld-Para without any mitigation applied. Since the object is left as written in the None column, reading down a column isolates the action axis, whereas reading across a row isolates the object axis. The three columns tend to decline in the order None, Addition, and Same-Polarity, and the drop into Same-Polarity is markedly larger than the gap between the first two. Row by row, the Same-Polarity cell is almost invariably the lowest of the three, the sole exception being Base-1.5B, which remains near the floor across the entire grid. As reported in § 4.1, the object axis dominates, and this pattern persists within each action type.

## B Analysis

## B.1 Analysis 1: Object Synonym Substitution Is the Dominant Failure Factor

Figure 19 unpacks Figure 2 in two ways. First, Figure 2 pools seven of the nine action types into a single Others group and shows only Question directive and Hint separately; here all nine are shown individually, grouped into lexical (Addition, Same-Polarity), structural (Coordination, Subordination), and pragmatic types (Embedded imperative, Need statement, Permission directive, Question directive, Hint). Second, Figure 2 reports action×object results only for SP; here each action type is crossed with both object paraphrases (second row of each group, Addition left and SP right). The encoding is unchanged: each marker is one model, marker size indicates parameter count, and the horizontal bar and diamond denote the median and mean across models. The expanded view confirms the pattern. Action types differ only modestly, with Question directive the weakest. Object SP, in contrast, lowers success far more than any action paraphrase, and the drop is nearly uniform across action types. Crossing with object Addition stays close to the action-only level. This holds for both pretrained and RL-finetuned models, indicating that object synonym substitution is the dominant failure factor rather than an artifact of any particular action type.

## B.2 Analysis 2: sLLMs Still Know What the Paraphrased Object Refers to

Probe set. We use the term object for both items and receptacles, the two kinds of entities an ALF-World instruction can refer to. The probe consists of every paraphrased object name that appears in ALFWorld-Para, taken verbatim from the benchmark instructions: 111 names, each paired with the entity type it replaces. Because no item is added, removed, or rewritten, the probe covers exactly the paraphrases the agents encounter during evaluation.

Fully observable lexical space. An agent perceives only the entity names it has observed thus far; its failure to bind a paraphrase may therefore stem from incomplete exploration rather than from a lack of knowledge. The probe eliminates this confound by presenting the complete entity vocabulary of the environment in every prompt. It thereby measures paraphrase knowledge with the environment’s object lexical space rendered fully observable: the model need only recognize which listed entity type the paraphrase denotes.

Prompt templates. Because estimates of a language model’s knowledge are sensitive to how the query is worded (Jiang et al., 2020; Mizrahi et al., 2024), each name is inserted into five templates that pose the same question in different forms: list choice, direct question, cloze completion, synonym question, and an agent-framed instruction (Figure 20). All templates list the 82 entity types in their environment form and require the answer to be exactly one name from the list, which eliminates free-form matching.

Decoding and scoring. We use greedy decoding for probing, with each model returning one deterministic answer per prompt. An answer is correct only if it is the gold entity type. The Paraphrase Knowledge score in Figure 3 is the accuracy over the 111 names, reported as the mean and standard deviation across the five templates.

Failure cases. To distinguish knowledge a model lacks from sensitivity to prompt wording, we regard a name as a persistent failure only if the model answers incorrectly under all five templates. Such failures cover only a small fraction of the 111 names for models of 7B parameters or more. Every failed name had been judged semantically equivalent to its target in our human evaluation (Figure 18); the failures therefore reflect gaps in model knowledge rather than errors in the benchmark.

Aside from a few names that admit more than one plausible reading, the persistent failures of the 7B models are rare idiomatic or compound expressions with an unambiguous referent, such as range eye (stoveburner), bath tissue spindle (toiletpaperhanger), and bathroom roll (toiletpaper). Common paraphrases of household objects are thus largely known to these models. RL fine-tuning does not introduce new kinds of failure: the names missed by the base model are largely the same names missed by its RL-finetuned variants, which add at most a few errors of their own. These failures are too few, and too concentrated on rare expressions, to account for the performance drop under paraphrase.

## B.3 Analysis 3: Agents Encounter the Target Object but Do Not Take It

Definition. The taken rate measures whether an agent takes the target once it is in view. The target is the item slot of the goal instruction, and any instance of its type (e.g., book 1 or book 2) counts; receptacles are excluded, since actions on them (e.g., go to, open) can also be part of task execution and are hard to attribute to binding. A step is observed if the target appears in the observation and has not yet been taken, including the step of the take itself; for two-instance tasks, this restarts after the first take. Let $N _ { e } ^ { \mathrm { o b s } }$ be the number of observed steps in episode $e ,$ and $N _ { e } ^ { \mathrm { t a k e } }$ the number of those at which the agent takes the target. For each object-axis paraphrase type (None, Addition, or Same-Polarity), the taken rate pools these counts over all episodes of that type, $\mathcal E _ { \mathrm { t y p e } } \mathrm { : }$

$$
\mathrm { T A K E N R A T E } ( \mathrm { t y p e } ) = \frac { \sum _ { e \in { \mathcal E } _ { \mathrm { t y p e } } } N _ { e } ^ { \mathrm { t a k e } } } { \sum _ { e \in { \mathcal E } _ { \mathrm { t y p e } } } N _ { e } ^ { \mathrm { o b s } } } .\tag{6}
$$

Step-level pooling. We pool at the step level because each step corresponds to a single action decision. The measure thus decomposes behavior into the individual choices of whether to take an observed target, rather than collapsing each episode into a single outcome.

## C Proposed Method

Algorithm 1 gives the per-step loop of BINDIntent and Algorithm 2 the three subroutines it calls. §C.1 defines the helper functions through which the loop reads the environment and rewrites the instruction, and §C.2–§C.4 describe one subroutine each; within them, §C.3.1 derives the shape posterior and the closed form of the posterior predictive probability $p _ { t }$

## C.1 Problem Setup and Notation

The environment is read through ENV, OBJECTS, UNITS, and EXPLORED, and the instruction through MATCH and REPLACE. AGENT denotes the frozen policy under evaluation, whose weights and decoding settings are identical across all compared methods. BOUND $( s ^ { ( i ) } ) \in \{ 0 , 1 \}$ is per-slot state: it starts at 0, is set by the Bernoulli draw of BIND, and is never cleared.

ENV. $\mathrm { E N V } \left( a _ { t - 1 } \right)$ executes the agent’s previous action and returns the resulting text observation $o _ { t } .$ BINDIntent neither selects nor modifies actions; it reads $o _ { t }$ only to update $V _ { t }$ and $R _ { t }$

```latex
Algorithm 1 BINDIntent.
Require: instruction x with slots $s ^ { ( 1 ) } , \ldots , s ^ { ( I ) }$ , horizon $T ,$ step size $\Delta \tau$ , fallback threshold $c$
1: $V _ { 0 }  \emptyset , \quad R _ { 0 }  \emptyset , \quad \tau _ { 0 }  0 ,$ $a _ { 0 } $ reset ▷ observed object labels $V ,$ , observed units R
2: $\{ \mathrm { B o U N D } ( s ^ { ( i ) } )  0 , P _ { 0 } ^ { \mathrm { m a x } ( i ) }  0 \} _ { i = 1 } ^ { I }$
3: for $t = 1 , \dots , T$ do
4: $o _ { t } \gets \mathrm { E N V } \big ( a _ { t - 1 } \big )$
5: $V _ { t }  V _ { t - 1 } \cup \mathrm { O B J E C T S } ( o _ { t } ) , \quad R _ { t }  R _ { t - 1 } \cup \mathrm { U N I T S } ( o _ { t } )$
6: $k _ { t } \gets | R _ { t } \setminus \mathrm { E X P L O R E D } ( R _ { t } ) |$ ▷ unexplored units, the chances left to observe a new object
7: $\tau _ { t } \gets t \Delta \tau$ ▷ lexical allowance widens
8: for $i = 1 , \dots , I$ do
9: $\boldsymbol v _ { t } ^ { \star ( i ) } , \boldsymbol D _ { t } ^ { ( i ) } \gets \mathrm { C A N D I D A T E D I S T A N C E } ( \boldsymbol s ^ { ( i ) } , V _ { t } , \tau _ { t } )$
10: $\bar { \mathbf { i f } } \neg \mathbf { B O U N D } ( s ^ { ( i ) } )$ then
11: $P _ { t } ^ { ( i ) } \gets \mathrm { P E R S I S T E N C E } ( s ^ { ( i ) } , v _ { t } ^ { \star ( i ) } , D _ { t } ^ { ( i ) } , V _ { t } , k _ { t } , c )$
12: $P _ { t } ^ { \operatorname* { m a x } ( i ) } \gets \mathrm { B I N D } ( s ^ { ( i ) } , P _ { t } ^ { ( i ) } , P _ { t - 1 } ^ { \operatorname* { m a x } ( i ) } )$
13: end if
14: end for
15: $x _ { t } \gets$ REPLACE $( x , \ \{ s ^ { ( i ) } \mapsto v _ { t } ^ { \star ( i ) }$ : BOUND $( s ^ { ( i ) } ) \}$
16: $a _ { t } \sim \operatorname { A G E N T } ( \cdot \mid x _ { t } )$
17: end for
```

Algorithm 2 Subroutines of BINDIntent (described in $\overline { { \ S { \bf C } . 2 - \ S { \bf C } . 4 ) } }$   
1: procedure CANDIDATEDISTANCE(s, V , τ )   
2: $v _ { t } ^ { \star } \gets \mathbf { M A T C H } ( s , V _ { t } )$ ▷ best match, tracked even after binding   
3: $D _ { t }  ( \delta ( s , v ) : v \in V _ { t } , \delta ( s , v ) \leq \tau _ { t } )$ ▷ distances of candidate referents   
4: return $v _ { t } ^ { \star } , D _ { t }$   
5: end procedure   
6: procedure PERSISTENCE(s, v<sup>⋆</sup>, D<sub>t</sub>, V<sub>t</sub>, k<sub>t</sub>, c)   
7: $n _ { t } \gets | D _ { t } |$   
8: if $n _ { t } = 0$ then   
9: return 0 ▷ no candidate referent, so no binding occurs   
10: else if $n _ { t } < c$ then   
11: return $\left( | V _ { t } | / ( | V _ { t } | + k _ { t } ) \right)$ ▷ too few candidate referents; exchangeability-based fallback   
12: end if   
13: $d _ { t } ^ { \star } \gets \delta ( s , v _ { t } ^ { \star } )$ ▷ best-match distance, raw rather than centered   
14: $\begin{array} { r } { \pi ( \beta \mid D _ { t } ) \propto \beta ^ { n _ { t } - 2 } S _ { t } ( \beta ) ^ { - n _ { t } } \prod _ { d \in D _ { t } } d ^ { \beta - 1 } } \end{array}$   
15: $p _ { t } \gets \int \pi \big ( \beta \mid D _ { t } \big ) \left( \frac { S _ { t } ( \beta ) } { S _ { t } ( \beta ) + ( d _ { t } ^ { \star } ) ^ { \beta } } \right)$ n<sub>t</sub> $d \beta \triangleright$ an unseen candidate is farther than the best match   
16: return $( p _ { t } ) ^ { k _ { t } }$ ▷ persistence over $k _ { t }$ unexplored units   
17: end procedure   
18: procedure BIND(s, $P _ { t } , P _ { t - 1 } ^ { \mathrm { m a x } } )$   
▷ $h _ { t }$ is chosen so that Pr[BOUND(s) by $t ] = P _ { t } ^ { \mathrm { m a x } }$   
19: $P _ { t } ^ { \operatorname* { m a x } } \gets \operatorname* { m a x } \bigl ( P _ { t - 1 } ^ { \operatorname* { m a x } } , P _ { t } \bigr )$ ▷ running max of persistence   
20: $h _ { t } \gets \frac { P _ { t } ^ { \mathrm { m a x } } - P _ { t - 1 } ^ { \mathrm { m a x } } } { 1 - P _ { t - 1 } ^ { \mathrm { m a x } } }$ ▷ this step’s increase, as a fraction of the probability still unbound   
21: BOUND(s) ∼ Bernoull $\mathrm { i } ( h _ { t } )$   
22: return $P _ { t } ^ { \mathrm { m a x } }$   
23: end procedure

OBJECTS & UNITS. ${ \mathrm { O B J E C T S } } { \left( o _ { t } \right) }$ returns the entity types mentioned in $o _ { t }$ with instance identity removed. Two drawers therefore contribute a single entry, drawer, and $| V _ { t } |$ counts distinct labels rather than physical objects. $\mathrm { U N I T S } \big ( o _ { t } \big )$ returns the places whose contents can be inspected, as individual instances, with $R _ { t }$ counting those two drawers twice. The asymmetry is deliberate: an instruction names a label, a unit’s own label included, whereas the agent opens one particular unit. Both sets are taken from what the environment exposes rather than from a list fixed in advance.

EXPLORED. A unit is EXPLORED once its contents have been revealed to the agent, including the case where it turns out to be empty; reaching a unit without seeing inside it does not count. Exploration is never undone, and $k _ { t }$ thus counts the known places whose contents remain unseen, that is, the chances left for a not-yet-observed object to appear.

REPLACE. REPLACE substitutes, for each bound slot, its current best match in place of the original mention, within the instruction and on whole words; unbound slots keep their original wording. Because $v _ { t } ^ { \star }$ is re-evaluated at every step, a bound slot continues to follow the best match as new objects are observed. Binding decides whether a mention is rewritten; MATCH decides into what.

## C.2 Candidate Distance

Embedding lookup. Multi-word expressions are embedded via their Numberbatch phrase entry when one exists (e.g., end\_table) and otherwise as the mean of their word vectors, a simple yet strong composition for short phrases (Mitchell and Lapata, 2010; Wieting et al., 2016). Concatenated environment labels (e.g., sidetable) are first split into their constituent words.

CANDIDATEDISTANCE selects $v _ { t } ^ { \star }$ and collects into $D _ { t }$ the distances of the candidate referents, the labels lying within the lexical allowance $\tau _ { t } \left( \mathrm { E q . } 2 \right)$ Labels observed together in a scene tend to share a common semantic component $( \mathrm { e . g . }$ , a household context), which inflates their similarity to any query. MATCH removes this component before ranking. With $\mu _ { t }$ the mean embedding of the labels in $V _ { t }$ and $\tilde { E } ( a ) = ( E ( a ) - \mu _ { t } ) / \| E ( a ) - \mu _ { t } \|$ , it returns

$$
v _ { t } ^ { \star } = \underset { v \in V _ { t } } { \arg \operatorname* { m a x } } \ \tilde { E } ( s ) ^ { \top } \tilde { E } ( v ) .\tag{7}
$$

Centering is used only for this selection; the dis-

tances collected in $D _ { t }$ are the uncentered $\delta$ of Eq. 1, which does not depend on $V _ { t }$

## C.3 Best-Match Persistence under Partial Observability

PERSISTENCE is called only while the slot is unbound. It turns $D _ { t }$ into $P _ { t }$ , the probability that $v _ { t } ^ { \star }$ remains the best match once the $k _ { t }$ unexplored units have been opened, and it separates three cases. With no candidate referent inside the allowance there is nothing to bind $\mathbf { t o } ,$ , and the procedure returns 0. With fewer than $c$ of them the distance distribution cannot be fit, and $P _ { t }$ is replaced by a countbased estimate: if the referent is equally likely to be any of the $| V _ { t } |$ observed labels or to lie behind one of the $k _ { t }$ unexplored units, it has already been observed with probability $| V _ { t } | / ( | V _ { t } | + k _ { t } )$ . Otherwise the Weibull of Eq. 3 is fit to $D _ { t }$ , giving the posterior predictive probability $p _ { t }$ that an unseen candidate lies farther than $d _ { t } ^ { \star }$ (Eq. 4); treating each unexplored unit as one independent chance for such a candidate yields $P _ { t } = p _ { t } ^ { \mathbf { \hat { k } } _ { t } }$ . The derivation is in $\ S { \bf C } . 3 . 1$

## C.3.1 Derivation of the Weibull Posterior and Posterior Predictive Probability

We expand the computation behind $\operatorname { E q . } \left( 3 \right) - ( 4 )$ of the main text. Dropping the time and slot indices, write $D = \{ d _ { 1 } , \ldots , d _ { n } \}$ for the observed candidate referent distances, $d ^ { \star }$ for the current best-match distance, and $\begin{array} { r } { S ( \beta ) : = \sum _ { i = 1 } ^ { n } d _ { i } ^ { \beta } } \end{array}$

Step 1: Likelihood. The density corresponding to the Weibull survival function $\operatorname* { P r } ( d \ > \ z )$ $\beta , \eta ) = \exp [ - ( z / \eta ) ^ { \beta } ]$ of $\operatorname { E q . } \left( 3 \right)$ is

$$
\begin{array} { r l r } & { } & { f ( d \mid \beta , \eta ) = \displaystyle \frac { \beta } { \eta } \left( \frac { d } { \eta } \right) ^ { \beta - 1 } \exp \left[ - ( d / \eta ) ^ { \beta } \right] } \\ & { } & { = \beta d ^ { \beta - 1 } \eta ^ { - \beta } \exp \left[ - d ^ { \beta } \eta ^ { - \beta } \right] . } \end{array}\tag{8}
$$

If the elements of D are conditionally i.i.d. given

(β, η), then

$$
\begin{array} { r l } { p ( D \mid \beta , \eta ) = \displaystyle \prod _ { i = 1 } ^ { n } f ( d _ { i } \mid \beta , \eta ) } \\ { = \displaystyle \prod _ { i = 1 } ^ { n } \left[ \beta \partial _ { i } ^ { \alpha _ { i } - 1 } \eta ^ { - \beta _ { i } } e ^ { - \alpha _ { i } ^ { \beta _ { i } - \eta } } \right] } \\ { = \beta ^ { n } \left( \displaystyle \prod _ { i = 1 } ^ { n } d _ { i } ^ { \beta - 1 } \right) \eta ^ { - n \delta } } \\ { \times \exp \left[ - \eta ^ { - \beta } \displaystyle \sum _ { i = 1 } ^ { n } d _ { i } ^ { \beta } \right] } \\ { = \beta ^ { n } \left( \displaystyle \prod _ { i = 1 } ^ { n } d _ { i } ^ { \beta - 1 } \right) \eta ^ { - n \delta } e ^ { - S ( \beta ) \eta ^ { - \beta } } . } \end{array}\tag{9}
$$

Step 2: Joint posterior. With the prior $\pi ( \beta , \eta ) \propto$ $( \beta \eta ) ^ { - 1 }$ , Bayes’ rule gives

$$
\begin{array} { l } { { \pi ( \beta , \eta \vert D ) \propto p ( D \vert \beta , \eta ) \pi ( \beta , \eta ) } } \\ { { \propto \beta ^ { n } \left( \displaystyle \prod _ { i = 1 } ^ { n } d _ { i } ^ { \beta - 1 } \right) \eta ^ { - n \beta } } } \\ { { \times e ^ { - S ( \beta ) \eta - \beta } \displaystyle \frac { 1 } { \beta \eta } } } \\ { { \propto \beta ^ { n - 1 } \left( \displaystyle \prod _ { i = 1 } ^ { n } d _ { i } ^ { \beta - 1 } \right) \eta ^ { - n \beta - 1 } } } \\ { { \times e ^ { - S ( \beta ) \eta - \beta } . } } \end{array}\tag{10}
$$

Here $\propto$ indicates that factors not depending on $( \beta , \eta )$ have been dropped.

Step 3: Marginalizing $\eta .$ We integrate η out,

$$
\pi ( \beta \mid D ) = \int _ { 0 } ^ { \infty } \pi ( \beta , \eta \mid D ) d \eta .\tag{11}
$$

Pulling the factors independent of $\eta$ out of Eq. (10) gives

$$
\pi ( \beta \mid D ) \propto \beta ^ { n - 1 } \left( \prod _ { i = 1 } ^ { n } d _ { i } ^ { \beta - 1 } \right) { \cal I } ( \beta ) ,\tag{12}
$$

where

$$
I ( \beta ) : = \int _ { 0 } ^ { \infty } \eta ^ { - n \beta - 1 } e ^ { - S ( \beta ) \eta ^ { - \beta } } d \eta .\tag{13}
$$

The substitution $u = S ( \beta ) \eta ^ { - \beta }$ gives

$$
\begin{array} { c } { { \eta = \left( \displaystyle \frac { S ( \beta ) } u \right) ^ { 1 / \beta } , } } \\ { { d \eta = - \displaystyle \frac { 1 } { \beta } S ( \beta ) ^ { 1 / \beta } u ^ { - 1 / \beta - 1 } d u , } } \end{array}\tag{14}
$$

$$
\eta ^ { - n \beta - 1 } = \left( \frac { u } { S ( \beta ) } \right) ^ { n + 1 / \beta } ,\tag{15}
$$

so that

$$
\eta ^ { - n \beta - 1 } d \eta = - \frac 1 \beta S ( \beta ) ^ { - n } u ^ { n - 1 } d u .\tag{16}
$$

Since $\eta : 0  \infty$ corresponds to $u : \infty  0$ reversing the limits gives

$$
\begin{array} { l } { { \displaystyle I ( \beta ) = \frac { 1 } { \beta } S ( \beta ) ^ { - n } \int _ { 0 } ^ { \infty } u ^ { n - 1 } e ^ { - u } d u } } \\ { { \displaystyle \ = \frac { \Gamma ( n ) } { \beta S ( \beta ) ^ { n } } . } } \end{array}\tag{17}
$$

Substituting into Eq. (12),

$$
\begin{array} { c } { { \pi ( \beta \mid D ) \propto \beta ^ { n - 1 } \left( \displaystyle \prod _ { i = 1 } ^ { n } d _ { i } ^ { \beta - 1 } \right) \displaystyle \frac { \Gamma ( n ) } { \beta S ( \beta ) ^ { n } } } } \\ { { \propto \beta ^ { n - 2 } S ( \beta ) ^ { - n } \displaystyle \prod _ { i = 1 } ^ { n } d _ { i } ^ { \beta - 1 } . } } \end{array}\tag{18}
$$

Writing the unnormalized posterior kernel as

$$
g ( \beta ; D ) : = \beta ^ { n - 2 } S ( \beta ) ^ { - n } \prod _ { i = 1 } ^ { n } d _ { i } ^ { \beta - 1 } ,\tag{19}
$$

the normalized posterior is

$$
\pi ( \beta \mid D ) = \frac { g ( \beta ; D ) } { \int _ { 0 } ^ { \infty } g ( b ; D ) d b } .\tag{20}
$$

Step 4: Posterior predictive probability. This step derives Eq. (4), the posterior predictive probability

$$
p : = \operatorname* { P r } ( d _ { \mathrm { n e w } } > d ^ { \star } \mid D ) .\tag{21}
$$

Given $( \beta , \eta )$ , the Weibull survival function gives

$$
\mathrm { P r } ( d _ { \mathrm { n e w } } > d ^ { \star } \mid \beta , \eta ) = e ^ { - ( d ^ { \star } ) ^ { \beta } \eta ^ { - \beta } } .\tag{22}
$$

Averaging the parameter uncertainty under the posterior,

$$
\begin{array} { l } { \displaystyle { p = \int _ { 0 } ^ { \infty } \int _ { 0 } ^ { \infty } e ^ { - ( d ^ { \star } ) ^ { \beta } \eta ^ { - \beta } } } } \\ { \displaystyle { \qquad \times \pi ( \beta , \eta \mid D ) d \eta d \beta . } } \end{array}\tag{23}
$$

For fixed $\beta ,$ , the conditional posterior of η is

$$
\pi ( \eta \mid \beta , D ) \propto \eta ^ { - n \beta - 1 } e ^ { - S ( \beta ) \eta ^ { - \beta } } .\tag{24}
$$

Normalizing with Eq. (17),

$$
\begin{array} { c } { { \pi ( \eta \mid \beta , D ) = \displaystyle \frac { \beta S ( \beta ) ^ { n } } { \Gamma ( n ) } \eta ^ { - n \beta - 1 } } } \\ { { \times e ^ { - S ( \beta ) \eta ^ { - \beta } } . } } \end{array}\tag{25}
$$

Writing the predictive survival at fixed $\beta$ as

$$
q ( \beta ) : = \mathrm { P r } ( d _ { \mathrm { n e w } } > d ^ { \star } \mid \beta , D ) ,\tag{26}
$$

we have

$$
\begin{array} { l } { { \displaystyle q ( \beta ) = \int _ { 0 } ^ { \infty } e ^ { - ( d ^ { \star } ) ^ { \beta } \eta ^ { - \beta } } \pi ( \eta \mid \beta , D ) d \eta } } \\ { { \displaystyle ~ = \frac { \beta S ( \beta ) ^ { n } } { \Gamma ( n ) } \int _ { 0 } ^ { \infty } \eta ^ { - n \beta - 1 } } } \\ { { \displaystyle ~ \times \exp \left[ - \left\{ S ( \beta ) + ( d ^ { \star } ) ^ { \beta } \right\} \eta ^ { - \beta } \right] d \eta } . } \end{array}\tag{27}
$$

As before, for any $A > 0$ we have

$$
\int _ { 0 } ^ { \infty } \eta ^ { - n \beta - 1 } e ^ { - A \eta ^ { - \beta } } d \eta = \frac { \Gamma ( n ) } { \beta A ^ { n } } ,\tag{28}
$$

so substituting $A = S ( \beta ) + ( d ^ { \star } ) ^ { \beta }$ gives

$$
\begin{array} { l } { { q ( \beta ) = \displaystyle \frac { \beta S ( \beta ) ^ { n } } { \Gamma ( n ) } \frac { \Gamma ( n ) } { \beta \left[ S ( \beta ) + ( d ^ { \star } ) ^ { \beta } \right] ^ { n } } \nonumber } } \\ { { \displaystyle ~ = \left( \frac { S ( \beta ) } { S ( \beta ) + ( d ^ { \star } ) ^ { \beta } } \right) ^ { n } . } } \end{array}\tag{29}
$$

Finally, averaging the uncertainty in $\beta$ under $\pi ( \beta \mid D )$

$$
p = \int _ { 0 } ^ { \infty } \pi ( \beta \mid D ) \left( \frac { S ( \beta ) } { S ( \beta ) + ( d ^ { \star } ) ^ { \beta } } \right) ^ { n } d \beta .\tag{30}
$$

Step 5: Restoring the indices. With $D = D _ { t }$ $n = n _ { t }$ , and $d ^ { \star } = d _ { t } ^ { \star }$

$$
\pi ( \beta \mid D _ { t } ) \propto \beta ^ { n _ { t } - 2 } S _ { t } ( \beta ) ^ { - n _ { t } } \prod _ { d \in D _ { t } } d ^ { \beta - 1 } ,\tag{31}
$$

$$
p _ { t } = \int _ { 0 } ^ { \infty } \pi ( \beta \mid D _ { t } ) \left( \frac { S _ { t } ( \beta ) } { S _ { t } ( \beta ) + ( d _ { t } ^ { \star } ) ^ { \beta } } \right) ^ { \frac { \ l } { t } }\tag{n<sub>t</sub>}
$$

dβ.

(32)

## C.4 Persistence-Driven Binding

BIND turns $P _ { t }$ into a decision. Since a binding cannot be undone, the probability already committed to it must be preserved: the cumulative binding probability is the running max $P _ { t } ^ { \mathrm { m a x } }$ of Eq. 5 rather than the current P<sub>t</sub>, and a step on which persistence falls adds nothing. The hazard $h _ { t }$ is the increase $P _ { t } ^ { \operatorname* { m a x } } - P _ { t - 1 } ^ { \operatorname* { m a x } }$ expressed as a fraction of the probability still unbound, which is what makes the per-step Bernoulli draws telescope to $\mathrm { P r } [ \mathbf { B O U N D } ( s )$ by $t ] = P _ { t } ^ { \mathrm { m a x } }$ . Once $\mathsf { B O U N D } \big ( s ^ { ( i ) } \big )$ is set, Algorithm 1 stops calling either subroutine for that slot, and REPLACE substitutes its best match into the instruction from that step onward.

## D Experiments

## D.1 Main Results

## D.1.1 Setup

This section details the methods compared in §6.

Shared pipeline. All methods share a single substitution layer and differ only in how they determine the referent of each slot. Before the episode, the parser extracts the object slots from the instruction (Appendix A.1.3). At every step, the layer accumulates the object labels that have appeared in the observations, determines a label for each slot, and replaces the slot in the instruction before passing it to the frozen agent.

Only candidates that have already been observed can be selected, and the replacement is confined to the instruction text, leaving the system prompt, few-shot examples, observations, and admissible actions unchanged. Since every method runs at every step under this layer, their task performance and per-step latency are directly comparable.

Baseline. The paraphrased instruction is passed to the agent unchanged, leaving grounding entirely to the LLM.

Greedy. Greedy uses the same candidate distance as BINDIntent, including mean-centering over observed labels (§5.2.2). At every step, it replaces each slot with the current best match without estimating persistence; an observed label that exactly matches the slot is selected directly. As new labels are observed, the replacement follows the updated best match. Greedy therefore isolates the effect of the binding decision in BINDIntent, as both methods select the same best match at every step and differ only in when they commit to it.

Self-clarification. Self-clarification adapts query-based disambiguation (Ren et al., 2023; Ramrakhya et al., 2025) to a setting without a user, delegating the binding decision to the agent LLM itself. At every step, after the agent acts, it constructs a query containing the instruction, the object types observed up to that step, and the referring expressions (Figure 7). The agent LLM answers this query in a separate generation call with the same weights, selecting one observed object or None for each expression. The answer replaces the expressions in the instruction at the next step; an expression answered with None or an unparsable answer remains unchanged. The query explicitly allows None when no observed object is the referent, and contains no examples or hints, ensuring that the method reflects the agent’s own binding ability. Its format mirrors the list-choice template of the paraphrase-knowledge probe (Appendix B.2), on which the same models identify most referents when all objects are listed. The query receives the same observed candidates as BINDIntent but no exploration progress, since estimating whether an unseen candidate may still be closer is the component that BINDIntent introduces.

BINDIntent<sub>SBERT</sub>. BINDIntent<sub>SBERT</sub> is identical to BINDIntent except that it computes candidate distances with the sentence encoder all-mpnet-base-v2 (Reimers and Gurevych, 2019) instead of Numberbatch. Because the encoder is applied to each newly observed label, its latency grows with the number of labels revealed per step (Appendix D.2).

BINDIntent hyperparameters. BINDIntent uses a lexical allowance step of ∆τ = 0.04 and a fallback threshold of c = 3, the smallest sample size exceeding the two Weibull parameters; results vary little with c (Table 11). The same values are used on ScienceWorld-Para without benchmark-specific tuning. For pretrained models we use $\begin{array} { r c l } { \Delta \tau } & { = } & { 0 . 0 2 5 } \end{array}$ . Since τ<sub>t</sub> widens with the step count, the allowance is effective only when it widens in step with exploration. Unlike RL-finetuned policies, pretrained models are not adapted to the environment and often spend steps on inadmissible or redundant actions that reveal no new objects. A smaller step keeps the allowance aligned with their slower exploration. All other settings are shared across models.

Figures 22–26 decompose Table 4 on ALFWorld-Para into all combinations of the nine action-axis types and the three object-axis types. Columns are the object axis and rows the action axis, with each panel’s color pattern showing how a method affects the two axes separately.

Under Baseline (Figure 22), the rightmost column, Same-Polarity, turns red in every panel. This is sharpest for RL-finetuned sLLMs, where the left two columns stay green and only the right column collapses, forming a vertical boundary. The dominance of the object axis found in § 4.1 is therefore consistent across action types.

![](images/c72f5696af6077377afc89b4f5bb5d486b5132600693e547328be9c184509845.jpg)  
Figure 7: Query used by Self-clarification at every step. {observed\_objects} lists the object types observed so far without instance indices, and {phrases} lists the quoted referring expressions of the instruction.

Greedy (Figure 23) erases this boundary but darkens the panel as a whole. Same-Polarity brightens while None and Addition fall below Baseline, since Greedy substitutes another object before the true referent is observed. StraTA-1.5B constitutes a notable exception: Greedy surpasses BINDIntent on SP (63.7 vs. 56.1) while scarcely degrading None (84.4 → 81.9), in contrast to StraTA-7B (89.6 → 67.2; Table 4). StraTA devises a strategy at the outset of each episode and acts upon it thereafter. Presumably owing to its limited capacity, StraTA-1.5B tends to compose vague initial strategies that prescribe indiscriminate search, whereas StraTA-7B formulates concrete strategies anchored to environment elements such as object and receptacle names. Under a premature substitution, the 1.5B agent consequently continues to explore, whereas the 7B agent commits to a strategy built around the substituted object.

Self-clarification (Figure 24) shows a pattern that varies with model size. For pretrained LLMs and 7B RL-finetuned sLLMs, Same-Polarity tends to brighten enough to offset the loss in None, whereas for 1.5B models it changes little while None and Addition drop. Because it also adds one LLM call per step even when the object is unchanged, the number of calls grows with every trajectory (Figure 6).

BINDIntent<sub>SBERT</sub> (Figure 25) raises Same-

<table><tr><td colspan="5">GiGPO-7B</td><td colspan="4">StraTA-1.5B</td></tr><tr><td> $\Delta \tau$ </td><td>None</td><td>Add</td><td>Same-P</td><td>All</td><td>None</td><td>Add</td><td>Same-P</td><td>All</td></tr><tr><td>0.20</td><td>87.4</td><td>82.1</td><td>50.2</td><td>72.7</td><td>81.4</td><td>74.2</td><td>49.8</td><td>68.0</td></tr><tr><td>0.10</td><td>84.6</td><td>85.6</td><td>54.6</td><td>74.6</td><td>86.1</td><td>80.8</td><td>54.3</td><td>73.3</td></tr><tr><td>0.05</td><td>88.5</td><td>87.0</td><td>58.1</td><td>77.5</td><td>84.5</td><td>82.1</td><td>55.8</td><td>73.8</td></tr><tr><td>0.04</td><td>88.9</td><td>86.8</td><td>57.6</td><td>77.4</td><td>85.2</td><td>81.2</td><td>56.1</td><td>73.8</td></tr><tr><td>0.02</td><td>89.0</td><td>86.7</td><td>56.4</td><td>77.0</td><td>84.7</td><td>77.4</td><td>55.7</td><td>72.2</td></tr><tr><td>0.00</td><td>89.2</td><td>86.8</td><td>12.1</td><td>61.8</td><td>84.4</td><td>71.3</td><td>7.0</td><td>53.2</td></tr></table>

Table 9: Ablation of the lexical allowance step $\Delta \tau \left( \tau _ { t } \right) =$ $t \Delta \tau ) ; \Delta \tau = 0$ permits only exact matches throughout the episode. Success rate (%) on ALFWorld-Para by object-axis paraphrase type; the shaded row is the setting used in all main experiments.

Polarity by a limited margin, blurring the vertical boundary without removing it.

Under BINDIntent (Figure 26) the boundary disappears in most panels, but how it closes differs by model. In GiGPO-7B, Same-Polarity spreads from 38.1 to 64.2 across action types, following the spread of its own None column (56.2–95.1). In StraTA-1.5B, Same-Polarity flattens to 50.9–59.7, more uniform than its None column, which still spans 72.1–90.3. Addition rises most in the rows lowest under Baseline, Question (58.0 → 73.9) and Hint (55.3 → 70.8), narrowing the spread across rows from 24.8 to 16.4. In both panels, Question and Hint are also the lowest rows in the None column, hence their darkness under Same-Polarity follows the difficulty of these action types rather than a residual failure of binding. Across the five heatmaps, the other methods pay for a gain on one axis with a loss on the other, while only BINDIntent raises Same-Polarity without giving up either.

The magnitude of this gain, however, differs between model groups: BINDIntent improves SP considerably more for RL-finetuned sLLMs (+17.5– 49.1 pp) than for pretrained LLMs (+9.6–21.1 pp). RL-finetuned agents possess environment-specific priors over exploration and action, and their failures under SP originate predominantly from binding, as evidenced by a wide None–SP gap (42.8– $7 7 . 4 \mathrm { p p ) ; }$ ; once binding is resolved, these priors translate more directly into task success. Pretrained LLMs lack such priors, and their narrower gap (27.4–45.8 pp) is attributable to two sources. Models with at least 14B parameters already resolve synonyms in part, retaining 41–58% of their None SR under SP, and the 7–8B models remain constrained by low SR even on unchanged objects. In both cases, determining how to proceed with the task persists as a bottleneck even after binding.

<table><tr><td rowspan="2"></td><td colspan="5">GiGPO-7B</td><td colspan="3">StraTA-1.5B</td></tr><tr><td>κ |None</td><td>Add</td><td>Same-P</td><td>All</td><td>None</td><td>Add</td><td>Same-P</td><td>All</td></tr><tr><td>0</td><td>85.5</td><td>81.6</td><td>65.4</td><td>77.2</td><td>82.1</td><td>75.9</td><td>63.4</td><td>73.5</td></tr><tr><td>0.25</td><td>88.0</td><td>86.7</td><td>63.1</td><td>78.9</td><td>84.9</td><td>81.0</td><td>60.8</td><td>75.2</td></tr><tr><td> $0 . 5$ </td><td>89.4</td><td>86.2</td><td>61.7</td><td>78.7</td><td>85.4</td><td>81.6</td><td>58.1</td><td>74.7</td></tr><tr><td>1</td><td>88.9</td><td>86.8</td><td>57.6</td><td>77.4</td><td>85.2</td><td>81.2</td><td>56.1</td><td>73.8</td></tr><tr><td>2</td><td>88.6</td><td>87.3</td><td>52.4</td><td>75.7</td><td>85.4</td><td>81.7</td><td>51.9</td><td>72.6</td></tr><tr><td>4</td><td>89.0</td><td>88.2</td><td>47.6</td><td>74.4</td><td>85.6</td><td>82.3</td><td>45.6</td><td>70.7</td></tr><tr><td>8</td><td>88.9</td><td>88.1</td><td>41.5</td><td>72.3</td><td>85.2</td><td>81.6</td><td>40.3</td><td>68.5</td></tr></table>

Table 10: Ablation of the scale κ on the number of remaining chances, $k _ { t } \ \to \ \kappa k _ { t } \colon \kappa \ = \ 1$ counts each unexplored unit once. Success rate (%) on ALFWorld-Para by object-axis paraphrase type; the shaded row is the setting used in all main experiments.

## D.1.2 Ablation Study

We ablate BINDIntent on GiGPO-7B and StraTA-1.5B, which differ in both scale and codebase, varying one component at a time with all others fixed. Results are averaged over three seeds, and shaded rows denote the setting used in all main experiments. BINDIntent accounts for unseen candidates through two mechanisms: the lexical allowance defers non-exact bindings early in an episode, and persistence defers them while unexplored units may still hold a closer candidate. The ablations show that both mechanisms contribute and that their default settings lie in a stable range.

Lexical allowance. With $\Delta \tau = 0 .$ , only exact matches are permitted. No paraphrase is ever bound and the results coincide with Baseline (Table 9; cf. Table 4). Increasing $\Delta \tau$ admits synonyms and recovers SP, which remains stable for $\Delta \tau$ between 0.02 and 0.05. A larger step admits distant candidates before exploration can rule out an unseen referent: at $\Delta \tau = 0 . 2 0$ , SP falls to 50.2 and 49.8, and None and Addition also decline, as expressions whose referents have not yet been observed are bound to other objects. The allowance is thus effective only when it widens in step with exploration.

Unexplored units. κ scales the number of chances for an unseen candidate to be closer than the current best match (Table 10). With $\kappa = 0 ,$ persistence ignores unexplored units, and a slot is bound as soon as a candidate enters the allowance. This yields the highest SP but lowers None and Addition by 3.1–5.3 pp relative to $\kappa = 1$ , since expressions are bound before their true referents are observed. As κ grows, binding is deferred more strongly, preserving None and Addition at the cost of SP. Accounting for unexplored units thus prevents premature binding on instructions that require no substitution. Although $\kappa = 0 . 2 5$ yields the highest All, calibrating κ amounts to estimating the density of objects per unit, which requires environment-specific knowledge. We therefore adopt κ = 1, which counts each unexplored unit as a single chance without tuning.

<table><tr><td rowspan="2">一</td><td colspan="5">GiGPO-7B</td><td colspan="3">StraTA-1.5B</td></tr><tr><td>C None</td><td>Add</td><td>Same-P</td><td>All</td><td>None</td><td></td><td>Add</td><td>Same-P</td><td>All</td></tr><tr><td>3</td><td>88.9</td><td>86.8</td><td>57.6</td><td>77.4</td><td>85.2</td><td>81.2</td><td>56.1</td><td>73.8</td></tr><tr><td>4</td><td>88.6</td><td>86.8</td><td>57.3</td><td>77.2</td><td>85.3</td><td>80.8</td><td>56.3</td><td>73.8</td></tr><tr><td>5</td><td>89.2</td><td>86.8</td><td>58.3</td><td>77.7</td><td>85.5</td><td>81.4</td><td>55.5</td><td>73.7</td></tr><tr><td>6</td><td>88.6</td><td>86.9</td><td>59.0</td><td>77.8</td><td>85.1</td><td>82.1</td><td>55.9</td><td>74.0</td></tr><tr><td>7</td><td>89.3</td><td>86.9</td><td>57.8</td><td>77.6</td><td>85.7</td><td>81.5</td><td>55.9</td><td>74.0</td></tr></table>

Table 11: Ablation of the fallback threshold c below which PERSISTENCE falls back to the exchangeabilitybased estimate. Success rate (%) on ALFWorld-Para by object-axis paraphrase type; the shaded row is the setting used in all main experiments.

<table><tr><td rowspan="2">Variant</td><td colspan="4">GiGPO-7B</td><td colspan="4">StraTA-1.5B</td></tr><tr><td>|None</td><td>Add</td><td>Same-P</td><td>All</td><td>None</td><td>Add</td><td>Same-P</td><td>All</td></tr><tr><td>BINDIntent</td><td>88.9</td><td>86.8</td><td>57.6</td><td>77.4</td><td>85.2</td><td>81.2</td><td>56.1</td><td>73.8</td></tr><tr><td>Deterministic binding</td><td>88.2</td><td>84.1</td><td>59.0</td><td>76.7</td><td>84.5</td><td>78.4</td><td>58.1</td><td>73.2</td></tr><tr><td>Fallback-only persistence</td><td>89.3</td><td>82.1</td><td>38.3</td><td>69.2</td><td>85.6</td><td>76.4</td><td>36.7</td><td>65.6</td></tr><tr><td>No centering</td><td>88.4</td><td>87.1</td><td>54.9</td><td>76.4</td><td>85.1</td><td>81.0</td><td>53.1</td><td>72.7</td></tr></table>

Table 12: Component ablation (SR, %) on ALFWorld-Para. Deterministic binding: bind once $P _ { t } ^ { \operatorname* { m a x } } > 0 . 5 ;$ Fallback-only persistence: $| V _ { t } | / ( | V _ { t } | + k _ { t } ) ^ { - }$ in place of the Weibull model; No centering: MATCH on raw embeddings. Shaded: full method.

Persistence estimate. Fallback-only persistence replaces the Weibull model with the exchangeability-based estimate, which ignores candidate distances (Table 12). SP drops to 38.3 and 36.7, as it cannot distinguish a distinctive best match from an ambiguous one. Modeling the distance distribution is thus essential for judging whether an unseen candidate may be closer. The fallback threshold c has little effect, with All varying within 0.6 pp for c = 3–7 (Table 11).

Binding rule and centering. Deterministic binding commits every slot once $P _ { t } ^ { \mathrm { m a x } }$ exceeds 0.5, slightly raising SP but lowering Addition by 2.7 and 2.8 pp and yielding a lower All. The hazard instead binds a slot by step t with probability equal to P<sup>max</sup><sub>t</sub> , calibrating commitment to persistence without a fixed threshold. No centering lowers SP by 2.7 and 3.0 pp, since components shared across a scene dominate the raw similarities and obscure the signal that distinguishes candidates.

<table><tr><td></td><td colspan="3">Per stage</td></tr><tr><td>Method</td><td>Embed Dist. Weibull</td><td>LLM Bind.</td><td>Total</td></tr><tr><td>Greedy</td><td>0.020.14</td><td>0.03 0.07</td><td>0.25 2.86</td></tr><tr><td>BINDIntentsBERT BINDIntent</td><td>2.41 0.20 0.10 0.02 0.15 0.09</td><td>0.06</td><td>0.38</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td colspan="4">Self-clarification, Pretrained Qwen2.5-1.5B</td></tr><tr><td>Qwen2.5-7B</td><td></td><td>29.92 0.32 60.84 0.29</td><td>30.43 61.30</td></tr><tr><td>Llama3.1-8B Phi-4-14B</td><td>46.76 71.04</td><td>0.36 0.30</td><td>47.33</td></tr><tr><td>Qwen3-14B</td><td>68.89</td><td>0.47</td><td>71.55 69.58</td></tr><tr><td>Mistral-S-24B</td><td>106.10</td><td>0.57</td><td>106.88</td></tr><tr><td colspan="4">Self-clarification, RL-inetuned</td></tr><tr><td>PPO-1.5B GRPO-1.5B</td><td>30.56 30.33</td><td>0.33</td><td>31.08</td></tr><tr><td></td><td></td><td>0.30</td><td>30.81</td></tr><tr><td>GiGPO-1.5B</td><td>30.74</td><td>0.33</td><td>31.26</td></tr><tr><td>StraTA-1.5B</td><td>31.34</td><td>0.33</td><td>31.87</td></tr><tr><td>PPO-7B</td><td>58.25</td><td>0.32</td><td>58.77</td></tr><tr><td>GRPO-7B</td><td>59.42</td><td>0.33</td><td>59.93</td></tr><tr><td>GiGPO-7B</td><td></td><td>56.67 0.33</td><td>57.20</td></tr><tr><td>StraTA-7B</td><td></td><td>61.83 0.33</td><td></td></tr><tr><td></td><td></td><td></td><td>62.36</td></tr></table>

Table 13: Per-step latency breakdown (ms). Embed: embedding of the referring expression and observed object names; Dist.: candidate distance computation; Weibull: persistence estimation; LLM: additional call to the agent LLM; Bind.: binding decision. Greedy, BINDIntent<sub>SBERT</sub>, and BINDIntent do not call the agent LLM and are averaged over all agent models; Selfclarification is reported per agent model. Total includes observation update and instruction rewriting.

## D.2 Latency

Principle. Latency must be measured under identical conditions for every method. The compared methods, however, rewrite the instruction, and measuring them on live rollouts would let each method alter the trajectory it is timed on. A rewrite can turn a failure into a success or vice versa and thereby change the episode length; since later steps contain many more observed objects, a different mix of short and long episodes would bias the per-step statistics. Attaching a method to a live policy would likewise alter the policy’s inputs, leaving the methods timed on different distributions. We therefore measure all methods by open-loop replay of a fixed set of trajectories: each logged step supplies the observation, the admissible actions, and the executed action, and every method is run on exactly the same sequence of steps.

Trajectory set. The trajectories are collected from Qwen2.5-7B-Instruct running as a ReAct agent on ALFWorld-Para without any binding method, to keep the observed scenes free of influence from the compared methods. To cover all task types and a range of episode lengths, we sample one success stratum and one failure stratum for each of the six ALFWorld task types, with five episodes per stratum, giving 60 episodes (30 successes and 30 failures). Failed episodes run to the 50-step limit, and half of the set accordingly consists of long-horizon trajectories in which many objects have been observed. Every method replays every step of these episodes in order, since its internal state accumulates over an episode.

Environment. All measurements are taken on a single NVIDIA A100 80GB PCIe GPU with batch size one, on a server with two Intel Xeon Gold 6338 CPUs (2.0 GHz, 64 cores in total) and 384 GB of RAM, running Ubuntu 22.04 with CUDA 12.2; the second GPU on the server is left idle. We use vLLM (Kwon et al., 2023) as an in-process engine, which keeps network and serialization overhead out of every measurement. All measurements start from a warm state: loading the language model and the methods and initializing the ALFWorld environment are excluded, and ten warm-up generations are discarded before timing begins. Each measurement starts when the observation of a step becomes available and ends when the corresponding computation completes, with GPU synchronization at both ends of every span that involves the GPU.

What is timed. Every method runs at every step. For BINDIntent and the greedy baseline, the measured span covers the entire computation of the method: updating the observed objects and units, obtaining embeddings, computing distances, estimating persistence, deciding the binding, and rewriting the instruction. No language-model call is involved. For BINDIntent<sub>SBERT</sub>, the span additionally includes the encoder forward pass. Selfclarification, by definition, delegates reference resolution to the language model itself; at every step it poses a single question covering all slots, and its span consists mainly of that generation. This cost is inherent to any approach that delegates clarification to an LLM: it is dominated by decoding rather than prefill, recurs at every step, and grows with the length of the answer, that is, with the number of slots in the instruction.

Caching. Since an embedding depends only on the word, a name embedded earlier in the same episode is reused, and the cache is cleared at the start of each episode. For BINDIntent<sub>SBERT</sub>, only the names newly observed at a step are encoded, in a single batched forward pass. Distances, centering, and the persistence estimate are recomputed at every step. The prefix cache of the language model is enabled within an episode and reset between episodes, as in the evaluation runs.

Reporting. We report the mean per-step time of each method over all replayed steps. Because the pretrained models are served at different precisions, methods should be compared within a model rather than by absolute times across models.

Results. Table 13 decomposes the per-step time of each method. Greedy, BINDIntent, and BINDIntent do not depend on the agent model, and their totals differ by less than 0.03 ms across models. BINDIntent takes 0.38 ms per step, 0.13 ms more than Greedy, which corresponds to the persistence estimate and the binding decision. Replacing Numberbatch with SBERT (all-mpnetbase-v2) raises the total to 2.86 ms, almost entirely due to the encoder forward pass (2.41 ms). Because only newly observed names are encoded, this cost is concentrated in the steps where new objects appear, and the median step takes only 0.53 ms. The subsequent stages are also slightly slower with SBERT (e.g., 0.20 vs. 0.15 ms for distance computation), as its embeddings have 768 dimensions compared with 300 for Numberbatch.

For Self-clarification, the additional LLM call accounts for more than 98% of the step time, and the cost grows with model size, from about 30 ms for 1.5B models to 107 ms for Mistral-Small-3-24B. Its binding stage, which builds the clarification query with the chat template of the model, varies with the tokenizer rather than with model size or training: the ten models sharing the Qwen2.5 tokenizer, from 1.5B to 7B and across all four RL recipes, cluster at 0.29–0.33 ms, whereas Qwen3-14B and Mistral-Small-3-24B take 0.47 and 0.57 ms.

Precision. Llama3.1-8B, Qwen3-14B, and Mistral-Small-3-24B are served as FP8 checkpoints. Since the A100 does not natively support FP8, vLLM runs them with weight-only FP8 kernels, which may be slower than on FP8-capable hardware. This affects only the LLM call of Self-clarification, not the LLM-free methods, and the gap between the two spans two orders of magnitude; the comparison is therefore unaffected.

<table><tr><td>Task</td><td>Action frame</td></tr><tr><td>Conductivity</td><td>Your task is to determine if [obj] is electrically conductive. The [obj] is located around the [obj]. First, focus on the [obj]. If it is electrically conductive, place it in the red [obj]. If it is electrically nonconductive, place it in the green [obj].</td></tr><tr><td>Melting point</td><td>Your task is to measure the melting point of [obj], which is located around the [obj]. First, focus on the [obj]. Next, focus on the [obj]. If the melting point of [obj] is above 150.0 degrees celsius, focus on the red [obj]. If the melting point of [obj] is below 150.0 degrees celsius, focus on the green [obj]. The [obj] are located around the [obj].</td></tr><tr><td>Find</td><td>Your task is to find a(n) [obj]. First, focus on the thing. Then, move it to the green [obj] in the [obj].</td></tr><tr><td>Life span</td><td>Your task is to find the [obj] with the longest life span. The [obj] are in the outside&#x27; location. Focus on the [obj] with the longest life span.</td></tr><tr><td>Chemistry</td><td>Your task is to use chemistry to create the substance [obj]&#x27;. A [obj] and some of the ingredients might be found near the [obj]. When you are done, focus on the [obj].</td></tr><tr><td>Paint mixing</td><td>Your task is to use chemistry to create green [obj]. When you are done, focus on the green [obj].</td></tr></table>

Table 14: Action frames of ScienceWorld-Para by task. [obj] denotes an object slot paraphrased on the object axis; all other text is kept fixed.

## E ScienceWorld-Para

ScienceWorld-Para follows the construction and evaluation protocol of ALFWorld-Para (Appendix A); this section describes only the components that differ.

## E.1 Benchmark Generation

Source episodes. We start from the Science-World (Wang et al., 2022) test split used by StraTA (Xue et al., 2026) (175 episodes, 11 task types), which covers the task types on which the RL policies were trained. Test episodes differ from training episodes only in task variation, and their instructions follow the same templates as those seen during training. As in ALFWorld-Para, paraphrasing is thus the sole source of linguistic variation relative to training. We exclude task types whose targets are distinguished only by alphanumeric labels (e.g., unknown substance B), since such names admit no synonym that preserves the referent. From the remaining episodes, we remove those with duplicate goal instructions and sample 82 instructions preserving the task-type distribution, yielding 82 × 29 = 2,378 instructions.

Paraphrase Generation. ScienceWorld-Para is produced with the factored pipeline of Appendix A.1.1, using the same generation model (Gemini 3 Flash (Google DeepMind, 2025)), the same type-specific suffixes (Figures 12–14 and 15), and the same verification and composition procedure; only the inputs and the environment-specific parts of the prompts differ.

Action Axis. We replace every object slot in the task instructions with a placeholder and define the remainder as the action frame (Table 14). Frames are paraphrased with the action axis prefix (Figure 11) adapted to ScienceWorld placeholders. Since a ScienceWorld instruction spans several sentences, each action axis type is applied once per instruction, and the remaining steps are kept verbatim.

Object Axis. Every object slot, including targets, containers, rooms, and category names, is a target of object axis paraphrase, and candidates are generated with the object axis prefix (Figure 16).

Verification and Composition. Candidates are filtered with the verifier prompts of Figure 17, and verified frames are composed with object paraphrases through the same slot-filling scheme with four inflected forms. As in ALFWorld-Para (Table 6), the 2 object types, 9 action types, and their 18 compositions yield 29 paraphrase types per instruction.

## E.1.1 Parser Stability

The same extraction procedure is applied without modification to ScienceWorld-Para, whose instructions span multiple sentences and contain 4.56 slots on average. Evaluated against the gold substitutions of all 2,378 instructions, the extraction recovers 99.96% of the 10,844 gold slots. Manual verification of 300 instructions, sampled with stratification by paraphrase type and assessed under the same criterion, found 296 to be correct (98.7%). Extraction is performed once per episode, requires 3.2 ms, and is likewise shared by all binding methods, leaving their relative cost unchanged.

## E.1.2 Human Evaluation

Annotation setup. We sampled 145 paraphrase– original pairs (6.1% of the full 2,378-item dataset) via stratified sampling across the 29 paraphrase types, five per type. Three annotators, blind to which label would favor the benchmark, independently judged each pair with a binary label (Figure 21): whether the two commands are consistent with the agent performing the same task. A pair was marked equivalent only if the target object, destination, handling (cleaning, heating, cooling), and count all remained the same despite differences in wording, tone, sentence structure, or directive form; it was marked non-equivalent if any of these changed or if the intent was too vague to determine. Annotators were given a reference list of the distinct entities in the environment—targets, rooms, and the animal, plant, living-thing, and non-livingthing categories—so that a renamed object could be checked against entities it might collide with. Since ScienceWorld tasks hinge on measurements, the same list fixed the properties that must survive unchanged: melting point, electrical conductivity, life span, and the temperature threshold with its unit. The three annotators were disjoint from those who annotated ALFWorld-Para; recruitment and the blinding protocol follow Appendix A.1.4.

<table><tr><td>Model</td><td>ScienceWorld</td><td>ScienceWorld-Para</td><td>Score Drop (%)</td></tr><tr><td>PPO-7B</td><td>45.2</td><td>38.1</td><td>-15.7</td></tr><tr><td>GRPO-7B</td><td>30.7</td><td>27.6</td><td>-10.1</td></tr><tr><td>GiGPO-7B</td><td>63.7</td><td>50.8</td><td>-20.3</td></tr><tr><td>StraTA-7B</td><td>63.9</td><td>55.3</td><td>-13.5</td></tr></table>

Table 15: Average score on ScienceWorld and ScienceWorld-Para. Score Drop: relative change.

Experimental Results. Under majority vote, 142 of 145 pairs (97.9%) were judged meaningpreserving, and 137 (94.5%) unanimously so; none was unanimously rejected. Adjudicating the 8 splitvote items against the entity reference, 7 were equivalent and 1 a failure: an object paraphrase that reduced the target to its head noun (jam sandwich → sandwich), leaving the substance the task asks the agent to synthesize under-specified. In total, 144 of 145 (99.3%) preserve the original task semantics. Unlike those in ALFWorld-Para, the disagreements are not confined to the lexical axis: three involve no object substitution and arise purely from pragmatic reframing, where one annotator read a question or hint form (e.g., Your task is to determine ifpaper clip is electrically conductive → It remains a mystery whether the paper clip is electrically conductive) as not explicitly instructing the action. The other five pair an object substitution with such a reframing, and the substituted terms, such as animal → beast, living room → front room, and green paint → green colorant, name no other distinct entity. No case altered the measured property, the temperature threshold, or the goal state.

Retention of flagged items. All 145 sampled pairs, including the eight that drew split votes, were retained in the benchmark by design. Terms such as beast and colorant are attested synonyms of the intended target, and the question and hint forms are exactly the pragmatic reframings the benchmark is built to measure; excluding the pairs that human annotators found hardest to judge would remove the upper end of the difficulty range rather than correct an error. We retain them to probe whether instruction understanding holds at the margins of valid paraphrase rather than only for clear-cut cases. The single adjudicated failure was also retained: although under-specified in this task, referring to an object by its head noun alone is an uncommon but natural user expression, and one pair in 145 cannot meaningfully shift the aggregate scores we report.

<table><tr><td></td><td>| ScienceWorld</td><td colspan="8">ScienceWorld-Para</td></tr><tr><td></td><td></td><td colspan="2">Action</td><td></td><td colspan="2">Object</td><td colspan="3">Action × Same-P</td></tr><tr><td>Model</td><td>None</td><td>Others</td><td>Quest.</td><td>Hint | Add</td><td></td><td>Same-P</td><td>Others</td><td>Quest. Hint</td><td></td></tr><tr><td>PPO-7B</td><td>45.2</td><td>46.7</td><td>46.7</td><td>42.3</td><td>43.7</td><td>26.5</td><td>25.1</td><td>27.8</td><td>23.2</td></tr><tr><td>GRPO-7B</td><td>30.7</td><td>36.2</td><td>36.2</td><td>33.6</td><td>32.1</td><td>14.7</td><td>16.3</td><td>13.0</td><td>17.1</td></tr><tr><td>GiGPO-7B</td><td>63.7</td><td>63.2</td><td>55.3</td><td>63.8</td><td>54.9</td><td>36.0</td><td>35.7</td><td>35.4</td><td>36.5</td></tr><tr><td>StraTA-7B</td><td>63.9</td><td>62.8</td><td>63.3</td><td>61.8</td><td>60.3</td><td>42.5</td><td>43.4</td><td>46.1</td><td>41.0</td></tr><tr><td>Mean</td><td>50.9</td><td>52.2</td><td>50.4</td><td>50.4 | 47.7</td><td></td><td>29.9</td><td>30.1</td><td>30.6</td><td>29.4</td></tr></table>

Table 16: Average score of RL-finetuned Qwen2.5-7B (Baseline) on ScienceWorld-Para by action and object paraphrase type. Others: mean over the seven action types other than Question and Hint.

Ethics. Annotators were unpaid volunteers from our research group who agreed to have their judgments used and reported in this work; given this affiliation, they were blind to which outcome would favor the benchmark (Appendix A.1.4). The judged text was public and non-sensitive, and no personally identifiable information was collected. We judged this not to constitute human subjects research under our institution’s guidelines and did not seek IRB review.

## E.2 Experimental Results

We evaluate RL-finetuned sLLMs trained with PPO, GRPO, GiGPO, and StraTA on the shared Qwen2.5-7B backbone, on both ScienceWorld and ScienceWorld-Para. Each agent runs with the step budget of its training codebase (20 steps for StraTA and 30 for BEACON; Appendix A.2.1), and following StraTA, we report the task score (0–100) provided by ScienceWorld. All models exhibit consistent degradation on ScienceWorld-Para, with relative score drops ranging from 10.1% to 20.3% (Table 15). The extent of degradation varies across RL algorithms despite the shared backbone. GiGPO-7B performs on par with StraTA-7B on Science-

![](images/1adc95162e2c064491b48f74a670e7ca42a728da39e723841f3cb2988a5838ec.jpg)  
Figure 8: Paraphrase knowledge (%) on ScienceWorld-Para of the pretrained Qwen2.5-7B backbone (top) and its RL-finetuned variants (bottom). The RL row shows four RL algorithms; error bars denote the standard deviation across the five probe templates.

World (63.7 vs. 63.9) but suffers the largest drop (20.3%), trailing it by 4.5 pp on ScienceWorld-Para (50.8 vs. 55.3). GRPO-7B shows the smallest relative drop (10.1%), although its low original score (30.7) makes it difficult to disentangle paraphrase sensitivity from baseline instability. These results indicate that, as on ALFWorld-Para, advantages on template-based instructions do not necessarily transfer to paraphrased ones. The following analyses examine action and object paraphrases separately to identify the transformations driving these losses.

## E.2.1 Analysis 1: Object Synonym Substitution Is the Dominant Failure Factor

We evaluate the four RL-finetuned sLLMs without any method and report the average score per paraphrase type, grouped as in Figure 2: Others averages the seven action types other than Question and Hint, and the action×object columns pair each action group with Same-Polarity (Table 16). Paraphrased instructions reduce scores by 10.1–20.3% relative to the original instructions (Table 15), and as on ALFWorld-Para, the drop is driven by the object axis. Action paraphrases leave the mean score nearly unchanged (50.4–52.2 vs. 50.9 under None), and Addition causes only a minor decline (47.7). Unlike on ALFWorld-Para, even Question and Hint cause little degradation: ScienceWorld instructions consist of sequential steps (Table 14), and since each action type is applied once per instruction, the remaining steps still state the required actions explicitly. Same-Polarity, in contrast, lowers the mean score to 29.9, and crossing it with any action group yields a similar level (29.4–30.6). This pattern holds for every policy, confirming that object synonym substitution is the dominant failure factor.

![](images/26b7683ef6fa4d3ad5fb9402f156f2fc372f0c8609729acb8e08057e19c8d076.jpg)  
Figure 9: Taken rate (%): among steps in which the target object is observed, up to and including its first take, the proportion of those at which the agent takes it. The top row denotes the shared Qwen2.5-7B backbone.

## E.2.2 Analysis 2: sLLMs Still Know What the Paraphrased Object Refers to

The probe follows Appendix B.2: it consists of every paraphrased object name in ScienceWorld-Para, and each prompt lists all entity types of ScienceWorld as answer choices. We probe the Qwen2.5-7B backbone and its four RL-finetuned variants with the same five templates and greedy decoding. The backbone achieves 71.2% paraphrase knowledge, and its RL-finetuned variants remain at 69.8–73.5% (Figure 8). As on ALFWorld-Para, RL fine-tuning largely preserves paraphrase knowledge, whereas the same policies lose most of their score under Same-Polarity. Knowledge loss during fine-tuning therefore does not account for the degradation.

## E.2.3 Analysis 3: Agents Encounter the Target Object but Do Not Take It

The taken rate follows the definition in Appendix B.3. Since a ScienceWorld instruction refers to several objects, the target at each step is the object that the agent must pick up at the current stage of the task, and it is observed when it appears in the observation at that step. The backbone rarely takes the target even when it is observed (16.4– 19.7%), consistent with its low score (11.5 overall; Table 17). RL fine-tuning raises taken rates under None (40.3–87.7%) and Addition (31.5–84.5%) for every policy (Figure 9). Under Same-Polarity, however, taken rates fall below None for every policy (4.2–65.9%), with GRPO dropping even below the backbone. Thus, as on ALFWorld-Para, RLfinetuned agents take observed targets reliably under original labels but far less consistently under synonyms.

## E.2.4 Main Results

Table 17 compares all methods on ScienceWorld-Para under the setup of §6. On RL-finetuned sLLMs, BINDIntent improves SP by 7.0–10.2 pp and All by 2.1–3.4 pp over Baseline, while None and Addition remain within 1.6 pp. In contrast, Base-7B gains little (+0.3 pp in All). Its score is low even when the object is unchanged (13.8 under None), and it rarely takes the target even when observed (16.4–19.7%; Appendix E.2.3). The bottleneck is task-solving rather than binding, as on ALFWorld-Para (§6.1). Greedy substitutes other objects before the true referent is observed, sharply lowering None (e.g., GiGPO-7B 62.6 → 24.7), and its All falls below Baseline on every RL-finetuned sLLM. Self-clarification raises SP on most policies, surpassing BINDIntent on PPO-7B and StraTA-7B, but lowers None and All on GiGPO-7B (50.8 → 46.8 in All) and requires an additional agent LLM call at every step, at over 160× the latency of BINDIntent. BINDIntent<sub>SBERT</sub> also improves SP, by a smaller margin than Numberbatch (+5.3– 7.8 pp). Overall, BINDIntent consistently improves over Baseline across all RL-finetuned sLLMs while preserving performance on unchanged objects, reproducing the trends observed on ALFWorld-Para.

<table><tr><td>Model</td><td colspan="4">Baseline</td><td colspan="4">Greedy</td><td colspan="4">Self-clarification</td><td colspan="4">BINDIntentsBERT</td><td colspan="4">BINDIntent (ours)</td></tr><tr><td>Split = Base-7B</td><td>None 13.8</td><td>Add 11.6</td><td>Same-P 9.0</td><td>All 11.5</td><td>None 14.0</td><td>Add 14.2</td><td>Same-P 11.6</td><td>All 13.3</td><td>None 14.2</td><td>Add 13.5</td><td>Same-P 12.0</td><td>All 13.2</td><td>None 14.0</td><td>Add 12.2</td><td>Same-P 9.5</td><td>All 11.9</td><td>None 13.5</td><td>Add 12.6</td><td>Same-P 9.4</td><td>All 11.8</td></tr><tr><td>PPO-7B</td><td>46.1</td><td>42.8</td><td>25.3</td><td>38.1</td><td>22.1</td><td>18.8</td><td>18.2</td><td>19.7</td><td>43.3</td><td>43.8</td><td>37.9</td><td>41.7</td><td>44.3</td><td>41.7</td><td>32.6</td><td>39.5</td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO-7B</td><td>35.4</td><td>31.5</td><td>15.9</td><td>27.6</td><td>22.3</td><td>19.9</td><td>18.2</td><td>20.1</td><td>35.3</td><td>32.8</td><td>23.6</td><td>30.6</td><td>32.9</td><td>33.5</td><td>23.7</td><td>30.0</td><td>45.8 36.5</td><td>43.0 30.3</td><td>34.3 26.1</td><td>41.0 31.0</td></tr><tr><td>GiGPO-7B</td><td>62.6</td><td>54.1</td><td>35.8</td><td>50.8</td><td>24.7</td><td>22.2</td><td>21.6</td><td>22.9</td><td>56.0</td><td>50.1</td><td>34.2</td><td>46.8</td><td>62.2</td><td>53.7</td><td>41.1</td><td>52.3</td><td>62.7</td><td>53.2</td><td>42.8</td><td>52.9</td></tr><tr><td>StraTA-7B</td><td>62.9</td><td>59.8</td><td>43.3</td><td>55.3</td><td>45.7</td><td>40.7</td><td>36.4</td><td>41.0</td><td>62.9</td><td>59.9</td><td>52.1</td><td>58.3</td><td>61.3</td><td>60.6</td><td>50.3</td><td>57.4</td><td>63.0</td><td>61.4</td><td>51.7</td><td>58.7</td></tr></table>

Table 17: Average score (0–100) on ScienceWorld-Para by object-axis paraphrase type: None (object unchanged), Add (Addition), Same-P (Same-Polarity), and All. BINDIntent<sub>SBERT</sub> replaces Numberbatch embeddings with SBERT. Best per row in bold for Same-P and All.

<table><tr><td rowspan="2">Method</td><td colspan="5">Qwen2.5-7B</td></tr><tr><td>Base</td><td>PPO</td><td>GRPO</td><td>GiGPO</td><td>StraTA</td></tr><tr><td>Greedy Self-clarification</td><td>0.51 140.5 12.4</td><td>0.51 129.5 12.4</td><td>0.51 160.3</td><td>0.51 139.9 12.4</td><td>0.51 132.9 12.4</td></tr></table>

Table 18: Per-step method latency (ms) on ScienceWorld-Para.

We measure per-step latency with the protocol of Appendix D.2 (Table 18). Greedy, BINDIntent, and BINDIntent<sub>SBERT</sub> never call the agent LLM, leaving their latency independent of the backbone. BINDIntent takes 0.78 ms per step, higher than on ALFWorld-Para (0.38 ms) because a Science-World instruction contains more object slots to bind. BINDIntent<sub>SBERT</sub> takes 12.4 ms on average because it runs an encoder forward pass for each newly observed name, and a ScienceWorld observation lists all objects in the current room, including container contents; its cost is thus concentrated in the steps that reveal new names, and its median latency is 0.92 ms. Self-clarification takes 129.5– 160.3 ms on 7B backbones, exceeding even its 24B cost on ALFWorld-Para (107 ms), since its single query per step must decode one referent per object slot and generation accounts for over 99% of its latency.

## F AI Assistants

Gemini 3 Flash (Google DeepMind, 2025) is a component of the benchmark construction pipeline rather than a writing aid: it generates the paraphrases of ALFWorld-Para and ScienceWorld-Para under the factored procedure described in Appendices A.1 and A.1.1, and its outputs were validated by the human evaluation reported in Appendices A.1.4 and E.1.2. Other AI assistants, namely Claude (Anthropic, 2026) and ChatGPT (OpenAI, 2026), were used only for writing polishing and code auto-completion. They were not used to develop research ideas, design experiments, analyze results, or write the scientific content of this paper. All experimental design, analysis, and conclusions are the authors’ own.

![](images/c03e29b684bbed24d0c18586cf68c4a60b3cd9a616bfb1ca3a2d6018fde3adfe.jpg)  
Figure 10: Example interaction trajectory on ALFWorld. Environment observations (left, blue) provide text observations and admissible actions. The agent (right, green) selects actions based on the goal instruction. StraTA (orange) additionally generates a global strategy at Step 0, which is prepended to every subsequent action turn.

![](images/6084203589a307577d0198764af5b01f2df4a4023c9364fbe4e33175916d2e86.jpg)  
Figure 11: Prompt used for generating action axis paraphrases. Placeholders {obj:FORM}, {recep:FORM}, and {toggle:FORM} are preserved in the output and filled with entity names at assembly time.

![](images/9078dd852429386d0650574b7ddc3eda8435e03e8f3b9c637456b94081df6f0d.jpg)  
Figure 12: Type-specific suffixes for the action axis (lexical types), appended to the action axis prefix (Figure 11).

![](images/1c169a234969d0aad9758545611cf7e299942f1fec6394748731d58238e22d45.jpg)  
Figure 13: Type-specific suffixes for the action axis (structural types), appended to the action axis prefix (Figure 11).

![](images/de7791043840d8a3563b0283c686a7da0fb3363eb07612bd8cb6b13531b6ca4c.jpg)  
Figure 14: Type-specific suffixes for the action axis (pragmatic types), appended to the action axis prefix (Figure 11).

![](images/0c52f964e5e0afacc6f8f4fefc7e330546ee5e45be377b28e4ff4c77c48a05f6.jpg)  
Figure 15: Type-specific suffixes for the object axis, appended to the object axis prefix (Figure 16).

![](images/a2fbd4168ad75e908d933da13fa7fd7059b7692ad1236a5fe78614ce4081ef4f.jpg)  
Figure 16: Prompt used for generating object axis paraphrases. The {ALFWORLD\_ENV\_OBJECTS} placeholder is replaced with the full entity list at generation time.

![](images/993ffea07e0750fbaaabdfc7b854dcc07769bbb1564f78a17f33dc3ba1a022ad.jpg)

![](images/0c7c0faa634bed34ff54975797c5e6162d3c279d07d929aa1f197e87a7b4262d.jpg)  
Figure 17: Verifier prompts for object axis (top) and action axis (bottom). Generated paraphrases are filtered through these verifiers before inclusion in ALFWorld-Para.

![](images/44ba4e0c9d277f6ab8595b9c4b9b60c0b2ac745e4d298573cae05dca03a22b74.jpg)  
Figure 18: Instruction (top) and object reference (bottom) provided to human evaluators for verifying semantic equivalence of paraphrased instructions.

![](images/6f1d6eaddcef3b4e215d83abc30dce5ab27b78ed183b2468ddac2ca5a5ebfa41.jpg)

![](images/c55a071d7d9847daf224a71c13f656b16151da528aa194e23e8e526815ae476b.jpg)

![](images/1443346e7bc0c734f2a0c183e85591bd5ebd484a6a1d9e511b617bc5e2bd038a.jpg)

![](images/a2bbb4ddb1c9aa38a3233946829f3f968e7209dded8f970ad00a10e33bb732c1.jpg)  
Figure 19: Per-type breakdown of Figure 2, showing every action paraphrase type individually. The top two rows show pretrained LLMs and the bottom two rows show RL-finetuned LLMs. In the first row of each group, success rate (%) is shown on original ALFWorld (None), on each of the nine action paraphrase types (grouped into lexical, structural, and pragmatic), and on the two object paraphrase types. The second row combines each action type with an object paraphrase, Addition (left) or Same-Polarity (right). As in Figure 2, each marker is one model, with marker size indicating parameter count; horizontal bars and diamonds denote the median and mean across models.

![](images/08a2ed6afe1073907f121f1b5733000eae8390172cce1b85a8bc4f4be67f3e33.jpg)  
Figure 20: The five prompt templates of the ALFWorld paraphrase-knowledge probe. {p} is the paraphrased noun; {CHOICES\_BLOCK} lists the 82 entity types in environment (lowercase) form, ordered as objects (42), receptacles (22), toggleable (1) and scenery (17); {TAIL} is identical to the ScienceWorld probe.

![](images/7d5eb51f79f1c17ffb3610e1b419b31dbaf22f9b137a6db305764fdb63a0a8a6.jpg)

![](images/c521469506a4b95349b475b0d8e215c262deebf9179a79cca529e624383f11f6.jpg)  
Figure 21: Instruction (top) and object reference (bottom) provided to human evaluators for verifying semantic equivalence of paraphrased instructions in ScienceWorld-Para.

![](images/980151fdf7f46e8e9bfe22508abcf379b4da6f50bcb1ec135febaac6750c4472.jpg)  
Figure 22: Success rate (SR) (%) on ALFWorld-Para for every action-axis (rows) and object-axis (columns) paraphrase type, without any mitigation (Baseline). Each panel is one model, grouped into ReAct-prompted pretrained LLMs (top) and RL-finetuned sLLMs on the Qwen2.5-7B (middle) and 1.5B (bottom) backbones.

![](images/e1defa237fcc8130cce8e8ddb736534f772881b2586c7fc70cf95f4685a4592a.jpg)  
Figure 23: Success rate (SR) (%) on ALFWorld-Para by paraphrase type under Greedy.

![](images/87c1d31fb903ebb67a6db0af005603b9c45ad214e6ad33dfa6373173a0b35615.jpg)  
Figure 24: Success rate (SR) (%) on ALFWorld-Para by paraphrase type under Self-clarification.

![](images/88dd733efd0882a329bc200262e9513c9bca2ade90b74f2693854e956f7d792a.jpg)  
Figure 25: Success rate (SR) (%) on ALFWorld-Para by paraphrase type under BINDIntent<sub>SBERT</sub>.

![](images/00bf684a58124a073a4c019f2ea770ac66d810e797c2f5439c0c45ce6164cb33.jpg)  
Figure 26: Success rate (SR) (%) on ALFWorld-Para by paraphrase type under BINDIntent (ours).