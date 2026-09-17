# When Edit Flows are Edit Jumps: replicating Edit Flows and EvoFlows

Gabriel Ben´ edict Melanie Buechler Gerard Riera-Sol´ a Chlo\` e de Ancos´ Yves Gaetan Nana Teukam Moritz Freidank

Visium, Switzerland {gabriel.benedict, melanie.buechler, gerard.riera, chloe.deancos, yves.nana, moritz.freidank}@visium.com

## Abstract

Antibody lead optimization calls for a small, bounded set of edits to an existing candidate: substitutions, but also insertions and deletions. Edit-based generative models are the only ones that allocate such an edit budget without fixing the edit positions, the edit count, or the output length in advance. However, the existing approaches Edit Flows and EvoFlows did not release code or complete training specifications. Here, we show that both methods follow the same underlying process — edits firing one at a time, at learned rates, in continuous time — the pure-jump case of generator matching over finite sequences. With EditJumps we introduce the first open implementation of this framework, with a single generalist antibody editor trained on 1.66M Observed Antibody Space homolog pairs to propose homolog-like variants of a seed sequence, editing unseen leads zero-shot, without the per-family retraining original approaches require. Replicating this system from scratch exposes why open code is essential for generative biology: reconciling published edit distributions required reverse-engineering an undocumented rate-scaling hyperparameter that dictates realized mutation counts. Moreover, we show that published evaluation metrics are highly sensitive to reference sample size, frequently flipping method rankings. We release our full codebase, automated test suite, and configurations at: https://github.com/VisiumCH/editjumps

## 1 Introduction

Lead optimization of a protein candidate aims to explore the local sequence landscape around a starting molecule by introducing small controlled edits that balance exploratory novelty with evolutionary naturalness. Doing this by rational design is difficult: mutations interact non-linearly across the sequence, so the effect of any one change is difficult to anticipate and the useful edits cannot be enumerated by inspection. A model for this task must therefore be able to place a small number of edits itself, by deciding where and what to change, while allowing the sequence length to vary as it would in natural homologs.

Standard generative architectures rarely provide this full combination. Masked protein language models such as ESM-2 [18] resample residues at given masked positions and return a sequence of the input length, so the positions are an input and no length change is possible. Antibody infilling models such as IgLM [25] are autoregressive: the span boundaries are supplied by the user, and the span is regenerated left-to-right, so the model can rewrite it entirely and the number of edits is not controlled. Discrete diffusion models such as EvoDiff [1] need no per-position choice unconditionally, but draw the length before decoding and require a mask for conditional generation. Edit-based models in natural language processing [27, 13, 24] place edits without committing to positions, but they apply them in a fixed refinement procedure and attach no rate to an edit, so nothing sets the expected number. Edi Flows [14] and EvoFlows [6] do attach rates and generate in continuous time. Despite the promising aspects of those models, the lack of released codebases and complete training specifications has hindered the adoption of those methods in the field. This paper aims to reconstruct both methods and ground them with the technical details needed for wider adoption.

![](images/883bfc756e8aa039d634364479eb8aefab8b761a2967dac47fcc8e65c1c57e62.jpg)  
Figure 1: Balancing protein edit diversity and property conservation of the reference family (i.e., naturalness) We measure diversity as the ratio of mean pairwise Levenshtein distance between the generated set and random real homologs pairs. Naturalness is the ratio of homolog-generated pairs correlation and homolog-homolog correlation. The real homolog baseline is thus homolog-homolog on both sides of the ratio and produces 1. Evotuning, and evotuning (forced) are simple finetuning methods on the ESM-2 transformer model, and EvoDiff-MSA is a diffusion model. EditJumps is our reformulation of EvoFlows. It defines the optimal frontier between naturalness and diversity. The data is pooled over two heldout antibody families, which motivated the calculation of ratios (see Table 1).

Both originate from flow matching, which regresses a velocity field onto the conditional velocity field of a prescribed probability path [19]. It was formulated on continuous vector spaces, thus the conditions of discrete tokens and variable length don’t apply. Later work lifts the first condition by running a continuous-time Markov chain (CTMC) over tokens at fixed length [4, 11]. Edit Flows lifts the second condition, defining a CTMC on sequences of length at most N with a rate for every single-token insertion, deletion and substitution available in the current state. EvoFlows adapts this to proteins by adding time conditioning and per-position rate and token heads to a pretrained ESM-2 encoder. Edit Flows trains a Llama-architecture transformer [12] from scratch, whereas EvoFlow fine-tunes its trunk jointly with the rate heads. We adopt the latter configuration.

While Edit Flows and EvoFlows were originally presented as extensions of flow matching, we re-formalize both methods as the pure-jump case of Generator Matching [17], reflecting that discrete sequences evolve through instantaneous jump events rather than continuous flows. This distinction is not merely notational. Because insertion opportunities scale with sequence length, the total jump rate from a state grows unbounded, requiring two necessary safeguards: a non-explosion condition to ensure well-defined waiting times (which Edit Flows achieves via a hard length cap, whereas EvoFlows leaves unconstrained), and a length-dependent clock normalization to prevent the realized jump count from growing with candidate length. Although EvoFlows notes the necessity of clock normalization, it omits both its mathematical formulation and operational value (Section 2).

We implement EditJumps, which unifies both methods under a shared pure-jump rate and kernel abstraction, and instantiate the protein editing configuration of EvoFlows. We replicate its training protocol, including source–target sequence coupling, pairwise alignment across unequal lengths, and the induced conditional probability paths. In the absence of a released reference codebase, we evaluate our replication against published derivations, pseudocode, and reported figures, uncovering two primary findings. First, published and replicated edit statistics can be reconciled only by identifying the clock normalization as the parameter governing realized edit count. Second, six of the ten evaluation metrics reported in EvoFlows cannot be deterministically computed from the text alone. Some other metrics like the spectrum Maximum Mean Discrepancy (MMD) estimator is sensitive to reference set size. Our grounded benchmark avoids this confounding normalization (Figure 1): at matched budgets of 4.1–4.6 edits per sequence, EditJumps produces greater sequence diversity than evotuning [2] and the alignment-based diffusion model EvoDiff-MSA [1] while achieving comparable distributional fidelity, matching unnormalized MMD with standard evotuning on the anti-SARS-CoV-2 nanobody Ty1 and falling within 3% on the anti-HER2 heavy chain HER2-VH. We note that our evaluation is strictly in silico, and the benchmark targets (the camelid single-domain V

Ty1 and isolated heavy chain HER2-VH) reflect single-domain constructs, contrasting with the paired heavy-light chain sequences used during generalist pretraining. We make the following contributions: (i) We identify Edit Flows and EvoFlows as instances of the pure-jump case of Generator Matching, with a common jump kernel, and state the non-explosion and clock-normalization conditions a variable-length space requires. (ii) We release EditJumps, an open-source PyTorch implementation covering both methods, with checkpoints, configurations and evaluation code for the EvoFlows setting. These are a specified baseline rather than a match to the published values, which the reported conditions do not allow us to recover (Section 4). (iii) We train one editor on 1.66M homologous pairs from the paired subset of the Observed Antibody Space [22] and apply it to held-out targets, removing the per-target retraining the original requires. No per-target-retrained EvoFlows exists to compare against. (iv) We report a replication analysis of EvoFlows: six of ten reported metrics are not deterministically recomputable from the text, replicating the published edit statistics requires an unstated clock normalization, and one metric reverses method rankings under a change of reference set.

## 2 Background

Although Edit Flows and EvoFlows are formulated in continuous time, the generated sequence itself does not evolve continuously. Instead, it remains unchanged until an insertion, deletion, or substitution moves it to a new sequence. It is the transition rates of these edits that vary with time. This combination of continuous time and discrete changes is characteristic of a jump process [7] “in a small time interval there is an overwhelming probability that the state will remain unchanged; however, if it changes, the change may be radical”. The term also appears in Discrete Walk-Jump Sampling [9], where a jump is the one-step denoising that projects a noisy sample back onto the data manifold, decoupled from the Langevin walk that precedes it. Here, we use jump throughout in the pure-jump sense above: a transition of the state itself, occurring at a rate.

Generator Matching provides a unified framework for generative modelling through infinitesimal generators, encompassing continuous flows, diffusion processes, and discrete jump processes [17]. We use this framework to formalise the sequence-editing dynamics of Edit Flows and EvoFlows as a pure-jump process. We first define the jump process and its generator, then express insertions, deletions, and substitutions as structured jumps. Finally, we establish the conditions under which this formulation holds and introduce its implementation, EditJumps.

## 2.1 Sequence evolution as a jump process

We now formalise the sequence-editing dynamics introduced above. Let $x = ( x ^ { 1 } , \ldots , x ^ { n } )$ denote a finite sequence of length $n ,$ with each element $x ^ { i }$ drawn from a finite vocabulary V. We denote by X the set of all such sequences. Although n is potentially unbounded, both V and each sequence are finite, making $\mathcal { X }$ countable. We consider a continuous-time stochastic process $( x _ { t } ) _ { t \in [ 0 , 1 ] }$ , where t is the time, with $x _ { 0 }$ drawn from a source distribution (e.g. noise or a start sequence) and $x _ { 1 }$ from the target data distribution. As described above, $x _ { t }$ remains constant between discrete events, which we call jumps.

Given a point in time t and the current state $x ,$ , the process is specified by a non-negative function $\lambda _ { t } ( \cdot \mid x _ { t } )$ , representing the rate at which the state jumps from x to a state $y \neq x$

We write $\begin{array} { r } { \Lambda _ { t } ( x ) = \sum _ { y \neq x } \lambda _ { t } ( y \mid x ) } \end{array}$ for the total rate of leaving x, and $J _ { t } ( y \mid x ) = \lambda _ { t } ( y \mid x ) / \Lambda _ { t } ( x )$ for the probability of the state landing on a new sequence y, given that the jump occurs.

The process is then defined by the infinitesimal sampling step:

$$
x _ { t + h } = \left\{ { \begin{array} { l l } { x _ { t } } & { { \mathrm { w i t h ~ p r o b a b i l i t y ~ } } 1 - \Lambda _ { t } ( x _ { t } ) h + o ( h ) , } \\ { y \sim J _ { t } ( \cdot \mid x _ { t } ) } & { { \mathrm { w i t h ~ p r o b a b i l i t y } } \qquad \Lambda _ { t } ( x _ { t } ) h + o ( h ) , } \end{array} } \right.\tag{1}
$$

following Holderrieth et al. [17, Eq. 80]. For small $h ,$ the process, therefore, remains in its current state with high probability, while jumps occur with probability proportional to $h$

Taking the rate of change as $h  0$ produces the generator $\mathcal { L } _ { t } ,$ which governs the infinitesimal transition from state x at time t. Its action on any test function $f$ is expressed as

$$
\begin{array} { r l } & { ( \mathcal { L } _ { t } f ) ( x ) = \underbrace { \Lambda _ { t } ( x ) \displaystyle \sum _ { y \neq x } J _ { t } ( y \mid x ) \left[ f ( y ) - f ( x ) \right] } _ { \mathrm { t o t a l j u m p r a t e } } = \underbrace { \sum _ { y \neq x } \lambda _ { t } ( y \mid x ) \left[ f ( y ) - f ( x ) \right] } _ { \mathrm { i n d i v i d u a l ~ t r a n s i t i o n ~ r a t e s } } . } \end{array}\tag{2}
$$

This is the pure-jump case of Holderrieth et al. [17] (see equations 81-86).

The transition rates $\lambda _ { t } ( \cdot \mid x )$ that define the target dynamics are generally not available in closed form, therefore the generator $\mathcal { L } _ { t }$ is parameterised by a neural network. At a given state x and time $t ,$ the network aims to approximate the marginal rate obtained by taking the expected value over the conditional rates of transitioning to x from any other state at time t. This can be achieved by optimising a Bregman divergence. A key property of this family of loss functions is that its minimiser is the conditional expectation of the corresponding rate, making it the adequate choice for this task. Depending on data modality, different Bregman divergences can be employed, including squared error for real-valued vectors and cross-entropy for categorical variables.

Once learned, these marginal transition rates fully specify the jump dynamics through (2). For sampling, they are decomposed into the total jump rate $\Lambda _ { t } ( x )$ , which governs the probability of jumping, followed by a sampling process from the jump kernel $J _ { t } ( \cdot \mid ^ { - } x )$ , which determines its destination. We derive the corresponding sampling procedures in Section B.

## 2.2 Sequence edits as structured jumps

Directly parameterising $\lambda _ { t } ( \cdot \mid x )$ over the sequence space $\mathcal { X }$ is intractable. Additionally, only states reachable from x through a valid edit have non-zero transition rate. We therefore formulate the transition rates in terms of three distinct single-token edit operations, each acting on an individual element (i.e. amino acid) of the current sequence x: del<sub>i</sub> removes $x ^ { i }$ , ins $^ { \dag , v }$ inserts v at position i, and su $\scriptstyle ) _ { i , v }$ replaces $x _ { i }$ by v, for $1 \leq i \leq n$ and $v \in \nu$ . Each jump has an explicit structure given by its edit type, position, and, where applicable, the resulting token. We refer to these as structured jumps.

Since the target sequence y may be reached through multiple edits $e ,$ the transition rate between states is obtained by summing the rates of all edits mapping x to $y \colon$

$$
\lambda _ { t } ( y \mid x ) = \sum _ { e : e ( x ) = y } \lambda _ { t } ( e \mid x ) .\tag{3}
$$

Further, we can factorise insertion and substitution rates into a position-specific edit rate and a conditional distribution over the resulting token:

$$
\lambda _ { t } ( \mathsf { i n s } _ { i , v } \mid x ) = \lambda _ { t } ^ { \mathsf { i n s } } ( i \mid x ) ~ q _ { t } ^ { \mathsf { i n s } } ( v \mid i , x ) , \qquad \lambda _ { t } ( \mathsf { s u b } _ { i , v } \mid x ) = \lambda _ { t } ^ { \mathsf { s u b } } ( i \mid x ) ~ q _ { t } ^ { \mathsf { s u b } } ( v \mid i , x ) ,\tag{4}
$$

with q a probability distribution over V. In contrast, deletion removes the token at position i without introducing a new token and therefore does not require a distribution over V. Its rate is given directly by $\lambda _ { t } ( \mathsf { d e l } _ { i } ^ { \mathsf { ^ { * } } } \mid x ) = \lambda _ { t } ^ { \mathsf { d e l } } ( i \mid x )$

For every time t, sequence x, and position i, the neural network therefore outputs five quantities: three position-specific rates for insertion, deletion, and substitution, and two token distributions for insertion and substitution. This corresponds to the parameterisation of Edit Flows [14, Eqs. 13–15] and the head structure used in our EvoFlows implementations, EditJumps.

## 2.3 Non-Explosion and Clock Normalization

On the countable state space X , the editing dynamics define a continuous-time Markov jump process whose infinitesimal generator is given by (2), corresponding to the pure-jump formulation of generator matching [17]. However, translating this construction to variable-length sequences introduces two distinct regularity challenges that are left unformalized in the original literature.

Non-explosion on unbounded sequence spaces. Because insertion operations can occur at any position, the total transition rate $\Lambda _ { t } ( x )$ scales with sequence length $\left| x _ { t } \right|$ . In CTMC, a pure birth process whose transition rates scale linearly (or superlinearly) with the current state can undergo explosion, accumulating infinitely many jumps and infinite length within a finite time horizon $t \leq 1$ [7]. A jump process is well-posed on [0, 1] only if the explosion time $\tau _ { \infty } = \operatorname* { i n f } \{ t : | x _ { t } | = \infty \}$ satisfies $\mathbb { P } ( \tau _ { \infty } > 1 ) = 1$ . In practice, Havasi et al. [14] circumvents explosion by restricting the state space to sequences below a fixed maximum length $L _ { \mathrm { m a x } }$ , while Deutschmann et al. [6] leaves the sequence space nominally unbounded.

Length-invariant clock normalization. Even when trajectories remain non-explosive, summing transition rates over positions induces an intrinsic length bias: longer sequences accumulate proportionally more edits per unit time than shorter sequences under identical per-position rates. In therapeutic protein design, lead optimization requires a consistent mutation rate per position regardless of whether the lead is a short peptide or a full variable domain. To decouple the expected edit fraction from sequence length, transition rates must be rescaled by a length-dependent clock factor. While Deutschmann et al. [6] notes that a clock normalization was applied, the published paper specifies neither the functional form nor the numerical parameter value. As we demonstrate in Section 4, this unstated hyperparameter directly governs the realized edit scale, and tuning it resolves the observed divergence discrepancy between published and reproduced runs.

## 3 Replication Setup

Neither Edit Flows [14] nor EvoFlows [6] released reference code or complete training data specifications. To reflect the practical demands of therapeutic lead optimization—where retraining per target is computationally prohibitive—we train a single generalist antibody editor across the Observed Antibody Space [22], enabling zero-shot editing of novel leads without per-family retraining. All configurations, data manifests, and model cards will be open-sourced.

Data curation and evaluation targets. Training pairs are constructed from paired heavy and light chains (VH.VL) in OAS, clustered via MMseqs2 at 0.5 identity and 0.8 coverage (see Appendix D). This antibody-wide pretraining restricts downstream evaluation to the two accessible antibody targets from Deutschmann et al. [6]: Ty1 (anti-SARS-CoV-2 VHH) and HER2-VH (anti-HER2 heavy chain), each evaluated across 20 templates and 200 held-out homologs. Non-antibody targets have no homologs in OAS, and anti-EphA2 is unrecoverable in public repositories (Appendix D).

Architecture and training protocol. We parameterize the editor with an unfrozen, pretrained ESM 2 encoder [18]. Prediction heads receive representations conditioned on sinusoidal time embeddings via Feature-wise Linear Modulation (FiLM): three per-position operation rate heads (insertion, deletion, substitution) and two token prediction heads. An ESM-2 35M-parameter trunk matches the 650M trunk in realized edit scale (0.180 vs. 0.182) and distributional fidelity at a fraction of the computational cost (Appendix E). The model is fine-tuned for 20,000 steps (batch size 16, Adam, learning rate $1 0 ^ { - 4 } )$ ; Table 1 reports individual runs and intervals across three seeds (Appendix D).

Supervision discrepancies and deviations. Training operates on aligned pairs. When an edit reaches an aligned gap, the published loss and code conflict: Equation 23 correctly assigns zero loss to gap sites, whereas an indexing bug in the reference code mistakenly supervises deletions on adjacent valid residues across 49.2% of training pairs. (Appendix C). We adhere to Equation 23. Four additional operational adaptations depart from the original specifications: linear rather than MLP rate heads (Figure 3), a linear schedule $\kappa _ { t } = t$ matching released code (Appendix C), frozen holdinginterval rates (Appendix B), and a 0.7 seed identity threshold for homolog retrieval (Appendix D). Further fine-grained deviations and interpretations are listed in the model cards<sup>1</sup>.

The evotuning baseline. Following Deutschmann et al. [6, §4.2], evotuning matches the realized mutation budget of EditJumps $( b = 5$ on Ty1, b = 4 on HER2-VH). Target positions are drawn without replacement from a per-column Shannon entropy profile derived from the training alignment (concentrating 50% of mask probability onto just 12% of sequence coordinates) and infilled iteratively with an evotuned ESM-2 MLM. Because the unforced MLM frequently re-predicts the template residue (71.7% unchanged on Ty1), we also evaluate a forced variant that is not allowed to edit

Table 1: EvoFlows’ six baselines, reimplemented and evaluated by us (top block) per family: anti-SARS-CoV-2 Ty1, and anti-HER2 construct. The two blocks are not comparable; each row compares against its own block’s floor and ceiling, respectively random pairing and random mutations.
<table><tr><td></td><td colspan="2">edits/seq</td><td colspan="2">diversity</td><td colspan="2">MMD↓</td><td colspan="2"> ${ \mathrm { K L } } \times 1 0 ^ { 3 } { \downarrow }$ </td></tr><tr><td>method</td><td>Ty1</td><td>HER2</td><td> $\mathrm { T y } 1$ </td><td>HER2</td><td>Ty1</td><td>HER2</td><td>Ty1</td><td>HER2</td></tr><tr><td colspan="9">ours — two antibody families</td></tr><tr><td>random pairing</td><td>23.38</td><td>23.25</td><td>23.53</td><td>23.02</td><td>0.60</td><td>0.47</td><td>0.14</td><td>0.10</td></tr><tr><td>EditJumps</td><td>4.58</td><td>4.08</td><td>24.47</td><td>24.73</td><td>0.98</td><td>1.02</td><td>0.50</td><td>0.63</td></tr><tr><td>evotuning</td><td>4.47</td><td>3.83</td><td>22.61</td><td>22.77</td><td>0.98</td><td>0.99</td><td>0.54</td><td>0.37</td></tr><tr><td>evotuning (forced)</td><td>5.00</td><td>4.00</td><td>25.22</td><td>25.00</td><td>1.09</td><td>1.15</td><td>0.53</td><td>0.59</td></tr><tr><td>EvoDiff-MSA</td><td>2.96</td><td>2.61</td><td>22.85</td><td>23.23</td><td>0.97</td><td>1.11</td><td>0.61</td><td>0.42</td></tr><tr><td>random mutations</td><td>4.81</td><td>3.89</td><td>30.01</td><td>29.08</td><td>1.41</td><td>1.38</td><td>0.77</td><td>0.79</td></tr><tr><td colspan="9">the original — the same two families, its own reported values</td></tr><tr><td>random pairing</td><td>30.52</td><td>93.86</td><td>30.28</td><td>93.27</td><td>0.38</td><td>0.61</td><td>9.84</td><td>8.72</td></tr><tr><td>EvoFlows</td><td>14.64</td><td>35.27</td><td>35.59</td><td>90.86</td><td>1.03</td><td>1.57</td><td>21.49</td><td>23.17</td></tr><tr><td>evotuning</td><td>2.54</td><td>11.75</td><td>30.99</td><td>96.14</td><td>0.40</td><td>0.66</td><td>10.40</td><td>10.66</td></tr><tr><td>evotuning (forced)</td><td>15.30</td><td>38.67</td><td>41.07</td><td>114.67</td><td>1.88</td><td>2.65</td><td>54.79</td><td>36.48</td></tr><tr><td>EvoDiff-MSA</td><td>4.43</td><td>13.80</td><td>30.01</td><td>93.06</td><td>0.42</td><td>0.66</td><td>11.63</td><td>10.16</td></tr><tr><td>random mutations</td><td>13.94</td><td>35.09</td><td>48.79</td><td>133.34</td><td>2.35</td><td>3.70</td><td>80.05</td><td>86.68</td></tr></table>

masks back to their value in the previous step, isolating how much of evotuning’s fidelity reflects generative editing versus non-intervention. Unlike our generalist editor, evotuning is restricted to substitutions at fixed sequence length and requires retraining a separate 650M trunk for each target family (Appendix A).

## 4 Results

Before interpreting our results in Table 1, we note that our computed values and published EvoFlows values are not comparable and we could not recover the original experimental conditions from the paper. Against evotuning baselines, EditJumps outperforms forced evotuning across distributional metrics, while demonstrating parity with standard evotuning on k-mer MMD despite higher composition divergence and $\mathbf { a } \sim 4 \%$ increase in edit budget. Performance across families exhibits domain heterogeneity: MMD differences are 0.08 on Ty1 and 0.035 on HER2-VH. On composition KL, EditJumps improves substantially over random mutation on Ty1 while remaining comparable to it on HER2-VH. All learned models outperform uniform random mutation, yet remain separated from the natural homology floor (0.98 MMD versus 0.60).

Sensitivity to the clock normalization hyperparameter. The edit budget is guided by the clock normalization hyperparameter (§3.3 of Deutschmann et al. [6]). A sweep across both families and three seeds reveals that a clock rate of approximately 95 replicates the published edit ratio of 0.43 in [6](Figure 2).

Identifying EvoFlows’ 10 evaluation metrics. Examining the ten evaluation panels in Figure 3 of Deutschmann et al. [6] reveals that six cannot be deterministically recomputed from the text alone. Four metrics are well-specified (one modulo an unstated kernel parameter for MMD). Of the remaining six, three are referenced solely by name but have standard mathematical formalization; two define pairwise matrices (pair covariance and mutual information product) without specifying scalar reductions; and one metric admits multiple incompatible reductions (Levenshtein Distance). Nevertheless, we replicate all 10 metrics in Appendix A and in code.

## 5 Discussion and Limitations

In computational protein engineering, in silico metrics such as sequence recovery, contact-map covariance, and Mutual Information Product (MIP) serve as common surrogates for structural fold preservation and functional viability prior to wet-lab assaying. However, our findings demonstrate that without grounded empirical baselines, these metrics are easily misinterpreted. An exact sequence recovery rate of 0.21 appears modest in isolation, yet represents a 300-fold increase over chance at matched edit budgets for antibody leads (Appendix E). Because covariance recovery shifts from 83.6% to 85.1% simply by varying reference partitions, ungrounded metrics risk rewarding sequence conservatism rather than genuine biophysical fitness.

![](images/5e85b4ddf00c74227e66280418beed4a993858ecdba7a73ce8961a0cbb8c411e.jpg)  
Figure 2: Sensitivity analysis of the clock normalization hyperparameter. Distributional distance (spectrum MMD) to held-out natural homologs as a function of the clock normalization parameter governing the edit budget, across two antibody families and three random seeds (solid: Ty1; dashed: HER2-VH). For our evaluation we set a scenario where a budget of ∼ 5 edits was allowed. A clock normalization rate that optimizes MMD sits around 80 or about 8.5 edits.

Furthermore, cross-study benchmarking frequently normalises scores. We show this holds for relative edit distance, but fails for distributional estimators like spectrum MMD: expanding the reference alignment from 200 to 800 homologs alters the normalised ratio by 50% on Ty1 and 26% on HER2- VH, a difference sufficiently large to invert method rankings. Because both the numerator and the baseline floor contract with sample size, internal normalisation cannot replace standardised reference sets. Encouragingly, core algorithmic formulations remain robust: the symmetric homolog pair enumeration protocol in Deutschmann et al. [6] replicated cleanly without discrepancy (Appendix D), showing that replication challenges in discrete protein editing stem primarily from underspecified evaluation protocols rather than algorithmic indeterminacy.

Limitations. For antibody lead optimisation, targeted mutations must balance liability removal against paratope disruption. A key limitation of clock normalisation is the absence of a closed-form mapping to realised edits: because jump rates vary dynamically with time and sequence context, achieving a specific edit budget requires empirical calibration rather than deterministic control. Biophysically, region-resolved analysis reveals that fidelity is substantially driven by conserved framework scaffolds; within hypervariable CDR loops—which dictate antigen binding—the model closes < 25% of the gap between random mutation and natural homologs (Appendix E). Methodologically, our training regime on concatenated variable domains (V H.V L) introduces a domain shift when evaluated on isolated heavy chains and nanobodies across two target families (Ty1 and HER2-VH). Finally, and exact comparisons against published baseline rows remain constrained by missing intermediate checkpoints in upstream releases (Appendix D).

## 6 Related Work

The methods reproduced here combine two lines of generative modelling: discrete generative processes and edit-based sequence generation.

Towards discrete generative processes. Diffusion models generate by reversing a fixed corruption of the data [26, 16], and structured discrete diffusion carries this construction to discrete token states [3], as do score-based and continuous-time formulations on those states [20, 4]. Flow matching changes the training regime: a conditional probability path from source to target is fixed in advance and regressed onto directly, without simulating a reverse process [19], and discrete flow matching applies this training to token sequences [11]. These models hold the sequence at a fixed length and change a position only by substitution. None of them inserts, deletes, or varies the length of a sequence.

Sequence editing. Edit-based models generate a sequence through a series of edits, a line of work older than the flow framing. The Insertion Transformer places tokens at arbitrary positions instead of filling a fixed grid of masks [27]. The Levenshtein Transformer adds deletion, so that a model produces the length of an output rather than being given a fixed length [13]. DiffusER [24] run insertion and deletion as the steps of a denoising process. These models apply edits through an iterative policy trained against an oracle, in a fixed number of discrete passes, and assign no time or rate to any edit.

Combining discrete generation and edits. Edit Flows [14] and EvoFlows [6] combine the two lines. They take insertion, deletion, and substitution from edit-based generation and assign each a rate under the conditional-path training of flow matching, so that generation runs in continuous time from one sequence to another. The rate makes the expected number of edits a parameter of the sampler, set by the scale of the rates, and makes the generative process a CTMC over sequences.

A jump process. Generator matching organises diffusion, flow matching, and pure jump processes as one design space, separated by the form of the generator [17]. On the countable state space of token sequences the combined construction is a pure jump process. The construction inherits the training path of flow matching and not the dynamics. This motivates our EditJumps framing.

Protein generative design, and replication. EvoFlows [6, §3.4] is framed on protein sequences using a pretrained ESM-2 trunk [18]. It sits within a body of generative protein models that produce whole outputs: sequences, whether autoregressive [21, 8], antibody infilling [25], or alignmentconditioned [23, 1], and structure-based backbone design [28, 5], against which the editing formulation outputs a bounded set of residue-level edits to one given sequence. While surrounding work explores active flow expansion using entropy metrics like the Vendi score [10], our benchmark strictly follows EvoFlows [6] using pooled pairwise distance and k-mer MMD. Finally, Discrete Walk-Jump Sampling [9] falls short for antibody lead optimization, because it requires fixing masks to infill, and it cannot calibrate an edit budget (similarly to DiffusER above).

## 7 Conclusions

We formulate the continuous-time sequence editing framework shared by Edit Flows and EvoFlows within a unified formulation of CTMC and pure jump processes. Framing discrete sequence design as a jump process provides a principled alternative to discrete diffusion and autoregressive generation, equipping discrete token mutations with continuous holding times and operation-specific transition rates (insertion, deletions, substitutions). For therapeutic lead optimisation, where conservative residue modifications must remediate biophysical liabilities without disrupting antigen binding, this framework establishes a direct bridge between continuous generative flow matching and variablelength sequence engineering.

Our empirical investigation resolves specification challenges in Edit Flows and EvoFlows (Table 1 and Table 5). We demonstrated that the apparent divergence in realised sequence novelty between published claims and standard implementations stems from an unstated clock rate normalisation. Calibrating this edit counts hyperparameter balances sequence diversification against structural preservation (see Figure 1). When evaluated against grounded empirical baselines, continuous-time jump models outperform uniform random mutation and forced evotuning, yet remain separated from natural homology floors, highlighting the ongoing challenge of introducing diverse edits while preserving natural sequence context.

Our findings suggest three concrete recommendations for discrete generative biology: (1) parameterizing operational clock rates or adaptive stopping criteria directly within the model, rather than relying on uncalibrated manual rate multipliers; (2) adopting standardized metrics to avoid sample-size arti facts in distributional metrics like MMD; and (3) evaluating generative baselines under both unforced and forced regimes to separate active generative design from passive wild-type conservation.

## AI Use Statement

LLM-based assistance was utilized across code development, data analysis, and manuscript preparation under direct author supervision.

Implementation and experiments. Training and evaluation pipelines, analysis scripts, and figuregeneration routines were developed with LLM-based coding assistance. All code implementations, algorithmic workflows, and metric definitions were audited, verified, and executed under the direct oversight of the authors.

Writing and analysis. LLM assistance was employed to draft and refine portions of the text, format comparison tables, and cross-reference citations against publisher records. All empirical claims and recovered data points were independently validated against raw artifacts by the authors, who take full intellectual and editorial responsibility for the entire manuscript.

## References

[1] Sarah Alamdari, Nitya Thakkar, Rianne van den Berg, Alex X. Lu, Nicolo Fusi, Ava P. Amini, and Kevin K. Yang. Protein generation with evolutionary diffusion: sequence is all you need. bioRxiv, 2023. doi: 10.1101/2023.09.11.556673.

[2] Ethan C. Alley, Gautam Khimulya, Surojit Biswas, Mohammed AlQuraishi, and George M. Church. Unified rational protein engineering with sequence-based deep representation learning. Nature Methods, 16(12):1315–1322, 2019. doi: 10.1038/s41592-019-0598-1.

[3] Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems 34 (NeurIPS), 2021. URL https://arxiv.org/abs/2107.03006.

[4] Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. In International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2402.04997.

[5] Justas Dauparas, Ivan Anishchenko, Nathaniel Bennett, Hua Bai, Robert J. Ragotte, Lukas F. Milles, Basile I. M. Wicky, Alexis Courbet, Rob J. de Haas, Neville Bethel, Philip J. Y. Leung, Timothy F. Huddy, Sam Pellock, Doug Tischer, Frederick Chan, Brian Koepnick, Hannah Nguyen, Alex Kang, Banumathi Sankaran, Asim K. Bera, Neil P. King, and David Baker. Robust deep learning-based protein sequence design using ProteinMPNN. Science, 378(6615): 49–56, 2022. doi: 10.1126/science.add2187.

[6] Nicolas Deutschmann, Constance Ferragu, Jonathan D. Ziegler, Shayan Aziznejad, and Eli Bixby. Evoflows: Evolutionary edit-based flow-matching for protein engineering, 2026. URL https://arxiv.org/abs/2603.11703. Accepted at the ICLR 2026 Workshop on Foundation Models for Science.

[7] William Feller. On the theory of stochastic processes, with particular reference to applications. In Proceedings of the (First) Berkeley Symposium on Mathematical Statistics and Probability, volume 1, pages 403–432. University of California Press, 1949.

[8] Noelia Ferruz, Steffen Schmidt, and Birte Hocker. ProtGPT2 is a deep unsupervised lan-¨ guage model for protein design. Nature Communications, 13:4348, 2022. doi: 10.1038/ s41467-022-32007-7.

[9] Nathan C. Frey, Daniel Berenberg, Karina Zadorozhny, Joseph Kleinhenz, Julien Lafrance-Vanasse, Isidro Hotzel, Yan Wu, Stephen Ra, Richard Bonneau, Kyunghyun Cho, Andreas Loukas, Vladimir Gligorijevic, and Saeed Saremi. Protein discovery with discrete walk-jump sampling. In International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2306.12360. Oral. Outstanding Paper Award.

[10] Dan Friedman and Adji Bousso Dieng. The Vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research, 2023. URL https://arxiv. org/abs/2210.02410.

[11] Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. In Advances in Neural Information Processing Systems 37 (NeurIPS), 2024. URL https://arxiv.org/abs/2407.15595.

[12] Aaron Grattafiori, Abhimanyu Dubey, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[13] Jiatao Gu, Changhan Wang, and Junbo Zhao. Levenshtein transformer. In Advances in Neural Information Processing Systems 32, 2019.

[14] Marton Havasi, Brian Karrer, Itai Gat, and Ricky T. Q. Chen. Edit flows: Variable length discrete flow matching with sequence-level edit operations. In Advances in Neural Information Processing Systems 38 (NeurIPS), 2025. URL https://arxiv.org/abs/2506.09018.

[15] Steven Henikoff and Jorja G. Henikoff. Amino acid substitution matrices from protein blocks. Proceedings of the National Academy of Sciences, 89(22):10915–10919, 1992. doi: 10.1073/ pnas.89.22.10915. URL https://doi.org/10.1073/pnas.89.22.10915.

[16] Jonathan Ho, Ajay N. Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems 33 (NeurIPS), 2020. URL https: //arxiv.org/abs/2006.11239.

[17] Peter Holderrieth, Marton Havasi, Jason Yim, Neta Shaul, Itai Gat, Tommi Jaakkola, Brian Karrer, Ricky T. Q. Chen, and Yaron Lipman. Generator matching: Generative modeling with arbitrary Markov processes. In International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2410.20587. Oral.

[18] Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Salvatore Candido, and Alexander Rives. Evolutionary-scale prediction of atomiclevel protein structure with a language model. Science, 379(6637):1123–1130, 2023. doi: 10.1126/science.ade2574. URL https://doi.org/10.1126/science.ade2574.

[19] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2210.02747. arXiv v1 2022; ICLR 2023.

[20] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2310.16834.

[21] Ali Madani, Ben Krause, Eric R. Greene, Subu Subramanian, Benjamin P. Mohr, James M. Holton, Jose Luis Olmos, Caiming Xiong, Zachary Z. Sun, Richard Socher, James S. Fraser, and Nikhil Naik. Large language models generate functional protein sequences across diverse families. Nature Biotechnology, 41(8):1099–1106, 2023. doi: 10.1038/s41587-022-01618-2.

[22] Tobias H. Olsen, Fergus Boyles, and Charlotte M. Deane. Observed antibody space: A diverse database of cleaned, annotated, and translated unpaired and paired antibody sequences. Protein Science, 31(1):141–146, 2022. doi: 10.1002/pro.4205.

[23] Roshan M. Rao, Jason Liu, Robert Verkuil, Joshua Meier, John Canny, Pieter Abbeel, Tom Sercu, and Alexander Rives. MSA transformer. In Proceedings of the 38th International Conference on Machine Learning, 2021.

[24] Machel Reid, Vincent J. Hellendoorn, and Graham Neubig. DiffusER: Diffusion via edit-based reconstruction. In International Conference on Learning Representations, 2023.

[25] Richard W. Shuai, Jeffrey A. Ruffolo, and Jeffrey J. Gray. IgLM: Infilling language modeling for antibody sequence design. Cell Systems, 14(11):979–989, 2023. doi: 10.1016/j.cels.2023. 10.001.

[26] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning (ICML), 2015. URL https://arxiv.org/abs/1503.03585.

[27] Mitchell Stern, William Chan, Jamie Kiros, and Jakob Uszkoreit. Insertion transformer: Flexible sequence generation via insertion operations. In Proceedings of the 36th International Conference on Machine Learning, 2019.

[28] Joseph L. Watson, David Juergens, Nathaniel R. Bennett, Brian L. Trippe, Jason Yim, Helen E. Eisenach, Woody Ahern, Andrew J. Borst, Robert J. Ragotte, Lukas F. Milles, Basile I. M. Wicky, Nikita Hanikel, Samuel J. Pellock, Alexis Courbet, William Sheffler, Jue Wang, Preetham Venkatesh, Isaac Sappington, Susana Vazquez Torres, Anna Lauko, Valentin De Bortoli, Emile´ Mathieu, Sergey Ovchinnikov, Regina Barzilay, Tommi S. Jaakkola, Frank DiMaio, Minkyung Baek, and David Baker. De novo design of protein structure and function with RFdiffusion. Nature, 620(7976):1089–1100, 2023. doi: 10.1038/s41586-023-06415-8.

Table 2: EvoFlows’ §4.2 reproduced results, on all ten metrics, both seed families, one frame: 20 templates × 20 variants, holdout 200, agreement ceiling 300 real homologs.
<table><tr><td>method</td><td>edits</td><td>pairw. Lev.</td><td>MMD KL pos. KL pool.</td><td></td><td> $\times \mathrm { \dot { 1 } 0 ^ { 3 } }$ </td><td>JS pos. ×10²</td><td>coV.</td><td>MIP</td><td>∆H</td><td>prof. LL</td></tr><tr><td></td><td></td><td>~</td><td>↓</td><td>↓</td><td>↓</td><td>↓</td><td>agr. ↑</td><td>agr. ↑</td><td>→ 0</td><td>↑</td></tr><tr><td>tyl</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ÉditJumps</td><td>4.58</td><td>24.47</td><td>0.98</td><td>0.07</td><td>0.50</td><td>1.26</td><td>0.91</td><td>0.83</td><td>0.07</td><td>-64.01</td></tr><tr><td>rand. mut.</td><td>4.81</td><td>30.01</td><td>1.41</td><td>0.16</td><td>0.77</td><td>2.23</td><td>0.85</td><td>0.66</td><td>0.15</td><td></td></tr><tr><td>rand. pair.</td><td>23.38</td><td>23.53</td><td>0.60</td><td>0.03</td><td>0.14</td><td>0.54</td><td>0.98</td><td>0.94</td><td>0.07</td><td></td></tr><tr><td>EvoDiff</td><td>2.96</td><td>22.85</td><td>0.97</td><td>0.07</td><td>0.61</td><td>1.38</td><td>0.89</td><td>0.78</td><td>0.02</td><td>-59.15</td></tr><tr><td>Evotune</td><td>4.47</td><td>22.61</td><td>0.98</td><td>0.06</td><td>0.54</td><td>1.14</td><td>0.91</td><td>0.81</td><td>0.01</td><td>-57.49</td></tr><tr><td>Evotune-f</td><td>5.00</td><td>25.22</td><td>1.09</td><td>0.09</td><td>0.53</td><td>1.40</td><td>0.87</td><td>0.74</td><td>0.09</td><td>-67.54</td></tr><tr><td>BF-DiT</td><td>19.77</td><td>33.76</td><td>1.58</td><td>0.22</td><td>0.91</td><td>2.17</td><td>0.38</td><td>0.23</td><td>0.25</td><td>-95.28</td></tr><tr><td>BF-ESM</td><td>31.69</td><td>44.28</td><td>4.31</td><td>0.76</td><td>10.65</td><td>6.28</td><td>0.12</td><td>0.15</td><td>0.33</td><td>-155.11</td></tr><tr><td>her2vh</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EditJumps</td><td>4.08</td><td>24.73</td><td>1.02</td><td>0.05</td><td>0.63</td><td>1.03</td><td>0.93</td><td>0.87</td><td>0.08</td><td>-59.03</td></tr><tr><td>rand. mut.</td><td>3.89</td><td>29.08</td><td>1.38</td><td>0.12</td><td>0.79</td><td>1.89</td><td>0.90</td><td>0.78</td><td>0.15</td><td></td></tr><tr><td>rand. pair.</td><td>23.25</td><td>23.02</td><td>0.47</td><td>0.03</td><td>0.10</td><td>0.47</td><td>0.98</td><td>0.94</td><td>0.10</td><td></td></tr><tr><td>EvoDiff</td><td>2.61</td><td>23.23</td><td>1.11</td><td>0.06</td><td>0.42</td><td>1.18</td><td>0.91</td><td>0.82</td><td>0.04</td><td>-55.03</td></tr><tr><td>Evotune</td><td>3.83</td><td>22.77</td><td>0.99</td><td>0.05</td><td>0.37</td><td>1.02</td><td>0.93</td><td>0.83</td><td>0.03</td><td>-52.87</td></tr><tr><td>Evotune-f</td><td>4.00</td><td>25.00</td><td>1.15</td><td>0.07</td><td>0.59</td><td>1.21</td><td>0.91</td><td>0.81</td><td>0.10</td><td>-62.17</td></tr><tr><td>BF-DiT</td><td>22.38</td><td>37.54</td><td>2.09</td><td>0.26</td><td>1.70</td><td>2.67</td><td>0.38</td><td>0.23</td><td>0.35</td><td>-104.87</td></tr><tr><td>BF-ESM</td><td>33.29</td><td>45.53</td><td>4.69</td><td>0.71</td><td>14.47</td><td>6.30</td><td>0.14</td><td>0.18</td><td>0.40</td><td>-153.82</td></tr></table>

## A Extended results on all metrics

In Table 2, we report on all 10 metrics in the original EvoFlows methodology. We provide an interpretation of those metrics here and in the code

Evotuning Baseline Implementation and Sampling Protocol. Following Alley et al. [2] and Deutschmann et al. [6, §2.2, §4.2], the evotuned baseline is constructed as a two-stage hybrid:

1. Family Adaptation (Evotuning): An esm2 t33 650M UR50D trunk (650M parameters) is fine-tuned independently on the training partition of each antibody family (36,417 sequences for Ty1; 40,489 for HER2-VH) for 2,000 steps (batch size 16, AdamW, learning rate $5 \times \mathrm { 1 \dot { 0 } ^ { - 5 } }$ , 15% masking probability).

2. Positional Entropy Profile (Where to edit): Multiple sequence alignments of the family training set define per-column amino-acid frequencies p<sub>l</sub>(a) (excluding gaps and nonstandard characters). The positional sampling distribution is governed by column-wise Shannon entropy:

$$
H ( l ) = - \sum _ { a \in \mathcal { A } } p _ { l } ( a ) \log \bigl ( p _ { l } ( a ) + \epsilon \bigr ) ,\tag{5}
$$

normalized across sequence positions and mapped to template coordinates. Zero-entropy columns (completely conserved framework residues) are sampled only after all positiveentropy positions are exhausted.

3. Iterative Infilling (What to edit): A budget of b positions is sampled from the entropy profile and replaced with <mask> tokens. The family-adapted MLM infills positions iteratively in random order, feeding predicted tokens back into context at temperature T = 1.0.

4. Budget Matching (Unforced vs. Forced): In the standard unforced regime, the MLM frequently predicts the template residue. To satisfy the requirement of matching the realized mutation count of EditJumps, our implementation tops up masks over up to 8 iterative rounds until b active substitutions are achieved. In theforced regime, template residues are masked from the output logits (−∞), guaranteeing b mutations in a single pass.

Crucially, evotuning is strictly substitution-only and cannot model length variability (indels).

## B Sampling Algorithms and Discretization Analysis

Continuous-Time Sampling Formulations. Inference in EditJumps simulates trajectories from t = 0 to t = 1 under learned transition rates:

1. Euler τ-leaping: Partitions the unit interval into $N$ uniform steps of width $h = 1 / N$ . At each step t, the rate field $\lambda _ { t } ( \cdot \mid x _ { t } )$ is evaluated, and candidate edits fire independently with probability $\lambda _ { t } ( e \mid x _ { t } ) h$ . This scheme is first-order accurate in h and computationally efficient, though multiple non-commuting edits may occasionally propose simultaneously within a step.

2. Exact Continuous-Time Simulation (Gillespie): Simulates event-to-event transitions gridfree. For time-dependent rates, the next jump time τ satisfies $\begin{array} { r } { \int _ { t } ^ { t + \tau } \Lambda _ { s } ( x _ { t } ) \mathrm { d } s = - \ln U } \end{array}$ where $U \sim \mathrm { U n i f o r m } ( 0 , 1 )$ . At the event time, the state transitions to $y \sim J _ { t + \tau } ( \cdot \mid x _ { t } )$ . In our continuous-time implementation, we freeze rates over the holding interval, yielding an efficient first-order continuous-time sampler.

Empirical Discretization Analysis: Euler vs. Exact Gillespie. To quantify discretization error introduced by Euler τ-leaping, we benchmarked Euler against exact Gillespie sampling on a fixed synthetic rate field across 2,000 independent trajectories starting from a sequence of length 120. Against an exact Gillespie mean of 71.29 ± 0.15 edits:

$\mathbf { A } \mathbf { t } N = 2$ steps, Euler over-fires by $4 . 4 4 \pm 0 . 2 0$ edits;

$\mathbf { A } \mathbf { t } \ N = 1 0$ steps, Euler over-fires by $0 . 7 6 \pm 0 . 2 1$ edits;

• At $N = 5 0$ steps (the setting used throughout all experimental runs), Euler over-fires by $0 . 0 8 \pm 0 . 2 1$ edits, representing a relative error of $< ~ 0 . 3 \%$ and demonstrating that discretization error is negligible.

Crucially, clock normalization dictates the direction of discretization error: under clock normalization, expanding sequence length lowers total transition rates, leading Euler’s frozen rates to slightly over fire; in unnormalized regimes, total rates increase with length, causing Euler to under-fire (by $- 4 . 0 3 \pm 0 . 4 2$ edits at $N \overset { \cdot } { = } 2 ,$ ). Consequently, sampler discretization does not account for observed empirical discrepancies.

## C Model Architectures, Training Paths, and Algorithmic Specifications

Augmented Training Path Construction. Coupling arbitrary source and target sequence pairs $( x _ { 0 } , x _ { 1 } )$ is non-trivial due to variable sequence lengths and combinatorial alignment paths. Havasi et al. [14] and Deutschmann et al. [6] resolve this by lifting sequences to an augmented space $\mathcal { Z } = ( \mathcal { V } \bar { \cup } \{ \varepsilon \} ) ^ { N }$ , where ε denotes an alignment gap. Pairs are coupled via optimal Needleman– Wunsch alignment [15]. Each aligned column $( z _ { 0 } ^ { i } , z _ { 1 } ^ { i } )$ specifies a categorical transition:

$\varepsilon $ v: insertion of token $v \in \mathcal V$ at position $i ;$

$u \to \varepsilon \colon$ deletion of token $u \in \mathcal V$ at position $i ;$

$u  v \mathrm { : }$ substitution of token u with v.

The conditional probability path is defined by coordinate-wise linear mixture:

$$
p _ { t } ( z ^ { i } \mid z _ { 0 } ^ { i } , z _ { 1 } ^ { i } ) = ( 1 - \kappa _ { t } ) \delta _ { z _ { 0 } ^ { i } } ( z ^ { i } ) + \kappa _ { t } \delta _ { z _ { 1 } ^ { i } } ( z ^ { i } ) ,\tag{6}
$$

where $\delta$ is the Dirac delta function and $\kappa _ { t } \in [ 0 , 1 ]$ is a schedule satisfying $\kappa _ { 0 } = 0$ and $\kappa _ { 1 } = 1$ . Thus, at an intermediate time $t ,$ the target sequence’s token at every position is sampled with probability $\kappa _ { t }$ and the source sequence’s token with probability $1 - \kappa _ { t }$

Finally, marginal transition rates are trained by regressing predicted rates onto conditional targets via cross-entropy and Bregman divergence objectives.

EvoFlows Architecture and Head Parameterization. EvoFlows utilizes a pretrained ESM-2 trunk fine-tuned jointly [18]. Time conditioning is introduced via sinusoidal embeddings processed through an $\begin{array} { r } { \mathbf { M L P } \colon \bar { \tau } _ { t } = \dot { \mathrm { M L P } } ( \operatorname { S i n u s o i d a l } ( t ) ) } \end{array}$ , modulating token representations across all heads via shared Feature-wise Linear Modulation (FiLM) scale and shift parameters. The five output quantities are parameterized as:

• Three position-specific rate heads $( \lambda ^ { \mathrm { i n s } } , \lambda ^ { \mathrm { d e l } } , \lambda ^ { \mathrm { s u b } } ) \mathrm { : }$ : parameterized as shallow MLPs mapped to positive values via softplus or bounded sigmoid activations;

![](images/4a23e1bf09579be73a2b0c5965f51a51f911a84bf6de9f7364fec5cd65e61ada.jpg)  
Figure 3: Head parameterization sensitivity analysis on Ty1. $\textup { A 2 } \times 2$ factorial comparison of training corpus (OAS vs. stock) against head parameterization: the published Appendix A specification (filled markers; adapting ESM-2 language model heads) versus lightweight linear rate heads with random token heads (open markers). Axes are normalized to natural homologs $( 1 , - 1 )$ . Novelty $\Delta$ measures the change in distance relative to template starting coordinates. Arrows indicate the effect of altering head parameterization at fixed corpus, shifting novelty by 0.52 and 1.24, dominating corpus shifts (0.15).

• Two token prediction heads $( q ^ { \mathsf { i n s } } , q ^ { \mathsf { s u b } } ) \colon$ parameterized either as MLPs from scratch or by adapting ESM-2’s pretrained masked language modeling head.

Figure 3 evaluates the sensitivity of generative dynamics to head parameterization. Using pretrained ESM-2 language prediction heads yields substantially more conservative edits (novelty $\bar { \Delta } \stackrel { - } { = } 0$ .45 vs. 0.61 on OAS; 0.50 vs. 0.89 on stock), with head parameterization accounting for shifts of 0.52–1.24 in novelty, significantly exceeding the effect of training corpus choice (0.15).

Discrepancy in Deletion Supervision in Edit Flows. In Havasi et al. [14, Eq. 23], deletion supervision is formally defined for positions where the source token is valid $( a \in \mathcal V )$ and the target is blank $( b = \varepsilon )$ . However, inspection of the reference implementation (Figure 13) reveals an off-by-one indexing error. The code introduces two distinct gap sentinels (epsilon 0 id and epsilon 1 id) and enters the deletion branch whenever the current token differs from epsilon 0 id and the target equals epsilon 1 id. Crucially, the active index into $x _ { t }$ advances only for non-sentinel tokens; when a deletion is encountered, the supervised rate corresponds to the preceding surviving residue. Consequently, the implementation supervises an already deleted position at an incorrect coordinate. Our reproduction rectifies this discrepancy by strictly following the mathematical formulation of Eq. 23.

Interpolation Schedule Analysis. Havasi et al. [14] reports using a cubic interpolation schedule $\kappa _ { t } = \bar { t } ^ { 3 }$ in its experimental narrative, yet Figure 13 specifies $\kappa _ { t } = t$ with the code comment # Using a linear schedule. Analytically, both schedules deposit identical total cross-entropy mass across training examples $\begin{array} { r } { ( \int _ { 0 } ^ { 1 } \mathrm { d } \kappa _ { t } = 1 ) } \end{array}$ . However, their supervision profiles differ substantially: the linear schedule yields $\mathbb { E } [ \kappa _ { t } ] = 0 . 5 0$ , whereas the cubic schedule yields $\mathbb { E } [ \kappa _ { t } ] = 0 . 2 5$ , shifting supervision mass toward later times $( t = 0 . 7 9 )$ and increasing the expected count of active edit targets per example from 32.7 to 50.1 across 400 homolog pairs. We adopted the linear schedule matching the reference code, and confirmed that both schedules remain fully selectable.

Implicit Library Specifications in Sequence Alignment. Deutschmann et al. [6] specifies Needleman–Wunsch alignment for training pairs [15], but omits substitution matrix parameters (e.g. BLOSUM62) and gap opening/extension penalties. Because alignment parameters directly determine intermediate corruption targets, training labels are partially governed by third-party library defaults, underscoring that algorithmic specifications in discrete flow models are frequently distributed across external library dependencies.

## D Dataset Curation, Training Protocol, and Verification

Corpus Curation and Homolog Clustering. All training pairs were constructed from the Observed Antibody Space [22]. Paired heavy and light chain variable domains (VH.VL) were joined by a delimiter token present in the ESM-2 vocabulary. Homolog clusters were constructed using MMseqs2 in easy-cluster mode with a minimum sequence identity threshold of 0.5 and coverage threshold of 0.8. To prevent high-frequency clonal expansions from dominating training, clusters were capped at 20 pairs per family, yielding 1,660,105 symmetric pairs. For evaluation targets (Ty1 and trastuzumab HER2-VH), MMseqs2 searches (E-value ≤ 0.1, coverage ≥ 0.8) were augmented with a 0.7 seed identity threshold to isolate genuine biological homologs from invariant framework regions.

Algorithmic Verification: Symmetric Pair Enumeration. Deutschmann et al. [6, §4.2] states that training pairs are formed by unordered enumeration of homolog pairs. We audited this property in our data generation pipeline. In a random sample of 2,000 training pairs, 44.9% exhibited a higher property score for the second sequence (44.3% in an independent 4,000-pair audit), consistent with unbiased symmetric enumeration (∼ 50%) and contrasting with oriented pairing regimes (100%).

Training Protocol and Compute Resources. Primary model training was executed on a single NVIDIA L4 GPU for 20,000 steps with a batch size of 16, using the Adam optimizer with a learning rate of 10<sup>−4</sup> (6 h 29 m wall-clock time, 1.09 s/step). Validation loss dropped rapidly from 798.0 at initialization to 149.6 at step 8,500, 138.7 at step 16,000, and reached 135.2 at step 19,999. The model remained non-converged at the 20,000-step budget, with the final four evaluations improving by 2.3 units over preceding checkpoints.

Operational Integrity and Artifact Verification. All code, model weights, and evaluation pipelines for EditJumps are released under an MIT license at https://github.com/VisiumCH/ editjumps. To safeguard experimental reproducibility, our artifact management protocols assert strict integrity constraints:

• Training jobs implement synchronous checkpoint validation, verifying that exported model state dictionaries load successfully before reporting job completion.

• Metric logging pipelines enforce continuous credential refresh to ensure that loss trajectories and evaluation metrics remain un-truncated across multi-hour runs.

• Baseline evaluation scripts verify sequence frame coordinates dynamically, preventing mismatched reference sets between candidate models and natural controls.

## E Evaluation Methodology, Metric Audits, and Ceilings

Vector Graphic Extraction and Method Resolution. Because Deutschmann et al. [6] reported quantitative results exclusively in Figure 3 without tabulated values, we recovered numerical markers by extracting vector paths from the published PDF and calibrating against axis tick marks. The six series were resolved unambiguously:

1. Rotated x-axis tick labels maintain a uniform pitch of $9 . 8 7 \pm 0 . 0 1$ pt, aligning with marker columns at a constant 2.4 pt offset.

2. Series ordering (random pairing, EvoFlows, EvoDiff-MSA, evotuning, forced evotuning, random mutation) matches domain endpoints: random pairing achieves optimal quality across all panels, while random mutation establishes the empirical floor.

3. Figure 5 (page 21) provides the explicit legend omitted in Figure 3, matching marker colors to method labels with zero discrepancy.

Evaluation Metric Completeness Audit. Table 3 categorizes the ten metrics reported in Deutschmann et al. [6]. Only four metrics are completely defined in the text; three are named without mathematical definitions, two provide covariance/MIP matrices without scalar reduction functions, and one leaves the reduction between pooled and per-template averages ambiguous.

Table 3: Completeness of evaluation metrics in Deutschmann et al. [6] Figure 3. Systematic audit of published metric specifications. Only four of ten panels are fully reproducible from the text alone (with MMD omitting the kernel hyperparameter k). Three metrics exist only as panel titles without mathematical definitions or formulas.
<table><tr><td>Metric Panel</td><td>Specification Status</td><td>Omitted Details / Ambiguities</td></tr><tr><td>Avg Levenshtein to x0</td><td>Defined</td><td></td></tr><tr><td>Avg Pairwise Levenshtein</td><td>Ambiguous</td><td>Pooled vs. within-template average</td></tr><tr><td>Covariance Matrix</td><td>Partially defined</td><td>Matrix specified; scalar reduction omitted</td></tr><tr><td>ESM-2 PLL</td><td>Defined</td><td></td></tr><tr><td>Entropy Delta</td><td>Undefined</td><td>Named in figure only</td></tr><tr><td>Jensen-Shannon Div.</td><td>Undefined</td><td>Named in figure only</td></tr><tr><td>KL Divergence</td><td>Defined</td><td></td></tr><tr><td>MIP Matrix</td><td>Partially defined</td><td>Matrix specified; scalar reduction omitted</td></tr><tr><td>Spectrum MMD</td><td>Defined</td><td>Kernel parameter k omitted</td></tr><tr><td>Profile Log-Likelihood</td><td>Undefined</td><td>Named in figure only</td></tr></table>

Table 4: Ceiling-normalized performance of EditJumps against EvoFlows. Both models are normalized against their respective natural homolog ceilings. EditJumps is evaluated on two antibody families (200 references); EvoFlows values are recovered from Figure 3 across six families. <sup>†</sup>Jensen– Shannon divergence is evaluated under our positional formulation as the original omits its definition. <sup>‡</sup>Pairwise distance is evaluated under the pooled reduction.
<table><tr><td>Metric</td><td>EditJumps (Ours)</td><td>EvoFlows (Published)</td></tr><tr><td>Covariance (% of ceiling)</td><td>84.3</td><td>98.4</td></tr><tr><td>MIP (% of ceiling)</td><td>78.8</td><td>95.5</td></tr><tr><td>Positional  $\mathbf { J } \mathbf { S } ^ { \dagger } \left( \times \mathbf { \hat { H } } \mathbf { o } \mathbf { o } \mathbf { r } \right)$ </td><td> $2 . 6 8 \pm 0 . 1 0$ </td><td>1.80</td></tr><tr><td>Pairwise Distance</td><td> $1 . 0 7 0 \pm 0 . 0 1 1$ </td><td>1.00</td></tr></table>

Sample-Size Dependency and Grounded Baseline Controls. Agreement metrics between generated and reference distributions exhibit severe sample-size dependency. Scoring real homologs against real homologs yields covariance scores of 0.790 at $N = 2 0 , 0 . 9 2 1$ at N = 100, and saturates at 0.985 at N = 800. Consequently, raw scores from evaluations with differing reference set sizes are mathematically non-comparable. Furthermore, evaluation metrics require grounded baselines: an exact sequence recovery rate of 0.21 appears modest in isolation but represents a 300-fold enrichment over chance at matched edit budgets; conversely, an apparent agreement score of 0.79 falls entirely within the random sampling variance of finite sets.

Reference Construction and Overlap Impact. Evaluation references (200 sequences per family) were drawn from held-out family members. Because families were mined from OAS, 113 of Ty1’s and 103 of HER2-VH’s reference sequences had been present in the training set. Re-evaluating with strictly disjoint reference sets shifted spectrum MMD by −36.7% (Ty1) and −12.6% (HER2-VH) and increased covariance agreement by 12–17 points, driven by greater sequence homogeneity in the disjoint subset (23.00 vs. 24.90 Levenshtein distance).

Ceiling Normalization and Alignment Frame Sensitivity. To establish valid cross-study comparisons, metrics must be normalized against attainable ceilings computed on held-out natural homologs (Table 4). However, ceilings are highly sensitive to multiple sequence alignment projections: computing ceilings in an unaligned or mismatched frame shifts MIP agreement by +2.3 percentage points, demonstrating that ceilings must be evaluated in identical coordinate frames.

Non-Explanations for the Edit Budget Discrepancy. We systematically evaluated alternative hypotheses for the edit budget gap (0.20 reproduced vs. 0.43 published):

• Trunk capacity: Scaling ESM-2 from 35M to 650M shifts the edit ratio from 0.180 to 0.182, well within seed variance (±0.01).

![](images/8814e0beb9e22111b927d5897efc6f871844d18fa97c9d537ff574fea8a9bace.jpg)  
Figure 4: Evaluation arms within published metric envelopes. Each of the six published series in Deutschmann et al. [6] is plotted as the range spanned across its six evaluation datasets (recovered from Figure 3). Our reproduced arms and baseline anchors fall entirely within the published metric envelope.

Table 5: Claim-by-claim reproducibility assessment. Operational evaluation of published claims in Edit Flows and EvoFlows. Operational completeness denotes whether published texts provide sufficient detail to reproduce reference behavior without external code artifacts.
<table><tr><td>Published Specification / Claim</td><td>Operational Reproduction Finding</td></tr><tr><td>Edit Flows [14]</td><td></td></tr><tr><td>Operational reproducibility from text</td><td>Not answerable (no public code or weights)</td></tr><tr><td>Deletion loss supervision</td><td>Discrepancy identified: Eq. 23 vs. Fig. 13 code</td></tr><tr><td>EvoFlows [6]</td><td></td></tr><tr><td>Operational reproducibility from text</td><td>Not answerable (no public code or weights)</td></tr><tr><td>Clock normalization hyperparameter</td><td>Omitted in text; governs edit budget</td></tr><tr><td>Coevolutionary structure recovery</td><td>Partially recovered (84.3% / 78.8% of ceiling)</td></tr></table>

• Head parameterization: Adhering to the published Appendix A head parameterization is more conservative (0.19) than lightweight linear heads (0.22–0.28).

• Training duration: Modulating learning rates at 650M $( 3 \times 1 0 ^ { - 4 } ~ \mathrm { v s } . 1 0 ^ { - 5 } )$ yields ratios of 0.17 and 0.25, demonstrating that optimization choices move the budget, but clock normalization remains the primary governing mechanism.

Biophysical Analysis: IMGT Regions and Disulfide Retention. Partitioning antibody sequences via IMGT numbering shows that 76% of residues occupy the structural framework and 24% reside in complementarity-determining regions (CDRs). In CDRs, where functional sequence diversity is concentrated, EditJumps closes 18.7% (Ty1) and 23.9% (HER2-VH) of the composition KL distance between random mutation and natural homologs. In framework positions, all methods approach the natural floor (0.0001–0.0011). Crucially, EditJumps preserves canonical structural disulfides (IMGT Cys23 and Cys104) in 100% (393/393 on Ty1 and 389/389 on HER2-VH) of numberable generations, compared to 89.0% and 93.8% under random mutation baselines.