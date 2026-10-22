# IMPLICIT FEATURE REWEIGHTING FOR LABEL-FREEGRAPH GENERALIZATION

Tianjun Yao<sup>1</sup> Chen Xi<sup>2,1</sup> Xinyang Li<sup>3,1</sup> Lixian Zhong<sup>4,1</sup>

<sup>1</sup>Shenzhen Loop Area Institute

<sup>2</sup>Guangzhou University

<sup>3</sup>North China Electric Power University

<sup>4</sup>University of Science and Technology of China

## ABSTRACT

Graph-level out-of-distribution (OOD) generalization is critical for deploying graph neural networks in real-world applications where distribution shifts are inevitable. Existing approaches typically require class labels during representation learning to decide which substructures are predictive, limiting their applicability. We propose SCARCE, a label-free representation-learning framework that yields distribution-shift-robust graph representations from an environment partition of the training graphs alone, whether that partition comes from dataset metadata or, when none is available, from clustering a self-supervised embedding. Our method operates in two stages: (1) a variational graph autoencoder is warm-started for graph reconstruction; and (2) an adversarial confusion mechanism forces the encoder to produce representations from which a discriminator cannot distinguish environments, while preserving graph-reconstruction fidelity. We provide theoretical analysis showing that environment confusion contracts environmentdiscriminative directions in the representation space, which, under regularized downstream classifiers, is equivalent to imposing a stronger directional penalty on environment-associated features. This feature-reweighting mechanism reduces reliance on spurious correlations without erasing information, offering a labelfree, continuous relaxation of last-layer reweighting methods. Experiments on a controlled synthetic model, the GOOD-Motif benchmark, and molecular property prediction under assay, scaffold, and size shifts show that SCARCE achieves stateof-the-art OOD generalization among label-free methods, using no class labels during representation learning.

## 1 INTRODUCTION

Machine learning models are routinely deployed on data whose distribution differs from that of their training set, and their tendency to latch onto features that are predictive only in training makes such shifts a primary cause of failure (Geirhos et al., 2020; Gulrajani & Lopez-Paz, 2021). Graph data sharpens the problem: molecules are tested on scaffolds absent from training, and graph size or base-graph structure drifts between collections (Gui et al., 2022; Ji et al., 2023; Li et al., 2025b). A substantial body of work now targets out-of-distribution (OOD) generalization on graphs, typically by extracting an invariant subgraph or enforcing invariance across environments (Wu et al., 2022b; Chen et al., 2022; Miao et al., 2022; Yang et al., 2022a; Wu et al., 2022a; Gui et al., 2023). These methods share a prerequisite that is easy to overlook: they need class labels to decide which substructures are predictive, environment labels to decide which correlations are unstable, or both.

Class labels are the expensive part. In drug discovery a single assay label is the outcome of a wet-lab experiment, and many DrugOOD assays contain only tens of labeled molecules, whereas unlabeled molecular graphs are available by the tens of millions (Ji et al., 2023; Hu et al., 2020; Rong et al., 2020). Environment information, by contrast, is cheap but rough: a benchmark may ship a domain index, a scaffold can be computed although which scaffolds form an environment is a modeling decision, and for proteins or social networks the partition has to be inferred from the data itself (Lin et al., 2022; Yao et al., 2024). A method that learns shift-robust representations from an environment partition alone, with no class labels, and that tolerates a coarse or inferred partition, would therefore cover the common case. This setting remains largely unexplored on graphs: the few existing label-free methods add invariance-inspired augmentations to a contrastive objective (Li et al., 2022b; 2024), and none establishes what the learned representation actually retains.

![](images/2ab5bb72090909d3947e4bb57217813d5e0839c36f06f3b2205fe60a076c4cea.jpg)  
Figure 1: Overview of the SCARCE framework. Given an environment partition of the training graphs (dataset metadata or, when unavailable, clusters of a frozen self-supervised embedding), (1) a VGAE is warm-started for graph reconstruction and (2) an adversarial confusion mechanism forces the encoder to produce representations indistinguishable across environments while preserving reconstruction fidelity.

Removing class supervision raises two obstacles, and each points to one of our contributions. The first concerns what the representation contains. Without labels one can neither tell the encoder which features to keep nor check afterwards which it kept, and a contrastive objective offers no guarantee: it is minimized by whichever features most easily discriminate instances and suppresses the rest (Robinson et al., 2021), and a representation that has discarded the core features cannot be repaired downstream. Likelihood objectives behave differently: because the latent must explain the whole input, they keep core features alongside shortcuts (Yang et al., 2022b; Li et al., 2025a; Muller et al.¨ , 2024; Qiang et al., 2025). We show that this holds for graph reconstruction without any auxiliary variable: for a variational graph autoencoder (Kipf & Welling, 2016), reconstruction alone forces the latent to retain both environment-specific and core factors (Theorem 4.1), and both remain linearly accessible. The second obstacle concerns what to do with them. No class label says which factor is spurious, and erasing environment information outright discards useful signal along with the shortcut (Belrose et al., 2023; Lin et al., 2024). We show that neither selection nor erasure is needed: it suffices to shrink the environment-associated directions, because any ℓ<sub>2</sub>- regularized downstream classifier then finds them expensive to use and shifts its weight onto the core features. We realize this shrinkage with an adversarial confusion loss against the training environments, which asks nothing of the partition beyond that it separate the spurious factor, and call the framework SCARCE (Shortcut Contraction via Adversarial Representation Compression for Environments).

## Our contributions are as follows.

① Retention by reconstruction. We prove that VGAE reconstruction alone forces the latent to retain both environment-specific and core features (Theorem 4.1), and verify that both are linearly accessible.

② Contraction as a ridge penalty. We propose a confusion loss that scales down environmentdiscriminative directions rather than erasing them, prove that this contraction (Lemma 4.2) is exactly equivalent to an anisotropic ridge penalty on the downstream probe (Proposition 4.3), and characterize exactly when it helps.

③ Empirical validation. Empirically, SCARCE attains state-of-the-art label-free OOD accuracy on GOOD-Motif and is best in 17 of 18 DrugOOD settings, where no environment annotation exists and clustered pseudo-environments suffice; a controlled synthetic study confirms the two preconditions identified by the theory.

Section 2 positions SCARCE against supervised and contrastive alternatives, Section 3 and Section 4 state the setting and the two guarantees, and Section 5 tests the mechanism from a synthetic model up to DrugOOD.

## 2 RELATED WORK

OOD generalization with labeled supervision. Most OOD methods prevent a model from relying on features whose correlation with the label is unstable (Geirhos et al., 2020; Gulrajani & Lopez-Paz, 2021): invariant risk minimization (Arjovsky et al., 2019), group-robust optimization (Sagawa et al., 2020), and domain-adversarial alignment (Ganin et al., 2016) use environment or group annotations, and environment inference (Creager et al., 2021) substitutes class labels for them. In every case some form of supervision decides which correlations are unstable.

Graph OOD generalization. Graph OOD methods either extract an invariant subgraph whose relation to the label is stable (Wu et al., 2022b; Chen et al., 2022; Miao et al., 2022; Li et al., 2022a; Yang et al., 2022a; Chen et al., 2023) or enforce invariance across given or generated environments (Wu et al., 2022a; Gui et al., 2023); in both cases a class label decides which substructures are predictive. SCARCE uses no class labels during representation learning; it needs only an environment partition, which may be given or inferred, and it does not select subgraphs.

Unsupervised graph representation learning. Unsupervised graph representation learning is mature (Sun et al., 2020; You et al., 2020; Hassani & Khasahmadi, 2020; Hou et al., 2022), and unsupervised domain generalization has been formulated for images (Zhang et al., 2022), but their intersection on graphs is thin. RGCL (Li et al., 2022b) adds rationale-inspired augmentations to a contrastive objective, and OOD-GCL (Li et al., 2024), the closest work to ours, learns a channelwise disentangled contrastive encoder and penalizes each factor’s contrastive loss for varying across inferred environments. Both are contrastive at the core, which suppresses features not needed to tell instances apart (Robinson et al., 2021), and both make environment information invariant; SCARCE instead trains by reconstruction, which provably retains core and environment factors alike (Theorem 4.1), and contracts environment directions rather than erasing them (Section 4.3). When no environment annotation is available we borrow the construction of Yao et al. (2024), who cluster a frozen self-supervised embedding in place of environment labels but then train a supervised graph OOD model; we use the same pseudo-environments inside a label-free objective (Section 5.3).

Feature reweighting and concept erasure. Two lines of work act on a representation after it is learned. Last-layer retraining (Kirichenko et al., 2023) re-fits the classifier on group-balanced data, reweighting core over spurious features at the cost of group labels. Linear concept erasure (Ravfogel et al., 2022; Belrose et al., 2023) removes every linearly decodable trace of a protected attribute, discarding task-useful environment signal as well (Lin et al., 2024), and the probes that certify removal are themselves unreliable (Kumar et al., 2022). SCARCE sits between the two: it reweights rather than erases, acts in representation space rather than on the classifier, and, since the contracted direction is chosen by an adversary against environments, needs neither group-balanced labeled data nor a protected attribute.

## 3 PROBLEM SETTING

We consider graph-level unsupervised domain generalization: learning graph representations that are robust to distribution shifts, without access to class labels during representation learning.

Let the training set contain N graphs $\left\{ G _ { 1 } , G _ { 2 } , \dots , G _ { N } \right\}$ , where each graph $G _ { i } = ( \mathbf { X } _ { i } , \mathbf { A } _ { i } )$ consists of a node feature matrix $\mathbf { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d _ { \mathrm { i n } } }$ and an adjacency structure $\mathbf { A } _ { i }$ . The training graphs are drawn from a mixture of environments (domains), and we seek representations that generalize to unseen test environments. Each training graph carries an environment index $e _ { i } \in \{ 1 , \ldots , K \}$ . The partition may be supplied by the dataset (a domain index) or, absent such metadata, inferred without labels by clustering a self-supervised embedding (Section 5.3); nothing below depends on which.

Latent generative factors. We posit that each graph is generated from three latent factors through an unknown map $^ { g , }$

$$
G = g ( C , S , \xi ) ,\tag{1}
$$

where $C$ is the core (invariant) factor that determines the label $Y$ in every environment, $S$ is the environment (spurious) factor whose correlation with $Y$ varies across environments $e ,$ and ξ collects residual variation independent of $( C , S )$ . A predictor that relies on S can succeed in training environments where the $\mathsf { \bar { S } } \mathrm { - } Y$ correlation is strong yet fail on test environments where it weakens or reverses, so $C$ is the only reliable predictor of $Y$ . We write $Z = f _ { \theta } ( G )$ for the latent representation produced by a graph encoder $f _ { \theta } { } _ { ; }$ , viewed as a random variable, and h $\mathbf { \Psi } : \in \mathbb { R } ^ { d }$ for its value on a given graph $G .$ . For a reconstruction-based encoder–decoder pair, $L _ { \mathrm { r e c } } = \mathbb { E } [ - \log p _ { \psi } ( G \mid Z ) ]$ denotes the expected reconstruction loss (negative log-likelihood, in nats), i.e., the reconstruction term of the evidence lower bound.

Encoder. We encode graphs with message-passing GNNs (Xu et al., 2019; Kipf & Welling, 2017): L layers of neighborhood aggregation produce node embeddings $\mathbf { h } _ { v } ^ { ( L ) }$ , and the graph-level latent is obtained by sum pooling, $\begin{array} { r } { \mathbf { h } _ { G } = \sum _ { v \in G } \mathbf { h } _ { v } ^ { ( L ) } } \end{array}$ . GIN is our default backbone; we also report GCN.

Objective. Let $e \in { \mathcal { E } }$ index the environments. The goal is to learn a graph encoder $f _ { \theta }$ whose representations, when paired with a downstream classifier $c _ { \phi } ,$ , minimize the worst-case risk across all environments including unseen ones:

$$
\operatorname* { m i n } _ { \theta } \operatorname* { s u p } _ { e \in { \mathcal E } _ { \mathrm { a l l } } } R ^ { e } ( \theta ) , \qquad R ^ { e } ( \theta ) = \mathbb { E } _ { ( G , Y ) \sim P ^ { e } } \big [ \ell \big ( c _ { \phi } \big ( f _ { \theta } ( G ) \big ) , Y \big ) \big ] .\tag{2}
$$

We call this setting label-free: no class label Y is available during representation learning, the environment index e is the only side information, and it may itself be inferred. Class labels are used only for fitting a linear probe at evaluation time. This constraint rules out existing invariant learning methods (Arjovsky et al., 2019; Wu et al., 2022a; Chen et al., 2022), which need class labels to decide which correlations are unstable.

## 4 METHOD

We introduce SCARCE, a framework for graph-level unsupervised domain generalization. The key insight is that a standard VGAE learns representations encoding all predictive variation, including environment-specific features that may serve as spurious shortcuts. SCARCE addresses this by augmenting the VGAE objective with an adversarial confusion mechanism that selectively suppresses environment-discriminative directions in the representation, while preserving graph-reconstruction fidelity. An overview is presented in Figure 1 and Algorithm 1.

## 4.1 WHAT FEATURES DOES A VGAE RETAIN?

Our method relies on a VGAE whose latent encodes both generative factors: without $S$ there is nothing for the adversary to contract, and without $C$ contraction cannot help. Identifiability results for VAE-style models (Khemakhem et al., 2020; Zheng et al., 2022; Kong et al., 2022; Sun et al., 2021; Wang et al., 2023) and the shortcut-robustness of generative classifiers (Li et al., 2025a; Qiang et al., 2025; Muller et al.¨ , 2024) both trace factor retention to the reconstruction objective. We ask whether VGAE reconstructionforces retention of both factors, and answer affirmatively.

Theorem 4.1 (Reconstruction forces retention of both factors). Let $G = g ( C , S , \xi )$ as in Equation (1) and let $Z = f _ { \theta } ( G )$ be the encoder latent. Assume (A1) the generative map g is injective in each argument with the others heldfixed, and (A2) G is discrete with $H ( S \mid C , \xi )$ and $H ( C \mid S , \xi )$ finite and positive. Then any encoder–decoder pair attaining reconstruction loss $L _ { \mathrm { r e c } }$ satisfies simultaneously

$$
I ( Z ; S \mid C , \xi ) \ge H ( S \mid C , \xi ) - L _ { \mathrm { r e c } } , \qquad I ( Z ; C \mid S , \xi ) \ge H ( C \mid S , \xi ) - L _ { \mathrm { r e c } } ,\tag{3}
$$

both bounds being strictly positive whenever $L _ { \mathrm { r e c } } < \operatorname* { m i n } \{ H ( S \mid C , \xi ) , H ( C \mid S , \xi ) \}$

The key distinction from discriminative training is that the reconstruction objective imposes perfactor constraints whose strength scales with each factor’s generative footprint on the observed graph, whereas a classification loss yields only a single aggregate constraint bounded by $H ( Y )$

![](images/97740d31878018c48f7ce622dfbc91b98396ece161cb3b7f091e80a52a463833.jpg)  
Figure 2: DFR on frozen GIN representations across training, GOOD-Motif concept shift (OOD accuracy, mean±std over 3 seeds; y-axis starts at 30%). ERM improves slowly, yet DFR on the same encoder is already above 87% at epoch 10; DFR on the unsupervised VGAE starts at 90.7% and degrades mildly with further training.

## Algorithm 1: SCARCE Training Pipeline

Require: graphs $\{ G _ { i } \} _ { i = 1 } ^ { N }$ , environment indices $\left\{ e _ { i } \right\}$ , weights   
{w<sub>i</sub>}   
1: Pre-train VGAE on $\mathcal { L } _ { \mathrm { V G A E } } ;$ save warm-start checkpoint   
2: Initialize VGAE from warm-start checkpoint; initialize ad  
versary ρ   
3: for $t = 1 , \dots , T$ do   
4: for each mini-batch B do   
5: Compute $\{ \mathbf { h } _ { G } ^ { ( i ) } \} _ { i \in B }$ with the VGAE encoder   
6: Adversary step: freeze encoder; $k _ { \mathrm { a d v } }$ <sub>v</sub> updates of $\rho$   
on $\mathrm { s g } ( \mathbf { h } _ { G } )$ with Equation (4)   
7: Encoder step: freeze ρ; update encoder with Equa  
tion (6)   
8: end for   
9: end for   
10: Evaluation: freeze encoder; fit a linear probe on training   
labels

which is satisfiable by an encoder that discards C entirely when S alone predicts the training labels. Importantly, the VGAE does not itself learn a “shortcut”: it retains S as a generative factor, and the shortcut arises only downstream when a predictor exploits the spurious S–Y correlation. Theorem 4.1 establishes information-theoretic retention; it does not guarantee linear accessibility or preservation after graph pooling. The full proof is deferred to Appendix A.1.

Empirical verification via feature reweighting. Theorem 4.1 is information-theoretic; we test whether both factors are also linearly accessible with Deep Feature Reweighting (DFR; Kirichenko et al. 2023), which freezes the encoder and refits a sparse linear head on a group-balanced subset, so its OOD accuracy measures the invariant signal the latent exposes once the spurious S–Y correlation is removed. We apply DFR to GIN checkpoints of a label-trained ERM encoder and of an unsupervised VGAE on GOOD-Motif concept shift (3 seeds; protocol in Appendix B). Figure 2 shows two things. First, ERM climbs slowly from 35.4% to 73.0%, yet DFR on the same frozen encoder is already at 87.5% at epoch 10: $C$ is linearly accessible early, and the ERM head merely underweights it relative to S. Second, DFR on the VGAE, which never sees a label, reaches 90.7% at epoch 10, on par with the label-trained encoder: reconstruction exposes C in linear form from the outset, and, since the same latent is what a naive probe exploits via $S ,$ it exposes S as well. With longer VGAE training, DFR accuracy drifts down to 85.4% with growing variance, suggesting that prolonged reconstruction slowly reallocates latent capacity away from the directions a linear head needs. Since DFR probes the pooled h rather than the latent the decoder sees, retention evidently survives pooling as well.

## 4.2 ADVERSARIAL ENVIRONMENT CONFUSION

Since the VGAE latent retains environment-specific features alongside invariant ones (Section 4.1), SCARCE adds an encoder–adversary alternating optimization that discourages reliance on the environment-discriminative directions.

Adversary. The adversarial classifier $\rho : \mathbb { R } ^ { d }  \mathbb { R } ^ { K }$ is a two-layer MLP that maps graph representations to environment logits. In each mini-batch, the encoder is frozen and the adversary is updated for $k _ { \mathrm { a d v } }$ steps on the detached representations $\mathbf { h } _ { \mathrm { a d v } } = \mathrm { s g } ( \mathbf { h } _ { G } )$ , minimizing the rebalanced cross-entropy:

$$
\mathcal { L } _ { \mathrm { a d v } } = \frac { \sum _ { i } w _ { i } \cdot \mathrm { C E } ( \rho ( \mathbf { h } _ { \mathrm { a d v } } ^ { ( i ) } ) , \ e _ { i } ) } { \sum _ { i } w _ { i } } ,\tag{4}
$$

where $e _ { i }$ are the environment indices of Section 3 and $w _ { i } \ \propto \ n _ { e _ { i } } ^ { - \alpha }$ are rebalancing weights that compensate for uneven environment sizes, with $n _ { e }$ the number of training graphs in environment

e and $\alpha \geq 0$ controlling the strength of rebalancing $( \alpha = 0$ gives uniform weights). Rebalancing matters most when the partition is inferred, since clusters are rarely of equal size.

Encoder update via uniform confusion. The adversary is then frozen, and the encoder is updated. The confusion loss drives the adversary’s output toward the uniform distribution ${ \bf u } \ =$ $( 1 / K , \ldots , 1 / K )$

$$
\mathcal { L } _ { \mathrm { c o n f } } = \frac { \sum _ { i } w _ { i } \cdot \left( - \sum _ { k = 1 } ^ { K } \frac { 1 } { K } \log \frac { \exp ( \rho _ { k } ( \mathbf { h } _ { G } ^ { ( i ) } ) ) } { \sum _ { j } \exp ( \rho _ { j } ( \mathbf { h } _ { G } ^ { ( i ) } ) ) } \right) } { \sum _ { i } w _ { i } } .\tag{5}
$$

For each sample, this equals $\begin{array} { r } { H ( U _ { K } , p _ { \rho } ) = \log K + \mathrm { K L } ( U _ { K } | | p _ { \rho } ) } \end{array}$ , minimized when the adversary outputs $1 / K$ for every environment. The alternating game works as follows: the adversary identifies representation directions that separate environments; the encoder receives gradients that weaken those distinctions; and the reconstruction objective constrains the encoder from discarding all variation.

Encoder total loss. The encoder loss combines reconstruction and confusion:

$$
\mathcal { L } _ { \mathrm { e n c } } = \mathcal { L } _ { \mathrm { V G A E } } + \beta \cdot \mathcal { L } _ { \mathrm { c o n f } } ,\tag{6}
$$

where $\beta$ controls the strength of the confusion penalty: ${ \mathcal { L } } _ { \mathrm { V G A E } }$ keeps the latent informative (Section 4.1) while ${ \mathcal { L } } _ { \mathrm { c o n f } }$ contracts the directions the adversary identifies.

## 4.3 THEORETICAL ANALYSIS: WHY CONFUSION HELPS OOD GENERALIZATION

Erasing environment information discards useful signal together with the shortcut (Belrose et al., 2023; Lin et al., 2024). We show instead that confusion training contracts environmentdiscriminative directions (Lemma 4.2), and that under a regularized downstream classifier this geometric change is exactly a directional penalty that makes spurious features expensive to use (Proposition 4.3). Proofs are in Appendix $\mathrm { A } .$

Lemma 4.2 (Local contraction). Fix a linear K-class softmax adversary with logits $z ( h ) = W h + b ,$ probabilities $p ( h ) =$ softmax $( z ( h ) )$ , and uniform target $q = K ^ { - 1 } \mathbf { 1 }$ . Define the centered adversary weights $W _ { c } \stackrel { \_ } { = } C W$ where $C \dot { = } \dot { I } - K ^ { - 1 } \mathbf { 1 } \dot { \mathbf { 1 } } ^ { \intercal }$ . Then a gradient step $\dot { h } ^ { + } = h - \alpha \beta \nabla _ { h } L _ { \mathrm { c o n f } } ( \dot { h } )$ contracts each adversary-discriminative direction $v _ { j }$ (right singular vector of $W _ { c }$ with singular value $s _ { j } )$ by a factor $1 - \alpha \beta s _ { i } ^ { 2 } / K$ to first order, while directions orthogonal to range $( W _ { c } ^ { \top } )$ are unchanged. In particular, when one singular value dominates $( s _ { 1 } \gg s _ { 2 } )$ , the cumulative effect of confusion training is well approximated by a rank-one contraction along $u _ { E } = v _ { 1 } .$

$$
\mathbf { h } _ { G } ^ { ( \beta ) } \approx T _ { \varepsilon } \mathbf { h } _ { G } ^ { ( 0 ) } , \qquad T _ { \varepsilon } = I - \left( 1 - \varepsilon \right) u _ { E } u _ { E } ^ { \top } , \quad 0 < \varepsilon \leq 1 ,\tag{7}
$$

where $\varepsilon = 1$ corresponds to no contraction and smaller $\varepsilon$ means the environment direction $u _ { E }$ receives less relative scale.

Uniform confusion is thus a targeted contraction: only the directions the adversary uses to separate environments are compressed, more strongly the larger their singular value $s _ { j }$ , while directions the adversary ignores, which may carry stable task-relevant information, are left intact. This selectivity is what distinguishes confusion from naive information removal: it narrows the environmentdiscriminative subspace without collateral damage to invariant features. The rank-one model equation $7$ is a first-order consequence of a dominant singular value, and it ties ε to the training hyperparameters: after $T$ encoder steps, $\begin{array} { r } { \varepsilon \approx \prod _ { t = 1 } ^ { T } ( 1 - \alpha \beta s _ { 1 } ^ { 2 } / K ) } \end{array}$ , so a larger confusion weight $\beta$ or longer training yields a stronger contraction. A neural encoder may additionally rotate or nonlinearly deform the representation, so we verify the model empirically rather than assume it globally.

Proposition 4.3 (Ridge identity). For any loss $\ell ,$ any $\lambda > 0$ , and any $\varepsilon \in ( 0 , 1 ] ,$ , the ℓ -regularized linear probe on the contracted representation is equivalent to an anisotropic ridge penalty on the original representation:

$$
\operatorname* { m i n } _ { w , b } \ \mathbb { E } \left[ \ell ( Y , w ^ { \top } T _ { \varepsilon } h + b ) \right] + \lambda \| w \| _ { 2 } ^ { 2 } = \operatorname* { m i n } _ { v , b } \ \mathbb { E } \left[ \ell ( Y , v ^ { \top } h + b ) \right] + \lambda \big [ \| v \| _ { 2 } ^ { 2 } + ( \varepsilon ^ { - 2 } - 1 ) ( v ^ { \top } u _ { E } ) ^ { 2 } \big ] ,\tag{8}
$$

where optima correspond under $v = T _ { \varepsilon } w$ . The effective penalty on span $( u _ { E } )$ is $\lambda / \varepsilon ^ { 2 }$ , while orthogonal directions retain penalty λ.

![](images/1842ed84e3fbc4b62911f4ba25d165f8c4286b31673864be162c01cb482b524c.jpg)

![](images/14055c724cd522ab834ec981b36d035e95a3b719382a249cfa2814de129982c8.jpg)  
Figure 3: Controlled contraction of an environment-associated direction. Cells report $\Delta R _ { \mathrm { O O D } }$ (10 seeds; negative = improvement). (a) Contraction helps when environment and task directions are separated but hurts at full alignment. (b) The effect vanishes as $\lambda  0$ and peaks at intermediate $\lambda ,$ , consistent with the induced penalty $\lambda / \varepsilon ^ { 2 }$

Contracting $u _ { E }$ in the representation is exactly equivalent to increasing the downstream ridge penalty in that direction from λ to $\lambda / \varepsilon ^ { 2 }$ , while leaving other directions unaffected.

The added term $( \varepsilon ^ { - 2 } - 1 ) ( v ^ { \top } u _ { E } ) ^ { 2 }$ charges only the component of v along $u _ { E }$ and leaves any $\ v \ \perp \ u _ { E }$ untouched. This is feature reweighting, not feature erasure: the probe can still use the environment-associated direction, but doing so becomes progressively more expensive as ε decreases, and the information along $u _ { E }$ is never removed. Proposition 4.3 thus performs the reweighting of DFR (Kirichenko et al., 2023) in representation space, as an anisotropic ridge penalty, and therefore needs neither class labels nor group annotations. Regularization is essential: $T _ { \varepsilon }$ is invertible for $\varepsilon > 0$ , so an unregularized classifier simply rescales its weight along $u _ { E }$ by $1 / \varepsilon$ and contraction has no downstream effect.

When does contraction help? A two-dimensional Gaussian model (Proposition $\mathrm { A . 1 }$ $\mathrm { A p \cdot }$ pendix A.4) makes the answer exact: when the environment feature is spuriously correlated with $Y$ in training but not at test time, the OOD risk is strictly decreasing in the contraction amount; as $\lambda  0 ^ { + }$ the probe undoes the contraction and the effect vanishes; and when $u _ { E } = u _ { Y }$ contraction hurts because the compressed direction carries the stable signal. The mechanism therefore requires exactly two things: ① a spurious training correlation along a direction separated from the core signal, and ② a finitely regularized probe.

## 5 EXPERIMENTS

## 5.1 SYNTHETIC EVALUATION

We isolate the mechanism of Section 4.3 by directly contracting an environment-associated direction in a synthetic representation and checking downstream behavior against the theory.

Setup. We sample $z _ { c } , z _ { e } \sim \mathcal { N } ( 0 , 1 ) , \eta \sim \mathcal { N } ( 0 , \sigma _ { \eta } ^ { 2 } )$ , and set $Y \ = \ z _ { c } .$ . The representation is ${ \bf h } = ( z _ { c } + \eta ) { \bf u } _ { Y } + \sigma _ { s } ( r z _ { c } + \sqrt { 1 - r ^ { 2 } } z _ { e } ) { \bf u } _ { E } $ , where $\mathbf { u } _ { E }$ and u<sub>Y</sub> have squared overlap $\cos ^ { 2 } \theta$ Training uses $r _ { \mathrm { t r } } = 0 . 9 ;$ testing uses $r _ { \mathrm { t e } } = 0$ . We apply $T _ { \varepsilon } = I - ( 1 - \varepsilon ) \mathbf { u } _ { E } \mathbf { u } _ { E } ^ { \top }$ directly and fit a ridge probe on 10k training examples, evaluating OOD MSE on 50k test examples. We report $\Delta R _ { \mathrm { O O D } } ^ { \mathrm { ~ - ~ } } = R _ { \mathrm { O O D } } ( \varepsilon ) - R _ { \mathrm { O O D } } ( 1 )$

Finding 1: Geometry governs the cost of contraction. We fix $\lambda = 0 . 1$ and sweep $\cos ^ { 2 } \theta$ . When the two directions are orthogonal, contraction reduces OOD error; the benefit shrinks with increasing overlap; and at full alignment contraction increases error because it suppresses stable signal alongside the shortcut (Figure 3a), matching the trade-off in Proposition A.1(iii).

Finding 2: Finite regularization is necessary. At near-zero regularization $( \lambda = 1 0 ^ { - 6 } ) .$ , contraction has negligible effect: the probe rescales its coefficient to undo the contraction, as Proposition A.1(ii) predicts. The benefit peaks at intermediate λ and diminishes under excessive regularization, which suppresses all directions uniformly (Figure 3b). Contraction acts through the regularization geometry, not by erasing information.

Table 1: GOOD-Motif concept shift, OOD test accuracy (%). Bold: best; underline: second best.
<table><tr><td>Method</td><td>Acc.</td></tr><tr><td>ERM (sup.) GraphCL</td><td> $5 3 . 0 \pm 8 . 5$   $6 0 . 9 \pm 3 . 4$ </td></tr><tr><td>MVGRL OOD-GCL</td><td> $5 4 . 5 \pm 4 . 7$   $6 4 . 1 \pm 3 . 6 $ </td></tr><tr><td>RGCL</td><td> ${ \bf 7 0 . 1 \pm 6 . 5 }$ </td></tr><tr><td>VGAE (β=0)</td><td> $6 5 . 9 \pm 2 . 0$ </td></tr><tr><td>SCARCE</td><td> $6 9 . 6 \pm 2 . 2$ </td></tr></table>

![](images/5fbe6839b6c9568c6564c9e065063c99bb8711132b8fb37533247d7d96fa7549.jpg)  
Figure 4: Probe regularization sweep on GOOD-Motif. Top: OOD accuracy for $\beta { = } 0 \ \mathrm { v s . } \ \beta { = } 0 . 1$ Bottom: gain $\Delta$ from confusion.

## 5.2 CONTROLLED GRAPH BENCHMARK

The synthetic study applies $T _ { \varepsilon }$ to a fixed representation; we now test whether the mechanism survives end-to-end training on realistic graphs.

Setup. We use GOOD-Motif concept shift (Gui et al., 2022), where the base-graph distribution shifts across domains while the motif–label relationship is stable. The benchmark provides a domain index for every training graph, which we use directly as the environment e (K equals the number of training domains); class labels are never seen by the encoder. The encoder is a 3-layer GCN with a 64-dimensional latent space, trained with the $\mathrm { V G A E }$ reconstruction loss and fine-tuned with adversarial confusion. We freeze the encoder and fit an $\ell _ { 2 }$ -regularized linear classifier on the training labels, sweeping the inverse regularization strength over $\mathsf { \bar { C } } \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , \dots , 1 0 ^ { 3 } \}$ . To isolate the effect of representation geometry from trivial scale differences, we fit a per-dimension standardizer on the $\beta { = } 0$ training representations and apply it, frozen, to all encoder variants before probing. We report OOD test accuracy averaged over 3 seeds.

Results. Table 1 reports OOD test accuracy. SCARCE outperforms GraphCL, MVGRL, and OOD-GCL, and improves over the plain VGAE warm-start. RGCL performs on par with SCARCE on this setting.

Ablation: probe regularization. Figure 4 sweeps $C$ over six orders of magnitude. Accuracy for both $\beta { = } 0$ and $\beta { = } 0 . 1$ is low when the probe is too constrained to use any feature, peaks near $C { = } 1 0 .$ and drops once weak regularization lets the probe exploit shortcut features. The gain $\Delta$ from confusion is non-monotonic: it is largest $( + 3 . 8 \% )$ at $C { = } 1 0 ^ { - 3 }$ , where a large base penalty λ is inflated to $\lambda / \varepsilon ^ { 2 }$ on the environment direction (Proposition 4.3); it vanishes at $C { = } 1$ , where the probe has enough capacity to partially undo the contraction (Proposition A.1(ii)); and it returns at $C \overset { \cdot } { \geq } 1 0 ^ { 2 }$ $( + 1 . 4 , + \bar { 2 } . 4 \% )$ , where the baseline probe freely exploits shortcuts that the confusion-trained encoder has already contracted.

## 5.3 REAL-WORLD MOLECULAR EVALUATION

Setup. We use EC50 and IC50 from DrugOOD (Ji et al., 2023) under assay, scaffold, and size splits. Each encoder is frozen and a logistic-regression probe is fitted on standardized representations using 5%, 10%, or 20% of the labeled training split; $( \alpha , \beta , K )$ and checkpoints are selected on validation and the test split is used once. Baselines are GraphCL, RGCL, OOD-GCL, MVGRL, and VGAE (β=0), i.e. SCARCE without confusion. Details are in Appendix C.

Environment partition. Unlike GOOD-Motif, DrugOOD offers no usable environment index inside the training split: the assay, scaffold, and size attributes define the train/validation/test split, but

Table 2: DrugOOD under limited labels. Test ROC-AUC (%), mean ± std over three seeds (IC50- size VGAE at 20%: two), for linear probes on frozen representations. Bold: best per column; Avg.: mean over the six splits; shaded row: SCARCE.
<table><tr><td>Method</td><td colspan="3">EC50</td><td colspan="4">IC50</td></tr><tr><td></td><td>Assay</td><td>Scaffold</td><td>Size</td><td>Assay</td><td>Scaffold</td><td>Size</td><td> $\operatorname { A v g } .$ </td></tr><tr><td> $5 \% { l a b e l s }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { M V G R L }$ </td><td> $6 8 . 9 { \pm } 3 . 7 $ </td><td> $5 2 . 1 { \pm } 3 . 7 $ </td><td> $5 7 . 8 { \pm } 2 . 9 $ </td><td> $6 7 . 1 { \pm } 0 . 3 $ </td><td> $5 7 . 4 { \pm } 1 . 7 $ </td><td> ${ \bf 6 2 . 1 { \pm 0 . 8 } }$ </td><td>60.9</td></tr><tr><td> ${ \mathsf { V G A E } } \left( \beta { = } 0 \right)$ </td><td> $5 8 . 2 { \pm } 2 . 8 $ </td><td> $5 0 . 0 { \pm } 6 . 2 $ </td><td> $5 2 . 9 { \pm } 4 . 1$ </td><td> $6 1 . 5 { \pm } 2 . 2 $ </td><td> $5 5 . 7 { \pm 2 . 2 }$ </td><td> $5 6 . 8 { \pm } 0 . 7 $ </td><td>55.9</td></tr><tr><td> $\mathrm { G r a p h C L }$ </td><td> $6 5 . 1 { \pm } 2 . 0 $ </td><td> $5 4 . 6 { \pm } 9 . 0 $ </td><td> $5 6 . 3 { \pm } 1 . 8 $ </td><td> $6 7 . 0 { \pm } 1 . 0 $ </td><td> $6 1 . 8 { \pm } 1 . 9 $ </td><td> $6 1 . 5 { \pm } 1 . 1 $ </td><td>61.1</td></tr><tr><td> $\operatorname { R G C L }$  OOD-GCL</td><td> $6 9 . 5 { \pm } 3 . 1 $ </td><td> $5 4 . 8 { \pm } 7 . 5 $ </td><td> $5 6 . 5 { \pm } 2 . 9 $ </td><td> $6 5 . 8 { \pm } 0 . 1 $ </td><td> $6 0 . 0 { \pm } 1 . 6 $ </td><td> $6 0 . 5 { \pm } 1 . 1 $ </td><td>61.2</td></tr><tr><td>SCARCE</td><td> $5 9 . 5 { \pm 6 . 5 }$ </td><td> $5 7 . 3 { \pm } 6 . 2 $ </td><td> $5 3 . 4 { \pm } 1 . 1$ </td><td> $5 8 . 1 { \pm } 2 . 2 $ </td><td> $5 7 . 3 { \pm } 4 . 0 $ </td><td> $5 7 . 1 { \pm } 2 . 9 $ </td><td>57.1 65.3</td></tr><tr><td></td><td> ${ \bf 7 0 . 6 { \pm 1 . 0 } }$ </td><td> ${ \bf 6 5 . 0 { \pm 1 . 0 } }$ </td><td> ${ \bf 6 1 . 0 { \pm } 1 . 8 }$ </td><td> ${ \bf 6 9 . 5 \pm 0 . 5 }$ </td><td> ${ \bf 6 4 . 3 \pm 1 . 2 }$ </td><td> $6 1 . 6 { \pm } 1 . 0 $ </td><td></td></tr><tr><td> $1 0 \% l a b e l s$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { M V G R L }$ </td><td> $6 7 . 5 { \pm } 2 . 5 $ </td><td> $5 4 . 4 { \pm } 3 . 5 $ </td><td> $5 6 . 0 { \pm } 1 . 8 $ </td><td> $6 7 . 8 { \pm } 0 . 2 $ </td><td> $5 9 . 8 { \pm } 1 . 0 $ </td><td> $6 1 . 7 { \pm } 1 . 1 $ </td><td>61.2</td></tr><tr><td> ${ \mathsf { V G A E } } \left( \beta { = } 0 \right)$ </td><td> $5 9 . 6 { \pm } 2 . 0 $ </td><td> $4 9 . 9 { \pm } 1 . 9 $ </td><td> $5 3 . 2 { \pm } 1 . 9 $ </td><td> $6 2 . 1 { \pm } 1 . 0 $ </td><td> $5 8 . 4 { \pm } 1 . 6 $ </td><td> $5 8 . 6 { \pm } 0 . 3 $ </td><td>57.0</td></tr><tr><td> $\operatorname { G r a p h C L }$ </td><td> $6 6 . 9 { \pm } 1 . 7 $ </td><td> $5 8 . 8 { \pm } 2 . 5 $ </td><td> $5 8 . 4 { \pm } 0 . 7 $ </td><td> $6 7 . 3 { \pm } 0 . 7 $ </td><td> $6 2 . 8 { \pm } 0 . 8 $ </td><td> $6 1 . 5 { \pm } 0 . 4 $ </td><td>62.6</td></tr><tr><td> $\mathsf { R G C L }$ </td><td> $6 9 . 8 { \pm } 1 . 5 $ </td><td> $5 8 . 8 { \pm } 2 . 3 $ </td><td> $5 7 . 7 { \pm } 2 . 6 $ </td><td> $6 6 . 9 { \pm } 1 . 3 $ </td><td> $6 0 . 8 { \pm } 1 . 7 $ </td><td> $6 1 . 3 { \pm } 0 . 6 $ </td><td>62.6</td></tr><tr><td> $_ { \mathrm { O O D - G C L } }$ </td><td> $6 3 . 9 2 4 . 7$ </td><td> $5 7 . 2 { \pm } 3 . 1 $ </td><td> $5 3 . 2 { \pm } 1 . 0 $ </td><td> $5 9 . 0 { \pm } 1 . 6 $ </td><td> $5 8 . 8 { \pm } 4 . 0 $ </td><td> $5 8 . 3 { \pm } 2 . 9 $ </td><td>58.4</td></tr><tr><td> $\operatorname { S C A R C E }$ </td><td> ${ \bf 7 1 . 8 \pm 1 . 7 }$ </td><td> ${ \bf 6 4 . 9 2 1 . 2 }$ </td><td> ${ \bf 6 2 . 9 2 1 . 5 }$ </td><td> ${ \bf 6 9 . 5 \pm 0 . 1 }$ </td><td> ${ \bf 6 6 . 0 { \pm 0 . 7 } }$ </td><td> ${ \bf 6 2 . 6 { \pm } 0 . 9 }$ </td><td>66.3</td></tr><tr><td> $2 0 \% l a b e l s$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { M V G R L }$ </td><td> $6 7 . 8 { \pm } 1 . 6 $ </td><td> $5 6 . 3 { \pm } 1 . 2 $ </td><td> $5 7 . 1 { \pm } 2 . 4 $ </td><td> $6 7 . 8 { \pm } 0 . 5 $ </td><td> $6 1 . 4 { \pm } 0 . 8 $ </td><td> $6 1 . 6 { \pm } 0 . 3 $ </td><td>62.0</td></tr><tr><td> ${ \mathsf { V G A E } } \left( \beta { = } 0 \right)$ </td><td> $6 2 . 5 { \pm } 4 . 6 $ </td><td> $5 3 . 1 { \pm } 0 . 6 $ </td><td> $5 5 . 6 { \pm } 1 . 4 $ </td><td> $6 3 . 9 { \pm } 0 . 4 $ </td><td> $6 0 . 4 { \pm } 0 . 7$ </td><td> $6 0 . 2 { \pm } 0 . 0 $ </td><td>59.3</td></tr><tr><td> $\mathrm { G r a p h C L }$ </td><td> $7 0 . 7 { \pm } 3 . 7 $ </td><td> $5 9 . 1 { \pm } 3 . 7 $ </td><td> $6 0 . 2 { \pm } 0 . 4 $ </td><td> $6 7 . 5 { \pm } 0 . 3 $ </td><td> $6 3 . 5 { \pm } 0 . 7 $ </td><td> $6 2 . 3 { \pm } 0 . 7 $ </td><td>63.9</td></tr><tr><td> $\mathsf { R G C L }$ </td><td> $7 3 . 0 { \pm } 6 . 2 $ </td><td> $5 5 . 0 { \pm } 4 . 3 $ </td><td> $6 1 . 0 { \pm } 0 . 6 $ </td><td> $6 7 . 2 { \pm } 0 . 4 $ </td><td> $6 1 . 3 { \pm } 1 . 1 $ </td><td> $6 2 . 3 { \pm } 1 . 0 $ </td><td>63.3</td></tr><tr><td> $_ { \mathrm { O O D - G C L } }$ </td><td> $6 4 . 3 { \pm } 4 . 0 $ </td><td> $5 8 . 7 { \pm } 4 . 3 $ </td><td> $5 3 . 9 { \pm } 1 . 0 $ </td><td> $6 0 . 1 { \pm } 1 . 9 $ </td><td> $5 8 . 7 { \pm } 3 . 6 $ </td><td> $5 9 . 0 { \pm } 3 . 1 $ </td><td>59.1</td></tr><tr><td> $\operatorname { S C A R C E }$ </td><td> ${ \bf 7 3 . 6 { \pm 1 . 3 } }$ </td><td> ${ \bf 6 4 . 4 \pm 1 . 7 }$ </td><td> ${ \bf 6 2 . 8 { \pm } 0 . 5 }$ </td><td> ${ \bf 6 9 . 7 \pm 0 . 1 }$ </td><td> ${ \bf 6 5 . 9 2 0 . 4 }$ </td><td> ${ \bf 6 3 . 2 \pm 0 . 5 }$ </td><td>66.6</td></tr></table>

within training they are respectively thousands of sparsely populated assay IDs, near-unique scaffolds, and a continuous graph size, none of which yields a partition of the kind Section 3 assumes. We therefore construct pseudo-environments without labels, following Yao et al. (2024): a frozen MVGRL encoder embeds the training graphs, k-means with K clusters is run on the embeddings, and the cluster index serves as $e _ { i } .$ Cluster imbalance is absorbed by the rebalancing exponent α of Section 4.2. No annotation of any kind is used during representation learning, and $\check { K }$ is treated as a hyperparameter selected on validation.

Results. Table 2 reports test ROC-AUC. SCARCE is best in 17 of the 18 dataset–ratio cells, the sole exception being IC50-size at 5% labels, and the margin is largest on the scaffold splits, where test molecules differ structurally from training ones. The VGAE $( \beta \mathrm { = } 0 )$ row is the ablation without confusion: it is the weakest representation on most splits, trailing SCARCE by 11–15 points on EC50-scaffold and by 3–10 points elsewhere, so the gain is attributable to confusion training rather than to the reconstruction backbone. MVGRL, whose embedding only supplies the pseudo-environment partition, is likewise well below SCARCE, so a rough, label-free partition is enough for confusion to act on. Averaged over the six splits, SCARCE leads the strongest baseline by 4.1, 3.7, and 2.7 points at 5%, 10%, and 20% labels, so the gain holds at every label budget.

Sensitivity. Across the top-ranked configurations (Appendix C.1), the strongest confusion weight $\beta = 1 0$ is preferred in nearly every setting, whereas the best K varies across splits and is selected on validation. The preference for large $\beta$ matches Proposition 4.3: a probe fitted on 5–20% of the labels sits in the strongly regularized regime where Figure 3(b) shows contraction to help most. Component-loss curves (Appendix C.2) confirm that the alternating optimization is stable.

## 6 CONCLUSION

We introduced SCARCE, a label-free framework for graph OOD generalization that pairs a VGAE with adversarial confusion against the training environments, given or, as on DrugOOD, inferred by clustering. Both components come with a guarantee: reconstruction forces the latent to retain core and environment factors alike, and confusion contracts, rather than erases, the environmentdiscriminative directions, which for any $\ell _ { 2 } \cdot$ -regularized probe is exactly an anisotropic ridge penalty. Experiments on a controlled synthetic model, GOOD-Motif, and DrugOOD confirm both the mechanism and its preconditions. The recipe, i.e., retain by reconstruction, then reweight by contraction, is not specific to graphs and extends naturally to other likelihood-based encoders.

## ETHICS STATEMENT

This work studies unsupervised representation learning for graph out-of-distribution generalization. All experiments use publicly available benchmarks that contain no personal data or human subjects. By removing the need for class labels during representation learning and tolerating coarse or inferred environment partitions, SCARCE lowers the cost of building predictors that remain reliable under distribution shift, which is beneficial in settings such as molecular property prediction where labeled and annotated data are scarce. The method does not create new data-collection or surveillance capabilities, and we do not foresee any direct negative societal impact beyond those generic to representation learning.

## REFERENCES

Martin Arjovsky, Leon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization.´ arXiv preprint arXiv:1907.02893, 2019.

Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman. LEACE: Perfect linear concept erasure in closed form. In NeurIPS, 2023.

Yongqiang Chen, Yonggang Zhang, Yatao Bian, Han Yang, Kaili Ma, Binghui Xie, Tongliang Liu, Bo Han, and James Cheng. Learning causally invariant representations for out-of-distribution generalization on graphs. In NeurIPS, 2022.

Yongqiang Chen, Yatao Bian, Kaiwen Zhou, Binghui Xie, Bo Han, and James Cheng. Does invariant graph learning via environment augmentation learn invariance? In NeurIPS, 2023. URL https: //openreview.net/forum?id=EqpR9Vtt13.

Elliot Creager, Jorn-Henrik Jacobsen, and Richard Zemel. Environment inference for invariant¨ learning. In ICML, 2021.

Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, Franc¸ois Laviolette, Mario March, and Victor Lempitsky. Domain-adversarial training of neural networks. Journal ofMachine Learning Research, 17(59):1–35, 2016.

Robert Geirhos, Jorn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel,¨ Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

Shurui Gui, Xiner Li, Limei Wang, and Shuiwang Ji. GOOD: A graph out-of-distribution benchmark. In NeurIPS Datasets and Benchmarks Track, 2022.

Shurui Gui, Meng Liu, Xiner Li, Youzhi Luo, and Shuiwang Ji. Joint learning of label and environment causal independence for graph out-of-distribution generalization. In NeurIPS, 2023.

Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In ICLR, 2021.

Kaveh Hassani and Amir Hosein Khasahmadi. Contrastive multi-view representation learning on graphs. In ICML, 2020.

Zhenyu Hou, Xiao Liu, Yukuo Cen, Yuxiao Dong, Hongxia Yang, Chunjie Wang, and Jie Tang. GraphMAE: Self-supervised masked graph autoencoders. In KDD, 2022.

Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. Strategies for pre-training graph neural networks. In ICLR, 2020.

Yuanfeng Ji, Lu Zhang, Jiaxiang Wu, Bingzhe Wu, Long-Kai Huang, Tingyang Xu, Yu Rong, Lanqing Li, Jie Ren, Ding Xue, et al. DrugOOD: Out-of-distribution dataset curator and benchmark for AI-aided drug discovery. In AAAI, 2023.

Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational autoencoders and nonlinear ICA: A unifying framework. In AISTATS, pp. 2207–2217. PMLR, 2020.

Thomas N Kipf and Max Welling. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308, 2016.

Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In ICLR, 2017.

Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson. Last layer re-training is sufficient for robustness to spurious correlations. In ICML, 2023.

Lingjing Kong, Shaoan Xie, Weiran Yao, Yujia Zheng, Guangyi Chen, Petar Stojanov, Victor Akinwande, and Kun Zhang. Partial disentanglement for domain adaptation. In ICML, pp. 11455– 11472, 2022.

Abhinav Kumar, Chenhao Tan, and Amit Sharma. Probing classifiers are unreliable for concept removal and detection. In NeurIPS, 2022.

Alexander C. Li, Ananya Kumar, and Deepak Pathak. Generative classifiers avoid shortcut solutions. In ICLR, 2025a.

Haoyang Li, Ziwei Zhang, Xin Wang, and Wenwu Zhu. Learning invariant graph representations for out-of-distribution generalization. In NeurIPS, 2022a.

Haoyang Li, Xin Wang, Zeyang Zhang, Haibo Chen, Ziwei Zhang, and Wenwu Zhu. Disentangled graph self-supervised learning for out-of-distribution generalization. In ICML, pp. 28890–28904, 2024.

Haoyang Li, Xin Wang, Ziwei Zhang, and Wenwu Zhu. Out-of-distribution generalization on graphs: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(11): 10490–10512, 2025b.

Sihang Li, Xiang Wang, An Zhang, Yingxin Wu, Xiangnan He, and Tat-Seng Chua. Let invariant rationale discovery inspire graph contrastive learning. In ICML, pp. 13052–13065. PMLR, 2022b.

Yong Lin, Shengyu Zhu, Lu Tan, and Peng Cui. Zin: When and how to learn invariance without environment partition? In NeurIPS, 2022.

Yong Lin, Lu Tan, Yifan Hao, Honam Wong, Hanze Dong, Weizhong Zhang, Yujiu Yang, and Tong Zhang. Spurious feature diversification improves out-of-distribution generalization. In ICLR, 2024.

Siqi Miao, Mia Liu, and Pan Li. Interpretable and generalizable graph learning via stochastic attention mechanism. In ICML, pp. 15524–15543. PMLR, 2022.

Nicolas M Muller, Simon Roschmann, Shahbaz Khan, Philip Sperl, and Konstantin B¨ ottinger. Short-¨ cut detection with variational autoencoders. In IJCNN, pp. 1–7. IEEE, 2024.

Wenwen Qiang, Jingyao Wang, Zeen Song, Jiangmeng Li, and Changwen Zheng. On the out-ofdistribution generalization of self-supervised learning. In ICML, pp. 50086–50112, 2025.

Shauli Ravfogel, Michael Twiton, Yoav Goldberg, and Ryan Cotterell. Linear adversarial concept erasure. In ICML, 2022.

Joshua Robinson, Li Sun, Ke Yu, Kayhan Batmanghelich, Stefanie Jegelka, and Suvrit Sra. Can contrastive learning avoid shortcut solutions? Advances in neural information processing systems, 34:4974–4986, 2021.

Yu Rong, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang. Self-supervised graph transformer on large-scale molecular data. Advances in neural information processing systems, 33:12559–12571, 2020.

Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In ICLR, 2020.

Fan-Yun Sun, Jordan Hoffmann, Vikas Verma, and Jian Tang. Infograph: Unsupervised and semisupervised graph-level representation learning via mutual information maximization. In ICLR, 2020.

Xinwei Sun, Botong Wu, Xiangyu Zheng, Chang Liu, Wei Chen, Tao Qin, and Tie-Yan Liu. Recovering latent causal factor for generalization to distributional shifts. Advances in Neural Information Processing Systems, 34:16846–16859, 2021.

Xinyi Wang, Michael Saxon, Jiachen Li, Hongyang Zhang, Kun Zhang, and William Yang Wang. Causal balancing for domain generalization. In ICLR, 2023.

Qitian Wu, Hengrui Zhang, Junchi Yan, and David Wipf. Handling distribution shifts on graphs: An invariance perspective. In ICLR, 2022a.

Ying-Xin Wu, Xiang Wang, An Zhang, Xiangnan He, and Tat-Seng Chua. Discovering invariant rationales for graph neural networks. In ICLR, 2022b.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In ICLR, 2019.

Nianzu Yang, Kaipeng Zeng, Qitian Wu, Xiaosong Jia, and Junchi Yan. Learning substructure invariance for out-of-distribution molecular representations. Advances in Neural Information Processing Systems, 35:12964–12978, 2022a.

Wanqian Yang, Polina Kirichenko, Micah Goldblum, and Andrew G Wilson. Chroma-vae: Miti gating shortcut learning with generative classifiers. Advances in Neural Information Processing Systems, 35:20351–20365, 2022b.

Tianjun Yao, Yongqiang Chen, Zhenhao Chen, Kai Hu, Zhiqiang Shen, and Kun Zhang. Empowering graph invariance learning with deep spurious infomax. In ICML, pp. 56860–56884, 2024.

Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen. Graph contrastive learning with augmentations. In NeurIPS, 2020.

Xingxuan Zhang, Linjun Zhou, Renzhe Xu, Peng Cui, Zheyan Shen, and Haoxin Liu. Towards unsupervised domain generalization. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4900–4910. IEEE, 2022.

Yujia Zheng, Ignavier Ng, and Kun Zhang. On the identifiability of nonlinear ICA: Sparsity and beyond. Advances in Neural Information Processing Systems, 35:16411–16422, 2022.

## APPENDIX

## A PROOFS

## A.1 PROOF OF THEOREM 4.1 (RECONSTRUCTION FORCES RETENTION)

Throughout, $G = g ( C , S , \xi )$ is a discrete random variable, $Z = f _ { \theta } ( G )$ is the (possibly stochastic) encoder latent, so that $( C , { \overset { \cdot } { S } } , \xi ) \to G \to Z$ forms a Markov chain, and $p _ { \psi } ( \cdot \mid Z )$ is the decoder distribution over graphs. All entropies are in nats.

Step 1: Reconstruction loss upper-bounds the conditional entropy of G. For every value z of $Z ,$ the Gibbs inequality gives $\mathbb { E } [ - \mathrm { \bar { l o g } } p _ { \psi } ( G \mid z ) \mid Z = z ] \ge H ( G \mid Z = z )$ , with equality iff $p _ { \psi } ( \cdot \mid z )$ equals the true posterior $p ( \cdot \mid z )$ . Taking expectation over $Z ,$

$$
L _ { \mathrm { r e c } } = \mathbb { E } [ - \log p _ { \psi } ( G \mid Z ) ] \ \geq \ H ( G \mid Z ) .\tag{9}
$$

This is the only place the decoder enters: any encoder–decoder pair achieving reconstruction loss $L _ { \mathrm { r e c } }$ leaves at most $L _ { \mathrm { r e c } }$ nats of $G$ unexplained by $Z .$

Step 2: Injectivity turns S into a function of $( G , C , \xi ) . { \mathrm { ~ B y ~ } } ( { \mathrm { A l } } )$ , for each fixed $( c , \xi ^ { \prime } )$ the map $s \mapsto g ( c , s , \xi ^ { \prime } )$ is injective, so there exists a measurable $\phi _ { S }$ with $S = \phi _ { S } ( G , C , \xi )$ almost surely. Symmetrically, injectivity in the first argument yields ϕ<sub>C</sub> with $C = \phi _ { C } ( G , \dot { S } , \xi )$

Step 3: Chain of conditioning inequalities. Since $S$ is a deterministic function of $( G , C , \xi )$ , conditioning on $( Z , C , \xi )$ and using that the entropy of a function is at most the entropy of its argument,

$$
H ( S \mid Z , C , \xi ) = H ( \phi _ { S } ( G , C , \xi ) \mid Z , C , \xi ) \le H ( G \mid Z , C , \xi ) \le H ( G \mid Z ) ,\tag{10}
$$

where the last step is “conditioning reduces entropy.” Combining with equation 9, $H ( S \mid Z , C , \xi ) \le$ $L _ { \mathrm { r e c } }$

Step 4: Conclusion. By the chain rule for conditional mutual information,

$$
I ( Z ; S \mid C , \xi ) = H ( S \mid C , \xi ) - H ( S \mid Z , C , \xi ) \geq H ( S \mid C , \xi ) - L _ { \mathrm { r e c } } ,\tag{11}
$$

which is the first bound in equation 3. The second bound follows by exchanging the roles of C and S in Steps 2–3 using $\phi _ { C }$ . Both bounds are strictly positive whenever $L _ { \mathrm { r e c } } < \operatorname* { m i n } \{ H ( S \mid C , \xi )$ , H(C | $S , \xi ) \big \}$ , and (A2) guarantees the right-hand sides are finite so the statement is non-vacuous. 口

Why the bounds are per-factor. The two inequalities hold simultaneously for the same $L _ { \mathrm { r e c } }$ because each factor is recoverable from $G$ once the others are fixed; the reconstruction loss therefore has to pay for every factor separately. A discriminative objective offers no such guarantee: minimizing $\mathbb { E } [ - \log q ( Y | Z ) ]$ only constrains $I ( Z ; Y ) \leq H ( Y )$ , and an encoder that discards $C$ can still attain this whenever S predicts $Y$ on the training environments.

## A.2 PROOF OF LEMMA 4.2 (LOCAL CONTRACTION)

We first establish the local contraction result in four steps (Steps 1–4): deriving the confusion gradient, introducing centering, linearizing near uniform predictions, and showing per-direction contraction. We then derive the rank-one cumulative approximation in Steps 5–8.

Step 1: Confusion gradient in representation space. The per-example confusion loss is crossentropy with the uniform target $q = K ^ { - 1 } \mathbf { 1 }$ :

$$
L _ { \mathrm { c o n f } } ( h ) = - q ^ { \top } \log p ( h ) , \qquad p ( h ) = \operatorname { s o f t m a x } ( W h + b ) .\tag{12}
$$

The standard logit-space gradient of cross-entropy with target $q$ is $\nabla _ { z } L _ { \mathrm { c o n f } } = p - q$ . Using $z =$ $W h + b$ , the chain rule gives:

$$
\begin{array} { c } { { \nabla _ { h } L _ { \mathrm { c o n f } } = \displaystyle \frac { \partial z } { \partial h } ^ { \top } \nabla _ { z } L _ { \mathrm { c o n f } } } } \\ { { = W ^ { \top } ( p - q ) . } } \end{array}\tag{13}
$$

That ${ \mathrm { i s } } ,$ the representation gradient is determined entirely by the directions the adversary uses to distinguish environments.

Step 2: Centering removes the common logit shift. Since both p and $\boldsymbol { q }$ are probability vectors summing to one, their difference satisfies $\mathbf { 1 } ^ { \top } ( p - q ) = 0$ , so $p - q \in { \mathbf { 1 } } ^ { \perp }$ and $C ( p - q ) = p - q$ where $\breve { C } = I - \breve { K } ^ { - 1 } \mathbf { 1 1 } ^ { \top }$ . Therefore:

$$
\begin{array} { r l } & { \nabla _ { h } L _ { \mathrm { c o n f } } = W ^ { \top } ( p - q ) } \\ & { \qquad = W ^ { \top } C ( p - q ) } \\ & { \qquad = ( C W ) ^ { \top } ( p - q ) } \\ & { \qquad = W _ { c } ^ { \top } ( p - q ) , } \end{array}\tag{14}
$$

which shows the gradient lies in range $( W _ { c } ^ { \top } )$ . Since C has rank $K - 1$ , this subspace has dimension at most $K - 1$ . Centering removes the common logit shift and keeps only the environmentdiscriminative directions.

Step 3: Linearization near equal-logit points. $\mathrm { A t } \ h _ { 0 }$ where $W _ { c } h _ { 0 } + b _ { c } = 0$ , all centered logits vanish, hence all logits are equal and $p ( h _ { 0 } ) \ = \ q .$ The Jacobian of softmax evaluated at equal probabilities is:

$$
\begin{array} { l } { { \displaystyle { \cal J } _ { \mathrm { s m } } ( z ( h _ { 0 } ) ) = \mathrm { d i a g } ( q ) - q q ^ { \top } } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { K } \left( I - \frac { 1 } { K } { \bf 1 1 } ^ { \top } \right) } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { K } C . } } \end{array}\tag{15}
$$

A first-order Taylor expansion of p around $h _ { 0 } ,$ using $\delta h = h - h _ { 0 }$ and noting that $C W \delta h = W _ { c } \delta h$ gives:

$$
p ( h ) - q = \frac { 1 } { K } W _ { c } \delta h + O ( \| W _ { c } \delta h \| _ { 2 } ^ { 2 } ) .\tag{16}
$$

Substituting into the gradient expression from Step 2:

$$
\begin{array} { r l } & { \nabla _ { h } L _ { \mathrm { c o n f } } = W _ { c } ^ { \top } ( p - q ) } \\ & { \qquad = W _ { c } ^ { \top } \left( \frac { 1 } { K } W _ { c } \delta h + O ( \| W _ { c } \delta h \| _ { 2 } ^ { 2 } ) \right) } \\ & { \qquad = \displaystyle \frac { 1 } { K } W _ { c } ^ { \top } W _ { c } \delta h + O ( \| W _ { c } \delta h \| _ { 2 } ^ { 2 } ) . } \end{array}\tag{17}
$$

Step 4: Contraction per singular direction. A gradient step $h ^ { + } = h - \alpha \beta \nabla _ { h } L _ { \mathrm { c o n f } }$ gives, in terms of δh:

$$
\delta h ^ { + } = \left( I - \frac { \alpha \beta } { K } W _ { c } ^ { \top } W _ { c } \right) \delta h + O ( \alpha \beta \| W _ { c } \delta h \| _ { 2 } ^ { 2 } ) .\tag{18}
$$

Let $W _ { c } = U \mathrm { d i a g } ( s _ { 1 } , \ldots , s _ { r } ) V ^ { \top }$ be a thin SVD. Then:

$$
\boldsymbol { W _ { c } ^ { \intercal } W _ { c } } = V \mathrm { d i a g } ( s _ { 1 } ^ { 2 } , \dots , s _ { r } ^ { 2 } ) \boldsymbol { V } ^ { \intercal } ,\tag{19}
$$

and projecting onto each right singular direction $v _ { j }$

$$
v _ { j } ^ { \top } \delta h ^ { + } = \left( 1 - \frac { \alpha \beta } { K } s _ { j } ^ { 2 } \right) v _ { j } ^ { \top } \delta h + O ( \alpha \beta \| W _ { c } \delta h \| _ { 2 } ^ { 2 } ) .\tag{20}
$$

When $0 < \alpha \beta s _ { i } ^ { 2 } / K < 1$ , the leading-order factor is a contraction. Directions with larger adversarial singular values are contracted more strongly, meaning the representation selectively suppresses environment-discriminative components. For any direction $v \perp \mathrm { \bar { r a n g e } } ( V ) = \mathrm { r a n g e } ( \dot { W _ { c } ^ { \top } } )$ ), we have $W _ { c } ^ { \top } W _ { c } v = 0 .$ , so $v ^ { \top } \delta h ^ { + } = v ^ { \top } \delta h$ exactly; non-discriminative directions are unchanged.

Step 5: Rank-one approximation. When the adversary’s centered weight matrix has a dominant singular value $s _ { 1 } \gg s _ { 2 }$ , the leading-order update from Step 4 is well approximated by retaining only the $j = 1$ term. Writing $u _ { E } : = v _ { 1 }$ for the leading right singular direction and defining $\gamma : =$ $\alpha \beta \dot { s } _ { 1 } ^ { 2 } / K$ , the single-step update becomes:

$$
\begin{array} { r } { h ^ { ( 1 ) } \approx \left( I - \gamma u _ { E } u _ { E } ^ { \top } \right) h ^ { ( 0 ) } . } \end{array}\tag{21}
$$

Step 6: Decomposition into environment and orthogonal components. To see how the update acts on different components, decompose any representation as $h \ = \ h _ { E } \ + \ h _ { \perp }$ , where $h _ { E } = ( u _ { E } ^ { \top } h )$ u is the environment-direction component and $u _ { E } ^ { \top } h _ { \bot } = 0$ . Denoting the projection $P = \overline { { u } } _ { E } u _ { E } ^ { \top } \left( \mathrm { s o } P ^ { 2 } = P \right)$ , one confusion step gives:

$$
\begin{array} { r l } & { h ^ { ( 1 ) } = ( I - \gamma P ) ( h _ { E } + h _ { \perp } ) } \\ & { ~ = h _ { E } + h _ { \perp } - \gamma P h _ { E } } \\ & { ~ = \left( 1 - \gamma \right) h _ { E } + h _ { \perp } . } \end{array}\tag{22}
$$

That is, the environment component shrinks by a factor $( 1 - \gamma )$ , while the orthogonal component is unchanged.

Step 7: Cumulative effect of m confusion steps. Repeating the same rank-one update m times and using $P ^ { 2 } = P $

$$
\begin{array} { l } { { \displaystyle h ^ { ( m ) } = ( I - \gamma P ) ^ { m } h ^ { ( 0 ) } } } \\ { { \displaystyle = ( 1 - \gamma ) ^ { m } h _ { E } ^ { ( 0 ) } + h _ { \perp } ^ { ( 0 ) } . } } \end{array}\tag{23}
$$

Each step multiplicatively shrinks the environment component while leaving the orthogonal complement intact. Defining $\dot { \varepsilon } : = ( 1 - \gamma ) ^ { m } \in ( 0 , 1 ]$ , we can write:

$$
\begin{array} { r l } & { \boldsymbol { h } ^ { ( m ) } = \varepsilon \boldsymbol { h } _ { E } ^ { ( 0 ) } + \boldsymbol { h } _ { \perp } ^ { ( 0 ) } } \\ & { \qquad = \boldsymbol { h } ^ { ( 0 ) } - ( 1 - \varepsilon ) ( \boldsymbol { u } _ { E } ^ { \top } \boldsymbol { h } ^ { ( 0 ) } ) \boldsymbol { u } _ { E } . } \end{array}\tag{24}
$$

Step 8: The contraction model $T _ { \varepsilon } .$ . The cumulative multi-step effect can therefore be expressed as a single linear map:

$$
\mathbf { h } _ { G } ^ { ( \beta ) } \approx T _ { \varepsilon } \mathbf { h } _ { G } ^ { ( 0 ) } , \qquad T _ { \varepsilon } = I - \left( 1 - \varepsilon \right) u _ { E } u _ { E } ^ { \top } , \quad 0 < \varepsilon \leq 1 .\tag{25}
$$

This map has a transparent geometric action:

$$
\begin{array} { r l } & { \boldsymbol { u } _ { E } ^ { \top } \boldsymbol { T } _ { \varepsilon } h = \varepsilon ( \boldsymbol { u } _ { E } ^ { \top } h ) , } \\ & { \boldsymbol { v } ^ { \top } \boldsymbol { T } _ { \varepsilon } h = \boldsymbol { v } ^ { \top } h \quad \mathrm { f o r } \boldsymbol { v } \perp \boldsymbol { u } _ { E } . } \end{array}\tag{26}
$$

When $\varepsilon = 1 , T _ { \varepsilon } = I$ (no contraction); as $\varepsilon  0$ , the environment-discriminative component is increasingly suppressed, while all other directions are preserved. ■

Scope. Steps 1–4 are exact for the gradient direction and first-order for the contraction magnitude, holding for a single step with a fixed linear adversary and a directly updated representation. Steps 5–8 model the cumulative multi-step effect as a rank-one contraction under three simplifying assumptions: (a) the adversary weights (hence $u _ { E } )$ remain approximately stable across steps; (b) the encoder update acts directly on h rather than through neural network parameters; and (c) the dominant singular value $s _ { 1 }$ governs the contraction. In the implemented method, the encoder is updated through its parameter Jacobian (introducing a preconditioner), the adversary is nonlinear and alternately retrained, and the reconstruction loss acts simultaneously. The rank-one model $T _ { \varepsilon }$ is therefore a first-order analytical approximation that captures the leading effect of confusion training, suppressing the dominant environment-discriminative direction while preserving orthogonal components, and is verified empirically rather than derived as a theorem.

## A.3 PROOF OF PROPOSITION 4.3 (RIDGE IDENTITY)

The key insight is that contraction $T _ { \varepsilon }$ does not remove information; rather, it changes the downstream regularization geometry. We establish this through a change of variables.

Step 1: Invertibility of $T _ { \varepsilon }$ . For $\varepsilon > 0 , T _ { \varepsilon }$ is symmetric and invertible with:

$$
\begin{array} { r } { T _ { \varepsilon } ^ { - 1 } = I + ( \varepsilon ^ { - 1 } - 1 ) u _ { E } u _ { E } ^ { \top } . } \end{array}\tag{27}
$$

To verify, using $u _ { E } ^ { \top } u _ { E } = 1 \mathrm { : }$

$$
\begin{array} { r l } & { T _ { \varepsilon } \cdot T _ { \varepsilon } ^ { - 1 } = \left( I - ( 1 - \varepsilon ) u _ { E } u _ { E } ^ { \top } \right) \left( I + ( \varepsilon ^ { - 1 } - 1 ) u _ { E } u _ { E } ^ { \top } \right) } \\ & { \qquad = I + \left[ ( \varepsilon ^ { - 1 } - 1 ) - ( 1 - \varepsilon ) - ( 1 - \varepsilon ) ( \varepsilon ^ { - 1 } - 1 ) \right] u _ { E } u _ { E } ^ { \top } } \\ & { \qquad = I , } \end{array}\tag{28}
$$

since the bracketed coefficient simplifies to $\varepsilon ^ { - 1 } - 1 - 1 + \varepsilon - \varepsilon ^ { - 1 } + 1 + 1 - \varepsilon = 0 .$

Step 2: Squared inverse. By squaring $T _ { \varepsilon } ^ { - 1 }$ using the identity $( I + \gamma u _ { E } u _ { E } ^ { \top } ) ^ { 2 } = I + ( ( 1 + \gamma ) ^ { 2 } -$ $1 ) u _ { E } u _ { E } ^ { \top }$ with $\gamma = \varepsilon ^ { - 1 } - 1$

$$
\begin{array} { r } { T _ { \varepsilon } ^ { - 2 } = I + ( \varepsilon ^ { - 2 } - 1 ) u _ { E } u _ { E } ^ { \top } . } \end{array}\tag{29}
$$

Step 3: Change of variables. Substitute $v : = T _ { \varepsilon } w$ , which is a bijection by Step 1, leaving b unchanged. The prediction function is unchanged after reparameterization:

$$
\begin{array} { r } { \boldsymbol { w } ^ { \top } \boldsymbol { T } _ { \varepsilon } h = ( T _ { \varepsilon } \boldsymbol { w } ) ^ { \top } h } \\ { = \boldsymbol { v } ^ { \top } h , \quad } \end{array}\tag{30}
$$

so the loss terms agree pointwise.

Step 4: Transform the regularizer. For the penalty term, $w = T _ { \varepsilon } ^ { - 1 } v ,$ , and only the environment direction receives an additional penalty:

$$
\begin{array} { r l } & { \| w \| _ { 2 } ^ { 2 } = \| T _ { \varepsilon } ^ { - 1 } v \| _ { 2 } ^ { 2 } } \\ & { \qquad = v ^ { \top } T _ { \varepsilon } ^ { - 2 } v } \\ & { \qquad = v ^ { \top } \big [ I + ( \varepsilon ^ { - 2 } - 1 ) u _ { E } u _ { E } ^ { \top } \big ] v } \\ & { \qquad = \| v \| _ { 2 } ^ { 2 } + ( \varepsilon ^ { - 2 } - 1 ) ( v ^ { \top } u _ { E } ) ^ { 2 } . } \end{array}\tag{31}
$$

The two objectives are therefore equal as functions under the bijection $v = T _ { \varepsilon } w .$ , so their infima agree and optima correspond whenever they exist. The penalty matrix $\lambda T _ { \varepsilon } ^ { - 2 }$ has eigenvalue $\lambda / \varepsilon ^ { 2 }$ on span $( u _ { E } )$ and λ on $u _ { E } ^ { \perp }$ . Therefore, contraction is equivalent to directional feature reweighting: using the environment direction becomes disproportionately expensive under $\ell _ { 2 }$ regularization. ■

## A.4 EXACT WORKED EXAMPLE: STATEMENT AND PROOF

The mechanism improves OOD generalization when the contracted direction captures unstable environment-associated variation while remaining separated from invariant task-relevant information. We formalize this intuition using a two-dimensional Gaussian model; the statement is the one summarized in Section 4.3. The analysis highlights two essential requirements: (1) the environmentassociated feature induces a spurious training correlation, and (2) the downstream classifier applies finite regularization.

Proposition A.1 (Exact worked example). Let $h = ( z _ { c } + \eta ) u _ { Y } + \sigma _ { s } ( r z _ { c } + \sqrt { 1 - r ^ { 2 } } z _ { e } ) u _ { E }$ with $Y = z _ { c } ,$ , orthonormal $( u _ { Y } , u _ { E } ) , z _ { c } , z _ { e } \sim \mathcal { N } ( 0 , 1 )$ and $\eta \sim \mathcal { N } ( 0 , \sigma _ { \eta } ^ { 2 } )$ mutually independent, $\sigma _ { \eta } ^ { 2 } > 0 ,$ $\sigma _ { s } > 0$ , and training correlation $r = r _ { \mathrm { t r } }$ . Under population ridge regression with regularization $\lambda > 0 ;$

(i) $I f r _ { \mathrm { t r } } \ne 0$ and $r _ { \mathrm { t e } } = 0$ (correlation shift), the OOD risk $R ( \varepsilon ; r _ { \mathrm { t e } } )$ is strictly decreasing in the contraction amount: every reduction of ε helps.

(ii) As $\lambda  0 ^ { + }$ (no regularization), both fitted coefficients become independent of ε: contraction has no downstream effect.

(iii) If u ${ \bf \nabla } _ { E } = u _ { Y }$ (full alignment), contraction hurts because the compressed direction also contains task-relevant information.

This proposition makes the two requirements concrete. Part (i) shows the positive case: when a spurious correlation exists in training $( r _ { \mathrm { t r } } \neq 0 )$ but vanishes at test time $( r _ { \mathrm { t e } } = 0 )$ , contraction strictly reduces OOD risk. Two effects contribute simultaneously: the probe’s loading on the spurious feature decreases, and its core coefficient, which was deflated at $\varepsilon = 1$ because the environment feature could proxy for the label, is restored toward its test-optimal value. Parts (ii) and (iii) characterize the failure modes: without regularization the probe can undo the contraction for free, and when the environment direction overlaps the stable signal, contraction damages invariant information. Together, these results precisely delineate when SCARCE is expected to help and when it is not.

Proof. The proof follows the chain: data model → covariance $\Sigma$ and cross-covariance $b $ ridge solution $\hat { w } _ { \varepsilon } \to$ effective coefficients $a _ { c } , a _ { s } \to \mathrm { O O D }$ risk.

Step 1: Setup and covariance derivation. We work in the orthonormal basis $( u _ { Y } , u _ { E } )$ , so each representation is $h = ( h _ { 1 } , h _ { 2 } )$ with $h _ { 1 } = z _ { c } + \eta$ (core feature with noise) and $h _ { 2 } = \sigma _ { s } ( r z _ { c } +$ $\sqrt { 1 - r ^ { 2 } } z _ { e } )$ (environment feature). The label is $Y = z _ { c } .$ . In this basis, the contraction $T _ { \varepsilon }$ acts as $S _ { \varepsilon } = \mathrm { d i a g } ( 1 , \varepsilon )$ , leaving the core feature unchanged and compressing the environment feature.

Since all variables are zero-mean and $z _ { c } , z _ { e } , \eta$ are mutually independent with $\mathbb { E } [ z _ { c } ^ { 2 } ] = \mathbb { E } [ z _ { e } ^ { 2 } ] = 1$ and $\mathbb { E } [ \eta ^ { 2 } ] = \sigma _ { \eta } ^ { 2 }$ , the core variance is:

$$
\begin{array} { r l } & { \mathbb { E } [ h _ { 1 } ^ { 2 } ] = \mathbb { E } [ ( z _ { c } + \eta ) ^ { 2 } ] } \\ & { \qquad = \mathbb { E } [ z _ { c } ^ { 2 } ] + \mathbb { E } [ \eta ^ { 2 } ] } \\ & { \qquad = 1 + \sigma _ { \eta } ^ { 2 } . } \end{array}\tag{32}
$$

The cross-covariance between core and environment features is:

$$
\begin{array} { r l } & { \mathbb { E } [ h _ { 1 } h _ { 2 } ] = \sigma _ { s } \mathbb { E } \Big [ ( z _ { c } + \eta ) ( r z _ { c } + \sqrt { 1 - r ^ { 2 } } z _ { e } ) \Big ] } \\ & { \qquad = r \sigma _ { s } . } \end{array}\tag{33}
$$

This off-diagonal term directly captures the spurious correlation. Similarly, $\mathbb { E } [ h _ { 2 } ^ { 2 } ] = \sigma _ { s } ^ { 2 } .$ , giving:

$$
\Sigma = \mathbb { E } [ h h ^ { \top } ] = \binom { 1 + \sigma _ { \eta } ^ { 2 } \quad r \sigma _ { s } } { r \sigma _ { s } } , \quad \quad b = \mathbb { E } [ h Y ] = \binom { 1 } { r \sigma _ { s } } .\tag{34}
$$

Step 2: Ridge solution on contracted features. The contracted features are $x = S _ { \varepsilon } h ,$ , so $\mathbb { E } [ x x ^ { \top } ] =$ $S _ { \varepsilon } \bar { \Sigma } S _ { \varepsilon }$ and $\breve { \mathbb { E } } [ x Y ] = S _ { \varepsilon } b$ . The population ridge regression solution is:

$$
\begin{array} { c } { \hat { w } _ { \varepsilon } = ( \mathbb { E } [ x x ^ { \top } ] + \lambda I ) ^ { - 1 } \mathbb { E } [ x Y ] } \\ { = ( S _ { \varepsilon } \Sigma S _ { \varepsilon } + \lambda I ) ^ { - 1 } S _ { \varepsilon } b . } \end{array}\tag{35}
$$

The determinant of the $2 \times 2$ system is:

$$
\Delta _ { \varepsilon } = ( 1 + \sigma _ { \eta } ^ { 2 } + \lambda ) ( \varepsilon ^ { 2 } \sigma _ { s } ^ { 2 } + \lambda ) - \varepsilon ^ { 2 } r ^ { 2 } \sigma _ { s } ^ { 2 } > 0 \quad ( \lambda > 0 ) .\tag{36}
$$

The effective coefficients on the original (unshrunk) features, defined by writing the prediction as $\hat { Y } = a _ { c } h _ { 1 } + a _ { s } h _ { 2 }$ with $a _ { c } = [ \hat { w } _ { \varepsilon } ]$ <sub>1</sub> and $a _ { s } = \varepsilon [ \hat { w } _ { \varepsilon } ] _ { 2 }$ , are:

$$
a _ { c } ( \varepsilon ) = \frac { \lambda + \varepsilon ^ { 2 } \sigma _ { s } ^ { 2 } ( 1 - r ^ { 2 } ) } { \Delta _ { \varepsilon } } , \qquad a _ { s } ( \varepsilon ) = \frac { \varepsilon ^ { 2 } r \sigma _ { s } ( \sigma _ { \eta } ^ { 2 } + \lambda ) } { \Delta _ { \varepsilon } } .\tag{37}
$$

Step 3: OOD risk derivation. At test time, $h _ { 2 } = \sigma _ { s } ( r _ { \mathrm { t e } } z _ { c } + \sqrt { 1 - r _ { \mathrm { t e } } ^ { 2 } } z _ { e } )$ . The prediction error is:

$$
\begin{array} { r l } & { Y - \hat { Y } = z _ { c } - a _ { c } ( z _ { c } + \eta ) - a _ { s } \sigma _ { s } \big ( r _ { \mathrm { t e } } z _ { c } + \sqrt { 1 - r _ { \mathrm { t e } } ^ { 2 } z _ { e } } \big ) } \\ & { \qquad = \left( 1 - a _ { c } - a _ { s } \sigma _ { s } r _ { \mathrm { t e } } \right) z _ { c } - a _ { c } \eta - a _ { s } \sigma _ { s } \sqrt { 1 - r _ { \mathrm { t e } } ^ { 2 } } z _ { e } . } \end{array}\tag{38}
$$

Since $z _ { c } , \eta , z _ { e }$ are mutually independent and zero-mean with $\mathbb { E } [ z _ { c } ^ { 2 } ] = \mathbb { E } [ z _ { e } ^ { 2 } ] = 1$ and $\mathbb { E } [ \eta ^ { 2 } ] = \sigma _ { \eta } ^ { 2 }$ squaring and taking expectations gives:

$$
\begin{array} { r l } & { R ( \varepsilon ; r _ { \mathrm { t e } } ) = \mathbb { E } [ ( Y - \hat { Y } ) ^ { 2 } ] } \\ & { \quad \quad \quad = ( 1 - a _ { c } - a _ { s } \sigma _ { s } r _ { \mathrm { t e } } ) ^ { 2 } } \\ & { \quad \quad \quad + a _ { c } ^ { 2 } \sigma _ { \eta } ^ { 2 } } \\ & { \quad \quad \quad + a _ { s } ^ { 2 } \sigma _ { s } ^ { 2 } ( 1 - r _ { \mathrm { t e } } ^ { 2 } ) . } \end{array}\tag{39}
$$

This decomposition is transparent: the first term is the squared bias from under- or over-relying on the core feature, the second is the noise penalty from using the noisy core channel, and the third is the variance penalty from loading on the environment feature whose correlation shifts.

We now prove each claim.

Claim (i): Strict benefit under correlation shift. Let $r = r _ { \mathrm { t r } } \neq 0 , r _ { \mathrm { t e } } = 0 , \lambda > 0$ . Define $t : = \varepsilon ^ { 2 }$ note that t increases as contraction weakens (larger ε), so it suffices to show R is strictly increasing

in t. Setting $r _ { \mathrm { t e } } = 0$ , the risk simplifies to $R ( \varepsilon ; 0 ) = ( 1 - a _ { c } ) ^ { 2 } + a _ { c } ^ { 2 } \sigma _ { \eta } ^ { 2 } + a _ { s } ^ { 2 } \sigma _ { s } ^ { 2 }$ . We show each summand is strictly increasing in t.

For the spurious term: $a _ { s } ( t ) = c _ { 1 } t / ( c _ { 2 } + c _ { 3 } t )$ with $c _ { 1 } = r \sigma _ { s } ( \sigma _ { \eta } ^ { 2 } + \lambda ) , c _ { 2 } = \lambda ( 1 + \sigma _ { \eta } ^ { 2 } + \lambda ) > 0$ , and $c _ { 3 } = \sigma _ { s } ^ { 2 } ( 1 + \sigma _ { \eta } ^ { 2 } + \lambda - r ^ { 2 } ) > 0$ . Its derivative $c _ { 1 } c _ { 2 } / ( c _ { 2 } + c _ { 3 } t ) ^ { 2 }$ has sign sign(r) and never vanishes, so $a _ { s } ^ { 2 }$ is strictly increasing in t.

For the core term, write $a _ { c } ( t ) = N ( t ) / \Delta ( t )$ where both $N ( t )$ and $\Delta ( t )$ are affine in $t .$ Direct differentiation gives:

$$
a _ { c } ^ { \prime } ( t ) = \frac { N ^ { \prime } ( t ) \Delta ( t ) - N ( t ) \Delta ^ { \prime } ( t ) } { \Delta ( t ) ^ { 2 } } .\tag{40}
$$

Since both functions are affine, the numerator $N ^ { \prime } \Delta - N \Delta ^ { \prime } = n _ { 1 } d _ { 0 } - n _ { 0 } d _ { 1 }$ is constant in t. Substituting $n _ { 0 } = \lambda , n _ { 1 } = \sigma _ { s } ^ { 2 } ( 1 - r ^ { 2 } ) , d _ { 0 } = \lambda ( 1 + \sigma _ { \eta } ^ { 2 } + \lambda ) , \mathrm { a n d } d _ { 1 } = \sigma _ { s } ^ { 2 } ( 1 \dot { + } \sigma _ { \eta } ^ { 2 } + \lambda - r ^ { 2 } )$

$$
\begin{array} { c } { { n _ { 1 } d _ { 0 } - n _ { 0 } d _ { 1 } = \sigma _ { s } ^ { 2 } ( 1 - r ^ { 2 } ) \lambda ( 1 + \sigma _ { \eta } ^ { 2 } + \lambda ) - \lambda \sigma _ { s } ^ { 2 } ( 1 + \sigma _ { \eta } ^ { 2 } + \lambda - r ^ { 2 } ) } } \\ { { { } } } \\ { { = - \lambda \sigma _ { s } ^ { 2 } r ^ { 2 } ( \sigma _ { \eta } ^ { 2 } + \lambda ) < 0 . } } \end{array}\tag{41}
$$

Since $\Delta ( t ) ^ { 2 } > 0$ , we conclude $a _ { c } ^ { \prime } ( t ) < 0 , \ s \ o { a _ { c } }$ is strictly decreasing in t. Since $a _ { c } < 1 / ( 1 + \sigma _ { \eta } ^ { 2 } ) = :$ $a ^ { * }$ (the test-optimal core coefficient), and $f ( a ) : = ( 1 - a ) ^ { 2 } + \sigma _ { \eta } ^ { 2 } a ^ { 2 }$ has $f ^ { \prime } ( a ) < 0$ for $a < a ^ { * }$ the composition $f ( a _ { c } ( t ) )$ is strictly increasing in t. Summing the two strictly increasing terms shows $\bar { R ( \varepsilon ; 0 ) }$ is strictly increasing in $t = \varepsilon ^ { 2 }$ , hence strictly decreasing as ε decreases (stronger contraction).

Claim (ii): Regularization is necessary. As $\lambda \to 0 ^ { + } , \Delta _ { \varepsilon } \to \varepsilon ^ { 2 } \sigma _ { s } ^ { 2 } ( 1 + \sigma _ { \eta } ^ { 2 } - r ^ { 2 } )$ and the factors of $\varepsilon ^ { 2 }$ cancel in both expressions of equation 37:

$$
\begin{array} { l } { { a _ { c } \to \displaystyle \frac { 1 - r ^ { 2 } } { 1 + \sigma _ { \eta } ^ { 2 } - r ^ { 2 } } , } } \\ { { a _ { s } \to \displaystyle \frac { r \sigma _ { \eta } ^ { 2 } } { \sigma _ { s } ( 1 + \sigma _ { \eta } ^ { 2 } - r ^ { 2 } ) } . } } \end{array}\tag{42}
$$

Both coefficients, hence the entire fitted predictor, become independent of $\varepsilon .$ This confirms that without regularization, the downstream probe can freely rescale its weight along the contracted direction by $1 / \varepsilon$ , undoing the contraction entirely.

Claim (iii): Alignment counterexample. When $u _ { E } ~ = ~ u _ { Y }$ , the representation collapses to one dimension $h = z _ { c } + \eta + \sigma _ { s } z _ { e }$ (taking $r = 0$ for clarity, $\sigma _ { n } ^ { 2 } = \sigma _ { s } ^ { 2 } = \mathrm { \bar { 1 } } , \lambda = 0 . 1 )$ . The contraction rescales this single coordinate by ε. The ridge coefficient on the unshrunk feature is:

$$
v ( \varepsilon ) = \frac { \varepsilon ^ { 2 } } { 3 \varepsilon ^ { 2 } + 0 . 1 } ,\tag{43}
$$

giving:

$$
R ( \varepsilon ) = 1 - 2 v ( \varepsilon ) + 3 v ( \varepsilon ) ^ { 2 } .\tag{44}
$$

Numerically, $R ( 1 ) \approx 0 . 6 6 7$ while $R ( 0 . 1 ) \approx 0 . 8 6 4$ : contraction strictly hurts because the contracted direction carries the core signal. This confirms that the separation between $u _ { E }$ and $u _ { Y }$ is essential for the mechanism to succeed. ■

## B DFR PROTOCOL

We follow Deep Feature Reweighting (DFR; Kirichenko et al. 2023), which retrains only the last layer of a fixed encoder and thereby reweights the features the encoder already exposes. Given an encoder checkpoint $f _ { \theta } ,$ , the protocol is:

1. Freeze and extract. Freeze $f _ { \theta } ,$ , discard its original classification head, and extract pooled representations $\mathbf { h } _ { G } = f _ { \theta } ( G )$ for the training, validation, and test graphs.

2. Group-balanced reweighting set. Define groups by the pair (label, environment). Keep all graphs of the smallest group and randomly subsample every other group to the same size, yielding a balanced set $\hat { \mathcal { D } }$ drawn from the group-labeled validation split; group labels are used only here and never during encoder training.

3. Standardize. Standardize each latent dimension with the mean and standard deviation computed on $\hat { \mathcal { D } }$ , and apply the same transform to validation and test representations.

4. Sparse linear head. Fit a logistic-regression head on $\hat { \mathcal { D } }$ with $\ell _ { 1 }$ regularization, min $\begin{array} { r } { \mathrm { \ i } _ { W , b } \sum _ { ( \mathbf { h } , y ) \in \hat { \mathcal { D } } } \ell ( W \mathbf { h } + b , y ) + \lambda \| W \| _ { 1 } } \end{array}$ <sub>1</sub>. The sparsity penalty suits the regime $| \hat { D } | \ll d$ and pushes the coefficients of irrelevant or spurious directions toward zero. The inverse strength $C \propto 1 / \lambda$ is selected from $\{ 1 , 0 . 7 , 0 . 3 , 0 . 1 , \dot { 0 . } 0 7 , 0 . 0 3 , 0 . 0 1 \}$ by worst-group accuracy on a heldout half of the validation split, after which the head is refit on the full balanced set.

5. Average over resamples. Steps 2–4 are repeated 10 times with independently resampled balanced subsets, and the resulting weights and biases are averaged, ${ \dot { W } } _ { \mathrm { D F R } } ~ { \stackrel { \sim } { = } } ~ { \frac { 1 } { 1 0 } } \sum _ { r = 1 } ^ { 1 { \stackrel { \cdot } { 0 } } } W ^ { ( r ) }$ which stabilizes the head.

The final predictor is ${ \hat { y } } = \arg \mathbf { \Sigma } _ { \mathrm { ~ } }$ max $W _ { \mathrm { D F R } } f _ { \theta } ( G ) + b _ { \mathrm { D F R } }$ , evaluated on the OOD test split. In Figure $2 , f _ { \theta }$ is a GIN encoder taken from checkpoints at epochs $\{ 1 0 , 3 0 , 6 0 , 9 0 , 1 2 0 , 1 5 0 \}$ of either supervised ERM training or unsupervised VGAE training on GOOD-Motif concept shift; the ERM curve reports the accuracy of the ERM model with its original head. All numbers are means and standard deviations over 3 seeds. DFR does not modify the representation: it changes only how the frozen features are weighted, so its accuracy measures what the encoder already makes linearly accessible.

## C ADDITIONAL REAL-WORLD RESULTS

The real-world experiments freeze the learned graph encoders and fit linear probes using 5%, 10%, or 20% of the labeled training split. Model and configuration selection uses validation performance, and the test split is reserved for final evaluation. This section expands the representative analyses in the main text across all available molecular shifts.

## C.1 HYPERPARAMETER SENSITIVITY ACROSS DRUGOOD

Tables 3, 4, and 5 list the six highest-scoring evaluated $( \alpha , \beta , K )$ configurations for each of the six DrugOOD settings at 5%, 10%, and 20% labeled-training ratios, respectively. Within each dataset– domain block, rows are ordered by test ROC-AUC at that ratio; untested configurations are not included or interpolated.

## C.2 COMPLETE TRAINING DYNAMICS

Across the complete set of runs, the three objectives remain bounded, although their scales and transient behavior depend on the dataset and split. These plots are optimization diagnostics: stable losses rule out persistent numerical divergence but do not by themselves establish invariance or causal removal of environment information.

Table 3: Top evaluated pseudo-environment configurations at 5% labels. Within each dataset– domain setting, the six displayed $( \alpha , \beta , K )$ configurations are ordered from highest to lowest test ROC-AUC (%). Untested configurations are not interpolated.
<table><tr><td>Dataset</td><td>Domain</td><td>α</td><td> $\beta$ </td><td>K</td><td>Test ROC-AUC (%)</td></tr><tr><td>EC50</td><td>Assay</td><td>0</td><td>10</td><td>3</td><td>69.84</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>20</td><td>65.70</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>20</td><td>65.63</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>65.61</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>3</td><td>64.28</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>20</td><td>63.96</td></tr><tr><td></td><td>Scaffold</td><td>1</td><td>10</td><td>20</td><td>63.75</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>20</td><td>61.71</td></tr><tr><td></td><td></td><td>0</td><td>0.01</td><td>20</td><td>61.00</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>60.30</td></tr><tr><td></td><td></td><td>0</td><td>1</td><td>20</td><td>60.27</td></tr><tr><td></td><td></td><td>0</td><td>0</td><td>20</td><td>50.0</td></tr><tr><td></td><td>Size</td><td>1</td><td>10</td><td>20</td><td>58.83</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>10</td><td>58.65</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>3</td><td>58.59</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>3</td><td>58.24</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>10</td><td>57.32</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>57.14</td></tr><tr><td>IC50</td><td>Assay</td><td>1</td><td>10</td><td>20</td><td>65.70</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>64.87</td></tr><tr><td></td><td></td><td>0.5</td><td>1</td><td>3</td><td>64.74</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>3</td><td>64.38</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>3</td><td>64.32</td></tr><tr><td></td><td></td><td>0</td><td>0.1</td><td>3</td><td>64.24</td></tr><tr><td></td><td>Scaffold</td><td>0</td><td>10</td><td>3</td><td>60.57</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>3</td><td>60.40</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>59.46</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>5</td><td>59.23</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>10</td><td>58.53</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>5</td><td>58.48</td></tr><tr><td></td><td>Size</td><td>0</td><td>10</td><td>20</td><td>58.22</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>20</td><td>58.14</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>58.11</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>3</td><td>57.83</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>3</td><td>57.71</td></tr><tr><td></td><td></td><td>0.5</td><td>1</td><td>20</td><td>57.55</td></tr></table>

Table 4: Top evaluated pseudo-environment configurations at 10% labels. Within each dataset– domain setting, the six displayed $( \alpha , \beta , K )$ configurations are ordered from highest to lowest test ROC-AUC (%). Untested configurations are not interpolated.
<table><tr><td>Dataset</td><td>Domain</td><td>α</td><td> $\beta$ </td><td>K</td><td>Test ROC-AUC (%)</td></tr><tr><td>EC50</td><td>Assay</td><td>0</td><td>10</td><td>3</td><td>70.38</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>10</td><td>68.45</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>68.32</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>5</td><td>67.17</td></tr><tr><td></td><td></td><td>0</td><td>0.1</td><td>3</td><td>66.59</td></tr><tr><td></td><td></td><td>0.5</td><td>1</td><td>20</td><td>66.39</td></tr><tr><td></td><td>Scaffold</td><td>1</td><td>10</td><td>10</td><td>61.03</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>10</td><td>61.00</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>10</td><td>60.99</td></tr><tr><td></td><td></td><td>0</td><td>0.01</td><td>20</td><td>60.53</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>20</td><td>60.31</td></tr><tr><td></td><td></td><td>0</td><td>0</td><td>20</td><td>49.9</td></tr><tr><td></td><td>Size</td><td>1</td><td>10</td><td>20</td><td>60.90</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>3</td><td>60.06</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>3</td><td>59.62</td></tr><tr><td></td><td></td><td>0.5</td><td>1</td><td>20</td><td>59.16</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>10</td><td>59.11</td></tr><tr><td></td><td></td><td>0</td><td>1</td><td>20</td><td>59.06</td></tr><tr><td>IC50</td><td>Assay</td><td>1</td><td>10</td><td>20</td><td>66.90</td></tr><tr><td></td><td></td><td>0</td><td>0.1</td><td>3</td><td>65.77</td></tr><tr><td></td><td></td><td>0.5</td><td>1</td><td>3</td><td>65.48</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>20</td><td>65.16</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>3</td><td>65.00</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>64.88</td></tr><tr><td></td><td>Scaffold</td><td>0</td><td>10</td><td>3</td><td>61.52</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>5</td><td>61.11</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>3</td><td>60.94</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>20</td><td>60.84</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>10</td><td>60.83</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>20</td><td>60.55</td></tr><tr><td></td><td>Size</td><td>0</td><td>10</td><td>20</td><td>59.44</td></tr><tr><td></td><td></td><td>0.5</td><td>1</td><td>20</td><td>59.44</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>20</td><td>59.18</td></tr><tr><td></td><td></td><td>0.5</td><td>1</td><td>3</td><td>58.94</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>3</td><td>58.84</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>3</td><td>58.74</td></tr></table>

Table 5: Top evaluated pseudo-environment configurations at 20% labels. Within each dataset– domain setting, the six displayed $( \alpha , \beta , K )$ configurations are ordered from highest to lowest test ROC-AUC (%). Untested configurations are not interpolated.
<table><tr><td>Dataset</td><td>Domain</td><td>α</td><td> $\beta$ </td><td>K</td><td>Test ROC-AUC (%)</td></tr><tr><td>EC50</td><td>Assay</td><td>0</td><td>10</td><td>3</td><td>73.07</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>3</td><td>71.55</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>69.67</td></tr><tr><td></td><td></td><td>0.5</td><td>1</td><td>20</td><td>69.25</td></tr><tr><td></td><td></td><td>0</td><td>1</td><td>3</td><td>68.31</td></tr><tr><td></td><td></td><td>0.5</td><td>1</td><td>3</td><td>68.31</td></tr><tr><td></td><td>Scaffold</td><td>0</td><td>10</td><td>10</td><td>61.23</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>10</td><td>60.59</td></tr><tr><td></td><td></td><td>0</td><td>0.01</td><td>20</td><td>60.42</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>20</td><td>60.06</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>59.93</td></tr><tr><td></td><td></td><td>0</td><td>0</td><td>20</td><td>53.1</td></tr><tr><td></td><td>Size</td><td>0.5</td><td>10</td><td>3</td><td>61.61</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>3</td><td>60.71</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>20</td><td>60.71</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>20</td><td>59.84</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>5</td><td>59.76</td></tr><tr><td></td><td></td><td>0</td><td>1</td><td>3</td><td>59.53</td></tr><tr><td>IC50</td><td>Assay</td><td>1</td><td>10</td><td>20</td><td>67.39</td></tr><tr><td></td><td></td><td>0</td><td>0.1</td><td>3</td><td>66.60</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>20</td><td>66.29</td></tr><tr><td></td><td></td><td>0.5</td><td>1</td><td>3</td><td>66.22</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>3</td><td>65.56</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>65.25</td></tr><tr><td></td><td>Scaffold</td><td>0.5</td><td>10</td><td>3</td><td>61.92</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>10</td><td>61.74</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>3</td><td>61.54</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>5</td><td>61.47</td></tr><tr><td></td><td></td><td>0.5</td><td>10</td><td>5</td><td>61.46</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>20</td><td>61.41</td></tr><tr><td></td><td>Size</td><td>0.5</td><td>1</td><td>20</td><td>60.73</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>20</td><td>60.09</td></tr><tr><td></td><td></td><td>1</td><td>10</td><td>20</td><td>60.04</td></tr><tr><td></td><td></td><td>0.5</td><td>1</td><td>3</td><td>59.91</td></tr><tr><td></td><td></td><td>0</td><td>1</td><td>20</td><td>59.43</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>59.36</td></tr><tr><td></td><td></td><td>0</td><td>10</td><td>3</td><td></td></tr></table>

Table 6: GOOD-Motif concept hyperparameter sensitivity. The evaluated configurations use fixed K = 8 and $k _ { \mathrm { a d v } } = 2 .$ Test accuracy is the best-epoch linear-probe result.
<table><tr><td rowspan=1 colspan=2>α          $\beta$      Test accuracy (%)</td></tr><tr><td rowspan=1 colspan=2>0     0.01                        65.0</td></tr><tr><td rowspan=1 colspan=2>0       0.1                        62.8</td></tr><tr><td rowspan=1 colspan=2>0         1                       66.0</td></tr><tr><td rowspan=1 colspan=2>0        10                        61.7</td></tr><tr><td rowspan=1 colspan=2>0.5     0.01                        62.3</td></tr><tr><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.1                        69.6</td></tr><tr><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>1                       63.6</td></tr><tr><td rowspan=1 colspan=2>0.5        10                       53.8</td></tr><tr><td rowspan=1 colspan=2>1.0     0.01                        65.3</td></tr><tr><td rowspan=1 colspan=2>1.0       0.1                        69.6</td></tr></table>

![](images/a97729e643c21284fba5ad64bbd03a5bc4e27663b9ef00c744f58340864bd3a6.jpg)  
Figure 5: Training dynamics on EC50-assay. The plot uses the validation-selected $( \alpha , \beta , K ) =$ (0, 10, 3). Reconstruction, confusion, and adversary losses use solid, dashed, and dash-dotted lines with triangle, circle, and square markers, respectively. Lines are means over three seeds and shaded regions denote mean ± sample standard deviation.

![](images/cd61f00202a43bf6f315b3a226d672a6f3db48e22860f2829d06fe05040c332a.jpg)  
Figure 6: Training dynamics on EC50-scaffold. The validation-selected $( \alpha , \beta , K )$ configuration is (1, 10, 20). Lines and shaded regions follow the convention of Figure 5.

![](images/6f586ca1bf18bc2913eea8f0d5c3a4ee5edaee8df014af9766297f1d98bc2332.jpg)  
Figure 7: Training dynamics on EC50-size. The validation-selected $( \alpha , \beta , K )$ configuration is (0.5, 10, 3). Lines and shaded regions follow the convention of Figure 5.

![](images/4055dba9386f5408b42cff62126c2a8ab0349c15f5b0d652a9a30b4c4d9ea5d7.jpg)  
Figure 8: Training dynamics on IC50-assay. The validation-selected $( \alpha , \beta , K )$ configuration is (1, 10, 20). Lines and shaded regions follow the convention of Figure 5.

$$
\begin{array} { r l r l } {  } & { { } \xrightarrow { } \mathrm { R e c o n s t r u c t i o n } } & { } & { { } \xrightarrow { } \mathrm { o } \circ - \mathrm { C o n f u s i o n } } & { } & { { } \xrightarrow { } \circ - \mathrm { A d v e r s a r y } } \end{array}
$$

![](images/a7fc87546851452dfe6365dd981b92b1293dfa90f999a7c13fb853a10ab79f8d.jpg)  
Figure 9: Training dynamics on IC50-scaffold. The validation-selected $( \alpha , \beta , K )$ configuration is $( 0 . 5 , 1 0 , 3 )$ . The vertical axis is truncated at 5 to show the stable-loss range; values above this limit are placed at the boundary and connected to the next sampled vertex. Lines and shaded regions follow the convention of Figure 5.

$$
\begin{array} { r l r l } {  } & { { } \xrightarrow { } \mathrm { R e c o n s t r u c t i o n } } & { } & { { } \xrightarrow { } \mathrm { o } \ldot { \mathrm { \textmu - } } \mathrm { C o n f u s i o n } } & { } & { { } \xrightarrow {  } \mathrm { A d v e r s a r y } } \end{array}
$$

![](images/b8575b441d61fea0065126941068e089e562021b5f2a72635279f16aa0bf20b4.jpg)  
Figure 10: Training dynamics on IC50-size. The validation-selected $( \alpha , \beta , K )$ configuration is (0.5, 10, 3). The vertical axis is logarithmic because reconstruction has a substantially larger transient scale. Lines and shaded regions follow the convention of Figure 5.

## D AI USAGE STATEMENT

The authors used generative AI tools to polish the writing, to assist with the literature survey, to assist with the implementation of experiments, and to check mathematical derivations. All research ideas, method design, experimental setups, results, and conclusions were developed and verified by the authors, who take full responsibility for the content of this paper.