# On the Sparsity of Direct Preference Optimization: Weight Disentanglement in the NTK Regime

Kensuke Sasaki Department of Computer Science The University of Tokyo sasaki-kensuke0222@g.ecc.u-tokyo.ac.jp

Issei Sato   
Department of Computer Science The University of Tokyo   
sato@g.ecc.u-tokyo.ac.jp

## Abstract

Direct Preference Optimization (DPO) has been widely adopted for aligning language models with human preferences. Recent empirical work has revealed that DPO induces remarkably sparse parameter updates, yet no theoretical explanation for this phenomenon has been offered. We analyze DPO’s optimization dynamics under a neural tangent kernel linearization and show that the logistic structure of the DPO loss gives rise to an implicit maximum-margin bias. Building on this observation, we introduce the concept of dual sparsity: the DPO update direction is a sparse linear combination (data-level sparsity, driven by support vectors) of sparse gradient feature vectors (parameter-level sparsity, arising from gradient cancellation between preferred and dispreferred responses). We formalize this structure in our main theorem, bound the number of support pairs by a margindependent spectral quantity, and present a framework relating dual sparsity to weight disentanglement—a condition favorable for model merging. Experiments on GPT-2 Small and Medium and on 7–9B instruction-tuned models (Llama-3- 8B, Mistral-7B, Gemma-2-9B) with the Anthropic HH-RLHF and UltraFeedback datasets confirm both sources of sparsity, and experiments on GPT-2 show that DPO reduces cross-task interference by up to 35× compared to supervised fine-tuning.

## 1 Introduction

Direct Preference Optimization [Rafailov et al., 2023] has emerged as a dominant paradigm for aligning large language models with human preferences. By casting the alignment objective as a logistic loss over preference pairs, DPO circumvents the instabilities of reinforcement learning from human feedback while retaining strong empirical performance. As practitioners increasingly train separate DPO models for distinct objectives—helpfulness, harmlessness, instruction following—there is growing interest in composing these independently trained models via model merging [Ilharco et al., 2023, Yu et al., 2024], which combines task-specific parameter updates (task vectors) into a single multitask model without additional training.

A striking but poorly understood empirical observation is that DPO fine-tuning induces highly sparse parameter updates. Mukherjee et al. [2025] recently demonstrated that reinforcement learning-based alignment concentrates its modifications onto a small subnetwork of the pre-trained model, leaving the vast majority of parameters essentially unchanged. This sparsity stands in marked contrast to supervised fine-tuning (SFT), which typically distributes updates broadly across the parameter space. Independently, Ortiz-Jiménez et al. [2023] identified weight disentanglement—the condition that task vectors act on non-overlapping regions of parameter space—as a key property enabling successful model merging. A natural question arises: is the sparsity of DPO a mere coincidence, or does it reflect a deeper structure in the optimization that is systematically favorable for composing task vectors?

In this paper, we provide a theoretical answer to this question. We analyze the implicit bias of DPO under a neural tangent kernel (NTK) linearization [Jacot et al., 2018] and show that the logistic loss structure, combined with the geometry of preference-pair gradients, gives rise to a phenomenon we call dual sparsity. The DPO update direction converges to the maximum-margin solution in the NTK feature space, inheriting two complementary forms of sparsity: data-level sparsity, where only a small subset of preference pairs (support vectors) contribute to the final update, and parameter-level sparsity, where the gradient feature vector for each pair is itself sparse due to gradient cancellation between preferred and dispreferred responses that share common tokens. Building on this, we present a framework relating dual sparsity to weight disentanglement, and verify empirically that DPO produces cross-task inner products an order of magnitude smaller than those under SFT.

Our main contributions are as follows:

(C1) We reduce DPO to maximum-margin classification under NTK linearization and introduce dual sparsity—a joint data-level and parameter-level sparsity structure—formalized in theorem 3. This provides the first theoretical explanation for why DPO induces sparse parameter updates without any explicit sparsity-promoting regularization.

(C2) We present a framework relating dual sparsity to weight disentanglement [Ortiz-Jiménez et al., 2023] via a triplet-space formulation of the DPO margin (section 3.3). The framework reduces the disentanglement condition to a per-triplet inner product whose magnitude is governed by the parameter-space overlap between tasks—an empirical question that we evaluate in section 4.

(C3) We validate both sources of sparsity on GPT-2 Small (124M) and Medium (355M) fine-tuned with the Anthropic HH-RLHF [Bai et al., 2022] and UltraFeedback [Cui et al., 2024] datasets, and at modern scale on Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, and Gemma-2-9Bit. DPO feature vectors are over 90% near-zero from 124M to 9B parameters, and the time-integrated weights concentrate on a small subset of pairs within three epochs of full fine-tuning at the 7–8B scale. On GPT-2, cross-task interference under DPO is reduced by 11× to 35× relative to SFT across all configurations.

The remainder of the paper is organized as follows. Section 2 reviews the DPO formulation and weight disentanglement framework. Section 3 presents our theoretical analysis, culminating in the dual sparsity theorem and a framework connecting it to weight disentanglement. Section 4 provides experimental validation. Section 5 discusses related work, and section 6 concludes with limitations and future directions.

## 2 Preliminaries

## 2.1 Direct Preference Optimization

Let $\pi _ { \mathrm { r e f } }$ denote a reference policy, typically a pre-trained or supervised fine-tuned (SFT) model [Rafailov et al., 2023]. Given a prompt x and a pair of responses $( y ^ { + } , y ^ { - } )$ where $y ^ { + }$ is preferred and $y ^ { - }$ is dispreferred, the log-probability margin under parameters θ is

$$
m _ { i } ( \theta ) = \beta \left( \log \frac { \pi _ { \theta } ( y _ { i } ^ { + } \mid x _ { i } ) } { \pi _ { \theta } ( y _ { i } ^ { - } \mid x _ { i } ) } - \log \frac { \pi _ { \mathrm { r e f } } ( y _ { i } ^ { + } \mid x _ { i } ) } { \pi _ { \mathrm { r e f } } ( y _ { i } ^ { - } \mid x _ { i } ) } \right) ,\tag{1}
$$

with temperature $\beta > 0$ . For a preference dataset $\mathcal { D } = \{ ( x _ { i } , y _ { i } ^ { + } , y _ { i } ^ { - } ) \} _ { i = 1 } ^ { n }$ , the DPO loss is the logistic loss

$$
\ell _ { i } ( \theta ) = \log \left( 1 + e ^ { - m _ { i } ( \theta ) } \right) ,\tag{2}
$$

whose gradient w.r.t. θ is weighted by $\sigma ( - m _ { i } ( \theta ) ) \in ( 0 , 1 )$ : the gradient vanishes for “easy” pairs already preferred by the policy, focusing optimization on the “hard” subset near the decision boundary.

## 2.2 Weight Disentanglement

We review the notion of weight disentanglement [Ortiz-Jiménez et al., 2023], which characterizes an ideal condition for composing independently trained task vectors without cross-task interference.

Consider a neural network $f \colon \mathcal { X } \times \Theta  \mathcal { Y }$ with pre-trained parameters $\theta _ { 0 }$ . A task t is specified by a data support ${ \mathcal { D } } _ { t } \subseteq { \mathcal { X } } ,$ an input distribution $\mu _ { t }$ with $\mathrm { s u p p } ( \mu _ { t } ) = \mathcal { D } _ { t }$ , and a target function $f _ { t } ^ { * } \colon D _ { t } \to \mathcal { V }$ . Fine-tuning $f$ on task t yields parameters $\theta _ { t } ^ { * }$ , and the corresponding task vector is defined as $\tau _ { t } : = \theta _ { t } ^ { * } - \theta _ { 0 }$

Definition 1 (Task Arithmetic, Ilharco et al., 2023). Let $\mathcal { T } = \{ \tau _ { t } \} _ { t \in [ T ] }$ be a set of task vectors with mutually disjoint supports $\mathcal { D } _ { t } \cap \mathcal { D } _ { t ^ { \prime } } = \emptyset$ for $t \neq t ^ { \prime }$ . The network $f$ satisfies the task arithmetic property around $\theta _ { 0 }$ if

$$
f \left( x ; \theta _ { 0 } + \sum _ { s = 1 } ^ { T } \alpha _ { s } \tau _ { s } \right) = \left\{ \begin{array} { l l } { f ( x ; \theta _ { 0 } + \alpha _ { t } \tau _ { t } ) } & { x \in \mathcal { D } _ { t } , } \\ { f ( x ; \theta _ { 0 } ) } & { x \notin \bigcup _ { t } \mathcal { D } _ { t } . } \end{array} \right.\tag{3}
$$

Definition 2 (Weight Disentanglement, Ortiz-Jiménez et al., 2023). The network $f$ is weightdisentangled with respect to $\tau$ and $\{ \mathcal { D } _ { t } \}$ if there exist functions $g _ { t } \colon \mathcal { X } \times \Theta  \mathcal { Y }$ with $g _ { t } ( x ; \cdot ) = 0$ for x $\notin { \mathcal { D } } _ { t }$ <sub>t</sub> and a residual $g _ { 0 }$ with $g _ { 0 } ( x ) = 0$ for $x \in \textstyle \bigcup _ { t } { \mathcal { D } } _ { t }$ , such that

$$
f \Biggl ( x ; \theta _ { 0 } + \sum _ { t = 1 } ^ { T } \alpha _ { t } \tau _ { t } \Biggr ) = \sum _ { t = 1 } ^ { T } g _ { t } ( x ; \alpha _ { t } \tau _ { t } ) + g _ { 0 } ( x ) .\tag{4}
$$

It is straightforward to verify that weight disentanglement (definition 2) implies the task arithmetic property (definition 1) by setting $g _ { t } ( x ; \alpha _ { t } \tau _ { t } ) ~ = ~ f ( x ; \theta _ { 0 } + \alpha _ { t } \tau _ { t } ) \mathbf { 1 } ( x ~ \in ~ \mathcal { D } _ { t } )$ and $g _ { 0 } ( x ) =$ $f ( x ; \boldsymbol { \theta } _ { 0 } ) \mathbf { 1 } \bar { ( } x \not \in \bigcup _ { t } \mathcal { D } _ { t } )$ ).

## 3 Theoretical Analysis

Recent empirical work has shown that alignment fine-tuning via DPO induces highly sparse parameter updates relative to the pre-trained weights [Mukherjee et al., 2025]. Independently, weight disentanglement has been identified as a desirable property for robust model merging [Ortiz-Jiménez et al., 2023]. The following questions remain open:

1. Why does DPO induce sparsity? What mechanism in the DPO optimization dynamics leads to sparse parameter updates, despite the absence of any explicit sparsity-promoting regularization?

2. What are the implications of this sparsity for cross-task interference? Specifically, does the sparsity inherent in DPO updates reduce the interference between independently trained task vectors?

This section addresses both questions. We analyze the implicit bias of DPO under a neural tangent kernel (NTK) linearization (section 3.1) and identify a dual sparsity structure that emerges from the interplay between the logistic loss and the geometry of the gradient features (section 3.2). We then present a framework relating this structure to weight disentanglement, with empirical evaluation deferred to section 4 (section 3.3).

## 3.1 DPO as Maximum-Margin Classification in Tangent Space

We begin by linearizing the DPO margin around the initial parameters $\theta _ { 0 }$ . Under the neural tangent kernel (NTK) regime [Jacot et al., 2018], the margin admits a first-order approximation:

$$
m _ { i } ( \theta ) \approx m _ { i } ( \theta _ { 0 } ) + \langle \Phi _ { i } , \Delta \theta \rangle ,\tag{5}
$$

where $\Delta \theta = \theta - \theta _ { 0 }$ and the gradientfeature vector $\Phi _ { i }$ is defined as

$$
\begin{array} { r } { \Phi _ { i } : = \beta \left( \nabla _ { \theta } \log \pi _ { \theta _ { 0 } } ( y _ { i } ^ { + } \mid x _ { i } ) - \nabla _ { \theta } \log \pi _ { \theta _ { 0 } } ( y _ { i } ^ { - } \mid x _ { i } ) \right) \in \mathbb { R } ^ { p } . } \end{array}\tag{6}
$$

Here $p$ denotes the total number of parameters. Each $\Phi _ { i }$ encodes the direction in parameter space along which a perturbation $\Delta \theta$ most effectively increases the preference margin for pair i. Under this linearization, the DPO loss reduces to a binary logistic regression over the features $\{ \Phi _ { i } \} _ { i = 1 } ^ { n } .$ with offsets $b _ { i } : = m _ { i } ( \theta _ { 0 } ) ;$

$$
\ell _ { i } ( \Delta \theta ) = \log ( 1 + \exp ( - \langle \Phi _ { i } , \Delta \theta \rangle - b _ { i } ) ) .\tag{7}
$$

With the standard DPO initialization $\pi _ { \theta _ { 0 } } = \pi _ { \mathrm { r e f } }$ , the offsets vanish: $b _ { i } = m _ { i } ( \theta _ { 0 } ) = 0$ by eq. (1).

We make the following assumption, which is standard in the analysis of implicit bias for logistic regression [Soudry et al., 2018]:

Assumption 1 (Linear Separability). There exist $\Delta \theta \in \mathbb { R } ^ { p }$ and $\gamma > 0$ such that $\left. \Phi _ { i } , \Delta \theta \right. + b _ { i } \geq \gamma$ for all $\bar { i } \in [ n ]$

Under assumption 1, the gradient flow on the total loss $\begin{array} { r } { L ( \Delta \theta ) = \sum _ { i } \ell _ { i } ( \Delta \theta ) } \end{array}$ starting from $\Delta \theta ( 0 ) = 0$ admits the explicit trajectory

$$
\Delta \theta ( T ) = \sum _ { i = 1 } ^ { n } A _ { i } ( T ) \Phi _ { i } , \quad A _ { i } ( T ) : = \int _ { 0 } ^ { T } \sigma ( - m _ { i } ( \theta ( t ) ) ) d t ,\tag{8}
$$

where $A _ { i } ( T )$ is the time-integrated gradient weight for pair i. Applying the implicit-bias result of Soudry et al. [2018] to this setup yields:

Lemma 1 (Implicit Bias of $\mathrm { D P O } ) .$ . Under assumption $^ { l , }$ the normalized parameter update converges to the $L _ { 2 }$ maximum-margin direction:

$$
\frac { \Delta \theta ( T ) } { \| \Delta \theta ( T ) \| } \xrightarrow { T \to \infty , } u ^ { \star } , \quad w h e r e u ^ { \star } s o l \nu e s \quad \operatorname* { m a x } _ { \| u \| \le 1 } \operatorname* { m i n } _ { i \in [ n ] } \left\{ \langle u , \Phi _ { i } \rangle + b _ { i } \right\} .\tag{9}
$$

$B y$ the Karush–Kuhn–Tucker (KKT) conditions, $u ^ { \star }$ admits the representation

$$
u ^ { \star } = \sum _ { i \in \mathrm { S V } } \alpha _ { i } ^ { \star } \Phi _ { i } , \qquad \alpha _ { i } ^ { \star } \geq 0 ,\tag{10}
$$

where $\mathrm { S V } : = \{ i : \alpha _ { i } ^ { \star } > 0 \}$ is the set of support pairs.

Remark 1 (Empirical observability of dual variables). The normalized time-integrated coefficients $\alpha _ { i } ( T ) : = \boldsymbol { A } _ { i } ( \dot { T } ) / \| \boldsymbol { \Delta } \theta ( T ) \|$ also converge to the dual variables $\alpha _ { i } ^ { \star }$ as $T \to \infty$ . This justifies tracking $A _ { i }$ as an empirical proxy for the support set in section 4.2.

## 3.2 Dual Sparsity of DPO Updates

Lemma 1 establishes that the DPO update direction is fully determined by the support pairs and the corresponding feature vectors. We now identify two distinct sources of sparsity in this representation.

Definition 3 (Dual Sparsity). The DPO update direction $\begin{array} { r } { u ^ { \star } = \sum _ { i \in \mathrm { S V } } \alpha _ { i } ^ { \star } \Phi _ { i } } \end{array}$ is said to exhibit dual sparsity if the following two conditions hold:

(i) Data-level sparsity: The number of support pairs is small relative to the dataset, i.e., $| \mathrm { S V } | \ll n$

(ii) Parameter-level sparsity: Each gradient feature vector $\Phi _ { i }$ is sparse, i.e., $\lVert \Phi _ { i } \rVert _ { 0 } \ll p$ for $i \in \mathrm { S V } .$

The data-level sparsity is a direct consequence of the maximum-margin structure:

Lemma 2 (Data-Level Sparsity). Suppose $\pi _ { \theta _ { 0 } } = \pi _ { \mathrm { r e f } }$ (so that $b _ { i } = 0 )$ and assumption 1 holds. Let $K : = \Phi ^ { \top } \Phi \in \mathbb { R } ^ { n \times n }$ , with $K _ { i j } = \langle \Phi _ { i } , \Phi _ { j } \rangle$ , be the Gram matrix of $\mathbf { \dot { \Phi } } \Phi : = [ \Phi _ { 1 } \Phi _ { 2 } \dotsm \Phi _ { n } ] ,$ , and let $\gamma _ { \star } : = \mathrm { m i n } _ { i \in [ n ] } \langle u ^ { \star } , \Phi _ { i } \rangle > 0$ be the maximum margin. Then,for the linearized model $e q . \ ( 7 )$

$$
| \mathrm { S V } | \gamma _ { \star } ^ { 2 } \le \lambda _ { \operatorname* { m a x } } ( K _ { \mathrm { S V } , \mathrm { S V } } ) \le \lambda _ { \operatorname* { m a x } } ( K ) , \qquad h e n c e \qquad | \mathrm { S V } | \le \operatorname* { m i n } \left\{ n , \bigg | \frac { \lambda _ { \operatorname* { m a x } } ( K ) } { \gamma _ { \star } ^ { 2 } } \bigg | \right\} ,\tag{11}
$$

where $K _ { \mathrm { S V , S V } }$ is the principal submatrix of K indexed by $\operatorname { S V } .$

The proof is given in section A.1.

This bound is meaningful because, in the DPO setting, the gradient flow concentrates its cumulative weight $A _ { i } ( T )$ on the pairs that remain difficult throughout training—those near the margin boundary— while the remaining pairs receive diminishing gradient signal. Since the bound is governed by the ratio $\lambda _ { \operatorname* { m a x } } ( K ) / \gamma _ { \star } ^ { 2 }$ rather than explicitly by $n ,$ the effective support set is provably $\ll n$ rather than merely ≤ rank(Φ) whenever the margin is large relative to the spectral scale of the features; we confirm empirically that this is the case on real preference data (see section 4.2).

For the parameter-level sparsity, we introduce the following assumption:

Assumption 2 (Sparsity of Gradient Features). For each $i \in [ n ]$ , the gradient feature vector $\Phi _ { i }$ satisfies $\| \Phi _ { i } \| _ { 0 } \leq s$ for some $s \ll p$

Remark 2 (Motivation for assumption 2). In autoregressive language models, the preferred and dispreferred responses $y _ { i } ^ { + }$ and $y _ { i } ^ { - }$ for a given prompt x<sub>i</sub> typically share a substantial number of tokens—particularly in the initial portion of the response. Since $\Phi _ { i }$ is defined as the difference of gradients $\begin{array} { r } { \beta ( \nabla _ { \theta } \log \pi _ { \theta _ { 0 } } ( y _ { i } ^ { + } \mid x _ { i } ) \stackrel { - } { - } \nabla _ { \theta } \log \pi _ { \theta _ { 0 } } ( y _ { i } ^ { - } \ \overline { { \mid x _ { i } ) } } ) } \end{array}$ , the parameter-level contributions from shared tokens cancel, leaving nonzero entries only in the parameters responsible for distinguishing between $y _ { i } ^ { + }$ and $y _ { i } ^ { - }$ . This cancellation effect provides a structural basis for assumption 2. We verify this assumption empirically in section 4.

Combining lemmas 1 and 2 with assumption 2, we arrive at the main result:

Theorem 3 (Dual Sparsity of DPO). Under assumptions 1 and 2, the DPO update direction $u ^ { \star }$ exhibits dual sparsity in the sense ofdefinition 3. That $i s , u ^ { \star }$ is a sparse linear combination of sparse vectors:

$$
u ^ { \star } = \sum _ { i \in \mathrm { S V } } \alpha _ { i } ^ { \star } \Phi _ { i } , \quad | \mathrm { S V } | \ll n , \quad \| \Phi _ { i } \| _ { 0 } \leq s \ll p .\tag{12}
$$

Consequently, $u ^ { \star }$ is itself sparse with $\| u ^ { \star } \| _ { 0 } \leq | \mathrm { S V } | \cdot s .$

## 3.3 Connection to Weight Disentanglement

Theorem 3 establishes that DPO updates exhibit dual sparsity. We now examine how this structure relates to the weight disentanglement framework reviewed in section 2.2. The original formulation of weight disentanglement (definition 2) is stated for a generic function $f \colon \mathcal { X } \times \Theta \stackrel { \cdot } { \to } \mathcal { Y } ;$ applying it to the DPO setting requires identifying the appropriate input space and function in preference learning.

Triplet-space formulation. We take the input space to be the space of preference triplets $\tilde { \mathcal { X } } : =$ $\{ ( x , y ^ { + } , y ^ { - } ) \}$ and define $f \colon \tilde { \mathcal { X } } \times \Theta  \mathbb { R }$ as the DPO log-probability margin

$$
f ( \tilde { x } ; \theta ) : = \beta \big ( \log \pi _ { \theta } ( y ^ { + } \mid x ) - \log \pi _ { \theta } ( y ^ { - } \mid x ) \big ) , \quad \tilde { x } = ( x , y ^ { + } , y ^ { - } ) .\tag{13}
$$

Differentiating at $\theta _ { 0 }$ recovers the gradient feature vector of section 3.1: $\nabla _ { \theta } f ( \tilde { x } ; \theta _ { 0 } ) = \Phi ( \tilde { x } )$ . The same features that drive dual sparsity therefore serve as the Jacobian relevant for weight disentanglement. A task t is associated with a preference dataset $\mathcal { D } _ { t } \subseteq \tilde { \mathcal { X } }$ collected under task-specific annotation criteria.

Linearized decomposition. For task vectors $\{ \tau _ { t } \} _ { t \in [ T ] }$ and coefficients $\{ \lambda _ { t } \} _ { t \in [ T ] }$ , a first-order Taylor expansion of f around $\theta _ { 0 }$ gives

$$
f \biggl ( \tilde { x } ; \theta _ { 0 } + \sum _ { t = 1 } ^ { T } \lambda _ { t } \tau _ { t } \biggr ) \approx f ( \tilde { x } ; \theta _ { 0 } ) + \sum _ { t = 1 } ^ { T } \lambda _ { t } \langle \tau _ { t } , \Phi ( \tilde { x } ) \rangle .\tag{14}
$$

Equation (14) matches the structural form of the weight disentanglement decomposition (definition 2) under the identification

$$
g _ { 0 } ( \tilde { x } ) : = f ( \tilde { x } ; \theta _ { 0 } ) , g _ { t } ( \tilde { x } ; \lambda _ { t } \tau _ { t } ) : = \lambda _ { t } \langle \tau _ { t } , \Phi ( \tilde { x } ) \rangle .\tag{15}
$$

With this identification, the disentanglement condition $g _ { t } ( \tilde { x } ; \lambda _ { t } \tau _ { t } ) = 0$ for $\tilde { x } \notin \mathcal { D } _ { t }$ reduces to the inner-product condition

$$
\langle \tau _ { t } , \Phi ( \tilde { x } ) \rangle \approx 0 \mathrm { f o r } \tilde { x } \notin \mathcal { D } _ { t } .\tag{16}
$$

Observation 1 (Linearized weight disentanglement under DPO). Under the NTK linearization, the DPO margin $f$ admits the decomposition eq. (14), and the identification eq. (15) places this decomposition in the form of weight disentanglement (definition 2). The disentanglement condition holds to the extent that the cross-task inner product eq. (16) is small.

Connection to dual sparsity. Lemma 1 establishes that, in the limit, the normalized task vector for task t aligns with the maximum-margin direction $\begin{array} { r } { \boldsymbol { u } _ { t } ^ { \star } = \sum _ { i \in \mathrm { S V } _ { t } } \boldsymbol { \alpha } _ { t , i } ^ { \star } \Phi ( \tilde { \boldsymbol { x } } _ { i } ^ { ( t ) } ) } \end{array}$ . Substituting into eq. (15), the local function $g _ { t }$ takes the form

$$
g _ { t } ( \tilde { x } ; \lambda _ { t } \tau _ { t } ) \propto \langle u _ { t } ^ { \star } , \Phi ( \tilde { x } ) \rangle = \Big \langle \sum _ { i \in \mathrm { S V } _ { t } } \alpha _ { t , i } ^ { \star } \Phi ( \tilde { x } _ { i } ^ { ( t ) } ) , \Phi ( \tilde { x } ) \Big \rangle .\tag{17}
$$

The cross-task interference eq. (16) is therefore governed by $\langle u _ { 1 } ^ { \star } , \Phi ( \tilde { x } ) \rangle$ ⟩ for $\tilde { x } \in \mathcal D _ { 2 }$ . By theorem $^ 3 \cdot$ $u _ { 1 } ^ { \star }$ is a sparse combination of sparse vectors; assumption $2$ ensures $\Phi ( \tilde { x } )$ is also sparse. When the active parameter supports of $u _ { 1 } ^ { \bar { \star } }$ and $\{ \Phi ( \tilde { x } ) \} _ { \tilde { x } \in \mathcal { D } _ { \tilde { x } } }$ have small overlap, this inner product is small. Whether the overlap is in fact small for naturally occurring preference tasks is an empirical question, which we examine in section 4.

Remark on the disjoint-support assumption. Definition 1 assumes disjoint task supports $\mathcal { D } _ { t } \cap$ $\mathcal { D } _ { t ^ { \prime } } = \varnothing$ . In the triplet-space formulation this is naturally satisfied: distinct preference datasets arise from distinct annotation processes, and an ordered triplet $( x , y ^ { + } , y ^ { - } )$ is tied to the task it was annotated under.

## 4 Experiments

We empirically validate the theoretical predictions of section 3. Specifically, we verify (i) data-level sparsity of the dual coefficients $\alpha _ { i } ^ { \star }$ (lemma 2), (ii) parameter-level sparsity of the gradient features $\Phi _ { i }$ (assumption 2), and (iii) suppression of cross-task interference under DPO compared to SFT (section 4.4).

## 4.1 Experimental Setup

Models. We use GPT-2 Small and GPT-2 Medium [Radford et al., 2019], together with three instruction-tuned models with 7–9B parameters: Llama-3-8B-Instruct [Grattafiori et al., 2024], Mistral-7B-Instruct-v0.2 [Jiang et al., 2023], and Gemma-2-9B-it [Gemma Team, 2024]. Each checkpoint serves as both the initialization and the reference policy, $\pi _ { \theta _ { 0 } } = \pi _ { \mathrm { r e f } }$

Datasets. For our primary experiments, we use the Anthropic HH-RLHF dataset [Bai et al., 2022], which consists of two subsets: helpful (evaluating which response is more useful to the user) and harmless (evaluating which response avoids harmful content). These two distinct subsets serve as the tasks $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ in our cross-task interference analysis. To assess generality, we additionally use the UltraFeedback Binarized dataset [Cui et al., 2024], a large-scale preference dataset annotated by GPT-4 across four evaluation axes (instruction-following, truthfulness, honesty, and helpfulness).

Gradient features. The gradient feature vectors $\Phi _ { i } = \beta ( \nabla _ { \theta } \log \pi _ { \theta _ { 0 } } ( y _ { i } ^ { + } \mid x _ { i } ) - \nabla _ { \theta } \log \pi _ { \theta _ { 0 } } ( y _ { i } ^ { - } \mid$ $x _ { i } ) )$ are computed empirically using the Jacobian of the model at initialization, since the analytical NTK cannot be obtained in closed form for self-attention architectures. Because storing full-dimensional gradient vectors $( \Phi _ { i } \in \mathbb { R } ^ { p }$ with $p > 1 0 ^ { 8 } )$ is memory-intensive, experiments that require $\Phi _ { i }$ are performed on subsamples of the datasets (see details in section B).

The models, datasets, and numbers of preference pairs used in each experiment, together with the computation of the time-integrated weights $A _ { i } .$ , are detailed in section B.

## 4.2 Data-Level Sparsity

Lemma $2$ predicts that the DPO update direction is driven by a small number of support pairs, with the remaining pairs contributing negligibly. To verify this, we examine how the time-integrated weights $A _ { i } ( \bar { T } )$ are distributed across pairs as training progresses.

Figure 1 visualizes $A _ { i }$ for GPT-2 Small and Medium on $N = 1 0 0$ pairs from HH-RLHF at 1, 100, and 1000 epochs. At early stages, the weights are distributed nearly uniformly. As training progresses, the distribution becomes increasingly polarized: a small subset of pairs accumulates large weights while the rest stop growing (see section C for the corresponding log-probabilities). This is the behavior predicted by lemma 1, under which the normalized coefficients $\alpha _ { i } ( T )$ converge to the dual variables $\alpha _ { i } ^ { \star }$ , which are nonzero only for the support pairs.

Table 1 quantifies this concentration on $N = 1 0 0 0$ pairs by the top-10% mass, i.e., the fraction of $\textstyle \sum _ { i } A _ { i }$ held by the top 10% of pairs (10% under a uniform distribution). For $\mathrm { G P T } { \cdot } 2 .$ , the top-10% mass grows from about 11% to 40–44% over 1000 epochs; for Llama-3-8B-Instruct and Mistral-7B-Instruct-v0.2, it grows from about 11% to 44–52% within three epochs of full fine-tuning DPO. The numerical evaluation of the bound in lemma 2 in the setting of fig. 1 is reported in section A.2.

![](images/e99d668e7764e8fc074ae9c3468f2ed1a7c3840c5a0d3fded9c84a5ad97346cd.jpg)  
(d) GPT-2 (M), Epoch 1

![](images/31c7068303438d6aa77c83e10ad85893918722edba3e0517cee03cc21c9e32e0.jpg)  
(e) GPT-2 (M), Epoch 100

![](images/afb00f8288ace27119e76537f03fe90611e7fd51f08c188dceae5d0311546547.jpg)  
(f)GPT-2 (M), Epoch 1000

![](images/24e9555e6511e8449fb630b47b00d0ef1ec8f2a96b59e7bfa24c5a36257d8ea7.jpg)

![](images/1557b038940c4c639a950ca6dcb24319d3d00b948c5a1069ff4aaf03128892f6.jpg)

![](images/15a02261f8595ea743f5119cbb234be3d1987f33c61671e5425d01c78d9b0645.jpg)  
Figure 1: Distribution of $A _ { i }$ across data pairs for GPT-2 Small (top) and Medium (bottom) at 1 epoch (left), 100 epochs (middle), and 1000 epochs (right) on HH-RLHF. Dark bars indicate the top 10% of pairs by weight. As training progresses, the distribution polarizes onto a small number of support pairs; this effect is more pronounced in the larger model.

Table 1: Top-10% mass of $A _ { i }$ on $N = 1 0 0 0$ pairs, i.e., the fraction of $\textstyle \sum _ { i } A _ { i }$ held by the top 10% of pairs (10% under a uniform distribution). Top: GPT-2 trained with DPO on HH-RLHF. Bottom: 7–8B models trained with full fine-tuning DPO on UltraFeedback Binarized.
<table><tr><td>Model</td><td>Dataset</td><td colspan="3">Top-10% mass (%)</td></tr><tr><td></td><td></td><td>Epoch 1</td><td>Epoch 100</td><td>Epoch 1000</td></tr><tr><td>GPT-2 Small</td><td>HH-RLHF</td><td>10.88</td><td>28.78</td><td>39.57</td></tr><tr><td>GPT-2 Medium</td><td>HH-RLHF</td><td>11.02</td><td>32.32</td><td>43.70</td></tr><tr><td></td><td></td><td>Epoch 1</td><td>Epoch 2</td><td>Epoch 3</td></tr><tr><td>Llama-3-8B-Instruct</td><td>UltraFeedback</td><td>11.75</td><td>41.82</td><td>52.48</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>UltraFeedback</td><td>11.23</td><td>39.92</td><td>43.77</td></tr></table>

## 4.3 Parameter-Level Sparsity

Assumption 2 posits that the DPO gradient feature vectors $\Phi _ { i }$ are sparse due to gradient cancellation between preferred and dispreferred responses. To verify this and to contrast it with SFT, fig. 2 visualizes the element-wise distribution of a representative Φ<sub>i</sub> under both methods (GPT-2 Medium, HH-RLHF, N = 100), and table 2 quantifies the fraction of near-zero entries for all five models $( N = 1 0 0 0$ , UltraFeedback Binarized).

Under DPO (fig. 2a), the vast majority of entries are concentrated near zero, with only a small fraction taking on appreciable magnitude. Under SFT (fig. 2b), the distribution is substantially broader, reflecting the fact that the SFT feature vector $\Phi _ { i } ^ { \mathrm { { S F T } } } = \nabla _ { \theta } \log \pi _ { \theta } ( y _ { i } ^ { + } \mid x _ { i } )$ involves only the preferred response and does not benefit from the gradient cancellation that arises in the DPO difference $\nabla _ { \theta } \log \pi _ { \theta _ { 0 } } ( y _ { i } ^ { + } \mid x _ { i } ) - \nabla _ { \theta } \log \pi _ { \theta _ { 0 } } ( y _ { i } ^ { - } \mid x _ { i } )$ . The same contrast holds at every scale we test (table 2): from 124M to 9B parameters, more than 90% of the entries of the DPO feature vector are near zero, whereas only 4–20% of those of the SFT feature vector are.

![](images/9eb313e11cf7b124b22dcdda112b86dd26819dfef826bbe3e5e2d0c3e4c36dbf.jpg)

![](images/8133ceff9b7b6320e394322c5b884d35872fb50bdbeee390cd07f24d510afa3f.jpg)  
Figure 2: Element-wise distribution of gradient feature vectors on a logarithmic scale for GPT-2 Medium on $N = 1 0 0$ pairs from HH-RLHF. (a) DPO: the vast majority of entries are near zero, confirming assumption 2. (b) SFT: the distribution is substantially denser, since no gradient cancellation occurs. Near-zero fractions for all five models, computed in a separate setting, are reported in table 2.

Table 2: Fraction of near-zero entries in a representative gradient feature vector under DPO $( \Phi _ { i } )$ and SFT $( \Phi _ { i } ^ { \mathrm { S F T } } )$ , computed from 1,000 pairs sampled from UltraFeedback Binarized. Higher values indicate sparser features.
<table><tr><td>Model</td><td>Params</td><td> $\mathbf { D P O } \Phi _ { i } \left( \% \right)$ </td><td> $\mathbf { S F T } \Phi _ { i } ^ { \mathrm { S F T } } \left( \% \right)$ </td></tr><tr><td>GPT-2 Small</td><td>124M</td><td>91.29</td><td>3.86</td></tr><tr><td>GPT-2 Medium</td><td>355M</td><td>94.82</td><td>4.34</td></tr><tr><td>Llama-3-8B-Instruct</td><td>8B</td><td>92.51</td><td>16.44</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>7B</td><td>90.93</td><td>19.72</td></tr><tr><td>Gemma-2-9B-it</td><td>9B</td><td>92.12</td><td>13.57</td></tr></table>

As discussed in remark 2, this contrast arises naturally from the structure of preference data in autoregressive models: since $y _ { i } ^ { + }$ and $\bar { y _ { i } ^ { - } }$ <sup>−</sup> typically share a large number of tokens—particularly in the initial portion of the response—the parameter-level gradients for these shared tokens cancel in the DPO feature vector, leaving nonzero contributions only for the parameters that distinguish between the two responses.

## 4.4 Cross-Task Interference

We evaluate whether the dual sparsity of DPO updates translates into reduced cross-task interference, as discussed in section 3.3. We consider two task pairs of increasing semantic distance:

(a) HH helpful ↔ harmless: two subsets of the same dataset (Anthropic HH-RLHF) with different annotation criteria—helpfulness vs. harmlessness.

(b) HH helpful ↔ UltraFeedback: two entirely different datasets with different annotation methods (human vs. AI), different prompt distributions, and different preference dimensions.

For the cross-dataset pair (b), only the gradient features $\Phi _ { j } ^ { ( 2 ) }$ of UltraFeedback at $\theta _ { 0 }$ are required; no DPO training on UltraFeedback is performed.

Baseline. We compare DPO against SFT. Unlike DPO, SFT does not produce the logistic loss structure that gives rise to implicit maximum-margin bias; consequently, all data pairs contribute to the update direction, and data-level sparsity does not emerge. Moreover, the SFT feature vector $\Phi _ { i } ^ { \mathrm { { S F T } } } \stackrel { * } { = } \nabla _ { \theta } \log \pi _ { \theta } ( y _ { i } ^ { + } \mid x _ { i } )$ involves only the preferred response, so it does not benefit from the gradient cancellation that induces parameter-level sparsity in DPO.

Table 3: Cross-task interference for DPO and SFT across two model scales (GPT-2 Small/Medium) and two task pairs: HH help↔harm (HH-RLHF helpful vs. harmless subsets) and HH help↔UF (HH-RLHF helpful vs. UltraFeedback). Lower values indicate greater disentanglement. Across all configurations, DPO reduces the interference $\langle u _ { 1 } ^ { \star } , V _ { 2 } \rangle$ by 11× to 35× relative to SFT.
<table><tr><td>Task Pair</td><td>Method</td><td>Model</td><td>Jaccard (%)</td><td> $\langle u _ { 1 } ^ { \star } , V _ { 2 } \rangle$ </td></tr><tr><td rowspan="4">HH help↔harm</td><td>SFT</td><td>Small</td><td>54.9</td><td>9.12</td></tr><tr><td>DPO</td><td>Small</td><td>18.9</td><td>0.258</td></tr><tr><td>SFT</td><td>Medium</td><td>49.3</td><td>3.22</td></tr><tr><td>DPO</td><td>Medium</td><td>11.4</td><td>0.191</td></tr><tr><td rowspan="4">HH help↔UF</td><td>SFT</td><td>Small</td><td>68.4</td><td>21.2</td></tr><tr><td>DPO</td><td>Small</td><td>25.2</td><td>1.91</td></tr><tr><td>SFT</td><td>Medium</td><td>60.5</td><td>19.8</td></tr><tr><td>DPO</td><td>Medium</td><td>17.7</td><td>0.852</td></tr></table>

Metrics. The disentanglement condition derived in section 3.3 is a per-triplet condition $\langle \tau _ { t } , \Phi ( { \tilde { x } } ) \rangle \approx 0$ for $\tilde { x } \notin \mathcal { D } _ { t } ^ { \overline { { \mathbf { \Lambda } } } }$ (eq. (16)). Verifying it directly would require evaluating $\bar { \langle { u } _ { 1 } ^ { \star } , { \Phi ( \tilde { x } ) } \rangle }$ for every $\tilde { x } \in \mathcal { D } _ { 2 }$ , which is computationally prohibitive given the scale of $\Phi ( { \tilde { x } } ) \in \mathbb { R } ^ { p }$ with $\bar { p } > 1 0 ^ { 8 }$ . We therefore aggregate the per-triplet quantity into a single scalar summary by introducing the mean feature vector for task t:

$$
V _ { t } : = \frac { 1 } { | \mathcal { D } _ { t } | } \sum _ { \tilde { x } \in \mathcal { D } _ { t } } \Phi ( \tilde { x } ) .\tag{18}
$$

On this aggregated quantity, we evaluate two measures of cross-task overlap between $V _ { 1 }$ and $V _ { 2 } { \mathrm { : } }$

• Top-1% Jaccard overlap: the Jaccard coefficient between the index sets of the top 1% largest-magnitude entries in each mean feature vector. This captures the degree of overlap between the most active parameters of the two tasks—the structural condition under which the per-triplet inner product $\langle u _ { 1 } ^ { \star } , \Phi ( \tilde { x } ) \rangle$ is small (section 3.3).

• Interference $\langle u _ { 1 } ^ { \star } , V _ { 2 } \rangle$ : the inner product between the DPO update direction for $\mathcal { T } _ { 1 }$ and the mean feature vector for $\mathcal { T } _ { 2 }$ . This is the average of $\langle u _ { 1 } ^ { \star } , \Phi ( \tilde { x } ) \rangle$ over $\tilde { x } \in \mathcal { D } _ { 2 }$ and serves as a tractable summary of the per-triplet disentanglement condition eq. (16).

Results. Table 3 reports the results. DPO suppresses cross-task interference by an order of magnitude or more in every configuration. Two qualitative patterns are notable: under DPO, both the Jaccard overlap and the interference decrease at the larger scale $( \mathbf { e . g . , 0 . 2 5 8 }  0 . 1 9 1$ on the withindataset pair and $1 . 9 1  0 . 8 5 2$ on the cross-dataset pair), consistent with the sharper polarization of $A _ { i }$ observed in section 4.2; and the same pattern holds for the cross-dataset pair (HH helpful UltraFeedback), demonstrating that the effect generalizes beyond subsets of a single dataset to entirely independent preference data sources. These observations empirically support the framework of section 3.3: DPO’s dual sparsity, combined with the small parameter-space overlap exhibited by semantically distinct preference tasks, yields cross-task inner products an order of magnitude smaller than those under SFT.

Model merging. To examine whether reduced interference carries over to merged models, we merge task vectors obtained by DPO or SFT on each task pair as $\theta _ { \mathrm { m e r g e d } } = \theta _ { 0 } + \alpha ( \bar { \tau } _ { \mathrm { t a r g e t } } + \tau _ { \mathrm { o t h e r } } )$ $\alpha \in [ 0 , 2 ]$ and measure the degradation $\Delta$ of each task’s performance relative to its single-task model (section D). As shown in table 4, DPO yields a smaller $\bar { \Delta }$ than SFT for both tasks in both task pairs, consistent with the reduced interference in table 3.

## 5 Related Work

Optimization dynamics of DPO. Since Rafailov et al. [2023] introduced DPO as a stable alternative to reinforcement learning from human feedback, several works have analyzed its dynamics theoretically. One line modifies the loss to address specific failure modes—IPO [Gheshlaghi Azar et al., 2024] avoids the Bradley–Terry assumption, and KTO [Ethayarajh et al., 2024] adopts a prospect-theoretic utility—but these are orthogonal to our analysis, which concerns the original objective. Closer to our approach, Ren and Sutherland [2025] linearize DPO under the NTK regime and use the resulting framework to characterize the squeezing effect, a counter-intuitive decrease in the probability of preferred responses also studied by Razin et al. [2025] under the term likelihood displacement. We adopt the same NTK lens but extract a structurally distinct property: the direction of the update in parameter space. Combining the classical implicit-bias result of Soudry et al. [2018] for separable logistic regression with the specific geometry of preference-pair gradients yields dual sparsity (theorem 3), which in turn provides a theoretical explanation for the small-subnetwork behav ior observed empirically by Mukherjee et al. [2025] on RL-based alignment. Likelihood displacement and dual sparsity are thus two complementary consequences of the same dynamics: one concerning the probabilities the update changes, the other the parameters it changes.

Table 4: Performance degradation ∆ (single-task − merged; lower is better) under task-vector merging on GPT-2 Medium. Task 1 is HH helpful; Task 2 is HH harmless or UltraFeedback. Full results are reported in section D.
<table><tr><td>Task Pair</td><td>Method</td><td>∆ Task 1</td><td>∆ Task 2</td></tr><tr><td rowspan="2">HH help↔harm</td><td>SFT</td><td>0.110</td><td>0.102</td></tr><tr><td>DPO</td><td>0.027</td><td>0.026</td></tr><tr><td rowspan="2">HH help↔UF</td><td>SFT</td><td>0.096</td><td>0.030</td></tr><tr><td>DPO</td><td>0.036</td><td>0.014</td></tr></table>

Task arithmetic, weight disentanglement, and model merging. Ilharco et al. [2023] showed that linear arithmetic on task vectors compositionally steers model behavior, and Ortiz-Jiménez et al. [2023] formalized weight disentanglement as a sufficient condition for this property, identifying via NTK tangent-space analysis that task vectors should act on disjoint regions of parameter space. A complementary line of work—DARE [Yu et al., 2024] and TIES-Merging [Yadav et al., 2023]— engineers sparsity into task vectors as a post-hoc intervention to suppress interference during merging. Our framework offers a different perspective: under DPO, sparsity emerges intrinsically from the optimization dynamics, and the resulting parameter-space overlap between semantically distinct preference tasks is small enough that the disentanglement condition holds approximately without any explicit intervention (section 4.4). Empirically, merging independently preference-optimized models has been reported to be competitive with or superior to data mixing, both for balancing safety and general-purpose capabilities [Aakanksha et al., 2024] and for balancing helpfulness, honesty, and harmlessness [Yang et al., 2025]. Our analysis provides theoretical support for these empirical findings, attributing the effectiveness of merging DPO-trained models to the dual sparsity of their updates and the resulting small cross-task interference. Our merging experiments (section 4.4) show that this translates into smaller performance degradation than SFT under task-vector merging; a larger-scale study of merging is left to future work.

## 6 Conclusion

We have presented a theoretical analysis of why DPO induces sparse parameter updates. Under the NTK linearization, the logistic structure of the DPO loss drives the update direction toward a maximum-margin solution that exhibits dual sparsity: it is determined by a small number of support pairs, and the gradient feature vector of each pair is itself sparse due to gradient cancellation between preferred and dispreferred responses. Relating this structure to weight disentanglement, we showed empirically that DPO reduces cross-task interference by 11× to 35× relative to SFT and degrades less under model merging, and that both sources of sparsity hold from GPT-2 to 7–9B models.

Limitations. Our analysis relies on the NTK linearization, which is not guaranteed to hold throughout DPO training of large models. More fundamentally, parameter-level sparsity (assumption 2) is an empirically motivated assumption rather than a consequence derived from the architecture. It relies on token overlap between the preferred and dispreferred responses: for pairs whose responses are entirely disjoint (e.g., different topics or formats), the gradient cancellation does not occur, and this source of sparsity may weaken or vanish. A formal derivation of assumption 2, and a characterization of the degree of overlap it requires in practice, are left to future work.

## References

Aakanksha, Arash Ahmadian, Seraphina Goldfarb-Tarrant, Beyza Ermis, Marzieh Fadaee, and Sara Hooker. Mix data or merge models? optimizing for diverse multi-task learning. arXiv preprint arXiv:2410.10801, 2024.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DaSilva, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, et al. Ultrafeedback: Boosting language models with scaled ai feedback. In International Conference on Machine Learning, pages 9722–9744. PMLR, 2024.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Model alignment as prospect theoretic optimization. In International Conference on Machine Learning, 2024.

Gemma Team. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, 2024.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In International Conference on Learning Representations, 2023.

Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, volume 31, pages 8571–8580, 2018.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al. Mistral 7B. arXiv preprint arXiv:2310.06825, 2023.

Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, and Hao Peng. Reinforcement learning finetunes small subnetworks in large language models. In Advances in Neural Information Processing Systems, 2025.

Guillermo Ortiz-Jiménez, Alessandro Favero, and Pascal Frossard. Task arithmetic in the tangent space: Improved editing of pre-trained models. In Advances in Neural Information Processing Systems, volume 36, pages 66727–66754, 2023.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI Blog, 2019.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pages 53728–53741, 2023.

Noam Razin, Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, and Boris Hanin. Unintentional unalignment: Likelihood displacement in direct preference optimization. In International Conference on Learning Representations, 2025.

Yi Ren and Danica J. Sutherland. Learning dynamics of LLM finetuning. In International Conference on Learning Representations, 2025.

Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. Journal of Machine Learning Research, 19(70):1–57, 2018.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. TIES-Merging: Resolving interference when merging models. In Advances in Neural Information Processing Systems, volume 36, 2023.

Jinluan Yang, Dingnan Jin, Anke Tang, Li Shen, Didi Zhu, Zhengyu Chen, Ziyu Zhao, Daixin Wang, Qing Cui, Zhiqiang Zhang, Jun Zhou, Fei Wu, and Kun Kuang. Mix data or merge models? balancing the helpfulness, honesty, and harmlessness of large language model via model merging. In Advances in Neural Information Processing Systems, volume 38, pages 124703–124732, 2025.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In Proceedings ofthe 41st International Conference on Machine Learning, 2024.

## A Proof and Numerical Evaluation of Lemma 2

## A.1 Proof

Throughout, we work with the frozen-NTK surrogate of the margin, $m _ { i } ^ { \operatorname * { l i n } } ( \theta ) : = m _ { i } ( \theta _ { 0 } ) + \langle \Phi _ { i } , \theta -$ $\theta _ { 0 } \rangle$ ⟩, under which the DPO loss reduces exactly to the logistic regression eq. $( 7 ) ;$ lemma $^ 2$ holds exactly for this model. Under $\pi _ { \theta _ { 0 } } ~ = ~ \pi _ { \mathrm { r e f } }$ , the offsets vanish, $b _ { i } = m _ { i } ( \theta _ { 0 } ) = 0 .$ so that $u ^ { \star }$ solves $\begin{array} { r } { \operatorname* { m a x } _ { \| u \| \leq 1 } \operatorname* { m i n } _ { i \in [ n ] } \langle u , \Phi _ { i } \rangle } \end{array}$ . By assumption 1, the optimal value γ is positive, and hence the constraint is active: $\| u ^ { \star } \| = 1$

KKT conditions. The problem is equivalent to the hard-margin problem $w ^ { \star } = \arg \operatorname* { m i n } _ { w } \frac { 1 } { 2 } \| w \| ^ { 2 }$ subject to $\langle w , \Phi _ { i } \rangle \geq 1$ for all i, with $\bar { u } ^ { \star } = w ^ { \star } / \| w ^ { \star } \|$ and $\gamma _ { \star } \doteq \bar { 1 / \| } w ^ { \star } \|$ . Its KKT conditions give $\begin{array} { r } { w ^ { \star } = \sum _ { i } \dot { \hat { \alpha } } _ { i } \Phi _ { i } } \end{array}$ with $\hat { \alpha } _ { i } \geq 0$ and $\hat { \alpha } _ { i } ( \langle w ^ { \star } , \Phi _ { i } \rangle \stackrel {  } { - } 1 ) \stackrel {  } { = } 0$ . Thus $\alpha _ { i } ^ { \star } = \hat { \alpha } _ { i } / \| w ^ { \star } \|$ in eq. (10), and complementary slackness yields

$$
\langle u ^ { \star } , \Phi _ { i } \rangle = \gamma _ { \star } \qquad \mathrm { f o r ~ a l l } ~ i \in \mathrm { S V } .\tag{19}
$$

Bound. Let $S : = \mathrm { S V } , s : = | S |$ , and $\mathbf { 1 } _ { S } \in \mathbb { R } ^ { s }$ the all-ones vector. Summing eq. (19) over $S$ and applying the Cauchy–Schwarz inequality with $\| u ^ { \star } \| = 1$

$$
s \gamma _ { \star } = \left. u ^ { \star } , \sum _ { i \in S } \Phi _ { i } \right. \leq \left\| \sum _ { i \in S } \Phi _ { i } \right\| = \left( \mathbf { 1 } _ { S } ^ { \top } K _ { S , S } \mathbf { 1 } _ { S } \right) ^ { 1 / 2 } .\tag{20}
$$

By the Rayleigh quotient, $\mathbf { 1 } _ { S } ^ { \top } K _ { S , S } \mathbf { 1 } _ { S } \le \lambda _ { \operatorname* { m a x } } ( K _ { S , S } ) \| \mathbf { 1 } _ { S } \| ^ { 2 } = s \lambda _ { \operatorname* { m a x } } ( K _ { S , S } )$ . Combining the two, $s ^ { \bar { 2 } } \gamma _ { \star } ^ { 2 } \leq s \bar { \lambda } _ { \operatorname* { m a x } } ( \bar { K _ { S , S } } )$ , i. $\mathrm { . e . , } s \gamma _ { \star } ^ { 2 } \leq \lambda _ { \operatorname* { m a x } } ( K _ { S , S } )$ . Since $K _ { S , S }$ is a principal submatrix of the positive semidefinite matrix K, Cauchy’s interlacing theorem gives $\lambda _ { \operatorname* { m a x } } ( \mathbf { \bar { \xi } } _ { K _ { S , S } } ) \leq \lambda _ { \operatorname* { m a x } } ( K )$ . Finally, since s is an integer with $s \leq n$ , we obtain $s \leq \operatorname* { m i n } \{ n , \lfloor \lambda _ { \operatorname* { m a x } } ( K ) / \gamma _ { \star } ^ { 2 } \rfloor \}$ □

## A.2 Numerical Evaluation

Computing $\gamma _ { \star }$ exactly requires solving the max-margin problem. We instead evaluate the bound through the finite-time margin of the normalized trajectory, which is justified by the following corollary.

Corollary 4 (Finite-time evaluation of the bound). Under the assumptions of lemma 2, let $\Delta \theta ( T ) \neq 0$ and define thefinite-time margin

$$
\tilde { \gamma } _ { T } : = \operatorname* { m i n } _ { i \in [ n ] } \left. \frac { \Delta \theta ( T ) } { \lVert \Delta \theta ( T ) \rVert } , \Phi _ { i } \right. .\tag{21}
$$

Then $\tilde { \gamma } _ { T } \leq \gamma _ { \star } , \ I f ,$ moreover, $\Delta \theta ( T )$ follows the gradient flow of the frozen-NTK surrogate, then $\tilde { \gamma } _ { T }  \gamma ,$ <sub>⋆</sub> as $T \to \infty$ . Consequently, whenever $\tilde { \gamma } _ { T } > 0$

$$
\left| \mathrm { S V } \right| \leq \operatorname* { m i n } \left\{ n , \left\lfloor \frac { \lambda _ { \operatorname* { m a x } } ( K ) } { \gamma _ { \star } ^ { 2 } } \right\rfloor \right\} \leq \operatorname* { m i n } \left\{ n , \left\lfloor \frac { \lambda _ { \operatorname* { m a x } } ( K ) } { \tilde { \gamma } _ { T } ^ { 2 } } \right\rfloor \right\} .\tag{22}
$$

Proof. Let $u _ { T } : = \Delta \theta ( T ) / \| \Delta \theta ( T ) \|$ . Since $\| u _ { T } \| = 1 , u _ { T }$ is feasible for the max-margin problem, and hence $\begin{array} { r } { \tilde { \gamma } _ { T } = \operatorname* { m i n } _ { i } \langle u _ { T } , \Phi _ { i } \rangle \leq \operatorname* { m a x } _ { \| u \| \leq 1 } } \end{array}$ min<sub>i</sub> $\langle u , \Phi _ { i } \rangle = \gamma _ { \star }$ Under the frozen-NTK gradient flow, lemma 1 gives $u _ { T } \to u ^ { \star } ;$ ; since $u \mapsto \operatorname* { m i n } _ { i } \langle u , \Phi _ { i } \rangle$ is continuous as the minimum of finitely many continuous functions, $\tilde { \gamma } _ { T }  \mathrm { m i n } _ { i } \langle u ^ { \star } , \Phi _ { i } \rangle = \gamma _ { \star }$ . Finally, $\mathrm { i f } \tilde { \gamma } _ { T } > 0$ , then $1 / \gamma _ { \star } ^ { 2 } \le 1 / \tilde { \gamma } _ { T } ^ { 2 }$ , which, combined with eq. (11), $| \mathrm { S V } | \leq n$ , and the integrality of |SV|, yields the chain of inequalities.

The values reported below are computed from the actual training trajectory of the network rather than from the frozen-NTK gradient flow. For such a trajectory, the convergence statement of corollary 4 does not apply, but the inequality $\tilde { \gamma } _ { T } \leq \gamma ,$ <sub>⋆</sub> still holds, since it uses only $\| u _ { T } \| = 1 ;$ the reported ratio $\lambda _ { \operatorname* { m a x } } \dot { ( } \dot { K ) } / \tilde { \gamma } _ { T } ^ { 2 }$ is therefore a conservative (upper) estimate of $\lambda _ { \operatorname* { m a x } } ( K ) \ddot { / } \gamma _ { \star } ^ { 2 }$ . Table 5 reports this quantity in the setting of fig. 1 (GPT-2 Small/Medium, HH-RLHF, $n = 1 0 0 )$ . For both models it is about $1 / 7$ to $1 / 9$ of n, guaranteeing that |SV| is roughly an order of magnitude smaller than $n ,$ consistent with the concentration observed in section 4.2.

Table 5: Conservative evaluation of the support-set bound eq. (11) in the setting of fig. $1 ( n = 1 0 0 )$
<table><tr><td>Model</td><td> $\lambda _ { \operatorname* { m a x } } ( K ) / \tilde { \gamma } _ { T } ^ { 2 }$ </td><td>Implied bound on |SV|</td></tr><tr><td>GPT-2 Small</td><td>14.58</td><td>≤ 14</td></tr><tr><td>GPT-2 Medium</td><td>11.33</td><td>≤11</td></tr></table>

## B Experimental Details

Time-integrated weights. The weights $\begin{array} { r } { A _ { i } ( T ) = \int _ { 0 } ^ { T } \sigma ( - m _ { i } ( \theta ( t ) ) ) } \end{array}$ dt are approximated by accumulating the discrete-time analogue $\begin{array} { r } { \sum _ { t } \sigma ( - m _ { i } ( \theta _ { t } ) ) } \end{array}$ ) over training steps. Since A<sub>i</sub> depends only on the margins, it requires no gradient features.

Experimental settings. Table 6 summarizes the models, datasets, and numbers of preference pairs used in each experiment. The full fine-tuning DPO runs for the 7–8B models use up to eight NVIDIA A100 GPUs.

Table 6: Experimental settings. “Pairs” is the number of preference pairs on which each quantity is computed. HH: Anthropic HH-RLHF; UF: UltraFeedback Binarized; ep.: epochs.
<table><tr><td>Experiment</td><td>Models</td><td>Dataset</td><td>Pairs</td><td>Training</td></tr><tr><td>Data-level (fig. 1)</td><td>GPT-2 S/M</td><td>HH</td><td>100</td><td>DPO, 1000 ep.</td></tr><tr><td>Data-level (table 1)</td><td>GPT-2 S/M</td><td>HH</td><td>1000</td><td>DPO, 1000 ep.</td></tr><tr><td rowspan="3"></td><td>Llama-3-8B</td><td>UF</td><td>1000</td><td>Full-FT DPO, 3 ep.</td></tr><tr><td>Mistral-7B</td><td>UF</td><td>1000</td><td>Full-FT DPO, 3 ep.</td></tr><tr><td>GPT-2 M</td><td>HH</td><td>100</td><td></td></tr><tr><td>Parameter-level (fig. 2) Parameter-level (table 2)</td><td>All five</td><td>UF</td><td>1000</td><td></td></tr><tr><td>Interference (table 3)</td><td>GPT-2 S/M</td><td>HH, UF</td><td>100</td><td>DPO / SFT</td></tr><tr><td>Model merging (table 4)</td><td>GPT-2 M</td><td>HH, UF</td><td>section D</td><td>DPO / SFT</td></tr></table>

## C Evolution of Log-Probabilities During DPO Training

We record the log-probabilities of the preferred and dispreferred responses along the training trajectory that produces fig. 1 (GPT-2 Medium, HH-RLHF, $N = 1 0 0 )$ . Each value is the sequence-level logprobability averaged over the 100 pairs, with logps/chosen $: = \log \pi _ { \theta } ( y ^ { + } \mid x ) \quad$ and logps/rejected = log $\pi _ { \theta } ( y ^ { - } \mid x )$

As shown in table 7, both log-probabilities decrease, but the rejected log-probability decreases considerably faster, so the gap widens from 11.7 to 128.6. Since $\pi _ { \theta _ { 0 } } = \pi _ { \mathrm { r e f } }$ , the margin satisfies $m _ { i } ( \theta ) = \beta \bar { ( \mathrm { g a p } _ { i } ( \theta ) - \mathrm { g a p } _ { i } ( \bar { \theta _ { 0 } } ) ) }$ ), where gap denotes the per-pair log-probability gap; a widening gap thus directly increases the margin. As a result, $\sigma ( - m _ { i } )  0$ for easy pairs, which stops their contribution to $A _ { i }$ and produces the polarization of $A _ { i }$ observed in fig. 1.

Table 7: Average log-probabilities of preferred (chosen) and dispreferred (rejected) responses during DPO training (GPT-2 Medium, HH-RLHF, N = 100).
<table><tr><td>Epoch</td><td>logps/chosen</td><td>logps/rejected</td><td>Gap (chosen – rejected)</td></tr><tr><td>1</td><td>-247.2</td><td>-258.9</td><td>11.7</td></tr><tr><td>100</td><td>-280.5</td><td>-334.5</td><td>54.0</td></tr><tr><td>1000</td><td>-308.3</td><td>-436.9</td><td>128.6</td></tr></table>

## D Model Merging Experiments

Setup. For each task, we fine-tune GPT-2 Medium with DPO or SFT until convergence and take the task vector $\tau = \theta ^ { \star } - \theta _ { 0 }$ from the best checkpoint. For each target task, we form $\theta _ { \mathrm { { m e r g e d } } } =$ $\theta _ { 0 } + \alpha \left( \tau _ { \mathrm { t a r g e t } } + \tau _ { \mathrm { o t h e r } } \right)$ , where $\alpha \in \{ 0 , 0 . 1 , \ldots , 2 . 0 \}$ is selected on a validation set, and report $\Delta = ( \mathrm { s i n g l e \mathrm { - } t a s k } ) - ( \mathrm { m e r g e d } )$ . We use the same two task pairs as in section 4.4.

Results. Table 8 reports the single-task and merged performance. On both task pairs, DPO degrades less than SFT under merging. The gap between the two methods is smaller than the order-ofmagnitude reduction in the cross-task inner product (table 3), since merged performance also depends on factors other than parameter-space overlap.

Table 8: Single-task and merged performance of GPT-2 under task-vector merging. ∆ is the singletask minus merged performance (lower is better).
<table><tr><td colspan="2"></td><td colspan="2">Single</td><td colspan="2">Merged</td><td colspan="2"> $\Delta$ </td></tr><tr><td>Task Pair</td><td>Method</td><td>Task 1</td><td>Task 2</td><td>Task 1</td><td>Task 2</td><td>Task 1</td><td>Task 2</td></tr><tr><td rowspan="2">HH help↔harm</td><td>SFT</td><td>0.631</td><td>0.625</td><td>0.521</td><td>0.523</td><td>0.110</td><td>0.102</td></tr><tr><td>DPO</td><td>0.668</td><td>0.645</td><td>0.641</td><td>0.619</td><td>0.027</td><td>0.026</td></tr><tr><td rowspan="2">HH help↔UF</td><td>SFT</td><td>0.631</td><td>0.535</td><td>0.518</td><td>0.505</td><td>0.096</td><td>0.030</td></tr><tr><td>DPO</td><td>0.668</td><td>0.572</td><td>0.632</td><td>0.558</td><td>0.036</td><td>0.014</td></tr></table>

## E Licenses

Table 9 lists the existing assets used in this paper together with their licenses and the versions (Hugging Face identifiers) we use.

Table 9: Existing assets, versions, and licenses.
<table><tr><td>Asset</td><td>Version (Hugging Face)</td><td>License</td></tr><tr><td>GPT-2 Small</td><td>openai-community/gpt2</td><td>Modified MIT</td></tr><tr><td>GPT-2 Medium</td><td>openai-community/gpt2-medium</td><td>Modified MIT</td></tr><tr><td>Llama-3-8B-Instruct</td><td>meta-1lama/Meta-Llama-3-8B-Instruct</td><td>Llama 3 Community License</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>mistralai/Mistral-7B-Instruct-vO.2</td><td>Apache 2.0</td></tr><tr><td>Gemma-2-9B-it</td><td>google/gemma-2-9b-it</td><td>Gemma Terms of Use</td></tr><tr><td>Anthropic HH-RLHF</td><td>Anthropic/hh-rlhf</td><td>MIT</td></tr><tr><td>UltraFeedback Binarized</td><td>HuggingFaceH4/ultrafeedback_binarized</td><td>MIT</td></tr></table>