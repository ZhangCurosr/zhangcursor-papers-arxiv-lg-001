# Towards a Statistical Understanding of Mixture-of-Experts

Siyuan He<sup>a</sup>, Bokai Yang<sup>∗a</sup>, Jie Hu<sup>b</sup>, Ziwen Gao<sup>c</sup>, and Yuhong Yang<sup>†b,d</sup>

<sup>a</sup>Qiuzhen College, Tsinghua University, Beijing, China.

<sup>b</sup>Yau Mathematical Sciences Center, Tsinghua University, Beijing, China. <sup>c</sup>KLATASDS-MOE, School of Statistics, East China Normal University, Shanghai, China.

<sup>d</sup>Beijing Institute of Mathematical Sciences and Applications (BIMSA), Beijing, China.

September 4, 2026

## Abstract

Mixture-of-experts (MoE) architectures increase model capacity by combining a collection of expert predictors through input-dependent routing, while often activating only a small subset of experts for each input. Despite their growing importance in modern large-scale models, the statistical roles of their design choices, especially routing, sparse activation, and shared experts, remain only partially understood, as existing theory has largely focused on parametric or correctly specified MoE models. In this paper, we view MoE as a form of localized aggregation and show how this localization reshapes the approximation-estimation-computation tradeof. We derive oracle risk bounds for learning dense and sparse routing with evolving experts, separating approximation, expertlearning, and router-estimation errors, and characterize how sparse Top-K routing can retain the benefits of localized aggregation while controlling per-input computation. We also interpret gating through the geometry of input space, relating routing performance to regions of local expert advantage, and show how shared experts, as adopted in architectures such as DeepSeekMoE, can extract common predictive structure so that routed experts focus on residual local variation. Together, these results provide a unified statistical framework for understanding MoE through input-dependent expert aggregation, in which expert specialization and computational tradeofs are governed by local predictive structure.

Keywords: expert specialization; localized aggregation; mixture of expert architectures; routing mechanisms; shared experts; sparse activation

## 1 Introduction

Modern artificial intelligence systems increasingly rely on architectures whose total capacity greatly exceeds the computation activated for any individual input (Brown et al., 2020; OpenAI et al., 2024; Rombach et al., 2022; Vaswani et al., 2017). A prominent example is the mixture-of-experts (MoE) architecture, in which a router assigns each input to a subset of expert networks and combines their outputs in an input-dependent manner. In transformer-based systems, MoE layers are often used as replacements for feed-forward blocks, as in GShard, Switch Transformer, GLaM, and Mixtral (Du et al., 2022; Fedus et al., 2022; Jiang et al., 2024; Lepikhin et al., 2021; Shazeer et al., 2017). These models use sparse expert activation to increase model capacity while keeping per-token computation manageable. More recent designs, such as DeepSeekMoE (Dai et al., 2024), also include shared experts that are always active and are intended to capture common knowledge across inputs. These developments suggest that the statistical roles of routing, sparsity, and shared components are central to understanding modern MoE systems.

This paper studies MoE architectures from the viewpoint of covariate-dependent aggregation. In this view, the expert networks provide a collection of candidate predictors, while the router determines how these predictors are locally combined. Dense softmax gating corresponds to smooth input-dependent averaging over all experts, while sparse Top-K routing performs localized selection and encourages expert specialization. Shared experts add a global component that is reused across the input space. Together, these mechanisms provide diferent ways to balance global structure, local specialization, and computational constraints.

The statistical study of MoE models has a long history. Classical works introduced MoE and hierarchical MoE architectures together with likelihood-based and EM-type estimation procedures (Jacobs et al., 1991; Jordan and Jacobs, 1994). Subsequent theory studied identifiability, convergence and asymptotic inference, often under parametric or correctly specified formulations (Jiang and Tanner, 1999b; Jordan and Xu, 1995; Olteanu and Rynkiewicz, 2011). More recent work has analyzed estimation rates for modern variants of MoE models, including dense softmax gating, sparse routing and architectures with shared experts (Nguyen et al., 2024a, 2025, 2024b). On the approximation side, classical and recent results show that softmax-gated MoE classes can have strong universal approximation properties (Jiang and Tanner, 1999a; Nguyen et al., 2016).

Despite these advances, there remains a gap between existing theory and the structural questions raised by modern MoE architectures. Much of the available theory treats the target function as belonging to a finite-dimensional MoE class and focuses on estimation of all model parameters. This perspective is less directly suited to large-scale systems, where the experts and router are highly overparameterized, the model is typically misspecified, and end-to-end optimization is non-convex. Moreover, existing results do not fully separate the statistical roles of dense averaging, sparse routing, adaptive partitioning and shared experts. In particular, it remains unclear how these architectural choices shape approximation, estimation, and computation outside a fully specified joint parametric MoE model.

Recent empirical MoE studies make these structural questions particularly salient. Largescale sparse MoE models increase the total number of experts while activating only a smal subset for each input (Du et al., 2022; Fedus et al., 2022; Jiang et al., 2024; Lepikhin et al., 2021; Shazeer et al., 2017). More recent architectures further refine this idea by using finegrained routed experts, multiple active experts, or shared experts (Dai et al., 2024; He, 2024; Ludziejewski et al., 2024; Yun et al., 2024). These design choices are often motivated by empirical eficiency and scalability considerations. They also raise an important statistical question: how do the size and composition of the expert pool, the sparsity and flexibility of the routing mechanism, and the inclusion of shared components jointly shape prediction accuracy, statistical complexity, and computational cost?

Addressing this question is challenging because joint optimization of experts and routers in large-scale MoE models is highly coupled and non-convex, making it dificult to disentangle the statistical contributions of diferent architectural components. We therefore base our analysis on a predictable expert-training trajectory, which provides a tractable way to study router learning without directly characterizing the full end-to-end training dynamics. Conditional on the past data, the current experts enter the next-step prediction problem as plug-in components, while their quality is allowed to evolve over time. Under this formulation, we develop a unified statistical framework that treats MoE models as flexible aggregation rules built from a collection of expert predictors and a gating mechanism. This framework then provides a common ground for comparing dense softmax routing, sparse Top-K routing, adaptive gating classes and shared-expert architectures in terms of their approximation power, statistical complexity and computational tradeofs.

Our first contribution is to clarify the approximation role of routing. We show that input-dependent routing turns a collection of experts into a localized aggregation class, which can be substantially more expressive than global averaging. Dense and sparse routing difer not only computationally but also statistically: dense softmax gating provides smooth averaging, while sparse Top-K gating induces localized expert selection. We characterize how these mechanisms afect approximation ability and complexity, and show when localization, through either dense or Top-K aggregation, can improve upon global averaging.

Our second contribution is to develop oracle risk bounds for estimating the gating mechanism along an evolving expert trajectory. We use a discretization-based scheme (Yang, 2001, 2004) that avoids assuming global optimization of a highly non-convex MoE objective. The resulting bounds separate the oracle approximation risk, the cumulative learning error of the evolving experts, and the statistical cost of learning the router. The framework applies to both dense and sparse routing and highlights the estimation price of routing flexibility.

Our third contribution is to interpret gating as a problem of input-space partitioning. Diferent routing functions induce diferent soft or hard partitions of the input space, and their performance depends on whether these partitions align with the regions on which diferent experts perform well. Linear, quadratic and kernel-based gating functions therefore represent diferent levels of geometric flexibility. This perspective explains why a single routing class may be efective for some target structures but inadequate for others, and motivates data-adaptive selection among gating mechanisms.

Our final contribution is to study the statistical role of shared experts. We show that shared experts are not merely additional predictors in the ensemble. Their main role is to extract common structure that would otherwise have to be repeatedly learned by routed experts. Once this common component is removed, the routed experts only need to learn residual local structure, which can reduce the efective complexity of the routed learning problem. This provides a statistical interpretation of recent MoE designs that combine shared and routed components.

The rest of the paper is organized as follows. Section 1.1 introduces the mathematical formulation of dense routing, sparse routing and shared experts. Section 2 studies the approximation properties of MoE predictors as localized aggregation rules. Sections 3 and

4 develop risk bounds for dense and Top-K gating estimation. Section 5 studies gating as input-space partitioning and discusses adaptive selection of routing classes. Section 6 analyzes the role of shared experts. Technical proofs and additional discussions are provided in the supplementary material.

Notation. Throughout the paper, for two nonnegative sequences $a _ { n }$ and $b _ { n } .$ , we write $a _ { n } \lesssim b _ { n }$ if there exists a constant $C > 0$ , independent of $n ,$ , such that $a _ { n } \leq C b _ { n }$ , and write $a _ { n } \asymp b _ { n }$ if both $a _ { n } \lesssim b _ { n }$ and $b _ { n } \lesssim a _ { n }$ hold. For two real numbers a and $b ,$ we write $a \vee b = \operatorname* { m a x } \{ a , b \}$ and $a \wedge b = \operatorname* { m i n } \{ a , b \}$ . For a positive integer $M , \left[ M \right] = \left\{ 1 , \ldots , M \right\}$ . For a finite set S, |S| denotes its cardinality. The set $\Delta ^ { M - 1 } = \{ w \in \mathbb { R } ^ { M } : w _ { j } \geq 0 , ~ \textstyle \sum _ { j = 1 } ^ { M } w _ { j } = 1 \}$ denotes the probability simplex in $\mathbb { R } ^ { M }$ . For a set $\mathcal { X } , \ \mathbb { 1 } _ { \mathcal { X } } ( x )$ denotes the indicator function taking the value one when $x \in \mathcal { X }$ and zero otherwise. For two vectors a and $b , \langle a , b \rangle = a ^ { \top } b$ denotes their inner product. Unless otherwise specified, $\| \cdot \|$ denotes the Euclidean norm for vectors and the spectral norm for matrices. For a diferentiable function $f , \nabla f$ denotes its gradient. For a measurable function $f = ( f _ { 1 } , \ldots , f _ { M } ) ^ { \top } : \mathcal { X } \to \mathbb { R } ^ { M }$ , with the scalar case corresponding to $M = 1$ , define $\begin{array} { r } { \| f \| _ { L _ { 2 } ( P _ { X } ) } = \left( \int _ { \mathcal { X } } \sum _ { m = 1 } ^ { M } f _ { m } ^ { 2 } ( x ) d P _ { X } ( x ) \right) ^ { 1 / 2 } } \end{array}$ . For a scalarvalued $f ,$ we also write $\| f \| _ { \infty } = \operatorname* { s u p } _ { x \in \mathcal { X } } | f ( x )$ |. Finally, for $x \in \mathbb { R } , \lfloor x \rfloor$ denotes the largest integer not exceeding x, and ⌈x⌉ denotes the smallest integer not smaller than x.

## 1.1 Mathematical formulation of MoE architecture

We begin with a routed-only MoE layer, which serves as the baseline formulation before shared experts are introduced. Let $\mathcal { X } \subset \mathbb { R } ^ { d }$ be a compact input space and let $x \in \mathcal { X }$ denote an input. The layer contains $M _ { R }$ routed experts. For $m \in [ M _ { R } ]$ , we write $f _ { R , m } ( x ; W _ { R , m } )$ for the output of the m-th routed expert, where $W _ { R , m }$ denotes its network parameter. In transformer implementations, such experts are typically feed-forward networks, although the statistical analysis below treats them only as candidate predictors.

A router maps the input x to a vector of gating weights

$$
g ( x ; \theta ) = ( g _ { 1 } ( x ; \theta ) , \ldots , g _ { M _ { R } } ( x ; \theta ) ) ^ { \top } \in \Delta ^ { M _ { R } - 1 } ,
$$

where $\theta$ is the gating parameter. The output of the conventional MoE (Jordan and Jacobs,

1994) is then

$$
F _ { \mathrm { M o E } } ( x ; W _ { R } , \theta ) = \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( x ; \theta ) f _ { R , m } ( x ; W _ { R , m } ) ,\tag{1.1}
$$

where $W _ { R } = \left( W _ { R , 1 } , \ldots , W _ { R , M _ { R } } \right)$ . Throughout the paper, we use router for the architectural module that assigns inputs to experts, and gating function for its mathematical representation $x \mapsto g ( x ; \theta )$ . Experts weighted or selected by the router are called routed experts. Experts that are always active, and hence do not depend on the router, are called shared experts.

Many MoE gating functions are constructed by first computing unnormalized routing scores and then normalizing them. Let $s _ { m } ( x ; \theta _ { m } )$ be the score assigned to the m-th routed expert, where $\theta _ { m }$ is the expert-specific score parameter. We also write $\boldsymbol { \theta } = ( \theta _ { 1 } ^ { \top } , \ldots , \theta _ { M _ { R } } ^ { \top } ) ^ { \top }$ We use two score classes as running examples. The first is the linear score,

$$
\begin{array} { r } { s _ { m } ^ { \mathrm { l i n } } ( x ; \theta _ { m } ) = \beta _ { m } ^ { \top } x + \alpha _ { m } , \qquad \theta _ { m } = ( \beta _ { m } ^ { \top } , \alpha _ { m } ) ^ { \top } \in \mathbb { R } ^ { d + 1 } , } \end{array}
$$

which induces routing regions separated by afine decision boundaries. The second is the quadratic score,

$$
\begin{array} { r } { s _ { m } ^ { \mathrm { q u a d } } ( \boldsymbol { x } ; \theta _ { m } ) = \boldsymbol { x } ^ { \top } B _ { m } \boldsymbol { x } + \beta _ { m } ^ { \top } \boldsymbol { x } + \alpha _ { m } , \qquad B _ { m } = B _ { m } ^ { \top } , } \end{array}
$$

where $\theta _ { m } = ( \mathrm { v e c h } ( B _ { m } ) ^ { \top } , \beta _ { m } ^ { \top } , \alpha _ { m } ) ^ { \top }$ , and $\operatorname { v e c h } ( B _ { m } )$ vectorizes the upper triangular part of the symmetric matrix $B _ { m }$ . The quadratic score allows curved decision boundaries. These two score classes will later be used to illustrate how gating geometry afects the input-space partition induced by the router.

Given scores $s _ { 1 } , \ldots , s _ { M _ { R } }$ , dense softmax gating assigns positive weight to every routed expert:

$$
g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) = \frac { \exp \{ s _ { m } ( x ; \theta _ { m } ) \} } { \sum _ { j = 1 } ^ { M _ { R } } \exp \{ s _ { j } ( x ; \theta _ { j } ) \} } , \qquad m = 1 , \ldots , M _ { R } .\tag{1.2}
$$

Sparse Top-K gating instead activates only the K experts with the largest scores, where $1 \le K \le M _ { R }$ . Let $\mathcal { T } _ { K } ( \boldsymbol { x } ; \boldsymbol { \theta } )$ denote the resulting set of exactly K selected indices. The

Top-K softmax gate is

$$
g _ { m } ^ { ( K ) } ( x ; \theta ) = \frac { \exp \{ s _ { m } ( x ; \theta _ { m } ) \} \mathbb { 1 } \{ m \in \mathcal { T } _ { K } ( x ; \theta ) \} } { \sum _ { j \in \mathcal { T } _ { K } ( x ; \theta ) } \exp \{ s _ { j } ( x ; \theta _ { j } ) \} } , \qquad m = 1 , \dots , M _ { R } .\tag{1.3}
$$

Thus dense softmax gating and Top-K gating represent two diferent forms of covariatedependent aggregation. Dense gating gives smooth averaging over the full expert collection, whereas Top-K gating gives a piecewise-defined rule whose active expert set changes across the input space. This partition viewpoint will be central to the approximation and estimation analyses below.

We next incorporate shared experts. Motivated by recent MoE architectures such as DeepSeekMoE (Dai et al., 2024), we consider a layer with $M _ { S }$ shared experts and $M _ { R }$ routed experts. The shared experts are always active, while the gating function is applied only to the routed experts. For input x, let $\{ f _ { S , \ell } ( x ; W _ { S , \ell } ) \} _ { \ell = 1 } ^ { M _ { S } }$ denote the shared expert outputs, and let $\{ f _ { R , m } ( x ; W _ { R , m } ) \} _ { m = 1 } ^ { M _ { R } }$ denote the routed expert outputs. The shared-routed MoE layer is

$$
F _ { \mathrm { s h a r e d } } ( x ; W _ { S } , W _ { R } , \theta ) = \sum _ { \ell = 1 } ^ { M _ { S } } f _ { S , \ell } ( x ; W _ { S , \ell } ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( x ; \theta ) f _ { R , m } ( x ; W _ { R , m } ) ,\tag{1.4}
$$

where $W _ { S } = ( W _ { S , 1 } , \ldots , W _ { S , M _ { S } } )$ . Combining this formulation with dense softmax or Top-K gating gives dense or sparse routing with shared experts. This decomposition separates an always-active shared component from an input-dependent routed component. Issues of identifiability, estimation and the statistical benefits of this separation are discussed in later sections.

## 2 Localized averaging in MoE

Building on the formulation in Section 1.1, we now examine the statistical role of inputdependent aggregation in MoE models. The key distinction is between global averaging, where the aggregation weights are fixed across the input space, and localized averaging, where the weights are generated by a gating function and therefore vary with the covariate value x. In modern MoE terminology, this localization is implemented by the router. Thus, the router determines not only which experts are used, but also how the prediction rule

varies across the input space.

Existing approximation results have established that input-dependent weighting generates rich approximation classes even when the experts are constants (Gao et al., 2026; Jiang and Tanner, 1999a; Nguyen et al., 2016; Zeevi et al., 1998). We take this constant expert setting as a starting point and first ask whether sparse Top-K routing retains universal approximation when only K experts are activated for each input. We then consider a fixed number of experts to characterize more explicitly the approximation ability and complexity induced by localized aggregation, before moving beyond constant experts to examine whether its expressive advantage persists for richer expert classes. Throughout this section, we consider routed experts without shared experts, write $M = M _ { R }$ and take $\mathcal { X } = [ 0 , 1 ] ^ { d }$

## 2.1 Constant experts: global versus localized averaging

Suppose that every expert belongs to the class of constant functions $\mathcal { F } _ { \mathrm { c } } : = \{ f : \mathcal { X }  \mathbb { R }$ $f ( x ) \equiv c , \ c \in \mathbb { R } \}$ . The corresponding global averaging class is

$$
\mathcal { A } _ { \mathrm { g l o b a l } } : = \left\{ \sum _ { m = 1 } ^ { M } w _ { m } f _ { m } \bigg | M \in \mathbb { N } _ { + } , \ f _ { m } \in \mathcal { F } _ { \mathrm { c } } , \ w _ { m } \geq 0 , \ \sum _ { m = 1 } ^ { M } w _ { m } = 1 \right\} .
$$

Since convex combinations of constant functions remain constant, $\mathcal { A } _ { \mathrm { g l o b a l } } = \mathcal { F } _ { \mathrm { c } }$ . In contrast, for softmax routing (1.2) with linear scores, define

$$
A _ { \mathrm { l o c a l } } ^ { \mathrm { s o f t } } : = \left\{ \sum _ { m = 1 } ^ { M } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { m } \ : \left| \begin{array} { l } { M \in \mathbb { N } _ { + } , \ f _ { m } \in \mathcal { F } _ { \mathrm { c } } , \ \theta _ { m } = ( \beta _ { m } ^ { \top } , \alpha _ { m } ) ^ { \top } \in \mathbb { R } ^ { d + 1 } , \ \theta = ( \theta _ { 1 } ^ { \top } , \ldots , \theta _ { M } ^ { \top } ) ^ { \top } } \end{array} \right. \right\} .
$$

Although the experts are constant, input-dependent softmax weighting induces a function class substantially richer than that generated by global averaging. The following existing result shows that this class can approximate any continuous function arbitrarily well on a compact input space.

Lemma 2.1 (Theorem 7 in (Nguyen et al., 2016)). For every compact $\chi , A _ { \mathrm { l o c a l } } ^ { \mathrm { s o f t } }$ is dense in $C ( \mathcal X )$ with respect to the uniform norm, where $C ( \mathcal X )$ denotes the class of all continuous real-valued functions on X.

In many modern large-scale MoE architectures, however, only the Top-K experts are activated at each input in order to control per-input computation. This widespread architectural choice motivates a corresponding approximation question: does activating only K of the M routed experts at each input fundamentally reduce the expressive power relative to dense softmax routing?

To study this question, we consider bounded constant experts and restrict the linear score parameters to a fixed box. Specifically, let

$$
\begin{array} { r } { \Theta _ { \mathrm { l i n } } ( C _ { 1 } ) : = \left\{ \vartheta = ( \beta ^ { \top } , \alpha ) ^ { \top } \in \mathbb { R } ^ { d + 1 } : \| \vartheta \| _ { \infty } \leq C _ { 1 } \right\} , \qquad C _ { 1 } > 0 . } \end{array}
$$

The bounded constant-expert class is defined by

$$
\mathcal F _ { \mathrm { c } } ^ { R } : = \{ f : \mathcal X \to \mathbb R : f ( x ) \equiv c , ~ | c | \leq R \} .
$$

For $1 \leq K \leq M$ , define the Top-K localized averaging class with M routed experts by

$$
\mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p \it - K } } ( M ) : = \left\{ \sum _ { m = 1 } ^ { M } g _ { m } ^ { ( K ) } ( x ; \theta ) f _ { m } ( x ) ~ \left| ~ f _ { m } \in \mathcal { F } _ { \mathrm { c } } ^ { R } , ~ \theta _ { m } \in \Theta _ { \mathrm { l i n } } ( C _ { 1 } ) , ~ \theta = ( \theta _ { 1 } ^ { \top } , \ldots , \theta _ { M } ^ { \top } ) ^ { \top } \right. \right\} ,
$$

where $g _ { m } ^ { ( K ) }$ is the Top-K softmax gate generated by the linear score $s _ { m } ^ { \mathrm { l i n } }$ . The full Top-K localized averaging class is

$$
\mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p - } K } : = \bigcup _ { M \geq K } \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p - } K } ( M ) .
$$

For comparison, the corresponding bounded global averaging class is

$$
\mathcal { A } _ { \mathrm { g l o b a l } } ^ { R } : = \left\{ \sum _ { m = 1 } ^ { M } w _ { m } f _ { m } \bigg | M \in \mathbb { N } _ { + } , \ f _ { m } \in \mathcal { F } _ { \mathrm { c } } ^ { R } , \ w _ { m } \geq 0 , \ \sum _ { m = 1 } ^ { M } w _ { m } = 1 \right\} = \mathcal { F } _ { \mathrm { c } } ^ { R } .
$$

The next proposition shows that fixing the per-input activation level K does not destroy universal approximation as the total number of routed experts increases: for every fixed K, the union over $M \geq K$ remains suficiently rich to approximate any bounded continuous target function.

Proposition 2.2. Let $\mathcal { X } = [ 0 , 1 ] ^ { d }$ and fix $K \in \mathbb { N } _ { + }$ . Then $\mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p } - K }$ is dense in

$$
C _ { R } ( \mathcal { X } ) : = \{ h \in C ( \mathcal { X } ) : \| h \| _ { \infty } \leq R \}
$$

with respect to the uniform norm, where $C ( \mathcal { X } )$ denotes the class of all continuous real-valued functions on $\mathcal { X }$

Proposition 2.2 shows that the expressive advantage of input-dependent routing over global averaging persists under Top-K activation: Top-K localized averaging of bounded constant experts remains dense in $C _ { R } ( \mathcal { X } )$ . For every fixed K, increasing M allows the router to generate increasingly fine local regions even though only K experts are active at each input and the score parameters remain in a fixed box. Thus, Top-K sparsity controls perinput computation without sacrificing universal approximation, provided that suficiently many routed experts are available.

## 2.2 Approximation and metric entropy with a fixed number of experts

We next consider the regime in which the number of routed experts M is fixed, complementing the preceding analysis that allows M to grow. While increasing M allows sparse Top-K routing to retain universal approximation, fixing M limits the number of efective local regions that the router can generate, leading to a more explicit approximation-complexity tradeof.

Throughout this subsection, $C _ { 1 }$ and R are fixed as above. We compare the Top-K localized averaging class $\mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p } - K } ( M )$ with the bounded global averaging class $\mathcal { A } _ { \mathrm { g l o b a l } } ^ { R }$ over the Lipschitz ball

$$
\mathcal { F } _ { b } ^ { R , L } : = \left\{ h : \mathcal { X } \to \mathbb { R } \ | \ \| h \| _ { \infty } \leq R , \ | h ( x ) - h ( y ) | \leq L \| x - y \| _ { 2 } , \ \forall x , y \in \mathcal { X } \right\} .
$$

The following proposition quantifies the approximation gain from localized routing with a fixed number of experts.

Proposition 2.3. Let $d \geq 1$ and fix M, $K \in \mathbb { N } _ { + }$ with $1 \leq K \leq M$ . Then

$$
\operatorname* { s u p } _ { h \in \mathcal { F } _ { b } ^ { R , L } } \operatorname* { i n f } _ { f \in \mathcal { A } _ { \mathrm { g l o b a l } } ^ { R } } \| f - h \| _ { \infty } = \operatorname* { m i n } \left\{ R , \frac { L \sqrt { d } } { 2 } \right\} ,
$$

and

$$
\operatorname* { s u p } _ { h \in \mathcal { F } _ { b } ^ { R , L } } \operatorname* { i n f } _ { f \in \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p } \cdot K } ( M ) } \| f - h \| _ { \infty } \leq \frac { L \sqrt { d } } { 2 \lfloor ( M / K ) ^ { 1 / d } \rfloor } .
$$

Proposition 2.3 shows that Top-K localized averaging can improve worst-case approximation over global averaging even with a fixed number of experts. Global averaging cannot adapt to local variation and therefore incurs a non-vanishing approximation error for general Lipschitz functions. By contrast, for fixed M, the Top-K approximation bound improves as K decreases, showing that greater sparsity yields a sharper worst-case approximation guarantee.

The preceding result concerns approximation accuracy. To understand the statistical price of this increased flexibility, we next examine metric entropy under the uniform norm. Similar to $\mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p } - K } ( M )$ , we define the localized averaging class with a fixed number of experts for dense softmax gating by

$$
\begin{array} { r } { \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { s o f f } } ( M ) : = \left\{ \displaystyle \sum _ { m = 1 } ^ { M } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { m } ( x ) ~ \left. ~ f _ { m } \in \mathcal { F } _ { \mathrm { c } } ^ { R } , ~ \theta _ { m } \in \Theta _ { \mathrm { l i n } } ( C _ { 1 } ) , ~ \theta = ( \theta _ { 1 } ^ { \top } , \ldots , \theta _ { M } ^ { \top } ) ^ { \top } \right. \right\} , } \end{array}
$$

where $g _ { m } ^ { \mathrm { s o f t } }$ is generated by the linear score $s _ { m } ^ { \mathrm { l i n } }$ . The bounded parameter box $\Theta _ { \mathrm { l i n } } ( C _ { 1 } )$ in this definition is introduced only for the metric entropy comparison below: it makes the covering numbers finite and directly comparable across global, Top-K, and dense softmax classes. For the other analyses of dense softmax gating in this section, this bounded-parameter restriction is not needed.

For a function class ${ \mathcal { F } } ,$ let $N ( \epsilon , \mathcal { F } , \| \cdot \| _ { \infty } )$ denote its ϵ-covering number with respect to the uniform norm. The following theorem compares three levels of flexibility: the global constant class, the dense softmax class, and the sparse Top-K class.

Theorem 2.4. Let $\mathcal { X } = [ 0 , 1 ] ^ { d }$ and fix $M \in \mathbb { N } _ { + }$ and $R , C _ { 1 } > 0$ . Then the following hold.

(i) Global constant class. $A s \ \epsilon \downarrow 0$

$$
\begin{array} { r } { \log N \left( \epsilon , \mathcal { A } _ { \mathrm { g l o b a l } } ^ { R } , \Vert \cdot \Vert _ { \infty } \right) = \log ( 1 / \epsilon ) + O ( 1 ) . } \end{array}
$$

(ii) Dense softmax class. If M = 1, the result in part (i) applies. If $M \geq 2$ , then as $\epsilon \downarrow 0 ,$

$$
\begin{array} { r } { ( d + 1 ) \log ( 1 / \epsilon ) + O ( 1 ) \le \log N \left( \epsilon , \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { s o f t } } ( M ) , \| \cdot \| _ { \infty } \right) \le [ M + ( M - 1 ) ( d + 1 ) ] \log ( 1 / \epsilon ) + O ( 1 ) . } \end{array}
$$

(iii) Sparse Top-K class. $I f 1 \leq K < M$ , then for any $0 < \epsilon \leq R / K$

$$
\log N \left( \epsilon , \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p } - K } ( M ) , \| \cdot \| _ { \infty } \right) = \infty .
$$

Theorem 2.4 shows that, for fixed M, input-dependent routing can substantially enlarge the complexity of the resulting prediction class, with sparse Top-K routing permitting particularly rich local variation through changes in the active expert set. For dense softmax routing, the larger logarithmic entropy coeficient, relative to global averaging, reflects the additional finite-dimensional complexity introduced by input-dependent weighting. By contrast, Top-K routing can change the selected expert set discontinuously across routing boundaries, which leads to infinite covering numbers at suficiently small scales. This infinite entropy is specific to the uniform norm. Under $L _ { 2 } ( P _ { X } )$ , boundary efects can be controlled by regularity or margin conditions on $P _ { X }$ , a distinction that becomes important in the estimation analysis below.

## 2.3 Beyond constant experts

The preceding subsections used constant experts to isolate the expressive contribution of the router. This simplification shows that input dependence in the gating weights alone can generate nontrivial approximation power. Modern MoE architectures, however, use highly expressive expert networks. It is therefore natural to ask whether the advantage of localized routing persists beyond constant experts, or whether it can be reproduced by suficiently rich input-independent aggregation.

To address this question, let E be a generic expert class. Define the global aggregation class

$$
\mathcal { A } _ { \mathrm { g l o b a l } } ( \mathcal { E } ; M ) : = \left\{ \sum _ { m = 1 } ^ { M } w _ { m } e _ { m } \biggm | e _ { m } \in \mathcal { E } , ~ w _ { m } \geq 0 , ~ \sum _ { m = 1 } ^ { M } w _ { m } = 1 \right\} ,
$$

and the localized softmax aggregation class

$$
A _ { \mathrm { l o c a l } } ^ { \mathrm { s o f t } } ( \mathcal { E } ; M ) : = \left\{ \sum _ { m = 1 } ^ { M } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) e _ { m } ( x ) ~ \left. ~ e _ { m } \in \mathcal { E } , ~ \theta _ { m } = ( \beta _ { m } ^ { \top } , \alpha _ { m } ) ^ { \top } \in \mathbb { R } ^ { d + 1 } , ~ \theta = ( \theta _ { 1 } ^ { \top } , \ldots , \theta _ { M } ^ { \top } ) ^ { \top } \right. \right\} ,
$$

where $g _ { m } ^ { \mathrm { s o f t } }$ is generated by the linear score $s _ { m } ^ { \mathrm { l i n } }$ . The next example shows that localization can create functions outside the finite global aggregation class even when the experts are already nonlinear.

Example 2.5. Let $\mathcal { X } = [ 0 , 1 ] ^ { d }$ and let $\begin{array} { r } { \sigma ( t ) : = \frac { 1 } { 1 + \exp ( - t ) } } \end{array}$ . Consider the expert class

$$
\xi _ { \sigma } : = \{ x \mapsto c _ { 0 } + c _ { 1 } \sigma ( a x _ { 1 } + b ) : c _ { 0 } , c _ { 1 } , a , b \in \mathbb { R } \} .
$$

Then the function $h ( x ) = \sigma ( x _ { 1 } ) ^ { 2 }$ belongs to $A _ { \mathrm { l o c a l } } ^ { \mathrm { s o f t } } ( \mathcal { E } _ { \sigma } ; 2 )$ , but

$$
h \not \in \bigcup _ { M \geq 1 } { \mathcal { A } } _ { \mathrm { g l o b a l } } ( { \mathcal { E } } _ { \sigma } ; M ) .
$$

Example 2.5 illustrates a structural distinction between global and localized aggregation. A global aggregator forms an input-independent convex combination of experts from $\mathcal { E } _ { \sigma }$ and hence remains within the finite linear span generated by such expert functions. In contrast, localized aggregation multiplies expert outputs by input-dependent gating weights. This interaction between the router and the experts creates multiplicative structure, and can generate functions, such as $\sigma ( x _ { 1 } ) ^ { 2 }$ , that are not representable by any finite inputindependent aggregation of the same expert class.

## 2.4 Two specialized experts: the advantage of localized routing

This subsection provides a risk-oriented illustration of how gating can amplify the strengths of specialized experts. For convenience, let $\mathcal { X } = [ 0 , 1 ] , \mathcal { X } _ { 1 } = [ 0 , 1 / 2 ]$ and $\mathcal { X } _ { 2 } = ( 1 / 2 , 1 ]$ . Let $f : \mathcal { X }  \mathbb { R }$ be the target regression function, with marginal law $P _ { X }$ . For any measurable $u : \mathcal { X }  \mathbb { R }$ , write

$$
\| u \| _ { \mathcal { X } _ { j } } : = \left\{ \int _ { \mathcal { X } _ { j } } u ( x ) ^ { 2 } d P _ { X } ( x ) \right\} ^ { 1 / 2 } , \qquad j = 1 , 2 .
$$

Assume that $\hat { f } _ { 1 }$ is specialized to $\mathcal { X } _ { 1 }$ and $\hat { f } _ { 2 }$ to $\mathcal { X } _ { 2 }$ in the sense that

$$
\| { \hat { f } } _ { 1 } - f \| _ { { \mathcal { X } } _ { 1 } } = \epsilon _ { 1 } , \qquad \| { \hat { f } } _ { 2 } - f \| _ { { \mathcal { X } } _ { 2 } } = \epsilon _ { 2 } , \qquad \| { \hat { f } } _ { 2 } - f \| _ { { \mathcal { X } } _ { 1 } } = \delta _ { 1 } , \qquad \| { \hat { f } } _ { 1 } - f \| _ { { \mathcal { X } } _ { 2 } } = \delta _ { 2 } ,
$$

where $\epsilon _ { j }$ is the good-region error, assumed to be relatively small, and $\delta _ { j }$ is the error of the other expert on $\chi _ { j }$ , typically relatively large.

For the fixed pair $( \hat { f } _ { 1 } , \hat { f } _ { 2 } )$ , define

$$
\begin{array} { r } {  { \mathcal { A } } _ { \mathrm { g l o b a l } } ( \hat { f } _ { 1 } , \hat { f } _ { 2 } ) : = \left\{ w \hat { f } _ { 1 } + ( 1 - w ) \hat { f } _ { 2 } \ \middle | \ 0 \leq w \leq 1 \right\} . } \end{array}
$$

This is the restriction of $\mathcal { A } _ { \mathrm { g l o b a l } } ( \mathcal { E } ; 2 )$ to the two fitted experts. By contrast, the oracle Top-1 routed predictor that uses the same two experts and splits the input space at $1 / 2$ is

$$
{ \hat { f } } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p - 1 } } ( x ) : = \left\{ { \begin{array} { l l } { { \hat { f } } _ { 1 } ( x ) , } & { x \in { \mathcal { X } } _ { 1 } , } \\ { } & { } \\ { { \hat { f } } _ { 2 } ( x ) , } & { x \in { \mathcal { X } } _ { 2 } . } \end{array} } \right.
$$

Here the routing rule assigns inputs in $\mathcal { X } _ { 1 }$ to expert 1 and inputs in $\mathcal { X } _ { 2 }$ to expert 2.

Proposition 2.6. Assume $\delta _ { 1 } + \delta _ { 2 } + \epsilon _ { 1 } + \epsilon _ { 2 } > 0$ . Let $\begin{array} { r } { \gamma : = \left[ \frac { \delta _ { 1 } \delta _ { 2 } - \epsilon _ { 1 } \epsilon _ { 2 } } { \delta _ { 1 } + \delta _ { 2 } + \epsilon _ { 1 } + \epsilon _ { 2 } } \right] _ { + } } \end{array}$ , where $[ a ] _ { + } : =$ max $\{ a , 0 \}$ , and write $\begin{array} { r } { R ( h ) : = \| h - f \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = \int _ { \mathcal { X } } \{ h ( x ) - f ( x ) \} ^ { 2 } d P _ { X } ( x ) } \end{array}$ . Then

$$
\operatorname* { i n f } _ { h \in A _ { \mathrm { g l o b a l } } ( \hat { f } _ { 1 } , \hat { f } _ { 2 } ) } R ( h ) \ge \gamma ^ { 2 } , \qquad R ( \hat { f } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p } - 1 } ) = \epsilon _ { 1 } ^ { 2 } + \epsilon _ { 2 } ^ { 2 } .
$$

Consequently, $\small { i f \epsilon _ { 1 } ^ { 2 } + \epsilon _ { 2 } ^ { 2 } < \gamma ^ { 2 } }$ , then the localized Top-1 rule using the same two experts has strictly smaller population $L _ { 2 } ( P _ { X } )$ risk than every global convex combination of them.

Remark 2.7. Proposition 2.6 formalizes the compromise faced by global aggregation. $\mathrm { A c } _ { - }$ curacy on $\mathcal { X } _ { 1 }$ pushes $w$ toward $\hat { f } _ { 1 }$ , while accuracy on $\mathcal { X } _ { 2 }$ pushes it toward ${ \hat { f } } _ { 2 } ;$ a single input-independent weight cannot generally satisfy both. The quantity $\gamma$ measures this tension. In the symmetric case $\delta _ { 1 } = \delta _ { 2 } = \delta$ and $\epsilon _ { 1 } = \epsilon _ { 2 } = \epsilon $ , we have $\gamma = [ ( \delta - \epsilon ) / 2 ] _ { + }$ When $\delta$ is much larger than $\epsilon ,$ each expert is genuinely specialized: it performs well on its own region but poorly on the other. In this regime, local routing has a larger advantage because it preserves the good-region performance of both experts instead of forcing a global compromise.

The results in this section give a simple but useful message. The expressive power of MoE models does not come only from the complexity of the individual experts. It also comes from the router, which turns a fixed collection of expert outputs into a locally adaptive prediction rule. Dense routing, sparse routing and global averaging therefore define genuinely diferent approximation classes.

## 3 Risk bounds for dense MoE

The previous section studied MoE architectures from an approximation perspective and showed that input-dependent routing can substantially enlarge the class of attainable predictors. We now turn to the corresponding statistical question: Given such a localized aggregation class, what is the cost of learning the router from finite data? This section focuses on dense softmax gating and derives oracle risk bounds for router learning along an online training trajectory. These bounds decompose the risk into approximation error, cumulative expert-learning error, and the statistical cost of learning the routing rule.

## 3.1 Basic setting and training procedure

We observe independent data $Z _ { i } = ( X _ { i } , Y _ { i } )$ generated from $Y _ { i } = f _ { 0 } ( X _ { i } ) + \varepsilon _ { i }$ , where $f _ { 0 } ( x ) =$ $\mathbb { E } ( Y _ { i } \mid X _ { i } = x )$ is an unknown regression function and $X _ { i } \sim P _ { X }$ on a compact input space $\mathcal { X } \subset \mathbb { R } ^ { d }$ with $\| x \| _ { 2 } \leq C _ { X } { \sqrt { d } }$ for every $x \in \mathcal { X }$ and some constant $C _ { X }$ . Throughout the remainder of the paper, the error $\varepsilon$ is assumed to be independent of X and to have a scalefamily density $h _ { \varepsilon } ( z ) = \sigma ^ { - 1 } h _ { \varepsilon , 0 } ( z / \sigma )$ , where $h _ { \varepsilon , 0 }$ is known, centered at zero, and normalized to have unit variance.

We adopt the shared-routed MoE formulation introduced in Section 1.1, specialize to dense softmax gating, and use an online decoupled formulation to accommodate sequential expert updating. Let U collect all auxiliary randomization used by the initialization and the blackbox expert training procedure, and assume that U is independent of the data. For each $t = 1 , \ldots , n - 1$ , the shared and routed expert functions available after processing $Z _ { 1 } , \ldots , Z _ { t }$ are measurable with respect to the σ-field generated by $Z _ { 1 } , \ldots , Z _ { t }$ and $u ,$ , namely, $\sigma ( Z _ { 1 } , \dots , Z _ { t } , \mathcal { U } )$ . After observing $X _ { t + 1 }$ , their evaluations and the candidate MoE predictions of $Y _ { t + 1 }$ are measurable with respect to the σ-field $\sigma ( Z _ { 1 } , \dots , Z _ { t } , \mathcal { U } , X _ { t + 1 } )$ . Once $Y _ { t + 1 }$ is observed, the resulting prediction loss is used to update the aggregation weights, and $Z _ { t + 1 }$ may also be used to update the experts for subsequent predictions.

Under this formulation, for each gating parameter θ and each time t, the candidate predictor takes the form

$$
\check { f } ^ { t } ( x ; \theta ) = \sum _ { \ell = 1 } ^ { M _ { S } } \hat { f } _ { S , \ell } ^ { t } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) \hat { f } _ { R , m } ^ { t } ( x ) ,\tag{3.1}
$$

where $\hat { f } _ { S , \ell } ^ { t }$ and $\hat { f } _ { R , m } ^ { t }$ denote the shared and routed experts available after processing the first t observations, respectively. Thus, conditional on the past data, the experts enter the next-step prediction problem as fixed plug-in components, while their quality is allowed to improve along the training trajectory.

This online formulation also simplifies identifiability issues. In a fully joint MoE model, the parameters are generally non-identifiable because experts can be permuted without changing the predictor. Here, for each time $t ,$ the experts $\hat { f } _ { S , \ell } ^ { t }$ and $\hat { f } _ { R , m } ^ { t }$ are viewed as fixed expert slots produced by the past training process, so this permutation ambiguity is irrelevant. The only remaining non-identifiability is the usual shift invariance of softmax scores: adding a common function to all scores does not change the normalized gating weights. Since our results concern the induced gating functions and predictors, this invariance has no efect on the risk bounds. When a parameter level normalization is convenient for softmax gating, we use a standard reference constraint, for example $s _ { M _ { R } } ( x ; \theta _ { M _ { R } } ) = 0$

Even along this online training trajectory, router learning remains statistically and computationally nontrivial. Conditional on the experts available at time t, estimating the dense softmax gate from the next observation or from a block of subsequent observations still involves a non-convex parametrization. We therefore use a discretization-based aggregation strategy, inspired by adaptive regression via mixing (Yang, 2001) and aggregated forecasting through exponential reweighting (Yang, 2004), to obtain risk guarantees without assuming global optimization of a non-convex empirical objective. The procedure constructs a finite net of candidate gates, aggregates the corresponding MoE predictors using likelihood-based weights, and, when a single MoE parameter is required, projects the aggregate back onto the discretized class. This construction is intended as a theoretical benchmark for router learning along the training trajectory, not as a practical substitute for end-to-end MoE training. The estimation procedure is summarized in Algorithm 1 for a general gating class. In this algorithm, i indexes the i.i.d. observations, while t is an algorithmic time index that records the sequential use of observations in the training procedure.

Remark 3.1. The shared and routed experts play diferent roles in the scaling considered below. Shared experts are intended to capture structure reused across the input space, so $M _ { S }$ is treated as fixed. Routed experts, by contrast, are used to represent local heterogeneity induced by the gate, and increasing their number allows a finer collection of local components.

Algorithm 1: Discretized Aggregation over a General Router Family   
Input: A burn-in size $n _ { 1 } ;$ ; a variance calibration size $n _ { 2 } ;$ a router family $g ( \cdot ; \theta )$ with   
$ { \boldsymbol { \theta } } \in \Theta _ { M _ { R } } ;$ a blackbox expert update rule; data $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$   
Output: An online aggregated sequence $\{ \tilde { f } ^ { t } ( x ) \} _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \}$ estimated gating   
parameter $\hat { \theta }$ and final estimator $\hat { f } _ { 0 } ( x )$   
(a) Choose an η-net of the normalized bounded gating parameter space $\Theta _ { M _ { R } } ,$ , denoted by $\Theta ( \eta )$   
Write $\Theta ( \eta ) \stackrel { \cdot } { = } \{ \theta ^ { ( 1 ) } , \ldots , \theta ^ { ( S _ { \eta } ) } \}$ where $S _ { \eta } = | \Theta ( \eta ) |$ . The mesh size $\eta$ is chosen according to the   
continuity and covering properties of the router family. These parameters induce the   
discretized routers $g ( \cdot ; \overleftarrow { \theta ^ { ( 1 ) } } ) , \cdot \cdot \cdot , g ( \cdot ; \theta ^ { ( S _ { \eta } ) } )$   
(b) Randomly permute the observations once, relabel them as $Z _ { 1 } , \ldots , Z _ { n } ,$ , and split the relabeled   
sample into three consecutive blocks, $Z ^ { ( 1 ) } = ( X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n _ { 1 } } , Z ^ { ( 2 ) } = ( X _ { i } , Y _ { i } ) _ { i = n _ { 1 } + 1 } ^ { n _ { 1 } + \hat { n _ { 2 } } }$ and   
$Z ^ { ( 3 ) } = ( X _ { i } , Y _ { i } ) _ { i = n _ { 1 } + n _ { 2 } + 1 } ^ { n } ;$ , and let $n _ { 3 } = n - n _ { 1 } - n _ { 2 } .$   
(c) Burn-in: Process $Z ^ { ( 1 ) }$ by the blackbox update rule to obtain $\{ \hat { f } _ { S , \ell } ^ { n _ { 1 } } \} _ { \ell = 1 } ^ { M _ { S } }$ and $\{ \hat { f } _ { R , m } ^ { n _ { 1 } } \} _ { m = 1 } ^ { M _ { R } }$   
(d) Calibration: For $t = n _ { 1 } , \ldots , n _ { 1 } + n _ { 2 } - 1 { : }$   
(i) For each $\theta ^ { ( s ) } \in \Theta ( \eta )$ , construct the candidate MoE predictor $\check { f } ^ { t } ( x ; \theta ^ { ( s ) } )$ as in (3.1) by   
replacing softmax gating $g ^ { \mathrm { s o f t } }$ by the specified gating $g$ and using experts trained at   
time t. After observing $Z _ { t + 1 }$ , compute the residual $\bar { e } _ { s , t + 1 } = Y _ { t + 1 } - \bar { \tilde { f } } ^ { t } ( X _ { t + 1 } ; \theta ^ { ( s ) } )$   
(ii) Update the experts by the blackbox rule to obtain $\{ \hat { f } _ { S , \ell } ^ { t + 1 } \} _ { \ell = 1 } ^ { M _ { S } }$ and $\{ \hat { f } _ { R , m } ^ { t + 1 } \} _ { m = 1 } ^ { M _ { R } }$   
Estimate the corresponding scale parameter by $\begin{array} { r } { \tilde { \sigma } _ { s } ^ { 2 } = 1 / n _ { 2 } \sum _ { t = n _ { 1 } + 1 } ^ { n _ { 1 } + n _ { 2 } } e _ { s , t } ^ { 2 } , } \end{array}$ , and let   
$\hat { \sigma } _ { s } ^ { 2 } = \Pi _ { [ \underline { { \sigma } } ^ { 2 } , \bar { \sigma } ^ { 2 } ] } ( \tilde { \sigma } _ { s } ^ { 2 } )$ where $\Pi _ { [ a , b ] } ( u ) =$ min $\{ b ,$ max $\{ a , u \} \}$   
(e) Aggregation: Set $\hat { w } _ { s , n _ { 1 } + n _ { 2 } } = 1 / S _ { \eta } .$ . For $t = n _ { 1 } + n _ { 2 } , \ldots , n - 1 { \mathrm { : } }$   
(i) Obtain the candidate predictors $\check { f } ^ { t } ( x ; \theta ^ { ( s ) } )$ similarly as before, and output   
$\begin{array} { r } { \tilde { f } ^ { t } ( x ) = \sum _ { s = 1 } ^ { S _ { \eta } } \hat { w } _ { s , t } \check { f } ^ { t } ( x ; \theta ^ { ( s ) } ) } \end{array}$   
(ii) After observing $Z _ { t + 1 }$ , update the weights $\hat { w } _ { s , t + 1 }$ by   
$\hat { w } _ { s , t } \hat { \sigma } _ { s } ^ { - 1 } h _ { \varepsilon , 0 } \left( ( Y _ { t + 1 } - \check { f } ^ { t } ( X _ { t + 1 } ; \theta ^ { ( s ) } ) ) / \hat { \sigma } _ { s } \right)$   
$w _ { s , t + 1 } = \frac { \hat { } } { \sum _ { r = 1 } ^ { S _ { \eta } } \hat { w } _ { r , t } \hat { \sigma } _ { r } ^ { - 1 } h _ { \varepsilon , 0 } \left( \left( Y _ { t + 1 } - \check { f } ^ { t } ( X _ { t + 1 } ; \theta ^ { ( r ) } ) \right) / \hat { \sigma } _ { r } \right) } .$ (3.2)   
(iii) Update the experts by the blackbox rule to obtain $\{ \hat { f } _ { S , \ell } ^ { t + 1 } \} _ { \ell = 1 } ^ { M _ { S } }$ and $\{ \hat { f } _ { R , m } ^ { t + 1 } \} _ { m = 1 } ^ { M _ { R } }$   
(f) Averaging the online outputs gives the predictor $\tilde { f } _ { 0 } .$ , while, for each $s \in [ S _ { \eta } ]$ , we define the   
corresponding time-averaged candidate predictor $f ^ { n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } )$ by   
$\tilde { f } _ { 0 } ( x ) = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \tilde { f } ^ { t } ( x ) , \qquad \bar { f } ^ { n _ { 3 } } ( x ; \theta ^ { ( s ) } ) = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \check { f } ^ { t } ( x ; \theta ^ { ( s ) } ) .$   
Project the aggregated predictor back onto the discretized MoE class by choosing the   
candidate predictor closest to $\tilde { f } _ { 0 }$ in $L _ { 2 } ( P _ { X } ) ;$   
$s ^ { * } = \underset { 1 \leq s \leq S _ { \eta } } { \arg \operatorname* { m i n } } \| \widetilde { f } _ { 0 } - \bar { f } ^ { n _ { 3 } } ( \cdot ; \boldsymbol { \theta } ^ { ( s ) } ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } , \qquad \hat { \boldsymbol { \theta } } = \boldsymbol { \theta } ^ { ( s ^ { * } ) } .$   
The final estimator is $\hat { f } _ { 0 } ( x ) = \bar { f } ^ { n _ { 3 } } ( x ; \hat { \theta } )$

Accordingly, we consider a triangular array setting in which $M _ { R } = M _ { R , n }$ is deterministic and may increase with n. This convention is consistent with recent shared-routed MoE designs, such as DeepSeekMoE, where a small number of shared experts is combined with a larger set of routed experts. The formal triangular array notation is introduced before the assumptions used in the oracle analysis.

Remark 3.2. The projection in Step (f) is defined at the population level. In the online setting, the aggregated predictor is obtained by averaging over both time and discretized gates. The projection is therefore taken onto the time-averaged MoE class whose elements are

$$
\bar { f } ^ { n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } ) = \sum _ { \ell = 1 } ^ { M _ { S } } \bar { f } _ { S , \ell } ^ { n _ { 3 } } + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( \cdot ; \theta ^ { ( s ) } ) \bar { f } _ { R , m } ^ { n _ { 3 } } ,
$$

where $\begin{array} { r } { \bar { f } _ { S , \ell } ^ { n _ { 3 } } = \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \hat { f } _ { S , \ell } ^ { t } / n _ { 3 } } \end{array}$ and $\begin{array} { r } { \bar { f } _ { R , m } ^ { n _ { 3 } } = \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \hat { f } _ { R , m } ^ { t } / n _ { 3 } } \end{array}$ are the corresponding timeaveraged experts. The population criterion involves $P _ { X }$ , which is unknown in practice. When X is supported on a compact set and $P _ { X }$ has a density bounded above and away from zero, the $L _ { 2 } ( P _ { X } )$ and Lebesgue $L _ { 2 }$ norms are equivalent, so a Lebesgue-integrated surrogate yields the same approximation order. For fixed $M _ { S }$ and $M _ { R }$ , Supplement Section F.2 gives a fully data-driven version of this projection step based on time-expanded crossentropy minimization; see also Remark 3.11.

Remark 3.3. If the noise scale is known, the calibration block can be omitted. If a rolling or external scale estimator is used instead, the same finite-grid aggregation argument yields an additional scale-estimation contribution. The formulation in Algorithm 1 keeps the main statement self-contained by absorbing scale estimation into the displayed learning cost below.

## 3.2 Oracle inequalities for dense softmax gating

We now analyze the estimator produced by Algorithm 1 when the gating class is the dense softmax class with linear scores. Recall that the m-th gating weight is given by

$$
g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) = \frac { \exp \{ s _ { m } ^ { \mathrm { l i n } } ( x ; \theta _ { m } ) \} } { \sum _ { j = 1 } ^ { M _ { R } } \exp \{ s _ { j } ^ { \mathrm { l i n } } ( x ; \theta _ { j } ) \} } , \qquad m = 1 , \ldots , M _ { R } ,
$$

where $s _ { m } ^ { \mathrm { l i n } } ( x ; \theta _ { m } ) = x ^ { \top } \beta _ { m } + \alpha _ { m }$ . Let $\hat { \theta }$ denote the parameter selected by Algorithm 1 under this dense softmax specification. Equivalently, the projected estimator in Step (f) can be written as

$$
\hat { f } _ { 0 } ( x ) : = \sum _ { \ell = 1 } ^ { M _ { S } } \bar { f } _ { S , \ell } ^ { n _ { 3 } } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \hat { \theta } ) \bar { f } _ { R , m } ^ { n _ { 3 } } ( x ) ,
$$

where $\bar { f } _ { S , \ell } ^ { n _ { 3 } }$ and $\bar { f } _ { R , m } ^ { n _ { 3 } }$ are the time-averaged experts introduced in Remark 3.2.

Since the number of routed experts can increase with the sample size, we make the triangular array dependence explicit and write the evolving expert predictors available at time t as $\{ \hat { f } _ { S , \ell } ^ { t , n } \} _ { \ell \in [ M _ { S } ] }$ and $\{ \hat { f } _ { R , m } ^ { t , n } \} _ { m \in [ M _ { R , n } ] }$ . An admissible pair $( t , n )$ refers to a sample size n and an algorithmic time index t at which these experts are used in Algorithm 1. We impose the following assumptions.

Assumption 3.4. For each sample size $n _ { \mathrm { : } }$ the dense softmax gating parameter belongs to a compact parameter space $\Theta _ { M _ { R } }$ . In addition, there exists a constant $C _ { 1 } > 1 / 2$ , independent of n and hence of $M _ { R }$ , such that $\Theta _ { M _ { R } }$ is contained in a box of radius $C _ { 1 }$ log $M _ { R }$ in $\mathbb { R } ^ { M _ { R } ( d + 1 ) }$ In particular, under softmax gating we impose the normalization $\theta _ { M _ { R } } = 0$ , so that the efective parameter space has dimension $( M _ { R } - 1 ) ( d + 1 )$ , although the ambient space is written as $\mathbb { R } ^ { M _ { R } ( d + 1 ) }$ .

The logarithmic growth of the parameter radius in Assumption 3.4 allows the dense softmax gate to remain suficiently selective as the number of routed experts increases. If the parameter box were fixed uniformly in $M _ { R } ,$ then the score diferences would remain uniformly bounded, and the largest attainable softmax weight assigned to any single expert would decrease as $M _ { R }$ grows. This limits the router’s ability to approximate hard specialization among a growing number of experts. The log $M _ { R }$ scaling avoids this degeneracy while preserving compactness of the parameter space for each fixed $M _ { R }$

Assumption 3.5. For each sample size $n ,$ the evolving experts admit deterministic population benchmarks $\{ f _ { S , \ell , n } ^ { * } \} _ { \ell \in [ M _ { S } ] }$ and $\{ f _ { R , m , n } ^ { * } \} _ { m \in [ M _ { R , n } ] }$ . There exist a constant $C _ { 2 } < \infty$ and a nonnegative deterministic triangular array $\{ a _ { t , n } \}$ such that, for every admissible pair

(t, n),

$$
\operatorname* { m a x } _ { \ell \leq M _ { S } } \mathbb { E } \left\| \hat { f } _ { S , \ell } ^ { t , n } - f _ { S , \ell , n } ^ { * } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \vee \operatorname* { m a x } _ { m \leq M _ { R , n } } \mathbb { E } \left\| \hat { f } _ { R , m } ^ { t , n } - f _ { R , m , n } ^ { * } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C _ { 2 } a _ { t , n } ^ { 2 } .
$$

Assumption 3.5 requires the evolving experts to be controlled relative to deterministic population benchmarks. The array $\boldsymbol { a } _ { t , n }$ quantifies the corresponding $L _ { 2 } ( P _ { X } )$ discrepancy at time t and sample size n. Pointwise convergence of $\boldsymbol { a } _ { t , n }$ along every admissible sequence is not required; the oracle inequality instead depends on the block averages $\bar { a } _ { n _ { 2 } } ^ { 2 }$ and $\bar { a } _ { n _ { 3 } } ^ { 2 }$ defined in Theorem 3.8.

Assumption 3.6. The regression function, the population benchmarks of the experts, and the evolving expert estimators are uniformly bounded. Specifically, there exists a constant $A < \infty$ such that the following bound holds almost surely for every admissible pair $( t , n )$ 2

$$
\lVert f _ { 0 } \rVert _ { \infty } \vee \operatorname* { m a x } _ { 1 \leq \ell \leq M _ { S } } \left\{ \lVert f _ { S , \ell , n } ^ { * } \rVert _ { \infty } \vee \lVert \hat { f } _ { S , \ell } ^ { t , n } \rVert _ { \infty } \right\} \vee \operatorname* { m a x } _ { 1 \leq m \leq M _ { R , n } } \left\{ \lVert f _ { R , m , n } ^ { * } \rVert _ { \infty } \vee \lVert \hat { f } _ { R , m } ^ { t , n } \rVert _ { \infty } \right\} \leq A .
$$

Assumption 3.6 requires $f _ { 0 }$ , the evolving experts and their population benchmarks to be uniformly bounded over all admissible $( t , n )$ . Whenever n is fixed, we suppress its index and write $M _ { R } = M _ { R , n } , \hat { f } _ { S , \ell } ^ { t } = \hat { f } _ { S , \ell } ^ { t , n } , \hat { f } _ { R , m } ^ { t } = \hat { f } _ { R , m } ^ { t , n } , f _ { S , \ell } ^ { * } = f _ { S , \ell , n } ^ { * } , f _ { R , m } ^ { * } = f _ { R , m , n } ^ { * } .$ , and $a _ { t } = a _ { t , n }$

Assumption 3.7. The error term ε is independent of X and has density $h _ { \varepsilon } ( z ) = \sigma ^ { - 1 } h _ { \varepsilon , 0 } ( z / \sigma )$ where the known standardized density $h _ { \varepsilon , 0 }$ satisfies $\textstyle { \int z h _ { \varepsilon , 0 } ( z ) d z = 0 , \int z ^ { 2 } h _ { \varepsilon , 0 } ( z ) d z = 1 }$ and $\begin{array} { r } { \int z ^ { 4 } h _ { \varepsilon , 0 } ( z ) d z < \infty } \end{array}$ . For any $0 < s _ { 0 } < 1$ and $T _ { 0 } > 0$ , there exists a constant $C _ { 3 } = C _ { 3 } ( s _ { 0 } , T _ { 0 } )$ such that

$$
\int h _ { \varepsilon , 0 } ( z ) \log \frac { h _ { \varepsilon , 0 } ( z ) } { s ^ { - 1 } h _ { \varepsilon , 0 } ( ( z - u ) / s ) } d z \leq C _ { 3 } \{ ( 1 - s ) ^ { 2 } + u ^ { 2 } \} ,
$$

for all $s _ { 0 } \leq s \leq s _ { 0 } ^ { - 1 }$ and $| u | \leq T _ { 0 }$ . Moreover, the scale parameter satisfies $0 < \underline { { \sigma } } \le \sigma \le \bar { \sigma } <$ ∞ where $\underline { { \sigma } }$ and ¯σ are known constants.

Assumption 3.7 standardizes $h _ { \varepsilon , 0 }$ to have mean zero and unit variance, so that $f _ { 0 }$ is the conditional mean function and $\sigma ^ { 2 }$ is the error variance. The remaining condition is a local Kullback–Leibler regularity requirement for location-scale perturbations. It is satisfied by Gaussian, double-exponential, and many other commonly used distributions.

Under the above assumptions, Algorithm 1 yields an oracle inequality for dense softmax gating. The bound separates two sources of error. The first is the oracle approximation risk, which measures how well the population shared and routed experts can approximate $f _ { 0 }$ under the best dense softmax gate in the prescribed class. The second is the statistical learning cost, which consists of the cumulative error of the evolving experts and the finitesample cost of estimating the gate. The following theorem gives the explicit form of these two terms.

Theorem 3.8. Assume Assumptions 3.4–3.7 hold. Run Algorithm 1 with dense softmax gating, $n _ { 1 } \asymp n _ { 2 } \asymp n _ { 3 }$ and mesh size $\eta \asymp ( M _ { R } \sqrt { n } ) ^ { - 1 }$ . Suppose that $M _ { S }$ is fixed. Then the online sequence $\{ \tilde { f } ^ { t } \} _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 }$ satisfies

$$
\frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \| \tilde { f } ^ { t } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + \mathfrak { L } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) \right\} ,\tag{3.3}
$$

where the softmax oracle approximation risk is

$$
\mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) : = \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \left. \mathscr { f } _ { 0 } - \sum _ { \ell = 1 } ^ { M _ { S } } \mathscr { f } _ { S , \ell } ^ { * } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) \mathscr { f } _ { R , m } ^ { * } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } ,
$$

and the softmax statistical learning cost is

$$
\mathfrak { L } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) : = M _ { R } ( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } ) + \frac { M _ { R } d } { n } \log ( M _ { R } d n ) ,
$$

where $\begin{array} { r } { \bar { a } _ { n _ { 2 } } ^ { 2 } = \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } a _ { t } ^ { 2 } / n _ { 2 } } \end{array}$ and $\begin{array} { r } { \bar { a } _ { n _ { 3 } } ^ { 2 } = \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } a _ { t } ^ { 2 } / n _ { 3 } } \end{array}$ . Here C is a constant independent of n and $M _ { R }$ . The same bound holds for the risk $\mathbb { E } \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 }$ of the final projected estimator.

Theorem 3.8 gives a non-asymptotic oracle inequality in which the number of routed experts $M _ { R }$ is allowed to grow with the sample size. The approximation term $\mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } )$ depends on the quality of the population experts and on the expressive power of the dense softmax gate. The learning term $\mathfrak { L } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } )$ has two components. The term $M _ { R } ( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } )$ measures the average quality of the expert trajectory during the calibration and aggregation stages, while the term $\begin{array} { r } { \frac { M _ { R } d } { n } \log ( M _ { R } d n ) } \end{array}$ is the cost of estimating the gate over a growing softmax class. Because $M _ { S }$ is fixed, the statistical contribution of the shared

experts is absorbed into the constant.

Thus $M _ { R }$ plays a dual role. Increasing $M _ { R }$ may improve localization and reduce the oracle approximation risk, but it also increases the statistical cost of learning the router. This is the statistical analogue of a familiar practical tradeof in MoE systems: adding routed experts increases capacity, but may also introduce load-balancing dificulties, communication overhead, and low expert utilization. Hence, a larger number of routed experts is not automatically preferable. The choice of $M _ { R }$ should balance the approximation gain from finer localization against the estimation cost of a more complex gating class. The next example illustrates how a shared expert can reduce the oracle approximation term by extracting a common component that would otherwise have to be approximated by the routed experts. This same shared-residual mechanism is studied more broadly in Section $6 ,$ with emphasis on how it changes the estimation problem faced by local learning rules.

Example 3.9. Let $X \sim \mathrm { U n i f } [ 0 , 1 ]$ and consider the target function $f _ { 0 } ( x ) = A \sin ( 2 \pi L x ) +$ $x ^ { 2 }$ , where $L \gg 1$ controls the complexity of the oscillatory component. Suppose that $M _ { S } = 1$ and that the shared expert captures the oscillatory component, namely $f _ { S , 1 } ^ { * } ( x ) =$ $A \sin ( 2 \pi L x )$ . The routed experts are then used to approximate the residual function $x ^ { 2 }$ through local aggregation. Specifically, suppose the population routed experts belong to the afine class $\mathcal F _ { R , \mathrm { l i n } } = \{ f ( x ) = a x + b : a , b \in \mathbb { R } \}$ . Then there exist routed experts $f _ { R , 1 } ^ { * } , \ldots , f _ { R , M _ { R } } ^ { * } \in { \mathcal { F } } _ { R , \operatorname* { l i n } }$ such that the softmax oracle approximation risk satisfies

$$
\mathfrak { A } _ { \mathrm { s o f t } } ( 1 , M _ { R } ) = \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \left. f _ { 0 } - f _ { S , 1 } ^ { * } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { R , m } ^ { * } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = O \left( \frac { 1 } { \log ^ { 2 } M _ { R } } \right) .
$$

Without the shared expert, the routed experts must approximate the full function $f _ { 0 }$ . In that case, the corresponding oracle approximation risk is of order $O ( L ^ { 4 } / \log ^ { 2 } M _ { R } )$

With the shared expert, the routed experts only need to approximate the smoother residual $x ^ { 2 }$ , and the approximation error no longer scales with the oscillation parameter $L .$ Without the shared expert, the routed experts must approximate both the oscillatory component and the residual structure, which inflates the oracle approximation risk by a factor of order $L ^ { 4 } .$ . Thus, in this example, the shared expert improves oracle approximation by removing a component that is dificult for the routed afine class to approximate.

The example also highlights the role of $M _ { R }$ . If $M _ { R }$ is fixed as $n  \infty$ , the oracle approximation risk generally does not vanish unless the fixed expert collection already yields an exact representation. Consistency therefore requires either a suficiently rich fixed expert collection or a growing number of routed experts.

Remark 3.10. A direct empirical risk minimization (ERM) analysis of the non-convex softmax class typically proceeds through uniform concentration bounds for the empirical squared loss. Even if the empirical objective is assumed to attain a global minimum, this route gives an excess risk bound of order $\sqrt { M _ { R } d \log ( M _ { R } d n ) / n } ;$ see Supplement Section F.1. By contrast, the excess risk bound obtained from the aggregation argument in Theorem 3.8 includes a term of order $M _ { R } d \log ( M _ { R } d n ) / n$ for estimating the gating function, in addition to the error contributed by the evolving experts. Thus, when the routed-expert complexity is moderate relative to the sample size, the aggregation approach provides a sharper oracle benchmark than the direct ERM route.

Remark 3.11. When $M _ { R }$ is fixed, the population projection step (f) in Algorithm 1 can be replaced by a fully data-driven surrogate. Recall that the aggregation step produces time-varying aggregated gating weights of the form $\begin{array} { r } { \tilde { g } _ { m , t } ( \boldsymbol { x } ) = \sum _ { s = 1 } ^ { S _ { \eta } } \hat { w } _ { s , t } g _ { m } ^ { \mathrm { s o f t } } ( \boldsymbol { x } ; \boldsymbol { \theta } ^ { ( s ) } ) } \end{array}$ . We estimate θ by projecting these aggregated weights back onto the dense softmax class through the time-expanded cross-entropy criterion

$$
\mathcal { L } _ { \mathrm { c e } } ( \theta ; n _ { 3 } , n _ { 4 } ) = - \frac { 1 } { n _ { 3 } n _ { 4 } } \sum _ { i = 1 } ^ { n _ { 4 } } \sum _ { \substack { t = n _ { 1 } + n _ { 2 } m = 1 } } ^ { n - 1 } \sum _ { \substack { m , t } } ^ { M _ { R } } \widetilde { g } _ { m , t } ( \widetilde { X } _ { i } ) \log g _ { m } ^ { \mathrm { s o f t } } ( \widetilde { X } _ { i } ; \theta ) ,
$$

where $\{ \widetilde { X } _ { i } \} _ { i = 1 } ^ { n _ { 4 } }$ is an independent validation sample from $P _ { X }$ with $n _ { 4 } \asymp n$ , and set ${ \widehat { \theta } } _ { \mathrm { c e } } =$ arg min $\mathsf { \iota } _ { \theta \in \Theta _ { M _ { R } } } \mathcal { L } _ { \mathrm { c e } } ( \theta ; n _ { 3 } , n _ { 4 } )$ . Under the linear-score softmax parametrization, this is a convex multinomial logistic regression problem with soft labels $\widetilde { g } _ { t } ( \widetilde { X } _ { i } )$ . Hence, in the fixed- $M _ { R }$ regime, the projection step admits a computationally tractable implementation without additional non-convex optimization. Under mild regularity conditions, the resulting estimator satisfies the oracle inequality

$$
\mathbb { E } \| \hat { f } _ { 0 , \mathrm { c e } } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C _ { M _ { R } } \left\{ \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } + \frac { \log n } { n } \right\} ,\tag{3.4}
$$

where $C _ { M _ { R } }$ is a constant depending on $M _ { R }$ , and $\mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } )$ is the softmax oracle approximation risk defined in Theorem 3.8. The proof is deferred to Supplement Section F.2.

This fixed- $M _ { R }$ implementation also clarifies the connection with adaptive regression by mixing. The classical adaptive regression by mixing (ARM) procedure (Yang, 2001) aggregates a finite set of candidate estimators using input-independent weights. Here, the aggregation is carried out over candidate gating functions, so the resulting aggregate induces covariate-dependent expert weights. This extension is essential for MoE models, because the relative performance of diferent experts may vary across the input space.

## 4 Risk bounds for sparse MoE

Dense softmax gating assigns positive weights to all routed experts for each input. This smooth averaging can be statistically efective, but it becomes computationally expensive when the number of routed experts $M _ { R }$ is large. Sparse Top-K gating addresses this issue by activating only a small subset of routed experts for each input. Since the seminal work of (Shazeer et al., 2017), this idea has become a central component of large-scale MoE architectures, including GShard, Switch Transformer, GLaM and Mixtral (Du et al., 2022; Fedus et al., 2022; Jiang et al., 2024; Lepikhin et al., 2021). Importantly, MoE systems are not restricted to Top-1 routing: many architectures activate multiple experts for each input, with K = 2 or larger values commonly adopted (Dai et al., 2024; Du et al., 2022; Jiang et al., 2024; Kimi Team et al., 2026). This motivates a statistical comparison between hard expert selection and local multi-expert aggregation.

This section studies the statistical efect of Top-K sparsification under the same online training formulation as Section 3. We ask whether sparse Top-K routing can retain the oracle-risk guarantees of dense softmax gating while reducing per-input computation.

## 4.1 Oracle inequalities for Top-K gating

We use the linear routing scores $s _ { m } ^ { \mathrm { l i n } } ( x ; \theta _ { m } ) = x ^ { \top } \beta _ { m } + \alpha _ { m } , m \in [ M _ { R } ]$ , as in Section 1.1 and write $s _ { m } ( x ; \theta _ { m } ) = s _ { m } ^ { \mathrm { l i n } } ( x ; \theta _ { m } )$ for simplicity in this section. As in the dense softmax case, we impose the reference normalization $\theta _ { M _ { R } } = 0$ to remove the shift invariance of the scores.

Fix $K \in [ M _ { R } ]$ . For a given input x and parameter θ, let $s _ { ( K ) } ( x ; \theta )$ denote the K-th

largest value among the scores $s _ { 1 } ( x ; \theta _ { 1 } ) , \ldots , s _ { M _ { R } } ( x ; \theta _ { M _ { R } } )$ . To resolve possible ties at the selection threshold, we adopt a deterministic tie-breaking rule, with one possible choice being to favor experts with smaller indices. The Top-K active set is then defined as

$$
\begin{array} { r } { \mathcal { T } _ { K } ( x ; \theta ) = \{ m \in [ M _ { R } ] : s _ { m } ( x ; \theta _ { m } ) \geq s _ { ( K ) } ( x ; \theta ) \} , } \end{array}
$$

so that $| \mathcal { T } _ { K } ( x ; \theta ) | = K$ . Under mild regularity conditions, for example, when X has a continuous distribution and the pairwise score diferences are nondegenerate, ties occur with probability zero for each fixed $\theta ,$ and the particular tie-breaking rule is immaterial.

We allow the selected scores to be transformed before normalization. Let $\phi : \mathbb { R } \to ( 0 , \infty )$ be a positive score transformation. The Top-K normalized gate associated with $\phi$ is

$$
g _ { m } ^ { ( K , \phi ) } ( x ; \theta ) = \frac { \phi \{ s _ { m } ( x ; \theta _ { m } ) \} \mathbb { 1 } \{ m \in \mathcal { T } _ { K } ( x ; \theta ) \} } { \sum _ { j \in \mathcal { T } _ { K } ( x ; \theta ) } \phi \{ s _ { j } ( x ; \theta _ { j } ) \} } , \qquad m \in [ M _ { R } ] .
$$

Thus unselected experts receive weight zero, while selected experts are normalized over the active set $\mathcal { T } _ { K } ( \boldsymbol { x } ; \boldsymbol { \theta } )$ . The choice $\phi ( t ) = \exp ( t )$ gives the Top-K softmax gate introduced earlier; in this case, we write $g ^ { ( K ) }$ for short. More generally, the analysis accommodates any regular score transformation satisfying the regularity conditions in Definition 4.1 below.

Given the evolving experts available at time t, the candidate predictor indexed by θ is

$$
\check { f } ^ { K , \phi , t } ( x ; \theta ) = \sum _ { \ell = 1 } ^ { M _ { S } } \hat { f } _ { S , \ell } ^ { t } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( K , \phi ) } ( x ; \theta ) \hat { f } _ { R , m } ^ { t } ( x ) .
$$

Running Algorithm 1 with the Top-K gate defined above gives scale estimates $\hat { \sigma } _ { s }$ and aggregation weights $\hat { w } _ { s , t }$ over the finite grid $\Theta ( \eta ) = \{ \theta ^ { ( 1 ) } , \dots , \theta ^ { ( S _ { \eta } ) } \}$ . During the aggregation block, the online aggregated predictor is $\begin{array} { r } { \tilde { f } ^ { K , \phi , t } ( x ) = \sum _ { s = 1 } ^ { S _ { \eta } } \hat { w } _ { s , t } \check { f } ^ { K , \phi , t } ( x ; \theta ^ { ( s ) } ) } \end{array}$ . Equivalently, this predictor can be written as a shared-expert MoE with efective gating weights $\tilde { g } _ { m , t } ^ { K , \phi } ( x ) =$ $\begin{array} { r l } { \sum _ { s = 1 } ^ { S _ { \eta } } \hat { w } _ { s , t } g _ { m } ^ { ( K , \phi ) } ( x ; \theta ^ { ( s ) } ) } & { { } } \end{array}$ . This resulting gate is a convex combination of Top-K gates and need not itself be Top-K for a single parameter. This distinction is harmless for the online oracle inequality, which evaluates the actual aggregate predictor. When a single sparse MoE predictor is required, Step (f) of Algorithm 1 projects the online-to-batch average back onto the discretized Top-K class.

To control the complexity of the resulting gating family, we impose a regularity condition on the normalized transformation over any fixed active set. This condition requires that, once the selected experts are fixed, the normalized weights vary smoothly with the scores.

Definition 4.1. Let $M _ { R } \ \geq \ 2$ . A function $\phi : \mathbb { R } \to ( 0 , \infty )$ is called a regular score transformation if it is continuous, strictly increasing, and satisfies the following Lipschitz condition after normalization. For any subset $J \subset [ M _ { R } ]$ with $| J | = K$ , define

$$
G _ { J , m } ^ { \phi } ( z ) : = \frac { \phi ( z _ { m } ) \mathbb { 1 } \{ m \in J \} } { \sum _ { j \in J } \phi ( z _ { j } ) } , \qquad m \in [ M _ { R } ] ,
$$

and write $G _ { J } ^ { \phi } ( z ) = ( G _ { J , 1 } ^ { \phi } ( z ) , \ldots , G _ { J , M _ { R } } ^ { \phi } ( z ) ) ^ { \top }$ . There exist constants $L _ { \phi } > 0$ and $l _ { \phi } \geq 0$ independent of $M _ { R }$ and $^ { J , }$ such that, for all $z , z ^ { \prime } \in \mathbb { R } ^ { M _ { R } }$ R ,

$$
\| G _ { J } ^ { \phi } ( z ) - G _ { J } ^ { \phi } ( z ^ { \prime } ) \| _ { 2 } \leq L _ { \phi } M _ { R } ^ { l _ { \phi } } \| z - z ^ { \prime } \| _ { 2 } .\tag{4.1}
$$

Typical examples covered by this condition include the exponential transformation $\phi ( t ) = \exp ( \alpha t )$ with $\alpha > 0$ , which corresponds to softmax-type routing, and the sigmoid transformation $\phi ( t ) = 1 / ( 1 + \exp ( - t ) )$ , which is used in normalized-sigmoid MoE routing, such as DeepSeek-V3 (DeepSeek-AI et al., 2025), and has been studied theoretically in related work (Nguyen et al., 2025). More generally, the condition holds whenever $\phi$ is continuously diferentiable, strictly increasing, and has uniformly bounded relative derivative, namely $\operatorname* { s u p } _ { t \in \mathbb { R } } \phi ^ { \prime } ( t ) / \phi ( t ) < \infty$ . This relative-derivative condition ensures that, after normalization over any fixed active set, the gating weights are Lipschitz functions of the score vector.

To analyze the statistical properties of Top-K gating, we restrict attention to a subset of the parameter space where pairwise score diferences are nondegenerate. Since Top-K selection depends on the relative ordering of the scores, the following separation condition keeps the corresponding score boundaries well conditioned.

Definition 4.2. For $r _ { M _ { R } } \in ( 0 , 1 )$ , let $\widetilde { \Theta } _ { M _ { R } } : = \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } )$ be the subset of $\Theta _ { M _ { R } }$ consisting of parameters satisfying min $_ { i \neq j } \| \beta _ { i } - \beta _ { j } \| _ { 2 } \geq r _ { M _ { R } } .$ , where $\beta _ { m }$ is the slope component of the linear score $s _ { m } ( x ; \theta ) = x ^ { \top } \beta _ { m } + \alpha _ { m }$ , with $\beta _ { M _ { R } } = 0$ under the reference normalization $\theta _ { M _ { R } } = 0$

Remark 4.3. The separation condition rules out nearly parallel or degenerate score differences without preventing changes in the Top-K active set. Combined with the nondegeneracy condition on $P _ { X }$ below, it allows the probability of active-set changes under small parameter perturbations to be controlled. Thus $r _ { M _ { R } }$ serves as a boundary-stability parameter rather than merely an identifiability condition.

The associated Top-K gating class is $\mathcal { G } _ { K , \phi } : = \{ g ^ { ( K , \phi ) } ( \cdot ; \theta ) : \theta \in \widetilde { \Theta } _ { M _ { R } } \}$ . We equip this class with the vector-valued $L _ { 2 } ( P _ { X } )$ metric $d _ { K , \phi } ( \theta , \theta ^ { \prime } ) : = \| g ^ { ( K , \phi ) } ( \cdot ; \theta ) - g ^ { ( K , \phi ) } ( \cdot ; \theta ^ { \prime } ) \| _ { L _ { 2 } ( P _ { X } ) }$ To use the separation condition in Definition 4.2 to control active-set changes, define the “good” set $G _ { \theta , \theta ^ { \prime } } : = \{ x \in \mathcal { X } : \mathcal { T } _ { K } ( x ; \theta ) = \mathcal { T } _ { K } ( x ; \theta ^ { \prime } ) \}$ and the “bad” set $B _ { \theta , \theta ^ { \prime } } : = \mathcal { X } \backslash G _ { \theta , \theta ^ { \prime } }$ . On $G _ { \theta , \theta ^ { \prime } }$ , the Top-K active sets coincide, so the normalized gating weights vary smoothly with the scores under the regularity condition on $\phi .$ On $B _ { \theta , \theta ^ { \prime } }$ , the selected expert set changes, and the gating weights may jump. To control the probability of this bad set, we impose the following nondegeneracy condition on the distribution of X.

Assumption 4.4. There exists a constant $C _ { 4 } > 0$ such that, for every a $\in \mathbb { R } ^ { d }$ with $\| a \| _ { 2 } = 1$ every $b \in \mathbb { R }$ , and every $t > 0 , \mathbb { P } ( | a ^ { \top } X + b | \leq t ) \leq C _ { 4 } t$

This assumption rules out excessive probability mass near afine decision boundaries. It ensures that inputs lying close to these decision boundaries have probability proportional to the width of the boundary band. Without such a condition, small perturbations of θ could change the Top-K active set on a set with non-negligible probability, and the gating map need not be continuous in $L _ { 2 } ( P _ { X } )$ .

Under Assumption 4.4, the Top-K gating map satisfies the following $L _ { 2 } ( P _ { X } )$ continuity property.

Proposition 4.5. Suppose Assumption $4 . 4$ holds. Then, for any regular score transformation $\phi$ , there exists a constant $C _ { K } > 0$ depending on K, $L _ { \phi }$ and $C _ { 4 }$ , such that for all $\theta \in \widetilde { \Theta } _ { M _ { R } }$ and $\theta ^ { \prime } \in \Theta _ { M _ { R } }$ satisfying $\| \theta - \theta ^ { \prime } \| _ { 2 } \leq 1$ , we have

$$
\| g ^ { ( K , \phi ) } ( \cdot ; \theta ) - g ^ { ( K , \phi ) } ( \cdot ; \theta ^ { \prime } ) \| _ { L _ { 2 } ( P _ { X } ) } \leq C _ { K } d ^ { 1 / 4 } M _ { R } ^ { 1 + l _ { \phi } } \left( \frac { \| \theta - \theta ^ { \prime } \| _ { 2 } } { r _ { M _ { R } } } \right) ^ { 1 / 2 } .
$$

Thus, uniformly over separated θ and nearby $\theta ^ { \prime } { } _ { ; }$ the map $\theta \mapsto g ^ { ( K , \phi ) } ( \cdot ; \theta )$ admits a onesided local H¨older modulus of order $1 / 2$ from the Euclidean parameter metric $\| \cdot \| _ { 2 }$ to the distributional $L _ { 2 } ( P _ { X } )$ metric on the Top-K gating class $\mathcal { G } _ { K , \phi }$

Remark 4.6. The one-sided nature of the proposition is important. The target parameter $\theta$ is required to lie in the separated set $\widetilde { \Theta } _ { M _ { R } }$ , but the approximation parameter $\theta ^ { \prime }$ may lie in the ambient parameter box $\Theta _ { M _ { R } }$ . This is suficient because, for nearby $\theta ^ { \prime } .$ , active-set switches are confined to narrow bands around the decision boundaries induced by θ. It is also convenient for covering arguments, since the covering centers can be chosen from the ambient parameter space. As a consequence, the covering number of the Top-K gating class under $L _ { 2 } ( P _ { X } )$ can be controlled by the covering number of the normalized parameter space. Let $N ( \eta , \mathcal { G } _ { K , \phi } , L _ { 2 } ( P _ { X } ) )$ denote the η-covering number of $\mathcal { G } _ { K , \phi }$ . Then, under Assumption 3.4,

$$
N ( \eta , \mathcal { G } _ { K , \phi } , L _ { 2 } ( P _ { X } ) ) \leq \mathcal { O } \left[ \left( \frac { M _ { R } ^ { 5 / 2 + 2 l _ { \phi } } d \log M _ { R } } { \eta ^ { 2 } r _ { M _ { R } } } \right) ^ { ( M _ { R } - 1 ) ( d + 1 ) } \right] .
$$

As in the dense softmax analysis, the risk bound separates an oracle approximation term from a statistical learning term. The following theorem gives the corresponding bound for Top-K routing.

Theorem 4.7. Assume Assumptions 3.4–3.7 hold. In addition, suppose Assumption $4 . 4$ holds and $\phi$ is regular. Run Algorithm 1 with Top-K gates $g ^ { ( K , \phi ) }$ with $n _ { 1 } \asymp n _ { 2 } \asymp n _ { 3 }$ , and mesh size $\begin{array} { r } { \eta \asymp \frac { \sqrt { d } r _ { M _ { R } } } { n _ { 3 } M _ { R } ^ { 2 + 2 l _ { \phi } } } } \end{array}$ . Suppose that $M _ { S }$ is fixed. Then the online sequence $\{ \tilde { f } ^ { K , \phi , t } \} _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 }$ satisfies

$$
\frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \| \tilde { f } ^ { K , \phi , t } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \mathfrak { A } _ { K , \phi } ( M _ { S } , M _ { R } ) + \mathfrak { L } _ { \mathrm { s p a r s e } } ( M _ { R } , n , r _ { M _ { R } } ) \right\} ,\tag{4.2}
$$

where the Top-K oracle approximation risk is

$$
\mathfrak { A } _ { K , \phi } ( M _ { S } , M _ { R } ) : = \operatorname* { i n f } _ { \theta \in \tilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } ) } \left. f _ { 0 } - \sum _ { \ell = 1 } ^ { M _ { S } } f _ { S , \ell } ^ { * } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( K , \phi ) } ( \cdot ; \theta ) f _ { R , m } ^ { * } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } ,
$$

and the sparse-routing statistical learning cost is

$$
\mathfrak { L } _ { \mathrm { s p a r s e } } ( M _ { R } , n , r _ { M _ { R } } ) : = M _ { R } ( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } ) + \frac { M _ { R } d } { n } \left\{ \log ( M _ { R } d n ) + \log \left( \frac { 1 } { r _ { M _ { R } } } \right) \right\} ,\tag{4.3}
$$

where $\begin{array} { r } { \bar { a } _ { n _ { 2 } } ^ { 2 } = \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } a _ { t } ^ { 2 } } \end{array}$ and $\begin{array} { r } { \bar { a } _ { n _ { 3 } } ^ { 2 } = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } a _ { t } ^ { 2 } } \end{array}$ . Here C is independent of $n , M _ { R }$ and $r _ { M _ { R } }$ . The same bound holds for the risk $\mathbb { E } \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 }$ of the final projected estimator.

Theorem 4.7 parallels the dense softmax oracle inequality in Theorem 3.8. As before, $M _ { R } ( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } )$ is the cost inherited from learning the routed experts, while $( M _ { R } d / n ) \log ( M _ { R } d n )$ is the cost of estimating the gate over a growing parameter space. The additional contribution $\left( M _ { R } d / n \right) \log ( 1 / r _ { M _ { R } } )$ is specific to Top-K routing and reflects the cost of controlling boundary-switch events. Equivalently, $r _ { M _ { R } }$ acts as a boundary-stability parameter: smaller $r _ { M _ { R } }$ enlarges the admissible class and may reduce the approximation term, but it also makes the active-set map less stable and increases the statistical learning cost.

When $\phi ( t ) \ : = \ : \exp ( t )$ , the result specializes to Top-K softmax gating. In this case, the statistical learning cost matches that of dense softmax gating up to the additional logarithmic term $\log ( 1 / r _ { M _ { R } } )$ . Thus, under the stated regularity conditions, sparsification does not substantially increase the router-learning rate. At the same time, after computing the router scores, only K routed experts are activated rather than all $M _ { R }$ experts, reducing the per-input expert-evaluation cost.

## 4.2 Regionwise performance: hard selection versus soft aggregation

The oracle inequalities above compare dense softmax gating and sparse Top-K gating in terms of approximation and estimation error. They do not, however, by themselves explain when hard selection should be preferred to soft aggregation. We now give a regionwise interpretation of this distinction. The main message is that Top-1 routing is statistically attractive in a specialist regime, where each region has a dominant expert, whereas soft or multi-expert aggregation becomes necessary when the best local predictor is a mixture of several experts.

For clarity, we omit shared experts in this subsection and focus on the routed component. Let $f _ { m } ^ { * }$ denote the population benchmark of the m-th routed expert. The same reasoning applies in the presence of shared experts after subtracting the population shared-expert contribution from $f _ { 0 }$ . We first work in a stylized oracle setting in which the relevant performance regions are represented by a measurable partition of the input space.

Let $\{ \mathcal { X } _ { r } \} _ { r = 1 } ^ { M _ { 0 } }$ be a measurable partition of $\boldsymbol { \mathcal { X } } \subset \mathbb { R } ^ { d }$ , with the regions disjoint up to sets of $P _ { X } \mathrm { - m e a s u r e }$ zero. To relate this partition to Top-1 linear routing, we assume that the regions are linearly separable.

Assumption 4.8. Let $\bar { x } = ( x ^ { \top } , 1 ) ^ { \top }$ . There exist vectors $v _ { r } = ( \zeta _ { r } ^ { \top } , \psi _ { r } ) ^ { \top } \in \mathbb { R } ^ { d + 1 } , r \in [ M _ { 0 } ]$ such that, for each r, $\mathcal { X } _ { r } \ : = \ : \{ x \in \mathcal { X } : \bar { x } ^ { \top } v _ { r } > \bar { x } ^ { \top } v _ { j } , \forall j \ : \neq \ : r \}$ , up to a boundary set of $P _ { X } \mathrm { - m e a s u r e \ z e r o }$

Under Assumption 4.8, Top-1 linear routing can realize the oracle region assignment. We assume $M _ { R } \ge M _ { 0 }$ so that there are enough routed experts to represent the regions. We first consider a specialist regime in which each region admits a single strong expert, and then turn to regimes where local mixtures of experts can achieve strictly better performance.

## 4.2.1 Regionwise existence of a strong expert

We first consider the specialist regime. Fix a separating representation $v _ { r } = ( \zeta _ { r } ^ { \top } , \psi _ { r } ) ^ { \top }$ , $r \in [ M _ { 0 } ]$ , satisfying Assumption 4.8, normalized so that max $_ { \mathrm { \ell } } { } _ { : r \leq M _ { 0 } } \| v _ { r } \| _ { 2 } = 1$ . Assume that the slope components are separated: $r _ { 0 } : = \mathrm { m i n } _ { 1 \le r < s \le M _ { 0 } } \| \zeta _ { r } - \zeta _ { s } \| _ { 2 } > 0$ . In each region $\mathcal { X } _ { r }$ , suppose there is a routed expert that provides a good local approximation to $f _ { 0 }$ . After relabeling the routed experts if necessary, write this region-specific specialist as $f _ { r } ^ { * } , r \in [ M _ { 0 } ]$ As in Section 2.4, we use $\| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 }$ to denote its approximation error on $\mathcal { X } _ { r }$ . The following result specializes the Top-K oracle inequality to the case $K = 1$ . It shows that, when the regions are linearly separable and each region has a strong specialist, Top-1 routing can achieve the corresponding regionwise oracle risk.

Corollary 4.9. Assume the conditions of Theorem 4.7 hold with $K = 1$ . To rule out degenerate parameter spaces, assume in addition to Assumption $\it 3 . 4$ that, after the normalization $\theta _ { M _ { R } } = 0$ , there exists a constant $c _ { 0 } > 0$ such that the centered box $[ - c _ { 0 } , c _ { 0 } ] ^ { ( M _ { R } - 1 ) ( d + 1 ) }$ is contained in $\Theta _ { M _ { R } }$ . Assume further that the partition satisfies Assumption $\it 4 . 8$ and that $M _ { R } \ge M _ { 0 }$ . Let $\hat { f } _ { 0 }$ denote the estimator produced by Algorithm 1 with Top-1 routing. Then there exists a suficiently small constant $c > 0$ , depending only on X and $c _ { 0 }$ , such that for any $0 < r _ { M _ { R } } \leq c \{ r _ { 0 } \land M _ { R } ^ { - 1 / d } \}$ , we have

$$
\mathbb { E } \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \sum _ { r = 1 } ^ { M _ { 0 } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + \mathfrak { L } _ { \mathrm { s p a r s e } } ( M _ { R } , n , r _ { M _ { R } } ) \right\} ,
$$

where $\mathfrak { L } _ { \mathrm { s p a r s e } }$ is defined in Theorem $4 . 7 .$ The same bound holds for the online average of the Top-1 aggregate, and C is independent of $n , M _ { 0 } , M _ { R }$ , and $r _ { M _ { R } }$

The bound in Corollary 4.9 highlights the advantage of Top-1 routing in a pure specialist regime. If each region admits a strong local expert, then the regionwise approximation term $\begin{array} { r } { \sum _ { r = 1 } ^ { M _ { 0 } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } } \end{array}$ is small, and the remaining error is the statistical price of learning the routing rule and using evolving experts. The choice $r _ { M _ { R } } \asymp r _ { 0 } \wedge M _ { R } ^ { - 1 / d }$ uses the largest feasible separation order for which the regionwise assignment can be realized by a separated system of linear scores. The factor $M _ { R } ^ { - 1 / d }$ arises from separating the dummy experts. For larger separation levels, such a representation is not guaranteed, so the oracle approximation risk need not be bounded by the regionwise approximation term above. Since the dependence of $\mathfrak { L } _ { \mathrm { s p a r s e } }$ on $r _ { M _ { R } }$ is through $M _ { R } d \log ( 1 / r _ { M _ { R } } ) / n$ , this choice contributes $O ( M _ { R } d \log M _ { R } / n )$ when $r _ { 0 }$ is fixed.

Dense softmax gating behaves diferently in this setting. Since every routed expert receives positive weight when the scores are finite, dense softmax cannot represent the hard regionwise assignment exactly. Under the additional conditions in Supplement Section $\mathrm { F . 3 } { } _ { \cdot }$ the resulting approximation error is bounded by $O ( M _ { 0 } / \log M _ { R } )$ . Thus, when each region is dominated by a single specialist, hard Top-1 selection is more naturally aligned with the approximation structure.

## 4.2.2 Improved performance via multi-expert aggregation

We now consider the opposite regime, in which selecting a single expert is not suficient. In such settings, the best local predictor may be a nontrivial combination of several experts. For clarity, we focus on mixtures of two experts. For $r \in [ M _ { 0 } ]$ , we say that distinct indices $i _ { r 1 } , i _ { r 2 } \in [ M _ { R } ]$ and weights $w _ { r } = ( w _ { r 1 } , w _ { r 2 } ) ^ { \top } \in \Delta ^ { 1 }$ solve the best two-expert approximation problem on $\mathcal { X } _ { r }$ if

$$
\begin{array} { r l } & { \left\| f _ { 0 } - w _ { r 1 } f _ { i _ { r 1 } } ^ { * } - w _ { r 2 } f _ { i _ { r 2 } } ^ { * } \right\| _ { \mathcal { X } _ { r } } ^ { 2 } = \underset { j _ { 1 } , j _ { 2 } \in [ M _ { R } ] , j _ { 1 } \neq j _ { 2 } } { \operatorname* { m i n } } \left\| f _ { 0 } - w _ { 1 } f _ { j _ { 1 } } ^ { * } - w _ { 2 } f _ { j _ { 2 } } ^ { * } \right\| _ { \mathcal { X } _ { r } } ^ { 2 } . } \\ & { \quad \quad \quad \quad w = ( w _ { 1 } , w _ { 2 } ) ^ { \top } \in \Delta ^ { 1 } } \end{array}\tag{4.4}
$$

The following assumption formalizes a regionwise advantage of two-expert mixtures.

Assumption 4.10. For each region $\mathcal { X } _ { r }$ , there exist two distinct routed expert indices $i _ { r 1 } , i _ { r 2 } \in [ M _ { R } ]$ and weights $( w _ { r 1 } , w _ { r 2 } ) ^ { \top } \in \Delta ^ { 1 }$ that solve (4.4) on $\mathcal { X } _ { r }$ . Write $f _ { r , 1 } ^ { * } = f _ { i _ { r 1 } } ^ { * }$ and $f _ { r , 2 } ^ { * } = f _ { i _ { r 2 } } ^ { * }$ . For some $\delta _ { r } > 0$

$$
\int _ { \mathcal { X } _ { r } } \{ f _ { 0 } ( x ) - w _ { r 1 } f _ { r , 1 } ^ { * } ( x ) - w _ { r 2 } f _ { r , 2 } ^ { * } ( x ) \} ^ { 2 } d P _ { X } ( x ) \leq \int _ { \mathcal { X } _ { r } } \operatorname* { m i n } _ { k \in [ M _ { R } ] } \{ f _ { 0 } ( x ) - f _ { k } ^ { * } ( x ) \} ^ { 2 } d P _ { X } ( x ) - \delta _ { r } .
$$

The optimal expert pairs are disjoint across regions: $\{ i _ { r 1 } , i _ { r 2 } \} \cap \{ i _ { r ^ { \prime } 1 } , i _ { r ^ { \prime } 2 } \} = \emptyset$ for $r \neq r ^ { \prime }$

Assumption 4.10 requires the best two-expert approximation on each region to improve on the pointwise best single-expert oracle by at least $\delta _ { r }$ . The disjointness condition further describes a heterogeneous setting in which diferent regions rely on diferent pairs of experts. In such regimes, Top-1 gating is intrinsically limited. It assigns one routed expert to each input and therefore cannot form a nontrivial convex combination of experts at the same point. We formalize this limitation through the population Top-1 oracle approximation risk

$$
\mathfrak { A } _ { 1 } ( M _ { R } ) : = \operatorname* { i n f } _ { \theta \in \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } ) } \left. f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( 1 , \phi ) } ( \cdot ; \theta ) f _ { m } ^ { * } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Proposition 4.11. Suppose Assumption $\it 4 . 1 0$ holds. Then

$$
{ \mathfrak { A } } _ { 1 } ( M _ { R } ) \geq \sum _ { r = 1 } ^ { M _ { 0 } } \| f _ { 0 } - w _ { r 1 } f _ { r , 1 } ^ { * } - w _ { r 2 } f _ { r , 2 } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + \sum _ { r = 1 } ^ { M _ { 0 } } \delta _ { r } .
$$

Moreover, $i f$ Assumption 3.5 holds, let $\hat { f } _ { 0 }$ denote the final projected estimator produced by Algorithm 1 with Top-1 routing. Then

$$
\mathbb { E } \Vert \hat { f } _ { 0 } - f _ { 0 } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \geq \left[ \sqrt { \mathfrak { A } _ { 1 } ( M _ { R } ) } - C \sqrt { M _ { R } } \bar { a } _ { n _ { 3 } } \right] _ { + } ^ { 2 } ,
$$

where $[ u ] _ { + } = \operatorname* { m a x } \{ u , 0 \}$ and $C$ is independent of $n , \ M _ { 0 }$ , and $M _ { R }$ . In particular, when $M _ { R } \bar { a } _ { n _ { 3 } } ^ { 2 }$ is negligible relative to the approximation gap $\textstyle \sum _ { r = 1 } ^ { M _ { 0 } } \delta _ { r }$ , the estimator lower bound reduces to the lower bound for $\mathfrak { A } _ { 1 } ( M _ { R } )$

Thus, relative to the best regionwise two-expert mixture, the Top-1 function class incurs an irreducible approximation error of at least $\textstyle \sum _ { r = 1 } ^ { M _ { 0 } } \delta _ { r }$ . This approximation barrier is intrinsic to the Top-1 function class and arises in addition to any error due to estimation.

Soft aggregation behaves diferently. Dense softmax routing, and sparse Top-K routing with $K \geq 2$ , can assign nontrivial weights to multiple experts and can therefore approximate regionwise mixtures. Under suitable conditions, dense softmax gating yields an upper bound

of the form

$$
\mathbb { E } \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \lesssim \sum _ { r = 1 } ^ { M _ { 0 } } \| f _ { 0 } - w _ { r 1 } f _ { r , 1 } ^ { * } - w _ { r 2 } f _ { r , 2 } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + \mathfrak { L } _ { \mathrm { s o f t } } ( 0 , M _ { R } ) + \frac { M _ { 0 } } { \log M _ { R } } ,
$$

where ${ \mathfrak { L } } _ { \mathrm { s o f t } }$ is defined in Theorem 3.8; see Section F.3 of the Supplement. The first term is the oracle error of the best regionwise two-expert mixture, the second is the statistical learning cost, and the final term bounds the error from approximating the regionwise selection weights with dense softmax gating. This contrasts with Top-1 routing, whose excess term $\textstyle \sum _ { r } \delta _ { r }$ remains at the oracle approximation level whenever mixtures are strictly superior.

These results clarify the role of K in sparse MoE. When each region has a dominant specialist, hard Top-1 selection can be suficient. When prediction is locally compositional, however, activating multiple experts has a statistical role: it allows the model to represent local mixtures that cannot be captured by single-expert assignments. This interpretation is consistent with practical MoE design, where Top-1 routing is used in models such as Switch Transformer (Fedus et al., 2022), while architectures such as Mixtral (Jiang et al., 2024) and DeepSeekMoE (Dai et al., 2024) activate multiple experts per token.

## 5 Adaptation of the gating function

## 5.1 Performance-induced regions and gating approximation losses

The previous section compared hard selection and soft aggregation under stylized oracle partitions. In practice, however, the relevant regions are not known in advance and may have a geometry that is far more complex than a linearly separable partition. This raises a more basic question: how should the gating class be chosen so that it adapts to the regions induced by expert performance?

We first introduce an oracle device that describes where each routed expert is locally most accurate. For clarity, we omit shared experts in this section as before. In contrast to the prespecified linearly separable partition in Assumption 4.8, the definition below constructs an oracle partition directly from pointwise expert performance. The resulting regions provide a benchmark for assessing how well a gating class aligns with expert specialization.

Definition 5.1. For each $x \in \mathcal { X }$ , define

$$
r ^ { * } ( x ) : = \operatorname* { m i n } \left. r \in [ M _ { R } ] : r \in \arg \operatorname* { m i n } _ { j \in [ M _ { R } ] } | f _ { j } ^ { * } ( x ) - f _ { 0 } ( x ) | \right. ,
$$

where ties are resolved in favor of the smallest expert index. For each routed expert $r \in$ $[ M _ { R } ]$ , define $\mathcal { X } _ { r } : = \{ x \in \mathcal { X } : r ^ { * } ( x ) = r \}$

Let $\mathcal { G } = \{ g _ { \theta } ( \cdot ) = ( g _ { \theta , 1 } ( \cdot ) , \cdot \cdot \cdot , g _ { \theta , M _ { R } } ( \cdot ) ) ^ { \top } : \theta \in \Theta _ { M _ { R } } \}$ be a gating class, such as dense softmax gating, Top-K gating, or quadratic-softmax gating. For a generic $g \in { \mathcal { G } }$ , write $g _ { m }$ for its m-th component. After omitting shared experts, the oracle approximation term in Theorems 3.8 and 4.7 has the form

$$
\operatorname* { i n f } _ { g \in \mathcal { G } } \left\| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( \cdot ) f _ { m } ^ { * } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Define the oracle assignment map $\begin{array} { r } { g ^ { * } ( x ) : = \sum _ { r = 1 } ^ { M _ { R } } e _ { r } \mathbb { 1 } _ { \mathcal { X } _ { r } } ( x ) } \end{array}$ , where $e _ { r }$ is the r-th canonical basis vector in $\mathbb { R } ^ { M _ { R } }$ . This map assigns each input to the expert that is pointwise closest to $f _ { 0 }$ . When the routed experts are uniformly bounded, this approximation term can be related to the geometry of the performance-induced partition through the decomposition

$$
\operatorname* { i n f } _ { g \in \mathcal { G } } \left\| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( \cdot ) f _ { m } ^ { * } \right\| _ { L _ { 2 } \left( P _ { X } \right) } ^ { 2 } \leq C \left\{ \sum _ { r = 1 } ^ { M _ { R } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + \operatorname* { i n f } _ { g \in \mathcal { G } } \| g - g ^ { * } \| _ { L _ { 2 } \left( P _ { X } \right) } ^ { 2 } \right\} .
$$

Here $C$ is independent of $n$ and $M _ { R }$ . The first term is the oracle expert approximation error since $f _ { r } ^ { * }$ is the pointwise best routed expert on $\mathcal { X } _ { r }$ . Once this expert approximation error is fixed, the remaining term in $\begin{array} { r } { \mathrm { f } _ { g \in \mathcal { G } } \| g - g ^ { * } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } \end{array}$ measures how well the gating class can represent the corresponding oracle regionwise assignment.

This motivates the following global region-assignment loss:

$$
\mathcal { L } ^ { ( 1 ) } ( \mathcal { G } ) : = \operatorname* { i n f } _ { g \in \mathcal { G } } \| g - g ^ { * } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = \operatorname* { i n f } _ { g \in \mathcal { G } } \sum _ { r = 1 } ^ { M _ { R } } \| ( e _ { r } - g ) \mathbb { 1 } _ { \mathcal { X } _ { r } } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .\tag{5.1}
$$

Equivalently, this loss sums the squared discrepancy between g and the corresponding canonical assignment vector over all oracle regions. The quantity ${ \mathcal { L } } ^ { ( 1 ) } ( { \mathcal { G } } )$ measures how well the gating class can approximate the oracle assignment on the whole input space.

Near region boundaries, however, the identity of the best expert may be intrinsically unstable: two experts may have nearly identical approximation error, and small perturbations of either the experts or the input can change the oracle label. We therefore also consider an interior version of the loss. Fix $\rho > 0$ , define the shrunken region

$$
\mathcal { X } _ { r } ^ { - \rho } : = \{ x \in \mathcal { X } _ { r } : \mathrm { d i s t } ( x , \partial \mathcal { X } _ { r } ) \geq \rho \} , \qquad r \in [ M _ { R } ] ,\tag{5.2}
$$

where $\partial \mathcal { X } _ { r }$ denotes the boundary of $\mathcal { X } _ { r }$ and dist $( x , A )$ is the Euclidean distance from $x$ to a set A. The interior region-assignment loss is

$$
\mathcal L ^ { ( 2 ) } ( \mathcal G ; \rho ) : = \operatorname* { i n f } _ { g \in \mathcal G } \sum _ { r = 1 } ^ { M _ { R } } \| ( e _ { r } - g ) \mathbb { 1 } _ { \chi _ { r } ^ { - \rho } } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .\tag{5.3}
$$

Thus, $\mathcal { L } ^ { ( 2 ) } ( \mathcal { G } ; \rho )$ measures how well the gating class approximates the oracle assignment on region interiors, while deliberately ignoring a $\rho \mathrm { - }$ neighborhood of the boundaries. The two losses capture complementary aspects of gating performance: $\mathcal { L } ^ { ( 1 ) } ( \mathcal { G } )$ evaluates global alignment including boundary behavior, whereas $\mathcal { L } ^ { ( 2 ) } ( \mathcal { G } ; \rho )$ focuses on whether the gate makes the correct assignments away from ambiguous boundary regions.

## 5.2 Linear and quadratic gating: examples of geometric alignment

We use two simple examples to illustrate the roles of $\mathcal { L } ^ { ( 1 ) }$ and $\boldsymbol { \mathcal { L } } ^ { ( 2 ) }$ . For concreteness, in addition to Assumption 3.4, we assume that each free gating parameter in these examples is restricted to $[ - C$ log $M _ { R } ,$ C log $M _ { R } ]$ for some constant $C > 1 / 2$ . The first example isolates boundary smoothing from interior assignment, while the second shows how the interior loss reflects geometric alignment between the gating class and the oracle partition. Details are given in Supplement Sections D.2 and D.3.

Example 5.2 (Global versus interior region-assignment loss). Consider $\chi = [ - 1 , 1 ]$ with $X \sim \mathrm { U n i f } [ - 1 , 1 ]$ . The two oracle regions are [−1, 0] and $( 0 , 1 ]$ , with oracle assignments $e _ { 1 }$ and $e _ { 2 }$ , respectively. Top-1 linear gating can implement the threshold at zero exactly, and hence $\mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \mathrm { T o p - 1 } } ) = 0$ and $\mathcal { L } ^ { ( 2 ) } ( \mathcal { G } _ { \mathrm { T o p - 1 } } ; \rho ) = 0$ . Dense linear-softmax gating with bounded scores cannot represent this discontinuous assignment exactly. Under the logarithmically growing parameter box in Assumption 3.4, its global boundary-smoothing error satisfies $\begin{array} { r } { \mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \mathrm { s o f t } } ) \asymp \frac { 1 } { \log M _ { R } } } \end{array}$ . In contrast, once a fixed ρ-neighborhood of the boundary is removed, dense softmax can create a large score margin on the remaining interior regions, so that $\mathcal { L } ^ { ( 2 ) } ( \mathcal { G } _ { \mathrm { s o f t } } ; \rho ) \lesssim M _ { R } ^ { - c \rho }$ for some constant $c > 0$ . Thus $\mathcal { L } ^ { ( 1 ) }$ captures the cost of smoothing sharp boundaries, whereas $\boldsymbol { \mathcal { L } } ^ { ( 2 ) }$ measures assignment accuracy away from the boundary.

Example 5.3 (Linear-softmax versus quadratic-softmax). The interior loss also reflects whether the gating geometry matches the oracle partition. Linear score diferences generate afine decision boundaries, whereas quadratic score diferences can generate curved boundaries. For a partition of a disk into an inner ball and an outer annulus, the oracle boundary is circular. Hence linear-softmax remains geometrically misspecified: for any fixed $\rho > 0 , \mathcal { L } ^ { ( 2 ) } ( \mathcal { G } _ { \mathrm { l i n } } ; \rho ) \ge C _ { L }$ for some constant $C _ { L } > 0$ , independent of $M _ { R }$ . By contrast, quadratic-softmax can generate radial score diferences, and under the same logarithmic parameter scaling, satisfies $\mathcal { L } ^ { ( 2 ) } ( \mathcal { G } _ { \mathrm { q u a d } } ; \rho ) \ : \leq \ : C _ { R } M _ { R } ^ { - \alpha }$ for some constants $C _ { R } , \alpha > 0$ , independent of $M _ { R }$ . Hence, $\mathcal { L } ^ { ( 2 ) }$ detects geometric mismatch: linear gating is adequate for afine partitions, while quadratic gating can substantially improve approximation for curved partitions.

Taken together, these examples show that $\mathcal { L } ^ { ( 1 ) }$ and $\mathcal { L } ^ { ( 2 ) }$ emphasize diferent aspects of gating approximation. The global loss penalizes boundary smoothing, while the interior loss focuses on assignment away from boundary layers and reveals geometric mismatch between the gating class and the expert-induced partition.

## 5.3 Gaussian-kernel-based gating function

The preceding examples show that the efectiveness of a gating class depends on its geometric alignment with the expert-induced regions. When the geometry of these regions cannot be captured by a given parametric score class, the resulting gating approximation error may remain nonnegligible. This motivates a more flexible construction whose approximation ability is not tied to a prespecified boundary geometry. We now introduce the Gaussian-kernel-based gating class that can approximate regionwise hard assignments and soft mixtures over partitions with controlled boundary complexity.

Following Definition 5.1, let $\mathcal { X } = [ 0 , 1 ] ^ { d }$ be partitioned into $M _ { R }$ regions $\{ \mathcal { X } _ { r } \} _ { r = 1 } ^ { M _ { R } }$ . To quantify the contribution of boundary layers to the kernel approximation error, we impose the following regularity condition on the partition.

Definition 5.4. Let $\mathcal { P } _ { M _ { R } } ~ = ~ \{ \mathcal { X } _ { i } \} _ { i = 1 } ^ { M _ { R } }$ be a measurable partition of $[ 0 , 1 ] ^ { d }$ . Recall the shrunken regions $\mathcal { X } _ { i } ^ { - \rho }$ defined in (5.2), and for $\rho > 0$ define $B _ { i } ( \rho ) : = \mathscr { X } _ { i } \backslash \mathscr { X } _ { i } ^ { - \rho }$ . We say that $\mathcal { P } _ { M _ { R } }$ has boundary-layer regularity if there exist $\rho _ { 0 } > 0$ and constants $\gamma _ { 1 } , . . . , \gamma _ { M _ { R } } < \infty$ such that $\mu ( B _ { i } ( \rho ) ) \leq \gamma _ { i } \rho$ for all $i \in [ M _ { R } ]$ and $0 < \rho \le \rho _ { 0 }$ , where $\mu$ denotes the Lebesgue measure on X. We write $\begin{array} { r } { \Gamma ( \mathcal { P } _ { M _ { R } } ) : = \sum _ { i = 1 } ^ { M _ { R } } \gamma _ { i } } \end{array}$ for the total boundary-layer complexity of the partition, and use $\Gamma _ { M _ { R } }$ as shorthand when the partition is clear.

This condition rules out pathological partitions whose boundary layers have disproportionately large volume. The quantity $\Gamma _ { M _ { R } }$ summarizes the cumulative boundary complexity of the partition. It may depend on $M _ { R }$ , so the approximation bound below explicitly records the cost of smoothing increasingly refined partitions.

Unlike the canonical assignment $x \mapsto e _ { r } .$ , we allow each region to be associated with a general target weight vector $w _ { r } ^ { * } = ( w _ { r , 1 } ^ { * } , \ldots , w _ { r , M _ { R } } ^ { * } ) ^ { \top } \in \Delta ^ { M _ { R } - 1 } , r \in [ M _ { R } ]$ . Thus, on region $\mathcal { X } _ { r }$ , the ideal gating target need not select a single routed expert; it may assign an arbitrary convex combination of the $M _ { R }$ routed experts.

We construct the gating function using Gaussian kernels centered on a fixed, dataindependent grid. Let $h > 0$ denote the mesh width of the grid, and define

$$
\mathcal C = \{ ( k _ { 1 } h , \dots , k _ { d } h ) : k _ { \ell } \in \{ 0 , 1 , \dots , \lfloor h ^ { - 1 } \rfloor \} \} \cap [ 0 , 1 ] ^ { d } .
$$

The number of kernel centers is of order $h ^ { - d }$ . Thus h controls the approximation power of the kernel grid, while through the number of centers it also determines the statistical complexity of the gating class.

For each region $\mathcal { X } _ { r }$ , let $C _ { r } = \mathcal { C } \cap \mathcal { X } _ { r }$ denote the set of grid centers lying in that region. Define the Gaussian kernel

$$
\varphi ( x , c ; \sigma ) = \exp \left\{ - \frac { \| x - c \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} , \qquad \sigma = \kappa h , \qquad \kappa \in \left[ \frac 1 2 , 1 \right] .
$$

The parameter space is

$$
\Theta _ { \mathrm { k e r } } = \left\{ w = ( w _ { j , c } ) _ { 1 \leq j \leq M _ { R } , \ c \in \mathcal { C } } : w _ { j , c } \geq 0 , \ \sum _ { j = 1 } ^ { M _ { R } } w _ { j , c } = 1 , \ \forall c \in \mathcal { C } \right\} .
$$

The induced Gaussian-kernel gating class is

$$
\mathcal { G } _ { \mathrm { k e r } } = \{ g ( \cdot ; w ) = ( g _ { 1 } ( \cdot ; w ) , \dots , g _ { M _ { R } } ( \cdot ; w ) ) ^ { \top } : w \in \Theta _ { \mathrm { k e r } } \} ,
$$

where

$$
g _ { j } ( x ; w ) = \frac { \sum _ { c \in \mathcal { C } } w _ { j , c } \varphi ( x , c ; \sigma ) } { \sum _ { \ell = 1 } ^ { M _ { R } } \sum _ { c \in \mathcal { C } } w _ { \ell , c } \varphi ( x , c ; \sigma ) } , \qquad j \in [ M _ { R } ] .
$$

Equivalently, each grid center c carries a local weight vector in $\Delta ^ { M _ { R } - 1 }$ . The Gaussian kernels interpolate these local weights across the input space, and the normalization in $g _ { j } ( x ; w )$ converts the kernel-weighted averages into a valid gating vector. The key point is that this construction can approximate not only hard region assignments, but also arbitrary regionwise target weight vectors.

To see the approximation mechanism, consider the following oracle choice

$$
w _ { j , c } = \sum _ { r = 1 } ^ { M _ { R } } w _ { r , j } ^ { * } \mathbb { 1 } _ { \{ c \in C _ { r } \} } , \qquad j \in [ M _ { R } ] , \qquad c \in \mathcal { C } .
$$

Under this assignment, each center $c \in C _ { r }$ carries the target weight vector $\boldsymbol { w } _ { r } ^ { * }$ . Since $w _ { r } ^ { * } \in \Delta ^ { M _ { R } - 1 }$ , this particular construction satisfies $\textstyle \sum _ { j = 1 } ^ { M _ { R } } w _ { j , c } = 1 , c \in { \mathcal { C } }$ . Thus, this oracle choice belongs to $\Theta _ { \mathrm { k e r } }$ . For this choice of $w .$ , the resulting gating function simplifies to

$$
g _ { j } ( x ; w ) = \frac { \sum _ { r = 1 } ^ { M _ { R } } w _ { r , j } ^ { * } \sum _ { c \in C _ { r } } \varphi ( x , c ; \sigma ) } { \sum _ { c \in \mathcal { C } } \varphi ( x , c ; \sigma ) } .
$$

This formula has a simple geometric interpretation. If x lies in the interior of $\mathcal { X } _ { r }$ , then the Gaussian kernels concentrate their weight on nearby centers in $C _ { r }$ , while contributions from centers in other regions become negligible. Consequently, $g ( x ; w ) \approx w _ { r } ^ { * }$ away from the region boundaries. Accordingly, we define $\mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \mathrm { k e r } } )$ and $\mathcal { L } ^ { ( 2 ) } ( \mathcal { G } _ { \mathrm { k e r } } ; \rho )$ as in (5.1) and (5.3), respectively, with the canonical vectors $e _ { r }$ replaced by the general target weights $\boldsymbol { w } _ { r } ^ { * }$

Theorem 5.5. Assume that C is a uniform grid with mesh width $h ,$ and that the density p<sub>X</sub> of $P _ { X }$ is continuous on $[ 0 , 1 ] ^ { d }$ . Then, for any measurable partition $\mathcal { P } _ { M _ { R } } = \{ \mathcal { X } _ { i } \} _ { i = 1 } ^ { M _ { R } }$ of $[ 0 , 1 ] ^ { d }$ satisfying the boundary-layer regularity condition in Definition $5 . \not \langle 4 ;$ , any target weights

$w _ { 1 } ^ { * } , \ldots , w _ { M _ { R } } ^ { * } \in \Delta ^ { M _ { R } - 1 }$ , and any h satisfying

$$
0 < h < \exp \left\{ - \frac { 1 } { d / 2 + 1 } \operatorname* { m a x } \left\{ 1 , \frac { d } { 1 6 \kappa ^ { 2 } } \right\} \right\} ,
$$

the Gaussian-kernel-based gating class satisfies

$$
\begin{array} { r } { \mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \mathrm { k e r } } ) \leq C \left\{ M _ { R } h ^ { d + 2 } + \Gamma _ { M _ { R } } h \sqrt { \log ( 1 / h ) } \right\} , } \end{array}
$$

where C is a constant independent of h, $\Gamma _ { M _ { R } }$ and $M _ { R }$

Theorem 5.5 shows that $\mathcal { G } _ { \mathrm { k e r } }$ can approximate arbitrary regionwise target weights over partitions with controlled boundary layers. The first term reflects the interior approximation error from the Gaussian-kernel grid, while the second term reflects the boundary smoothing error. In particular, the boundary contribution is governed by $\Gamma _ { M _ { R } } .$ , which may grow with $M _ { R }$ . The canonical hard-assignment case is recovered by taking $w _ { r } ^ { \ast } = e _ { r }$ , so the result extends regionwise expert selection to regionwise expert mixing.

Remark 5.6 (Resolution and learning cost). If max<sub>1</sub> $\leq i \leq M _ { R } \gamma _ { i } = O ( 1 )$ , then $\Gamma _ { M _ { R } } = { \cal O } ( M _ { R } )$ Taking, for example, $h \asymp M _ { R } ^ { - 2 }$ yields

$$
M _ { R } h ^ { d + 2 } + \Gamma _ { M _ { R } } h \sqrt { \log ( 1 / h ) } \lesssim \frac { \sqrt { \log M _ { R } } } { M _ { R } } .
$$

Thus, a suficiently fine kernel grid yields a vanishing oracle approximation error. At the same time, a smaller h requires more kernel centers and hence more parameters, which increases the statistical learning cost. Therefore, in estimation problems, the choice of h should balance the oracle approximation error against the statistical cost of learning the enlarged gating class.

The preceding theorem evaluates the global loss and therefore includes a boundary-layer contribution. Restricting attention to points that stay a fixed distance away from the region boundaries removes this contribution and yields the sharper interior bound below.

Corollary 5.7. Under the same conditions as in Theorem 5.5, fix a constant $\rho > 0$ . Then,

for any h satisfying $\textstyle 0 < h < \operatorname* { m i n } \left\{ { \frac { 2 \rho } { \sqrt { d } } } , { \frac { \rho } { 2 \kappa } } \right\}$ , the Gaussian-kernel-based gating class satisfies

$$
\mathcal { L } ^ { ( 2 ) } ( \mathcal { G } _ { \mathrm { k e r } } ; \rho ) \leq C M _ { R } \exp \left\{ - \frac { \rho ^ { 2 } } { 2 \kappa ^ { 2 } h ^ { 2 } } \right\} ,
$$

where C is a constant independent of h and $M _ { R }$

Remark 5.8. Compared with the global loss $\mathcal { L } ^ { ( 1 ) }$ , the interior bound eliminates the boundarylayer term $\Gamma _ { M _ { R } } h \sqrt { \log ( 1 / h ) }$ . For fixed $\rho > 0$ , the remaining approximation error decays exponentially in $h ^ { - 2 }$ , up to the multiplicative factor $M _ { R }$

The preceding analysis shows that the approximation ability of a gating class depends on how well it captures the geometry of expert-induced regions. To illustrate this point in finite samples, Table 1 compares linear, quadratic and Gaussian-kernel-based gating under partitions with diferent boundary geometries. In each design, the lowest gating weight errors are attained by the gating class most closely aligned with the geometry of the underlying partition. This pattern illustrates the role of geometric alignment in the choice of gating class; see Supplement Section G.1 for simulation details and visualizations of the region partitions.

Table 1: Monte Carlo performance of gating classes under three DGPs
<table><tr><td rowspan="2">Gating class</td><td colspan="3">Linear DGP</td><td colspan="3">Quadratic DGP</td><td colspan="3">Nonlinear DGP</td></tr><tr><td>n</td><td> $\ell _ { 1 }$  error (SD)</td><td> $\ell _ { 2 } ^ { 2 }$  error (SD)</td><td>n</td><td> $\ell _ { 1 }$  error (SD)</td><td> $\ell _ { 2 } ^ { 2 }$  error (SD)</td><td>n</td><td> $\ell _ { 1 }$  error (SD)</td><td> $\ell _ { 2 } ^ { 2 }$  error (SD)</td></tr><tr><td rowspan="2"> $\mathcal { G } _ { \mathrm { l i n } }$ </td><td>200</td><td>0.100 (0.022)</td><td>0.006 (0.003)</td><td>200</td><td>0.230 (0.033)</td><td>0.030 (0.012)</td><td>500</td><td>0.745 (0.062)</td><td>0.339 (0.051)</td></tr><tr><td>400</td><td>0.081 (0.023)</td><td>0.004 (0.002)</td><td>400</td><td>0.220 (0.019)</td><td>0.027 (0.006)</td><td>1000</td><td>0.713 (0.033)</td><td>0.322 (0.025)</td></tr><tr><td rowspan="2"> $\mathcal { G } _ { \mathrm { q u a d } }$ </td><td>200</td><td>0.178 (0.045)</td><td>0.018 (0.011)</td><td>200</td><td>0.141 1 (0.062)</td><td>0.014 (0.016)</td><td>500</td><td>0.840 (0.041)</td><td>0.352 (0.035)</td></tr><tr><td>400</td><td>0.151 (0.040)</td><td>0.013 (0.007)</td><td>400</td><td>0.133 (0.061)</td><td>0.013 (0.013)</td><td>1000</td><td>0.821 (0.030)</td><td>0.339 (0.025)</td></tr><tr><td rowspan="2"> $\mathcal { G } _ { \mathrm { k e r } }$ </td><td>200</td><td>0.211 (0.052)</td><td>0.026 (0.013)</td><td>200</td><td>0.204 (0.042)</td><td>0.023 (0.011)</td><td>500</td><td>0.711 (0.028)</td><td>0.267 (0.021)</td></tr><tr><td>400</td><td>0.188 (0.043)</td><td>0.020 (0.009)</td><td>400</td><td>0.181 (0.027)</td><td>0.018 (0.006)</td><td>1000</td><td>0.713 (0.019)</td><td>0.266 (0.014)</td></tr></table>

Note: Linear, quadratic and nonlinear DGPs refer to region partitions with linear, quadratic and nonlinear boundaries, respectively. The gating classes $\mathcal { G } _ { \mathrm { l i n } } , \mathcal { G } _ { \mathrm { q u a d } }$ and $\mathcal { G } _ { \mathrm { k e r } }$ denote dense softmax gating with linear scores, dense softmax gating with quadratic scores and Gaussian-kernel-based gating, respectively. The $\ell _ { 1 }$ error is the sample average of the $\ell _ { 1 }$ distance between the estimated and true gating-weight vectors, while the $\ell _ { 2 } ^ { 2 }$ error is the sample average of their squared Euclidean distance. Reported values are Monte Carlo means, with standard deviations in parentheses. For each DGP and sample size, the smallest value of each error measure across gating classes is shown in bold. The results show that gating performance is closely related to the geometric alignment between the gating class and the underlying region partition: linear, quadratic and Gaussian-kernel-based gating perform best under linear, quadratic and nonlinear designs, respectively.

## 5.4 Selection procedure for diferent classes of gating functions

The preceding subsections show that the performance of a gating rule depends on its alignment with the geometry of the expert-induced regions. In practice, neither the relative performance of the experts nor the induced region partition is known a priori. Although the Gaussian-kernel construction provides a flexible approximation template, it may not always be the most eficient choice: a simpler linear or quadratic gate can be preferable when the region geometry is correspondingly simple. This motivates a data-adaptive procedure that selects among candidate gating classes according to their validation performance.

Suppose that we are given $L \geq 2$ candidate gating classes $\{ \mathcal { G } _ { \ell } \} _ { \ell = 1 } ^ { L }$ . For each class $\mathcal { G } _ { \ell } .$ define the corresponding population MoE oracle class

$$
{ \mathcal { F } } _ { \ell } ^ { * } = \left\{ x \mapsto \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( x ) f _ { m } ^ { * } ( x ) : g = \left( g _ { 1 } , \ldots , g _ { M _ { R } } \right) ^ { \top } \in { \mathcal { G } } _ { \ell } \right\} .
$$

Write $\begin{array} { r } { \mathfrak { A } _ { \ell } : = \operatorname* { i n f } _ { f \in \mathcal { F } _ { \ell } ^ { \ast } } \| f _ { 0 } - f \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } \end{array}$ for the oracle approximation risk of the ℓ-th class. Using Algorithm 1, or another class-specific estimation method with the same type of oracle guarantee, we construct a final projected estimator $\check { f } _ { \ell }$ for each $\ell \in [ L ]$ . We assume a common envelope bound max $_ { \ell \in [ L ] } \| \check { f } _ { \ell } \| _ { \infty } \leq A _ { 0 }$ , where $A _ { 0 }$ is independent of n and L. We also assume that these candidate estimators satisfy the uniform class-specific bound

$$
\begin{array} { r } { \mathbb { E } \| f _ { 0 } - \check { f } _ { \ell } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C _ { \ell } \left\{ \mathfrak { A } _ { \ell } + \mathfrak { L } _ { \ell } \right\} , \qquad \ell \in [ L ] , } \end{array}\tag{5.4}
$$

where ${ \mathfrak { L } } _ { \ell }$ denotes the statistical learning cost associated with the gating class $\mathcal { G } _ { \ell }$ , and the constants $C _ { \ell }$ are uniformly bounded by a constant independent of n and L. This condition summarizes the class-specific results established in the previous sections. For example, a dense softmax candidate has $\mathfrak { A } _ { \ell } = \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } )$ and $\mathfrak { L } _ { \ell } = \mathfrak { L } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } )$ , whereas a Top-K candidate has $\mathfrak { A } _ { \ell } = \mathfrak { A } _ { K , \phi } ( M _ { S } , M _ { R } )$ and $\mathfrak { L } _ { \ell } = \mathfrak { L } _ { \mathrm { s p a r s e } } ( M _ { R } , n , r _ { M _ { R } } )$ , with the sample sizes in these learning costs replaced by the corresponding block sizes inside the estimation sample.

To select the best-performing estimator among the candidates, we retain the aggregationand-projection principle of Algorithm 1, but adapt the procedure to aggregate the candidate estimators using an independent sample and project the aggregate back onto the candidate list, as presented in Algorithm 2.

We now analyze the selected estimator and show that the procedure adapts to the unknown gating structure by nearly matching the best candidate class, up to a logarithmic selection cost. We first present the result in a global $L _ { 2 } ( P _ { X } )$ risk framework and then derive a regionwise counterpart that explicitly separates the expert approximation error from the gating approximation error. We begin with the global case.

Algorithm 2: Aggregation-projection over gating-induced MoE classes   
Input: A data set $\overline { { \mathcal { D } _ { \mathrm { s e l } } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n } } }$ reserved for candidate construction and class   
selection; candidate classes $\{ \mathcal { G } _ { \ell } \} _ { \ell = 1 } ^ { L }$   
Output: Selected gating class $\mathcal { G } _ { \ell ^ { * } }$ and final estimator $\check { f } _ { \ell ^ { * } }$   
(a) Sample splitting. Randomly permute and relabel the observations, and split the resulting   
sample into three independent blocks:   
$Z ^ { ( 1 ) } = ( X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n _ { 1 } } , \qquad Z ^ { ( 2 ) } = ( X _ { i } , Y _ { i } ) _ { i = n _ { 1 } + 1 } ^ { n _ { 1 } + n _ { 2 } } , \qquad Z ^ { ( 3 ) } = ( X _ { i } , Y _ { i } ) _ { i = n _ { 1 } + n _ { 2 } + 1 } ^ { n } ,$   
and define $n _ { 3 } = n - n _ { 1 } - n _ { 2 } .$   
(b) Candidate construction. For each candidate class $\mathcal { G } _ { \ell } ,$ use $Z ^ { ( 1 ) }$ to construct a final   
projected estimator $\check { f } _ { \ell }$ satisfying the class-specific oracle bound   
$\begin{array} { r } { \mathbb { E } \| f _ { 0 } - \check { f } _ { \ell } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C _ { \ell } \left\{ \mathfrak { A } _ { \ell } + \mathfrak { L } _ { \ell } \right\} , \qquad \ell \in [ L ] . } \end{array}$   
(c) Calibration and aggregation. Conditional on the fitted list $\{ \check { f } _ { \ell } \} _ { \ell = 1 } ^ { L }$ , compute the   
calibration residual variance   
$\tilde { \sigma } _ { \ell } ^ { 2 } = \frac { 1 } { n _ { 2 } } \sum _ { \substack { \ell = - \tilde { \sigma } _ { \ell } \left( X _ { i } \right) } } \big ( Y _ { i } - \tilde { f } _ { \ell } ( X _ { i } ) \big ) ^ { 2 } , \qquad \hat { \sigma } _ { \ell } ^ { 2 } = \operatorname* { m i n } \left\{ \bar { \sigma } ^ { 2 } , \operatorname* { m a x } \left\{ \underline { { \sigma } } ^ { 2 } , \tilde { \sigma } _ { \ell } ^ { 2 } \right\} \right\} , \qquad \ell \in [ L ] .$   
(X<sub>i</sub>,Y<sub>i</sub>)∈Z<sup>(2)</sup>   
Initialize $\hat { \omega } _ { \ell , n _ { 1 } + n _ { 2 } } = 1 / L$ for $\ell \in [ L ]$ . For $t = n _ { 1 } + n _ { 2 } , \ldots , n - 1$ , first form the one-step-ahead   
weighted predictor   
$\tilde { f } ^ { t } ( x ) = \sum _ { \ell = 1 } ^ { L } \hat { \omega } _ { \ell , t } \check { f } _ { \ell } ( x ) ,$   
and then, after observing $( X _ { t + 1 } , Y _ { t + 1 } )$ , update the aggregation weights by   
$\hat { \omega } _ { \ell , t + 1 } = \frac { \hat { \omega } _ { \ell , t } \hat { \sigma } _ { \ell } ^ { - 1 } h _ { \varepsilon , 0 } \{ \left( Y _ { t + 1 } - \check { f } _ { \ell } ( X _ { t + 1 } ) \right) / \hat { \sigma } _ { \ell } \} } { \sum _ { r = 1 } ^ { L } \hat { \omega } _ { r , t } \hat { \sigma } _ { r } ^ { - 1 } h _ { \varepsilon , 0 } \{ \left( Y _ { t + 1 } - \check { f } _ { r } ( X _ { t + 1 } ) \right) / \hat { \sigma } _ { r } \} } .$   
Define the aggregated predictor by the online-to-batch average   
$\tilde { f } _ { n } ( x ) = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \tilde { f } ^ { t } ( x ) = \sum _ { \ell = 1 } ^ { L } \bar { \omega } _ { \ell } \tilde { f } _ { \ell } ( x ) , \qquad \bar { \omega } _ { \ell } = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \hat { \omega } _ { \ell , t } .$   
(d) Projection and output. Project the aggregate $\tilde { f } _ { n }$ back to the candidate list by the   
population $L _ { 2 } ( P _ { X } )$ criterion,   
$\ell ^ { * } = \mathop { \arg \operatorname* { m i n } } _ { \ell \in [ L ] } \| \check { f } _ { \ell } - \tilde { f } _ { n } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .$   
Output the selected gating class $\mathcal { G } _ { \ell ^ { * } }$ and the final estimator $\check { f } _ { \ell ^ { * } }$

Theorem 5.9. Suppose Assumptions 3.6 and 3.7 hold. Assume that the candidate estimators constructed in Algorithm $\mathcal { Q }$ satisfy (5.4). If $n _ { 1 } \asymp n _ { 2 } \asymp n _ { 3 } \asymp n ,$ then the selected estimator $\check { f } _ { \ell ^ { * } }$ satisfies

$$
\mathbb { E } \Vert f _ { 0 } - \check { f } _ { \ell ^ { * } } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left[ \operatorname* { i n f } _ { \ell \in [ L ] } \{ \mathfrak { A } _ { \ell } + \mathfrak { L } _ { \ell } \} + \frac { \log L } { n } \right] ,
$$

where $C$ is a constant independent of n and $L$

The leading term inf $\dot { \boldsymbol { \ell } } { \in } [ L ] \left\{ \mathfrak { A } _ { \ell } + \mathfrak { L } _ { \ell } \right\}$ is the best achievable risk among the candidate gating-induced MoE classes. It balances the approximation power of each gating class with its statistical learning cost. The additive term (log $L ) / n$ is the price of adapting over L candidate classes.

The same result can be expressed in regionwise form.

Corollary 5.10. Under the conditions of Theorem 5.9, and with the expert-wise optimal regions defined as in Definition 5.1, the selected estimator $\check { f } _ { \ell ^ { * } }$ satisfies

$$
\mathbb { E } \Vert f _ { 0 } - \check { f } _ { \ell ^ { * } } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left[ \sum _ { r = 1 } ^ { M _ { R } } \Vert f _ { 0 } - f _ { r } ^ { * } \Vert _ { \mathcal { X } _ { r } } ^ { 2 } + \operatorname* { i n f } _ { \ell \in [ L ] } \left\{ \mathscr { L } ^ { ( 1 ) } ( \mathscr { G } _ { \ell } ) + \mathfrak { L } _ { \ell } \right\} + \frac { \log L } { n } \right] ,
$$

where $\mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \ell } )$ is defined in (5.1) and C is a constant independent of n and $L$

This regionwise inequality separates three components: the approximation error of the population experts on their locally optimal regions, the best achievable gating approximation error among the candidate classes, and the logarithmic cost of selecting the class from data. Thus, the procedure adapts to the unknown geometry of expert performance by choosing a gating class whose approximation complexity and statistical learning cost are well balanced.

## 6 The role of shared experts

The previous sections allowed for shared experts as given components, without isolating their statistical contribution. We now examine what is gained by combining a shared expert with routed experts. In contrast to routed experts, shared experts remain active for all inputs and are designed to capture information that is useful across diferent regions of the input space. This architectural idea is particularly prominent in recent large-scale models such as DeepSeekMoE (Dai et al., 2024), where shared experts are introduced to reduce redundancy among routed experts and to encourage local specialization. Recent statistical work has also begun to analyze the sample-eficiency benefit of shared experts in DeepSeek-type architectures (Nguyen et al., 2025). Our analysis is complementary: instead of studying full parameter convergence in a specified MoE model, we focus on how the presence of a shared component changes the statistical learning problem faced by the routed experts.

In this section, we study the statistical role of shared experts from a simplified but revealing perspective. Rather than analyzing the full joint training dynamics of experts and the router, which is highly non-convex, we fix the routing partition and the regional learning rules and compare the estimation problems induced by shared-routed and purerouted representations. This comparison isolates the statistical efect of introducing an explicit shared component.

The resulting comparison is learner-specific. The benefit of a shared expert does not require the shared component to be simpler in an absolute sense. Rather, the benefit appears when the common component is better matched to the shared learning rule than to the routed learning rules. In that case, separating common from region-specific structure can reduce the efective complexity of the targets assigned to the routed experts and improve estimation eficiency.

## 6.1 Model and learning rules

Let $\mathcal { X } = \mathcal { X } _ { 1 } \cup \mathcal { X } _ { 2 }$ be a fixed Top-1 routing partition with $\chi _ { 1 } \cap \chi _ { 2 } = \emptyset$ . Suppose that $Y _ { i } =$ $f _ { 0 } ( X _ { i } ) + \varepsilon _ { i }$ , where $\mathbb { E } [ \varepsilon _ { i } \mid X _ { i } ] = 0$ and $\varepsilon _ { i }$ is conditionally sub-Gaussian with variance proxy

$\sigma ^ { 2 }$ . We assume that the target regression function admits the common-local decomposition

$$
f _ { 0 } ( x ) = f _ { S } ( x ) + \mathbb { 1 } _ { \boldsymbol { \mathcal { X } } _ { 1 } } ( x ) f _ { R , 1 } ( x ) + \mathbb { 1 } _ { \boldsymbol { \mathcal { X } } _ { 2 } } ( x ) f _ { R , 2 } ( x ) , \qquad x \in \boldsymbol { \mathcal { X } } ,\tag{6.1}
$$

where $f _ { S } \in \mathcal { F } _ { S }$ is a common component and $f _ { R , j } \in \mathcal { F } _ { j } , j = 1 , 2$ , are region-specific residual components. The corresponding shared-representation class is

$$
\begin{array} { r } { \mathcal { A } _ { \mathrm { s h } } : = \{ f _ { S } + \mathbb { 1 } _ { \mathcal { X } _ { 1 } } f _ { R , 1 } + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } f _ { R , 2 } : f _ { S } \in \mathcal { F } _ { S } , \ f _ { R , j } \in \mathcal { F } _ { j } , \ j = 1 , 2 \} . } \end{array}
$$

To understand the role of the shared expert, we compare this representation with a purerouted representation using the same routing partition. Without an explicit shared component, the branch function on region $\chi _ { j }$ is $H _ { j } ( x ) : = f _ { S } ( x ) + f _ { R , j } ( x )$ for $x \in \mathcal { X } _ { j }$ . Thus the pure-routed architecture represents the same target as $f _ { 0 } ( x ) = \mathbb { 1 } _ { \mathcal { X } _ { 1 } } ( x ) H _ { 1 } ( x ) + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } ( x ) H _ { 2 } ( x )$ ， provided that the regional routed class is enlarged to $\mathcal { H } _ { j } : = \mathcal { F } _ { S } | _ { \mathcal { X } _ { j } } + \mathcal { F } _ { j }$ , where $\mathcal { F } _ { S } | _ { \mathcal { X } _ { j } }$ denotes the restriction of the shared class to region $\chi _ { j }$ . Thus, at the level of approximation, the pure-routed representation can match the shared representation if its regional learners are rich enough to contain the full branch functions. The distinction is statistical rather than purely representational: with a shared expert, the routed learner on region $\chi _ { j }$ estimates only the residual component $f _ { R , j } ;$ without a shared expert, the same routed learner must estimate the full branch function $H _ { j } = f _ { S } + f _ { R , j }$

Let $\mathcal { M } _ { S }$ denote the learning rule designed for the shared class $\mathcal { F } _ { S }$ , and let ${ \mathcal { M } } _ { j }$ denote the routed learning rule used on region $\chi _ { j }$ . Under the shared representation, write $\mathcal { M } _ { \mathrm { s h } }$ for the combined shared-residual learning rule that uses $\mathcal { M } _ { S } , \mathcal { M } _ { 1 }$ , and $\mathcal { M } _ { 2 }$ . Given the full training sample $\mathcal { D } _ { n } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ , this combined rule produces $( \hat { f } _ { S } , \hat { f } _ { R , 1 } , \hat { f } _ { R , 2 } ) = \mathcal { M } _ { \mathrm { s h } } ( \mathcal { D } _ { n } )$ . The combined rule may be implemented, for example, by joint least squares or penalized least squares over the additive shared-residual class. It may also be implemented by a two-stage procedure that first estimates the shared component and then applies the regional learners to residuals, provided that the resulting residualization step is well posed. The resulting shared estimator is

$$
\hat { f } _ { 0 , \mathrm { s h } } ( x ) : = \hat { f } _ { S } ( x ) + \mathbb { 1 } _ { \mathcal { X } _ { 1 } } ( x ) \hat { f } _ { R , 1 } ( x ) + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } ( x ) \hat { f } _ { R , 2 } ( x ) .
$$

Under the pure-routed representation, there is no explicit shared expert. The same regional learning rule $\mathcal { M } _ { j }$ is applied directly to the full branch target $H _ { j } ( x )$ . Equivalently, the routed learner is now required to estimate an element of the larger branch class $\mathcal { H } _ { j } =$ $\mathcal { F } _ { S } | _ { \mathcal { X } _ { j } } + \mathcal { F } _ { j }$ , rather than only the residual class ${ \mathcal { F } } _ { j }$ . Let $\mathcal { D } _ { n , j } = \{ ( X _ { i } , Y _ { i } ) : X _ { i } \in \mathcal { X } _ { j } \}$ } denote the regional training sample. The branch estimators are $\hat { H } _ { j } = \mathcal { M } _ { j } ( \mathcal { D } _ { n , j } ) , j \in \{ 1 , 2 \}$ , and the corresponding pure-routed estimator is

$$
\hat { f } _ { 0 , \mathrm { r t } } ( \boldsymbol x ) : = \mathbb { 1 } _ { \boldsymbol \chi _ { 1 } } ( \boldsymbol x ) \hat { H } _ { 1 } ( \boldsymbol x ) + \mathbb { 1 } _ { \boldsymbol \chi _ { 2 } } ( \boldsymbol x ) \hat { H } _ { 2 } ( \boldsymbol x ) .
$$

Thus the comparison keeps the routing partition and the regional learning rules fixed, but changes the target faced by the routed learner: the shared representation replaces the full branch-learning problem by a residual-learning problem on each region.

This formulation reflects the practical motivation for shared experts. As argued in (Dai et al., 2024), routed experts may otherwise redundantly acquire common knowledge in their parameters; a shared expert can absorb this common information and leave the routed experts to specialize locally.

## 6.2 A learner-specific comparison

We now formalize the preceding idea through the risk profiles of the learning rules. For a function g on $\chi _ { j }$ , write $\| g \| _ { j } ^ { 2 } : = \mathbb { E } \{ g ( X ) ^ { 2 } \mid X \in \mathcal { X } _ { j } \}$ , so that $\| \cdot \| _ { j }$ denotes the $L _ { 2 }$ norm under the conditional distribution of X given $X \in \mathcal { X } _ { j }$ . Let $p _ { j } : = P _ { X } ( \mathcal { X } _ { j } ) > 0 , j = 1 , 2$ . For a target $h ,$ let $\mathcal { D } _ { m , j } ^ { h }$ and $\mathcal { D } _ { m } ^ { h }$ denote training samples of size m generated from $Z = h ( X ) + \xi$ where $\xi$ has conditional mean zero and is conditionally sub-Gaussian with variance proxy $\sigma ^ { 2 }$ . Here, $X \sim P _ { X } ( \cdot \mid X \in \mathcal { X } _ { j } )$ for $\mathcal { D } _ { m , j } ^ { h }$ and $X \sim P _ { X }$ for $\mathcal { D } _ { m } ^ { h }$

For a local target h on $\chi _ { j }$ , define the learner-specific risk of the regional rule $\mathcal { M } _ { j }$ by

$$
\mathcal { R } _ { j } ^ { \mathcal { M } } ( h ; m , \sigma ) : = \mathbb { E } _ { h , \sigma } \left\| \mathcal { M } _ { j } ( \mathcal { D } _ { m , j } ^ { h } ) - h \right\| _ { j } ^ { 2 } .
$$

This is the risk of a fixed learning rule at a specified target, not a minimax risk over a function class. Similarly, for a global target h on $x ,$ define the shared learner risk

$$
\mathcal { R } _ { S } ^ { \mathcal { M } } ( h ; m , \sigma ) : = \mathbb { E } _ { h , \sigma } \left\| \mathcal { M } _ { S } ( \mathcal { D } _ { m } ^ { h } ) - h \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Recall that the full branch class in the pure-routed representation is $\mathcal { H } _ { j } = \mathcal { F } _ { S } | _ { \mathcal { X } _ { j } } + \mathcal { F } _ { j }$ for $j \in \{ 1 , 2 \}$

Assumption 6.1. There exist positive rate profiles $\varrho _ { S } ( m ) , \varrho _ { j } ( m )$ and $\tilde { \varrho } _ { j } ( m ) , j \in \{ 1 , 2 \}$ , where m denotes the number of observations available to the corresponding learning rule, and constants $0 < c < C < \infty$ such that the following conditions hold:

(a) $\begin{array} { r } { \operatorname* { s u p } _ { h \in \mathcal { F } _ { S } } \mathcal { R } _ { S } ^ { \mathcal { M } } ( h ; m , \sigma ) \leq C \varrho _ { S } ( m ) ; } \end{array}$

(b) $\begin{array} { r } { \operatorname* { s u p } _ { h \in \mathcal { F } _ { i } } \mathcal { R } _ { j } ^ { \mathcal { M } } ( h ; m , \sigma ) \leq C \varrho _ { j } ( m ) } \end{array}$ for $j \in \{ 1 , 2 \}$ ;

(c) for each $j \in \{ 1 , 2 \}$ , there exists a nonempty subclass $B _ { j } \subseteq { \mathcal { H } } _ { j }$ , independent of $m _ { \colon }$ such that $\begin{array} { r } { \operatorname* { i n f } _ { h \in \mathcal { B } _ { j } } \mathcal { R } _ { j } ^ { \mathcal { M } } ( h ; m , \sigma ) \geq c \tilde { \varrho } _ { j } ( m ) } \end{array}$

In addition, the rate profiles are nonincreasing and regular under fixed rescaling: for every fixed $0 < a < \infty , \varrho ( a m ) \asymp \varrho ( m )$ for each $\varrho \in \{ \varrho _ { S } , \varrho _ { 1 } , \varrho _ { 2 } , \tilde { \varrho } _ { 1 } , \tilde { \varrho } _ { 2 } \}$ . Finally, writing $p _ { j } =$ $P _ { X } ( \mathcal { X } _ { j } )$ , the shared representation estimator satisfies

$$
\operatorname* { s u p } _ { f _ { 0 } \in \mathcal A _ { \mathrm { s h } } } \mathbb E _ { f _ { 0 } , \sigma } \Big \lVert \widehat f _ { 0 , \mathrm { s h } } - f _ { 0 } \Big \rVert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C \{ \varrho _ { S } ( n ) + p _ { 1 } \varrho _ { 1 } ( n p _ { 1 } ) + p _ { 2 } \varrho _ { 2 } ( n p _ { 2 } ) \} ,\tag{6.2}
$$

Assumption 6.1 compares the performance of fixed learning rules across diferent targets. In particular, the same regional learner $\mathcal { M } _ { j }$ may perform well on residual targets in ${ \mathcal { F } } _ { j }$ but poorly on full branch targets in $\mathcal { H } _ { j }$ , either because the latter are harder to estimate or because they are less compatible with the learner’s approximation class.

The bound in (6.2) requires a joint estimator of the shared and regional components to attain the displayed rate. A naive sequential procedure that first regresses Y on $\mathcal { F } _ { S }$ need not recover $f _ { S } ,$ , since the regional residuals may have nonzero projection onto $\mathcal { F } _ { S }$ Such a sequential interpretation therefore requires additional normalization conditions, such as regional orthogonality or residualization. Alternatively, the shared upper bound (6.2) can be justified directly by a joint procedure, for example penalized least squares over the additive class $\{ s + \mathbb { 1 } _ { \mathcal { X } _ { 1 } } r _ { 1 } + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } r _ { 2 } : s \in \mathcal { F } _ { S } , \ r _ { j } \in \mathcal { F } _ { j } \}$ . Under standard oracle inequalities for such sieve or penalized estimators, the risk decomposes at the scale $\varrho _ { S } ( n ) + p _ { 1 } \varrho _ { 1 } ( n p _ { 1 } ) +$ $p _ { 2 } \varrho _ { 2 } ( n p _ { 2 } )$ ; see Supplement Section E.2 for details. We use this high-level formulation to keep the comparison independent of a particular implementation, such as sieve estimation or a specific orthogonalization scheme.

The preceding assumption leads to the following generic rate comparison.

Proposition 6.2. Assume the routing partition is known and Assumption 6.1 holds. Suppose also that, for each $j = 1 , 2$ , the hard subclass $B _ { j } \subseteq { \mathcal { H } } _ { j }$ in Assumption 6.1 is compatible with the shared representation, in the sense that its elements can arise as branch functions $H _ { j } = f _ { S } | _ { \mathcal { X } _ { j } } + f _ { R , j }$ for some $f _ { 0 } \in \mathcal { A } _ { \mathrm { s h } }$ . Then, for all suficiently large $n ,$ there exist constants $0 < c < C < \infty$ such that the shared representation satisfies

$$
\operatorname* { s u p } _ { f _ { 0 } \in \mathcal A _ { \mathrm { s h } } } \mathbb E _ { f _ { 0 } , \sigma } \Big \lVert \widehat f _ { 0 , \mathrm { s h } } - f _ { 0 } \Big \rVert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C \{ \varrho _ { S } ( n ) + p _ { 1 } \varrho _ { 1 } ( n p _ { 1 } ) + p _ { 2 } \varrho _ { 2 } ( n p _ { 2 } ) \} ,
$$

whereas the pure-routed architecture trained with the same regional learners satisfies

$$
\operatorname* { s u p } _ { f _ { 0 } \in \mathcal A _ { \mathrm { s h } } } \mathbb E _ { f _ { 0 } , \sigma } \left\| \hat { f } _ { 0 , \mathrm { r t } } - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \geq c \{ p _ { 1 } \tilde { \varrho } _ { 1 } ( n p _ { 1 } ) \vee p _ { 2 } \tilde { \varrho } _ { 2 } ( n p _ { 2 } ) \} .
$$

Consequently, if

$$
\varrho _ { S } ( n ) + p _ { 1 } \varrho _ { 1 } ( n p _ { 1 } ) + p _ { 2 } \varrho _ { 2 } ( n p _ { 2 } ) = o \{ p _ { 1 } \tilde { \varrho } _ { 1 } ( n p _ { 1 } ) \vee p _ { 2 } \tilde { \varrho } _ { 2 } ( n p _ { 2 } ) \} ,
$$

then the shared architecture has asymptotically smaller worst-case risk for this fixed routed training procedure.

Remark 6.3. Proposition 6.2 compares the shared-routed and pure-routed representations under the same regional learning rules, with residual targets in the shared-routed case and full branch targets in the pure-routed case. The latter may be statistically harder, or may simply be less well approximated by the chosen regional learner. This comparison is motivated by practical MoE training, where routed experts are typically trained under the same optimization pipeline rather than by oracle procedures tailored to each induced branch class. In special cases, for instance when $\mathcal { H } _ { j }$ is finite dimensional and well matched to the routed learner, the pure-routed minimax rate over $\mathcal { H } _ { j }$ may match the shared minimax rate up to constants.

## 6.3 An illustrative example

We now consider a setting in which the regional learning rules estimate the residual components more eficiently than the full branch functions. The shared component is a linear trend, whereas the routed residuals are finite periodic Fourier signals. A routed learner based only on periodic Fourier sieves can estimate the residuals eficiently after the shared trend has been extracted, but it is poorly matched to the full branch functions because the linear trend is nonperiodic in the local coordinates. For simplicity, both regions use the same Fourier basis. The same learner-specific mechanism can also arise with region-specific bases when each regional learning rule estimates its residual target more eficiently than the corresponding full branch target.

Example 6.4 (A Fourier-sieve illustration). Let $\mathcal { X } = [ 0 , 1 ] , \ : X \sim \mathrm { U n i f } [ 0 , 1 ] , \ : \mathcal { X } _ { 1 } = [ 0 , 1 / 2 ]$ 2 $\mathcal { X } _ { 2 } = ( 1 / 2 , 1 ]$ , and $p _ { 1 } = p _ { 2 } = 1 / 2$ . For $x \in \mathcal { X } _ { j }$ , define the local coordinate as $t _ { 1 } ( x ) = 2 x$ 2 $t _ { 2 } ( x ) = 2 x - 1$ , so that $t _ { j } ( X ) \sim \mathrm { U n i f } [ 0 , 1 ]$ conditional on $X \in \mathcal { X } _ { j }$ . Consider the shared-plusrouted target

$$
f _ { 0 } ( x ) = f _ { S } ( x ) + \mathbb { 1 } _ { \mathcal { X } _ { 1 } } ( x ) f _ { R , 1 } ( x ) + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } ( x ) f _ { R , 2 } ( x ) ,
$$

where $f _ { S } ( x ) = \gamma x$ with $\gamma _ { 0 } \leq | \gamma | \leq B$ , for fixed constants $0 < \gamma _ { 0 } \le B < \infty$ . The routed residuals are finite periodic Fourier signals: for a fixed integer $K _ { 0 } > 0$

$$
f _ { R , j } ( x ) = \sum _ { k = 1 } ^ { K _ { 0 } } \{ a _ { j , k } \sin ( 2 k \pi t _ { j } ( x ) ) + b _ { j , k } \cos ( 2 k \pi t _ { j } ( x ) ) \} , \qquad \operatorname* { m a x } _ { 1 \leq k \leq K _ { 0 } } \{ | a _ { j , k } | , | b _ { j , k } | \} \leq B .
$$

Assume that the noise $\varepsilon \sim N ( 0 , \sigma ^ { 2 } )$ , independent of $X$

Let the routed learner be a periodic Fourier-sieve learner on $t \in [ 0 , 1 ]$ , based on ${ \cal { S } } _ { K } = $ $\operatorname { s p a n } \{ 1 , \sin ( 2 \pi k t ) , \cos ( 2 \pi k t ) : 1 \leq k \leq K \}$ . Under the shared representation, use a joint sieve least-squares estimator over the additive class

$$
\mathcal { A } _ { \mathrm { s h } , K _ { 0 } } = \{ \alpha x + \mathbb { 1 } _ { \mathcal { X } _ { 1 } } ( x ) r _ { 1 } ( t _ { 1 } ( x ) ) + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } ( x ) r _ { 2 } ( t _ { 2 } ( x ) ) : \alpha \in \mathbb { R } , \ r _ { j } \in \mathcal { S } _ { K _ { 0 } } \} .
$$

The true function $f _ { 0 }$ belongs to this finite-dimensional class. Since $K _ { 0 }$ is fixed, standard least-squares theory gives $\mathbb { E } \| \hat { f } _ { 0 , \mathrm { s h } } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \lesssim \sigma ^ { 2 } / n$ . Thus, in the notation of Assumption 6.1, for a sample of size m, the corresponding rate profiles can be taken as $\varrho _ { S } ( m ) \asymp \sigma ^ { 2 } / m$ and $\varrho _ { j } ( m ) \asymp \sigma ^ { 2 } / m$ for $j \in \{ 1 , 2 \}$ . Consequently,

$$
\varrho _ { S } ( n ) + p _ { 1 } \varrho _ { 1 } ( n p _ { 1 } ) + p _ { 2 } \varrho _ { 2 } ( n p _ { 2 } ) \asymp \frac { \sigma ^ { 2 } } { n } ,
$$

which verifies the shared upper-rate condition for the joint Fourier-sieve estimator.

Now consider the pure-routed representation trained with the same regional Fouriersieve learner. On region $\chi _ { j }$ , the routed learner must estimate the full branch function $H _ { j } = f _ { S } ( x ) + f _ { R , j } ( x )$ . In the local coordinate, the shared component becomes a nonperiodic linear ramp. Specifically, $f _ { S } ( x ) = \gamma t _ { 1 } ( x ) / 2$ on $\mathcal { X } _ { 1 }$ and $f _ { S } ( x ) = \gamma \{ t _ { 2 } ( x ) + 1 \} / 2$ on $\mathcal { X } _ { 2 }$ The constant part is contained in $\boldsymbol { \mathcal { S } } _ { K }$ , but the ramp $t \mapsto t$ is not periodic. Its squared approximation tail under the periodic Fourier basis is of order $K ^ { - 1 }$ . Therefore, even if the sieve dimension $K$ is chosen optimally, the regional Fourier learner faces the bias-variance tradeof

$$
\mathcal { R } _ { j } ^ { \mathcal { M } } ( H _ { j } ; m , \sigma ) \gtrsim \operatorname* { i n f } _ { K \geq 1 } \left\{ \gamma _ { 0 } ^ { 2 } K ^ { - 1 } + \frac { \sigma ^ { 2 } K } { m } \right\} .
$$

The first term is the Fourier approximation tail of the nonperiodic linear ramp, and the second term is the variance cost of estimating K Fourier modes. Optimizing over $K$ yields $\mathcal { R } _ { j } ^ { \mathcal { M } } ( H _ { j } ; m , \sigma ) \gtrsim \gamma _ { 0 } \sigma / \sqrt { m }$ . Therefore, for balanced regions, $\mathbb { E } \Vert \hat { f } _ { 0 , \mathrm { r t } } - f _ { 0 } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \gtrsim \gamma _ { 0 } \sigma / \sqrt { n }$

This example illustrates a learner-specific complexity separation. After the linear common trend is extracted by the shared expert, the routed experts only need to estimate finite Fourier residuals and achieve the parametric rate $n ^ { - 1 }$ . In contrast, a pure-routed Fourier learner must estimate the full branch function $H _ { j } = f _ { S } + f _ { R , j }$ , whose approximation by periodic Fourier sieves is limited by the nonperiodic linear component. This yields the slower learner-specific rate $n ^ { - 1 / 2 }$

The rate gap in this example arises from the periodic Fourier basis used by the routed learner. A pure-routed learner whose dictionary contains the linear function $x ,$ or whose basis is adapted to $\mathcal { H } _ { j }$ , can avoid the corresponding approximation error.

Remark 6.5. A complementary form of separation arises when the shared component is complex but globally smooth, while the routed residuals are simple discontinuities. Suppose $\begin{array} { r } { f _ { 0 } ( x ) = h ( x ) + \sum _ { r = 1 } ^ { M _ { 0 } } a _ { r } \mathbb { 1 } _ { \mathcal { X } _ { r } } ( x ) } \end{array}$ , where $h$ is a smooth nonparametric function and the routed residual is piecewise constant over the routing regions. If the shared learner is well suited to estimating $h ,$ while the routed learners are simple piecewise-constant or low-order local learners, then the shared representation leaves only the discontinuous regional ofsets to the routed learners. In contrast, a pure-routed learner must approximate $h + a _ { r }$ separately on each region. When h is not well represented by the routed learner, the pure-routed estimator may sufer a large approximation bias, even if the residual ofsets themselves are easy to learn. Figure 1 illustrates the two complementary settings considered in the preceding example and this remark.

![](images/28a615382a693675e973bdae1f785bd37f28ab25987a371f7c2162173e3fb327.jpg)  
(a) Simple shared component, complex routed components

![](images/4f88a0903a963a62a7c0128a81f3d6971bbf191e7376d7978aed9952da91015c.jpg)  
(b) Complex shared component, simple routed components  
Figure 1: Prediction MSE for the shared-routed and pure-routed estimators on log-log axes. Panel (a) considers a simple shared component and complex routed components, while Panel (b) considers a complex shared component and simple routed components. Error bars represent one standard deviation across Monte Carlo replications. The shared-routed estimator has lower prediction MSE across the displayed sample sizes in both settings. See Supplement Section G.2 for implementation details.

Together, the two settings show that the gain from a shared expert can arise through either a slower estimation rate or approximation bias for the pure-routed learner, depending on how the induced targets interact with the local learning rules.

## 7 Conclusions and outlook

## 7.1 Main conclusions

This paper develops a statistical perspective on MoE architectures. The central message is that the statistical role of MoE stems from how the gating function makes the combination of experts depend on the input, thereby changing the approximation-estimation-computation tradeof. From this viewpoint, dense routing, sparse routing, adaptive gating classes, and shared experts are diferent mechanisms for allocating modeling capacity locally across the

input space.

The analysis highlights three complementary mechanisms. First, sparse Top-K routing admits an oracle inequality with a router-learning cost of the same order as dense softmax gating, up to a boundary-stability logarithmic term, while reducing per-input expertevaluation cost. Second, the choice of gating class is a geometric approximation problem: linear, quadratic, and kernel-based gates encode diferent assumptions about the geometry of the expert-induced partition of the input space. Third, shared experts can change the targets faced by routed learners, replacing full branch-learning problems by residuallearning problems when the common component is better matched to the shared learning rule. Taken together, these results suggest that the statistical value of MoE architectures lies not only in using more experts, but in matching routing, sparsity, and shared structure to the local complexity of the target.

These conclusions also provide a statistical interpretation of several heuristics in largescale MoE practice. Increasing the total number of routed experts is useful only insofar as it creates a richer library of locally accurate predictors; otherwise the router pays a larger estimation and search cost without a corresponding approximation gain. Activating more than one routed expert per input is useful when the locally best predictor is not a single expert but a nondegenerate mixture of several experts. Shared experts are useful when a common component is repeatedly needed across many regions, because they prevent routed experts from redundantly learning the same global structure. Thus, the empirical success of fine-grained experts, Top-K routing with K > 1, and shared-routed designs can be viewed as diferent ways of improving the local approximation-estimation-computation tradeof.

## 7.2 Limitations and open questions

The theory developed here is intentionally modular. It isolates several statistical mechanisms: localized aggregation, sparse activation, geometry-adaptive gating, and sharedresidual decomposition. This modular view clarifies which architectural component is responsible for which statistical efect and the conditions under which each efect should be expected: Top-K routing requires control of boundary-switching events; flexible gating classes require enough data to ofset their larger estimation cost; and shared experts improve eficiency only when the induced residual learning problem is better matched to the

routed learners.

Several limitations remain. First, although the dense and sparse oracle inequalities allow the experts to evolve predictably along the training trajectory, our analysis relies on high-level expert-error control and boundedness conditions and does not characterize the underlying joint gradient-based expert-router dynamics. Other approximation and sharedexpert results condition on population experts, oracle partitions, or fixed learning rules. Second, some arguments use oracle or performance-induced partitions to describe the geometry of expert specialization. These partitions are useful for interpretation and approximation analysis, but they are not directly observed in practice. Third, the framework abstracts away from engineering features that are central in large-scale MoE systems, including load balancing, token capacity constraints, optimization dynamics, and interactions between routing and representation learning.

These limitations point to several open questions. A first question is how to empirically validate the local-performance premise underlying the analysis. The oracle or performanceinduced partitions used in the theory are not meant to be directly observed objects, but they suggest a testable diagnostic question: do trained experts, or trained mixtures of experts, exhibit systematic predictive advantages on diferent parts of the input space? Answering this question would require data-driven measures of local expert performance and methods for checking whether the learned router aligns with regions where the selected experts are genuinely stronger. This would help distinguish meaningful expert specialization from purely architectural sparsity.

A second question concerns end-to-end expert-router co-adaptation. The modular perspective adopted here isolates the statistical role of routing and localized aggregation, but it leaves open how these mechanisms interact with gradient-based training, representation learning, router regularization, load balancing, and token capacity constraints in large-scale MoE systems. A more complete theory should explain not only how a given collection of experts should be combined, but also when training dynamics produce experts that are diverse, stable, and locally useful.

A third direction is to develop data-driven procedures for choosing the composition of the expert pool under a given computational or model-capacity budget. Our analysis shows that increasing the number of routed experts can improve local approximation, but in practice this expansion cannot be separated from the resulting costs in expert capacity, router learning and system coordination. This naturally leads to the question of expert granularity: how finely should a fixed modeling budget be divided across experts in order to achieve the best predictive performance? The answer may depend jointly on the number and capacity of routed experts, the allocation to shared experts, and the complexity of the routing mechanism. Developing principled procedures for selecting such budget-constrained expert granularity remains an important open problem.

Finally, extending the approximation and risk analyses to deep MoE architectures with compositional routing across layers and high-dimensional input spaces, establishing minimax lower bounds, and incorporating computational constraints into the statistical analysis remain important steps toward a more complete theory of modern MoE architectures.

## 7.3 A statistical outlook for future MoE architectures

The preceding analysis suggests several architecture-level directions that may be worth exploring in future MoE systems.

Specialization-aware routing under system constraints. A central message of our analysis is that routing should preserve and exploit local predictive advantages among experts, thereby maintaining statistically meaningful specialization. In large-scale systems, this objective must coexist with load, capacity, and communication constraints. Recently, DeepSeek-V3 (DeepSeek-AI et al., 2025) and Kimi K3 (Kimi Team et al., 2026) highlight the tension between expert specialization and load control in large-scale MoE systems. This motivates designing routing mechanisms that maintain acceptable system balance while disturbing predictive specialization as little as possible. More broadly, predictive specialization, routing stability, and system feasibility may be better treated as jointly designed objectives rather than separate considerations.

Input-adaptive expert activation. Conditional on a given expert pool, Section 4 shows that the statistically appropriate level of expert activation depends on the local prediction regime: a dominant specialist may be suficient for some inputs, whereas others benefit from combining several experts. This suggests going beyond the choice of a single common value of K. Rather than fixing the same Top-K rule for all inputs, the router could adapt the number of active experts to local predictive structure, activating fewer experts where one or a few specialists sufice and more experts where prediction is intrinsically compositional. Related token- and sequence-adaptive routing mechanisms have already begun to explore variable expert activation (Wen et al., 2025; Zeng et al., 2024). Developing statistically principled procedures for such input-dependent activation, including how the activation level should be selected from data, is therefore a natural direction. Such input-dependent expert allocation may be viewed as a first step toward a broader framework of adaptive expert specialization.

Multiscale shared–routed architectures. Section 6 suggests that shared experts are useful when they absorb predictive structure that would otherwise be repeatedly learned by routed experts. The same principle may extend beyond the binary distinction between globally shared and fully routed components. Classical hierarchical MoE architectures (Jordan and Jacobs, 1994) illustrate how experts can be organized at intermediate levels, while more recent constructions such as CartesianMoE (Su et al., 2025) introduce structured forms of sharing across expert components. We therefore envision shared–routed architectures with multiple scales of reuse, where some components are shared globally, others within groups of related inputs, and the remaining components are routed locally. The scope of each shared component could therefore adapt to the set of inputs over which it captures common predictive structure.

Boundary-aware counterfactual expert discovery. Section 5 defines meaningful specialization through local predictive advantage, but standard sparse training observes only the executed route and may therefore reinforce early routing decisions without suficient information about alternatives. Counterfactual routing and exploration have recently been investigated as ways to provide such information (Fedus et al., 2022; Kim and Kang, 2025; Yoon et al., 2026). Our analysis of routing geometry and active-set stability motivates investigating whether this additional information may be particularly valuable near routing boundaries. Rather than exploring alternative experts uniformly, a limited exploration budget could be concentrated on inputs with unstable expert assignments, for example near routing boundaries where small perturbations may change the active set. Such boundaryaware exploration may reduce specialization driven by early routing noise without increasing the number of experts activated at inference time.

Taken together, these directions suggest that future MoE design may benefit from making specialization, expert granularity, expert activation, sharing, and exploration increasingly adaptive to the local prediction structure, while respecting the computational constraints that make sparse MoE architectures attractive in the first place.

## References

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. (2020). Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.

Dai, D., Deng, C., Zhao, C., Xu, R. X., Gao, H., Chen, D., Li, J., Zeng, W., Yu, X., Wu, Y., Xie, Z., Li, Y. K., Huang, P., Luo, F., Ruan, C., Sui, Z., and Liang, W. (2024). DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1280–1297. Association for Computational Linguistics.

DeepSeek-AI, Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Guo, D., Yang, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Zhang, H., Ding, H., Xin, H., Gao, H., Li, H., Qu, H., Cai, J. L., Liang, J., Guo, J., Ni, J., Li, J., Wang, J., Chen, J., Chen, J., Yuan, J., Qiu, J., Li, J., Song, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Xu, L., Xia, L., Zhao, L., Wang, L., Zhang, L., Li, M., Wang, M., Zhang, M., Zhang, M., Tang, M., Li, M., Tian, N., Huang, P., Wang, P., Zhang, P., Wang, Q., Zhu, Q., Chen, Q., Du, Q., Chen, R. J., Jin, R. L., Ge, R., Zhang, R., Pan, R., Wang, R., Xu, R., Zhang, R., Chen, R., Li, S. S., Lu,

S., Zhou, S., Chen, S., Wu, S., Ye, S., Ye, S., Ma, S., Wang, S., Zhou, S., Yu, S., Zhou,

S., Pan, S., Wang, T., Yun, T., Pei, T., Sun, T., Xiao, W. L., Zeng, W., Zhao, W., An,

W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Li, X. Q., Jin, X., Wang, X., Bi,

X., Liu, X., Wang, X., Shen, X., Chen, X., Zhang, X., Chen, X., Nie, X., Sun, X., Wang,

X., Cheng, X., Liu, X., Xie, X., Liu, X., Yu, X., Song, X., Shan, X., Zhou, X., Yang, X.,

Li, X., Su, X., Lin, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhu, Y. X., Zhang, Y., Xu,

Y., Xu, Y., Huang, Y., Li, Y., Zhao, Y., Sun, Y., Li, Y., Wang, Y., Yu, Y., Zheng, Y.,

Zhang, Y., Shi, Y., Xiong, Y., He, Y., Tang, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y.,

Liu, Y., Guo, Y., Wu, Y., Ou, Y., Zhu, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Zha,

Y., Xiong, Y., Ma, Y., Yan, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Wu, Z. F., Ren,

Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Huang, Z., Zhang, Z., Xie, Z., Zhang, Z., Hao,

Z., Gou, Z., Ma, Z., Yan, Z., Shao, Z., Xu, Z., Wu, Z., Zhang, Z., Li, Z., Gu, Z., Zhu,

Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Gao, Z., and Pan, Z. (2025). DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437.

Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., Zoph, B., Fedus, L., Bosma, M. P., Zhou, Z., Wang, T., Wang, E., Webster, K., Pellat, M., Robinson, K., Meier-Hellstern, K., Duke, T., Dixon, L., Zhang, K., Le, Q., Wu, Y., Chen, Z., and Cui, C. (2022). GLaM: Eficient scaling of language models with mixture-of-experts. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 5547–5569. PMLR.

Fedus, W., Zoph, B., and Shazeer, N. (2022). Switch Transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal of Machine Learning Research, 23(120):1–39.

Gao, Z., He, B., and Yang, Y. (2026). Combining pre-trained models via localized model averaging. arXiv preprint arXiv:2605.13421.

He, X. O. (2024). Mixture of a million experts. arXiv preprint arXiv:2407.04153.

Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E. (1991). Adaptive mixtures of local experts. Neural Computation, 3(1):79–87.

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., de las Casas, D., Hanna, E. B., Bressand, F., Lengyel, G., Bour, G., Lample, G., Lavaud, L. R., Saulnier, L., Lachaux, M.-A., Stock, P., Subramanian, S., Yang, S., Antoniak, S., Scao, T. L., Gervet, T., Lavril, T., Wang, T., Lacroix, T., and Sayed, W. E. (2024). Mixtral of Experts. arXiv preprint arXiv:2401.04088.

Jiang, W. and Tanner, M. A. (1999a). Hierarchical mixtures-of-experts for generalized linear models: Some results on denseness and consistency. In Heckerman, D. and Whittaker, J., editors, Proceedings of the Seventh International Workshop on Artificial Intelligence and Statistics, volume R2 of Proceedings of Machine Learning Research. PMLR. Reissued by PMLR on 20 August 2020.

Jiang, W. and Tanner, M. A. (1999b). On the identifiability of mixtures-of-experts. Neural Networks, 12(9):1253–1258.

Jordan, M. I. and Jacobs, R. A. (1994). Hierarchical mixtures of experts and the EM algorithm. Neural Computation, 6(2):181–214.

Jordan, M. I. and Xu, L. (1995). Convergence results for the EM approach to mixtures of experts architectures. Neural Networks, 8(9):1409–1431.

Kim, G. and Kang, S. (2025). Exploration-driven reinforcement learning for expert routing improvement in mixture-of-experts language models. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 23592–23605.

Kimi Team, Bai, T., Bai, Y., Bao, Y., C., M., Cai, J., Cai, X., Cao, P., Cao, Y., Chai, Z., Charles, Y., Che, H. S., Chen, G., Chen, G., Chen, G., Chen, H., Chen, J., Chen, J., Chen, J., Chen, K., Chen, P., Chen, R., Chen, W., Chen, X., Chen, Y., Chen, Y., Chen, Y., Chen, Y., Chen, Y., Chen, Y., Chen, Y., Chen, Z., Cheng, D., Cheng, Y., Cui, J., Cui, J., Dai, A., Deng, J., Ding, H., Ding, R., Ding, S., Dong, M., Dong, M., Dong, Y., Dong, Y., Du, A., Du, C., Du, D., Du, J., Du, Y., Fan, Y., Feng, J., Feng, Q., Feng, Y., Fu, K., Fu, Q., Gao, F., Gao, H., Gao, J., Gao, T., Gao, W., Geng, S., Gong, J., Gong, L., Gong, S., Gong, X., Gu, Q., Gu, Y., Guan, S., Guo, H., Guo, S., Guo, X., Guo, Z., Hao, B., Hao, W., Hao, X., He, D., He, H., He, L., He, Q., He, W., He, X., He, X., He, Y., He, Y., Hong, C., Hong, T., Hu, H., Hu, J., Hu, R., Hu, W., Hu, Y., Hu, Z., Hua, L.,

Huang, J., Huang, K., Huang, R., Huang, S., Huang, W., Huang, Y., Huang, Z., Huang,

Z., Hui, Y., Jia, C., Jiang, Y., Jiang, Z., Jiang, Z., Jin, W., Jin, X., Jing, Y., Kong, H.,

Lai, G., Li, A., Li, C., Li, C., Li, C., Li, F., Li, G., Li, H., Li, J., Li, J., Li, L., Li, L.,

Li, L., Li, W., Li, W., Li, X., Li, Y., Li, Y., Li, Y., Li, Y., Li, Z., Li, Z., Li, Z., Li, Z.,

Li, Z., Lin, J., Lin, X., Lin, Y., Lin, Z., Lin, Z., Liu, B., Liu, B., Liu, C., Liu, L., Liu,

S., Liu, S., Liu, S., Liu, T., Liu, W., Liu, Y., Liu, Y., Liu, Y., Liu, Y., Liu, Z., Liu, Z.,

Lu, E., Lu, H., Lu, L., Lu, T., Lu, Z., Luo, A., Luo, G., Luo, J., Luo, Y., Lyu, B., Lyu,

W., Mao, S., Mei, Y., Men, X., Ni, M., Niu, Y., Pan, S., Peng, S., Qi, Z., Qin, R., Qin,

Z., Qin, Z., Qiu, H., Qiu, J., Qiu, J., Qu, B., Qu, Y., Shang, Z., Shao, Y., Shen, H., Shi,

J., Shi, J., Shi, L., Shi, S., Siu, W., Song, P., Song, X., Su, J., Su, Y., Su, Z., Sui, L.,

Sun, J., Sun, J., Sun, S., Sun, S., Sun, T., Sun, Y., Tai, Y., Tang, C., Tang, H., Tang, S.,

Tang, Z., Tian, C., Tian, R., Tian, Y., Tu, W., Wang, C., Wang, C., Wang, C., Wang,

D., Wang, F., Wang, H., Wang, H., Wang, H., Wang, H., Wang, H., Wang, H., Wang, J.,

Wang, J., Wang, J., Wang, J., Wang, L., Wang, S., Wang, S., Wang, S., Wang, S., Wang,

S., Wang, T., Wang, W., Wang, X., Wang, X., Wang, X., Wang, X., Wang, Y., Wang,

Y., Wang, Y., Wang, Y., Wang, Y., Wang, Y., Wang, Y., Wang, Y., Wang, Z., Wang,

Z., Wang, Z., Wang, Z., Wang, Z., Wang, Z., Wei, C., Wei, M., Wei, S., Wen, Z., Wu,

F., Wu, H., Wu, R., Wu, W., Wu, X., Wu, Y., Wu, Y., Wu, Y., Wu, Z., Xian, X., Xiang,

C., Xiang, Y., Xiao, B., Xiao, C., Xiao, X., Xie, J., Xie, X., Xie, Y., Xie, Z., Xing, B.,

Xiong, Y., Xu, B., Xu, B., Xu, J., Xu, J., Xu, J., Xu, J., Xu, L. H., Xu, Q., Xu, S., Xu,

S., Xu, T., Xu, T., Xu, W., Xu, X., Xu, Y., Xu, Y., Xu, Y., Xu, Z., Xue, H., Yan, J.,

Yan, Y., Yang, F., Yang, G., Yang, H., Yang, J., Yang, R., Yang, W., Yang, X., Yang,

X., Yang, Y., Yang, Y., Yang, Y., Yang, Y., Yang, Z., Yang, Z., Yang, Z., Yang, Z., Yao,

H., Ye, D., Ye, H., Ye, W., Ye, Z., Yin, B., Yin, H., Yin, X., Yu, C., Yu, H., Yu, L., Yu,

S., Yu, S., Yu, T., Yuan, E., Yuan, M., Yue, T., Yue, W., Yue, Y., Zha, D., Zhan, H.,

Zhang, B. H., Zhang, D., Zhang, F., Zhang, H., Zhang, H., Zhang, H., Zhang, J., Zhang,

J., Zhang, J., Zhang, K., Zhang, M., Zhang, P., Zhang, Q., Zhang, R., Zhang, R., Zhang,

S., Zhang, S., Zhang, X., Zhang, X., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang,

Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Z., Zhang, Z., Zhao,

B., Zhao, C., Zhao, F., Zhao, J., Zhao, J., Zhao, S., Zhao, W., Zhao, X., Zhao, X., Zhao,

Y., Zhao, Z., Zheng, H., Zheng, H., Zheng, R., Zheng, S., Zheng, T., Zhong, H., Zhong,

L., Zhong, L., Zhou, M., Zhou, Q., Zhou, R., Zhou, R., Zhou, X., Zhou, Y., Zhou, Z., Zhu, J., Zhu, L., Zhu, X., Zhu, Y., Zhu, Y., Zhu, Z., Zhuang, C., Zhuang, W., and Zu, X. (2026). Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653.

Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., and Chen, Z. (2021). GShard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations.

Ludziejewski, J., Krajewski, J., Adamczewski, K., Pi´oro, M., Krutul, M., Antoniak, S., Ciebiera, K., Kr´ol, K., Odrzyg´o´zd´z, T., Sankowski, P., Cygan, M., and Jaszczur, S. (2024). Scaling laws for fine-grained mixture of experts. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 33270–33288. PMLR.

Nguyen, H., Akbarian, P., Yan, F., and Ho, N. (2024a). Statistical perspective of topk sparse softmax gating mixture of experts. In International Conference on Learning Representations.

Nguyen, H., Doan, T. T., Pham, Q., Bui, N. D., Ho, N., and Rinaldo, A. (2025). On DeepSeekMoE: Statistical benefits of shared experts and normalized sigmoid gating. arXiv preprint arXiv:2505.10860.

Nguyen, H., Ho, N., and Rinaldo, A. (2024b). On least square estimation in softmax gating mixture of experts. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 37707–37735. PMLR.

Nguyen, H. D., Lloyd-Jones, L. R., and McLachlan, G. J. (2016). A universal approximation theorem for mixture-of-experts models. Neural Computation, 28(12):2585–2593.

Olteanu, M. and Rynkiewicz, J. (2011). Asymptotic properties of mixture-of-experts models. Neurocomputing, 74(9):1444–1449.

OpenAI, Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., Avila, R., Babuschkin, I., Balaji, S., Balcom, V., Baltescu, P., Bao, H., Bavarian, M., Belgum, J., Bello, I., Berdine, J.,

Bernadett-Shapiro, G., Berner, C., Bogdonof, L., Boiko, O., Boyd, M., Brakman, A.-L., Brockman, G., Brooks, T., Brundage, M., Button, K., Cai, T., Campbell, R., Cann, A., Carey, B., Carlson, C., Carmichael, R., Chan, B., Chang, C., Chantzis, F., Chen, D., Chen, S., Chen, R., Chen, J., Chen, M., Chess, B., Cho, C., Chu, C., Chung, H. W., Cummings, D., Currier, J., Dai, Y., Decareaux, C., Degry, T., Deutsch, N., Deville, D., Dhar, A., Dohan, D., Dowling, S., Dunning, S., Ecofet, A., Eleti, A., Eloundou, T., Farhi, D., Fedus, L., Felix, N., Fishman, S. P., Forte, J., Fulford, I., Gao, L., Georges, E., Gibson, C., Goel, V., Gogineni, T., Goh, G., Gontijo-Lopes, R., Gordon, J., Grafstein, M., Gray, S., Greene, R., Gross, J., Gu, S. S., Guo, Y., Hallacy, C., Han, J., Harris, J., He, Y., Heaton, M., Heidecke, J., Hesse, C., Hickey, A., Hickey, W., Hoeschele, P., Houghton, B., Hsu, K., Hu, S., Hu, X., Huizinga, J., Jain, S., Jain, S., Jang, J., Jiang, A., Jiang, R., Jin, H., Jin, D., Jomoto, S., Jonn, B., Jun, H., Kaftan, T., Lukasz Kaiser, Kamali, A., Kanitscheider, I., Keskar, N. S., Khan, T., Kilpatrick, L., Kim, J. W., Kim, C., Kim, Y., Kirchner, J. H., Kiros, J., Knight, M., Kokotajlo, D., Lukasz Kondraciuk, Kondrich, A., Konstantinidis, A., Kosic, K., Krueger, G., Kuo, V., Lampe, M., Lan, I., Lee, T., Leike, J., Leung, J., Levy, D., Li, C. M., Lim, R., Lin, M., Lin, S., Litwin, M., Lopez, T., Lowe, R., Lue, P., Makanju, A., Malfacini, K., Manning, S., Markov, T., Markovski, Y., Martin, B., Mayer, K., Mayne, A., McGrew, B., McKinney, S. M., McLeavey, C., McMillan, P., McNeil, J., Medina, D., Mehta, A., Menick, J., Metz, L., Mishchenko, A., Mishkin, P., Monaco, V., Morikawa, E., Mossing, D., Mu, T., Murati, M., Murk, O., M´ely, D., Nair, A., Nakano, R., Nayak, R., Neelakantan, A., Ngo, R., Noh, H., Ouyang, L., O’Keefe, C., Pachocki, J., Paino, A., Palermo, J., Pantuliano, A., Parascandolo, G., Parish, J., Parparita, E., Passos, A., Pavlov, M., Peng, A., Perelman, A., de Avila Belbute Peres, F., de Oliveira Pinto, H. P., Michael, Pokorny, Pokrass, M., Pong, V. H., Powell, T., Power, A., Power, B., Proehl, E., Puri, R., Radford, A., Rae, J., Ramesh, A., Raymond, C., Real, F., Rimbach, K., Ross, C., Rotsted, B., Roussez, H., Ryder, N., Saltarelli, M., Sanders, T., Santurkar, S., Sastry, G., Schmidt, H., Schnurr, D., Schulman, J., Selsam, D., Sheppard, K., Sherbakov, T., Shieh, J., Shoker, S., Shyam, P., Sidor, S., Sigler, E., Simens, M., Sitkin, J., Slama, K., Sohl, I., Sokolowsky, B., Song, Y., Staudacher, N., Such, F. P., Summers, N., Sutskever, I., Tang, J., Tezak, N., Thompson, M. B., Tillet, P., Tootoonchian, A., Tseng, E., Tuggle, P., Turley, N., Tworek, J., Uribe, J. F. C., Vallone,

A., Vijayvergiya, A., Voss, C., Wainwright, C., Wang, J. J., Wang, A., Wang, B., Ward, J., Wei, J., Weinmann, C., Welihinda, A., Welinder, P., Weng, J., Weng, L., Wiethof, M., Willner, D., Winter, C., Wolrich, S., Wong, H., Workman, L., Wu, S., Wu, J., Wu, M., Xiao, K., Xu, T., Yoo, S., Yu, K., Yuan, Q., Zaremba, W., Zellers, R., Zhang, C., Zhang, M., Zhao, S., Zheng, T., Zhuang, J., Zhuk, W., and Zoph, B. (2024). GPT-4 technical report. arXiv preprint arXiv:2303.08774.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. (2022). High-resolution image synthesis with latent difusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations.

Su, Z., Wu, X., Lin, Z., Xiong, Y., Lv, M., Ma, G., Chen, H., Hu, S., and Ding, G. (2025). CartesianMoE: Boosting knowledge sharing among experts via cartesian product routing in mixture-of-experts. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 10040–10055.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pages 5998–6008. Curran Associates, Inc.

Wen, T., Wang, Y., Feng, A., Ma, L., Liu, X., Wang, Y., Guo, L., Chen, B., Jegelka, S., and You, C. (2025). Route experts by sequence, not by token. arXiv preprint arXiv:2511.06494.

Yang, Y. (2001). Adaptive regression by mixing. Journal of the American Statistical Association, 96(454):574–588.

Yang, Y. (2004). Combining forecasting procedures: Some theoretical results. Econometric Theory, 20(1):176–222.

Yoon, Y., Wang, S., Chen, W., and Ok, J. (2026). When are experts misrouted? counterfactual routing analysis in mixture-of-experts language models. arXiv preprint arXiv:2605.07260.

Yun, L., Zhuang, Y., Fu, Y., Xing, E. P., and Zhang, H. (2024). Toward inference-optimal mixture-of-expert large language models. arXiv preprint arXiv:2404.02852.

Zeevi, A. J., Meir, R., and Maiorov, V. (1998). Error bounds for functional approximation and estimation using mixtures of experts. IEEE Transactions on Information Theory, 44(3):1010–1025.

Zeng, Z., Miao, Y., Gao, H., Zhang, H., and Deng, Z. (2024). AdaMoE: Token-adaptive routing with null experts for mixture-of-experts language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 6223–6235.

# Supplementary Materials for “Towards a Statistical Understanding of Mixture-of-Experts”

This supplementary file contains the proofs of lemmas, propositions and theorems in the main article, as well as simulation details. This file is organized as follows. Appendix A provides the proofs for Section 2, including the approximation properties, metric entropy and the comparison between global and localized routing. Appendix B proves the oracle risk bounds for dense softmax gating in Section 3. Appendix C establishes the oracle results for sparse Top-K gating in Section 4, together with the corresponding Top-1 results. Appendix D contains the gating-approximation decomposition and the gating-class selection results in Section 5. Appendix E proves the shared-expert results in Section 6. Appendix F collects additional technical results on empirical risk minimization, cross-entropy-based parameter estimation, and dense-softmax upper bounds. Finally, Appendix G provides the implementation details for the simulation studies.

Additional Notation. We will use the following notations in this file additional to the main article. 1 denotes the all-ones vector, with its dimension understood from context. For a vector $v = ( v _ { 1 } , \ldots , v _ { m } ) ^ { \top }$ and $1 \leq p < \infty$ , we denote its p-norm by $\begin{array} { r } { \| v \| _ { p } = \left( \sum _ { j = 1 } ^ { m } | v _ { j } | ^ { p } \right) ^ { 1 / p } } \end{array}$ 2 while $\left\| v \right\| _ { \infty } = \operatorname* { m a x } _ { 1 \leq j \leq m } \left| v _ { j } \right|$ |. For a matrix $A , \| A \| _ { \mathrm { o p } } = \operatorname* { s u p } _ { \| v \| _ { 2 } = 1 } \| A v \| _ { 2 }$ denotes its operator norm induced by the Euclidean norm. For $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and $r > 0 , B _ { d } ( x , r ) = \{ y \in \mathbb { R } ^ { d } : \| y - x \| _ { 2 } \leq$ r} denotes the closed Euclidean ball centered at x with radius r. For a finite set ${ \mathcal { S } } , \# { \mathcal { S } }$ denotes its cardinality. Finally, for a Lebesgue-measurable set $\mathcal { A } \subseteq \mathbb { R } ^ { d } , \operatorname { V o l } ( \mathcal { A } )$ denotes its volume with respect to the Lebesgue measure.

## Contents

A Proof of Section 2 67   
A.1 Proof of Proposition 2.2 67   
A.2 Proof of Proposition 2.3 68   
A.3 Proof of Theorem 2.4 . 70   
A.4 Proof of Example 2.5 . 75   
A.5 Proof of Proposition 2.6 77   
B Proof of Section 3 78   
B.1 Proof of Theorem 3.8 . 78   
B.2 Proof of Example 3.9 . 91   
C Proof of Section 4 94   
C.1 Proof of Proposition 4.5 and related remarks 95   
C.2 Proof of Theorem 4.7 . 99   
C.3 Proof of Corollary 4.9 103   
C.4 Proof of Proposition 4.11 105   
D Proof of Section 5 108   
D.1 Proof of the oracle decomposition for gating approximation 108   
D.2 Proof of Example 5.2 . 108   
D.3 Proof of Example 5.3 . 112   
D.4 Proof of Theorem 5.5 . 115   
D.5 Proof of Corollary 5.7 120   
D.6 Proof of Theorem 5.9 . 121   
D.7 Proof of Corollary 5.10 123   
E Proof of Section 6 125   
E.1 Proof of Proposition 6.2 125   
E.2 A Sieve Justification for the Shared Upper Bound . 126   
E.3 Proof of Example 6.4 . 129   
F Additional Materials 133   
F.1 Order of Empirical Risk Minimization under Sub-Gaussian Noise 133   
F.2 Details for Empirical Parameter Estimation Using Cross-Entropy 140   
F.3 Specific Upper Bounds for Dense Softmax Gating in Section 4.2 . 151   
F.3.1 Dense Softmax Upper Bound after Corollary 4.9 151   
F.3.2 Dense Softmax Upper Bound after Proposition 4.11 156   
G Simulation Details 161   
G.1 Comparison of diferent gating classes 161   
G.2 MSE Comparison With and Without a Shared Expert . . . . . . . . . . . . 164

## A Proof of Section 2

## A.1 Proof of Proposition 2.2

Proof of Proposition 2.2. Fix $\varepsilon > 0$ and $h \in C _ { R } ( \mathcal { X } )$ . Since $\mathcal { X } = [ 0 , 1 ] ^ { d }$ is compact, h is uniformly continuous. Hence there exists $q \in \mathbb { N } _ { + }$ such that

$$
| h ( x ) - h ( y ) | < \varepsilon \qquad { \mathrm { w h e n e v e r } } \qquad \| x - y \| _ { 2 } \leq { \frac { \sqrt { d } } { q } } .
$$

Partition $[ 0 , 1 ] ^ { d }$ into the $q ^ { d }$ cubes

$$
Q _ { \nu } = \left[ \frac { \nu _ { 1 } - 1 } { q } , \frac { \nu _ { 1 } } { q } \right] \times \left[ \frac { \nu _ { 2 } - 1 } { q } , \frac { \nu _ { 2 } } { q } \right] \times \cdots \times \left[ \frac { \nu _ { d } - 1 } { q } , \frac { \nu _ { d } } { q } \right] , \qquad \nu = ( \nu _ { 1 } , \dots , \nu _ { d } ) \in \{ 1 , \ldots , q \} ^ { d } ,
$$

and let

$$
m _ { \nu } = \left( \frac { 2 \nu _ { 1 } - 1 } { 2 q } , \ldots , \frac { 2 \nu _ { d } - 1 } { 2 q } \right)
$$

be the center of $Q _ { \nu }$ . Set

$$
n = q ^ { d } , \qquad M = K n , \qquad \lambda = \frac { C _ { 1 } } { \mathrm { m a x } \{ 2 , d \} } .
$$

For each $\nu \in \{ 1 , \ldots , q \} ^ { d }$ and each $r \in [ K ]$ , define the constant expert

$$
f _ { \nu , r } ( x ) \equiv h ( m _ { \nu } )
$$

and the linear score

$$
u _ { \nu , r } ( \boldsymbol { x } ) = 2 \lambda m _ { \nu } ^ { \top } \boldsymbol { x } - \lambda \| m _ { \nu } \| _ { 2 } ^ { 2 } .
$$

These scores belong to the parameter box $\Theta _ { \mathrm { l i n } } ( C _ { 1 } )$ . Indeed, for every coordinate,

$$
| 2 \lambda m _ { \nu , j } | \leq 2 \lambda \leq C _ { 1 } ,
$$

and the intercept is bounded by

$$
\lambda \| m _ { \nu } \| _ { 2 } ^ { 2 } \leq \lambda d \leq C _ { 1 } .
$$

For fixed $x \in [ 0 , 1 ] ^ { d }$ , maximizing $u _ { \nu , r } ( x )$ over ν is equivalent to maximizing $2 m _ { \nu } ^ { \top } x$ $\| m _ { \nu } \| _ { 2 } ^ { 2 }$ , or equivalently, to minimizing $\| \boldsymbol { x } - \boldsymbol { m } _ { \nu } \| _ { 2 } ^ { 2 }$ . Thus the largest scores are attained by the grid center or centers closest to x. Since each center has K identical copies, the Top-K rule selects copies attached to nearest centers. If the nearest center is unique, it selects exactly the K copies attached to that center. The resulting MoE output is therefore

$$
\ell ( x ) = \sum _ { \nu , r } g _ { \nu , r } ^ { ( K ) } ( x ) f _ { \nu , r } ( x ) , \qquad g _ { \nu , r } ^ { ( K ) } ( x ) = \frac { \exp \{ u _ { \nu , r } ( x ) \} \mathbb { 1 } _ { \{ ( \nu , r ) \in \mathcal { T } _ { K } ( x ) \} } } { \sum _ { ( \nu ^ { \prime } , r ^ { \prime } ) \in \mathcal { T } _ { K } ( x ) } \exp \{ u _ { \nu ^ { \prime } , r ^ { \prime } } ( x ) \} } ,
$$

where $\ell ( x )$ is a convex combination of values $h ( \boldsymbol { m } _ { \nu } )$ at nearest grid centers. Each such center belongs to a cube whose closure contains x. Consequently,

$$
\| x - m _ { \nu } \| _ { 2 } \leq \frac { \sqrt { d } } { 2 q } < \frac { \sqrt { d } } { q } ,
$$

and therefore

$$
| h ( x ) - h ( m _ { \nu } ) | < \varepsilon .
$$

Since $\ell ( x )$ is a convex combination of such values $h ( \boldsymbol { m } _ { \nu } )$ , it follows that $| \ell ( x ) - h ( x ) | < \varepsilon$ for all $x \in \mathcal { X }$ . Thus

$$
\| \ell - h \| _ { \infty } < \varepsilon .
$$

Because $\varepsilon > 0$ was arbitrary, $\mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p } - K }$ is dense in $C _ { R } ( \mathcal { X } )$

## A.2 Proof of Proposition 2.3

Proof of Proposition 2.3. We first analyze the global class. Since $\mathcal { A } _ { \mathrm { g l o b a l } } ^ { R } = \mathcal { F } _ { \mathrm { c } } ^ { R }$ , every $f \in$ $\mathcal { A } _ { \mathrm { g l o b a l } } ^ { R }$ is a constant $f ( x ) \equiv c \mathrm { w i t h } \ | c | \le R .$ . For a fixed $h \in \mathcal { F } _ { b } ^ { R , L }$ , the midpoint of the range of h belongs to $[ - R , R ]$ , and hence the best constant approximation error equals half of the

oscillation:

$$
\operatorname* { i n f } _ { f \in A _ { \mathrm { g l o b a l } } ^ { R } } \| f - h \| _ { \infty } = { \frac { 1 } { 2 } } \left( \operatorname* { s u p } _ { x \in \mathcal { X } } h ( x ) - \operatorname* { i n f } _ { x \in \mathcal { X } } h ( x ) \right) .
$$

Because h is L-Lipschitz and $\mathcal { X } = [ 0 , 1 ] ^ { d }$ has diameter ${ \sqrt { d } } ,$ the oscillation is at most $L { \sqrt { d } }$ Also $| h | \leq R$ , so the oscillation is at most 2R. Therefore

$$
\operatorname* { i n f } _ { f \in A _ { \mathrm { g l o b a l } } ^ { R } } \Vert f - h \Vert _ { \infty } \leq \operatorname* { m i n } \left\{ R , { \frac { L { \sqrt { d } } } { 2 } } \right\} .
$$

To show that this bound is sharp, let

$$
a : = \operatorname* { m i n } \left\{ R , { \frac { L { \sqrt { d } } } { 2 } } \right\} , \qquad h _ { a } ( x ) = { \frac { a } { d } } \sum _ { j = 1 } ^ { d } ( 2 x _ { j } - 1 ) .
$$

Then $\| h _ { a } \| _ { \infty } \leq a \leq R .$ , and

$$
\| \nabla h _ { a } \| _ { 2 } = \frac { 2 a } { d } \sqrt { d } = \frac { 2 a } { \sqrt { d } } \leq L .
$$

Thus $h _ { a } \in \mathcal { F } _ { b } ^ { R , L }$ . Its range is exactly $[ - a , a ]$ , and therefore

$$
\operatorname* { i n f } _ { f \in \mathcal { A } _ { \mathrm { g l o b a l } } ^ { R } } \| f - h _ { a } \| _ { \infty } = a .
$$

This proves

$$
\operatorname* { s u p } _ { h \in \mathcal { F } _ { b } ^ { R , L } } \operatorname* { i n f } _ { f \in \mathcal { A } _ { \mathrm { g l o b a l } } ^ { R } } \| f - h \| _ { \infty } = \operatorname* { m i n } \left\{ R , \frac { L \sqrt { d } } { 2 } \right\} .
$$

We next prove the local upper bound. Set

$$
q : = \left\lfloor ( M / K ) ^ { 1 / d } \right\rfloor .
$$

Then $q \geq 1$ and $K q ^ { d } \leq M$ . Partition $[ 0 , 1 ] ^ { d }$ into the $q ^ { d }$ cubes $Q _ { \nu }$ defined in the proof of Proposition 2.2, and let $m _ { \nu }$ be their centers. For any fixed $h \in \mathcal { F } _ { b } ^ { R , L }$ , we use the same nearest-center construction a before. Assign K copies of an expert to each center, with

constant value $f _ { \nu , r } ( x ) \equiv h ( m _ { \nu } )$ , and take the logits

$$
u _ { \nu , r } ( x ) = 2 \lambda m _ { \nu } ^ { \top } x - \lambda \| m _ { \nu } \| _ { 2 } ^ { 2 } , \qquad \lambda = \frac { C _ { 1 } } { 2 \operatorname* { m a x } \{ 2 , d \} } .
$$

If $M > K q ^ { d }$ , assign the remaining experts the constant value 0 and the logit $- C _ { 1 }$ . Since $u _ { \nu , r } ( x ) \geq - \lambda d \geq - C _ { 1 } / 2$ for every constructed copy and every $x \in \mathcal { X }$ , the remaining experts have strictly smaller scores than all constructed copies and are never selected by the Top-K rule.

For any $x \in \mathcal { X }$ , maximizing $u _ { \nu , r } ( x )$ over ν is equivalent to minimizing $\Vert \boldsymbol { x } - \boldsymbol { m } _ { \nu } \Vert _ { 2 } ^ { 2 }$ . Hence the selected constructed experts are attached to nearest grid centers. Therefore the output $\ell ( x )$ is a convex combination of the values $h ( \boldsymbol { m } _ { \nu } )$ associated with centers $m _ { \nu } .$ , whose cubes contain x in their closure. For each such center,

$$
\| x - m _ { \nu } \| _ { 2 } \leq \frac { \sqrt { d } } { 2 q } ,
$$

so the Lipschitz condition gives

$$
| h ( x ) - h ( m _ { \nu } ) | \leq \frac { L \sqrt { d } } { 2 q } .
$$

Taking convex combinations yields

$$
| h ( x ) - \ell ( x ) | \leq { \frac { L { \sqrt { d } } } { 2 q } } , \qquad x \in \mathcal { X } .
$$

Hence

$$
\| h - \ell \| _ { \infty } \leq \frac { L \sqrt { d } } { 2 \left\lfloor ( M / K ) ^ { 1 / d } \right\rfloor } .
$$

Taking the supremum over $h \in \mathcal { F } _ { b } ^ { R , L }$ proves the claim.

## A.3 Proof of Theorem 2.4

Proof of Theorem ${ 2 . 4 }$ . For part (i), since $A _ { \mathrm { g l o b a l } } ^ { R } = \mathcal { F } _ { \mathrm { c } } ^ { R }$ is isometric to the interval $[ - R , R ]$ under the metric $\| f _ { c } - f _ { c ^ { \prime } } \| _ { \infty } = | c - c ^ { \prime } |$ , its covering number is the covering number of $[ - R , R ]$

For $0 < \epsilon \le 2 R$ , this is

$$
N \left( \epsilon , \mathcal { A } _ { \mathrm { g l o b a l } } ^ { R } , \| \cdot \| _ { \infty } \right) = \left\lceil \frac { R } { \epsilon } \right\rceil .
$$

Taking logarithm yields the desired order.

For part (ii), if $M = 1$ , then $\mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { s o f t } } ( 1 ) = \mathcal { A } _ { \mathrm { g l o b a l } } ^ { R }$ , so part (i) applies. Assume $M \geq 2$ For the proof, define $T _ { M } : = 2 ( d \mathopen { } \mathclose \bgroup ( d \mathopen { } \mathclose \bgroup ( d \mathopen { } \mathclose \bgroup ( M \aftergroup \egroup ) C _ { 1 } + \log ( M - 1 )$ and $m _ { M } : = \exp \{ - T _ { M } \} / ( 1 { + } \exp \{ -  T _ { M } \} ) ^ { 2 }$ For $\beta \in [ 0 , 2 C _ { 1 } ] ^ { d }$ and $\alpha \in [ 0 , 2 C _ { 1 } ]$ , define

$$
\sigma ( x ) = \frac { 1 } { 1 + \exp \{ - x \} } , \qquad f _ { \beta , \alpha } ( x ) = R \sigma \left( \beta ^ { \top } x + \alpha - \log ( M - 1 ) \right) .
$$

We first check $f _ { \beta , \alpha } \in \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { s o f t } } ( M )$ . Take $c _ { 1 } = R , c _ { 2 } = \cdot \cdot \cdot = c _ { M } = 0$ , and choose the scores

$$
u _ { 1 } ( x ) = { \frac { 1 } { 2 } } ( { \boldsymbol { \beta } } ^ { \top } { \boldsymbol { x } } + { \boldsymbol { \alpha } } ) , \qquad u _ { 2 } ( { \boldsymbol { x } } ) = \dots = u _ { M } ( { \boldsymbol { x } } ) = - { \frac { 1 } { 2 } } ( { \boldsymbol { \beta } } ^ { \top } { \boldsymbol { x } } + { \boldsymbol { \alpha } } ) .
$$

All coeficients lie in $[ - C _ { 1 } , C _ { 1 } ]$ , since each component of $\beta / 2$ and the intercept $\alpha / 2$ belong to $[ 0 , C _ { 1 } ]$ . The softmax weight of expert 1 is

$$
\frac { \exp \{ u _ { 1 } ( x ) \} } { \exp \{ u _ { 1 } ( x ) \} + \sum _ { j = 2 } ^ { M } \exp \{ u _ { j } ( x ) \} } = \frac { 1 } { 1 + ( M - 1 ) \exp \{ - ( \beta ^ { \top } x + \alpha ) \} } = \sigma \left( \beta ^ { \top } x + \alpha - \log ( M - 1 ) \right) .
$$

Thus the resulting output is exactly $f _ { \beta , \alpha }$

For every $x \in [ 0 , 1 ] ^ { d } , \beta \in [ 0 , 2 C _ { 1 } ] ^ { d }$ , and $\alpha \in [ 0 , 2 C _ { 1 } ]$ , the argument of σ satisfies

$$
\begin{array} { r } { - \log ( M - 1 ) \leq \beta ^ { \top } x + \alpha - \log ( M - 1 ) \leq 2 ( d + 1 ) C _ { 1 } - \log ( M - 1 ) . } \end{array}
$$

Since $T _ { M } = 2 ( d + 1 ) C _ { 1 } + \log ( M - 1 )$ , this argument lies in $[ - T _ { M } , T _ { M } ]$ . Hence

$$
\sigma ^ { \prime } ( t ) \geq m _ { M } \qquad \mathrm { f o r ~ a l l ~ } | t | \leq T _ { M } .
$$

For the lower bound on the covering number, take $\begin{array} { r } { \delta : = \frac { 3 \epsilon } { R m _ { M } } } \end{array}$ . Let $\Gamma _ { \beta }$ be a grid of $[ 0 , 2 C _ { 1 } ] ^ { d }$ with mesh size δ in each coordinate, and let $\Gamma _ { \alpha }$ be a grid of $[ 0 , 2 C _ { 1 } ]$ with mesh size

δ. Then

$$
| \Gamma _ { \beta } | = \bigg ( \bigg | \frac { 2 R m _ { M } C _ { 1 } } { 3 \epsilon } \bigg | + 1 \bigg ) ^ { d } , \qquad | \Gamma _ { \alpha } | = \bigg | \frac { 2 R m _ { M } C _ { 1 } } { 3 \epsilon } \bigg | + 1 .
$$

Consider $\mathcal { H } = \{ f _ { \beta , \alpha } : \beta \in \Gamma _ { \beta } , \alpha \in \Gamma _ { \alpha } \}$ . Distinct elements of H are 3ϵ-separated under $\| \cdot \| _ { \infty }$ Indeed, take $( \beta , \alpha ) \neq ( \beta ^ { \prime } , \alpha ^ { \prime } )$ . If $\alpha \neq \alpha ^ { \prime }$ , then evaluate at $x = 0$ gives

$$
\begin{array} { r } { \| f _ { \beta , \alpha } - f _ { \beta ^ { \prime } , \alpha ^ { \prime } } \| _ { \infty } \geq | f _ { \beta , \alpha } ( 0 ) - f _ { \beta ^ { \prime } , \alpha ^ { \prime } } ( 0 ) | \geq R m _ { M } | \alpha - \alpha ^ { \prime } | \geq 3 \epsilon . } \end{array}
$$

If $\alpha = \alpha ^ { \prime }$ , then some coordinate $\beta _ { j } \neq \beta _ { j } ^ { \prime }$ . Evaluating at $\boldsymbol { x } = \boldsymbol { e } _ { j }$ gives

$$
\| f _ { \beta , \alpha } - f _ { \beta ^ { \prime } , \alpha } \| _ { \infty } \geq | f _ { \beta , \alpha } ( e _ { j } ) - f _ { \beta ^ { \prime } , \alpha } ( e _ { j } ) | \geq R m _ { M } | \beta _ { j } - \beta _ { j } ^ { \prime } | \geq 3 \epsilon .
$$

Therefore $\mathcal { H }$ is 3ϵ-separated. Since every ball of radius ϵ has diameter at most $2 \epsilon .$ each such ball contains at most one element of $\mathcal { H } .$ . It follows that

$$
N \left( \epsilon , \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { s o f t } } ( M ) , \lVert \cdot \rVert _ { \infty } \right) \ge | \mathcal { H } | = \left( \left\lfloor \frac { 2 R m _ { M } C _ { 1 } } { 3 \epsilon } \right\rfloor + 1 \right) ^ { d + 1 } .
$$

We next prove the upper bound of the covering number. Fix $h \in \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { s o f t } } ( M )$ . Then

$$
h ( x ) = \sum _ { i = 1 } ^ { M } s _ { i } ( x ) c _ { i } , \qquad | c _ { i } | \leq R ,
$$

where

$$
s _ { i } ( \boldsymbol { x } ) = g _ { i } ^ { \mathrm { s o f t } } ( \boldsymbol { x } ; \theta ) = \frac { \exp \{ \beta _ { i } ^ { \top } \boldsymbol { x } + \alpha _ { i } \} } { \sum _ { j = 1 } ^ { M } \exp \{ \beta _ { j } ^ { \top } \boldsymbol { x } + \alpha _ { j } \} } , \qquad ( \beta _ { i } ^ { \top } , \alpha _ { i } ) ^ { \top } \in \Theta _ { \mathrm { l i n } } ( C _ { 1 } ) .
$$

Using expert M as a reference, define

$$
\begin{array} { r l r l r l r l } { v _ { 0 } : = c _ { M } , } & & { \boldsymbol { v } _ { 1 } : = c _ { i } - c _ { M } \in [ - 2 R , 2 R ] , } & & { \boldsymbol { a } _ { i } : = \beta _ { i } - \beta _ { M } \in [ - 2 C _ { 1 } , 2 C _ { 1 } ] ^ { d } , } & & { \boldsymbol { b } _ { i } : = \alpha _ { i } - \alpha _ { M } \in [ - 2 C _ { 1 } , 2 C _ { 1 } ] } \end{array}
$$

for $i = 1 , \dots , M - 1$ . Then $\begin{array} { r } { h ( x ) = v _ { 0 } + \sum _ { i = 1 } ^ { M - 1 } v _ { i } q _ { i } ( x ) } \end{array}$ , where

$$
q _ { i } ( x ) = \frac { \exp \{ a _ { i } ^ { \top } x + b _ { i } \} } { 1 + \sum _ { j = 1 } ^ { M - 1 } \exp \{ a _ { j } ^ { \top } x + b _ { j } \} } .
$$

Let h and h<sup>′</sup> correspond to parameter tuples $( v _ { 0 } , v , a , b )$ and $( v _ { 0 } ^ { \prime } , v ^ { \prime } , a ^ { \prime } , b ^ { \prime } )$ . Then

$$
\left| h ( x ) - h ^ { \prime } ( x ) \right| \leq | v _ { 0 } - v _ { 0 } ^ { \prime } | + \sum _ { i = 1 } ^ { M - 1 } | v _ { i } - v _ { i } ^ { \prime } | q _ { i } ( x ) + \sum _ { i = 1 } ^ { M - 1 } | v _ { i } ^ { \prime } | | q _ { i } ( x ) - q _ { i } ^ { \prime } ( x ) | .
$$

Since $\textstyle \sum _ { i = 1 } ^ { M - 1 } q _ { i } ( x ) \leq 1$

$$
\sum _ { i = 1 } ^ { M - 1 } | v _ { i } - v _ { i } ^ { \prime } | q _ { i } ( x ) \leq \| v - v ^ { \prime } \| _ { \infty } .
$$

Also $| v _ { i } ^ { \prime } | \leq 2 R .$

It remains to control the diference between q and $q ^ { \prime } .$ . Let

$$
S ( z _ { 1 } , \dots , z _ { M } ) = \left( \frac { \exp \{ z _ { 1 } \} } { \sum _ { j = 1 } ^ { M } \exp \{ z _ { j } \} } , \dots , \frac { \exp \{ z _ { M } \} } { \sum _ { j = 1 } ^ { M } \exp \{ z _ { j } \} } \right)
$$

be the M-dimensional softmax map. Its Jacobian satisfies

$$
D S ( z ) v = p \odot ( v - \langle p , v \rangle \mathbf { 1 } ) , \qquad p = S ( z ) ,
$$

where $\odot$ denotes the Hadamard product. Therefore,

$$
\| D S ( z ) v \| _ { 1 } = \sum _ { i = 1 } ^ { M } p _ { i } | v _ { i } - \langle p , v \rangle | \leq 2 \| v \| _ { \infty } .
$$

By the mean value theorem, $\| S ( z ) - S ( z ^ { \prime } ) \| _ { 1 } \leq 2 \| z - z ^ { \prime } \| _ { \infty }$ . Applying this to

$$
z ( x ) = ( a _ { 1 } ^ { \top } x + b _ { 1 } , \ldots , a _ { M - 1 } ^ { \top } x + b _ { M - 1 } , 0 ) , \qquad z ^ { \prime } ( x ) = ( a _ { 1 } ^ { \prime \top } x + b _ { 1 } ^ { \prime } , \ldots , a _ { M - 1 } ^ { \prime } { } ^ { \top } x + b _ { M - 1 } ^ { \prime } , 0 ) ,
$$

we obtain

$$
\sum _ { i = 1 } ^ { M - 1 } | q _ { i } ( x ) - q _ { i } ^ { \prime } ( x ) | \leq 2 \| z ( x ) - z ^ { \prime } ( x ) \| _ { \infty } .
$$

Since $x \in [ 0 , 1 ] ^ { d }$

$$
\| z ( x ) - z ^ { \prime } ( x ) \| _ { \infty } \leq d \operatorname* { m a x } _ { 1 \leq i \leq M - 1 } \| a _ { i } - a _ { i } ^ { \prime } \| _ { \infty } + \operatorname* { m a x } _ { 1 \leq i \leq M - 1 } | b _ { i } - b _ { i } ^ { \prime } | .
$$

Combining the estimates gives

$$
\| h - h ^ { \prime } \| _ { \infty } \leq | v _ { 0 } - v _ { 0 } ^ { \prime } | + \| v - v ^ { \prime } \| _ { \infty } + 4 R \left( d \operatorname* { m a x } _ { 1 \leq i \leq M - 1 } \| a _ { i } - a _ { i } ^ { \prime } \| _ { \infty } + \operatorname* { m a x } _ { 1 \leq i \leq M - 1 } | b _ { i } - b _ { i } ^ { \prime } | \right) .
$$

Now choose parameter nets satisfying

$$
| v _ { 0 } - v _ { 0 } ^ { \prime } | \le \frac { \epsilon } { 3 } , \qquad \| v - v ^ { \prime } \| _ { \infty } \le \frac { \epsilon } { 3 } , \qquad \operatorname* { m a x } _ { i } \| a _ { i } - a _ { i } ^ { \prime } \| _ { \infty } \le \frac { \epsilon } { 2 4 R d } , \qquad \operatorname* { m a x } _ { i } | b _ { i } - b _ { i } ^ { \prime } | \le \frac { \epsilon } { 2 4 R } .
$$

Then $\| h - h ^ { \prime } \| _ { \infty } \leq \epsilon$ . The interval $[ - R , R ]$ can be covered at accuracy $\epsilon / 3$ by $\lceil \frac { 3 R } { \epsilon } \rceil$ points. Each interval $[ - 2 R , 2 R ]$ can be covered at the same accuracy by $\lceil \frac { 6 R } { \epsilon } \rceil$ points. Each interval $[ - 2 C _ { 1 } , 2 C _ { 1 } ]$ can be covered at accuracy $\epsilon / ( 2 4 R d )$ by $\left\lceil \frac { 4 8 R d C _ { 1 } } { \epsilon } \right\rceil$ points, and at accuracy $\epsilon / ( 2 4 R )$ by $\lceil \frac { 4 8 R C _ { 1 } } { \epsilon } \rceil$ points. Since there are $M - 1$ scalar parameters $v _ { i } , ( M - 1 ) d$ coordinates in the vectors ${ { a } _ { i } } ,$ , and $M - 1$ intercept diferences $b _ { i }$ , the stated upper bound follows.

Finally, taking logarithms yields

$$
( d + 1 ) \log ( 1 / \epsilon ) + O ( 1 ) \leq \log N \left( \epsilon , \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { s o f t } } ( M ) , \lVert \cdot \rVert _ { \infty } \right) \leq [ M + ( M - 1 ) ( d + 1 ) ] \log ( 1 / \epsilon ) + O ( 1 )
$$

for fixed $M \geq 2$

For part (iii), fix $1 \leq K < M$ . For each $t \in ( 0 , 1 )$ , define experts

$$
f _ { 1 } \equiv R , \qquad f _ { 2 } \equiv - R , \qquad f _ { 3 } = \cdots = f _ { M } \equiv 0 ,
$$

and scores

$$
u _ { 1 } ^ { ( t ) } ( x ) = C _ { 1 } ( x _ { 1 } - t ) , \qquad u _ { 2 } ^ { ( t ) } ( x ) = - C _ { 1 } ( x _ { 1 } - t ) , \qquad u _ { 3 } ^ { ( t ) } ( x ) = \cdots = u _ { M } ^ { ( t ) } ( x ) = 0 .
$$

All these parameters lie in $\Theta _ { \mathrm { l i n } } ( C _ { 1 } )$ . Let $h _ { t } \in \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p } - K } ( M )$ be the resulting function.

Take $s < t$ and set $x = ( ( s + t ) / 2 , 0 , \ldots , 0 ) ^ { \top }$ . Then $x _ { 1 } = ( s + t ) / 2$ lies strictly between s and t. Hence

$$
u _ { 1 } ^ { ( s ) } ( x ) > 0 , \qquad u _ { 2 } ^ { ( s ) } ( x ) < 0 ,
$$

Thus the Top-K rule for $h _ { s }$ selects expert 1 together with K − 1 zero experts. Therefore

$$
h _ { s } ( x ) = R \cdot \frac { \exp \{ u _ { 1 } ^ { ( s ) } ( x ) \} } { \exp \{ u _ { 1 } ^ { ( s ) } ( x ) \} + ( K - 1 ) } > \frac { R } { K } .
$$

Similarly,

$$
u _ { 1 } ^ { ( t ) } ( x ) < 0 , \qquad u _ { 2 } ^ { ( t ) } ( x ) > 0 .
$$

Thus the Top-K rule for $h _ { t }$ selects expert 2 together with $K - 1$ zero experts, and

$$
h _ { t } ( x ) = - R \cdot \frac { \exp \{ u _ { 2 } ^ { ( t ) } ( x ) \} } { \exp \{ u _ { 2 } ^ { ( t ) } ( x ) \} + ( K - 1 ) } < - \frac { R } { K } .
$$

Consequently,

$$
\| h _ { s } - h _ { t } \| _ { \infty } \geq | h _ { s } ( x ) - h _ { t } ( x ) | > \frac { 2 R } { K } .
$$

Thus $\{ h _ { t } : t \in ( 0 , 1 ) \}$ is an uncountable 2R/K-separated family. Therefore, for every $0 < \epsilon \leq R / K$ , every ball of radius ϵ has diameter at most $2 \epsilon \leq 2 R / K$ and hence contains at most one element of this family. Consequently,

$$
N \left( \epsilon , \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p - } K } ( M ) , \| \cdot \| _ { \infty } \right) = \infty .
$$

## A.4 Proof of Example 2.5

Proof of Example 2.5. For the local representation, take two experts

$$
e _ { 1 } ( x ) = \sigma ( x _ { 1 } ) , \qquad e _ { 2 } ( x ) \equiv 0 .
$$

Both experts belong to $\mathcal { E } _ { \sigma }$ . Choose the scores

$$
u _ { 1 } ( x ) = x _ { 1 } , \qquad u _ { 2 } ( x ) = 0 .
$$

Then the softmax weight of expert 1 is

$$
{ \frac { \exp ( x _ { 1 } ) } { \exp ( x _ { 1 } ) + 1 } } = \sigma ( x _ { 1 } ) ,
$$

and the resulting local average is

$$
{ \frac { \exp ( x _ { 1 } ) } { \exp ( x _ { 1 } ) + 1 } } e _ { 1 } ( x ) + { \frac { 1 } { \exp ( x _ { 1 } ) + 1 } } e _ { 2 } ( x ) = \sigma ( x _ { 1 } ) ^ { 2 } = h ( x ) .
$$

Thus $h \in \mathcal { A } _ { \mathrm { l o c a l } } ^ { \mathrm { s o f t } } ( \mathcal { E } _ { \sigma } ; 2 )$

We next show that h /∈ $\textstyle \bigcup _ { M > 1 } { \mathcal { A } } _ { \mathrm { g l o b a l } } ( { \mathcal { E } } _ { \sigma } ; M )$ . Suppose otherwise. Then for some $M ,$

$$
h ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { M } w _ { i } e _ { i } ( \boldsymbol { x } ) , \qquad w _ { i } \geq 0 , \qquad \sum _ { i = 1 } ^ { M } w _ { i } = 1 ,
$$

with $e _ { i } ( x ) = c _ { 0 , i } + c _ { 1 , i } \sigma ( a _ { i } x _ { 1 } + b _ { i } )$ . Collecting the constants, we obtain an identity on $[ 0 , 1 ]$

$$
\sigma ( t ) ^ { 2 } = \beta _ { 0 } + \sum _ { i = 1 } ^ { M } \beta _ { i } \sigma ( a _ { i } t + b _ { i } ) , \qquad t \in [ 0 , 1 ] ,
$$

for suitable real coeficients $\beta _ { 0 } , \ldots , \beta _ { M }$ . Although the aggregation weights $w _ { i }$ are nonnegative and sum to one, the coeficients $\beta _ { i }$ need not be nonnegative, since the expert coeficients $c _ { 1 , i }$ are arbitrary real numbers.

Now view both sides as meromorphic functions of a complex variable z. The logistic function $\begin{array} { r } { \sigma ( z ) = \frac { 1 } { 1 + \exp ( - z ) } } \end{array}$ has simple poles at $z = ( 2 \ell + 1 ) \pi i$ for $\ell \in \mathbb { Z }$ . Therefore each nonconstant function $\sigma ( a _ { i } z + b _ { i } )$ has only simple poles, while if $a _ { i } = 0$ , the function is constant and has no pole. Hence any finite linear combination

$$
\beta _ { 0 } + \sum _ { i = 1 } ^ { M } \beta _ { i } \sigma ( a _ { i } z + b _ { i } )
$$

has at most simple poles at every point of $\mathbb { C } .$ On the other hand, $\sigma ( z ) ^ { 2 }$ has a second-order pole at $z = \pi i$

The two meromorphic functions agree on the real interval [0, 1], which has an accumulation point in a domain where both sides are analytic. $\mathrm { B y }$ the identity theorem for meromorphic functions, they must agree as meromorphic functions. This is impossible,

because the left-hand side has a second-order pole at $z = \pi i .$ while the right-hand side can have at most a simple pole there. Hence h $\begin{array} { r } { \notin \bigcup _ { M \ge 1 } \mathcal { A } _ { \mathrm { g l o b a l } } ( \mathcal { E } _ { \sigma } ; M ) } \end{array}$ □

## A.5 Proof of Proposition 2.6

Proof of Proposition 2.6. Fix any $w \in [ 0 , 1 ]$ and write $\hat { f } _ { w } ^ { \mathrm { g l o b a l } } : = w \hat { f } _ { 1 } + ( 1 - w ) \hat { f } _ { 2 }$ . On $\mathcal { X } _ { 1 }$ the reverse triangle inequality gives

$$
\begin{array} { r l } & { \rVert \hat { f } _ { w } ^ { \mathrm { g l o b a l } } - f \rVert _ { \mathcal { X } _ { 1 } } = \rVert w ( \hat { f } _ { 1 } - f ) + ( 1 - w ) ( \hat { f } _ { 2 } - f ) \rVert _ { \mathcal { X } _ { 1 } } } \\ & { \qquad \ge ( 1 - w ) \rVert \hat { f } _ { 2 } - f \rVert _ { \mathcal { X } _ { 1 } } - w \rVert \hat { f } _ { 1 } - f \rVert _ { \mathcal { X } _ { 1 } } } \\ & { \qquad = ( 1 - w ) \delta _ { 1 } - w \epsilon _ { 1 } . } \end{array}
$$

Similarly, on $\mathcal { X } _ { 2 }$

$$
\begin{array} { r l } & { \rVert \hat { f } _ { w } ^ { \mathrm { g l o b a l } } - f \rVert _ { \mathcal { X } _ { 2 } } = \rVert w ( \hat { f } _ { 1 } - f ) + ( 1 - w ) ( \hat { f } _ { 2 } - f ) \rVert _ { \mathcal { X } _ { 2 } } } \\ & { \qquad \ge w \rVert \hat { f } _ { 1 } - f \rVert _ { \mathcal { X } _ { 2 } } - ( 1 - w ) \rVert \hat { f } _ { 2 } - f \rVert _ { \mathcal { X } _ { 2 } } } \\ & { \qquad = w \delta _ { 2 } - ( 1 - w ) \epsilon _ { 2 } . } \end{array}
$$

Hence

$$
\operatorname* { m a x } _ { j = 1 , 2 } \| \hat { f } _ { w } ^ { \mathrm { g l o b a l } } - f \| _ { \mathcal { X } _ { j } } \ge \operatorname* { m a x } \left\{ ( 1 - w ) \delta _ { 1 } - w \epsilon _ { 1 } , w \delta _ { 2 } - ( 1 - w ) \epsilon _ { 2 } \right\} _ { + } .
$$

The first afine term is decreasing in $w ,$ , while the second is increasing in $w$ . Balancing them gives

$$
w _ { \star } = { \frac { \delta _ { 1 } + \epsilon _ { 2 } } { \delta _ { 1 } + \delta _ { 2 } + \epsilon _ { 1 } + \epsilon _ { 2 } } } ,
$$

and the common value is $( \delta _ { 1 } \delta _ { 2 } - \epsilon _ { 1 } \epsilon _ { 2 } ) / ( \delta _ { 1 } + \delta _ { 2 } + \epsilon _ { 1 } + \epsilon _ { 2 } )$ . Taking the positive part yields the lower bound $\gamma$ .

Since

$$
R ( \hat { f } _ { \boldsymbol { w } } ^ { \mathrm { g l o b a l } } ) = \| \hat { f } _ { \boldsymbol { w } } ^ { \mathrm { g l o b a l } } - f \| _ { \mathcal { X } _ { 1 } } ^ { 2 } + \| \hat { f } _ { \boldsymbol { w } } ^ { \mathrm { g l o b a l } } - f \| _ { \mathcal { X } _ { 2 } } ^ { 2 } ,
$$

at least one regional norm being no smaller than $\gamma$ implies $R ( \hat { f } _ { w } ^ { \mathrm { g l o b a l } } ) \ge \gamma ^ { 2 }$ . Taking the infimum over $w \in [ 0 , 1 ]$ gives the desired lower bound for $\mathscr { A } _ { \mathrm { g l o b a l } } ( \hat { f } _ { 1 } , \hat { f } _ { 2 } )$ . Finally, by the definition of $\hat { f } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p - 1 } }$ ，

$$
\begin{array} { r } { R ( \hat { f } _ { \mathrm { l o c a l } } ^ { \mathrm { T o p - 1 } } ) = \| \hat { f } _ { 1 } - f \| _ { \mathcal { X } _ { 1 } } ^ { 2 } + \| \hat { f } _ { 2 } - f \| _ { \mathcal { X } _ { 2 } } ^ { 2 } = \epsilon _ { 1 } ^ { 2 } + \epsilon _ { 2 } ^ { 2 } . } \end{array}
$$

This proves the proposition.

## B Proof of Section 3

## B.1 Proof of Theorem 3.8

We consider the same regression model as in Section 3.1 of the main article. We first introduce the following generalized version of AFTER algorithm (Yang, 2004) in Algorithm B.1, which allows the aggregation of covariate-dependent forecasting procedures. Throughout this subsection, all candidate forecasts and scale estimators are understood to satisfy the non-anticipation convention stated in Section 3.1. In particular, the quantities used to predict $Y _ { t + 1 }$ may depend on the observations available up to time $t ,$ the auxiliary algorithmic randomization, and the newly observed covariate $X _ { t + 1 }$ , but not on $Y _ { t + 1 }$ or future observations.

We first introduce the assumptions needed to establish the theory for the introduced algorithm.

Assumption B.1 (Uniform boundedness). There exists a constant $A < \infty$ such that, almost surely for $t \geq n _ { 1 }$

$$
\| f _ { 0 } \| _ { \infty } \vee \operatorname* { m a x } _ { s \in [ S ] } \| \mu _ { s , t } \| _ { \infty } \le A .
$$

Assumption B.2. Assume that the produced scale estimators $\{ \hat { \sigma } _ { s , t } \} _ { s = 1 } ^ { S }$ satisfies $0 < \underline { { \sigma } } \le$ $\hat { \sigma } _ { s , t } \leq \bar { \sigma } < \infty$ , where $\underline { { \sigma } }$ and ¯σ are constants in Assumption 3.7.

Here, Assumption B.2 can be reached through simple truncation. Then, we have the following result.

Algorithm B.1: General AFTER Algorithm   
Input: A burn-in size $n _ { 1 } ;$ a class of S forecasts $\{ \mu _ { s , t } ( x ) \} _ { s = 1 } ^ { S } ;$ a black-box scale   
estimator; data $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n } ;$ prior weights $\{ \pi _ { s } \} _ { s = 1 } ^ { S } .$   
Output: An online aggregated sequence $\{ \tilde { \mu } _ { t } ( x ) \} _ { t = n _ { 1 } } ^ { n - 1 } .$   
(a) Randomly partition the data into two subsamples $\bar { Z } ^ { ( 1 ) } = ( X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n _ { 1 } }$ and   
$Z ^ { ( 2 ) } = ( \bar { X _ { i } , Y _ { i } } ) _ { i = n _ { 1 } + 1 } ^ { n } ,$ and let $n _ { 2 } = n - n _ { 1 }$ . The first subsample is used for burn-in, and the   
second is for aggregation.   
(b) Burn-in: Process $Z ^ { ( 1 ) }$ to obtain the initial forecasts $\mu _ { s , n _ { 1 } } ( x )$ and initial scale estimators   
$\sigma _ { s , n _ { 1 } }$   
(c) Aggregation: Initialize $w _ { s , n _ { 1 } } = \pi _ { s }$ . For $t = n _ { 1 } , \ldots , n - 1 { \mathrm { : } }$   
(i) Construct the aggregated forecast estimator   
$\tilde { \mu } _ { t } ( x ) = \sum _ { s = 1 } ^ { S } w _ { s , t } \mu _ { s , t } ( x ) .$   
(ii) Denote the one-step-ahead joint density of $\boldsymbol Z _ { t + 1 } = ( X _ { t + 1 } , Y _ { t + 1 } )$ under candidate s at   
time t, with respect to the product measure $P _ { X } ( d x ) d y$ , by   
$q _ { s , t } ( x , y ) = \frac { 1 } { \hat { \sigma } _ { s , t } } h _ { \varepsilon , 0 } \left( \frac { y - \mu _ { s , t } ( x ) } { \hat { \sigma } _ { s , t } } \right)$   
(iii) After observing $Z _ { t + 1 }$ , update the weights $w _ { s , t + 1 }$ by   
$w _ { s , t + 1 } = \frac { w _ { s , t } q _ { s , t } ( X _ { t + 1 } , Y _ { t + 1 } ) } { \sum _ { r = 1 } ^ { S } w _ { r , t } q _ { r , t } ( X _ { t + 1 } , Y _ { t + 1 } ) } .$ (B.1)   
(iv) Using $Z _ { t + 1 }$ and previous data, update the scale estimators by its black-box rule to   
obtain $\{ \hat { \sigma } _ { s , t + 1 } \} _ { s = 1 } ^ { S }$ , then evolve forecasts to obtain $\{ \mu _ { s , t + 1 } ( x ) \} _ { s = 1 } ^ { S }$

Proposition B.3 (Risk bound for general AFTER algoritm). Let $\{ \tilde { \mu } _ { t } \} _ { t = n _ { 1 } } ^ { n - 1 }$ be the online   
sequence produced by Algorithm B.1. Assume the data generation mechanism satisfies As  
sumption 3.7, and Assumptions B.1, B.2 hold. Then, there exists a constant C depending   
only on the boundedness and error-distribution constants, such that   
$\frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n - 1 } \mathbb { E } \| \tilde { \mu } _ { t } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \operatorname* { i n f } _ { s \leq S } \left( \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n - 1 } \mathbb { E } \| \mu _ { s , t } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \frac { \log ( 1 / \pi _ { s } ) } { n _ { 2 } } + \mathcal { V } _ { s } \right)$ , (B.2)   
where   
$\mathcal { V } _ { s } = \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n - 1 } \mathbb { E } \left( \hat { \sigma } _ { s , t } - \sigma \right) ^ { 2 } .$

Proof of Proposition B.3. Let

$$
p _ { t } ( x , y ) = \frac { 1 } { \sigma } h _ { \varepsilon , 0 } \left( \frac { y - f _ { 0 } ( x ) } { \sigma } \right)
$$

be the true joint density of $Z _ { t + 1 }$ . Define the mixture density

$$
q _ { t } ( x , y ) = \sum _ { s = 1 } ^ { S } w _ { s , t } q _ { s , t } ( x , y ) .
$$

The AFTER recursion (B.1) yields

$$
w _ { s , t } = \frac { \pi _ { s } \prod _ { \ell = n _ { 1 } } ^ { t - 1 } q _ { s , \ell } ( X _ { \ell + 1 } , Y _ { \ell + 1 } ) } { \sum _ { r = 1 } ^ { S } \pi _ { r } \prod _ { \ell = n _ { 1 } } ^ { t - 1 } q _ { r , \ell } ( X _ { \ell + 1 } , Y _ { \ell + 1 } ) } ,
$$

which results in

$$
q _ { t } ( X _ { t + 1 } , Y _ { t + 1 } ) = \frac { \sum _ { r = 1 } ^ { S } \pi _ { r } \prod _ { \ell = n _ { 1 } } ^ { t } q _ { r , \ell } ( X _ { \ell + 1 } , Y _ { \ell + 1 } ) } { \sum _ { r = 1 } ^ { S } \pi _ { r } \prod _ { \ell = n _ { 1 } } ^ { t - 1 } q _ { r , \ell } ( X _ { \ell + 1 } , Y _ { \ell + 1 } ) } ,
$$

and thus we have the standard likelihood identity

$$
\prod _ { t = n _ { 1 } } ^ { n - 1 } q _ { t } ( X _ { t + 1 } , Y _ { t + 1 } ) = \sum _ { r = 1 } ^ { S } \pi _ { r } \prod _ { t = n _ { 1 } } ^ { n - 1 } q _ { r , t } ( X _ { t + 1 } , Y _ { t + 1 } ) .
$$

Thus, for a fixed $s \in [ S ]$ , using a similar argument as in the proof of Theorem 1 in (Yang, 2001), we have

$$
\begin{array} { r l } {  { \sum _ { t = n _ { 1 } } ^ { n - 1 } \mathbb { E } [ D ( p _ { t } \| q _ { t } ) ] = \mathbb { E } [ \log \frac { \prod _ { t = n _ { 1 } } ^ { n - 1 } p _ { t } ( X _ { t + 1 } , Y _ { t + 1 } ) } { \prod _ { t = n _ { 1 } } ^ { n - 1 } q _ { t } ( X _ { t + 1 } , Y _ { t + 1 } ) } ] } \quad } & { } \\ & { \leq \mathbb { E } [ \log \frac { \prod _ { t = n _ { 1 } } ^ { n - 1 } p _ { t } ( X _ { t + 1 } , Y _ { t + 1 } ) } { \pi _ { s } \prod _ { t = n _ { 1 } } ^ { n - 1 } q _ { s , t } ( X _ { t + 1 } , Y _ { t + 1 } ) } ] } \\ & { = \log ( 1 / \pi _ { s } ) + \sum _ { t = n _ { 1 } } ^ { n - 1 } \mathbb { E } [ D ( p _ { t } \| q _ { s , t } ) ] . } \end{array}\tag{B.3}
$$

Here, the densities $p _ { t } , \ q _ { t }$ , and ${ { q } _ { s , t } }$ are conditional one-step-ahead densities given the online history up to time t. Accordingly, $D ( p _ { t } \| q _ { t } )$ is a conditional KL divergence and is itself a random variable through its dependence on the past data and the algorithmic randomness.

Thus $\mathbb { E } [ D ( p _ { t } \| q _ { t } ) ]$ denotes the outer expectation of this conditional KL divergence. The expectations in the logarithmic display are taken over the full data path and the algorithmic randomness.

Now, a similar argument as in (Yang, 2001) shows that

$$
\begin{array} { l }  \displaystyle { { D ( p _ { t } | | q _ { s , t } ) = \int ( \int \frac { 1 } { \sigma } h _ { \varepsilon , 0 } ( \frac { y - f _ { 0 } ( x ) } { \sigma } ) \log ( \frac { \sigma ^ { - 1 } h _ { \varepsilon , 0 } ( ( y - f _ { 0 } ( x ) ) / \sigma ) } { \widehat { \sigma } _ { s , t } ^ { - 1 } h _ { \varepsilon , 0 } ( ( y - \mu _ { s , t } ( x ) ) / \widehat { \sigma } _ { s , t } ) } ) d y ) P _ { X } ( d x ) } } \\ { \displaystyle { \quad \quad = \int ( h _ { \varepsilon , 0 } ( y ) \log ( \frac { h _ { \varepsilon , 0 } ( y ) } { \sigma / \widehat { \sigma } _ { s , t } h _ { \varepsilon , 0 } ( \sigma ( y - \sigma ^ { - 1 } ( \mu _ { s , t } ( x ) - f _ { 0 } ( x ) ) ) / \widehat { \sigma } _ { s , t } ) } ) d y ) P _ { X } ( d x ) } } \\ { \displaystyle { \quad \leq C \int ( ( 1 - \frac { \widehat { \sigma } _ { s , t } } { \sigma } ) ^ { 2 } + ( \frac { \mu _ { s , t } ( x ) - f _ { 0 } ( x ) } { \sigma } ) ^ { 2 } ) P _ { X } ( d x ) } } \\ { \displaystyle { \quad \leq C ( \| \mu _ { s , t } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + ( \widehat { \sigma } _ { s , t } - \sigma ) ^ { 2 } ) . } } \end{array}\tag{4}
$$

Here the third line is by Assumption 3.7, where Assumption B.2 ensures that $\hat { \sigma } _ { s , t } / \sigma$ is bounded and Assumption B.1 together with $\underline { { \sigma } } \le \sigma \le \bar { \sigma }$ ensure that $( \mu _ { s , t } ( x ) - f _ { 0 } ( x ) ) / \sigma$ is uniformly bounded.

Now, consider the squared Hellinger distance $H ^ { 2 } ( p , q )$ . We use the convention

$$
H ^ { 2 } ( p , q ) : = \int ( { \sqrt { p } } - { \sqrt { q } } ) ^ { 2 } d \mu .
$$

The standard Hellinger inequality gives

$$
H ^ { 2 } ( p , q ) \leq D ( p | | q ) .
$$

Moreover, for densities $p$ and $q$ with uniformly bounded means $m _ { p } , m _ { q }$ and variance $v _ { p } , v _ { q }$ an elementary moment-Hellinger bound

$$
H ^ { 2 } ( p , q ) \geq \frac { ( m _ { p } - m _ { q } ) ^ { 2 } } { 2 ( v _ { p } + v _ { q } ) + ( m _ { p } - m _ { q } ) ^ { 2 } } .
$$

Indeed, let $\Delta = m _ { p } - m _ { q }$ and $a = ( m _ { p } + m _ { q } ) / 2$ , then using Cauchy Schwarz’s inequality, we have

$$
\Delta ^ { 2 } = \left( \int x ( p - q ) d \mu \right) ^ { 2 } = \left( \int ( x - a ) ( p - q ) d x \right) ^ { 2 }
$$

$$
\begin{array} { l } { \displaystyle \leq \left( \int ( \sqrt { p } - \sqrt { q } ) ^ { 2 } d x \right) \left( \int ( x - a ) ^ { 2 } ( \sqrt { p } + \sqrt { q } ) ^ { 2 } d x \right) } \\ { \displaystyle \leq 2 H ^ { 2 } ( p , q ) \left( \int ( x - a ) ^ { 2 } ( p + q ) d x \right) } \\ { \displaystyle = 2 H ^ { 2 } ( p , q ) \left( v _ { p } + v _ { q } + \frac { \Delta ^ { 2 } } { 2 } \right) , } \end{array}
$$

and thus the desired inequality holds. Now, if $H ^ { 2 } ( p , q ) \leq 1 / 2$ , then $( m _ { p } - m _ { q } ) ^ { 2 } \leq 4 ( v _ { p } +$ $v _ { q } ) H ^ { 2 } ( p , q )$ . If $H ^ { 2 } ( p , q ) > 1 / 2$ , the uniform mean bound gives $( m _ { p } - m _ { q } ) ^ { 2 } \leq C H ^ { 2 } ( p , q )$ after enlarging C. Thus, in all cases,

$$
\begin{array} { r } { ( m _ { p } - m _ { q } ) ^ { 2 } \leq C H ^ { 2 } ( p , q ) \leq C D ( p \| q ) . } \end{array}
$$

We now apply this inequality conditionally on the online history and on the covariate value. For fixed x and time $t ,$ let

$$
p _ { t } ^ { x } ( y ) = \sigma ^ { - 1 } h _ { \varepsilon , 0 } \left( \frac { y - f _ { 0 } ( x ) } { \sigma } \right) , \qquad q _ { t } ^ { x } ( y ) = \sum _ { s = 1 } ^ { S } w _ { s , t } \hat { \sigma } _ { s , t } ^ { - 1 } h _ { \varepsilon , 0 } \left( \frac { y - \mu _ { s , t } ( x ) } { \hat { \sigma } _ { s , t } } \right) .
$$

The conditional mean of $p _ { t } ^ { x }$ is $f _ { 0 } ( x )$ , while the conditional mean of $q _ { t } ^ { x }$ is

$$
\sum _ { s = 1 } ^ { S } w _ { s , t } \mu _ { s , t } ( x ) = \tilde { \mu } _ { t } ( x ) .
$$

Let $v _ { 0 }$ denote the variance of $h _ { \varepsilon , 0 }$ . The conditional variance of $p _ { t } ^ { x }$ is $\sigma ^ { 2 } v _ { 0 }$ . The conditional variance of $q _ { t } ^ { x }$ is

$$
\sum _ { s = 1 } ^ { S } w _ { s , t } \hat { \sigma } _ { s , t } ^ { 2 } v _ { 0 } + \sum _ { s = 1 } ^ { S } w _ { s , t } \{ \mu _ { s , t } ( x ) - \tilde { \mu } _ { t } ( x ) \} ^ { 2 } .
$$

By Assumptions B.1 and B.2, this variance is bounded above by $\bar { \sigma } ^ { 2 } v _ { 0 } + 4 A ^ { 2 }$ . Moreover, both conditional means are uniformly bounded. Hence the preceding moment-Hellinger bound implies that, for a constant C independent of s, t and $x ,$

$$
\begin{array} { r } { ( \tilde { \mu } _ { t } ( x ) - f _ { 0 } ( x ) ) ^ { 2 } \le C H ^ { 2 } ( p _ { t } ^ { x } , q _ { t } ^ { x } ) \le C D ( p _ { t } ^ { x } \| q _ { t } ^ { x } ) . } \end{array}
$$

Integrating with respect to $P _ { X } ( d x )$ gives, conditionally on the online history,

$$
\| \tilde { \mu } _ { t } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C D ( p _ { t } \| q _ { t } ) ,
$$

and thus taking outer expectation yields

$$
\begin{array} { r } { \mathbb { E } \left\| \tilde { \mu } _ { t } - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \mathbb { E } D ( p _ { t } \| q _ { t } ) . } \end{array}\tag{B.5}
$$

Combining (B.5) with (B.3) and (B.4), then taking infimum over all $s \in [ S ]$ , one can get the desired result. □

We now turn to the specific scenario considered in Algorithm 1, that is, the burn-in block in Algorithm B.1 are further separated into a burn-in block and a calibration block, and in the aggregation block, we have $\mu _ { s , t } ( x ) = \check { f } ^ { t } ( x ; \theta ^ { ( s ) } ) , \pi _ { s } \equiv 1 / S _ { \eta }$ , and $\hat { \sigma } _ { s , t } \equiv \hat { \sigma } _ { s }$ . Denote $\mathcal { F } _ { t }$ be the σ-field generated by $\{ Z _ { \ell } \} _ { \ell = 1 } ^ { t }$ , and the randomness in the black-box algorithm up to time t. We first state the following lemma that removes the scale estimation term $\gamma _ { s }$ in (B.2).

Lemma B.4 (Calibration-scale error for predictable candidate forecasts). Let $\{ \mu _ { s , t } : s \in$ [S], $t = n _ { 1 } , \ldots , n _ { 1 } + n _ { 2 } - 1 \}$ be a collection of predictable candidate forecasts such that $\mu _ { s , t }$ is measurable with respect to $\sigma ( Z _ { 1 } , \dots , Z _ { t } , U )$ as a random function and

$$
\| f _ { 0 } \| _ { \infty } \vee \operatorname* { s u p } _ { s \in [ S ] } \operatorname* { s u p } _ { t = n _ { 1 } , \ldots , n _ { 1 } + n _ { 2 } - 1 } \| \mu _ { s , t } \| _ { \infty } \leq B
$$

almost surely for some constant $B < \infty$ . Define

$$
\widetilde { \sigma } _ { s } ^ { 2 } = \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \left\{ Y _ { t + 1 } - \mu _ { s , t } ( X _ { t + 1 } ) \right\} ^ { 2 } ,
$$

and let $\hat { \sigma } _ { s } ^ { 2 } = \Pi _ { [ \underline { { { \sigma } } } ^ { 2 } , \bar { \sigma } ^ { 2 } ] } \left( \widetilde { \sigma } _ { s } ^ { 2 } \right)$ with $\hat { \sigma } _ { s } = ( \hat { \sigma } _ { s } ^ { 2 } ) ^ { 1 / 2 }$ . Suppose Assumption 3.7 holds. Then, for every $s \in [ S ]$

$$
\mathbb { E } \left( \hat { \sigma } _ { s } - \sigma \right) ^ { 2 } \leq C \left. \frac { 1 } { n _ { 2 } } + \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \mathbb { E } \left\| \mu _ { s , t } - f _ { 0 } \right\| _ { L _ { 2 } \left( P _ { X } \right) } ^ { 2 } \right. ,\tag{B.6}
$$

where C depends only on B, σ, σ¯, and the fourth moment of the standardized error distribution.

Proof of Lemma $B . 4 .$ Fix $s \in [ S ]$ and define $r _ { s , t } ( x ) = f _ { 0 } ( x ) - \mu _ { s , t } ( x ) , t = n _ { 1 } , \ldots , n _ { 1 } + n _ { 2 } - 1$ By predictability, ${ r } _ { s , t }$ is measurable with respect to $\sigma ( Z _ { 1 } , \dots , Z _ { t } , U )$ as a random function. Moreover, $Y _ { t + 1 } - \mu _ { s , t } ( X _ { t + 1 } ) = \varepsilon _ { t + 1 } + r _ { s , t } ( X _ { t + 1 } )$ . Hence

$$
\widetilde { \sigma } _ { s } ^ { 2 } - \sigma ^ { 2 } = \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \left( \varepsilon _ { t + 1 } ^ { 2 } - \sigma ^ { 2 } \right) + \frac { 2 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \varepsilon _ { t + 1 } r _ { s , t } ( X _ { t + 1 } ) + \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } r _ { s , t } ^ { 2 } ( X _ { t + 1 } )
$$

The finite fourth moment of the error gives

$$
\mathbb { E } I ^ { 2 } \leq { \frac { C } { n _ { 2 } } } .\tag{B.7}
$$

Since $\mathbb { E } ( \varepsilon _ { t + 1 } | Z _ { 1 } , \dots , Z _ { t } , U , X _ { t + 1 } ) = 0$ , the summands constituting II form a martingalediference sequence. Moreover, because $\varepsilon _ { t + 1 }$ is independent of $( Z _ { 1 } , \dots , Z _ { t } , U , X _ { t + 1 } ) , \mathbb { E } ( \varepsilon _ { t + 1 } ^ { 2 } ) =$ $\sigma ^ { 2 }$ . Then we have

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } I I ^ { 2 } = \frac { 4 } { n _ { 2 } ^ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \mathbb { E } \left[ \varepsilon _ { t + 1 } ^ { 2 } r _ { s , t } ^ { 2 } ( X _ { t + 1 } ) \right] } \\ & { \displaystyle \quad = \frac { 4 \sigma ^ { 2 } } { n _ { 2 } ^ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \mathbb { E } \left\| r _ { s , t } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } \\ & { \displaystyle \quad \leq \frac { C } { n _ { 2 } } \left\{ \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \mathbb { E } \left\| r _ { s , t } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \right\} . } \end{array}\tag{B.8}
$$

By Cauchy’s inequality, $\begin{array} { r } { I I I ^ { 2 } \le \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } r _ { s , t } ^ { 4 } ( X _ { t + 1 } ) } \end{array}$ . The uniform boundedness of $f _ { 0 }$ and $\mu _ { s , t }$ implies

$$
\mathbb { E } I I I ^ { 2 } \leq C \left\{ \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \mathbb { E } \left. r _ { s , t } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \right\} .\tag{B.9}
$$

Combining (B.7)–(B.9) gives

$$
\mathbb { E } \left( \widetilde { \sigma } _ { s } ^ { 2 } - \sigma ^ { 2 } \right) ^ { 2 } \leq C \left\{ \frac { 1 } { n _ { 2 } } + \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \mathbb { E } \left. \mu _ { s , t } - f _ { 0 } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \right\} .\tag{B.10}
$$

Since $\sigma ^ { 2 } \in [ \underline { { \sigma } } ^ { 2 } , \bar { \sigma } ^ { 2 } ]$ and projection onto a closed interval is nonexpansive, $| \hat { \sigma } _ { s } ^ { 2 } - \sigma ^ { 2 } | \le | \widetilde { \sigma } _ { s } ^ { 2 } - \sigma ^ { 2 } |$ Furthermore,

$$
\left| \hat { \sigma } _ { s } - \sigma \right| = \frac { \left| \hat { \sigma } _ { s } ^ { 2 } - \sigma ^ { 2 } \right| } { \hat { \sigma } _ { s } + \sigma } \leq \frac { 1 } { 2 \underline { { \sigma } } } \left| \hat { \sigma } _ { s } ^ { 2 } - \sigma ^ { 2 } \right| .
$$

The result follows.

Lemma B.5 (Evolving-to-population comparator reduction). Let Θ be a parameter set and let $g ( \cdot ; \theta ) = \left( g _ { 1 } ( \cdot ; \theta ) , \ldots , g _ { M _ { R } } ( \cdot ; \theta ) \right) ^ { \top } , \theta \in \Theta$ be any gating rule satisfying $g _ { m } ( x ; \theta ) \ge 0$ and $\begin{array} { r } { \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( x ; \theta ) = 1 } \end{array}$ for every $x \in \mathcal { X }$ and $\theta \in \Theta$ . Define

$$
\begin{array} { l } { { \displaystyle { \check { f } } _ { g } ^ { t } ( x ; \theta ) = \sum _ { \ell = 1 } ^ { M _ { S } } { \hat { f } } _ { S , \ell } ^ { t } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( x ; \theta ) { \hat { f } } _ { R , m } ^ { t } ( x ) } , } \\ { { \displaystyle f _ { g } ^ { * } ( x ; \theta ) = \sum _ { \ell = 1 } ^ { M _ { S } } f _ { S , \ell } ^ { * } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( x ; \theta ) f _ { R , m } ^ { * } ( x ) } . } \end{array}
$$

Under Assumption 3.5,

$$
\mathbb { E } \operatorname* { s u p } _ { \theta \in \Theta } \left. \check { f } _ { g } ^ { t } ( \cdot ; \theta ) - f _ { g } ^ { * } ( \cdot ; \theta ) \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left( M _ { S } ^ { 2 } + M _ { R } \right) a _ { t } ^ { 2 } .\tag{B.11}
$$

Consequently, for every $\theta \in \Theta$

$$
\begin{array} { r } { \mathbb { E } \left\| \check { f } _ { g } ^ { t } ( \cdot ; \theta ) - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq 2 \left\| f _ { g } ^ { * } ( \cdot ; \theta ) - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + C \left( M _ { S } ^ { 2 } + M _ { R } \right) a _ { t } ^ { 2 } . } \end{array}\tag{B.12}
$$

In particular, when $M _ { S }$ is fixed and $M _ { R } \geq 2$ , the last term is bounded by $C M _ { R } a _ { t } ^ { 2 }$

Proof. Write

$$
U _ { t } = \sum _ { \ell = 1 } ^ { M _ { S } } \left( \hat { f } _ { S , \ell } ^ { t } - f _ { S , \ell } ^ { * } \right) , \qquad V _ { t } ( \theta ) = \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( \cdot ; \theta ) \left( \hat { f } _ { R , m } ^ { t } - f _ { R , m } ^ { * } \right) .
$$

Then $\check { f } _ { g } ^ { t } ( \cdot ; \theta ) - f _ { g } ^ { * } ( \cdot ; \theta ) = U _ { t } + V _ { t } ( \theta )$ . For the shared component,

$$
\left. U _ { t } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le M _ { S } \sum _ { \ell = 1 } ^ { M _ { S } } \left. \hat { f } _ { S , \ell } ^ { t } - f _ { S , \ell } ^ { * } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

For the routed component, Jensen’s inequality gives, pointwise in $x ,$

$$
| V _ { t } ( x ; \theta ) | ^ { 2 } \leq \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( x ; \theta ) \left| \hat { f } _ { R , m } ^ { t } ( x ) - f _ { R , m } ^ { * } ( x ) \right| ^ { 2 } .
$$

Therefore,

$$
\operatorname* { s u p } _ { \theta \in \Theta } \| V _ { t } ( \theta ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq \sum _ { m = 1 } ^ { M _ { R } } \left\| \hat { f } _ { R , m } ^ { t } - f _ { R , m } ^ { * } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Assumption 3.5 now gives

$$
\mathbb { E } \left\| U _ { t } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C M _ { S } ^ { 2 } a _ { t } ^ { 2 } , \qquad \mathbb { E } \operatorname* { s u p } _ { \theta \in \Theta } \| V _ { t } ( \theta ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C M _ { R } a _ { t } ^ { 2 } .
$$

Moreover,

$$
\operatorname* { s u p } _ { \theta \in \Theta } \left\| U _ { t } + V _ { t } ( \theta ) \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq 2 \left\| U _ { t } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + 2 \operatorname* { s u p } _ { \theta \in \Theta } \left\| V _ { t } ( \theta ) \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Combining this inequality with the preceding bounds proves (B.11).

The second conclusion follows from

$$
\begin{array} { r } { \left\| \check { f } _ { g } ^ { t } ( \cdot ; \theta ) - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le 2 \left\| f _ { g } ^ { * } ( \cdot ; \theta ) - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + 2 \left\| \check { f } _ { g } ^ { t } ( \cdot ; \theta ) - f _ { g } ^ { * } ( \cdot ; \theta ) \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } . } \end{array}
$$

Here we complete the proof.

Proof of Theorem 3.8. We first verify that under our scenario with $\mu _ { s , t } ( x ) = \check { f } ^ { t } ( x ; \theta ^ { ( s ) } )$ , $\pi _ { s } ~ \equiv ~ 1 / S _ { \eta }$ , and $\hat { \sigma } _ { s , t } \equiv \hat { \sigma } _ { s }$ , Assumptions B.1 and B.2 required by the general AFTER algorithm hold. For any $\theta ^ { ( s ) } \in \Theta ( \eta )$

$$
\check { f } ^ { t } ( x ; \theta ^ { ( s ) } ) = \sum _ { \ell = 1 } ^ { M _ { S } } \hat { f } _ { S , \ell } ^ { t } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ^ { ( s ) } ) \hat { f } _ { R , m } ^ { t } ( x ) .
$$

Since $\begin{array} { r } { \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ^ { ( s ) } ) = 1 } \end{array}$ , Assumption 3.6 gives

$$
\| \ b { \check { f } } ^ { t } ( \cdot ; \theta ^ { ( s ) } ) \| _ { \infty } \leq \sum _ { \ell = 1 } ^ { M _ { S } } \| \ b { \hat { f } } _ { S , \ell } ^ { t } \| _ { \infty } + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ^ { ( s ) } ) \| \ b { \hat { f } } _ { R , m } ^ { t } \| _ { \infty } \leq \boldsymbol { A } ( M _ { S } + 1 ) .
$$

Thus the candidate forecast predictors are uniformly bounded. Moreover, since ˆσ $\hat { \sigma } _ { s }$ are projected onto $[ \underline { { \sigma } } , \bar { \sigma } ]$ , Assumption B.2 is satisfied automatically.

Thus, applying Proposition B.3 with burn-in time $n _ { 1 } + n _ { 2 }$ and aggregation legnth $n _ { 3 }$ with the uniform prior $\pi _ { s } = 1 / S _ { \eta }$ , and then using Lemma B.4 with $\mu _ { s , t } = \check { f } ^ { t } ( \cdot ; \theta ^ { ( s ) } )$ , we obtain

$$
\begin{array} { r } { \displaystyle \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \left\| f _ { 0 } - \widetilde { f } ^ { t } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \displaystyle \operatorname* { i n f } _ { 1 \leq s \leq S _ { \eta } } \left\{ \frac { 1 } { n _ { 3 } } \displaystyle \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \left\| f _ { 0 } - \widetilde { f } ^ { t } ( \cdot ; \theta ^ { ( s ) } ) \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \right. } \\ { \displaystyle \left. + \frac { 1 } { n _ { 2 } } \displaystyle \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \mathbb { E } \left\| f _ { 0 } - \widetilde { f } ^ { t } ( \cdot ; \theta ^ { ( s ) } ) \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \frac { 1 } { n _ { 2 } } + \displaystyle \frac { \log S _ { \eta } } { n _ { 3 } } \right\} . } \end{array}\tag{B.13}
$$

Define $\begin{array} { r } { f ^ { * } ( x ; \theta ) = \sum _ { \ell = 1 } ^ { M _ { S } } f _ { S , \ell } ^ { * } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { R , m } ^ { * } ( x ) } \end{array}$ , and let $\begin{array} { r } { \theta ^ { \circ } \in \arg \operatorname* { m i n } _ { \theta \in \Theta _ { M _ { R } } } \| f _ { 0 } - f ^ { * } ( \cdot ; \theta ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } . } \end{array}$ Choose $\theta ^ { ( s ^ { \circ } ) } \in \Theta ( \eta )$ such that $\left\| \theta ^ { ( s ^ { \circ } ) } - \theta ^ { \circ } \right\| _ { 2 } \leq \eta$ . By little abuse of notation, for $z \in \mathbb { R } ^ { M _ { R } }$ we denote $g ^ { \mathrm { s o f t } } ( z )$ as the softmax function on $z ,$ with $g _ { m } ^ { \mathrm { s o f t } } ( z ) = \exp \{ z _ { m } \} / \sum _ { j = 1 } ^ { M _ { R } } \exp \{ z _ { j } \}$ For $z , z ^ { \prime } \in \mathbb { R } ^ { M _ { R } }$ , the softmax map satisfies

$$
\left\| g ^ { \mathrm { s o f t } } ( z ) - g ^ { \mathrm { s o f t } } ( z ^ { \prime } ) \right\| _ { 1 } \leq \left\| z - z ^ { \prime } \right\| _ { \infty } .
$$

In fact, we have for the softmax map,

$$
\frac { \partial g _ { m } ^ { \mathrm { s o f t } } } { \partial z _ { j } } = g _ { m } ^ { \mathrm { s o f t } } ( \delta _ { m j } - g _ { j } ^ { \mathrm { s o f t } } ) ,
$$

and thus the Jacobian can be written as $J ( z ) = \mathrm { d i a g } ( g ^ { \mathrm { s o f t } } ) - g ^ { \mathrm { s o f t } } ( g ^ { \mathrm { s o f t } } ) ^ { \top }$ . We now compute the $( \infty , 1 )$ norm of the Jacobian. Recall that it can be computed as

$$
\| J ( z ) \| _ { \infty \to 1 } = \operatorname* { m a x } _ { \| x \| _ { \infty } \leq 1 } \| J ( z ) x \| _ { 1 } .
$$

For any $\| x \| _ { \infty } \leq 1$ , we have

$$
( J ( z ) x ) _ { j } = g _ { j } ^ { \mathrm { s o f t } } x _ { j } - g _ { j } ^ { \mathrm { s o f t } } \sum _ { j ^ { \prime } = 1 } ^ { M _ { R } } g _ { j ^ { \prime } } ^ { \mathrm { s o f t } } x _ { j ^ { \prime } } = g _ { j } ^ { \mathrm { s o f t } } ( x _ { j } - c ) ,
$$

where $\begin{array} { r } { c = \langle g ^ { \mathrm { s o f t } } , x \rangle = \sum _ { j ^ { \prime } = 1 } ^ { M _ { R } } g _ { j ^ { \prime } } ^ { \mathrm { s o f t } } x _ { j ^ { \prime } } } \end{array}$ , which can be understood as the expectation of x under probability distribution $g ^ { \mathrm { s o f t } }$ . Then, $\begin{array} { r } { \| J ( z ) x \| _ { 1 } = \sum _ { i = 1 } ^ { M _ { R } } g _ { j } ^ { \mathrm { s o f t } } | x _ { j } - c | } \end{array}$

Now, let $S ^ { + } = \{ j : x _ { j } \geq c \}$ and $S ^ { - } ~ = ~ \{ j ~ : ~ x _ { j } ~ < ~ c \} ~$ , and denote $\begin{array} { r } { p = \sum _ { j \in S ^ { + } } g _ { j } ^ { \mathrm { s o f t } } } \end{array}$ Moreover, define $\textstyle { \bar { x } } _ { + } = \sum _ { j \in S ^ { + } } g _ { j } ^ { \mathrm { s o f t } } x _ { j } / p$ be the conditional mean of $x _ { j }$ in the positive index set $S ^ { + }$ , and $\begin{array} { r } { \bar { x } _ { - } = \sum _ { j \in S ^ { - } } g _ { j } ^ { \mathrm { s o f t } } x _ { j } / ( 1 - p ) } \end{array}$ similarly. We then have

$$
\| J ( z ) x \| _ { 1 } = 2 \sum _ { i \in S ^ { + } } g _ { i } ^ { \mathrm { s o f t } } ( x _ { i } - c ) = 2 \left( p \bar { x } _ { + } - p ( p \bar { x } _ { + } + ( 1 - p ) \bar { x } _ { - } ) \right) = 2 p ( 1 - p ) ( \bar { x } _ { + } - \bar { x } _ { - } ) .
$$

Since $\| x \| _ { \infty } \leq 1$ , we have $\bar { x } _ { + } - \bar { x } _ { - } \leq 2$ , and thus

$$
\| J ( z ) x \| _ { 1 } \leq 4 p ( 1 - p ) \leq 1 \quad \Longrightarrow \quad \| J ( z ) \| _ { \infty \to 1 } \leq 1 .
$$

Finally, by mean-value theorem and Minkowski’s inequality, we have for any $z , z ^ { \prime } \in \mathbb { R } ^ { M _ { R } }$

$$
\begin{array} { r l r } { \left. { \| g ^ { \mathrm { s o f t } } ( z ) - g ^ { \mathrm { s o f t } } ( z ^ { \prime } ) \| _ { 1 } \leq \int _ { 0 } ^ { 1 } \| J ( t z + ( 1 - t ) z ^ { \prime } ) ( z - z ^ { \prime } ) \| _ { 1 } \mathrm { d } t } \right.} \\ & { } & \\ & { } & { \leq \int _ { 0 } ^ { 1 } \| J ( t z + ( 1 - t ) z ^ { \prime } ) \| _ { \infty  1 } \| z - z ^ { \prime } \| _ { \infty } \mathrm { d } t } \\ & { } & \\ & { } & { \leq \displaystyle \int _ { 0 } ^ { 1 } \| z - z ^ { \prime } \| _ { \infty } \mathrm { d } t = \| z - z ^ { \prime } \| _ { \infty } . } \end{array}
$$

Since $\| ( x ^ { \top } , 1 ) ^ { \top } \| _ { 2 } \leq C \sqrt { d }$ uniformly over $x \in \mathcal { X }$ , by Cauchy’s inequality, we have

$$
\operatorname* { m a x } _ { m \leq M _ { R } } \left| s _ { m } ^ { \mathrm { l i n } } ( x ; \theta _ { m } ^ { ( s ^ { \circ } ) } ) - s _ { m } ^ { \mathrm { l i n } } ( x ; \theta _ { m } ^ { \circ } ) \right| \leq C \sqrt { d } \eta .
$$

It follows from Assumption 3.6 that

$$
\begin{array} { r } { \left\| f ^ { * } ( \cdot ; \theta ^ { ( s ^ { \circ } ) } ) - f ^ { * } ( \cdot ; \theta ^ { \circ } ) \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C d \eta ^ { 2 } . } \end{array}
$$

Consequently,

$$
\left\| f ^ { * } ( \cdot ; \theta ^ { ( s ^ { \circ } ) } ) - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C \left\{ \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + d \eta ^ { 2 } \right\} .
$$

Applying Lemma B.5 with $g = g ^ { \mathrm { s o f t } }$ gives

$$
\begin{array} { r l } {  { \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \mathbb { E } \| f _ { 0 } - \check { f } ^ { t } ( \cdot ; \theta ^ { ( s ^ { \circ } ) } ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \| f _ { 0 } - \check { f } ^ { t } ( \cdot ; \theta ^ { ( s ^ { \circ } ) } ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } } \\ & { \leq C \{ \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + M _ { R } ( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } ) + d \eta ^ { 2 } \} . } \end{array}
$$

Substituting the comparator $\theta ^ { ( s ^ { \circ } ) }$ into (B.13) yields

$$
\frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \left\| f _ { 0 } - \widetilde { f } ^ { t } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \mathfrak { A } _ { \mathrm { s o t t } } ( M _ { S } , M _ { R } ) + M _ { R } \left( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } \right) + d \eta ^ { 2 } + \frac { 1 } { n _ { 2 } } + \frac { \log S _ { \eta } } { n _ { 3 } } \right\} .\tag{B.14}
$$

We next bound $S _ { \eta }$ . Under the normalization $\theta _ { M _ { R } } = 0$ , the efective parameter dimension is $D = ( M _ { R } - 1 ) ( d + 1 )$ . Assumption 3.4 implies that $\Theta _ { M _ { R } }$ is contained in a D-dimensional box of radius $C _ { 1 }$ log $M _ { R }$ . Hence a Euclidean η-net may be chosen such that

$$
\log S _ { \eta } \leq C M _ { R } d \left[ \log ( M _ { R } d ) + \log \left( \frac { 1 } { \eta } \right) \right] .
$$

Choosing $\begin{array} { r } { \eta \asymp \frac { \sqrt { M _ { R } } } { \sqrt { n _ { 3 } } } } \end{array}$ gives $\begin{array} { r } { d \eta ^ { 2 } \le { \frac { C d M _ { R } } { n _ { 3 } } } } \end{array}$ . Moreover,

$$
{ \frac { \log S _ { \eta } } { n _ { 3 } } } \leq C { \frac { M _ { R } d } { n _ { 3 } } } \log ( M _ { R } d n _ { 3 } ) .
$$

Since $n _ { 1 } \asymp n _ { 2 } \asymp n _ { 3 }$ $M _ { R } \ge 2$ , and $d \geq 1$ , we have

$$
\frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \left\| f _ { 0 } - \widetilde { f } ^ { t } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + M _ { R } \left( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } \right) + \frac { M _ { R } d } { n } \log ( M _ { R } d n ) \right\} .
$$

It remains to pass from the time-averaged $\tilde { f } _ { 0 }$ to the projected estimator $\hat { f } _ { 0 }$ in Step (f). By Jensen’s inequality,

$$
\mathbb { E } \| f _ { 0 } - \tilde { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \| f _ { 0 } - \tilde { f } ^ { t } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 }
$$

$$
\leq C \left[ \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + M _ { R } ( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } ) + \frac { M _ { R } d } { n } \log ( M _ { R } d n ) \right] .\tag{B.15}
$$

By definition,

$$
\hat { f } _ { 0 } = \underset { \theta ^ { ( s ) } \in \Theta ( \eta ) } { \arg \operatorname* { m i n } } \ \lVert \tilde { f } _ { 0 } - \bar { f } ^ { n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } ) \rVert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Therefore, for every $\theta ^ { ( s ) } \in \Theta ( \eta )$

$$
\| \tilde { f } _ { 0 } - \hat { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } \leq \| \tilde { f } _ { 0 } - \bar { f } ^ { n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } ) \| _ { L _ { 2 } ( P _ { X } ) } .
$$

Using the triangle inequality,

$$
\begin{array} { r } { \| f _ { 0 } - \hat { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq 2 \| f _ { 0 } - \tilde { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + 2 \| \tilde { f } _ { 0 } - \hat { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } . } \end{array}
$$

Hence, for every $\theta ^ { ( s ) } \in \Theta ( \eta )$

$$
\begin{array} { r } { \| f _ { 0 } - \hat { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq 6 \| f _ { 0 } - \tilde { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + 4 \| f _ { 0 } - \bar { f } ^ { n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } , } \end{array}
$$

where we use the inequality $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ . Recall that

$$
\bar { f } ^ { n _ { 3 } } ( x ; \theta ^ { ( s ) } ) = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \check { f } ^ { t } ( x ; \theta ^ { ( s ) } ) .
$$

Taking expectations and then the infimum over $\theta ^ { ( s ) } \in \Theta ( \eta )$ , Jensen’s inequality gives

$$
\mathbb { E } \left. f _ { 0 } - \hat { f } _ { 0 } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le 6 \mathbb { E } \left. f _ { 0 } - \widetilde f _ { 0 } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \frac { 4 } { n _ { 3 } } \operatorname* { i n f } _ { \theta ^ { ( s ) } \in \Theta ( \eta ) } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \left. f _ { 0 } - \check { f } ^ { t } ( \cdot ; \theta ^ { ( s ) } ) \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Choosing the grid comparator $\theta ^ { ( s ^ { \circ } ) }$ constructed above and applying the preceding comparator bound yield

$$
\mathbb { E } \left\| f _ { 0 } - \hat { f } _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + M _ { R } \left( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } \right) + \frac { M _ { R } d } { n } \log ( M _ { R } d n ) \right\} .
$$

This proves the desired oracle inequality.

## B.2 Proof of Example 3.9

Proof of Example 3.9. Let $I _ { m } = [ ( m - 1 ) / M _ { R } , m / M _ { R } ]$ be a partition into $M _ { R }$ regions, and $x _ { m } = ( 2 m - 1 ) / ( 2 M _ { R } )$ be the center of $I _ { m }$ . Denote $h = 1 / M _ { R }$ . For suficiently large $M _ { R }$ set $\tau = c _ { \Theta } / \lfloor \sqrt { \log M _ { R } } \rfloor$ , where $c _ { \Theta } > 0$ is chosen large enough so that the normalized softmax parameters constructed below belong to the logarithmic parameter box in Assumption 3.4. Finitely many smaller values of $M _ { R }$ can be absorbed into the constants. Let

$$
\theta _ { m } = \left( \frac { x _ { m } } { \tau ^ { 2 } } , - \frac { x _ { m } ^ { 2 } } { 2 \tau ^ { 2 } } \right) , \qquad s _ { m } ^ { \mathrm { l i n } } ( x ; \theta _ { m } ) = ( x , 1 ) ^ { \top } \theta _ { m } , \qquad g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) = \frac { \exp \{ s _ { m } ^ { \mathrm { l i n } } ( x ; \theta _ { m } ) \} } { \sum _ { j = 1 } ^ { M _ { R } } \exp \{ s _ { j } ^ { \mathrm { l i n } } ( x ; \theta _ { j } ) \} }
$$

for $m \in [ M _ { R } ]$ . The displayed parameters are unnormalized. By the shift invariance of the softmax logits, replacing $\theta _ { m }$ by $\theta _ { m } - \theta _ { M _ { R } }$ imposes the reference normalization without changing the softmax weights. Since the coordinates of the normalized parameters are of order $\tau ^ { - 2 } = O ( \log M _ { R } )$ , the constructed parameter belongs to $\Theta _ { M _ { R } }$ by the choice of $c _ { \Theta }$

We can rewrite the gating function as

$$
\begin{array} { l c r } { g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) = \displaystyle \frac { \exp \{ - ( x - x _ { m } ) ^ { 2 } / ( 2 \tau ^ { 2 } ) \} \exp \{ x ^ { 2 } / ( 2 \tau ^ { 2 } ) \} } { \sum _ { j = 1 } ^ { M _ { R } } \exp \{ - ( x - x _ { j } ) ^ { 2 } / ( 2 \tau ^ { 2 } ) \} \exp \{ x ^ { 2 } / ( 2 \tau ^ { 2 } ) \} } } \\ { \displaystyle = \frac { \exp \{ - ( x - x _ { m } ) ^ { 2 } / ( 2 \tau ^ { 2 } ) \} } { \sum _ { j = 1 } ^ { M _ { R } } \exp \{ - ( x - x _ { j } ) ^ { 2 } / ( 2 \tau ^ { 2 } ) \} } . } \end{array}
$$

We now prove that this gating mechanism yields the desired upper bounds.

We first consider the case where the shared expert is included. Consistent with the main text, set $M _ { S } = 1$ and take $f _ { S , 1 } ^ { * } ( x ) = A \sin ( 2 \pi L x )$ . The routed experts only need to approximate the residual $f _ { 0 } - f _ { S , 1 } ^ { * } = x ^ { 2 }$ . Take

$$
f _ { R , m } ^ { * } ( x ) = a _ { m } x + b _ { m } , \qquad a _ { m } = 2 x _ { m } = { \frac { 2 m - 1 } { M _ { R } } } , \qquad b _ { m } = - x _ { m } ^ { 2 } = - \left( { \frac { 2 m - 1 } { 2 M _ { R } } } \right) ^ { 2 } ,
$$

for $m \in [ M _ { R } ]$ . Since $X \sim \mathrm { U n i f } [ 0 , 1 ]$ , the approximation error of this construction satisfies

$$
\int _ { 0 } ^ { 1 } \left( x ^ { 2 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) ( a _ { m } x + b _ { m } ) \right) ^ { 2 } d x = \int _ { 0 } ^ { 1 } \left( \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) ( x - x _ { m } ) ^ { 2 } \right) ^ { 2 } d x .
$$

We next bound the local second moment of the softmax weights. Set $\displaystyle { u _ { m } = ( x - x _ { m } ) / h }$

and $\varpi = h ^ { 2 } / \tau ^ { 2 }$ . Then exp $\{ - ( x - x _ { m } ) ^ { 2 } / ( 2 \tau ^ { 2 } ) \} = \exp \{ - \varpi u _ { m } ^ { 2 } / 2 \}$

For $M _ { R }$ large enough, $\tau \geq 2 h$ . Uniformly over $x \in [ 0 , 1 ]$ , including boundary points, there are at least a constant multiple of $\tau / h$ grid centers satisfying $| x - x _ { m } | \leq \tau / 2$ . Hence

$$
\sum _ { m = 1 } ^ { M _ { R } } \exp \{ - \varpi u _ { m } ^ { 2 } / 2 \} \ge c \frac { \tau } { h } ,
$$

for some universal constant $c > 0$

Moreover, for each $k \geq 1$ , there are at most two indices m such that $k - 1 / 2 < | u _ { m } | \leq$ $k + 1 / 2$ . For those indices,

$$
u _ { m } ^ { 2 } \exp \{ - \varpi u _ { m } ^ { 2 } / 2 \} \leq \left( k + \frac 1 2 \right) ^ { 2 } \exp \{ - \varpi ( k - 1 / 2 ) ^ { 2 } / 2 \} .
$$

Therefore,

$$
\begin{array} { r l r } {  { \sum _ { m = 1 } ^ { M _ { R } } \exp \{ - \varpi u _ { m } ^ { 2 } / 2 \} u _ { m } ^ { 2 } \le \frac 1 4 + 2 \sum _ { k = 1 } ^ { \infty } \bigg ( k + \frac 1 2 \bigg ) ^ { 2 } \exp \{ - \varpi ( k - 1 / 2 ) ^ { 2 } / 2 \} } } \\ & { } & { \lesssim \int _ { 0 } ^ { \infty } u ^ { 2 } \exp \{ - \varpi u ^ { 2 } / 4 \} d u } \\ & { } & { \asymp \varpi ^ { - 3 / 2 } = \frac { \tau ^ { 3 } } { h ^ { 3 } } . } \end{array}
$$

Combining the denominator lower bound and the numerator bound gives

$$
\sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) ( x - x _ { m } ) ^ { 2 } = h ^ { 2 } \frac { \sum _ { m = 1 } ^ { M _ { R } } \exp \{ - \varpi u _ { m } ^ { 2 } / 2 \} u _ { m } ^ { 2 } } { \sum _ { m = 1 } ^ { M _ { R } } \exp \{ - \varpi u _ { m } ^ { 2 } / 2 \} } \lesssim h ^ { 2 } \frac { \tau ^ { 3 } / h ^ { 3 } } { \tau / h } = \tau ^ { 2 } .
$$

Thus,

$$
\left| x ^ { 2 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) ( a _ { m } x + b _ { m } ) \right| \lesssim \tau ^ { 2 } , \qquad x \in [ 0 , 1 ] .
$$

Consequently,

$$
\mathfrak { A } _ { \mathrm { s o f t } } ( 1 , M _ { R } ) = \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \left. f _ { 0 } - f _ { S , 1 } ^ { * } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { R , m } ^ { * } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = O ( \tau ^ { 4 } ) = O \left( \frac { 1 } { \log ^ { 2 } M _ { R } } \right) .
$$

We now consider the case without shared experts. Set $M _ { S } = 0 _ { : }$ so that the routed experts must approximate the full target function $f _ { 0 } ( x ) = A \sin ( 2 \pi L x ) + x ^ { 2 }$ . For each $m \in [ M _ { R } ]$ , define the local afine approximation at $x _ { m }$ by

$$
f _ { R , m } ^ { * } ( x ) = a _ { m } x + b _ { m } : = f _ { 0 } ( x _ { m } ) + f _ { 0 } ^ { \prime } ( x _ { m } ) ( x - x _ { m } ) ,
$$

where

$$
a _ { m } = f _ { 0 } ^ { \prime } ( x _ { m } ) = 2 \pi A L \cos ( 2 \pi L x _ { m } ) + 2 x _ { m } , \qquad b _ { m } = f _ { 0 } ( x _ { m } ) - a _ { m } x _ { m } .
$$

We assume, as in the example statement, that the routed expert class contains these local afine approximations. The constants below retain their dependence on the curvature of $f _ { 0 }$ By Taylor’s theorem, for every $x \in [ 0 , 1 ]$ and each $m \in [ M _ { R } ]$ , there exists $\xi _ { m , x }$ between x and $x _ { m }$ such that

$$
f _ { 0 } ( x ) - f _ { R , m } ^ { * } ( x ) = \frac { 1 } { 2 } f _ { 0 } ^ { \prime \prime } ( \xi _ { m , x } ) ( x - x _ { m } ) ^ { 2 } .
$$

Since

$$
f _ { 0 } ^ { \prime \prime } ( x ) = - ( 2 \pi L ) ^ { 2 } A \sin ( 2 \pi L x ) + 2 ,
$$

we have

$$
\operatorname* { s u p } _ { x \in [ 0 , 1 ] } | f _ { 0 } ^ { \prime \prime } ( x ) | \leq 4 \pi ^ { 2 } | A | L ^ { 2 } + 2 = : C _ { L } .
$$

Therefore,

$$
\begin{array} { r l r } & { } & { \Bigg | f _ { 0 } ( x ) - \displaystyle \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { R , m } ^ { * } ( x ) \Bigg | = \Bigg | \displaystyle \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) \left( f _ { 0 } ( x ) - f _ { R , m } ^ { * } ( x ) \right) \Bigg | } \\ & { } & { \leq \displaystyle \frac { C _ { L } } { 2 } \displaystyle \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) ( x - x _ { m } ) ^ { 2 } . } \end{array}
$$

Using the same softmax second-moment bound as above,

$$
\sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) ( x - x _ { m } ) ^ { 2 } \leq C \tau ^ { 2 } ,
$$

uniformly in $x \in [ 0 , 1 ]$ . Hence,

$$
\left| f _ { 0 } ( x ) - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { R , m } ^ { * } ( x ) \right| \leq C C _ { L } { \tau } ^ { 2 } .
$$

Integrating over $x \in [ 0 , 1 ]$ yields

$$
\left\| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { R , m } ^ { * } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C C _ { L } ^ { 2 } \tau ^ { 4 } .
$$

Since $C _ { L } ^ { 2 } = O ( L ^ { 4 } + 1 )$ , we conclude that

$$
\operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \left\| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { R , m } ^ { * } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = O \left( \frac { L ^ { 4 } + 1 } { \log ^ { 2 } M _ { R } } \right) .
$$

In particular, when L is large, this is of order $O ( L ^ { 4 } / \log ^ { 2 } M _ { R } )$ . This completes the proof.

## C Proof of Section 4

Lemma C.1 (Regularity from bounded relative derivative). Suppose that $\phi : \mathbb { R } \to ( 0 , \infty )$ is continuously diferentiable, strictly increasing, and satisfies

$$
B _ { \phi } : = \operatorname* { s u p } _ { t \in \mathbb { R } } \frac { \phi ^ { \prime } ( t ) } { \phi ( t ) } < \infty .
$$

Then $\phi$ is regular in the sense of Definition $\it 4 . 1$ with $l _ { \phi } = 0$ . More precisely, for every $J \subset [ M _ { R } ]$ with $| J | = K$ and every $z , z ^ { \prime } \in \mathbb { R } ^ { M _ { R } }$

$$
\left\| G _ { J } ^ { \phi } ( z ) - G _ { J } ^ { \phi } ( z ^ { \prime } ) \right\| _ { 2 } \leq \frac { B _ { \phi } } { 2 } \| z - z ^ { \prime } \| _ { 2 } .
$$

Proof. Fix $J \subset [ M _ { R } ]$ with $| J | = K$ and write

$$
p _ { m } ( z ) = G _ { J , m } ^ { \phi } ( z ) = \frac { \phi ( z _ { m } ) } { \sum _ { r \in J } \phi ( z _ { r } ) } , \qquad m \in J .
$$

Coordinates outside J are identically zero. For $m , j \in J ,$ , the quotient rule gives

$$
\cfrac { \partial p _ { m } ( z ) } { \partial z _ { j } } = p _ { m } ( z ) \{ \delta _ { m j } - p _ { j } ( z ) \} \frac { \phi ^ { \prime } ( z _ { j } ) } { \phi ( z _ { j } ) } .
$$

Hence the only nonzero Jacobian block is

$$
\nabla _ { J } G _ { J } ^ { \phi } ( z ) = \{ \mathrm { d i a g } ( p _ { J } ( z ) ) - p _ { J } ( z ) p _ { J } ( z ) ^ { \top } \} \mathrm { d i a g } \left( \frac { \phi ^ { \prime } ( z _ { j } ) } { \phi ( z _ { j } ) } \right) _ { j \in J } .
$$

For any $p \in \Delta ^ { K - 1 }$ , the symmetric matrix $A ( p ) = \mathrm { d i a g } ( p ) - p p ^ { \top }$ satisfies

$$
\| A ( p ) \| _ { \mathrm { o p } } \leq \operatorname* { m a x } _ { i } \sum _ { j = 1 } ^ { K } | A ( p ) _ { i j } | = \operatorname* { m a x } _ { i } 2 p _ { i } ( 1 - p _ { i } ) \leq { \frac { 1 } { 2 } } .
$$

By the definition of $B _ { \phi }$ , the diagonal factor has operator norm at most $B _ { \phi }$ . Therefore $\| \nabla G _ { J } ^ { \phi } ( z ) \| _ { \mathrm { o p } } \leq B _ { \phi } / 2$ uniformly in $z , J , K$ , and $M _ { R }$

For $z _ { t } = z ^ { \prime } + t ( z - z ^ { \prime } )$ , the fundamental theorem of calculus gives

$$
G _ { J } ^ { \phi } ( z ) - G _ { J } ^ { \phi } ( z ^ { \prime } ) = \int _ { 0 } ^ { 1 } \nabla G _ { J } ^ { \phi } ( z _ { t } ) ( z - z ^ { \prime } ) d t ,
$$

and hence

$$
\Big \| G _ { J } ^ { \phi } ( z ) - G _ { J } ^ { \phi } ( z ^ { \prime } ) \Big \| _ { 2 } \leq \frac { B _ { \phi } } { 2 } \| z - z ^ { \prime } \| _ { 2 } .
$$

This is Definition 4.1 with $L _ { \phi } = B _ { \phi } / 2$ and $l _ { \phi } = 0$

## C.1 Proof of Proposition 4.5 and related remarks

Proof of Proposition 4.5. Write $\delta : = \lVert \boldsymbol { \theta } - \boldsymbol { \theta } ^ { \prime } \rVert _ { 2 }$ and $\bar { \boldsymbol { x } } : = ( x ^ { \top } , 1 ) ^ { \top }$ . We prove the result for $\theta \in \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } ) , \theta ^ { \prime } \in \Theta _ { M _ { R } }$ , and $\delta \leq 1$ . By the standing covariate bound $\| X \| _ { 2 } \leq C _ { X } \sqrt { d } .$ , we may take $R _ { X } : = ( C _ { X } ^ { 2 } d + 1 ) ^ { 1 / 2 }$ , so that $\| { \bar { X } } \| _ { 2 } \leq R _ { X } \leq C { \sqrt { d } }$ almost surely. In what follows, C may depend on $K , L _ { \phi }$ , and $C _ { 4 }$ in Assumption 4.4, but not on d, $M _ { R } , r _ { M _ { R } } , \theta _ { \mathrm { { f } } }$ , or $\theta ^ { \prime }$ .

For each pair $( \theta , \theta ^ { \prime } )$ , decompose $\mathcal { X } = G _ { \theta , \theta ^ { \prime } } \cup B _ { \theta , \theta ^ { \prime } }$ , where

$$
G _ { \theta , \theta ^ { \prime } } = \{ x : \mathcal { T } _ { K } ( x ; \theta ) = \mathcal { T } _ { K } ( x ; \theta ^ { \prime } ) \} , \qquad B _ { \theta , \theta ^ { \prime } } = \mathcal { X } \setminus G _ { \theta , \theta ^ { \prime } } .
$$

Then

$$
\begin{array} { r l } & { \left. g ^ { ( K , \phi ) } ( \cdot ; \theta ) - g ^ { ( K , \phi ) } ( \cdot ; \theta ^ { \prime } ) \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = \mathbb { E } \left[ \left. g ^ { ( K , \phi ) } ( X ; \theta ) - g ^ { ( K , \phi ) } ( X ; \theta ^ { \prime } ) \right. _ { 2 } ^ { 2 } \mathbb { 1 } _ { \{ X \in G _ { \theta , \theta ^ { \prime } } \} } \right] } \\ & { \qquad + \mathbb { E } \left[ \left. g ^ { ( K , \phi ) } ( X ; \theta ) - g ^ { ( K , \phi ) } ( X ; \theta ^ { \prime } ) \right. _ { 2 } ^ { 2 } \mathbb { 1 } _ { \{ X \in B _ { \theta , \theta ^ { \prime } } \} } \right] . } \end{array}
$$

On $G _ { \theta , \theta ^ { \prime } }$ , the selected Top-K set is fixed. Hence, by the regularity of $\phi ,$

$$
\begin{array} { l } { \displaystyle \left\| g ^ { ( K , \phi ) } ( x ; \theta ) - g ^ { ( K , \phi ) } ( x ; \theta ^ { \prime } ) \right\| _ { 2 } \leq L _ { \phi } M _ { R } ^ { l _ { \phi } } \left( \displaystyle \sum _ { m \in \mathcal { T } _ { K } ( x ; \theta ) } | s _ { m } ( x ; \theta _ { m } ) - s _ { m } ( x ; \theta _ { m } ^ { \prime } ) | ^ { 2 } \right) ^ { 1 / 2 } } \\ { = L _ { \phi } M _ { R } ^ { l _ { \phi } } \left( \displaystyle \sum _ { m \in \mathcal { T } _ { K } ( x ; \theta ) } | \bar { x } ^ { \top } ( \theta _ { m } - \theta _ { m } ^ { \prime } ) | ^ { 2 } \right) ^ { 1 / 2 } } \\ { \leq L _ { \phi } R _ { X } M _ { R } ^ { l _ { \phi } } \delta . } \end{array}
$$

Since both gating vectors lie in the simplex, their squared Euclidean distance is at most 2. Therefore,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left. g ^ { ( K , \phi ) } ( X ; \theta ) - g ^ { ( K , \phi ) } ( X ; \theta ^ { \prime } ) \right. _ { 2 } ^ { 2 } \mathbb { 1 } _ { \left\{ X \in G _ { \theta , \theta ^ { \prime } } \right\} } \right] \leq \operatorname* { m i n } \left\{ L _ { \phi } ^ { 2 } R _ { X } ^ { 2 } M _ { R } ^ { 2 l _ { \phi } } \delta ^ { 2 } , 2 \right\} } \\ & { \qquad \leq C R _ { X } M _ { R } ^ { l _ { \phi } } \delta . } \end{array}
$$

Next consider $B _ { \theta , \theta ^ { \prime } }$ . If $x \in B _ { \theta , \theta ^ { \prime } }$ , then the two Top-K active set difers. Hence there exist indices $i \in \mathcal { T } _ { K } ( x ; \theta ) \setminus \mathcal { T } _ { K } ( x ; \theta ^ { \prime } )$ and $j \in \mathcal { T } _ { K } ( x ; \theta ^ { \prime } ) \setminus \mathcal { T } _ { K } ( x ; \theta )$ . By the definition of the Top-K set, these indices satisfy

$$
s _ { i } ( x ; \theta _ { i } ) \geq s _ { j } ( x ; \theta _ { j } ) , \qquad s _ { i } ( x ; \theta _ { i } ^ { \prime } ) \leq s _ { j } ( x ; \theta _ { j } ^ { \prime } ) .
$$

Thus

$$
\begin{array} { r } { | s _ { i } ( x ; \theta _ { i } ) - s _ { j } ( x ; \theta _ { j } ) | \leq | s _ { i } ( x ; \theta _ { i } ) - s _ { i } ( x ; \theta _ { i } ^ { \prime } ) | + | s _ { j } ( x ; \theta _ { j } ) - s _ { j } ( x ; \theta _ { j } ^ { \prime } ) | \leq C R _ { X } \delta . } \end{array}
$$

Since $s _ { i } ( \boldsymbol { x } ; \boldsymbol { \theta } _ { i } ) - s _ { j } ( \boldsymbol { x } ; \boldsymbol { \theta } _ { j } ) = \boldsymbol { x } ^ { \top } ( \beta _ { i } - \beta _ { j } ) + ( \alpha _ { i } - \alpha _ { j } )$ , we have

$$
B _ { \theta , \theta ^ { \prime } } \subset \bigcup _ { i \neq j } \left\{ x : \left| x ^ { \top } ( \beta _ { i } - \beta _ { j } ) + ( \alpha _ { i } - \alpha _ { j } ) \right| \leq C R _ { X } \delta \right\} .
$$

By the definition of $\widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } ) , \lVert \beta _ { i } - \beta _ { j } \rVert _ { 2 } \ge r _ { M _ { R } } , i \ne j$ . Applying Assumption 4.4 with

$$
a = \frac { \beta _ { i } - \beta _ { j } } { \| \beta _ { i } - \beta _ { j } \| _ { 2 } } , \qquad b = \frac { \alpha _ { i } - \alpha _ { j } } { \| \beta _ { i } - \beta _ { j } \| _ { 2 } } ,
$$

gives

$$
\mathbb { P } \left( \left| X ^ { \top } ( \beta _ { i } - \beta _ { j } ) + ( \alpha _ { i } - \alpha _ { j } ) \right| \leq C R _ { X } \delta \right) \leq C R _ { X } \frac { \delta } { r _ { M _ { R } } } .
$$

A union bound over at most $M _ { R } ^ { 2 }$ pairs yields

$$
\mathbb { P } ( X \in B _ { \theta , \theta ^ { \prime } } ) \leq C R _ { X } M _ { R } ^ { 2 } \frac { \delta } { r _ { M _ { R } } } .
$$

Since both gating vectors lie in the simplex,

$$
\begin{array} { r } { \left\| g ^ { ( K , \phi ) } ( X ; \theta ) - g ^ { ( K , \phi ) } ( X ; \theta ^ { \prime } ) \right\| _ { 2 } ^ { 2 } \leq 2 , } \end{array}
$$

and therefore

$$
\mathbb { E } \left[ \left. g ^ { ( K , \phi ) } ( X ; \theta ) - g ^ { ( K , \phi ) } ( X ; \theta ^ { \prime } ) \right. _ { 2 } ^ { 2 } \boldsymbol { \mathbb { 1 } } _ { \{ X \in B _ { \theta , \theta ^ { \prime } } \} } \right] \leq C R _ { X } M _ { R } ^ { 2 } \frac { \delta } { r _ { M _ { R } } } .
$$

Combining the two bounds gives

$$
\left\| g ^ { ( K , \phi ) } ( \cdot ; \theta ) - g ^ { ( K , \phi ) } ( \cdot ; \theta ^ { \prime } ) \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C R _ { X } M _ { R } ^ { l _ { \phi } } \delta + C R _ { X } M _ { R } ^ { 2 } \frac { \delta } { r _ { M _ { R } } } .
$$

Since $r _ { M _ { R } } \in ( 0 , 1 ) , M _ { R } \geq 2$ , and $l _ { \phi } \geq 0$ , both terms are bounded by a constant multiple of $R _ { X } M _ { R } ^ { 2 + 2 l _ { \phi } } \delta / r _ { M _ { R } }$ . Using $R _ { X } \leq C \sqrt { d } .$ , we obtain

$$
\left. g ^ { ( K , \phi ) } ( \cdot ; \theta ) - g ^ { ( K , \phi ) } ( \cdot ; \theta ^ { \prime } ) \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C R _ { X } M _ { R } ^ { 2 + 2 l _ { \phi } } \frac { \delta } { r _ { M _ { R } } } \leq C \sqrt { d } M _ { R } ^ { 2 + 2 l _ { \phi } } \frac { \delta } { r _ { M _ { R } } } .
$$

Taking square roots and recalling $\delta = \lVert \boldsymbol { \theta } - \boldsymbol { \theta } ^ { \prime } \rVert _ { 2 }$ gives

$$
\left\| g ^ { ( K , \phi ) } ( : ; \theta ) - g ^ { ( K , \phi ) } ( \cdot ; \theta ^ { \prime } ) \right\| _ { L _ { 2 } ( P _ { X } ) } \le C d ^ { 1 / 4 } M _ { R } ^ { 1 + l _ { \phi } } \left( \frac { \| \theta - \theta ^ { \prime } \| _ { 2 } } { r _ { M _ { R } } } \right) ^ { 1 / 2 } .
$$

This completes the proof.

Proof of remark 4.6. By Proposition 4.5, for any $\theta \in \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } )$ and $\theta ^ { \prime } \in \Theta _ { M _ { R } }$ satisfying $\| \theta - \theta ^ { \prime } \| _ { 2 } \leq 1$ ,

$$
\left\| g ^ { ( K , \phi ) } ( : ; \theta ) - g ^ { ( K , \phi ) } ( \cdot ; \theta ^ { \prime } ) \right\| _ { L _ { 2 } ( P _ { X } ) } \le C d ^ { 1 / 4 } M _ { R } ^ { 1 + l _ { \phi } } \left( \frac { \| \theta - \theta ^ { \prime } \| _ { 2 } } { r _ { M _ { R } } } \right) ^ { 1 / 2 } .
$$

Fix $0 < \varepsilon \le 1$ and set

$$
u _ { \varepsilon } : = \frac { c \varepsilon ^ { 2 } r _ { M _ { R } } } { \sqrt { d } M _ { R } ^ { 2 + 2 l _ { \phi } } } ,
$$

with $c > 0$ suficiently small. We may assume $u _ { \varepsilon } \leq 1$ after decreasing c.

Take a u<sub>ε</sub>-cover of the larger parameter space $\left( \Theta _ { M _ { R } } , \Vert \cdot \Vert _ { 2 } \right)$

$$
\Theta _ { M _ { R } } \subset \bigcup _ { j = 1 } ^ { N } B _ { 2 } ( \theta ^ { j } , u _ { \varepsilon } ) , \qquad \theta ^ { j } \in \Theta _ { M _ { R } } .
$$

For every $\theta \in \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } )$ , choose $j$ such that $\lVert \theta - \theta ^ { j } \rVert _ { 2 } \leq u _ { \varepsilon }$ . By the choice of $u _ { \varepsilon }$ and Proposition 4.5,

$$
\left\| g ^ { ( K , \phi ) } ( \cdot ; \theta ) - g ^ { ( K , \phi ) } ( \cdot ; \theta ^ { j } ) \right\| _ { L _ { 2 } ( P _ { X } ) } \leq C d ^ { 1 / 4 } M _ { R } ^ { 1 + l _ { \phi } } \left( \frac { u _ { \varepsilon } } { r _ { M _ { R } } } \right) ^ { 1 / 2 } \leq \varepsilon .
$$

Thus the ambient grid induces an ε-cover of $\mathcal { G } _ { K , \phi }$

It remains to bound N. Since the reference normalization fixes $\theta _ { M _ { R } } = 0$ , only $M _ { R } - 1$ parameter vectors vary. Since each $\theta _ { m } = ( \beta _ { m } ^ { \top } , \alpha _ { m } ) ^ { \top } \in \mathbb { R } ^ { d + 1 }$ , the efective dimension is $D = ( M _ { R } - 1 ) ( d + 1 )$ . By Assumption 3.4, $\Theta _ { M _ { R } }$ is contained in a D-dimensional cube of radius C<sup>′</sup> log $M _ { R }$ . Hence

$$
N \left( u _ { \varepsilon } , \Theta _ { M _ { R } } , \| \cdot \| _ { 2 } \right) \leq \left( \frac { C ^ { \prime } \log M _ { R } \sqrt { ( M _ { R } - 1 ) ( d + 1 ) } } { u _ { \varepsilon } } \right) ^ { ( M _ { R } - 1 ) ( d + 1 ) } .
$$

Substituting the value of $u _ { \varepsilon }$ gives

$$
N \left( \varepsilon , \mathcal { G } _ { K , \phi } , L _ { 2 } ( P _ { X } ) \right) \leq \left( \frac { C ^ { \prime } \log M _ { R } \sqrt { \left( M _ { R } - 1 \right) d ( d + 1 ) } M _ { R } ^ { 2 + 2 l _ { \phi } } } { \varepsilon ^ { 2 } r _ { M _ { R } } } \right) ^ { ( M _ { R } - 1 ) ( d + 1 ) } .
$$

Since $\sqrt { ( M _ { R } - 1 ) ( d + 1 ) } \leq C M _ { R } ^ { 1 / 2 } \sqrt { d } ,$ we obtain

$$
N \left( \varepsilon , \mathcal { G } _ { K , \phi } , L _ { 2 } ( P _ { X } ) \right) \leq \left( \frac { C ^ { \prime } M _ { R } ^ { 5 / 2 + 2 l _ { \phi } } d \log M _ { R } } { \varepsilon ^ { 2 } r _ { M _ { R } } } \right) ^ { ( M _ { R } - 1 ) ( d + 1 ) } .
$$

Here we complete the proof.

## C.2 Proof of Theorem 4.7

Proof of Theorem 4.7. Let $\Theta ( \eta ) = \{ \theta ^ { ( 1 ) } , \dots , \theta ^ { ( S _ { \eta } ) } \}$ be the Euclidean grid used by Algorithm 1. For $\theta \in \Theta _ { M _ { R } }$ define

$$
\begin{array} { r l } & { R _ { \mathrm { a g } } ^ { K , \phi } ( \theta ) = \displaystyle \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \left\| \check { f } ^ { K , \phi , t } ( \cdot ; \theta ) - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } , } \\ & { R _ { \mathrm { c a l } } ^ { K , \phi } ( \theta ) = \displaystyle \frac { 1 } { n _ { 2 } } \sum _ { t = n _ { 1 } } ^ { n _ { 1 } + n _ { 2 } - 1 } \mathbb { E } \left\| \check { f } ^ { K , \phi , t } ( \cdot ; \theta ) - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } . } \end{array}
$$

Applying Proposition B.3 with burn-in time $n _ { 1 }$ , calibration time $^ { n _ { 2 } , }$ aggregation time $n _ { 3 }$ and with

$$
\mu _ { s , t } ( \boldsymbol { x } ) = \check { f } ^ { K , \phi , t } ( \boldsymbol { x } ; \boldsymbol { \theta } ^ { ( s ) } ) , \qquad \pi _ { s } = S _ { \eta } ^ { - 1 } ,
$$

gives the corresponding oracle inequality. Applying Lemma B.4 to the same candidate forecasts then yields

$$
\frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \left\| \widetilde { f } ^ { K , \phi , t } - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \operatorname* { i n f } _ { 1 \leq s \leq S _ { \eta } } \left\{ R _ { \mathrm { a g } } ^ { K , \phi } ( \theta ^ { ( s ) } ) + R _ { \mathrm { c a l } } ^ { K , \phi } ( \theta ^ { ( s ) } ) + \frac { \log S _ { \eta } } { n _ { 3 } } + \frac { 1 } { n _ { 2 } } \right\} .\tag{C.1}
$$

The boundedness conditions required by Proposition B.3 hold because the Top-K gate lies in the simplex and the experts are uniformly bounded.

Define

$$
F ^ { * , K , \phi } ( x ; \theta ) = \sum _ { \ell = 1 } ^ { M _ { S } } f _ { S , \ell } ^ { * } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( K , \phi ) } ( x ; \theta ) f _ { R , m } ^ { * } ( x ) ,
$$

and let

$$
\theta ^ { \circ } \in \underset { \theta \in \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } ) } { \arg \operatorname* { m i n } } \left\| f _ { 0 } - F ^ { * , K , \phi } ( \cdot ; \theta ) \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Choose $\theta ^ { ( s ^ { \circ } ) } \in \Theta ( \eta )$ such that $\left\| \theta ^ { ( s ^ { \circ } ) } - \theta ^ { \circ } \right\| _ { 2 } \leq \eta$ . By Proposition 4.5,

$$
\left\| g ^ { ( K , \phi ) } ( \cdot ; \theta ^ { ( s ^ { 0 } ) } ) - g ^ { ( K , \phi ) } ( \cdot ; \theta ^ { 0 } ) \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C _ { K } \sqrt { d } M _ { R } ^ { 2 + 2 l _ { \phi } } \frac { \eta } { r _ { M _ { R } } } .
$$

For every x, the union of the supports of $g ^ { ( K , \phi ) } ( x ; \theta ^ { ( s ^ { \circ } ) } )$ and $g ^ { ( K , \phi ) } ( x ; \theta ^ { \circ } )$ contains at most 2K indices. Hence,

$$
\left. F ^ { \ast , K , \phi } ( x ; \theta ^ { ( s ^ { 0 } ) } ) - F ^ { \ast , K , \phi } ( x ; \theta ^ { \circ } ) \right. ^ { 2 } \leq 2 K A ^ { 2 } \left. g ^ { ( K , \phi ) } ( x ; \theta ^ { ( s ^ { 0 } ) } ) - g ^ { ( K , \phi ) } ( x ; \theta ^ { \circ } ) \right. _ { 2 } ^ { 2 } .
$$

Therefore,

$$
\Bigl \| F ^ { * , K , \phi } ( \cdot ; \theta ^ { ( s ^ { 0 } ) } ) - F ^ { * , K , \phi } ( \cdot ; \theta ^ { \circ } ) \Bigr \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C _ { K } \sqrt { d } M _ { R } ^ { 2 + 2 l _ { \phi } } \frac { \eta } { r _ { M _ { R } } } .
$$

It follows that

$$
\begin{array} { r } { \left\| F ^ { * , K , \phi } ( \cdot ; \theta ^ { ( s ^ { 0 } ) } ) - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C _ { K } \left\{ \mathfrak { A } _ { K , \phi } ( M _ { S } , M _ { R } ) + \sqrt { d } M _ { R } ^ { 2 + 2 l _ { \phi } } \frac { \eta } { r _ { M _ { R } } } \right\} . } \end{array}
$$

Applying Lemma B.5 with $g = g ^ { ( K , \phi ) }$ gives

$$
\begin{array} { r } { R _ { \mathrm { a g } } ^ { K , \phi } ( \theta ^ { ( s ^ { o } ) } ) + R _ { \mathrm { c a l } } ^ { K , \phi } ( \theta ^ { ( s ^ { o } ) } ) \leq C _ { K } \left\{ \mathfrak { A } _ { K , \phi } ( M _ { S } , M _ { R } ) + M _ { R } \left( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } \right) + \sqrt { d } M _ { R } ^ { 2 + 2 l _ { \phi } } \frac { \eta } { r _ { M _ { R } } } \right\} . } \end{array}
$$

Substituting this comparator into (C.1) yields

$$
\begin{array} { r } { \displaystyle \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \left\| \widetilde { f } ^ { K , \phi , t } - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C _ { K } \left\{ \mathfrak { A } _ { K , \phi } ( M _ { S } , M _ { R } ) + M _ { R } \left( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } \right) \right. } \\ { \displaystyle \left. + \sqrt { d } M _ { R } ^ { 2 + 2 l _ { \phi } } \frac { \eta } { r _ { M _ { R } } } + \frac { 1 } { n _ { 2 } } + \frac { \log S _ { \eta } } { n _ { 3 } } \right\} . } \end{array}\tag{C.2}
$$

We now choose the mesh size. Since $\Theta _ { M _ { R } }$ is contained in a box of radius $C _ { 1 }$ log $M _ { R }$ and has efective dimension $D = ( M _ { R } - 1 ) ( d + 1 )$ , a Euclidean grid with mesh η satisfies

$$
\log S _ { \eta } \leq C M _ { R } d \log \left( \frac { C \sqrt { M _ { R } d } \log M _ { R } } { \eta } \right) .
$$

With $\begin{array} { r } { \eta \asymp \frac { \sqrt { d } r _ { M _ { R } } } { n _ { 3 } M _ { R } ^ { 2 + 2 l _ { \phi } } } } \end{array}$ , truncated at a suficiently small absolute constant if necessary, the discretization term satisfies $\begin{array} { r } { \sqrt { d } M _ { R } ^ { 2 + 2 l _ { \phi } } \frac { \eta } { r _ { M _ { R } } } \leq C \frac { d } { n _ { 3 } } } \end{array}$ . Moreover,

$$
\frac { \log S _ { \eta } } { n _ { 3 } } \leq C \frac { M _ { R } d } { n _ { 3 } } \log \left( \frac { C M _ { R } ^ { 5 / 2 + 2 l _ { \phi } } n _ { 3 } \log M _ { R } } { r _ { M _ { R } } } \right) .
$$

Since $l _ { \phi }$ is fixed and $r _ { M _ { R } } \in ( 0 , 1 )$

$$
\frac { \log S _ { \eta } } { n _ { 3 } } \leq C \frac { M _ { R } d } { n _ { 3 } } \left\{ \log ( M _ { R } d n _ { 3 } ) + \log \left( \frac { 1 } { r _ { M _ { R } } } \right) \right\} .
$$

Since $n _ { 2 } \asymp n _ { 3 } \asymp n$ , both the discretization term and the calibration term $1 / n _ { 2 }$ are absorbed into

$$
C \frac { M _ { R } d } { n } \left\{ \log ( M _ { R } d n ) + \log \left( \frac { 1 } { r _ { M _ { R } } } \right) \right\} .
$$

Combining these bounds with the preceding online inequality proves (4.2).

For the online-to-batch average

$$
\tilde { f } _ { 0 } ^ { K , \phi } : = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \tilde { f } ^ { K , \phi , t } ,
$$

Jensen’s inequality gives

$$
\mathbb { E } \Vert \tilde { f } _ { 0 } ^ { K , \phi } - f _ { 0 } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq \frac { 1 } { n _ { 3 } } \sum _ { { t = n _ { 1 } + n _ { 2 } } } ^ { n - 1 } \mathbb { E } \Vert \tilde { f } ^ { K , \phi , t } - f _ { 0 } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } ,
$$

so the online-to-batch average satisfies the same bound. Finally, let

$$
\bar { f } ^ { K , \phi , n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } ) = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \check { f } ^ { K , \phi , t } ( \cdot ; \theta ^ { ( s ) } )
$$

be the time-averaged candidate used in the population projection step. By definition of the projection,

$$
\hat { f } _ { 0 } = \bar { f } ^ { K , \phi , n _ { 3 } } ( \cdot ; \hat { \theta } ) , \qquad \hat { \theta } \in \arg \operatorname* { m i n } _ { \theta ^ { ( s ) } \in \Theta ( \eta ) } \| \tilde { f } _ { 0 } ^ { K , \phi } - \bar { f } ^ { K , \phi , n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Thus, for every $\theta ^ { ( s ) } \in \Theta ( \eta )$

$$
\begin{array} { r } { \| \tilde { f } _ { 0 } ^ { K , \phi } - \hat { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } \leq \| \tilde { f } _ { 0 } ^ { K , \phi } - \bar { f } ^ { K , \phi , n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } ) \| _ { L _ { 2 } ( P _ { X } ) } . } \end{array}
$$

Using the triangle inequality and then $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ , we obtain, for every $\theta ^ { ( s ) }$

$$
\begin{array} { r l } & { \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq 2 \| \tilde { f } _ { 0 } ^ { K , \phi } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + 2 \| \tilde { f } _ { 0 } ^ { K , \phi } - \hat { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } \\ & { \qquad \leq 6 \| \tilde { f } _ { 0 } ^ { K , \phi } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + 4 \| \bar { f } ^ { K , \phi , n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } ) - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } . } \end{array}
$$

Taking expectations, then infimizing over the grid, gives

$$
\mathbb { E } \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq 6 \mathbb { E } \| \tilde { f } _ { 0 } ^ { K , \phi } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + 4 \operatorname* { i n f } _ { 1 \leq s \leq S _ { \eta } } \mathbb { E } \| \bar { f } ^ { K , \phi , n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } ) - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

For the second term, Jensen’s inequality over the aggregation block gives

$$
\mathbb { E } \| \bar { f } ^ { K , \phi , n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } ) - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \| \check { f } ^ { K , \phi , t } ( \cdot ; \theta ^ { ( s ) } ) - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Taking $s = s ^ { \circ }$ , where $\theta ^ { ( s ^ { \circ } ) }$ is the oracle grid comparator constructed above, and using the

preceding population-comparison and evolving-expert bounds, we obtain

$$
\operatorname* { i n f } _ { 1 \leq s \leq S _ { \eta } } \mathbb { E } \left\| \bar { f } ^ { K , \phi , n _ { 3 } } ( \cdot ; \theta ^ { ( s ) } ) - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C _ { K } \left\{ \mathfrak { A } _ { K , \phi } ( M _ { S } , M _ { R } ) + M _ { R } \bar { a } _ { n _ { 3 } } ^ { 2 } + \sqrt { d } M _ { R } ^ { 2 + 2 l _ { \phi } } \frac { \eta } { r _ { M _ { R } } } \right\} .
$$

Together with the online-to-batch bound, this proves that the projected estimator satisfies the same oracle rate. This completes the proof. □

## C.3 Proof of Corollary 4.9

Proof of Corollary 4.9. We apply Theorem 4.7 with $K = 1$ and $M _ { S } = 0$ . It remains to bound the Top-1 oracle approximation term

$$
\operatorname* { i n f } _ { \theta \in \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } ) } \left\| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( 1 , \phi ) } ( \cdot ; \theta ) f _ { m } ^ { * } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

For Top-1 routing, the selected expert has weight one, independently of $\phi .$ . We therefore only need to exhibit a separated linear score system that selects the region-wise specialist. By relabeling the routed experts, use experts $1 , \ldots , M _ { 0 } - 1 , M _ { R }$ as the specialists for regions $1 , \ldots , M _ { 0 }$ , respectively, with the normalization $\theta _ { M _ { R } } = 0$ . For a fixed small $\lambda > 0$ , set

$$
\theta _ { r } = ( \beta _ { r } ^ { \top } , \alpha _ { r } ) ^ { \top } = \lambda ( v _ { r } - v _ { M _ { 0 } } ) , \qquad r = 1 , \ldots , M _ { 0 } - 1 .
$$

Choose $\lambda \leq ( 1 / 2 ) \land ( c _ { 0 } / 4 )$ . Since max $_ { \cdot r \leq M _ { 0 } } \| v _ { r } \| _ { 2 } = 1$ , this gives

$$
\| \theta _ { r } \| _ { 2 } \leq 2 \lambda \leq 1 , \qquad \| \theta _ { r } \| _ { \infty } \leq 2 \lambda \leq c _ { 0 } / 2 , \qquad r = 1 , \ldots , M _ { 0 } - 1 .
$$

Thus the active parameters satisfy both the Euclidean normalization and the centered-box constraint. Moreover, their slopes are separated from each other and from the reference slope by at least $\lambda r _ { 0 }$

It remains only to fill the unused routed experts without changing the Top-1 assignment. Let $B _ { X } = 1 \vee \operatorname* { s u p } _ { x \in { \mathcal { X } } } \| x \| _ { 2 }$ , and define

$$
b = { \frac { 1 } { 2 } } { \big ( } c _ { 0 } \wedge 1 { \big ) } , \qquad \delta = { \frac { b } { 2 B _ { X } } } , \qquad \mathcal { B } _ { \mathrm { a c t } } = \{ \beta _ { 1 } , \dots , \beta _ { M _ { 0 } - 1 } , 0 \} .
$$

Set $a = c _ { 1 } \delta M _ { R } ^ { - 1 / d }$ , where $c _ { 1 } > 0$ will be chosen suficiently small. By the volumetric packing bound, there exists an a-separated set $\mathcal { P } \subset B _ { d } ( 0 , \delta )$ with

$$
| \mathcal { P } | \geq c _ { d } \left( \frac { \delta } { a } \right) ^ { d } = c _ { d } c _ { 1 } ^ { - d } M _ { R } ,
$$

where $c _ { d } > 0$ depends only on the dimension. Define

$$
\mathcal { P } _ { 0 } = \left\{ p \in \mathcal { P } : \operatorname* { m i n } _ { \beta \in \mathcal { B } _ { \mathrm { a c t } } } \| p - \beta \| _ { 2 } \geq a \right\} .
$$

For each fixed $\beta \in B _ { \mathrm { a c t } }$ , the balls $\{ B _ { d } ( p , a / 2 ) : p \in \mathcal { P } \cap B _ { d } ( \beta , a ) \}$ are disjoint and are contained in $B _ { d } ( \beta , 3 a / 2 )$ . Hence

$$
| \mathcal { P } \cap B _ { d } ( \beta , a ) | \leq 3 ^ { d } .
$$

Since $| B _ { \mathrm { a c t } } | = M _ { 0 } \leq M _ { R }$ , at most $3 ^ { d } M _ { R }$ points are removed from $\mathcal { P } .$ . Choose $c _ { 1 }$ small enough so that $c _ { d } c _ { 1 } ^ { - d } - 3 ^ { d } \geq 1$ . Then $| \mathcal { P } _ { 0 } | \geq M _ { R }$ , and we may choose distinct points $\beta _ { j } ^ { \mathrm { d u m } } \in \mathcal { P } _ { 0 }$ 2 $j = M _ { 0 } , \ldots , M _ { R } - 1$ . These dummy slopes satisfy

$$
\| \beta _ { j } ^ { \mathrm { d u m } } \| _ { 2 } \leq \delta , \qquad \| \beta _ { j } ^ { \mathrm { d u m } } - \beta _ { k } ^ { \mathrm { d u m } } \| _ { 2 } \geq a \quad ( j \neq k ) , \qquad \operatorname* { m i n } _ { \beta \in \mathcal { B } _ { \mathrm { a c t } } } \| \beta _ { j } ^ { \mathrm { d u m } } - \beta \| _ { 2 } \geq a .
$$

Set

$$
\theta _ { j } = \big ( \big ( \beta _ { j } ^ { \mathrm { d u m } } \big ) ^ { \top } , - b \big ) ^ { \top } .
$$

Then

$$
\| \theta _ { j } \| _ { 2 } ^ { 2 } \leq \delta ^ { 2 } + b ^ { 2 } \leq \frac { 5 } { 4 } b ^ { 2 } \leq 1 , \qquad \| \theta _ { j } \| _ { \infty } \leq c _ { 0 } ,
$$

so the dummy parameters also satisfy the required feasibility constraints. Moreover, for all $x \in \mathcal { X }$ 2

$$
s _ { j } ( x ; \theta _ { j } ) \leq B _ { X } \delta - b = - b / 2 < 0 = s _ { M _ { R } } ( x ) .
$$

Thus no dummy expert can beat the reference expert.

For the active specialists, Assumption 4.8 implies that, up to a $P _ { X ^ { - } \mathrm { n u l l } }$ boundary set, subtracting $\bar { x } ^ { \top } v _ { M _ { 0 } }$ and multiplying by λ preserves all region-wise score comparisons. Hence the constructed parameter realizes

$$
\begin{array} { r } { \mathcal { T } _ { 1 } ( x ; \theta ) = \left\{ \begin{array} { l l } { \{ r \} , } & { x \in \mathcal { X } _ { r } , \quad r = 1 , \ldots , M _ { 0 } - 1 , } \\ { \{ M _ { R } \} , } & { x \in \mathcal { X } _ { M _ { 0 } } . } \end{array} \right. } \end{array}
$$

Moreover, the slope construction gives membership in $\widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } )$ for every $r _ { M _ { R } } \ \le \ c ( r _ { 0 } \wedge$ $M _ { R } ^ { - 1 / d } )$ , after reducing the constant $c > 0$ if necessary.

Consequently,

$$
\begin{array} { r l } {  { \operatorname* { i n f } _ { \theta ^ { \prime } \in \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } ) } \| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( 1 , \phi ) } ( \cdot ; \theta ^ { \prime } ) f _ { m } ^ { * } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le \| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( 1 , \phi ) } ( \cdot ; \theta ) f _ { m } ^ { * } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } \quad } & { } \\ & { \qquad = \sum _ { r = 1 } ^ { M _ { 0 } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } . } \end{array}
$$

Applying Theorem 4.7 gives

$$
\mathbb { E } \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \sum _ { r = 1 } ^ { M _ { 0 } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + \mathfrak { L } _ { \mathrm { s p a r s e } } ( M _ { R } , n , r _ { M _ { R } } ) \right\} .
$$

The same bound holds for the online average by Jensen’s inequality applied to the online sequence bound in Theorem 4.7. This proves the result. □

## C.4 Proof of Proposition 4.11

Proof of Proposition $4 . 1 1 .$ The oracle lower bound only uses the measurable partition fixed in the main text and the mixture advantage in Assumption 4.10. For any $\theta \in \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } )$ the Top-1 gating mechanism assigns each $x \in \mathcal { X }$ to exactly one expert. Denote by $\mathcal { X } _ { m } ^ { \prime } ( \theta )$ the region consisting of inputs assigned to $f _ { m } ^ { * }$ . Then $\{ \mathcal { X } _ { m } ^ { \prime } ( \theta ) \} _ { m \in [ M _ { R } ] }$ is a partition of $\mathcal { X }$ and, because the active set has size one,

$$
\sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( 1 , \phi ) } ( x ; \theta ) f _ { m } ^ { * } ( x ) = \sum _ { m = 1 } ^ { M _ { R } } \mathbb { 1 } _ { \mathcal { X } _ { m } ^ { \prime } ( \theta ) } ( x ) f _ { m } ^ { * } ( x ) .
$$

Therefore, for every $\theta \in \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } )$

$$
\begin{array} { r l } { \displaystyle \left\| { f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { _ m } ^ { ( 1 , \phi ) } ( \cdot ; \theta ) f _ { _ m } ^ { * } } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = } & { \displaystyle \sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \chi _ { _ r } } \left( f _ { 0 } ( x ) - \sum _ { m = 1 } ^ { M _ { R } } \mathbb { 1 } _ { \chi _ { _ m } ^ { \prime } ( \theta ) } ( x ) f _ { _ m } ^ { * } ( x ) \right) ^ { 2 } d P _ { X } ( x ) } \\ & { \geq \displaystyle \sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \chi _ { _ r } } \operatorname* { m i n } _ { k \in [ M _ { R } ] } ( f _ { 0 } ( x ) - f _ { k } ^ { * } ( x ) ) ^ { 2 } d P _ { X } ( x ) } \\ & { \geq \displaystyle \sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \chi _ { _ r } } \left( f _ { 0 } ( x ) - w _ { r 1 } f _ { r , 1 } ^ { * } ( x ) - w _ { r 2 } f _ { r , 2 } ^ { * } ( x ) \right) ^ { 2 } d P _ { X } ( x ) + \displaystyle \sum _ { r = 1 } ^ { M _ { 0 } } \delta _ { r } . } \end{array}
$$

The last inequality follows from Assumption 4.10. Hence

$$
\left\| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( 1 , \phi ) } ( \cdot ; \theta ) f _ { m } ^ { * } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \geq \sum _ { r = 1 } ^ { M _ { 0 } } \| f _ { 0 } - w _ { r 1 } f _ { r , 1 } ^ { * } - w _ { r 2 } f _ { r , 2 } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + \sum _ { r = 1 } ^ { M _ { 0 } } \delta _ { r } .
$$

Taking the infimum over $\theta \in \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } )$ gives

$$
{ \mathfrak { A } } _ { 1 } ( M _ { R } ) \geq \sum _ { r = 1 } ^ { M _ { 0 } } \| f _ { 0 } - w _ { r 1 } f _ { r , 1 } ^ { * } - w _ { r 2 } f _ { r , 2 } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + \sum _ { r = 1 } ^ { M _ { 0 } } \delta _ { r } .
$$

We next prove the second claim. By the projection step of Algorithm 1, the final Top-1 estimator has the form

$$
\hat { f } _ { 0 } ( x ) = \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( 1 , \phi ) } ( x ; \hat { \theta } ) \bar { f } _ { R , m } ^ { n _ { 3 } } ( x ) ,
$$

for a random grid parameter $\hat { \theta } \in \widetilde { \Theta } _ { M _ { R } } ( r _ { M _ { R } } )$ , where the projected Top-1 class is the same separated class used in the definition of $\mathfrak { A } _ { 1 } ( M _ { R } )$ . Here $\bar { f } _ { R , m } ^ { n _ { 3 } }$ denotes the aggregation-block average of the m-th routed expert. Define the population-expert counterpart

$$
F ^ { * } ( x ; \hat { \theta } ) = \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( 1 , \phi ) } ( x ; \hat { \theta } ) f _ { m } ^ { * } ( x ) .
$$

For every realization of $\hat { \theta } .$

$$
\| F ^ { * } ( \cdot ; \hat { \theta } ) - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \geq \mathfrak { A } _ { 1 } ( M _ { R } ) .
$$

Moreover, for every realization of the data and every $x \in \mathcal { X }$

$$
\hat { f } _ { 0 } ( x ) - F ^ { * } ( x ; \hat { \theta } ) = \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( 1 , \phi ) } ( x ; \hat { \theta } ) \{ \bar { f } _ { R , m } ^ { n _ { 3 } } ( x ) - f _ { m } ^ { * } ( x ) \} .
$$

Since $g ^ { ( 1 , \phi ) } ( x ; \hat { \theta } ) \in \Delta ^ { M _ { R } - 1 }$ , Jensen’s inequality with respect to the simplex weights gives

$$
| \hat { f } _ { 0 } ( x ) - F ^ { * } ( x ; \hat { \theta } ) | ^ { 2 } \leq \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { ( 1 , \phi ) } ( x ; \hat { \theta } ) | \bar { f } _ { R , m } ^ { n _ { 3 } } ( x ) - f _ { m } ^ { * } ( x ) | ^ { 2 } .
$$

Integrating with respect to $P _ { X }$ and using $0 \leq g _ { m } ^ { ( 1 , \phi ) } \leq 1$ yields

$$
\begin{array} { r l } {  { \mathbb { E } \| \hat { f } _ { 0 } - F ^ { * } ( \cdot ; \hat { \theta } ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le \displaystyle \sum _ { m = 1 } ^ { M _ { R } } \mathbb { E } \| \bar { f } _ { R , m } ^ { n _ { 3 } } - f _ { m } ^ { * } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } } \\ & { \le M _ { R } \operatorname* { m a x } _ { m \le M _ { R } } \mathbb { E } \| \bar { f } _ { R , m } ^ { n _ { 3 } } - f _ { m } ^ { * } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } \\ & { \le C M _ { R } \bar { a } _ { n _ { 3 } } ^ { 2 } . } \end{array}
$$

The last inequality follows from Jensen’s inequality over the aggregation block and Assumption 3.5:

$$
\mathbb { E } \Vert \bar { f } _ { R , m } ^ { n _ { 3 } } - f _ { m } ^ { * } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \Vert \hat { f } _ { R , m } ^ { t } - f _ { m } ^ { * } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \bar { a } _ { n _ { 3 } } ^ { 2 } .
$$

The reverse triangle inequality in $L _ { 2 } ( \Omega \times P _ { X } )$ gives

$$
\Big ( \mathbb { E } \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \Big ) ^ { 1 / 2 } \geq \sqrt { \mathfrak { A } _ { 1 } ( M _ { R } ) } - C \sqrt { M _ { R } } \bar { a } _ { n _ { 3 } } .
$$

Squaring the positive part proves the claimed lower bound. This proves the claim. □

## D Proof of Section 5

## D.1 Proof of the oracle decomposition for gating approximation

Proof. Fix any $g \in { \mathcal { G } }$ . For $x \in \mathcal { X } _ { r }$ , we have $g ^ { * } ( x ) = e _ { r }$ and hence

$$
f _ { 0 } ( x ) - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ( x ) f _ { m } ^ { * } ( x ) = \left( f _ { 0 } ( x ) - f _ { r } ^ { * } ( x ) \right) + \sum _ { m = 1 } ^ { M _ { R } } \bigl ( g _ { m } ^ { * } ( x ) - g _ { m } ( x ) \bigr ) f _ { m } ^ { * } ( x ) .
$$

Because $g ( x ) \in \Delta ^ { M _ { R } - 1 }$

$$
\lVert g ( x ) - g ^ { * } ( x ) \rVert _ { 1 } = 2 \{ 1 - g _ { r } ( x ) \} \leq 2 \lVert g ( x ) - g ^ { * } ( x ) \rVert _ { 2 } .
$$

Therefore, Assumption 3.6 gives

$$
\left| \sum _ { m = 1 } ^ { M _ { R } } \bigl ( g _ { m } ^ { * } ( x ) - g _ { m } ( x ) \bigr ) f _ { m } ^ { * } ( x ) \right| \leq A \| g ( x ) - g ^ { * } ( x ) \| _ { 1 } \leq 2 A \| g ( x ) - g ^ { * } ( x ) \| _ { 2 } .
$$

Applying $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ , integrating over each $\mathcal { X } _ { r }$ , and using that $\{ \mathcal { X } _ { r } \} _ { r = 1 } ^ { M _ { R } }$ partitions $\mathcal { X }$ , we obtain

$$
\begin{array} { r l r } {  { \| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } f _ { m } ^ { * } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le 2 \sum _ { r = 1 } ^ { M _ { R } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + 8 A ^ { 2 } \| g - g ^ { * } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } } \\ & { } & { \le C \{ \displaystyle \sum _ { r = 1 } ^ { M _ { R } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + \| g - g ^ { * } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \} , } \end{array}
$$

where $C = \operatorname* { m a x } \{ 2 , 8 A ^ { 2 } \}$ is independent of n and $M _ { R }$ . Taking the infimum over $g \in { \mathcal { G } }$ proves the decomposition stated in the main text. □

## D.2 Proof of Example 5.2

Proof of Example 5.2. Recall that $\mathcal { X } _ { 1 } = [ - 1 , 0 ] , \mathcal { X } _ { 2 } = ( 0 , 1 ]$ and $X \sim \mathrm { U n i f } [ - 1 , 1 ]$ . The oracle assignment is $e _ { 1 }$ on $\mathcal { X } _ { 1 }$ and $e _ { 2 }$ on $\mathcal { X } _ { 2 }$ . Thus for any gating rule $g ( x ) = ( g _ { 1 } ( x ) , \ldots , g _ { M _ { R } } ( x ) ) ^ { \top }$

the global and interior losses are

$$
\begin{array} { l } { \displaystyle { \mathcal { L } ^ { ( 1 ) } ( g ) = \frac { 1 } { 2 } \int _ { - 1 } ^ { 0 } \left( ( 1 - g _ { 1 } ( x ) ) ^ { 2 } + \sum _ { j \ne 1 } g _ { j } ( x ) ^ { 2 } \right) d x + \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \left( ( 1 - g _ { 2 } ( x ) ) ^ { 2 } + \sum _ { j \ne 2 } g _ { j } ( x ) ^ { 2 } \right) d x , } } \\ { \displaystyle { \mathcal { L } ^ { ( 2 ) } ( g ; \rho ) = \frac { 1 } { 2 } \int _ { - 1 } ^ { - \rho } \left( ( 1 - g _ { 1 } ( x ) ) ^ { 2 } + \sum _ { j \ne 1 } g _ { j } ( x ) ^ { 2 } \right) d x + \frac { 1 } { 2 } \int _ { \rho } ^ { 1 } \left( ( 1 - g _ { 2 } ( x ) ) ^ { 2 } + \sum _ { j \ne 2 } g _ { j } ( x ) ^ { 2 } \right) d x . } } \end{array}
$$

Now we prove the lower bound for $\mathcal { L } ^ { ( 1 ) }$ under softmax gating. Let

$$
s _ { r } ^ { \mathrm { l i n } } ( x ; \theta _ { r } ) = \beta _ { r } x + \alpha _ { r } , \qquad g _ { r } ( x ) = \frac { \exp \{ s _ { r } ^ { \mathrm { l i n } } ( x ; \theta _ { r } ) \} } { \sum _ { j = 1 } ^ { M _ { R } } \exp \{ s _ { j } ^ { \mathrm { l i n } } ( x ; \theta _ { j } ) \} } .
$$

Since

$$
1 - g _ { 1 } ( x ) \geq \frac { \exp \{ s _ { 2 } ^ { \mathrm { l i n } } ( x ; \theta _ { 2 } ) \} } { \exp \{ s _ { 1 } ^ { \mathrm { l i n } } ( x ; \theta _ { 1 } ) \} + \exp \{ s _ { 2 } ^ { \mathrm { l i n } } ( x ; \theta _ { 2 } ) \} } , \qquad 1 - g _ { 2 } ( x ) \geq \frac { \exp \{ s _ { 1 } ^ { \mathrm { l i n } } ( x ; \theta _ { 1 } ) \} } { \exp \{ s _ { 1 } ^ { \mathrm { l i n } } ( x ; \theta _ { 1 } ) \} + \exp \{ s _ { 2 } ^ { \mathrm { l i n } } ( x ; \theta _ { 2 } ) \} } ,
$$

we have

$$
\mathcal { L } ^ { ( 1 ) } ( g ) \geq \frac { 1 } { 2 } \int _ { - 1 } ^ { 0 } \Bigg ( \frac { 1 } { 1 + \exp \{ s _ { 1 } ^ { \mathrm { i n } } ( x ; \theta _ { 1 } ) - s _ { 2 } ^ { \mathrm { i n } } ( x ; \theta _ { 2 } ) \} } \Bigg ) ^ { 2 } d x + \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \bigg ( \frac { 1 } { 1 + \exp \{ s _ { 2 } ^ { \mathrm { i n } } ( x ; \theta _ { 2 } ) - s _ { 1 } ^ { \mathrm { i n } } ( x ; \theta _ { 1 } ) \} } \bigg ) ^ { 2 } d x .
$$

$$
\mathrm { S e t } ~ \Delta ( x ) : = s _ { 2 } ^ { \mathrm { l i n } } ( x ; \theta _ { 2 } ) - s _ { 1 } ^ { \mathrm { l i n } } ( x ; \theta _ { 1 } ) = ( \beta _ { 2 } - \beta _ { 1 } ) x + ( \alpha _ { 2 } - \alpha _ { 1 } ) . \mathrm { ~ B y ~ t h e ~ p a r a m e t e r ~ b o u n d } ,
$$

$$
| \beta _ { 2 } - \beta _ { 1 } | \le 2 C \log M _ { R } , \qquad | \alpha _ { 1 } - \alpha _ { 2 } | \le 2 C \log M _ { R } .
$$

Let $L : = 2 C$ log $M _ { R }$ and $k : = s _ { 1 } ^ { \mathrm { l i n } } ( 0 ; \theta _ { 1 } ) - s _ { 2 } ^ { \mathrm { l i n } } ( 0 ; \theta _ { 2 } )$ . We first consider the case $k \geq 0$ . Then $0 \le k \le L$ , and for $x \in [ 0 , 1 ]$ ，

$$
\Delta ( x ) = - k + ( \beta _ { 2 } - \beta _ { 1 } ) x \leq - k + L x .
$$

Since $t \mapsto ( 1 + \exp \{ t \} ) ^ { - 2 }$ is decreasing,

$$
\mathcal { L } ^ { ( 1 ) } ( g ) \geq \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \left( \frac { 1 } { 1 + \exp \{ \Delta ( x ) \} } \right) ^ { 2 } d x \geq \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \left( \frac { 1 } { 1 + \exp \{ - k + L x \} } \right) ^ { 2 } d x .
$$

Now split the integral at $x = k / L$ , we obtain

$$
\begin{array} { l } { \displaystyle \mathcal { L } ^ { ( 1 ) } ( g ) \geq \frac { 1 } { 2 } \int _ { 0 } ^ { k / L } \left( \frac { 1 } { 1 + \exp \{ - k + L x \} } \right) ^ { 2 } d x + \frac { 1 } { 2 } \int _ { k / L } ^ { 1 } \left( \frac { 1 } { 1 + \exp \{ - k + L x \} } \right) ^ { 2 } d x } \\ { \displaystyle \qquad \geq \frac { 1 } { 2 } \cdot \frac { k } { 4 L } + \frac { 1 } { 2 } \int _ { k / L } ^ { 1 } \frac { 1 } { 4 \exp \{ - 2 k + 2 L x \} } d x } \\ { \displaystyle \qquad = \frac { k } { 8 L } + \frac { 1 } { 1 6 L } \left( 1 - \exp \{ 2 k - 2 L \} \right) } \\ { \displaystyle \qquad = \frac { 1 } { 1 6 L } \left( 2 k + 1 - \exp \{ 2 k - 2 L \} \right) . } \end{array}
$$

Since $0 \le k \le L$ , the last display implies $\begin{array} { r } { \mathcal { L } ^ { ( 1 ) } ( g ) \ge \frac { 1 - \exp \{ - 2 L \} } { 1 6 L } } \end{array}$ . If $k < 0 ,$ , the same argument is applied on the interval [−1, 0], with the two sided reversed. This gives the same lower bound. Therefore, for every softmax gate g,

$$
\mathcal { L } ^ { ( 1 ) } ( g ) \geq \frac { 1 - \exp \{ - 2 L \} } { 1 6 L } .
$$

Since $L = 2 C$ log $M _ { R }$

$$
\mathcal { L } ^ { ( 1 ) } ( g ) \geq \frac { 1 - M _ { R } ^ { - 4 C } } { 3 2 C \log M _ { R } } \gtrsim \frac { 1 } { \log M _ { R } } .
$$

As $g \in \mathcal { G } _ { \mathrm { s o f t } }$ was arbitrary,

$$
\mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \mathrm { s o f t } } ) \gtrsim \frac { 1 } { \log M _ { R } } .
$$

Next we prove the upper bound for $\mathcal { L } ^ { ( 1 ) }$ under softmax gating. Choose

$$
\begin{array} { r } { s _ { 1 } ^ { \mathrm { l i n } } ( x ; \theta _ { 1 } ) = B - a x , \quad \quad s _ { 2 } ^ { \mathrm { l i n } } ( x ; \theta _ { 2 } ) = B + a x , \quad \quad s _ { j } ^ { \mathrm { l i n } } ( x ; \theta _ { j } ) = - B , \quad \quad j \geq 3 , } \end{array}
$$

with $a = B = C \log M _ { R }$ . Set $s : = ( M _ { R } - 2 ) \exp \{ - 2 B \} = ( M _ { R } - 2 ) M _ { R } ^ { - 2 C }$ . Since $C > 1 / 2$ we have $0 \leq s \leq 1$ for all suficiently large $M _ { R } ;$ the finitely many smaller values of $M _ { R }$ are absorbed into the constant. For $x \in [ 0 , 1 ]$ 2

$$
1 - g _ { 2 } ( x ) = \frac { \exp \{ - a x \} + s } { \exp \{ - a x \} + \exp \{ a x \} + s } \leq \exp \{ - 2 a x \} + s \exp \{ - a x \} ,
$$

Similarly, for $x \in [ - 1 , 0 ]$ 2

$$
1 - g _ { 1 } ( x ) \leq \exp \{ - 2 a | x | \} + s \exp \{ - a | x | \} .
$$

Also,

$$
( 1 - g _ { 1 } ( x ) ) ^ { 2 } + \sum _ { j \neq 1 } g _ { j } ( x ) ^ { 2 } \leq 2 ( 1 - g _ { 1 } ( x ) ) ^ { 2 } , \qquad ( 1 - g _ { 2 } ( x ) ) ^ { 2 } + \sum _ { j \neq 2 } g _ { j } ( x ) ^ { 2 } \leq 2 ( 1 - g _ { 2 } ( x ) ) ^ { 2 } .
$$

Hence, by symmetry,

$$
\mathcal { L } ^ { ( 1 ) } ( g ) \leq 2 \int _ { 0 } ^ { 1 } \left( \exp \{ - 2 a x \} + s \exp \{ - a x \} \right) ^ { 2 } d x .
$$

Using $( u + v ) ^ { 2 } \leq 2 ( u ^ { 2 } + v ^ { 2 } )$ and $s \leq 1$

$$
\mathcal { L } ^ { ( 1 ) } ( g ) \leq 4 \int _ { 0 } ^ { 1 } \exp \{ - 4 a x \} d x + 4 \int _ { 0 } ^ { 1 } \exp \{ - 2 a x \} d x = \frac { 1 - \exp \{ - 4 a \} } { a } + \frac { 2 ( 1 - \exp \{ - 2 a \} ) } { a } \leq \frac { 3 } { a } .
$$

Therefore

$$
\mathcal { L } ^ { ( 1 ) } ( g ) \leq \frac { 3 } { C \log M _ { R } } ,
$$

and hence

$$
\mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \mathrm { s o f t } } ) \leq \frac { 3 } { C \log M _ { R } } .
$$

Now we prove the upper bound for $\mathcal { L } ^ { ( 2 ) }$ under softmax gating. For the same choice of scores, by symmetry,

$$
\mathcal { L } ^ { ( 2 ) } ( g ; \rho ) \leq 2 \int _ { \rho } ^ { 1 } \left( \exp \{ - 2 a x \} + s \exp \{ - a x \} \right) ^ { 2 } d x .
$$

Using again $( u + v ) ^ { 2 } \leq 2 ( u ^ { 2 } + v ^ { 2 } )$ and $s \leq 1$

$$
\mathcal { L } ^ { ( 2 ) } ( g ; \rho ) \leq 4 \int _ { \rho } ^ { 1 } \exp \{ - 4 a x \} d x + 4 \int _ { \rho } ^ { 1 } \exp \{ - 2 a x \} d x \leq \frac { \exp \{ - 4 a \rho \} } { a } + \frac { 2 \exp \{ - 2 a \rho \} } { a } \leq \frac { 3 \exp \{ - 2 a \rho \} } { a } .
$$

Since $a = C$ log $M _ { R }$

$$
\mathcal { L } ^ { ( 2 ) } ( g ; \rho ) \leq \frac { 3 } { C \log M _ { R } } M _ { R } ^ { - 2 C \rho } \leq \frac { 3 } { C \log 2 } M _ { R } ^ { - 2 C \rho } .
$$

Therefore

$$
\mathcal { L } ^ { ( 2 ) } ( \mathcal { G } _ { \mathrm { s o f t } } ; \rho ) \leq \frac { 3 } { C \log 2 } M _ { R } ^ { - 2 C \rho } .
$$

Finally we prove the results for Top-1 gating. Choose scores

$$
\Bigl ( s _ { 1 } ^ { \mathrm { l i n } } ( x ; \theta _ { 1 } ) , \ldots , s _ { M _ { R } } ^ { \mathrm { l i n } } ( x ; \theta _ { M _ { R } } ) \Bigr ) = ( - x , x , - 2 , \ldots , - 2 ) .
$$

Then the Top-1 rule assigns expert 1 on $[ - 1 , 0 ]$ and expert 2 on $( 0 , 1 ]$ , up to the boundary point $x = 0$ , which has $P _ { X }$ -measure zero. Therefore the oracle partition is realized exactly, and

$$
\begin{array} { r } { \mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \mathrm { T o p - 1 } } ) = 0 , \qquad \mathcal { L } ^ { ( 2 ) } ( \mathcal { G } _ { \mathrm { T o p - 1 } } ; \rho ) = 0 . } \end{array}
$$

This complete the proof.

## D.3 Proof of Example 5.3

Proof of Example 5.3. This proof supplies the formal disk example underlying Example 5.3. Let $\mathcal { X } = \{ x \in \mathbb { R } ^ { 2 } : \| x \| _ { 2 } \leq R _ { 2 } \}$ with $X \sim \operatorname { U n i f } ( \mathcal { X } )$ , and let the oracle regions be

$$
\begin{array} { r } { \mathcal { X } _ { 1 } = \{ x \in \mathcal { X } : \| x \| _ { 2 } \leq R _ { 1 } \} , \qquad \mathcal { X } _ { 2 } = \{ x \in \mathcal { X } : R _ { 1 } < \| x \| _ { 2 } \leq R _ { 2 } \} , } \end{array}
$$

where $0 < R _ { 1 } < R _ { 2 }$ . We prove that, for every fixed $0 < \rho < \operatorname* { m i n } \{ R _ { 1 } , R _ { 2 } - R _ { 1 } \}$ , there exist constants $C _ { L } , C _ { R } , \alpha > 0$ independent of $M _ { R }$ , such that

$$
\begin{array} { r } {  { \mathcal { L } } ^ { ( 2 ) } (  { \mathcal { G } } _ { \mathrm { l i n } } ; \rho ) \geq C _ { L } , \qquad { \mathcal { L } } ^ { ( 2 ) } (  { \mathcal { G } } _ { \mathrm { q u a d } } ; \rho ) \leq C _ { R } M _ { R } ^ { - \alpha } } \end{array}
$$

Define

$$
\chi _ { 1 } ^ { ( - \rho ) } : = \{ x \in \mathbb { R } ^ { 2 } : \| x \| _ { 2 } \leq R _ { 1 } - \rho \} , \qquad \chi _ { 2 } ^ { ( - \rho ) } : = \{ x \in \mathbb { R } ^ { 2 } : R _ { 1 } + \rho \leq \| x \| _ { 2 } \leq R _ { 2 } \} .
$$

Since $X \sim \operatorname { U n i f } ( \mathcal { X } )$ , we have

$$
\mathcal { L } ^ { ( 2 ) } ( g ; \rho ) = \frac { 1 } { \pi R _ { 2 } ^ { 2 } } \int _ { \mathcal { X } _ { 1 } ^ { ( - \rho ) } } \left( ( 1 - g _ { 1 } ) ^ { 2 } + \sum _ { j \neq 1 } g _ { j } ^ { 2 } \right) d x + \frac { 1 } { \pi R _ { 2 } ^ { 2 } } \int _ { \mathcal { X } _ { 2 } ^ { ( - \rho ) } } \left( ( 1 - g _ { 2 } ) ^ { 2 } + \sum _ { j \neq 2 } g _ { j } ^ { 2 } \right) d x .
$$

We first prove the lower bound for $\mathcal { L } ^ { ( 2 ) }$ under linear-softmax gating. For $g \in \mathcal { G } _ { \mathrm { l i n } }$ , write $\Delta ( x ) : = s _ { 2 } ^ { \mathrm { l i n } } ( x ; \theta _ { 2 } ) - s _ { 1 } ^ { \mathrm { l i n } } ( x ; \theta _ { 1 } ) = c + b ^ { \top } x$ . First suppose $c \geq 0$ . Since $\Delta ( x ) + \Delta ( - x ) = 2 c \geq 0$ , symmetry implies that $\Delta ( x ) \geq 0$ on a subset of $\mathcal { X } _ { 1 } ^ { ( - \rho ) }$ with Lebesgue measure at least $\mathrm { V o l } ( \mathcal { X } _ { 1 } ^ { ( - \rho ) } ) / 2$ . On this subset, $g _ { 1 } ( x ) \leq 1 / 2$ . Thus

$$
( 1 - g _ { 1 } ) ^ { 2 } + \sum _ { j \neq 1 } g _ { j } ^ { 2 } \geq \frac { 1 } { 4 } ,
$$

on this subset, and therefore

$$
\mathcal { L } ^ { ( 2 ) } ( g ; \rho ) \ge \frac { \mathrm { V o l } ( \mathcal { X } _ { 1 } ^ { ( - \rho ) } ) } { 8 \pi R _ { 2 } ^ { 2 } } = \frac { ( R _ { 1 } - \rho ) ^ { 2 } } { 8 R _ { 2 } ^ { 2 } } .
$$

If $c \leq 0$ , then $\Delta ( x ) + \Delta ( - x ) = 2 c \leq 0$ , so by the same symmetry argument $\Delta ( x ) \leq 0$ on at least half of $\ X _ { 2 } ^ { ( - \rho ) }$ . On that subset, $g _ { 2 } ( x ) \leq 1 / 2$ and hence

$$
\mathscr { L } ^ { ( 2 ) } ( g ; \rho ) \geq \frac { \mathrm { V o l } ( \chi _ { 2 } ^ { ( - \rho ) } ) } { 8 \pi R _ { 2 } ^ { 2 } } = \frac { R _ { 2 } ^ { 2 } - ( R _ { 1 } + \rho ) ^ { 2 } } { 8 R _ { 2 } ^ { 2 } } .
$$

Consequently,

$$
\mathcal { L } ^ { ( 2 ) } ( \mathcal { G } _ { \mathrm { l i n } } ; \rho ) \ge C _ { L } : = \frac { 1 } { 8 R _ { 2 } ^ { 2 } } \operatorname* { m i n } \left\{ ( R _ { 1 } - \rho ) ^ { 2 } , R _ { 2 } ^ { 2 } - ( R _ { 1 } + \rho ) ^ { 2 } \right\} > 0 .
$$

We next prove the upper bound for $\mathcal { L } ^ { ( 2 ) }$ under quadratic-softmax gating. Since $C > 1 / 2$ , choose

$$
0 < c _ { 0 } < \operatorname * { m i n } \left\{ \frac { 2 C - 1 } { R _ { 1 } ^ { 2 } } , \frac { C } { R _ { 1 } ^ { 2 } } , C \right\} .
$$

Then the interval $( \operatorname* { m a x } \{ 0 , 1 - C \} , C - c _ { 0 } R _ { 1 } ^ { 2 } )$ is nonempty. Choose b in this interval. Define

$$
s _ { 1 } ^ { \mathrm { q u a d } } ( x ; \theta _ { 1 } ) : = b \log M _ { R } + a R _ { 1 } ^ { 2 } - a \| x \| _ { 2 } ^ { 2 } ,
$$

$$
s _ { 2 } ^ { \mathrm { q u a d } } ( x ; \theta _ { 2 } ) : = b \log M _ { R } - a R _ { 1 } ^ { 2 } + a \| x \| _ { 2 } ^ { 2 } ,
$$

$$
s _ { j } ^ { \mathrm { q u a d } } ( x ; \theta _ { j } ) : = - B , \qquad j \geq 3 ,
$$

with $a : = c _ { 0 }$ log $M _ { R }$ and $B : = C \log M _ { R }$ . The choice of $c _ { 0 }$ and b ensures that all quadratic coeficients and intercepts in the first two scores are bounded by C log $M _ { R }$ in absolute value. The dummy scores also lie in the same box. Set

$$
\eta _ { \rho } : = R _ { 1 } ^ { 2 } - ( R _ { 1 } - \rho ) ^ { 2 } = 2 R _ { 1 } \rho - \rho ^ { 2 } > 0 , \qquad s : = ( M _ { R } - 2 ) \exp \{ - ( B + b \log M _ { R } ) \} .
$$

Since $C + b > 1$ , we have $s \leq 1$ for al suficiently large $M _ { R } ;$ smaller values of $M _ { R }$ can be absorbed into the constant. On $\mathcal { X } _ { 1 } ^ { ( - \rho ) }$

$$
\begin{array} { r } { s _ { 1 } ^ { \mathtt { q u a d } } ( x ; \theta _ { 1 } ) \geq b \log M _ { R } + a \eta _ { \rho } , \qquad s _ { 2 } ^ { \mathtt { q u a d } } ( x ; \theta _ { 2 } ) \leq b \log M _ { R } - a \eta _ { \rho } . } \end{array}
$$

Therefore,

$$
1 - g _ { 1 } \leq \frac { \exp \{ s _ { 2 } ^ { \mathrm { q u a d } } ( x ; \theta _ { 2 } ) \} + ( M _ { R } - 2 ) \exp \{ - B \} } { \exp \{ s _ { 1 } ^ { \mathrm { q u a d } } ( x ; \theta _ { 1 } ) \} } \leq \exp \{ - 2 a \eta _ { \rho } \} + \exp \{ - a \eta _ { \rho } \} \leq 2 \exp \{ - a \eta _ { \rho } \} .
$$

Similarly, on $\mathcal { X } _ { 2 } ^ { ( - \rho ) } , 1 - g _ { 2 } \le 2 \exp \{ - a \eta _ { \rho } \}$ . Thus on both interior regions,

$$
( 1 - g _ { i } ) ^ { 2 } + \sum _ { j \ne i } g _ { j } ^ { 2 } \le 2 ( 1 - g _ { i } ) ^ { 2 } \le 8 \exp \{ - 2 a \eta _ { \rho } \} , \qquad i = 1 , 2 .
$$

It follows that

$$
\mathcal { L } ^ { ( 2 ) } ( g ; \rho ) \leq \frac { 8 \exp \{ - 2 a \eta _ { \rho } \} } { \pi R _ { 2 } ^ { 2 } } \mathrm { V o l } ( \mathcal { X } ) = 8 \exp \{ - 2 a \eta _ { \rho } \} = 8 M _ { R } ^ { - 2 c _ { 0 } \eta _ { \rho } } ,
$$

where the last equality follows from $a = c _ { 0 }$ log $M _ { R }$ . Therefore

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { L } ^ { ( 2 ) } ( \mathcal { G } _ { \mathrm { q u a d } } ; \rho ) \leq 8 M _ { R } ^ { - \alpha } , \qquad \alpha : = 2 c _ { 0 } ( 2 R _ { 1 } \rho - \rho ^ { 2 } ) > 0 . } \end{array} } \end{array}
$$

Combining the lower bound for linear-softmax gating and the upper bound for quadraticsoftmax gating proves the claim. □

## D.4 Proof of Theorem 5.5

We first recall the grid notation used in the kernel construction. For a mesh width $h > 0$ define

$$
\mathcal { C } : = \{ 0 , h , 2 h , \ldots , \lfloor h ^ { - 1 } \rfloor h \} ^ { d } \cap [ 0 , 1 ] ^ { d }
$$

as the set of grid centers. The Gaussian kernel bandwidth is set to $\sigma = \kappa h$ where $\kappa \in [ 1 / 2 , 1 ]$ The following lemma bound the total Gaussian tail mass contributed by grid centers in C that lie at Euclidean distance at least r from a fixed point x.

Lemma D.1 (Gaussian tail sum over a uniform grid). Let $\sigma = \kappa h$ for some fixed $\kappa \in$ $[ 1 / 2 , 1 ]$ . Then there exists a constant $A _ { d , \kappa } > 0$ , depending only on $( d , \kappa )$ , such that for every $x \in [ 0 , 1 ] ^ { d }$ and every $r \geq 2 \sigma$ 2

$$
\sum _ { \stackrel { c \in \mathcal { C } } { | | x - c | | \geq r } } \exp \left\{ - \frac { | | x - c | | ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} \leq A _ { d , \kappa } \exp \left\{ - \frac { r ^ { 2 } } { 4 \sigma ^ { 2 } } \right\} .
$$

Proof of Lemma D.1. Let $\Lambda : = h \mathbb { Z } ^ { d } .$ . Since $\mathcal { C } \subset \Lambda$ and all terms are nonnegative,

$$
\sum _ { \stackrel { c \in \mathcal { C } } { | | x - c | | \geq r } } \exp \left\{ - \frac { | | x - c | | ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} \leq \sum _ { \stackrel { \lambda \in \Lambda } { | | x - \lambda | | \geq r } } \exp \left\{ - \frac { | | x - \lambda | | ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} = : S .
$$

It sufices to bound S. For $R \geq 0$ , let $N ( R ) : = \# \{ \lambda \in \Lambda : \| x - \lambda \| \leq R \}$ . Since the balls $B _ { d } ( \lambda , h / 2 ) , \lambda \in \Lambda$ , are pairwise disjoint up to boundaries of measure zero,

$$
N ( R ) { \mathrm { V o l } } ( B _ { d } ( 0 , h / 2 ) ) \leq { \mathrm { V o l } } ( B _ { d } ( 0 , R + h / 2 ) ) ,
$$

hence

$$
N ( R ) \leq \left( 1 + \frac { 2 R } { h } \right) ^ { d } \leq C _ { d } \left( 1 + \frac { R } { h } \right) ^ { d } .
$$

Now decompose the lattice points into dyadic shells

$$
A _ { m } : = \{ \lambda \in \Lambda : 2 ^ { m } r \leq \| x - \lambda \| < 2 ^ { m + 1 } r \} , \qquad m \geq 0 .
$$

Then

$$
S = \sum _ { m = 0 } ^ { \infty } \sum _ { \lambda \in A _ { m } } \exp \left\{ - \frac { \| x - \lambda \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} \leq \sum _ { m = 0 } ^ { \infty } \# A _ { m } \exp \left\{ - \frac { ( 2 ^ { m } r ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} .
$$

Using $\# A _ { m } \leq N ( 2 ^ { m + 1 } r )$ , we get

$$
S \leq C _ { d } \sum _ { m = 0 } ^ { \infty } \left( 1 + \frac { 2 ^ { m + 1 } r } { h } \right) ^ { d } \exp \left\{ - \frac { 2 ^ { 2 m } r ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} .
$$

Set $\scriptstyle t : = r / \sigma$ . Since $\textstyle r \geq 2 \sigma , t = { \frac { r } { \sigma } } \geq 2$ . Also $r / h = \kappa t$ . Therefore

$$
S \le C _ { d } \sum _ { m = 0 } ^ { \infty } ( 1 + 2 ^ { m + 1 } \kappa t ) ^ { d } \exp \{ - 2 ^ { 2 m - 1 } t ^ { 2 } \} .
$$

Because $t \geq 2$ and $\kappa \geq 1 / 2$ , we have $2 ^ { m + 1 } \kappa t \geq 2$ , hence

$$
( 1 + 2 ^ { m + 1 } \kappa t ) ^ { d } \leq C _ { d , \kappa } ( 2 ^ { m } t ) ^ { d } .
$$

Thus

$$
S \leq C _ { d , \kappa } \sum _ { m = 0 } ^ { \infty } ( 2 ^ { m } t ) ^ { d } \exp \{ - 2 ^ { 2 m - 1 } t ^ { 2 } \} .
$$

Now use the fact that $y \mapsto y ^ { d } \exp \{ - y ^ { 2 } / 8 \}$ is bounded on $[ 0 , \infty )$ . Hence there exists $C _ { d } ^ { \prime } > 0$ such that

$$
y ^ { d } \leq C _ { d } ^ { \prime } \exp \{ y ^ { 2 } / 8 \} , \qquad y \geq 0 .
$$

Applying this with $y = 2 ^ { m } t$ , we obtain

$$
\begin{array} { r } { ( 2 ^ { m } t ) ^ { d } \exp \{ - 2 ^ { 2 m - 1 } t ^ { 2 } \} \leq C _ { d } ^ { \prime } \exp \{ - 2 ^ { 2 m - 1 } t ^ { 2 } + 2 ^ { 2 m } t ^ { 2 } / 8 \} = C _ { d } ^ { \prime } \exp \{ - ( 3 / 8 ) 2 ^ { 2 m } t ^ { 2 } \} \leq C _ { d } ^ { \prime } \exp \{ - 2 ^ { 2 m - 2 } t ^ { 2 } \} . } \end{array}
$$

Therefore

$$
S \le C _ { d , \kappa } C _ { d } ^ { \prime } \sum _ { m = 0 } ^ { \infty } \exp \{ - 2 ^ { 2 m - 2 } t ^ { 2 } \} .
$$

Since $4 ^ { m } \geq m + 1$ ，

$$
\sum _ { m = 0 } ^ { \infty } \exp \{ - 2 ^ { 2 m - 2 } t ^ { 2 } \} = \sum _ { m = 0 } ^ { \infty } \exp \{ - ( t ^ { 2 } / 4 ) 4 ^ { m } \} \le \sum _ { m = 0 } ^ { \infty } \exp \{ - ( t ^ { 2 } / 4 ) ( m + 1 ) \} .
$$

For $\begin{array} { r } { t \ge 2 , \sum _ { m = 0 } ^ { \infty } \exp \{ - 2 ^ { 2 m - 2 } t ^ { 2 } \} \le \frac { e } { e - 1 } \exp \{ - t ^ { 2 } / 4 \} } \end{array}$ . Combining the preceding displays, and defining $A _ { d , \kappa } = e C _ { d , \kappa } C _ { d } ^ { \prime } / ( e - 1 )$ , we obtain

$$
S \le A _ { d , \kappa } \exp \{ - t ^ { 2 } / 4 \} = A _ { d , \kappa } \exp \left\{ - \frac { r ^ { 2 } } { 4 \sigma ^ { 2 } } \right\} .
$$

This proves the lemma.

Extension to general weights. We use the general region-wise target-weight notation introduced in Section 5 of the main text. Thus each cell $\mathcal { X } _ { i }$ is associated with a target vector $w _ { i } ^ { * } = ( w _ { i 1 } ^ { * } , \ldots , w _ { i M _ { R } } ^ { * } ) \in \Delta ^ { M _ { R } - 1 }$ . For the uniform grid $\mathcal { C } \subset [ 0 , 1 ] ^ { d }$ , assign each grid center $c \in { \mathcal { C } }$ to a unique region by a fixed deterministic tie-breaking rule and let

$$
C _ { i } : = \{ c \in \mathcal { C } : c \mathrm { ~ i s ~ a s s i g n e d ~ t o ~ } \mathcal { X } _ { i } \} , \qquad i = 1 , \dotsc , M _ { R } .
$$

Then $\{ C _ { i } \} _ { i = 1 } ^ { M _ { R } }$ forms a disjoint partition of C. We consider the choice

$$
w _ { r , c } : = \sum _ { i = 1 } ^ { M _ { R } } w _ { i r } ^ { * } \mathbb { 1 } _ { \{ c \in C _ { i } \} } , \qquad 1 \le r \le M _ { R } , \quad c \in \mathcal { C } .
$$

Thus, whenever $c \in C _ { i }$ , one has $w _ { r , c } = w _ { i r } ^ { * }$ . Define

$$
k _ { r } ( x ) : = \sum _ { i = 1 } ^ { M _ { R } } w _ { i r } ^ { * } \sum _ { c \in C _ { i } } \varphi ( x , c ; \sigma ) , \qquad g _ { r } ( x ) : = \frac { k _ { r } ( x ) } { \sum _ { s = 1 } ^ { M _ { R } } k _ { s } ( x ) } = \frac { k _ { r } ( x ) } { \sum _ { c \in \mathcal { C } } \varphi ( x , c ; \sigma ) } .
$$

The last equality follows because the vectors $\boldsymbol { w } _ { i } ^ { * }$ lie in the simplex and the sets $C _ { i }$ form a partition of C. The loss function is now defined as

$$
\mathcal { L } ^ { ( 1 ) } ( g ) : = \sum _ { i = 1 } ^ { M _ { R } } \Vert ( w _ { i } ^ { * } - g ( x ) ) \mathbb { 1 } _ { \mathcal { X } _ { i } } ( x ) \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } , \qquad \mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \mathrm { k e r } } ) : = \operatorname* { i n f } _ { g \in \mathcal { G } _ { \mathrm { k e r } } } \mathcal { L } ^ { ( 1 ) } ( g ) .
$$

Proof of Theorem 5.5. Set

$$
\rho _ { h } : = 2 \kappa h \sqrt { \left( \frac { d } { 2 } + 1 \right) \log ( 1 / h ) } .
$$

The boundary-layer regularity condition is only assumed for radii at most $\rho _ { 0 }$ . Since $\rho _ { h }  0$ as $h \downarrow 0$ , there exists $h _ { 0 } > 0$ , depending only on $( d , \kappa , \rho _ { 0 } )$ , such that $\rho _ { h } \leq \rho _ { 0 }$ whenever $0 < h \leq h _ { 0 }$ . If $h > h _ { 0 }$ , then, because both $g ( x )$ and every $\boldsymbol { w } _ { i } ^ { * }$ belong to $\Delta ^ { M _ { R } - 1 }$ ，

$$
\begin{array} { r } { \mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \mathrm { k e r } } ) \leq 2 \leq 2 h _ { 0 } ^ { - ( d + 2 ) } M _ { R } h ^ { d + 2 } , } \end{array}
$$

which already has the asserted form. It therefore remains to consider $0 < h \leq h _ { 0 }$ , and in this case $\rho _ { h } \le \rho _ { 0 }$ . Fix $i \in [ M _ { R } ]$ . We use the notation $\mathscr { X } _ { i } ^ { - \rho _ { h } }$ and $B _ { i } ( \rho _ { h } )$ from this definition. The assumed upper bound on h implies $\rho _ { h } \ge 2 \sigma$ , so Lemma D.1 is applicable with $r = \rho _ { h }$

We first estimate the error on $\mathscr { X } _ { i } ^ { - \rho _ { h } }$ . If $x \in \mathcal { X } _ { i } ^ { - \rho _ { h } }$ , then every $c \notin C _ { i }$ satisfies $\| x { - } c \| \geq \rho _ { h }$ Hence Lemma D.1 gives

$$
\sum _ { c \notin C _ { i } } \varphi ( x , c ; \sigma ) \leq A _ { d , \kappa } \exp \left\{ - \frac { \rho _ { h } ^ { 2 } } { 4 \sigma ^ { 2 } } \right\} .
$$

On the other hand, since C is a uniform grid with mesh width $h ,$ and taking $h ^ { - 1 } \in \mathbb { N }$ for notational simplicity, there exists a grid point $c _ { x } \in \mathcal { C }$ such that

$$
\| x - c _ { x } \| \leq { \frac { \sqrt { d } } { 2 } } h .
$$

The case of a general mesh is identical after changing the constant ${ \sqrt { d } } / 2$ to a comparable mesh constant. The assumed upper bound on $h$ ensures $\frac { \sqrt { d } } { 2 } h < \rho _ { h }$ . Since $x \in \mathcal { X } _ { i } ^ { - \rho _ { h } }$ , this implies $c _ { x } \in \mathcal { X } _ { i }$ . Moreover, $c _ { x }$ is strictly inside $\mathcal { X } _ { i } .$ , so the deterministic tie-breaking rule

assigns $c _ { x }$ to $C _ { i }$ . Therefore

$$
\sum _ { c \in C _ { i } } \varphi ( x , c ; \sigma ) \geq \varphi ( x , c _ { x } ; \sigma ) \geq \exp \left\{ - \frac { d } { 8 \kappa ^ { 2 } } \right\} .
$$

Using the definition of $g$ and $k _ { r }$ , for every $r _ { \mathrm { { ; } } }$

$$
\left| g _ { r } ( x ) - w _ { i r } ^ { * } \right| = \left| \frac { \sum _ { k \neq i } ( w _ { k r } ^ { * } - w _ { i r } ^ { * } ) \sum _ { c \in C _ { k } } \varphi ( x , c ; \sigma ) } { \sum _ { k = 1 } ^ { M _ { R } } \sum _ { c \in C _ { k } } \varphi ( x , c ; \sigma ) } \right| \leq \frac { \sum _ { c \notin C _ { i } } \varphi ( x , c ; \sigma ) } { \sum _ { c \in C _ { i } } \varphi ( x , c ; \sigma ) } .
$$

Hence

$$
| g _ { r } ( x ) - w _ { i r } ^ { * } | \leq A _ { d , \kappa } \exp \left\{ \frac { d } { 8 \kappa ^ { 2 } } \right\} \exp \left\{ - \frac { \rho _ { h } ^ { 2 } } { 4 \sigma ^ { 2 } } \right\} .
$$

Since $\begin{array} { r } { \frac { \rho _ { h } ^ { 2 } } { 4 \sigma ^ { 2 } } = ( d / 2 + 1 ) \log ( 1 / h ) } \end{array}$ , we have, for all $x \in \mathcal { X } _ { i } ^ { - \rho _ { h } }$

$$
| g _ { r } ( x ) - w _ { i r } ^ { * } | \leq C _ { d , \kappa } h ^ { d / 2 + 1 } .
$$

Therefore,

$$
\int _ { \chi _ { i } ^ { - \rho _ { h } } } \sum _ { r = 1 } ^ { M _ { R } } ( g _ { r } ( x ) - w _ { i r } ^ { \ast } ) ^ { 2 } d P _ { X } ( x ) \leq P _ { X } ( \chi _ { i } ) M _ { R } C _ { d , \kappa } h ^ { d + 2 } .
$$

We next control the boundary layer $B _ { i } ( \rho _ { h } )$ . Since $g ( x ) \ \in \ \Delta ^ { M _ { R } - 1 }$ and $w _ { i } ^ { * } \in { \Delta ^ { M _ { R } - 1 } }$ ， $\begin{array} { r } { \sum _ { r = 1 } ^ { M _ { R } } ( g _ { r } ( x ) - w _ { i r } ^ { * } ) ^ { 2 } \leq 2 } \end{array}$ . Thus

$$
\int _ { B _ { i } ( \rho _ { h } ) } \sum _ { r = 1 } ^ { M _ { R } } ( g _ { r } ( x ) - w _ { i r } ^ { * } ) ^ { 2 } d P _ { X } ( x ) \leq 2 P _ { X } ( B _ { i } ( \rho _ { h } ) ) .
$$

Because $p _ { X }$ is continuous on the compact set $[ 0 , 1 ] ^ { d }$ , we have $\| p _ { X } \| _ { \infty } < \infty$ . By the boundarylayer regularity of the partition, Vol $\begin{array} { r } { \left( B _ { i } ( \rho _ { h } ) \right) = \int _ { B _ { i } ( \rho _ { h } ) } 1 d x \leq \gamma _ { i } \rho _ { h } } \end{array}$ , and therefore

$$
P _ { X } ( B _ { i } ( \rho _ { h } ) ) \leq \| p _ { X } \| _ { \infty } \mathrm { V o l } \left( B _ { i } ( \rho _ { h } ) \right) \leq \| p _ { X } \| _ { \infty } \gamma _ { i } \rho _ { h } .
$$

Hence

$$
\int _ { B _ { i } ( \rho _ { h } ) } \sum _ { r = 1 } ^ { M _ { R } } ( g _ { r } ( x ) - w _ { i r } ^ { * } ) ^ { 2 } d P _ { X } ( x ) \leq 2 \| p _ { X } \| _ { \infty } \gamma _ { i } \rho _ { h } .
$$

Summing over $i = 1 , \ldots , M _ { R }$ , and using $\begin{array} { r } { \sum _ { i } P _ { X } ( \mathcal { X } _ { i } ) = 1 } \end{array}$ , we conclude that

$$
\begin{array} { r } { \mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \mathrm { k e r } } ) \leq C _ { d , \kappa } M _ { R } { h } ^ { { d } + 2 } + 2 \| \boldsymbol { p } _ { X } \| _ { \infty } \Gamma _ { M _ { R } } \rho _ { h } . } \end{array}
$$

Since $\rho _ { h } = 2 \kappa h \sqrt { ( d / 2 + 1 ) \log ( 1 / h ) }$ , the preceding display implies

$$
\begin{array} { r } { \mathcal { L } ^ { ( 1 ) } ( \mathcal { G } _ { \mathrm { k e r } } ) \leq C \left( M _ { R } h ^ { d + 2 } + \Gamma _ { M _ { R } } h \sqrt { \log ( 1 / h ) } \right) , } \end{array}
$$

for a constant C independent of $h , \Gamma _ { M _ { R } }$ , and $M _ { R }$

## D.5 Proof of Corollary 5.7

Proof of Corollary 5.7. Fix $i \in [ M _ { R } ]$ . The condition $h < 2 \rho / \sqrt { d }$ gives $\begin{array} { r } { \frac { \sqrt { d } } { 2 } h < \rho . } \end{array}$ . Moreover, since $\sigma = \kappa h$ , the condition $h < \rho / ( 2 \kappa )$ implies $\rho \geq 2 \sigma$ . Thus Lemma D.1 applies with $r = \rho .$ . The same interior-grid-point argument used in the proof of Theorem 5.5, with $\rho$ in place of $\rho _ { h }$ , gives that for every $x \in \mathcal { X } _ { i } ^ { - \rho }$ and every $r = 1 , \ldots , M _ { R }$

$$
| g _ { r } ( x ) - w _ { i r } ^ { * } | \leq C _ { d , \kappa } \exp \left\{ - \frac { \rho ^ { 2 } } { 4 \sigma ^ { 2 } } \right\} , \qquad \sigma = \kappa h .
$$

Therefore,

$$
\| g ( x ) - w _ { i } ^ { * } \| _ { 2 } ^ { 2 } = \sum _ { r = 1 } ^ { M _ { R } } ( g _ { r } ( x ) - w _ { i r } ^ { * } ) ^ { 2 } \leq M _ { R } C _ { d , \kappa } \exp \left\{ - \frac { \rho ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} .
$$

Integrating over $\mathcal { X } _ { i } ^ { - \rho }$ and summing over $i ,$ we obtain

$$
\begin{array} { l } { \displaystyle \sum _ { i = 1 } ^ { M _ { R } } \Big \| ( g ( x ) - w _ { i } ^ { * } ) \mathbb { 1 } _ { \displaystyle \chi _ { i } ^ { - \rho } } \Big \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = \displaystyle \sum _ { i = 1 } ^ { M _ { R } } \int _ { \chi _ { i } ^ { - \rho } } \| g ( x ) - w _ { i } ^ { * } \| _ { 2 } ^ { 2 } d P _ { X } ( x ) } \\ { \displaystyle \qquad \leq M _ { R } C _ { d , \kappa } \exp \left\{ - \frac { \rho ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} \displaystyle \sum _ { i = 1 } ^ { M _ { R } } P _ { X } ( \chi _ { i } ^ { - \rho } ) } \end{array}
$$

$$
\leq M _ { R } C _ { d , \kappa } \exp \left\{ - \frac { \rho ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} ,
$$

since the sets $\mathcal { X } _ { i } ^ { - \rho }$ are pairwise disjoint. Using $\sigma = \kappa h$ , we get

$$
\mathcal { L } ^ { ( 2 ) } ( \mathcal { G } _ { \mathrm { k e r } } ; \rho ) \leq C M _ { R } \exp \left\{ - \frac { \rho ^ { 2 } } { 2 \kappa ^ { 2 } h ^ { 2 } } \right\} ,
$$

for a constant C independent of h and $M _ { R }$ . This completes the proof.

## D.6 Proof of Theorem 5.9

We prove the guarantee for the aggregation-projection procedure in Algorithm 2 of the main text. Recall that each candidate estimator satisfies the class-specific oracle inequality (5.4) in the main text.

Proof of Theorem 5.9. Define

$$
R _ { \ell } : = \mathfrak { A } _ { \ell } + \mathfrak { L } _ { \ell } , \qquad \ell \in [ L ] .
$$

Throughout the proof, C denotes a positive constant that may change from line to line. It is independent of n, L, and $\ell ,$ but may depend on the fixed boundedness constants, the common envelope bound $A _ { 0 }$ , the error-distribution constants, and the uniform bound for the constants $C _ { \ell }$ in (5.4). By the sample splitting in Algorithm 2, after conditioning on the fitted candidate list $\{ \check { f } _ { \ell } \} _ { \ell = 1 } ^ { L }$ , we may apply Proposition B.3 to the calibration block $Z ^ { ( 2 ) }$ and the aggregation block $Z ^ { ( 3 ) }$ . In the notation of Proposition B.3, take

$$
S = L , \qquad \mu _ { \ell , t } ( x ) = \tilde { f } _ { \ell } ( x ) , \qquad \pi _ { \ell } = 1 / L , \qquad \hat { \sigma } _ { \ell , t } = \hat { \sigma } _ { \ell } ,
$$

for $t = n _ { 1 } + n _ { 2 } , \ldots , n - 1$ . Assumption 3.6 bounds $f _ { 0 } .$ , and the common envelope bound on $\{ \check { f } _ { \ell } \} _ { \ell = 1 } ^ { L }$ before (5.4) gives Assumption B.1. The truncation of $\hat { \sigma } _ { \ell }$ in Algorithm 2 gives Assumption B.2. Hence

$$
\frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } _ { \mathrm { s e l } } \| f _ { 0 } - \tilde { f } ^ { t } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \operatorname* { i n f } _ { \ell \in [ L ] } \left\{ \| f _ { 0 } - \tilde { f } _ { \ell } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \frac { \log L } { n _ { 3 } } + \mathbb { E } _ { \mathrm { s e l } } ( \hat { \sigma } _ { \ell } - \sigma ) ^ { 2 } \right\} .
$$

Moreover, the same calculation as in Lemma $\mathrm { B . 4 } .$ now with the fixed residual $r _ { \ell } ( x ) =$ $f _ { 0 } ( x ) - \check { f } _ { \ell } ( x )$ on the calibration block, gives

$$
\mathbb { E } _ { \mathrm { s e l } } ( \hat { \sigma } _ { \ell } - \sigma ) ^ { 2 } \leq C \left\{ \frac { 1 } { n _ { 2 } } + \| f _ { 0 } - \check { f } _ { \ell } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \right\} .
$$

Finally, since $\begin{array} { r } { \tilde { f } _ { n } = n _ { 3 } ^ { - 1 } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \tilde { f } ^ { t } } \end{array}$ , Jensen’s inequality implies

$$
\lVert f _ { 0 } - \tilde { f } _ { n } \rVert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \lVert f _ { 0 } - \tilde { f } ^ { t } \rVert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Combining these displays yields

$$
\mathbb { E } _ { \mathrm { s e l } } \Vert f _ { 0 } - \tilde { f } _ { n } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left( \operatorname* { i n f } _ { \ell \in [ L ] } \Vert f _ { 0 } - \check { f } _ { \ell } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \frac { \log L } { n _ { 3 } } + \frac { 1 } { n _ { 2 } } \right) ,
$$

where $\mathbb { E } _ { \mathrm { s e l } }$ denotes conditional expectation over $Z ^ { ( 2 ) }$ and $Z ^ { ( 3 ) }$ , given the fitted candidate list. Taking expectation over $Z ^ { ( 1 ) }$ and using (5.4), we obtain

$$
\begin{array} { r l } & { \mathbb { E } \| f _ { 0 } - \tilde { f } _ { n } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left( \underset { \ell \in [ L ] } { \operatorname* { i n f } } \ \mathbb { E } \| f _ { 0 } - \check { f } _ { \ell } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \frac { \log L } { n _ { 3 } } + \frac { 1 } { n _ { 2 } } \right) } \\ & { \qquad \leq C \left( \underset { \ell \in [ L ] } { \operatorname* { i n f } } \ R _ { \ell } + \frac { \log L } { n _ { 3 } \wedge n _ { 2 } } \right) . } \end{array}\tag{D.1}
$$

By the population $L _ { 2 } ( P _ { X } )$ -projection definition of $\ell ^ { * }$ , for every sample realization,

$$
\| \tilde { f } _ { n } - \check { f } _ { \ell ^ { * } } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = \operatorname* { i n f } _ { \ell \in [ L ] } \| \tilde { f } _ { n } - \check { f } _ { \ell } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Therefore,

$$
\mathbb { E } \| \tilde { f } _ { n } - \check { f } _ { \ell ^ { * } } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq \operatorname* { i n f } _ { \ell \in [ L ] } \mathbb { E } \| \tilde { f } _ { n } - \check { f } _ { \ell } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Using the inequality $\| a - b \| ^ { 2 } \leq 2 \| a - c \| ^ { 2 } + 2 \| c - b \| ^ { 2 }$ , we have

$$
\begin{array} { r l r } {  { \mathbb { E } \| f _ { 0 } - \check { f } _ { \ell ^ { * } } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le 2 ( \mathbb { E } \| f _ { 0 } - \tilde { f } _ { n } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \mathbb { E } \| \tilde { f } _ { n } - \check { f } _ { \ell ^ { * } } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } ) } } \\ & { } & { \le 2 ( \mathbb { E } \| f _ { 0 } - \tilde { f } _ { n } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \underset { \ell \in [ L ] } { \operatorname* { i n f } } \ \mathbb { E } \| \tilde { f } _ { n } - \check { f } _ { \ell } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } ) } \end{array}
$$

$$
\begin{array} { r l } & { \le 2 \left( \mathbb { E } \| f _ { 0 } - \tilde { f } _ { n } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + 2 \displaystyle \operatorname* { i n f } _ { \ell \in [ L ] } \left( \mathbb { E } \| f _ { 0 } - \tilde { f } _ { n } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \mathbb { E } \| f _ { 0 } - \tilde { f } _ { \ell } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \right) \right) } \\ & { \le C \left( \displaystyle \operatorname* { i n f } _ { \ell \in [ L ] } R _ { \ell } + \displaystyle \frac { \log L } { n _ { 3 } \wedge n _ { 2 } } \right) , } \end{array}
$$

where the last inequality follows from (D.1) and (5.4). Finally, if the split proportions are fixed so that $n _ { 1 } \asymp n _ { 2 } \asymp n _ { 3 } \asymp n .$ then the selection-stage cost is of order log $L / n$ . Hence

$$
\mathbb { E } \| f _ { 0 } - \check { f } _ { \ell ^ { * } } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left( \operatorname* { i n f } _ { \ell \in [ L ] } \{ \mathfrak { A } _ { \ell } + \mathfrak { L } _ { \ell } \} + \frac { \log L } { n } \right) ,
$$

where C does not depend on n or L. This proves the theorem.

## D.7 Proof of Corollary 5.10

Proof of Corollary 5.10. Throughout the proof, C denotes a positive constant that may change from line to line and is independent of n and L. By Theorem 5.9,

$$
\mathbb { E } \| f _ { 0 } - \check { f } _ { \ell ^ { * } } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left( \operatorname* { i n f } _ { \ell \in [ L ] } ( \mathfrak { A } _ { \ell } + \mathfrak { L } _ { \ell } ) + \frac { \log L } { n } \right) .
$$

It remains to upper bound the approximation term inf $\mathsf { \Gamma } _ { f \in \mathcal { F } _ { \ell } ^ { * } } \| f _ { 0 } - f \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 }$ for each fixed $\ell \in [ L ]$

Fix $\ell \in [ L ]$ , and let $f \in \mathcal { F } _ { \ell } ^ { * }$ be induced by some gating function $g \in { \mathcal { G } } _ { \ell }$ . By definition,

$$
f ( x ) = \sum _ { r = 1 } ^ { M _ { R } } g _ { r } ( x ) f _ { r } ^ { * } ( x ) .
$$

Since the oracle region assignment is $e _ { r }$ on $\mathcal { X } _ { r }$ , we may write

$$
f _ { 0 } ( x ) - f ( x ) = \sum _ { r = 1 } ^ { M _ { R } } ( f _ { 0 } ( x ) - f _ { r } ^ { * } ( x ) ) \mathbb { 1 } _ { \mathcal { X } _ { r } } ( x ) + \sum _ { r = 1 } ^ { M _ { R } } f _ { r } ^ { * } ( x ) \left( \mathbb { 1 } _ { \mathcal { X } _ { r } } ( x ) - g _ { r } ( x ) \right) .
$$

Therefore, using $( a + b ) ^ { 2 } \leq C ( a ^ { 2 } + b ^ { 2 } )$

$$
\| f _ { 0 } - f \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C ( T _ { 1 } + T _ { 2 } ) ,
$$

where

$$
T _ { 1 } : = \left\| \sum _ { r = 1 } ^ { M _ { R } } ( f _ { 0 } - f _ { r } ^ { * } ) \mathbb { 1 } _ { \boldsymbol { X } _ { r } } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } , \qquad T _ { 2 } : = \left\| \sum _ { r = 1 } ^ { M _ { R } } f _ { r } ^ { * } ( \mathbb { 1 } _ { \boldsymbol { X } _ { r } } - g _ { r } ) \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Since $\{ \mathcal { X } _ { r } \} _ { r = 1 } ^ { M _ { R } }$ form a partition of $\mathcal { X } ,$

$$
T _ { 1 } = \sum _ { r = 1 } ^ { M _ { R } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } .
$$

Next, for $x \in \mathcal { X } _ { r }$ , by $| f _ { j } ^ { * } ( x ) | \leq A$ in Assumption 3.6 and $\begin{array} { r } { \sum _ { j = 1 } ^ { M _ { R } } g _ { j } ( x ) = 1 } \end{array}$

$$
\left| \sum _ { j = 1 } ^ { M _ { R } } f _ { j } ^ { * } ( x ) \left( \mathbb { 1 } _ { \mathcal { X } _ { j } } ( x ) - g _ { j } ( x ) \right) \right| \leq A ( 1 - g _ { r } ( x ) ) + A \sum _ { j \neq r } g _ { j } ( x ) = 2 A ( 1 - g _ { r } ( x ) ) .
$$

Hence

$$
T _ { 2 } = \sum _ { r = 1 } ^ { M _ { R } } \left\| \sum _ { j = 1 } ^ { M _ { R } } f _ { j } ^ { * } ( \mathbb { 1 } _ { \mathcal { X } _ { j } } - g _ { j } ) \right\| _ { \mathcal { X } _ { r } } ^ { 2 } \leq C \sum _ { r = 1 } ^ { M _ { R } } \| 1 - g _ { r } \| _ { \mathcal { X } _ { r } } ^ { 2 } \leq C \sum _ { r = 1 } ^ { M _ { R } } \| ( e _ { r } - g ( x ) ) \mathbb { 1 } _ { \mathcal { X } _ { r } } ( x ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Combining the two bounds gives

$$
\| f _ { 0 } - f \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \sum _ { r = 1 } ^ { M _ { R } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + \sum _ { r = 1 } ^ { M _ { R } } \| ( e _ { r } - g ( x ) ) \mathbb { 1 } _ { \mathcal { X } _ { r } } ( x ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \right\} .
$$

Taking the infimum over all $f \in \mathcal { F } _ { \ell } ^ { * }$ , equivalently over the associated $g \in { \mathcal { G } } _ { \ell }$ , yields

$$
\operatorname* { i n f } _ { f \in { \mathscr F } _ { \ell } ^ { * } } \| f _ { 0 } - f \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \sum _ { r = 1 } ^ { M _ { R } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { { \mathscr X } _ { r } } ^ { 2 } + \operatorname* { i n f } _ { g \in { \mathscr G } _ { \ell } } \sum _ { r = 1 } ^ { M _ { R } } \| ( e _ { r } - g ( x ) ) \mathbb { 1 } _ { { \mathscr X } _ { r } } ( x ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \right\} .
$$

Substituting this bound into Theorem 5.9, and absorbing constants, gives

$$
\mathbb { E } \Vert f _ { 0 } - \check { f } _ { \ell ^ { * } } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left[ \sum _ { r = 1 } ^ { M _ { R } } \Vert f _ { 0 } - f _ { r } ^ { * } \Vert _ { \mathcal { X } _ { r } } ^ { 2 } + \operatorname* { i n f } _ { \ell \in [ L ] } \left\{ \operatorname* { i n f } _ { g \in \mathcal { G } _ { \ell } } \sum _ { r = 1 } ^ { M _ { R } } \Vert ( e _ { r } - g ( x ) ) \Vert \chi _ { r } ( x ) \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \mathfrak { L } _ { \ell } \right\} + \frac { \log L } { n } \right] .
$$

This proves the corollary.

## E Proof of Section 6

## E.1 Proof of Proposition 6.2

Proof of Proposition 6.2. The upper bound follows directly from Assumption 6.1 in the main text. The final comparison statement is also immediate once the lower bound for the pure-routed mechanism is established. We therefore focus on this lower bound.

Let $\begin{array} { r } { N _ { j } = \sum _ { i = 1 } ^ { n } \mathbb { 1 } _ { \{ X _ { i } \in \mathcal { X } _ { j } \} } } \end{array}$ be the number of observations in the j-th region. Conditional on $\left( N _ { 1 } , N _ { 2 } \right) = \left( m _ { 1 } , m _ { 2 } \right)$ , the observations falling in $\chi _ { j }$ , after relabelling, are i.i.d. from the conditional distribution $P _ { X } ( \cdot \ | \ X \in \mathcal { X } _ { j } )$ , with the same conditional noise distribution as in the definition of $\mathcal { R } _ { j } ^ { \mathcal { M } } ( h ; m , \sigma )$ . Thus, conditional on the regional counts, each regional learning rule is trained with a fixed sample size.

For a given $f _ { 0 } \in \mathcal { A } _ { \mathrm { s h } }$ , write $f _ { 0 } = f _ { S } + \mathbb { 1 } _ { \mathcal { X } _ { 1 } } f _ { R , 1 } + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } f _ { R , 2 }$ . Recall that $H _ { j } = f _ { S } + f _ { R , j }$ on $\chi _ { j }$ . By the definition of the pure-routed training rule and the disjointness of the two regions,

$$
\left. \hat { f } _ { \mathrm { 0 , r t } } - f _ { 0 } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = \sum _ { j = 1 } ^ { 2 } p _ { j } \left. \mathcal { M } _ { j } ( \mathcal { D } _ { m _ { j } , j } ^ { H _ { j } } ) - H _ { j } \right. _ { j } ^ { 2 } .
$$

Taking conditional expectation given $\left( N _ { 1 } , N _ { 2 } \right) = \left( m _ { 1 } , m _ { 2 } \right)$ , we obtain

$$
\begin{array} { r l } { \mathbb { E } _ { f _ { 0 } , \sigma } \left[ \left\| \hat { f } _ { 0 , \mathrm { r t } } - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \bigm | N _ { 1 } = m _ { 1 } , N _ { 2 } = m _ { 2 } \right] = \displaystyle \sum _ { j = 1 } ^ { 2 } p _ { j } \mathbb { E } _ { f _ { 0 } , \sigma } \left[ \left\| { \mathcal M } _ { j } ( { \mathcal P } _ { m _ { j } , j } ^ { H _ { j } } ) - { \mathcal H } _ { j } \right\| _ { j } ^ { 2 } \bigm | N _ { 1 } = m _ { 1 } , N _ { 2 } = m _ { 2 } \right] } & { } \\ { = \displaystyle \sum _ { j = 1 } ^ { 2 } p _ { j } { \mathcal R } _ { j } ^ { M } ( { H } _ { j } ; m _ { j } , \sigma ) } & { } \\ { \geq \displaystyle \operatorname* { m a x } _ { j = 1 , 2 } p _ { j } { \mathcal R } _ { j } ^ { M } ( { H } _ { j } ; m _ { j } , \sigma ) . } \end{array}
$$

Without loss of generality, suppose

$$
p _ { 1 } \tilde { \varrho } _ { 1 } \big ( n p _ { 1 } \big ) = p _ { 1 } \tilde { \varrho } _ { 1 } \big ( n p _ { 1 } \big ) \vee p _ { 2 } \tilde { \varrho } _ { 2 } \big ( n p _ { 2 } \big ) .
$$

Choose a branch function $H _ { 1 } ^ { \dagger } \in B _ { 1 }$ satisfying the branchwise lower bound $\mathcal { R } _ { 1 } ^ { \mathcal { M } } ( H _ { 1 } ^ { \dagger } ; m _ { 1 } , \sigma ) \geq$ $c \tilde { \varrho } _ { 1 } ( m )$ for all m in the relevant range $m \asymp n p _ { 1 }$ . By our setting, $H _ { 1 } ^ { \dagger }$ has the form $f _ { S } ^ { \dagger } | _ { X _ { 1 } } + f _ { R , 1 } ^ { \dagger }$ take $f _ { R , 2 } ^ { \dagger }$ be an arbitrary function in $\mathcal { F } _ { 2 }$ and let $f _ { 0 } ^ { \dagger } = f _ { S } ^ { \dagger } + \mathbb { 1 } _ { \mathcal { X } _ { 1 } } f _ { R , 1 } ^ { \dagger } + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } f _ { R , 2 } ^ { \dagger }$ . Therefore,

for every fixed $( m _ { 1 } , m _ { 2 } )$ with $m _ { 1 } \asymp n p _ { 1 }$ ，

$$
\begin{array} { r } { \mathbb { E } _ { f _ { 0 } ^ { \dagger } , \sigma } \left[ \left. \hat { f } _ { 0 , \mathrm { r t } } - f _ { 0 } ^ { \dagger } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \mid N _ { 1 } = m _ { 1 } , N _ { 2 } = m _ { 2 } \right] \geq p _ { 1 } \mathcal { R } _ { 1 } ^ { \mathcal { M } } ( H _ { 1 } ^ { \dagger } ; m _ { 1 } , \sigma ) \geq c p _ { 1 } \tilde { \varrho } _ { 1 } ( m _ { 1 } ) . } \end{array}\tag{E.1}
$$

Let $A _ { n } : = \{ n p _ { j } / 2 \leq N _ { j } \leq 2 n p _ { j } , \ j = 1 , 2 \}$ , and denote $p _ { \operatorname* { m i n } } : = \operatorname* { m i n } \{ p _ { 1 } , p _ { 2 } \}$ . By a Chernof bound, $\mathbb { P } ( A _ { n } ^ { c } ) \le 2 \exp \{ - c ^ { \prime } n p _ { \mathrm { m i n } } \}$ for some constant $c ^ { \prime } > 0$ . On $A _ { n } ,$ the regularity condition in Assumption 6.1 gives $\tilde { \varrho } _ { 1 } ( N _ { 1 } ) \asymp \tilde { \varrho } _ { 1 } ( n p _ { 1 } )$ . Thus, for all suficiently large n such that $\mathbb { P } ( A _ { n } )$ is bounded below by an absolute positive constant,

$$
\operatorname* { s u p } _ { f _ { 0 } \in \mathcal A _ { \mathrm { s h } } } \mathbb E _ { f _ { 0 } , \sigma } \left. \widehat f _ { 0 , \mathrm { r t } } - f _ { 0 } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \geq \mathbb E _ { f _ { 0 } ^ { \dagger } , \sigma } \left. \widehat f _ { 0 , \mathrm { r t } } - f _ { 0 } ^ { \dagger } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \geq c p _ { 1 } \mathbb E \left[ \widetilde \varrho _ { 1 } ( N _ { 1 } ) \mathbb { 1 } _ { A _ { n } } \right] \geq c p _ { 1 } \widetilde \varrho _ { 1 } ( n p _ { 1 } ) .
$$

Since $p _ { 1 } \tilde { \varrho } _ { 1 } ( n p _ { 1 } )$ is the larger of the two branch lower bounds by construction, this proves

$$
\operatorname* { s u p } _ { f _ { 0 } \in \mathcal A _ { \mathrm { s h } } } \mathbb E _ { f _ { 0 } , \sigma } \left\| \hat { f } _ { 0 , \mathrm { r t } } - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \geq c \{ p _ { 1 } \tilde { \varrho } _ { 1 } ( n p _ { 1 } ) \vee p _ { 2 } \tilde { \varrho } _ { 2 } ( n p _ { 2 } ) \} .
$$

If the second branch gives the maximum, the same argument is applied with $j = 2$ . This completes the proof. □

## E.2 A Sieve Justification for the Shared Upper Bound

Recall that Assumption 6.1 imposes the joint-estimation upper bound

$$
\operatorname* { s u p } _ { f _ { 0 } \in \mathcal A _ { \mathrm { s h } } } \mathbb E _ { f _ { 0 } , \sigma } \left. \hat { f } _ { 0 , \mathrm { s h } } - f _ { 0 } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C \left[ \varrho _ { S } ( n ) + p _ { 1 } \varrho _ { 1 } ( n p _ { 1 } ) + p _ { 2 } \varrho _ { 2 } ( n p _ { 2 } ) \right] .
$$

This subsection gives a suficient condition under standard sieve least squares.

Assume that, for the shared class $\mathcal { F } _ { S }$ and the residual classes ${ \mathcal { F } } _ { j }$ , there exist sieves $\mathcal { F } _ { S } ( k _ { S } )$ and $\mathcal { F } _ { j } ( k _ { j } )$ such that

$$
A _ { S } ( k _ { S } ) : = \operatorname* { s u p } _ { s \in \mathcal { F } _ { S } } \operatorname* { i n f } _ { u \in \mathcal { F } _ { S } ( k _ { S } ) } \| s - u \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } ,
$$

and

$$
A _ { j } ( k _ { j } ) : = \operatorname* { s u p } _ { r _ { j } \in \mathscr { F } _ { j } } \operatorname* { i n f } _ { v _ { j } \in \mathscr { F } _ { j } ( k _ { j } ) } \| r _ { j } - v _ { j } \| _ { j } ^ { 2 } , \qquad j = 1 , 2 .
$$

For the component classes, define the corresponding sieve oracle rates by

$$
\varrho _ { S } ( m ) : = \operatorname* { i n f } _ { k _ { S } } \left\{ A _ { S } ( k _ { S } ) + \frac { \sigma ^ { 2 } k _ { S } } { m } \right\} ,
$$

and

$$
\varrho _ { j } ( m ) : = \operatorname* { i n f } _ { k _ { j } } \left\{ A _ { j } ( k _ { j } ) + \frac { \sigma ^ { 2 } k _ { j } } { m } \right\} , \qquad j = 1 , 2 .
$$

These are the usual approximation–estimation tradeofs for sieve least-squares estimators: the first term is the approximation error of the sieve, and the second term is the variance cost of estimating k coeficients from m observations. Any additional logarithmic factors incurred by a particular sieve procedure can be absorbed into the definitions of $\varrho _ { S }$ and $\varrho _ { j }$

For the shared representation, we combine the three sieve learning rules through least squares over the additive sieve class

$$
\begin{array} { r } { S _ { \mathrm { s h } } ( \boldsymbol { k } _ { S } , \boldsymbol { k } _ { 1 } , \boldsymbol { k } _ { 2 } ) = \left\{ s + \mathbb { 1 } _ { \mathcal { X } _ { 1 } } r _ { 1 } + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } r _ { 2 } : s \in \mathcal { F } _ { S } ( \boldsymbol { k } _ { S } ) , \ r _ { j } \in \mathcal { F } _ { j } ( \boldsymbol { k } _ { j } ) \right\} . } \end{array}
$$

We impose the following regularity condition on least squares over this additive sieve.

Assumption E.1 (Compatible additive complexity). Assume that the least-squares estimator $\hat { h } _ { k _ { S } , k _ { 1 } , k _ { 2 } }$ over $S _ { \mathrm { s h } } ( k _ { S } , k _ { 1 } , k _ { 2 } )$ satisfies the oracle inequality

$$
\mathbb { E } _ { f _ { 0 } , \sigma } \Vert \hat { h } _ { k _ { S } , k _ { 1 } , k _ { 2 } } - f _ { 0 } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C \left[ \operatorname* { i n f } _ { h \in S _ { \mathrm { s h } } ( k _ { S } , k _ { 1 } , k _ { 2 } ) } \Vert h - f _ { 0 } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \frac { \sigma ^ { 2 } ( k _ { S } + k _ { 1 } + k _ { 2 } ) } { n } \right] .
$$

The following proposition shows that this standard sieve oracle inequality is suficient for the shared upper bound used in Assumption 6.1.

Proposition E.2 (Joint shared upper bound). Assume the known-partition model

$$
f _ { 0 } ( x ) = f _ { S } ( x ) + \mathbb { 1 } _ { \mathcal { X } _ { 1 } } ( x ) f _ { R , 1 } ( x ) + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } ( x ) f _ { R , 2 } ( x ) ,
$$

where $f _ { S } \in \mathcal { F } _ { S }$ and $f _ { R , j } \in \mathcal { F } _ { j }$ . Suppose Assumption E.1 holds. Then there exists a joint shared estimator

$$
\hat { f } _ { 0 , \mathrm { s h } } ( x ) = \hat { f } _ { S } ( x ) + \mathbb { 1 } _ { \chi _ { 1 } } ( x ) \hat { f } _ { R , 1 } ( x ) + \mathbb { 1 } _ { \chi _ { 2 } } ( x ) \hat { f } _ { R , 2 } ( x )
$$

obtained by least squares over a suitable additive sieve, such that

$$
\operatorname* { s u p } _ { f _ { 0 } \in \mathcal A _ { \mathrm { s h } } } \mathbb E _ { f _ { 0 } , \sigma } \| \hat { f } _ { 0 , \mathrm { s h } } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \varrho _ { S } ( n ) + p _ { 1 } \varrho _ { 1 } ( n p _ { 1 } ) + p _ { 2 } \varrho _ { 2 } ( n p _ { 2 } ) \right\} .
$$

Proof of Proposition E.2. Fix $k _ { S } , k _ { 1 } , k _ { 2 }$ . For any $f _ { S } \in \mathcal { F } _ { S }$ and $f _ { R , j } \in \mathcal { F } _ { j }$ , choose $s _ { k _ { S } } ~ \in$ $\mathcal { F } _ { S } ( k _ { S } )$ and $r _ { j , k _ { j } } \in \mathcal { F } _ { j } ( k _ { j } )$ such that

$$
\| s _ { k _ { S } } - f _ { S } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq 2 A _ { S } ( k _ { S } ) ,
$$

and

$$
\| r _ { j , k _ { j } } - f _ { R , j } \| _ { j } ^ { 2 } \leq 2 A _ { j } ( k _ { j } ) .
$$

Define

$$
h _ { k } ( \boldsymbol { x } ) = s _ { k _ { S } } ( \boldsymbol { x } ) + \mathbb { 1 } _ { \boldsymbol { \mathcal { X } } _ { 1 } } ( \boldsymbol { x } ) r _ { 1 , k _ { 1 } } ( \boldsymbol { x } ) + \mathbb { 1 } _ { \boldsymbol { \mathcal { X } } _ { 2 } } ( \boldsymbol { x } ) r _ { 2 , k _ { 2 } } ( \boldsymbol { x } ) .
$$

Then $h _ { k } \in  { S _ { \mathrm { s h } } } ( k _ { S } , k _ { 1 } , k _ { 2 } )$ . Using $( a + b + c ) ^ { 2 } \leq 3 ( a ^ { 2 } + b ^ { 2 } + c ^ { 2 } )$ , we obtain

$$
\begin{array} { r l } & { \| h _ { k } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \| s _ { k _ { S } } - f _ { S } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + C p _ { 1 } \| r _ { 1 , k _ { 1 } } - f _ { R , 1 } \| _ { 1 } ^ { 2 } + C p _ { 2 } \| r _ { 2 , k _ { 2 } } - f _ { R , 2 } \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq C \left\{ A _ { S } ( k _ { S } ) + p _ { 1 } A _ { 1 } ( k _ { 1 } ) + p _ { 2 } A _ { 2 } ( k _ { 2 } ) \right\} . } \end{array}
$$

By the oracle inequality in Assumption E.1,

$$
\mathbb { E } _ { f _ { 0 } , \sigma } \Vert \hat { h } _ { k _ { S } , k _ { 1 } , k _ { 2 } } - f _ { 0 } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C \left\{ A _ { S } ( k _ { S } ) + p _ { 1 } A _ { 1 } ( k _ { 1 } ) + p _ { 2 } A _ { 2 } ( k _ { 2 } ) + \frac { \sigma ^ { 2 } ( k _ { S } + k _ { 1 } + k _ { 2 } ) } { n } \right\} .
$$

Because $k _ { j } / n = p _ { j } k _ { j } / ( n p _ { j } )$ , thus

$$
{ \frac { k _ { S } + k _ { 1 } + k _ { 2 } } { n } } = { \frac { k _ { S } } { n } } + p _ { 1 } { \frac { k _ { 1 } } { n p _ { 1 } } } + p _ { 2 } { \frac { k _ { 2 } } { n p _ { 2 } } } .
$$

Taking the infimum over $k _ { S } , k _ { 1 } , k _ { 2 }$ , we get

$$
\mathbb { E } _ { f _ { 0 } , \sigma } \Vert \hat { f } _ { \boldsymbol { 0 } , \mathrm { s h } } - f _ { 0 } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left[ \operatorname* { i n f } _ { k _ { S } } \left\{ A _ { S } ( k _ { S } ) + \frac { \sigma ^ { 2 } k _ { S } } { n } \right\} + \sum _ { j = 1 } ^ { 2 } p _ { j } \operatorname* { i n f } _ { k _ { j } } \left\{ A _ { j } ( k _ { j } ) + \frac { \sigma ^ { 2 } k _ { j } } { n p _ { j } } \right\} \right] .
$$

Choosing sieve dimension $( k _ { S } ^ { * } , k _ { 1 } ^ { * } , k _ { 2 } ^ { * } )$ attaining the preceding infima, or within a fixed constant factor if the infima are not attained, and set

$$
\hat { f } _ { 0 , \mathrm { s h } } : = \hat { h } _ { { k _ { S } ^ { * } } , { k _ { 1 } ^ { * } } , { k _ { 2 } ^ { * } } } .
$$

By the definition of $\varrho _ { S }$ and $\varrho _ { j }$ , this gives

$$
\begin{array} { r } { \mathbb { E } _ { f _ { 0 } , \sigma } \| \hat { f } _ { 0 , \mathrm { s h } } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \varrho _ { S } ( n ) + p _ { 1 } \varrho _ { 1 } ( n p _ { 1 } ) + p _ { 2 } \varrho _ { 2 } ( n p _ { 2 } ) \right\} . } \end{array}
$$

Since the bound is uniform in $f _ { 0 } \in \mathcal { A } _ { \mathrm { s h } }$ , taking the supremum over $f _ { 0 }$ yields

$$
\operatorname* { s u p } _ { f _ { 0 } \in \mathcal A _ { \mathrm { s h } } } \mathbb E _ { f _ { 0 } , \sigma } \lVert \hat { f } _ { 0 , \mathrm { s h } } - f _ { 0 } \rVert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \varrho _ { S } ( n ) + p _ { 1 } \varrho _ { 1 } ( n p _ { 1 } ) + p _ { 2 } \varrho _ { 2 } ( n p _ { 2 } ) \right\} .
$$

This proves the result.

## E.3 Proof of Example 6.4

Lemma E.3 (Risk lower bound for the Fourier-sieve learner). Let $X \ \sim \ \mathrm { U n i f } [ 0 , 1 ]$ , and suppose

$$
Y = h ( X ) + \xi , \qquad \mathbb { E } ( \xi \mid X ) = 0 , \qquad \operatorname { V a r } ( \xi \mid X ) \geq c _ { \xi } \sigma ^ { 2 } .
$$

Let $S _ { K } = \operatorname { s p a n } \{ 1 , \sin ( 2 \pi k t ) , \cos ( 2 \pi k t ) : 1 \leq k \leq K \}$ , and let $\hat { h } _ { K }$ be the Fourier orthogonalseries estimator over $\boldsymbol { \mathcal { S } } _ { K }$ . Then

$$
\mathbb { E } \| \hat { h } _ { K } - h \| _ { L _ { 2 } [ 0 , 1 ] } ^ { 2 } \geq c \left\{ \| h - \Pi _ { K } h \| _ { L _ { 2 } [ 0 , 1 ] } ^ { 2 } + \frac { \sigma ^ { 2 } K } { m } \right\} ,
$$

where $\Pi _ { K }$ is the L<sub>2</sub>[0, 1]-projection onto $\scriptstyle { S _ { K } }$

Proof of Lemma E.3. Write $\hat { h } _ { K } - h = \left( \hat { h } _ { K } - \Pi _ { K } h \right) + \left( \Pi _ { K } h - h \right)$ . Since $\hat { h } _ { K } - \Pi _ { K } h \in { \cal S } _ { K }$ and $\Pi _ { K } h - h \perp S _ { K }$ , the Pythagorean identity gives

$$
\begin{array} { r } { \| \hat { h } _ { K } - h \| _ { L _ { 2 } [ 0 , 1 ] } ^ { 2 } = \| \hat { h } _ { K } - \Pi _ { K } h \| _ { L _ { 2 } [ 0 , 1 ] } ^ { 2 } + \| h - \Pi _ { K } h \| _ { L _ { 2 } [ 0 , 1 ] } ^ { 2 } . } \end{array}
$$

It remains to lower-bound the first term. Let $\{ \phi _ { \ell } \} _ { \ell = 1 } ^ { d _ { K } }$ be an orthonormal basis of $\boldsymbol { \mathcal { S } } _ { K }$ , where

$d _ { K } = 2 K + 1$ . Then

$$
\Pi _ { K } h ( t ) = \sum _ { \ell = 1 } ^ { d _ { K } } \theta _ { \ell } \phi _ { \ell } ( t ) , \qquad \theta _ { \ell } = \mathbb { E } \{ Y \phi _ { \ell } ( X ) \} ,
$$

where the identity follows from $\mathbb { E } ( \xi \mid X ) = 0$ . The orthogonal-series estimator is

$$
\widehat { h } _ { K } ( t ) = \sum _ { \ell = 1 } ^ { d _ { K } } \widehat { \theta } _ { \ell } \phi _ { \ell } ( t ) , \qquad \widehat { \theta } _ { \ell } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } Y _ { i } \phi _ { \ell } ( X _ { i } ) .
$$

Therefore,

$$
\Vert \widehat { h } _ { K } - \Pi _ { K } h \Vert ^ { 2 } = \sum _ { \ell = 1 } ^ { d _ { K } } ( \widehat { \theta } _ { \ell } - \theta _ { \ell } ) ^ { 2 } .
$$

For each $\ell ,$

$$
\mathbb { E } ( \widehat { \theta } _ { \ell } - \theta _ { \ell } ) ^ { 2 } = \frac { 1 } { m } \operatorname { V a r } \{ Y \phi _ { \ell } ( X ) \} \geq \frac { 1 } { m } \mathbb { E } \{ \operatorname { V a r } ( Y \phi _ { \ell } ( X ) \mid X ) \} .
$$

Since $Y = h ( X ) + \xi { \mathrm { ~ a n d ~ } } \operatorname { V a r } ( \xi \mid X ) \geq c _ { \xi } \sigma ^ { 2 }$

$$
\operatorname { V a r } ( Y \phi _ { \ell } ( X ) \mid X ) = \phi _ { \ell } ( X ) ^ { 2 } \operatorname { V a r } ( \xi \mid X ) \geq c _ { \xi } \sigma ^ { 2 } \phi _ { \ell } ( X ) ^ { 2 } .
$$

Thus, using $X \sim \mathrm { U n i f } [ 0 , 1 ]$ and the orthonormality of $\phi _ { \ell } .$

$$
\mathbb { E } ( \widehat { \theta } _ { \ell } - \theta _ { \ell } ) ^ { 2 } \geq \frac { c _ { \xi } \sigma ^ { 2 } } { m } \mathbb { E } \phi _ { \ell } ( X ) ^ { 2 } = \frac { c _ { \xi } \sigma ^ { 2 } } { m } .
$$

Summing over $\ell = 1 , \ldots , d _ { K }$ gives

$$
\mathbb { E } \| \widehat { h } _ { K } - \Pi _ { K } h \| _ { L _ { 2 } [ 0 , 1 ] } ^ { 2 } = \sum _ { \ell = 1 } ^ { d _ { K } } \mathbb { E } ( \widehat { \theta } _ { \ell } - \theta _ { \ell } ) ^ { 2 } \geq c \frac { \sigma ^ { 2 } K } { m } ,
$$

Because $d _ { K } = 2 K + 1$ . Combining this variance lower bound with the approximation term proves the claim. □

Proof of Example $6 . 4 \cdot$ . We first verify the shared upper bound. Consider the finite-dimensional

additive class

$$
\mathcal { A } _ { \mathrm { s h } , K _ { 0 } } = \{ \alpha x + \mathbb { 1 } _ { \mathcal { X } _ { 1 } } ( x ) r _ { 1 } ( t _ { 1 } ( x ) ) + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } ( x ) r _ { 2 } ( t _ { 2 } ( x ) ) : \alpha \in \mathbb { R } , \ r _ { j } \in \mathcal { S } _ { K _ { 0 } } \} .
$$

By construction, the true regression function belongs to $\mathcal { A } _ { \mathrm { s h } , K _ { 0 } }$ . The class $\mathcal { A } _ { \mathrm { s h } , K _ { 0 } }$ is finitedimensional with fixed dimension $d _ { \mathrm { s h } }$ . Moreover, the basis functions defining $\mathcal { A } _ { \mathrm { s h } , K _ { 0 } }$ are linearly independent under the uniform design. Indeed, if

$$
\alpha x + \mathbb { 1 } _ { \mathcal { X } _ { 1 } } ( x ) r _ { 1 } ( t _ { 1 } ( x ) ) + \mathbb { 1 } _ { \mathcal { X } _ { 2 } } ( x ) r _ { 2 } ( t _ { 2 } ( x ) ) = 0 \quad \mathrm { a . e . , }
$$

with $r _ { j } \in \mathcal S _ { K _ { 0 } }$ , then on each region $r _ { j }$ must coincide with an afine function of the local coordinate $t _ { j }$ . A finite trigonometric polynomial on [0, 1] is periodic at the endpoints, and therefore cannot coincide with a nonconstant afine function. Hence $\alpha = 0$ , and then $r _ { 1 } = r _ { 2 } = 0$ . Thus the population Gram matrix is positive definite. Since its dimension is fixed, its smallest eigenvalue is bounded away from zero by a constant. Consequently, ordinary least squares over $\mathcal { A } _ { \mathrm { s h } , K _ { 0 } }$ satisfies

$$
\mathbb { E } \Vert \hat { f } _ { 0 , \mathrm { s h } } - f _ { 0 } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \lesssim \frac { \sigma ^ { 2 } d _ { \mathrm { s h } } } { n } \lesssim \frac { \sigma ^ { 2 } } { n } .\tag{E.2}
$$

We next derive the Fourier-sieve risk for the pure-routed branch. It is enough to work on one region in the local coordinate $t \in [ 0 , 1 ]$ . Since the noise is Gaussian, the variance condition in Lemma E.3 holds with $c _ { \xi } = 1$ . Thus, by Lemma E.3, it remains to control the approximation error term.

In the local coordinate, the branch target has the form

$$
H _ { j } ( t ) = \gamma \left( c _ { j } + \frac { t } { 2 } \right) + \sum _ { k = 1 } ^ { K _ { 0 } } \left( a _ { j , k } \sin ( 2 k \pi t ) + b _ { j , k } \cos ( 2 k \pi t ) \right) ,
$$

where $c _ { 1 } = 0$ and $c _ { 2 } = 1 / 2$ . For $K \geq K _ { 0 }$ , the constant term and the first sine-cosine component belong to $\boldsymbol { S _ { K } }$ . Hence the approximation error is determined by the ramp component t.

The Fourier expansion of t on [0, 1] is

$$
t = \frac { 1 } { 2 } - \frac { 1 } { \pi } \sum _ { k = 1 } ^ { \infty } \frac { \sin ( 2 \pi k t ) } { k } \mathrm { i n } { \cal L } _ { 2 } [ 0 , 1 ] .
$$

Using the orthonormal sine basis $\{ { \sqrt { 2 } } \sin ( 2 \pi k t ) \} _ { k \geq 1 }$ , the corresponding Fourier coeficient is

$$
\left. t , \sqrt { 2 } \sin ( 2 \pi k t ) \right. = - \frac { 1 } { \sqrt { 2 } \pi k } .
$$

Therefore,

$$
\operatorname* { i n f } _ { g \in S _ { K } } \| t - g \| _ { L _ { 2 } [ 0 , 1 ] } ^ { 2 } = \frac { 1 } { 2 \pi ^ { 2 } } \sum _ { k > K } \frac { 1 } { k ^ { 2 } } \asymp K ^ { - 1 } .
$$

Since $| \gamma | \geq \gamma _ { 0 }$ , the ramp contribution in $H _ { j }$ yields, for all $K \geq K _ { 0 }$ 2

$$
\operatorname* { i n f } _ { g \in S _ { K } } \| H _ { j } - g \| _ { L _ { 2 } [ 0 , 1 ] } ^ { 2 } \geq c \gamma _ { 0 } ^ { 2 } K ^ { - 1 } .
$$

For the finitely many choices $K < K _ { 0 }$ , the same lower bound is harmless after changing the constant: the approximation error is bounded below by a positive constant depending only on $K _ { 0 }$ and $\gamma _ { 0 }$ , which dominates $K ^ { - 1 }$ over this finite range. Hence, up to a universal constant depending on $K _ { 0 }$

$$
\operatorname* { i n f } _ { g \in S _ { K } } \| H _ { j } - g \| _ { L _ { 2 } [ 0 , 1 ] } ^ { 2 } \geq c \gamma _ { 0 } ^ { 2 } K ^ { - 1 } , \qquad K \geq 1 .
$$

Thus, by Lemma E.3, the oracle risk of the Fourier-orthogonal-series learner on $H _ { j }$ is bounded below by

$$
\operatorname * { i n f } _ { K \geq 1 } \left\{ c \gamma _ { 0 } ^ { 2 } K ^ { - 1 } + c ^ { \prime } \frac { \sigma ^ { 2 } K } { m } \right\} .
$$

The minimum is achieved at $K \asymp \gamma _ { 0 } \sqrt { m } / \sigma$ , and the corresponding value is of order $\gamma _ { 0 } \sigma / { \sqrt { m } }$ Therefore,

$$
\mathcal { R } _ { j } ^ { \mathcal { M } } ( H _ { j } ; m , \sigma ) \stackrel { > } { \sim } \frac { \gamma _ { 0 } \sigma } { \sqrt { m } } .
$$

Finally, the pure-routed estimator has disjoint regional loss decomposition:

$$
\begin{array} { r } { \lVert \hat { f } _ { 0 , \mathrm { r t } } - f _ { 0 } \rVert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = p _ { 1 } \lVert \hat { H } _ { 1 } - H _ { 1 } \rVert _ { 1 } ^ { 2 } + p _ { 2 } \lVert \hat { H } _ { 2 } - H _ { 2 } \rVert _ { 2 } ^ { 2 } . } \end{array}
$$

Let $\begin{array} { r } { N _ { j } = \sum _ { i = 1 } ^ { n } \Im \{ X _ { i } \in \mathcal { X } _ { j } \} } \end{array}$ be the number of training observations in region $\chi _ { j }$ . Taking expectations first conditional on $\left( N _ { 1 } , N _ { 2 } \right) = \left( m _ { 1 } , m _ { 2 } \right)$ and using the same concentrationevent argument as in the proof of Proposition 6.2, with $p _ { 1 } = p _ { 2 } = 1 / 2$ , gives

$$
\mathbb { E } \| \hat { f } _ { 0 , \mathrm { r t } } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \gtrsim \sum _ { j = 1 } ^ { 2 } p _ { j } \frac { \gamma _ { 0 } \sigma } { \sqrt { n p _ { j } } } \asymp \frac { \gamma _ { 0 } \sigma } { \sqrt { n } } .
$$

Combining this lower bound with the shared upper bound (E.2) proves the stated procedurespecific rate gap. □

## F Additional Materials

## F.1 Order of Empirical Risk Minimization under Sub-Gaussian Noise

We now examine what the traditional empirical risk minimization (ERM) theory would yield for the gating-estimation problem considered in the main text. The purpose of this discussion is not to advocate ERM computationally, since the corresponding optimization problem is generally non-convex. Rather, it is to show that even if a global empirical minimizer were available, the standard uniform-concentration argument leads to a squareroot complexity term.

Recall that in Algorithm 1, we use the first sample $Z ^ { ( 1 ) }$ to burn-in, the second sample $Z ^ { ( 2 ) }$ to calibrate the scale, and the last sample to do online aggregation and estimation for θ. For $t = n _ { 1 } + n _ { 2 } , . . . , n - 1$ , the one-step-ahead dense-softmax predictor is denoted as

$$
\check { f } ^ { t } ( x ; \theta ) = \sum _ { \ell = 1 } ^ { M _ { S } } \hat { f } _ { S , \ell } ^ { t } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) \hat { f } _ { R , m } ^ { t } ( x ) , \qquad \theta \in \Theta _ { M _ { R } } .\tag{F.1}
$$

Consequently, for $M _ { R } \ge 2$ , define the fixed-expert dense softmax class

$$
\mathcal { F } _ { M _ { R } } ^ { \mathrm { o n } } : = \left\{ \{ \check { f } ^ { t } ( \cdot ; \theta ) \} _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } : \theta \in \Theta _ { M _ { R } } \right\} .
$$

For comparison with the oracle terms in Theorem 3.8 of the main text, also define the corresponding limit-expert predictor

$$
f ^ { * } ( x ; \theta ) = \sum _ { \ell = 1 } ^ { M _ { S } } f _ { S , \ell } ^ { * } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { R , m } ^ { * } ( x ) .
$$

The direct online ERM benchmark minimizes the one-step-ahead squared loss over the aggregation block:

$$
R _ { n _ { 3 } } ( \theta ) : = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \{ Y _ { t + 1 } - \check { f } ^ { t } ( X _ { t + 1 } ; \theta ) \} ^ { 2 } .\tag{F.2}
$$

Let

$$
\hat { \theta } _ { \mathrm { E R M } } \in \mathop { \mathrm { a r g } } _ { \theta \in \Theta _ { M _ { R } } } R _ { n _ { 3 } } ( \theta ) .
$$

This is a theoretical oracle benchmark, because solving this requires global optimization of a non-convex softmax objective. The resulting online sequence and its time-averaged version are

$$
\tilde { f } _ { \mathrm { E R M } } ^ { t } ( x ) = \check { f } ^ { t } ( x ; \hat { \theta } _ { \mathrm { E R M } } ) , \qquad \bar { f } _ { \mathrm { E R M } } ( x ) = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \check { f } ^ { t } ( x ; \hat { \theta } _ { \mathrm { E R M } } ) .
$$

Equivalently,

$$
\bar { f } _ { \mathrm { E R M } } ( x ) = \sum _ { \ell = 1 } ^ { M _ { S } } \bar { f } _ { S , \ell } ^ { n _ { 3 } } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \hat { \theta } _ { \mathrm { E R M } } ) \bar { f } _ { R , m } ^ { n _ { 3 } } ( x ) ,
$$

where

$$
\bar { f } _ { S , \ell } ^ { n _ { 3 } } = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \hat { f } _ { S , \ell } ^ { t } , \qquad \bar { f } _ { R , m } ^ { n _ { 3 } } = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \hat { f } _ { R , m } ^ { t } .
$$

Moreover, define

$$
\mathcal { R } _ { n _ { 3 } } ( \theta ) : = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \big \lVert \check { f } ^ { t } ( \cdot ; \theta ) - f _ { 0 } \big \rVert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .\tag{F.3}
$$

This is the random prequential prediction risk of the fixed gate θ along the online expert trajectory. The next proposition gives the ERM benchmark.

Proposition F.1 (ERM under sub-Gaussian noise). Suppose that $Y _ { i } = f _ { 0 } ( X _ { i } ) + \varepsilon _ { i }$ and $\mathbb { E } ( \varepsilon _ { i } \mid X _ { i } ) = 0$ , where $\varepsilon _ { i }$ is conditionally sub-Gaussian with parameter $\sigma _ { \varepsilon } ,$ and the observations are independent. Assume Assumptions 3.4 and 3.6 in the main text hold. Then there exist constants $C , c > 0$ , depending only on the constants in Assumptions $\ 3 . 4$ and 3.6, $\sigma _ { \varepsilon }$ and the fixed number of shared experts $M _ { S }$ , such that, conditional on the pretrained experts, with probability at least $1 - n _ { 3 } ^ { - c }$

$$
\mathcal { R } _ { n _ { 3 } } ( \widehat { \theta } _ { \mathrm { E R M } } ) - \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \mathcal { R } _ { n _ { 3 } } ( \theta ) \leq C \sqrt { \frac { M _ { R } d \log ( M _ { R } d n _ { 3 } ) } { n _ { 3 } } } .\tag{F.4}
$$

Moreover, if Assumption 3.5 in the main text also holds, then

$$
\mathbb { E } \mathcal { R } _ { n _ { 3 } } ( \widehat { \theta } _ { \mathrm { E R M } } ) \leq C \left\{ \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + M _ { R } \bar { a } _ { n _ { 3 } } ^ { 2 } + \sqrt { \frac { M _ { R } d \log ( M _ { R } d n _ { 3 } ) } { n _ { 3 } } } \right\} ,\tag{F.5}
$$

where we recall that

$$
\mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) : = \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \Vert f _ { 0 } - f ^ { * } ( \cdot ; \theta ) \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } , \qquad \bar { a } _ { n _ { 3 } } ^ { 2 } = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } a _ { t } ^ { 2 } .
$$

Moreover, the same bound holds for the time-averaged output <sup>¯</sup>f<sub>ERM</sub> as

$$
\mathbb { E } \Vert \bar { f } _ { \mathrm { E R M } } - f _ { 0 } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left\{ \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + M _ { R } \bar { a } _ { n _ { 3 } } ^ { 2 } + \sqrt { \frac { M _ { R } d \log ( M _ { R } d n _ { 3 } ) } { n _ { 3 } } } \right\} .\tag{F.6}
$$

Proof of Proposition F.1. Let $\mathcal { F } _ { t }$ be the natural online filtration generated by $Z _ { 1 } , \ldots , Z _ { t }$ and by the algorithmic randomness used up to time t. By the online update order, $\check { f } ^ { t } ( \cdot ; \theta )$ is $\mathcal { F } _ { t ^ { - } }$ measurable for each θ. Moreover, by Assumption 3.6 and the identity $\begin{array} { r } { \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) = 1 } \end{array}$ ， there exists a constant $B _ { 0 } > 0$ , depending on the uniform bounds and on the fixed number $M _ { S }$ , such that

$$
| f _ { 0 } ( x ) | \vee | \check { f } ^ { t } ( x ; \theta ) | \le B _ { 0 } , \qquad x \in \mathcal { X } , \quad \theta \in \Theta _ { M _ { R } } \quad t \ge n _ { 1 } + n _ { 2 } .\tag{F.7}
$$

For fixed θ, define

$$
\begin{array} { r } { V _ { t + 1 } ( \theta ) : = Y _ { t + 1 } - \check { f } ^ { t } ( X _ { t + 1 } ; \theta ) = \varepsilon _ { t + 1 } + f _ { 0 } ( X _ { t + 1 } ) - \check { f } ^ { t } ( X _ { t + 1 } ; \theta ) . } \end{array}
$$

By (F.7), the conditional mean component $f _ { 0 } ( X _ { t + 1 } ) - \check { f } ^ { t } ( X _ { t + 1 } ; \theta )$ is uniformly bounded. Therefore, conditional on $\mathcal { F } _ { t }$ , the centered variable

$$
V _ { t + 1 } ( \theta ) - \mathbb { E } [ V _ { t + 1 } ( \theta ) \mid \mathcal { F } _ { t } ]
$$

is sub-Gaussian with a parameter depending only on $A , M _ { S }$ , and $\sigma _ { \varepsilon }$ . By Lemma F.2, for every fixed θ and every $0 < u \leq 1$ ，

$$
\begin{array} { r } { \mathbb { P } \left( \left| R _ { n _ { 3 } } ( \theta ) - \bar { R } _ { n _ { 3 } } ( \theta ) \right| \geq u \right) \leq 2 \exp \{ - c _ { 1 } n _ { 3 } u ^ { 2 } \} , } \end{array}\tag{F.8}
$$

where

$$
{ \bar { R } } _ { n _ { 3 } } ( \theta ) : = { \frac { 1 } { n _ { 3 } } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \mathbb { E } \left[ \{ Y _ { t + 1 } - { \check { f } } ^ { t } ( X _ { t + 1 } ; \theta ) \} ^ { 2 } \mid { \mathcal { F } } _ { t } \right] .
$$

Since $\mathbb { E } ( \varepsilon _ { t + 1 } \mid X _ { t + 1 } ) = 0$ and $\varepsilon _ { t + 1 }$ is independent of $\mathcal { F } _ { t }$ , we have

$$
\begin{array} { r } { \bar { R } _ { n _ { 3 } } ( \theta ) = \mathcal { R } _ { n _ { 3 } } ( \theta ) + \mathbb { E } \varepsilon _ { 1 } ^ { 2 } . } \end{array}\tag{F.9}
$$

We next pass from a fixed θ to the full parameter space. Let $\Theta ( \eta ) = \{ \theta ^ { ( 1 ) } , \dots , \theta ^ { ( S _ { \eta } ) } \}$ be an η-net of $\Theta _ { M _ { R } }$ . By Assumption 3.4, the parameter space $\Theta _ { M _ { R } }$ is contained in a Euclidean box of dimension at most $M _ { R } ( d + 1 )$ and side length of order log $M _ { R }$ . Hence a standard grid construction gives, for $0 < \eta \leq 1$

$$
\log S _ { \eta } \leq C M _ { R } d \log \left( \frac { C M _ { R } d \log M _ { R } } { \eta } \right) .
$$

Therefore, by the union bound applying to (F.8), we have

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq s \leq S _ { \eta } } | R _ { n _ { 3 } } ( \theta ^ { ( s ) } ) - \bar { R } _ { n _ { 3 } } ( \theta ^ { ( s ) } ) | \geq u \right) \leq 2 \exp \left\{ C M _ { R } d \log \left( \frac { C M _ { R } d \log M _ { R } } { \eta } \right) - c _ { 1 } n _ { 3 } u ^ { 2 } \right\} .\tag{F.10}
$$

It remains to control the discretization error. Using the Lipschitz property of the softmax map and (F.7), we have

$$
| \check { f } ^ { t } ( x ; \theta ) - \check { f } ^ { t } ( x ; \theta ^ { \prime } ) | \leq L _ { M _ { R } } \| \theta - \theta ^ { \prime } \| _ { 2 } , \qquad L _ { M _ { R } } \leq C M _ { R } ^ { 3 / 2 } \sqrt { d } ,
$$

for all $t , x ,$ and $\theta , \theta ^ { \prime } \in \Theta _ { M _ { R } }$ . For each $\theta \in \Theta _ { M _ { R } } .$ , choose $\tau ( \theta ) \in \Theta ( \eta )$ such that $\| \theta - \pi ( \theta ) \| _ { 2 } \leq$ $\eta .$ . Then by boundness, (F.9) and the fact that $a ^ { 2 } - b ^ { 2 } \leq ( a - b ) ( a + b )$ , we have

$$
| \bar { R } _ { n _ { 3 } } ( \theta ) - \bar { R } _ { n _ { 3 } } ( \pi ( \theta ) ) | \le C L _ { M _ { R } } \eta .
$$

Similarly,

$$
| R _ { n _ { 3 } } ( \theta ) - R _ { n _ { 3 } } ( \pi ( \theta ) ) | \leq L _ { M _ { R } } \eta \cdot \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \{ 4 B _ { 0 } + 2 | \varepsilon _ { t + 1 } | \} .
$$

Since $\varepsilon _ { i }$ is sub-Gaussian, $\left| \varepsilon _ { i } \right|$ is sub-exponential. Hence Bernstein’s inequality implies that there exist constants $c _ { \varepsilon } , C _ { \varepsilon } > 0$ such that

$$
{ \frac { 1 } { n _ { 3 } } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } | \varepsilon _ { t + 1 } | \leq C _ { \varepsilon }
$$

with probability at least $1 - \exp \{ - c _ { \varepsilon } n _ { 3 } \}$ . On this event,

$$
\operatorname* { s u p } _ { \theta \in \Theta _ { M _ { R } } } | R _ { n _ { 3 } } ( \theta ) - \bar { R } _ { n _ { 3 } } ( \theta ) | \leq \operatorname* { m a x } _ { 1 \leq s \leq S _ { \eta } } | R _ { n _ { 3 } } ( \theta ^ { ( s ) } ) - \bar { R } _ { n _ { 3 } } ( \theta ^ { ( s ) } ) | + C L _ { M _ { R } } \eta .
$$

Set

$$
u = K _ { 0 } \sqrt { \frac { M _ { R } d \log ( M _ { R } d n _ { 3 } ) } { n _ { 3 } } } , \qquad \eta = \operatorname * { m i n } \left\{ 1 , \frac { u } { L _ { M _ { R } } } \right\} ,
$$

into (F.10), where $K _ { 0 } > 0$ is suficiently large. If $u > 1$ , the desired bound is trivial after enlarging the constant, since both $f _ { 0 }$ and $\check { f } ^ { t } ( \cdot ; \theta )$ are uniformly bounded. Thus assume $u \leq 1$ . Since $L _ { M _ { R } }$ is polynomial in $M _ { R }$ up to logarithmic factors,

$$
\log \left( \frac { C M _ { R } d \log M _ { R } } { \eta } \right) \stackrel { } { \ \sim } \log ( M _ { R } d n _ { 3 } ) .
$$

Choosing $K _ { 0 }$ suficiently large and combining the preceding bounds gives

$$
\operatorname* { s u p } _ { \theta \in \Theta _ { M _ { R } } } | R _ { n _ { 3 } } ( \theta ) - \bar { R } _ { n _ { 3 } } ( \theta ) | \leq C u
$$

with probability at least $1 - n _ { 3 } ^ { - c }$

Finally, let $\begin{array} { r } { \theta _ { n _ { 3 } } ^ { * } \in \arg \operatorname* { m i n } _ { \theta \in \Theta _ { M _ { R } } } \bar { R } _ { n _ { 3 } } ( \theta ) } \end{array}$ . By empirical optimality, $R _ { n _ { 3 } } ( \hat { \theta } _ { \mathrm { E R M } } ) \leq R _ { n _ { 3 } } ( \theta _ { n _ { 3 } } ^ { * } )$ Therefore,

$$
\bar { R } _ { n _ { 3 } } ( \hat { \theta } _ { \mathrm { E R M } } ) - \bar { R } _ { n _ { 3 } } ( \theta _ { n _ { 3 } } ^ { * } ) \leq 2 \operatorname* { s u p } _ { \theta \in \Theta _ { M _ { R } } } | R _ { n _ { 3 } } ( \theta ) - \bar { R } _ { n _ { 3 } } ( \theta ) | \leq C \sqrt { \frac { M _ { R } d \log ( M _ { R } d n _ { 3 } ) } { n _ { 3 } } } .
$$

By (F.9), this is exactly (F.2).

We now translate the random evolving-expert oracle into the population-expert oracle. Fix any $\theta \in \Theta _ { M _ { R } }$ . By Assumption 3.5 and the same triangle-inequality calculation as in the proof of Theorem 3.8,

$$
\mathbb { E } \operatorname* { s u p } _ { \theta \in \Theta _ { M _ { R } } } \left. \check { f } ^ { t } ( \cdot ; \theta ) - f ^ { * } ( \cdot ; \theta ) \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C M _ { R } a _ { t } ^ { 2 } .
$$

Averaging over t gives

$$
\mathbb { E } \operatorname* { s u p } _ { \theta \in \Theta _ { M _ { R } } } \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \Vert \check { f } ^ { t } ( \cdot ; \theta ) - f ^ { * } ( \cdot ; \theta ) \Vert _ { L _ { 2 } \left( P _ { X } \right) } ^ { 2 } \leq C M _ { R } \bar { a } _ { n _ { 3 } } ^ { 2 } .
$$

Consequently,

$$
\mathbb { E } \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \mathcal { R } _ { n _ { 3 } } ( \theta ) \leq C \left\{ \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + M _ { R } \bar { a } _ { n _ { 3 } } ^ { 2 } \right\} .
$$

Combining this bound with (F.4), and integrating the high-probability statement over the failure event using the uniform boundedness of the risks, yields (F.5).

Finally, by convexity of the squared norm,

$$
\left. \bar { f } _ { \mathrm { E R M } } - f _ { 0 } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = \left. \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } ( \check { f } ^ { t } ( \cdot ; \hat { \theta } _ { \mathrm { E R M } } ) - f _ { 0 } ) \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 }
$$

$$
\leq \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n - 1 } \left. \check { f } ^ { t } ( \cdot ; \hat { \theta } _ { \mathrm { E R M } } ) - f _ { 0 } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Taking expectation and applying (F.5) proves (F.6).

Lemma F.2 (Concentration for predictable squared sub-Gaussian losses). Let $\{ \mathcal { G } _ { i } \} _ { i = 0 } ^ { T }$ be a filtration. Suppose $V _ { i } , i = 1 , \dots , T$ , is adapted to $\mathcal { G } _ { i }$ and, conditional on $\mathcal { G } _ { i - 1 }$ , the centered variable $V _ { i } - \mathbb { E } ( V _ { i } \mid { \mathcal { G } } _ { i - 1 } )$ is sub-Gaussian with parameter $\sigma _ { V }$ , while $\left| \mathbb { E } ( V _ { i } \mid { \mathcal { G } } _ { i - 1 } ) \right| \leq m _ { 0 }$ almost surely. Then there exists $c > 0$ , depending only on $\sigma _ { V }$ and $m _ { 0 }$ , such that for every $u > 0$

$$
\mathbb { P } \left( \left| \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \{ V _ { i } ^ { 2 } - \mathbb { E } ( V _ { i } ^ { 2 } \mid \mathcal { G } _ { i - 1 } ) \} \right| \geq u \right) \leq 2 \exp \{ - c T \operatorname* { m i n } ( u ^ { 2 } , u ) \} .
$$

In particular, for $0 < u \leq 1$ , the right-hand side is bounded by $2 \exp \{ - c T u ^ { 2 } \}$

Proof of Lemma F.2. Write $\mu _ { i } \ : = \ \mathbb { E } ( V _ { i } \ | \ Q _ { i - 1 } )$ and $W _ { i } : = V _ { i } - \mu _ { i }$ . Then $\{ W _ { i } \}$ is a martingale-diference sequence and $W _ { i }$ is conditionally sub-Gaussian with parameter $\sigma _ { V }$ . Conditional sub-Gaussianity implies that $W _ { i } ^ { 2 } { - } \mathbb { E } ( W _ { i } ^ { 2 } \mid { \mathcal { G } } _ { i - 1 } )$ is conditionally sub-exponential: there exist constants $\nu , \alpha > 0$ , depending only on $\sigma _ { V } ,$ such that

$$
{ \mathbb E } \left[ \exp \{ \lambda ( W _ { i } ^ { 2 } - { \mathbb E } ( W _ { i } ^ { 2 } \mid { \mathcal G } _ { i - 1 } ) ) \} \mid { \mathcal G } _ { i - 1 } \right] \leq \exp \left\{ \frac { \nu ^ { 2 } \lambda ^ { 2 } } { 2 } \right\} , \qquad | \lambda | \leq \frac { 1 } { \alpha } .
$$

By the Bernstein inequality for martingale diferences with conditional sub-exponential increments,

$$
\mathbb { P } \left( \left| \frac { 1 } { n _ { 3 } } \sum _ { i = 1 } ^ { T } \{ W _ { i } ^ { 2 } - \mathbb { E } ( W _ { i } ^ { 2 } \mid \mathcal { G } _ { i - 1 } ) \} \right| \geq s \right) \leq 2 \exp \{ - c _ { 1 } T \operatorname* { m i n } ( s ^ { 2 } , s ) \} .\tag{F.11}
$$

Next,

$$
V _ { i } ^ { 2 } - \mathbb { E } ( V _ { i } ^ { 2 } \mid \mathcal { G } _ { i - 1 } ) = W _ { i } ^ { 2 } - \mathbb { E } ( W _ { i } ^ { 2 } \mid \mathcal { G } _ { i - 1 } ) + 2 \mu _ { i } W _ { i } .
$$

Since $| \mu _ { i } | \le m _ { 0 }$ and $W _ { i }$ is a conditionally sub-Gaussian martingale diference, Azuma–

Hoefding’s inequality for sub-Gaussian martingale diferences gives

$$
\mathbb { P } \left( \left| \frac { 1 } { n _ { 3 } } \sum _ { i = 1 } ^ { T } \mu _ { i } W _ { i } \right| \geq r \right) \leq 2 \exp \{ - c _ { 2 } T r ^ { 2 } \} ,\tag{F.12}
$$

where $c _ { 2 }$ depends only on $\sigma _ { V }$ and $m _ { 0 }$ . Combining (F.11) and (F.12) with $s = u / 2$ and $r = u / 4$ gives

$$
\mathbb { P } \left( \left| \frac { 1 } { n _ { 3 } } \sum _ { i = 1 } ^ { T } \{ V _ { i } ^ { 2 } - \mathbb { E } ( V _ { i } ^ { 2 } \mid \mathcal { G } _ { i - 1 } ) \} \right| \geq u \right) \leq 2 \exp \{ - c T \operatorname* { m i n } ( u ^ { 2 } , u ) \}
$$

after decreasing c if necessary. The final statement follows because min $( u ^ { 2 } , u ) = u ^ { 2 }$ for $0 < u \leq 1$ □

## F.2 Details for Empirical Parameter Estimation Using Cross-Entropy

This subsection gives the details for the fixed- $M _ { R }$ implementation described in the main text. Throughout this subsection, $M _ { R }$ and $M _ { S }$ are treated as fixed, and constants denoted by $C _ { M _ { R } }$ may depend on $M _ { R } , M _ { S }$ , the fixed-dimensional softmax class, the compact parameter set, and the uniform boundedness constants, but not on the sample sizes.

The population projection in Algorithm 1 selects a representative element from a timeaveraged discretized MoE class. We now replace this population $L _ { 2 } ( P _ { X } )$ projection by an empirical cross-entropy minimization. Notice that the online aggregate generally cannot be written as a MoE predictor with one single averaged gate, because the aggregation weights and the experts both vary over time. We therefore fit a single softmax gate to the whole sequence of online aggregated gates.

Specifically, we choose the same η-net $\Theta ( \eta )$ of $\Theta _ { M _ { R } }$ as in step (a) of Algorithm 1. Then we randomly split the data into four subsamples, $Z ^ { ( 1 ) } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n _ { 1 } } , Z ^ { ( 2 ) } =$ $\{ ( X _ { i } , Y _ { i } ) \} _ { i = n _ { 1 } + 1 } ^ { n _ { 1 } + n _ { 2 } } , \ Z ^ { ( 3 ) } \ = \ \{ ( X _ { i } , Y _ { i } ) \} _ { i = n _ { 1 } + n _ { 2 } + 1 } ^ { n _ { 1 } + n _ { 2 } + n _ { 3 } }$ and ${ \cal Z } ^ { ( 4 ) } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = n _ { 1 } + n _ { 2 } + n _ { 3 } + 1 } ^ { n } ,$ and we denote $n _ { 4 } = n - n _ { 1 } - n _ { 2 } - n _ { 3 }$ . For notational simplicity, denote $n _ { \mathrm { o n } } = n _ { 1 } + n _ { 2 } + n _ { 3 }$ . We use steps (c), (d) and (e) in Algorithm 1 on $Z ^ { ( 1 ) } , Z ^ { ( 2 ) }$ and $Z ^ { ( 3 ) }$ respectively, to obtain weights $\{ \hat { w } _ { s , t } \} _ { s = 1 } ^ { S _ { \eta } }$ over a discretized softmax net $\Theta ( \eta ) = \{ \theta ^ { ( 1 ) } , \dots , \theta ^ { ( S _ { \eta } ) } \}$ for $n _ { 1 } + n _ { 2 } \le t \le n _ { \mathrm { o n } } - 1$ ， and online aggregated estimators $\{ \tilde { f } ^ { t } ( x ) \} _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 }$ . Define the time-dependent aggregated

gate

$$
\tilde { g } _ { m , t } ( \boldsymbol { x } ) : = \sum _ { s = 1 } ^ { S _ { \eta } } \hat { w } _ { s , t } g _ { m } ^ { \mathrm { s o f t } } ( \boldsymbol { x } ; \theta ^ { ( s ) } ) , \qquad m \in [ M _ { R } ] , \quad n _ { 1 } + n _ { 2 } \leq t \leq n _ { \mathrm { o n } } - 1 .
$$

For any $\theta \in \Theta _ { M _ { R } }$ , recall the corresponding one-step-ahead candidate predictor

$$
\check { f } ^ { t } ( x ; \theta ) = \sum _ { \ell = 1 } ^ { M _ { S } } \hat { f } _ { S , \ell } ^ { t } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) \hat { f } _ { R , m } ^ { t } ( x ) .
$$

The time-averaged online aggregate and the time-averaged candidate predictor are

$$
\tilde { f } _ { 0 } ( x ) : = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \tilde { f } ^ { t } ( x ) , \qquad \bar { f } ^ { n _ { 3 } } ( x ; \theta ) : = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \check { f } ^ { t } ( x ; \theta ) .\tag{F.13}
$$

Equivalently,

$$
\bar { f } ^ { n _ { 3 } } ( x ; \theta ) = \sum _ { \ell = 1 } ^ { M _ { S } } \bar { f } _ { S , \ell } ^ { n _ { 3 } } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) \bar { f } _ { R , m } ^ { n _ { 3 } } ( x ) ,\tag{F.14}
$$

where

$$
\bar { f } _ { S , \ell } ^ { n _ { 3 } } : = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \hat { f } _ { S , \ell } ^ { t } , \qquad \bar { f } _ { R , m } ^ { n _ { 3 } } : = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \hat { f } _ { R , m } ^ { t } .
$$

The online aggregate $\tilde { f } _ { 0 }$ , however, does not generally admit the same representation with a single averaged gate, because

$$
\frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \tilde { g } _ { m , t } ( x ) \hat { f } _ { R , m } ^ { t } ( x ) \neq \left\{ \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \tilde { g } _ { m , t } ( x ) \right\} \left\{ \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \hat { f } _ { R , m } ^ { t } ( x ) \right\}
$$

in general.

Define the empirical cross-entropy objective

$$
\mathcal { L } _ { \mathrm { s o f t } } ^ { \mathrm { o n } } ( \theta ; Z ^ { ( 4 ) } ) : = - \frac { 1 } { n _ { 4 } n _ { 3 } } \sum _ { \substack { i = n _ { \mathrm { o n } } + 1 } } ^ { n } \sum _ { \substack { t = n _ { 1 } + n _ { 2 } } } ^ { n _ { \mathrm { o n } } - 1 } \sum _ { m = 1 } ^ { M _ { R } } \tilde { g } _ { m , t } ( X _ { i } ) \log g _ { m } ^ { \mathrm { s o f t } } ( X _ { i } ; \theta ) .\tag{F.15}
$$

Thus cross-entropy implementation estimates the softmax weights by solving

$$
\hat { \theta } _ { \mathrm { c e } } \in \operatorname * { a r g m i n } _ { \theta \in \Theta _ { M _ { R } } } \mathcal { L } _ { \mathrm { s o f t } } ^ { \mathrm { o n } } ( \theta ; Z ^ { ( 4 ) } ) .\tag{F.16}
$$

One can verify that with softmax gating, the above objective is convex in $\theta ,$ thus this optimization problem is numerically feasible. The final estimator is

$$
\hat { f } _ { 0 , \mathrm { c e } } ( x ) : = \bar { f } ^ { n _ { 3 } } ( x ; \hat { \theta } _ { \mathrm { c e } } ) = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \check { f } ^ { t } ( x ; \hat { \theta } _ { \mathrm { c e } } ) .\tag{F.17}
$$

The full algorithm is summarized in Algorithm F.2.

Algorithm F.2: Discritized Aggregation with Empirical Parameter Estimation   
using Cross-Entropy   
Input: Pre-trained shared experts $\{ \hat { f } _ { S , \ell } ^ { N } \} _ { \ell = 1 } ^ { M _ { S } }$ and routed experts $\{ \hat { f } _ { R , m } ^ { N } \} _ { m = 1 } ^ { M _ { R } }$ ;   
softmax-router family $g ^ { \mathrm { s o f t } } ( \cdot ; \theta )$ with $ { \boldsymbol { \theta } } \in \Theta _ { M _ { R } } ;$ data $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n } .$   
Output: Estimated gating parameter $\hat { \theta }$ and final estimator $\hat { f } _ { 0 } ( x )$   
(a) Choose an $\eta -$ net of the normalized bounded gating parameter space $\Theta _ { M _ { R } } ,$ denoted   
by $\Theta ( \eta )$ , similarly as in Algorithm 1. Write $\bar { \Theta } ( \eta ) = \{ \theta ^ { ( 1 ) } , \dots , \theta ^ { ( S _ { \eta } ) } \}$ where   
$S _ { \eta } = | \Theta ( \eta ) |$ , which induce the discretized routers $g ( \cdot ; \partial ^ { ( 1 ) } ) , \dots , g ( \cdot ; \overset { \cdot } { \theta } ^ { ( S _ { \eta } ) } )$   
(b) Randomly partition the data into three subsamples, $Z ^ { ( 1 ) } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n _ { 1 } }$   
$Z ^ { ( 2 ) } = \{ ( \bar { X _ { i } } , Y _ { i } ) \} _ { i = n _ { 1 } + 1 } ^ { n _ { 1 } + n _ { 2 } } , Z ^ { ( 3 ) } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = n _ { 1 } + n _ { 2 } + 1 } ^ { n _ { \mathrm { o n } } }$ and $\overline { { { Z } } } ^ { ( 4 ) } = \{ ( X _ { i } , \overline { { { Y _ { i } } } } ) \} _ { i = n _ { \mathrm { o n } } + 1 } ^ { n _ { \mathrm { o n } } + n _ { 4 } }$   
where $n = n _ { \mathrm { o n } } + n _ { 4 }$ . The first three subsamples are used for burn-in, calibration and   
aggregation, and the last subsample is used for empirical parameter estimation.   
(c) Use steps (c), (d) and $\mathrm { ( e ) }$ in Algorithm 1 on $Z ^ { ( 1 ) } , Z ^ { ( 2 ) }$ and $Z ^ { ( 3 ) }$ respectively, to   
obtain weights $\{ \hat { w } _ { s , t } \} _ { s = 1 } ^ { S _ { \eta } }$ for $n _ { 1 } + n _ { 2 } \le t \le n _ { \mathrm { o n } } - 1$ . Denote   
$\tilde { g } _ { m , t } ( \boldsymbol { x } ) : = \sum _ { s = 1 } ^ { S _ { \eta } } \hat { w } _ { s , t } g _ { m } ^ { \mathrm { s o f t } } ( \boldsymbol { x } ; \theta ^ { ( s ) } ) , \qquad m \in [ M _ { R } ] , \quad n _ { 1 } + n _ { 2 } \leq t \leq n _ { \mathrm { o n } } - 1 .$   
(d) Solve the convex optimization $\begin{array} { r } { \hat { \theta } _ { \mathrm { c e } } = \arg \operatorname* { m i n } _ { \theta \in \Theta _ { M _ { R } } } \mathcal { L } _ { \mathrm { s o f t } } ^ { \mathrm { o n } } ( \theta ; Z ^ { ( 4 ) } ) } \end{array}$ , where   
$\mathcal { L } _ { \mathrm { s o f t } } ^ { \mathrm { o n } } ( \theta ; Z ^ { ( 4 ) } ) : = - \frac { 1 } { n _ { 4 } n _ { 3 } } \sum _ { \substack { i = n _ { \mathrm { o n } } + 1 } } ^ { n } \sum _ { \substack { t = n _ { 1 } + n _ { 2 } m = 1 } } ^ { n _ { \mathrm { o n } } - 1 } \sum _ { \substack { m = 1 } } ^ { M _ { R } } \tilde { g } _ { m , t } ( X _ { i } ) \log g _ { m } ^ { \mathrm { s o f t } } ( X _ { i } ; \theta ) .$   
Obtain the resulting the final output $\hat { f } _ { 0 , \mathrm { c e } } ( x ) = \bar { f } ^ { n _ { 3 } } ( x ; \hat { \theta } _ { \mathrm { c e } } )$ as in (F.17).

For the cross-entropy objective, its population counterpart is

$$
\mathcal { L } _ { \mathrm { s o f t } } ^ { \mathrm { o n } } ( \theta ) : = - \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \int _ { \mathcal { X } } \sum _ { m = 1 } ^ { M _ { R } } \tilde { g } _ { m , t } ( x ) \log g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) d P _ { X } ( x ) .
$$

The compactness, boundedness, and error-distribution assumptions used in the main online oracle inequality are kept in force here and are not repeated. We impose one additional fixed- $. M _ { R }$ compatibility condition to convert time-averaged weight approximation into timeaveraged function approximation.

Assumption F.3 (Additional fixed- $M _ { R }$ cross-entropy regularity). Let

$$
\mathcal { G } _ { M _ { R } } ^ { \mathrm { s o f t } } : = \left\{ x \mapsto g ^ { \mathrm { s o f t } } ( x ; \theta ) = \left( g _ { 1 } ^ { \mathrm { s o f t } } ( x ; \theta ) , \ldots , g _ { M _ { R } } ^ { \mathrm { s o f t } } ( x ; \theta ) \right) ^ { \top } : \theta \in \Theta _ { M _ { R } } \right\}
$$

be the softmax gating class. Define the online softmax-diference class $\Delta _ { M _ { R } } ^ { \mathrm { o n } }$ as the collection of all time-indexed vector-valued functions $\delta = ( \delta _ { t } ) _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 }$ , where

$$
\delta _ { t } ( \boldsymbol { x } ) = \sum _ { s = 1 } ^ { S _ { \eta } } p _ { s , t } g ^ { \mathrm { s o f t } } ( \boldsymbol { x } ; \boldsymbol { \theta } ^ { ( s ) } ) - g ^ { \mathrm { s o f t } } ( \boldsymbol { x } ; \boldsymbol { \theta } ) , \qquad p _ { s , t } \in [ 0 , 1 / 2 ] , \qquad \sum _ { s = 1 } ^ { S _ { \eta } } p _ { s , t } = 1 , \qquad \boldsymbol { \theta } \in \Theta _ { M _ { R } } .
$$

The routed experts satisfy the online compatibility condition that, for some $c _ { M _ { R } } > 0$ and all $\delta \in \Delta _ { M _ { R } } ^ { \mathrm { o n } }$

$$
\frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \left\| \sum _ { m = 1 } ^ { M _ { R } } \delta _ { m , t } \hat { f } _ { R , m } ^ { t } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \geq c _ { M _ { R } } \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \sum _ { m = 1 } ^ { M _ { R } } \left\| \delta _ { m , t } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .\tag{F.18}
$$

Remark F.4. Assumption F.3 can be viewed as a positive minimum singular-value condition for a sequence of bounded linear operator. For each time $t ,$ let

$$
T _ { t } : \{ L _ { 2 } ( P _ { X } ) \} ^ { M _ { R } } \to L _ { 2 } ( P _ { X } ) , \qquad T _ { t } ( \delta _ { t } ) ( x ) : = \sum _ { m = 1 } ^ { M _ { R } } \delta _ { m , t } ( x ) \hat { f } _ { R , m } ^ { t } ( x ) .
$$

Since the routed experts are uniformly bounded, $T _ { t }$ is bounded. The condition can be viewed as requiring the average operator $( T _ { t } ) _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { 0 \mathrm { n } } - 1 }$ to have a positive minimum “singular value” on the class of softmax diferences generated by the online aggregated gates and a single candidate softmax gate. It prevents diferent gate discrepancies from being invisible after multiplication by the evolving routed experts.

As a concrete intuition, suppose $\hat { f } _ { R , m } ^ { t } ( x ) = \hat { f } _ { R , m } ( x ) : = \operatorname* { m a x } \{ a _ { m } ^ { \top } x + b _ { m } , 0 \}$ are nondegenerate ReLU experts and does not vary with respect to t. Then all the above defined operators $T _ { t }$ are identical to a specific $T _ { \hat { f } }$ . The kink hyperplanes of these ReLU functions make $T _ { \hat { f } }$ injective on analytic coeficient functions under suitable nondegeneracy conditions. If one restricts to a truncated class $\Delta _ { M _ { R } , N } ^ { \mathrm { o n } }$ with $L \leq N$ , then compactness of the parameter space and continuity of the softmax map imply a positive constant $c _ { M _ { R } , N }$ , provided that the normalized class is separated from the kernel of $T _ { \hat { f } }$ . More generally, the same conclusion holds whenever the normalized softmax-diference class is contained in a compact subset of analytic functions that does not intersect the kernel.

We first record the elementary comparison between cross-entropy and squared weight distance that will be used in both empirical and population forms.

Lemma F.5 (KL–L<sub>2</sub> comparison for softmax weights). Let $p = \left( p _ { 1 } , \ldots , p _ { M _ { R } } \right)$ and $q =$ $( q _ { 1 } , \dots , q _ { M _ { R } } )$ be probability vectors. If $q _ { m } \ge \underline { { s } } _ { M _ { R } } > 0$ for all $m \in [ M _ { R } ]$ , then

$$
\frac 1 2 \| p - q \| _ { 2 } ^ { 2 } \leq \sum _ { m = 1 } ^ { M _ { R } } p _ { m } \log \frac { p _ { m } } { q _ { m } } \leq \frac { 1 } { \underline { { s } } _ { M _ { R } } } \| p - q \| _ { 2 } ^ { 2 } .
$$

Consequently, the same inequalities hold after averaging over any empirical distribution or integrating with respect to $P _ { X }$

Proof of Lemma F.5. The lower bound follows from Pinsker’s inequality:

$$
\sum _ { m = 1 } ^ { M _ { R } } p _ { m } \log { \frac { p _ { m } } { q _ { m } } } \geq \frac 1 2 \| p - q \| _ { 1 } ^ { 2 } \geq \frac 1 2 \| p - q \| _ { 2 } ^ { 2 } .
$$

For the upper bound, use the standard inequality

$$
\sum _ { m = 1 } ^ { M _ { R } } p _ { m } \log { \frac { p _ { m } } { q _ { m } } } \leq \sum _ { m = 1 } ^ { M _ { R } } { \frac { ( p _ { m } - q _ { m } ) ^ { 2 } } { q _ { m } } } \leq { \frac { 1 } { \underline { { s } } _ { M _ { R } } } } \| p - q \| _ { 2 } ^ { 2 } .
$$

Averaging or integrating these pointwise inequalities gives the empirical and population versions. □

Define the empirical and population time-expanded squared weight errors by

$$
\begin{array} { r l } & { R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ) : = \displaystyle \frac { 1 } { n _ { 4 } n _ { 3 } } \sum _ { i = n _ { \mathrm { o n } } + 1 } ^ { n } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \sum _ { m = 1 } ^ { M _ { R } } \Big \{ \tilde { g } _ { m , t } ( X _ { i } ) - g _ { m } ^ { \mathrm { s o f t } } ( X _ { i } ; \theta ) \Big \} ^ { 2 } , } \\ & { R _ { w } ^ { \mathrm { o n } } ( \theta ) : = \displaystyle \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \sum _ { m = 1 } ^ { M _ { R } } \| \tilde { g } _ { m , t } - g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } . } \end{array}
$$

Lemma F.6 (Time-averaged weight estimation via cross-entropy). Suppose the assumptions of the main softmax oracle inequality hold and assume $n _ { 1 } \asymp n _ { 2 } \asymp n _ { 3 } \asymp n _ { 4 } \asymp n$ . Let $\mathfrak { F } _ { \mathrm { o n } }$ denote the σ-field generated by by the data and algorithmic randomness used before the cross-entropy step. Then, conditional on $\mathfrak { F } _ { \mathrm { o n } }$ , for fixed $M _ { R }$

$$
\mathbb { E } \left[ R _ { w } ^ { \mathrm { o n } } ( \hat { \theta } _ { \mathrm { c e } } ) \mid \mathfrak { F } _ { \mathrm { o n } } \right] \leq C _ { M _ { R } } \left\{ \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } R _ { w } ^ { \mathrm { o n } } ( \theta ) + \frac { \log n } { n } \right\} .\tag{F.19}
$$

Proof of Lemma F.6. Throughout the proof, condition on $\mathfrak { F } _ { \mathrm { o n } }$ . Under this conditioning, $\tilde { g } _ { m , t } , \hat { f } _ { S , \ell } ^ { t } ,$ , and $\hat { f } _ { R , m } ^ { t }$ are fixed functions, while the covariates in the third split sample remain i.i.d. from $P _ { X }$ . Constants denoted by $C _ { M _ { R } }$ may change from line to line and may absorb additional fixed constants depending only on the fixed- $M _ { R }$ softmax class, the compact parameter set, and the sample-splitting proportions.

Since the compactness assumption is in force and $M _ { R }$ is fixed, the softmax probabilities are uniformly bounded away from zero on $\Theta _ { M _ { R } } ;$ : there exists $\underline { { s } } _ { M _ { R } } > 0$ such that

$$
\begin{array} { r } { g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) \geq \underline { { s } } _ { M _ { R } } , \qquad x \in \mathcal { X } , \qquad \theta \in \Theta _ { M _ { R } } , \qquad m \in [ M _ { R } ] . } \end{array}
$$

Define the empirical time-expanded KL loss

$$
D _ { n _ { 4 } } ^ { \mathrm { o n } } ( \tilde { g } , g _ { \theta } ) : = \frac { 1 } { n _ { 4 } n _ { 3 } } \sum _ { i = n _ { \mathrm { o n } } + 1 } ^ { n } \sum _ { \substack { t = n _ { 1 } + n _ { 2 } } } ^ { n _ { \mathrm { o n } } - 1 } \sum _ { m = 1 } ^ { M _ { R } } \tilde { g } _ { m , t } ( X _ { i } ) \log \frac { \tilde { g } _ { m , t } ( X _ { i } ) } { g _ { m } ^ { \mathrm { s o f t } } ( X _ { i } ; \theta ) } ,
$$

with the convention 0 log $0 = 0$ . The empirical cross-entropy objective in (F.15) difers from $D _ { n _ { 4 } } ^ { \mathrm { o n } } ( \tilde { g } , g _ { \theta } )$ by a term independent of θ. Therefore, by the definition of $\widehat { \theta } _ { \mathrm { c e } }$

$$
D _ { n _ { 4 } } ^ { \mathrm { o n } } ( \tilde { g } , g _ { \hat { \theta } _ { \mathrm { c e } } } ) \leq D _ { n _ { 4 } } ^ { \mathrm { o n } } ( \tilde { g } , g _ { \theta } ) .
$$

Applying Lemma F.5 to both sides gives the empirical $L _ { 2 }$ oracle inequality

$$
R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \hat { \theta } _ { \mathrm { c e } } ) \leq C _ { M _ { R } } R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ) , \qquad \theta \in \Theta _ { M _ { R } } .
$$

Taking the infimum over θ yields

$$
R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \hat { \theta } _ { \mathrm { c e } } ) \leq C _ { M _ { R } } \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ) .
$$

It remains to transfer this empirical inequality to its population version. For $\theta \in \Theta _ { M _ { R } } ,$ write

$$
\ell _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) : = \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \sum _ { m = 1 } ^ { M _ { R } } \left( \tilde { g } _ { m , t } ( \boldsymbol { x } ) - g _ { m } ^ { \mathrm { s o f t } } ( \boldsymbol { x } ; \boldsymbol { \theta } ) \right) ^ { 2 } .
$$

Then

$$
R _ { w } ^ { \mathrm { o n } } ( \theta ) = \mathbb { E } _ { P _ { X } } \ell _ { \theta } ( X ) , \qquad R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ) = \frac { 1 } { n _ { 4 } } \sum _ { i = n _ { \mathrm { o n } } + 1 } ^ { n } \ell _ { \theta } ( X _ { i } ) .
$$

Since both $\tilde { g } _ { t } ( x )$ and $g ^ { \mathrm { s o f t } } ( x ; \theta )$ are probability vectors, $0 \leq \ell _ { \theta } ( x ) \leq 4$ , and hence

$$
\operatorname { V a r } \{ \ell _ { \theta } ( X ) \} \leq \mathbb { E } \ell _ { \theta } ^ { 2 } ( X ) \leq 4 R _ { w } ^ { \mathrm { o n } } ( \theta ) .
$$

Bernstein’s inequality then gives, for every fixed θ and every $t > 0$

$$
\mathbb { P } \bigg ( | R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ) - R _ { w } ^ { \mathrm { o n } } ( \theta ) | \geq \frac { 1 } { 2 } R _ { w } ^ { \mathrm { o n } } ( \theta ) + u \bigg | \mathfrak { F } _ { \mathrm { o n } } \bigg ) \leq 2 \exp \{ - c n _ { 4 } u \} ,
$$

where $c > 0$ is an absolute constant.

Next, since $M _ { R }$ is fixed, $\Theta _ { M _ { R } }$ is finite-dimensional and compact. Moreover, the map $\theta \mapsto \ell _ { \theta } ( x )$ is uniformly Lipschitz on $\Theta _ { M _ { R } } ;$ that is, for some constant $L _ { M _ { R } } < \infty$

$$
\operatorname* { s u p } _ { x \in \mathcal { X } } | \ell _ { \theta } ( x ) - \ell _ { \theta ^ { \prime } } ( x ) | \leq L _ { M _ { R } } \| \theta - \theta ^ { \prime } \| _ { 2 } .
$$

Let $\Theta ( \eta ) = \{ \theta ^ { ( 1 ) } , \dots , \theta ^ { ( S _ { \eta } ) } \}$ be an η-net of $\Theta _ { M _ { R } }$ satisfying

$$
\log S _ { \eta } \leq C _ { M _ { R } } \log \frac { C _ { M _ { R } } } { \eta } .
$$

Therefore, by the union bound,

$$
\mathbb { P } ( \operatorname* { m a x } _ { 1 \leq j \leq S _ { \eta } } \{ | R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ^ { ( j ) } ) - R _ { w } ^ { \mathrm { o n } } ( \theta ^ { ( j ) } ) | - \frac { 1 } { 2 } R _ { w } ^ { \mathrm { o n } } ( \theta ^ { ( j ) } ) \} \geq u  \mathfrak { F } _ { \mathrm { o n } } ) \leq 2 \exp \{ C _ { M _ { R } } \log \frac { C _ { M _ { R } } } { \eta } - c n _ { 4 } u \} .
$$

Take $u = C _ { M _ { R } }$ log n/n with $C _ { M _ { R } }$ suficiently large, and set $\eta = u / ( 4 L _ { M _ { R } } )$ . Then

$$
\log \frac { C _ { M _ { R } } } { \eta } = \log \frac { 4 C _ { M _ { R } } L _ { M _ { R } } } { u } \leq C _ { M _ { R } } \log n .
$$

Since $n _ { 4 } \ \asymp \ n$ , the preceding probability is bounded by $C _ { M _ { R } } n ^ { - 2 }$ after enlarging $C _ { M _ { R } }$ if necessary. Hence, with conditional probability at least $1 - C _ { M _ { R } } n ^ { - 2 }$

$$
| R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ^ { ( j ) } ) - R _ { w } ^ { \mathrm { o n } } ( \theta ^ { ( j ) } ) | \leq \frac { 1 } { 2 } R _ { w } ^ { \mathrm { o n } } ( \theta ^ { ( j ) } ) + C _ { M _ { R } } \frac { \log n } { n } , \qquad 1 \leq j \leq S _ { \eta } .
$$

For an arbitrary $\theta \in \Theta _ { M _ { R } }$ , choose $\theta ^ { ( j ) } \in \Theta ( \eta )$ such that $\lVert \theta - \theta ^ { ( j ) } \rVert _ { 2 } \leq \eta$ . The Lipschitz bound implies

$$
| R _ { w } ^ { \mathrm { o n } } ( \theta ) - R _ { w } ^ { \mathrm { o n } } ( \theta ^ { ( j ) } ) | + | R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ) - R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ^ { ( j ) } ) | \le 2 L _ { M _ { R } } \eta \le \frac { 1 } { 2 } C _ { M _ { R } } \frac { \log n } { n } .
$$

Also $R _ { w } ^ { \mathrm { o n } } ( \theta ^ { ( j ) } ) \leq R _ { w } ^ { \mathrm { o n } } ( \theta ) + L _ { M _ { R } } \eta$ . Combining these bounds gives the uniform comparison

$$
| R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ) - R _ { w } ^ { \mathrm { o n } } ( \theta ) | \leq \frac { 1 } { 2 } R _ { w } ^ { \mathrm { o n } } ( \theta ) + C _ { M _ { R } } \frac { \log n } { n } , \qquad \forall \theta \in \Theta _ { M _ { R } } .
$$

Consequently, on this event,

$$
R _ { w } ^ { \mathrm { o n } } ( \theta ) \leq 2 R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ) + C _ { M _ { R } } \frac { \log n } { n } , \qquad R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ) \leq \frac 3 2 R _ { w } ^ { \mathrm { o n } } ( \theta ) + C _ { M _ { R } } \frac { \log n } { n } ,
$$

uniformly over $\theta \in \Theta _ { M _ { R } }$ . Using these two inequalities together with the empirical oracle inequality, we obtain

$$
R _ { w } ^ { \mathrm { o n } } ( \hat { \theta } _ { \mathrm { c e } } ) \leq 2 R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \hat { \theta } _ { \mathrm { c e } } ) + C _ { M _ { R } } \frac { \log n } { n }
$$

$$
\begin{array} { l } { \displaystyle \le C _ { { M _ { R } } } \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } R _ { n _ { 4 } , w } ^ { \mathrm { o n } } ( \theta ) + C _ { { M _ { R } } } \frac { \log n } { n } } \\ { \displaystyle \le C _ { { M _ { R } } } \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } R _ { w } ^ { \mathrm { o n } } ( \theta ) + C _ { { M _ { R } } } \frac { \log n } { n } . } \end{array}
$$

Since both $R _ { w } ^ { \mathrm { o n } }$ and $R _ { n _ { 4 } , w } ^ { \mathrm { o n } }$ are uniformly bounded, the complement of the high-probability event contributes only $O ( n _ { 4 } ^ { - 3 } )$ to the conditional expectation. Therefore,

$$
\mathbb { E } \left[ R _ { w } ^ { \mathrm { o n } } ( \hat { \theta } _ { \mathrm { c e } } ) \mid \mathfrak { F } _ { \mathrm { o n } } \right] \leq C _ { M _ { R } } \left\{ \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } R _ { w } ^ { \mathrm { o n } } ( \theta ) + \frac { \log n } { n } \right\} .
$$

This proves the lemma.

Proposition F.7 (Function estimation via time-averaged cross-entropy weights). Suppose the assumptions of the main softmax oracle inequality and Assumption F.3 hold. Assume that the sample is split so that $n _ { 1 } \asymp n _ { 2 } \asymp n _ { 3 } \asymp n _ { 4 } \asymp n$ . Let $\widehat { \theta } _ { \mathrm { c e } }$ be obtained from the time-expanded cross-entropy criterion (F.16), and define $\hat { f } _ { 0 , \mathrm { c e } } ( x ) = \bar { f } ^ { n _ { 3 } } ( x ; \hat { \theta } _ { \mathrm { c e } } )$ as in (F.17). Then, for fixed $M _ { R }$ , with expectation taken over all randomness, including data and algorithmic randomness, we have

$$
\mathbb { E } \left\| \hat { f } _ { 0 , \mathrm { c e } } - f _ { 0 } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C _ { M _ { R } } \left\{ \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } + \frac { \log n } { n } \right\} .\tag{F.20}
$$

Proof of Proposition F.7. Recall the σ-field $\mathfrak { F } _ { \mathrm { o n } }$ defined in Lemma F.6. We first work conditionally on $\mathfrak { F } _ { \mathrm { o n } }$ . For each $n _ { 1 } + n _ { 2 } \le t \le n _ { \mathrm { o n } } - 1$ , the shared part cancels in the diference between $\tilde { f } ^ { t }$ and $\check { f } ^ { t } ( \cdot ; \hat { \theta } _ { \mathrm { c e } } )$ . Thus

$$
\tilde { f } ^ { t } ( x ) - \check { f } ^ { t } ( x ; \hat { \theta } _ { \mathrm { c e } } ) = \sum _ { m = 1 } ^ { M _ { R } } \left\{ \tilde { g } _ { m , t } ( x ) - g _ { m } ^ { \mathrm { s o f t } } ( x ; \hat { \theta } _ { \mathrm { c e } } ) \right\} \hat { f } _ { R , m } ^ { t } ( x ) .
$$

By uniform boundedness of the routed experts and Cauchy’s inequality,

$$
| \widetilde { f } ^ { t } ( x ) - \check { f } ^ { t } ( x ; \hat { \theta } _ { \mathrm { c e } } ) | ^ { 2 } \leq C _ { M _ { R } } \sum _ { m = 1 } ^ { M _ { R } } \left| \tilde { g } _ { m , t } ( x ) - g _ { m } ^ { \mathrm { s o f t } } ( x ; \hat { \theta } _ { \mathrm { c e } } ) \right| ^ { 2 } .
$$

Integrating with respect to $P _ { X }$ , averaging over $n _ { 1 } + n _ { 2 } \le t \le n _ { \mathrm { o n } } - 1$ , and applying

Lemma F.6 gives

$$
\mathbb { E } \left[ \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \lVert \tilde { f } ^ { t } - \tilde { f } ^ { t } ( \cdot ; \hat { \theta } _ { \mathrm { c e } } ) \rVert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \Bigg | \mathfrak { F } _ { \mathrm { o n } } \right] \leq C _ { M _ { R } } \left\{ \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } R _ { w } ^ { \mathrm { o n } } ( \theta ) + \frac { \log n } { n } \right\} .
$$

It remains to relate the weight approximation error to the corresponding prediction approximation error. For any $\theta \in \Theta _ { M _ { R } }$ , define

$$
\delta _ { m , t } ^ { \theta } ( x ) : = \widetilde { g } _ { m , t } ( x ) - g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) , \qquad m \in [ M _ { R } ] , \ n _ { 1 } + n _ { 2 } \le t \le n _ { \mathrm { o n } } - 1 .
$$

Since both $\tilde { g } _ { m , t } ( x )$ and $g ^ { \mathrm { s o f t } } ( x ; \theta )$ are probability vectors, $\textstyle \sum _ { m = 1 } ^ { M _ { R } } \delta _ { m , t } ^ { \theta } ( x )$ . Moreover, the ARM construction writes $\tilde { g } _ { m , t } ( x )$ as a finite convex combination of softmax weight vectors. For a time $t ,$ if one aggregation coeficient is larger than $1 / 2 ,$ we split it by keeping mass $1 / 2$ on the original softmax vector and assigning the excess mass to the softmax vector with the nearest parameter; by compactness of $\Theta _ { M _ { R } }$ and continuity of the softmax map, this replacement can be made arbitrarily close in $L _ { 2 } ( P _ { X } )$ and is absorbed by the closure in the definition of $\Delta _ { M _ { R } } ^ { \mathrm { o n } }$ . Hence $\delta ^ { \theta } = ( \delta _ { t } ^ { \theta } ) _ { n _ { 1 } + n _ { 2 } \leq t \leq n _ { \mathrm { o n } } - 1 }$ belongs to the softmax-diference class in Assumption F.3. Consequently,

$$
\begin{array} { r l } & { c _ { M _ { R } } R _ { w } ^ { \mathrm { o n } } ( \theta ) \leq \displaystyle \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \left\| \sum _ { m = 1 } ^ { M _ { R } } \left( \tilde { g } _ { m , t } - g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) \right) \hat { f } _ { R , m } ^ { t } \right\| _ { L _ { 2 } \left( P _ { X } \right) } ^ { 2 } } \\ & { \quad \quad \quad = \displaystyle \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \| \tilde { f } ^ { t } - \check { f } ^ { t } ( \cdot ; \theta ) \| _ { L _ { 2 } \left( P _ { X } \right) } ^ { 2 } . } \end{array}
$$

Taking the infimum over $\theta \in \Theta _ { M _ { R } }$ and absorbing $c _ { M _ { R } } ^ { - 1 }$ into ${ \cal C } _ { M _ { R } } ,$ we obtain

$$
\operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } R _ { w } ^ { \mathrm { o n } } ( \theta ) \leq C _ { M _ { R } } \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \| \tilde { f } ^ { t } - \check { f } ^ { t } ( \cdot ; \theta ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

By Jensen’s inequality, we have

$$
\lVert \tilde { f } _ { 0 } - \hat { f } _ { 0 , \mathrm { c e } } \rVert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = \left. \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \left\{ \tilde { f } ^ { t } - \check { f } ^ { t } ( \cdot ; \hat { \theta } _ { \mathrm { c e } } ) \right\} \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 }
$$

$$
\leq \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \Vert \tilde { f } ^ { t } - \check { f } ^ { t } ( \cdot ; \hat { \theta } _ { \mathrm { c e } } ) \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Taking expectations in the preceding conditional bound and using the above bounds, we obtain

$$
\mathbb { E } \| \hat { f } _ { 0 , \mathrm { c e } } - \tilde { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C _ { M _ { R } } \left( \mathbb { E } \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \| \tilde { f } ^ { t } - \check { f } ^ { t } ( \cdot ; \theta ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + \frac { \log n } { n } \right) .
$$

To control the projection term on the right-hand side, notice that

$$
\begin{array} { r l } { \displaystyle \underset { \theta \in \Theta _ { M _ { R } } } { \operatorname* { i n f } } \frac { 1 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \| \tilde { f } ^ { t } - \tilde { f } ^ { t } ( \cdot ; \theta ) \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq \frac { 2 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \| \tilde { f } ^ { t } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } & { } \\ { + \displaystyle \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \frac { 2 } { n _ { 3 } } \sum _ { t = n _ { 1 } + n _ { 2 } } ^ { n _ { \mathrm { o n } } - 1 } \| \tilde { f } ^ { t } ( \cdot ; \theta ) - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } . } & { } \end{array}
$$

Taking expectations and applying the ARM oracle bound for the online aggregated sequence $\{ \tilde { f } ^ { t } \}$ , together with the expert approximation bounds used in the proof of Theorem 3.8, we obtain

$$
\mathbb { E } \| \hat { f } _ { 0 , \mathrm { c e } } - \tilde { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C _ { M _ { R } } \bigg ( \mathfrak { A } _ { \mathrm { s o f t } } ( M _ { S } , M _ { R } ) + a _ { n _ { 2 } } ^ { 2 } + a _ { n _ { 3 } } ^ { 2 } + \frac { \log n } { n } \bigg ) .
$$

Finally, by the squared triangle inequality,

$$
\begin{array} { r } { \| \hat { f } _ { 0 , \mathrm { c e } } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq 2 \| \hat { f } _ { 0 , \mathrm { c e } } - \tilde { f } _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + 2 \| \tilde { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } , } \end{array}
$$

and by the ARM oracle bound for time-averaged $\tilde { f } _ { 0 }$ , we obtain the stated bound for $\hat { f } _ { 0 , \mathrm { c e } }$

Here we still need to confirm that under the potential adjustment for ˆw, the ARM oracle bound for $\tilde { f } _ { 0 }$ still holds. In fact, for a given time t, recall that $\begin{array} { r } { \tilde { g } _ { m , t } ( x ) = \sum _ { s = 1 } ^ { S _ { \eta } } \hat { w } _ { s , t } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ^ { ( s ) } ) } \end{array}$ Without loss of generalization, assume $\hat { w } _ { 1 , t } > 1 / 2$ and $\theta ^ { ( 2 ) }$ is the nearest point to $\theta ^ { ( 1 ) }$ in $\Theta ( \eta )$ . Then, $\| \theta ^ { ( 2 ) } - \theta ^ { ( 1 ) } \| _ { 2 } \leq \eta$ . We adjust the weight as $\hat { w } _ { 1 , t } ^ { \prime } : = 1 / 2 , \hat { w } _ { 2 , t } ^ { \prime } : = \hat { w } _ { 1 , t } + \hat { w } _ { 2 , t } - 1 / 2$ $\hat { w } _ { s , t } ^ { \prime } : = \hat { w } _ { s , t }$ for $s \geq 3$

$$
\tilde { g } _ { m , t } ^ { \prime } ( x ) : = \sum _ { s = 1 } ^ { S _ { \eta } } \hat { w } _ { s , t } ^ { \prime } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ^ { ( s ) } ) ,
$$

and

$$
\tilde { f } ^ { t ^ { \prime } } ( x ) : = \sum _ { \ell = 1 } ^ { M _ { S } } \hat { f } _ { \ell , S } ^ { t } ( x ) + \sum _ { m = 1 } ^ { M _ { R } } \tilde { g } _ { m , t } ^ { \prime } ( x ) \hat { f } _ { m , R } ^ { t } ( x ) .
$$

Then, we have

$$
\tilde { f } ^ { t ^ { \prime } } ( x ) - \tilde { f } ^ { t } ( x ) = \left( \hat { w } _ { 1 , t } - 1 / 2 \right) \left( \check { f } ^ { t } ( x ; \theta ^ { ( 1 ) } ) - \check { f } ^ { t } ( x ; \theta ^ { ( 2 ) } ) \right) .
$$

By the proof of Theorem 3.8, we know that under the same choice of $\eta ,$

$$
\Big \| \tilde { f } ^ { t } ( x ; \theta ^ { ( 1 ) } ) - \check { f } ^ { t } ( x ; \theta ^ { ( 2 ) } ) \Big \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq \frac { C M _ { R } d } { n } : = \frac { C _ { M _ { R } } } { n } .
$$

Thus this adjustment only brings an additional $1 / n$ term, which can be absorbed into the original ARM oracle bound for the online aggregated sequence $\{ \tilde { f } ^ { t } \}$ □

## F.3 Specific Upper Bounds for Dense Softmax Gating in Section 4.2

## F.3.1 Dense Softmax Upper Bound after Corollary 4.9

Recall that we have a partition of X into $\chi = \cup _ { r = 1 } ^ { M _ { 0 } } \chi _ { r }$ , up to $P _ { X }$ -null boundary sets, suppose that there exist vectors $\{ v _ { r } \in \mathbb { R } ^ { d + 1 } \} _ { r = 1 } ^ { M _ { 0 } }$ , normalized so that $\mathrm { m a x } _ { r \in [ M _ { 0 } ] } \| v _ { r } \| _ { 2 } \leq 1$ , such that, for $\bar { x } = ( x ^ { \top } , 1 ) ^ { \top }$ and each $r \in [ M _ { 0 } ]$ 2

$$
\mathcal { X } _ { r } = \{ x \in \mathcal { X } : \bar { x } ^ { \top } v _ { r } > \bar { x } ^ { \top } v _ { j } , \ \forall j \neq r \} .
$$

For each region $\mathcal { X } _ { r } .$ , define the pointwise margin function

$$
m _ { r } ( x ) : = v _ { r } ^ { \top } \bar { x } - \operatorname* { m a x } _ { j \neq r } v _ { j } ^ { \top } \bar { x } \geq 0 , \qquad x \in \mathcal { X } _ { r } .\tag{F.21}
$$

We impose the following condition on the distribution of $X$

Assumption F.8 (Margin regularity for linearly separable regions). Let $P _ { X }$ be the distribution of X and, for each $r ,$ denote by $G _ { r }$ the random variable $G _ { r } : = m _ { r } ( X )$ conditional on

$X \in \mathcal { X } _ { r }$ . Suppose that each $G _ { r }$ admits a density $f _ { G _ { r } }$ on $[ 0 , \infty )$ that is uniformly bounded:

$$
\operatorname* { s u p } _ { r \leq M _ { 0 } } \operatorname* { s u p } _ { u \geq 0 } f _ { G _ { r } } ( u ) \leq L < \infty .
$$

Moreover, assume that the normalized parameter space contains a logarithmic inner box: for some constant $C _ { \mathrm { i n } } > 2$ 2

$$
[ - C _ { \mathrm { i n } } \log M _ { R } , C _ { \mathrm { i n } } \log M _ { R } ] ^ { ( M _ { R } - 1 ) ( d + 1 ) } \subseteq \Theta _ { M _ { R } } .
$$

The density condition controls the probability mass near the interfaces between regions. In particular, it implies that the margin variable $m _ { r } ( X )$ has at most linear probability mass near zero. This condition is satisfied, for example, when X has a bounded density on a bounded domain and the separated hyperplanes are non-degenerate.

We next record a dense-softmax approximation bound for the hard Top-1 region assignment.

Proposition F.9. Under the linearly separable partition condition above and Assumption $F . 8 ,$ for $M _ { R } \ge M _ { 0 }$

$$
\operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \mathcal { X } _ { r } } \| g ^ { \mathrm { s o f t } } ( x ; \theta ) - e _ { r } \| _ { 2 } ^ { 2 } d P _ { X } ( x ) = O \left( \frac { M _ { 0 } } { \log M _ { R } } \right) .
$$

In particular, when $M _ { 0 }$ is fixed, this bound is of order $O ( 1 / \log M _ { R } )$

Proof of Proposition F.9. Since the softmax weights are invariant under a simultaneous relabeling of the routed experts and the corresponding assignment vectors, we may assume, without loss of generality, that the specialist for ${ \mathcal { X } } _ { M _ { 0 } }$ is used as the reference expert. We therefore impose the reference normalization $\theta _ { M _ { 0 } } = 0$ in this proof. Define

$$
B _ { \mathcal { X } } ^ { \prime } = 1 \vee \operatorname* { m a x } _ { x \in \mathcal { X } } \| \bar { x } \| _ { 2 } , \qquad \bar { x } = ( x ^ { \top } , 1 ) ^ { \top } ,
$$

and choose

$$
t : = \frac { C _ { \mathrm { i n } } \log M _ { R } } { 4 B _ { \chi } ^ { \prime } } .
$$

For $r = 1 , \ldots , M _ { 0 } - 1$ , define $\tilde { \theta } _ { r } : = t ( v _ { r } - v _ { M _ { 0 } } )$ and set ${ \tilde { \theta } } _ { M _ { 0 } } = 0$ . For the dummy experts $j = M _ { 0 } + 1 , \ldots , M _ { R } .$ , define $\tilde { \theta } _ { j } = ( 0 , \ldots , 0 , - C _ { \mathrm { i n } } \log M _ { R } ) ^ { \top }$ . Because $\| v _ { r } \| _ { 2 } \leq 1$ and $B _ { \mathcal { X } } ^ { \prime } \geq 1$ the above choice of t ensures that the constructed parameters lie in the logarithmic inner box. Hence $\tilde { { \boldsymbol { \theta } } } \in \Theta _ { M _ { R } }$

Fix $r \in [ M _ { 0 } ]$ and $x \in \mathcal { X } _ { r }$ , and let $g ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } )$ denote the corresponding softmax vector. Write

$$
a _ { i } ( x ) = v _ { i } ^ { \top } \bar { x } , \quad i \in [ M _ { 0 } ] , \qquad a _ { ( - r ) } ( x ) : = \operatorname* { m a x } _ { j \neq r } v _ { j } ^ { \top } \bar { x } .
$$

Then $m _ { r } ( x ) = a _ { r } ( x ) - a _ { ( - r ) } ( x ) \geq 0$ . For the constructed logits $\ell _ { j } ( x ) : = \tilde { \theta } _ { j } ^ { \top } \bar { x }$ , we have

$$
\ell _ { j } ( x ) = t ( a _ { j } ( x ) - a _ { M _ { 0 } } ( x ) ) , \quad j < M _ { 0 } , \quad \ell _ { M _ { 0 } } ( x ) = 0 , \quad \ell _ { j } ( x ) = - C _ { \mathrm { i n } } \log M _ { R } , \quad j > M _ { 0 } .
$$

Therefore

$$
g _ { r } ^ { \mathrm { s o f t } } ( x ; \widetilde { \theta } ) \geq \frac { 1 } { 1 + ( M _ { 0 } - 1 ) \exp \{ - t m _ { r } ( x ) \} + M _ { R } ^ { - ( C _ { \mathrm { i n } } - 2 ) / 2 } } .
$$

Indeed,

$$
1 + \sum _ { j \neq r } \exp \{ \ell _ { j } ( x ) - \ell _ { r } ( x ) \} \leq 1 + \sum _ { j \in [ M _ { 0 } ] , j \neq r } \exp \{ - t m _ { r } ( x ) \} + \sum _ { j > M _ { 0 } } \exp \{ - C _ { \mathrm { i n } } \log M _ { R } + t ( a _ { M _ { 0 } } ( x ) - a _ { r } ( x ) ) \} .
$$

The first sum is bounded by $( M _ { 0 } - 1 ) \exp \{ - t m _ { r } ( x ) \}$ . For the dummy experts, we use

$$
| a _ { M _ { 0 } } ( x ) - a _ { r } ( x ) | \leq \| v _ { M _ { 0 } } - v _ { r } \| _ { 2 } \| \bar { x } \| _ { 2 } \leq 2 B _ { \mathcal { X } } ^ { \prime } ,
$$

and hence

$$
t ( a _ { M _ { 0 } } ( x ) - a _ { r } ( x ) ) \leq 2 t B _ { \mathcal { X } } ^ { \prime } = \frac { C _ { \mathrm { i n } } } { 2 } \log M _ { R } .
$$

Thus,

$$
\sum _ { j > M _ { 0 } } \exp \{ - C _ { \mathrm { i n } } \log M _ { R } + t ( a _ { M _ { 0 } } ( x ) - a _ { r } ( x ) ) \} \leq ( M _ { R } - M _ { 0 } ) \exp \{ C _ { \mathrm { i n } } \log M _ { R } / 2 \} \leq M _ { R } ^ { - ( C _ { \mathrm { i n } } - 2 ) / 2 } .
$$

Combining the preceding bounds gives

$$
1 + \sum _ { j \neq r } \exp \{ \ell _ { j } ( x ) - \ell _ { r } ( x ) \} \leq 1 + \left( M _ { 0 } - 1 \right) \exp \{ - t m _ { r } ( x ) \} + M _ { R } ^ { - \left( C _ { \mathrm { i n } } - 2 \right) / 2 } ,
$$

and therefore the displayed lower for $g _ { r } ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } )$ follows.

It follows that

$$
1 - g _ { r } ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) \leq \left( M _ { 0 } - 1 \right) \exp \{ - t m _ { r } ( x ) \} + { M _ { R } ^ { - } } ^ { ( C _ { \mathrm { i n } } - 2 ) / 2 } .
$$

Since $0 \leq g _ { j } ^ { \mathrm { s o f t } } \leq 1$ and $\begin{array} { r } { \sum _ { j } g _ { j } ^ { \mathrm { s o f t } } = 1 } \end{array}$

$$
\| g ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) - e _ { r } \| _ { 2 } ^ { 2 } = ( 1 - g _ { r } ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) ) ^ { 2 } + \sum _ { j \ne r } \{ g _ { j } ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) \} ^ { 2 } \le 2 ( 1 - g _ { r } ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) ) .
$$

Consequently,

$$
\| g ^ { \mathrm { s o f t } } ( x ; \widetilde { \theta } ) - e _ { r } \| _ { 2 } ^ { 2 } \le 2 ( M _ { 0 } - 1 ) \exp \{ - t m _ { r } ( x ) \} + 2 M _ { R } ^ { - ( C _ { \mathrm { i n } } - 2 ) / 2 } .
$$

Integrating over $\mathcal { X } _ { r }$ and summing over r yields

$$
\sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \mathcal { X } _ { r } } \| g ^ { \mathrm { s o t } } ( x ; \widetilde { \theta } ) - c _ { r } \| _ { 2 } ^ { 2 } d P _ { X } ( x ) \le 2 ( M _ { 0 } - 1 ) \sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \mathcal { X } _ { r } } \exp \{ - t m _ { r } ( x ) \} d P _ { X } ( x ) + 2 M _ { 0 } { \cal M } _ { R } ^ { - ( C _ { \mathrm { i n } } - 2 ) / 2 } .
$$

Let $p _ { r } : = \mathbb { P } ( X \in \mathcal { X } _ { r } )$ . By the definition of $G _ { r }$

$$
\int _ { \chi _ { r } } \exp \{ - t m _ { r } ( x ) \} d P _ { X } ( x ) = p _ { r } \int _ { 0 } ^ { \infty } \exp \{ - t u \} f _ { G _ { r } } ( u ) d u .
$$

Using the uniform density bound,

$$
\int _ { \chi _ { r } } \exp \{ - t m _ { r } ( x ) \} d P _ { X } ( x ) \leq p _ { r } L \int _ { 0 } ^ { \infty } \exp \{ - t u \} d u = p _ { r } \frac { L } { t } .
$$

Summing over r and using $\begin{array} { r } { \sum _ { r } p _ { r } = 1 } \end{array}$ , we obtain

$$
\sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \chi _ { r } } \Vert g ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) - e _ { r } \Vert _ { 2 } ^ { 2 } d P _ { X } ( x ) \leq 2 ( M _ { 0 } - 1 ) \frac { L } { t } + 2 M _ { 0 } M _ { R } ^ { - ( C _ { \mathrm { i n } } - 2 ) / 2 } .
$$

Since $t = C _ { \mathrm { i n } }$ log $M _ { R } / ( 4 B _ { \chi } ^ { \prime } )$ , the first term is of order $M _ { 0 } / \log M _ { R }$ . The second term is polynomially small in $M _ { R }$ because $C _ { \mathrm { i n } } > 2$ . Hence

$$
\sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \mathcal { X } _ { r } } \Vert g ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) - e _ { r } \Vert _ { 2 } ^ { 2 } d P _ { X } ( x ) = O \left( \frac { M _ { 0 } } { \log M _ { R } } \right) .
$$

Taking the infimum over $\theta \in \Theta _ { M _ { R } }$ completes the proof.

Proposition F.9 yields the following risk upper bound for dense softmax gating in the setting of Corollary 4.9.

Corollary F.10. Assume the dense-softmax oracle assumptions in the main text and Assumption F.8 hold. Let $\hat { f } _ { 0 }$ be the estimator produced by Algorithm 1 in the main text. Then

$$
\mathbb { E } \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le C \left( \sum _ { r = 1 } ^ { M _ { 0 } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + M _ { R } ( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } ) + \frac { M _ { R } d } { n } \log ( M _ { R } d n ) + \frac { M _ { 0 } } { \log M _ { R } } \right) ,
$$

where C is a constant independent of $n , M _ { 0 }$ and $M _ { R }$

Proof of Corollary F.10. We first apply the dense-softmax oracle inequality in Theorem 3.8.

$$
\mathbb { E } \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left( \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \left\| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { m } ^ { * } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + M _ { R } ( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } ) + \frac { M _ { R } d } { n } \log ( M _ { R } d n ) \right) .
$$

It remains to bound the approximation term. For any $\theta \in \Theta _ { M _ { R } }$ 2

$$
\Bigg \Vert f _ { 0 } ( x ) - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { m } ^ { * } ( x ) \Bigg \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = \sum _ { r = 1 } ^ { M _ { 0 } } \Bigg \Vert \Bigg ( f _ { 0 } ( x ) - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { m } ^ { * } ( x ) \Bigg ) \mathbb { 1 } _ { \mathscr { X } _ { r } } \Bigg \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Using the squared triangle inequality on each region, we obtain

$$
\left. f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o t h } } ( \cdot ; \theta ) f _ { m } ^ { * } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \le 2 \sum _ { r = 1 } ^ { M _ { 0 } } \left. ( f _ { 0 } - f _ { r } ^ { * } ) \mathbb { 1 } _ { \mathcal { X } _ { r } } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + 2 \sum _ { r = 1 } ^ { M _ { 0 } } \left. \left( f _ { r } ^ { * } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o t h } } ( \cdot ; \theta ) f _ { m } ^ { * * } \right) \mathbb { 1 } _ { \mathcal { X } _ { r } } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Taking the infimum over $\theta \in \Theta _ { M _ { R } }$ , we get

$$
\begin{array} { r l } & { \underset { \theta \in \Theta _ { M _ { R } } } { \underline { { \operatorname* { i n f } } } } \left. f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { m } ^ { * } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq 2 \displaystyle \sum _ { r = 1 } ^ { M _ { 0 } } \Vert ( f _ { 0 } - f _ { r } ^ { * } ) \mathbb { 1 } _ { \mathcal { X } _ { r } } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } \\ & { \quad + \ 2 \underset { \theta \in \Theta _ { M _ { R } } } { \underline { { \operatorname* { i n f } } } } \displaystyle \sum _ { r = 1 } ^ { M _ { 0 } } \left. \left( f _ { r } ^ { * } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { m } ^ { * } \right) \mathbb { 1 } _ { \mathcal { X } _ { r } } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } . } \end{array}
$$

We now control the second term. Since the expert functions are uniformly bounded by $A ,$ for $x \in \mathcal { X } _ { r }$ 2

$$
\begin{array} { r l r } {  { \bigg | f _ { r } ^ { * } ( x ) - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { m } ^ { * } ( x ) \bigg | = \bigg | ( 1 - g _ { r } ^ { \mathrm { s o f t } } ( x ; \theta ) ) f _ { r } ^ { * } ( x ) - \sum _ { m \neq r } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { m } ^ { * } ( x ) \bigg | } } \\ & { } & { \leq ( 1 - g _ { r } ^ { \mathrm { s o f t } } ( x ; \theta ) ) | f _ { r } ^ { * } ( x ) | + \sum _ { m \neq r } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) | f _ { m } ^ { * } ( x ) | } \\ & { } & { \leq 2 A ( 1 - g _ { r } ^ { \mathrm { s o f t } } ( x ; \theta ) ) . } \end{array}
$$

Therefore,

$$
\sum _ { r = 1 } ^ { M _ { 0 } } \left\| \left( f _ { r } ^ { * } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { m } ^ { * } \right) \mathbb { 1 } _ { \mathcal { X } _ { r } } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq 4 A ^ { 2 } \sum _ { r = 1 } ^ { M _ { 0 } } \| ( e _ { r } - g ^ { \mathrm { s o f t } } ( \cdot ; \theta ) ) \mathbb { 1 } _ { \mathcal { X } _ { r } } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } ,
$$

where we used $( 1 - g _ { r } ^ { \mathrm { s o f t } } ( x ; \theta ) ) ^ { 2 } \leq \| e _ { r } - g ^ { \mathrm { s o f t } } ( x ; \theta ) \| _ { 2 } ^ { 2 }$ . Combining the preceding displays and applying Proposition F.9 gives

$$
\operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \left\| f _ { 0 } ( x ) - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { m } ^ { * } ( x ) \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left( \sum _ { r = 1 } ^ { M _ { 0 } } \| f _ { 0 } - f _ { r } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + \frac { M _ { 0 } } { \log M _ { R } } \right) .
$$

Substituting this bound into the dense-softmax oracle inequality yields the desired result.

## F.3.2 Dense Softmax Upper Bound after Proposition 4.11

After relabeling the routed experts, assume that for each $r \in [ M _ { 0 } ]$ , the best two-expert convex combination on region $\mathcal { X } _ { r }$ is $w _ { r 1 } f _ { 2 r - 1 } ^ { * } + w _ { r 2 } f _ { 2 r } ^ { * }$ . This is only a notational simplification of the general two-expert pair in the main text. Denote $\mathbf { w } _ { r } : = ( 0 , \ldots , 0 , w _ { r 1 } , w _ { r 2 } , 0 , \ldots , 0 )$ ，

where $w _ { r 1 }$ is on the $( 2 r - 1 )$ -th entry and $w _ { r 2 }$ is on the 2r-th entry. We next give the corresponding dense-softmax approximation bound analogue to Proposition F.9.

Proposition F.11. Under the linearly separable partition condition above and Assumption F.8, further assume that $w _ { r i }$ are bounded away from 0 in the sense that ma $\mathrm { x } _ { r \in [ M _ { R } ] , i = 1 , 2 } | \log w _ { r i } | <$ $C _ { w }$ for some constant $C _ { w } < \infty$ . Then, for $M _ { R } \ge 2 M _ { 0 }$ ，

$$
\operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \mathcal { X } _ { r } } \| g ^ { \mathrm { s o f t } } ( x ; \theta ) - \mathbf { w } _ { r } \| _ { 2 } ^ { 2 } d P _ { X } ( x ) = O \left( \frac { M _ { 0 } } { \log M _ { R } } \right) .
$$

In particular, when $M _ { 0 }$ is fixed, this bound is of order $O ( 1 / \log M _ { R } )$

Proof of Proposition F.11. As in the proof of Proposition F.9, we may relabel the routed experts and use the specialist indexed by $2 M _ { 0 }$ as the reference expert. Thus, in the proof we impose the reference normalization $\theta _ { 2 M _ { 0 } } = 0$ . Let

$$
B _ { \mathcal { X } } ^ { \prime } = 1 \vee \operatorname* { m a x } _ { x \in \mathcal { X } } \| \bar { x } \| _ { 2 } , \qquad \bar { x } = ( x ^ { \top } , 1 ) ^ { \top } ,
$$

and choose

$$
t = \frac { C _ { \mathrm { i n } } \log M _ { R } } { 4 B _ { \mathcal { X } } ^ { \prime } } .
$$

For $r = 1 , \ldots , M _ { 0 }$ , define $c _ { r } = \log w _ { r 1 } - \log w _ { r 2 }$ . Then $| c _ { r } | \leq 2 C _ { w }$ . Define the candidate parameters by

$$
\tilde { \theta } _ { 2 r - 1 } = t ( v _ { r } - v _ { M 0 } ) + ( 0 , \ldots , 0 , c _ { r } ) ^ { \top } , \qquad \tilde { \theta } _ { 2 r } = t ( v _ { r } - v _ { M 0 } ) , \qquad r = 1 , \ldots , M _ { 0 } ,
$$

and set $\tilde { \theta } _ { j } = ( 0 , \ldots , 0 , - C _ { \mathrm { i n } } \log M _ { R } ) ^ { \top }$ for $j > 2 M _ { 0 }$ . The boundedness of $c _ { r } ,$ , the normalization $\| v _ { r } \| ~ \leq ~ 1$ , and the logarithmic inner-box condition imply that $\tilde { { \boldsymbol { \theta } } } \in \Theta _ { M _ { R } }$ for all suficiently large $M _ { R }$ . Finitely many smaller values of $M _ { R }$ are absorbed into the constant in the final bound.

Fix $r \in [ M _ { 0 } ]$ and $x \in \mathcal { X } _ { r }$ . Write

$$
a _ { i } ( \boldsymbol { x } ) = \boldsymbol { v } _ { i } ^ { \top } \bar { \boldsymbol { x } } , \quad i \in [ M _ { 0 } ] , \qquad a _ { ( - r ) } ( \boldsymbol { x } ) = \operatorname* { m a x } _ { j \neq r } \boldsymbol { v } _ { j } ^ { \top } \bar { \boldsymbol { x } } .
$$

Then $m _ { r } ( x ) = a _ { r } ( x ) - a _ { ( - r ) } ( x ) \geq 0$ . Let $\ell _ { j } ( x ) : = \tilde { \theta } _ { j } ^ { \top } \bar { x }$ . For the two experts assigned to region $\mathcal { X } _ { r } , \ell _ { j } ( x ) = - C _ { \mathrm { i n } } \log M _ { R } , j > 2 M _ { 0 }$ . Define the relative contribution of all experts outside the pair {2r − 1, 2r} by

$$
D _ { r } ( x ) = \sum _ { j \notin \{ 2 r - 1 , 2 r \} } \exp \{ \ell _ { j } ( x ) - \ell _ { 2 r } ( x ) \} .
$$

Then

$$
g _ { 2 r - 1 } ^ { \mathrm { s o f t } } ( x ; \widetilde { \theta } ) = \frac { \exp \{ c _ { r } \} } { 1 + \exp \{ c _ { r } \} + D _ { r } ( x ) } , \qquad g _ { 2 r } ^ { \mathrm { s o f t } } ( x ; \widetilde { \theta } ) = \frac { 1 } { 1 + \exp \{ c _ { r } \} + D _ { r } ( x ) } .
$$

We next bound $D _ { r } ( x )$ . For $q \neq r ,$ , the margin condition gives $a _ { q } ( x ) - a _ { r } ( x ) \leq - m _ { r } ( x )$ Therefore, the contribution from the non-target region pairs satisfies

$$
\sum _ { q \neq r } \{ \exp \{ \ell _ { 2 q - 1 } ( x ) - \ell _ { 2 r } ( x ) \} + \exp \{ \ell _ { 2 q } ( x ) - \ell _ { 2 r } ( x ) \} \} \leq C ( M _ { 0 } - 1 ) \exp \{ - t m _ { r } ( x ) \} ,
$$

where C depends only on the weight lower bound through $C _ { w }$ . For the dummy experts, using $| a _ { M _ { 0 } } ( x ) - a _ { r } ( x ) | \leq \| v _ { M _ { 0 } } - v _ { r } \| _ { 2 } \| \bar { x } \| _ { 2 } \leq 2 B _ { \mathcal { X } } ^ { \prime }$ , we obtain

$$
\sum _ { j > 2 M _ { 0 } } \exp \{ - C _ { \mathrm { i n } } \log M _ { R } + t ( a _ { M _ { 0 } } ( x ) - a _ { r } ( x ) ) \} \le M _ { R } ^ { - ( C _ { \mathrm { i n } } - 2 ) / 2 } .
$$

Hence

$$
D _ { r } ( x ) \leq C ( M _ { 0 } - 1 ) \exp \{ - t m _ { r } ( x ) \} + { M } _ { R } ^ { - ( C _ { \mathrm { i n } } - 2 ) / 2 } .
$$

Since exp $\{ c _ { r } \} / ( 1 + \exp \{ c _ { r } \} ) = w _ { r 1 }$ and $1 / ( 1 + \exp \{ c _ { r } \} ) = w _ { r 2 }$ , the two target softmax weights have the same relative proportions as $( w _ { r 1 } , w _ { r 2 } )$ . Moreover, $g _ { 2 r - 1 } ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) + g _ { 2 r } ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) =$ $\frac { 1 + \exp \{ c _ { r } \} } { 1 + \exp \{ c _ { r } \} + D _ { r } ( x ) }$ , so

$$
1 - g _ { 2 r - 1 } ^ { \mathrm { s o f t } } ( x ; \widetilde { \theta } ) - g _ { 2 r } ^ { \mathrm { s o f t } } ( x ; \widetilde { \theta } ) = \frac { D _ { r } ( x ) } { 1 + \exp \{ c _ { r } \} + D _ { r } ( x ) } \le D _ { r } ( x ) .
$$

Consequently

$$
\| g ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) - \mathbf { w } _ { r } \| _ { 2 } ^ { 2 } = ( w _ { r 1 } - g _ { 2 r - 1 } ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) ) ^ { 2 } + ( w _ { r 2 } - g _ { 2 r } ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) ) ^ { 2 } + \sum _ { j \not \in \{ 2 r - 1 , 2 r \} } g _ { j } ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) ^ { 2 } .
$$

Because the first two softmax weights are proportional to $( w _ { r 1 } , w _ { r 2 } )$ , their deviation from $( w _ { r 1 } , w _ { r 2 } )$ is controlled by the missing mass assigned outside the pair. Thus

$$
\| g ^ { \mathrm { s o f t } } ( \boldsymbol { x } ; \widetilde { \boldsymbol { \theta } } ) - \mathbf { w } _ { r } \| _ { 2 } ^ { 2 } \leq 2 ( 1 - g _ { 2 r - 1 } ^ { \mathrm { s o f t } } ( \boldsymbol { x } ; \widetilde { \boldsymbol { \theta } } ) - g _ { 2 r } ^ { \mathrm { s o f t } } ( \boldsymbol { x } ; \widetilde { \boldsymbol { \theta } } ) ) \leq 2 D _ { r } ( \boldsymbol { x } ) .
$$

Combining this with the preceding bound on $D _ { r } ( x )$ , we obtain the pointwise inequality

$$
\begin{array} { r } { \| g ^ { \mathrm { s o f t } } ( x ; \widetilde { \theta } ) - \mathbf { w } _ { r } \| _ { 2 } ^ { 2 } \leq C ( M _ { 0 } - 1 ) \exp \{ - t m _ { r } ( x ) \} + 2 M _ { R } ^ { - ( C _ { \mathrm { i n } } - 2 ) / 2 } . } \end{array}
$$

Integrating over $\mathcal { X } _ { r }$ and summing over r gives

$$
\sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \mathcal { X } _ { r } } \| g ^ { \mathrm { s o f t } } ( x ; \widetilde { \theta } ) - \mathbf { w } _ { r } \| _ { 2 } ^ { 2 } d P _ { X } ( x ) \leq C ( M _ { 0 } - 1 ) \sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \mathcal { X } _ { r } } \exp \{ - t m _ { r } ( x ) \} d P _ { X } ( x ) + 2 M _ { 0 } { \cal M } _ { R } ^ { - ( C _ { \mathrm { i n } } - 2 ) / 2 } .
$$

The same margin-distribution calculation as in the proof of Proposition F.9 gives

$$
\sum _ { r = 1 } ^ { M _ { 0 } } \int _ { \mathcal { X } _ { r } } \| g ^ { \mathrm { s o f t } } ( x ; \tilde { \theta } ) - \mathbf { w } _ { r } \| _ { 2 } ^ { 2 } d P _ { X } ( x ) \leq C ( M _ { 0 } - 1 ) \frac { L } { t } + 2 M _ { 0 } M _ { R } ^ { - ( C _ { \mathrm { i n } } - 2 ) / 2 } .
$$

Since $t \asymp$ log $M _ { R }$ and $C _ { \mathrm { i n } } > 2$ , the right-hand side is of order $O ( M _ { 0 } / \log M _ { R } )$ . Taking the infimum over $\theta \in \Theta _ { M _ { R } }$ completing the proof. □

This yields the following risk upper bound for dense softmax gating in the setting of Corollary 5.7.

Corollary F.12. Assume the dense-softmax oracle assumptions in the main text and Assumption F.8 hold. Assume further that $w _ { r 1 } + w _ { r 2 } = 1 , w _ { r i } > 0$ , and the weights are uniformly bounded away from zero in the sense that ma $\mathrm { x } _ { r , i }$ | log $w _ { r i } | < C _ { w }$ for some constant $C _ { w } \ < \ \infty$ . Let $\hat { f } _ { 0 }$ denote the estimator produced by Algorithm 1 in the main text.

Then

$$
\mathbb { E } \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left( \sum _ { r = 1 } ^ { M _ { 0 } } \| f _ { 0 } - w _ { r 1 } f _ { r , 1 } ^ { * } - w _ { r 2 } f _ { r , 2 } ^ { * } \| _ { \mathcal { X } _ { r } } ^ { 2 } + M _ { R } ( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } ) + \frac { M _ { R } d } { n } \log ( M _ { R } d n ) + \frac { M _ { 0 } } { \log M _ { R } } \right) ,
$$

where $C$ is a constant independent of $n , M _ { 0 }$ and $M _ { R }$

Proof of Corollary F.12. We first apply the dense-softmax oracle inequality in Theorem 3.8.

This gives

$$
\mathbb { E } \| \hat { f } _ { 0 } - f _ { 0 } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \left( \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } } \left\| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { m } ^ { * } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } + M _ { R } ( \bar { a } _ { n _ { 2 } } ^ { 2 } + \bar { a } _ { n _ { 3 } } ^ { 2 } ) + \frac { M _ { R } d } { n } \log ( M _ { R } d n ) \right) .
$$

It remains to bound the approximation term. For any $\theta \in \Theta _ { M _ { R } }$

$$
\left\| f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { m } ^ { * } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } = \sum _ { r = 1 } ^ { M _ { 0 } } \left\| \left( f _ { 0 } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { m } ^ { * } \right) \mathbb { 1 } _ { \mathcal { X } _ { r } } \right\| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Using the squared inequality on each region, we obtain

$$
\begin{array} { r l } { \bigg \| f _ { 0 } - \displaystyle \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { m } ^ { * } \bigg \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \displaystyle \sum _ { r = 1 } ^ { M _ { 0 } } \| ( f _ { 0 } - w _ { r 1 } f _ { r , 1 } ^ { * } - w _ { r 2 } f _ { r , 2 } ^ { * } ) \mathbb { 1 } _ { \mathcal { X } _ { r } } \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } } & \\ & { \quad \quad + C \displaystyle \sum _ { r = 1 } ^ { M _ { 0 } } \bigg \| \bigg ( w _ { r 1 } f _ { r , 1 } ^ { * } + w _ { r 2 } f _ { r , 2 } ^ { * } - \displaystyle \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { m } ^ { * } \bigg ) \mathbb { 1 } _ { \mathcal { X } _ { r } } \bigg \| _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } . } \end{array}
$$

Taking the infimum over $\theta \in \Theta _ { M _ { R } }$ , it remains to control the second term. Since the expert

functions are uniformly bounded by A, for $x \in \mathcal { X } _ { r }$

$$
\begin{array} { r l } { \displaystyle  w _ { r 1 } f _ { r , 1 } ^ { * } ( x ) + w _ { r 2 } f _ { r , 2 } ^ { * } ( x ) - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) f _ { m } ^ { * } ( x )  = \displaystyle  ( w _ { r 1 } - g _ { 2 r - 1 } ^ { \mathrm { s o f t } } ( x ; \theta ) ) f _ { r , 1 } ^ { * } ( x ) + ( w _ { r 2 } - g _ { 2 r } ^ { \mathrm { s o f t } } ( x ; \theta ) ) f _ { r , 2 } ^ { * } ( x ) - \sum _ { m \neq \ell \ge 0 } \sum _ { \ell \ge 0 } \sqrt { 1 - \xi _ { m } ^ { \mathrm { s o f t } } ( x ; \theta ) } } & { \quad } \\ { \le A (  w _ { r 1 } - g _ { 2 r - 1 } ^ { \mathrm { s o f t } } ( x ; \theta )  +  w _ { r 2 } - g _ { 2 r } ^ { \mathrm { s o f t } } ( x ; \theta )  + 1 - g _ { 2 r - 1 } ^ { \mathrm { s o f t } } ( x ; \theta ) -  } & { \quad } \\ { \le  2 A ( | w _ { r 1 } - g _ { 2 r - 1 } ^ { \mathrm { s o f t } } ( x ; \theta ) | + | w _ { r 2 } - g _ { 2 r } ^ { \mathrm { s o f t } } ( x ; \theta ) | ) ) , } & { \quad } \end{array}
$$

where the last inequality holds because $w _ { r 1 } + w _ { r 2 } = 1$ . Therefore, using $( a + b ) ^ { 2 } \leq 2 ( a ^ { 2 } + b ^ { 2 } )$

$$
\sum _ { r = 1 } ^ { M _ { 0 } } \left. \left( w _ { r 1 } f _ { r , 1 } ^ { * } + w _ { r 2 } f _ { r , 2 } ^ { * } - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { m } ^ { * } \right) \mathbb { 1 } _ { \mathcal { X } _ { r } } \right. _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq 8 A ^ { 2 } \sum _ { r = 1 } ^ { M _ { 0 } } \Vert ( \mathbf { w } _ { r } - g ^ { \mathrm { s o f t } } ( \cdot ; \theta ) ) \mathbb { 1 } _ { \mathcal { X } _ { r } } \Vert _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } .
$$

Combining the preceding displays and apply Proposition F.11 gives

$$
 \operatorname* { i n f } _ { \theta \in \Theta _ { M _ { R } } }  f _ { 0 } ( x ) - \sum _ { m = 1 } ^ { M _ { R } } g _ { m } ^ { \mathrm { s o f t } } ( \cdot ; \theta ) f _ { m } ^ { * }  _ { L _ { 2 } ( P _ { X } ) } ^ { 2 } \leq C ( \sum _ { r = 1 } ^ { M _ { 0 } } \Vert f _ { 0 } - w _ { r 1 } f _ { r , 1 } ^ { * } - w _ { r 2 } f _ { r , 2 } ^ { * } \Vert _ { X _ { r } } ^ { 2 } + \frac { M _ { 0 } } { \log M _ { R } } ) .
$$

Substituting this bound into the dense-softmax oracle inequality yields the desired result.

## G Simulation Details

## G.1 Comparison of diferent gating classes

We conduct Monte Carlo simulations to illustrate how the performance of diferent gating classes depends on the geometry of the underlying routing regions. The goal is not to provide a comprehensive benchmark comparison, but rather to examine whether a gating class performs best when its geometric structure is aligned with that of the true partition.

We consider three data-generating processes (DGPs) corresponding to region partitions with linear, quadratic and highly nonlinear boundaries. These designs form a hierarchy of geometric complexity and allow us to assess how diferent gating classes adapt to diferent partition structures. Throughout, the expert functions are fixed and treated as known at the estimation stage, so that diferences across methods primarily reflect the ability of the gating mechanism to approximate the true routing structure.

For each design, we compare three candidate gating specifications: linear gating, quadratic gating, and Gaussian-kernel-based gating, as defined in the main text. Observations are generated according to

$$
y _ { i } = \sum _ { m = 1 } ^ { 3 } g _ { m } ^ { * } ( x _ { i } ) f _ { m } ( x _ { i } ) + \varepsilon _ { i } , \qquad i = 1 , \dots , n ,\tag{G.1}
$$

where $\varepsilon _ { i } \sim N ( 0 , \sigma ^ { 2 } )$ and $x _ { i } = ( x _ { i 1 } , x _ { i 2 } ) ^ { \top }$ . In all designs, the covariates $x _ { i 1 }$ and $x _ { i 2 }$ are

generated independently from the uniform distribution on [−1, 1]. The gating function follows the softmax form

$$
g _ { m } ^ { * } ( x ) = \frac { \exp \{ \eta _ { m } ^ { * } ( x ) \} } { \sum _ { \ell = 1 } ^ { 3 } \exp \{ \eta _ { \ell } ^ { * } ( x ) \} } , \qquad m = 1 , 2 , 3 ,\tag{G.2}
$$

where the latent scores $\eta _ { m } ^ { * } ( x )$ vary across the three designs and induce expert-dominance partitions of difering geometric complexity.

Linear DGP. We first consider a design in which the true routing regions are induced by linear scores. The latent scores are given by

$$
\eta _ { 1 } ^ { * } ( x ) = a ^ { * } x _ { 1 } , \qquad \eta _ { 2 } ^ { * } ( x ) = a ^ { * } x _ { 2 } , \qquad \eta _ { 3 } ^ { * } ( x ) = 0 ,
$$

where $a ^ { * } = 2$ . Under this construction, the third expert serves as a baseline component, and the resulting expert-dominance regions are separated by linear decision boundaries. The expert functions are fixed as

$$
f _ { 1 } ( x ) = 2 + 2 x _ { 1 } - x _ { 2 } , \qquad f _ { 2 } ( x ) = - 1 - x _ { 1 } + 2 x _ { 2 } , \qquad f _ { 3 } ( x ) = 1 + \sin ( \pi x _ { 1 } ) - 1 . 5 x _ { 2 } ^ { 2 } .
$$

We set $n \in \{ 2 0 0 , 4 0 0 \}$ and $\sigma = 0 . 5$

Quadratic DGP. The second design considers routing regions generated by quadratic scores:

$$
\eta _ { 1 } ^ { * } ( x ) = 2 x _ { 1 } - 1 . 2 x _ { 1 } ^ { 2 } - 0 . 8 x _ { 2 } ^ { 2 } , \qquad \eta _ { 2 } ^ { * } ( x ) = 2 x _ { 2 } - 0 . 8 x _ { 1 } ^ { 2 } - 1 . 2 x _ { 2 } ^ { 2 } , \qquad \eta _ { 3 } ^ { * } ( x ) = 0 .
$$

The third expert again serves as the baseline component. Compared with the linear DGP, the quadratic terms generate curved decision boundaries, so that the resulting dominance regions are no longer linearly separable. This setting therefore allows us to examine the performance loss from using a geometrically misspecified linear gate.

The expert functions are the same as those in the linear DGP:

$$
f _ { 1 } ( x ) = 2 + 2 x _ { 1 } - x _ { 2 } , \qquad f _ { 2 } ( x ) = - 1 - x _ { 1 } + 2 x _ { 2 } , \qquad f _ { 3 } ( x ) = 1 + \sin ( \pi x _ { 1 } ) - 1 . 5 x _ { 2 } ^ { 2 } .
$$

We again consider sample sizes $n \in \{ 2 0 0 , 4 0 0 \}$ and set $\sigma = 0 . 5$

Nonlinear DGP. The third design considers routing regions with highly nonlinear boundaries. Unlike the previous two settings, the true gating structure is generated through a nonlinear deformation of angular regions. This construction cannot be represented exactly by either linear or quadratic gating specifications and therefore provides a setting in which more flexible gates may be beneficial. Specifically, let $\theta = \operatorname { a t a n 2 } ( x _ { 2 } , x _ { 1 } )$ and $r = ( x _ { 1 } ^ { 2 } + x _ { 2 } ^ { 2 } ) ^ { 1 / 2 }$ denote the polar angle and radial coordinate, and define the deformed angular coordinate $\widetilde { \theta } = \theta + 0 . 8 5 \sin ( 4 . 5 r + 1 . 8 \theta ) + 0 . 3 5 \sin ( 4 \theta )$ . We then set $\eta _ { m } ^ { * } ( x ) = \tau \cos ( \widetilde { \theta } - \phi _ { m } )$ , with $\tau = 7$ and angular centers $\phi _ { 1 } = \pi , \phi _ { 2 } = \pi / 3$ , and $\phi _ { 3 } = - \pi / 3$ . Under this construction, the three experts dominate diferent regions of the covariate space, while the boundaries between regions become highly irregular due to the angular deformation.

The expert functions are specified as $f _ { 1 } ( x ) = 3 + 3 x _ { 1 } - 2 x _ { 2 } , f _ { 2 } ( x ) = - 2 - 2 x _ { 1 } + 3 x _ { 2 }$ and $f _ { 3 } ( x ) = 2 . 5 \sin ( \pi x _ { 1 } x _ { 2 } ) - 3 x _ { 1 } + 2 x _ { 2 }$ . We set $n \in \{ 5 0 0 , 1 0 0 0 \}$ and $\sigma = 0 . 3$

Implementation of gating estimators. The implementation details difer slightly across the three designs. In the linear DGP, the linear gating parameters are searched over the common grid $\{ 0 , 0 . 5 , 1 , \ldots , 3 \}$ , and the quadratic gating parameters over $\{ 0 , 0 . 6 , 1 . 2 , \ldots , 3 \}$ The Gaussian-kernel-based gating uses a regular $3 \times 3$ grid of kernel centers on $[ - 1 , 1 ] ^ { 2 }$ with bandwidth $\sigma _ { K } = \kappa h$ , where $\kappa = 0 . 7 5$ and $h = 1$ is the spacing between neighboring grid points.

In the quadratic DGP, the linear and quadratic grids are centered around the true quadratic specification. For linear gating, the four coeficients are searched over $\{ - 2 . 5 , - 1 . 2 5 , 0 , 1 . 2 5 , 2 . 5 \}$ , $\{ - 1 , 0 , 1 \} , \{ - 1 , 0 , 1 \}$ , and $\{ - 2 . 5 , - 1 . 2 5 , 0 , 1 . 2 5 , 2 . 5 \}$ }, respectively. For quadratic gating, the linear coeficients are searched over the same corresponding grids, while each quadratic coeficient is searched over $\{ - 1 . 6 , - 1 . 2 , - 0 . 8 , 0 \}$ . The Gaussian-kernel-based gating again uses a $3 \times 3$ grid with $\kappa = 0 . 7 5$

In the nonlinear DGP, the linear gating parameters are searched over $\{ - 3 , - 2 , \ldots , 3 \}$ , and the quadratic gating parameters over $\{ - 2 , - 1 , 0 , 1 , 2 \}$ . The Gaussian-kernel-based gating uses a denser $4 \times 4$ grid of kernel centers on $[ - 1 , 1 ] ^ { 2 }$ with $\kappa = 0 . 5$

Monte Carlo evaluation. We conduct 100 Monte Carlo replications. For each replication, let $\hat { g } ( x _ { i } ) = ( \hat { g } _ { 1 } ( x _ { i } ) , \ldots , \hat { g } _ { 3 } ( x _ { i } ) ) ^ { \top }$ and $g ^ { * } ( x _ { i } ) = ( g _ { 1 } ^ { * } ( x _ { i } ) , \ldots , g _ { 3 } ^ { * } ( x _ { i } ) ) ^ { \top }$ denote the estimated and true gating-weight vectors at observation $i ,$ respectively. We evaluate gatingweight estimation using the mean $\ell _ { 1 }$ error and mean squared $\ell _ { 2 }$ error, defined as

$$
{ \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \| { \hat { g } } ( x _ { i } ) - g ^ { * } ( x _ { i } ) \| _ { 1 } \qquad { \mathrm { a n d } } \qquad { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \| { \hat { g } } ( x _ { i } ) - g ^ { * } ( x _ { i } ) \| _ { 2 } ^ { 2 } ,
$$

respectively. For each gating class, we report the Monte Carlo mean and standard deviation of both error measures.

In addition to numerical summaries, Figure 2 compares the true and estimated expertdominance regions, defined by $\arg \operatorname* { m a x } _ { m } g _ { m } ^ { * } ( x )$ and $\arg \operatorname* { m a x } _ { m } \hat { g } _ { m } ( x )$ , respectively, using the first Monte Carlo replication. These visualizations illustrate how diferent gating classes adapt to the geometry of the underlying partition.

## G.2 MSE Comparison With and Without a Shared Expert

We conduct two simulation studies to examine whether a shared-routed representation improves prediction accuracy relative to a pure-routed representation when the routing threshold is unknown. Both studies are based on the Fourier-sieve illustration in Example 6.4, but emphasize two complementary regimes: a simple shared component with complex routed components, and a complex shared component with simple routed components.

In both designs, we generate $X \sim \mathrm { U n i f } [ 0 , 1 ]$ and $Y = f _ { 0 } ( X ) + \varepsilon$ , where $\varepsilon \sim N ( 0 , \sigma ^ { 2 } )$ , $\sigma = 0 . 5$ , and the true threshold is $c _ { 0 } = 1 / 2$

Simple shared, complex routed. The first design sets

$$
f _ { 0 } ( x ) = \theta x + \mathbb { 1 } \left\{ x \leq c _ { 0 } \right\} f _ { R , 1 } ( 2 x ) + \mathbb { 1 } \left\{ x > c _ { 0 } \right\} f _ { R , 2 } ( 2 x - 1 ) ,
$$

with $\theta = 1$ . The routed components are Fourier series of order $K _ { 0 } = 3$ without intercepts,

$$
f _ { R , j } ( t ) = \sum _ { k = 1 } ^ { 3 } \left\{ a _ { j , k } \sin ( 2 \pi k t ) + b _ { j , k } \cos ( 2 \pi k t ) \right\} , \qquad j = 1 , 2 ,
$$

where $a _ { 1 } = ( 1 . 0 , 0 . 6 , 0 . 4 ) , b _ { 1 } = ( 0 . 7 , - 0 . 5 , 0 . 3 ) , a _ { 2 } = ( - 0 . 8 , 0 . 5 , - 0 . 3 )$ , and $b _ { 2 } = ( 0 . 6 , 0 . 4 , - 0 . 5 )$

Complex shared, simple routed. The second design sets

$$
f _ { 0 } ( x ) = f _ { S } ( x ) + \mathbb { 1 } \{ x \leq c _ { 0 } \} f _ { R , 1 } ( 2 x ) + \mathbb { 1 } \{ x > c _ { 0 } \} f _ { R , 2 } ( 2 x - 1 ) ,
$$

![](images/9b9658147bdfe952aa2d3ae80807b05c25c90a146a0ecc153165f7e5f7bd9308.jpg)  
(a) Linear gating

![](images/d2ae395d73926e33493883919bec00caa66572bf31c963024c63eb2f0176e758.jpg)  
(b) Quadratic gating

![](images/8e7589a6180356838423f56ccf0a62c10b06867ea76be7ecca9358971b3257b2.jpg)  
(c) GK-based gating

![](images/97f7926ee32363674aa3caceaf9cf8d94e91548bd498674ac72dc851294422a8.jpg)  
(d) Linear gating

![](images/0383084f3e9827fb04883d0997cdd394cdb9f6f2200cf22100534f571836138d.jpg)  
(e) Quadratic gating

![](images/173637b07b92df26456d41ef684437825ffc16af068e2f3e4c2206a7f02ca529.jpg)  
(f) GK-based gating

![](images/7095a22c3b434ef79f91a9592912f4b7af6c3e81ad93b5d5898c0480cefcaf92.jpg)  
(g) Linear gating

![](images/f0c8d28ec063bfb654fc9abf263b65d37006fa6b7bd90fb01d6477c367da403d.jpg)  
(h) Quadratic gating

![](images/812c4d5170dfc6f462e173f46f5849df0d9977a77e11136e7079592417ccefbb.jpg)  
(i) GK-based gating  
Figure 2: Comparison of estimated routing regions under diferent expert-induced partition geometries and gating classes. Rows correspond to the linear, quadratic, and nonlinear DGPs, while columns correspond to the linear, quadratic, and Gaussian-kernel-based gating. The results illustrate the importance of matching the geometric flexibility of the gating class to the complexity of the expert-induced partition.

where $f _ { S } ( x ) = A \sin ( 2 \pi q x )$ with $A = 1$ and $q = 3$ . The routed components are simple linear functions with opposite slopes, $f _ { R , 1 } ( t ) = \delta ( t - 1 / 2 )$ and $f _ { R , 2 } ( t ) = - \delta ( t - 1 / 2 )$ for $t \in [ 0 , 1 ]$ 2 where $\delta = 0 . 6$

For each design, we compare a shared-routed estimator with a pure-routed estimator. In the first design, the shared-routed estimator includes a linear shared component and routed Fourier sieves of order $K = 3$ with intercepts; the pure-routed estimator uses the same routed Fourier sieves but omits the shared component. In the second design, the shared-routed estimator includes a Fourier shared component of order $Q = 3$ and routed linear experts in span $\{ 1 , t \}$ ; the pure-routed estimator uses the same routed linear experts but omits the shared component.

For both estimators and both designs, the threshold c is unknown and is estimated by profile least squares over a 200-point grid on [0.1, 0.9]. Conditional on each candidate threshold, all linear coeficients are estimated by ordinary least squares, and the threshold minimizing the residual sum of squares is selected.

For each $n \in \{ 2 0 0 , 4 0 0 , 8 0 0 , 1 6 0 0 \}$ , the experiment is repeated 200 times. Prediction accuracy is evaluated by the integrated squared error

$$
\mathrm { M S E } = \int _ { 0 } ^ { 1 } \{ \widehat { f } ( x ) - f _ { 0 } ( x ) \} ^ { 2 } d x ,
$$

which is approximated on a dense grid of 5000 equally spaced points in $[ 0 , 1 ]$