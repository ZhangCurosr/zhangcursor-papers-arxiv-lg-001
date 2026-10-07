# HOW BREGMAN DIVERGENCES SHAPE SHAMPOO

Bing Liu<sup>1</sup>, Wenjie Zhou<sup>2</sup>, Chengcheng Zhao<sup>†</sup> <sup>1</sup>, Hongtao Zhang<sup>2</sup>, Boao Kong<sup>3</sup>, Felix Dangel<sup>\*</sup> <sup>4</sup>, and Wu Lin<sup>\*</sup> <sup>5</sup>

<sup>1</sup>Zhejiang University; <sup>2</sup>University of the Chinese Academy of Sciences; <sup>3</sup>Peking University <sup>4</sup>Concordia University & Mila; <sup>5</sup>University of Central Florida

## ABSTRACT

Understanding the principles behind Shampoo has recently guided the development of more effective neural network optimizers. These methods learn a preconditioner by optimizing the Frobenius or Kullback-Leibler (KL) divergence against the gradient second moment. In this work, we investigate how the choice of divergence shapes preconditioning, which remains unclear and blocks further improvements. To do so, we develop a unified Bregman divergence framework that connects all popular divergences, allowing us to study them jointly. Through empirical spectral analysis of gradient second moments, we examine how divergence choice shapes Kronecker approximation and interacts with finite-sample error in preconditioning. We find that some divergences can better compensate for finitesample underestimation of the empirical second moment, helping explain the differing behavior of their corresponding Shampoo variants. We further validate this explanation through GPT-2 pretraining experiments. By connecting divergence choice to practical training behavior, we believe our framework provides principled guidance for understanding the foundations of, and further improving, Shampoo.

## 1 INTRODUCTION

Shampoo’s empirical success (Dahl et al., 2023; Kasimbeg et al., 2025) raises a fundamental question: what makes its preconditioner effective? This question has motivated a growing body of work (Morwani et al., 2025; Vyas et al., 2025; An et al., 2025; Eschenhagen et al., 2025; 2026; Lin et al., 2026) with the goal of developing more effective neural network optimizers. A promising approach is to view Shampoo’s update scheme as a Kronecker-factored approximation of the full-matrix gradient second moment via Bregman divergence minimization (Lin et al., 2026). This approach recovers existing variants and produces new ones, including F-Shampoo with Frobenius divergence, VN-Shampoo with von Neumann divergence, and KL-Shampoo with Kullback-Leibler divergence. Recent studies have reported strong performance of KL-Shampoo in LLM pretraining (Eschenhagen et al., 2026; Zhang et al., 2026). However, the variants derived from different divergences perform very differently in practice (Figure 1, right). The framework of Lin et al. (2026) does not explain this performance gap. Given that improvements to Shampoo have been driven by understanding its principles, we argue that explaining this gap is essential for guiding the design of better variants, since many other Bregman divergences remain to be explored and evaluated. For example, preconditioned stochastic gradient descent (PSGD, Li, 2018; 2019) is a related method that has shown promising results (Frans et al., 2025) but whose success remains poorly understood. Surprisingly, we show that PSGD’s strong performance can also be understood through its connection to a new Bregman divergence. Thus, our central question is:

How and why do different divergence objectives lead to differences in training performance?

To answer this question, we consider effects that are often conflated in practice. (1) The Kronecker factorization does not exactly capture the second moment. (2) Finite samples introduce errors both in the second moment and in the estimated Kronecker factors. (3) They also interact with each other

F VN SQ KL C<sub>0</sub>

Table 1: Overview of the divergence properties studied in this work. The exponent q<sub>D</sub> indexes the trace-power family; spectral weighting and error asymmetry determine how approximation error is allocated, while few-sample error measures the stability of Kronecker factor estimation.
<table><tr><td>Property Divergence</td><td>Frobenius</td><td>von Neumann</td><td>Square root</td><td>KL</td></tr><tr><td>Trace-power exponent qD</td><td>2</td><td>1 (limit)</td><td> $^ 1 / 2$ </td><td>0 (limit)</td></tr><tr><td>Weight on large eigenvalues</td><td>Strong</td><td>Moderate</td><td>Mild</td><td>None</td></tr><tr><td>Penalized more</td><td>Overestimation</td><td>Overestimation</td><td>Symmetric</td><td>Underestimation</td></tr><tr><td>Error with few samples</td><td>Highest</td><td>Higher</td><td>Lower</td><td>Lowest</td></tr></table>

![](images/62ca3021325d105b34ac7e8112ecbd124eb7a306591264a637951ea704e64e14.jpg)

![](images/4b25fb7a2e67634e32043be6bd61973e80c13df88978e4c4abbaa1c02a0c9c56.jpg)

![](images/a089f4f48fd5cdee286625f1241aa3040ba8014641b78d30264d0bc3bdeb4e92.jpg)  
Figure 1: Divergences shape preconditioning and training. Left and middle: A GPT-2 checkpoint (output proj) at step 2,500, ordered by decreasing eigenvalue. Left: Empirical second-moment (C ) eigenvalues become less reliable toward the tail, increasingly underestimating held-out scales (Section 4.1). Middle (smoothed curves, asinh y-axis for better resolution near zero): KL stays closest to the held-out target (dashed zero line), followed by SQ, compensating $C _ { 0 } \mathrm { { ^ \circ } s }$ scale errors (Section 5). Right: GPT-2 validation loss relative to KL with asinh y-axis; lower is better (Section 6).

when used for preconditioning in practice. Divergence choice determines how these effects shape the resulting preconditioner. Thus, we decompose the question into the following sub-questions:

(Q1) How does divergence choice shape approximation errors?

(Q2) How does finite sampling affect Kronecker approximation?

(Q3) How do divergence choices and finite sampling jointly affect preconditioning?

By answering these questions, we make the following contributions:

(C1) A unified analysis of Kronecker approximation. We unify the three divergences from Lin et al. (2026) into a trace-power Bregman family and introduce a fourth, the square-root (SQ) divergence, which recovers the gradient-whitening objective of PSGD and yields SQ-Shampoo. We show how the family exponent determines which parts of the spectrum are prioritized, and whether over- or underestimation is penalized more (Table 1).

(C2) Finite-sample effects on Kronecker approximation. Finite sampling enters in two places. On GPT-2, we show that empirical gradient second moments severely underestimate tail eigenvalues even when gradients outnumber dimensions (Figure 1, left). As for the Kronecker factor estimation, the same exponent $q _ { D }$ also controls the finite-sample error, with KL achieving the lowest error (Table 1, last row).

(C3) Explaining differences in inverse-root preconditioning. Last, we study how divergencedependent approximation (C1) interacts with the two finite-sample effects in (C2) during preconditioning. The divergence properties help explain why SQ and KL better compensate for underestimated empirical eigenvalues (Figure 1, middle) and more reliably recover their large-sample gains from short gradient histories (Table 3). Consistently, SQ and KL also achieve lower GPT-2 validation losses than F and VN, with KL’s advantage growing at larger widths.

Together, these results help explain how divergences affect preconditioning and training performance, and provide practical criteria for designing better Kronecker-factored optimizers.

## 2 BACKGROUND

Notation. Let $G \in \mathbb { R } ^ { m \times n }$ be the stochastic gradient of a matrix parameter Θ. We use row-wise vectorization $g = \operatorname { v e c } ( G )$ and write $C : = \mathbb { E } [ \breve { g } g ^ { \top } ] \in \mathbb { S } _ { + + } ^ { m n }$ for its gradient second moment, where ${ \mathbb S } _ { + + } ^ { d }$ denotes symmetric positive-definite (SPD) matrices. For simplicity, we omit momentum, as in prior work.

Kronecker Approximation and Preconditioning For factors $L \in \mathbb { S } _ { + + } ^ { m } , R \in \mathbb { S } _ { + + } ^ { n }$ , we consider methods that approximate C by a Kronecker product $S : = L \otimes R$ , with inverse-root preconditioning:

$$
S ^ { - 1 / 2 } g = ( L ^ { - 1 / 2 } \otimes R ^ { - 1 / 2 } ) g = \mathrm { v e c } ( L ^ { - 1 / 2 } G R ^ { - 1 / 2 } ) .
$$

Shampoo. Shampoo (Gupta et al., 2018; Anil et al., 2020) maintains averages $L _ { t } , R _ { t }$ of $G _ { t } G _ { t } ^ { \top }$ and $G _ { t } ^ { \top } G _ { t }$ . Its preconditioning step is $\Theta _ { t + 1 } = \Theta _ { t } - \eta _ { t } L _ { t } ^ { - p } G _ { t } R _ { t } ^ { - p }$ . Motivated by gradient secondmoment approximation, modern Shampoo variants (Shi et al., 2023; Morwani et al., 2025; Eschenhagen et al., 2025) use $p = 1 / 2$ , while original Shampoo (Gupta et al., 2018) uses $p = 1 / 4$

PSGD. Preconditioned stochastic gradient descent (PSGD, Li, 2018; 2019) learns a preconditioner P that acts on the stochastic gradient as $P g$ . Its Fisher-type gradient-whitening objective is

$$
\operatorname* { m i n } _ { P \in \mathbb { S } _ { + + } ^ { m n } } \mathrm { T r } ( P C ) + \mathrm { T r } ( P ^ { - 1 } ) ,\tag{1}
$$

with unique dense solution $P = C ^ { - 1 / 2 }$ . PSGD-Kron restricts the preconditioner to a Kronecker product $\bar { P } = P _ { L } \otimes P _ { R }$ and learns the factors through relative-gradient updates (Li, 2019).

Although PSGD and Shampoo were developed independently, we will show in the next section that PSGD-Kron is closely related to a Shampoo variant under a new Bregman divergence. This allows us to understand them and explain their strong performance in a unified way.

Bregman matrix divergences. For a strictly convex, differentiable $f : ( 0 , \infty ) \to \mathbb { R }$ , we apply f spectrally to SPD matrices. Given the eigendecomposition $\begin{array} { r } { M = \sum _ { i = 1 } ^ { d } \mu _ { i } u _ { i } u _ { i } ^ { \top } , f ( M ) = } \end{array}$ $\textstyle \sum _ { i = 1 } ^ { d } f ( \mu _ { i } ) u _ { i } u _ { i } ^ { \intercal }$ and $\Phi ( M ) = \operatorname { T r } ( f ( M ) )$ ). Since $\begin{array} { r } { \Phi ( M ) = \sum _ { i } f ( \mu _ { i } ) } \end{array}$ , Φ is also strictly convex on SPD matrices, whenever f is strictly convex. For $X , Y \in \mathbb { S } _ { + + } ^ { d }$ , the Bregman matrix divergence (Dhillon & Tropp, 2008) is

$$
B _ { \Phi } ( X , Y ) = \Phi ( X ) - \Phi ( Y ) - \langle \nabla \Phi ( Y ) , X - Y \rangle _ { \mathrm { F } } .\tag{2}
$$

where $\langle A , B \rangle _ { \mathrm { F } } = \operatorname { T r } ( A ^ { \top } B )$ is the Frobenius inner product. It is nonnegative and vanishes exactly when ${ \dot { X } } = { \dot { Y } }$ . Choosing $f ( x ) = x ^ { 2 } / 2$ , x log x − x, or − log x gives Frobenius (F), von Neumann (VN), or Kullback-Leibler (KL) divergence, respectively. The KL divergence is also known as the LogDet divergence in the literature. Appendix B gives matrix formulas for these divergences.

Deriving Shampoo methods via Bregman divergence minimization. Lin et al. (2026) derive F-, VN-, and KL-Shampoo variants from the stationarity condition of the following Bregman objective:

$$
\operatorname* { m i n } _ { L \in \mathbb { S } _ { + + } ^ { m } , \ R \in \mathbb { S } _ { + + } ^ { n } } B _ { \Phi } ( C , L \otimes R ) ,\tag{3}
$$

with different generators Φ. However, these variants perform differently in practice (Figure 1, right) when training neural networks. The relationship between the choice of Φ and empirical training performance remains unknown, as the authors do not explain it. Understanding this connection helps guide the design of better Kronecker-factored optimizers. Appendix A discusses related work in detail.

## 3 KRONECKER APPROXIMATION ERROR ALLOCATION

We first address Q1: How does divergence choice shape approximation errors? We use a tracepower Bregman family to unify Shampoo variants and connect the Kronecker-constrained PSGD objective in Equation (1) to the Bregman formulation in Equation (3). Then, within this family, we compare how the divergences allocate spectral approximation error.

## 3.1 A TRACE-POWER FAMILY OF KRONECKER PRECONDITIONERS

Unified analysis of Bregman objectives. We formulate the objectives above through a trace-power Bregman family, obtained by extending scalar divergence generators (Cichocki & Amari, 2010) to SPD matrices via spectral functions (Amari, 2014). Our unification further clarifies the choice of the generators and lets us connect PSGD-Kron and study its related variant within the same framework.

Proposition 1 (Unification of Shampoo and PSGD-Kron via the Trace-Power Bregman Objective). The objectives of $F _ { \overline { { \mathbf { \Gamma } } } }$ VN-, and KL-Shampoo and the gradient-whitening objective ofPSGD-Kron belong to the same trace-power Bregmanfamily, generated by $\Phi _ { q } ( M ) = \operatorname { T r } ( f _ { q } ( M ) )$ . Here, $f _ { q }$ is applied spectrally to the eigenvalues ofM as defined in Section 2. The four scalar generators are shown in Figure 2:

![](images/3615118d9b5be641fc3be2b4d741b05ccd8f0dfb5857d18ae01ece656137ac85.jpg)  
Figure 2: The family of scalar generators $f _ { q } ( x )$ inducing tracepower Bregman divergences we study through one unifying lens. Shown with an asinh vertical scale.

$$
f _ { q } ( x ) : = \frac { x ^ { q } - q x + q - 1 } { q ( q - 1 ) } \left( q \neq 0 , 1 \right) , f _ { 1 } ( x ) : = x \log x - x + 1 , f _ { 0 } ( x ) : = - \log x + x - 1 ,
$$

Let $q _ { \mathrm { D } }$ denote the exponent for divergence D. Frobenius has $q _ { \mathrm { F } } = 2 ,$ , while von Neumann and KL divergences arise as $q $ 1 and $q  0 ,$ , respectively,for which we denote the exponents by $q _ { \mathrm { V N } } = 1$ and $q _ { \mathrm { K L } } = 0$ . The square-root (SQ) member, $q _ { \mathrm { S Q } } = { } ^ { 1 } / 2 ,$ , recovers PSGD-Kron’s gradient-whitening objective equation 1 under $P = S ^ { - 1 / 2 }$ , up to a positive scale and an additive constant.

Setting $S = L \otimes R$ gives the coupled stationarity equations:

$$
L _ { q } = \frac { \mathbb { E } [ G R ^ { q - 1 } G ^ { \top } ] } { \operatorname { T r } ( R ^ { q } ) } , R _ { q } = \frac { \mathbb { E } [ G ^ { \top } L ^ { q - 1 } G ] } { \operatorname { T r } ( L ^ { q } ) } .\tag{4}
$$

Every joint stationary point satisfies these coupled equations. See proofin Appendix B.1.

Square-root (SQ) Shampoo. To better understand how the choice of Bregman divergence shapes optimization, we consider a new Shampoo variant—square-root $( S Q )$ -Shampoo—that is induced by the Bregman family and recovers PSGD’s objective function by setting $q = 1 / 2$ in Equation (4). We consider this variant because the existing PSGD method does not employ a Shampoo-like scheme.

We consider four divergences in this work. For a fair comparison, we adopt the estimation technique suggested by Lin et al. (2026) in our study of divergence-induced update schemes, which reduces cost and resembles Shampoo’s update scheme. Implementation details are given in Appendix C. We then examine how these divergences allocate approximation error under limited Kronecker capacity (Section 3.2).

## 3.2 SPECTRAL ERROR ALLOCATION UNDER LIMITED KRONECKER CAPACITY

As our theory (Appendix B.2) and empirical results (Figure 5, left) show, a single Kronecker product generally cannot match C exactly. The divergence therefore determines how the remaining approximation errors are penalized. These penalty properties are not specific to gradient second moments, so we first derive them for arbitrary SPD matrices to isolate the effect of the divergence itself.

Proposition 2 (Spectral Error Decomposition and Scaling). Let $A = U \Lambda U ^ { \top } , B = V \Omega V ^ { \top } \in \mathbb { S } _ { + - } ^ { d }$ + have eigenpairs $( \lambda _ { i } , u _ { i } )$ and $( \omega _ { i } , v _ { i } )$ ordered by decreasing eigenvalue, and set $P _ { i j } = \langle u _ { i } , v _ { j } \rangle ^ { 2 } .$ Writing d<sub>f</sub> for the scalar Bregman divergence generated by $f ,$ , we have

$$
\begin{array} { r } { \mathcal { B } _ { \Phi } ( A , B ) = \underbrace { \sum _ { i = 1 } ^ { d } d _ { f } ( \lambda _ { i } , \omega _ { i } ) } _ { e i g e n v a l u e m i s m a t c h } + \underbrace { \Delta _ { f } ( \Lambda , \Omega ; P ) } _ { e i g e n s p a c e m i s a l i g n m e n t } , \quad \Delta _ { f } ( \Lambda , \Omega ; P ) \ge 0 , } \end{array}
$$

For the trace-power scalar generator $f _ { q } ,$ , write $d _ { q } : = d _ { f _ { q } }$ and $\Delta _ { q } : = \Delta _ { f _ { q } }$ . Both mismatch costs behave similarly when eigenvalues are re-scaled: for any $\dot { r } > 0 ;$ , they satisfy

$$
d _ { q } ( \lambda , r \lambda ) = \lambda ^ { q } d _ { q } ( 1 , r ) , \quad \Delta _ { q } ( r \Lambda , r \Omega ; P ) = r ^ { q } \Delta _ { q } ( \Lambda , \Omega ; P ) .\tag{5}
$$

Thus q controls how strongly both mismatch costs depend on eigenvalue scale.

Eigenspace mismatch. Keeping the eigenvectors fixed (and hence $P$ unchanged), multiplying all eigenvalues of both matrices by $r > 1$ multiplies the eigenspace-mismatch cost by $r ^ { q }$ . Thus larger q places greater relative weight on the same eigenspace misalignment at larger eigenvalue scales.

Eigenvalue (spectral) mismatch. Writing $d _ { \mathrm { F } } = d _ { 2 } , d _ { \mathrm { V N } } = d _ { 1 } , d _ { \mathrm { S Q } } = d _ { 1 / 2 }$ , and $d _ { \mathrm { K L } } = d _ { 0 }$ , the scalar eigenvalue penalties are

$$
\begin{array} { r l } & { d _ { \mathrm { F } } ( \lambda , r \lambda ) = \frac { 1 } { 2 } \lambda ^ { 2 } ( r - 1 ) ^ { 2 } , \qquad d _ { \mathrm { V N } } ( \lambda , r \lambda ) = \lambda ( r - 1 - \log r ) , } \\ & { d _ { \mathrm { S Q } } ( \lambda , r \lambda ) = 2 \lambda ^ { 1 / 2 } \left( \sqrt { r } + 1 / \sqrt { r } - 2 \right) , \qquad d _ { \mathrm { K L } } ( \lambda , r \lambda ) = \log r + 1 / r - 1 . } \end{array}\tag{6}
$$

For fixed relative mismatch $r ,$ the weights $\lambda ^ { 2 } , \lambda , \lambda ^ { 1 / 2 } ,$ 1 show that larger q puts more emphasis on large eigenvalues $( { \mathrm { F i g u r e ~ } } 3 ,$ left). We call their associated eigenvectors leading directions.

Over- versus under-estimation and asymmetry. For $r > 1$ , the fitted values $r \lambda$ and $\lambda / r$ over- and under-estimate λ by the same multiplicative factor. Their penalties satisfy (Figure 3, right)

![](images/5a6d03ceef5411e625c8b0ac9c3d2b69098401f5e3e8bcb41f1e2264712fec68.jpg)

$$
d _ { \mathrm { F } } ( \lambda , r \lambda ) > d _ { \mathrm { F } } ( \lambda , \lambda / r ) ,
$$

$$
d _ { \mathrm { V N } } ( \lambda , r \lambda ) > d _ { \mathrm { V N } } ( \lambda , \lambda / r ) ,\tag{7}
$$

$$
d _ { \mathrm { S Q } } ( \lambda , r \lambda ) = d _ { \mathrm { S Q } } ( \lambda , \lambda / r ) ,
$$

![](images/4aa222de84dd845118943fd6cc5935735aa501f7898d3308dd62badc61b02f74.jpg)

$$
d _ { \mathrm { K L } } ( \lambda , r \lambda ) < d _ { \mathrm { K L } } ( \lambda , \lambda / r ) .
$$

Figure 3: Left: Normalized penalties ${ \frac { d _ { \mathrm { D } } ( \lambda , r \lambda ) } { d _ { \mathrm { D } } ( 1 , r ) } } = \lambda ^ { q _ { \mathrm { D } } }$ for $r ~ \neq ~ 1$ Smaller $q$ places less emphasis on large eigenvalues; KL is scale-independent. Right: $d _ { \mathrm { D } } ( 1 , r )$ versus $^ { r , }$ with a log horizontal scale and an asinh vertical scale. F and VN favor underestimation over reciprocal overestimation, KL shows the opposite preference, and SQ is symmetric.

Table 1 summarizes these properties together with the finite-sample results from Section 4.2. $\mathrm { A p \mathrm { - } }$   
pendix B.3.4 gives the Kronecker decomposition and shows that the same penalty rules still hold.

Takeaway. Smaller $q _ { \mathrm { D } }$ gives more weight to the tail spectrum and penalizes underestimation more strongly, encouraging larger fitted eigenvalues. These larger eigenvalues can help avoid excessive inverse-root amplification, motivating smaller $q _ { \mathrm { D } }$ in preconditioner design.

## 4 FINITE-SAMPLE EFFECTS ON KRONECKER APPROXIMATION

We now turn to Q2: How does finite sampling affect Kronecker approximation? Finite sampling enters in two places: the empirical second-moment target and the estimated Kronecker factors in Equation (4). We examine the reliability of the target before Kronecker fitting, and then study divergence-dependent factor estimation from limited samples.

## 4.1 RELIABILITY OF THE EMPIRICAL SECOND-MOMENT TARGET

We test the reliability of the empirical gradient second moment by computing its eigenvectors on one data split and comparing second-moment values along the same directions on other independent data splits (Figure 1, left).

Shared experimental design (full details in Appendix E). Sections 4.1 and 5 use the same setup. First, we train a model (GPT-2 (Jordan et al., 2024) on FineWeb, 12 layers, hidden size 128) with AdamW and collect a large number of mini-batch gradients (batch size 64, sequence length 512), at selected checkpoints (100, 2,500, and 4,500) during training. We then split those gradients into four disjoint splits (each of size $N = 2 0 { , } 4 8 0 )$ . Each split gives rise to an empirical gradient second moment $\begin{array} { r } { C _ { s } = 1 / N { \sum _ { n = 1 } ^ { N } } g _ { s , n } g _ { s , n } ^ { \top } } \end{array}$ with $g _ { s , n }$ the nth mini-batch gradient in split s. In what follows, we focus on the query, value, and output projection weights of the model’s 5th block, for which $N > d = 1 2 8 ^ { 2 }$ . Thus, rank deficiency is not forced by the sample count. We repeated the analyses on checkpoints generated by training with Shampoo and find similar results (Appendix E.6). This large-sample setting provides an offline reference for subsequent preconditioning comparisons.

Are empirical second-moments reliable? Following high-dimensional covariance estimation work (Ledoit & Wolf, 2012), we estimate eigenvectors on a source split s and test their directiona scales on a held-out split $s ^ { \prime } \neq s .$ . Let $\{ ( \lambda _ { s , i } , u _ { s , i } ) \} _ { i = 1 } ^ { d }$ denote the eigenpairs of the empirical moment $C _ { s }$ on split s. To assess the representativeness of the source eigenvalues, we keep the source directions $u _ { s , i }$ fixed and compare their directional scales across splits. Along each direction, the source scale is $\lambda _ { s , i }$ and the held-out scale is $u _ { s , i } ^ { \top } C _ { s ^ { \prime } } u _ { s , i }$ . These scales can agree even if $u _ { s , i }$ is not an eigenvector of $C _ { s ^ { \prime } }$ . We measure held-out scale error by comparing held-out and source scales:

$$
\xi _ { s  s ^ { \prime } , i } : = \log _ { 1 0 } \frac { u _ { s , i } ^ { \top } C _ { s ^ { \prime } } u _ { s , i } } { u _ { s , i } ^ { \top } C _ { s } u _ { s , i } } = \log _ { 1 0 } \frac { u _ { s , i } ^ { \top } C _ { s ^ { \prime } } u _ { s , i } } { \lambda _ { s , i } } .\tag{8}
$$

Interpretation. Zero indicates cross-split agreement; positive values indicate underestimation and negative values indicate overestimation. Larger absolute values indicate greater mismatch. Thus, values close to zero across the spectrum indicate reliable empirical second moments.

Remark 3. Wefocus on eigenvalue (scale) reliability, without eigenvector stability. These directions need not remain eigenvectors across splits. What matters is whether the scales along these directions reproduce on independent data, supporting the scaling applied by the preconditioner is stable.

Empirical eigenvalues become less reliable toward the tail. In Figure 1 (left), using the AdamW step-2,500 output projection, held-out scale error is near zero for large eigenvalues and rises toward smaller eigenvalues, showing increasingly severe underestimation. Held-out scale error remains near zero in leading directions and strongly positive in the tail across checkpoints and projections from both AdamW and Distributed Shampoo (Table 9 and Figure 9). We also compare scales along directions selected from a third split $s ^ { \prime \prime }$ , independently of both splits above. The resulting crossfit errors stay near zero across the spectrum, supporting sampling noise as the source of the scale differences (see Appendix E.1.1).

Held-out scale as the correction. Covariance estimation and mini-batch curvature debiasing consider a correction based on separate data: keep the estimated eigenvectors, but replace their sample eigenvalues with scales measured along those directions on an independent batch (Ledoit & Wolf, 2012; Tatzel et al., 2025). Here, the replacement for $\lambda _ { s , i }$ is precisely the held-out scale $u _ { s , i } ^ { \top } C _ { s ^ { \prime } } u _ { s , i }$ in Equation (8). This motivates using it as the reference for preconditioner gains in Section 5.2

Takeaway. Even with many gradients, empirical second moments can underestimate the tail of the spectrum. Using one data split to estimate the eigenvectors and another to measure their scales reveals this finite-sample bias. Under inverse-root preconditioning, such underestimation can lead to overly large gains, and is even more pronounced in practice with short gradient histories.

## 4.2 FINITE-SAMPLE STABILITY OF KRONECKER FACTORS

The previous subsection shows finite-sample bias in the second moment. In practice, Kronecker factors are also estimated from shorter gradient histories, introducing another source of sampling error. We use a controlled model to isolate how divergence choice affects this factor estimation.

Model and isolated empirical factor estimation. To separate sampling error from approximation error in Kronecker approximation, we consider an exact Kronecker Gaussian model $C = L _ { \star } \otimes R _ { \star }$

$$
\begin{array} { r } { G = L _ { \star } ^ { 1 / 2 } Z R _ { \star } ^ { 1 / 2 } , \quad Z _ { i j } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , 1 ) . } \end{array}\tag{9}
$$

Fixing $R = R _ { \star }$ and replacing the expectation in Equation (4) with the sample expectation over N independent gradients $\dot { \{ G _ { b } = L _ { \star } ^ { 1 / 2 } Z _ { b } R _ { \star } ^ { 1 / 2 } \} } _ { b = 1 } ^ { N }$ gives the empirical estimate

$$
\widehat { L } _ { \mathrm { D } } = \frac { 1 } { N \mathrm { T r } ( R _ { \star } ^ { q _ { \mathrm { D } } } ) } \sum _ { b = 1 } ^ { N } G _ { b } R _ { \star } ^ { q _ { \mathrm { D } } - 1 } G _ { b } ^ { \top } = L _ { \star } ^ { 1 / 2 } \left[ \frac { 1 } { N \mathrm { T r } ( R _ { \star } ^ { q _ { \mathrm { D } } } ) } \sum _ { b = 1 } ^ { N } Z _ { b } R _ { \star } ^ { q _ { \mathrm { D } } } Z _ { b } ^ { \top } \right] L _ { \star } ^ { 1 / 2 } .\tag{10}
$$

Analogously, we define the right-side counterpart $\widehat { R } _ { \mathrm { D } }$ of $\widehat { L } _ { \mathrm { D } }$ to be the empirically estimated right factor on N samples when fixing the left factor to be $L _ { \star }$

Proposition 4 (Finite-Sample Factor Stability). Define the effective rank $r _ { \mathrm { e f f } } ( A ) : = \mathrm { T r } ( A ) ^ { 2 } \big / \mathrm { T r } ( A ^ { 2 } )$ and the relative factors $\widetilde { L } _ { \mathrm { D } } ^ { \mathrm { n } } : = L _ { \star } ^ { - 1 / 2 } \widehat { L } _ { \mathrm { D } } L _ { \star } ^ { - \check { \tau } / 2 }$ and $\widetilde { R } _ { \mathrm { D } } : = \widetilde { R } _ { \star } ^ { - 1 / 2 } \widehat { R } _ { \mathrm { D } } R _ { \star } ^ { - 1 / 2 }$ , which in the case of perfect estimation simplify to the identity matrix. Under Equation (9), the estimator in Equation (10) and its right-factor counterpart, with the opposite factor fixed at its population value, satisfy

$$
\mathrm { E r r } _ { \mathrm { D } } ^ { L } : = \frac { 1 } { m } \mathbb { E } \Vert \widetilde { L } _ { \mathrm { D } } - I \Vert _ { F } ^ { 2 } = \frac { m + 1 } { N r _ { \mathrm { e f f } } \left( R _ { \star } ^ { q _ { \mathrm { D } } } \right) } , \quad \mathrm { E r r } _ { \mathrm { D } } ^ { R } : = \frac { 1 } { n } \mathbb { E } \Vert \widetilde { R } _ { \mathrm { D } } - I \Vert _ { F } ^ { 2 } = \frac { n + 1 } { N r _ { \mathrm { e f f } } \left( L _ { \star } ^ { q _ { \mathrm { D } } } \right) } .
$$

Since $r _ { \mathrm { e f f } } ( A ^ { q } )$ is non-increasing in $q \geq 0 ,$ , as shown in Figure 4, left, these errors satisfy

$$
\mathrm { E r r } _ { \mathrm { K L } } ^ { s } \leq \mathrm { E r r } _ { \mathrm { S Q } } ^ { s } \leq \mathrm { E r r } _ { \mathrm { V N } } ^ { s } \leq \mathrm { E r r } _ { \mathrm { F } } ^ { s } , \quad s \in \{ L , R \} .
$$

with equality when the oppositefactor has aflat spectrum. See proofin Appendix B.4.

Why Frobenius fitting has larger estimation error. For a long-tailed $R _ { \star } , \mathrm { F } \left( q = 2 \right)$ puts most of its weight on a few large eigenvalues, reducing the effective number of directions that average out sampling noise. Equivalently, $R _ { \star } ^ { 2 }$ has a smaller effective rank, leading to larger estimation error. KL $( q = 0 )$ weights all directions equally, giving the largest effective rank and the smallest error under the assumptions of Proposition 4.

Scope, limitations, and empirical evidence. Here, our aim is to compare finite-sample factor stability across F, VN, SQ, and KL using the common exponent q introduced in Section 3.1. Joint Kronecker Gaussian estimation has been extensively studied in statistics. In particular, McCormack & Hoff (2025) characterize the asymptotic efficiency loss of VN relative to KL, while Franks et al. (2026) give high-probability error bounds for KL. However, these results do not directly provide a finite-sample comparison across all four divergences in our setting.

Our analysis estimates one factor with the other fixed under an exact Kronecker Gaussian model. This removes the coupling between factors and isolates the effect of spectral weighting, yielding a unified exact error formula for all four divergences. To test whether the ordering persists under joint fitting beyond these assumptions, we fit both factors on synthetic long-tailed non-Kronecker Gaussian gradients. The mean errors follow the predicted ordering at every tested sample size (Figure 4, right). Section 5.3 further examines the recovery of large-sample inverseroot gains on real gradients, without Gaussian or Kronecker assumptions.

![](images/d57ea34de9e40a8fe46dac186e15a3f4eead835778b2f8a8534f9eb886f3158c.jpg)

![](images/8afa68dc75755cf710acc076e9b83cd3c94fde4d7291b4233403382035294048.jpg)

Figure 4: Left: Effective rank $r _ { \mathrm { e f f } } ( R _ { \star } ^ { q } )$ for eigenvalues $\rho _ { j } ^ { \overline { { } } } = j ^ { - 1 . 8 }$ . Right: Mean squared relative Frobenius error of the jointly fitted ${ \cal S } \doteq { \cal L } \otimes { \cal R }$ (Equation (19)) on non-Kronecker gradients over 20 repeats, relative to each method’s 4096-sample fit. Both results favor smaller $q _ { \mathrm { D } } .$ , consistent with the theoretical prediction in Proposition 4.

Takeaway. With limited gradient samples, smaller $q _ { \mathrm { D } }$ gives more accurate Kronecker factor estimates under our model, with KL achieving the lowest expected relative error. The same trend also holds empirically beyond these assumptions. This favors smaller $q _ { \mathrm { D } }$ for preconditioners, where more stable factor estimates lead to reliable preconditioning from limited gradient histories.

## 5 FROM KRONECKER APPROXIMATIONS TO PRECONDITIONING

Q1 characterizes divergence-dependent approximation error allocation, while Q2 shows how finite sampling affects both the empirical target and Kronecker factor estimation. We now address Q3: how do divergence choice and finite sampling jointly affect preconditioning? We first examine whether large-sample Kronecker fits on real gradients follow the spectral predictions from Q1. Then, we ask how these fits interact with the finite-sample bias identified in $\boldsymbol { \mathrm { Q 2 } }$ when converted into inverse-root gains, and whether these gains remain reliable with short gradient histories.

## 5.1 LARGE-SAMPLE KRONECKER FITS

We fit each Kronecker state $S _ { D } ^ { \mathrm { r e f } }$ to the empirical second moment $C _ { 0 }$ , using all gradients in split 0. These large-sample fits show how different divergences allocate approximation error and whether the observed results, shown in Figure 5, match the predictions in Section 3.2.

Eigenvalues. We compare empirical and fitted eigenvalues in decreasing order. F underestimates much of the empirical spectrum, while VN follows it; SQ and KL have larger fitted tail eigenvalues. The same pattern holds across the other checkpoints and projections; see Figure 13.

Eigenspaces. Let $U _ { k }$ and $V _ { D , k }$ contain the leading k eigenvectors of $C _ { 0 }$ and $S _ { D } ^ { \mathrm { r e f } }$ . Following Song et al. (2025), we measure their subspace overlap by $O _ { D } ( k ) \ : =$ $\| U _ { k } ^ { \top } V _ { D , k } \| _ { \mathrm { F } } ^ { 2 } / k \overset { \cdot } { \in } [ \bar { 0 } , 1 ]$ , where 1 means identical subspaces and 0 means orthogonal subspaces. F best matches the leading eigenvector, while KL achieves higher overlap for most larger subspaces. Since inverse-root gains depend on both eigenspace overlap and fitted eigenvalues, we examine their joint effect in Section 5.2.

![](images/e8f22e2f039ef3007994ec97f373379316a6b7906dcfcdf73f2ac86399bb1278.jpg)  
Figure 5: Large-sample Kronecker fits. Left: empirical and fitted eigenvalues. Right: overlap of the leading subspaces. The fitted spectra and eigenspaces follow the divergence-dependent preferences predicted in Section 3.2.

Takeaway. The Kronecker fits reflect the spectral preferences predicted in Section 3.2.

## 5.2 PRECONDITIONING GAINS OF LARGE-SAMPLE FITS

The previous subsection shows that divergence choice leads to different eigenvalues and eigenspaces in the Kronecker fits. To answer Q3, we now ask how these differences translate into preconditioning through $S _ { D } ^ { - 1 / 2 }$ . We compare the resulting preconditioning gains with held-out inverse-root scales to test whether each fit compensates for the finite-sample scale errors identified in Section 4.1.

Following the held-out correction in Section 4.1, we use scales measured on independent data along the fixed source eigenvectors as the reference for preconditioning. Let $\{ ( \lambda _ { 0 , i } , u _ { 0 , i } ) \}$ be the eigenpairs of the source second moment $C _ { 0 }$ . We average the three held-out directional scales, $h _ { i } : = \overline { { { ^ { 1 } } / { 3 } } } \overline { { { \textstyle \sum _ { s ^ { \prime } = 1 } ^ { 3 } u _ { 0 , i } ^ { \top } C _ { s ^ { \prime } } u _ { 0 , i } } } }$ , and use ${ h _ { i } ^ { - 1 / 2 } }$ as the reference gain along $u _ { 0 , i }$ . For a fitted state $S _ { D }$ we define the held-out gain error as

$$
e _ { D , i } : = \log _ { 1 0 } \frac { \lvert \lvert S _ { D } ^ { - 1 / 2 } u _ { 0 , i } \rvert \rvert _ { 2 } } { h _ { i } ^ { - 1 / 2 } } = \frac { 1 } { 2 } \log _ { 1 0 } \left( h _ { i } u _ { 0 , i } ^ { \top } S _ { D } ^ { - 1 } u _ { 0 , i } \right) .\tag{11}
$$

Interpretation. Zero means matching the scaling factor $^ 1 / \sqrt { h _ { i } }$ estimated from held-out data; positive and negative values mean larger and smaller scales, respectively. Our goal is to keep this error close to zero across the whole spectrum.

![](images/90b5e0ae0f5ac1270a0897df7644db8f47af34934e753da6e5d050bc40da27c6.jpg)

![](images/9776c5f3384a8055599885799286dc7847edf91790627c5302e0fd979d790964.jpg)

![](images/c601df2d99cb1b7e9845df7427d92dda80e69350431e1d02afe5377b714e095c.jpg)  
Figure 6: Smoothed log held-out gain error for the output projection at AdamW step 2,500. Closer to zero is better; with asinh vertical scale. Left: large-sample fit; middle: 40-sample fits (4 repeats); right: EMA (16 repeats). Black solid: large-sample $\big ( C _ { 0 } ^ { - 1 / 2 }$ in all panels; dashed: the held-out target.

Divergence choice shapes inverse-root gains. KL and SQ better compensate for empirical secondmoment scaling error and stay closer to the held-out target across the whole spectrum, with KL achieving the lowest error (Figure 6, left, and Table 2). Results across the remaining checkpoint and projections show the same pattern (Appendix E.3).

Link to divergence properties. The behavior above follows the spectral preferences in Section 3.2. F and VN emphasize the leading spectrum and tend to underestimate the remaining eigenvalues,

Table 2: Absolute held-out gain error (Equation (11), ↓). KL has the lowest mean error, followed by SQ. Each experiment contributes the median $| e _ { D , i } |$ over all directions and repeats; entries show mean ± sample SD across nine checkpoint-projection pairs per training optimizer.
<table><tr><td>Optimizer</td><td>Method</td><td>Large-sample</td><td>40-sample</td><td>EMA</td></tr><tr><td rowspan="4">AdamW</td><td>F</td><td> $0 . 9 7 5 9 \pm 0 . 4 6 2 3$ </td><td> $2 . 4 9 4 5 \pm 0 . 8 2 6 5$ </td><td> $0 . 7 6 4 5 \pm 0 . 3 4 5 3$ </td></tr><tr><td>VN</td><td> $0 . 3 3 2 8 \pm 0 . 1 7 6 9$ </td><td> $0 . 5 5 4 9 \pm 0 . 2 9 8 6$ </td><td> $0 . 2 8 9 7 \pm 0 . 1 4 7 3$ </td></tr><tr><td>SQ</td><td> $0 . 1 4 6 7 \pm 0 . 0 5 7 3$ </td><td> $0 . 2 1 7 5 \pm 0 . 0 8 9 3$ </td><td> $0 . 1 2 9 4 \pm 0 . 0 4 1 2$ </td></tr><tr><td>KL</td><td> $\mathbf { 0 . 1 0 2 4 } \pm 0 . 0 1 8 2$ </td><td> $\mathbf { 0 . 1 5 6 1 } \pm 0 . 0 2 9 7$ </td><td> $\mathbf { 0 . 1 0 7 1 } \pm 0 . 0 2 0 3$ </td></tr><tr><td rowspan="4">Distributed Shampoo</td><td>F</td><td> $0 . 8 8 0 1 \pm 0 . 3 2 3 4$ </td><td> $2 . 6 9 2 2 \pm 0 . 5 6 7 4$ </td><td> $0 . 7 0 0 8 \pm 0 . 2 2 6 7$ </td></tr><tr><td>VN</td><td> $0 . 3 3 5 3 \pm 0 . 1 1 0 9$ </td><td> $0 . 6 2 0 7 \pm 0 . 1 9 6 6$ </td><td> $0 . 2 8 0 6 \pm 0 . 0 9 3 5$ </td></tr><tr><td>SQ</td><td> $0 . 1 2 7 3 \pm 0 . 0 2 8 5$ </td><td> $0 . 1 8 7 6 \pm 0 . 0 4 0 7$ </td><td> $0 . 1 2 5 7 \pm 0 . 0 3 0 0$ </td></tr><tr><td>KL</td><td> $\mathbf { 0 . 1 0 0 3 \ : \pm { \ : 0 . 0 1 7 7 } }$ </td><td> $\mathbf { 0 . 1 4 8 2 } \pm 0 . 0 2 4 9$ </td><td> $\mathbf { 0 . 1 0 7 2 } \pm 0 . 0 2 0 4$ </td></tr></table>

whereas KL weights errors uniformly across the spectrum and tends to overestimate; SQ lies in between. This yields larger eigenvalues for SQ and KL, which can reduce excessive inverse-root amplification during preconditioning and brings the gains closer to the held-out targets (more details in Appendix E.4).

Takeaway. KL, closely followed by SQ, best compensates for the finite-sample underestimation in the empirical second moment, keeping inverse-root gains closer to the desired scaling across the spectrum.

## 5.3 FEW-SAMPLE FITS AND ONLINE EMA

The large-sample behavior is useful in practice only if it can be recovered from short gradient histories. We therefore evaluate both few-sample fitting and online EMA using two metrics: recovery error, which measures how closely each method reproduces its own large-sample gains, and the held-out gain error from Section 5.2, which measures agreement with the held-out target.

Configuration. All four methods follow the EMA scheme of Lin et al. (2026), detailed in $\mathrm { A p \cdot }$ pendix C.5, with $\beta _ { 2 } = 0 . 9 5$ and eigendecomposition every five steps. Equal-weight fits use $N = 4 0$ gradients, close to the nominal EMA effective sample size $( 1 + \dot { \beta } _ { 2 } ) / ( \dot { 1 } - \beta _ { 2 } ) \stackrel { - } { = } 3 9 .$ <sup>1</sup> Each EMA repeat uses 512 gradients to reduce initialization effects. Table 8 summarizes these settings.

Recovery error. For this metric, the reference is each method’s large-sample fit $S _ { D } ^ { \mathrm { r e f } }$ . Let $\hat { S } _ { D }$ be the few-sample or EMA fit, and let $\{ \omega _ { D , j } , v _ { D , j } \} _ { j = 1 } ^ { d }$ be the eigenpairs of $S _ { D } ^ { \mathrm { r e f } }$ . We compare inverse-root amplification along the reference directions $v _ { D , j } \colon$

$$
\gamma _ { D , j } ^ { \mathrm { r e c } } : = \log _ { 1 0 } \frac { \| \widehat { S } _ { D } ^ { - 1 / 2 } v _ { D , j } \| _ { 2 } } { \| ( S _ { D } ^ { \mathrm { r e f } } ) ^ { - 1 / 2 } v _ { D , j } \| _ { 2 } } = \frac 1 2 \log _ { 1 0 } ( \omega _ { D , j } v _ { D , j } ^ { \top } \widehat { S } _ { D } ^ { - 1 } v _ { D , j } ) .\tag{12}
$$

Interpretation. The gain mismatch $\gamma _ { D , j } ^ { \mathrm { r e c } }$ is zero when the large-sample scaling is recovered; positive and negative values indicate stronger and weaker amplification, respectively. We also want to keep this error close to zero across the whole spectrum.

Table 3: Recovery error (Equation (12), ↓). KL has the lowest mean error, followed by SQ. Each experiment contributes the median $| \gamma _ { D } ^ { \mathrm { r e c } } |$ over all reference directions and repeats; entries show mean ± sample SD across nine checkpoint-projection pairs per training optimizer.
<table><tr><td rowspan="2">Divergence</td><td colspan="2">AdamW</td><td colspan="2">Distributed Shampoo</td></tr><tr><td>Few-sample</td><td>EMA</td><td>Few-sample</td><td>EMA</td></tr><tr><td>F</td><td> $1 . 6 6 8 \pm 0 . 6 0 1$ </td><td> $0 . 1 7 3 \pm 0 . 0 7 0$ </td><td> $1 . 9 6 7 \pm 0 . 4 3 0$ </td><td> $0 . 1 5 4 \pm 0 . 0 1 7$ </td></tr><tr><td>VN</td><td> $0 . 2 5 9 \pm 0 . 1 4 1$ </td><td> $0 . 0 8 2 \pm 0 . 0 3 2$ </td><td> $0 . 3 3 9 \pm 0 . 1 2 1$ </td><td> $0 . 0 8 7 \pm 0 . 0 1 7$ </td></tr><tr><td>SQ</td><td> $0 . 0 8 9 \pm 0 . 0 4 2$ </td><td> $0 . 0 6 3 \pm 0 . 0 2 0$ </td><td> $0 . 0 7 7 \pm 0 . 0 2 1$ </td><td> $0 . 0 4 8 \pm 0 . 0 0 8$ </td></tr><tr><td>KL</td><td> $\mathbf { 0 . 0 6 6 \ : \pm { \ : 0 . 0 2 0 } }$ </td><td> $\mathbf { 0 . 0 4 4 } \pm \ : 0 . 0 1 1$ </td><td> $\mathbf { 0 . 0 6 0 \ : \pm { \ : 0 . 0 1 5 } }$ </td><td> $\mathbf { 0 . 0 4 0 \ : \pm { \ : 0 . 0 0 9 } }$ </td></tr></table>

Finite-sample gain results. KL has the lowest recovery error, followed by SQ, VN, and F, across both 40-sample fits and EMA and both checkpoint sources (Table 3). The same ordering holds for aggregate held-out gain error (Table 2 and Figure 6, middle and right). Additional results in Appendix E.5 show the same trend.

Connecting Q1 and Q2. Q1 shows that SQ and KL put more emphasis on the entire spectrum and produce larger fitted tail eigenvalues. Q2 reveals two finite-sample effects: the empirical second moment underestimates the tail, while lower-q factor estimates are more stable with limited gradients. Together, these effects explain why SQ and KL better compensate for the empirical second moment bias and recover their large-sample gains more reliably.

Takeaway. Divergence choice and finite sampling jointly shape preconditioning. KL, closely followed by SQ, best compensates for tail underestimation and yields more reliable gains from short gradient histories.

## 6 GPT-2 TRAINING

We next test whether the gain differences observed above are reflected in GPT-2 training.

Prediction. More generally, we expect preconditioners whose inverse-root gains stay close to the held-out targets across the spectrum and remain reliable with short gradient histories to perform better in training. This suggests the held-out gain and recovery errors in Section 5 as practical metrics for evaluating optimizer designs. For the methods studied here, we therefore expect KL and SQ to perform similarly and better than F and VN, with a slight advantage for KL.

We compare F-, VN-, SQ-, and KL-Shampoo by their final validation losses after training 12-layer GPT-2 models on FineWeb at several widths. Across widths 256–1,536, all methods use 5,100 updates and spectral EMA $( \beta _ { 2 } = 0 . 9 5 )$ . Appendix F gives the full settings and training curves.
<table><tr><td>Method / width</td><td>256</td><td>512</td><td>768</td><td>1024</td><td>1536</td></tr><tr><td>F</td><td>3.8281</td><td>3.5082</td><td>3.3993</td><td>3.3528</td><td>3.3111</td></tr><tr><td>VN</td><td>3.7573</td><td>3.4377</td><td>3.2876</td><td>3.1979</td><td>3.0972</td></tr><tr><td>SQ</td><td>3.7450</td><td>3.4173</td><td>3.2661</td><td>3.1824</td><td>3.0817</td></tr><tr><td>KL</td><td>3.7475</td><td>3.4180</td><td>3.2660</td><td>3.1769</td><td>3.0758</td></tr></table>

Table 4: Final GPT-2 validation loss on FineWeb after 5, 100 updates (↓); one run per configuration. SQ and KL outperform F and VN at every width. Bold marks the lowest loss in each column.

SQ and KL achieve lower validation losses than F and VN at every width, consistent with this prediction (Figure 1, right, and Table 4). Their losses are similar at smaller widths, while KL achieves lower losses at the two largest widths.

## 7 CONCLUSION

We used a trace-power Bregman family to unify divergence-based Shampoo variants and connect SQ-Shampoo to PSGD-Kron, showing how divergence choice shapes Kronecker approximation error. We then identified two finite-sample effects: empirical second moments can underestimate tail eigenvalues, while divergence choice also affects the stability of Kronecker factor estimation. Together, these effects explain why SQ and KL better compensate for the underestimation of empirical second moments and recover their large-sample gains more reliably from short gradient histories. Consistently, SQ and KL achieve lower GPT-2 validation losses than F and VN. Our held-out gain and recovery errors provide practical metrics for evaluating new preconditioners.

## REFERENCES

Naman Agarwal, Rohan Anil, Elad Hazan, Tomer Koren, and Cyril Zhang. Disentangling adaptive gradient methods from learning rates. arXiv preprint arXiv:2002.11803, 2020.

Shun-Ichi Amari. Natural gradient works efficiently in learning. Neural Computation, 10(2):251– 276, 1998.

Shun-ichi Amari. Information geometry of positive measures and positive-definite matrices: Decomposable dually flat structure. Entropy, 16(4):2131–2145, 2014.

Kang An, Yuxing Liu, Rui Pan, Yi Ren, Shiqian Ma, Donald Goldfarb, and Tong Zhang. ASGO: Adaptive structured gradient optimization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, and Yoram Singer. Scalable second order optimization for deep learning. arXiv preprint arXiv:2002.09018, 2020.

Zhidong Bai and Jack W Silverstein. Spectral analysis of large dimensional random matrices, volume 20. Springer, 2010.

Daniel Bartz. Cross-validation based nonlinear shrinkage. arXiv preprint arXiv:1611.00798, 2016.

Andrzej Cichocki and Shun-ichi Amari. Families of alpha-beta-and gamma-divergences: Flexible and robust measures of similarities. Entropy, 12(6):1532–1568, 2010.

George E Dahl, Frank Schneider, Zachary Nado, Naman Agarwal, Chandramouli Shama Sastry, Philipp Hennig, Sourabh Medapati, Runa Eschenhagen, Priya Kasimbeg, Daniel Suo, et al. Benchmarking neural network training algorithms. arXiv preprint arXiv:2306.07179, 2023.

Inderjit S Dhillon and Joel A Tropp. Matrix nearness problems with bregman divergences. SIAM Journal on Matrix Analysis and Applications, 29(4):1120–1146, 2008.

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(61):2121–2159, 2011.

Pierre Dutilleul. The mle algorithm for the matrix normal distribution. Journal of Statistical Computation and Simulation, 64(2):105–123, 1999.

Runa Eschenhagen, Alexander Immer, Richard Turner, Frank Schneider, and Philipp Hennig. Kronecker-factored approximate curvature for modern neural network architectures. Advances in Neural Information Processing Systems, 36:33624–33655, 2023.

Runa Eschenhagen, Aaron Defazio, Tsung-Hsien Lee, Richard E. Turner, and Hao-Jun Michael Shi. Purifying shampoo: Investigating shampoo’s heuristics by decomposing its preconditioner. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Runa Eschenhagen, Anna Cai, Tsung-Hsien Lee, and Hao-Jun Michael Shi. Clarifying shampoo: Adapting spectral descent to stochasticity and the parameter trajectory. arXiv preprint arXiv:2602.09314, 2026.

Cole Franks, Rafael Oliveira, Akshay Ramachandran, and Michael Walter. Near optimal sample complexity for matrix and tensor normal models via geodesic convexity. The Annals ofStatistics, 54(1):93–119, 2026.

Kevin Frans, Sergey Levine, and Pieter Abbeel. A stable whitening optimizer for efficient neural network training. Advances in Neural Information Processing Systems, 38:174086–174110, 2025.

Kaixin Gao, Xiaolei Liu, Zhenghai Huang, Min Wang, Zidong Wang, Dachuan Xu, and Fan Yu. A trace-restricted kronecker-factored approximation to natural gradient. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 35, pp. 7519–7527, 2021.

Roger Grosse and James Martens. A kronecker-factored approximate fisher matrix for convolution layers. In International Conference on Machine Learning, pp. 573–582. PMLR, 2016.

Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization. In International Conference on Machine Learning, pp. 1842–1850. PMLR, 2018.

Tom Heskes. On “natural” learning and pruning in multilayered perceptrons. Neural Computation, 12(4):881–901, 2000.

Keller Jordan, Jeremy Bernstein, Brendan Rappazzo, @fernbear.bsky.social, Boza Vlado, You Jiacheng, Franz Cesista, Braden Koszarsky, and @Grad62304977. modded-nanogpt: Speedrunning the nanogpt baseline, 2024. URL https://github.com/KellerJordan modded-nanogpt.

Priya Kasimbeg, Frank Schneider, Runa Eschenhagen, Juhan Bae, Chandramouli Shama Sastry, Mark Saroufim, Boyuan Feng, Less Wright, Edward Z. Yang, Zachary Nado, Sourabh Medapati, Philipp Hennig, Michael Rabbat, and George E. Dahl. Accelerating neural network training: An analysis of the algoperf competition. In The Thirteenth International Conference on Learning Representations, 2025.

Juno Kim, Eshaan Nichani, Denny Wu, Alberto Bietti, and Jason D Lee. Sharp capacity scaling of spectral optimizers in learning associative memory. arXiv preprint arXiv:2603.26554, 2026.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

Brian Kulis, Maty´ as A Sustik, and Inderjit S Dhillon. Low-rank kernel learning with bregman matrix´ divergences. Journal ofMachine Learning Research, 10(13):341–376, 2009.

Frederik Kunstner, Lukas Balles, and Philipp Hennig. Limitations of the empirical fisher approximation for natural gradient descent. Advances in Neural Information Processing Systems, 32, 2019.

Olivier Ledoit and Michael Wolf. Nonlinear shrinkage estimation of large-dimensional covariance matrices. The Annals ofStatistics, 40(2):1024–1060, 2012. doi: 10.1214/12-AOS989.

Xi-Lin Li. Preconditioned stochastic gradient descent. IEEE Transactions on Neural Networks and Learning Systems, 29(5):1454–1466, 2018.

Xi-Lin Li. Preconditioner on matrix lie group for sgd. In International Conference on Learning Representations, 2019.

Wu Lin, Felix Dangel, Runa Eschenhagen, Kirill Neklyudov, Agustinus Kristiadi, Richard E Turner, and Alireza Makhzani. Structured inverse-free natural gradient descent: Memory-efficient & numerically-stable kfac. In Forty-first International Conference on Machine Learning, 2024.

Wu Lin, Scott C. Lowe, Felix Dangel, Runa Eschenhagen, Zikun Xu, and Roger Baker Grosse. Understanding and improving shampoo and SOAP via kullback-leibler minimization. In The Fourteenth International Conference on Learning Representations, 2026.

Bing Liu, Wenjie Zhou, and Chengcheng Zhao. Rethinking Bregman divergences in Kroneckerfactored optimizers. In ICML Workshop on High-dimensional Learning Dynamics, 2026.

James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International Conference on Machine Learning, pp. 2408–2417. PMLR, 2015.

James Martens, Jimmy Ba, and Matt Johnson. Kronecker-factored curvature approximations for recurrent neural networks. In International Conference on Learning Representations, 2018.

Andrew McCormack and Peter Hoff. Information geometry and asymptotics for kronecker covariances. Bernoulli, 31(4):3165–3186, 2025.

Depen Morwani, Itai Shapira, Nikhil Vyas, Eran Malach, Sham M Kakade, and Lucas Janson. A new perspective on shampoo’s preconditioner. In The Thirteenth International Conference on Learning Representations, 2025.

Yi Ren and Donald Goldfarb. Tensor normal training for deep learning models. Advances in Neural Information Processing Systems, 34:26040–26052, 2021.

Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, and Michael Rabbat. A distributed data-parallel pytorch implementation of the distributed shampoo optimizer for training neural networks at-scale. arXiv preprint arXiv:2309.06497, 2023.

Minhak Song, Kwangjun Ahn, and Chulhee Yun. Does sgd really happen in tiny subspaces? In 13th International Conference on Learning Representations. International Conference on Learning Representations (ICLR), 2025.

Lukas Nicola Tatzel, Balint Mucs ´ anyi, Osane Hackel, and Philipp Hennig. Debiasing mini-batch ´ quadratics for applications in deep learning. In International Conference on Learning Representations, volume 2025, pp. 41298–41337, 2025.

Valentin Thomas, Fabian Pedregosa, Bart Merrienboer, Pierre-Antoine Manzagol, Yoshua Bengio,¨ and Nicolas Le Roux. On the interplay between noise and curvature and its effect on optimization and generalization. In International Conference on Artificial Intelligence and Statistics, pp. 3503– 3513. PMLR, 2020.

Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham Kakade. Soap: Improving and stabilizing shampoo using adam for language modeling. In International Conference on Learning Representations, volume 2025, pp. 93423–93444, 2025.

Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, Cunxiao Du, Chao Du, Tianyu Pang, Zhuoran Yang, Mingyi Hong, and Vincent Tan. Muon outperforms adam in tail-end associative memory learning. In International Conference on Learning Representations, volume 2026, pp. 37382–37419, 2026.

Lei Wu, Mingze Wang, and Weijie Su. The alignment property of sgd noise and how it helps select flat minima: A stability analysis. Advances in Neural Information Processing Systems, 35:4680– 4693, 2022.

Ashley Zhang, Ben Keigwin, Dhruv Pai, and Alec Dewulf. Online kl shampoo. 2026. URL https://tilderesearch.com/blog/online-kl-shampoo.

Zhanxing Zhu, Jingfeng Wu, Bing Yu, Lei Wu, and Jinwen Ma. The anisotropic noise in stochastic gradient descent: Its behavior of escaping from sharp minima and regularization effects. In International Conference on Machine Learning, pp. 7654–7663. PMLR, 2019.

## APPENDIX CONTENTS

A Related Work 15   
B Proofs 16   
B.1 Proof of Proposition 1 . . . 16   
B.2 Limited Kronecker Capacity 18   
B.3 Spectral Error Decomposition and Kronecker Approximations 18   
B.3.1 Proof of Proposition 2 18   
B.3.2 Divergence-Specific Spectral Penalties . 19   
B.3.3 Reciprocal Asymmetry . 19   
B.3.4 Kronecker Approximations . 20   
B.4 Proof of Proposition 4 . . 20   
B.5 Local Geometry of SQ and KL 21   
C Preconditioner Update Details 22   
C.1 Original Shampoo . . . 22   
C.2 PSGD-Kron, SQ-Shampoo, and KL-Shampoo 22   
C.3 Divergence-Based Factor Statistics . 22   
C.4 Offline Fixed-Point Fits 23   
C.5 Online EMA Updates 23   
D Synthetic Validation 23   
D.1 Experimental Setup 23   
D.2 Metrics and Results . 24   
D.3 Directional Gain Interventions on a Quadratic 25   
E Real-Gradient Experimental Details 26   
E.1 Empirical Second Moments and Finite-Sample Effects 26   
E.1.1 Finite-Sample Tail Selection . 26   
E.1.2 High-Dimensional Sampling and Tail Signals . 27   
E.2 Reliability Across Checkpoints . 27   
E.3 Inverse-Root Gain Across Checkpoints . 28   
E.4 Kronecker Fits of Different Divergences 30   
E.5 Gain Recovery with Short Gradient Histories 31   
E.6 Diagnostics on Distributed Shampoo Checkpoints 33   
F GPT-2 Training Details and Full Results 34

## A RELATED WORK

Natural gradients and structured curvature. Natural gradient descent uses the Fisher metric to account for parameter-space geometry (Amari, 1998). Layered Fisher approximations (Heskes, 2000) and K-FAC (Martens & Grosse, 2015) make this approach tractable by exploiting network structure. Subsequent work extends Kronecker factorizations to convolutional layers (Grosse & Martens, 2016), recurrent networks (Martens et al., 2018), and general linear layers with weight sharing (Eschenhagen et al., 2023). Tensor Normal Training uses a tensor-normal model of sampled gradients to approximate the Fisher (Ren & Goldfarb, 2021), TKFAC imposes trace constraints (Gao et al., 2021), and SINGD incorporates inverse-free updates to avoid explicit matrix inversion (Lin et al., 2024).

Shampoo and Kronecker approximation. AdaGrad preconditions gradients using their accumulated second moments (Duchi et al., 2011), while Adam uses an exponential moving average of squared gradients for coordinate-wise scaling (Kingma & Ba, 2015). Shampoo exploits tensor structure to construct a Kronecker-factored bound on full-matrix AdaGrad (Gupta et al., 2018). Distributed computation, infrequent inverse-root updates, and learning-rate grafting make Shampoo practical at scale (Anil et al., 2020; Agarwal et al., 2020; Shi et al., 2023). SOAP applies Adam in Shampoo’s eigenbasis to improve training performance and stability (Vyas et al., 2025). Morwani et al. (2025) show that the Kronecker product of Shampoo’s statistics corresponds to one poweriteration step from identity initialization toward the Frobenius-optimal Kronecker approximation. In their experiments, its cosine similarity to the target closely tracks that of the optimal approximation. They also examine how batch gradients and label sampling affect the estimation.

Matrix divergences and Kronecker estimation. Bregman matrix divergences provide alternatives to Frobenius approximation (Dhillon & Tropp, 2008), with applications to positive-semidefinite kernel learning (Kulis et al., 2009). For Kronecker approximation, the LogDet (KL divergence be tween two zero-mean multivariate Gaussian distributions) objective corresponds to matrix-normal maximum likelihood, which estimates row and column covariance factors alternately (Dutilleul, 1999). Franks et al. (2026) establish nearly optimal finite-sample guarantees for joint maximum likelihood estimation without condition-number assumptions. McCormack & Hoff (2025) study partial-trace estimation, corresponding to VN fitting, and relate its statistical error and asymptotic efficiency loss to the factor spectra. Lin et al. (2026) formulate gradient second-moment estimation through matrix divergences, yielding F-, VN-, and KL-Shampoo. We study spectral error allocation and finite-sample stability across F, VN, SQ, and KL in a trace-power family. Our fixed-factor analysis is complemented by joint fits on synthetic non-Kronecker and real gradients.

Reliability of empirical second moments. Empirical gradient second moments can differ substantially from curvature matrices like the Fisher and Hessian (Kunstner et al., 2019; Thomas et al., 2020). Furthermore, with finite samples, empirical eigenvalues systematically misestimate population scales along their eigenvectors. Prior work addresses this bias through nonlinear covariance shrinkage or cross-validation (Ledoit & Wolf, 2012; Bartz, 2016) in high-dimensional covariance estimation. A similar selection bias occurs in deep-learning Hessian and GGN estimates: directions with extreme curvature on one mini-batch have less extreme scales on others (Tatzel et al., 2025). We examine these estimation effects in mini-batch gradient second moments and their consequences for divergence-dependent Kronecker preconditioning.

## B PROOFS

Throughout the theoretical analysis, following Eschenhagen et al. (2025); Lin et al. (2026), we assume the gradient second moment C to be positive definite.

Matrix notation and divergence formulas. For $M = U \mathrm { d i a g } ( \mu _ { 1 } , \dots , \mu _ { d } ) U ^ { \top } \in \mathbb { S } _ { + + } ^ { d }$ , matrix functions act on its eigenvalues:

$$
f ( M ) = U \mathrm { d i a g } ( f ( \mu _ { 1 } ) , \dots , f ( \mu _ { d } ) ) U ^ { \top } , \quad \Phi ( M ) = \sum _ { i = 1 } ^ { d } f ( \mu _ { i } ) , \quad \nabla \Phi ( M ) = f ^ { \prime } ( M ) .
$$

We use $\| A \| _ { F } ^ { 2 } = \langle A , A \rangle _ { \mathrm { F } }$ and write $\nabla ^ { 2 } \Phi ( { \cal S } ) [ H ]$ for the Hessian on a perturbation H. Substituting the generators $f _ { \mathrm { F } } ( x ) = x ^ { 2 } / 2 , f _ { \mathrm { V N } } ( x ) = x \log { x } - x$ , and $f _ { \mathrm { K L } } ( x ) = - \log x$ into Equation (2) gives

$$
\mathcal { B } _ { F } ( X , Y ) = \frac { 1 } { 2 } \| X - Y \| _ { F } ^ { 2 } , \quad \mathcal { B } _ { \mathrm { V N } } ( X , Y ) = \mathrm { T r } ( X \log X - X \log Y - X + Y ) ,
$$

$$
{ \mathcal { B } } _ { \mathrm { K L } } ( X , Y ) = \operatorname { T r } ( Y ^ { - 1 } X ) - \log \operatorname* { d e t } ( Y ^ { - 1 } X ) - d .
$$

In particular, ${ \mathcal { B } } _ { \mathrm { K L } } ( X , Y ) = 2 D _ { \mathrm { K L } } ( { \mathcal { N } } ( 0 , X ) \| { \mathcal { N } } ( 0 , Y ) )$ . For row-wise vectorization, $\operatorname { v e c } ( A G B ) =$ $( A ^ { ' } \otimes B ^ { \top } ) \operatorname { v e c } ( G )$

## B.1 PROOF OF PROPOSITION 1

For $q = 2 , f _ { 2 } ( x ) = ( x - 1 ) ^ { 2 } / 2$ and $f _ { 2 } ^ { \prime } ( x ) = x - 1$ . Substitution into Equation (2) gives

$$
\mathcal { B } _ { \Phi _ { 2 } } ( X , Y ) = \frac { 1 } { 2 } \operatorname { T r } ( X ^ { 2 } - Y ^ { 2 } ) - \operatorname { T r } ( Y ( X - Y ) ) = \frac { 1 } { 2 } \| X - Y \| _ { F } ^ { 2 } = \mathcal { B } _ { F } ( X , Y ) .
$$

L’Hopital’s rule givesˆ $f _ { q } ( x ) $ x log $x - x + 1$ as $q \to 1$ and $f _ { q } ( x )  - \log x + x - 1$ as $q  0$ The derivative $f _ { q } ^ { \prime } ( x ) = \bar { ( } x ^ { q - 1 } - 1 ) / ( q - 1 )$ converges to log x and $1 - x ^ { - 1 }$ , respectively. For fixed $X , Y \in \mathbb { S } _ { + + } ^ { d }$ , we can obtain

$$
\operatorname* { l i m } _ { q \to 1 } \mathcal { B } _ { \Phi _ { q } } ( X , Y ) = \mathcal { B } _ { \Phi _ { 1 } } ( X , Y ) = \operatorname { T r } ( X \log X - X \log Y - X + Y ) = \mathcal { B } _ { \operatorname { V N } } ( X , Y ) ,
$$

$$
\operatorname* { l i m } _ { q \to 0 } \mathcal { B } _ { \Phi _ { q } } ( X , Y ) = \mathcal { B } _ { \Phi _ { 0 } } ( X , Y ) = \operatorname { T r } ( Y ^ { - 1 } X ) - \log \operatorname* { d e t } ( Y ^ { - 1 } X ) - d = \mathcal { B } _ { \mathrm { K L } } ( X , Y ) .
$$

For $\begin{array} { r } { q = \frac { 1 } { 2 } } \end{array}$ , which we call square root (SQ), $f _ { 1 / 2 } ( x ) = - 4 \sqrt { x } + 2 x + 2$ and $f _ { 1 / 2 } ^ { \prime } ( x ) = 2 - 2 x ^ { - 1 / 2 }$ Substituting into Equation (2) yields

$$
{ \frac { 1 } { 2 } } { \mathcal { B } } _ { \Phi _ { 1 / 2 } } ( C , S ) = \operatorname { T r } ( S ^ { 1 / 2 } ) + \operatorname { T r } ( C S ^ { - 1 / 2 } ) - 2 \operatorname { T r } ( C ^ { 1 / 2 } ) .\tag{13}
$$

Under $P = { S } ^ { - 1 / 2 }$ , this becomes

$$
{ \frac { 1 } { 2 } } { \mathcal { B } } _ { \Phi _ { 1 / 2 } } ( C , S ) = \operatorname { T r } ( P C ) + \operatorname { T r } ( P ^ { - 1 } ) - 2 \operatorname { T r } ( C ^ { 1 / 2 } ) .
$$

The first two terms are exactly the PSGD objective in Equation (1); the last term is independent of P. Thus SQ and PSGD have the same objective up to a positive scale and an additive constant. For $S = L \otimes R ,$ , the change of variables gives $P = L ^ { - 1 / 2 } \otimes R ^ { - 1 / 2 }$ <sup>2</sup>, so the correspondence also holds under the Kronecker constraint.

We now derive the factor equation in Equation (4). For $q \neq 0 , 1$ ，

$$
\nabla \Phi _ { q } ( S ) = \frac { S ^ { q - 1 } - I } { q - 1 } .
$$

By substituting this into the Bregman divergence and keeping only the terms that depend on S, we can derive

$$
{ \mathcal B } _ { \Phi _ { q } } ( C , S ) = \frac 1 q \mathrm { T r } ( S ^ { q } ) - \frac { 1 } { q - 1 } \mathrm { T r } ( C S ^ { q - 1 } ) + \mathrm { c o n s t . }
$$

For $S = L \otimes R ,$

$$
\mathrm { T r } ( { \cal S } ^ { q } ) = \mathrm { T r } ( { \cal L } ^ { q } ) \mathrm { T r } ( R ^ { q } ) ,
$$

and the definition of $C = \mathbb { E } [ g g ^ { \top } ]$ gives

$$
\operatorname { T r } ( C S ^ { q - 1 } ) = \mathbb { E } \left[ \operatorname { T r } \left( L ^ { q - 1 } G R ^ { q - 1 } G ^ { \top } \right) \right] .
$$

Fix R, and write

$$
A _ { R } : = \mathbb { E } [ G R ^ { q - 1 } G ^ { \top } ] , \quad a _ { R } : = \operatorname { T r } ( R ^ { q } ) .
$$

Combining all the equations above, we can write the objective $B _ { \Phi _ { q } }$ that depends on L as

$$
J _ { q } ( L ; R ) = \frac { a _ { R } } { q } \operatorname { T r } ( L ^ { q } ) - \frac { 1 } { q - 1 } \operatorname { T r } ( A _ { R } L ^ { q - 1 } ) .
$$

For $q \neq 0 , 1$ 1, let $X = L ^ { q - 1 }$ . Then $L ^ { q } = X ^ { q / q - 1 }$ , and

$$
\nabla _ { X } J _ { q } = { \frac { 1 } { q - 1 } } \left( a _ { R } X ^ { 1 / q - 1 } - A _ { R } \right) = { \frac { 1 } { q - 1 } } ( a _ { R } L - A _ { R } ) .
$$

Thus the stationary equation gives $L = A _ { R } / a _ { R } = \mathbb { E } [ G R ^ { q - 1 } G ^ { \top } ] / \operatorname { T r } ( R ^ { q } )$ . The objective is strictly convex in X, so this stationary point is the unique minimizer.

For the VN defined by $q \to 1 , \Phi _ { 1 } ( S ) = \operatorname { T r } ( S \log S - S + I )$ . The limit established above gives

$$
B _ { \Phi _ { 1 } } ( C , S ) = \mathrm { T r } ( S ) - \mathrm { T r } ( C \log S ) + \mathrm { c o n s t . }
$$

Using

$$
\mathrm { T r } ( L \otimes R ) = \mathrm { T r } ( L ) \mathrm { T r } ( R ) , \qquad \log ( L \otimes R ) = \log L \otimes I _ { n } + I _ { m } \otimes \log R ,
$$

and the row-wise vectorization identity

$$
\operatorname { v e c } ( G ) ^ { \top } ( \log L \otimes I _ { n } ) \operatorname { v e c } ( G ) = \operatorname { T r } ( G G ^ { \top } \log L ) ,
$$

the terms depending on L reduce to

$$
J _ { 1 } ( L ; R ) = \operatorname { T r } ( R ) \operatorname { T r } ( L ) - \operatorname { T r } ( A _ { R } \log L ) , \qquad A _ { R } : = \mathbb { E } [ G G ^ { \top } ] .
$$

Here R is fixed, so the term involving $I _ { m } \otimes$ log R is constant in L. Set $X = \log L$ . Then

$$
J _ { 1 } ( e ^ { X } ; R ) = \mathrm { T r } ( R ) \mathrm { T r } ( e ^ { X } ) - \mathrm { T r } ( A _ { R } X ) ,
$$

whose gradient with respect to X is

$$
\nabla _ { X } J _ { 1 } = \mathrm { T r } ( { \cal R } ) e ^ { X } - A _ { R } .
$$

Setting this to zero gives $L = e ^ { X } = A _ { R } / \operatorname { T r } ( R )$ . Since $\operatorname { T r } ( e ^ { X } )$ is strictly convex on symmetric matrices, this is the unique minimizer.

For the KL defined by $q  0 , \Phi _ { 0 } ( S ) = - \log \operatorname * { d e t } S + \mathrm { T r } ( S ) - m n$ . Its affine terms cancel in the divergence, giving

$$
\mathcal { B } _ { \Phi _ { 0 } } ( C , S ) = \mathrm { T r } ( C S ^ { - 1 } ) + \log \operatorname* { d e t } S + \mathrm { c o n s t } .
$$

Using $S ^ { - 1 } = L ^ { - 1 } \otimes R ^ { - 1 }$ and log det $( L \otimes R ) = n$ log det $L + m$ log det R, the terms that depend on L are

$$
\begin{array} { r } { J _ { 0 } ( L ; R ) = \operatorname { T r } ( A _ { R } L ^ { - 1 } ) + n \log \operatorname* { d e t } L , \quad A _ { R } : = \mathbb { E } [ G R ^ { - 1 } G ^ { \top } ] . } \end{array}
$$

Setting $X = L ^ { - 1 }$ , we can derive:

$$
\begin{array} { r } { J _ { 0 } \big ( X ^ { - 1 } ; R \big ) = \operatorname { T r } \big ( A _ { R } X \big ) - n \log \operatorname* { d e t } X , \qquad \nabla _ { X } J _ { 0 } = A _ { R } - n X ^ { - 1 } . } \end{array}
$$

Setting the gradient to zero gives $X ^ { - 1 } = A _ { R } / n$ , hence

$$
L = { \frac { A _ { R } } { n } } = { \frac { \mathbb { E } [ G R ^ { - 1 } G ^ { \top } ] } { \operatorname { T r } ( R ^ { 0 } ) } } .
$$

Since − log det X is strictly convex, this is the unique minimizer.

The cases $q = 1$ and $q = 0$ give the same formula, so all cases can be written as

$$
L _ { q } ( R ) = \frac { \mathbb { E } [ G R ^ { q - 1 } G ^ { \top } ] } { \operatorname { T r } ( R ^ { q } ) } .
$$

Exchanging L and R gives

$$
R _ { q } ( L ) = \frac { \mathbb { E } [ G ^ { \top } L ^ { q - 1 } G ] } { \operatorname { T r } ( L ^ { q } ) } .
$$

## B.2 LIMITED KRONECKER CAPACITY

A single Kronecker product generally cannot represent a gradient second moment exactly. The factors L and R have m $( m + 1 ) / 2 + n ( n + 1 ) / 2 - 1$ free parameters after accounting for reciprocal rescaling, fewer than the $m n ( \dot { m } n + 1 ) / 2$ parameters of a general symmetric matrix when $m , n \geq 2 .$ Exact matching is possible for second moments with Kronecker structure; otherwise, the choice of divergence determines how approximation errors are allocated.

## B.3 SPECTRAL ERROR DECOMPOSITION AND KRONECKER APPROXIMATIONS

## B.3.1 PROOF OF PROPOSITION 2

Notation and spectral costs. Write $U = [ u _ { 1 } , \dots , u _ { d } ] , V = [ v _ { 1 } , \dots , v _ { d } ] , \Lambda = \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { d } )$ and $\boldsymbol { \Omega } = \mathrm { d i a g } ( \omega _ { 1 } , \dots , \omega _ { d } )$ , with both spectra ordered decreasingly. The scalar divergence and cost on eigenspace misalignment are

$$
\begin{array} { l } { { d _ { f } ( \lambda , \omega ) : = f ( \lambda ) - f ( \omega ) - f ^ { \prime } ( \omega ) ( \lambda - \omega ) , } } \\ { { \displaystyle \Delta _ { f } ( \Lambda , \Omega ; P ) : = - \sum _ { i , j = 1 } ^ { d } \lambda _ { i } f ^ { \prime } ( \omega _ { j } ) ( P _ { i j } - \delta _ { i j } ) , \quad P _ { i j } = \langle u _ { i } , v _ { j } \rangle ^ { 2 } , } } \end{array}
$$

where $\delta _ { i j }$ is the Kronecker delta.

Spectral decomposition. By the definition of the Bregman matrix divergence,

$$
{ \mathcal { B } } _ { \Phi } ( A , B ) = \Phi ( A ) - \Phi ( B ) - \operatorname { T r } \left( \nabla \Phi ( B ) ( A - B ) \right) .\tag{14}
$$

Since $\Phi ( M ) = \operatorname { T r } ( f ( M ) )$ ) is a spectral function, its gradient is given by $\nabla \Phi ( M ) = f ^ { \prime } ( M )$ . Therefore, for $B = V \Omega V ^ { \top }$

$$
\begin{array} { r } { \nabla \Phi ( B ) = V f ^ { \prime } ( \Omega ) V ^ { \top } . } \end{array}
$$

Let $Q = U ^ { \top } V$ . Then $Q$ is orthogonal and $Q _ { i j } = \langle u _ { i } , v _ { j } \rangle$ , so that $P _ { i j } = Q _ { i j } ^ { 2 }$ . Using the cyclic property of the trace,

$$
\mathrm { T r } \left( \nabla \Phi ( B ) A \right) = \mathrm { T r } \left( V f ^ { \prime } ( \Omega ) V ^ { \top } U \Lambda U ^ { \top } \right) = \mathrm { T r } \left( f ^ { \prime } ( \Omega ) Q ^ { \top } \Lambda Q \right) = \sum _ { i = 1 } ^ { d } \sum _ { j = 1 } ^ { d } \lambda _ { i } f ^ { \prime } ( \omega _ { j } ) P _ { i j } .\tag{15}
$$

Similarly, using $V ^ { \top } V = I$

$$
\operatorname { T r } \left( \nabla \Phi ( B ) B \right) = \operatorname { T r } \left( V f ^ { \prime } ( \Omega ) V ^ { \top } V \Omega V ^ { \top } \right) = \operatorname { T r } \left( f ^ { \prime } ( \Omega ) V ^ { \top } V \Omega V ^ { \top } V \right) = \sum _ { j = 1 } ^ { d } \omega _ { j } f ^ { \prime } ( \omega _ { j } ) .\tag{16}
$$

Substituting Equations (15) and (16) into Equation (14) gives

$$
\mathcal { B } _ { \Phi } ( A , B ) = \sum _ { i = 1 } ^ { d } f ( \lambda _ { i } ) - \sum _ { j = 1 } ^ { d } f ( \omega _ { j } ) + \sum _ { j = 1 } ^ { d } \omega _ { j } f ^ { \prime } ( \omega _ { j } ) - \sum _ { i = 1 } ^ { d } \sum _ { j = 1 } ^ { d } \lambda _ { i } f ^ { \prime } ( \omega _ { j } ) P _ { i j } .
$$

Adding and subtracting $\scriptstyle \sum _ { i = 1 } ^ { d } \lambda _ { i } f ^ { \prime } ( \omega _ { i } )$ yields

$$
\begin{array} { l } { \displaystyle \mathcal { B } _ { \Phi } ( A , B ) = \sum _ { i = 1 } ^ { d } [ f ( \lambda _ { i } ) - f ( \omega _ { i } ) - f ^ { \prime } ( \omega _ { i } ) ( \lambda _ { i } - \omega _ { i } ) ] + \sum _ { i = 1 } ^ { d } \lambda _ { i } f ^ { \prime } ( \omega _ { i } ) - \sum _ { i = 1 } ^ { d } \sum _ { j = 1 } ^ { d } \lambda _ { i } f ^ { \prime } ( \omega _ { j } ) P _ { i j } } \\ { \displaystyle \quad = \sum _ { i = 1 } ^ { d } d _ { f } ( \lambda _ { i } , \omega _ { i } ) + \sum _ { i = 1 } ^ { d } \sum _ { j = 1 } ^ { d } - \lambda _ { i } f ^ { \prime } ( \omega _ { j } ) \left( P _ { i j } - \delta _ { i j } \right) } \\ { \displaystyle \quad = \sum _ { i = 1 } ^ { d } d _ { f } ( \lambda _ { i } , \omega _ { i } ) + \Delta _ { f } ( \Lambda , \Omega ; P ) . } \end{array}
$$

The first term measures eigenvalue mismatch for aligned eigenspaces; $\Delta _ { f }$ is the cost of eigenspace misalignment.

Nonnegativity of $\Delta _ { f } ( \Lambda , \Omega ; P )$ . Since $Q$ is orthogonal, $P _ { i j } \ = \ Q _ { i j } ^ { 2 }$ is doubly stochastic. The Birkhoff–von Neumann theorem therefore gives $\begin{array} { r } { P = \sum _ { k } \alpha _ { k } \Pi _ { k } } \end{array}$ , where $\Pi _ { k }$ are permutation matrices, $\alpha _ { k } \geq 0$ , and $\textstyle \sum _ { k } \alpha _ { k } = 1$ . Both $\lambda _ { i }$ and ${ \bar { f } } ^ { \prime } ( \omega _ { i } )$ are decreasing, since the eigenvalues are ordered and f is convex. The rearrangement inequality says that pairing them in this order maximizes their product sum. Thus, for each permutation $\sigma _ { k }$ associated with $\Pi _ { k }$

$$
\sum _ { i } \lambda _ { i } f ^ { \prime } ( \omega _ { \sigma _ { k } ( i ) } ) \leq \sum _ { i } \lambda _ { i } f ^ { \prime } ( \omega _ { i } ) .
$$

Taking the weighted average gives

$$
\sum _ { i , j } \lambda _ { i } f ^ { \prime } ( \omega _ { j } ) P _ { i j } \leq \sum _ { i } \lambda _ { i } f ^ { \prime } ( \omega _ { i } ) ,
$$

which proves $\Delta _ { f } ( \Lambda , \Omega ; P ) \ge 0$

## B.3.2 DIVERGENCE-SPECIFIC SPECTRAL PENALTIES

We overlook the affine terms. For Frobenius, $f _ { \mathrm { F } } ( x ) = { \textstyle \frac { 1 } { 2 } } x ^ { 2 }$ and $f _ { \mathrm { F } } ^ { \prime } ( x ) = x$ , which gives

$$
d _ { \mathrm { F } } ( \lambda , \omega ) = \frac { 1 } { 2 } ( \lambda - \omega ) ^ { 2 } .
$$

For von Neumann, $f _ { \mathrm { V N } } ( x ) = x \log x - x$ and $f _ { \mathrm { V N } } ^ { \prime } ( x ) = \log x .$ , and hence

$$
d _ { \mathrm { V N } } ( \lambda , \omega ) = \lambda \log \frac { \lambda } { \omega } - \lambda + \omega .
$$

For the square-root member, $f _ { \mathrm { S Q } } ( x ) = - 4 \sqrt { x }$ and $f _ { \mathrm { S Q } } ^ { \prime } ( x ) = - 2 x ^ { - 1 / 2 }$ , giving

$$
d _ { \mathrm { S Q } } ( \lambda , \omega ) = 2 \left( \sqrt { \omega } + \frac { \lambda } { \sqrt { \omega } } - 2 \sqrt { \lambda } \right) .
$$

For LogDet, $f _ { \mathrm { K L } } ( x ) = - \log x$ and $f _ { \mathrm { K L } } ^ { \prime } ( x ) = - 1 / x$ , which yields

$$
d _ { \mathrm { K L } } ( \lambda , \omega ) = \log \frac { \omega } { \lambda } + \frac { \lambda } { \omega } - 1 .
$$

Setting $\omega = r \lambda$ gives Equation (6), which establishes the scalar scale dependence in Equation (5). By using the exponents $q _ { \mathrm { D } }$ in Proposition 1, Frobenius and KL give

$$
\Delta _ { \mathrm { F } } ( r \Lambda , r \Omega ; P ) = r ^ { 2 } \Delta _ { \mathrm { F } } ( \Lambda , \Omega ; P ) , \qquad \Delta _ { \mathrm { K L } } ( r \Lambda , r \Omega ; P ) = \Delta _ { \mathrm { K L } } ( \Lambda , \Omega ; P ) .
$$

For von Neumann, using $\begin{array} { r } { \sum _ { j = 1 } ^ { d } ( P _ { i j } - \delta _ { i j } ) = 0 } \end{array}$ , we obtain

$$
\begin{array} { l } { { \displaystyle \Delta _ { \mathrm { V N } } ( r \Lambda , r \Omega ; P ) = - r \sum _ { i = 1 } ^ { d } \sum _ { j = 1 } ^ { d } \lambda _ { i } \log ( r \omega _ { j } ) \left( P _ { i j } - \delta _ { i j } \right) } } \\ { ~ } \\ { { \displaystyle = - r \log r \sum _ { i = 1 } ^ { d } \lambda _ { i } \sum _ { j = 1 } ^ { d } \left( P _ { i j } - \delta _ { i j } \right) + r \Delta _ { \mathrm { V N } } ( \Lambda , \Omega ; P ) } } \\ { ~ } \\ { { \displaystyle = r \Delta _ { \mathrm { V N } } ( \Lambda , \Omega ; P ) } . } \end{array}
$$

For SQ, $f _ { \mathrm { S O } } ^ { \prime } ( \omega ) = - 2 \omega ^ { - 1 / 2 }$ gives $\Delta _ { \mathrm { S Q } } ( r \Lambda , r \Omega ; P ) = r ^ { 1 / 2 } \Delta _ { \mathrm { S Q } } ( \Lambda , \Omega ; P )$ directly. Therefore, for $\mathrm { D } \in \{ \mathrm { F } , \mathrm { V N } , \mathrm { S Q } , \mathrm { K L } \}$

$$
\Delta _ { \mathrm { D } } ( r \Lambda , r \Omega ; P ) = r ^ { q _ { \mathrm { D } } } \Delta _ { \mathrm { D } } ( \Lambda , \Omega ; P ) , \quad r > 0 .
$$

## B.3.3 RECIPROCAL ASYMMETRY

Substituting r and $r ^ { - 1 }$ into Equation (6) proves Equation (7). For $\mathrm { F } , ( r - 1 ) ^ { 2 } > ( r ^ { - 1 } - 1 ) ^ { 2 }$ when $r > 1 ;$ for SQ, the two penalties are equal. For VN and KL, the differences are $\lambda h ( r ) { \mathrm { ~ a n d } } - h ( r )$ respectively, where $h ( r ) : = r - r ^ { - 1 } - 2 \log r$ . Since $h ( 1 ) = 0$ and $h ^ { \prime } ( r ) = ( r - \mathrm { i } ) ^ { 2 } / r ^ { 2 } > 0$ for $r > 1 , \mathsf { V N }$ penalizes overestimation more, while KL penalizes underestimation more.

## B.3.4 KRONECKER APPROXIMATIONS

Let $C \in \mathbb { S } _ { + + } ^ { m n }$ have eigenpairs $( \lambda _ { i } , u _ { i } )$ ordered by decreasing eigenvalue, and let L and R have eigenpairs $( \dot { \ell } _ { a } , v _ { L , a } ) , a = 1 , \dots , m$ , and $( \rho _ { b } , v _ { R , b } ) , b = 1 , . . . , n$ , respectively. Sort the mn eigenvalue products $\ell _ { a } \rho _ { b }$ in decreasing order. If the j-th product comes from the a-th eigenvalue of L and the b-th eigenvalue of $R ,$ write $a _ { j } = a$ and $b _ { j } = b .$ The corresponding eigenpair of $S = L \otimes R$ is

$$
\omega _ { j } = \ell _ { a _ { j } } \rho _ { b _ { j } } , \qquad v _ { j } = v _ { L , a _ { j } } \otimes v _ { R , b _ { j } } , \qquad j = 1 , \ldots , m n .
$$

Set $\Lambda = \mathrm { d i a g } ( \lambda _ { i } ) , \Omega = \mathrm { d i a g } ( \omega _ { j } )$ , and $P _ { i j } = | \langle u _ { i } , v _ { j } \rangle | ^ { 2 }$ as in Proposition 2. We can obtain:

$$
\mathcal { B } _ { \Phi _ { q } } ( C , L \otimes R ) = \underbrace { \sum _ { i = 1 } ^ { m n } d _ { q } ( \lambda _ { i } , \ell _ { a _ { i } } \rho _ { b _ { i } } ) } _ { \mathrm { c i g e n v a l u e ~ m i s m a t c h } } + \underbrace { \vphantom { \sum _ { i = 1 } ^ { m } } \Delta _ { q } ( \Lambda , \Omega ; P ) } _ { \mathrm { e i g e n s p a c e ~ m i s a l i g n m e n t } } , \quad \Delta _ { q } ( \Lambda , \Omega ; P ) \geq 0 .
$$

For Kronecker approximations, both mismatch terms retain the scaling relations in Equation $( 5 )$ , and the scalar penalties still satisfy the reciprocal-error comparisons in Equation (7).

## B.4 PROOF OF PROPOSITION 4

We only prove the result for the left factor; the right-factor result follows symmetrically. Let $A : =$ $R _ { \star } ^ { q _ { \mathrm { D } } } , \widetilde { L } _ { \mathrm { D } } : = L _ { \star } ^ { - 1 / 2 } \widehat { L } _ { \mathrm { D } } L _ { \star } ^ { - 1 / 2 }$ . By Equation (10), we have

$$
\widetilde L _ { \mathrm { D } } = \frac { 1 } { N } \sum _ { b = 1 } ^ { N } M _ { b } , \quad M _ { b } : = \frac { Z _ { b } A Z _ { b } ^ { \top } } { \operatorname { T r } ( A ) } .
$$

Let $z _ { i } ^ { \top }$ denote the i-th row of $Z _ { b }$ , so $z _ { i } \sim \mathcal { N } ( 0 , I _ { n } )$ . Then

$$
\mathbb { E } \left[ ( Z _ { b } A Z _ { b } ^ { \top } ) _ { i j } \right] = \mathbb { E } [ z _ { i } ^ { \top } A z _ { j } ] = { \left\{ \begin{array} { l l } { { \mathrm { T r } } ( A ) , } & { i = j , } \\ { 0 , } & { i \neq j , } \end{array} \right. }
$$

where we use $\mathbb { E } [ z _ { i } z _ { i } ^ { \top } ] = I$ and the independence of different rows. Hence, $\mathbb { E } [ Z _ { b } A Z _ { b } ^ { \top } ] = \operatorname { T r } ( A ) I$ and $\mathbb { E } [ M _ { b } ] = I .$ Using $\widetilde { L } _ { \mathrm { D } } - I = 1 / N \sum _ { b = 1 } ^ { N } ( M _ { b } - I )$ , we obtain

$$
\mathbb { E } \left[ \left. \widetilde { L } _ { \mathrm { D } } - I \right. _ { F } ^ { 2 } \right] = \frac { 1 } { N ^ { 2 } } \sum _ { b , c = 1 } ^ { N } \mathbb { E } \left[ \langle M _ { b } - I , M _ { c } - I \rangle _ { \mathrm { F } } \right] .
$$

For $b \neq c ,$ , independence together with $\mathbb { E } [ M _ { b } - I ] = 0$ gives

$$
\mathbb { E } \left[ \langle M _ { b } - I , M _ { c } - I \rangle _ { \mathrm { F } } \right] = 0 .
$$

Hence, only the N diagonal terms remain, yielding

$$
\mathbb { E } \left[ \left. \widetilde { L } _ { \mathrm { D } } - I \right. _ { F } ^ { 2 } \right] = \frac { 1 } { N } \mathbb { E } \left[ \left. M _ { 1 } - I \right. _ { F } ^ { 2 } \right] .\tag{17}
$$

Let $z _ { i } ^ { \top }$ denote the i-th row of $Z _ { 1 }$ . For the diagonal entries, $\begin{array} { r } { ( M _ { 1 } ) _ { i i } = \frac { z _ { i } ^ { \top } A z _ { i } } { \operatorname { T r } ( A ) } } \end{array}$ . Since $z _ { i } \sim \mathcal { N } ( 0 , I _ { n } )$ and A is symmetric, the Gaussian quadratic-form identity $\mathrm { V a r } ( z _ { i } ^ { \top } A z _ { i } ) = 2 \mathrm { T r } ( A ^ { 2 } )$ gives

$$
\mathbb { E } \left[ \left( ( M _ { 1 } ) _ { i i } - 1 \right) ^ { 2 } \right] = \frac { 2 \operatorname { T r } ( A ^ { 2 } ) } { \operatorname { T r } ( A ) ^ { 2 } } .
$$

For $i \neq j$ , using the independence of $z _ { i }$ and $z _ { j }$ and $\mathbb { E } [ z _ { j } z _ { j } ^ { \top } ] = I$

$$
\begin{array} { r } { \mathbb { E } \left[ ( z _ { i } ^ { \top } A z _ { j } ) ^ { 2 } \right] = \mathbb { E } \left[ z _ { i } ^ { \top } A \mathbb { E } [ z _ { j } z _ { j } ^ { \top } ] A z _ { i } \right] = \operatorname { T r } ( A ^ { 2 } ) , } \end{array}
$$

and therefore

$$
\mathbb { E } \left[ ( M _ { 1 } ) _ { i j } ^ { 2 } \right] = \frac { \operatorname { T r } ( A ^ { 2 } ) } { \operatorname { T r } ( A ) ^ { 2 } } .
$$

Summing over the m diagonal and $m ( m - 1 )$ off-diagonal entries yields

$$
\mathbb { E } \left[ \left. M _ { 1 } - I \right. _ { F } ^ { 2 } \right] = m ( m + 1 ) \frac { \mathrm { T r } ( A ^ { 2 } ) } { \mathrm { T r } ( A ) ^ { 2 } } = \frac { m ( m + 1 ) } { r _ { \mathrm { e f f } } ( A ) } .
$$

Combining this with Equation (17) and dividing by m gives

$$
\mathbb { E } \left[ \frac { 1 } { m } \left. L _ { \star } ^ { - 1 / 2 } \widehat { L } _ { \mathrm { D } } L _ { \star } ^ { - 1 / 2 } - I \right. _ { F } ^ { 2 } \right] = \frac { m + 1 } { N r _ { \mathrm { e f f } } ( R _ { \star } ^ { q _ { \mathrm { D } } } ) } .
$$

The right-factor identity follows by exchanging $L _ { \star } , m$ with $R _ { \star } , n$

It remains to compare the four exponents $q _ { \mathrm { D } }$ . For positive eigenvalues $\{ \rho _ { i } \}$ , define $h ( q ) : =$ log $\textstyle \sum _ { i } \rho _ { i } ^ { q }$ and $p _ { i } ( \boldsymbol { \dot { q } } ) : = \rho _ { i } ^ { q } / \sum _ { j } \rho _ { j } ^ { q }$ . Then

$$
h ^ { \prime \prime } ( q ) = \sum _ { i } p _ { i } ( q ) ( \log \rho _ { i } ) ^ { 2 } - \left( \sum _ { i } p _ { i } ( q ) \log \rho _ { i } \right) ^ { 2 } \geq 0 ,
$$

so $h ^ { \prime }$ is non-decreasing. Noting that the log effective rank is log $r _ { \mathrm { e f f } } ( R _ { \star } ^ { q } ) = 2 h ( q ) - h ( 2 q )$ , we have

$$
\frac { d } { d q } \log r _ { \mathrm { e f f } } ( R _ { \star } ^ { q } ) = 2 h ^ { \prime } ( q ) - 2 h ^ { \prime } ( 2 q ) \leq 0 , \quad q \geq 0 .
$$

Thus effective rank is decreasing in $q _ { D }$ . Evaluating at $q = q _ { \mathrm { D } }$ gives

$$
r _ { \mathrm { e f f } } ( R _ { \star } ^ { 2 } ) \leq r _ { \mathrm { e f f } } ( R _ { \star } ) \leq r _ { \mathrm { e f f } } ( R _ { \star } ^ { 1 / 2 } ) \leq r _ { \mathrm { e f f } } ( I ) = n .
$$

Since the expected estimation error is proportional to the reciprocal of effective rank, we obtain

$$
\mathrm { E r r } _ { \mathrm { K L } } \leq \mathrm { E r r } _ { \mathrm { S Q } } \leq \mathrm { E r r } _ { \mathrm { V N } } \leq \mathrm { E r r } _ { \mathrm { F } } .
$$

The inequalities become equalities for an isotropic opposite-factor spectrum.

## B.5 LOCAL GEOMETRY OF SQ AND KL

We compare the local penalties of SQ and KL through their Hessians. KL matches the Gaussian Fisher geometry up to a constant factor; SQ does not.

For $S \in \mathbb { S } _ { + + } ^ { d }$ and symmetric H, Taylor expansion gives

$$
\mathcal { B } _ { \Phi _ { q _ { \mathrm { D } } } } ( S + \varepsilon H , S ) = \frac { \varepsilon ^ { 2 } } { 2 } \left. H , \nabla ^ { 2 } \Phi _ { q _ { \mathrm { D } } } ( S ) [ H ] \right. _ { \mathrm { F } } + o ( \varepsilon ^ { 2 } ) .
$$

Thus the Hessian determines the penalty for small perturbations. For eigenvalue perturbations, $f _ { \boldsymbol { q } _ { \mathrm { D } } } ^ { \prime \prime } ( s ) = s ^ { q _ { \mathrm { D } } - 2 }$ , giving $s ^ { - 3 / 2 }$ for $\mathrm { S Q }$ and $s ^ { - 2 }$ for $\mathrm { K L }$

To compare this geometry with Gaussian Fisher information, consider the auxiliary model $x \sim$ $\mathcal { N } ( 0 , S )$ with density function $p _ { S }$ . The directional score for a symmetric perturbation H is

$$
s _ { H } ( x ) : = D _ { S } \log p _ { S } ( x ) [ H ] = \frac { 1 } { 2 } \left( x ^ { \top } S ^ { - 1 } H S ^ { - 1 } x - \mathrm { T r } ( S ^ { - 1 } H ) \right) .
$$

Its second moment gives the Fisher quadratic form:

$$
\mathbb { E } _ { { x } \sim { p } _ { S } } [ s _ { H } ( x ) ^ { 2 } ] = \frac { 1 } { 2 } \mathrm { T r } ( S ^ { - 1 } H S ^ { - 1 } H ) = \langle H , \mathcal { T } _ { S } [ H ] \rangle _ { \mathrm { F } } , \quad \mathcal { T } _ { S } [ H ] : = \frac { 1 } { 2 } S ^ { - 1 } H S ^ { - 1 } .
$$

The KL generator satisfies

$$
\nabla ^ { 2 } \Phi _ { 0 } ( S ) [ H ] = S ^ { - 1 } H S ^ { - 1 } = 2  { \mathcal { T } } _ { S } [ H ] .
$$

Hence its Hessian equals twice the Gaussian Fisher information operator (Lin et al., 2026). SQ does not share the Gaussian Fisher geometry. At $S = s I .$ , its Hessian is $s ^ { - 3 / 2 } H$ , whereas the Gaussian Fisher operator is $H / ( 2 s ^ { 2 } )$ . Their ratio, $2 { \sqrt { s } } ,$ , varies with s, so they cannot agree. Among the four divergences, only KL has a Hessian proportional to the Gaussian Fisher operator for every $S \succ 0$

## C PRECONDITIONER UPDATE DETAILS

We describe the factor updates of original Shampoo, PSGD-Kron, SQ-Shampoo, and KL-Shampoo, then specify the factor statistics, offline fitting procedure, and online EMA scheme used in our experiments.

## C.1 ORIGINAL SHAMPOO

In the EMA form of Shampoo (Gupta et al., 2018; Anil et al., 2020), the marginal statistics satisfy

$$
\begin{array} { r } { L _ { t } = \beta _ { 2 } L _ { t - 1 } + ( 1 - \beta _ { 2 } ) G _ { t } G _ { t } ^ { \top } , \quad R _ { t } = \beta _ { 2 } R _ { t - 1 } + ( 1 - \beta _ { 2 } ) G _ { t } ^ { \top } G _ { t } . } \end{array}
$$

Ignoring momentum, the parameter update is $\Theta _ { t + 1 } = \Theta _ { t } - \eta _ { t } L _ { t } ^ { - p } G _ { t } R _ { t } ^ { - p }$ . This applies the inverse square root of $S _ { t } = L _ { t } ^ { 2 p } \otimes R _ { t } ^ { 2 p } { : \ : p = 1 / 4 }$ for original Shampoo, while $p = 1 / 2$ applies the inverse square root of $L _ { t } \otimes R _ { t }$

## C.2 PSGD-KRON, SQ-SHAMPOO, AND KL-SHAMPOO

Under $P = S ^ { - 1 / 2 }$ , SQ-Shampoo shares PSGD’s population gradient-whitening objective (Equation (13)), but uses different factor updates.

Factor updates. PSGD parameterizes the preconditioner as $P = Q ^ { \top } Q$ and optimizes $Q$ on a matrix Lie group (Li, 2019). In the Kronecker form,

$$
Q = Q _ { L } \otimes Q _ { R } , \quad P = P _ { L } \otimes P _ { R } , \quad P _ { L } : = Q _ { L } ^ { \top } Q _ { L } , \quad P _ { R } : = Q _ { R } ^ { \top } Q _ { R } .
$$

With row-wise vectorization, the parameter update is $\Theta _ { t + 1 } = \Theta _ { t } - \eta _ { t } P _ { L , t } G _ { t } P _ { R , t }$ . For the original relative-gradient coordinates $d Q = { \mathcal { E } } Q$ , the factors follow

$$
Q _ { L , t + 1 } = ( I - \mu _ { t } \mathcal { R } _ { L , t } ) Q _ { L , t } , \quad Q _ { R , t + 1 } = ( I - \mu _ { t } \mathcal { R } _ { R , t } ) Q _ { R , t } ,
$$

where $\mathcal { R } _ { L , t } , \mathcal { R } _ { R , t }$ are the factor-wise relative gradients of the whitening objective in Equation (1). The official implementation also provides other update geometries, including $d Q = Q { \mathcal { E } } Q$

Our SQ- and KL-Shampoo implementations follow the spectral EMA scheme of Lin et al. (2026). Both average the factor statistics over time, update eigenvalue estimates in the current eigenbases, and periodically refresh bases. The two methods differ in the statistics supplied to this scheme: SQ uses the square-root divergence statistics, while KL uses the KL statistics in Table 6. Algorithm 1 gives the full update.

Table 5: Objectives and update schemes discussed here. Objectives are expressed in $S ,$ up to positive scale and additive constants; the PSGD row refers to gradient whitening.
<table><tr><td>Method</td><td>Population objective</td><td>Factor updates discussed here</td></tr><tr><td>PSGD-Kron</td><td>SQ divergence</td><td>Original relative-gradient steps; other geometries are available</td></tr><tr><td>SQ-Shampoo</td><td>SQ divergence</td><td>Spectral EMA of SQ statistics</td></tr><tr><td>KL-Shampoo</td><td>KL divergence</td><td>Spectral EMA of KL statistics</td></tr></table>

## C.3 DIVERGENCE-BASED FACTOR STATISTICS

Table 6 gives the factor statistics for F-, VN-, SQ-, and KL-Shampoo, obtained by setting $q = q _ { \mathrm { D } }$ in Equation (4). These statistics define both the offline and online updates that follow. For SQ-Shampoo, $q = 1 / 2$ gives the bilateral fixed-point iteration

$$
L ^ { ( k + 1 ) } = \frac { \mathbb { E } [ G ( R ^ { ( k ) } ) ^ { - 1 / 2 } G ^ { \top } ] } { \operatorname { T r } ( ( R ^ { ( k ) } ) ^ { 1 / 2 } ) } , \quad R ^ { ( k + 1 ) } = \frac { \mathbb { E } [ G ^ { \top } ( L ^ { ( k ) } ) ^ { - 1 / 2 } G ] } { \operatorname { T r } ( ( L ^ { ( k ) } ) ^ { 1 / 2 } ) } .
$$

Table 6: Instantaneous factor statistics for the Shampoo variants.
<table><tr><td>Method</td><td> $\Psi _ { L } ( G ; L , R )$ </td><td> $\Psi _ { R } ( G ; L , R )$ </td></tr><tr><td>F-Shampoo</td><td> $\boldsymbol { G R G ^ { \intercal } } / \operatorname { T r } ( \boldsymbol { R } ^ { 2 } )$ </td><td> $\boldsymbol { G } ^ { \top } \boldsymbol { L } \boldsymbol { G } / \operatorname { T r } ( \boldsymbol { L } ^ { 2 } )$ </td></tr><tr><td>VN-Shampoo</td><td> $G G ^ { \top } / \operatorname { \dot { T r } } ( \dot { R } )$ </td><td> $G ^ { \top } G / \operatorname { \dot { T r } } ( L )$ </td></tr><tr><td>SQ-Shampoo</td><td> $G R ^ { - \mathrm { i } / 2 } G ^ { \dagger } / \operatorname { T r } ( R ^ { 1 / 2 } )$ </td><td> $\boldsymbol { G } ^ { \top } \boldsymbol { L } ^ { - 1 / 2 } \boldsymbol { G } / \operatorname { T r } ( \boldsymbol { L } ^ { 1 / 2 } )$ </td></tr><tr><td>KL-Shampoo</td><td> $G R ^ { - 1 } G ^ { \top } / n$ </td><td> $G ^ { \top } L ^ { - 1 } G / \dot { m }$ </td></tr></table>

## C.4 OFFLINE FIXED-POINT FITS

Given N gradients, the offline bilateral rule for F-, VN-, SQ-, and KL-Shampoo is

$$
\widetilde { L } _ { \mathrm { D } } ^ { ( k + 1 ) } : = \frac { 1 } { N } \sum _ { b = 1 } ^ { N } \Psi _ { \mathrm { D } , L } ( G _ { b } ; L _ { \mathrm { D } } ^ { ( k ) } , R _ { \mathrm { D } } ^ { ( k ) } ) , \quad \widetilde { R } _ { \mathrm { D } } ^ { ( k + 1 ) } : = \frac { 1 } { N } \sum _ { b = 1 } ^ { N } \Psi _ { \mathrm { D } , R } ( G _ { b } ; L _ { \mathrm { D } } ^ { ( k ) } , R _ { \mathrm { D } } ^ { ( k ) } ) ,
$$

followed by under-relaxation,

$$
{ \cal L } _ { \mathrm { D } } ^ { ( k + 1 ) } = ( 1 - \alpha ) { \cal L } _ { \mathrm { D } } ^ { ( k ) } + \alpha \widetilde { \cal L } _ { \mathrm { D } } ^ { ( k + 1 ) } , \quad { \cal R } _ { \mathrm { D } } ^ { ( k + 1 ) } = ( 1 - \alpha ) { \cal R } _ { \mathrm { D } } ^ { ( k ) } + \alpha \widetilde { \cal R } _ { \mathrm { D } } ^ { ( k + 1 ) } .
$$

We initialize $L ^ { ( 0 ) } = I _ { m }$ and $R ^ { ( 0 ) } = I _ { n }$ , use $\alpha = 0 . 3$ , and iterate to numerical convergence.

Remark on few-sample fitting. We later consider fits with $N < m n$ , where the empirical second moment $\begin{array} { r } { \widehat { C } _ { N } : = \overline { { N ^ { - 1 } \sum _ { b = 1 } ^ { N } g _ { b } g _ { b } ^ { \top } } } } \end{array}$ is necessarily singular. This does not affect the iterations above: they require only positive-definite factors. For KL, the updates correspond to the criterion $\mathrm { T r } ( \widehat { C } _ { N } S ^ { - 1 } )$ + log det S, with $S = L \otimes R ,$ which remains finite for singular ${ \widehat { C } } _ { N }$

## C.5 ONLINE EMA UPDATES

Following the KL-Shampoo implementation of (Lin et al., 2026), the optimizer maintains factor accumulators $( A _ { L } , A _ { R } )$ for the eigenbases and separate eigenvalue estimates $\left( \lambda _ { L } , \lambda _ { R } \right)$ . Writing $\bar { L } = Q _ { L } \mathrm { D i a g } ( \lambda _ { L } ) \dot { Q } _ { L } ^ { \top }$ and $\begin{array} { r } { \bar { R } = \bar { Q _ { R } } \operatorname { D i a g } ( \lambda _ { R } ) Q _ { R } ^ { \top } , } \end{array}$ Algorithm 1 applies to $\mathrm { D } \in \{ \mathrm { F } , \mathrm { V N } , \mathrm { S Q } , \mathrm { K L } \}$

Algorithm 1 Online EMA update for a Shampoo variant   
Require: Matrix gradients $G _ { t } ,$ current momentum $M _ { t } ,$ , method D, second-moment EMA decay $\beta _ { 2 }$   
basis period $\check { T , }$ and initialized state $( A _ { L } , A _ { R } , Q _ { L } , Q _ { R } , \lambda _ { L } , \lambda _ { R } )$   
1: for $t \bar { = } 1 , 2 , . . .$ . do   
2: Form $\begin{array} { r } { \dot { L } = Q _ { L } \operatorname { D i a g } ( \lambda _ { L } ) Q _ { L } ^ { \top } } \end{array}$ and $\bar { R } = Q _ { R } \mathrm { D i a g } ( \lambda _ { R } ) Q _ { R } ^ { \top }$   
3: Precondition M<sub>t</sub> as $\widetilde { M } _ { t } \gets \bar { Q } _ { L } \mathrm { D i a g } ( \lambda _ { L } ^ { - 1 / 2 } ) Q _ { L } ^ { \top } M _ { t } Q _ { R } \mathrm { D i a g } ( \lambda _ { R } ^ { - 1 / 2 } ) Q _ { R } ^ { \top }$   
4: $\Delta _ { L } \gets \Psi _ { \mathrm { D } , L } ( G _ { t } ; \bar { L } , \bar { R } ) , \Delta _ { R } \gets \Psi _ { \mathrm { D } , R } ( \bar { G } _ { t } ; \bar { L } , \bar { R } )$   
5: $A _ { L }  \beta _ { 2 } \dot { A } _ { L } + ( 1 - \beta _ { 2 } ) \Delta _ { L } , A _ { \underline { { R } } }  \beta _ { 2 } A _ { R } + ( 1 - \beta _ { 2 } ) \Delta _ { R }$   
6: $\lambda _ { L }  \beta _ { 2 } \lambda _ { L } + ( 1 - \beta _ { 2 } ) \mathrm { d i a g } ( Q _ { L } ^ { \top } \Delta _ { L } Q _ { L } )$   
7: $\lambda _ { R }  \beta _ { 2 } \lambda _ { R } + ( 1 - \beta _ { 2 } ) \mathrm { d i a g } ( Q _ { R } ^ { \top } \Delta _ { R } Q _ { R } )$   
8: if t mod $T = 0$ then   
9: Refresh $Q _ { L } , Q _ { R }$ from $A _ { L } , A _ { R }$ by eigendecomposition/QR, while preserving $\lambda _ { L } , \lambda _ { R }$   
10: end if   
11: Return the preconditioned statistic $\widetilde { M } _ { t }$   
12: end for

The input $M _ { t } = \beta _ { 1 } M _ { t - 1 } + ( 1 - \beta _ { 1 } ) G _ { t }$ is the current momentum, with $M _ { t } = G _ { t }$ when momentum is disabled. It is preconditioned before $G _ { t }$ is added into the preconditioner statistics. The coefficient $\beta _ { 2 }$ controls the EMA of both the factor accumulators and eigenvalue estimates. In the EMA diagnostics, $T = 5$ for all four methods.

## D SYNTHETIC VALIDATION

We examine spectral error allocation and finite-sample stability using a smooth, long-tailed second moment that is not exactly Kronecker.

## D.1 EXPERIMENTAL SETUP

For matrix gradients $G _ { b } \in \mathbb { R } ^ { m \times n }$ , the empirical gradient second moment is

$$
{ \widehat C } : = \frac { 1 } { N } \sum _ { b = 1 } ^ { N } g _ { b } g _ { b } ^ { \top } , \quad g _ { b } : = \mathrm { v e c } _ { \mathrm { r o w } } ( G _ { b } ) .
$$

F-, VN-, SQ-, and KL-Shampoo share samples and use the offline updates in Appendix C. Table 7 shows the detailed settings.

Table 7: Settings for the synthetic experiments.
<table><tr><td>Setting</td><td> $m \times n$ </td><td>N</td><td>Additional parameters</td></tr><tr><td>Smooth long-tailed</td><td> $1 6 \times 2 4$ </td><td>4096</td><td> $\mathrm { s c a l e } = 1 0 ^ { - 1 } , \sigma = 0 . 0 2$ </td></tr></table>

Smooth long-tailed spectrum. We combine three matrix-normal components with different eigenvalue decay rates and orientations:

$$
G _ { b } : = 1 0 ^ { - 1 } \left( \sum _ { k = 1 } ^ { 3 } w _ { k } L _ { k } Z _ { b k } R _ { k } ^ { \top } + \sigma E _ { b } \right) ,\tag{18}
$$

where $Z _ { b k }$ and $E _ { b }$ have i.i.d. standard Gaussian terms. The weights $( w _ { 1 } , w _ { 2 } , w _ { 3 } ) = ( 1 , 0 . 3 , 0 . 1 5 )$ give the components different strengths. We control their eigenvalue decay through

$$
L _ { k } : = U _ { k } \operatorname { D i a g } \left( i ^ { - ^ { p _ { k } / 2 } } \right) _ { i = 1 } ^ { m } , \quad R _ { k } : = V _ { k } \operatorname { D i a g } \left( j ^ { - ^ { p _ { k } / 2 } } \right) _ { j = 1 } ^ { n } ,
$$

where $U _ { k } , V _ { k }$ are independently sampled orthogonal matrices and $( p _ { 1 } , p _ { 2 } , p _ { 3 } ) = ( 1 . 8 , 1 . 2 , 0 . 8 )$ . The second-moment factors $L _ { k } L _ { k } ^ { \top }$ and $\bar { R } _ { k } R _ { k } ^ { \top }$ have eigenvalues $i ^ { - p _ { k } }$ and $j ^ { - p _ { k } }$ , respectively, so larger p gives faster decay. The resulting second moment is generally not a single Kronecker product. We also add isotropic noise with $\sigma = 0 . 0 2$ and scale the gradients by $1 0 ^ { - 1 }$

Fitting. We generate a fixed pool of 4096 gradients. All methods start from $L = I _ { 1 6 }$ and $R = I _ { 2 4 }$ and use a relaxation coefficient of 0.3. We stop when the sum of the two factors’ relative Frobenius changes falls below $1 0 ^ { - 1 0 }$ , or after 10,000 iterations.

## D.2 METRICS AND RESULTS

For $S : = L \otimes R _ { \ l }$ , let $\lambda _ { i } ( M )$ denote descending eigenvalues and $U _ { M , k }$ the leading k eigenvectors. We measure cumulative eigenvalue error and leading-subspace overlap by

$$
E ( k ) : = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \left| \log _ { 1 0 } \frac { \lambda _ { i } ( S ) } { \lambda _ { i } ( \widehat { C } ) } \right| , \quad O ( k ) : = \frac { 1 } { k } \left\| U _ { \widehat { C } , k } ^ { \top } U _ { S , k } \right\| _ { F } ^ { 2 } .
$$

Finite-sample stability. For each divergence D, let $S _ { \mathrm { D } } ^ { \mathrm { r e f } }$ be its fit to the full 4096-sample pool and $S _ { N , \mathrm { ~ ~ } } ^ { \mathrm { { D } } }$ its fit to N samples in repeat r. We measure the squared relative Frobenius error

$$
E _ { \mathrm { r e l } , r } ^ { \mathrm { D } } ( N ) : = \frac { 1 } { m n } \left. ( S _ { \mathrm { D } } ^ { \mathrm { r e f } } ) ^ { - 1 / 2 } S _ { N , r } ^ { \mathrm { D } } ( S _ { \mathrm { D } } ^ { \mathrm { r e f } } ) ^ { - 1 / 2 } - I _ { m n } \right. _ { F } ^ { 2 } .\tag{19}
$$

This error is invariant to reciprocal factor rescaling. If $S _ { \mathrm { D } } ^ { \mathrm { r e f } } = L _ { \star } \otimes R _ { \star }$ and $S _ { N , r } ^ { \mathrm { D } } = \widehat { L } _ { \mathrm { D } } \otimes R _ { \star }$ it reduces to $1 / m \| \widetilde { L } _ { \mathrm { D } } - I _ { m } \| _ { F } ^ { 2 }$ , whose expectation is $\mathrm { E r r } _ { \mathrm { D } } ^ { L }$ in Proposition 4. Here both factors are estimated jointly, and the reference is each method’s large-sample Kronecker fit.

We use 20 random permutations of the same pool and fit the first $N \in \{ 1 0 , 2 0 , 4 0 , 8 0 , 1 6 0 , 3 2 0 , 6 4 0 \}$ gradients. Figure 4 (right) plots $1 / 2 0 \sum _ { r = 1 } ^ { 2 0 } E _ { \mathrm { r e l } , r } ^ { \mathrm { D } } ( N )$

Figure 7 shows that F prioritizes large eigenvalues but severely underestimates small ones; VN follows more mildly. SQ and KL have larger fitted tail eigenvalues. The mean stability errors follow $\mathrm { K L } < \mathrm { S Q } < \mathrm { V N } < \mathrm { \bar { F } }$ at every tested N, consistent with the theoretical ordering despite jointly fitting the non-Kronecker second moment.

![](images/aa2a089127ffc2eb69f404cd2988b51fb2be8b8d137a47ffe48cf83a70001e7c.jpg)

![](images/d2ab8f8ff9c78ebfd813e17178df94880d4041f4e5dad34e019bc800c860ca96.jpg)

![](images/98673d045fddb10efdfbbaed7b0c31525b4e5727524359cb1b400a47f3a94d0e.jpg)  
Figure 7: Smooth synthetic setting. From left to right: empirical and fitted spectra, cumulative spectral error, and leading-subspace overlap. Finite-sample stability is shown in Figure 4 (right).

## D.3 DIRECTIONAL GAIN INTERVENTIONS ON A QUADRATIC

We test whether gain differences explain the methods’ training performance by replacing KL’s leading or tail gains with those of F or SQ in a stochastic quadratic.

Setup. To model stochastic gradient noise, we use linear regression with squared loss $\ell ( x ; z , y ) =$ $\textstyle { \frac { 1 } { 2 } } ( z ^ { \top } x - y ) ^ { 2 }$ , where $z \sim \bar { \mathcal { N } } ( 0 , H )$ and $y \sim \mathcal { N } ( 0 , 1 / 6 )$ are independent. Its population loss is $\begin{array} { r } { \bar { \mathbb { E } } [ \ell ( x ; z , y ) ] = \frac { 1 } { \gamma } x ^ { \top } H x + 1 / 1 2 } \end{array}$ , so it is equivalent to the quadratic $\begin{array} { r } { f ( x ) = \frac { 1 } { 2 } x ^ { \top } H x } \end{array}$

We set $d = 3 8 \bar { 4 }$ and $H = \mathop { d C _ { \mathrm { t r u e } } } / \mathrm { T r } ( C _ { \mathrm { t r u e } } )$ , where $C _ { \mathrm { t r u e } }$ is the population second moment of the generator in Equation (18). Each minibatch gradient averages $B = 6 4$ sample gradients. Its population second moment is $C ( x ) = \mathbb { E } [ g _ { B } ( x ) g _ { B } ( x ) ^ { \top } ]$ , with expectation over minibatch draws:

$$
g _ { B } ( x ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } z _ { b } ( z _ { b } ^ { \top } x - y _ { b } ) , \qquad C ( x ) = \frac { ( x ^ { \top } H x + 1 / 6 ) H } { B } + \left( 1 + \frac { 1 } { B } \right) ( H x ) ( H x ) ^ { \top } .
$$

At each step, we fit the Kronecker factors to 40 independent mini-batch gradients reshaped into $1 6 \times 2 4$ , then update with a separate mini-batch. We fit the factors using the offline updates in Appendix C, with no EMA.

We run F, VN, SQ, and KL for 1,000 steps at a shared learning rate $\eta = 2 \times 1 0 ^ { - 3 } \ : ( \mathrm { F i g u r e \ : 8 } )$ . Within each seed, all methods share the same initialization and mini-batches. We initialize $x _ { 0 } = H ^ { - 1 / 2 } v .$ with a Gaussian vector v normalized to norm 2, giving $f ( x _ { 0 } ) = 2 { \mathrm { . } }$

Gain replacement. At the current x, fit KL and the comparison method $\mathrm { D } \in \{ \mathrm { F } , \mathrm { S Q } \}$ to the same gradients. Write $C ( x ) = U \Lambda U ^ { \top }$ in decreasing eigenvalue order and $P _ { \mathrm { D } } = ( \bar { L } _ { \mathrm { D } } \otimes \bar { R } _ { \mathrm { D } } ) ^ { - 1 / 2 }$ . For selected directions I, set

$$
r _ { i } = \left\{ \begin{array} { l l } { \lVert P _ { \mathrm { D } } u _ { i } \rVert / \lVert P _ { \mathrm { K L } } u _ { i } \rVert , } & { i \in \mathcal { T } , } \\ { 1 , } & { i \not \in \mathcal { T } , } \end{array} \right. \qquad \widetilde { P } = P _ { \mathrm { K L } } U \mathrm { D i a g } ( r _ { i } ) U ^ { \top } .
$$

We replace KL’s gains in the first 39 (top 10%) or last 96 (bottom 25%) directions by rescaling each $P _ { \mathrm { K L } } u _ { i }$ to match $\left. P _ { \mathrm { D } } u _ { i } \right.$ . Starting from the same KL iterate at step 900, we continue to step 1,000 using the original training minibatches and learning rate, recomputing the factors and population eigenbasis at each step.

Results. F reaches the highest loss (Figure 8, left). Averaged over the last 20 updates and 3 seeds, KL achieves a loss of 0.00844. Replacing KL’s leading or tail gains with F’s raises the loss to 1.441 and 0.03333, respectively (middle and right). The corresponding SQ replacements yield 0.00857 and 0.00855, close to KL.

![](images/7a430e84ad5553c3905fab8d25f25cc9d5c79d56725daa44a36bdae22c6b0712.jpg)

![](images/553dde94a2395eef613bceb57971c033d3e1e3f9bb27f0eabd1107c1799cad47.jpg)

![](images/b3216796c82c036cfb161390aca398e87a7c317b34a8c99c428a107dd751365a.jpg)  
Figure 8: Quadratic training loss averaged over 3 seeds. Left: four methods. Middle and right: replacing KL gains from step 900. F gains raise loss in either region; SQ gains leave it close to KL.

## E REAL-GRADIENT EXPERIMENTAL DETAILS

We analyze $1 2 8 \times 1 2 8$ query, value, and attention-output gradients from block 5 of a 12-layer, fourhead GPT-2 model trained with AdamW. We freeze the model at steps 100, 2500, and 4500, giving nine checkpoint-projection pairs. Table 8 summarizes the shared settings; Appendix E.6 extends the diagnostics to Distributed Shampoo checkpoints.

Table 8: Shared settings for real-gradient diagnostics.  
Setting Specification   
Gradient pools Four splits per experiment; 20,480 FP32 mini-batch gradients per split, assigned by   
round-robin interleaving   
Gradient collection Batch size 64; sequence length 512   
Large-sample fit All 20,480 gradients in split 0   
Few-sample fit 40 gradients from split 0; four repeats   
Fixed-point solver Relaxation α = 0.3; tolerance $1 0 ^ { - 8 } ;$ ; at most 1,000 steps   
Online EMA 16 disjoint chronological repeats from split 0, each with 512 gradients; $\beta _ { 2 } = 0 . 9 5 ;$   
eigendecomposition every five steps

All four methods retain their eigenvalue EMAs during each eigenbasis refresh. Reliability uses all splits. For gain calibration, split 0 supplies the fits and source eigenvectors; splits 1, 2, and 3 supply the held-out directional scales in Section 5.2.

Numerical details. Second moments and eigendecompositions use FP64. We sort eigenvalues in decreasing order and retain those exceeding $d { \check { \times } } 2 ^ { - 5 2 }$ times the largest eigenvalue, where $d = 1 6$ ,384 and $2 ^ { - 5 2 }$ is FP64 machine precision. For reliability summaries, retained directions are grouped by rank into top 0–10%, middle 10–40%, lower 40–75%, and tail 75–100%.

## E.1 EMPIRICAL SECOND MOMENTS AND FINITE-SAMPLE EFFECTS

## E.1.1 FINITE-SAMPLE TAIL SELECTION

Cross-fit control. We take eigenvectors $u _ { b , i }$ of $C _ { b }$ from a third split b and compare their scales on splits s and t:

$$
\xi _ { b ; s , t , i } ^ { \mathrm { c f } } : = \log _ { 1 0 } \frac { u _ { b , i } ^ { \top } C _ { t } u _ { b , i } } { u _ { b , i } ^ { \top } C _ { s } u _ { b , i } } , \quad b \notin \{ s , t \} .
$$

The direction $u _ { b , i }$ is therefore selected independently of both second-moment estimates being compared. We compare the six pairs with $s < t$ , using each remaining split to select directions. This gives 12 comparisons without reverse pairs that would cancel the signed log-ratios. Cross-fit scale errors have medians and IQRs near zero across the spectrum (Figure 9), supporting sampling-induced selection bias rather than distribution shift.

Selection bias. Another independent split estimates the population second moment along each source eigenvector. The identity below shows that underestimating the source eigenvalue raises the expected held-out ratio, with larger relative errors giving larger ratios. Let $\rho _ { s \to t , i } : = 1 0 ^ { \xi _ { s \to t , i } }$ denote the ratio before taking the logarithm in Equation (8).

Proposition 5 (Finite-Sample Tail Selection). Let $C _ { s } = C + E _ { s }$ and $C _ { t } = C + E _ { t }$ be independent empirical gradient second moments drawnfrom the same distribution, where $E _ { s }$ and $E _ { t }$ denote their zero-mean finite-sample estimation errors. For a source eigenpair $C _ { s } u _ { s , i } = \lambda _ { s , i } u _ { s , i }$ with $\lambda _ { s , i } > 0$

$$
\mathbb { E } [ \rho _ { s  t , i } \mid C _ { s } ] = \frac { u _ { s , i } ^ { \top } C u _ { s , i } } { \lambda _ { s , i } } = 1 - \frac { u _ { s , i } ^ { \top } E _ { s } u _ { s , i } } { \lambda _ { s , i } } .\tag{20}
$$

Thus, if the source error along this direction is negative, $u _ { s , i } ^ { \top } E _ { s } u _ { s , i } < 0 ,$ its conditional expected held-out ratio is larger than one.

Proof. Conditional on $C _ { s }$ , the vector $u _ { s , i }$ and the denominator $\lambda _ { s , i }$ are fixed. Independence and $\begin{array} { r } { \mathbb { E } [ C _ { t } ] = C \mathrm { ~ g i v e ~ } \mathbb { E } [ \rho _ { s  t , i } \ | \ C _ { s } ] = u _ { s , i } ^ { \top } C u _ { s , i } / \lambda _ { s , i } } \end{array}$ . Substituting $C = C _ { s } - E _ { s }$ yields Equation (20). □

Why ratios rise toward the tail. Selecting directions with small sample eigenvalues can favor downward sampling fluctuations, so their scales may increase when evaluated on an independent split. In Equation (20), the downward estimation error is divided by the source eigenvalue $\lambda _ { s , i }$ Small tail eigenvalues can therefore turn downward errors into large held-out ratios.

## E.1.2 HIGH-DIMENSIONAL SAMPLING AND TAIL SIGNALS

Finite-sample spectral distortion is well documented in high-dimensional covariance estimation (Ledoit & Wolf, 2012; Bartz, 2016), and similar selection bias has also been observed in minibatch Hessian and GGN estimates (Tatzel et al., 2025). With $d / N = 0 . 8$ , the spectral distortion observed here may partly be explained by Marcenko–Pastur-type spreading (ˇ Bai & Silverstein, 2010). However, this does not explain why the leading scales are much more consistent across splits than the tail. Prior work has reported alignment between the leading subspaces of the empirical second moment and the Hessian (Zhu et al., 2019; Wu et al., 2022; Liu et al., 2026). This may help explain the stronger reproducibility of the leading scales, although the connection remains to be established.

Tail directions and learning signals. If class k occurs with probability $p _ { k }$ and produces gradient $g = a _ { k } u _ { k }$ , with orthonormal $u _ { k }$ , then $\begin{array} { r } { C = \sum _ { k } p _ { k } a _ { k } ^ { 2 } u _ { k } u _ { k } ^ { \top } } \end{array}$ . For comparable $\left| a _ { k } \right|$ , infrequent classes correspond to directions with small second-moment scales. This illustrates how tail directions can carry useful signals (Wang et al., 2026; Kim et al., 2026). However, finite-sample noise can still severely underestimate their scales.

## E.2 RELIABILITY ACROSS CHECKPOINTS

Figure 9 shows held-out scale error for all 9 experiments. Table 9 reports all four regions, including absolute held-out scale error. We take the median over all directions in each region and all 12 comparisons within an experiment, then report the mean and sample standard deviation across the nine checkpoint-projection pairs per training optimizer.

Table 9: Held-out scale error across all four eigenvalue regions. Zero indicates agreement; positive ξ indicates underestimation, and larger |ξ| indicates greater mismatch. Entries show mean ± sample SD across nine checkpoint-projection pairs per training optimizer, using each’s regional median.
<table><tr><td colspan="5">AdamW checkpoints</td></tr><tr><td>Metric</td><td>Top (0–10%)</td><td></td><td>Middle (10–40%)Lower (40–75%)</td><td>Tail (75–100%)</td></tr><tr><td>|ξ| ()</td><td> $0 . 0 2 7 \pm 0 . 0 1 6$ </td><td> $0 . 1 2 4 \pm 0 . 0 3 7$ </td><td> $0 . 4 3 1 \pm 0 . 0 3 0$ </td><td> $1 . 0 0 9 \pm 0 . 1 2 0$ </td></tr><tr><td>Signed ξ  $\boldsymbol { \xi } ^ { \mathrm { c f } }$ </td><td> $0 . 0 1 0 \pm 0 . 0 3 0$ </td><td> $0 . 1 2 4 \pm 0 . 0 3 7$ </td><td> $0 . 4 3 1 \pm 0 . 0 3 0$ </td><td> $1 . 0 0 9 \pm 0 . 1 2 0$ </td></tr><tr><td>Cross-fit</td><td> $- 0 . 0 0 8 \pm 0 . 0 0 4$ </td><td> $- 0 . 0 0 6 \pm 0 . 0 0 4$ </td><td> $- 0 . 0 0 6 \pm 0 . 0 0 5$ </td><td> $- 0 . 0 0 7 \pm 0 . 0 0 6$ </td></tr><tr><td colspan="5">Distributed Shampoo checkpoints</td></tr><tr><td>Metric</td><td>Top (0–10%)</td><td>Middle (10–40%) Lower (40–75%)</td><td></td><td>Tail (75–100%)</td></tr><tr><td>|ξ| (↓)</td><td> $0 . 0 2 0 \pm 0 . 0 0 9$ </td><td> $0 . 1 0 4 \pm 0 . 0 3 0$ </td><td> $0 . 4 2 9 \pm 0 . 0 2 5$ </td><td> $1 . 0 4 8 \pm 0 . 0 1 7$ </td></tr><tr><td>Signed ξ</td><td> $0 . 0 0 2 \pm 0 . 0 2 0$ </td><td> $0 . 1 0 4 \pm 0 . 0 3 0$ </td><td> $0 . 4 2 9 \pm 0 . 0 2 5$ </td><td> $1 . 0 4 8 \pm 0 . 0 1 7$ </td></tr><tr><td>Cross-fit  $\boldsymbol { \xi } ^ { \mathrm { c f } }$ </td><td> $- 0 . 0 0 9 \pm 0 . 0 0 5$ </td><td> $- 0 . 0 0 8 \pm 0 . 0 0 4$ </td><td> $- 0 . 0 0 7 \pm 0 . 0 0 4$ </td><td> $- 0 . 0 0 7 \pm 0 . 0 0 4$ </td></tr></table>

Figure 9: Held-out scale error. Rows: steps 100, 2500, 4500. Bands show split-comparison IQRs.  
![](images/e5d011fa0cadc7dbe23302b35fffeb995a6a2428c705ece3fe8ab04c0e0ab62a.jpg)  
E.3 INVERSE-ROOT GAIN ACROSS CHECKPOINTS

Figures 10–12 show the gains of large-sample fits, few-sample fits, and EMA states at all three checkpoints. Within each checkpoint-projection pair, all fits use the same split-0 directions and the same scales averaged over splits 1, 2, and 3 (Equation (11)). Evaluations use the numerically retained directions from split 0. The black solid curve shows the source preconditioner $C _ { 0 } ^ { - 1 / 2 }$ , with error ${ \scriptstyle \frac { 1 } { 2 } } \log _ { 1 0 } ( h _ { i } / \lambda _ { 0 , i } ) $ ; the dashed line marks zero.

Large-sample curves use one fit per method. Few-sample and EMA curves average held-out gain error over 4 repeats and 16 repeats, respectively. For Table 2, we take the median absolute held-out gain error over all retained directions and repeats within each experiment, then report the mean and sample standard deviation across nine experiments. This is median $( | e _ { D , i } | )$ , not | median $( e _ { D , i } )$ |.

Figure 10: Held-out gain error on AdamW checkpoints at step 100. Rows: large-sample fit, 40- sample fit, EMA with eigendecomposition every five steps.

![](images/3c489a382f3a60da92e08704d8045fe70ad839c3f0009946e5395cab42d47eef.jpg)

![](images/2ca61dbd8f4dc3aede5f236957dcb3e2b28e786dc3d2dbfca6156e3800c8e93c.jpg)

![](images/043049084eaca57abd3004a5e08e96890f7f439cfc2f0bd37290a4966d528788.jpg)

Figure 10: Held-out gain error on AdamW checkpoints at step 100 (continued).

![](images/b2eeb465b93f5b41ada475b84794224a971e84d49408c67260069bb21e3d4849.jpg)

Figure 11: Held-out gain error on AdamW checkpoints at step 2500. Rows: large-sample fit, 40- sample fit, EMA with eigendecomposition every five steps.  
![](images/138d8bde678cc9b293d45f3597c196bcce4285329d6e5b5723d9e19e2b44a2af.jpg)

Figure 12: Held-out gain error on AdamW checkpoints at step 4500. Rows: large-sample fit, 40- sample fit, EMA with eigendecomposition every five steps.  
![](images/e517bbe55e52ed7df3e85c8eb988a83ffb8bfe8a6c082d47581b91832b7531e1.jpg)  
E.4 KRONECKER FITS OF DIFFERENT DIVERGENCES

We compare fitted spectra across all nine checkpoint-projection pairs and show how fitted eigenvalues and eigenvectors determine gains. Each fit uses the same split-0 gradients as the empirical second moment (Figure 13).

Spectral preferences. F and VN put more weight on large eigenvalues and penalize overestimation more than reciprocal underestimation (Table 1). SQ gives large eigenvalues less weight and penalizes reciprocal errors equally. KL weights relative errors equally across scales and penalizes underestimation more strongly. These preferences help explain the larger fitted tail eigenvalues of SQ and KL in Section 5.1.

How small eigenvalues increase gains. Let $( \omega _ { D , j } , v _ { D , j } )$ be the eigenpairs of $S _ { D } ^ { \mathrm { r e f } }$ in decreasing order, and write $P _ { D , i j } : = \langle u _ { 0 , i } , v _ { D , j } \rangle ^ { 2 }$ . Then

$$
1 0 ^ { 2 e _ { D , i } } = h _ { i } \sum _ { j } \frac { P _ { D , i j } } { \omega _ { D , j } } .
$$

Each overlap is divided by a fitted eigenvalue, so even a small overlap can matter when that eigenvalue is tiny. Since $\begin{array} { r } { \sum _ { j } \dot { P _ { D , i j } } = 1 } \end{array}$ , we can also write

$$
1 0 ^ { 2 e _ { D , i } } = \frac { h _ { i } } { \omega _ { D , i } } + h _ { i } \sum _ { j \neq i } { P _ { D , i j } \left( { \frac { 1 } { \omega _ { D , j } } - \frac { 1 } { \omega _ { D , i } } } \right) } .
$$

With aligned eigenvectors, only the first term remains. The second term shows how gains change when the fitted eigenvectors differ from the source eigenvectors.

Let $\varepsilon _ { D , i }$ be the total squared overlap with fitted eigenvectors whose eigenvalues satisfy $\omega _ { D , j } \leq \tau$ Then

$$
1 0 ^ { 2 e _ { D , i } } \geq h _ { i } \varepsilon _ { D , i } / \tau .
$$

If $h _ { i } \varepsilon _ { D , i } > \tau$ , the gain exceeds the held-out target, even if the leading eigenvalues are fitted well. Figure 13: Large-sample spectra. Rows: steps 100, 2500, 4500.

![](images/3be22757f88461a1dcb02f95372494e0b24ddad5920ba37f57a6e358d135004b.jpg)  
E.5 GAIN RECOVERY WITH SHORT GRADIENT HISTORIES

Recovery is measured against each method’s own large-sample fit over all 16,384 reference directions (Equation (12)). For Figure 14, we first average the log gain ratios across 4 offline or 16 EMA repeats, and then average every 64 consecutive directions. For Table 3, we compute the median absolute log gain ratio over all directions and repeats without binning, and report the mean and sample SD across nine checkpoint-projection pairs.

Figure 14: Large-sample fit recovery on AdamW checkpoints at step 2500. Rows: F, VN, SQ, KL.   
Solid: 40-sample fits; dashed: EMA with eigendecomposition every five steps.

![](images/c45f11a915d9c04eab5c88a93370494b40f50bca72d5d45f8929213c4cad5679.jpg)

![](images/f607b5e1155a4c5ab83e228c9ebcd20478884718c9fc0a11b2d4441b2da03393.jpg)

![](images/60dfe0c069326e6895b35df635923269df5e0cccc0b3038e90cf2fd8872c0a92.jpg)

Figure 14: Large-sample fit recovery on AdamW checkpoints at step 2500 (continued).  
![](images/54072e10db33a50f31c7bf66ab0a41b63e831c84d104a29e5b2e208dca387096.jpg)

## E.6 DIAGNOSTICS ON DISTRIBUTED SHAMPOO CHECKPOINTS

We train the same GPT-2 model with Distributed Shampoo (Shi et al., 2023), using inverse-squareroot factors and Adam grafting.

We use the same checkpoint steps, projections, gradient collection, fitting settings, and aggregation as for AdamW (Appendix E). All four divergences are fitted to the new gradients. Figure 15 shows step 2500; the main-text tables summarize all nine checkpoint-projection pairs.

Figure 15: Distributed Shampoo diagnostics at step 2500. Top: held-out scale error and cross-fit controls. Remaining rows: held-out gain error for large-sample fits, 40-sample fits (four repeats), and EMA (512 gradients, 16 repeats, eigendecomposition every five steps).

![](images/be67398123d5373d558b206400bb91ebcf620234e564023459e42a583e74e356.jpg)

## F GPT-2 TRAINING DETAILS AND FULL RESULTS

Models and data. The 12-layer models use the architecture from (Jordan et al., 2024). FineWeb provides 3B training tokens. Each run processes 2.67B tokens with sequence length 1024 and a global batch of 512 sequences.

Training settings. Grid searches at model widths up to 768 found nearly identical optimal learning rates across the tested optimizers. We therefore use a shared learning rate across optimizers at each larger width without further searches. Learning-rate tuning can absorb an overall gain offset, but cannot correct gain errors that vary across directions, layers, or checkpoints. Within each width, methods share initialization, data order, and learning rate (Table 10). We follow (Lin et al., 2026) for the other hyperparameters. The schedule uses 250 warmup steps, a constant-rate phase, and 1450 steps of linear decay. The embedding and other non-matrix valued parameters use AdamW with learning rate $0 . 0 0 3 6 \dot { , } \beta = ( 0 . 9 , 0 . 9 5 )$ , and no weight decay.

Table 10: Final validation loss after 5, 100 steps; one run per configuration.
<table><tr><td>Width</td><td>Params (M)</td><td>LR</td><td>F</td><td>VN</td><td>SQ</td><td>KL</td></tr><tr><td>256</td><td>22.3</td><td>0.0024</td><td>3.8281</td><td>3.7573</td><td>3.7450</td><td>3.7475</td></tr><tr><td>512</td><td>63.5</td><td>0.0024</td><td>3.5082</td><td>3.4377</td><td>3.4173</td><td>3.4180</td></tr><tr><td>768</td><td>123.6</td><td>0.0018</td><td>3.3993</td><td>3.2876</td><td>3.2661</td><td>3.2660</td></tr><tr><td>1024</td><td>202.5</td><td>0.0012</td><td>3.3528</td><td>3.1979</td><td>3.1824</td><td>3.1769</td></tr><tr><td>1536</td><td>417.0</td><td>0.0008</td><td>3.3111</td><td>3.0972</td><td>3.0817</td><td>3.0758</td></tr></table>

Training results. Table 10 supplements Table 4 with parameter counts and learning rates. Figure 16 shows the validation-loss trajectories from step 2,000 to the end of training.

![](images/f4d3c654d980fa7781608e6397d26d6a7a36752ea8fa99c3b8266aec3b091de1.jpg)

![](images/32e2f572c277378e93517aa57bbeea34f683cd6a0c4840fce0407704f3989925.jpg)

![](images/7947a65ad1dd7f938dbf7d04d575a6ec2167a099cc9367aab51d81cac35c2e18.jpg)

![](images/109c030abf23689bd0982b679443bcabe5720bb312fa0dd2eec3d7453ec49279.jpg)

![](images/af1feb5fc8aba7d99cb70a7e54565b1b8dfa793430f8fd41183fc0a5b97c5cc0.jpg)  
Figure 16: Validation loss from step 2, 000 at all five widths.