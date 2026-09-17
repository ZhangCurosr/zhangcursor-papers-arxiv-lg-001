# STABLE FILTERS FOR GENERATIVE MODELING OF GRAPH SIGNALS

Martin Schmidt and Gonzalo Mateos

University of Rochester, Rochester, NY, USA

## ABSTRACT

Generating signals on graphs requires permutation-equivariant models that exhibit stability with respect to relative structural perturbations. While recent graph-aware Schrodinger bridge models incor-¨ porate topology information directly into their reference dynamics, it is unclear how perturbations of the graph propagate through these dynamics and affect the resulting generated distributions. In this paper, we analyze the structural stability of graph-aware continuoustime generative models whose drift combines a graph filter with a learned graph neural network. We derive explicit Wasserstein stability bounds that quantify the effect of relative graph perturbations on the generated distributions. Motivated by these bounds, we introduce a principled framework for designing stable graph filters that preserve the smoothing behavior of graph heat diffusion, while boosting structural stability. Experiments on synthetic and fMRI signals show our stable filters enhance structural robustness while matching or exceeding the generative quality of the heat equation baseline.

Index Terms— Graph signal processing, generative models, stability, permutation equivariance, graph neural networks.

## 1. INTRODUCTION

Mapping one probability distribution to another is a fundamental problem in generative modeling. While these models have seen remarkable success in Euclidean spaces, there is a growing need to transport distributions on non-Euclidean settings. In particular, graphs provide a natural representation for these irregular domains, where vertex-supported signals can describe, e.g., neural activity over brain connectomes or traffic flows on transportation networks [1–3]. Learning to faithfully generate graph signals has broad applicability, ranging from synthetic data augmentation and implicit priors to realistic system simulations across domains like recommender systems, financial forecasting, and wireless networks [4, 5].

Among the many stochastic processes that match a given pair of endpoint distributions, the dynamic Schrodinger bridge problem¨ provides a principled solution that favors the process closest to a prescribed reference dynamics [6–8]. In the classical formulation, the reference is typically chosen as a standard Wiener process. However, when transporting distributions of graph signals, it is prudent to incorporate the underlying structure directly into the generative model by replacing the Wiener process with graph-aware reference dynamics [9–11]. This promotes trajectories that are consistent with the graph topology, and the corresponding dynamics can be learned from samples using bridge and flow matching techniques [12, 13].

Crucially and by design, these generative models explicitly depend on the graph that parameterizes both the reference dynamics and the learned vector field. In practice, the observed graph is only a partial reflection of an underlying complex system [14, Ch. 7], typically subject to missing edges or estimation errors, leading to an imperfect graph representation [15, 16]. Consequently, unavoidable perturbations of the graph can steer the entire generative trajectory and, ultimately, the distribution of the generated signals.

Recent efforts have started to examine the robustness properties of continuous-time generative models on graphs. Specifically, prior work established the structural stability of generative processes governed solely by a learned graph neural network (GNN) vector field [17]. However, the analysis therein does not extend to frameworks that involve graph-aware reference dynamics, which introduce an additional graph-dependent drift term governed by a graph filter. Since this filter is a fixed design choice (e.g., graph heat diffusion) rather than a learned component, its inclusion not only alters the stability of the overall dynamics, but also raises the critical question of how to design stable graph filters for these generative models.

Our contributions address this foundational gap. First, in Section 3 we characterize the structural stability of graph-aware continuoustime models by deriving bounds on the Wasserstein distance, quantifying the error between the generated distributions caused by graph perturbations. Leveraging these theoretical guarantees, in Section 4 we introduce a principled optimization framework for reference graph filter design. Specifically, our approach synthesizes filters that respect the induced smoothing of heat diffusion on graphs without sacrificing robustness to structural perturbations. Reproducible tests on synthetic and fMRI signals demonstrate that our optimal filter design exhibits enhanced stability to graph perturbations, while matching or exceeding the generative quality of the heat equation baseline (Section 5). Concluding remarks are given in Section 6.

## 2. PRELIMINARIES

## 2.1. Graph Signal Processing

Let $\mathcal { G } = ( \nu , \mathcal { E } , \mathcal { W } )$ be a known undirected graph with N nodes, edge set $\mathcal { E } \subseteq \mathcal { V } \times \mathcal { V } ,$ and edge weight map $\mathcal { W } : \mathcal { E }  \mathbb { R }$ . Its structure is represented by a graph shift operator (GSO) $\mathbf { L } \in \mathbb { R } ^ { N \times N }$ such as the graph Laplacian. A graph signal $x : \mathcal { V } $ R is represented by $\mathbf { x } \in \mathbf { \hat { \mathbb { R } } } ^ { N }$ , with x<sub>i</sub> denoting its value at node i. Assuming the eigendecomposition $\mathbf { L } = \mathbf { V } \mathbf { A } \mathbf { \bar { V } }$ <sup>⊤</sup>, with eigenvalues $\{ \lambda _ { i } \} _ { i = 1 } ^ { N } .$ the graph Fourier transform is $\hat { \mathbf { x } } = \mathbf { V } ^ { \top } \mathbf { x } \left[ 1 8 \right]$ . A graph convolutional filter of order P is defined as $\begin{array} { r } { \mathbf { F } ( \mathbf { L } ) \dot { \mathbf { \Psi } } : = \sum _ { p = 0 } ^ { \tilde { P } } \dot { \theta } _ { p } \mathbf { L } ^ { p } } \end{array}$ , with coefficients $\{ \theta _ { p } \}$ [19, 20], and corresponding frequency response $\begin{array} { r } { f ( \lambda ) : = \sum _ { p = 0 } ^ { P } \theta _ { p } \lambda ^ { p } } \end{array}$ . A GNN extends this construction by cascading graph filters with pointwise, normalized Lipschitz nonlinearities $\sigma ( \cdot ) ~ [ 2 1 - 2 3 ]$ . For an input x, we denote the output of a GNN with learnable parameters θ and fixed graph support L by u<sub>θ</sub>(x; L).

## 2.2. Stability of Graph Filters and GNNs

Let P denote the set of $N \times N$ permutation matrices. A fundamental property of both graph filters and GNNs is permutation equivariance [24]. For any permutation $\mathbf { P } \in \mathcal { P }$ , applying the operator to a permuted signal $\mathbf { P } ^ { \top } \mathbf { x }$ on a similarly permuted graph $\mathbf { P } ^ { \top } \mathbf { L P }$ yields a permuted output, meaning $u _ { \pmb \theta } ( \mathbf { P } ^ { \dagger } \mathbf { \hat { x } } ; \mathbf { P } ^ { \top } \mathbf { L P } ) = \mathbf { \tilde { P } } ^ { \top } u _ { \pmb \theta } ( \mathbf { x } ; \dot { \mathbf { L } } )$

In practice, the observed graph structure is often noisy. To analyze the robustness of graph filters and GNNs to structural errors, we adopt a well-documented relative perturbation model [24,25]. Given a nominal GSO L and a perturbed one L<sup>˜</sup>, we evaluate the symmetric relative error matrix $\mathbf { E } \stackrel { \cdot } { \in } \mathcal { S } : = \{ \mathbf { S } \in \mathbb { R } ^ { N \times N } : \mathbf { S } = \mathbf { S } ^ { \top } \}$ at the permutation $\mathbf { P } _ { 0 } \in \mathcal { P }$ that minimizes the error’s spectral norm, i.e.,

$$
\begin{array} { r } { \{ \mathbf { E } ^ { \star } , \mathbf { P } _ { 0 } \} = \arg \underset { \mathbf { E } \in \mathcal { S } , \mathbf { P } \in \mathcal { P } } { \operatorname* { m i n } } \| \mathbf { E } \| _ { 2 } } \\ { \mathrm { s . t . } \quad \mathbf { P } ^ { \top } \tilde { \mathbf { L } } \mathbf { P } = \mathbf { L } + \frac { 1 } { 2 } ( \mathbf { E } \mathbf { L } + \mathbf { L } \mathbf { E } ) . } \end{array}\tag{1}
$$

Under this perturbation model, both graph filters and GNNs are structurally stable. Specifically, for a relative perturbation bounded by $\| \mathbf { E } ^ { \star } \| _ { 2 } \leq \varepsilon ,$ the pointwise deviation of the output is bounded by

$$
\| \mathbf { P } _ { 0 } ^ { \top } u _ { \theta } ( \mathbf { x } ; \tilde { \mathbf { L } } ) - u _ { \theta } ( \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } ; \mathbf { L } ) \| \leq \left( \Gamma \varepsilon + \mathcal { O } ( \varepsilon ^ { 2 } ) \right) \| \mathbf { x } \| ,
$$

for all $\mathbf { x } \in \mathbb { R } ^ { N }$ . The structural stability constant $\Gamma > 0$ dictates the sensitivity of the architecture to structural noise. For a graph filter, Γ depends primarily on the integral Lipschitz constant of its frequency response; see [24, Thm. 2]. For a GNN, however, Γ is strictly larger as it compounds with the network’s depth, width, and the uniform spectral bounds of its intermediate filters [24, Thm. 4]. We build on these foundations to analyze the propagation of structural errors through the continuous-time dynamics of our generative model.

## 2.3. Topological Schrodinger Bridge¨

Let p<sub>0</sub> and $p _ { 1 }$ be two probability distributions on $\mathbb { R } ^ { N }$ , and let M denote the set of probability measures on the path space $\mathcal { C } ( [ 0 , 1 ] , \mathbb { R } ^ { N } )$ Given a reference path measure $\mathbb { Q } .$ the Schrodinger bridge prob-¨ lem [26] seeks (KL stands for Kullback-Leibler divergence)

$$
\mathbb { P } ^ { \star } \in \arg \operatorname* { m i n } _ { \mathbb { P } \in \mathfrak { M } } \mathrm { K L } ( \mathbb { P } \| \mathbb { Q } ) \quad \mathrm { s . t . } \quad \mathbb { P } _ { 0 } = p _ { 0 } , \mathbb { P } _ { 1 } = p _ { 1 } .\tag{2}
$$

Thus, the choice of $\mathbb { Q }$ determines the reference dynamics with respect to which the interpolation between $p _ { 0 }$ and $p _ { 1 }$ is optimized.

For graph-supported data, this reference process can be chosen to explicitly incorporate the underlying topology [9]. Let L denote a GSO and consider the graph-aware reference diffusion

$$
d \mathbf { y } _ { t } = \left[ \mathbf { H } _ { t } ( \mathbf { L } ) \mathbf { y } _ { t } + \pmb { \alpha } _ { t } \right] d t + \pmb { \Sigma } _ { t } d \mathbf { w } _ { t } ,\tag{3}
$$

where $\mathbf { w } _ { t }$ is a standard Brownian motion in $\mathbb { R } ^ { N } , \ \pmb { \Sigma } _ { t } \ \in \ \mathbb { R } ^ { N \times N }$ controls the diffusion, ${ \pmb { \alpha } } _ { t } \in \mathbb { R } ^ { N }$ is a bias term and ${ \bf H } _ { t } ( { \bf L } ) \ : = \mathbf { \Gamma }$ $\scriptstyle \sum _ { k = 0 } ^ { K } h _ { k } ( t ) \mathbf { L } ^ { k }$ is a time-dependent graph filter of order K. Then $\mathbb { Q }$ is the path measure law induced by (3). Substituting Q into (2) yields a Schrodinger bridge whose optimal trajectories are defined relative¨ to the graph-aware dynamics. For general endpoint distributions, the drift of the resulting bridge is not available in closed form. Bridge and flow matching methods learn the required correction from samples by regressing vector fields associated with conditional reference bridges [9, 10]. The resulting dynamics can be written as

$$
d { \mathbf { x } } _ { t } = b _ { t } ( { \mathbf { x } } _ { t } ; { \mathbf { L } } ) d t + \Sigma _ { t } d { \mathbf { w } } _ { t } ,\tag{4}
$$

where

$$
b _ { t } ( \mathbf { x } ; \mathbf { L } ) : = \mathbf { H } _ { t } ( \mathbf { L } ) \mathbf { x } + \pmb { \alpha } _ { t } + u _ { t } ^ { \pmb { \theta } } ( \mathbf { x } ; \mathbf { L } ) .\tag{5}
$$

Here, $u _ { t } ^ { \pmb \theta } \big ( \cdot ; \mathbf { L } \big )$ denotes a learned vector field that corrects the reference drift so that the law induced by (4) solves the Schrodinger¨ bridge problem (2). Specifically, we let $u _ { t } ^ { \theta } ( \mathbf { x } ; \mathbf { L } ) : = u _ { \theta } ( g ( \mathbf { x } , t ) ; \mathbf { L } )$ where $u _ { \boldsymbol { \theta } } ( \cdot ; \mathbf { L } )$ is a GNN and $g ( \mathbf { x } , t ) ~ \in ~ \mathbb { R } ^ { N }$ is a continuous, permutation-equivariant conditioning map that incorporates the temporal dependence; see [17, Sec. III] for a detailed discussion.

## 3. STABILITY PROPERTIES

## 3.1. Problem Formulation

We consider the graph-aware generative dynamics (4)–(5) and study their stability to perturbations of the underlying graph G. In particular, let L denote the nominal GSO and L<sup>˜</sup> be a perturbed GSO satisfying the relative perturbation model (1). Let $\Phi _ { t } ( \mathbf { x } _ { 0 } ; \mathbf { L } )$ denote the solution of (4) at time t with initial condition $\mathbf { x } _ { 0 } \sim p _ { 0 }$ . Because node indexing is arbitrary, we compare the nominal and perturbed dynamics modulo permutations. For a permutation $\mathbf { P } \in \mathcal { P }$ , define

$$
\Phi _ { t } ( \mathbf { P } ^ { \top } \mathbf { x } _ { 0 } ; \mathbf { L } ) \sim \pi _ { t } ^ { \mathbf { P } } , \quad \mathbf { P } ^ { \top } \Phi _ { t } ( \mathbf { x } _ { 0 } ; \tilde { \mathbf { L } } ) \sim \tilde { \pi } _ { t } ^ { \mathbf { P } } .\tag{6}
$$

Our goal is to characterize the Wasserstein distance between these distributions as a function of the structural perturbation $\| \mathbf { E } ^ { \star } \| _ { 2 } \leq \varepsilon .$

## 3.2. Permutation Equivariance

To ensure the generative dynamics respect the graph’s structural symmetries, the stochastic differential equation (SDE) in (4) must be permutation equivariant. The graph filter H (L) and the GNN vector field $u _ { t } ^ { \theta } ( \cdot ; \hat { \mathbf { L } } )$ are equivariant by design (cf. Section 2.2). We further assume that for any $\mathbf { P } \in \mathcal { P }$ , the bias and diffusion satisfy $\mathbf { P } ^ { \top } \alpha _ { t } = \alpha _ { t } ~ ( { \mathrm { e . g . , ~ } } \mathfrak { a }$ uniform drift $\mathbf { \alpha } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf { { \alpha \alpha } } \mathbf \mathbf { { \alpha \alpha } } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha \alpha } \mathbf { \alpha } \mathbf \mathbf { \alpha \alpha } \mathbf { \alpha \alpha } \mathbf \mathbf { \alpha \alpha } \mathbf \mathbf { \alpha \alpha \alpha } \mathbf \mathbf { \alpha \alpha \alpha \alpha \alpha } \mathbf \mathbf \alpha \mathbf \alpha \mathbf  \alpha \alpha \alpha \alpha \alpha \alpha \alpha \alpha \alpha \alpha \alpha \alpha \alpha \alpha \alpha \delta \alpha \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta $ ) and $\mathbf { P } ^ { \top } \pmb { \Sigma } _ { t } \mathbf { P } = \pmb { \Sigma } _ { t }$ (e.g., isotropic noise $\Sigma _ { t } = \sigma _ { t } \mathbf { I } )$ for all $t \in [ 0 , 1 ]$ . Under these conditions, node relabeling commutes with the stochastic dynamics<sup>1</sup>.

Proposition 1. For any initial distribution $\mathbf { x } _ { 0 } \sim p _ { 0 } ,$ the probability laws of the generated trajectories of (4) satisfy

$$
\begin{array} { r } { \mathbf { P } ^ { \top } \Phi _ { t } ( \mathbf { x } _ { 0 } ; \mathbf { L } ) \stackrel { d } { = } \Phi _ { t } ( \mathbf { P } ^ { \top } \mathbf { x } _ { 0 } ; \mathbf { P } ^ { \top } \mathbf { L } \mathbf { P } ) , } \end{array}
$$

for all $\mathbf { P } \in \mathcal { P }$ and $t \in [ 0 , 1 ]$ , where $\circeq$ means equal in distribution.

## 3.3. Stability

To establish the stability of the generated distributions, we first formally introduce the structural stability and Lipschitz continuity constants characterizing the drift vector field components in (5). As stated in Section 2.2, graph filters and GNNs are fundamentally stable operators. For a perturbation $\| \mathbf { E } ^ { \star } \| _ { 2 } \leq \varepsilon$ and its corresponding optimal permutation $\mathbf { P } _ { 0 }$ (cf. (1)), we define the structural stability constants $\Gamma _ { H }$ and $\Gamma _ { u }$ such that the filter and GNN in (5) satisfy

$$
\begin{array} { r l } & { \quad \| \mathbf { P } _ { 0 } ^ { \top } \mathbf { H } _ { t } ( \tilde { \mathbf { L } } ) \mathbf { x } - \mathbf { H } _ { t } ( \mathbf { L } ) \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } \| \leq \left( \Gamma _ { H } \varepsilon + \mathcal { O } ( \varepsilon ^ { 2 } ) \right) \| \mathbf { x } \| , } \\ & { \| \mathbf { P } _ { 0 } ^ { \top } u _ { t } ^ { \theta } ( \mathbf { x } ; \tilde { \mathbf { L } } ) - u _ { t } ^ { \theta } ( \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } ; \mathbf { L } ) \| \leq \left( \Gamma _ { u } \varepsilon + \mathcal { O } ( \varepsilon ^ { 2 } ) \right) \| g ( \mathbf { x } , t ) \| , } \end{array}
$$

for all $\mathbf { x } \in \mathbb { R } ^ { N }$ and $t \in [ 0 , 1 ]$ . Furthermore, graph filters and GNNs are Lipschitz continuous [17, Prop. 3]. We denote their respective one-sided Lipschitz constants as m<sub>H</sub> and $m _ { u } .$ satisfying

$$
\begin{array} { r } { \langle \mathbf { H } _ { t } ( \mathbf { L } ) ( \mathbf { x } - \mathbf { y } ) , \mathbf { x } - \mathbf { y } \rangle \leq m _ { H } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } , } \\ { \langle u _ { t } ^ { \theta } ( \mathbf { x } ; \mathbf { L } ) - u _ { t } ^ { \theta } ( \mathbf { y } ; \mathbf { L } ) , \mathbf { x } - \mathbf { y } \rangle \leq m _ { u } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } , } \end{array}
$$

for all $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { N }$ and $t \in [ 0 , 1 ]$ . Combining these properties yields the stability and continuity of the drift vector field $b _ { t } ( \mathbf { x } ; \mathbf { L } )$

Lemma 1. The vectorfield b (x; L) in (5) is structurally stable, i.e., for the optimal permutation $\mathbf { P } _ { 0 }$ defining the perturbation one has

$$
\begin{array} { r } { \| \mathbf { P } _ { 0 } ^ { \top } b _ { t } ( \mathbf { x } ; \tilde { \mathbf { L } } ) - b _ { t } \big ( \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } ; \mathbf { L } \big ) \| \leq \varepsilon \big ( \Gamma _ { H } \| \mathbf { x } \| + \Gamma _ { u } \| g ( \mathbf { x } , t ) \| \big ) } \\ { + \mathcal { O } ( \varepsilon ^ { 2 } ) ( \| \mathbf { x } \| + \| g ( \mathbf { x } , t ) \| ) . } \end{array}
$$

Lemma 2. The vector $\hbar e l d b _ { t } ( \mathbf { x } ; \mathbf { L } )$ is one-sided Lipschitz with combined constant $m _ { b } = m _ { H } + m _ { u } , i . e .$

$$
\begin{array} { r } { \langle b _ { t } ( \mathbf { x } ; \mathbf { L } ) - b _ { t } ( \mathbf { y } ; \mathbf { L } ) , \mathbf { x } - \mathbf { y } \rangle \leq m _ { b } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } . } \end{array}
$$

Equipped with these properties, we can establish the following key stability result for the distributions generated by the SDE (4).

Theorem 1. For any initial distribution $\mathbf { x } _ { 0 } \sim p _ { 0 }$ , assume that the components ofthe drift $b _ { t } ( \mathbf { x } ; \mathbf { L } )$ in (5) are permutation equivariant, and that $b _ { t }$ satisfies the structural stability and one-sided Lipschitz properties stated in Lemmata 1 and 2. Then, the Wasserstein- $- 2 ( W _ { 2 } )$ distance between the probability laws generated by the nominal and perturbed dynamics (cf. (6)) satisfy

$$
\operatorname* { m i n } _ { \mathbf { P } \in \mathcal { P } } W _ { 2 } \Big ( \tilde { \pi } _ { t } ^ { \mathbf { P } } , \pi _ { t } ^ { \mathbf { P } } \Big ) \leq \Omega _ { t } \big ( \Gamma _ { H } C _ { H , t } + \Gamma _ { u } C _ { u , t } \big ) \varepsilon + \mathcal { O } ( \varepsilon ^ { 2 } ) ,\tag{7}
$$

for all $t \in [ 0 , 1 ] ,$ , where $\begin{array} { r } { \Omega _ { t } : = \int _ { 0 } ^ { t } e ^ { m _ { b } ( t - s ) } d s } \end{array}$ and

$$
\begin{array} { r l } & { C _ { H , t } : = \underset { \mathbf { P } \in \mathcal { P } } { \operatorname* { m a x } } \underset { s \in [ 0 , t ] } { \operatorname* { s u p } } \sqrt { \mathbb { E } \left[ \left\| \Phi _ { s } ( \mathbf { P } ^ { \top } \mathbf { x } _ { 0 } ; \mathbf { L } ) \right\| ^ { 2 } \right] } , } \\ & { C _ { u , t } : = \underset { \mathbf { P } \in \mathcal { P } } { \operatorname* { m a x } } \underset { s \in [ 0 , t ] } { \operatorname* { s u p } } \sqrt { \mathbb { E } \left[ \left\| g \big ( \Phi _ { s } ( \mathbf { P } ^ { \top } \mathbf { x } _ { 0 } ; \mathbf { L } ) , s \big ) \right\| ^ { 2 } \right] } . } \end{array}
$$

Proofsketch. Recall $\mathbf { P } _ { 0 } \in \mathcal { P }$ in (1), let $\mathbf { x } _ { t } : = \boldsymbol { \Phi } _ { t } ( \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } _ { 0 } ; \mathbf { L } )$ and $\tilde { \mathbf { x } } _ { t } : = \boldsymbol { \Phi } _ { t } ( \mathbf { x } _ { 0 } ; \tilde { \mathbf { L } } )$ , and define $\mathbf { z } _ { t } : = \mathbf { P } _ { 0 } ^ { \top } \tilde { \mathbf { x } } _ { t } - \mathbf { x } _ { t }$ . Consider a synchronous coupling sharing the initial state $\mathbf { x } _ { 0 }$ and driving the perturbed trajectory with the standard Brownian motion $\tilde { \mathbf { w } } _ { t } : = \mathbf { P } _ { 0 } \mathbf { w } _ { t }$ By the equivariance ${ \bf P } _ { 0 } ^ { \top } { \bf { \Sigma } } { \bf { \Sigma } } { \bf { \Sigma } } _ { t } { \bf { P } } _ { 0 } = { \bf { \Sigma } } { \bf { \Sigma } } _ { t }$ , the stochastic noise cancels, yielding the ordinary differential equation (ODE)

$$
d \mathbf { z } _ { t } = \left[ \mathbf { P } _ { 0 } ^ { \top } b _ { t } \big ( \tilde { \mathbf { x } } _ { t } ; \tilde { \mathbf { L } } \big ) - b _ { t } \big ( \mathbf { x } _ { t } ; \mathbf { L } \big ) \right] d t .
$$

Adding and subtracting $\mathbf { P } _ { 0 } ^ { \top } b _ { t } ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \tilde { \mathbf { L } } )$ , we have

$$
\begin{array} { r } { \displaystyle \frac { 1 } { 2 } \frac { d } { d t } \| \mathbf { z } _ { t } \| ^ { 2 } = \langle \mathbf { z } _ { t } , \mathbf { P } _ { 0 } ^ { \top } \left( b _ { t } \big ( \tilde { \mathbf { x } } _ { t } ; \tilde { \mathbf { L } } \big ) - b _ { t } \big ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \tilde { \mathbf { L } } \big ) \right) \rangle } \\ { + \langle \mathbf { z } _ { t } , \mathbf { P } _ { 0 } ^ { \top } b _ { t } ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \tilde { \mathbf { L } } ) - b _ { t } \big ( \mathbf { x } _ { t } ; \mathbf { L } \big ) \rangle . } \end{array}
$$

Applying Lemma 2 to the first term, and Cauchy-Schwarz with Lemma 1 to the second, yields

$$
\begin{array} { r } { \displaystyle \frac { 1 } { 2 } \frac { d } { d t } \| { \bf z } _ { t } \| ^ { 2 } \leq m _ { b } \| { \bf z } _ { t } \| ^ { 2 } + \Big ( \varepsilon \big ( \Gamma _ { H } \| { \bf x } _ { t } \| + \Gamma _ { u } \| g ( { \bf x } _ { t } , t ) \| \big ) } \\ { \displaystyle + \mathcal { O } ( \varepsilon ^ { 2 } ) \big ( \| { \bf x } _ { t } \| + \| g ( { \bf x } _ { t } , t ) \| \big ) \Big ) \| { \bf z } _ { t } \| . } \end{array}
$$

Dividing by $\left\| \mathbf { z } _ { t } \right\|$ and applying Gronwall’s inequality¨ $( \mathbf { z } _ { 0 } = \mathbf { 0 } )$ gives

$$
\begin{array} { r } { \| \mathbf { z } _ { t } \| \leq \displaystyle \int _ { 0 } ^ { t } e ^ { m _ { b } ( t - s ) } \Big ( \varepsilon \big ( \Gamma _ { H } \| \mathbf { x } _ { s } \| + \Gamma _ { u } \| g ( \mathbf { x } _ { s } , s ) \| \big ) } \\ { + O ( \varepsilon ^ { 2 } ) \big ( \| \mathbf { x } _ { s } \| + \| g ( \mathbf { x } _ { s } , s ) \| \big ) \Big ) d s . } \end{array}
$$

The 2-Wasserstein distance is bounded by this coupling, namely $W _ { 2 } ( \tilde { \pi } _ { t } ^ { \mathbf { P } _ { 0 } } , \pi _ { t } ^ { \mathbf { P } _ { 0 } } ) \leq \sqrt { \mathbb { E } \big [ \| \mathbf { z } _ { t } \| ^ { 2 } \big ] }$ . Applying Minkowski’s integral inequality and the $L ^ { 2 }$ triangle inequality, yields

$$
\begin{array} { r l } & { \sqrt { \mathbb { E } \left[ \| { \bf z } _ { t } \| ^ { 2 } \right] } \leq } \\ & { \varepsilon \int _ { 0 } ^ { t } e ^ { m _ { b } ( t - s ) } \Big ( \Gamma _ { H } \sqrt { \mathbb { E } \left[ \| { \bf x } _ { s } \| ^ { 2 } \right] } + \Gamma _ { u } \sqrt { \mathbb { E } \left[ \| g ( { \bf x } _ { s } , s ) \| ^ { 2 } \right] } \Big ) d s } \\ & { + \mathcal { O } ( \varepsilon ^ { 2 } ) \int _ { 0 } ^ { t } e ^ { m _ { b } ( t - s ) } \Big ( \sqrt { \mathbb { E } \left[ \| { \bf x } _ { s } \| ^ { 2 } \right] } + \sqrt { \mathbb { E } \left[ \| g ( { \bf x } _ { s } , s ) \| ^ { 2 } \right] } \Big ) d s . } \end{array}
$$

Using $C _ { H , t } \geq \sqrt { \mathbb { E } [ \| \mathbf { x } _ { s } \| ^ { 2 } ] }$ and $C _ { u , t } \geq \sqrt { \mathbb { E } [ \| g ( { \bf x } _ { s } , s ) \| ^ { 2 } ] }$ ] yields the bound in (7), completing the proof sketch. □

We find stability is governed by three terms: $( \mathrm { i } ) \ \Gamma _ { H }$ and $\Gamma _ { u } ,$ which capture the inherent stability of the graph filter and GNN; (ii) $\Omega _ { t }$ , which is exponential in the combined Lipschitz constant and dictates how the continuous dynamics amplify errors; and $( \mathrm { i i i } ) C _ { H , i }$ <sub>t</sub> and $C _ { u , t }$ , the trajectory suprema across all initial condition permutations.

## 4. STABLE GRAPH FILTERS

Theorem 1 offers actionable insights to promote robustness in the generative pipeline. While the GNN constants $\{ \Gamma _ { u } , m _ { u } \}$ can be regularized during training [17, Sec. V], the filter constants $\{ \Gamma _ { H } , m _ { H } \}$ are determined by the choice of reference dynamics. Instead of relying on the standard heat equation whereby $\mathbf { H ( L ) } = - \kappa \mathbf { L }$ in (5), here we formulate a principled filter design problem that enforces similar signal smoothness while improving its structural stability.

We henceforth assume that L is a normalized graph Laplacian with spectrum in $[ 0 , \lambda _ { \operatorname* { m a x } } ]$ and restrict our attention to timeindependent reference filters by setting $\begin{array} { l c l } { h _ { k } ( t ) } & { \equiv } & { h _ { k } } \end{array}$ for $k \_ =$ $0 , \ldots , K ,$ so that $\begin{array} { r } { \mathbf { H } ( \mathbf { L } ) : = \sum _ { k = 0 } ^ { K } h _ { k } \mathbf { L } ^ { k } } \end{array}$ . Nulling the bias term, $h _ { 0 } = 0$ , we parameterize the corresponding frequency response as $\begin{array} { r } { h ( \lambda ) : = \sum _ { k = 1 } ^ { \tilde { K } } h _ { k } \lambda ^ { k } } \end{array}$ , with $\mathbf { h } : = [ \bar { h } _ { 1 } , \ldots , \bar { h } _ { K } ] ^ { \dagger } \in \mathbf { \bar { \mathbb { R } } } ^ { K }$ . Further imposing no signal amplification, $h ( \lambda ) ~ \leq ~ 0 ,$ yields a one-sided Lipschitz constant m $_ H = 0$ . Thus, enhancing robustness reduces to minimizing the structural stability constant $\Gamma _ { H }$ . Since $\Gamma _ { H }$ is proportional to the integral Lipschitz constant of the filter [24], our design objective is to select h so as to minimize max<sub>λ</sub> $| \lambda h ^ { \prime } ( \lambda )$ |.

To also guarantee the filter generates smooth signals, we bound the Dirichlet energy $\mathcal { E } ( \mathbf { x } ) = \mathbf { x } ^ { \top }$ Lx of its output. Under the linear dynamics $d \mathbf { x } _ { t } = \mathbf { H } ( \dot { \mathbf { L } } ) \mathbf { x } _ { t } d t ,$ , the spectral response at $t \ : = \ : 1$ and frequency $\lambda _ { i }$ is $\hat { x } _ { 1 , i } = e ^ { h ( \lambda _ { i } ) } \hat { x } _ { 0 , i }$ . For a normalized initial signal $\| \mathbf { x } _ { 0 } \| \leq 1$ , imposing a prescribed smoothness bound $\eta > 0$ yields

$$
\mathcal { E } ( \mathbf { x } _ { 1 } ) \leq \operatorname* { m a x } _ { \lambda \in [ 0 , \lambda _ { \operatorname* { m a x } } ] } \lambda e ^ { 2 h ( \lambda ) } \leq \eta .
$$

Taking the logarithm directly simplifies this smoothness constraint to $\begin{array} { r } { h ( \bar { \lambda } ) \leq \frac { 1 } { 2 } \bar { \ln ( \frac { \eta } { \lambda } ) } } \end{array}$ , for all $\lambda > 0$

Introducing an auxiliary variable $\rho \geq 0$ , we formulate the minmax filter design problem in epigraph form as

$$
\begin{array} { r l r } & { \underset { \mathbf { h } , \rho } { \mathrm { m i n } } \rho } \\ & { \mathrm { s . t . } } & { - \rho \leq \lambda h ^ { \prime } ( \lambda ) \leq \rho , \qquad \forall \lambda \in ( 0 , \lambda _ { \operatorname* { m a x } } ] , } \\ & { h ( \lambda ) \leq \displaystyle \frac { 1 } { 2 } \ln \left( \frac { \eta } { \lambda } \right) , \qquad } & { \forall \lambda \in ( 0 , \lambda _ { \operatorname* { m a x } } ] , } \\ & { h ( \lambda ) \leq 0 , \qquad } & { \forall \lambda \in [ 0 , \lambda _ { \operatorname* { m a x } } ] . } \end{array}\tag{8}
$$

Since $h ( \lambda )$ and $h ^ { \prime } ( \lambda )$ are linear in the polynomial coefficients h, (8) is a semi-infinite linear program. In practice, it can be approximated by discretizing $( 0 , \lambda _ { \operatorname* { m a x } } ]$ over a dense grid and solving the resulting finite-dimensional linear program [27, Sec. 7].

## 5. NUMERICAL EXPERIMENTS

Here we test the proposed stable filters on a graph signal generation task using samples from a target data distribution. We consider two test cases: (i) a synthetic setting based on a Stochastic Block Model (SBM) graph [28]; and (ii) a real-world scenario using fMRI data [29], where functional brain connectivity defines the graph structure. Since our prime objective is to perform an ablation study on the graph filter design, we utilize a fixed architecture and training procedure for the learned vector field $u _ { t } ^ { \theta }$ across all configurations. For an approach to robustify $u _ { t } ^ { \theta }$ itself; see [17]. Code to reproduce the experiments is available at github.com/mschmi21/stable graph fm.

![](images/155819891b371c21512feed95fb97790cfc2f826ce2af7b0fbef622175a326b9.jpg)

![](images/b0bd567e4d3346263d956d2c95725e34b553ed00847c4182fa5ad9c24d323831.jpg)

![](images/00a4e849a323196268059611ff7076a53762037826565c8d8037b69b65af3e96.jpg)

![](images/c6c05afb8b9dfdf5701c2033ded7ca1eab5c780828adc216dde75a5231895421.jpg)  
Fig. 1. Performance and stability under structural graph perturbations. Top: SBM graph under controlled synthetic relative perturbations (ε). Bottom: fMRI signals under data-driven graph estimation errors, varying the fraction of training samples used. Generative quality (left) shows higher-order optimized filters $( K = 2 , 4 )$ achieve competitive or better $W _ { 1 }$ distances compared to the $K = 1$ baseline (heat equation). Empirical stability (right) demonstrates that increasing the optimized filter order yields significantly lower output variation under perturbations. Lines and shaded regions represent medians and 25th–75th percentiles across 10 independent runs.

## 5.1. Experimental Setup

Datasets. In the synthetic setting, we use an SBM graph with two communities of 10 nodes each $( N = 2 0 )$ . Graph signals are drawn from a Gaussian distribution with standard deviation 1. One community has a mean of 1, while the other has a mean of −1. For the real-data test case, we use a single subject from the HCP dataset, producing data in $\mathbb { R } ^ { 3 6 0 \times 1 1 9 0 }$ $( N = 3 6 0$ brain regions, 1190 time points). Each time point is treated as an individual graph signal.

Filter Design and Architecture. To isolate the effect of the reference dynamics, we synthesize stable filters of orders $K \in \{ 1 , 2 , 4 \}$ by solving the semi-infinite linear program in (8). Notably, for $K =$ 1, this optimization naturally recovers the heat equation filter, which serves as our baseline. The smoothness constraint parameter η is selected via a grid search on the validation set. For the learned vector field $u _ { t } ^ { \theta }$ , we employ the GCNPolicy model introduced in [9]. For simplicity, we henceforth set $\mathbf { \alpha } _ { \alpha _ { t } } = \mathbf { 0 }$ and $\Sigma _ { t } = \mathbf { 0 }$ for all $t \in [ 0 , 1 ]$

Perturbations and Evaluation. We train all models on the unperturbed, nominal graph, using an independent coupling and filterdependent optimal paths (cf. [10]). Robustness is evaluated at test time. For the SBM graphs, we introduce a controlled synthetic relative perturbation of the form $\tilde { \mathbf { L } } = \mathbf { L } + \frac { 1 } { 2 } ( \mathbf { E } \mathbf { L } + \mathbf { L } \mathbf { E } )$ , with $\| \mathbf { E } \| _ { 2 } = \varepsilon$ For the fMRI data, perturbations are data-driven: test-time graphs are constructed using empirical correlation matrices estimated from varying data subsets. We report two metrics: (i) the Wasserstein-1 $( W _ { 1 } )$ distance to quantify generative performance; and (ii) the variation $\| \Phi _ { 1 } ( \mathbf { x } _ { 0 } ; \tilde { \mathbf { L } } ) - \Phi _ { 1 } ( \mathbf { x } _ { 0 } ; \mathbf { L } ) \|$ in generated outputs induced by graph perturbations, estimated via an Euler sampler.

## 5.2. Results and Discussion

Fig. 1 depicts the results for the synthetic SBM (top) and fMRI (bottom) experiments. In terms of generative quality measured by the $W _ { 1 }$ distance, the higher-order stable filters $( K = 2 , 4 )$ achieve competitive or slightly better performance compared to the first-order baseline $( K = 1 )$ under low perturbations, while clearly outperforming it under larger perturbations. More importantly, the empirical stability evaluations demonstrate that increasing the filter order offers added robustness to structural errors. By actively minimizing the structural stability constant $\Gamma _ { H }$ , the generative trajectories of the higher-order filters exhibit markedly lower output divergence. Finally, while the theoretical stability bounds conservatively overestimate the empirical errors, they remain highly informative and align with the relative differences observed in practice.

## 6. CONCLUSION

In this work, we established explicit stability bounds for a general class of graph-aware continuous-time generative dynamics (cf. Theorem 1). By quantifying the Wasserstein distance between distributions generated under nominal and perturbed graphs, our analysis reveals how the choice of reference dynamics directly impacts the structural robustness of the model. Guided by these theoretical insights, we introduced a principled optimization framework to design stable graph filters. Our empirical evaluations confirm that appropriately selecting this reference filter significantly enhances the stability of the generative model against graph perturbations, without sacrificing the quality of the generated signals.

## 7. COMPLIANCE WITH ETHICAL STANDARDS

This study uses previously collected, de-identified Human Connectome Project (HCP) data [29]. No new human-subject data were collected, and no additional ethical approval was required.

## 8. REFERENCES

[1] W. Huang, T. A. Bolton, J. D. Medaglia, D. S. Bassett, A. Ribeiro, and D. Van De Ville, “A graph signal processing perspective on functional brain imaging,” Proc. IEEE, vol. 106, no. 5, pp. 868–885, 2018.

[2] M. Schmidt, S. Silva, F. Larroca, G. Mateos, and P. Muse,´ “Graph contrastive learning for connectome classification,” in Proc. Asilomar Conf. Signals, Syst., Computers, 2025, pp. 1027–1032.

[3] B. Yu, H. Yin, and Z. Zhu, “Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting,” in Proc. Int. Joint Cont. Artificial Intell. (IJCAI), 2018, pp. 3634–3640.

[4] Y. Zhu, C. Wang, Q. Zhang, and H. Xiong, “Graph signal diffusion model for collaborative filtering,” in Proc. ACM SIGIR Conf. Res. Develop. Inf. Retrieval, 2024, pp. 1380–1390.

[5] Y. B. Uslu, S. Hadou, S. Rozada, S. S. Bidokhti, and A. Ribeiro, “Generative diffusion models of stochastic graph signals,” arXiv preprint arXiv:2607.06833, 2026.

[6] V. De Bortoli, J. Thornton, J. Heng, and A. Doucet, “Diffusion Schrodinger bridge with applications to score-based generative¨ modeling,” in Proc. Adv. Neural. Inf. Process. Syst. (NeurIPS), 2021, pp. 1–15.

[7] T. Chen, G.-H. Liu, and E. Theodorou, “Likelihood training of Schrodinger bridge using forward-backward SDEs theory,”¨ in Proc. Int. Conf. Learn. Representations (ICLR), 2022, pp. 1–27.

[8] G.-H. Liu, Y. Lipman, M. Nickel, B. Karrer, E. Theodorou, and R. T. Chen, “Generalized Schrodinger bridge matching,”¨ in Proc. Int. Conf. Learn. Representations (ICLR), 2024, pp. 1–26.

[9] M. Yang, “Topological Schrodinger bridge matching,” in ¨ Proc. Int. Conf. Learn. Representations (ICLR), vol. 2025, 2025, pp. 1–42.

[10] K. Wyrwal, I. I. Ceylan, and A. Tong, “Topological flow matching,” in Proc. Int. Conf. Learn. Representations (ICLR), vol. 2026, 2026, pp. 1–26.

[11] S. Rozada, Vimal K B, A. Cavallo, A. G. Marques, H. Jamali-Rad, and E. Isufi, “Graph-aware diffusion for signal generation,” in Proc. IEEE Intl. Conf. Acoustics, Speech and Signal Process. (ICASSP), 2026, pp. 461–465.

[12] Y. Shi, V. De Bortoli, A. Campbell, and A. Doucet, “Diffusion Schrodinger bridge matching,” in ¨ Proc. Adv. Neural. Inf. Process. Syst. (NeurIPS), 2023, pp. 1–41.

[13] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in Proc. Int. Conf. Learn. Representations (ICLR), 2023, pp. 1–28.

[14] E. D. Kolaczyk, Statistical Analysis ofNetwork Data: Methods and Models. New York, NY: Springer, 2009.

[15] G. Mateos, S. Segarra, A. G. Marques, and A. Ribeiro, “Connecting the dots: Identifying network structure via graph signal processing,” IEEE Signal Process. Mag., vol. 36, no. 3, pp. 16– 43, May 2019.

[16] X. Dong, D. Thanou, M. Rabbat, and P. Frossard, “Learning graphs from data: A signal representation perspective,” IEEE Signal Process. Mag., vol. 36, no. 3, pp. 44–63, 2019.

[17] M. Schmidt and G. Mateos, “Stability of flow models for graph signals,” arXiv preprint arXiv:2607.07510, 2026.

[18] A. Ortega, P. Frossard, J. Kovaceviˇ c, J. M. Moura, and P. Van-´ dergheynst, “Graph signal processing: Overview, challenges, and applications,” Proc. IEEE, vol. 106, no. 5, pp. 808–828, 2018.

[19] A. Sandryhaila and J. M. F. Moura, “Discrete signal processing on graphs: Frequency analysis,” IEEE Trans. Signal Process., vol. 62, no. 12, pp. 3042–3054, 2014.

[20] E. Isufi, F. Gama, D. I. Shuman, and S. Segarra, “Graph filters for signal processing and machine learning on graphs,” IEEE Trans. Signal Process., vol. 72, pp. 4745–4781, 2024.

[21] F. Gama, A. G. Marques, G. Leus, and A. Ribeiro, “Convolutional neural network architectures for signals supported on graphs,” IEEE Trans. Signal Process., vol. 67, no. 4, p. 1034–1049, 2019.

[22] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. Int. Conf. Learn. Representations (ICLR), 2017, pp. 1–14.

[23] M. Defferrard, X. Bresson, and P. Vandergheynst, “Convolutional neural networks on graphs with fast localized spectral filtering,” in Proc. Adv. Neural. Inf. Process. Syst. (NeurIPS), 2016, pp. 1–9.

[24] F. Gama, J. Bruna, and A. Ribeiro, “Stability properties of graph neural networks,” IEEE Trans. Signal Process., vol. 68, p. 5680–5695, 2020.

[25] L. Ruiz, F. Gama, and A. Ribeiro, “Graph neural networks: Architectures, stability and transferability,” Proc. IEEE, vol. 109, no. 5, pp. 660–682, 2021.

[26] C. Leonard, “A survey of the Schr´ odinger problem and some of¨ its connections with optimal transport,” Discrete & Continuous Dynamical Systems - A, vol. 34, no. 4, pp. 1533–1574, 2014.

[27] R. Hettich and K. O. Kortanek, “Semi-infinite programming: Theory, methods, and applications,” SIAM J. Optim., vol. 35, no. 3, pp. 380–429, 1993.

[28] P. W. Holland, K. B. Laskey, and S. Leinhardt, “Stochastic blockmodels: First steps,” Social Networks, vol. 5, no. 2, pp. 109–137, 1983.

[29] Human Connectome Project, https://www.humanconnectome. org/.