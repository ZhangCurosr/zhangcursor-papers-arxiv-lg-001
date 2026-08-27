# Multi-output Gaussian process prediction of physical fields under linear equality constraints

Mahamat Hamdan Nassouradine<sup>a,c,∗</sup>, Clément Gauchy<sup>a</sup>, Pierre-Emmanuel Angeli<sup>b</sup> and Sébastien Da Veiga<sup>c</sup>

<sup>a</sup>Université Paris-Saclay, CEA, Service de Génie Logiciel pour la Simulation, Gif-sur-Yvette, 91191, France

<sup>b</sup>Université Paris-Saclay, CEA, Service de Thermohydraulique et de Mécanique des Fluides, Gif-sur-Yvette, 91191, France <sup>c</sup>Univ Rennes, Ensai, CNRS, CREST - UMR 9194, Rennes, F-35000, France

## A R T I C L E I N F O

Keywords:   
Metamodeling   
linear equality constraints   
physical fields   
Gaussian process   
PCA

## A BS T R AC T

We address the simultaneous prediction of multiple high-dimensional physical fields governed by linear equality constraints, a setting that arises in many real-world applications in physics machine learning. Gaussian process (GP) regression is a widely used surrogate modeling approach due to its efectiveness in small-sample regimes and its ability to provide uncertainty quantification. However, applying GP models in this setting raises two major challenges: the high dimensionality of the discretized output fields and the enforcement of the physical constraint in predictions. For the latter, a common strategy consists in deducing one output from the others via the constraint relation. Through an empirical benchmark, we show that this deductive approach is sensitive to the arbitrary choice of which output to deduce, afecting both predictive accuracy and uncertainty quantification. Consequently, there is a need for an approach that treats all fields symmetrically while strictly respecting the underlying physics. Motivated by these limitations, we propose a robust framework for jointly modeling constrained multi-field data. Our approach first leverages a specific PCA procedure for multi-field data, coined row-wise PCA, which has the interesting property of preserving the constraint in the latent space. Since standard PCA strategies for multi-field data (field-wise, column-wise) do not preserve such constraints, we investigate theoretically the optimality of the row-wise choice and provide conditions under which it incurs a negligible reconstruction cost. In a second step, we consider a linearly-constrained multi-output GP approach based on a specific kernel parametrization which is trained on the latent space of row-wise PCA. The proposed framework is validated on a population dynamics problem and on an industrial computational fluid dynamics application, which involves the prediction of Reynolds stress tensor components under the incompressibility constraint. We demonstrate competitive predictive performance while guaranteeing strict constraint satisfaction.

## 1. Introdution.

Metamodeling is now a fundamental approach for approximating computationally expensive numerical simulations or costly experiments (O’Hagan, 2006; Kennedy and O’Hagan, 2000; Santner, Williams and Notz, 2003). This technique is essential for various engineering applications, such as structural optimization (Jansson, Nilsson and Redhe, 2003), aeronautics (Sobieszczanski-Sobieski and Haftka, 1997), multifidelity optimization (Forrester, Sóbester and Keane, 2007; Forrester, Sobester and Keane, 2008), sensitivity analysis (Nanty, Helbert, Marrel, Pérot and Prieur, 2016), or reliability analysis (Su, Peng and Hu, 2017; Li, Sadoughi, Hu and Hu, 2020). In this work, we specifically focus on building a metamodel that simultaneously predicts multiple spatial or temporal fields governed by linear equality constraints, which are ubiquitous in physics-based simulations. Gaussian process (GP) regression, also known as kriging in geostatistics (Matheron, 1963), has established itself as a method of choice in metamodeling context, thanks to its flexibility in capturing nonlinear relationships, its principled quantification of predictive uncertainty, and its efectiveness in small-sample regimes (Rasmussen and Williams, 2005). This approach has been successfully applied to approximate scalar outputs across various domains: uncertainty quantification (O’Hagan, 2006), thermohydraulic studies in nuclear safety (Marrel, Iooss and Chabridon, 2022), or magnetic field reconstruction in physics (Macêdo and Castro, 2010). To capture dependencies across multiple scalar outputs, this framework has been naturally extended to Multi-Output Gaussian Processes (MOGP) using structured covariance models such as the linear model of coregionalization (LCM) (Goulard and Voltz, 1992; Rougier, 2008), the intrinsic coregionalization model (ICM), or process convolutions (Higdon, 2002; Alvarez, Rosasco and Lawrence, 2012).

In many physical applications of interest, however, the output fields are subject to linear equality constraints: for instance, an incompressibility condition on the velocity field, or a conservation law imposing that species mass fractions sum to unity at every point. Accounting for these constraints is crucial: first, it guarantees physically consistent predictions. Second, we expect that accounting for the constraints will produce a more accurate metamodel. When such constraints are of the form $\begin{array} { r } { \sum _ { j = 1 } ^ { Q } \alpha _ { j } ( \mathbf { x } ) f _ { j } ( \mathbf { x } ) ~ = ~ c ( \mathbf { x } ) } \end{array}$ , a seemingly straightforward strategy consists in exploiting the constraint algebraically: one output is deduced from the others via the constraint relation, and a standard (unconstrained) GP surrogate is built only on the remaining � − 1 fields. We refer to this as the deductive approach. It is appealing because it allows the direct reuse of any existing unconstrained framework and at first sight guarantees constraint satisfaction on the predictive mean by construction. However, this approach may sufer from a fundamental structural limitation: the treatment of the outputs becomes arbitrary, since the deduced field is not modeled jointly with the others but is instead recovered a posteriori through the constraint relation. As a consequence, the deduced outpu accumulates the prediction errors of all modeled fields, and its predictive quality depends on the arbitrary choice of the deduced output. This phenomenon is closely related to the well-known dependence on the reference category in the additive log-ratio transformation of compositional data analysis (Aitchison, 1986), where the statistical properties of the transformed variables are sensitive to the choice of the component used as denominator.

The structural limitations of this deductive approach motivate the search for methods that integrate constraints directly into the GP model. Several methods have been developed for integrating constraints into GP regression (Swiler, Gulian, Frankel, Safta and Jakeman, 2020). For inequality constraints, approaches include data augmentation (Abrahamsen and Benth, 2001), finite-dimensional projections (Da Veiga and Marrel, 2020; Maatouk and Bay, 2017), and shape constraint methods (Wang and Berger, 2016). For equality constraints, early work on linear forms dates back to Graepel (2003), leading to more general approaches that encode constraints directly in the kernel by restricting the GP to the solution space of a linear operator (Jidling, Wahlström, Wills and Schön, 2017; Lange-Hegermann, 2018). The parametrization approach of Jidling et al. (2017), in particular, constructs a covariance kernel such that every sample path of the GP satisfies the constraint over the entire domain, thus ensuring constraint satisfaction on both the predictive mean and posterior samples. More recently, Pilar, Jidling, Schön and Wahlström (2022) specifically addressed linear and non linear constraints in multitask GPs. However, these constrained GP methods have been developed and demonstrated for problems with low-dimensional outputs, and they inherit the cubic scaling of full GP inference $\mathcal { O } ( N ^ { 3 } P ^ { 3 } )$ or $\mathcal { O } ( N ^ { 3 } + P ^ { 3 } )$ in favorable cases (Stegle, Lippert, Mooij, Lawrence and Borgwardt, 2011), where � is the sample size and � the output dimension, which precludes their direct application to high-dimensional discretized fields.

To overcome the dimensional limitation in general case, a widely adopted strategy in the GP literature combines a dimensionality reduction step with GP regression in the resulting low-dimensional latent space. The seminal contribution of Higdon, Gattiker, Williams and Rightley (2008) proposed applying Principal Component Analysis (PCA) to the output fields and training independent GPs on the principal component scores. More recently, Xing, Yu, Leung, Li, Wang and Shah (2021) extended this framework by independently reducing the dimension of each individual field and coupling these representations with MOGP through the LCM kernel, enabling thejoint modeling of several correlated fields in high dimensions. Other strategies have explored nonlinear alternatives such as Isomap-GP or KPCA-GP (Xing, Shah and Nair, 2015; Zhou and Peng, 2020), developed extended formulations such as the extended LCM (Wang, Zhang, Meng and Xing, 2022) that use invertible neural networks to capture spatial nonlinearities, or introduced higher-order GPs (Zhe, Xing and Kirby, 2019) exploiting Kronecker product properties to further reduce computational cost. We also mention the PCA-GP emulators of Herman, Stewart and Dingreville (2020) for predicting high-dimensional microstructure morphology fields, and the eficient surrogate of Mak, Sung, Wang, Yeh, Chang, Joseph, Yang and Wu (2018) for large eddy simulations, which also relies on dimensionality reduction combined with GP regression. While all these methods successfully address the dimensionality challenge, they do not naturally guarantee compliance with linear equality constraints unless they are embedded within a deductive approach, which, as discussed above, introduces its own structural limitations.

In this paper, we propose an approach that addresses both the dimensionality and the constraint challenges in a unified framework. The key idea is to perform dimensionality reduction in a way that preserves the constraint structure in the latent space, so that constrained GP regression can then be applied in this reduced space. We focus on PCA as the dimensionality reduction tool, since nonlinear alternatives are both less robust in small-sample regimes and incompatible with the linear constraint structure. For multi-field data, PCA admits several formulations depending on how the component fields are arranged before decomposition. In the geoscience and spatio-temporal statistics literature, this extension of PCA to vector fields is known as multivariate Empirical Orthogonal Functions (EOFs) (Preisendorfer and Mobley, 1988; Hannachi, Jollife and Stephenson, 2007; Sparnocchia, Pinardi and Demirov, 2003), with applications in climatology (Alvera-Azcárate, Barth, Beckers and Weisberg, 2007), oceanography (Liang, Mazlof, Rosso, Fang and Yu, 2018), and ecology (Petitgas, Doray, Huret, Massé and Woillez, 2014). Two main global formulations exist (Preisendorfer and Mobley, 1988): the time-modulation form (TMF, or column-wise PCA), which concatenates the fields column-wise and captures inter-field dependencies in the spatial dimension, and the spacemodulation form (SMF, or row-wise PCA), which stacks the fields row-wise and produces a shared spatial basis. In contrast to these global methods, a third option consists in applying the reduction independently to each component. Following Xing et al. (2021), we formally refer to this strategy as field-wise PCA. Theoretical underpinnings for such decoupled representations of vector fields can also be found in functional data analysis (Happ and Greven, 2018) and generalized Karhunen–Loève expansions (Perrin, Soize, Duhamel and Funfschilling, 2013). Among these strategies, the row-wise approach has a distinctive property: it preserves linear equality constraints exactly in the latent space, regardless of the truncation level. This makes it uniquely suited for coupling with a constrained MOGP model. We propose the row-wise PCA-Constrained MOGP framework (coined Row-CMO), which combines the row-wise PCA with constrained MOGP regression based on the parametrization approach of Jidling et al. (2017), ensuring that both the predictive mean and posterior samples satisfy the physical constraint to machine precision. The column-wise and field-wise PCA strategies, which do not preserve the constraint after truncation, can only be used within a deductive approach for constrained problems. A natural question then arises: what is the reconstruction cost of choosing row-wise PCA over these alternatives ? To address this, we provide a theoretical analysis comparing the three strategies from the perspective of reconstruction error. The main contributions of this paper are the following:

1. We propose the Row-CMO framework for predicting multi-field outputs under linear equality constraints.

2. Through empirical benchmarking, we demonstrate the inherent heterogeneity of deductive approaches, showing that performance can degrade significantly depending on the arbitrary choice of the deduced output.

3. We provide a theoretical analysis of multi-field PCA strategies (field-wise, column-wise, and row-wise) by bounding the excess reconstruction error associated with global methods. Based on classical perturbation theory, our analysis identifies inter-field spectral heterogeneity as the sole factor driving this discrepancy.

4. We validate Row-CMO on two case studies: the Lotka–Volterra population dynamics system (Wangersky, 1978), subject to a sum-to-zero conservation constraint, and a Computational Fluid Dynamics (CFD) application predicting Reynolds stress tensor components on the Buice–Eaton 2D difuser (Buice and Eaton, 2000a), governed by the incompressibility constraint.

The paper is organized as follows. Section 2 establishes the problem statement and recalls the necessary background on Gaussian process regression. Building on this framework, Section 3 exposes the vulnerabilities of the deductive approach to constraint enforcement. The core theoretical analysis of the diferent multi-field PCA strategies is then presented in Section 4. These theoretical insights and the proposed model are subsequently validated through numerical experiments in Section 5. All the numerical experiments, except for the CFD application, are available at https: //github.com/nasrmht/paper\_benchmark\_repo.

## 2. Problem statement

## 2.1. Problem formulation

We consider the output of a parameterized numerical simulator that produces � coupled scalar fields, each evaluated over a fixed set of � discretes coordinates (e.g spatial locations or time steps). The simulator is governed by a vector of input parameters $\mathbf { x } \in \mathcal { X } \subset \mathbb { R } ^ { D }$ , which may represent physical constants, boundary conditions, or model closure coeficients. For a given input $\mathbf { x } ^ { ( i ) }$ , the simulator returns � discretized outputs $\mathbf { y } _ { 1 } ^ { ( i ) } , \ldots , \mathbf { y } _ { Q } ^ { ( i ) }$ , where each $\mathbf { y } _ { j } ^ { ( i ) } = ( u _ { j } ( \boldsymbol { v } _ { 1 } , \mathbf { x } ^ { ( i ) } ) , \ldots , u _ { j } ( \boldsymbol { v } _ { S } , \mathbf { x } ^ { ( i ) } ) ) ^ { \top } \in \mathbb { R } ^ { S }$ collects the values of the �-th scalar field $u _ { j }$ evaluated at the fixed coordinates $\boldsymbol { \mathsf { v } } _ { 1 } , \ldots , \boldsymbol { \mathsf { v } } _ { S }$ . We denote by $\mathbf { y } ^ { ( i ) } = [ \mathbf { y } _ { 1 } ^ { ( i ) \top } , \ldots , \mathbf { y } _ { Q } ^ { ( i ) \top } ] ^ { \top } \in \mathbb { R } ^ { P }$ , with $P = Q \times S$ , the concatenated output vector. We assume that a training set of � simulation runs is available: $\big ( \mathbf { x } ^ { ( i ) } , \mathbf { y } ^ { ( i ) } = [ \mathbf { y } _ { 1 } ^ { ( i ) \top } , \dots , \mathbf { y } _ { Q } ^ { ( i ) \top } ] ^ { \top } \big ) _ { 1 \leq i \leq N ^ { \top } }$ , where � is typically small (on the order of tens to hundreds) due to the computational cost of each simulation. The data are assumed noise-free, as is the case for outputs of deterministic numerical simulators. Our goal is to approximate the mapping $\mathbf { f } = [ \mathbf { f } _ { 1 } ^ { \top } , \ldots , \mathbf { f } _ { O } ^ { \top } ] ^ { \top } : \mathcal { X }  \mathbb { R } ^ { P }$ such that

$$
\mathbf { y } ^ { ( i ) } = \mathbf { f } ( \mathbf { x } ^ { ( i ) } ) ,\tag{1}
$$

$i = 1 , \ldots , N$ . We further assume that the fields satisfy a linear equality constraint of the form

$$
\mathcal { F } [ \mathbf { f } ( \mathbf { x } ) ] = \sum _ { j = 1 } ^ { Q } \alpha _ { j } ( \mathbf { x } ) \mathbf { f } _ { j } ( \mathbf { x } ) = \mathbf { c } ( \mathbf { x } )\tag{2}
$$

for all $\textbf { x } \in { \mathcal { X } } .$ , where $\alpha _ { 1 } ( \cdot ) , \ldots , \alpha _ { Q } ( \cdot )$ are known, non-zero input-dependent coeficients and $\textbf { c } : \boldsymbol { \mathcal { X } }  \mathbb { R } ^ { S }$ is a known constraint function. For example, for two fields constrained to sum to 1 on each discrete coordinate, we have $\alpha _ { 1 } ( \mathbf { x } ) = \alpha _ { 2 } ( \mathbf { x } ) = 1$ and ${ \bf c } = 1 _ { \mathbb { R } ^ { S } }$ for all $\mathbf { x } \in { \mathcal { X } } .$ . We focus first on the case where the constraint coeficients $\alpha _ { j }$ are independent of the input � and the second member is zero $( \mathbf { c } ( \mathbf { x } ) = \mathbf { 0 } )$ . Under these assumptions, the constraint reads

$$
\sum _ { j = 1 } ^ { Q } \alpha _ { j } \mathbf { f } _ { j } ( \mathbf { x } ) = \mathbf { 0 } _ { S } , \quad \forall \mathbf { x } \in \mathcal { X } .\tag{3}
$$

The generalizations to input-dependent coeficients $\alpha _ { j } ( \mathbf { x } )$ and non-zero second member $\mathbf { c } ( \mathbf { x } ) \neq \mathbf { 0 }$ are discussed in Section 4.6.2, while the extension to noisy observations is discussed in Section 4.6.1.

## 2.2. Gaussian process regression

We adopt Gaussian process regression as our modeling tool for functions $\mathbf { f } _ { 1 } , \ldots , \mathbf { f } _ { Q }$ , given its well-established efectiveness in small-sample regimes and its principled uncertainty quantification (Rasmussen and Williams, 2005). We briefly recall the single-output and multi-output formulations, which serve both as building blocks for our framework and as components of the competing methods in our benchmark.

## 2.2.1. Single-output Gaussian process

Given a training set $( \mathbf { x } ^ { ( i ) } , z ^ { ( i ) } ) _ { 1 \leq i \leq N } \subset \mathbb { R } ^ { D } \times \mathbb { R }$ with $z ^ { ( i ) } = f ( \mathbf { x } ^ { ( i ) } )$ , a GP prior $f ( \mathbf { x } ) \sim \mathcal { G P } ( \mu ( \mathbf { x } ) , k ( \mathbf { x } , \mathbf { x } ^ { \prime } ) )$ is placed on $f .$ Conditioning on the observations, the posterior distribution at a test point $\mathbf { x } _ { * }$ is Gaussian, $f _ { \ast } \mid \mathbf { X } , \mathbf { z } , \mathbf { x } _ { \ast } \sim$ $\mathcal { N } \big ( \bar { f } _ { * } , \mathrm { V a r } ( f _ { * } ) \big )$ , with

$$
\bar { f } _ { * } = \mu ( \mathbf { x } _ { * } ) + \mathbf { k } _ { * } ^ { \top } \mathbf { K } ^ { - 1 } ( \mathbf { z } - \mu ) , \qquad \mathrm { V a r } ( f _ { * } ) = k ( \mathbf { x } _ { * } , \mathbf { x } _ { * } ) - \mathbf { k } _ { * } ^ { \top } \mathbf { K } ^ { - 1 } \mathbf { k } _ { * } ,\tag{4}
$$

where $\mathbf { z } = [ z _ { 1 } , \dots , z _ { N } ] ^ { \top } , \boldsymbol { \mu } = [ \mu ( \mathbf { x } _ { 1 } ) , \dots , \mu ( \mathbf { x } _ { N } ) ] ^ { \top } , [ \mathbf { K } ] _ { i j } = k ( \mathbf { x } ^ { ( i ) } , \mathbf { x } ^ { j } )$ , and $\mathbf { k } _ { * } = [ k ( \mathbf { x } ^ { 1 } , \mathbf { x } _ { * } ) , \ldots , k ( \mathbf { x } ^ { N } , \mathbf { x } _ { * } ) ] ^ { \top }$ . The kernel hyperparameters are estimated by maximizing the marginal log-likelihood (Rasmussen and Williams, 2005). In practice, a small nugget $\boldsymbol { \sigma } ^ { 2 } \mathbf { I } _ { N }$ is added to � for numerical stability.

## 2.2.2. Multi-output Gaussian process

Now consider the multi-output regression task of learning a vector-valued function $\textbf { f } : \mathcal { X } \subset \mathbb { R } ^ { D } \to \mathbb { R } ^ { Q }$ from a training set $( \mathbf { x } ^ { ( i ) } , \mathbf { z } ^ { ( i ) } = [ z _ { 1 } ^ { ( i ) } , \dots , z _ { Q } ^ { ( i ) } ] ^ { \top } ) _ { 1 \leq i \leq N }$ with $z _ { q } ^ { ( i ) } = f _ { q } ( \mathbf { x } ^ { ( i ) } )$ . A multi-output GP prior is placed on �:

$$
\mathbf { f } ( \mathbf { x } ) \sim \mathcal { G P } \big ( \mu ( \mathbf { x } ) , \mathbf { k } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \big ) ,
$$

where $\mu : \mathcal { X }  \mathbb { R } ^ { Q }$ is a vector-valued mean function and � $: \mathcal { X } \times \mathcal { X } \to \mathbb { R } ^ { Q \times Q }$ is a matrix-valued covariance kernel with entries $[ \mathbf { k } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) ] _ { q q ^ { \prime } } = \mathbf { C o v } [ f _ { q } ( \mathbf { x } ) , f _ { q ^ { \prime } } ( \mathbf { x } ^ { \prime } ) ]$ . Denoting by $\mathbf { Z } = [ \mathbf { z } _ { 1 } ^ { \top } , \ldots , \mathbf { z } _ { N } ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { N Q }$ the concatenated observations, the posterior at a test point $\mathbf { x } _ { * }$ is Gaussian with mean and covariance

$$
\begin{array} { r } { \bar { \mathbf { f } } _ { * } = \mathbf { \mu } ( \mathbf { x } _ { * } ) + \mathbf { k } ( \mathbf { X } , \mathbf { x } _ { * } ) ^ { \top } \mathbf { k } ( \mathbf { X } , \mathbf { X } ) ^ { - 1 } ( \mathbf { Z } - \mathbf { \mu } \mathbf { \mu } ( \mathbf { X } ) ) , } \\ { \mathbf { C } \mathrm { o v } ( \mathbf { f } _ { * } ) = \mathbf { k } ( \mathbf { x } _ { * } , \mathbf { x } _ { * } ) - \mathbf { k } ( \mathbf { x } _ { * } , \mathbf { X } ) \mathbf { k } ( \mathbf { X } , \mathbf { X } ) ^ { - 1 } \mathbf { k } ( \mathbf { X } , \mathbf { x } _ { * } ) , } \end{array}\tag{5}
$$

where $\mathbf { k } ( \mathbf { X } , \mathbf { X } ) \in \mathbb { R } ^ { N Q \times N Q }$ is the block covariance matrix. A popular class of matrix-valued kernels is the linear model of coregionalization (LCM) (Goulard and Voltz, 1992; Rougier, 2008), which expresses � as a sum of separable terms:

$$
\mathbf { k } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \sum _ { r = 1 } ^ { R } k _ { r } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \mathbf { B } _ { r } ,\tag{6}
$$

where each $k _ { r } ( \cdot , \cdot )$ is a scalar kernel and each $\mathbf { B } _ { r } \in \mathbb { R } ^ { Q \times Q }$ is a positive semi-definite coregionalization matrix capturing inter-output dependencies at the �-th scale. The special case � = 1 reduces to the intrinsic coregionalization model (ICM): ${ \bf k } ( { \bf x } , { \bf x } ^ { \prime } ) = k ( { \bf x } , { \bf x } ^ { \prime } )$ �. Hyperparameters, including the kernel parameters and the entries of $\mathbf { B } _ { r }$ , are estimated by maximizing the marginal log-likelihood

$$
\begin{array} { r } { \log p ( \mathbf { Z } \mid \mathbf { X } , \theta ) = - \frac { 1 } { 2 } \mathbf { Z } ^ { \top } \mathbf { k } ( \mathbf { X } , \mathbf { X } ) ^ { - 1 } \mathbf { Z } - \frac { 1 } { 2 } \log \left| \mathbf { k } ( \mathbf { X } , \mathbf { X } ) \right| - \frac { N Q } { 2 } \log 2 \pi , } \end{array}\tag{7}
$$

using gradient-based optimization (Nocedal and Wright, 2006). For more details on the choice of kernel structure and discussion about multi-output GPs, we refer the reader to Alvarez et al. (2012).

## 2.3. Multi-output GP regression under linear equality constraints

We now recall how linear equality constraints can be enforced within the MOGP framework using the parametrization approach proposed by Jidling et al. (2017). Consider a multi-output function $\mathbf { f } : \mathcal { X }  \mathbb { R } ^ { Q }$ subject to the linear constraint

$$
\mathcal { L } [ \mathbf { f } ] ( \mathbf { x } ) = \sum _ { j = 1 } ^ { Q } \alpha _ { j } f _ { j } ( \mathbf { x } ) = 0 , \quad \forall \mathbf { x } \in \mathcal { X } ,\tag{8}
$$

which can be written in vector form as $\alpha ^ { \top } \mathbf { f } ( \mathbf { x } ) = 0$ with $\alpha = [ \alpha _ { 1 } , \ldots , \alpha _ { Q } ] ^ { \top }$ . Considering a MOGP prior on �, the problem amounts to imposing the linear constraint on a vector-valued Gaussian process $\mathbf { f } \sim \mathcal { G P } ( \mu _ { f } ( \mathbf { x } ) , \mathbf { k } _ { f } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) )$ .

The parametrization approach constrains the search space of � to the set of functions satisfying (8) by expressing � as a linear transformation of an unconstrained function. Specifically, one assumes that � is related to another function $\mathbf { g } : \mathcal { X }  \mathbb { R } ^ { \ell }$ through a linear operator $\mathcal { G } \mathrm { : }$

$$
\mathbf { f } ( \mathbf { x } ) = { \mathcal { G } } _ { \mathbf { x } } [ \mathbf { g } ] .\tag{9}
$$

With this formulation, the constraint on � becomes $\mathcal { L } [ \mathcal { G } _ { \mathbf { x } } [ \mathbf { g } ] ] = 0$ , and we require this to hold for any function �, thereby transferring the restriction from � to the operator  itself. Since both  and  are linear, they can be represented by matrices in our setting. Let $\mathbf { C } _ { \mathcal { L } } \in \mathbb { R } ^ { Q \times \ell }$ denote the matrix representation of $\mathcal { G } .$ The relation (9) becomes

$$
\mathbf { f } ( \mathbf { x } ) = \mathbf { C } _ { \mathcal { L } } \mathbf { g } ( \mathbf { x } )\tag{10}
$$

and the constraint (8) holds for any � if and only if

$$
\begin{array} { r } { \mathbf { \alpha } \propto ^ { \top } \mathbf { C } _ { \mathcal { L } } = \mathbf { 0 } _ { \mathcal { C } } ^ { \top } . } \end{array}\tag{11}
$$

Since Gaussian processes are stable under linear transformations, placing a GP prior $\mathbf { g } \ \sim \ G P ( \mu _ { g } , \mathbf { k } _ { g } )$ on the unconstrained function, where $\mathbf { k } _ { g } : \mathcal { X } \times \mathcal { X } \to \mathbb { R } ^ { \ell \times \ell }$ is a matrix-valued kernel, induces a valid GP prior on �, with mean and covariance

$$
\begin{array} { r } { \mathsf { \Pi } \mathsf { \Pi } \mathsf { \Pi } \mathsf { \Pi } \mathsf { \Pi } _ { F } ( \mathbf { x } ) = \mathbf { C } _ { \mathcal { L } } \mathsf { \Pi } \mathsf { \Pi } \mathsf { \Pi } \mathsf { \Pi } _ { g } ( \mathbf { x } ) , \qquad \mathsf { \Pi } \mathsf { \Pi } \mathsf { \Pi } \mathsf { \Pi } \mathsf { \Pi } \mathsf { \Pi } \bigl ( \mathbf { x } , \mathbf { x } ^ { \prime } \bigr ) = \mathbf { C } _ { \mathcal { L } } \mathsf { \Pi } \mathsf { \Pi } \mathsf { \Sigma } \bigl \mathbf { \Sigma } \mathsf { \Sigma } \bigl \mathbf { \Sigma } \bigl \mathbf { \Sigma } \bigl ( \mathbf { x } , \mathbf { x } ^ { \prime } \bigr ) \mathbf { C } _ { \mathcal { L } } ^ { \top } . } \end{array}\tag{12}
$$

By construction, every realization of � lies in the column space of $\mathbf { C } _ { \mathcal { L } }$ , which is contained in the null space of $\alpha ^ { \top }$ and therefore satisfies the constraint exactly. Crucially, this property is inherited by the posterior distribution: both the posterior mean and any sample drawn from the posterior of � satisfy (8) at every point in $\mathcal { X } .$

Remark 1. The parametrization approach is presented herefor the specific case ofa single linear equality constraint with constant coeficients. It applies, however, in a more general framework involving multiple constraints defined by a matrix-valued linear operator . See Lange-Hegermann (2018) for a systematic algorithmic construction of the parametrization using Gröbner bases. In that setting, $\mathbf { C } _ { \mathcal { L } }$ is chosen in the intersection of the null spaces of the rows of .

Solution space and hyperparameter estimation The parametrization matrix $\mathbf { C } _ { \mathcal { L } }$ is not unique: any matrix whose columns lie in ker $( \pmb { \alpha } ^ { \top } )$ is valid. For linear equality constraint (8), ker $( \pmb { \alpha } ^ { \top } )$ has dimension $Q - 1$ . Since we assumed $\alpha _ { j } \neq 0$ , the set of admissible matrices is

$$
S _ { \ell } = \Big \{ \mathbf { C } _ { \mathcal { L } } \in \mathbb { R } ^ { Q \times \ell } \ : \ C _ { \mathcal { L } , 1 , r } = - \frac { 1 } { \alpha _ { 1 } } \sum _ { j = 2 } ^ { Q } \alpha _ { j } C _ { \mathcal { L } , j , r } , r = 1 , \ldots , \ell \Big \} .\tag{13}
$$

Each column of $\mathbf { C } _ { \mathcal { L } }$ has $Q - 1$ free entries (the first one being determined by the constraint). Crucially, unlike deductive approaches, this construction is stable with respect to the choice of the constrained entry. Consequently, the non-uniqueness of $\mathbf { C } _ { \mathcal { L } }$ introduces free parameters that are estimated jointly with the kernel hyperparameters of � by maximizing the marginal log-likelihood (7). This lets the optimization select the parametrization that best fits the data while guaranteeing constraint satisfaction by construction.

Choice ofkernelfor � A natural choice is to model the $\ell$ components of � as independent scalar GPs, each with its own kernel $k _ { r } ( \mathbf { x } , \mathbf { x } ^ { \prime } )$ , so that $\mathbf { k } _ { g } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \mathrm { d i a g } \big ( k _ { 1 } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) , \dots , k _ { \ell } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \big )$ . Substituting into (12), the induced kernel on � reads

$$
\mathbf { k } _ { f } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \mathbf { C } _ { \mathcal { L } } \operatorname { d i a g } \left( k _ { 1 } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) , \dots , k _ { \ell } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \right) \mathbf { C } _ { \mathcal { L } } ^ { \top } ,\tag{14}
$$

where the constraint matrix $\mathbf { C } _ { \mathcal { L } } \in \mathbb { R } ^ { Q \times \ell }$ satisfies $\mathbf { C } _ { \mathcal { L } } \in S _ { \ell } $ , as defined in (13). Instead of placing independent priors on the components of ${ \bf g } ,$ here we consider a linear model of coregionalization, to allow a richer prior structure on �. Following (6), we set

$$
{ \bf k } _ { g } ( { \bf x } , { \bf x } ^ { \prime } ) = \sum _ { r = 1 } ^ { R } k _ { r } ( { \bf x } , { \bf x } ^ { \prime } ) { \bf B } _ { r } ,\tag{15}
$$

where each $k _ { r }$ is a scalar kernel and each $\mathbf { B } _ { r } \in \mathbb { R } ^ { \ell \times \ell }$ is a positive semi-definite coregionalization matrix. Substituting this LCM prior into (12) yields the constrained kernel on �

$$
\begin{array} { r } { \mathbf { k } _ { f } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \mathbf { C } _ { \mathcal { L } } \Big ( \sum _ { r = 1 } ^ { R } k _ { r } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \mathbf { B } _ { r } \Big ) \mathbf { C } _ { \mathcal { L } } ^ { \top } = \sum _ { r = 1 } ^ { R } k _ { r } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \mathbf { C } _ { \mathcal { L } } \mathbf { B } _ { r } \mathbf { C } _ { \mathcal { L } } ^ { \top } . } \end{array}\tag{16}
$$

The parametrization thus acts solely on the coregionalization matrices of the LCM, replacing each $\mathbf { B } _ { r } \in \mathbb { R } ^ { \ell \times \ell }$ by $\mathbf { C } _ { \mathcal { L } } \mathbf { B } _ { r } \mathbf { C } _ { \mathcal { r } } ^ { \top } \in \mathbb { R } ^ { Q \times Q }$ , while the scalar kernels $k _ { r }$ remain free. Each $\mathbf { C } _ { \mathcal { L } } \mathbf { B } _ { r } \mathbf { C } _ { \mathcal { L } } ^ { \top }$ is positive semi-definite and its columns lie in $\ker ( \pmb { \alpha } ^ { \top } )$ , so the constraint is encoded entirely through the coregionalization structure. For estimation by maximum likelihood (7), in practice we parametrize each $\mathbf { C } _ { \mathcal { L } } \mathbf { B } _ { r } \mathbf { C } _ { \mathcal { L } } ^ { \top } = \widetilde { \mathbf { W } } _ { r } \widetilde { \mathbf { W } } _ { r } ^ { \top } \succeq 0$ with a single matrix $\widetilde { \mathbf { W } } _ { r } \in S _ { \ell }$ , where each column of $\widetilde { \mathbf { W } } _ { r }$ is constrained to be in ker $( \pmb { \alpha } ^ { \top } )$ .

Illustrative example To illustrate the construction, consider the case $Q = 2$ with the constraint $\alpha _ { 1 } f _ { 1 } ( \mathbf { x } ) \mathbf { + } \alpha _ { 2 } f _ { 2 } ( \mathbf { x } ) = 0$ and $\ell = 1$ . So $\mathbf { C } _ { \mathcal { L } } \in \mathbb { R } ^ { 2 \times 1 }$ and takes the form $\mathbf { C } _ { \mathcal { L } } = [ - \alpha _ { 2 } / \alpha _ { 1 } , \ 1 ] ^ { \top }$ . Placing a scalar GP prior $g \sim \mathcal { G P } ( \mu _ { g } , k _ { g } )$ , the induced kernel on � is

$$
\mathbf { k } _ { f } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \mathbf { C } _ { \mathcal { L } } \mathbf { C } _ { \mathcal { L } } ^ { \top } k _ { g } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \left[ \begin{array} { c c } { \alpha _ { 2 } ^ { 2 } / \alpha _ { 1 } ^ { 2 } } & { - \alpha _ { 2 } / \alpha _ { 1 } } \\ { - \alpha _ { 2 } / \alpha _ { 1 } } & { 1 } \end{array} \right] k _ { g } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) .\tag{17}
$$

For instance, with $\alpha _ { 1 } = 1 , \alpha _ { 2 } = 1 \mathrm { ( i . e . , } f _ { 1 } + f _ { 2 } = 0 \mathrm { ) }$ , one recovers $\mathbf { k } _ { f } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \left[ \mathbf { \varphi } _ { - 1 } ^ { 1 } \mathbf { \varphi } _ { 1 } ^ { - 1 } \right] k _ { g } ( \mathbf { x } , \mathbf { x } ^ { \prime } )$ . Any prior or posterior sample from this model satisfies $f _ { 1 } ( { \bf x } ) + f _ { 2 } ( { \bf x } ) = 0$ everywhere. In comparison, an unconstrained ICM model where $\mathbf { k } _ { f } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \mathbf { B } k ( \mathbf { x } , \mathbf { x } ^ { \prime } )$ would estimate the coregionalization matrix � from data. While the estimated matrix may converge to the same structure after optimization, it can never guarantee exact constraint satisfaction, even when the training data are noise-free.

## 3. Linear equality constraints and deductive approaches

As discussed in the introduction, a natural strategy for enforcing a linear equality constraint $\begin{array} { r } { \sum _ { j = 1 } ^ { Q } \alpha _ { j } f _ { j } ( { \bf x } ) = 0 } \end{array}$ consists in selecting one output index $l \in \{ 1 , \ldots , Q \}$ and deducing its prediction from the others via the constraint

relation:

$$
\hat { f } _ { l } ( \mathbf { x } ) = - \frac { 1 } { \alpha _ { l } } \sum _ { j \neq l } \alpha _ { j } \hat { f } _ { j } ( \mathbf { x } ) ,\tag{18}
$$

where ${ \hat { f } } _ { j } ( \mathbf { x } )$ denotes the prediction of the �-th output by an unconstrained surrogate model trained on the $Q - 1$ remaining outputs. This deductive approach can be combined with any standard multi-output regression method, including independent GPs or LCM-based MOGPs, without any modification to their training procedure. Constraint satisfaction on the predictive mean is guaranteed by construction through (18). Moreover, posterior samples can also be made constraint-compliant: ones sample jointly from the posterior of the $Q - 1$ modeled outputs and recovers the deduced output via (18) applied to each sample. However, this strategy introduces an arbitrary choice across outputs: the deduced output � is not modeled from data but recovered algebraically, which means its prediction error accumulates the individual errors of all modeled outputs. The choice of � is a priori arbitrary, and diferent choices may lead to substantially diferent predictive performances. In this section, we investigate these efects empirically through a benchmark that compares the deductive approach, under all possible deduction scenarios, against the constrained MOGP (CMOGP) presented in Section 2.3, which models all outputsjointly and encodes the constraint in its covariance structure.

## 3.1. Benchmark setup

We consider a regression problem with $Q = 3$ scalar outputs subject to the constraint $f _ { 1 } ( \mathbf { x } ) + f _ { 2 } ( \mathbf { x } ) + f _ { 3 } ( \mathbf { x } ) = 0$ . The three underlying functions are defined on $\mathcal { X } = [ 0 , 1 ] ^ { 3 }$ as follows: $f _ { 1 }$ is the Ishigami function (Ishigami and Homma, 1990), $f _ { 2 }$ is the Branin function (Jamil and Yang, 2013) (rescaled to $[ 0 , 1 ] ^ { 3 } )$ , and $f _ { 3 }$ is determined by the constraint $f _ { 3 } = - ( f _ { 1 } + f _ { 2 } )$ . This choice is deliberate: $f _ { 1 }$ and $f _ { 2 }$ have markedly diferent complexity and smoothness properties, so that the dificulty of the regression task is heterogeneous across outputs. The constraint output $f _ { 3 }$ inherits the combined complexity of both. Three methods are compared:

• CMOGP: the constrained multi-output GP of Section 2.3, which models all three outputs jointly using a constrained kernel (16) with g modeled by an LCM $( R = 3 , \ell = 1 )$ and anisotropic Matérn $5 / 2$ covariance functions used for each scalar kernel $k _ { r } \left( r \in { 1 , 2 , 3 } \right)$ .

• Indep. GP: independent single-output GPs, one per modeled output, each with an anisotropic Matérn $5 / 2$ kernel, combined with deduction (18).

• LCM: a multi-output GP with an LCM kernel $( R = 2$ , rank-1 coregionalization matrices, anisotropic Matérn $5 / 2$ scalar kernels), trained on the $Q - 1 = 2$ modeled outputs, combined with deduction (18).

We consider three deduction scenarios $l \in \{ 1 , 2 , 3 \}$ , for every deductive baseline. A key practical distinction is that CMOGP requires only a single training phase to predict all outputs jointly (each deductive configuration mandates a separately trained model). Our evaluation relies on 200 independent replications. Within a single replication, we sample a fixed test set of 200 points alongside varying training sets of size $N \in \{ 2 0 , 5 0 , 1 0 0 \}$ using Latin Hypercube Sampling (LHS). Model hyperparameters are optimized by maximizing the marginal log-likelihood (via L-BFGS-B), using a multi-start strategy with 50 random initializations for multi-output formulations (CMOGP and LCM) and 30 for standard single-output GPs. Predictive performance is evaluated by the Root Mean Square Error (RMSE) on each output, defined for $N _ { t e s t }$ test samples as:

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { N _ { t e s t } } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } .\tag{19}
$$

Furthermore, the quality of the uncertainty estimates is evaluated through the average length of the 90% predictive intervals.

## 3.2. Results

We first present the RMSE results on the output $f _ { 2 }$ (Branin) to analyze the impact of the arbitrary choice of deduced output, and then turn to a win rate analysis covering all three outputs

![](images/ba16f54ac05fe8549a928fb58fa974d3e9664fec2c94e5a8c09289ba3ae54e32.jpg)  
Figure 1: Distribution of the RMSE on $f _ { 2 }$ (Branin) across 200 replications, for training set sizes $N \in \{ 2 0 , 5 0 , 1 0 0 \}$ and the three deduction scenarios. Each panel corresponds to a diferent choice of deduced output �. The CMOGP (blue) is identical across panels. The deductive methods exhibit sensitivity to the choice of �, with the independent GP showing systematic degradation when $f _ { 2 }$ is deduced (middle panel).

RMSE across deduction scenarios Figure 1 presents the empirical distributions of the RMSE associated with the prediction of $f _ { 2 } ,$ for training set sizes $N \in 2 0 , 5 0 , 1 0 0$ and across the three deduction scenarios. Each panel corresponds to a specific choice of deduced output �, while the CMOGP model yields identical results across panels, as it is independent of this choice. A first insight is obtained by contrasting the three panels. When $f _ { 2 }$ is explicitly modeled (i.e., $l \in \{ 1 , 3 \} ,$ , the three approaches exhibit comparable performance, as reflected by largely overlapping RMSE distributions. In contrast, a markedly diferent behavior emerges in the intermediate panel (� = 2), where $f _ { 2 }$ is reconstructed. In this setting, the independent GP displays a noticeable degradation in performance, characterized by both an upward shift in RMSE values. This discrepancy highlights a pronounced sensitivity of the independent GP to the choice of deduced output. By comparison, the LCM model appears more stable, with RMSE distributions that remain at a similar level regardless of whether $f _ { 2 }$ is directly modeled or inferred. While these distributions suggest comparable behavior between the LCM and the CMOGP, such a visual assessment remains limited, as it does not capture how often one method efectively outperforms the others. To address this point, we now turn to a complementary win rate analysis.

Win rate analysis Figure 2 reports the win rate of each method for $\begin{array} { r } { { \cal N } } { { \bf \Pi } = { \bf \nabla } 2 0 , } \end{array}$ , defined as the proportion of replications in which a given method achieves the lowest RMSE for a given output. The three panels correspond to performance evaluated on $f _ { 1 } , f _ { 2 } ,$ and $f _ { 3 } ,$ respectively, while the horizontal axis indicates the deduced output �. This view complement Figure 1 in two ways. First, for each evaluated output, the win rates results change across deduction scenarios, with a noticeable shifts when the evaluated quantity is deduced. Since the CMOGP approach yields identica RMSE distributions regardless of the deduction choice, the observed variations in win rates necessarily reflect changes in the relative performance of the two deductive methods (independent GP and LCM). Figure 2 does not allow one to attribute this variability to one deductive method alone. However, the result is consistent with the heterogeneity already observed on the independent GP in Figure 1. Second, Figure 2 separates the LCM from the CMOGP in a way that the violin plot did not allow. Across outputs and deduction scenarios, the CMOGP wins more often than the LCM in most configurations, even when their RMSE distributions looked close in Figure 1. The gap in favor of the CMOGP grows with �. The corresponding plots for � = 50 and � = 100 are reported in C.

## 3.3. Discussion

Two main observations emerge from these results. First, the deductive approach can be sensitive to the choice of which output is deduced. This sensitivity is directly visible on the RMSE of the independent GP, which shifts towards higher values when the evaluated output is the deduced one (Figure 1, middle panel), and this gap does not vanish with the training set size. The win rate analysis provides an additional piece of evidence: since the CMOGP behaves identically across scenarios, the variability of the win rate ranking between scenarios necessarily comes from the deductive methods. In practice, this means that with a deductive strategy, the choice of the deduced output cannot be made without trying all � configurations, which multiplies the cost by � and weakens the scalability of the approach.

![](images/a2cccc7d4a73b2f4193d565e086b8ec99d51e1e6cd433eec77abe98ec712eccf.jpg)  
Figure 2: Win rate (%) of each method at � = 20 on the RMSE of each output $\left( f _ { 1 } , f _ { 2 } , f _ { 3 } \right)$ , across the three deduction scenarios $\left( l \in \{ 1 , 2 , 3 \} \right)$ . The CMOGP (blue) is the same model in all scenarios. Aggregated over 200 replications.

Second, between the two deductive methods, the LCM looks more stable than the independent GP on the RMSE distributions of Figure 1: modeling the � − 1 retained outputs jointly seems to partly absorb the efect of the deductive choice. Among the joint approaches in the benchmark, the win rate analysis shows that the CMOGP is the best choice. The LCM approach still treats one output as a quantity computed from the others, so the constraint information is only used at the post-processing step. The CMOGP, on the other hand, models all � outputs together and encodes the constraint analytically in its kernel. This combination of joint modeling of all � outputs and analytical use of the constraint translates into the win rate plots, where the CMOGP wins more often than the LCM in most configurations, with a gap that grows with �. We also examined the lengths of the predictive intervals, after verifying that all three methods achieve comparable empirical coverage rates close to the nominal level. The patterns observed on the interval length distributions are similar to those of the RMSE in Figure 1 and do not bring additional information beyond what is already discussed above, see Figure 12 in C.1.

These findings clearly demonstrate the inherent instability of deductive approaches, which illustrates the need to handle equality constraints through robust frameworks such as the CMOGP. While this motivates to adopt the CMOGP framework for our current problem, a major challenge arises regarding the dimensionality of the outputs. Since constrained MOGP regression cannot be applied directly when the output dimension $P = Q \times S$ is large, a dimensionality reduction step that preserves the constraint structure in the latent space is needed. This is the subject of the next section.

## 4. Principal component analysis for multi-field data

In this section, we address the problem of reducing the dimension of the output fields while preserving the linear constraint structure, so that the constrained MOGP of Section 2.3 can be applied in the resulting latent space. We present three PCA strategies for multi-field data, analyze their relationship with the constraint, and establish theoretical bounds comparing their reconstruction quality.

## 4.1. PCA strategies for multi-field data

For each field $k = 1 , \dots , Q$ , we collect the training observations into the data matrix $\mathbf { Y } _ { k } \in \mathbb { R } ^ { N \times S }$ whose �-th row is $\mathbf { y } _ { k } ^ { \left( i \right) ^ { \top } }$ , where $\mathbf { y } _ { k } ^ { ( i ) }$ is the �-th component of the simulator output introduced in Section 2.1. We assume that the sample mean has been subtracted from each row of these data matrices. The non-centered case will be dealt with in Section 4.6.1. The singular value decomposition (SVD) of each field is $\mathbf { Y } _ { k } = \mathbf { U } _ { k } \mathbf { D } _ { k } \mathbf { V } _ { k } ^ { \top }$ , where $\mathbf { V } _ { k } \in \mathbb { R } ^ { S \times S }$ contains the right singular vectors (spatial basis), ${ \bf U } _ { k } \in \mathbb { R } ^ { N \times N }$ the left singular vectors (observation basis), and $\mathbf { D } _ { k }$ the singular values in decreasing order. We define the spatial covariance matrix $\begin{array} { r } { \pmb { \Sigma } _ { k } = \frac { 1 } { N } \pmb { \mathrm { Y } } _ { k } ^ { \top } \pmb { \mathrm { Y } } _ { k } \in \mathbb { R } ^ { S \times S } } \end{array}$ and the Gram matrix $\begin{array} { r } { \mathbf { K } _ { k } = \frac { 1 } { S } \mathbf { Y } _ { k } \mathbf { Y } _ { k } ^ { \top } \in \mathbb { R } ^ { N \times N } } \end{array}$ . For a truncation rank �, we denote by $\mathbf { V } _ { k , m }$ and $\mathbf { U } _ { k , m }$ the matrices of the first � right and left singular vectors, and by $\mathbf { P } _ { k , m } = \mathbf { V } _ { k , m } \mathbf { V } _ { k , m } ^ { \top }$ and $\boldsymbol { \Pi } _ { k , m } = \mathbf { U } _ { k , m } \mathbf { U } _ { k , m } ^ { \top }$ the corresponding spatial and observation projectors. Three strategies can be considered for simultaneously reducing the dimension of all � fields

Field-wise PCA Each field $\mathbf { Y } _ { k }$ is reduced independently onto its own optimal rank-� spatial basis $\mathbf { V } _ { k , m }$ . The reconstruction error for field � is

$$
E _ { k , m } ^ { ( \mathrm { f i e l d } ) } = \| \mathbf { Y } _ { k } ( \mathbf { I } _ { S } - \mathbf { P } _ { k , m } ) \| _ { F } ^ { 2 } .\tag{20}
$$

This approach minimizes the reconstruction error for each field individually, but uses a diferent basis per field, which prevents any coupling between the latent representations.

Column-wise PCA The fields are concatenated column-wise as $\mathbf { Y } _ { \mathrm { c o l } } = [ \mathbf { Y } _ { 1 } , \dots , \mathbf { Y } _ { O } ] \in \mathbb { R } ^ { N \times Q S }$ , and a single PCA is performed on $\mathbf { Y } _ { \mathrm { c o l } }$ . The projector operates in the observation space: $\Pi _ { \mathrm { c o l } , m } = \mathbf { U } _ { \mathrm { c o l } , m } \mathbf { U } _ { \mathrm { c o l } , m } ^ { \top }$ , where $\mathbf { U } _ { \mathrm { c o l } , m }$ contains the left singular vectors associated with the first � eigenvalues of the SVD decomposition of $\mathbf { Y } _ { \mathrm { c o l } }$ . The global and per-field reconstruction errors are

$$
E _ { m } ^ { ( \mathrm { c o l } ) } = \sum _ { k = 1 } ^ { Q } E _ { k , m } ^ { ( \mathrm { c o l } ) } , \qquad E _ { k , m } ^ { ( \mathrm { c o l } ) } = \| ( \mathbf { I } _ { N } - \mathbf { \Pi } _ { \mathrm { c o l } , m } ) \mathbf { Y } _ { k } \| _ { F } ^ { 2 } .\tag{21}
$$

This approach captures inter-field correlations at the dimensionality reduction stage and produces scalar latent variables that can later be modeled by independent GPs.

Row-wise PCA The fields are stacked row-wise as $\mathbf { Y } _ { \mathrm { r o w } } = [ \mathbf { Y } _ { 1 } ^ { \top } , \ldots , \mathbf { Y } _ { O } ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { N Q \times S }$ . Through the SVD of $\mathbf { Y } _ { r o w } ,$ we obtain a shared spatial basis $\mathbf { V } _ { \mathrm { r o w } , m }$ formed by its first � leading right singular vectors, along with the associated projector $\mathbf { P } _ { \mathrm { r o w } , m } = \mathbf { V } _ { \mathrm { r o w } , m } \mathbf { V } _ { \mathrm { r o w } , m } ^ { \top }$ . The errors are

$$
E _ { m } ^ { ( \mathrm { r o w } ) } = \sum _ { k = 1 } ^ { Q } E _ { k , m } ^ { ( \mathrm { r o w } ) } , \qquad E _ { k , m } ^ { ( \mathrm { r o w } ) } = \| \mathbf { Y } _ { k } ( \mathbf { I } _ { S } - \mathbf { P } _ { \mathrm { r o w } , m } ) \| _ { F } ^ { 2 } .\tag{22}
$$

Because all fields share the same spatial basis, the projection of each sample $\mathbf { y } ^ { ( i ) } = [ \mathbf { y } _ { 1 } ^ { ( i ) \top } , \ldots , \mathbf { y } _ { Q } ^ { ( i ) \top } ] ^ { \top }$ onto $\mathbf { V } _ { \mathrm { r o w } , m }$ produces a vector-valued weight $\mathbf { w } _ { i } \in \mathbb { R } ^ { Q \times m }$ , whose components of each column can be modeled jointly by a MOGP. A key structural property distinguishes the row-wise approach. The row-wise covariance matrix satisfies $\pmb { \Sigma } _ { \mathrm { r o w } } =$ $\begin{array} { r } { \frac { 1 } { Q } \sum _ { k = 1 } ^ { Q } \pmb { \Sigma } _ { k } } \end{array}$ , and similarly $\begin{array} { r } { \mathbf { K } _ { \mathrm { c o l } } = \frac { 1 } { Q } \sum _ { k = 1 } ^ { Q } \mathbf { K } _ { k } } \end{array}$ for the column-wise Gram matrix.

## 4.2. Constraint preservation in the latent space

Consider the constraint $\begin{array} { r } { \sum _ { i = 1 } ^ { Q } \alpha _ { j } \mathbf { Y } _ { j } = \mathbf { 0 } _ { N \times S } } \end{array}$ satisfied by the training data. Forming the row-wise concatenation and denoting by $\mathbf { U } = [ \alpha _ { 1 } \mathbf { I } _ { N } , \hdots , \alpha _ { Q } \mathbf { I } _ { N } ] \in \mathbb { R } ^ { N \times N Q }$ the constraint matrix, the constraint reads $\mathbf { U Y } _ { \mathrm { r o w } } = \mathbf { 0 }$ . Since the row-wise projector $\mathbf { P } _ { \mathrm { r o w } , m }$ acts on the right (spatial dimension), we have:

$$
\mathbf { U } \mathbf { Y } _ { \mathrm { r o w } } = \mathbf { 0 } \implies \mathbf { U } \mathbf { W } = \mathbf { U } \mathbf { Y } _ { \mathrm { r o w } } \mathbf { V } _ { \mathrm { r o w } , m } = \mathbf { 0 } ,\tag{23}
$$

where $\mathbf { W } \in \mathbb { R } ^ { N Q \times m }$ denotes the latent weight matrix. The constraint is thus exactly transferred to the latent space, regardless of the truncation rank �. Furthermore, the reconstruction $\tilde { \mathbf { Y } } _ { \mathrm { r o w } } = \mathbf { W } \mathbf { V } _ { \mathrm { r o w } , m } ^ { \top }$ also satisfies the constraint:

$$
\mathbf { U } \tilde { \mathbf { Y } } _ { \mathrm { r o w } } = \mathbf { U } \mathbf { W } \mathbf { V } _ { \mathrm { r o w } , m } ^ { \top } = \mathbf { 0 } .\tag{24}
$$

This property is specific to the row-wise approach. For the column-wise strategy, the projector operates on the observation dimension, and no analogous preservation holds in general. For the field-wise approach, the use of diferent spatial bases per field destroys any linear relation between the latent representations. Consequently, the row-wise PCA is the only strategy among the three that can be coupled with the constrained MOGP of Section 2.3 to guarantee constraint satisfaction throughout the entire algorithm.

## 4.3. Theoretical analysis of the reconstruction error of multi-field approaches

Since the row-wise approach is the only one that preserves the constraint, while the column-wise and field-wise approaches can only be used within a deductive framework, a natural question is whether they can also be compared in terms of reconstruction error ? This allows us to assess whether adopting the row-wise approach requires a significant trade-of in accuracy. To evaluate this, we compare the row-wise and col-wise approaches (referred to as multi-field approaches) with the field-wise optimum, which by construction minimizes the reconstruction error for each field individually. The proofs of all results in this section are given in A.

Theorem 1 (Row-wise excess error bound). Let $\delta _ { k } = \lambda _ { m } ^ { ( k ) } - \lambda _ { m + 1 } ^ { ( k ) }$ denote the spectral gap of $\scriptstyle \mathbf { \dot { \Sigma } } \mathbf { \Sigma } _ { k }$ at rank �. $I f \delta _ { k } > 0 ,$ then:

$$
\Delta E _ { k , m } ^ { ( r o w ) } : = E _ { k , m } ^ { ( r o w ) } - E _ { k , m } ^ { ( f i e l d ) } \ \leq \ \frac { 2 N \sqrt { 2 } \left\| \pmb { \Sigma } _ { k } \right\| _ { F } } { \delta _ { k } } \cdot \operatorname* { m i n } \{ \sqrt { m } \left\| \Delta \pmb { \Sigma } _ { k } \right\| _ { o p } , \ \left\| \Delta \pmb { \Sigma } _ { k } \right\| _ { F } \} ,\tag{25}
$$

where $\begin{array} { r } { \Delta \Sigma _ { k } = \Sigma _ { r o w } - \Sigma _ { k } = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } ( \Sigma _ { j } - \Sigma _ { k } ) a n d \parallel \cdot \parallel _ { o p } } \end{array}$ denotes the spectral norm.

Theorem 2 (Column-wise excess error bound). Let $\delta _ { k } ^ { ( c o l ) } = \lambda _ { m } ^ { ( k ) } - \lambda _ { m + 1 } ^ { ( k ) }$ denote the spectral gap of $\mathbf { K } _ { k }$ at rank �. If $\delta _ { k } ^ { ( c o l ) } > 0 ,$ , then:

$$
\Delta E _ { k , m } ^ { ( c o l ) } : = E _ { k , m } ^ { ( c o l ) } - E _ { k , m } ^ { ( f i e l d ) } \ \leq \ \frac { 2 S \sqrt { 2 } \| \mathbf { K } _ { k } \| _ { F } } { \delta _ { k } ^ { ( c o l ) } } \cdot \operatorname* { m i n } \big \{ \sqrt { m } \| \Delta K _ { k } \| _ { o p } , \ \| \Delta K _ { k } \| _ { F } \big \} ,\tag{26}
$$

where $\begin{array} { r } { \Delta \boldsymbol { K } _ { k } = \mathbf { K } _ { c o l } - \mathbf { K } _ { k } = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } ( \mathbf { K } _ { j } - \mathbf { K } _ { k } ) . } \end{array}$

The two bounds share a common structure: the excess error for field � is controlled by the inter-field spectral heterogeneity, the distance between the covariance (or Gram) matrix of field � and the average over all fields. When all fields share similar structures, both bounds tend to zero and the multi-field approaches incur essentially no penalty relative to the field-wise optimum. The two bounds involve diferent matrices operating in diferent spaces $( \mathbb { R } ^ { S \times S }$ vs. $\mathbb { R } ^ { N \times N } )$ ), with diferent spectral gaps, so their relative ordering is problem-dependent.

Corrolary 1 (Vanishing excess error under shared basis). If all fields share the same spatial basis $( \mathbf { V } _ { 1 } = \cdots = \mathbf { V } _ { Q } ) ,$ then $\Delta E _ { k , m } ^ { ( r o w ) } = 0$ for all �. Similarly, ifallfields share the same observation basis $( \mathbf { U } _ { 1 } = \cdots = \mathbf { U } _ { Q } )$ , then $\Delta E _ { k , m } ^ { ( c o l ) } = 0$ for all �.

## 4.4. Direct comparison of row-wise and column-wise

The previous bounds compare each multi-field approach to a separate reference (the field-wise optimum). Theorem 3 below provides an exact criterion to compare row-wise and column-wise PCA against each other, when both are applied to the full set of � fields.

Theorem 3 (Row-wise vs. column-wise global error). The diference between the global reconstruction errors of the column-wise and row-wise approaches at rank � is exactly:

$$
E _ { m } ^ { ( c o l ) } - E _ { m } ^ { ( r o w ) } = T r ( \mathbf { D } _ { r o w , m } ^ { 2 } ) - T r ( \mathbf { D } _ { c o l , m } ^ { 2 } ) ,\tag{27}
$$

where $\mathbf { D } _ { r o w , m } ^ { 2 }$ and $\mathbf { D } _ { c o l , m } ^ { 2 }$ are the diagonal matrices of the � largest squared singular values of $\mathbf { Y } _ { r o w }$ and $\mathbf { Y } _ { c o l } ,$ respectively. Equivalently, denoting by $\lambda _ { 1 } ^ { ( r o w ) } \ge \cdots \ge \lambda _ { m } ^ { ( r o w ) }$ and $\lambda _ { 1 } ^ { ( c o l ) } \ge \cdots \ge \lambda _ { m } ^ { ( c o l ) }$ the � largest eigenvalues of $\begin{array} { r } { \Sigma _ { r o w } = \frac { 1 } { Q } \sum _ { k } \Sigma _ { k } } \end{array}$ and $\begin{array} { r } { { \bf K } _ { c o l } = \frac { 1 } { Q } \sum _ { k } { \bf K } _ { k } . } \end{array}$

$$
E _ { m } ^ { ( c o l ) } - E _ { m } ^ { ( r o w ) } = N Q \sum _ { p = 1 } ^ { m } \lambda _ { p } ^ { ( r o w ) } - Q S \sum _ { p = 1 } ^ { m } \lambda _ { p } ^ { ( c o l ) } .\tag{28}
$$

In particular, neither approach dominates the other in general.

Algorithm 1: Row-CMO   
Input: Training data $\left( ( \mathbf { x } ^ { ( i ) } ) _ { i = 1 } ^ { N } , \mathbf { Y } _ { 1 } , \ldots , \mathbf { Y } _ { Q } \right)$ , coeficients $\alpha = ( \alpha _ { 1 } , \ldots , \alpha _ { Q } ) .$ , and rank $m .$   
Output: Constrained predictive model <sup>̂</sup>� satisfying $\begin{array} { r } { \sum _ { j = 1 } ^ { Q } \alpha _ { j } \hat { \mathbf { f } } _ { j } = \mathbf { 0 } _ { S } . } \end{array}$   
Step 1: Data Pre-processing (Training)   
1: Center each field: $\mathbf Y _ { j }  \mathbf { \bar { Y } } _ { j } - \mathbf 1 _ { N } \bar { \mathbf y } _ { j } ^ { \top }$ , where $\bar { \mathbf { y } } _ { j }$ is the empirical mean of field �.   
2 Stack the centered fields row-wise into $\mathbf { Y } _ { \mathrm { r o w } } \in \mathbb { R } ^ { N Q \times S }$ and extract the �-rank spatial basis $\mathbf { V } _ { \mathrm { r o w } , m } \in \mathbb { R } ^ { S \times m }$ via   
SVD.   
3: Project data into the latent space: $\mathbf { w } _ { j } ^ { ( i ) } = \mathbf { y } _ { j } ^ { ( i ) \top } \mathbf { V } _ { \mathrm { r o w } , m } \in \mathbb { R } ^ { m }$   
Step 2: Model Training   
4: for each latent dimension $s = 1 , \ldots , m$ do   
5: Fit a constrained MOGP on $\big ( ( \mathbf { x } ^ { ( i ) } ) _ { i = 1 } ^ { N } , ( w _ { 1 , s } ^ { ( i ) } , \ldots , w _ { Q , s } ^ { ( i ) } ) _ { i = 1 } ^ { N } \big )$ using the kernel of Eq. (12).   
6: end for   
Step 3: Prediction at a new input $\mathbf { x } _ { \ast }$   
7: for each latent dimension $s = 1 , \ldots , m$ do   
8: Compute the latent prediction $\hat { \mathbf { w } } _ { * , s } = [ \hat { w } _ { * , 1 , s } , \ldots , \hat { w } _ { * , Q , s } ] ^ { \intercal }$ from the �-th CMOGP.   
9: end for   
10: Assemble per-field latent predictions: $\hat { \mathbf { w } } _ { * , j } = [ \hat { w } _ { * , j , 1 } , \ldots , \hat { w } _ { * , j , m } ] ^ { \top }$ for $j = 1 , \dots , Q .$   
11: Reconstruct each physical field $\hat { \mathbf { y } } _ { * , j } = \bar { \mathbf { y } } _ { j } + \mathbf { V } _ { \mathrm { r o w } , m } \hat { \mathbf { w } } _ { * , j } \mathrm { ~ f o r ~ } j = 1 , \ldots , Q .$

## 4.5. Discussion

The theoretical results of this section provide guarantees for the row-wise PCA strategy that underlies our framework. Theorems 1 and 2 identify inter-field spectral heterogeneity as the sole quantity governing the excess reconstruction error of each multi-field approach relative to the field-wise optimum. For problems where the component fields share similar spatial covariance structures, which is common in physical applications where the fields arise from the same PDE system and are defined on the same mesh, the row-wise approach incurs a negligible penalty while ofering the crucial advantage of preserving the linear equality constraint. Theorem 3 gives an exact criterion to compare row-wise and column-wise PCA on the full set of � fields. We note that, in the constrained setting, columnwise PCA is in fact applied on � − 1 fields rather than �, since one field is removed and recovered by deduction. By optimality of PCA on the larger set, this can only increase the reconstruction error compared to the column-wise strategy on � fields. As a consequence, whenever the criterion of Theorem 3 indicates equivalence or row-wise dominance on the � fields, the row-wise approach also dominates the deductive column-wise variant.

From a practical standpoint, the bounds of Theorems 1 and 2 and the criterion of Theorem 3 can be evaluated directly from the data. They give a way of assessing which regime one is in: similar fields, heterogeneous fields, row wise or column-wise dominance. This is useful in practice when no prior information is available on the choice between the strategies, and even more so when one suspects the fields to be correlated.

## 4.6. The Row-CMO framework

We now combine the row-wise PCA with the constrained MOGP of Section 2.3 to form the Row-CMO framework. Recall that the constraint under consideration is $\begin{array} { r } { \sum _ { j = 1 } ^ { Q } \alpha _ { j } \mathbf { f } _ { j } ( \mathbf { x } ) = \mathbf { 0 } _ { S } } \end{array}$ , where the coeficients $\alpha _ { j }$ are non-zero scalars independent of �. The complete procedure is summarized in Algorithm 1.

By construction, the constraint is satisfied at every stage of Algorithm 1: in the latent weights (line 5), in the MOGP predictions and posterior samples (line 8), and in the reconstructed fields (line 11). The resulting predictive mean and any posterior sample thus satisfy $\begin{array} { r } { \sum _ { j } \alpha _ { j } \hat { \bf f } _ { j } ( { \bf x } ) = { \bf 0 } _ { S } } \end{array}$ to machine precision, for all $\mathbf { x } \in \mathcal { X }$

## 4.6.1. Extension to noisy data

The framework presented above assumes noise-free data, which is the typical situation for outputs of deterministic numerical simulators. In applications where the outputs fields are corrupted by observation noise, two steps of the procedure require an adjustment: the centering step in Algorithm 1, and the GP regression in the latent space.

Constraint-consistent centering In the noise-free case, the field means $\bar { \mathbf { y } } _ { k }$ naturally satisfy $\begin{array} { r } { \sum _ { j } \alpha _ { j } \bar { \bf y } ^ { j } = { \bf \nabla } { \bf 0 } _ { S } , } \end{array}$ , so standard centering preserves the constraint on the centered data. When the outputs data are corrupted by noise, however,

the empirical means may not satisfy the constraint exactly: $\textstyle \sum _ { i } \alpha _ { j } { \bar { \mathbf { y } } } ^ { j } = \mathbf { r } \neq \mathbf { 0 } _ { S }$ , where $\mathbf { r } \in \mathbb { R } ^ { S }$ is a residual vector. If left uncorrected, this residual propagates into the centered data and breaks the constraint structure that is essential for the row-wise PCA to transfer the constraint to the latent space.

To address this, we use a constraint-consistent centering that distributes the residual proportionally across the field means. Specifically, we define adjusted means as

$$
\tilde { \bar { \bf y } } _ { k } = \bar { \bf y } _ { k } - \frac { \alpha _ { k } } { \sum _ { j = 1 } ^ { Q } ( \alpha _ { j } ) ^ { 2 } } { \bf r } , \qquad k = 1 , \ldots , Q ,\tag{29}
$$

which satisfy $\begin{array} { r } { \sum _ { j } \alpha _ { j } \tilde { \bar { \bf y } } ^ { j } = { \bf 0 } _ { S } } \end{array}$ by construction. This adjustment is the minimum-norm correction (in the Euclidean sense on the vector $[ \bar { \mathbf { y } } ^ { 1 } , \dots , \bar { \mathbf { y } } ^ { Q } ] )$ that restores the constraint on the means. The centering of Algorithm 1 (line 1) is then performed using $\tilde { \bar { \bf y } } _ { k }$ in place of $\bar { \mathbf { y } } _ { k }$

Constrained MOGP regression with noise The constrained MOGP fitted on the latent components in Step 2 of Algorithm 1 can be extended to handle noisy observations through the standard MOGP likelihood. Assuming an observation model $\mathbf { w } _ { i } ^ { s } = \mathbf { h } _ { s } ( \mathbf { x } ^ { ( i ) } ) + \boldsymbol { \eta } _ { i } ^ { s }$ for each latent dimension �, with independent Gaussian noise $\boldsymbol \eta _ { i } ^ { s } \sim \mathcal { N } ( \mathbf 0 , \pmb \Sigma ^ { s } )$ and $\Sigma ^ { s } = \mathrm { d i a g } ( \sigma _ { s , 1 } ^ { 2 } , \dots , \sigma _ { s , Q } ^ { 2 } )$ collecting the per-output noise variances, the covariance matrix to invert in the posterior expressions becomes $\mathbf { k } ( \mathbf { X } , \mathbf { \tilde { X } } ) + \mathbf { \Sigma } ^ { s } \otimes \mathbf { I } _ { N }$ . The noise variances are estimated jointly with the kernel hyperparameters by maximizing the marginal log-likelihood. The rest of the procedure (Steps 1–3 of Algorithm 1) is unchanged.

## 4.6.2. Handling more general constraints

Algorithm 1 is formulated for the constraint $\begin{array} { r } { \sum _ { j } \alpha _ { j } \mathbf { f } _ { j } ( \mathbf { x } ) = \mathbf { 0 } _ { S } } \end{array}$ with scalar coeficients $\alpha _ { j }$ independent of �. We now discuss two generalizations of practical interest.

Input-dependent coeficients When the constraint coeficients depend on the input, $\begin{array} { r } { \sum _ { i } \alpha _ { j } ( \mathbf { x } ) f _ { j } ( \mathbf { x } ) = 0 } \end{array}$ , with each $\alpha _ { j } ( \cdot )$ known and non-vanishing on $x ,$ , one can introduce the transformed variables $z _ { i } ^ { j } = { \alpha _ { j } } ( { \bf x } ^ { ( i ) } ) y _ { i } ^ { j }$ for each observation � and field �. These satisfy $\begin{array} { r } { \sum _ { j } z _ { i } ^ { j } = 0 } \end{array}$ , which is a constraint with unit scalar coeficients. Algorithm 1 is then applied to the matrices $\mathbf { Z } _ { k }$ with $\alpha _ { j } = 1$ for all �. The inverse transformation $\hat { y } _ { \ast } ^ { j } = \hat { z } _ { \ast } ^ { j } / \alpha _ { j } ( \mathbf { x } _ { \ast } )$ is well defined since the coeficients are non-zero by assumption.

Non-zero second member When the constraint takes the form $\begin{array} { r } { \sum _ { j } \alpha _ { j } \mathbf { f } _ { j } ( \mathbf { x } ) = \mathbf { c } ( \mathbf { x } ) } \end{array}$ with a known, non-zero function $\mathbf { c } : \boldsymbol { \mathcal { X } } \to \mathbb { R } ^ { S }$ , the goal is to reduce the problem to a homogeneous constraint by defining transformed variables that sum to zero. This reduction requires subtracting an appropriate share of �(�) from each output field. A natural approach is to define, for each field �:

$$
\tilde { \mathbf { y } } _ { k } ^ { ( i ) } = \mathbf { y } _ { k } ^ { ( i ) } - \frac { \boldsymbol { \beta } _ { k } } { \sum _ { j } \alpha _ { j } \beta _ { j } } \mathbf { c } ( \mathbf { x } ^ { ( i ) } ) , \quad k = 1 , \ldots , Q ,\tag{30}
$$

where $\beta _ { k } > 0$ are weights satisfying $\begin{array} { r } { \sum _ { j } \alpha _ { j } \beta _ { j } \ne 0 } \end{array}$ . The transformed variables satisfy $\begin{array} { r } { \sum _ { j } \alpha _ { j } \tilde { \mathbf { y } } _ { j } ^ { ( i ) } = 0 } \end{array}$ for any choice of $\beta ,$ , and Algorithm 1 can be applied to $\{ \tilde { \mathbf { Y } } _ { k } \}$

The choice of β, however, is not neutral: subtracting a spatially complex function �(�) from one or several fields modifies their variance structure, which in turn afects the row-wise PCA reconstruction quality. In particular, assigning the entire correction to a single output (e.g., $\beta _ { l } = 1 / \alpha _ { l }$ and ${ \beta } _ { k } = 0$ for $k \neq l )$ would reintroduce an arbitrary analogous to the arbitrary output selection in the deductive approach. Uniform weights $( \beta _ { k } = \alpha _ { k }$ , recovering (29)) avoid this issue but may not be optimal when the fields have heterogeneous variance levels. In practice, the weights can be guided by domain knowledge or by a variance-based criterion: a field � for which the transformation $\mathbf { y } _ { k } ^ { ( i ) } - \beta _ { k } \mathbf { c } ( \mathbf { x } ^ { ( i ) } )$ does not significantly increase the empirical variance is a natural candidate for absorbing a larger share of �(�). A systematic study of optimal redistribution strategies in this setting is an interesting direction for future work.

## 5. Numerical experiments

We now validate the Row-CMO framework on two case studies of increasing complexity. The first is a population dynamics problem based on the Lotka–Volterra system, which serves as a controlled test bed for evaluating the theoretical results of Section 4 and confirming the sensitivity of deductive approaches observed in Section 3. The second is a Computational Fluid Dynamics (CFD) application involving the prediction of Reynolds stress tensor components under an incompressibility constraint, which demonstrates the applicability of Row-CMO to a realistic high-dimensional industrial problem.

In both case studies, we compare Row-CMO against deductive alternatives that combine diferent PCA strategies with unconstrained GP regression. The competing methods are:

• Row-CMO (our model): Row-wise PCA + Constrained MOGP;

• Col-Indep(�): Column-wise PCA + Independent GPs with deduced output $l ;$

• Fw-Indep(�): Field-wise PCA + Independent GPs with deduced output $l ;$

• Fw-LCM(�): Field-wise PCA + LCM with deduced output �.

For each deductive method, � ranges over all � possible deduction scenarios. The comparison focuses on predictive accuracy as a function of the latent dimension � and the training set size �. The Relative Root Mean Square Error (RRMSE) provides a global measure of accuracy for each predicted field $\hat { \mathbf { y } } ^ { ( i ) }$ relative to the simulated reference $\mathbf { y } ^ { ( i ) }$ :

$$
\mathrm { R R M S E } ^ { 2 } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \frac { \| \hat { \mathbf { y } } ^ { ( i ) } - \mathbf { y } ^ { ( i ) } \| _ { 2 } ^ { 2 } } { S \| \mathbf { y } ^ { ( i ) } \| _ { \infty } ^ { 2 } }\tag{31}
$$

on a test set of size $N _ { t e s t }$ . Hyperparameters are optimized by maximum likelihood via L-BFGS-B with 50 multi-start initializations for multi-output models and 30 for single-output GPs.

## 5.1. Population dynamics: Lotka–Volterra system

## 5.1.1. Problem setup

We consider the Lotka–Volterra prey-predator system (Wangersky, 1978):

$$
\left\{ \begin{array} { l } { { p ^ { \prime } ( t ) = a p ( t ) - b p ( t ) q ( t ) } } \\ { { q ^ { \prime } ( t ) = - c q ( t ) + d p ( t ) q ( t ) } } \end{array} \right. ~ ,\tag{32}
$$

where $p ( t )$ and �(�) denote the prey and predator population densities. The positive parameters � and � are the prey growth rate and the predator mortality rate, while � and � control the prey–predator interaction. This system admits a Hamiltonian structure (Nutku, 1990), which we linearize to obtain the constraint:

$$
\begin{array} { r } { d p ( t ) - c s ( t ) + b q ( t ) - a r ( t ) = C _ { 0 } , } \end{array}\tag{33}
$$

where $s = \log ( p ) , r = \log ( q )$ , and $C _ { 0 }$ is a constant determined by the initial conditions. The metamodeling task consists in simultaneously predicting � = 4 output fields $( p , q , r , s )$ , each discretized on a uniform temporal grid of $S = 2 0 0 0 0$ points in [0, 20], as a function of the input parameters $\mathbf { x } = ( b , d )$ . The inputs � and � are assumed to be independent and uniformly distributed over [0.37, 0.40] and [0.0, 0.06], respectively, with fixed parameters $a = 1 . 1 , c = 0 . 4$ and initial conditions $( p ( 0 ) , q ( 0 ) ) = ( 1 . 9 , 0 . 3 )$

Training sets of size $N ~ \in ~ \{ 1 0 , 1 5 , 3 0 \}$ with corresponding test sets of size $N _ { { \mathrm { t e s t } } } ~ \in ~ \{ 9 0 , 8 5 , 7 0 \}$ such that $N + N _ { \mathrm { t e s t } } = 1 0 0$ , are independently sampled from this joint uniform distribution. The constrained kernel in Row-CMO uses an LCM kernel $( R = 2 , l = 2$ , anisotropic Matérn $5 / 2$ kernels) on the latent function � (cf. Section 2.3), while the LCM in Fw-LCM uses � = 2 scalar kernels (coregionalization matrices of rank at most 2, anisotropic Matérn $5 / 2$ kernels). All results are averaged over 10 independent replications with randomized train and test sets, and optimizer initializations. The latent dimension � varies from 1 to 10. All methods enforce the constraint to machine precision on the predictive mean: Row-CMO does so by construction, while the deductive methods achieve it through (18). We therefore focus the comparison on predictive accuracy (RRMSE) as a function of � and �. We illustrate the results on the output � (prey density), and analogous patterns are observed on the other outputs and reported in D.

![](images/1186e7f1fcba74ad665d10c46da8b86cd3bd14af84252b4dad4abf0be7369b0b.jpg)  
Figure 3: Left: normalized dominance criterion of Theorem 3 as a function of �, for $N \in \{ 3 0 , 1 5 , 1 0 \}$ . Right: RRMSE on output � for all methods and the three training set sizes $( m = 1 , \ldots , 1 0 )$ . Error bars represent ±1 standard deviation over 10 seeds. The dominance criterion converges to zero beyond $m \approx 4 ,$ and Row-CMO surpasses all alternatives at higher latent dimensions.

## 5.1.2. Results

Alignment ofTheorem 3 with empirical RRMSE Figure 3 (left panel) displays the normalized dominance criterion of Theorem 3 as a function of � for $N \in \{ 3 0 , 1 5 , 1 0 \}$ . The criterion is positive for the first few modes indicating that column-wise on � fields captures slightly more variance and converges rapidly toward zero beyond $m \approx 4 ,$ , after which the two strategies explain essentially the same amount of variance. Notably, the convergence is faster for smaller �: with fewer training samples, the estimated covariance matrices become more similar, reducing the gap between the two PCA strategies. The RRMSE panels (right) confirm this alignment: at small �, Row-CMO starts with a higher RRMSE than Col-Indep and Fw-Indep variants, but the gap narrows as � increases and vanishes around $m = 5  – 6$ , precisely where the criterion indicates near-equivalence.

Predictive advantage of the row-wise PCA-Constrained MOGP combination Although the dominance criterion indicates near-equivalence of the PCA strategies on the training data for $m \geq 5 ,$ , the RRMSE which measures predictive performance on unseen test data points to a clear advantage for our approach. Figure 4 provides a zoomed view of the RRMSE for $m \in \{ 5 , \ldots , 1 0 \}$ , where Row-CMO consistently achieves the lowest RRMSE across all three training set sizes. The gap between Row-CMO and the alternatives widens as � decreases: for $N \ = \ 1 0 .$ Row-CMO clearly separates from all deductive methods, whereas for � = 30 the diferences are smaller but Row-CMO remains the best-performing method.

This discrepancy between the near-equivalence observed on training data (via the criterion) and the clear superiority observed on test data (via the RRMSE) points to a generalization advantage of the combination of row-wise PCA with constrained MOGP. Two mechanisms contribute to this efect. First, the row-wise PCA benefits from an informationpooling efect when data are scarce: the shared spatial basis is estimated from �� stacked observations, providing a more stable estimate than the � separate bases of the field-wise approach (each estimated from � observations) or the single column-wise basis (estimated from � observations in a ��-dimensional space). This improved stability translates into better out-of-sample reconstruction. Second, the constrained MOGP encodes the constraint analytically in the kernel rather than estimating the coregionalization structure from data, which reduces the efective number of hyperparameters and avoids the sensitivity to the deduction choice that penalizes the alternatives. The fact that this advantage is amplified for smaller � supports the interpretation in terms of generalization capacity.

We also note that Row-CMO outperforms not only Col-Indep but also Fw-Indep and Fw-LCM , despite the fact that field-wise PCA is optimal for each individual field on the training data (Theorem 1).

Heterogeneity on deductive approaches results The spread of the deductive method curves across deduction scenarios is clearly visible in both Figures 3 and 4. This variability is present across all values of � and all latent dimensions, confirming that it is an inherent feature of the deductive strategy.

In summary, the Lotka–Volterra experiments confirm the three key messages of this paper: (i) the dominance criterion of Theorem 3 reliably predicts the relative PCA performance from the training data, (ii) the combination of row-wise PCA with constrained MOGP exhibits a generalization advantage driven by the information-pooling efect of the shared basis and the analytical constraint encoding of the CMOGP that becomes most pronounced in data-scarce regimes, and (iii) the deductive approaches sufer from a sensitivity to the output selection that persists across all configurations. These findings motivate the application of Row-CMO to the industrial-scale CFD problem that follows.

![](images/db81b03db8c1cf6890436ac10a6f056849c2554ea1e68d6b992c536f9e678167.jpg)  
Figure 4: Zoomed view of the RRMSE on output $p$ for $m \in \{ 5 , \ldots , 1 0 \}$ and training set sizes $N \in \{ 3 0 , 1 5 , 1 0 \}$ . Row-CMO (red, solid) consistently achieves the lowest RRMSE. The spread of Col-Indep, Fw-Indep, and Fw-LCM curves across deduction scenarios $\left( l \in \{ 0 , 1 , 2 , 3 \} \right)$ ) illustrates the sensitivity of the deductive approaches to the arbitrary choice of deduced output.

## 5.2. Application to a CFD test case: uncertainty propagation of parameters of a turbulence model on a two-dimensional difuser

In this section, we present an application that was the primary motivation for the methods described above, which concerns the construction of metamodels for CFD fields using Gaussian processes. We propose addressing an uncertainty propagation problem within the context of the turbulent fluid flow in a difuser, represented by a channel with a progressive widening. Its geometry is represented in Figure 5. Experimental measurements on this difuser has been performed in Buice and Eaton (2000b) with a Reynolds number of 18,000. Experimentally, a fluid recirculation is observed between points $R _ { 1 }$ and $R _ { 2 }$ in Figure 5. This recirculation is due to the appearance of an opposing pressure gradient (opposite to the main flow direction) caused by the widening of the cross-section, which in turn causes the boundary layer to detach. The main challenge of this application for RANS models lies in accurately predicting both the recirculation amplitude and limits, along with the velocity field throughout the entire domain.

Numerical simulations of the Buice-Eaton 2D difuser were carried out using the solver TrioCFD (Puscas, Angeli, Nouaime and Jamelot, 2025), exploring diferent sets of input parameters. Each simulation employed a mesh of 200, 000 triangular elements which correspond to $S ~ = ~ 1 4 1 , 0 3 9$ mesh coordinates. To enhance computational eficiency, a domain decomposition strategy combined with a message-passing interface (MPI) was implemented for parallel computations on 11 CPU cores. The standard � − ε turbulence model (Launder and Spalding, 1974) was used, with inlet boundary conditions for velocity, turbulent kinetic energy (�), and dissipation rate (ε) obtained from a preliminary periodic simulation. A zero-pressure boundary condition was applied at the channel outlet. For the uncertainty propagation study, the input parameters correspond to the closure coeficients of the RANS standard $k - \varepsilon$ turbulence model:

$$
{ \bf x } = ( C _ { \parallel } , \sigma _ { k } , \sigma _ { \varepsilon } , C _ { \varepsilon _ { 1 } } , C _ { \varepsilon _ { 2 } } ) .\tag{34}
$$

These five parameters govern the transport equations for � and ε: $C _ { \mu }$ is the proportionality constant between the eddy viscosity and $k ^ { 2 } / \varepsilon , \sigma _ { k }$ and $\sigma _ { \varepsilon }$ are the turbulent Prandtl numbers for � and ε respectively, and $C _ { \varepsilon _ { 1 } }$ and $C _ { \varepsilon _ { \gamma } }$ are the source and sink term coeficients in the equation for ε respectively. Their uncertainties are represented by probability distributions chosen based on physical considerations. Further details on the RANS modeling, the derivation of these distributions and the sampling procedure are provided in B.

![](images/f427381847abf363389fa3919e82995fe7889209e5d73867e2b466435df21a50.jpg)  
Figure 5: Scheme of the experiment carried out in (Buice and Eaton, 2000b). The wall of the channel are defined by straight lines and circular arcs of centers $O _ { 1 }$ and $O _ { 2 }$

The quantity of interest considered as outputs in this study are the diagonal terms of the Reynolds stress tensor (RST) $\tau _ { i j } = u _ { i } ^ { \prime } u _ { j } ^ { \prime }$ , and the turbulent kinetic energy. This selection is motivated by the goal of constructing a physicsbased surrogate model for the Reynolds stress tensor, as represented in linear eddy-viscosity turbulence models such as the standard �-ε model:

$$
\tau _ { i j } = - \nu _ { t } \left( \frac { \partial \bar { u } _ { i } } { \partial x _ { j } } + \frac { \partial \bar { u } _ { j } } { \partial x _ { i } } \right) + \frac { 2 } { 3 } k \delta _ { i j }\tag{35}
$$

where $\boldsymbol { v } _ { t }$ is the eddy viscosity. This expression is known as the Boussinesq assumption (Boussinesq, 1897), written here for incompressible flow. Combined with the periodicity in the spanwise direction $\begin{array} { r } { ( \mathrm { i . e } \ \frac { \partial \bar { u } _ { 3 } } { \partial x _ { 3 } } = 0 ) } \end{array}$ , it implies the following linear equality constraint between the diagonal components of the Reynolds stress tensor and the turbulent kinetic energy:

$$
\tau _ { 1 1 } + \tau _ { 2 2 } - \frac { 4 } { 3 } k = 0 .\tag{36}
$$

Training and testing datasets generation A candidate set $( \mathbf { x } ^ { ( i ) } ) _ { 1 \leq i \leq N _ { c } }$ of size $N _ { c } = 1 { , } 0 0 0$ is generated following the procedure described in B. From this candidate set, a sub-sample of size $\dot { n } = 5 0$ is selected using a space-filling greedy algorithm that maximizes the energy distance (Mak et al., 2018; Teymur, Gorham, Riabiz and Oates, 2021), thereby defining the training dataset. The testing dataset is then constructed using a Monte Carlo sample of size $N = 1 0 0$ of the input parameters, generated according to the same procedure. Results are averaged over 10 training runs with diferent optimizer initialization seeds. For the kernel configuration, the constrained kernel in Row-CMO uses an LCM $( \ R = 2 , \ell = 2$ , anisotropic Matérn $5 / 2 )$ while the LCM in Fw-LCM uses $R = 2$ scalar kernels with the corresponding coregionalization matrices $\mathbf { B } _ { 1 }$ and ${ \bf B } _ { 2 }$ having ranks at most 1 and 2, respectively. To assess the spatial distribution of prediction errors, we use the Spatial Relative Squared Error (SRSE), defined at each mesh location ν for a given test input � as:

$$
\mathrm { S R S E } ( \hat { y } ^ { ( i ) } ( \ v v ) , y ^ { ( i ) } ( \ v v ) ) = \frac { ( \hat { y } ^ { ( i ) } ( \ v v ) - y ^ { ( i ) } ( \ v v ) ) ^ { 2 } } { \lVert \mathbf { y } ^ { ( i ) } \rVert _ { \infty } ^ { 2 } } .\tag{37}
$$

The global RRMSE defined in (31) is recovered as the joint average of the SRSE over the spatial mesh and the test set: $\begin{array} { r } { \mathbf { R R M S E } ^ { 2 } = \frac { 1 } { N _ { \mathrm { t e s t } } S } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \sum _ { j = 1 } ^ { S } { \mathrm { S R S E } \big ( \hat { y } ^ { ( i ) } ( \mathbf { v } _ { j } ) , y ^ { ( i ) } ( \mathbf { v } _ { j } ) \big ) } } \end{array}$

## 5.2.1. Results

A priori diagnostic and predictive accuracy Figure 6 displays the dominance criterion of Theorem 3 (left panel) alongside the RRMSE of all methods on the three output fields as a function of � (right panels). The criterion serves here as a practical diagnostic: it is evaluated directly from the training data, before any GP model is trained, and indicates whether the row-wise PCA strategy is competitive with column-wise for this dataset. The values are positive but small (on the order of $1 0 ^ { - 3 } )$ and monotonically decreasing, confirming that the column-wise PCA on � fields captures only marginally more variance than row-wise, and that the two strategies are essentially equivalent for this problem. This is a direct consequence of a strong spatial correlation among the three fields. Figure 8 provides visual confirmation: the spatial structures of $\tau _ { 1 1 } , \tau _ { 2 2 }$ , and � are clearly similar, with intensity concentrated in the recirculation zone of the difuser. The RRMSE panels confirm this diagnostic. At small �, all methods achieve very similar error levels unlike the Lotka–Volterra case where Row-CMO started above the alternatives. This alignment is expected: when the dominance criterion indicates near-equivalence from the outset, the PCA strategies contribute similarly to the reconstruction quality, regardless of whether row-wise, column-wise, or field-wise is used. As � increases beyond $m \approx 5 ,$ Row-CMO progressively separates from the alternatives and achieves the lowest RRMSE on all three outputs at $m \ : = \ : 8 .$ . This pattern is consistent with the observations on the Lotka–Volterra problem: once the PCA strategies are equivalent, the generalization advantage of Row-CMO, the information-pooling efect of the shared basis and the analytical constraint encoding of the CMOGP drives the performance gap, especially in this regime where only � = 50 training points are available. Figure 7 provides a detailed view at � = 8: Row-CMO achieves the lowest mean RRMSE on each output with a narrower variability across seeds. The deductive methods exhibit a tighter grouping of their variants than in the Lotka–Volterra case, which is expected given the strong inter-field correlation. Nevertheless, a residual sensitivity to the deduction scenario persists: the spread across variants within each deductive family (Col-Indep, Fw-Indep, Fw-LCM ) remains visible. This confirms on a realistic industrial problem, that the sensitivity of the deductive approach to the arbitrary output selection does not vanish even when the fields are strongly correlated.

![](images/798947efbb33ff0485864fae73bbc2bcb5768b0a8c6e72effb1e5ef3d262667c.jpg)  
Figure 6: Left: dominance criterion of Theorem 3 for the CFD test case, showing the normalized diference $\ T r ( { \bf D } _ { \mathsf { c o l } , m } ^ { 2 } ) -$ ${ \mathsf { T r } } ( \mathbf { D } _ { \mathsf { r o w } , m } ^ { 2 } )$ as a function of �. The small and decreasing values confirm near-equivalence of the two PCA strategies. Right: RRMSE on each output field $\left( \tau _ { 1 1 } , \tau _ { 2 2 } , k \right)$ for all methods $( m = 1 , \ldots , 1 0 )$ . achieves the lowest RRMSE at $m \geq 6 .$ Error bars represent ±1 standard deviation over 10 seeds

![](images/ad3da79ddfb68f761e867fb0addc6c0de0df6437ddbe223bd24e911d31837820.jpg)  
Figure 7: RRMSE at $m = 8$ for each output field $\left( \tau _ { 1 1 } , ~ \tau _ { 2 2 } , ~ k \right)$ across all methods and deduction scenarios. Row-CMO achieves the lowest mean RRMSE with the smallest variability. The spread across deduction scenarios within each family (Col-Indep, Fw-Indep, Fw-LCM ) and the larger error bars of the deductive methods illustrate the residual sensitivity to the output selection, even in a regime of strong inter-field correlation.

![](images/ecd6ed4f0fa7e784086637910b163951bae792c983d633dfef82fa7fc9150cc1.jpg)  
Figure 8: Row-CMO predictions on the CFD test case for a representative test input (� = 8). From top to bottom: $\tau _ { 1 1 } , \tau _ { 2 2 } ,$ �. Left to right: simulated field, predicted field, predictive standard deviation, and SRSE. The model accurately captures the spatial structure. Errors and uncertainty are concentrated in the recirculation zone of the difuser.

Spatial distribution of predictions and errors Figure 8 illustrates the predictive quality of Row-CMO for a representative test input, using � = 8 latent dimensions. The left two columns compare the simulated and predicted fields for $\tau _ { 1 1 } , \tau _ { 2 2 }$ , and �: the model accurately reproduces the spatial structure of all three fields, including the intensity and extent of the recirculation zone. The third column displays the predictive standard deviation, which is highest in the recirculation region where the flow dynamics are most sensitive to the input parameters and lowest in the upstream channel, reflecting the model’s ability to identify regions of high and low predictive confidence. The rightmost column shows the SRSE: the prediction errors are localized in the same recirculation region, with magnitudes on the order of $1 0 ^ { - 4 } ~ \mathrm { t o } ~ 1 0 ^ { - 3 }$ , confirming the overall accuracy of the model.

These results confirm that the proposed approach is well suited to this class of problems where the output fields share strong spatial correlations. In this regime, the row-wise PCA incurs a negligible reconstruction cost, and the generalization advantage of Row-CMO enables it to outperform the deductive alternatives at moderate-to-high latent dimensions, while guaranteeing strict constraint satisfaction.

## 6. Conclusion

We propose a framework for the simultaneous prediction of multiple high-dimensional physical fields governed by linear equality constraints. The approach combines a row-wise PCA strategy, which uniquely preserves the constraint structure in the latent space with constrained multi-output GP regression based on a kernel parametrization that analytically encodes the constraint. The motivation for joint constrained modeling, as opposed to a trivial deductive approach, was established through a benchmark on scalar-output functions. This benchmark demonstrated that the deductive strategy introduces a sensitivity to the arbitrary choice of which output to deduce, afecting both predictive accuracy and uncertainty quantification, and that this sensitivity persists regardless of the regression method used for the remaining outputs. The constrained MOGP avoids these issues by treating all outputs symmetrically and encoding the constraint directly in the covariance structure. To assess the reconstruction cost of the constraint-preserving rowwise PCA relative to the more common column-wise and field-wise strategies, we provided a theoretical analysis based on perturbation bounds and an exact dominance criterion. The framework was validated on a population dynamics problem based on the Lotka–Volterra system and on an industrial CFD application involving the prediction of Reynolds stress tensor components under the incompressibility constraint. In both cases, our method achieved competitive or superior predictive performance compared to deductive alternatives combining column-wise, field-wise, or LCMbased regression with output deduction. The generalization advantage of the proposed method was most pronounced in data-scarce regimes, where the information-pooling efect of the shared spatial basis and the analytical constraint encoding jointly contributed to improved out-of-sample accuracy.

Several directions for future work can be identified. First, the parametrization approach in latent space should make it possible to handle settings where the diferent outputs fields are not observed at the same input locations, by exploiting the constraint structure to couple the fields. Second, extending the methodology to spatio-temporal data from physical simulations, following the approach of Mak et al. (2018) is a natural perspective. Third, while the present framework handles linear equality constraints, many physical systems involve also inequality constraints (e.g, positivity of energy or concentration fields). Extending the parametrization approach to such settings would broaden the applicability of the method. Finally, for problems exhibiting strong nonlinearities in the output fields, the linear PCA step may become a bottleneck, and coupling constrained GP regression with nonlinear dimensionality reduction techniques such as autoencoders or kernel PCA, represents a promising but challenging research direction.

## 7. Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used Claude in order to improve software development for the numerical experiments. After using Claude, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## 8. Acknowledgments

This work has received funding from the France 2030 NumPEx Exa-MA (ANR-22-EXNU-0002) project managed by the French National Research Agency (ANR)

## A. Proofs of the theoretical results

We recall the Davis–Kahan sin Θ theorem (Davis and Kahan, 1970; Stewart and Sun, 1990) in the form used throughout our proofs.

Theorem 4 (Davis–Kahan). Let $\pmb { \Sigma } _ { A } , \pmb { \Sigma } _ { B } \in \mathbb { R } ^ { S \times S }$ be symmetric matrices with eigenvalues $\lambda _ { 1 } ^ { A } \ \ge \ \cdots \ \ge \ \lambda _ { S } ^ { A }$ and $\lambda _ { 1 } ^ { B } \geq \cdots \geq \lambda _ { S } ^ { B } .$ . Let $\mathbf { A } = ( \mathbf { a } _ { 1 } , \ldots , \mathbf { a } _ { m } )$ and $\mathbf { B } = (  { \mathbf { b } } _ { 1 } , \dots ,  { \mathbf { b } } _ { m } )$ be the matrices of the first � eigenvectors $o f \Sigma _ { A }$ and $\Sigma _ { B }$ respectively, and let $\delta _ { A } = \lambda _ { m } ^ { A } - \lambda _ { m + 1 } ^ { A } > 0$ . Then:

$$
\| \sin \Theta ( \mathbf { A } , \mathbf { B } ) \| _ { F } \leq \frac { \operatorname* { m i n } \big \{ \sqrt { m } \| \Sigma _ { B } - \Sigma _ { A } \| _ { o p } , \| \Sigma _ { B } - \Sigma _ { A } \| _ { F } \big \} } { \delta _ { A } } ,\tag{38}
$$

where $\Theta ( \mathbf { A } , \mathbf { B } )$ denotes the diagonal matrix of principal angles between the column spaces of � and �.

We also use the relation between the projector distance and the principal angles: for projectors $\mathbf { P } _ { A } = \mathbf { A } \mathbf { A } ^ { \top }$ and $\mathbf { P } _ { B } = \mathbf { B } \mathbf { B } ^ { \top }$ of rank $m ,$

$$
\lVert \mathbf { P } _ { A } - \mathbf { P } _ { B } \rVert _ { F } = \sqrt { 2 } \lVert \sin \Theta ( \mathbf { A } , \mathbf { B } ) \rVert _ { F } .\tag{39}
$$

## A.1. Proof of Theorem 1 (Row-wise excess error bound)

Proof. Using the trace formulation and the idempotence of projectors:

$$
\begin{array} { r l } & { \Delta E _ { k , m } ^ { ( \mathrm { r o w } ) } = \| \mathbf { Y } _ { k } ( \mathbf { I } _ { S } - \mathbf { P } _ { \mathrm { r o w } , m } ) \| _ { F } ^ { 2 } - \| \mathbf { Y } _ { k } ( \mathbf { I } _ { S } - \mathbf { P } _ { k , m } ) \| _ { F } ^ { 2 } } \\ & { \qquad = \operatorname { T r } \bigl ( N \Sigma _ { k } ( \mathbf { P } _ { k , m } - \mathbf { P } _ { \mathrm { r o w } , m } ) \bigr ) . } \end{array}\tag{40}
$$

By the Cauchy–Schwarz inequality for the trace inner product:

$$
\begin{array} { r } { \Delta E _ { k , m } ^ { \mathrm { ( r o w ) } } \leq N \left. \Sigma _ { k } \right. _ { F } \Vert \mathbf { P } _ { k , m } - \mathbf { P } _ { \mathrm { r o w } , m } \Vert _ { F } . } \end{array}\tag{41}
$$

Now, we apply Theorem 4 with $\Sigma _ { A } = \Sigma _ { k }$ and $\Sigma _ { B } = \Sigma _ { \mathrm { r o w } }$ . The perturbation is:

$$
\Delta \Sigma _ { k } = \Sigma _ { \mathrm { r o w } } - \Sigma _ { k } = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } ( \Sigma _ { j } - \Sigma _ { k } ) .
$$

Using (39), the projector distance satisfies:

$$
\begin{array} { r } { \| \mathbf { P } _ { k , m } - \mathbf { P } _ { \mathrm { r o w } , m } \| _ { F } = \sqrt { 2 } \| \sin \Theta ( \mathbf { V } _ { k , m } , \mathbf { V } _ { \mathrm { r o w } , m } ) \| _ { F } \leq \frac { \sqrt { 2 } \operatorname* { m i n } \left\{ \sqrt { m } \| \Delta \Sigma _ { k } \| _ { \mathrm { o p } } , \| \Delta \Sigma _ { k } \| _ { F } \right\} } { \delta _ { k } } . } \end{array}\tag{42}
$$

Substituting (42) into (41):

$$
\Delta E _ { k , m } ^ { ( \mathrm { r o w } ) } \leq \frac { 2 N \sqrt { 2 } \left. \pmb { \Sigma } _ { k } \right. _ { F } } { \delta _ { k } } \cdot \operatorname* { m i n } \bigr \{ \sqrt { m } \left. \Delta \pmb { \Sigma } _ { k } \right. _ { \mathrm { o p } } , \ \left. \Delta \pmb { \Sigma } _ { k } \right. _ { F } \bigr \} .
$$

## A.2. Proof of Theorem 2 (Column-wise excess error bound)

Proof. The argument is analogous to Theorem 1 but operates in the observation space. Using the equivalence of left and right projections for the SVD $( \mathbf { Y } _ { k } \mathbf { P } _ { k , m } = \mathbf { I I } _ { k , m } \mathbf { Y } _ { k } )$ , the excess error becomes:

$$
\begin{array} { r } { \Delta E _ { k , m } ^ { \mathrm { ( c o l ) } } = \| ( \mathbf { I } _ { N } - \mathbf { I } _ { \mathrm { c o l } , m } ) \mathbf { Y } _ { k } \| _ { F } ^ { 2 } - \| ( \mathbf { I } _ { N } - \mathbf { I } _ { k , m } ) \mathbf { Y } _ { k } \| _ { F } ^ { 2 } = \operatorname { T r } \big ( S \mathbf { K } _ { k } ( \mathbf { I } _ { k , m } - \mathbf { I } _ { \mathrm { c o l } , m } ) \big ) . } \end{array}\tag{43}
$$

By Cauchy–Schwarz:

$$
\Delta E _ { k , m } ^ { ( \mathrm { c o l } ) } \leq S \Vert \mathbf { K } _ { k } \Vert _ { F } \Vert \mathbf { I } _ { k , m } - \mathbf { I } \mathbf { I } _ { \mathrm { c o l } , m } \Vert _ { F }\tag{44}
$$

Then we apply Theorem 4 with $\pmb { \Sigma } _ { A } = \mathbf { K } _ { k }$ and $\Sigma _ { B } = \mathbf { K } _ { \mathrm { c o l } }$ . The perturbation is $\begin{array} { r } { \Delta K _ { k } = \mathbf { K } _ { \mathrm { c o l } } - \mathbf { K } _ { k } = \frac { 1 } { Q } \sum _ { j } ( \mathbf { K } _ { j } - \mathbf { K } _ { k } ) } \end{array}$ and:

$$
\| \mathbf { I } _ { k , m } - \mathbf { I } _ { \mathrm { c o l } , m } \| _ { F } \leq \frac { \sqrt { 2 } \operatorname* { m i n } \bigl \{ \sqrt { m } \| \Delta K _ { k } \| _ { \mathrm { o p } } , \ \| \Delta K _ { k } \| _ { F } \bigr \} } { \delta _ { k } ^ { \mathrm { ( c o l ) } } } .\tag{45}
$$

Substituting (45) into (44) yields:

$$
\Delta E _ { k , m } ^ { ( \mathrm { c o l ) } } \leq \frac { 2 S \sqrt { 2 } \left\| \mathbf { K } _ { k } \right\| _ { F } } { \delta _ { k } ^ { ( \mathrm { c o l ) } } } \cdot \operatorname* { m i n } \bigr \{ \sqrt { m } \left\| \Delta K _ { k } \right\| _ { \mathrm { o p } } , \ \left\| \Delta K _ { k } \right\| _ { F } \bigr \} .
$$

## A.3. Proof of Theorem 3 (Row-wise vs. column-wise)

Proof. Both global errors decompose as total energy minus explained variance. For the row-wise approach:

$$
E _ { m } ^ { ( \mathrm { r o w } ) } = \| \mathbf { Y } _ { \mathrm { r o w } } \| _ { F } ^ { 2 } - \mathrm { T r } ( \mathbf { D } _ { \mathrm { r o w } , m } ^ { 2 } ) ,
$$

and for the column-wise approach:

$$
E _ { m } ^ { ( \mathrm { c o l } ) } = \| \mathbf { Y } _ { \mathrm { c o l } } \| _ { F } ^ { 2 } - \mathrm { T r } ( \mathbf { D } _ { \mathrm { c o l } , m } ^ { 2 } ) .
$$

Since $\mathbf { Y } _ { \mathrm { r o w } }$ and $\mathbf { Y } _ { \mathrm { c o l } }$ contain exactly the same elements rearranged, their Frobenius norms coincide: $\| \mathbf { Y } _ { \mathrm { r o w } } \| _ { F } ^ { 2 } =$ $\| \mathbf { Y } _ { \mathrm { c o l } } \| _ { F } ^ { 2 }$ . Subtracting yields

$$
E _ { m } ^ { ( \mathrm { c o l } ) } - E _ { m } ^ { ( \mathrm { r o w } ) } = \mathrm { T r } ( \mathbf { D } _ { \mathrm { r o w } , m } ^ { 2 } ) - \mathrm { T r } ( \mathbf { D } _ { \mathrm { c o l } , m } ^ { 2 } ) .
$$

Since $\begin{array} { r } { \mathrm { T r } ( \mathbf { D } _ { \mathrm { r o w } , m } ^ { 2 } ) = N Q \sum _ { p = 1 } ^ { m } \lambda _ { p } ^ { \mathrm { ( r o w ) } } } \end{array}$ and $\begin{array} { r } { \mathrm { T r } ( \mathbf { D } _ { \mathrm { c o l } , m } ^ { 2 } ) = Q S \sum _ { p = 1 } ^ { m } \lambda _ { p } ^ { \mathrm { ( c o l ) } } } \end{array}$ , the equivalence (28) follows.

## A.4. Proof of Corollary 1

Proof. If $\mathbf { V } _ { j } = \mathbf { V } _ { 0 }$ for all �, then $\begin{array} { r } { \pmb { \Sigma } _ { j } = \mathbf { V } _ { 0 } ( \frac { 1 } { N } \mathbf { D } _ { j } ^ { 2 } ) \mathbf { V } _ { 0 } ^ { \top } } \end{array}$ and $\begin{array} { r } { \pmb { \Sigma } _ { \mathrm { r o w } } = \mathbf { V } _ { 0 } ( \frac { 1 } { N O } \sum _ { j } \mathbf { D } _ { j } ^ { 2 } ) \mathbf { V } _ { 0 } ^ { \top } } \end{array}$ . Since $\begin{array} { r } { \frac { 1 } { N O } \sum _ { j } \mathbf { D } _ { j } ^ { 2 } } \end{array}$ is diagonal, $\mathbf { V } _ { 0 }$ is also the eigenvector matrix of $\pmb { \Sigma } _ { \mathrm { r o w } }$ . Truncating at rank � then yields $\mathbf { V } _ { \mathrm { r o w } , m } = \mathbf { V } _ { 0 , m } = \mathbf { V } _ { k , m } ^ { } $ , hence $\mathbf { P } _ { \mathrm { r o w } , m } = \mathbf { P } _ { k , m }$ and $\Delta E _ { k , m } ^ { ( \mathrm { r o w } ) } = 0$ . The column-wise case is analogous with � and �. □

## B. Uncertainty quantification of the standard �–ε turbulence model parameters

The standard �–ε turbulence model (Launder and Spalding, 1974) is widely used in the Computational Fluid Dynamics (CFD) community. It provides reliable predictions for a wide range of flows, including industrial applications, while maintaining reasonable computational cost. The principal hypotheses underlying the model are as follows:

• Turbulence has the same efect as viscous forces. Hence, an eddy viscosity coeficient $\boldsymbol { v } _ { t }$ is introduced. It varies spatially depending on the local flow conditions.

• The eddy viscosity can be expressed as a function of turbulent kinetic energy � and turbulent dissipation rate ε. A dimensional analysis gives:

$$
v _ { t } = C _ { \mu } \frac { k ^ { 2 } } { \varepsilon }\tag{46}
$$

where $C _ { \mu }$ is a constant calibrated on experimental data.

The turbulent kinetic energy is modeled by a transport equation that can be derived exactly from the Reynolds decomposition of the flow velocity $u _ { i } = \bar { u } _ { i } + u _ { i } ^ { \prime } .$ The exact transport equation for the turbulent dissipation rate is highly complex and is therefore represented in a simplified modeled form. The equations of the standard �–ε model are written as follows:

<table><tr><td rowspan=1 colspan=1> $\underline { { C _ { \mu } } }$ </td><td rowspan=1 colspan=1> $\sigma _ { k }$ </td><td rowspan=1 colspan=1> $\sigma _ { \varepsilon }$ </td><td rowspan=1 colspan=1> $C _ { \varepsilon _ { 1 } }$ </td><td rowspan=1 colspan=1> $C _ { \varepsilon _ { 2 } }$ </td></tr><tr><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.30</td><td rowspan=1 colspan=1>1.44</td><td rowspan=1 colspan=1>1.92</td></tr></table>

Table 1

Usual values of the parameters of the standard � − ε model.

$$
\left\{ \begin{array} { l } { \displaystyle \frac { \partial k } { \partial t } + \bar { u } _ { j } \displaystyle \frac { \partial k } { \partial x _ { j } } = \frac { \partial } { \partial x _ { j } } \left[ \left( \boldsymbol { v } + \frac { \boldsymbol { v } _ { t } } { \sigma _ { k } } \right) \displaystyle \frac { \partial k } { \partial x _ { j } } \right] + P _ { k } - \varepsilon } \\ { \displaystyle \frac { \partial \varepsilon } { \partial t } + \bar { u } \displaystyle \frac { \partial \varepsilon } { \partial x _ { j } } = \frac { \partial } { \partial x _ { j } } \left[ \left( \boldsymbol { v } + \frac { \boldsymbol { v } _ { t } } { \sigma _ { \varepsilon } } \right) \displaystyle \frac { \partial \varepsilon } { \partial x _ { j } } \right] + \frac { \varepsilon } { k } ( C _ { \varepsilon _ { 1 } } P _ { k } - C _ { \varepsilon _ { 2 } } \varepsilon ) } \end{array} \right.\tag{47}
$$

where

$$
P _ { k } = - \tau _ { i j } \frac { { \partial } \bar { u } _ { i } } { { \partial } x _ { j } }\tag{48}
$$

is the production of turbulent kinetic energy. In the equation for the turbulent dissipation rate, three parameters have to be calibrated:

• the turbulent Prandtl number for dissipation $\sigma _ { \varepsilon }$ ;

• the coeficient for the source term $C _ { \varepsilon 1 }$ ;

• the coeficient for the sink term $C _ { \varepsilon 2 }$

Ultimately, the model contains five parameters requiring experimental or semi-analytical calibration, with their standard values reported in Table 1.

## B.1. Calibration of $C _ { \mu }$

In the case of a parallel shear flow, the � − ε model equations can be simplified to yield the following expression of $C _ { \mu }$ :

$$
C _ { \mu } = \frac { \varepsilon } { P _ { k } } \left( \frac { \overline { { u _ { 1 } ^ { \prime } u _ { 2 } ^ { \prime } } } } { k } \right) ^ { 2 } .\tag{49}
$$

Assuming local equilibrium between production and dissipation, it follows that $P _ { k } = \varepsilon$ . Therefore, $C _ { \mu }$ can be computed as:

$$
C _ { \mu } = \left( \frac { 2 \tau _ { 1 2 } } { \tau _ { 1 1 } + \tau _ { 2 2 } + \tau _ { 3 3 } } \right) ^ { 2 } .\tag{50}
$$

We can thus determine the value of $C _ { \mu }$ and its uncertainties using the experimental data and information from Buice and Eaton (2000b). Figure 9a shows measured values of the Reynolds stress tensor in the channel upstream of the difuser, indexed by their position along the axis perpendicular to the flow direction. $\mathrm { A t } y / H = 4$ , which is close to the channel wall, the local equilibrium hypothesis may hold. A nominal value of $C _ { \mu } = 0 . 0 6$ is obtained at this location, which is close to the classical value of $C _ { \mu } = 0 . 0 9$

In Buice and Eaton (2000b), information about the measurement uncertainties for the diferent component Reynolds stress tensor are reported as follows: 4% for $\tau _ { 1 1 }$ , 5% for $\tau _ { 2 2 }$ and 10% for $\tau _ { 1 2 }$ . Note that $\tau _ { 3 3 }$ is not given in the Buice-Eaton

![](images/9a1ecaf0d95bfce8e93e4d2c898c0fc111d4e3eed20541fa1c51207cb153174b.jpg)

(a) Experimental data from (Buice and Eaton, 2000b).  
![](images/c72da8b667a8aab4f74896ec561afa168f5dee48a79da367d8cfc273976bd9c5.jpg)  
(b) Probability density function (PDF) estimate of $C _ { \mu }$

Figure 9: Experimental data of Buice and Eaton (2000b) and probability density estimate of $C _ { \mu }$
<table><tr><td rowspan=1 colspan=1> $\tau _ { 1 1 }$ </td><td rowspan=1 colspan=1> $\tau _ { 2 2 }$ </td><td rowspan=1 colspan=1> $\tau _ { 3 3 }$ </td><td rowspan=1 colspan=1> $\tau _ { 1 2 }$ </td></tr><tr><td rowspan=1 colspan=1>4.020e-3</td><td rowspan=1 colspan=1>1.576e-3</td><td rowspan=1 colspan=1>2.798e-3</td><td rowspan=1 colspan=1>1.020e-3</td></tr></table>

## Table 2

Measured values for the Reynolds stress tensor component for $y / H = 4$

experimental campaign data. An ad-hoc value of $( \tau _ { 1 1 } + \tau _ { 2 2 } ) / 2$ is used with an assigned uncertainty of 4%. Accordingly, the probability distributions of the $\tau _ { i j }$ are assumed to be Gaussian, such that:

$$
\tau _ { i j } \sim \mathcal { N } \left( \tau _ { i j , \mathrm { m e s } } , \frac { \Delta \tau _ { i j } \times \tau _ { i j , \mathrm { m e s } } } { 2 } \right)\tag{51}
$$

where $\tau _ { i j , \mathrm { m e s } }$ are the measured values for �∕� = 4 specified in Table 2, and $\Delta \tau _ { i j }$ the uncertainties values for the corresponding component of the Reynolds stress tensor. Figure 9b shows the nonparametric estimate of the PDF of $C _ { \mu } .$ , based on a simulated sample of 1,000 draws from the Gaussian distributions of the components $\tau _ { i j }$ , which are then used to compute $C _ { \mu }$

![](images/071e5d37b7519ed588df38709f6bcf627b826ba27e0d8742596342239c1620f3.jpg)  
Figure 10: Histogram and gaussian KDE of the distribution of $C _ { \varepsilon _ { 2 } }$ using a 1,000-sized sample of β values.

## B.2. $C _ { \varepsilon _ { 2 } }$ calibration

The parameter $C _ { \varepsilon _ { 2 } }$ is calibrated for an isotropic, homogeneous turbulence regime in which the turbulent kinetic energy decays according to a power law $k ( t ) \propto t ^ { - \beta }$ . In this case, the spatial gradients of the flow velocity � and ε are zero. From the standard �–ε model, it follows that:

$$
C _ { \varepsilon _ { 2 } } = \frac { \beta + 1 } { \beta }\tag{52}
$$

The probability density function of $\beta$ is inferred from Figure 1 of Meldi and Sagaut (2012), which shows a histogram of a sample of 650 β values obtained from experimental data and Direct Numerical Simulations (Moin and Mahesh, 1998). Based on this information, we assume that the power-law exponent $\beta$ follows a uniform distribution such that:

$$
\begin{array} { r } { \beta \sim \mathcal { V } ( [ 0 . 9 5 , 1 . 2 5 ] ) . } \end{array}\tag{53}
$$

Figure 10 shows an histogram and a Kernel Density Estimation (KDE) using a Gaussian kernel on a 1,000-sized sample of $C _ { \varepsilon _ { \gamma } }$ values, determined using a 1,000-sized sample of β values using the distribution defined in Equation (53) and the relationship between $C _ { \varepsilon _ { 2 } }$ and β given in Equation (52).

## B.3. Prandtl numbers $\sigma _ { k }$ and $\sigma _ { \varepsilon }$

The parameters $\sigma _ { k }$ and $\sigma _ { \varepsilon }$ are assigned a 10% uncertainty, as reported in Foken (2006). Additionally, the constraint $2 \sigma _ { k } - \sigma _ { \varepsilon } \leq 1$ must be enforced to ensure the correct behavior of the standard $k { - } \varepsilon$ model at the turbulent/nonturbulent interface (Cazalbou, Spalart and Bradshaw, 1994). Accordingly, $\sigma _ { k }$ and $\sigma _ { \varepsilon }$ are assumed to be uniformly distributed over the ranges $[ \sigma _ { k } ^ { * } \times 0 . 9 , \sigma _ { k } ^ { * } \times 1 . 1 ]$ and $[ \sigma _ { \varepsilon } ^ { * } \times 0 . 9 , \sigma _ { \varepsilon } ^ { * } \times 1 . 1 ]$ , respectively. From an i.i.d. sample $( \sigma _ { k , i } , \sigma _ { \varepsilon , i } ) _ { 1 \leq i \leq N }$ , only those samples satisfying $2 \sigma _ { k , i } - \sigma _ { \varepsilon , i } \leq 1$ are retained.

## B.4. $C _ { \varepsilon _ { 1 } }$ calibration

Considering the fluid flow in the boundary layer and local equilibrium $( \varepsilon = P _ { k } )$ , we obtain the following equation between $C _ { \varepsilon _ { 1 } }$ and $C _ { \varepsilon _ { 2 } }$

$$
C _ { \varepsilon _ { 1 } } = C _ { \varepsilon _ { 2 } } - \frac { \kappa ^ { 2 } } { \sigma _ { \varepsilon } \sqrt { C _ { \mu } } } ,\tag{54}
$$

where κ is the Von Karman constant. A dataset of $n = 8$ values are obtained from Foken (2006). A Gaussian is fitted on this dataset with its parameters $( m _ { \kappa } , \sigma _ { \kappa } ^ { 2 } )$ estimated empirically.

## B.5. Sampling procedure for the standard � − ε turbulence model parameters

After properly assessing the sources of uncertainty in the �–ε model, Algorithm 2 outlines the procedure for sampling values of the standard �–ε parameter vector $\mathbf { x } = ( C _ { \mu } , \sigma _ { k } , \sigma _ { \varepsilon } , C _ { \varepsilon _ { 1 } } , C _ { \varepsilon _ { 2 } } )$

Algorithm 2: Sampling algorithm of the standard � − ε turbulence model parameters.   
Step 1: Sample $( \sigma _ { k } , \sigma _ { \varepsilon } )$ from their respective Uniform distribution conditioned by $2 \sigma _ { k } - \sigma _ { \varepsilon } \leq 1$   
Step 2: Sample $\kappa \sim \bar { \mathcal { N } } ( m _ { \kappa } , \sigma _ { \kappa } ^ { 2 } )$   
Step 3: Sample β ∼ ([0.95, 1.25]).   
Step 4: Sample $\tau _ { 1 1 } , \tau _ { 2 2 } , \tau _ { 1 2 } , \tau _ { 3 3 }$ from Normal distributions with parameters determined using Buice and   
Eaton (2000b).   
Step 5: Compute $C _ { \varepsilon _ { 2 } } = ( \beta + 1 ) / \beta .$   
Step 6: Compute $C _ { \mu } = \Bigg ( \frac { 2 \tau _ { 1 2 } } { \tau _ { 1 1 } + \tau _ { 2 2 } + \tau _ { 3 3 } } \Bigg ) ^ { 2 } .$   
Step 7: Compute $C _ { \varepsilon _ { 1 } } = C _ { \varepsilon _ { 2 } } - \frac { \kappa ^ { 2 } } { \sigma _ { \varepsilon } \sqrt { C _ { \mu } } } .$

## C. Complete win rate analysis of the deductive approach benchmark

This appendix complements the analysis of Section 3 with the complete win rate comparison for all three outputs. Figure 11 reports the win rate analysis at $N \in \{ 2 0 , 5 0 , 1 0 0 \}$ . Three observations stand out. First, the winrates of the two deductive methods change across deduction scenarios at every training set size, while the CMOGP varies as a mechanical consequence of its fixed RMSE. The variability of the deductive methods across scenarios is therefore present at � = 20, 50, and 100, and does not vanish as the training set grows. Second, the independent GP has a markedly low win rate in the middle panel of each row at the scenario where $f _ { 2 }$ is the deduced output, with values ranging from 6% at � = 20 to 0% at � = 100 : this is consistent with the shift of its RMSE distribution towards higher values when $f _ { 2 }$ is deduced. Third, and most importantly, the CMOGP is the most frequent best method in nearly all configurations and across all evaluated outputs, with a clear margin over the two deductive methods.

![](images/3a11f9b6ede9c428ee06624a21d06c5c3d908f1cc9da791dab64b6ec0c2ae046.jpg)

(a) � = 20  
![](images/1c02bb83611026e5bb548bfd8faf22566f628dcad281270ec9400d2fb7bb9c53.jpg)

(b) � = 50  
![](images/d46a9af60b3f35cc1342e25d9fc69746fe847b88b7f3bcb29589e985a86bf505.jpg)  
(c) $N = 1 0 0$  
Figure 11: Win rate (%) of each method on the RRMSE of each output $\left( f _ { 1 } , \ f _ { 2 } , \ f _ { 3 } \right)$ for $N \in \{ 2 0 , 5 0 , 1 0 0 \}$ , across all deduction scenarios $\left( k _ { * } \in \left\{ 1 , 2 , 3 \right\} \right)$ . The highlighted column indicates the scenario where the evaluated output is the deduced one. The CMOGP (blue) is the same model in all scenarios. Results are aggregated over 200 independent replications.

![](images/1f81b44d074024829c0e3ad2a34623b6f57004c7008299d7c622adcb5bafc02e.jpg)  
Figure 12: Distribution of the average 90% predictive interval length for $f _ { 2 }$ across 200 replications, for training set sizes $N \in 2 0 , 5 0 .$ , 100 and the three deduction scenarios. All methods achieve comparable empirical coverage rates. Shorter intervals thus indicate more informative predictions. The CMOGP maintains stable and shorter intervals across all scenarios.

## C.1. Predictive intervals

Figure 12 shows the distribution of the average 90% predictive interval length for $f _ { 2 }$ across all deduction scenarios. For the deductive methods, predictive intervals on the deduced output are obtained by propagating the posterior samples of the modeled outputs through (18).

## D. Complete RRMSE results for the Lotka–Volterra experiment

Figure 13 displays the RRMSE as a function of the latent dimension � for all four outputs $\left( p , q , r , s \right)$ and the three training set sizes $( N \in \{ 3 0 , 1 5 , 1 0 \} )$ , comparing Row-CMO against Col-Indep and Fw-Indep with all deduction scenarios. The patterns discussed in Section 5.1 on the output � are consistently observed across all outputs: Row-CMO starts above the alternatives at small � and surpasses them at moderate-to-high latent dimensions, with the gap widening as � decreases.

![](images/ca1910317dece7cbb77fffc436dbe78ff18c940725c7ee997304d13465269cae.jpg)  
Figure 13: RRMSE as a function of � for all four Lotka–Volterra outputs $( p , \ q , \ r , \ s ,$ columns) and training set sizes $( N \ : = \ : 3 0 , 1 5 , 1 0$ , rows). Row-CMO (red) is compared against Col-Indep and Fw-Indep with all deduction scenarios $k _ { * } \in \{ 0 , 1 , 2 , 3 \}$ . Error bars represent ±1 standard deviation over 10 seeds.

## References

Abrahamsen, P., Benth, F.E., 2001. Kriging with inequality constraints. Mathematical Geology 33, 719–744.

Aitchison, J., 1986. The Statistical Analysis of Compositional Data. Chapman and Hall.

Alvarez, M.A., Rosasco, L., Lawrence, N.D., 2012. Kernels for vector-valued functions: A review. Foundations and Trends® in Machine Learning 4, 195–266.

Alvera-Azcárate, A., Barth, A., Beckers, J.M., Weisberg, R.H., 2007. Multivariate reconstruction of missing data in sea surface temperature, chlorophyll, and wind satellite fields. Journal of Geophysical Research: Oceans 112.

Boussinesq, J., 1897. Théorie de l’écoulement tourbillonnant et tumultueux des liquides dans les lits rectilignes à grande section. volume 1. Gauthier-Villars.

Buice, C.U., Eaton, J.K., 2000a. Experimental investigation of flow through an asymmetric plane difuser. Journal of Fluids Engineering 122, 433–435.

Buice, C.U., Eaton, J.K., 2000b. Experimental investigation of flow through an asymmetric plane difuser: (data bank contribution)1. Journal of Fluids Engineering 122, 433–435.

Cazalbou, J.B., Spalart, P.R., Bradshaw, P., 1994. On the behavior of two-equation models at the edge of a turbulent region. Physics of Fluids 6, 1797–1804.

Da Veiga, S., Marrel, A., 2020. Gaussian process regression with linear inequality constraints. Reliability Engineering & System Safety 195, 106732.

Davis, C., Kahan, W.M., 1970. The rotation of eigenvectors by a perturbation. III. SIAM Journal on Numerical Analysis 7, 1–46.

Foken, T., 2006. 50 years of the monin–obukhov similarity theory. Boundary-Layer Meteorology 119, 431–447.

Forrester, A., Sobester, A., Keane, A., 2008. Engineering Design via Surrogate Modelling: A Practical Guide. John Wiley & Sons.

Forrester, A.I., Sóbester, A., Keane, A.J., 2007. Multi-fidelity optimization via surrogate modelling. Proceedings of the Royal Society A 463, 3251–3269.

Goulard, M., Voltz, M., 1992. Linear coregionalization model: tools for estimation and choice of cross-variogram matrix. Mathematical Geology 24, 269–286.

Graepel, T., 2003. Solving noisy linear operator equations by Gaussian processes: Application to ordinary and partial diferential equations, in: Proceedings of the Twentieth International Conference on Machine Learning, pp. 234–241.

Hannachi, A., Jollife, I.T., Stephenson, D.B., 2007. Empirical orthogonal functions and related techniques in atmospheric science: A review. International Journal of Climatology 27, 1119–1152.

Happ, C., Greven, S., 2018. Multivariate functional principal component analysis for data observed on diferent (dimensional) domains. Journal of the American Statistical Association 113, 649–659.

Herman, E., Stewart, J.A., Dingreville, R., 2020. A data-driven surrogate model to rapidly predict microstructure morphology during physical vapor deposition. Applied Mathematical Modelling 88, 589–603.

Higdon, D., 2002. Space and space-time modeling using process convolutions, in: Quantitative Methods for Current Environmental Issues. Springer, pp. 37–56.

Higdon, D., Gattiker, J., Williams, B., Rightley, M., 2008. Computer model calibration using high-dimensional output. Journal of the American Statistical Association 103, 570–583.

Ishigami, T., Homma, T., 1990. An importance quantification technique in uncertainty analysis for computer models, in: Proceedings of the First International Symposium on Uncertainty Modeling and Analysis, IEEE. pp. 398–403.

Jamil, M., Yang, X.S., 2013. A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematica Modelling and Numerical Optimisation 4, 150–194.

Jansson, T., Nilsson, L., Redhe, M., 2003. Using surrogate models and response surfaces in structural optimization–with application to crashworthiness design and sheet metal forming. Structural and Multidisciplinary Optimization 25, 129–140.

Jidling, C., Wahlström, N., Wills, A., Schön, T.B., 2017. Linearly constrained Gaussian processes, in: Advances in Neural Information Processing Systems, pp. 1–8.

Kennedy, M.C., O’Hagan, A., 2000. Predicting the output from a complex computer code when fast approximations are available. Biometrika 87, 1–13.

Lange-Hegermann, M., 2018. Algorithmic linearly constrained Gaussian processes, in: Advances in Neural Information Processing Systems (NeurIPS), pp. 1–9.

Launder, B., Spalding, D., 1974. The numerical computation of turbulent flows. Computer Methods in Applied Mechanics and Engineering 3, 269–289.

Li, M., Sadoughi, M., Hu, Z., Hu, C., 2020. A hybrid Gaussian process model for system reliability analysis. Reliability Engineering & System Safety 197, 106816.

Liang, Y.C., Mazlof, M.R., Rosso, I., Fang, S.W., Yu, J.Y., 2018. A multivariate empirical orthogonal function method to construct nitrate maps in the southern ocean. Journal of Atmospheric and Oceanic Technology 35, 1505–1519.

Maatouk, H., Bay, X., 2017. Gaussian process emulators for computer experiments with inequality constraints. Mathematical Geosciences 49, 557–582.

Macêdo, I., Castro, R., 2010. Learning divergence-free and curl-free vector fields with matrix-valued kernels. IMPA IMPA Technical Report.

Mak, S., Sung, C.L., Wang, X., Yeh, S.T., Chang, Y.H., Joseph, V.R., Yang, V., Wu, C.F.J., 2018. An eficient surrogate model for emulation and physics extraction of large eddy simulations. Journal of the American Statistical Association 113, 1443–1456.

Marrel, A., Iooss, B., Chabridon, V., 2022. The ICSCREAM methodology: Identification of penalizing configurations in computer experiments using screening and metamodel—applications in thermal hydraulics. Nuclear Science and Engineering 196, 301–321.

Matheron, G., 1963. Principles of geostatistics. Economic Geology 58, 1246–1266.

Meldi, M., Sagaut, P., 2012. On non-self-similar regimes in homogeneous isotropic turbulence decay. Journal of Fluid Mechanics 711, 364–393.

Moin, P., Mahesh, K., 1998. Direct numerical simulation: a tool in turbulence research. Annual review of fluid mechanics 30, 539–578.

Nanty, S., Helbert, C., Marrel, A., Pérot, N., Prieur, C., 2016. Sampling, metamodeling, and sensitivity analysis of numerical simulators with functional stochastic inputs. SIAM/ASA Journal on Uncertainty Quantification 4, 636–659.

Nocedal, J., Wright, S.J., 2006. Numerical Optimization. 2nd ed., Springer.

Nutku, Y., 1990. Hamiltonian structure of the Lotka–Volterra equations. Physics Letters A 145, 27–28.

O’Hagan, A., 2006. Bayesian analysis of computer code outputs: A tutorial. Reliability Engineering & System Safety 91, 1290–1300.

Perrin, G., Soize, C., Duhamel, D., Funfschilling, C., 2013. Karhunen–Loève expansion revisited for vector-valued random fields: Scaling, errors and optimal basis. Journal of Computational Physics 242, 607–622.

Petitgas, P., Doray, M., Huret, M., Massé, J., Woillez, M., 2014. Modelling the variability in fish spatial distributions over time with empirical orthogonal functions: anchovy in the bay of biscay. ICES Journal of Marine Science 71, 2379–2389.

Pilar, P., Jidling, C., Schön, T.B., Wahlström, N., 2022. Incorporating sum constraints into multitask gaussian processes. Transactions on Machine Learning Research .

Preisendorfer, R.W., Mobley, C.D., 1988. Principal Component Analysis in Meteorology and Oceanography. Elsevier.

Puscas, M.A., Angeli, P.E., Nouaime, N., Jamelot, E., 2025. Description and convergence order analysis of the finite element-volume spatial discretization method. International Journal for Numerical Methods in Fluids 97, 1189–1208.

Rasmussen, C.E., Williams, C.K.I., 2005. Gaussian Processes for Machine Learning. MIT Press.

Rougier, J., 2008. Eficient emulators for multivariate deterministic functions. Journal of Computational and Graphical Statistics 17, 827–843.

Santner, T.J., Williams, B.J., Notz, W.I., 2003. The Design and Analysis of Computer Experiments. Springer New York.

Sobieszczanski-Sobieski, J., Haftka, R.T., 1997. Multidisciplinary aerospace design optimization: survey of recent developments. Structura Optimization 14, 1–23.

Sparnocchia, S., Pinardi, N., Demirov, E., 2003. Multivariate empirical orthogonal function analysis of the upper thermocline structure of the Mediterranean Sea from observations and model simulations, in: Annales Geophysicae, Copernicus GmbH. pp. 167–187.

Stegle, O., Lippert, C., Mooij, J.M., Lawrence, N., Borgwardt, K., 2011. Eficient inference in matrix-variate Gaussian models with i.i.d. observation noise, in: Advances in Neural Information Processing Systems, pp. 1–9.

Stewart, G.W., Sun, J.G., 1990. Matrix Perturbation Theory. Academic Press.

Su, G., Peng, L., Hu, L., 2017. A Gaussian process-based dynamic surrogate model for complex engineering structural reliability analysis. Structura Safety 68, 97–109.

Swiler, L.P., Gulian, M., Frankel, A.L., Safta, C., Jakeman, J.D., 2020. A survey of constrained Gaussian process regression: Approaches and implementation challenges. Journal of Machine Learning for Modeling and Computing 1, 119–156.

Teymur, O., Gorham, J., Riabiz, M., Oates, C., 2021. Optimal quantisation of probability measures using maximum mean discrepancy, in: Banerjee, A., Fukumizu, K. (Eds.), Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, PMLR. pp. 1027–1035.

Wang, S., Zhang, X., Meng, Y., Xing, W.W., 2022. E-LMC: Extended linear model of coregionalization for spatial field prediction, in: 2022 International Joint Conference on Neural Networks (IJCNN), IEEE. pp. 1–8.

Wang, X., Berger, J.O., 2016. Estimating shape constrained functions using Gaussian processes. SIAM/ASA Journal on Uncertainty Quantification 4, 1–25.

Wangersky, P.J., 1978. Lotka–Volterra population models. Annual Review of Ecology and Systematics 9, 189–218.

Xing, W., Shah, A.A., Nair, P.B., 2015. Reduced dimensional Gaussian process emulators of parametrized partial diferential equations based on Isomap. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 471, 20140697.

Xing, W., Yu, F., Leung, P., Li, X., Wang, P., Shah, A., 2021. A new multi-task learning framework for fuel cell model outputs in high-dimensional spaces. Journal of Power Sources 482, 228930.

Zhe, S., Xing, W., Kirby, R.M., 2019. Scalable high-order Gaussian process regression, in: Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, pp. 2611–2620.

Zhou, T., Peng, Y., 2020. Kernel principal component analysis-based Gaussian process regression modelling for high-dimensional reliability analysis. Computers & Structures 241, 106358.