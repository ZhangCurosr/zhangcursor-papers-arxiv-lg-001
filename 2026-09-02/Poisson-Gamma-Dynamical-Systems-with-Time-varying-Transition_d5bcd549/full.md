# Poisson-Gamma Dynamical Systems with Time-varying Transition Dynamics

Jiahao Wang<sup>1,†</sup>, Yijun Wang<sup>2,3,4,†</sup>, Nan Fang<sup>2</sup>, Sikun Yang<sup>2,4,‡</sup>

<sup>1</sup> Department of Electrical and Computer Engineering, University of Arizona, US

<sup>2</sup> School of Computing and Information Technology, Great Bay University, China, <sup>3</sup> Tsinghua University, China

Guangdong Provincial Key Laboratory of Mathematical and Neural Dynamical Systems

Email: jiahaow@arizona.edu, wangyj@gbu.edu.cn, 2310275052@email.szu.edu.cn, sikunyang@gbu.edu.cn

Abstract—Bayesian methodologies for handling count-valued time series have gained prominence due to their ability to infer interpretable latent structures and to estimate uncertainties. Among these Bayesian models, Poisson-Gamma Dynamical Systems (PGDSs) are proven to be effective in capturing the evolving dynamics underlying observed count sequences. However, the state-of-the-art PGDS still falls short in capturing the timevarying transition dynamics that are commonly observed in realworld count time series. To mitigate this limitation, a PGDS with time-varying transition kernel (TV-PGDS), is proposed to allow the underlying transition matrices to evolve over time. Three specifically-designed Dirichlet Markov chains (Dir-Dir, Dir-Gam-Dir, PR-Gam-Dir) are constructed to accommodate heterogeneous structural mutations within these dependencies. Leveraging Dirichlet-Multinomial-Beta data augmentation techniques, a fully-conjugate and efficient Gibbs sampler is developed to perform posterior simulation. Experiments show that, in comparison with related models, the proposed PGDS achieves improved predictive performance due to its capacity to learn time-varying dependency structure captured by the time-evolving transition matrices.

Index Terms—Count Sequences, Poisson-Gamma Dynamical Systems, Time-varying Transition Kernels

## I. INTRODUCTION

There has been an increasing interest in modeling count time series. For instance, some previous works [1]–[3] are concerned with how to learn the evolving topics behind text corpus (frequencies of words) over time. Some works [4]– [7] try to predict global immigrant trends underlying international population movements. Count time series are often overdispersed, sparse, high-dimensional, and thus can not be well modeled by widely used dynamic models such as linear dynamical systems (LDS) [8], [9].

Recently, many works [10]–[16] prefer to adopt nonnegative distributions from the gamma-Poisson family to construct hierarchical Bayesian models for count time series. In particular, these models enjoy strong explainability and can estimate uncertainty especially when the observations are noisy and incomplete. Among these works, Poisson-Gamma Dynamical Systems (PGDSs) [13] received a lot of attention. PGDS preserves the dynamical mechanism of LDS by evolving the latent state ${ \boldsymbol { \theta } } ^ { ( t ) }$ via a gamma Markov chain: $\theta ^ { ( t ) } \sim \mathsf { \bar { G } } a m ( \tau _ { 0 } \Pi \theta ^ { ( t - 1 ) } , \tau _ { 0 } )$ . Here, the stationary transition kernel Π captures complex dynamics by learning how latent dimensions excite one another. For example, a very inspiring research paper may motivate other researchers to publish papers on related topics [17]. In particular, PGDS can be efficiently learned with a tractable Gibbs sampling scheme via Poisson-Logarithmic data augmentation and marginalization technique [11]. Due to its strong flexibility, PGDS achieves better performance in predicting missing entities and future observations, compared with related models.

Despite these advantages, although PGDS considered the interdependencies of the latent variables within the Markovian stochastic process, it remains limited by its reliance on static interaction mechanism among these latent variables. This timeinvariant transition kernel hinders its ability to deeply describe shifting dynamics across time, which are ubiquitous in realworld scenarios [18]. For instance, during the initial stage of the COVID-19 pandemic, the worldwide counts of infectious patients were significantly affected by various local policies, government interventions, and emergent events [19]–[21]. The cross transition dynamics among the different monitoring areas were also evolving as the corresponding policies and interventions changed over time. Hence, PGDS unavoidably makes a certain amount of approximation error in capturing the aforementioned time-varying count time series, using a time-invariant transition kernel.

To mitigate this limitation, Time-Varying Poisson-Gamma Dynamical Systems (TV-PGDS), a novel kind of Poissongamma dynamical systems with time-varying transition dynamics are developed. In particular, TV-PGDS models the time-varying transition dynamics using tailored Dirichlet Markov chains. Via the Dirichlet-Multinomial-Beta data augmentation strategy, TV-PGDS can be inferred with a conjugate-yet-efficient Gibbs sampler. While ensuring efficient inference, TV-PGDS explicitly captures how does the latent dependencies changes over time as illustrated in Figure 1. As far as we know, this is not realised in existing models. Our contributions are summarized as follows:

1) We propose a time-varying Poisson-Gamma Dynamical System (TV-PGDS), a novel Poisson-gamma dynamical system with time-evolving transition matrices that can well capture time-varying transition dynamics underlying observed count series.

2) Three Dirichlet Markov chains are dedicated to improving the flexibility and expressiveness of TV-PGDS, for capturing the complex transition dynamics behind sequential count data.

![](images/a79a092fe333aa05bec04999230075fff03c3ad21e176bc7e8b271e893908a4a.jpg)

![](images/ed3641a567d49f4c307738a924b2413dc9292c7f6c9432f1d5361216a4dbb14d.jpg)  
Fig. 1. An example illustrates the Poisson-gamma dynamical systems with time-varying transition kernels. The three gamma dynamic processes independently evolve over time during the $( i - \overline { { 1 } } ) \ – \ t h$ interval. During i-th interval, $\theta _ { 1 } ^ { ( t ) }$ and $\theta _ { 2 } ^ { ( t ) }$ gradually starts to interact with each other while $\theta _ { 3 } ^ { ( t ) }$ remains independent to the other two dimensions. During (i + 1)-th interval all the three latent components start to interact with each other.

3) Fully-conjugate-yet-efficient Gibbs samplers are developed via Dirichlet-Multinomial-Beta augmentation techniques to perform posterior simulation for the proposed Dirichlet Markov chains.

4) Extensive experiments are conducted on four real-world datasets, to evaluate the performance of the proposed TV-PGDS in predicting missing and future unseen observations. We also provide exploratory analysis to demonstrate the explainable latent structure inferred by the proposed TV-PGDS.

## II. RELATED WORK

To capture latent dynamics underlying count sequences, previous works [22], [23] employed a probabilistic framework $\bar { \boldsymbol { y } } ^ { ( t ) } = p ( \boldsymbol { z } ^ { ( t ) } )$ , with $z ^ { ( t ) } = \dot { f } ^ { - 1 } ( \theta ^ { ( \bar { t } ) } )$ , where $\boldsymbol { y } ^ { ( t ) } \in \mathbb { N } ^ { V }$ represents count-valued observations at time t. The function $p ( \cdot )$ is the likelihood, and $f ( \cdot )$ maps parameters to continuous latent variables ${ { \bf \nabla } \theta ^ { ( t ) } } ~ { \bf { \Lambda } } \in { { \bf { \bar { \mathbb { R } } } } ^ { \dot { K } } }$ . In linear dynamical systems $\mathrm { ( L D S ) } , \ \theta ^ { ( t ) }$ evolves via a Gaussian Markov chain: $\bar { \boldsymbol { \theta } } ^ { ( t ) }$ ∼ $\mathcal { N } ( \Pi ^ { ( t , t - 1 ) } \theta ^ { ( t - 1 ) } , \Lambda ^ { - 1 } )$ . Π is the state transition matrix, and Λ is the inverse covariance matrix. While some work [22] adopted link functions within the LDS framework to model count observations, it incurs high computational complexity $( \mathcal { O } ( ( K + V ) ^ { 3 } ) )$ , limiting applicability to large-scale sequences. Although LDS and its variants perform well on real-valued sequences, their Gaussian state-space assumption hinders inference efficiency and fails to accurately model burstiness in count time series. Recently, Poisson-gamma family models [13], [15], [16] have emerged for count sequences. Gamma Process Dynamic Poisson Factor Analysis (GP-DPFA) [15] models count data as $\begin{array} { r } { y _ { v } ^ { ( t ) } \sim \mathrm { P o i s } ( \sum _ { k = 1 } ^ { \bar { K } } \lambda _ { k } \phi _ { v k } \theta _ { k } ^ { ( t ) } ) } \end{array}$ , where $\theta _ { k } ^ { ( t ) }$ represents the strength of k-th latent factor at time $t ,$ and $\phi _ { v k }$ captures the involvement degree of k-th factor to v-th observed dimension, regularized by a Dirichlet prior $\begin{array} { l l l } { \phi _ { k } } & { \sim } & { D i r ( \epsilon _ { 0 } , \cdot { \bf \sigma } \cdot { \bf \sigma } \cdot , \epsilon _ { 0 } ) } \end{array}$ . The latent factor evolves via a gamma Markov chain: $\theta _ { k } ^ { ( t ) } \ \sim \ G a m ( \theta _ { k } ^ { ( t - 1 ) } , c _ { t } )$ . Although GP-DPFA fits one-dimensional sequences well, it fails to learn interactions among latent dimensions. To address this concern, Poisson-Gamma Dynamical Systems (PGDS) [13] captures underlying transition dynamics by evolving $\dot { \theta } _ { k } ^ { ( t ) }$ as $\begin{array} { r l r } { \theta _ { k } ^ { ( \hat { t } ) } } & { \sim } & { \mathrm { G a m } ( \bar { \tau } _ { 0 } \sum _ { k _ { 2 } = 1 } ^ { K } \pi _ { k k _ { 2 } } \theta _ { k _ { 2 } } ^ { ( \hat { t } - 1 ) } , \tau _ { 0 } ) } \end{array}$ , where $\pi _ { k k _ { 2 } }$ represents how $k _ { 2 } \mathrm { { \cdot } }$ -th latent factor excites the k-th latent factor at next time step, and $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \pi _ { k k _ { 2 } } = 1 } \end{array}$ . Several extensions of PGDS are developed to further enhance the flexibility of the gamma Markov chain in capturing dynamics. Switching PGDS (SPGDS) [24] models non-linear dynamics via discrete switching among finite transition regimes as $\theta _ { k } ^ { ( t ) } \sim$ $\begin{array} { r } { \prod _ { c = 1 } ^ { C } G a m ( \tau _ { 0 } \Pi _ { c } \theta _ { k _ { 2 } } ^ { ( t - 1 ) } , \tau _ { c } ) } \end{array}$ . However, they are constrained by stationary assumptions—DPGDS employs fixed kernels, while SPGDS aggregates discrete states—making them inadequate for continuously evolving non-stationary environments.

Another stream of studies models dynamics by chaining the rate parameter of the gamma distribution. For example, GMC-RATE [25] formulates $\theta _ { k } ^ { ( t ) } \ \sim \ \mathrm { G a m } ( \alpha , \beta / \theta _ { k } ^ { ( t - 1 ) } )$ . GMC-HIER [26] introduces hierarchical auxiliary variables, such as $z _ { k } ^ { ( t ) } \stackrel {  } { \sim } \mathrm { \bar { G } a m } ( \alpha _ { z } , \beta _ { z } \theta _ { k } ^ { ( t - 1 ) } )$ and $\theta _ { k } ^ { ( t ) } \sim \mathsf { \bar { G } a m } ( a _ { \theta } , \beta _ { \theta } z _ { k } ^ { ( t ) } )$ However, since the rate parameter governs variance, these models are prone to high volatility, potentially causing numerical instability or model collapse.

## III. TIME-VARYING POISSON-GAMMA DYNAMICAL SYSTEMS

Real-world count time sequences are often time-varying because the external interventional environments are always changing over time. The stationary PGDS with a time-invariant transition kernel fails to capture such time-varying transition dynamics. Hence, to mitigate this limitation, we model the count sequences as

$$
\begin{array} { r } { \boldsymbol { y } _ { \boldsymbol { v } } ^ { ( t ) } \sim \mathrm { P o i s } \left( \delta ^ { ( t ) } \sum _ { k = 1 } ^ { K } \phi _ { \boldsymbol { v } k } \boldsymbol { \theta } _ { k } ^ { ( t ) } \right) , } \end{array}
$$

in which, the latent factors are specified by

$$
\begin{array} { r } { \theta _ { k } ^ { ( t ) } \sim \mathrm { G a m } \left( \tau _ { 0 } \sum _ { k _ { 2 } = 1 } ^ { K } \pi _ { k k _ { 2 } } ^ { ( t - 1 ) } \theta _ { k _ { 2 } } ^ { ( t - 1 ) } , \tau _ { 0 } \right) , } \end{array}\tag{1}
$$

where the multiplicative term $\delta ^ { ( t ) } \sim \mathrm { G a m } ( \epsilon _ { 0 } , \epsilon _ { 0 } )$ and the transition matrices are time-varying as $\boldsymbol { \Pi } ^ { ( t ) } \equiv [ \pi _ { k k _ { 2 } } ^ { ( t ) } ] _ { k , k _ { 2 } = 1 } ^ { K } .$

As shown in Figure 1, to model the time-varying transition dynamics, we assume the whole time interval can be divided into I equally-spaced sub-intervals. The transition kernel behind complicated dynamic counts is assumed to be static within each sub-interval, while evolving over subintervals, to capture time-varying behaviours. In other words, the proposed model allows the latent factors to evolve over time steps while the transition matrices change over subintervals but assumed to be stationary within each sub-interval, as shown in Figure 2. In particular, we let each sub-interval contains M time steps, and the i-th interval contains time steps $\{ t \mid t = ( i - 1 ) M + 1 , \cdots , i M \}$ . We define i (t) as the function that maps time step t to its corresponding subinterval.

![](images/dd0b19bb435cbb656b8fc874395a71170b1ba3d417201d5fa60d9e0c6c4716b6.jpg)

![](images/b95b9f32425449af3cb2b96f60092731f76826e816cf29f3025d2dd215aa32f1.jpg)  
Fig. 2. The graphical representation of the TV-PGDS. The time interval is divided into equally-spaced sub-intervals. Each sub-interval contains M time steps The transition dynamics is stationary within a sub-interval. In particular, the transition matrices evolve over sub-intervals via Dirichlet Markov processes while latent factors evolve over time steps via Eq.(1).

By setting local stationarity within each sub-interval, the transition matrices mitigate the overfitting risks inherent in high-dimensional temporal modeling. The sub-interval length, $M ,$ governs a fundamental trade-off: while $M = 1$ characterizes a fully dynamic system with maximum granularity, such a configuration often leads to high computational costs or unstable convergence. Hence, the value of M should be determined by the patterns of empirical data and model performance. An analysis of the influence of interval length M on model performance is provided in the following section. Dirichlet-Dirichlet Markov processes. To capture how the underlying transition kernel smoothly evolves over subintervals, we first propose the Dirichlet-Dirichlet (Dir-Dir) Markov chain as

$$
\pi _ { k } ^ { ( i ) } \mid \pi _ { k } ^ { ( i - 1 ) } \sim \operatorname { D i r } \left( \eta K \pi _ { 1 k } ^ { ( i - 1 ) } , \cdot \cdot \cdot , \eta K \pi _ { K k } ^ { ( i - 1 ) } \right) ,\tag{2}
$$

where $\pi _ { k } ^ { ( i ) }$ represents the k-th column of $\boldsymbol { \Pi } ^ { ( i ) }$ , and the prior of the scaling parameter η is given by $\eta \sim$ Gam $( e _ { 0 } , f _ { 0 } )$

The initial states are defined as $\begin{array} { r } { \dot { \theta } _ { k } ^ { ( 1 ) } \sim \operatorname { G a m } \left( \tau _ { 0 } \nu _ { k } , \tau _ { 0 } \right) } \end{array}$ The prior for the transition kernel of the first sub-interval is given by $\pi _ { k } ^ { ( 1 ) } \sim$ Dir $\left( \nu _ { 1 } \nu _ { k } , \cdot \cdot \cdot , \xi \nu _ { k } , \cdot \cdot \cdot , \nu _ { K } \nu _ { k } \right)$ , where $\begin{array} { r } { \nu _ { k } \ \sim \ \mathrm { G a m } ( \frac { \gamma _ { 0 } } { K } , \beta ) } \end{array}$ and $\xi , \beta \sim \mathrm { G a m } ( \epsilon _ { 0 } , \epsilon _ { 0 } ) . \ \nu _ { k }$ captures the global prominence/activity of latent factor k, while $\nu _ { k _ { 1 } } \nu _ { k _ { 2 } }$ provides a separable prior preference for transitions $k _ { 1 }  k _ { 2 }$ Note that the expectation and variance of the transition kernel at i-th sub-interval can be calculated as

$$
\begin{array} { r l } & { \mathsf { E } \left[ \pmb { \pi } _ { k } ^ { ( i ) } \mid \pmb { \pi } _ { k } ^ { ( i - 1 ) } \right] = \pmb { \pi } _ { k } ^ { ( i - 1 ) } , } \\ & { \mathsf { V a r } \left[ \pmb { \pi } _ { k _ { 1 } k } ^ { ( i ) } \mid \pmb { \pi } _ { k } ^ { ( i - 1 ) } \right] = \frac { \pmb { \pi } _ { k _ { 1 } k } ^ { ( i - 1 ) } \left( 1 - \pmb { \pi } _ { k _ { 1 } k } ^ { ( i - 1 ) } \right) } { \pmb { \eta } \pmb { K } + 1 } , } \end{array}
$$

respectively. The transition dynamics of i-th sub-interval inherits the information of the previous sub-interval, and also adapts to the data observed in the current sub-interval. The parameter η controls the variance of the transition matrices.

The prior specification defined in $\operatorname { E q . } ( 2 )$ by rescaling the transition matrix at the previous sub-interval allows the transition dynamics to change smoothly, and thus might be insufficient to capture the rapid mutations observed in complicated dynamics. To further improve the flexibility of the transition structure, two modified Dirichlet Markov chains are studied

(a)

(b)

(c)

Fig. 3. Diagrams of the proposed Dirichlet Markov constructions. (a) is the Dir-Dir construction. (b) is the Dir-Gam-Dir construction which takes mutation into account. (c) illustrates the PR-Gam-Dir construction which adopts Poisson randomized gamma distribution and can be equivalently represented as Eq.(7).

to capture the correlation structure between the dimensions of the transition matrices over time.

Dirichlet-Gamma-Dirichlet Markov processes. We first introduce the Dirichlet-Gamma-Dirichlet (Dir-Gam-Dir) Markov chain to model the evolving transition matrices as

$$
\begin{array} { r l } & { \pi _ { k } ^ { \left( i \right) } \sim \operatorname { D i r } \left( \alpha _ { 1 k } ^ { \left( i \right) } , \cdots , \alpha _ { K k } ^ { \left( i \right) } \right) , } \\ & { \alpha _ { k _ { 1 } k } ^ { \left( i \right) } \sim \operatorname { G a m } \left( \gamma _ { k } ^ { \left( i - 1 \right) } \sum _ { k _ { 2 } = 1 } ^ { K } \psi _ { k k _ { 1 } k _ { 2 } } ^ { \left( i - 1 \right) } \pi _ { k _ { 2 } k } ^ { \left( i - 1 \right) } , c _ { k } ^ { \left( i \right) } \right) , } \end{array}\tag{3}
$$

where we use $\psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) }$ to capture the mutation between two consecutive sub-intervals, and its prior is given by

$$
\left( \boldsymbol { \psi } _ { k 1 k _ { 2 } } ^ { \left( i - 1 \right) } , \cdot \cdot \cdot , \boldsymbol { \psi } _ { k K k _ { 2 } } ^ { \left( i - 1 \right) } \right) \sim \mathrm { D i r } \left( \epsilon _ { 0 } , \cdot \cdot \cdot , \epsilon _ { 0 } \right) ,\tag{4}
$$

and $\gamma _ { k } ^ { ( i ) } , c _ { k } ^ { ( i ) } \sim \mathrm { G a m } \left( \epsilon _ { 0 } , \epsilon _ { 0 } \right)$ . Compared with the construction defined by Eq.(2), the expectation of Dirichlet-Gamma-Dirichlet Markov chain is

$$
\mathsf E \left[ \boldsymbol { \pi } _ { k } ^ { ( i ) } \mid \boldsymbol { \pi } _ { k } ^ { ( i - 1 ) } \right] = \Psi _ { k } ^ { ( i - 1 ) } \boldsymbol { \pi } _ { k } ^ { ( i - 1 ) } .
$$

This construction takes interactions among components of columns into account. Hence it will improve the flexibility of our model and thus better fit more complicated dynamics, compared with Dir-Dir Markov chains that only yield smoothing transition dynamics.

Poisson-randomized-gamma-Dirichlet Markov processes. By leveraging the Poisson-randomized gamma distribution [27], we introduce another type of time-varying transition kernels, which also model the interactions among components like Dir-Gam-Dir construction but may further encode a unique inductive bias tailored to sparsity and burstiness. The Poisson-randomized-gamma-Dirichlet (PR-Gam-Dir) Markov chain can be formulated as

$$
\pmb { \pi } _ { k } ^ { ( i ) } \sim \mathrm { D i r } \left( \alpha _ { 1 k } ^ { ( i ) } , \cdot \cdot \cdot , \alpha _ { K k } ^ { ( i ) } \right) ,\tag{5}
$$

$$
\begin{array} { r } { \alpha _ { k _ { 1 } k } ^ { ( i ) } \sim \mathrm { R G 1 } \left( \epsilon ^ { \alpha } , \gamma _ { k } ^ { ( i - 1 ) } \sum _ { k 2 = 1 } ^ { K } \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } \pi _ { k _ { 2 } k } ^ { ( i - 1 ) } , c _ { k } ^ { ( i ) } \right) , } \end{array}
$$

where RG1 (·) denotes the randomized gamma distribution of the first type. Similarly, for $\psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } , \gamma _ { k } ^ { ( i ) }$ , and $c _ { k } ^ { ( i ) }$ , the priors are given by

$$
\begin{array} { r l } & { \left( \psi _ { k 1 k _ { 2 } } ^ { ( i - 1 ) } , \cdot \cdot \cdot , \psi _ { k K k _ { 2 } } ^ { ( i - 1 ) } \right) \sim \mathrm { D i r } \left( \epsilon _ { 0 } , \cdot \cdot \cdot , \epsilon _ { 0 } \right) , } \\ & { \qquad \gamma _ { k } ^ { ( i ) } , c _ { k } ^ { ( i ) } \sim \mathrm { G a m } \left( \epsilon _ { 0 } , \epsilon _ { 0 } \right) , \mathrm { r e s p e c t i v e l y . } } \end{array}\tag{6}
$$

The PR-Gam-Dir Markov chain in $\operatorname { E q . } ( 5 )$ can be equivalently written as

$$
\begin{array} { r l } & { \pi _ { k } ^ { ( i ) } \sim \mathrm { D i r } \left( \alpha _ { 1 k } ^ { ( i ) } , \cdots , \alpha _ { K k } ^ { ( i ) } \right) , } \\ & { \alpha _ { k _ { 1 } k } ^ { ( i ) } \sim \mathrm { G a m } \left( g _ { k _ { 1 } k } ^ { ( i ) } + \epsilon ^ { \alpha } , c _ { k } ^ { ( i ) } \right) , } \\ & { g _ { k _ { 1 } k } ^ { ( i ) } \sim \mathrm { P o i s } \left( \gamma _ { k } ^ { ( i - 1 ) } \sum _ { k 2 = 1 } ^ { K } \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } \pi _ { k _ { 2 } k } ^ { ( i - 1 ) } \right) . } \end{array}\tag{7}
$$

When we marginalize out $g _ { k _ { 1 } k } ^ { ( i ) }$ in Eq.(7), we then obtain $\begin{array} { r } { \alpha _ { k _ { 1 } k } ^ { ( i ) } \sim \mathrm { R G 1 } \left( \epsilon ^ { \alpha } , \gamma _ { k } ^ { ( i - 1 ) } \sum _ { k 2 = 1 } ^ { K } \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } \pi _ { k _ { 2 } k } ^ { ( i - 1 ) } , c _ { k } ^ { ( i ) } \right) } \end{array}$ in $\operatorname { E q . } ( 5 )$ Compared with gamma distribution, RG1 enables a deeper decomposition of complex dynamics [16]. The Dir-Dir and Dir-Gam-Dir Markov chains were restricted by similar patterns between transitional matrices. The Poisson latent variable $\mathbf { \pmb { g } } _ { k } ^ { ( i ) }$ in the PR-Gam-Dir Markov chain decouples the component ratios, further enhancing the model’s representation capability. Thus, this modification may allow for large mutations in latent distributions and accommodate to burstiness.

The diagrams of three Dirichlet Markov constructions are shown in Figure 3.

## IV. MARKOV CHAIN MONTE CARLO INFERENCE

In this section, we present the Gibbs sampler for the proposed TV-PGDS. We only illustrate the key points of the derivation and the details can be found in the appendix. In the following inference, we repeatedly used two lemmas.

Lemma 1. If $\begin{array} { r l r } { \boldsymbol { y } } & { { } \sim } & { \mathrm { N B } \left( a , \boldsymbol { g } \left( \zeta \right) \right) } \end{array}$ and $l \ \sim \ \mathrm { C R T } \left( y , a \right)$ where NB (·) refers to negative-binomial distribution, CRT (·) represents Chinese restaurant table distribution [28], and $g \left( z \right) = 1 - \exp \left( - z \right)$ . Then the joint distribution of y and l can be equivalently distributed as $y \sim \operatorname { S u m L o g } \left( l , g \left( \zeta \right) \right)$ and $l \sim \mathrm { P o i s } \left( a \zeta \right) [ I I I ] , i . e .$

$$
\begin{array} { r l } & { \mathrm { N B } \left( y ; a , g \left( \zeta \right) \right) \mathrm { C R T } \left( l ; y , a \right) = } \\ & { \mathrm { S u m L o g } \left( y ; l , g \left( \zeta \right) \right) \mathrm { P o i s } \left( l ; a \zeta \right) , } \end{array}
$$

where SumLog $\begin{array} { r } { ( l , g \left( \zeta \right) ) = \sum _ { i = 1 } ^ { l } x _ { i } } \end{array}$ and $x _ { i } \sim \operatorname { L o g } \left( g \left( \zeta \right) \right)$ are independently and identically logarithmic distributed random variables [29].

Lemma 2. Suppose $\mathbf { n } = \left( n _ { 1 } , \cdots , n _ { K } \right)$ and

$$
\mathrm { \bf ~ n } \mid n \sim \mathrm { D i r M u l t } \left( n , r _ { 1 } , \cdots , r _ { K } \right) ,
$$

where DirMult (·) refers to Dirichlet-multimonial distribution. We sample the augmented variable $q \mid n \sim \operatorname { B e t a } \left( n , r . \right)$ , where $\begin{array} { r } { r . = \dot { \sum _ { k = 1 } ^ { K } } r _ { k } . } \end{array}$ . According to [30], conditioning on q, we have $n _ { k } \sim \mathrm { N B } \left( r _ { k } , q \right)$

Sampling $y _ { v k } ^ { ( t ) } :$ Use the relationship between Poisson and multinomial distributions, we sample

$$
\left( \left( \boldsymbol { y } _ { v k } ^ { ( t ) } \right) _ { k = 1 } ^ { K } | - \right) \sim \mathrm { M u l t } \left( \boldsymbol { y } _ { v } ^ { ( t ) } , \left( \frac { \phi _ { v k } \theta _ { k } ^ { ( t ) } } { \sum _ { k = 1 } ^ { K } \phi _ { v k } \theta _ { k } ^ { ( t ) } } \right) _ { k = 1 } ^ { K } \right) .\tag{8}
$$

Sampling $\phi _ { k }$ : Via Dirichlet-multinomial conjugacy, the posterior of $\phi _ { k }$ is

$$
\begin{array} { r } { ( \phi _ { k } \mid - ) \sim \operatorname { D i r } \left( \epsilon _ { 0 } + \sum _ { t = 1 } ^ { T } y _ { 1 k } ^ { ( t ) } , \cdot \cdot \cdot , \epsilon _ { 0 } + \sum _ { t = 1 } ^ { T } y _ { V k } ^ { ( t ) } \right) . } \end{array}\tag{9}
$$

Sampling $\theta _ { k } ^ { ( t ) } :$ To sample from the posterior of $\theta _ { k } ^ { ( t ) }$ , we first sample the auxiliary variables. The auxiliary variables $l _ { k \cdot } ^ { ( t ) }$ are used to decouple $\theta _ { k } ^ { ( t ) }$ from the Markov chain. The backward decoupling process is detailed in the appendix. Setting $l _ { \cdot k } ^ { ( T + 1 ) } = 0$ and $\zeta ^ { ( T + 1 ) } = 0$ , we sample the augmented variables backwards from $t = T , \cdots , 2$

$$
\begin{array} { r l } & { \left( l _ { k \cdot } ^ { ( t ) } \mid - \right) \sim \mathrm { C R T } \left( y _ { \cdot k } ^ { ( t ) } + l _ { \cdot k } ^ { ( t + 1 ) } , \tau _ { 0 } \sum _ { k _ { 2 } = 1 } ^ { K } \pi _ { k k _ { 2 } } ^ { i ( t - 1 ) } \theta _ { k _ { 2 } } ^ { ( t - 1 ) } \right) , } \\ & { \left( l _ { k 1 } ^ { ( t ) } , \cdots , l _ { k K } ^ { ( t ) } \mid - \right) \sim } \\ & { \mathrm { M u l t } \left( l _ { k \cdot } ^ { ( t ) } , \left( \frac { \pi _ { k 1 } ^ { i ( t - 1 ) } \theta _ { 1 } ^ { ( t - 1 ) } } { \sum _ { k _ { 2 } = 1 } ^ { K } \pi _ { k k _ { 2 } } ^ { i ( t - 1 ) } \theta _ { k _ { 2 } } ^ { ( t - 1 ) } } , \cdots , \frac { \pi _ { k K } ^ { i ( t - 1 ) } \theta _ { K } ^ { ( t - 1 ) } } { \sum _ { k _ { 2 } = 1 } ^ { K } \pi _ { k k _ { 2 } } ^ { i ( t - 1 ) } \theta _ { k _ { 2 } } ^ { ( t - 1 ) } } \right) \right) . } \end{array}\tag{10}
$$

Let us define $\begin{array} { r } { \boldsymbol { l } _ { \cdot k } ^ { ( t ) } = \sum _ { k _ { 1 } = 1 } ^ { K } \boldsymbol { l } _ { k _ { 1 } k } ^ { ( t ) } } \end{array}$ and $\begin{array} { r } { \zeta ^ { ( t ) } = \ln ( 1 + \frac { \delta ^ { ( t ) } } { \tau _ { 0 } } + } \end{array}$ $\zeta ^ { ( t + 1 ) } )$ . Using Lemma $I , l _ { k } ^ { ( t ) }$ can also be written in

$$
l _ { \cdot k } ^ { ( t ) } \sim \operatorname { P o i s } \left( \zeta ^ { ( t ) } \tau _ { 0 } \theta _ { k } ^ { ( t - 1 ) } \right) .\tag{11}
$$

After sampling the auxiliary variables, for $t = 1 , \cdots , T$ , by Poisson-gamma conjugacy, we obtain

$$
\begin{array} { r } { \left( \theta _ { k } ^ { ( t ) } \mid - \right) \sim \mathrm { G a m } ( y _ { \cdot k } ^ { ( t ) } + l _ { \cdot k } ^ { ( t + 1 ) } + \tau _ { 0 } \sum _ { k _ { 2 } = 1 } ^ { K } \pi _ { k k _ { 2 } } ^ { i ( t - 1 ) } \theta _ { k _ { 2 } } ^ { ( t - 1 ) } , } \\ { \tau _ { 0 } + \delta ^ { ( t ) } + \zeta ^ { ( t + 1 ) } \tau _ { 0 } ) . \qquad } \end{array}\tag{12}
$$

Sampling $\boldsymbol { \Pi } ^ { ( i ) }$ : We only illustrate Gibbs sampling algorithm for PR-Gam-Dir construction. Sampling algorithms for other constructions can be found in the appendix. The probabilistic graphical model of PR-GM-Dir construction is shown in Figure 4. We define M denote the length of each sub-interval and I be the total number of intervals. For $i \in \{ 2 , \ldots , I \}$ and $k _ { 1 } \in \{ 1 , \ldots , K \} , \ l _ { k _ { 1 } k } ^ { ( i ) }$ and $g _ { \cdot k _ { 1 } k } ^ { ( i + 1 ) }$ are characterized by Poisson distributions, as formulated in $\operatorname { E q . } ( 7 )$ and Eq.(11). Here, $\begin{array} { r } { l _ { k _ { 1 } k } ^ { ( i ) } ~ = ~ \sum _ { ( i - 1 ) M + 1 } ^ { i M } l _ { k _ { 1 } k } ^ { ( t ) } } \end{array}$ refers to the summation of $l _ { k _ { 1 } k } ^ { ( t ) }$ over i-th sub-interval; analogous notation applies to other variables. By exploiting the conjugacy between Poisson and multinomial distributions, the vectors $( l _ { 1 k } ^ { ( i ) } , \cdots , l _ { K k } ^ { ( i ) } )$ and $( g _ { \cdot 1 k } ^ { ( i + 1 ) } , \cdot \cdot \cdot , g _ { \cdot K k } ^ { ( i + 1 ) } )$ are multinomially distributed with parameter $( \pi _ { 1 k } ^ { ( i ) } , \cdot \cdot \cdot , \pi _ { K k } ^ { ( i ) } ) .$ as in Figure 4. Upon marginalizing out $\pi _ { k } ^ { \left( i \right) } , \ l _ { \cdot k } ^ { \left( i \right) }$ and $g _ { \cdot k } ^ { ( i + 1 ) }$ follow Dirichlet-Multinomial distributions.

Consequently, we construct an auxiliary variable $q _ { k } ^ { ( i ) }$ via Lemma 2, defining $g _ { k _ { 1 } k } ^ { ( I + 1 ) } = 0$ , as

$$
( q _ { k } ^ { ( i ) } \mid - ) \sim \mathrm { B e t a } ( l _ { \cdot k } ^ { ( i ) } + g _ { \cdot k } ^ { ( i + 1 ) } , \alpha _ { \cdot k } ^ { ( i ) } ) .\tag{13}
$$

Then we have $( l _ { k _ { 1 } k } ^ { ( i ) } + g _ { \cdot k _ { 1 } k } ^ { ( i + 1 ) } ) \sim \mathrm { N B } ( \alpha _ { k _ { 1 } k } ^ { ( i ) } , q _ { k } ^ { ( i ) } )$ . Furthermore, by applying the NB-CRT augmentation (Lemma 1), we sample another auxiliary variable $\breve { h _ { k _ { 1 } k } ^ { ( i ) } }$ as

$$
( h _ { k _ { 1 } k } ^ { ( i ) } \mid - ) \sim \mathrm { C R T } ( l _ { k _ { 1 } k } ^ { ( i ) } + g _ { \cdot k _ { 1 } k } ^ { ( i + 1 ) } , \alpha _ { k _ { 1 } k } ^ { ( i ) } ) ,\tag{14}
$$

which, according to the properties of the CRT distribution, yields $h _ { k _ { 1 } k } ^ { ( i ) } \sim \bar { \mathrm { P o i s } } ( - \alpha _ { k _ { 1 } k } ^ { ( i ) } \bar { \ln ( 1 - q _ { k } ^ { ( i ) } ) } )$ . By Poisson-gamma conjugacy, we have $( \alpha _ { k _ { 1 } k } ^ { ( i ) ^ { \ast } \cdot \cdot \cdot } | - ) \sim \mathrm { G a m } ( g _ { k _ { 1 } k } ^ { ( i ) } + \epsilon ^ { \alpha } + h _ { k _ { 1 } k } ^ { ( i ) } , c _ { k } ^ { ( i ) } -$ $\ln ( 1 - q _ { k } ^ { ( i ) } ) )$

We define $\begin{array} { r } { \lambda _ { k _ { 1 } k } ^ { ( i - 1 ) } \triangleq \gamma _ { k } ^ { ( i - 1 ) } \sum _ { k 2 = 1 } ^ { K } \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } \pi _ { k _ { 2 } k } ^ { ( i - 1 ) } } \end{array}$ for notation conciseness. When sampling $g _ { k _ { 1 } k } ^ { ( i ) } .$ the posterior distribution—arising from the Poisson prior and the gamma likelihood—exhibits a closed-form analytical solution via the Bessel distribution [27]. Specifically, if $\epsilon ^ { \alpha } > 0$ , then we can sample the posterior of $g _ { k _ { 1 } k } ^ { ( i ) }$ via $( \dot { g } _ { k _ { 1 } k } ^ { ( i ) } \ | \ - ) \sim$ Bessel $( \epsilon ^ { \alpha } -$ $1 , 2 \sqrt { \alpha _ { k _ { 1 } k } ^ { ( i ) } c _ { k } ^ { ( i ) } \lambda _ { k _ { 1 } k } ^ { ( i - 1 ) } } )$ , where Bessel (·) denotes Bessel distribution. If $\epsilon ^ { \alpha } = 0 .$ , to avoid the absorbing condition between $\alpha _ { k _ { 1 } k } ^ { ( i ) }$ and $g _ { k _ { 1 } k } ^ { ( i ) } .$ , we sample $g _ { k _ { 1 } k } ^ { ( i ) }$ via

$$
\begin{array} { r } { ( g _ { k _ { 1 } k } ^ { ( i ) } \mid - ) \sim \left\{ \begin{array} { l l } { \mathrm { P o i s } \left( \frac { c _ { k } ^ { ( i ) } \lambda _ { k _ { 1 } k } ^ { ( i - 1 ) } } { c _ { k } ^ { ( i ) } - \ln ( 1 - q _ { k } ^ { ( i ) } ) } \right) } & { \mathrm { i f } \ h _ { k _ { 1 } k } ^ { ( i ) } = 0 } \\ { \mathrm { S C H } \left( h _ { k _ { 1 } k } ^ { ( i ) } , \frac { c _ { k } ^ { ( i ) } \lambda _ { k _ { 1 } k } ^ { ( i - 1 ) } } { c _ { k } ^ { ( i ) } - \ln ( 1 - q _ { k } ^ { ( i ) } ) } \right) } & { \mathrm { o t h e r w i s e , } } \end{array} \right. } \end{array}\tag{15}
$$

where SCH (·) denotes the shifted confluent hypergeometric distribution [16]. Both Bessel and the shifted confluent hypergeometric distribution can be sampled efficiently [16], [31]. Defining $\begin{array} { r } { g _ { k _ { 1 } k } ^ { ( i ) } = g _ { k _ { 1 } \cdot k } ^ { ( i ) } = \sum _ { k 2 = 1 } ^ { K } g _ { k _ { 1 } k _ { 2 } k } ^ { ( \ i ) } , } \end{array}$ , we first augment

$$
\begin{array} { r } { \left( g _ { k _ { 1 } 1 k } ^ { ( i ) } , \cdots , g _ { k _ { 1 } K k } ^ { ( i ) } \right) \sim \mathrm { M u l t } \Big ( g _ { k _ { 1 } k } ^ { ( i ) } , \big ( \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } \pi _ { k _ { 2 } k } ^ { ( i - 1 ) } \big ) _ { k _ { 2 } = 1 } ^ { K } \Big ) . } \end{array}\tag{16}
$$

Then we obtain $g _ { k _ { 1 } k _ { 2 } k } ^ { ( i ) } \ \sim \ \mathrm { P o i s } ( \gamma ^ { ( i - 1 ) } \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } \pi _ { k _ { 2 } k } ^ { ( i - 1 ) } )$ . By Dirichlet-multinomial conjugacy, we have

$$
\Big ( \big ( \psi _ { k 1 k _ { 2 } } ^ { ( i - 1 ) } , \dots , \psi _ { k K k _ { k } } ^ { ( i - 1 ) } \big ) \ | \ - \Big ) \sim \mathrm { D i r } \Big ( \epsilon _ { 0 } + g _ { 1 k _ { 2 } k } ^ { ( i ) } , \dots ,
$$

$$
\epsilon _ { 0 } + g _ { K k _ { 2 } k } ^ { ( i ) } \bigg ) ,\tag{17}
$$

$$
\begin{array} { r } { \left( \pi _ { k } ^ { \left( i - 1 \right) } \mid - \right) \sim \operatorname { D i r } \left( \alpha _ { 1 k } ^ { \left( i - 1 \right) } + l _ { 1 k } ^ { \left( i - 1 \right) } + g _ { \cdot 1 k } ^ { \left( i \right) } , \cdot \cdot \cdot , \right. } \\ { \left. \alpha _ { K k } ^ { \left( i - 1 \right) } + l _ { K k } ^ { \left( i - 1 \right) } + g _ { \cdot K k } ^ { \left( i \right) } \right) . \qquad } \end{array}\tag{18}
$$

Specifically, we have $\alpha _ { k _ { 1 } k } ^ { ( 1 ) } = \nu _ { k _ { 1 } } \nu _ { k }$ , if $\boldsymbol { k } _ { 1 } \neq \boldsymbol { k }$ , and $\alpha _ { k _ { 1 } k } ^ { ( 1 ) } =$ $\xi \nu _ { k } , { \mathrm { i f } } \ k _ { 1 } = k$

## V. EXPERIMENTS

We conducted experiments for both predictive and exploratory analysis to demonstrate the ability of the proposed model in capturing count time sequences with timevarying dynamics. The baseline models included in the experiments are: 1) Gamma process dynamic Poisson factor analysis (GP-DPFA) [15]. 2) Poisson-gamma dynamical system (PGDS) [13]. 3) Gamma Markov chains on the rate parameter of gamma distribution (GMC-RATE) [25]. 4) Gamma Markov chains on the rate parameter with hierarchical auxiliary variable (GMC-HIER) [26]. 5) Autogressive beta-gamma procecss (BGAR) [32]. BGAR is also a gamma Markov model. We included it as one of the baseline models since a review [26] highlighted its superiority in well-defined stationary distribution compared with the above-mentioned models.

![](images/68310f18e6396e1112f3b2a571d6d020e949494e1d2e2a56ba1dee38e7895c83.jpg)  
Fig. 4. The probabilistic graphical model of the proposed PR-Gam-Dir construction as Eq.(7). Dashed nodes and edges denote the augmented auxiliary variables.

The real-world datasets used in the experiments are: 1) Integrated Crisis Early Warning System (ICEWS) [13]: ICEWS is an international relations event dataset, comprising interaction events between countries extracted from news corpora. For ICEWS dataset, we have $T = 3 6 5$ time steps and $V \ = \ 6 1 9 7 . \ 2 )$ NIPS [33]: NIPS dataset contains the papers published in the NeurIPS conference from 1987 to 2015. We have $T \ = \ 2 8$ time steps and $V ~ = ~ 2 0 0 0 . ~ 3 )$ U.S. Earthquake Intensity (USEI) [34]: USEI contains a collection of damage and felt reports for U.S. (and a few other countries) earthquakes. We use the monthly reports from 1957-1986 and have $T = 3 4 8 , V = 6 4 . 4 )$ COVID-19 [35]: This dataset contains daily death cases data for states in the United States, spanning from March 2020 to June 2020. For this dataset, we have $V = 5 1$ and $T = 9 0$ time steps.

Predictive Analysis. To compare the predictive performance of the proposed model with the baselines, we considered two standard tasks: data smoothing and forecasting. For data smoothing task, our objective is to predict $\mathbf { \boldsymbol { y } } ^ { ( t ) }$ given the remaining data observation $\mathbf { \Delta } _ { \mathbf { Y \backslash \boldsymbol { y } ^ { ( t ) } } }$ . To this end, we randomly masked 10 percents of the observed data over non-adjacent time steps, and predicted the masked values. For forecasting task, we held out data of the last S time steps, and predicted $\pmb { y } ^ { ( T + 1 ) } , \cdot \cdot \cdot , \pmb { y } ^ { ( T + S ) }$ given $\pmb { y } ^ { ( 1 ) } , \cdots , \pmb { y } ^ { ( T ) }$ . In this experiment we set $S \ = \ 2$ . We ran the baseline models including GP-DPFA, PGDS, GMC-RATE, GMC-HIER, BGAR, using their default settings as provided in [13], [15], [26]. For the TV-PGDS, we set $\tau _ { 0 } ~ = ~ 1 , \gamma _ { 0 } ~ = ~ 5 0 , \epsilon _ { 0 } ~ = ~ 0 . 1$ . To maintain experimental consistency and ensure comparison, we fixed the number of sub-intervals at $I \ = \ 6$ for all datasets, with M determined accordingly. Additionally, we set $K \ : = \ : 1 0 0$ for the ICEWS dataset to capture its higher dimensionality, while $K \ : = \ : 1 0$ was applied to others. These configurations were kept strictly uniform across all baseline models to eliminate any potential bias. While these empirical settings demonstrate robust performance, we further provide a sensitivity analysis of M (or I) in the following section to evaluate their influence on model behavior.

TABLE I  
RESULTS OF PREDICTIVE ANALYSIS. ”S” MEANS DATA SMOOTHING AND ”F” MEANS DATA FORECASTING.
<table><tr><td></td><td></td><td></td><td>GP-DPFA</td><td>GMC-RATE</td><td>GMC-HIER</td><td></td><td>BGAR</td><td>PGDS</td><td>TV-PGDS (Dir-Dir)</td><td>TV-PGDS (Dir-Gam-Dir)</td><td></td><td>TV-PGDS (PR-Gam-Dir)</td></tr><tr><td>ICEWS</td><td>MAE</td><td>S F</td><td>0.259 ±0.005</td><td>0.258 ±0.005</td><td>0.256 ±0.006</td><td></td><td>0.264 ±0.006</td><td>0.215 ±0.007</td><td>0.215 ±0.008</td><td>0.214 ±0.008</td><td></td><td>0.215 ±0.008</td></tr><tr><td></td><td></td><td></td><td>0.176 ±0.005</td><td>0.187 ±0.003</td><td>0.185 ±0.016</td><td>0.222 ±0.043</td><td></td><td>0.185 ±0.003</td><td>0.167 ±0.009</td><td>0.169 ±0.006</td><td></td><td>0.169 ±0.009</td></tr><tr><td></td><td>MRE S</td><td></td><td>0.125 ±0.003</td><td>0.124 ±0.002</td><td>0.122 ±0.003</td><td>0.130 ±0.004</td><td></td><td>0.102 ±0.005</td><td>0.101 ±0.005</td><td>0.101 ±0.005</td><td>0.102</td><td>±0.005</td></tr><tr><td></td><td>F</td><td></td><td>0.099 ±0.006</td><td>0.114 ±0.003</td><td>0.111 ±0.018</td><td>0.142 ±0.036</td><td></td><td>0.108 ±0.001</td><td>0.094 ±0.005</td><td>0.097 ±0.004</td><td>0.097</td><td>±0.008</td></tr><tr><td>NIPS</td><td>MAE S</td><td></td><td>18.299 ±6.545</td><td>17.105 ±6.449</td><td>17.098 ±6.441</td><td>17.935 ±6.450</td><td></td><td>14.706 ±4.414</td><td>14.032 ±4.401</td><td>14.026 ±4.405</td><td></td><td>14.014 ±4.387</td></tr><tr><td></td><td>F</td><td></td><td>48.355 ±1.461</td><td>46.234 ±1.629</td><td>102.506 ±39.932</td><td>62.449 ±14.463</td><td></td><td>51.562 ±0.679</td><td>45.979 ±1.342</td><td>46.710 ±1.152</td><td></td><td>46.582 ±1.196</td></tr><tr><td></td><td>MRE S</td><td></td><td>0.729 ±0.412</td><td>0.684 ±0.316</td><td>0.664 ±0.315</td><td>0.769 ±0.366</td><td></td><td>0.590 ±0.097</td><td>0.581 ±0.090</td><td>0.581 ±0.090</td><td></td><td>0.580 ±0.090</td></tr><tr><td></td><td>F</td><td></td><td>0.415 ±0.016</td><td>0.387 ±0.023</td><td>0.580 ±0.148</td><td>0.465 ±0.049</td><td></td><td>0.459 ±0.006</td><td>0.399 ±0.003</td><td>0.395</td><td>±0.006 0.397</td><td>±0.003</td></tr><tr><td>USEI MAE</td><td>S</td><td></td><td>4.681 ±0.564</td><td>4.931 ±0.872</td><td>4.748 ±0.829</td><td>5.244 ±0.939</td><td></td><td>4.703 ±0.538</td><td>4.399 ±0.540</td><td>4.318 ±0.533</td><td></td><td>4.219 ±0.537</td></tr><tr><td></td><td>F</td><td></td><td>11.665 ±0.367</td><td>9.454 ±0.809</td><td>12.423 ±1.060</td><td>21.948 ±0.133</td><td></td><td>11.118 ±0.220</td><td>7.637 ±1.225</td><td>7.193 ±1.205</td><td></td><td>7.164 ±1.121</td></tr><tr><td>MRE</td><td>S</td><td></td><td>1.458 ±0.177</td><td>1.128 ±0.189</td><td>1.088 ±0.162</td><td>1.941 ±0.209</td><td></td><td>1.279 ±0.257</td><td>1.201 ±0.214</td><td>1.177 ±0.221</td><td></td><td>1.031 ±0.193</td></tr><tr><td></td><td>F</td><td></td><td>7.473 ±0.623</td><td>6.508 ±0.571</td><td>8.929 ±2.514</td><td>13.706 ±1.268</td><td></td><td>4.238 ±0.325</td><td>2.591 ±0.355</td><td>2.565 ±0.333</td><td></td><td>2.583 ±0.353</td></tr><tr><td>COVID-19</td><td>MAE S</td><td></td><td>7.935 ±0.751</td><td>7.144 ±1.159</td><td>7.240 ±0.848</td><td>7.819 ±1.348</td><td></td><td>7.566 ±1.095</td><td>6.599 ±1.101</td><td>6.427 ±1.043</td><td></td><td>6.539 ±1.012</td></tr><tr><td></td><td>F</td><td></td><td>9.137 ±1.102</td><td>9.600 ±1.257</td><td>10.409 ±1.910</td><td>12.550 ±2.156</td><td></td><td>9.314 ±0.236</td><td>8.696 ±0.426</td><td>8.572 ±0.435</td><td></td><td>8.635 ±0.447</td></tr><tr><td>MRE</td><td>S</td><td>0.564 ±0.126</td><td></td><td>0.493 ±0.136</td><td>0.504 ±0.109</td><td>0.769 ±0.169</td><td></td><td>0.558 ±0.130</td><td>0.346 ±0.105</td><td>0.335</td><td>±0.104 0.342</td><td>2 ±0.113</td></tr><tr><td></td><td>F</td><td>0.627 ±0.106</td><td></td><td>0.556 ±0.052</td><td>0.585 ±0.067</td><td>0.759 ±0.150</td><td></td><td>0.585 ±0.007</td><td>0.520 ±0.019</td><td>0.507 ±0.017</td><td></td><td>0.515 ±0.015</td></tr></table>

We performed 4000 Gibbs sampling iterations. In the experiments, we found the Gibbs sampler started to converge after 1000 iterations, and thus we set the burn-in time be 2000 iterations. We retained every hundredth sample, and averaged the predictions over the samples. Mean relative error (MRE) and mean absolute error (MAE) are adopted to evaluate the model’s predictive capability, which are defined as MRE = $\begin{array} { r } { \frac { 1 } { T V } \sum _ { t } \sum _ { v } ^ { } \frac { | y _ { v } ^ { ( t ) } - \hat { y } _ { v } ^ { ( t ) } | } { 1 + y _ { v } ^ { ( t ) } } } \end{array}$ and $\begin{array} { r } { \mathrm { M A E } = \frac { 1 } { T V } \sum _ { t } \sum _ { v } \mid y _ { v } ^ { ( t ) } - \hat { y } _ { v } ^ { ( t ) } \mid } \end{array}$ where $y _ { v } ^ { ( t ) }$ indicates the true count and $\hat { y } _ { v } ^ { ( t ) }$ is the prediction.

As the experiment results shown in Table I, the TV-PGDS exhibits improved performance in both data smoothing and forecasting tasks. We attribute this enhanced capability to the time-varying transition kernels, which effectively adapt to the non-stationary environment, and thus achieve improved predictive performance. For some datasets (e.g. ICEWS) and tasks, the effectiveness of the Dir-Gam-Dir and PR-Gam-Dir constructions is not evident in the numerical results. This indicates that the selection of these three Dirichlet Markov constructions should depend on different data features. However, the Dir-Gam-Dir and PR-Gam-Dir constructions indeed induce more informative patterns compared with Dir-Dir construction, as shown in the exploratory analysis.

Exploratory Analysis. We used ICEWS and NIPS datasets for exploratory analysis, and chose the TV-PGDS with Dirichlet-Dirichlet Markov chains for illustration. Figure 5(a) and Figure 5(b) demonstrate the top 2 latent factors inferred by TV-PGDS from ICEWS dataset. From Figure 5(a) we can see that the main labels are “Iraq (IRQ)–United States (USA)”, “Iraq (IRQ)–United Kingdom (UK)”, “Russia (RUS)–United States (USA)”, and so on. This latent factor probably corresponds to the topic about Iraq war. Besides, in Figure 5(a), there is a peak around March, 2003, and we know that the Iraq war broke out exactly on 20 March, 2003. In addition, the most dominant labels shown in Figure 5(b) are “Japan (JPN)–United States (USA)”, “China (CHN)–United States (USA)”, “North Korea (PRK)–United States (USA)”, “South Korea (KOR)– United States (USA)”, and so on. We can infer that this latent factor corresponds to “Six-Party Talks” and other accidents about it.

Figure 5(c) demonstrates the evolving trends of the top 5 latent factors inferred by the TV-PGDS from NIPS dataset, and the legend indicates the representative words of the corresponding latent factors. Clearly, the green and blue lines correspond to the latent factors of neural network research which started to decline from the 1990s. From the 1990s, the latent factors about statistical and probabilistic methods began to dominate the NeurIPS conference. In addition, the TV-PGDS also captured the revival of neural networks (blue line) from the 2010s. The above observations from the latent structure inferred by the TV-PGDS match our prior knowledge.

Next, we explored the time-varying transition matrices inferred by the TV-PGDS. We chose NIPS dataset for illustratiuon, and set K = 10 and the interval length M to be 5. The time-varying transition matrices are shown from Figure 6(b) to Figure 6(f). At the beginning, matrices shown in Figure 6(b) and Figure 6(c) are close to identity matrices. Then the transition matrices tend to become block diagonal matrices with 2 blocks, as shown in Figure 6(d)-6(f). The representative words for latent factors in the first block are “state-linear-classification”, “network-neural-networks”, “kernel-image-space”, “networkneural-networks”, “neural-networks-state”. The representative words for latent factors in the second block are “imagesparse-matrix”, “kernel-supervised-random”, “matrix-samplerandom”, “inference-prior-latent”, “state-policy-gamma”. The first block primarily captured the correlations among the research topics about neural networks. The second block reflects that, from the 1990s, statistical learning and Bayesian methods began to dominate, and these topics are highly correlated.

![](images/d20264d58035c1c6c61371cdbfe9c0c1fb75404d56b01fb096c5d4f71389af21.jpg)

![](images/3559a85f0a519609595b9f29b0dfea004d7739b4932b8dbcb2e2c500418190ae.jpg)

![](images/fcad52f571412f31128cccc6516a373352d02d26feae7c5653a9af26ebe91acf.jpg)  
(a)

![](images/66825e38f71e888f2281810649a03c842468f4cfda3c7859480064ea3e52873e.jpg)  
(b)

![](images/04dbfe8756ca1021e7516f989d1b56827155260a8517cc4b833ef6f9666a0b70.jpg)  
(c)  
Fig. 5. The latent factors inferred by the TV-PGDS. (a) and (b) illustrate the top 2 latent factors inferred from ICEWS dataset, (a) corresponds to Iraq war and (b) corresponds to the Six-Party Talks. (c) illustrates the evolving trends of the top 5 latent factors inferred from NIPS dataset.

![](images/2574f70ab74a10a4caa60fb48361a186f6f105976ab09e84f2278a6ca35594cf.jpg)

![](images/7fb45aa70d79bdbc6979544e53592d8a3bc3d22c97f3ac003ec244d7214ec782.jpg)

Fig. 6. Transition matrices inferred from NIPS dataset. (a) The transition matrix inferred by the PGDS. (b)-(f) The time-varying transition matrices inferred by the TV-PGDS.  
![](images/60dadcdcbf0e8460aee1d161e10de76086f6dddbe76b25cf12e2919047e4ec04.jpg)  
Fig. 7. From top to bottom are the first four transition matrices inferred by different Dirichlet Markov chains from ICEWS dataset. Top row: Matrices inferred by the Dir-Dir construction. Middle row: Matrices inferred by the Dir-Gam-Dir construction. Bottom row: Matrices inferred by the PR-Gam-Dir construction.

Figure 6(a) illustrates the transition matrix inferred by the PGDS, which is averaged over all time steps. Compared with the TV-PGDS, the PGDS can not capture the informative timevarying transition dynamics.

We also compared the theoretical and empirical properties of the three proposed Dirichlet Markov chain variants. The top row of Figure 7 demonstrates transition matrices of the first four sub-intervals of ICEWS dataset inferred by the TV-PGDS (Dir-Dir). Because of the Dir-Dir construction, the consecutive transition matrices smoothly change over time and thus the TV-PGDS may lack sufficient flexibility to capture rapid dynamics. The middle row of Figure 7 illustrates the transition matrices inferred by the TV-PGDS (Dir-Gam-Dir), which takes mutations among latent components into account and captured more complicated patterns. Transition matrices inferred by the PR-Gam-Dir construction are shown in the bottom row of Figure 7, these matrices not only exhibited sufficient flexibility but also captured sparser patterns compared with the Dir-Gam-Dir construction.

Effect of Sub-interval Length. The length of each subinterval plays a critical role in the performance of the proposed model. Figure 8 illustrates, under various sub-interval lengths, the performance of TV-PGDS (Dir-Dir) varies across the ICEWS and COVID-19 datasets. Intuitively, allowing the transition matrix to change at every time step offers greater modeling flexibility compared to dividing the whole interval into fixed-length sub-intervals. A shorter sub-interval length enables the model to more closely track the temporal dynamics of the underlying process. However, shorter sub-intervals also result in fewer data points available for estimating the transition matrices within each segment, which increases estimation variance and may lead to degraded performance. Conversely, longer sub-intervals provide more data for reliable parameter estimation, thereby reducing variance. Nonetheless, overly long sub-intervals may hinder the model’s ability to capture fine-grained temporal variations, as the assumption of a fixed transition matrix over an extended period becomes less realistic. We believe that incorporating a change-point detection mechanism—which adaptively partitions the entire time horizon into irregular sub-intervals based on shifts in transition dynamics—would further enhance the model’s flexibility and accuracy. We leave this as our future work.

## VI. CONCLUSION

Poisson-gamma dynamical systems with time-varying transition matrices, have been proposed to capture time-varying dynamics observed in count sequences. Dirichlet Markov chains are constructed to allow the underlying transition matrices to evolve over time. Although the Dirichlet Markov processes lack conjugacy, we have developed tractable-butefficient Gibbs sampling algorithms to perform posterior simulation. For future work, we plan to design methods that can find the point of change, and thus the length of each subinterval can be determined automatically. We also consider to capture the time-varying dynamics occurring in temporal social networks [36]–[39] and irregularly-observed social event data [40]–[42].

![](images/e609cd9ac860cc02d684a725280cb9dab0cb8cce3c984c4582a076e1a718f1e3.jpg)  
Fig. 8. The mean absolute error (MAE) for smoothing and forecasting tasks with different length of sub-interval. Top: the MAE for ICEWS dataset. Down: the MAE for COVID-19 dataset.

## ACKNOWLEDGEMENTS

Sikun Yang is supported by National Natural Science Foundation of China(NSFC)(Grant No.62476047), Peking University Mathematics Challenge Funding Program(Grant No.2024SRMC10), the Guangdong-Dongguan Joint Research Fund(Grant No.2025A1515140098), and Dongguan Key Laboratory for AI and Dynamical Systems, Dongguan Key Laboratory for Intelligence and Information Technology, Dongguan Key Laboratory for Data Science and Intelligent Medicine, Guangdong Provincial Key Laboratory of Mathematical and Neural Dynamical Systems (Grant No.2024B1212010004), and Guangdong Multidisciplinary Innovation Research Group for New-Generation Intelligent Systems and Diagnostic-Therapeutic Applications(Grant No.2025KCXTD031).

## REFERENCES

[1] D. M. Blei and J. D. Lafferty, “Dynamic topic models,” in Proceedings of the 23rd International Conference on Machine Learning, 2006, pp. 113–120.

[2] X. Wang and A. McCallum, “Topics over time: a non-markov continuous-time model of topical trends,” in Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2006, pp. 424–433.

[3] P. Jahnichen, F. Wenzel, M. Kloft, and S. Mandt, “Scalable general-¨ ized dynamic topic models,” in International Conference on Artificial Intelligence and Statistics, 2018, pp. 1427–1435.

[4] D. Sheldon and T. G. Dietterich, “Collective graphical models,” in Proceedings ofthe 24th International Conference on Neural Information Processing Systems, 2011, pp. 1161–1169.

[5] J. Raymer, A. Wisniowski, J. J. Forster, P. W. Smith, and J. Bijak,´ “Integrated modeling of european migration,” Journal of the American Statistical Association, vol. 108, no. 503, pp. 801–819, 2013.

[6] T. Wilson, “Methods for estimating sub-state international migration: The case of australia,” Spatial Demography, vol. 5, no. 3, pp. 171–192, 2017.

[7] P. Wanner, “How well can we estimate immigration trends using google data?” Quality & Quantity, vol. 55, no. 4, pp. 1181–1202, 2021.

[8] R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,” Journal of Basic Engineering, vol. 82, no. 1, pp. 35–45, 1960.

[9] Z. Ghahramani and S. T. Roweis, “Learning nonlinear dynamical systems using an em algorithm,” in Proceedings of the 11th International Conference on Neural Information Processing Systems, 1998, pp. 431– 437.

[10] M. Zhou and L. Carin, “Augment-and-conquer negative binomial processes,” Advances in Neural Information Processing Systems, vol. 4, pp. 2546–2554, 2012.

[11] ——, “Negative binomial process count and mixture modeling,” IEEE Transactions on Pattern Analysis & Machine Intelligence, vol. 37, no. 02, pp. 307–320, 2015.

[12] A. Schein, J. Paisley, D. M. Blei, and H. Wallach, “Bayesian poisson tensor factorization for inferring multilateral relations from sparse dyadic event counts,” in Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2015, pp. 1045– 1054.

[13] A. Schein, M. Zhou, and H. Wallach, “Poisson-gamma dynamical systems,” in Proceedings ofthe 30th International Conference on Neural Information Processing Systems, 2016, pp. 5012–5020.

[14] A. Schein, M. Zhou, D. Blei, and H. Wallach, “Bayesian poisson tucker decomposition for learning the structure of international relations,” in International Conference on Machine Learning, 2016, pp. 2810–2819.

[15] A. Acharya, J. Ghosh, and M. Zhou, “Nonparametric bayesian factor analysis for dynamic count matrices,” in Artificial Intelligence and Statistics, 2015, pp. 1–9.

[16] A. Schein, S. W. Linderman, M. Zhou, D. M. Blei, and H. Wallach, “Poisson-randomized gamma dynamical systems,” in Proceedings of the 33rd International Conference on Neural Information Processing Systems, 2019, pp. 782–793.

[17] J. Chang and D. Blei, “Relational topic models for document networks,” in Proceedings of the Twelth International Conference on Artificial Intelligence and Statistics, 2009, pp. 81–88.

[18] R. Winkelmann, Econometric Analysis of Count Data, 5th ed. Springer Publishing Company, Incorporated, 2008.

[19] G. Grossman, S. Kim, J. M. Rexer, and H. Thirumurthy, “Political partisanship influences behavioral responses to governors’ recommendations for covid-19 prevention in the united states,” Proceedings ofthe National Academy of Sciences, vol. 117, no. 39, pp. 24 144–24 153, 2020.

[20] I. C.-. F. Team, “Modeling covid-19 scenarios for the united states,” Nature Medicine, vol. 27, no. 1, pp. 94–105, 2021.

[21] L. Feng, T. Zhang, Q. Wang, Y. Xie, Z. Peng, J. Zheng, Y. Qin, M. Zhang, S. Lai, D. Wang et al., “Impact of covid-19 outbreaks and interventions on influenza in china and the united states,” Nature Communications, vol. 12, no. 1, p. 3249, 2021.

[22] S. Han, L. Du, E. Salazar, and L. Carin, “Dynamic rank factor model for text streams,” in Proceedings of the 27th International Conference on Neural Information Processing Systems-Volume 2, 2014, pp. 2663–2671.

[23] R. Kalantari and M. Zhou, “Graph gamma process generalized linear dynamical systems,” arXiv preprint arXiv:2007.12852, 2020.

[24] W. Chen, B. Chen, Y. Liu, Q. Zhao, and M. Zhou, “Switching poisson gamma dynamical systems,” in Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence, 2021, pp. 2029–2036.

[25] C. Fevotte, N. Bertin, and J.-L. Durrieu, “Nonnegative matrix factor-´ ization with the itakura-saito divergence: With application to music analysis,” Neural computation, vol. 21, no. 3, pp. 793–830, 2009.

[26] L. Filstroff, O. Gouvert, C. Fevotte, and O. Capp´ e, “A comparative study´ of gamma markov chains for temporal non-negative matrix factorization,” IEEE Transactions on Signal Processing, vol. 69, pp. 1614–1626, 2021.

[27] L. Yuan and J. D. Kalbfleisch, “On the bessel distribution and related problems,” Annals of the Institute of Statistical Mathematics, vol. 52, pp. 438–447, 2000.

[28] Y. W. Teh, M. I. Jordan, M. J. Beal, and D. M. Blei, “Hierarchical dirichlet processes,” Journal ofthe American Statistical Association, vol. 101, no. 476, pp. 1566–1581, 2006.

[29] N. L. Johnson, A. W. Kemp, and S. Kotz, Univariate discrete distributions. John Wiley & Sons, 2005, vol. 444.

[30] M. Zhou, “Nonparametric bayesian negative binomial factor analysis,” Bayesian Analysis, vol. 13, no. 4, pp. 1065–1093, 2018.

[31] L. Devroye, “Simulating bessel random variables,” Statistics & probability letters, vol. 57, no. 3, pp. 249–257, 2002.

[32] P. A. Lewis, E. McKenzie, and D. K. Hugus, “Gamma processes,” Stochastic Models, vol. 5, no. 1, pp. 1–30, 1989.

[33] V. Perrone, “NIPS Conference Papers 1987-2015,” UCI Machine Learning Repository, 2016, DOI: https://doi.org/10.24432/C5KC80.

[34] United States Geological Survey (USGS), “U.S. Earthquake Intensity Database (1638–1985),” 2025, DOI: https://doi.org/10.25921/wa4zd240.

[35] Bing COVID-19 Tracker, “Bing covid-19 tracker,” https://www.bing.com/covid, 2024.

[36] S. Yang and H. Koeppl, “A Poisson gamma probabilistic model for latent node-group memberships in dynamic networks,” in Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), S. A. McIlraith and K. Q. Weinberger, Eds. AAAI Press, 2018, pp. 4366–4373.

[37] ——, “Dependent relational gamma process models for longitudinal networks,” in Proceedings of the 35th International Conference on Machine Learning, ICML, vol. 80, 2018, pp. 5547–5556.

[38] X. Yu, N. Fang, H. Liao, and S. Yang, “Tracking latent communities evolution with hierarchical edge partition models,” in IEEE International Conference on Data Mining, ICDM, 2025, pp. 1672–1681.

[39] N. Fang, Y. Wang, H. Liao, and S. Yang, “Poisson–gamma modeling of inter-relational dependencies in dynamic knowledge graphs,” in Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence, 2026, pp. 1520–1539.

[40] S. Yang and H. Koeppl, “The Hawkes edge partition model for continuous-time event-based temporal networks,” in Proceedings of the Thirty-Sixth Conference on Uncertainty in Artificial Intelligence, UAI, vol. 124, 2020, pp. 460–469.

[41] S. Yang and H. Zha, “Estimating latent population flows from aggregated data via inversing multi-marginal optimal transport,” in Proceedings of the SIAM International Conference on Data Mining, SDM, 2023, pp. 181–189.

[42] ——, “A variational autoencoder for neural temporal point processes with dynamic latent graphs,” in Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI, 2024, pp. 16 343–16 351.

## VII. APPENDIX A: MCMC INFERENCE

Negative Binomial Distribution Let $y \sim$ Pois (cλ), and $\lambda \sim$ Gam (a, b). If we marginalize over λ, then $\begin{array} { r } { y \sim \mathrm { N B } \left( a , \frac { c } { b + c } \right) } \end{array}$ We can further parameterize it as $y \sim \mathrm { N B } \left( a , g \left( \zeta \right) \right)$ , where $g \left( z \right) = 1 - \exp \left( - z \right)$ and $\begin{array} { r } { \zeta = \ln \left( 1 + \frac { c } { b } \right) } \end{array}$

Backward decoupling $\theta _ { k } ^ { ( t ) } \colon$ : Following the backward decoupling strategy of [13], we recursively marginalize the latent states $\theta _ { k } ^ { ( \overline { { t } } ) }$ backward in time to decouple the Markov chain. Defining $\begin{array} { r } { y _ { \cdot k } ^ { ( t ) } ~ = ~ \sum _ { v = 1 } ^ { V } y _ { v k } ^ { ( t ) } ~ \sim ~ P \acute { o i s } ( \delta ^ { ( t ) } \theta _ { k } ^ { ( t ) } ) } \end{array}$ , we start at the boundary $t ~ = ~ T$ to marginalize $\dot { \theta } _ { k } ^ { ( T ) }$ and obtain $\begin{array} { r } { y _ { \cdot k } ^ { ( T ) } \ \sim \ N B ( \bar { \tau _ { 0 } } \sum _ { k _ { 2 } = 1 } ^ { K } \pi _ { k k _ { 2 } } ^ { i ( T - 1 ) } \theta _ { k _ { 2 } } ^ { ( \overline { { T } } - 1 ) } , g ( \zeta ^ { ( \overline { { T } } ) } ) ) } \end{array}$ , where $\zeta ^ { ( T ) } = \mathrm { l n } ( 1 + \delta ^ { ( T ) } / \tau _ { 0 } )$

To proceed backward, we introduce auxiliary variables $\begin{array} { r l r } { l _ { k } ^ { ( T ) } } & { \sim } & { C R T ( y _ { \cdot k } ^ { ( T ) } , \tau _ { 0 } \sum _ { k _ { 2 } = 1 } ^ { K } \pi _ { k k _ { 2 } } ^ { i ( T - 1 ) } \theta _ { k _ { 2 } } ^ { ( T - 1 ) } ) } \end{array}$ .By Lemma 1, the joint distribution of $y _ { \cdot k } ^ { ( T ) }$ and $\bar { l } _ { k } ^ { ( T ) }$ can be expressed as $y _ { \cdot k } ^ { ( T ) } \sim$ SumL $\mathrm { \ddot { o g } } ( l _ { k } ^ { ( T ) } , g ( \ddot { \zeta ^ { ( T ) } } ) )$ , $l _ { k } ^ { ( T ) } \sim$ $\begin{array} { r } { \operatorname { \tilde { P } o i s } \zeta ^ { ( T ) } \tau _ { 0 } \sum _ { k _ { 2 } = 1 } ^ { K ^ { ' ^ { \circ } } } \pi _ { k k _ { 2 } } ^ { i ( T - 1 ) } \theta _ { k _ { 2 } } ^ { ( T - 1 ) } ) } \end{array}$ . Next, note that $y _ { \cdot k } ^ { \binom { T - 1 } { T } } \sim$ $\mathrm { P o i s } ( \delta ^ { ( T - 1 ) } \theta _ { k } ^ { ( \tilde { T } - \bar { 1 } ) } )$ , if we introduce $m _ { k } ^ { ( T - 1 ) } = y _ { \cdot k } ^ { ( T - 1 ) ^ { \because } } + l _ { \cdot k } ^ { ( T ) }$ then we have $m _ { k } ^ { ( { \cal T } - 1 ) } ~ \sim ~ \mathrm { P o i s } ( \theta _ { k } ^ { ( { \cal T } - 1 ) } ( \delta ^ { ( { \cal T } - 1 ) } + \zeta ^ { ( { \cal T } ) } \tau _ { 0 } ^ { " } ) )$ We can again marginalize over $\dot { \theta _ { k } ^ { ( \overset { n } { T } - 1 ) } }$ to obtain $m _ { k } ^ { ( T - 1 ) } \sim$ $\begin{array} { r } { \mathrm { N B } ( \tau _ { 0 } \sum _ { k _ { 2 } = 1 } ^ { \overline { { K } } } \pi _ { k k _ { 2 } } ^ { i ( T ^ { - } - 2 ) } \theta _ { k _ { 2 } } ^ { ( T - 2 ) } , g ( \zeta ^ { ( T - 1 ) } ) ) } \end{array}$ , where $\zeta ^ { ( \overleftarrow { T } - 1 ) } =$ $\begin{array} { r } { \ln ( 1 + \frac { \delta ^ { ( T - 1 ) } } { \tau _ { 0 } } + \zeta ^ { \bar { ( T ) } } ) } \end{array}$ . Then we introduce auxiliary variables $\begin{array} { r } { \boldsymbol { l } _ { k } ^ { ( T - 1 ) } \sim \mathrm { { C R T } } ( m _ { k } ^ { ( T - 1 ) } , \tau _ { 0 } \sum _ { k _ { 2 } = 1 } ^ { K } \pi _ { k k _ { 2 } } ^ { i ( T - 2 ) } \boldsymbol { \theta } _ { k _ { 2 } } ^ { ( T - 2 ) } ) } \end{array}$

Repeating this marginalization recursively down to $t = 1$ with $\zeta ^ { ( t ) } = \ln ( 1 + \delta ^ { \overline { { ( t ) } } } / \tau _ { 0 } + \zeta ^ { ( t + 1 ) } )$ to maginalize over all the $\theta _ { k } ^ { ( t ) }$ , we decouple the sequence. To sample $\theta _ { k } ^ { ( t ) }$ , we set $l _ { \cdot k } ^ { ( T + 1 ) ^ { \wedge } } = 0$ and $\zeta ^ { ( T + 1 ) } = 0 ,$ and first sample the auxiliary variables $l _ { k \cdot } ^ { ( t ) }$ and $l _ { k k _ { 2 } } ^ { ( t ) }$ backward for $t = T , \dots , 2$ as Eq. (10).

And using the relationship between Poisson and multinomial distributions, we have

$$
\left( l _ { 1 k } ^ { ( t ) } , \cdots , l _ { K k } ^ { ( t ) } \right) \sim \mathrm { M u l t } \left( l _ { \cdot k } ^ { ( t ) } , \pi _ { 1 k } ^ { i ( t - 1 ) } , \cdots , \pi _ { K k } ^ { i ( t - 1 ) } \right) .\tag{19}
$$

Sampling $\boldsymbol { \Pi } ^ { ( i ) }$ : For $i = I ,$ by $\operatorname { E q . } ( 1 9 ) , \ \left( l _ { 1 k } ^ { ( I ) } , \cdot \cdot \cdot , l _ { K k } ^ { ( I ) } \right)$ is multinomial distributed. Thus by multinomial-Dirichlet conjugacy, we obtain

$$
\left( \pi _ { k } ^ { ( I ) } \mid - \right) \sim \mathrm { D i r } \left( \alpha _ { 1 k } ^ { ( I ) } + l _ { 1 k } ^ { ( I ) } , \cdot \cdot \cdot , \alpha _ { K k } ^ { ( I ) } + l _ { K k } ^ { ( I ) } \right) .
$$

Inference for Dirichlet-Dirichlet Markov chains. For Dirichlet-Dirichlet Markov chains, $\alpha _ { k _ { 1 } k } ^ { ( i ) } ~ = ~ \eta K \pi _ { k _ { 1 } k } ^ { ( i - 1 ) }$ . If we marginalize $\Big ( \pi _ { 1 k } ^ { ( I ) } , \cdot \cdot \cdot , \pi _ { K k } ^ { ( I ) } \Big ) , \Big ( l _ { 1 k } ^ { ( \bar { I } ) } , \cdot \cdot \cdot , l _ { K k } ^ { ( I ) } \Big )$ will be Dirichlet-multinomial distributed as

$$
\left( l _ { 1 k } ^ { ( I ) } , \cdots , l _ { K k } ^ { ( I ) } \right) \sim \mathrm { D i r M u l t } \left( l _ { k \cdot } ^ { ( I ) } , \left( \eta K \pi _ { 1 k } ^ { ( I - 1 ) } , \cdots , \eta K \pi _ { K k } ^ { ( I - 1 ) } \right) \right)
$$

Thus by Lemma 2, for $i = I ,$ we first sample the auxiliary variables as

$$
\begin{array} { r l } & { \left( q _ { k } ^ { ( I ) } \mid - \right) \sim \mathrm { B e t a } \left( l _ { \cdot k } ^ { ( I ) } , \eta K \right) , } \\ & { \left( h _ { k _ { 1 } k } ^ { ( I ) } \mid - \right) \sim \mathrm { C R T } \left( l _ { k _ { 1 } k } ^ { ( I ) } , \eta K \pi _ { k _ { 1 } k } ^ { ( I - 1 ) } \right) . } \end{array}
$$

Via Lemma 1 and 2, $h _ { \cdot k } ^ { ( i ) }$ is Poisson distributed and $\left( h _ { 1 k } ^ { \left( i \right) } , \cdot \cdot \cdot , h _ { K k } ^ { \left( i \right) } \right)$ is Dirichlet-multinomial distributed. Thus for $i \dot { = } I - 1 , \cdots , \stackrel { \zeta } { 2 } ,$ we sample the auxiliary variables as

$$
\begin{array} { r l } & { \left( \boldsymbol { q } _ { k } ^ { ( i ) } \mid - \right) \sim \mathrm { B e t a } \left( l _ { \cdot k } ^ { ( i ) } + h _ { \cdot k } ^ { ( i + 1 ) } , \eta K \right) , } \\ & { \left( h _ { k _ { 1 } k } ^ { ( i ) } \mid - \right) \sim \mathrm { C R T } \left( l _ { k _ { 1 } k } ^ { ( i ) } + h _ { k _ { 1 } k } ^ { ( i + 1 ) } , \eta K \pi _ { k _ { 1 } k } ^ { ( i - 1 ) } \right) , } \end{array}
$$

where $\begin{array} { r } { l _ { k _ { 1 } k } ^ { ( i ) } = \sum _ { ( i - 1 ) M + 1 } ^ { i M } l _ { k _ { 1 } k } ^ { ( t ) } } \end{array}$ refers to the summation of $l _ { k _ { 1 } k } ^ { ( t ) }$ over i-th interval. Via Lemma $^ { 2 , }$ conditioning on $q _ { k } ^ { ( i ) }$ , we have

$$
\begin{array} { r } { \left( l _ { k _ { 1 } k } ^ { ( i ) } + h _ { k _ { 1 } k } ^ { ( i + 1 ) } \right) \sim \mathrm { N B } \left( \eta K \pi _ { k _ { 1 } k } ^ { ( i - 1 ) } , q _ { k } ^ { ( i ) } \right) . } \end{array}
$$

Then via Lemma 1, $\left( l _ { k _ { 1 } k } ^ { ( i ) } + h _ { k _ { 1 } k } ^ { ( i + 1 ) } \right)$ also follows a Poisson distribution, $\left( l _ { k } ^ { ( i ) } + h _ { k } ^ { ( i + 1 ) } \right)$ thus is mutinomial distributed. Via Dirichlet-multinomial conjugacy, for $i = I - 1 , \cdot \cdot , 2 ,$ we obtain

$$
\begin{array} { r l } & { \left( \pi _ { k } ^ { ( i ) } \mid - \right) \sim \mathrm { D i r } \Big ( \eta K \pi _ { 1 k } ^ { ( i - 1 ) } + l _ { 1 k } ^ { ( i ) } + h _ { 1 k } ^ { ( i + 1 ) } } \\ & { \qquad , \cdots , \eta K \pi _ { K k } ^ { ( i - 1 ) } + l _ { K k } ^ { ( i ) } + h _ { K k } ^ { ( i + 1 ) } \Big ) . } \end{array}
$$

Specifically, for $i = 1$ , we have

$$
\begin{array} { r l } & { \left( \pi _ { k } ^ { ( 1 ) } \mid - \right) \sim \operatorname { D i r } ( \nu _ { 1 } \nu _ { k } + l _ { 1 k } ^ { ( 1 ) } + h _ { 1 k } ^ { ( 2 ) } , \cdots , \xi \nu _ { k } + l _ { k k } ^ { ( 1 ) } + h _ { k k } ^ { ( 2 ) } } \\ & { \qquad , \cdots , \nu _ { K } \nu _ { k } + l _ { K k } ^ { ( 1 ) } + h _ { K k } ^ { ( 2 ) } ) . } \end{array}
$$

For sampling η, note that

$$
\left( h _ { k _ { 1 } k } ^ { ( i ) } \mid - \right) \sim \mathrm { P o i s } \left( - \eta K \pi _ { k _ { 1 } k } ^ { ( i - 1 ) } \mathrm { l n } \left( 1 - q _ { k } ^ { ( i ) } \right) \right) , i = I , \cdots , 2 .
$$

Given the prior $\eta \sim \mathrm { G a m } \left( e _ { 0 } , f _ { 0 } \right)$ ,via Poisson-gamma conjugacy, we obtain

$$
\begin{array} { c } { { ( \eta \mid - ) \sim \mathrm { \normalfont ~ G a m } ( e _ { 0 } + \displaystyle \sum _ { i = 2 } ^ { I } \sum _ { k _ { 1 } = 1 } ^ { K } \sum _ { k _ { 2 } = 1 } ^ { K } h _ { k _ { 1 } k _ { 2 } } ^ { ( i ) } , } } \\ { { f _ { 0 } - K \displaystyle \sum _ { i = 2 } ^ { I } \sum _ { k = 1 } ^ { K } \ln ( 1 - q _ { k } ^ { ( i ) } ) ) . } } \end{array}
$$

Inference for Dirichlet-Gamma-Dirichlet Markov chains. $\left( l _ { 1 k } ^ { \left( i \right) } , \cdots , l _ { K k } ^ { \left( i \right) } \right)$ will be Dirichlet-multinomial distributed when marginalizing $\left( \pi _ { 1 k } ^ { \left( i \right) } , \cdot \cdot \cdot , \pi _ { K k } ^ { \left( i \right) } \right)$ . Thus by Lemma 2, for $i = I ,$ , we first sample the auxiliary variables as

$$
\begin{array} { r l } & { \left( \boldsymbol { q } _ { k } ^ { ( I ) } \mid - \right) \sim \mathrm { B e t a } \left( \boldsymbol { l } _ { \cdot k } ^ { ( I ) } , \boldsymbol { \alpha } _ { \cdot k } ^ { ( I ) } \right) , } \\ & { \left( h _ { k _ { 1 } k } ^ { ( I ) } \mid - \right) \sim \mathrm { C R T } \left( \boldsymbol { l } _ { k _ { 1 } k } ^ { ( I ) } , \boldsymbol { \alpha } _ { k _ { 1 } k } ^ { ( I ) } \right) . } \end{array}
$$

Similarly, by Eq.(20), we construct $\left( g _ { \cdot 1 k } ^ { \left( i \right) } , \cdot \cdot \cdot , g _ { \cdot K k } ^ { \left( i \right) } \right)$ which is also Dirichlet-multinomial distributed. Thus for $\textit { i } = I -$ $1 , \cdots , 2 ,$ , we sample the auxiliary variables as

$$
\begin{array} { r l } & { \left( q _ { k } ^ { \left( i \right) } \mid - \right) \sim \mathrm { B e t a } \left( l _ { \cdot k } ^ { \left( i \right) } + g _ { \cdot k } ^ { \left( i + 1 \right) } , \alpha _ { \cdot k } ^ { \left( i \right) } \right) , } \\ & { \left( h _ { k _ { 1 } k } ^ { \left( i \right) } \mid - \right) \sim \mathrm { C R T } \left( l _ { k _ { 1 } k } ^ { \left( i \right) } + g _ { \cdot k _ { 1 } k } ^ { \left( i + 1 \right) } , \alpha _ { k _ { 1 } k } ^ { \left( i \right) } \right) . } \end{array}
$$

Via Lemma 2, conditioning on $\begin{array} { r } { q _ { k } ^ { ( i ) } , } \end{array}$ we have $\begin{array} { r l r } { \left( l _ { k _ { 1 } k } ^ { ( i ) } + g _ { \cdot k _ { 1 } k } ^ { ( i + 1 ) } \right) } & { { } \sim } & { \mathrm { N B } \left( \alpha _ { k _ { 1 } k } ^ { ( i ) } , \overline { { q _ { k } ^ { ( i ) } } } \right) } \end{array}$ . Then via Lemma 1, we obtain $\begin{array} { r } { \dot { h } _ { k _ { 1 } k } ^ { ( i ) } \sim \operatorname { P o i s } \dot { \left( - \alpha _ { k _ { 1 } k } ^ { ( i ) } \mathrm { l n } \left( 1 - q _ { k } ^ { ( i ) } \right) \right) } } \end{array}$ . Thus via Poisson-gamma conjugacy, we obtain

$$
\begin{array} { c } { { ( \alpha _ { k _ { 1 } k } ^ { ( i ) } \mid - ) \sim \mathrm { G a m } ( \gamma _ { k } ^ { ( i - 1 ) } \displaystyle \sum _ { k 2 = 1 } ^ { K } \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } \pi _ { k _ { 2 } k } ^ { ( i - 1 ) } + h _ { k _ { 1 } k } ^ { ( i ) } , } } \\ { { c _ { k } ^ { ( i ) } - \ln ( 1 - q _ { k } ^ { ( i ) } ) ) . } } \end{array}
$$

Marginalizing over $\alpha _ { k _ { 1 } k } ^ { ( i ) } ,$ and via the definition of negative binomial distribution, we have

$$
h _ { k _ { 1 } k } ^ { ( i ) } \sim \mathrm { N B } \left( \gamma _ { k } ^ { ( i - 1 ) } \sum _ { k 2 = 1 } ^ { K } \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } \pi _ { k _ { 2 } k } ^ { ( i - 1 ) } , \frac { - \mathrm { l n } \left( 1 - q _ { k } ^ { ( i ) } \right) } { c _ { k } ^ { ( i ) } - \mathrm { l n } \left( 1 - q _ { k } ^ { ( i ) } \right) } \right)
$$

Then using Lemma 1, we sample

$$
\left( g _ { k _ { 1 } k } ^ { ( i ) } \mid - \right) \sim \mathrm { C R T } \left( h _ { k _ { 1 } k } ^ { ( i ) } , \gamma _ { k } ^ { ( i - 1 ) } \sum _ { k 2 = 1 } ^ { K } \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } \pi _ { k _ { 2 } k } ^ { ( i - 1 ) } \right) ,
$$

and obtain $g _ { k _ { 1 } k } ^ { ( i ) } \ \sim \ \mathrm { P o i s } ( \gamma _ { k } ^ { ( i - 1 ) } \sum _ { k 2 = 1 } ^ { K } \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } \pi _ { k _ { 2 } k } ^ { ( i - 1 ) } \mathrm { l n } ( 1 \ - $ $\ln ( 1 - q _ { k } ^ { ( i ) } ) / c _ { k } ^ { ( i ) } ) )$ . Using the relationship between Poisson and multinomial distributions and $\begin{array} { r } { \sum _ { k _ { 1 } } ^ { K } \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( \bar { i } - 1 ) } = 1 } \end{array}$ , we have

$$
\left( g _ { \cdot 1 k } ^ { ( i ) } , \cdot \cdot \cdot , g _ { \cdot K k } ^ { ( i ) } \right) \sim \mathrm { M u l t } \left( g _ { \cdot k } ^ { ( i ) } , \left( \pi _ { k _ { 1 } k } ^ { ( i - 1 ) } \right) _ { k _ { 1 } = 1 } ^ { K } \right) ,\tag{20}
$$

$$
\left( g _ { 1 k _ { 2 } k } ^ { ( i ) } , \cdots , g _ { K k _ { 2 } k } ^ { ( i ) } \right) \sim \mathrm { M u l t } \left( g _ { \cdot k _ { 2 } k } ^ { ( i ) } , \left( \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } \right) _ { k 1 = 1 } ^ { K } \right) .
$$

Thus by Dirichlet-multinomial conjugacy, for $i = I , \cdots , 2$ , we can obtain the posterior distributions of $( \psi _ { k 1 k _ { 2 } } ^ { ( i - 1 ) } , \cdot \cdot \cdot , \psi _ { k K k _ { 2 } } ^ { ( i - 1 ) } )$ and $\pi _ { k } ^ { ( i - 1 ) }$ , which is the same as in Eq.(17) and (18). For sampling $\gamma _ { k } ^ { ( i - 1 ) }$ , note that $\begin{array} { r } { \sum _ { k _ { 1 } } ^ { K } \psi _ { k k _ { 1 } k _ { 2 } } ^ { ( i - 1 ) } = 1 } \end{array}$ , we have

$$
g _ { \cdot k } ^ { ( i ) } \sim \mathrm { P o i s } \left( \gamma _ { k } ^ { ( i - 1 ) } \mathrm { l n } \left( 1 - \mathrm { l n } \left( 1 - q _ { k } ^ { ( i ) } \right) / c _ { k } ^ { ( i ) } \right) \right) .
$$

Thus via Poisson-gamma conjugacy, we obtain

$$
\left( \gamma _ { k } ^ { ( i - 1 ) } \mid - \right) \sim \mathrm { G a m } \left( \epsilon _ { 0 } + g _ { . k } ^ { ( i ) } , \epsilon _ { 0 } + \ln \left( 1 - \ln \left( 1 - q _ { k } ^ { ( i ) } \right) \right) \right) .
$$

By gamma-gamma conjugacy, we have

$$
\left( c _ { k } ^ { ( i ) } \mid - \right) \sim { \mathrm { G a m } } \left( \epsilon _ { 0 } + \gamma _ { k } ^ { ( i - 1 ) } , \epsilon _ { 0 } + \sum _ { k _ { 1 } = 1 } ^ { K } \alpha _ { k _ { 1 } k } ^ { ( i ) } \right) .
$$

Sampling $\nu _ { k }$ and $\xi : \operatorname { A s }$ we sample $\boldsymbol { \Pi } ^ { ( i ) }$ , by the definition of Dirichlet-multinomial distribution, we obtain

$$
\begin{array} { r l } & { \left( l _ { 1 k } ^ { ( 1 ) } + g _ { \cdot 1 k } ^ { ( 2 ) } , \cdot \cdot \cdot , l _ { K k } ^ { ( 1 ) } + g _ { \cdot K k } ^ { ( 2 ) } \right) } \\ & { \sim \mathrm { D i r M u l t } \left( \nu _ { 1 } \nu _ { K } , \cdot \cdot \cdot , \xi \nu _ { k } , \cdot \cdot \cdot , \nu _ { K } \nu _ { k } \right) . } \end{array}
$$

In particular, with a little abuse of notation here, for Dir-Dir construction, we take $g _ { \cdot k _ { 1 } k } ^ { ( 2 ) } = h _ { k _ { 1 } k } ^ { ( 2 ) }$ . We first sample

$$
\begin{array} { r } { \left( h _ { k _ { 1 } k } ^ { ( 1 ) } \mid - \right) \sim \left\{ \begin{array} { l l } { \mathrm { C R T } \left( l _ { k _ { 1 } k } ^ { ( 1 ) } + g _ { \cdot k _ { 1 } k } ^ { ( 2 ) } , \nu _ { k _ { 1 } } \nu _ { k } \right) } & { k _ { 1 } \neq k } \\ { \mathrm { C R T } \left( l _ { k _ { 1 } k } ^ { ( 1 ) } + g _ { \cdot k _ { 1 } k } ^ { ( 2 ) } , \xi \nu _ { k } \right) } & { k _ { 1 } = k . } \end{array} \right. } \end{array}
$$

Then we sample

$$
q _ { k } ^ { ( 1 ) } \sim \mathrm { B e t a } \left( l _ { \cdot k } ^ { ( 1 ) } + g _ { \cdot k } ^ { ( 2 ) } , \nu _ { k } \left( \sum _ { k _ { 1 } \neq k } \nu _ { k 1 } + \xi \right) \right) .
$$

We further introduce

$$
\begin{array} { l } { { \displaystyle n _ { k } = h _ { k k } ^ { ( 1 ) } + \sum _ { k _ { 1 } \neq k } h _ { k _ { 1 } k } ^ { ( 1 ) } + \sum _ { k _ { 2 } \neq k } h _ { k k _ { 2 } } ^ { ( 1 ) } + l _ { k \cdot } ^ { ( 1 ) } , } } \\ { { \displaystyle \rho _ { k } = \tau _ { 0 } \zeta ^ { ( 1 ) } - \ln \left( 1 - q _ { k } ^ { ( 1 ) } \right) \left( \xi + \sum _ { k _ { 1 } \neq k } \nu _ { k _ { 1 } } \right) } } \\ { { \displaystyle ~ - \sum _ { k _ { 2 } \neq k } \ln ( 1 - q _ { k _ { 2 } } ^ { ( 1 ) } ) \nu _ { k _ { 2 } } . } } \end{array}
$$

Via Poisson-gamma conjugacy, we have

$$
\begin{array} { l } { { \displaystyle \left( \xi \mid - \right) \sim \mathrm { G a m } \left( \frac { \gamma _ { 0 } } { K } + \sum _ { k } h _ { k k } ^ { ( 1 ) } , \beta - \sum _ { k } \nu _ { k } \mathrm { l n } \left( 1 - q _ { k } ^ { ( 1 ) } \right) \right) , } } \\ { { \displaystyle \left( \nu _ { k } \mid - \right) \sim \mathrm { G a m } \left( \frac { \gamma _ { 0 } } { K } + n _ { k } , \beta + \rho _ { k } \right) . } } \end{array}
$$

Sampling $\delta ^ { ( t ) }$ and $\beta$ : Via Poisson-gamma conjugacy

$$
\left( \delta ^ { ( t ) } \mid - \right) \sim { \mathrm { G a m } } \left( \epsilon _ { 0 } + \sum _ { v = 1 } ^ { V } y _ { v } ^ { ( t ) } , \epsilon _ { 0 } + \sum _ { k = 1 } ^ { K } \theta _ { k } ^ { ( t ) } \right) .
$$

And by gamma-gamma conjugacy, we obtain

$$
( \beta \mid - ) \sim { \mathrm { G a m } } \left( \epsilon _ { 0 } + \gamma _ { 0 } , \epsilon _ { 0 } + \sum _ { k = 1 } ^ { K } \nu _ { k } \right) .
$$