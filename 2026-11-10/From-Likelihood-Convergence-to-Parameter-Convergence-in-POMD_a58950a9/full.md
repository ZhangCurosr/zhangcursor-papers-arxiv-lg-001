# From Likelihood Convergence to Parameter Convergence in POMDPs

Jack Zhang<sup>1</sup> Saurabh Amin<sup>1</sup> Jiawei Zhang<sup>2</sup> Patrick Jaillet<sup>1</sup>

<sup>1</sup>Operations Research Center, Massachusetts Institute of Technology Cambridge, MA 02139

<sup>2</sup>Computer Sciences Department, University of Wisconsin–Madison Madison, WI 53706

{jackz111,amins,jaillet}@mit.edu jzhang2924@wisc.edu

## Abstract

Likelihood-based methods are widely used to learn the parameters of partially observable Markov decision processes (POMDPs). While finite-sample guarantees for policy suboptimality exist for certain POMDP subclasses, analogous guarantees for parameter recovery remain less well understood. For any candidate POMDP parameter, we bound its parameter recovery error in terms of its likelihood gap to the ground truth, the structural conditioning of the observation and transition kernels, and the behavior policy used to collect data. The bound applies in both the undercomplete and overcomplete regimes, with the latter following naturally from the undercomplete analysis. Because the result applies to any sufficiently good candidate, it yields finite-sample guarantees for the maximum-likelihood estimator and any procedure producing a parameter with controlled likelihood gap. This is particularly useful for near-optimal learning in infinite-horizon POMDPs, where existing approaches largely rely on model-based planning with accurate transition and observation kernel estimates. The bound separates the contributions of the behavior policy and model, showing how data-collection choices, including action coverage, and their effects on mixing can accelerate parameter learning.

## 1 Introduction

Partially observable Markov decision processes (POMDPs) provide a general framework for sequential decision making under partial observability, with applications in robotics [Koenig and Simmons, 1996], healthcare [Li et al., 2018], recommendation systems [Boutilier, 2002], and operations research [Bravo et al., 2019]. POMDPs introduce uncertainty about the underlying state: the agent must make decisions based on its observation-action history or belief state. This uncertainty substantially increases the difficulty of planning and learning [Kaelbling et al., 1998]. Indeed, even finite-horizon planning with a known POMDP model is PSPACE-complete [Papadimitriou and Tsitsiklis, 1987].

A prominent line of work studies model-based learning for POMDPs, in which model parameters are estimated from data and used for planning. The learned model enables recursive belief-state updates from observation and action histories. The resulting belief states serve as sufficient statistics for decision-making within the learned POMDP [Kaelbling et al., 1998]. This belief-state representation allows the agent to account for both reward maximization and information gathering [Spaan, 2012], while leveraging the extensive body of POMDP planning methods developed over the past two decades [Shani et al., 2013]. In particular, existing methods for learning in infinite-horizon POMDPs have largely followed this model-based paradigm, relying on estimates of the transition and observation kernels to compute belief states and obtain near-optimal policies [Ross et al., 2009].

Two major approaches to model-based POMDP learning are spectral methods and likelihood-based methods. Spectral methods estimate cross-moment statistics of observation sequences and employ matrix or tensor decompositions to recover the model kernels, providing finite-sample guarantees under suitable assumptions while avoiding direct optimization of the nonconvex likelihood. Likelihoodbased methods, by contrast, seek to maximize the likelihood of observed trajectories, often using the Baum–Welch algorithm [Baum et al., 1970], and are widely used in practice.

## 1.1 Motivation

Spectral and likelihood-based methods face complementary limitations, leaving an important gap in finite-sample kernel recovery. Standard spectral methods for POMDPs require a full-column-rank observation kernel for identifiability, thereby restricting their guarantees to undercomplete POMDPs, where the number of latent states does not exceed the number of observations. Moreover, stable recovery requires the observation kernel to admit a well-conditioned left inverse, ensuring that estimation errors in empirical moments are not excessively amplified. In overcomplete POMDPs, where the number of latent states exceeds the number of observations, the observation kernel is necessarily rank-deficient. Consequently, distinct latent-state distributions can induce identical one-step observation distributions, and standard spectral recovery guarantees no longer apply.

Likelihood-based methods do not require the observation kernel to have full column rank and can be formulated in both undercomplete and overcomplete regimes. However, their finite-sample guarantees for parameter recovery remain limited. Classical results establish consistency and asymptotic normality [Leroux, 1992, Bickel et al., 1998], while finite-sample analyses have been developed for specific algorithms such as Baum–Welch [Yang et al., 2017]. A central remaining challenge is to establish a general relationship that translates the likelihood gap of an arbitrary candidate model relative to the true model into a bound on parameter recovery error. Such a relationship would provide an estimator-agnostic recovery guarantee, allowing any candidate with a certified likelihood-gap bound to inherit a corresponding parameter-error bound, regardless of how it was computed.

Our focus on parameter recovery, rather than policy suboptimality, is motivated by the role of accurate transition and observation kernels in belief-state estimation and near-optimal control for infinitehorizon POMDPs. For finite-horizon POMDPs, near-optimal learning can sometimes be achieved by estimating predictive distributions of future observations without recovering the full model parameters. In contrast, existing approaches to near-optimal learning in infinite-horizon POMDPs have largely been model-based, relying on estimates of the transition and observation kernels (up to latent-state relabeling) for belief-state updates [Ross et al., 2009, Russo et al., 2026]. However, whether accurate kernel recovery is fundamentally necessary for near-optimal learning remains unclear. Finite-sample guarantees directly targeting policy suboptimality primarily address the finite-horizon setting, whereas spectral recovery guarantees generally require undercomplete models. Consequently, these two lines of results do not fully cover infinite-horizon POMDPs, particularly in the overcomplete regime.

Beyond its role in policy learning, parameter recovery is also a well-established objective in latentvariable modeling. Hidden Markov models (HMMs), which can be viewed as POMDPs with a singleton action set, have long been studied independently of control objectives: recovering their transition and observation kernels enables latent-state inference, forecasting, simulation, and interpretation of the underlying dynamics. Similarly, in POMDPs, the transition and observation kernels characterize the underlying dynamics and observation mechanism independently of the reward specification. Recovering these kernels therefore yields a model reusable across multiple downstream tasks and analyses with different reward kernels. Appendix A provides a concrete example.

The theoretical and practical motivations raise the central question: Can we establish finite-sample guarantees for POMDP parameter recovery that translate the likelihood gap of an arbitrary candidate model into a quantitative parameter-error bound in both undercomplete and overcomplete regimes?

## 1.2 Related work

Spectral methods. Spectral learning for discrete POMDPs was introduced by Azizzadenesheli et al. [2016a], with subsequent extensions to rich-observation MDPs [Azizzadenesheli et al., 2016b] and episodic PAC learning [Guo et al., 2016]. These approaches build on the broader spectral literature for latent-variable models [Anandkumar et al., 2012, 2014, Hsu et al., 2012] and require undercompleteness. More recently, Xiong et al. [2022] used spectral methods to establish sublinear regret for infinite-horizon POMDPs with state-dependent rewards. Russo et al. [2025] developed an action-wise spectral estimator achieving $\widetilde { O } ( \sqrt { T } )$ regret with known observation kernels. Russo et al. [2026] later extended this approach to unknown observation kernels through mixed spectral estimation, recovering both kernels from data collected under multiple policies while retaining the $\widetilde { O } ( \sqrt { T } )$ regret guarantee. Depiste these advances, the methods of Xiong et al. [2022] and Russo et al. [2025, 2026] continue to require undercompleteness. Shaw et al. [2026] extended spectral learning beyond this restriction by employing predictive state representations and tensor decomposition to recover model parameters up to a similarity transformation. Their results accommodate certain overcomplete POMDPs but provide asymptotic rather than finite-sample guarantees.

Likelihood-based methods for HMMs. Classical results establish consistency and asymptotic normality of maximum likelihood estimators (MLEs) for HMMs [Leroux, 1992, Bickel et al., 1998], with extensions to general state spaces [Douc and Matias, 2001, Douc et al., 2011]. Finitesample guarantees for parameter recovery, however, remain limited. Lehéricy [2021] bounds the KL divergence between the observation distributions induced by the MLE and the true model, rather than directly controlling parameter error. Yang et al. [2017] establish finite-sample contraction of Baum–Welch’s iterates to a neighborhood of the true parameters under local regularity, mixing, and a uniform-positivity assumption on the transition kernel. These results characterize how closely the estimated model reproduces the true distribution of observation sequences and the convergence of Baum–Welch, respectively, rather than directly relating likelihood gaps to parameter recovery errors.

Likelihood-based methods for POMDPs. The model-dependent component of POMDP trajectory likelihood has an input-output HMM structure, even under adaptive behavior policies [Krishnamurthy, 2016]. This connection allows likelihood-based techniques for HMMs to be adapted to POMDPs. In POMDP learning, likelihood-based estimation is often incorporated into broader frameworks, including Optimistic MLE approaches of Chen et al. [2023] and Liu et al. [2022, 2023], as well as related procedures for hindsight-observable variants [Lee et al., 2023, Guo et al., 2024, 2023]. These works primarily establish sample-complexity guarantees for policy suboptimality in episodic POMDPs. Separately, Jahromi et al. [2022] employ Bayesian posterior sampling to establish sublinear regret for infinite-horizon POMDPs with known observation kernels, although their general-case guarantee relies on additional assumptions concerning parameter and belief estimation.

## 1.3 Contributions

Finite-sample, candidate-agnostic bounds linking likelihood gaps to parameter recovery errors in undercomplete and overcomplete POMDPs. For any candidate model, we bound its parameter recovery error, up to a permutation of the latent states, in terms of its likelihood gap relative to the true model, the structural conditioning of the kernels, and the sample size. This guarantee applies to MLE and, more generally, to any procedure that produces a parameter estimate with a controlled likelihood gap. Both the undercomplete and overcomplete regimes are covered by a unified framework, with the analysis of the latter building on that of the former. To the best of our knowledge, this is the first finite-sample guarantee for full parameter recovery in overcomplete POMDPs.

Full kernel recovery provides a reusable model beyond fixed episodic tasks. As discussed in Section 1.2, near-optimal policy-learning guarantees without full parameter recovery typically concern episodic POMDPs with fixed horizons, fixed initial-state distributions, and reset-based data collection. By contrast, our result provides an estimator-agnostic parameter-recovery guarantee that supports model-based planning in infinite-horizon POMDPs [Ross et al., 2009]. The resulting model supports planning from arbitrary initial-state distributions, replanning without system resets, and other downstream tasks requiring accurate kernel estimates.

Explicit dependence on the policy. Our bound distinguishes the structural properties of the model from the effects of the behavior policy, explicitly quantifying the role of the behavior policy through its action coverage and the stationary distribution and mixing rate of the induced Markov chain.

## 1.4 Paper organization

Section 2 formalizes the POMDP model and the likelihood formulation. Section 3 establishes our main results, beginning with the undercomplete case and extending to the overcomplete case. Section 4 summarizes our main contributions. Section 5 discusses limitations and future work.

## 2 Problem setup

We consider a POMDP with finite state, action, and observation spaces, $\begin{array} { r l } { \mathcal { M } } & { { } : = } \end{array}$ $( S , \mathcal { A } , \{ T _ { \beta ^ { * } } ^ { a } \} _ { a \in \mathcal { A } } , R , \Omega , O _ { \varphi ^ { * } } , \gamma , \mu _ { \beta ^ { * } } )$ , where $s , A ,$ and Ω denote the state, action, and observation spaces, respectively; $\{ T _ { \beta ^ { * } } ^ { a } \} _ { a \in \mathcal A }$ is the transition kernel; $R : S \times \mathcal { A }  \mathbb { R }$ is the reward function; $O _ { \varphi ^ { * } } : S  \Delta ( \Omega )$ is the observation kernel; $\mu _ { \beta ^ { * } }$ is the initial-state distribution. We use $\beta$ to parameterize the transition kernels and initial-state distribution, and $\varphi$ to parametrize the observation kernel. The ground-truth parameter is denoted by $\theta ^ { * } = ( \beta ^ { * } , \varphi ^ { * } )$

We focus on the single-trajectory setting, where temporal dependence requires mixing-based concentration. With independent trajectories, the same analysis applies to the pooled data. Hence the single-trajectory case captures the main dependence-related difficulty.

We consider a trajectory ${ \boldsymbol { \tau } } = ( o _ { 1 : n } , a _ { 1 : n } )$ collected under a behavior policy π and the ground-truth parameter $\theta ^ { * }$ , and let $s _ { 1 : n }$ denote the corresponding latent state sequence. The assumptions on π are specified in Sections 3.1 and 3.2 for under- and overcomplete regimes. For any $\bar { { \boldsymbol { \theta } } } = ( \beta , \varphi )$ the model-dependent component of the complete-data trajectory likelihood is $f _ { \theta } ( o _ { 1 } ^ { n } , s _ { 1 } ^ { n } ; a _ { 1 } ^ { n } ) ~ =$ $\begin{array} { r } { \mu _ { \beta } ( s _ { 1 } ) \prod _ { i = 2 } ^ { n } \mathbf { \hat { T } } _ { \beta } ^ { a _ { i - 1 } } ( s _ { i } | s _ { i - 1 } ) \prod _ { i = 1 } ^ { n } O _ { \varphi } ( o _ { i } | s _ { i } ) } \end{array}$

Since $\pi$ is fixed and independent of $\theta ,$ its contribution to the trajectory likelihood does not affect the relative likelihood between any pair of model parameters $( \theta , \theta ^ { \prime } )$ Accordingly, we define the model-dependent component of the observed-data log-likelihood, as $\ell _ { n } ( \theta ) : =$ log $\begin{array} { r } { \sum _ { s _ { 1 : n } \in S ^ { n } } f _ { \theta } \big ( o _ { 1 : n } , s _ { 1 : n } ; a _ { 1 : n } \big ) } \end{array}$

A small log-likelihood gap in $\ell _ { n }$ does not, in general, guarantee accurate recovery of the transition and observation kernels. We therefore seek to identify conditions under which such a gap yields a quantitative bound on parameter recovery error, up to latent-state relabeling.

## 3 Main result

## 3.1 Undercomplete POMDPs $( | S | \leq | \Omega | )$

For $\theta = ( \beta , \varphi )$ , let

$$
b _ { t , \theta } ( s ) : = \mathbb { P } _ { \theta } ( s _ { t } = s \mid o _ { 1 : t } , a _ { 1 : t - 1 } ) , \qquad { \bar { b } } _ { 1 , \theta } : = \mu _ { \beta } , \qquad { \bar { b } } _ { t , \theta } ( s ) : = \mathbb { P } _ { \theta } ( s _ { t } = s \mid o _ { 1 : t - 1 } , a _ { 1 : t - 1 } ) .
$$

The standard Bayes filter gives

$$
\bar { b } _ { t , \theta } ( s ) = \sum _ { s ^ { \prime } } T _ { \beta } ^ { a _ { t - 1 } } ( s \mid s ^ { \prime } ) b _ { t - 1 , \theta } ( s ^ { \prime } ) , \qquad b _ { t , \theta } ( s ) = \frac { O _ { \varphi } ( o _ { t } \mid s ) \bar { b } _ { t , \theta } ( s ) } { \sum _ { s ^ { \prime } } O _ { \varphi } ( o _ { t } \mid s ^ { \prime } ) \bar { b } _ { t , \theta } ( s ^ { \prime } ) } .
$$

Since $\bar { b } _ { t , \theta }$ is the pre-observation belief at time $t ,$ it induces the predictive distribution of $o _ { t } .$ For this reason, we call $\bar { b } _ { t , \theta }$ the predictive belief at time t induced by θ, while $b _ { t , \theta }$ is the posterior belief.

Having defined the predictive and posterior beliefs, we next specify the behavior policy used to collect trajectory data, as the subsequent assumptions and analysis depend on this choice.

Assumption 3.1. The behavior policy satisfies $\pi \in \Pi _ { \mathrm { o b s } } : = \{ \pi ^ { \prime } : \Omega \to \Delta ( A ) \}$

Assumption 3.2. $\pi ( a | o ) \geq \pi _ { \operatorname* { m i n } } > 0$ for all $a \in { \mathcal { A } }$ and $o \in \Omega$

We then use predictive beliefs to express the trajectory likelihood in two equivalent forms: one as a product over individual observations and the other as a product over consecutive observation pairs.

Proposition 3.3. F $\begin{array} { r } { r \theta = ( \beta , \varphi ) , \ell _ { n } ( \theta ) = \sum _ { i = 1 } ^ { n } \log \left( \sum _ { s \in S } \bar { b } _ { i , \theta } ( s ) O _ { \varphi } ( o _ { i } \mid s ) \right) } \end{array}$

Proof. See Appendix B.

Proposition 3.4. Assume n is even for notational simplicity. Then, for any $\theta = ( \beta , \varphi )$ , we have $\begin{array} { r } { \ell _ { n } ( \theta ) = \sum _ { k = 1 } ^ { n / 2 } \log \Big ( \sum _ { s \in \mathcal { S } } \bar { b } _ { 2 k - 1 , \theta } ( s ) O _ { \varphi } ( o _ { 2 k - 1 } \mid s ) \sum _ { s ^ { \prime } \in \mathcal { S } } T _ { \beta } ^ { a _ { 2 k - 1 } } ( s ^ { \prime } \mid s ) O _ { \varphi } ( o _ { 2 k } \mid s ^ { \prime } ) \Big ) } \end{array}$

Proof. See Appendix C.

The one-step and two-step likelihood representations allow us to relate likelihood gaps to discrepancies between the corresponding predictive distributions. Specifically, we establish two high-probability inequalities showing that the cumulative squared discrepancy between the candidate and ground-truth predictive distributions is bounded by the log-likelihood gap plus a statistical error term.

Lemma 3.5. For any candidate parameter $\hat { \theta } = ( \hat { \beta } , \hat { \varphi } )$ , there exists a universal constant $c > 0$ such that for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$ , it holds that

$$
\sum _ { i = 1 } ^ { n } \left( \sum _ { o } \left| \sum _ { s } \bar { b } _ { i , \hat { \theta } } ( s ) O _ { \hat { \varphi } } ( o | s ) - \sum _ { s } \bar { b } _ { i , \theta ^ { * } } ( s ) O _ { \varphi ^ { * } } ( o | s ) \right| \right) ^ { 2 } \leq 4 \left( \ell _ { n } ( \theta ^ { * } ) - \ell _ { n } ( \hat { \theta } ) + 2 \log ( 1 / \delta ) \right) .\tag{1}
$$

Proof. See Appendix D.

Lemma 3.6. For any candidate parameter $\hat { \theta } = ( \hat { \beta } , \hat { \varphi } )$ , for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$ , it holds that

$$
\begin{array} { r l } & { ~ \displaystyle \sum _ { k = 1 } ^ { n } \Biggl ( \sum _ { o , \sigma ^ { \prime } } \Biggl | \sum _ { s \in \mathcal { S } } \bar { b } _ { k , \hat { \theta } } ( s ) \mathbb { P } _ { \hat { \theta } } ( o , o ^ { \prime } \mid s , a _ { k } ) - \sum _ { s \in \mathcal { S } } \bar { b } _ { k , \theta ^ { * } } ( s ) \mathbb { P } _ { \theta ^ { * } } ( o , o ^ { \prime } \mid s , a _ { k } ) \Biggr | \Biggr ) ^ { 2 } } \\ & { ~ \leq \frac { 8 } { \pi _ { \operatorname* { m i n } } } \left( \ell _ { n } ( \theta ^ { * } ) - \ell _ { n } ( \hat { \theta } ) + 2 \log ( 2 / \delta ) \right) + 4 , } \\ & { \mathbb { P } _ { \theta } ( o , o ^ { \prime } \mid s , a _ { k } = a ) = O _ { \varphi } ( o \mid s ) \sum _ { s ^ { \prime } \in \mathcal { S } } T _ { \beta } ^ { a } ( s ^ { \prime } \mid s ) O _ { \varphi } ( o ^ { \prime } \mid s ^ { \prime } ) . } \end{array}\tag{2}
$$

Proof. See Appendix F.

Remark 3.7. Lemmas 3.5 and 3.6 provide pointwise guarantees. Their uniform counterparts are given in Appendices E and G, respectively.

Lemma 3.5 and 3.6 relate likelihood gaps to the cumulative discrepancies between one- and two-step predictive observation distributions. We next translate these bounds into recovery guarantees for the underlying POMDP kernels. This step requires identifiability, as distinct kernels can otherwise induce identical distributions over observed trajectories.

For $A \in \mathbb { R } ^ { m \times n }$ let $\sigma _ { \mathrm { m i n } } ( A )$ be the min $\{ m , n \}$ -th largest singular value of $A$

Assumption 3.8. There exists $\sigma _ { O } > 0$ such that $\sigma _ { \mathrm { m i n } } ( O _ { \varphi ^ { * } } ) \geq \sigma _ { O }$

This is the α-weakly revealing condition of Liu et al. [2022], with $\alpha = \sigma _ { O }$ . It ensures that $O _ { \varphi ^ { * } }$ has full column rank and admits a well-conditioned left inverse, which is possible only when $| \boldsymbol { S } | \le | \boldsymbol { \Omega } |$ Consequently, distinct latent-state distributions induce distinct observation distributions. Our analysis additionally relies on a nondegeneracy condition on the transition kernel, introduced next.

Assumption 3.9. There exists $\sigma _ { T } > 0$ such that, for every $a \in \mathcal { A } , \sigma _ { \operatorname* { m i n } } ( T _ { \beta ^ { * } } ^ { a } ) \geq \sigma _ { T }$

Remark 3.10. Assumption 3.9 can be relaxed to an aggregate conditioning requirement on the transition kernel. For clarify, we retain Assumption 3.9 in the main text and present the relaxed condition and corresponding proof modifications in Appendix L.

We next investigate how closeness of predictive distributions translates into recovery guarantees for the underlying POMDP kernels. To this end, we collect the predictive beliefs along the trajectory into empirical belief matrices. Define $\bar { B } : = \left[ \bar { b } _ { 1 , \theta ^ { * } } \ \cdot \cdot \cdot \ \bar { b } _ { n , \theta ^ { * } } \right] ^ { \top } , \widehat { B } : = \left[ \bar { b } _ { 1 , \hat { \theta } } \ \cdot \cdot \cdot \ \bar { b } _ { n , \hat { \theta } } \right] ^ { \top }$ . Furthermore, for each action $a \in { \mathcal { A } } ,$ , let $\mathcal { T } _ { a } : = \{ t | a _ { t } = a \} , \mathcal { N } _ { a } : = | \mathcal { T } _ { a } |$ , and define the corresponding action-restricted belief matrices $\bar { B } _ { a } : = \left[ \bar { b } _ { \mathcal { T } _ { a } ( 1 ) , \theta ^ { * } } , \therefore , \bar { b } _ { \mathcal { T } _ { a } ( \mathcal { N } _ { a } ) , \theta ^ { * } } \right] ^ { \top } , \widehat { B } _ { a } : = \left[ \bar { b } _ { \mathcal { T } _ { a } ( 1 ) , \hat { \theta } } , \cdot \cdot \cdot , \bar { b } _ { \mathcal { T } _ { a } ( \mathcal { N } _ { a } ) , \hat { \theta } } \right] ^ { \top }$

Well-conditioned left inverses for $\bar { B }$ and ${ \bar { B } } _ { a }$ ensures that the predictive beliefs are sufficiently informative to establish kernel-recovery bounds from discrepancies in the one-step and two-step predictive distributions. Rather than imposing these conditioning requirements as separate assumptions, we derive them from two properties of the data-generating process: uniform action coverage and ergodicity of the policy-induced latent-state Markov chain. Together with the preceding structural assumptions, these properties ensure sufficient state-action coverage and, after burn-in, well-conditioned belief matrices with high probability. We define ergodicity in the following assumption.

Assumption 3.11. The Markov chain induced by π with the transition kernel $\mathbb { P } _ { \theta ^ { * } } ^ { \pi } ( s ^ { \prime } \mid s ) =$ $\begin{array} { r } { \sum _ { o \in \Omega } \hat { O } _ { \varphi ^ { * } } ( o \mid s ) \sum _ { a \in \mathcal { A } } \pi ( a \mid o ) T _ { \beta ^ { * } } ^ { a } ( s ^ { \prime } \mid s ) } \end{array}$ is ergodic. Let $d ^ { \pi }$ denote the unique stationary distribution of this chain, and define $\begin{array} { r } { d _ { \operatorname* { m i n } } ^ { \pi } : = \operatorname* { m i n } _ { s \in \cal S } d ^ { \pi } ( s ) > 0 } \end{array}$

Assumption 3.11 ensures geometric convergence of the policy-induced latent-state Markov chain to stationarity. We state the corresponding mixing bound below (see Kontorovich and Weiss [2014]).

Corollary 3.12. Let $\| \mu - \nu \| _ { \mathrm { T V } } = \operatorname* { s u p } _ { A \subset S } | \mu ( A ) - \nu ( A ) |$ . Then, there exist constants $1 \leq G ( \pi ) <$ ∞ and $0 < \rho ( \pi ) < 1$ such that for every initial state $s \in S$ and every integer $t \geq 1$

$$
\| \mathbb { P } _ { \theta ^ { * } } ^ { \pi } ( s _ { t } \in \cdot \vert s _ { 1 } = s ) - d ^ { \pi } \| _ { \mathrm { T V } } \leq G ( \pi ) \rho ( \pi ) ^ { t - 1 } ,\tag{3}
$$

where G(π) and $\rho ( \pi )$ characterize geometric mixing ofthe policy-induced Markov chain.

We discuss the above assumptions and their practical implications in Appendix H, including which can be relaxed and which are technical artifacts of the analysis rather than intrinsic to the setting.

We now combine the structural, coverage, and mixing conditions to establish quantitative conditioning of the action-restricted predictive-belief matrices. Assumptions 3.8 and 3.9 provide separation and invertibility, while Assumption 3.2 ensures action coverage and Assumption 3.11 together with Corollary 3.12 guarantees state coverage and geometric mixing. These conditions yield the following finite-sample lower bound on the smallest singular value of these matrices.

Lemma 3.13 (Informal). Suppose that Assumptions 3.2, 3.8, 3.9, and 3.11 hold. For any $\delta \in ( 0 , 1 )$ ifn is bounded below by a universal polynomial with nonnegative exponents in

$$
| S | , ~ | \Omega | , ~ \frac { 1 } { \pi _ { \mathrm { m i n } } } , ~ \frac { 1 } { d _ { \mathrm { m i n } } ^ { \pi } } , ~ \frac { 1 } { \sigma _ { T } } , ~ \frac { 1 } { \sigma _ { O } } , ~ \log ( 1 / \delta ) , ~ \log ( G ( \pi ) ) , ~ \frac { 1 } { - \log ( \rho ( \pi ) ) } ,
$$

then with probability at least $1 - \delta$ there exists a universal constant c such thatfor all $a \in { \mathcal { A } }$

$$
\begin{array} { r } { \sigma _ { \operatorname* { m i n } } ( \bar { B } _ { a } ) ^ { 2 } \geq c ( n - t _ { \mathrm { b u r n } } ) | A | \pi _ { \operatorname* { m i n } } ^ { 2 } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } ( d _ { \operatorname* { m i n } } ^ { \pi } ) ^ { 2 } . } \end{array}
$$

With sufficient post-burn-in samples, the action-restricted predictive-beliefmatrices are uniformly well conditioned with high probability, with smallest singular values scaling optimally as $\sqrt { n }$

Proof. See Appendix J for the formal lemma statement and proof.

Lemmas 3.5 and 3.6 bound discrepancies between candidate and ground-truth predictive distributions via the likelihood gap, while Lemma 3.13 establishes quantitative conditioning of the action-restricted predictive-belief matrices. Together, they yield a finite-sample POMDP kernel-recovery guarantee with explicit dependence on the likelihood gap, kernel conditioning, sample size, and behavior policy.

Theorem 3.14. Given any parameter ${ \widehat { \theta } } ,$ suppose Assumptions 3.2, 3.8, 3.9, and 3.11 hold.

Fix a confidence level $\delta \in ( 0 , 1 )$ and define $d = ( | A | | S | + 1 ) ( | S | - 1 ) + | S | ( | \Omega | - 1 ) , H _ { n } ( \delta ) =$ $G _ { n } + 2 \bar { d } \log ( 2 | \Omega | n ) + 2 \log ( 4 / \delta ) + 2 .$ . If n is bounded below by a universal polynomial, with nonnegative exponents, in

$$
| A | , \ { \frac { 1 } { \pi _ { \operatorname* { m i n } } } } , \ { \frac { 1 } { d _ { \operatorname* { m i n } } ^ { \pi } } } , \ { \frac { 1 } { \sigma _ { T } } } , \ { \frac { 1 } { \sigma _ { O } } } , \ | S | , \ | \Omega | , \ H _ { n } ( \delta ) , \ \log ( G ( \pi ) ) , \ { \frac { 1 } { - \log ( \rho ( \pi ) ) } } ,\tag{4}
$$

then with probability at least $1 - \delta ,$ the following holds simultaneously for all $\hat { \theta } = ( \hat { \beta } , \hat { \varphi } ) \in \Theta$ satisfying $\ell _ { n } ( \theta ^ { * } ) - \ell _ { n } ( { \hat { \theta } } ) \leq G _ { n } .$ : there exists a permutation matrix P such that

$$
\| O _ { \varphi ^ { * } } P - O _ { \hat { \rho } } \| _ { 1 , \mathrm { e n t } } \leq C _ { O } \sqrt { \frac { H _ { n } ( \delta ) } { n } } , \qquad \Big \| P ^ { \top } T _ { \beta ^ { * } } ^ { a } P - T _ { \hat { \beta } } ^ { a } \Big \| _ { 1 , \mathrm { e n t } } \leq C _ { T } \sqrt { \frac { H _ { n } ( \delta ) } { n } } ,
$$

for every $a \in A ,$ where for a matrix A, we define $\begin{array} { r } { \| A \| _ { 1 , \mathrm { e n t } } : = \sum _ { i } \sum _ { i } | A _ { i j } | . } \end{array}$ . Here $C _ { O } , C _ { T }$ are problem-dependent constants that depend polynomially on the first seven quantities in (4).

In particular, a sufficiently small likelihood gap guarantees accurate recovery ofthe POMDP kernels, up to a common permutation of the latent states and uniformly over the parameter space, with high probability.

Proof. See Appendix K for the formal theorem statement and proof.

With observed rewards, the same latent-state permutation aligns the reward kernel, as follows.

Corollary 3.15. Under the assumptions of Theorem 3.14, suppose additionally that rewards are observed, take values in afinite alphabet, and are conditionally independent ofobservations given the state and action. Assume further that <sup>ˆ</sup>θ satisfies the corresponding likelihood-gap condition for the joint observation–action–reward data. Then, with probability at least $1 - \delta ,$ , there exists a permutation matrix P satisfying the conclusions of Theorem $3 . I 4$ and the additional bound

$$
\operatorname* { m a x } _ { a \in \mathcal { A } } \Vert R _ { \theta ^ { * } } ^ { a } P - R _ { \hat { \theta } } ^ { a } \Vert _ { 1 , \mathrm { e n t } } \leq C _ { R } \sqrt { \frac { H _ { n } ( \delta ) } { n } } ,
$$

where $R _ { \theta } ^ { a } { \mathrm { , } }$ <sub>∗</sub> and $R _ { \hat { \theta } } ^ { a }$ denote the true and estimated conditional reward kernelsfor action a, respectively. The constant $C _ { R }$ depends on the problem parameters analogously to $C _ { O }$ and $C _ { T }$ as well as the to size ofthe reward alphabet.

## Proof. See Appendix M.

Thus, the observation and transition kernels, together with the relevant reward quantities, are recovered up to a common permutation of the latent states, providing a consistent model for planning.

Furthermore, our recovery guarantee does not require optimization over the unrestricted POMDP class. In particular, prior structural knowledge can be incorporated by restricting the candidate model class, provided that the true model remains feasible. This yields the following immediate corollary.

Corollary 3.16. Let $\Theta _ { \mathrm { s t r } } \subseteq \Theta$ be any structured $P O M D P$ class containing the true parameter $\theta ^ { * }$ where Θ is the unrestricted model class. Then any $ { \hat { \theta } } \in \Theta _ { \mathrm { s t r } }$ <sub>r</sub> satisfying the likelihood-gap condition of Theorem 3.14 enjoys the same likelihood-to-parameter recovery guarantee.

Thus, known structural constraints may be imposed on the candidate class without modifying our recovery analysis (see Appendix N for an example).

## 3.2 Overcomplete POMDPs $( | S | > | \Omega | )$

For overcomplete POMDPs, we extend the framework in Section 3.1 by replacing one-step observability with its multi-step analogue. The main proof strategy, including likelihood decomposition, the relationship between likelihood gaps and predictive-distribution discrepancies, and predictivebelief conditioning, remains applicable. We therefore focus on modified assumptions and present streamlined bounds, suppressing polynomial dependencies analogous to those in Section 3.1.

Assumption 3.8 cannot hold in the overcomplete regime, since $O _ { \varphi ^ { \ast } } \in \mathbb { R } ^ { | \Omega | \times | S | }$ cannot have full column rank when $| S | > | \Omega |$

For $m \in \mathbb { N }$ and a prescribed action sequence w $\in \mathcal { A } ^ { m - 1 }$ , define the corresponding m-step observation kernel $[ \mathcal { M } _ { \theta } ^ { ( m ) } [ w ] ]$ under $\theta = ( \beta , \varphi ) \colon [ \mathcal { M } _ { \theta } ^ { ( m ) } [ w ] ] ( z , s ) = \mathbb { P } _ { \theta } \big ( o _ { 1 : m } = z \mid s _ { 1 } = s , a _ { 1 : m - 1 } = w \big )$ , for $z \in \Omega ^ { m }$ and $s \in S$ . This leads to the following multi-step identifiability assumption.

Assumption 3.17. There exist $m \in \mathbb { N }$ and $\sigma _ { M } ~ > ~ 0$ such that the set $\mathcal { W } : = \{ w \in \mathcal { A } ^ { m - 1 }$ $\sigma _ { \operatorname* { m i n } } ( \mathcal { M } _ { \theta ^ { * } } ^ { ( m ) } [ w ] ) \geq \sigma _ { M } \}$ is nonempty.

In the rest of Section 3.2, we assume that Assumption 3.17 holds for a fixed $m .$ . It requires a prescribed action sequence w $\in \mathcal { A } ^ { m - 1 }$ for which $\mathcal { M } _ { \theta ^ { * } } ^ { ( m ) } [ w ]$ admits a well-conditioned left inverse. Since $\mathcal { M } _ { \theta ^ { \ast } } ^ { ( m ) } [ w ] \in \mathbb { R } ^ { | \Omega | ^ { m } \times | S | }$ , this requires $| \Omega | ^ { m } \geq | S |$ . Under such an action sequence, distinct latent-state distributions induce discint distributions over length-m observation blocks, even if indistinguishable from one-step observations. When $m = 1$ , this reduces to Assumption 3.8.

Let ⊕ denote concatenation of action strings, and define $\mathcal { A } \oplus \mathcal { A } \oplus \mathcal { W } \oplus \mathcal { A }$ as the set of all strings $a _ { 1 } \oplus a _ { 2 } \oplus w \oplus a ^ { \prime }$ , where $a _ { 1 } , a _ { 2 } , a ^ { \prime } \in \mathcal { A }$ and $w \in \mathcal { W }$ . We formalize the behavior policy below.

Assumption 3.18. π is blockwise: at the start of each block, it samples a string $( a _ { 1 } , a _ { 2 } , w , a ^ { \prime } ) \in$ $\mathcal { A } \oplus \bar { \mathcal { A } } \oplus \mathcal { W } \oplus \mathcal { A }$ from π and executes it throughout the block. Moreover, there exists $\pi _ { \mathrm { m i n } } ^ { \mathrm { b l k } } \ > \ 0$ such that for every $a _ { 1 } \in \ A .$ , there exists $a _ { 2 } \in \mathsf { \Gamma } .$ such that the set $\begin{array} { r l } { \mathcal { W } _ { \bar { a } } } & { { } : = } \end{array}$ $\begin{array} { r } { \left\{ \overline { { w } } \in \mathcal { W } : \sum _ { a ^ { \prime } \in \mathcal { A } } \pi ( a _ { 1 } \oplus a _ { 2 } \oplus w \oplus a ^ { \prime } ) \geq \pi _ { \operatorname* { m i n } } ^ { \mathrm { b l k } } \right\} } \end{array}$ is nonempty, where we denote $( a _ { 1 } , a _ { 2 } )$ as a¯.

Assumption 3.18 ensures that, for every $a _ { 1 } \in { \mathcal { A } } .$ , there exist $a _ { 2 } \in { \mathcal { A } }$ and $w \in \mathcal { W }$ such that the behavior policy samples a block beginning with $a _ { 1 } \oplus a _ { 2 } \oplus$ w with probability at least $\pi _ { \mathrm { m i n } } ^ { \mathrm { b l k } } > 0$

Remark 3.19. Assumption 3.18 can be relaxed to allow action strings outside W, provided the required informative strings are sampled with probability bounded below by a positive constant. We retain the stronger assumption for simplicity of exposition and proof.

The following proposition decomposes the trajectory likelihood into consecutive blocks of possibly different lengths, expressing each factor using the multi-step observation kernels introduced above.

Proposition 3.20. For any $\theta = ( \beta , \varphi )$ , let $1 = \tau _ { 1 } < \tau _ { 2 } < \cdot \cdot \cdot < \tau _ { N + 1 } = n + 1$ be a sequence ofcut points, and define $m _ { i } = \tau _ { i + 1 } - \tau _ { i } f o r$ each $i \in [ N ]$ . Then

$$
\ell _ { n } ( \theta ) = \sum _ { i = 1 } ^ { N } \log \left( \sum _ { s \in \mathcal { S } } \bar { b } _ { \tau _ { i } , \theta } ( s ) \left[ \mathcal { M } _ { \theta } ^ { ( m _ { i } ) } \big [ a _ { \tau _ { i } : \tau _ { i + 1 } - 2 } \big ] \right] \left( o _ { \tau _ { i } : \tau _ { i + 1 } - 1 } , s \right) \right) .
$$

Here, when $m _ { i } = 1$ , the action sequence $a _ { \tau _ { i } : \tau _ { i + 1 } - 2 }$ is understood as the empty sequence.

Proof. The result follows by the same argument used to derive the two-step factorization in Proposition 3.4 from the one-step factorization in Proposition 3.3. Specifically, grouping consecutive one-step likelihood factors into blocks of lengths $m _ { i }$ yields the claimed expression, with each block represented by its corresponding m<sub>i</sub>-step observation kernel. □

We now extend Lemmas 3.5 and 3.6 to multi-step observation blocks.

Lemma 3.21. For each $i \in [ N ] ,$ , let $t _ { i } = 1 + ( i - 1 ) ( m + 2 )$ and define the block predictive distribution $\begin{array} { r } { p _ { t _ { i } , \theta } ( x ) : = \sum _ { s \in S } \bar { b } _ { t _ { i } , \theta } ( s ) \left[ \mathcal { M } _ { \theta } ^ { ( m + 2 ) } [ a _ { t _ { i } : t _ { i } + m } ] \right] ( x , s ) , x \in \Omega ^ { m + 2 } } \end{array}$ , where $N = \lfloor n / ( m + 2 ) \rfloor$ . Then, given a candidate parameter $\hat { \theta } = ( \hat { \beta } , \hat { \varphi } )$ , for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$

$$
\sum _ { i = 1 } ^ { N } \| p _ { t _ { i } , \hat { \theta } } - p _ { t _ { i } , \theta ^ { * } } \| _ { 1 } ^ { 2 } \leq \frac { 8 } { \pi _ { \operatorname* { m i n } } ^ { \mathrm { b l k } } } \left( \ell _ { n } ( \theta ^ { * } ) - \ell _ { n } ( \hat { \theta } ) + 2 \log ( 2 / \delta ) \right) + 4 .
$$

Proof. The claim follows by applying the argument of Lemma 3.6 to the blockwise likelihood factorization in Proposition 3.20. □

Under Assumption 3.18, the blockwise behavior policy induces a Markov chain on the latent states at block boundaries, $\left( s _ { 1 + k \left( m + 2 \right) } \right) _ { k \geq 0 }$ . We impose the following ergodicity condition on this chain.

Assumption 3.22. The block-level Markov chain induced by π is ergodic with stationary distribution $d _ { \mathrm { b l k } } ^ { \pi }$ . Define $\begin{array} { r } { \bigl ( d _ { \mathrm { b l k } } ^ { \pi } \bigr ) _ { \mathrm { m i n } } : = \operatorname* { m i n } _ { s \in \cal S } d _ { \mathrm { b l k } } ^ { \pi } \mathopen { } \mathclose \bgroup \left( s \aftergroup \egroup \right) > 0 } \end{array}$ , and let $G _ { \mathrm { b l k } } ( \pi )$ and $\rho _ { \mathrm { b l k } } ( \pi )$ denote its geometric mixing constants, defined analogously to those in Corollary 3.12.

Remark 3.23. Assumptions 3.18 and 3.22 are sufficient, rather than minimal, conditions on π. We impose them to simplify the analysis.

To identify POMDP kernels up to a permutation of the latent states, we impose an additional condition requiring distinct latent states to induce distinct observation distributions.

Assumption 3.24. We observation distributions of distinct latent states satisfy $\begin{array} { r l } { \Delta _ { O } } & { { } : = } \end{array}$ $\begin{array} { r } { \operatorname* { m i n } _ { s \neq s ^ { \prime } } \| O _ { \varphi ^ { * } } ( \cdot \mid s ) - O _ { \varphi ^ { * } } ( \cdot \mid s ^ { \prime } ) \| > 0 } \end{array}$

Appendix O shows that, without this assumption, kernel recovery can be information-theoretically impossible in general overcomplete POMDPs.

A more detailed discussion of Assumptions 3.17, 3.18, 3.22, and 3.24 is provided in Appendix H.

We next investigate the conditioning of predictive-belief matrices. To this end, we introduce their blockwise counterparts. Let $N = \lfloor \bar { n / ( m + 2 ) } \rfloor , t _ { i } = 1 + ( i - 1 ) ( m + 2 )$ , and $\bar { b } _ { i , \theta } ^ { \mathrm { b l k } } : = \bar { b } _ { t _ { i } , \theta } , i \in [ N ]$ Define $\bar { B } ^ { \mathrm { b l k } } : = [ \bar { b } _ { 1 , \theta ^ { * } } ^ { \mathrm { b l k } } ~ \cdot \cdot ~ \bar { b } _ { N , \theta ^ { * } } ^ { \mathrm { b l k } } ] ^ { \top }$ and $\widehat { B } ^ { \mathrm { b l k } } : = [ \bar { b } _ { 1 , \widehat { \theta } } ^ { \mathrm { b l k } } ~ \cdot \cdot \cdot ~ \bar { b } _ { N , \widehat { \theta } } ^ { \mathrm { b l k } } ] ^ { \top }$ . For each $\tilde { a } = ( a _ { 1 } , a _ { 2 } ) \in \mathcal { A } \oplus \mathcal { A }$ and $w \in \mathcal { W }$ , let $\mathcal { Z } _ { \bar { a } , w } ^ { \mathrm { b l k } } : = \{ i \in [ k ] : ( a _ { t _ { i } } , a _ { t _ { i } + 1 } , \ldots , a _ { t _ { i } + m } ) = ( a _ { 1 } , a _ { 2 } , w ) \} , N _ { \bar { a } , w } ^ { \mathrm { b l k } } : = | \mathcal { Z } _ { \bar { a } , w } ^ { \mathrm { b l k } } |$ , and $\bar { B } _ { \tilde { a } , w } ^ { \mathrm { b l k } } : = [ \bar { b } _ { \mathcal { T } _ { \tilde { a } , w } ^ { \mathrm { b l k } } ( 1 ) , \theta ^ { * } } ^ { \mathrm { b l k } } \ \cdot \ \cdot \ \cdot \ \ \bar { b } _ { \mathcal { T } _ { \tilde { a } , w } ^ { \mathrm { b l k } } ( N _ { \tilde { a } , w } ^ { \mathrm { b l k } } ) , \theta ^ { * } } ^ { \mathrm { b l k } } ] ^ { \top }$ . The matrix $\widehat { B } _ { \tilde { a } , w } ^ { \mathrm { b l k } }$ is defined analogously.

The conditioning of the block-level matrices is central to our subsequent analysis. However, the assumptions introduced thus far are insufficient to guarantee polynomial sample complexity for kernel recovery, as demonstrated by the following proposition.

Proposition 3.25. There exist overcomplete POMDPs satisfying Assumptions 3.9, 3.17, 3.18, 3.22, $3 . 2 4 ,$ such that learning their transition and observation kernels to accuracy $O ( 1 / | S | )$ , up to a common state permutation, requires $\exp ( \Omega ( | S | ) )$ observations.

## Proof. See Appendix P.

The pathological instances underlying Proposition 3.25 exhibit exponentially ill-conditioned backward observability: certain latent-state directions are nearly indistinguishable from past observation histories. To rule out such instances, let $\rho$ be a distribution over $\mathcal { A } ^ { \breve { L } }$ and define

$$
\Phi _ { L } ^ { \rho } ( x ) : = \sum _ { a } \rho ( a ) \sum _ { y \in \Omega ^ { L } } \left| x ^ { \top } K _ { a , y } d _ { \mathrm { b l k } } ^ { \pi } \right| , \qquad K _ { a , y } : = T _ { \beta ^ { * } } ^ { a _ { L } } D _ { o _ { L } } \cdot \cdot \cdot T _ { \beta ^ { * } } ^ { a _ { 1 } } D _ { o _ { 1 } } ,
$$

where $\pmb { a } \in \mathcal { A } ^ { L } \sim \rho$ is a block of L actions, $y = \left( o _ { 1 } , \ldots , o _ { L } \right)$ , and $D _ { o } = \mathrm { d i a g } ( O _ { \varphi ^ { * } } ( o \mid \cdot ) )$ . The quantity $\Phi _ { L } ^ { \rho } ( x )$ measures how strongly length-L observation histories excite a latent-state direction x when the corresponding action blocks are sampled according to $\rho .$

For $L = q H$ , where $H = m + 2 ,$ , let $\begin{array} { r } { \rho _ { L } ^ { \pi } ( \pmb { a } ) : = \prod _ { j = 0 } ^ { q - 1 } \pi ( a _ { j H + 1 } , \dots , a _ { ( j + 1 ) H } ) } \end{array}$ , denote the length-L action-string distribution induced by $\pi .$ . We write $\Phi _ { L } ^ { \pi } : = \Phi _ { L } ^ { \rho _ { L } ^ { \pi } }$ . We may restrict to $L \in H \mathbb { N } .$ , since $\Phi _ { L } ^ { \pi } ( x )$ is nondecreasing in L (see Appendix Q). Then, we impose the following assumption.

Assumption 3.26 (Quantitative backward excitation). For fixed universal constants $c , C > 0$ , there exists a horizon $L \in ( m + 2 ) \mathbb { N }$ with $L \leq | S | ^ { C }$ such that $\Phi _ { L } ^ { \pi } ( x ) \geq c \vert S \vert ^ { - C } \Vert x \Vert _ { 2 } , \forall x \in \mathbb { R } ^ { \vert S \vert }$

We discuss the assumption further in Appendix H. The following proposition establishes its sufficiency for polynomial sample complexity in kernel recovery and shows that, without it, the remaining assumptions permit instances with exponential sample complexity.

Proposition 3.27. Suppose Assumptions 3.9, 3.17, 3.18, 3.22, 3.24, and 3.26 hold, with all relevant model parameters bounded by $\mathrm { p o l y } ( | S | )$ . Then the kernels can be estimated, up to a common permutation of the latent states, to accuracy ε with probability at least $1 - \delta$ using a trajectory of length $\dot { \operatorname { p o l y } } ( | S | , \varepsilon ^ { - 1 } , \log ( 1 / \delta ) )$ . Conversely, without Assumption 3.26, the remaining assumptions admit instances requiring $\exp ( \Omega ( | S | ) )$ ) observations for any estimator, even at accuracy $\varepsilon = \bar { \Theta } ( | S | ^ { - 1 } )$ and constant confidence.

## Proof. See Appendix Q.

We can now establish the quantitative conditioning of the block-level predictive-belief matrices.

Lemma 3.28. Suppose Assumption 3.22 holds $f o r \quad \pi ,$ Assumption 3.26 holds $f o r ~ \pi$ with parameters L and $\eta ,$ and Assumption 3.18 holds. Fix a confidence level $\delta \in \mathsf { \Gamma }$ $( 0 , 1 ) . \ J ^ { \textit { n } }$ is bounded below by a universal polynomial with nonnegative powers in $\begin{array} { r } { | \dot { S } | , ~ | \mathcal { A } | , ~ \frac { 1 } { \pi _ { \mathrm { m i n } } ^ { \mathrm { b l k } } } } \end{array}$ , log(1/δ), log $\begin{array} { r } { ( G _ { \mathrm { b l k } } ( \dot { \pi } ) ) , \frac { 1 } { - \log ( \rho _ { \mathrm { b l k } } ( \pi ) ) } } \end{array}$ , L, 1/η, then, with probability at least $1 - \delta ,$ , there exists a universal constant $c ^ { \mathrm { b l k } }$ such that for every $a _ { 1 } \in { \mathcal { A } } ,$ , there exists $a _ { 2 } \in { \mathcal { A } }$ and $w \in \mathcal { W } _ { \tilde { a } } ^ { \pi }$ , where $\tilde { a } = ( a _ { 1 } , a _ { 2 } )$ , satisfying $\sigma _ { \mathrm { m i n } } ^ { 2 } ( \bar { B } _ { \tilde { a } , w } ^ { \mathrm { b l k } } ) \geq c ^ { \mathrm { b l k } } \pi _ { \mathrm { m i n } } ^ { \mathrm { b l k } } \frac { \eta ^ { 2 } } { L } n$

## Proof. See Appendix R.

We now have the necessary ingredients to extend the kernel-recovery guarantee of Theorem 3.14 to overcomplete POMDPs.

Theorem 3.29. Suppose Assumptions $3 . 9 , 3 . 1 7 , 3 . 1 8 , 3 . 2 2 , 3 . 2 4 ,$ and 3.26 hold. Fix a confidence level $\delta \in ( 0 , 1 )$ , and define $d = ( \hat { | } A | | S | + 1 ) ( | S | - 1 ) + | S | ( | \Omega | - 1 ) , H _ { n } ( \delta ) = G _ { n } + 2 d \log ( 2 | \Omega | n ) +$ $2 \log ( 4 / \delta ) \stackrel { \cdot } { + } 2 . \ i$ Ifn is bounded below by a universal polynomial, with nonnegative powers, in

$$
| \mathcal { A } | , \frac { 1 } { \pi _ { \mathrm { m i n } } ^ { \mathrm { b l k } } } , \frac { 1 } { ( d _ { \mathrm { m i n } } ^ { \pi } ) ^ { \mathrm { b l k } } } , \frac { 1 } { \sigma _ { T } } , \frac { 1 } { \sigma _ { M } } , \frac { 1 } { \Delta _ { O } } , | \mathcal { S } | , | \Omega | , \frac { L } { \eta ^ { 2 } } , H _ { n } ( \delta ) , \log ( G _ { \mathrm { b l k } } ( \pi ) ) , \frac { 1 } { - \log ( \rho _ { \mathrm { b l k } } ( \pi ) ) } , \frac { 1 } { \rho _ { M } ( \delta ) } , \frac { 1 } { \log ( \rho _ { \mathrm { b l k } } ( \pi ) ) } .\tag{5}
$$

then with probability at least $1 - \delta ,$ , the following holds simultaneously for all $\hat { \theta } = ( \hat { \beta } , \hat { \varphi } ) \in \Theta$ satisfying $\ell _ { n } ( \theta ^ { * } ) - \ell _ { n } ( { \hat { \theta } } ) \leq G _ { n } .$ : there exists a permutation matrix P such that for every $a \in { \mathcal { A } } ,$

$$
\| O _ { \varphi ^ { * } } P - O _ { \hat { \varphi } } \| _ { 1 , \mathrm { e n t } } = O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) , \qquad \left\| P ^ { \top } T _ { \beta ^ { * } } ^ { a } P - T _ { \hat { \beta } } ^ { a } \right\| _ { 1 , \mathrm { e n t } } = O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) ,
$$

where the constants in $O ( \cdot )$ are bounded by a universal polynomial in the first nine quantities in (5).

Proof. See Appendix S.

Thus, despite the failure of one-step observability in overcomplete POMDPs, likelihood-based kernel recovery remains polynomially sample-efficient under multi-step observability, blockwise coverage and ergodicity, pairwise distinct observation columns, and inverse-polynomial distinguishability of latent-state directions from past observations.

## 4 Discussion

In this work, we establish finite-sample guarantees that translate likelihood gaps into parameter recovery error bounds. By combining trajectory likelihood decompositions with quantitative conditioning bounds for predictive-belief matrices, our analysis covers both undercomplete and overcomplete POMDPs. To the best of our knowledge, such finite-sample guarantees have not previously been established in the likelihood-based literature, while existing spectral methods have not established finite-sample guarantees for full kernel recovery in the overcomplete regime. Furthermore, our guarantees apply to any parameter with a controlled likelihood gap, regardless of how it is obtained. Thus, they can be combined with POMDP learning algorithms that produce such estimates.

## 5 Limitations and future work

Our analysis adapts the observation-kernel assumptions from the undercomplete to the overcomplete regime, while retaining the same full-rank condition on the transition kernels in both. Investigating whether this transition-kernel condition can be weakened is an important direction for future work.

As noted in Remark 3.23, Assumptions 3.18 and 3.22 are sufficient, rather than minimal, conditions for extending the undercomplete analysis to the overcomplete regime. Characterizing weaker behaviorpolicy assumptions under which likelihood gaps still yield parameter recovery guarantees remains an important direction for future work. This is particularly relevant in the overcomplete setting, where the behavior policy plays a more active role: by selecting informative action sequences, it can help distinguish latent-state mixtures that cannot be separated through one-step observations alone.

While our theoretical results identify key properties of effective behavior policies, they do not provide a tractable procedure for selecting them. In practice, transition and observation kernels are unknown, motivating robust policy design over uncertainty sets of plausible models. Developing efficient algorithms that balance action coverage, stationary state coverage, and mixing, while accounting for informative action blocks in the overcomplete regime, remains an important direction for future work.

A separate practical consideration arises when latent states have semantic labels and a known, externally supplied reward kernel is tied to those labels. Since our results recover the model only up to a permutation of the latent states, the recovered kernels must be aligned with the external labeling. Such alignment is possible when states have known or approximately known transition or observation signatures, such as distinctive sensor patterns or transition behavior, that allow the recovered states to be matched to their semantic labels. Exact signatures are not required; sufficiently separated signatures or observations strongly associated with particular states may suffice.

Although Section 1 highlights the benefits of POMDP kernel recovery for latent-state inference, forecasting, simulation, and downstream tasks, it remains unclear whether full recovery is necessary for near-optimal policy learning in infinite-horizon POMDPs. Existing guarantees in the infinitehorizon setting typically rely on accurate estimation of the transition and observation kernels. If full recovery is necessary in this setting, it would be valuable to characterize subclasses where weaker forms of model identification suffice; otherwise, an important direction is to develop near-optimal algorithms that circumvent full kernel recovery for infinite-horizon POMDPs.

## Acknowledgments and Disclosure of Funding

This research was partially supported by the AFOSR grant FA9550-23-1-0190.

The authors declare no competing interests.

## References

Animashree Anandkumar, Daniel Hsu, and Sham M Kakade. A method of moments for mixture models and hidden Markov models. In Conference on learning theory, pages 33–1. JMLR Workshop and Conference Proceedings, 2012.

Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. The Journal of Machine Learning Research, 15(1):2773–2832, 2014.

Kamyar Azizzadenesheli, Alessandro Lazaric, and Animashree Anandkumar. Reinforcement learning of POMDPs using spectral methods. In Conference on Learning Theory, pages 193–256. PMLR, 2016a.

Kamyar Azizzadenesheli, Alessandro Lazaric, and Animashree Anandkumar. Reinforcement learning in rich-observation MDPs using spectral methods. arXiv preprint arXiv:1611.03907, 2016b.

Leonard E Baum, Ted Petrie, George Soules, and Norman Weiss. A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. The annals ofmathematical statistics, 41(1):164–171, 1970.

H. C. P. Berbee. Random Walks with Stationary Increments and Renewal Theory, volume 112 of Mathematical Centre Tracts. Mathematisch Centrum, Amsterdam, 1979.

Aditya Bhaskara, Moses Charikar, and Aravindan Vijayaraghavan. Uniqueness of tensor decompositions with applications to polynomial identifiability. In Conference on Learning Theory, pages 742–778. PMLR, 2014.

Peter J Bickel, Ya’acov Ritov, and Tobias Ryden. Asymptotic normality of the maximum-likelihood estimator for general hidden Markov models. The Annals of Statistics, 26(4):1614–1635, 1998.

Craig Boutilier. A POMDP formulation of preference elicitation problems. In AAAI/IAAI, pages 239–246. Edmonton, AB, 2002.

Richard C. Bradley. Basic properties of strong mixing conditions. a survey and some open questions. Probability Surveys, 2:107–144, 2005.

Raissa Zurli Bittencourt Bravo, Adriana Leiras, and Fernando Luiz Cyrino Oliveira. The use of uavs in humanitarian relief: An application of POMDP-based methodology for finding victims. Production and Operations Management, 28(2):421–440, 2019.

Fan Chen, Yu Bai, and Song Mei. Partially observable RL with B-stability: Unified structural condition and sharp sample-efficient algorithms. In International Conference on Learning Representations, 2023.

Randal Douc and Catherine Matias. Asymptotics of the maximum likelihood estimator for general hidden Markov models. Bernoulli, 7(3):381–420, 2001.

Randal Douc, Eric Moulines, Jimmy Olsson, and Ramon van Handel. Consistency of the maximum likelihood estimator for general hidden Markov models. The Annals ofStatistics, 39(1):474–513, 2011.

John C Duchi. Introductory lectures on stochastic optimization. The mathematics of data, 25:99–186, 2018.

Eyal Even-Dar, Sham M Kakade, and Yishay Mansour. The value of observation for monitoring dynamic systems. In IJCAI, pages 2474–2479, 2007.

Noah Golowich, Ankur Moitra, and Dhruv Rohatgi. Learning in observable POMDPs, without computationally intractable oracles. Advances in neural information processing systems, 35: 1458–1473, 2022a.

Noah Golowich, Ankur Moitra, and Dhruv Rohatgi. Planning in observable POMDPs in quasipolynomial time. arXiv preprint arXiv:2201.04735, 2022b.

Jiacheng Guo, Zihao Li, Huazheng Wang, Mengdi Wang, Zhuoran Yang, and Xuezhou Zhang. Provably efficient representation learning with tractable planning in low-rank POMDP. In International Conference on Machine Learning, pages 11967–11997. PMLR, 2023.

Jiacheng Guo, Minshuo Chen, Huan Wang, Caiming Xiong, Mengdi Wang, and Yu Bai. Sampleefficient learning of POMDPs with multiple observations in hindsight. In International Conference on Learning Representations, 2024.

Zhaohan Daniel Guo, Shayan Doroudi, and Emma Brunskill. A PAC RL algorithm for episodic POMDPs. In Artificial Intelligence and Statistics, pages 510–518. PMLR, 2016.

Daniel Hsu, Sham M Kakade, and Tong Zhang. A spectral algorithm for learning hidden Markov models. Journal ofComputer and System Sciences, 78(5):1460–1480, 2012.

Mehdi Jafarnia Jahromi, Rahul Jain, and Ashutosh Nayyar. Online learning for unknown partially observable MDPs. In International Conference on Artificial Intelligence and Statistics, pages 1712–1732. PMLR, 2022.

Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra. Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2):99–134, 1998.

Sven Koenig and Reid G Simmons. Unsupervised learning of probabilistic models for robot navigation. In Proceedings ofIEEE International Conference on Robotics and Automation, volume 3, pages 2301–2308. IEEE, 1996.

Aryeh Kontorovich and Roi Weiss. Uniform Chernoff and Dvoretzky-Kiefer-Wolfowitz-type inequalities for Markov chains and related processes. Journal ofApplied Probability, 51(4):1100–1113, 2014.

Vikram Krishnamurthy. Partially observed Markov decision processes: fromfiltering to controlled sensing. Cambridge University Press, 2016.

Jonathan Lee, Alekh Agarwal, Christoph Dann, and Tong Zhang. Learning in POMDPs is sampleefficient with hindsight observability. In International Conference on Machine Learning, pages 18733–18773. PMLR, 2023.

Luc Lehéricy. Nonasymptotic control of the MLE for misspecified nonparametric hidden Markov models. Electronic Journal ofStatistics, 15(2):4916–4965, 2021.

Brian G Leroux. Maximum-likelihood estimation for hidden Markov models. Stochastic processes and their applications, 40(1):127–143, 1992.

Luchen Li, Matthieu Komorowski, and Aldo A Faisal. The actor search tree critic (astc) for off-policy POMDP learning in medical decision making. arXiv preprint arXiv:1805.11548, 2018.

Qinghua Liu, Alan Chung, Csaba Szepesvári, and Chi Jin. When is partially observable reinforcement learning not scary? In Conference on Learning Theory, pages 5175–5220. PMLR, 2022.

Qinghua Liu, Praneeth Netrapalli, Csaba Szepesvari, and Chi Jin. Optimistic MLE: A generic model-based algorithm for partially observable sequential decision making. In Proceedings ofthe 55th Annual ACM Symposium on Theory of Computing, pages 363–376, 2023.

S Murugesan, Z Jiang, MJ Risbeck, J Amores, C Zhang, V Ramamurti, KH Drees, and YM Lee. Less is more: Simplified state-action space for deep reinforcement learning based hvac control. In Proceedings of the 1st international workshop on reinforcement learning for energy management in buildings & cities, pages 20–23, 2020.

Christos H Papadimitriou and John N Tsitsiklis. The complexity of Markov decision processes. Mathematics ofoperations research, 12(3):441–450, 1987.

Emmanuel Rio et al. Asymptotic theory of weakly dependent random processes, volume 80. Springer, 2017.

Stephane Ross, Masoumeh Izadi, Mark Mercer, and David Buckeridge. Sensitivity analysis of POMDP value functions. In 2009 International Conference on Machine Learning and Applications, pages 317–323. IEEE, 2009.

Alessio Russo, Alberto Maria Metelli, Marcello Restelli, et al. Achieving $\mathcal { O } ( \sqrt { T } )$ regret in averagereward POMDPs with known observation models. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS 2025), volume 258, pages 4168–4176, 2025.

Alessio Russo, Alberto Maria Metelli, and Marcello Restelli. Spectral learning for infinite-horizon average-reward POMDPs. Advances in Neural Information Processing Systems, 38:52330–52380, 2026.

Guy Shani, Joelle Pineau, and Robert Kaplow. A survey of point-based POMDP solvers. Autonomous Agents and Multi-Agent Systems, 27(1):1–51, 2013.

Seiji Shaw, Travis Manderson, Chad Kessens, and Nicholas Roy. Toward learning POMDPs beyond full-rank actions and state observability. In World Symposium on the Algorithmic Foundations of Robotics (WAFR), 2026.

Matthijs TJ Spaan. Partially observable Markov decision processes. In Reinforcement learning: State-of-the-art, pages 387–414. Springer, 2012.

Joel A. Tropp. User-friendly tail bounds for matrix martingales. ACM Technical Report 2011-01, California Institute of Technology, 2011.

Roman Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.

VA Volkonskii and Yu A Rozanov. Some limit theorems for random functions. i. Theory of Probability & Its Applications, 4(2):178–197, 1959.

VA Volkonskii and Yu A Rozanov. Some limit theorems for random functions. ii. Theory of Probability & Its Applications, 6(2):186–198, 1961.

Yi Xiong, Ningyuan Chen, Xuefeng Gao, and Xiang Zhou. Sublinear regret for learning POMDPs. Production and Operations Management, 31(9):3491–3504, 2022.

Fanny Yang, Sivaraman Balakrishnan, and Martin J Wainwright. Statistical and computational guarantees for the Baum-Welch algorithm. Journal ofMachine Learning Research, 18(125):1–53, 2017.

## A

Example: AI-operated warehouse-robot fleet. Consider an autonomous warehouse in which mobile robots repeatedly transport inventory while their drivetrain condition is only partially observed. At each maintenance epoch, the latent condition (state) of a robot is

$$
{ \cal { S } } = \{ H , D , C \} ,
$$

where H denotes healthy, D denotes degraded but operational, and C denotes maintenance-critical. The controller chooses among

$$
\mathcal { A } = \{ a _ { \mathrm { r e d } } , a _ { \mathrm { h i g h } } , a _ { \mathrm { s v c } } \} ,
$$

corresponding to reduced-speed operation, high-throughput operation, and service/repair. The available telemetry is discretized into

$$
\Omega = \{ \mathrm { n o m i n a l , v i b r a t i o n ~ a n o m a l y , t h e r m a l / c u r r e n t ~ a n o m a l y } , \mathrm { a m b i g u o u s } \} .
$$

The transition kernel describes action-dependent deterioration and recovery, while the observation kernel describes the vibration, temperature, motor-current, and diagnostic signatures generated by each health regime. These kernels characterize the robot and its sensing system rather than a particular control objective. By contrast, the reward can be changed to emphasize throughput, energy consumption, safety, delivery deadlines, maintenance cost, or equipment longevity.

To be more specific, because the kernels describe the underlying environment independently of the chosen reward, recovering them once supports a range of downstream objectives and analyses. First, the kernels provide a system-identification model of how operating intensity affects deterioration and how each latent health condition appears in the observable telemetry. Second, the same recovered model can be reused when the downstream reward changes, for example from peak-throughput maximization to energy-efficient operation or safety-critical maintenance. Third, one may perform sensitivity analysis by perturbing wear, repair-success, or sensor-error probabilities; policy-level counterfactual simulation by evaluating alternative service schedules or operating intensities; and model-based off-policy evaluation of a candidate policy. Thus the transition and observation kernels form a reusable description of the environment, whereas the reward specifies a particular downstream task.

## B

Recall that $a _ { 1 } ^ { n }$ is the realized action sequence under the fixed, parameter-independent behavior policy π. By the chain rule, the joint trajectory likelihood satisfies

$$
\mathbb { P } _ { \theta } \big ( o _ { 1 : n } , a _ { 1 : n } \big ) = \prod _ { i = 1 } ^ { n } \mathbb { P } _ { \theta } \big ( o _ { i } \mid o _ { 1 : i - 1 } , a _ { 1 : i - 1 } \big ) \pi \big ( a _ { i } \mid o _ { i } \big ) .
$$

Define $f _ { \theta } ( o _ { 1 : n } , a _ { 1 : n } )$ as the model-dependent component of the observed-data likelihood. Then, since the behavior policy π is independent of θ, we have

$$
f _ { \theta } \big ( o _ { 1 : n } , a _ { 1 : n } \big ) = \prod _ { i = 1 } ^ { n } \mathbb { P } _ { \theta } \big ( o _ { i } \mid o _ { 1 : i - 1 } , a _ { 1 : i - 1 } \big ) .\tag{6}
$$

Next, condition on the latent state $s _ { i }$ and marginalize:

$$
\begin{array} { l } { \displaystyle \mathbb { P } _ { \theta } \big ( o _ { i } \mid o _ { 1 : i - 1 } , a _ { 1 : i - 1 } \big ) = \sum _ { s } \mathbb { P } _ { \theta } \big ( o _ { i } , s _ { i } = s \mid o _ { 1 : i - 1 } , a _ { 1 : i - 1 } \big ) } \\ { \displaystyle \qquad = \sum _ { s } \mathbb { P } _ { \theta } \big ( s _ { i } = s \mid o _ { 1 : i - 1 } , a _ { 1 : i - 1 } \big ) \mathbb { P } _ { \theta } \big ( o _ { i } \mid s _ { i } = s , o _ { 1 : i - 1 } , a _ { 1 : i - 1 } \big ) . } \end{array}\tag{7}
$$

By the conditional independence of emissions given the current state $( { \mathrm { i . e . , ~ } } o _ { i } \bot ( o _ { 1 : i - 1 } , a _ { 1 : i - 1 } ) \mid s _ { i } )$

$$
\mathbb { P } _ { \theta } ( o _ { i } \mid s _ { i } = s , o _ { 1 : i - 1 } , a _ { 1 : i - 1 } ) = \mathbb { P } _ { \theta } ( o _ { i } \mid s _ { i } = s ) = O _ { \varphi } ( o _ { i } \mid s ) .\tag{8}
$$

Substituting (8) into (7) and using the definition $\bar { b } _ { i , \theta } ( s ) : = \mathbb { P } _ { \theta } ( s _ { i } = s \mid o _ { 1 : i - 1 } , a _ { 1 : i - 1 } )$ gives

$$
\mathbb { P } _ { \theta } \big ( o _ { i } \mid o _ { 1 : i - 1 } , a _ { 1 : i - 1 } \big ) = \sum _ { s } \bar { b } _ { i , \theta } ( s ) O _ { \varphi } ( o _ { i } \mid s ) .\tag{9}
$$

Combining (6) and (9) yields

$$
f _ { \theta } \bigl ( o _ { 1 : n } , a _ { 1 : n } \bigr ) = \prod _ { i = 1 } ^ { n } \Big ( \sum _ { s } \bar { b } _ { i , \theta } ( s ) O _ { \varphi } ( o _ { i } \mid s ) \Big ) .
$$

Taking logs proves the claimed expression for $\ell _ { n } ( \theta )$

## C

Recall that the model-dependent component of the trajectory likelihood satisfies

$$
f _ { \theta } \bigl ( o _ { 1 : n } , a _ { 1 : n } \bigr ) = \prod _ { i = 1 } ^ { n } \mathbb { P } _ { \theta } \bigl ( o _ { i } \mid o _ { 1 : i - 1 } , a _ { 1 : i - 1 } \bigr ) .
$$

For notational convenience, let

$$
q _ { i } : = \mathbb { P } _ { \theta } ( o _ { i } \mid o _ { 1 : i - 1 } , a _ { 1 : i - 1 } ) .
$$

Recall that from (9), we have

$$
q _ { i } = \sum _ { s \in S } \bar { b } _ { i , \theta } ( s ) O _ { \varphi } ( o _ { i } \mid s ) .
$$

Next, we express ${ \bar { b } _ { 2 k , \theta } }$ in terms of $\bar { b } _ { 2 k - 1 , \theta }$ . By Bayes’ rule,

$$
\begin{array} { r } { { \mathbb { P } _ { \theta } ( s _ { 2 k - 1 } = s \mid o _ { 1 : 2 k - 1 } , a _ { 1 : 2 k - 1 } ) } } \\ { = \frac { \bar { b } _ { 2 k - 1 , \theta } ( s ) O _ { \varphi } ( o _ { 2 k - 1 } \mid s ) } { q _ { 2 k - 1 } } , } \end{array}
$$

where conditioning on $a _ { 2 k - 1 }$ provides no additional information about the latent state given $O _ { 2 k - 1 }$ since the behavior policy is observation-based. Applying the transition kernel and marginalizing over $s _ { 2 k - 1 }$ then gives

$$
\begin{array} { r l r } & { } & { \bar { b } _ { 2 k , \theta } ( s ^ { \prime } ) = \displaystyle \sum _ { s \in S } \mathbb { P } _ { \theta } ( s _ { 2 k - 1 } = s \mid o _ { 1 : 2 k - 1 } , a _ { 1 : 2 k - 1 } ) T _ { \beta } ^ { a _ { 2 k - 1 } } ( s ^ { \prime } \mid s ) } \\ & { } & { = \displaystyle \frac { 1 } { q _ { 2 k - 1 } } \sum _ { s \in S } \bar { b } _ { 2 k - 1 , \theta } ( s ) O _ { \varphi } ( o _ { 2 k - 1 } \mid s ) T _ { \beta } ^ { a _ { 2 k - 1 } } ( s ^ { \prime } \mid s ) . } \end{array}
$$

Consequently,

$$
\begin{array} { l } { q _ { 2 k - 1 } q _ { 2 k } = q _ { 2 k - 1 } \displaystyle \sum _ { s ^ { \prime } \in \mathcal { S } } \bar { b } _ { 2 k , \theta } ( s ^ { \prime } ) { O } _ { \varphi } ( { o } _ { 2 k } \mid s ^ { \prime } ) } \\ { = \displaystyle \sum _ { s \in \mathcal { S } } \bar { b } _ { 2 k - 1 , \theta } ( s ) { O } _ { \varphi } ( { o } _ { 2 k - 1 } \mid s ) \displaystyle \sum _ { s ^ { \prime } \in \mathcal { S } } T _ { \beta } ^ { a _ { 2 k - 1 } } ( s ^ { \prime } \mid s ) { O } _ { \varphi } ( { o } _ { 2 k } \mid s ^ { \prime } ) . } \end{array}
$$

Finally, since n is even, grouping consecutive terms in the likelihood decomposition yields

$$
\begin{array} { l } { \displaystyle \ell _ { n } ( \theta ) = \sum _ { k = 1 } ^ { n / 2 } \log ( q _ { 2 k - 1 } q _ { 2 k } ) } \\ { = \displaystyle \sum _ { k = 1 } ^ { n / 2 } \log \left( \sum _ { s \in \mathcal { S } } \bar { b } _ { 2 k - 1 , \theta } ( s ) O _ { \varphi } ( \sigma _ { 2 k - 1 } \mid s ) \sum _ { s ^ { \prime } \in \mathcal { S } } T _ { \beta } ^ { a _ { 2 k - 1 } } ( s ^ { \prime } \mid s ) O _ { \varphi } ( \sigma _ { 2 k } \mid s ^ { \prime } ) \right) , } \end{array}
$$

as claimed.

## D

First, we acknowledge that Lemma 3.5 is inspired by the high-level argument underlying Propositions 13 and 14 of Liu et al. [2022], but the statement and proof are tailored to our predictivedistribution formulation.

Assume we have a series of predictive belief states $\{ \bar { b } _ { i , \theta ^ { * } } \} _ { i = 1 } ^ { n }$ . Then, for each $i \in [ n ]$ , we sample $o _ { i }$ according to the distribution $\begin{array} { r } { \mathbb { P } _ { i } ^ { * } ( o ) : = \sum _ { s } \bar { b } _ { i , \theta ^ { * } } ( s ) O _ { \varphi ^ { * } } ( o | s ) } \end{array}$ . Assume we have another series of predictive belief states $\{ \bar { b } _ { i , \hat { \theta } } \} _ { i = 1 } ^ { n }$ . Then, define $D : = \{ ( o _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ) \} _ { i = 1 } ^ { n }$ . Similarly, define $\begin{array} { r } { \widehat { \mathbb { P } } _ { i } ( o ) : = \sum _ { s } \bar { b } _ { i , \hat { \theta } } ( s ) O _ { \hat { \varphi } } ( o | s ) } \end{array}$ . For a certain $o _ { i }$ , let ${ \tilde { o } } _ { i }$ be a “tangent" observation independently sampled from the same probability distribution $\mathbb { P } _ { i } ^ { * }$ . Then, we define $\widetilde { D } : = \{ ( \tilde { o } _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ) \} _ { i = 1 } ^ { n }$ . Then, we introduce an auxiliary lemma adapted from Liu et al. [2022]. See Appendix D.1 for the proof.

Lemma D.1. Let $\ell : = \ell ( o _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } )$ be a real-valuedfunction that maps an observation $o _ { i }$ and two predictive belief states $\bar { b } _ { i , \theta ^ { * } }$ and $\bar { b } _ { i , \hat { \theta } }$ to R. Let $\begin{array} { r } { L ( D ) = \sum _ { i = 1 } ^ { n } \ell ( o _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ) } \end{array}$ , and $L ( \widetilde { D } ) =$ $\textstyle \sum _ { i = 1 } ^ { n } \ell ( \tilde { o } _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } )$ . Then,

$$
\begin{array} { r } { \mathbb { E } \Big [ \exp \Big ( L ( D ) - \log \mathbb { E } \big [ \exp ( L ( \tilde { D } ) ) | D \big ] \Big ) \Big ] = 1 . } \end{array}
$$

Given Lemma D.1, we define

$$
\begin{array} { r } { \ell _ { \hat { \theta } } ( o , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ) = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \log \left( \frac { \sum _ { s } \bar { b } _ { i , \hat { \theta } } ( s ) O _ { \hat { \varphi } } ( o | s ) } { \sum _ { s } \bar { b } _ { i , \theta ^ { * } } ( s ) O _ { \varphi ^ { * } } ( o | s ) } \right) = \frac { 1 } { 2 } \log \left( \frac { \widehat { \mathbb { P } } _ { i } ( o ) } { \mathbb { P } _ { i } ^ { * } ( o ) } \right) } & { \mathrm { i f ~ } \mathbb { P } _ { i } ^ { * } ( o ) \neq 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}
$$

By Lemma D.1 and Markov’s inequality, with probability at least $1 - \delta .$ , we have

$$
- \log \mathbb { E } _ { \widetilde { D } } [ \exp ( L ( \widetilde { D } ) ) \mid D ] < - L ( D ) + \log ( 1 / \delta )
$$

Then, by the definition of L and by the inequality $- \log x \geq 1 - x $

$$
\begin{array} { r } { \displaystyle - \log \mathbb { E } _ { \widetilde { D } } [ \exp ( L ( \widetilde { D } ) ) \mid D ] = - \sum _ { i = 1 } ^ { n } \log \mathbb { E } _ { o \sim \mathbb { P } _ { i } ^ { * } } \left[ \sqrt { \frac { \widehat { \mathbb { P } } _ { i } ( o ) } { \mathbb { P } _ { i } ^ { * } ( o ) } } \right] } \\ { \displaystyle \geq \sum _ { i = 1 } ^ { n } \left( 1 - \mathbb { E } _ { o \sim \mathbb { P } _ { i } ^ { * } } \left[ \sqrt { \frac { \widehat { \mathbb { P } } _ { i } ( o ) } { \mathbb { P } _ { i } ^ { * } ( o ) } } \right] \right) . } \end{array}
$$

Furthermore, by algebra,

$$
\begin{array} { r l } & { \quad \displaystyle \sum _ { i = 1 } ^ { n } \left( 1 - \mathrm { E } _ { i } \mathrm { e } _ { i } \mathrm { e } _ { i } ^ { \kappa } \bigg \lbrace \left[ \sqrt { \frac { \hat { \mathcal { F } } _ { i } ( \theta ) } { \hat { \mathcal { F } } _ { i } ( \theta ) } } \right] \right) } \\ & { = \displaystyle \sum _ { i = 1 } ^ { n } \left( 1 - \sum _ { o } \sqrt { \sqrt { \hat { \mathcal { F } } _ { i } ( o ) \Psi _ { i } ^ { * } ( o ) } } \right) } \\ & { = \displaystyle \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } \left( \sum _ { o } \left( \sqrt { \hat { \mathcal { F } } _ { i } ( o ) } - \sqrt { \mathrm { F } _ { i } ^ { * } ( o ) } \right) ^ { 2 } \right) } \\ & { \geq \displaystyle \sum _ { \hat { \mathbf { b } } } \displaystyle \sum _ { i = 1 } ^ { n } \left[ \sum _ { o } \left( \sqrt { \hat { \mathcal { F } } _ { i } ( o ) } - \sqrt { \mathrm { F } _ { i } ( o ) } \right) ^ { 2 } \right] \left[ \sum _ { o } \left( \sqrt { \hat { \mathcal { F } } _ { i } ( o ) } + \sqrt { \mathrm { F } _ { i } ^ { * } ( o ) } \right) ^ { 2 } \right] } \\ & { \geq \displaystyle \sum _ { \hat { \mathbf { b } } } \frac { 1 } { \mathrm { S o } } \left( \sum _ { o } \left| \hat { \mathcal { F } } _ { i } ( o ) - \mathrm { P } _ { i } ^ { * } ( o ) \right| \right) ^ { 2 } . } \end{array}\tag{10}
$$

where the last inequality follows from the Cauchy–Schwarz inequality. Putting all relations together, we have that with probability at least $1 - \delta$

$$
\frac { 1 } { 8 } \sum _ { i = 1 } ^ { n } \left( \sum _ { o } \Big | \widehat { \mathbb { P } } _ { i } ( o ) - \mathbb { P } _ { i } ^ { * } ( o ) \Big | \right) ^ { 2 } \leq - L ( D ) + \log ( 1 / \delta ) .
$$

We plug in the definition of $\widehat { \mathbb { P } } _ { i } ( o )$ and $\mathbb { P } _ { i } ^ { * } ( o )$ to obtain the left-hand side of (1), and expand $- L ( D )$ and use the result from Proposition 3.3 to obtain the right-hand side of (1).

## D.1

It is important to mention that all expectations that appear in the lemma and in this proof exist because each $o _ { i }$ and ${ \tilde { o } } _ { i }$ can only take finitely many different values whereas $\{ \bar { b } _ { i , \theta ^ { * } } \} _ { i = 1 } ^ { n }$ and $\{ \bar { b } _ { i , \hat { \theta } } \} _ { i = 1 } ^ { n }$ are fixed and do not affect the expectations.

Define $E _ { i } = \mathbb { E } [ \exp \left( \ell ( \tilde { o } _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ) \right) \mid \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ]$ and

$$
\begin{array} { r } { u _ { n } = \mathbb { E } \left[ \exp \left( L ( \mathcal { D } ) - \log \mathbb { E } [ \exp ( L ( \tilde { \mathcal { D } } ) ) \mid \mathcal { D } ] \right) \right] = \mathbb { E } \left[ \frac { \exp \left( \sum _ { i = 1 } ^ { n } \ell ( o _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \theta } ) \right) } { \mathbb { E } [ \exp \left( \sum _ { i = 1 } ^ { n } \ell ( \tilde { \omega } _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ) \right) \mid \mathcal { D } ] } \right] . } \end{array}
$$

We want to show that $u _ { n } = 1$ . By the definition of $\{ \tilde { o } _ { i } \} _ { i = 1 } ^ { n }$ , some calculation gives

$$
\mathbb { E } \left[ \exp \left( \sum _ { i = 1 } ^ { n } \ell ( \tilde { o } _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ) \right) \bigg | \mathcal { D } \right] = \prod _ { i = 1 } ^ { n } E _ { i } .
$$

Plugging this into the last expression obtained for $u _ { n }$ and using the tower rule with

$$
\mathcal { D } _ { n - 1 } = ( \bar { b } _ { 1 , \theta ^ { * } } , \bar { b } _ { 1 , \hat { \theta } } , o _ { 1 } , \dots , \bar { b } _ { n - 1 , \theta ^ { * } } , \bar { b } _ { n - 1 , \hat { \theta } } , o _ { n - 1 } , \bar { b } _ { n , \theta ^ { * } } , \bar { b } _ { n , \hat { \theta } } )
$$

we get

$$
u _ { n } = \mathbb { E } [ \mathbb { E } [ \frac { \exp ( \sum _ { i = 1 } ^ { n } \ell ( o _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ) } { \prod _ { i = 1 } ^ { n } E _ { i } } | D _ { n - 1 } | ] = \mathbb { E } [ \frac { \mathbb { E } [ \exp ( \sum _ { i = 1 } ^ { n } \ell ( o _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ) ) \mid D _ { n - 1 } ] } { \prod _ { i = 1 } ^ { n } E _ { i } } ] ,
$$

where we used that $E _ { 1 } , \ldots , E _ { n }$ are $\sigma ( D _ { t - 1 } )$ -measurable. Now,

$$
\mathbb { E } \left[ \exp \left( \sum _ { i = 1 } ^ { n } \ell ( o _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ) \right) \bigg | D _ { n - 1 } \right] = \mathbb { E } \left[ \exp \left( \sum _ { i = 1 } ^ { n - 1 } \ell ( o _ { i } , \bar { b } _ { i , \theta ^ { * } } , \bar { b } _ { i , \hat { \theta } } ) \right) \bigg | D _ { n - 1 } \right] E _ { n } .
$$

Plugging this back into our previous expression,

$$
\begin{array} { r l } & { u _ { n } = \mathbb { E } \left[ \frac { \mathbb { E } \left[ \exp { \left( \sum _ { i = 1 } ^ { n - 1 } \ell ( o _ { i } , \bar { b } _ { i } , \theta ^ { * } , \bar { b } _ { i } , \bar { \theta } ) \right) } \ | D _ { n - 1 } \right] E _ { n } } { \left( \prod _ { i = 1 } ^ { n - 1 } E _ { i } \right) E _ { n } } \right] } \\ & { \quad = \mathbb { E } \left[ \mathbb { E } \left[ \frac { \exp { \left( \sum _ { i = 1 } ^ { n - 1 } \ell ( o _ { i } , \bar { b } _ { i } , \theta ^ { * } , \bar { b } _ { i } , \theta ) \right) } } { \prod _ { i = 1 } ^ { n - 1 } E _ { i } } \Big | D _ { n - 1 } \right] \right] } \\ & { \quad = \mathbb { E } \left[ \frac { \exp { \left( \sum _ { i = 1 } ^ { n - 1 } \ell ( o _ { i } , \bar { b } _ { i } , \theta ^ { * } , \bar { b } _ { i } , \theta ) \right) } } { \prod _ { i = 1 } ^ { n - 1 } E _ { i } } \right] } \\ & { \quad = u _ { n - 1 } = \cdots = u _ { 0 } = 1 , } \end{array}
$$

where the second equality used that $E _ { 1 } , \dots , E _ { t - 1 }$ is $\sigma ( D _ { t - 1 } )$ -measurable.

## E

In this appendix, we extend the pointwise guarantee of Lemma 3.5 to a uniform statement over the parameter space. We begin by formally stating the resulting global guarantee.

Corollary E.1. Let

$$
\Theta = \Delta ( \mathcal { S } ) \times \prod _ { a \in \mathcal { A } , s \in \mathcal { S } } \Delta ( \mathcal { S } ) \times \prod _ { s \in \mathcal { S } } \Delta ( \Omega )
$$

be the POMDP parameter space, and define

$$
d = ( | A | | S | + 1 ) ( | S | - 1 ) + | S | ( | \Omega | - 1 ) , \qquad K = \operatorname* { m a x } \{ | S | , | \Omega | \} = | \Omega | .
$$

For $\theta = ( \beta , \varphi ) \in \Theta$ , let

$$
p _ { i , \theta } ( o ) = \sum _ { s \in S } \bar { b } _ { i , \theta } ( s ) O _ { \varphi } ( o \mid s )
$$

denote the predictive observation probability at time $i .$

There exists a universal constant $c > 0$ such that, for any $\varepsilon \in ( 0 , 1 ]$ and $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$ , the following holds simultaneously for all ${ \hat { \theta } } \in \Theta .$ : there exists $\tilde { \theta } \in \Theta$ satisfying $\| { \tilde { \theta } } - { \hat { \theta } } \| _ { \infty } \leq \varepsilon$ and

$$
\begin{array} { r l } & { \displaystyle \sum _ { i = 1 } ^ { n } \left( \sum _ { o \in \Omega } \left. p _ { i , \bar { \theta } } ( o ) - p _ { i , \theta ^ { * } } ( o ) \right. \right) ^ { 2 } } \\ & { \quad \le c \left( \displaystyle \sum _ { i = 1 } ^ { n } \log \frac { p _ { i , \theta ^ { * } } ( o _ { i } ) } { p _ { i , \hat { \theta } } ( o _ { i } ) } + d \log \left( 2 K \operatorname* { m a x } \left\{ n , \frac { 1 } { \varepsilon } \right\} \right) + \log ( 1 / \delta ) + 2 \right) . } \end{array}\tag{11}
$$

Here, a zero denominator in the likelihood ratio is interpreted as giving an infinite right-hand side.

Remark E.2. Corollary E.1 shows that a small likelihood gap for any candidate parameter $\hat { \theta }$ implies a small prediction-error bound for some nearby discretized parameter ${ \tilde { \theta } } .$ Since $\| { \hat { \theta } } - { \tilde { \theta } } \| _ { \infty } \leq \varepsilon$ , any later bound on $\lVert \tilde { { \boldsymbol { \theta } } } - { \boldsymbol { \theta } } ^ { * } \rVert$ then be transferred back to $\hat { \theta }$ by paying only an additional ε-level discretization error. Furthermore, the discretization cost appears only through a logarithmic term.

Proof. We discretize the parameter space using a deterministic grid. Let

$$
h = { \frac { \operatorname* { m i n } \{ \varepsilon , 1 / n \} } { K } } .
$$

For each simplex in $\Theta ,$ we partition its first $k - 1$ coordinates into intervals of length at most $h ,$ where k is the dimension of that simplex. The last coordinate is determined by the simplex constraint. Taking the Cartesian product of these partitions gives a collection of nonempty grid cells $C _ { 1 } , \ldots , C _ { N _ { h } }$ covering Θ.

Since $\Theta$ has d free coordinates, the number of cells satisfies

$$
N _ { h } \leq \left\lceil \frac { 1 } { h } \right\rceil ^ { d } \leq \left( \frac { 2 } { h } \right) ^ { d } ,
$$

where the last inequality follows from $h \leq 1$ . Consequently,

$$
\log N _ { h } \leq d \log \left( 2 K \operatorname* { m a x } \left\{ n , \frac { 1 } { \varepsilon } \right\} \right) .\tag{12}
$$

We next construct a deterministic representative $\theta _ { j }$ for each cell $C _ { j }$ . In particular, $\theta _ { j }$ is deterministic, and will depend only on the cell and not on the observed trajectory.

Consider one simplex row of a cell, denoted by $C \subseteq \Delta ( k )$ . For each coordinate $r ,$ define

$$
u _ { r } = \operatorname* { s u p } _ { p \in C } p _ { r } , \qquad Z = \sum _ { r = 1 } ^ { k } u _ { r } ,
$$

and let

$$
q _ { r } = \frac { u _ { r } } { Z } , \qquad r = 1 , \dots , k .
$$

Clearly, $q \in \Delta ( k )$

We first bound the normalization factor $Z .$ Since $u _ { r } \geq p _ { r }$ for every $p \in C ,$ , we have $Z \geq 1$ . Let $p ^ { \circ }$ be a point in the closure of $C$ maximizing the last coordinate $p _ { k }$ . Then $u _ { k } = p _ { k } ^ { \circ }$ . Since the first $k - 1$ coordinates vary by at most h within the cell,

$$
u _ { r } \le p _ { r } ^ { \circ } + h , \qquad r = 1 , \ldots , k - 1 .
$$

Therefore,

$$
Z = \sum _ { r = 1 } ^ { k - 1 } u _ { r } + u _ { k } \leq \sum _ { r = 1 } ^ { k - 1 } ( p _ { r } ^ { \circ } + h ) + p _ { k } ^ { \circ } = 1 + ( k - 1 ) h .
$$

Thus,

$$
1 \leq Z \leq 1 + ( k - 1 ) h \leq 1 + K h .\tag{13}
$$

For every $p \in C$ and every coordinate r, we have $u _ { r } \geq p _ { r }$ . It follows that

$$
q _ { r } = \frac { u _ { r } } { Z } \geq \frac { p _ { r } } { 1 + K h } .\tag{14}
$$

Moreover, the variation of each of the first $k - 1$ coordinates within the cell is at most $h ,$ while the variation of the last coordinate is at most $( k - 1 ) h$ . Hence,

$$
0 \leq u _ { r } - p _ { r } \leq ( k - 1 ) h
$$

for every r. Using (13), we obtain

$$
\begin{array} { c } { \displaystyle { \left| q _ { r } - p _ { r } \right| = \frac { \left| ( u _ { r } - p _ { r } ) - p _ { r } ( Z - 1 ) \right| } { Z } } } \\ { \displaystyle { \leq ( k - 1 ) h \leq K h , } } \end{array}
$$

where we used the fact that both $u _ { r } - p _ { r }$ and $p _ { r } ( Z - 1 )$ belong to $[ 0 , ( k - 1 ) h ]$

We apply this construction independently to every simplex row of each grid cell $C _ { j }$ . The resulting parameter $\theta _ { j } \in \Theta$ is deterministic and satisfies, for every $\theta \in C _ { j }$

$$
\| \theta _ { j } - \theta \| _ { \infty } \leq K h \leq \varepsilon ,\tag{15}
$$

and

$$
( \theta _ { j } ) _ { r } \geq \frac { \theta _ { r } } { 1 + K h }\tag{16}
$$

for every probability coordinate r of the initial distribution, transition kernel, and observation kernel.   
The representative $\theta _ { j }$ need not belong to $C _ { j }$ itself.

We now compare the likelihoods under $\theta _ { j }$ and an arbitrary $\theta \in C _ { j }$ . Recall that

$$
\ell _ { n } ( \theta ) = \sum _ { s _ { 1 : n } } \mu _ { \beta } ( s _ { 1 } ) \prod _ { i = 1 } ^ { n } O _ { \varphi } ( o _ { i } \mid s _ { i } ) \prod _ { i = 1 } ^ { n - 1 } T _ { \beta } ^ { a _ { i } } ( s _ { i + 1 } \mid s _ { i } ) .
$$

From Proposition 3.3,

$$
\ell _ { n } ( \theta ) = \prod _ { i = 1 } ^ { n } p _ { i , \theta } ( o _ { i } ) .
$$

Each product inside the sum defining $\ell _ { n } ( \theta )$ contains one initial-state probability, n observation probabilities, and $n - 1$ transition probabilities, for a total of 2n model-dependent factors. By (16), every such factor under $\theta _ { j }$ is at least the corresponding factor under θ divided by $1 + K h$ . Thus, for every hidden-state trajectory,

$$
\begin{array} { r l } {  { \mu _ { \beta _ { j } } ( s _ { 1 } ) \prod _ { i = 1 } ^ { n } O _ { \varphi _ { j } } ( o _ { i } \mid s _ { i } ) \prod _ { i = 1 } ^ { n - 1 } T _ { \beta _ { j } } ^ { a _ { i } } ( s _ { i + 1 } \mid s _ { i } ) } } \\ & { \geq ( 1 + K h ) ^ { - 2 n } \mu _ { \beta } ( s _ { 1 } ) \prod _ { i = 1 } ^ { n } O _ { \varphi } ( o _ { i } \mid s _ { i } ) \prod _ { i = 1 } ^ { n - 1 } T _ { \beta } ^ { a _ { i } } ( s _ { i + 1 } \mid s _ { i } ) . } \end{array}
$$

Therefore,

$$
\ell _ { n } ( \theta _ { j } ) \geq ( 1 + K h ) ^ { - 2 n } \ell _ { n } ( \theta ) .
$$

Taking logarithms, we obtain

$$
\begin{array} { r l } & { \log \frac { \ell _ { n } ( \theta ^ { * } ) } { \ell _ { n } ( \theta _ { j } ) } \le \log \frac { \ell _ { n } ( \theta ^ { * } ) } { \ell _ { n } ( \theta ) } + 2 n \log ( 1 + K h ) } \\ & { \qquad \le \log \frac { \ell _ { n } ( \theta ^ { * } ) } { \ell _ { n } ( \theta ) } + 2 n K h } \\ & { \qquad \le \log \frac { \ell _ { n } ( \theta ^ { * } ) } { \ell _ { n } ( \theta ) } + 2 , } \end{array}\tag{17}
$$

where the last inequality follows from $K h \le 1 / n$ . This comparison holds deterministically for every observed trajectory and every $\theta \in C _ { j }$

Finally, we apply Lemma 3.5 to the deterministic representatives $\theta _ { 1 } , \ldots , \theta _ { N _ { h } }$ , with failure probability $\delta / N _ { h }$ for each one. A union bound implies that, with probability at least $1 - \delta .$ , simultaneously for every $j \in \{ 1 , \ldots , N _ { h } \}$

$$
\begin{array} { r l r } {  { \sum _ { i = 1 } ^ { n } ( \sum _ { o \in \Omega } \vert p _ { i , \theta _ { j } } ( o ) - p _ { i , \theta ^ { * } } ( o ) \vert ) ^ { 2 } } } \\ & { } & { \leq c ( \log \frac { L _ { \theta ^ { * } } ( D ) } { L _ { \theta _ { j } } ( D ) } + \log ( N _ { h } / \delta ) ) . } \end{array}\tag{18}
$$

This union bound is valid because the representatives are fixed before observing the data.

On this event, consider any $\hat { \theta } \in \Theta$ , possibly selected using the observed trajectory. Let $C _ { j }$ be a grid cell containing ${ \hat { \theta } } ,$ and choose $\tilde { \theta } = \theta _ { j }$ . By (15),

$$
\| { \tilde { \theta } } - { \hat { \theta } } \| _ { \infty } \leq \varepsilon .
$$

Furthermore, combining (18), (17), and (12), we have

$$
\begin{array} { r l } {  { \sum _ { i = 1 } ^ { n } ( \sum _ { o \in \Omega } \vert p _ { i , \tilde { \theta } } ( o ) - p _ { i , \theta ^ { * } } ( o ) \vert ) ^ { 2 } } } \\ & { \leq c ( \log \frac { L _ { \theta ^ { * } } ( D ) } { L _ { \hat { \theta } } ( D ) } + \log ( N _ { h } / \delta ) + 2 ) } \\ & { \leq c ( \sum _ { i = 1 } ^ { n } \log \frac { p _ { i , \theta ^ { * } } ( o _ { i } ) } { p _ { i , \hat { \theta } } ( o _ { i } ) } + d \log ( 2 K \operatorname* { m a x } \{ n , \frac { 1 } { \varepsilon } \} ) + \log ( 1 / \delta ) + 2 ) . } \end{array}
$$

Since the high-probability event holds simultaneously for all deterministic representatives, the conclusion holds for every $\hat { \theta } \in \Theta$ on the same event. This proves the claim. □

## F

First, we acknowledge that Lemma 3.6 is inspired by the high-level argument underlying Propositions 13 and 14 of Liu et al. [2022], but the statement and proof are substantially modified to tailor to our predictive-distribution formulation.

For notational convenience, define

$$
\mathbb { P } _ { k } ^ { * } ( o , o ^ { \prime } \mid a ) : = \sum _ { s \in S } \bar { b } _ { k , \theta ^ { * } } ( s ) \mathbb { P } _ { \theta ^ { * } } \big ( o , o ^ { \prime } \mid s , a \big )
$$

and

$$
\widehat { \mathbb { P } } _ { k } \big ( o , o ^ { \prime } \mid a \big ) : = \sum _ { s \in \mathcal { S } } \bar { b } _ { k , \widehat { \theta } } ( s ) \mathbb { P } _ { \widehat { \theta } } \big ( o , o ^ { \prime } \mid s , a \big ) .
$$

Let $H _ { k - 1 }$ denote the observed history preceding $o _ { k }$ , including the previously selected actions. Since the behavior policy chooses $a _ { k }$ after observing $O _ { k }$ , the conditional joint distributions of $\left( o _ { k } , a _ { k } , o _ { k + 1 } \right)$ are

$$
\begin{array} { r } { R _ { k } ^ { * } ( o , a , o ^ { \prime } ) : = \pi ( a \mid o ) \mathbb { P } _ { k } ^ { * } ( o , o ^ { \prime } \mid a ) , } \\ { \widehat { R } _ { k } ( o , a , o ^ { \prime } ) : = \pi ( a \mid o ) \widehat { \mathbb { P } } _ { k } ( o , o ^ { \prime } \mid a ) . } \end{array}
$$

In particular, $R _ { k } ^ { * }$ is the true conditional distribution of $\left( o _ { k } , a _ { k } , o _ { k + 1 } \right)$ given $H _ { k - 1 }$ Define the Hellinger affinity

$$
C _ { k } : = \sum _ { o , a , o ^ { \prime } } \sqrt { R _ { k } ^ { * } ( o , a , o ^ { \prime } ) \widehat { R } _ { k } ( o , a , o ^ { \prime } ) } .
$$

Both $R _ { k } ^ { * }$ and $\widehat { R } _ { k }$ are probability distributions, so $0 \leq C _ { k } \leq 1$

For $k < n$ , let

$$
\lambda _ { k } : = \frac { 1 } { 2 } \log \frac { \widehat { \mathbb { P } } _ { k } ( o _ { k } , o _ { k + 1 } \mid a _ { k } ) } { \mathbb { P } _ { k } ^ { * } ( o _ { k } , o _ { k + 1 } \mid a _ { k } ) } .
$$

The behavior-policy factors cancel from this likelihood ratio. Moreover,

$$
\begin{array} { l } { { \mathbb { E } _ { \theta ^ { * } } \left[ \exp \left( \lambda _ { k } \right) \bigm | H _ { k - 1 } \right] } } \\ { { = \displaystyle \sum _ { o , a , o ^ { \prime } } R _ { k } ^ { * } ( o , a , o ^ { \prime } ) \sqrt { \frac { \widehat { \mathbb { P } } _ { k } \left( o , o ^ { \prime } \bigm | a \right) } { \mathbb { P } _ { k } ^ { * } \left( o , o ^ { \prime } \bigm | a \right) } } } } \\ { { = \displaystyle \sum _ { o , a , o ^ { \prime } } \pi ( a \mid o ) \sqrt { \mathbb { P } _ { k } ^ { * } ( o , o ^ { \prime } \bigm | a ) \widehat { \mathbb { P } } _ { k } ( o , o ^ { \prime } \mid a ) } } } \\ { { = C _ { k } . } } \end{array}\tag{19}
$$

We also define the one-step predictive observation distributions

$$
\begin{array} { l } { { g _ { k } ^ { \ast } ( o ) : = \displaystyle \sum _ { s } \bar { b } _ { k , \theta ^ { \ast } } ( s ) O _ { \varphi ^ { \ast } } ( o \mid s ) , } } \\ { { \displaystyle \widehat { g } _ { k } ( o ) : = \displaystyle \sum _ { s } \bar { b } _ { k , \widehat { \theta } } ( s ) O _ { \widehat { \varphi } } ( o \mid s ) . } } \end{array}
$$

For the boundary observations $j \in \{ 1 , n \}$ , set

$$
\xi _ { j } : = \frac { 1 } { 2 } \log \frac { \widehat { g } _ { j } ( o _ { j } ) } { g _ { j } ^ { \ast } ( o _ { j } ) }
$$

and

$$
B _ { j } : = \sum _ { o } \sqrt { g _ { j } ^ { * } ( o ) \widehat { g } _ { j } ( o ) } .
$$

Again, $0 \leq B _ { j } \leq 1$ , and

$$
\mathbb { E } _ { \theta ^ { * } } \left[ \exp \left( \xi _ { j } \right) | H _ { j - 1 } \right] = B _ { j } .\tag{20}
$$

We now divide the likelihood factors into two nonoverlapping collections, as in the original proof. Let

$$
\begin{array} { r l } & { I _ { 1 } : = \{ k < n : k \mathrm { i s ~ o d d } \} , } \\ & { I _ { 2 } : = \{ k < n : k \mathrm { i s ~ e v e n } \} . } \end{array}
$$

To account for the boundary observations, define

$$
J _ { 1 } : = { \left\{ \begin{array} { l l } { \emptyset , } & { n { \mathrm { ~ i s ~ e v e n } } , } \\ { \{ n \} , } & { n { \mathrm { ~ i s ~ o d d } } , } \end{array} \right. }
$$

and

$$
J _ { 2 } : = \left\{ \begin{array} { l l } { \{ 1 , n \} , } & { n \mathrm { i s } \mathrm { e v e n } , } \\ { \{ 1 \} , } & { n \mathrm { i s } \mathrm { o d d } . } \end{array} \right.
$$

For $r \in \{ 1 , 2 \}$ , write

$$
L _ { r } : = \sum _ { k \in I _ { r } } \lambda _ { k } + \sum _ { j \in J _ { r } } \xi _ { j }
$$

and define

$$
M _ { r } : = \frac { \exp ( L _ { r } ) } { \displaystyle \prod _ { k \in I _ { r } } C _ { k } \prod _ { j \in J _ { r } } B _ { j } } .
$$

Within each collection, the observation pairs do not overlap, and any remaining observations appear as singleton boundary terms. By (19), (20), and iterated conditional expectations,

$$
\mathbb { E } _ { \theta ^ { * } } [ M _ { r } ] = 1 , \qquad r = 1 , 2 .
$$

If the observed trajectory has zero likelihood under ${ \hat { \theta } } ,$ the desired bound holds trivially. Thus, we assume its likelihood is positive.

Applying Markov’s inequality and a union bound, with probability at least $1 - \delta$ we have, simultaneously for $r = 1 , 2$

$$
\log M _ { r } \leq \log { \frac { 2 } { \delta } } .
$$

Since − log $x \ge 1 - x$ for $x > 0 ,$ , and $B _ { j } \leq 1$ , it follows that

$$
\begin{array} { r } { \displaystyle \sum _ { k \in I _ { r } } ( 1 - C _ { k } ) \leq - \sum _ { k \in I _ { r } } \log C _ { k } } \\ { \leq - L _ { r } + \log \displaystyle \frac { 2 } { \delta } . } \end{array}\tag{21}
$$

Next, by the definition of $C _ { k }$

$$
\begin{array} { r } { 1 - C _ { k } = \displaystyle \frac { 1 } { 2 } \sum _ { o , a , o ^ { \prime } } \pi ( a \mid o ) \left( \sqrt { \widehat { \mathbb { P } } _ { k } ( o , o ^ { \prime } \mid a ) } - \sqrt { \mathbb { P } _ { k } ^ { * } ( o , o ^ { \prime } \mid a ) } \right) ^ { 2 } } \\ { \geq \displaystyle \frac { \pi _ { \operatorname* { m i n } } } { 2 } \sum _ { o , o ^ { \prime } } \left( \sqrt { \widehat { \mathbb { P } } _ { k } ( o , o ^ { \prime } \mid a _ { k } ) } - \sqrt { \mathbb { P } _ { k } ^ { * } ( o , o ^ { \prime } \mid a _ { k } ) } \right) ^ { 2 } . } \end{array}\tag{22}
$$

The last inequality follows from $\pi ( a \mid o ) \geq \pi _ { \operatorname* { m i n } }$ and by retaining only the terms corresponding to the realized action $a _ { k }$

Furthermore, by Cauchy–Schwarz,

$$
\begin{array} { r l } & { \left( \displaystyle \sum _ { \alpha , \sigma ^ { \prime } } \left| \widehat { \mathbb { P } } _ { k } ( o , \sigma ^ { \prime } \mid a _ { k } ) - \mathbb { P } _ { k } ^ { * } ( o , \sigma ^ { \prime } \mid a _ { k } ) \right| \right) ^ { 2 } } \\ & { \le \left( \displaystyle \sum _ { \sigma , \sigma ^ { \prime } } \left( \sqrt { \widehat { \mathbb { P } } _ { k } ( o , \sigma ^ { \prime } \mid a _ { k } ) } - \sqrt { \mathbb { P } _ { k } ^ { * } ( o , \sigma ^ { \prime } \mid a _ { k } ) } \right) ^ { 2 } \right] } \\ & { ~ \cdot ~ \left[ \displaystyle \sum _ { \alpha , \sigma ^ { \prime } } \left( \sqrt { \widehat { \mathbb { P } } _ { k } ( o , \sigma ^ { \prime } \mid a _ { k } ) } + \sqrt { \mathbb { P } _ { k } ^ { * } ( o , \sigma ^ { \prime } \mid a _ { k } ) } \right) ^ { 2 } \right] } \\ & { \le 4 \displaystyle \sum _ { \sigma , \sigma ^ { \prime } } \left( \sqrt { \widehat { \mathbb { P } } _ { k } ( o , \sigma ^ { \prime } \mid a _ { k } ) } - \sqrt { \mathbb { P } _ { k } ^ { * } ( o , \sigma ^ { \prime } \mid a _ { k } ) } \right) ^ { 2 } , } \end{array}
$$

where the final inequality uses the fact that both $\widehat { \mathbb { P } } _ { k } ( \cdot , \cdot \mid a _ { k } )$ and $\mathbb { P } _ { k } ^ { * } ( \cdot , \cdot \mid a _ { k } )$ are probability distributions.

Combining this with (22) gives

$$
1 - C _ { k } \ge \frac { \pi _ { \operatorname* { m i n } } } { 8 } \left( \sum _ { o , o ^ { \prime } } \left| \widehat { \mathbb { P } } _ { k } ( o , o ^ { \prime } \mid a _ { k } ) - \mathbb { P } _ { k } ^ { * } ( o , o ^ { \prime } \mid a _ { k } ) \right| \right) ^ { 2 } .\tag{23}
$$

Substituting (23) into (21) and summing over $r = 1 , 2 $ , we obtain

$$
\begin{array} { r l } {  { \frac { \pi _ { \mathrm { m i n } } } { 8 } \sum _ { k = 1 } ^ { n - 1 } ( \sum _ { o , o ^ { \prime } } | \widehat { \mathbb { P } } _ { k } ( o , o ^ { \prime } \mid a _ { k } ) - \mathbb { P } _ { k } ^ { * } ( o , o ^ { \prime } \mid a _ { k } ) | ) ^ { 2 } } \quad } & { } \\ & { \leq - L _ { 1 } - L _ { 2 } + 2 \log \displaystyle \frac { 2 } { \delta } . } \end{array}\tag{24}
$$

It remains to relate $L _ { 1 } + L _ { 2 }$ to the model-dependent likelihood. By the sequential predictive factorization, for every realized observation pair,

$$
\begin{array} { r } { \mathbb { P } _ { k } ^ { * } ( o _ { k } , o _ { k + 1 } \mid a _ { k } ) = g _ { k } ^ { * } ( o _ { k } ) g _ { k + 1 } ^ { * } ( o _ { k + 1 } ) , } \\ { \mathbb { \widehat P } _ { k } ( o _ { k } , o _ { k + 1 } \mid a _ { k } ) = \widehat { g } _ { k } ( o _ { k } ) \widehat { g } _ { k + 1 } ( o _ { k + 1 } ) . } \end{array}
$$

Here the predictive distributions at time $k + 1$ incorporate the observed $o _ { k }$ and the selected action $a _ { k }$ Consequently,

$$
\lambda _ { k } = \frac { 1 } { 2 } \log \frac { \widehat { g } _ { k } ( o _ { k } ) } { g _ { k } ^ { * } ( o _ { k } ) } + \frac { 1 } { 2 } \log \frac { \widehat { g } _ { k + 1 } ( o _ { k + 1 } ) } { g _ { k + 1 } ^ { * } ( o _ { k + 1 } ) } .
$$

Since every pair index $k < n$ appears exactly once in $I _ { 1 } \cup I _ { 2 }$ , and the two boundary terms appear exactly once in $J _ { 1 } \cup J _ { 2 }$ , we have

$$
\begin{array} { c } { { \displaystyle L _ { 1 } + L _ { 2 } = \sum _ { k = 1 } ^ { n - 1 } \lambda _ { k } + \xi _ { 1 } + \xi _ { n } } } \\ { { \displaystyle = \sum _ { k = 1 } ^ { n } \log \frac { \widehat { g } _ { k } ( o _ { k } ) } { g _ { k } ^ { * } ( o _ { k } ) } } } \\ { { \displaystyle = \ell _ { n } ( \widehat { \theta } ) - \ell _ { n } ( \theta ^ { * } ) . } } \end{array}
$$

The final equality follows because the behavior-policy factors are parameter-independent and therefore cancel from the likelihood ratio.

Substituting this identity into (24) yields

$$
\begin{array} { r l } {  { \frac { \pi _ { \operatorname* { m i n } } } { 8 } \sum _ { k = 1 } ^ { n - 1 } ( \sum _ { o , o ^ { \prime } } | \widehat { \mathbb { P } } _ { k } ( o , o ^ { \prime } \mid a _ { k } ) - \mathbb { P } _ { k } ^ { * } ( o , o ^ { \prime } \mid a _ { k } ) | ) ^ { 2 } } \ } \\ & { \leq \ell _ { n } ( \theta ^ { * } ) - \ell _ { n } ( \hat { \theta } ) + 2 \log \displaystyle \frac { 2 } { \delta } . } \end{array}\tag{25}
$$

Finally, since the total $\ell _ { 1 }$ distance between two probability distributions is at most 2,

$$
\left( \sum _ { o , o ^ { \prime } } \left| \widehat { \mathbb { P } } _ { n } ( o , o ^ { \prime } \mid a _ { n } ) - \mathbb { P } _ { n } ^ { * } ( o , o ^ { \prime } \mid a _ { n } ) \right| \right) ^ { 2 } \leq 4 .
$$

Combining this with (25) establishes

$$
\sum _ { k = 1 } ^ { n } \left( \sum _ { o , o ^ { \prime } } \left| \widehat { \mathbb { P } } _ { k } ( o , o ^ { \prime } \mid a _ { k } ) - \mathbb { P } _ { k } ^ { * } ( o , o ^ { \prime } \mid a _ { k } ) \right| \right) ^ { 2 } \leq \frac { 8 } { \pi _ { \operatorname* { m i n } } } \left( \ell _ { n } ( \theta ^ { * } ) - \ell _ { n } ( \hat { \theta } ) + 2 \log \frac { 2 } { \delta } \right) + 4 .
$$

This completes the proof.

## G

In this Appendix, we state the global counterpart of Lemma 3.6. Moreover, we simply state the result without proving it since the same argument from Appendix E applies.

Corollary G.1. Let

$$
\Theta = \Delta ( \mathcal { S } ) \times \prod _ { a \in \mathcal { A } , s \in \mathcal { S } } \Delta ( \mathcal { S } ) \times \prod _ { s \in \mathcal { S } } \Delta ( \Omega )
$$

be the POMDP parameter space, and define

$$
d = ( | A | | S | + 1 ) ( | S | - 1 ) + | S | ( | \Omega | - 1 ) , \qquad K = \operatorname* { m a x } \{ | S | , | \Omega | \} = | \Omega | .
$$

For $\theta = ( \beta , \varphi ) \in \Theta$ , let

$$
\mathbb { P } _ { \theta } \big ( o , o ^ { \prime } \mid s , a \big ) = O _ { \varphi } ( o \mid s ) \sum _ { s ^ { \prime } \in S } T _ { \beta } ^ { a } ( s ^ { \prime } \mid s ) O _ { \varphi } ( o ^ { \prime } \mid s ^ { \prime } ) .
$$

For any $\varepsilon \in ( 0 , 1 ]$ and $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$ , the following holds simultaneously for all $\hat { \theta } \in \Theta$ : there exists $\tilde { \theta } \in \Theta$ satisfying $\| { \tilde { \theta } } - { \hat { \theta } } \| _ { \infty } \leq \varepsilon$ and

$$
\begin{array} { r l } & { \displaystyle \sum _ { k = 1 } ^ { n } \bigg ( \sum _ { o , o ^ { \prime } \in \Omega }  \sum _ { s \in \mathcal { S } } \bar { b } _ { k , \tilde { \theta } } ( s ) \mathbb { P } _ { \tilde { \theta } } ( o , o ^ { \prime } \mid s , a _ { k } ) - \sum _ { s \in \mathcal { S } } \bar { b } _ { k , \theta ^ { * } } ( s ) \mathbb { P } _ { \theta ^ { * } } ( o , o ^ { \prime } \mid s , a _ { k } )  \bigg ) ^ { 2 } } \\ & { \quad \leq \frac { 8 } { \pi _ { \operatorname* { m i n } } } ( \ell _ { n } ( \theta ^ { * } ) - \ell _ { n } ( \hat { \theta } ) + 2 d \log \bigg ( 2 K \operatorname* { m a x } \bigg \{ n , \frac { 1 } { \varepsilon } \bigg \} ) + 2 \log ( 2 / \delta ) + 2 ) + 4 . } \end{array}\tag{26}
$$

Here, the bound is understood to be vacuous $i f \ell _ { n } ( { \widehat { \theta } } ) = - \infty$

Remark G.2. Corollary G.1 shows that a small likelihood gap for any candidate parameter $\hat { \theta }$ implies a small length-2 prediction-error bound for some nearby discretized parameter ${ \tilde { \theta } } .$ Since $\| \hat { \theta } - \widetilde { \theta } \| _ { \infty } \leq \varepsilon .$ any later bound on $\lVert \tilde { { \boldsymbol { \theta } } } - { \boldsymbol { \theta } } ^ { * } \rVert$ then be transferred back to $\hat { \theta }$ by paying only an additional ε-level discretization error. Furthermore, the discretization cost appears only through a logarithmic term.

## H

Table 1 and 2 summarize the main assumptions used in our analysis, their roles, possible relaxations, and practical implications. The key takeaway is that several assumptions are convenient sufficient conditions rather than minimal requirements: some form of observability and coverage is fundamental, while the particular one-step, conditioning, policy, and mixing conditions can often be relaxed.

## I

In this appendix, we prove the following proposition, which constitutes the main technical component of the proof of Lemma 3.13 in Appendix J.

Proposition I.1. Under Assumption 3.8, 3.9, 3.2, and $3 . l I ,$ let

$$
t _ { \mathrm { b u r n } } = 1 + \left\lceil \frac { \log \bigl ( 8 G ( \pi ) / d _ { \mathrm { m i n } } ^ { \pi } \bigr ) } { \log \bigl ( 1 / \rho ( \pi ) \bigr ) } \right\rceil ,
$$

$$
L = \operatorname* { m i n } \left( \frac { C | \mathcal { S } | ^ { 2 } } { \sigma _ { \mathcal { O } } ^ { 4 } } \log \left( \frac { 1 0 2 4 \left( 1 + 1 / \delta _ { 2 } \right) | \mathcal { S } | } { 4 9 | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { \mathcal { O } } ^ { 2 } \left( d _ { \operatorname* { m i n } } ^ { \pi } \right) ^ { 2 } } \right) , \frac { C | \mathcal { S } | } { \sigma _ { \mathcal { O } } ^ { 2 } } \log ^ { 2 } \left( \frac { 1 0 2 4 \left( 1 + 1 / \delta _ { 2 } \right) \sqrt { | \mathcal { S } | \log | \mathcal { S } | } } { 4 9 | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { \mathcal { O } } ^ { 2 } \left( d _ { \operatorname* { m i n } } ^ { \pi } \right) ^ { 2 } } \right) \right) ,
$$

and

$$
D = \frac { 1 4 \left( \log ( 8 / \delta _ { 1 } ) + | S | \log 9 \right) } { | \mathcal { A } | ^ { 2 } \pi _ { \operatorname* { m i n } } ^ { 2 } \sigma _ { T } ^ { 4 } \sigma _ { O } ^ { 4 } ( d _ { \operatorname* { m i n } } ^ { \pi } ) ^ { 4 } } .
$$

If

$$
n \geq 2 D L + \frac { 2 D } { \log ( \frac { 1 } { \rho ( \pi ) } ) } \left[ \log \left( \frac { 4 G ( \pi ) } { \delta _ { 1 } } \right) + \operatorname* { m a x } \left\{ \log \left( \frac { 2 D } { \log ( \frac { 1 } { \rho ( \pi ) } ) } \right) , 0 \right\} \right] + t _ { \mathrm { b u r n } } ,\tag{27}
$$

then, with probability at least $1 - \delta _ { 1 } - \delta _ { 2 }$

$$
\sigma _ { \operatorname* { m i n } } \mathopen { } \mathclose \bgroup \left( \bar { B } \aftergroup \egroup \right) ^ { 2 } \geq \frac { 1 } { 1 2 } \mathopen { } \mathclose \bgroup \left( n - t _ { \mathrm { b u r n } } \aftergroup \egroup \right) \mathopen { } \mathclose \bgroup \left| A \aftergroup \egroup \right| \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( d _ { \operatorname* { m i n } } ^ { \pi } \aftergroup \egroup \right) ^ { 2 } .
$$

Due to the length of the proof, we further divide the proof into subsections.

## I.1

We start by proving several results regarding the expected behavior of the true belief vectors $( \left\{ b _ { t , \theta ^ { * } } \right\} )$ and predictive belief vectors $( \{ \bar { b } _ { t , \theta ^ { * } } \} )$ . Moreover, for notational simplicity, throughout Appendix I we omit the $\theta ^ { * }$ and denote the true belief vectors as $b _ { t }$ and true predictive belief vectors as ${ { \bar { b } } _ { t } }$

In addition, we use $\| \cdot \| _ { o p }$ to denote the matrix spectral norm in the rest of Appendix I.

Recall the definition of history $H _ { t } = ( o _ { 1 } ^ { t } , a _ { 1 } ^ { t } )$ . Then, we have the following lemma.

Table 1: Summary of the main assumptions, their roles in the analysis, possible relaxations, and practical implications.
<table><tr><td colspan="2">Assumption Why needed</td><td>How relaxable</td><td>Practical implications</td></tr><tr><td>3.1</td><td>Makes the induced latent process a stationary Markov chain, enabling mixing and concentration analysis.</td><td>Imposed for convenience. Finite-memory policies can be handled by augmenting the latent state with policy memory.</td><td>Arbitrary full-history adaptation is not covered, but state augmentation handles the broad class of finite-memory adaptive exploration. When certain actions are</td></tr><tr><td>3.2</td><td>Ensures that each action is selected with uniformly positive probability under every observation.</td><td>Some form of observation-action excitation is necessary for uniform recovery guarantees. Without sufficient coverage, belief matrices can become rank deficient, rendering distinct POMDPs observationally</td><td>rarely selected under some observations, the collected data may be insufficient for reliable kernel recovery. Randomizing action selection or combining data from different behavior policies can</td></tr><tr><td>3.8</td><td>Makes latent states distinguishable from one-step observations and controls the rate through σo.</td><td>indistinguishable. Some observability is necessary for latent-state identification. If one-step observability fails but Assumption 3.17 holds, the overcomplete analysis applies.</td><td>improve coverage. Small  $\sigma _ { O }$  makes recovery ill-conditioned and raises sample complexity; longer observation windows may restore separation under Assumption 3.17.</td></tr><tr><td>3.9</td><td>Ensures stable inversion of the transition matrices in recovery steps that require their inverses.</td><td>It is mentioned in Remark 3.10 that 3.9 can be replaced by the aggregate conditioning condition in Appendix L; the stated assumption is only a convenient sufficient</td><td>Near-singularity amplifies predictive-distribution error and increases sample complexity.</td></tr><tr><td>3.11</td><td>Ergodicity ensures positive visitation of all states; mixing controls dependence and enables concentration for a single trajectory.</td><td>condition. Without ergodicity, unvisited states are unlearnable. Multiple independent trajectories can replace within-trajectory mixing if their reset distribution provides adequate coverage.</td><td>Longer runs help with slow mixing but not missing recurrent classes; covering them requires resets or broadly initialized trajectories.</td></tr></table>

## Lemma I.2. Under Assumptions 3.8 and 3.11, the following statements hold.

1. For every $t \geq 1$

$$
\mathbb { E } \big [ b _ { t } b _ { t } ^ { \top } \mid H _ { t - 1 } \big ] \ \succeq \ \sigma _ { O } ^ { 2 } \mathrm { D i a g } ( \bar { b } _ { t } ) ^ { 2 } a l m o s t s u r e l y .
$$

2. If

$$
d _ { t } ^ { \pi } ( s ) = \mathbb { P } ^ { \pi } ( X _ { t } = s ) , \qquad d _ { t , \mathrm { m i n } } ^ { \pi } = \operatorname* { m i n } _ { s \in \mathcal { S } } d _ { t } ^ { \pi } ( s ) ,
$$

then for every $t \geq 1$

$$
\mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \ \succeq \ \sigma _ { O } ^ { 2 } \operatorname { D i a g } ( d _ { t } ^ { \pi } ) ^ { 2 } \ \succeq \ \sigma _ { O } ^ { 2 } ( d _ { t , \operatorname* { m i n } } ^ { \pi } ) ^ { 2 } I .
$$

3. Define

$$
t _ { \mathrm { b u r n } } = 1 + \left\lceil \frac { \log \bigl ( 8 G ( \pi ) / d _ { \mathrm { m i n } } ^ { \pi } \bigr ) } { \log \bigl ( 1 / \rho ( \pi ) \bigr ) } \right\rceil .
$$

Then for every $t \geq t _ { \mathrm { b u r n } }$

$$
\mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \ \succeq \ \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I .
$$

Table 2: Summary of the main assumptions, their roles in the analysis, possible relaxations, and practical implications.
<table><tr><td colspan="2">Assumption Why needed</td><td>How relaxable</td><td>Practical implications</td></tr><tr><td>3.17</td><td>Uses length-m observation blocks and informative action strings when one-step observations do not separate latent mixtures; m = 1 recovers Assumption 3.8.</td><td>Some multi-step separation is necessary when the observation kernel is rank-deficient or when the POMDP is overcomplete. Assumption 3.17 is sufficient, but is not claimed to be minimal.</td><td>Choose informative action sequences that produce distinguishable observation patterns across latent states. Use the shortest observation horizon m that provides sufficient separation, since longer sequences increase sample requirements.</td></tr><tr><td>3.18, 3.22</td><td>Assumption 3.18 ensures positive sampling of informative strings and a Markov block process; Assumption 3.22 supplies ergodicity for mixing and concentration.</td><td>Kernel identification requires coverage of informative action strings and latent states, but Assumptions 3.18 and 3.22 are sufficient rather than claimed minimal conditions.</td><td>Rare informative actions or slow mixing reduce the effective sample size. Uniform randomization lower-bounds informative-action sampling probabilities, while independent trajectories reduce dependence.</td></tr><tr><td>3.24</td><td>Without this, observationally indistinguishable latent states can admit distinct, observationally equivalent POMDP representations that are not related by state permutations, making kernel identification up to permutation impossible in general.</td><td>It is necessary as established in Appendix O.</td><td>If two latent states produce identical patterns of observable signals, distinguishing them requires additional information, and their corresponding transition dynamics may not be uniquely identifiable from data.</td></tr><tr><td>3.26</td><td>Prevents exponentially ill-conditioned backward observability, where latent-state directions are nearly indistinguishable from past observation histories, leading to exponential sample complexity.</td><td>Necessary in general, as dropping it permits exponentially hard instances even under the remaining assumptions. More strongly, it is strictly necessary for our proof, since polynomial conditioning of the predictive belief matrix requires polynomial backward excitation.</td><td>Design data collection to generate sufficiently diverse beliefs about the current latent state. Otherwise, repeatedly observing nearly identical mixtures of latent states makes it difficult to disentangle their individual dynamics, even when future observations are informative.</td></tr></table>

4. Consequently, for every integer n $\ r _ { \ r } \geq t _ { \mathrm { b u r n } }$

$$
\frac { 1 } { n - t _ { \mathrm { b u r n } } + 1 } \sum _ { t = t _ { \mathrm { b u r n } } } ^ { n } \mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \succeq \sigma _ { O } ^ { 2 } \bigg ( \frac { 7 d _ { \mathrm { m i n } } ^ { \pi } } { 8 } \bigg ) ^ { 2 } I ,
$$

and

$$
\frac { 1 } { n } \sum _ { t = 1 } ^ { n } \mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \ \succeq \ \frac { n - t _ { \mathrm { b u r n } } + 1 } { n } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I .
$$

Proof. We prove the claims one by one.

The first claim

Fix any $t \geq 1$ , and condition on $H _ { t - 1 }$ throughout this step. For each $o \in \Omega$ , define

$$
p _ { t } ( o ) = \mathbb { P } ( o _ { t } = o \mid H _ { t - 1 } ) .
$$

Since

$$
{ \bar { b } } _ { t } ( s ) = \mathbb { P } ( s _ { t } = s \mid H _ { t - 1 } ) ,
$$

we have

$$
p _ { t } ( o ) = \sum _ { s \in \cal S } \bar { b } _ { t } ( s ) O _ { \varphi ^ { * } } ( o \mid s ) .
$$

Whenever $p _ { t } ( o ) > 0$ , Bayes’ rule gives

$$
b _ { t } = \mathbb { P } ( s _ { t } = \cdot \mid H _ { t - 1 } , o _ { t } ) = \frac { \operatorname { D i a g } ( \bar { b } _ { t } ) \big ( O _ { \varphi ^ { * } } ( o _ { t } \mid s ) \big ) _ { s \in \mathcal { S } } } { p _ { t } ( o _ { t } ) } .
$$

Hence

$$
\begin{array} { r } { \mathbb { E } [ b _ { t } b _ { t } ^ { \top } \ | \ H _ { t - 1 } ] = \displaystyle \sum _ { o \in \Omega } p _ { t } ( o ) \mathbb { P } ( s _ { t } = \cdot \ | \ H _ { t - 1 } , o _ { t } = o ) \mathbb { P } ( s _ { t } = \cdot \ | \ H _ { t - 1 } , o _ { t } = o ) ^ { \top } } \\ { = \displaystyle \sum _ { o : p _ { t } ( o ) > 0 } \frac { \mathrm { D i a g } ( \bar { b } _ { t } ) \left( O _ { \varphi ^ { * } } ( o \mid s ) O _ { \varphi ^ { * } } ( o \mid s ^ { \prime } ) \right) _ { s , s ^ { \prime } \in \mathcal { S } } \mathrm { D i a g } ( \bar { b } _ { t } ) } { p _ { t } ( o ) } . } \end{array}
$$

Since $0 < p _ { t } ( o ) \le 1$ whenever $p _ { t } ( o ) > 0$ , we have $1 / p _ { t } ( o ) \geq 1$ , so

$$
\mathbb { E } [ b _ { t } b _ { t } ^ { \top } \mid H _ { t - 1 } ] \ \succeq \ \sum _ { o : p _ { t } ( o ) > 0 } \mathrm { D i a g } ( \bar { b } _ { t } ) \left( O _ { \varphi ^ { * } } ( o \mid s ) O _ { \varphi ^ { * } } ( o \mid s ^ { \prime } ) \right) _ { s , s ^ { \prime } \in S } \mathrm { D i a g } ( \bar { b } _ { t } ) .
$$

If $p _ { t } ( o ) = 0$ , then

$$
0 = \sum _ { s \in \cal S } \bar { b } _ { t } ( s ) { \cal O } _ { \varphi ^ { * } } ( o \mid s ) ,
$$

and all terms in this sum are nonnegative, so

$$
\bar { b } _ { t } ( s ) O _ { \varphi ^ { * } } ( o \mid s ) = 0 \qquad \mathrm { f o r } \mathrm { e v e r y } s \in \mathcal { S } .
$$

Hence the corresponding matrix term is zero. Therefore we may sum over all $o \in \Omega :$

$$
\mathbb { E } [ b _ { t } b _ { t } ^ { \top } \mid H _ { t - 1 } ] \ \succeq \ \sum _ { o \in \Omega } \mathrm { D i a g } ( \bar { b } _ { t } ) \left( O _ { \varphi ^ { * } } ( o \mid s ) O _ { \varphi ^ { * } } ( o \mid s ^ { \prime } ) \right) _ { s , s ^ { \prime } \in S } \mathrm { D i a g } ( \bar { b } _ { t } ) .
$$

Since

$$
\sum _ { o \in \Omega } \bigl ( O _ { \varphi ^ { * } } ( o \mid s ) O _ { \varphi ^ { * } } ( o \mid s ^ { \prime } ) \bigr ) _ { s , s ^ { \prime } \in S } = O _ { \varphi ^ { * } } ^ { \top } O _ { \varphi ^ { * } } ,
$$

we obtain

$$
\mathbb { E } [ u _ { t } u _ { t } ^ { \top } \mid H _ { t - 1 } ] \succeq \mathrm { D i a g } ( \bar { b } _ { t } ) O _ { \varphi ^ { * } } ^ { \top } O _ { \varphi ^ { * } } \mathrm { D i a g } ( \bar { b } _ { t } ) .
$$

By Assumption 3.8,

$$
O _ { \varphi ^ { * } } ^ { \top } O _ { \varphi ^ { * } } \succeq \sigma _ { O } ^ { 2 } I .
$$

Therefore

$$
\mathbb { E } [ b _ { t } b _ { t } ^ { \top } \mid H _ { t - 1 } ] \ \succeq \ \sigma _ { O } ^ { 2 } \mathrm { D i a g } ( \bar { b } _ { t } ) ^ { 2 } .
$$

This proves the first claim.

The second claim

Taking total expectation in the previous inequality yields

$$
\mathbb { E } \big [ b _ { t } b _ { t } ^ { \top } \big ] = \mathbb { E } \big [ \mathbb { E } \big [ b _ { t } b _ { t } ^ { \top } \mid H _ { t - 1 } \big ] \big ] \ \succeq \ \sigma _ { O } ^ { 2 } \mathbb { E } \big [ \mathrm { D i a g } ( \bar { b } _ { t } ) ^ { 2 } \big ] .
$$

Now $\mathrm { D i a g } ( \bar { b } _ { t } ) ^ { 2 }$ is diagonal, and its $( s , s )$ entry is $\bar { b } _ { t } ( s ) ^ { 2 }$ . Therefore

$$
\mathbb { E } \big [ \mathrm { D i a g } ( \bar { b } _ { t } ) ^ { 2 } \big ] = \mathrm { D i a g } \big ( \mathbb { E } [ \bar { b } _ { t } ( s ) ^ { 2 } ] \big ) _ { s \in \mathcal { S } } .
$$

For each fixed state $s \in S ,$

$$
\bar { b } _ { t } ( s ) = \mathbb { P } ( s _ { t } = s \mid H _ { t - 1 } ) = \mathbb { E } [ \mathbf { 1 } _ { \{ s _ { t } = s \} } \mid H _ { t - 1 } ] ,
$$

so the tower property implies

$$
\mathbb { E } [ \bar { b } _ { t } ( s ) ] = \mathbb { E } [ \mathbf { 1 } _ { \{ s _ { t } = s \} } ] = \mathbb { P } ( s _ { t } = s ) = d _ { t } ^ { \pi } ( s ) .
$$

Since $x \mapsto x ^ { 2 }$ is convex on [0, 1], Jensen’s inequality gives

$$
\mathbb { E } [ \bar { b } _ { t } ( s ) ^ { 2 } ] \geq \left( \mathbb { E } [ \bar { b } _ { t } ( s ) ] \right) ^ { 2 } = d _ { t } ^ { \pi } ( s ) ^ { 2 } .
$$

Hence

$$
\mathbb { E } \big [ \mathrm { D i a g } ( \bar { b } _ { t } ) ^ { 2 } \big ] \ \succeq \ \mathrm { D i a g } ( d _ { t } ^ { \pi } ) ^ { 2 } .
$$

Combining the previous inequalities yields

$$
\mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \ \succeq \ \sigma _ { O } ^ { 2 } \operatorname { D i a g } ( d _ { t } ^ { \pi } ) ^ { 2 } .
$$

Since

$$
\mathrm { D i a g } ( d _ { t } ^ { \pi } ) ^ { 2 } \succeq ( d _ { t , \operatorname* { m i n } } ^ { \pi } ) ^ { 2 } I ,
$$

we conclude that

$$
\mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \ \succeq \ \sigma _ { O } ^ { 2 } ( d _ { t , \operatorname* { m i n } } ^ { \pi } ) ^ { 2 } I .
$$

This proves the second claim.

## The third claim

We now derive a lower bound on $d _ { t , \mathrm { m i n } } ^ { \pi }$ after a sufficiently long burn-in period. Because $( s _ { t } ) _ { t \geq 1 }$ is a Markov chain with transition kernel P<sup>π</sup>, and conditioning on the initial state, we have

$$
d _ { t } ^ { \pi } ( \boldsymbol { s } ^ { \prime } ) = \sum _ { \boldsymbol { s } \in \mathcal { S } } \mathbb { P } ^ { \pi } ( \boldsymbol { s } _ { 1 } = \boldsymbol { s } ) \mathbb { P } ^ { \pi } ( \boldsymbol { s } _ { t } = \boldsymbol { s } ^ { \prime } \mid \boldsymbol { s } _ { 1 } = \boldsymbol { s } ) ,
$$

where we abuse the notation and let $\mathbb { P } ^ { \pi }$ denote the probability measure under the Markov chain induced by π.

Using the convexity of the total variation norm in the first argument and Corollary 3.12,

$$
\begin{array} { r l } & { \| d _ { t } ^ { \pi } - d ^ { \pi } \| _ { \mathrm { T V } } = \left\| \displaystyle \sum _ { s \in \mathcal { S } } \mathbb { P } ^ { \pi } ( s _ { 1 } = s ) \left( \mathbb { P } ^ { \pi } ( s _ { t } = \cdot \mid s _ { 1 } = s ) - d ^ { \pi } \right) \right\| _ { \mathrm { T V } } } \\ & { \qquad \le \displaystyle \sum _ { s \in \mathcal { S } } \mathbb { P } ^ { \pi } ( s _ { 1 } = s ) \left\| \mathbb { P } ^ { \pi } ( s _ { t } = \cdot \mid s _ { 1 } = s ) - d ^ { \pi } \right\| _ { \mathrm { T V } } } \\ & { \qquad \le G ( \pi ) \rho ( \pi ) ^ { t - 1 } . } \end{array}
$$

For any fixed state $s \in S$ , taking the set $A = \{ s \}$ in the definition of total variation in Corollary 3.12 gives

$$
\begin{array} { r } { | d _ { t } ^ { \pi } ( s ) - d ^ { \pi } ( s ) | \leq \| d _ { t } ^ { \pi } - d ^ { \pi } \| _ { \mathrm { T V } } . } \end{array}
$$

Therefore

$$
d _ { t } ^ { \pi } ( s ) \geq d ^ { \pi } ( s ) - \| d _ { t } ^ { \pi } - d ^ { \pi } \| _ { \mathrm { T V } } \geq d _ { \operatorname* { m i n } } ^ { \pi } - G ( \pi ) \rho ( \pi ) ^ { t - 1 } .
$$

Thus we can require

$$
G ( \pi ) \rho ( \pi ) ^ { t - 1 } \leq \frac { d _ { \operatorname* { m i n } } ^ { \pi } } { 8 }
$$

in order to guarantee that

$$
d _ { t } ^ { \pi } ( s ) \geq { \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } } \qquad { \mathrm { f o r ~ e v e r y ~ } } s \in { \mathcal { S } } .
$$

Because $0 < \rho ( \pi ) < 1$ , this inequality is equivalent to

$$
t \geq 1 + \frac { \log \left( 8 G ( \pi ) / d _ { \operatorname* { m i n } } ^ { \pi } \right) } { \log \left( 1 / \rho ( \pi ) \right) } .
$$

Therefore, if

$$
t _ { \mathrm { b u r n } } = 1 + \left\lceil \frac { \log \bigl ( 8 G ( \pi ) / d _ { \mathrm { m i n } } ^ { \pi } \bigr ) } { \log \bigl ( 1 / \rho ( \pi ) \bigr ) } \right\rceil ,
$$

then for every $t \geq t _ { \mathrm { b u r r } }$ <sub>n</sub>,

$$
d _ { t , \mathrm { m i n } } ^ { \pi } = \operatorname* { m i n } _ { s \in \mathcal { S } } d _ { t } ^ { \pi } ( s ) \geq \frac { 7 d _ { \mathrm { m i n } } ^ { \pi } } { 8 } .
$$

Substituting this into the inequality from the second claim gives

$$
\mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \ \succeq \ \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \mathrm { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I \qquad \mathrm { f o r } \mathrm { e v e r y } t \ge t _ { \mathrm { b u r n } } .
$$

This proves the third claim.

The fourth claim

For every $t \geq t _ { \mathrm { b u r n } }$ , we have already shown that

$$
\mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \ \succeq \ \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I .
$$

Averaging over $t = t _ { \mathrm { b u r n } } , \ldots , n$ gives

$$
\frac { 1 } { n - t _ { \mathrm { b u r n } } + 1 } \sum _ { t = t _ { \mathrm { b u r n } } } ^ { n } \mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \ \succeq \ \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I .
$$

This proves the first average bound.

For the full average from $t = 1$ to $t = n$ , note that each matrix $\mathbb { E } [ b _ { t } b _ { t } ^ { \top } ]$ is positive semidefinite. Therefore

$$
\frac { 1 } { n } \sum _ { t = 1 } ^ { n } \mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \succeq \frac { 1 } { n } \sum _ { t = t _ { \mathrm { b u r n } } } ^ { n } \mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] = \frac { n - t _ { \mathrm { b u r n } } + 1 } { n } \left( \frac { 1 } { n - t _ { \mathrm { b u r n } } + 1 } \sum _ { t = t _ { \mathrm { b u r n } } } ^ { n } \mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \right) .
$$

Using the previous inequality,

$$
\frac { 1 } { n } \sum _ { t = 1 } ^ { n } \mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \ \succeq \ \frac { n - t _ { \mathrm { b u r n } } + 1 } { n } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I .
$$

This proves the fourth claim.

Lemma I.3 (Transport of isotropy from the filtered belief to the predictive belief). For every $t \geq t _ { \mathrm { b u r n } } + 1$

$$
\mathbb { E } [ \bar { b } _ { t } \bar { b } _ { t } ^ { \top } ] \succ \lvert \cal { A } \rvert \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { \cal { O } } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I .
$$

Proof. For every $t \geq 1$ , the predictive belief satisfies

$$
\bar { b } _ { t + 1 } = T _ { \beta ^ { * } } ^ { a _ { t } } b _ { t } .
$$

Fix $t \geq t _ { \mathrm { b u r n } } .$ . For any $x \in \mathbb { R } ^ { | S | }$

$$
x ^ { \top } \mathbb { E } [ \bar { b } _ { t + 1 } \bar { b } _ { t + 1 } ^ { \top } ] x = \mathbb { E } \left[ ( x ^ { \top } \bar { b } _ { t + 1 } ) ^ { 2 } \right] = \mathbb { E } \left[ ( x ^ { \top } T _ { \beta ^ { * } } ^ { a _ { t } } b _ { t } ) ^ { 2 } \right] .
$$

Let

$$
\mathcal G _ { t } : = \sigma ( H _ { t - 1 } , o _ { t } ) .
$$

By definition, $b _ { t }$ is $\mathcal { G } _ { t }$ -measurable. Therefore, for every fixed $a \in A .$ , the quantity

$$
( x ^ { \top } T _ { \beta ^ { * } } ^ { a } b _ { t } ) ^ { 2 }
$$

is also $\mathcal { G } _ { t }$ -measurable. Therefore, we have

$$
{ \mathbb P } ( a _ { t } = a \mid { \mathcal G } _ { t } ) = \sum _ { s } b _ { t } ( s ) \sum _ { o } { \pi } ( a \mid o ) O _ { \varphi ^ { * } } ( o | s ) \ge { \pi } _ { \operatorname* { m i n } } \sum _ { s } b _ { t } ( s ) = { \pi } _ { \operatorname* { m i n } } \qquad \mathrm { f o r ~ e v e r y ~ } a \in \mathcal A ,\tag{28}
$$

where the inequality comes from Assumption 3.2.

Using the law of total expectation, we obtain

$$
\mathbb { E } \left[ ( x ^ { \top } T _ { \beta ^ { * } } ^ { a _ { t } } b _ { t } ) ^ { 2 } \right] = \mathbb { E } \left[ \mathbb { E } \left[ ( x ^ { \top } T _ { \beta ^ { * } } ^ { a _ { t } } b _ { t } ) ^ { 2 } \mid \mathcal { G } _ { t } \right] \right] .
$$

Now

$$
( x ^ { \top } T _ { \beta ^ { * } } ^ { a _ { t } } b _ { t } ) ^ { 2 } = \sum _ { a \in \mathcal { A } } \mathbf { 1 } \{ a _ { t } = a \} ( x ^ { \top } T _ { \beta ^ { * } } ^ { a } b _ { t } ) ^ { 2 } ,
$$

so

$$
\mathbb { E } \big [ ( { \boldsymbol { x } } ^ { \top } { \boldsymbol { T } } _ { \beta ^ { * } } ^ { a _ { t } } b _ { t } ) ^ { 2 } \mid { \boldsymbol { \mathcal { G } } } _ { t } \big ] = \sum _ { a \in { \mathcal { A } } } \mathbb { E } \big [ \mathbf { 1 } \{ a _ { t } = a \} ( { \boldsymbol { x } } ^ { \top } { \boldsymbol { T } } _ { \beta ^ { * } } ^ { a } b _ { t } ) ^ { 2 } \mid { \boldsymbol { \mathcal { G } } } _ { t } \big ] .
$$

Because $( x ^ { \top } T _ { \beta ^ { * } } ^ { a } b _ { t } ) ^ { 2 }$ is $\mathcal { G } _ { t }$ -measurable, it can be taken outside the conditional expectation:

$$
\mathbb { E } \big [ ( x ^ { \top } T _ { \beta ^ { * } } ^ { a _ { t } } b _ { t } ) ^ { 2 } \ | \ \mathcal { G } _ { t } \big ] = \sum _ { a \in \cal A } ( x ^ { \top } T _ { \beta ^ { * } } ^ { a } b _ { t } ) ^ { 2 } \mathbb { E } [ \mathbf { 1 } \{ a _ { t } = a \} \ | \ \mathcal { G } _ { t } ] .
$$

Since $\mathbb { E } [ \mathbf { 1 } \{ a _ { t } = a \} \mid \mathcal { G } _ { t } ] = \mathbb { P } ( a _ { t } = a \mid \mathcal { G } _ { t } ) \geq \pi _ { \operatorname* { m i n } }$ from (28),

$$
\mathbb { E } \big [ ( x ^ { \top } T _ { \beta ^ { * } } ^ { a _ { t } } b _ { t } ) ^ { 2 } \mid \mathcal { G } _ { t } \big ] \ge \pi _ { \operatorname* { m i n } } \sum _ { a \in \mathcal { A } } ( x ^ { \top } T _ { \beta ^ { * } } ^ { a } b _ { t } ) ^ { 2 } .
$$

Therefore,

$$
\mathbb { E } \big [ ( x ^ { \top } T _ { \beta ^ { * } } ^ { a _ { t } } b _ { t } ) ^ { 2 } \big ] \geq \pi _ { \operatorname* { m i n } } \sum _ { a \in \mathcal { A } } \mathbb { E } \big [ ( x ^ { \top } T _ { \beta ^ { * } } ^ { a } b _ { t } ) ^ { 2 } \big ] .
$$

Hence

$$
x ^ { \top } \mathbb { E } [ \bar { b } _ { t + 1 } \bar { b } _ { t + 1 } ^ { \top } ] x \geq \pi _ { \operatorname* { m i n } } \sum _ { a \in \mathcal { A } } \mathbb { E } \big [ ( x ^ { \top } T _ { \beta ^ { * } } ^ { a } b _ { t } ) ^ { 2 } \big ] .
$$

Since each $T _ { \beta ^ { * } } ^ { a }$ is deterministic,

$$
\mathbb { E } \big [ ( x ^ { \top } T _ { \beta ^ { * } } ^ { a } b _ { t } ) ^ { 2 } \big ] = x ^ { \top } T _ { \beta ^ { * } } ^ { a } \mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] ( T _ { \beta ^ { * } } ^ { a } ) ^ { \top } x .
$$

Therefore

$$
x ^ { \top } \mathbb { E } [ \bar { b } _ { t + 1 } \bar { b } _ { t + 1 } ^ { \top } ] x \geq \pi _ { \operatorname* { m i n } } \sum _ { a \in \mathcal { A } } x ^ { \top } T _ { \beta ^ { * } } ^ { a } \mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] ( T _ { \beta ^ { * } } ^ { a } ) ^ { \top } x .
$$

Using the lower bound from Lemma I.2

$$
\mathbb { E } [ b _ { t } b _ { t } ^ { \top } ] \succeq \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I ,
$$

we conclude that

$$
x ^ { \top } \mathbb { E } [ \bar { b } _ { t + 1 } \bar { b } _ { t + 1 } ^ { \top } ] x \geq \pi _ { \operatorname* { m i n } } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } \sum _ { a \in \mathcal { A } } x ^ { \top } T _ { \beta ^ { * } } ^ { a } ( T _ { \beta ^ { * } } ^ { a } ) ^ { \top } x .
$$

Since this holds for every $x \in \mathbb { R } ^ { | S | }$ , we obtain the exact matrix lower bound

$$
\mathbb { E } [ \bar { b } _ { t + 1 } \bar { b } _ { t + 1 } ^ { \top } ] \succeq \pi _ { \operatorname* { m i n } } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } \sum _ { a \in \mathcal { A } } T _ { \beta ^ { * } } ^ { a } ( T _ { \beta ^ { * } } ^ { a } ) ^ { \top } .\tag{29}
$$

Now, Assumption 3.9 gives

$$
\sum _ { a \in \mathcal { A } } T _ { \beta ^ { * } } ^ { a } ( T _ { \beta ^ { * } } ^ { a } ) ^ { \top } \succeq \vert \mathcal { A } \vert \sigma _ { T } ^ { 2 } I .
$$

Substituting this into the exact matrix lower bound yields

$$
\mathbb { E } [ \bar { b } _ { t + 1 } \bar { b } _ { t + 1 } ^ { \top } ] \succeq \vert \boldsymbol { \mathcal { A } } \vert \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I .
$$

Corollary I.4. For every integer $n \geq t _ { \mathrm { b u r n } } + 1$

$$
\frac { 1 } { n - t _ { \mathrm { b u r n } } } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \mathbb { E } [ \bar { b } _ { t } \bar { b } _ { t } ^ { \top } ] \succeq | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { \mathcal { O } } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I ,
$$

and

$$
\frac { 1 } { n } \sum _ { t = 1 } ^ { n } \mathbb { E } [ \bar { b } _ { t } \bar { b } _ { t } ^ { \top } ] \succeq \frac { n - t _ { \mathrm { b u r n } } } { n } \left. A \right. \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I .
$$

Proof. This is immediate from the pointwise lower bound from Lemma I.3:

$$
\mathbb { E } [ \bar { b } _ { t } \bar { b } _ { t } ^ { \top } ] \succeq | { \cal A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { { \cal O } } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I \qquad \mathrm { f o r ~ e v e r y ~ } t \geq t _ { \mathrm { b u r n } } + 1 .
$$

## I.2

In Appendix I.1, we have proved several results regarding the properties of $\mathbb { E } [ \bar { b } _ { t } \bar { b } _ { t } ^ { \top } ]$ . The next step is to convert these results into a high probability bound on $\bar { \sigma } _ { \mathrm { m i n } } ( \bar { B } )$ . In this subsection, we introduce two useful concepts and then establish a number of “helper" lemmas and propositions using the two concepts before finishing the proof of the lemma in Appendix I.3.

First, we define the concept of $\beta$ -mixing (also known as absolute regularity) and relate it to the mixing condition of the Markov chain induced by π described in Corollary 3.12.

For two sigma-fields $\mathcal { F }$ and ${ \mathcal { G } } ,$ define their absolute-regularity coefficient by

$$
\beta ( \mathcal { F } , \mathcal { G } ) = \operatorname* { s u p } \frac { 1 } { 2 } \sum _ { i = 1 } ^ { I } \sum _ { j = 1 } ^ { J } \left| \mathbb { P } ( F _ { i } \cap G _ { j } ) - \mathbb { P } ( F _ { i } ) \mathbb { P } ( G _ { j } ) \right| ,
$$

where the supremum is over all finite measurable partitions $\{ F _ { i } \} _ { i = 1 } ^ { I } \subseteq { \mathcal { F } }$ and $\{ G _ { j } \} _ { j = 1 } ^ { J } \subseteq \mathcal { G }$

For a sequence $( R _ { t } )$ , define its absolute-regularity coefficient by

$$
\beta _ { R } ( \ell ) = \operatorname* { s u p } _ { r \geq 1 } \beta \Big ( \sigma ( R _ { 1 } , \ldots , R _ { r } ) , \sigma ( R _ { r + \ell } , R _ { r + \ell + 1 } , \ldots ) \Big ) .
$$

The random sequence $( R _ { t } )$ is said to be β-mixing (or absolutely regular) if $\beta _ { R } ( \ell ) \to 0$ when $\ell \to \infty$ [Volkonskii and Rozanov, 1959, 1961].

Second, we define the concept of $\gamma \cdot$ -separability.

Initially introduced in Even-Dar et al. [2007], under the name “value of observation" and in the context of HMMs, Golowich et al. [2022b] defined a non-stationary POMDP to be γ-observable if for every $O _ { h }$ , the observation kernel at time h, for any beliefs $b , b ^ { \prime }$

$$
\begin{array} { r } { \| \boldsymbol { O } _ { h } ^ { \top } \boldsymbol { b } - \boldsymbol { O } _ { h } \boldsymbol { b } ^ { \prime } \| _ { 1 } \geq \gamma \| \boldsymbol { b } - \boldsymbol { b } ^ { \prime } \| _ { 1 } . } \end{array}
$$

Since a stationary POMDP is a special case of non-stationary POMDPs, we can similarly define under our notation a stationary POMDP to be γ-observable if for any beliefs $b , b ^ { \prime }$

$$
\begin{array} { r } { \| O _ { \varphi ^ { * } } ^ { \top } b - O _ { \varphi ^ { * } } b ^ { \prime } \| _ { 1 } \geq \gamma \| b - b ^ { \prime } \| _ { 1 } . } \end{array}\tag{30}
$$

Now, we introduce the “helper" lemmas and propositions.

Proposition I.5. From Bradley [2005], Corollary 3.12 implies that for every $t \geq 1$

$$
\beta _ { S } ( t ) \le G ( \pi ) \rho ( \pi ) ^ { t - 1 } ,
$$

where we abuse the notation and denote $s$ as the random sequence of latent states. In other words, the Markov chain induced by π is β-mixing.

In other words, Assumption 3.11 implies Corollary 3.12, which in turn implies β-mixing for the induced Markov chain.

Lemma I.6. IfAssumption 3.9 holds, then the POMDP is $\frac { \sigma _ { O } } { \sqrt { | \boldsymbol { S } | } }$ -observable.

Proof. Under Assumption 3.9, $\sigma _ { \mathrm { m i n } } ( O _ { \varphi ^ { \ast } } ) \geq \sigma _ { O } > 0$ , which means for any nonzero d $\in \mathbb { R } ^ { | \cal S | }$

$$
\| O _ { \varphi ^ { * } } d \| _ { 2 } \geq \sigma _ { O } \| d \| _ { 2 } .
$$

In particular, taking $d = b - b ^ { \prime }$ for any beliefs b and $b ^ { \prime } \mathrm { i }$

$$
\| O _ { \varphi ^ { * } } b - O _ { \varphi ^ { * } } b ^ { \prime } \| _ { 2 } \geq \sigma _ { O } \| b - b ^ { \prime } \| _ { 2 } .
$$

Therefore,

$$
\| O _ { \varphi ^ { * } } b - O _ { \varphi ^ { * } } b ^ { \prime } \| _ { 1 } \geq \| O _ { \varphi ^ { * } } b - O _ { \varphi ^ { * } } b ^ { \prime } \| _ { 2 } \geq \sigma _ { O } \| b - b ^ { \prime } \| _ { 2 } \geq \frac { \sigma _ { O } } { \sqrt { | \cal { S } | } } \| b - b ^ { \prime } \| _ { 1 } .
$$

Lemma I.7 (Monotonicity of absolute regularity under sigma-field inclusion). $I f { \mathcal { F } } _ { 1 } \subseteq { \mathcal { F } } _ { 2 }$ and $\mathcal { G } _ { 1 } \subseteq \mathcal { G } _ { 2 } ,$ , then

$$
\beta ( \mathcal { F } _ { 1 } , \mathcal { G } _ { 1 } ) \leq \beta ( \mathcal { F } _ { 2 } , \mathcal { G } _ { 2 } ) .
$$

Proof. Every finite measurable partition admissible for the pair $( \mathcal { F } _ { 1 } , \mathcal { G } _ { 1 } )$ is also admissible for the pair $\dot { ( \mathcal { F } _ { 2 } , \mathcal { G } _ { 2 } ) }$ . The result follows immediately from the definition of $\beta ( \mathcal { F } , \mathcal { G } )$ . □

Proposition I.8 (Mixing of the observation-action process). Proposition I.5 implies that the process

$$
W _ { t } = \left( o _ { t } , a _ { t } \right)
$$

is geometrically β-mixing. More precisely,

$$
\begin{array} { r } { \beta _ { W } ( \ell ) \le \beta _ { S } ( \ell - 1 ) \le G ( \pi ) \rho ( \pi ) ^ { \ell - 2 } , \qquad \ell \ge 2 . } \end{array}
$$

Together with $\beta _ { W } ( 1 ) \leq 1$ , this yields

$$
\begin{array} { r } { \beta _ { W } ( \ell ) \leq \widetilde { G } ( \pi ) \rho ( \pi ) ^ { \ell - 1 } , \qquad \ell \geq 1 , } \end{array}
$$

where $\widetilde { G } ( \pi ) = \operatorname* { m a x } \{ 1 , G ( \pi ) / \rho ( \pi ) \} f o r 0 < \rho ( \pi ) < 1 .$

Proof. Fix $r \geq 1$ and $\ell \geq 2$ . Since the behavior policy is memoryless, the latent state process $\left( { { s _ { t } } } \right)$ is Markov. Moreover, conditional on the current state, the future observation-action process is independent of the past. Consequently,

$$
( W _ { 1 } , \dots , W _ { r } ) \longrightarrow s _ { r + 1 } \longrightarrow s _ { r + \ell } \longrightarrow ( W _ { r + \ell } , W _ { r + \ell + 1 } , \dots )
$$

forms a Markov chain.

By the data-processing inequality for absolute regularity,

$$
\begin{array} { r l } & { \beta \big ( \sigma ( W _ { 1 } , \ldots , W _ { r } ) , \sigma ( W _ { r + \ell } , W _ { r + \ell + 1 } , \ldots ) \big ) } \\ & { \qquad \le \beta \big ( \sigma ( s _ { r + 1 } ) , \sigma ( s _ { r + \ell } ) \big ) } \\ & { \qquad \le \beta _ { S } ( \ell - 1 ) , } \end{array}
$$

where the last inequality follows from the definition of $\beta _ { S } ( \ell - 1 )$

Taking the supremum over r and applying Proposition I.5 gives

$$
\begin{array} { r } { \beta _ { W } ( \ell ) \le \beta _ { S } ( \ell - 1 ) \le G ( \pi ) \rho ( \pi ) ^ { \ell - 2 } , \qquad \ell \ge 2 . } \end{array}
$$

Since $\beta _ { W } ( 1 ) \leq 1$ , the claimed geometric bound follows with $\widetilde G ( \pi ) = \operatorname* { m a x } \{ 1 , G ( \pi ) / \rho ( \pi ) \}$ □

Corollary I.9 (Belief contraction for all history-dependent policies). Consider any γ-observable POMDP P, any $\epsilon > 0 ,$ , and any $L \in \mathbb { N }$ satisfying the lower bound in Theorem 2.1 ofGolowich et al. [2022a]. Let µ be any nonanticipatory history-dependent randomized policy, meaning thatfor each h and each history $( a _ { 1 : h - 1 } , o _ { 2 : h } )$ , the policy chooses an action according to a distribution

$$
\mu _ { h } ( \cdot \mid a _ { 1 : h - 1 } , o _ { 2 : h } ) \in \Delta ( { \cal A } ) .
$$

Suppose that $\mathcal { D } \in \Delta ^ { s }$ satisfies the same domination condition as in Theorem 2.1, namely

$$
\frac { \mathbf { b } _ { h } ^ { \mathcal { P } } ( a _ { 1 : h - L - 1 } , o _ { 2 : h - L } ) ( s ) } { \mathcal { D } ( s ) } \leq \frac { 1 } { \phi }
$$

for every relevant history and every $s \in S .$ . Then

$$
\mathbb { E } _ { ( a _ { 1 : h - 1 } , o _ { 2 : h } ) \sim \mu } ^ { \mathcal { P } } \left[ \left\| \mathbf { b } _ { h } ^ { \mathcal { P } } ( a _ { 1 : h - 1 } , o _ { 2 : h } ) - \mathbf { b } _ { h } ^ { \mathrm { a p x } , \mathcal { P } } ( a _ { h - L : h - 1 } , o _ { h - L + 1 : h } ; \mathcal { D } ) \right\| _ { 1 } \right] \leq \epsilon ,
$$

where $b _ { h } ^ { \mathcal { P } }$ and $b _ { h } ^ { \mathrm { a p x } , \mathcal { P } }$ are defined in Golowich et al. [2022a].

Proof. We show that the trajectory distribution induced by $\mu$ can be represented by some general policy in $\Pi ^ { \mathrm { g e n } }$ . The policy class $\mathbf { \dot { I } } ^ { \mathrm { g e n } }$ is defined in Section 2.2 of Golowich et al. [2022a]: each $\pi \in \Pi ^ { \mathrm { g e n } }$ is a distribution over deterministic history-dependent policies, and is executed by first sampling $\sigma \sim \pi$ and then following σ for the entire episode. For every deterministic policy

$$
\sigma = ( \sigma _ { 1 } , \dots , \sigma _ { H - 1 } ) , \qquad \sigma _ { h } : \mathcal { A } ^ { h - 1 } \times \Omega ^ { h - 1 } \to \mathcal { A } ,
$$

define a distribution $\pi _ { \mu }$ over deterministic policies by sampling, independently for every h and every history $( a _ { 1 : h - 1 } , o _ { 2 : h } )$

$$
\sigma _ { h } ( a _ { 1 : h - 1 } , o _ { 2 : h } ) \sim \mu _ { h } ( \cdot \mid a _ { 1 : h - 1 } , o _ { 2 : h } ) .
$$

Then $\pi _ { \mu } \in \Pi ^ { \mathrm { g e n } }$

Fix any trajectory $\left( s _ { 1 : H } , a _ { 1 : H - 1 } , o _ { 2 : H } \right)$ . Under the stagewise randomized policy $\mu ,$ its probability is

$$
\mathbb { P } _ { \mu } ^ { P } ( s _ { 1 : H } , a _ { 1 : H - 1 } , o _ { 2 : H } ) = \rho ( s _ { 1 } ) \prod _ { h = 1 } ^ { H - 1 } \mu _ { h } ( a _ { h } \mid a _ { 1 : h - 1 } , o _ { 2 : h } ) T _ { \beta ^ { s } } ^ { a _ { h } } ( s _ { h + 1 } \mid s _ { h } ) O _ { \varphi ^ { * } } ( o _ { h + 1 } \mid s _ { h + 1 } ) .
$$

On the other hand, under the general policy $\pi _ { \mu } ,$ a deterministic policy $\sigma$ is first sampled from $\pi _ { \mu }$ and then followed for the whole episode. Therefore,

$$
\begin{array} { l } { { \mathbb { P } _ { \pi _ { \mu } } ^ { \mathcal { P } } \big ( s _ { 1 : H } , a _ { 1 : H - 1 } , o _ { 2 : H } \big ) } } \\ { { \mathrm { } } } \\ { { = \rho ( s _ { 1 } ) \displaystyle \prod _ { h = 1 } ^ { H - 1 } T _ { \beta ^ { \star } } ^ { a _ { h } } \big ( s _ { h + 1 } \mid s _ { h } \big ) O _ { \varphi ^ { \ast } } \big ( o _ { h + 1 } \mid s _ { h + 1 } \big ) \mathbb { P } _ { \sigma \sim \pi _ { \mu } } \big ( \sigma _ { h } \big ( a _ { 1 : h - 1 } , o _ { 2 : h } \big ) = a _ { h } \mathrm { ~ f o r ~ } h = 1 , \dots , H - 1 \big ) . } } \end{array}
$$

By the construction of $\pi _ { \mu }$ , the sampled actions assigned to the histories appearing along the trajectory are independent, and hence

$$
\mathbb { P } _ { \sigma \sim \pi _ { \mu } } \left( \sigma _ { h } ( a _ { 1 : h - 1 } , o _ { 2 : h } ) = a _ { h } { \mathrm { ~ f o r ~ a l l ~ } } h = 1 , \ldots , H - 1 \right) = \prod _ { h = 1 } ^ { H - 1 } \mu _ { h } ( a _ { h } \mid a _ { 1 : h - 1 } , o _ { 2 : h } ) .
$$

Thus,

$$
\mathbb { P } _ { \pi _ { \mu } } ^ { \mathcal { P } } \big ( s _ { 1 : H } , a _ { 1 : H - 1 } , o _ { 2 : H } \big ) = \mathbb { P } _ { \mu } ^ { \mathcal { P } } \big ( s _ { 1 : H } , a _ { 1 : H - 1 } , o _ { 2 : H } \big )
$$

for every trajectory. Therefore, the two policies induce the same distribution over action-observation histories. Consequently,

$$
\mathbb { E } _ { ( a _ { 1 : h - 1 } , o _ { 2 : h } ) \sim \mu } ^ { \mathcal { P } } \left[ \left\| \mathbf { b } _ { h } ^ { \mathcal { P } } ( a _ { 1 : h - 1 } , o _ { 2 : h } ) - \mathbf { b } _ { h } ^ { \mathrm { a p x } , \mathcal { P } } ( a _ { h - L : h - 1 } , o _ { h - L + 1 : h } ; \mathcal { D } ) \right\| _ { 1 } \right]
$$

equals

$$
\mathbb { E } _ { ( a _ { 1 : h - 1 } , o _ { 2 : h } ) \sim \pi _ { \mu } } ^ { \mathcal { P } } \left[ \left\| \mathbf { b } _ { h } ^ { \mathcal { P } } ( a _ { 1 : h - 1 } , o _ { 2 : h } ) - \mathbf { b } _ { h } ^ { \mathrm { a p x } , \mathcal { P } } ( a _ { h - L : h - 1 } , o _ { h - L + 1 : h } ; \mathcal { D } ) \right\| _ { 1 } \right] .
$$

Since $\pi _ { \mu } \in \Pi ^ { \mathrm { g e n } }$ , Theorem 2.1 of Golowich et al. [2022a] applies to $\pi _ { \mu } .$ , and the last display is at most ϵ. This proves the claim. □

Proposition I.10 (Finite-window approximation of the predictive belief). Given Lemma I.6, there exists a constant $C \geq 1$ such thatfor every integer $L \geq \bar { 2 } ,$ , there exists a measurablefunction

$$
g _ { t } ^ { L } : ( \Omega \times \mathcal { A } ) ^ { L } \to \Delta ( \mathcal { S } )
$$

such that

$$
\bar { b } _ { t } ^ { ( L ) } = g _ { t } ^ { L } ( W _ { t - L } , W _ { t - L + 1 } , \dots , W _ { t - 1 } )
$$

satisfies

$$
\mathbb { E } \Big [ \| \bar { b } _ { t } - \bar { b } _ { t } ^ { ( L ) } \| _ { 1 } \Big ] \leq \operatorname* { m i n } \left( | \mathcal { S } | \exp \left( - \frac { \sigma _ { \mathcal { O } } ^ { 4 } } { C | \mathcal { S } | ^ { 2 } } L \right) , \sqrt { | \mathcal { S } | \log ( | \mathcal { S } | ) } \exp \left( - \sigma _ { \mathcal { O } } \sqrt { \frac { L } { C | \mathcal { S } | } } \right) \right) .
$$

Proof. The proof of this proposition combines the belief contraction result of Theorem 2.1 from Golowich et al. [2022a] and Corollary I.9.

We divide the proof into two cases.

Case 1: $t \geq L + 1$

Let

$$
D = { \mathrm { U n i f } } ( S ) , \qquad D ( s ) = { \frac { 1 } { | S | } } \quad { \mathrm { f o r ~ e v e r y ~ } } s \in { \mathcal { S } } .
$$

For any belief vector $b \in \Delta ( S )$ and any state $s \in S$

$$
{ \frac { b ( s ) } { D ( s ) } } = | S | b ( s ) \leq | S | .
$$

Therefore the coverage condition in Theorem 2.1 of Golowich et al. [2022a] is automatically satisfied with

$$
\phi = \frac { 1 } { | S | } .
$$

For any integer $L \geq 1$ , define the truncated filtered belief

$$
b _ { t } ^ { ( L ) } = b _ { t } ^ { \mathrm { a p x } } ( a _ { t - L : t - 1 } , o _ { t - L + 1 : t } ; D ) ,
$$

where $b _ { t } ^ { \mathrm { a p x } }$ is the approximate belief state from Theorem 2.1 of Golowich et al. [2022a] which computes an approximation of the true filtered belief by using only the observations and actions from the most recent length-L window of the history and assuming an initial belief of $D$ at the start of the length-L window.

Furthermore, recall from Lemma I.6 that the POMDP satisfies the observability condition with the constant $\scriptstyle { \frac { \sigma _ { O } } { \sqrt { | { \boldsymbol { S } } | } } }$

Then, if we choose

$$
\varepsilon = | S | \exp \left( - \frac { \sigma _ { O } ^ { 4 } } { C | S | ^ { 2 } } L \right) ,
$$

where the constant C is the same as the constant in Theorem 2.1 of Golowich et al. [2022a]. Then

$$
\varepsilon \phi = \exp \left( - \frac { \sigma _ { O } ^ { 4 } } { C | S | ^ { 2 } } L \right) ,
$$

and therefore

$$
L = C | S | ^ { 2 } \cdot \frac { \log ( 1 / ( \varepsilon \phi ) ) } { \sigma _ { O } ^ { 4 } } ,
$$

which means $L$ exceeds the lower bound required for Theorem 2.1 of Golowich et al. [2022a] to hold.   
There is another choice of $( \varepsilon , L )$ , which we also include for completeness.

Let

$$
L = C \frac { | \cal S | } { \sigma _ { O } ^ { 2 } } \log \frac { 1 } { \epsilon \phi } \cdot \log \frac { \log ( 1 / \phi ) } { \epsilon }
$$

so that L exceeds the lower bound required for Theorem 2.1 of Golowich et al. [2022a].

Furthermore, assume $\phi \in ( 0 , 1 / e )$ (so $\log ( 1 / \phi ) > 1$ and hence log $\log ( 1 / \phi ) > 0 ) \epsilon \in ( 0 , 1 )$ . Note that assuming $\phi \in ( 0 , 1 / e )$ is without loss of generality since $\begin{array} { r } { \phi = \frac { 1 } { | S | } } \end{array}$ , and the only case where $\phi \notin ( 0 , 1 / e )$ is when $| S | = 1 \mathrm { o r } | S | = 2 .$ , in which case we set $\begin{array} { r } { \phi = \frac { 1 } { 3 } } \end{array}$ , which is still a valid choice of $\phi$ that fulfills the coverage assumption of Theorem 2.1 of Golowich et al. [2022a].

Using log $( 1 / ( \epsilon \phi ) ) = \log ( 1 / \epsilon ) + \log ( 1 / \phi )$ and lo $\begin{array} { r } { \small \int ( \log ( 1 / \phi ) / \epsilon ) = \log ( 1 / \epsilon ) + \log \log ( 1 / \phi ) } \end{array}$ , the equation becomes

$$
\left( \log ( 1 / \epsilon ) + \log ( 1 / \phi ) \right) \left( \log ( 1 / \epsilon ) + \log \log ( 1 / \phi ) \right) = L \frac { \sigma _ { O } ^ { 2 } } { C | S | } ,
$$

which is quadratic in $\log ( 1 / \epsilon )$

Expanding,

$$
\begin{array} { r } { \left( \log ( 1 / \epsilon ) \right) ^ { 2 } + \left( \log ( 1 / \phi ) + \log \log ( 1 / \phi ) \right) \log ( 1 / \epsilon ) + \left( \log ( 1 / \phi ) \cdot \log \log ( 1 / \phi ) - L \frac { \sigma _ { O } ^ { 2 } } { C | \mathcal { S } | } \right) = 0 . } \end{array}
$$

Using the identity $( p + q ) ^ { 2 } - 4 p q = ( p - q ) ^ { 2 } \mathrm { w i t h } p = \log ( 1 / \phi ) , q = \log \log ( 1 / \phi ) ,$

$$
\begin{array} { r c l } { \Delta } & { = } & { \displaystyle \left( \log ( 1 / \phi ) + \log \log ( 1 / \phi ) \right) ^ { 2 } - 4 \big ( \log ( 1 / \phi ) \cdot \log \log ( 1 / \phi ) - L \frac { \sigma _ { O } ^ { 2 } } { C | \mathcal { S } | } \big ) } \\ & { = } & { \displaystyle \left( \log ( 1 / \phi ) - \log \log ( 1 / \phi ) \right) ^ { 2 } + 4 L \frac { \sigma _ { O } ^ { 2 } } { C | \mathcal { S } | } \geq 0 . } \end{array}
$$

The quadratic formula gives two algebraic solutions:

$$
\log ( 1 / \epsilon ) = \frac { - \bigl ( \log ( 1 / \phi ) + \log \log ( 1 / \phi ) \bigr ) \ \pm \ \sqrt { \bigl ( \log ( 1 / \phi ) - \log \log ( 1 / \phi ) \bigr ) ^ { 2 } + 4 L \frac { \sigma _ { 0 } ^ { 2 } } { C | S | } } . } { 2 } .
$$

Only the + root is admissible. For the − root, since

$$
\sqrt { \left( \log ( 1 / \phi ) - \log \log ( 1 / \phi ) \right) ^ { 2 } + 4 L \frac { \sigma _ { O } ^ { 2 } } { C | \mathcal { S } | } } \geq \left| \log ( 1 / \phi ) - \log \log ( 1 / \phi ) \right| = \log ( 1 / \phi ) - \log \log ( 1 / \phi ) ,
$$

which further implies

$$
\log ( 1 / \epsilon _ { - } ) \leq \frac { - \left( \log ( 1 / \phi ) + \log \log ( 1 / \phi ) \right) - \left( \log ( 1 / \phi ) - \log \log ( 1 / \phi ) \right) } { 2 } = - \log ( 1 / \phi ) ,
$$

so $\epsilon _ { - } \geq 1 / \phi > 1$ , violating $\epsilon \in ( 0 , 1 )$ . This − root corresponds to the spurious regime where both factors of the original product are negative (their product still equals $L \gamma ^ { 2 } > 0$ , but this sign pattern is ruled out by $\epsilon \in ( 0 , 1 ) )$ .

Taking the + root,

$$
\log ( 1 / \epsilon ) = \frac { - \left( \log ( 1 / \phi ) + \log \log ( 1 / \phi ) \right) + \sqrt { \left( \log ( 1 / \phi ) - \log \log ( 1 / \phi ) \right) ^ { 2 } + 4 L \frac { \sigma _ { O } ^ { 2 } } { C | S | } } , } { 2 } ,
$$

so, exponentiating,

$$
\epsilon = \exp \biggl ( \frac { \log ( 1 / \phi ) + \log \log ( 1 / \phi ) } { 2 } \biggr ) \exp \biggl ( - \frac { 1 } { 2 } \sqrt { \bigl ( \log ( 1 / \phi ) - \log \log ( 1 / \phi ) \bigr ) ^ { 2 } + 4 L \frac { \sigma _ { O } ^ { 2 } } { C | S | } } \biggr ) .
$$

The prefactor simplifies via

$$
\exp \biggl ( \frac { \log ( 1 / \phi ) + \log \log ( 1 / \phi ) } { 2 } \biggr ) = \frac { 1 } { \sqrt { \phi } } \cdot \sqrt { \log ( 1 / \phi ) } = \sqrt { \frac { \log ( 1 / \phi ) } { \phi } } ,
$$

giving

$$
\epsilon = \sqrt { { \frac { \log ( 1 / \phi ) } { \phi } } } ~ \exp \left( - { \scriptstyle { \frac { 1 } { 2 } } } \sqrt { \left( \log ( 1 / \phi ) - \log \log ( 1 / \phi ) \right) ^ { 2 } + 4 L { \frac { \sigma _ { O } ^ { 2 } } { C | S | } } } \right) .
$$

Since $\left( \log ( 1 / \phi ) - \log \log ( 1 / \phi ) \right) ^ { 2 } \geq 0 .$

$$
\sqrt { \left( \log ( 1 / \phi ) - \log \log ( 1 / \phi ) \right) ^ { 2 } + 4 L \frac { \sigma _ { O } ^ { 2 } } { | S | } } \geq 2 \sigma _ { O } \sqrt { \frac { L } { C | S | } } ,
$$

hence

$$
\epsilon \leq \sqrt { \frac { \log ( 1 / \phi ) } { \phi } } \exp \left( - \sigma \sigma \sqrt { \frac { L } { C | S | } } \right) = \sqrt { | S | \log ( | S | ) } \exp \left( - \sigma \sigma \sqrt { \frac { L } { C | S | } } \right) .
$$

To summarize, Theorem 2.1 of Golowich et al. [2022a] and Corollary I.9 gives us that, our two choices of $( L , \epsilon )$ imply

$$
\mathbb { E } \Big [ \| b _ { t } - b _ { t } ^ { ( L ) } \| _ { 1 } \Big ] \leq \operatorname* { m i n } \left( | { \cal S } | \exp \left( - \frac { \sigma _ { 0 } ^ { 4 } } { C | { \cal S } | ^ { 2 } } L \right) , \sqrt { | { \cal S } | \log ( | { \cal S } | ) } \exp \left( - \sigma _ { { \cal O } } \sqrt { \frac { L } { C | { \cal S } | } } \right) \right)
$$

where the randomness in the expectation of the left-hand side comes from the randomness from POMDP dynamics and the randomness from the behavior policy π.

Now define, for every $L \geq 2$ and $t \geq L + 1$

$$
\bar { b } _ { t } ^ { ( L ) } = T _ { \beta ^ { * } } ^ { a _ { t - 1 } } b _ { t - 1 } ^ { ( L - 1 ) } .
$$

Since $b _ { t - 1 } ^ { ( L - 1 ) }$ depends only on

$$
a _ { t - L : t - 2 } \quad \mathrm { a n d } \quad o _ { t - L + 1 : t - 1 } ,
$$

the vector $\bar { b } _ { t } ^ { ( L ) }$ depends only on

$$
a _ { t - L : t - 1 } \quad \mathrm { a n d } \quad o _ { t - L + 1 : t - 1 } ,
$$

and hence it is a measurable function of

$$
W _ { t - L } , W _ { t - L + 1 } , \dots , W _ { t - 1 } .
$$

This gives the measurable map $g _ { t } ^ { L }$

Finally,

$$
\bar { b } _ { t } = T _ { \beta ^ { * } } ^ { a _ { t - 1 } } b _ { t - 1 } ,
$$

so

$$
\| \bar { b } _ { t } - \bar { b } _ { t } ^ { ( L ) } \| _ { 1 } = \| T _ { \beta ^ { * } } ^ { a _ { t - 1 } } ( b _ { t - 1 } - b _ { t - 1 } ^ { ( L - 1 ) } ) \| _ { 1 } .
$$

Because $T _ { \beta ^ { * } } ^ { a _ { t - 1 } }$ is column-stochastic, it is nonexpansive in $\ell _ { 1 }$

$$
\| T _ { \beta ^ { * } } ^ { a _ { t - 1 } } x \| _ { 1 } \leq \| x \| _ { 1 } \qquad \mathrm { f o r e v e r y v e c t o r } x .
$$

Therefore

$$
\| \bar { b } _ { t } - \bar { b } _ { t } ^ { ( L ) } \| _ { 1 } \leq \| b _ { t - 1 } - b _ { t - 1 } ^ { ( L - 1 ) } \| _ { 1 } .
$$

Taking expectations and using the filtered-belief estimate with $L - 1$ in place of L yields

$$
\mathbb { E } \Big [ \| \bar { b } _ { t } - \bar { b } _ { t } ^ { ( L ) } \| _ { 1 } \Big ] \leq \operatorname* { m i n } \left( | \mathcal { S } | \exp \left( - \frac { \sigma _ { O } ^ { 4 } } { C | \mathcal { S } | ^ { 2 } } L \right) , \sqrt { | \mathcal { S } | \log ( | \mathcal { S } | ) } \exp \left( - \sigma _ { O } \sqrt { \frac { L } { C | \mathcal { S } | } } \right) \right) .
$$

Case $2 \colon t \leq L$ In this case, we simply define $b _ { t } ^ { ( L ) } = b _ { t } = b ^ { \mathrm { e x t } } ( a _ { 1 : t - 1 } , o _ { 1 : t } , b _ { 0 } )$ , where $b ^ { \mathrm { e x t } }$ is the exact update function for POMDP beliefs. Then

$$
\mathbb { E } \bigg [ \| b _ { t } - b _ { t } ^ { ( L ) } \| _ { 1 } \bigg ] = 0 .
$$

Again, define

$$
\bar { b } _ { t } ^ { ( L ) } = T _ { \beta ^ { * } } ^ { a _ { t - 1 } } b _ { t - 1 } ^ { ( L - 1 ) } ,
$$

we then have

$$
\mathbb { E } \left[ \Vert \bar { b } _ { t } - \bar { b } _ { t } ^ { ( L ) } \Vert _ { 1 } \right] = 0 \leq \operatorname* { m i n } \left( \vert \mathcal { S } \vert \exp \left( - \frac { \sigma _ { O } ^ { 4 } } { C \vert \mathcal { S } \vert ^ { 2 } } L \right) , \sqrt { \vert \mathcal { S } \vert \log ( \vert \mathcal { S } \vert ) } \exp \left( - \sigma _ { O } \sqrt { \frac { L } { C \vert \mathcal { S } \vert } } \right) \right) ,
$$

Similar to Case 1, since $b _ { t - 1 } ^ { ( L - 1 ) }$ depends only on

$$
a _ { 1 : t - 2 } \quad \mathrm { a n d } \quad o _ { 1 : t - 1 } ,
$$

the vector $\bar { b } _ { t } ^ { ( L ) }$ depends only on

$$
a _ { 1 : t - 1 } \quad \mathrm { a n d } \quad o _ { 1 : t - 1 } ,
$$

and hence it is a measurable function of

$$
W _ { 1 } , W _ { t - L + 1 } , \dots , W _ { t - 1 } .
$$

To obtain a function of the form

$$
g _ { t } ^ { L } : ( \Omega \times \mathcal { A } ) ^ { L } \to \Delta ( \mathcal { S } ) ,
$$

we simply prepend a random string $W _ { t - L } , \cdots , W _ { 0 }$ to obtain the measurable map $g _ { t } ^ { L }$

Proposition I.11 (Finite-window approximation of the matrix process). For every integer $L \geq 2$ and every t, define

$$
Z _ { t } ^ { ( L ) } = \bar { b } _ { t } ^ { ( L ) } \big ( \bar { b } _ { t } ^ { ( L ) } \big ) ^ { \top } .
$$

Then

$$
\mathbb { E } \Big [ \| Z _ { t } - Z _ { t } ^ { ( L ) } \| _ { \mathrm { o p } } \Big ] \leq 2 \operatorname* { m i n } \left( | { \cal S } | \exp \left( - \frac { \sigma _ { \cal O } ^ { 4 } } { { \cal C } | \mathcal { S } | ^ { 2 } } L \right) , \sqrt { | { \cal S } | \log ( | { \cal S } | ) } ~ \exp \left( - \sigma _ { \cal O } \sqrt { \frac { L } { { \cal C } | \mathcal { S } | } } \right) \right) .
$$

Proof. For any two vectors $u , v \in \Delta ( S )$

$$
u u ^ { \top } - v v ^ { \top } = u ( u - v ) ^ { \top } + ( u - v ) v ^ { \top } .
$$

Hence

$$
\begin{array} { r } { \| u u ^ { \top } - v v ^ { \top } \| _ { \mathrm { o p } } \leq \| u \| _ { 2 } \| u - v \| _ { 2 } + \| u - v \| _ { 2 } \| v \| _ { 2 } . } \end{array}
$$

Because $u , v \in \Delta ( S )$ , we have

$$
\begin{array} { r } { \| u \| _ { 2 } \leq \| u \| _ { 1 } = 1 , \qquad \| v \| _ { 2 } \leq \| v \| _ { 1 } = 1 , } \end{array}
$$

so

$$
\begin{array} { r } { \| u u ^ { \top } - v v ^ { \top } \| _ { \mathrm { o p } } \leq 2 \| u - v \| _ { 2 } \leq 2 \| u - v \| _ { 1 } . } \end{array}
$$

Apply this with $u = \bar { b } _ { t }$ and $v = \bar { b } _ { t } ^ { ( L ) }$ , then take expectations and use Proposition I.10. □

Proposition I.12 (Finite-window transforms shift the effective mixing gap). Let $( R _ { t } )$ be any process and let

$$
V _ { t } ^ { ( L ) } = h _ { L } \bigl ( R _ { t - L } , R _ { t - L + 1 } , \ldots , R _ { t - 1 } \bigr ) \qquad f o r e \nu e r y t \geq L + 1 .
$$

Then for every $\ell \geq L ,$

$$
\beta _ { V ^ { ( L ) } } ( \ell ) \leq \beta _ { R } ( \ell - L + 1 ) .
$$

Informally: a length-L truncated object shifts the effective mixing gap down by $L - 1$

Proof. Fix $r \geq L + 1$ and $\ell \geq L$ . By construction, the sigma-field generated by

$$
V _ { L + 1 } ^ { ( L ) } , V _ { L + 2 } ^ { ( L ) } , \dots , V _ { r } ^ { ( L ) }
$$

is contained in the sigma-field generated by

$$
R _ { 1 } , \ldots , R _ { r - 1 } .
$$

Indeed, the latest variable on the past side is $V _ { r } ^ { ( L ) }$ , and that variable only depends on

$$
R _ { r - L } , R _ { r - L + 1 } , \ldots , R _ { r - 1 } .
$$

Therefore

$$
\sigma ( V _ { L + 1 } ^ { ( L ) } , V _ { L + 2 } ^ { ( L ) } , \ldots , V _ { r } ^ { ( L ) } ) \subseteq \sigma ( R _ { 1 } , R _ { 2 } , \ldots , R _ { r - 1 } ) .
$$

Similarly, the sigma-field generated by

$$
V _ { r + \ell } ^ { ( L ) } , V _ { r + \ell + 1 } ^ { ( L ) } , \dots .
$$

is contained in the sigma-field generated by

$$
R _ { r + \ell - L } , R _ { r + \ell - L + 1 } , \dots .
$$

Indeed, the earliest variable on the future side is $V _ { r + \ell } ^ { ( L ) }$ , and that variable only depends on

$$
R _ { r + \ell - L } , R _ { r + \ell - L + 1 } , \ldots , R _ { r + \ell - 1 } .
$$

Therefore

$$
\sigma ( V _ { r + \ell } ^ { ( L ) } , V _ { r + \ell + 1 } ^ { ( L ) } , \ldots ) \subseteq \sigma ( R _ { r + \ell - L } , R _ { r + \ell - L + 1 } , \ldots ) .
$$

Now apply Lemma I.7:

$$
\begin{array} { r l } & { \quad \beta \Big ( \sigma ( V _ { L + 1 } ^ { ( L ) } , \ldots , V _ { r } ^ { ( L ) } ) , \sigma ( V _ { r + \ell } ^ { ( L ) } , V _ { r + \ell + 1 } ^ { ( L ) } , \ldots ) \Big ) } \\ & { \quad \le \beta \Big ( \sigma ( R _ { 1 } , \ldots , R _ { r - 1 } ) , \sigma ( R _ { r + \ell - L } , R _ { r + \ell - L + 1 } , \ldots ) \Big ) . } \end{array}
$$

The lag on the right-hand side is

$$
( r + \ell - L ) - ( r - 1 ) = \ell - L + 1 .
$$

So

$$
\beta \Big ( \sigma ( V _ { L + 1 } ^ { ( L ) } , \ldots , V _ { r } ^ { ( L ) } ) , \sigma ( V _ { r + \ell } ^ { ( L ) } , V _ { r + \ell + 1 } ^ { ( L ) } , \ldots ) \Big ) \leq \beta _ { R } ( \ell - L + 1 ) .
$$

Taking the supremum over r proves the result.

Lemma I.13 (Blocking concentration for bounded geometrically mixing scalar sequences). Let $( R _ { t } ) _ { t = 1 } ^ { m }$ be a scalar process with

$$
0 \leq R _ { t } \leq 1 \qquad a l m o s t s u r e l y f o r e \nu e r y t .
$$

Fix an integer q with $1 \leq q \leq m$ . Then for every $\varepsilon > 0 ,$

$$
\mathbb { P } \left( \left| \frac { 1 } { m } \sum _ { t = 1 } ^ { m } ( R _ { t } - \mathbb { E } [ R _ { t } ] ) \right| \geq \varepsilon \right) \leq \left( \frac { m } { q } + 1 \right) \beta _ { R } ( q ) + 4 \exp \left( - \frac { m \varepsilon ^ { 2 } } { q } \frac { m } { m + q } \right) .
$$

Proof. Let

$$
J = \left\lfloor { \frac { m } { q } } \right\rfloor .
$$

Partition $\{ 1 , 2 , \ldots , m \}$ into the J full blocks

$$
I _ { j } = \{ ( j - 1 ) q + 1 , \ldots , j q \} , \qquad j = 1 , \ldots , J ,
$$

and one remainder block

$$
\{ J q + 1 , \ldots , m \} ,
$$

whose size is strictly less than q.

For each full block define the block average

$$
U _ { j } = \frac { 1 } { q } \sum _ { t \in I _ { j } } R _ { t } .
$$

Similarly, for the remainder block, define the block average

$$
U _ { J + 1 } = \frac { 1 } { m - J q } \sum _ { t = J q + 1 } ^ { m } R _ { t } .
$$

Since $0 \leq R _ { t } \leq 1$ , we have

$$
0 \leq U _ { j } \leq 1 .
$$

Now decompose the centered empirical average:

$$
{ \frac { 1 } { m } } \sum _ { t = 1 } ^ { m } { \bigl ( } R _ { t } - \mathbb { E } [ R _ { t } ] { \bigr ) } = { \frac { q } { m } } \sum _ { j = 1 } ^ { J } { \bigl ( } U _ { j } - \mathbb { E } [ U _ { j } ] { \bigr ) } + { \frac { m - J q } { m } } { \bigl ( } U _ { J + 1 } - \mathbb { E } [ U _ { J + 1 } ] { \bigr ) } .
$$

Split the full-block sum into the odd blocks and even blocks. More specifically, if $J + 1$ is odd,

$$
S _ { \mathrm { o d d } } = \frac { q } { m } \sum _ { \stackrel { 1 \leq j \leq J } { j \mathrm { ~ o d d } } } \left( U _ { j } - \mathbb { E } [ U _ { j } ] \right) + \frac { m - J q } { m } \big ( U _ { J + 1 } - \mathbb { E } [ U _ { J + 1 } ] \big ) , \qquad S _ { \mathrm { e v e n } } = \frac { q } { m } \sum _ { \stackrel { 1 \leq j \leq J } { j \mathrm { ~ e v e n } } } \big ( U _ { j } - \mathbb { E } [ U _ { j } ] \big ) ,
$$

and if $J + 1$ is even,

$$
S _ { \mathrm { o d d } } = \frac { q } { m } \sum _ { \stackrel { 1 \leq j \leq J } { j \mathrm { ~ o d d } } } \left( U _ { j } - \mathbb { E } [ U _ { j } ] \right) , \qquad S _ { \mathrm { e v e n } } = \frac { q } { m } \sum _ { \stackrel { 1 \leq j \leq J } { j \mathrm { ~ e v e n } } } \left( U _ { j } - \mathbb { E } [ U _ { j } ] \right) + \frac { m - J q } { m } \left( U _ { J + 1 } - \mathbb { E } [ U _ { J + 1 } ] \right) .
$$

Then

$$
{ \frac { 1 } { m } } \sum _ { t = 1 } ^ { m } \bigl ( R _ { t } - \mathbb { E } [ R _ { t } ] \bigr ) = S _ { \mathrm { o d d } } + S _ { \mathrm { e v e n } } .
$$

Therefore, if

$$
\left| { \frac { 1 } { m } } \sum _ { t = 1 } ^ { m } ( R _ { t } - \mathbb { E } [ R _ { t } ] ) \right| \geq \varepsilon ,
$$

then

$$
| S _ { \mathrm { o d d } } + S _ { \mathrm { e v e n } } | \geq \varepsilon ,
$$

Hence at least one of the two block sums must satisfy

$$
| S _ { \mathrm { o d d } } | \geq \frac { 1 } { 2 } \varepsilon \qquad \mathrm { o r } \qquad | S _ { \mathrm { e v e n } } | \geq \frac { 1 } { 2 } \varepsilon .
$$

By the union bound,

$$
\mathbb P \Bigg ( \Bigg | \frac 1 m \sum _ { t = 1 } ^ { m } \big ( R _ { t } - \mathbb E [ R _ { t } ] \big ) \Bigg | \geq \varepsilon \Bigg ) \leq \mathbb P \big ( | S _ { \mathrm { o d d } } | \geq \frac 1 2 \varepsilon \big ) + \mathbb P \big ( | S _ { \mathrm { e v e n } } | \geq \frac 1 2 \varepsilon \big ) .
$$

We now bound the odd-block term. Let $r _ { \mathrm { o d d } }$ be the number of odd full blocks. Consecutive odd blocks are separated by a gap of length q. By Berbee’s coupling lemma (first introduced in Berbee [1979], also see Lemma 5.1 of Rio et al. [2017]), there exist independent copies

$$
\widetilde { U } _ { 1 } , \ldots , \widetilde { U } _ { r _ { \mathrm { o d d } } }
$$

with the same marginals as the odd block averages, such that the total coupling failure probability is at most

$$
r _ { \mathrm { o d d } } \beta _ { R } ( q ) .
$$

Hence

$$
{ \mathbb P } \big ( | S _ { \mathrm { o d d } } | \ge \varepsilon \big ) \le r _ { \mathrm { o d d } } \beta _ { R } ( q ) + { \mathbb P } \left( \left| \frac { q } { m } \sum _ { j = 1 } ^ { r _ { \mathrm { o d d } } } \bigl ( \widetilde { U } _ { j } - { \mathbb E } [ \widetilde { U } _ { j } ] \bigr ) \right| \ge \varepsilon \right) .
$$

To elaborate, the first term on the right-hand side corresponds to when there is a failure in coupling (i.e., there exists $1 \leq j \leq r _ { \mathrm { o d d } }$ such that $\widetilde { U } _ { j } \neq U _ { j } )$ . The second term corresponds to when the coupling is successful $( \mathrm { i . e . , } \tilde { U } _ { j } = U _ { j } , \forall 1 \leq j \leq r _ { \mathrm { o d d } } )$ . Conditioned on successful coupling,

$$
\mathbb { P } \left( \left| \frac { q } { m } \sum _ { j = 1 } ^ { r _ { \mathrm { o d d } } } ( \widetilde { U } _ { j } - \mathbb { E } [ \widetilde { U } _ { j } ] ) \right| \geq \frac { 1 } { 2 } \varepsilon \right) = \mathbb { P } \left( \left| \frac { q } { m } \sum _ { j = 1 } ^ { r _ { \mathrm { o d d } } } ( U _ { j } - \mathbb { E } [ U _ { j } ] ) \right| \geq \frac { 1 } { 2 } \varepsilon \right)
$$

(recall that $\mathbb { E } [ \widetilde { U } _ { j } ] = \mathbb { E } [ U _ { j } ]$ for all j by construction).

Equivalently,

$$
\mathbb { P } \big ( | S _ { \mathrm { o d d } } | \ge \frac { 1 } { 2 } \varepsilon \big ) \le r _ { \mathrm { o d d } } \beta _ { R } ( q ) + \mathbb { P } \left( \left| \frac { 1 } { r _ { \mathrm { o d d } } } \sum _ { j = 1 } ^ { r _ { \mathrm { o d d } } } \bigl ( \widetilde { U } _ { j } - \mathbb { E } [ \widetilde { U } _ { j } ] \bigr ) \right| \ge \frac { m \varepsilon } { 2 q r _ { \mathrm { o d d } } } \right) .
$$

Since the $\widetilde { U } _ { j }$ are independent and lie in [0, 1], Hoeffding’s inequality gives

$$
\begin{array} { r } { \mathbb { P } \big ( | S _ { \mathrm { o d d } } | \ge \frac { 1 } { 2 } \varepsilon \big ) \le r _ { \mathrm { o d d } } \beta _ { R } ( q ) + 2 \exp \left( - 2 r _ { \mathrm { o d d } } \left( \frac { m \varepsilon } { 2 q r _ { \mathrm { o d d } } } \right) ^ { 2 } \right) } \\ { = r _ { \mathrm { o d d } } \beta _ { R } ( q ) + 2 \exp \left( - \frac { 1 } { 2 } r _ { \mathrm { o d d } } \left( \frac { m \varepsilon } { q r _ { \mathrm { o d d } } } \right) ^ { 2 } \right) . } \end{array}
$$

Because

$$
r _ { \mathrm { o d d } } = \frac { \lfloor \frac { m } { q } \rfloor + 1 } { 2 } \leq \frac { 1 } { 2 } \frac { m } { q } + \frac { 1 } { 2 } ,
$$

we obtain

$$
\mathbb { P } \big ( | S _ { \mathrm { o d d } } | \geq \frac { 1 } { 2 } \varepsilon \big ) \leq r _ { \mathrm { o d d } } \beta _ { R } ( q ) + 2 \exp \left( - \frac { m \varepsilon ^ { 2 } } { q } \frac { m } { m + q } \right) .
$$

Exactly the same argument yields

$$
\mathbb { P } \big ( | S _ { \mathrm { e v e n } } | \geq \frac { 1 } { 2 } \varepsilon \big ) \leq r _ { \mathrm { e v e n } } \beta _ { R } ( q ) + 2 \exp \left( - \frac { m \varepsilon ^ { 2 } } { q } \frac { m } { m + q } \right) ,
$$

where $r _ { \mathrm { e v e n } }$ is the number of even full blocks.

Since

$$
r _ { \mathrm { o d d } } + r _ { \mathrm { e v e n } } = J + 1 \leq \frac { m } { q } + 1 ,
$$

combining the odd and even bounds gives

$$
\mathbb { P } \left( \left| \frac { 1 } { m } \sum _ { t = 1 } ^ { m } ( R _ { t } - \mathbb { E } [ R _ { t } ] ) \right| \geq \varepsilon \right) \leq \left( \frac { m } { q } + 1 \right) \beta _ { R } ( q ) + 4 \exp \left( - \frac { m \varepsilon ^ { 2 } } { q } \frac { m } { m + q } \right) .
$$

Lemma I.14 (Net argument for symmetric matrices). Let $\mathcal { N }$ be a 1/4-net ofthe unit sphere in $\mathbb { R } ^ { d } .$ and let M be a symmetric $d \times d$ matrix. Then

$$
\| M \| _ { \mathrm { o p } } \leq 2 \operatorname* { s u p } _ { v \in \mathcal { N } } | v ^ { \top } M v | .
$$

Proof. Let x be a unit vector such that

$$
| x ^ { \top } M x | = \| M \| _ { \mathrm { o p } } .
$$

Choose $v \in \mathcal N$ so that

$$
\| x - v \| _ { 2 } \leq { \frac { 1 } { 4 } } .
$$

Then

$$
\begin{array} { r l } & { | x ^ { \top } M x - v ^ { \top } M v | = | ( x - v ) ^ { \top } M x + v ^ { \top } M ( x - v ) | } \\ & { \qquad \leq \| x - v \| _ { 2 } \| M \| _ { \mathrm { o p } } \| x \| _ { 2 } + \| v \| _ { 2 } \| M \| _ { \mathrm { o p } } \| x - v \| _ { 2 } } \\ & { \qquad \leq \frac { 1 } { 2 } \| M \| _ { \mathrm { o p } } . } \end{array}
$$

Therefore

$$
\| M \| _ { \mathrm { o p } } = | x ^ { \top } M x | \leq | v ^ { \top } M v | + \frac 1 2 \| M \| _ { \mathrm { o p } } .
$$

Taking the supremum over $v \in \mathcal N$ gives

$$
\| M \| _ { \mathrm { o p } } \leq \operatorname* { s u p } _ { v \in \mathcal { N } } | v ^ { \top } M v | + \frac 1 2 \| M \| _ { \mathrm { o p } } .
$$

Rearranging yields the claim.

## I.3

In this subsection, we use the facts developed in Appendix I.2 to convert the results in Appendix I.1 into a high-probability lower-bound on $\sigma _ { \mathrm { m i n } } ( \bar { B } )$ .

Step 1: Concentrate the truncated empirical covariance around its mean. For each $t \geq L + 1$ the truncated matrix

$$
Z _ { t } ^ { ( L ) } = \bar { b } _ { t } ^ { ( L ) } \big ( \bar { b } _ { t } ^ { ( L ) } \big ) ^ { \top }
$$

is a measurable function of

$$
W _ { t - L } , W _ { t - L + 1 } , \dots , W _ { t - 1 } ,
$$

where recall that $W _ { t } = \left( o _ { t } , a _ { t } \right)$

For $t \leq L , Z _ { t } ^ { ( L ) }$ is a measurable function of

$$
W _ { 1 } , W _ { t - L + 1 } , \dots , W _ { t - 1 } .
$$

Furthermore, recall Case 2 of Proposition I.10, we can prepend a random string $W _ { t - L } , \cdots , W _ { 0 }$ and $Z _ { t } ^ { ( L ) }$ is still a measurable function of

$$
W _ { t - L } , \cdot \cdot \cdot , W _ { 0 } , W _ { 1 } , \cdot \cdot \cdot , W _ { t } .
$$

Then, by Proposition I.8 and Proposition I.12,

$$
\qquad \beta _ { Z ^ { ( L ) } } ( \ell ) \leq \beta _ { W } ( \ell - L + 1 ) \leq G ( \pi ) \rho ( \pi ) ^ { \ell - L } \qquad { \mathrm { f o r ~ e v e r y ~ } } \ell \geq L .
$$

Let $m = n - t _ { \mathrm { b u r n } }$ . Then, fix a unit vector $v \in \mathbb { R } ^ { | S | }$ and define the scalar process

$$
R _ { t } ( v ) = v ^ { \top } Z _ { t _ { \mathrm { b u r n } } + t } ^ { ( L ) } v , \qquad t = 1 , \ldots , m .
$$

Because $Z _ { t _ { \mathrm { b u r n } } + t } ^ { ( L ) }$ is positive semidefinite and $\bar { b } _ { t _ { \mathrm { b u r n } } + t } ^ { ( L ) } \in \Delta ( S )$

$$
0 \leq R _ { t } ( v ) \leq 1 .
$$

Moreover, since $R _ { t } ( v )$ is a measurable function of $Z _ { t _ { \mathrm { b u r n } } + t } ^ { ( L ) } ,$ it has no larger absolute-regularity coefficient:

$$
\begin{array} { r } { \beta _ { R ( v ) } ( \ell ) \le \beta _ { Z ^ { ( L ) } } ( \ell ) \le G ( \pi ) \rho ( \pi ) ^ { \ell - L } \qquad \mathrm { ~ f o r ~ e v e r y ~ } \ell \ge L . } \end{array}
$$

Now apply Lemma I.13 with block length q. Because $q \geq L ,$ , we obtain, for every $\varepsilon > 0$

$$
\begin{array} { r l r } {  { \mathbb P ( | \displaystyle \frac { 1 } { m } \sum _ { t = 1 } ^ { m } ( R _ { t } ( v ) - \mathbb E [ R _ { t } ( v ) ] ) | \geq \varepsilon ) \leq ( \displaystyle \frac { m } { q } + 1 ) \beta _ { R ( v ) } ( q ) + 4 \exp ( - \displaystyle \frac { m \varepsilon ^ { 2 } } { q } \displaystyle \frac { m } { m + q } ) } } \\ & { } & { \leq ( \displaystyle \frac { m } { q } + 1 ) G ( \pi ) \rho ( \pi ) ^ { q - L + 1 } + 4 \exp ( - \displaystyle \frac { m \varepsilon ^ { 2 } } { q } \displaystyle \frac { m } { m + q } ) . } \end{array}
$$

Setting

$$
q = L + \left\lceil \frac { \log { ( 2 ( m + 1 ) G ( \pi ) / \delta _ { 1 } ) } } { \log ( 1 / \rho ( \pi ) ) } \right\rceil\tag{31}
$$

is sufficient for

$$
\left( { \frac { m } { q } } + 1 \right) G ( \pi ) \rho ( \pi ) ^ { q - L } \leq { \frac { \delta _ { 1 } } { 2 } } .
$$

For readability, we retain q as a symbol and substitute (31) only at the end. The choice for L will be determined in Step 3.

Therefore,

$$
\mathbb { P } \left( \left| \frac { 1 } { m } \sum _ { t = 1 } ^ { m } \bigl ( R _ { t } ( v ) - \mathbb { E } [ R _ { t } ( v ) ] \bigr ) \right| \geq \varepsilon \right) \leq \frac { \delta _ { 1 } } { 2 } + 4 \exp \left( - \frac { m \varepsilon ^ { 2 } } { q } \frac { m } { m + q } \right) .
$$

Let $\mathcal { N }$ be a $1 / 4 \AA$ -net of the unit sphere in $\mathbb { R } ^ { | \boldsymbol { S } | }$ . It is a standard result (see Corollary 4.2.13 of Vershynin [2018]) that

$$
| { \mathcal { N } } | \leq 9 ^ { | { \mathcal { S } } | } .
$$

Choose

$$
\varepsilon = \frac { 1 } { m } \sqrt { q ( m + q ) \big ( \log ( 8 / \delta _ { 1 } ) + | S | \log 9 \big ) } .
$$

Then

$$
4 \exp \left( - \frac { m \varepsilon ^ { 2 } } { q } \frac { m } { m + q } \right) = \frac { \delta _ { 1 } } { 2 \cdot 9 ^ { | \cal S | } } .
$$

Hence, for each fixed $v \in \mathcal N$

$$
\mathbb { P } \Bigg ( \Bigg | \frac { 1 } { m } \sum _ { t = 1 } ^ { m } \big ( R _ { t } ( v ) - \mathbb { E } [ R _ { t } ( v ) ] \big ) \Bigg | \geq \varepsilon \Bigg ) \leq \frac { \delta _ { 1 } } { 2 } + \frac { \delta _ { 1 } } { 2 \cdot 9 ^ { | S | } } .
$$

Now, notice that the first term on the right-hand side corresponds to the probability of the coupling process of $R _ { t } ( v )$ being unsuccessful (recall the proof of Lemma I.13). In particular, by construction, the coupling of $R _ { t } ( v )$ would be unsuccessful only if the coupling of $Z _ { t } ^ { ( L ) }$ is unsuccessful. This means the probability of unsuccessful coupling for $Z _ { t } ^ { ( L ) }$ upper-bounds the probability that there exists some $v \in \mathcal N$ where $R _ { t } ( v )$ is unsuccessfully coupled. Applying again Berbee’s coupling lemma on $Z ^ { ( L ) }$ gives us that the coupling failure probability is

$$
\left( { \frac { m } { q } } + 1 \right) G ( \pi ) \rho ( \pi ) ^ { q - L } \leq { \frac { \delta _ { 1 } } { 2 } } .
$$

On the other hand, the second term $\textstyle { \bigl ( } { \frac { \delta _ { 1 } } { 2 \cdot 9 ^ { | S | } } } { \bigr ) }$ depends on the choice of v and is designed to be summable over the net.

Therefore, we take a union bound of the term $\frac { \delta _ { 1 } } { 2 . 9 ^ { | S | } }$ over $\mathcal { N }$ and it yields that, with probability at least $1 - \delta _ { 1 }$

$$
\operatorname* { s u p } _ { v \in \mathcal { N } } \left. v ^ { \top } \left( \frac { 1 } { m } \sum _ { { t = t _ { \mathrm { b u r n } + 1 } } } ^ { n } \big ( Z _ { t } ^ { ( L ) } - \mathbb { E } [ Z _ { t } ^ { ( L ) } ] \big ) \right) v \right. \leq \varepsilon .
$$

Applying Lemma I.14 to the symmetric matrix

$$
\frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \big ( Z _ { t } ^ { ( L ) } - \mathbb { E } [ Z _ { t } ^ { ( L ) } ] \big ) ,
$$

we conclude that with probability at least $1 - \delta _ { 1 }$

$$
\left\| \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \bigl ( Z _ { t } ^ { ( L ) } - \mathbb { E } [ Z _ { t } ^ { ( L ) } ] \bigr ) \right\| _ { \mathrm { o p } } \leq 2 \varepsilon .
$$

Substituting the choice of ε gives

$$
\left\| \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \left( Z _ { t } ^ { ( L ) } - \mathbb { E } [ Z _ { t } ^ { ( L ) } ] \right) \right\| _ { \mathrm { o p } } \leq 2 \frac { 1 } { m } \sqrt { q ( m + q ) \big ( \log ( 8 / \delta _ { 1 } ) + | \mathcal { S } | \log 9 \big ) } .
$$

Step 2: Control the truncation remainder. Write

$$
\begin{array} { r l r } {  { \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } ( Z _ { t } - \mathbb { E } [ Z _ { t } ] ) = \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } ( Z _ { t } ^ { ( L ) } - \mathbb { E } [ Z _ { t } ^ { ( L ) } ] ) } } \\ & { } & { \quad + \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } [ ( Z _ { t } - Z _ { t } ^ { ( L ) } ) - \mathbb { E } [ Z _ { t } - Z _ { t } ^ { ( L ) } ] ] . } \end{array}
$$

Define

$$
R = \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \Big [ \big ( Z _ { t } - Z _ { t } ^ { ( L ) } \big ) - \mathbb { E } \big [ Z _ { t } - Z _ { t } ^ { ( L ) } \big ] \Big ] .
$$

By the triangle inequality,

$$
\| R \| _ { \mathrm { o p } } \leq \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \| Z _ { t } - Z _ { t } ^ { ( L ) } \| _ { \mathrm { o p } } + \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \mathbb { E } \Big [ \| Z _ { t } - Z _ { t } ^ { ( L ) } \| _ { \mathrm { o p } } \Big ] .
$$

Proposition I.11 gives

$$
\mathbb { E } \Big [ \| Z _ { t } - Z _ { t } ^ { ( L ) } \| _ { \mathrm { o p } } \Big ] \leq 2 \operatorname* { m i n } \left( | { \cal S } | \exp \left( - \frac { \sigma _ { \cal O } ^ { 4 } } { C | { \cal S } | ^ { 2 } } L \right) , \sqrt { | { \cal S } | \log ( | { \cal S } | ) } \exp \left( - \sigma _ { \cal O } \sqrt { \frac { L } { C | { \cal S } | } } \right) \right)
$$

for every $t \geq t _ { \mathrm { b u r n } } + 1$ . Hence

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \mathbb { E } \Big [ \| Z _ { t } - Z _ { t } ^ { ( L ) } \| _ { \mathrm { o p } } \Big ] } \\ & { \displaystyle \leq 2 \operatorname* { m i n } \left( | \mathcal { S } | \exp \left( - \frac { \sigma _ { O } ^ { 4 } } { C | \mathcal { S } | ^ { 2 } } L \right) , \sqrt { | \mathcal { S } | \log ( | \mathcal { S } | ) } \exp \left( - \sigma _ { O } \sqrt { \frac { L } { C | \mathcal { S } | } } \right) \right) . } \end{array}
$$

The same bound holds for the expectation of the random average:

$$
\begin{array} { r l } & { \mathbb { E } \Bigg [ \frac { 1 } { m } \displaystyle \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \| Z _ { t } - Z _ { t } ^ { ( L ) } \| _ { \mathrm { o p } } \Bigg ] } \\ & { \le 2 \operatorname* { m i n } \left( | \mathcal { S } | \exp \left( - \frac { \sigma _ { O } ^ { 4 } } { C | \mathcal { S } | ^ { 2 } } L \right) , \sqrt { | \mathcal { S } | \log ( | \mathcal { S } | ) } \exp \left( - \sigma _ { O } \sqrt { \frac { L } { C | \mathcal { S } | } } \right) \right) . } \end{array}
$$

Therefore, by Markov’s inequality, with probability at least $1 - \delta _ { 2 }$

$$
\frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \| Z _ { t } - Z _ { t } ^ { ( L ) } \| _ { \mathrm { \partial p } } \leq \frac { 2 \operatorname* { m i n } \left( | S | \exp \left( - \frac { \sigma _ { O } ^ { 4 } } { C | S | ^ { 2 } } L \right) , \sqrt { | S | \log ( | S | ) } \exp \left( - \sigma _ { O } \sqrt { \frac { L } { C | S | } } \right) \right) } { \delta _ { 2 } } .
$$

On the same event,

$$
\| R \| _ { \mathrm { o p } } \leq 2 \operatorname* { m i n } \left( | \mathcal { S } | \exp \left( - \frac { \sigma _ { O } ^ { 4 } } { C | \mathcal { S } | ^ { 2 } } L \right) , \sqrt { | \mathcal { S } | \log ( | \mathcal { S } | ) } \exp \left( - \sigma _ { O } \sqrt { \frac { L } { C | \mathcal { S } | } } \right) \right) \left( 1 + \frac { 1 } { \delta _ { 2 } } \right) .
$$

Step 3: Combine the mean lower bound, the truncated concentration bound, and the truncation remainder bound. We have the exact identity

$$
\frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } Z _ { t } = \Sigma + \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \big ( Z _ { t } ^ { ( L ) } - \mathbb { E } [ Z _ { t } ^ { ( L ) } ] \big ) + R ,
$$

where

$$
\Sigma = \frac { 1 } { n - t _ { \mathrm { b u r n } } } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \mathbb { E } [ Z _ { t } ] = \frac { 1 } { n - t _ { \mathrm { b u r n } } } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \mathbb { E } [ \bar { b } _ { t } \bar { b } _ { t } ^ { \top } ] .
$$

Hence, by Weyl’s inequality,

$$
\lambda _ { \operatorname* { m i n } } \left( \frac { 1 } { m } \sum _ { { t = t _ { \mathrm { b u r n } } } + 1 } ^ { n } Z _ { t } \right) \ge \lambda _ { \operatorname* { m i n } } ( \Sigma ) - \left\| \frac { 1 } { m } \sum _ { { t = t _ { \mathrm { b u r n } } } + 1 } ^ { n } \big ( Z _ { t } ^ { ( L ) } - \mathbb { E } [ Z _ { t } ^ { ( L ) } ] \big ) \right\| _ { \mathrm { o p } } - \| R \| _ { \mathrm { o p } } .
$$

From Corollary I.4,

$$
\frac { 1 } { m } \sum _ { \substack { t = t _ { \mathrm { b u r n } } + 1 } } ^ { n } \mathbb { E } [ Z _ { t } ] \succeq \left| \boldsymbol { \mathcal { A } } \right| \pi _ { \mathrm { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \mathrm { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I ,
$$

By Step 1, with probability at least $1 - \delta _ { 1 }$

$$
\left\| \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \left( Z _ { t } ^ { ( L ) } - \mathbb { E } [ Z _ { t } ^ { ( L ) } ] \right) \right\| _ { \mathrm { o p } } \leq 2 \frac { 1 } { m } \sqrt { q ( m + q ) \big ( \log ( 8 / \delta _ { 1 } ) + | \mathcal { S } | \log 9 \big ) } .
$$

By Step 2, with probability at least $1 - \delta _ { 2 }$

$$
\| R \| _ { \mathrm { o p } } \leq 2 \operatorname* { m i n } \left( | \mathcal { S } | \exp \left( - \frac { \sigma _ { O } ^ { 4 } } { C | \mathcal { S } | ^ { 2 } } L \right) , \sqrt { | \mathcal { S } | \log ( | \mathcal { S } | ) } \exp \left( - \sigma _ { O } \sqrt { \frac { L } { C | \mathcal { S } | } } \right) \right) \left( 1 + \frac { 1 } { \delta _ { 2 } } \right) .
$$

By another union bound, with probability at least $1 - \delta _ { 1 } - \delta _ { 2 }$ all three estimates hold simultaneously. Therefore

$$
\begin{array} { r l } & { \lambda _ { \operatorname* { m i n } } \left( \cfrac { 1 } { m } \displaystyle \sum _ { t = t _ { \mathrm { b a r n } } + 1 } ^ { n } Z _ { t } \right) } \\ & { \geq | A | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { \mathscr { O } } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } } \\ & { - 2 \cfrac { 1 } { m } \sqrt { q ( m + q ) \big ( \log ( 8 / \delta _ { 1 } ) + | \mathcal { S } | \log 9 \big ) } } \\ & { - 2 \operatorname* { m i n } \left( | \mathcal { S } | \exp \left( - \frac { \sigma _ { \mathscr { O } } ^ { 4 } } { C | \mathcal { S } | ^ { 2 } } L \right) , \sqrt { | \mathcal { S } | \log ( | \mathcal { S } | ) } \exp \left( - \sigma \sigma \sqrt { \frac { L } { C | \mathcal { S } | } } \right) \right) \left( 1 + \frac { 1 } { \delta _ { 2 } } \right) . } \end{array}
$$

The choice

$$
\begin{array} { r } { L = \operatorname* { m i n } { \left( { \frac { C | S | ^ { 2 } } { \sigma _ { O } ^ { 4 } } } \log { \left( { \frac { 1 0 2 4 ( 1 + 1 / \delta _ { 2 } ) | S | } { 4 9 | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } ( d _ { \operatorname* { m i n } } ^ { \pi } ) ^ { 2 } } } \right) } , \right. } \\ { \left. { \frac { C | S | } { \sigma _ { O } ^ { 2 } } } \log ^ { 2 } { \left( { \frac { 1 0 2 4 ( 1 + 1 / \delta _ { 2 } ) \sqrt { | S | \log { | S | } } } { 4 9 | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } ( d _ { \operatorname* { m i n } } ^ { \pi } ) ^ { 2 } } } \right) } \right) } , } \end{array}\tag{32}
$$

gives

$$
\begin{array} { l } { { \mathrm { 2 m i n } \left( | \displaystyle \mathcal { S } | \exp \left( - \frac { \sigma _ { O } ^ { 4 } } { C | \mathcal { S } | ^ { 2 } } L \right) , \sqrt { | \mathcal { S } | \log ( | \mathcal { S } | ) } \exp \left( - \sigma _ { O } \sqrt { \frac { L } { C | \mathcal { S } | } } \right) \right) \left( 1 + \frac { 1 } { \delta _ { 2 } } \right) } } \\ { { \mathrm { } \leq \frac { 1 } { 8 } | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } . } } \end{array}
$$

This leads to a simpler bound:

$$
\begin{array} { r } { \lambda _ { \operatorname* { m i n } } \Bigg ( \displaystyle \frac { 1 } { m } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } Z _ { t } \Bigg ) \geq \displaystyle \frac { 7 } { 8 } | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } } \\ { - 2 \displaystyle \frac { 1 } { m } \sqrt { q ( m + q ) \big ( \log ( 8 / \delta _ { 1 } ) + | \mathcal { S } | \log 9 \big ) } . } \end{array}
$$

A simple calculation shows that it is sufficient to have

$$
m \ge \frac { 1 4 \left( \log ( 8 / \delta _ { 1 } ) + | S | \log 9 \right) } { | \mathcal { A } | ^ { 2 } \pi _ { \operatorname* { m i n } } ^ { 2 } \sigma _ { T } ^ { 4 } \sigma _ { O } ^ { 4 } ( d _ { \operatorname* { m i n } } ^ { \pi } ) ^ { 4 } } q
$$

for

$$
2 \frac { 1 } { m } \sqrt { q ( m + q ) \big ( \log ( 8 / \delta _ { 1 } ) + | \mathcal { S } | \log 9 \big ) } \leq \frac { 7 } { 8 } \cdot \frac { 7 } { 8 } | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } ,
$$

which means

$$
\begin{array} { r l } & { \lambda _ { \operatorname* { m i n } } \left( \cfrac { 1 } { m } \sum _ { { t = t _ { \mathrm { b u r n } } } + 1 } ^ { n } Z _ { t } \right) \ge \displaystyle \frac { 7 } { 8 } | { \cal A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( \cfrac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } - \frac { 7 } { 8 } \cdot \frac { 7 } { 8 } | { \cal A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( \cfrac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } } \\ & { \phantom { a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a } \left( \cfrac { 1 } { 2 } \right) } \\ & { \phantom { a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a } \left( \cfrac { 1 } { 2 } \right) } \end{array}
$$

Furthermore, we plug in the value of $q$ from Equation (31) and obtain

$$
\begin{array} { r l } & { q = L + \left\lceil \frac { \log \left( 2 ( m + 1 ) G ( \pi ) / \delta _ { 1 } \right) } { \log ( 1 / \rho ( \pi ) ) } \right\rceil } \\ & { \quad = \operatorname* { m i n } \left( \frac { C | \mathcal { S } | ^ { 2 } } { \sigma _ { O } ^ { 4 } } \log \left( \frac { 1 0 2 4 \left( 1 + 1 / \delta _ { 2 } \right) | \mathcal { S } | } { 4 9 | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( d _ { \operatorname* { m i n } } ^ { \pi } \right) ^ { 2 } } \right) , } \\ & { \quad \frac { C | \mathcal { S } | } { \sigma _ { O } ^ { 2 } } \log ^ { 2 } \left( \frac { 1 0 2 4 \left( 1 + 1 / \delta _ { 2 } \right) \sqrt { | \mathcal { S } | \log | \mathcal { S } | } } { 4 9 | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( d _ { \operatorname* { m i n } } ^ { \pi } \right) ^ { 2 } } \right) \right) + \left\lceil \frac { \log \left( 2 ( m + 1 ) G ( \pi ) / \delta _ { 1 } \right) } { \log \left( 1 / \rho ( \pi ) \right) } \right\rceil . } \end{array}
$$

Recall that $\begin{array} { r } { m = n - t _ { \mathrm { b u r n } } = n - \left\lceil \frac { \log \bigl ( 8 G ( \pi ) / d _ { \mathrm { m i n } } ^ { \pi } \bigr ) } { \log \bigl ( 1 / \rho ( \pi ) \bigr ) } \right\rceil - 1 . } \end{array}$

Now

$$
\sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } Z _ { t } = \left[ \begin{array} { c } { \overline { { b } } _ { t _ { \mathrm { b u r n } } + 1 } ^ { \top } } \\ { \overline { { b } } _ { t _ { \mathrm { b u r n } } + 2 } ^ { \top } } \\ { \vdots } \\ { \overline { { b } } _ { n } ^ { \top } } \end{array} \right] ^ { \top } \left[ \begin{array} { c } { \overline { { b } } _ { t _ { \mathrm { b u r n } } + 1 } ^ { \top } } \\ { \overline { { b } } _ { t _ { \mathrm { b u r n } } + 2 } ^ { \top } } \\ { \vdots } \\ { \overline { { b } } _ { n } ^ { \top } } \end{array} \right] ,
$$

so

$$
\sigma _ { \operatorname* { m i n } } \left( \left[ \begin{array} { c } { \left[ \overline { { b } } _ { t b u r n } ^ { \top } + 1 \right] } \\ { \overline { { b } } _ { t b u r n } ^ { \top } + 2 } \\ { \vdots } \\ { \overline { { b } } _ { n } ^ { \top } } \end{array} \right] \right) ^ { 2 } = m \lambda _ { \operatorname* { m i n } } \left( \frac { 1 } { m } \sum _ { t = t _ { \operatorname { b u r n } } + 1 } ^ { n } Z _ { t } \right) \geq \frac { 1 } { 1 2 } m | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( d _ { \operatorname* { m i n } } ^ { \pi } \right) ^ { 2 } .\tag{33}
$$

Finally, since

$$
\bar { B } _ { n } ^ { \top } \bar { B } _ { n } = \sum _ { t = 1 } ^ { n } Z _ { t } \succeq \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } Z _ { t } ,
$$

we have

$$
\sigma _ { \operatorname* { m i n } } ( { \bar { B } } _ { n } ) ^ { 2 } = \lambda _ { \operatorname* { m i n } } ( { \bar { B } } _ { n } ^ { \top } { \bar { B } } _ { n } ) \geq \lambda _ { \operatorname* { m i n } } \left( \sum _ { { t = t _ { \operatorname { b u r n } } + 1 } } ^ { n } Z _ { t } \right) .
$$

Thus the same lower bound also holds for $\sigma _ { \operatorname* { m i n } } ( { \bar { B } } _ { n } )$

In addition, we derive a sufficient condition on the lower bound of n for a more interpretable result. For ease of notation, let

$$
\gamma _ { \pi } : = \log \frac { 1 } { \rho ( \pi ) } > 0
$$

and

$$
D : = \frac { 1 4 \left( \log ( 8 / \delta _ { 1 } ) + | S | \log 9 \right) } { | A | ^ { 2 } \pi _ { \operatorname* { m i n } } ^ { 2 } \sigma _ { T } ^ { 4 } \sigma _ { O } ^ { 4 } ( d _ { \operatorname* { m i n } } ^ { \pi } ) ^ { 4 } } .
$$

Furthermore, define $\log _ { + } ( x ) : = \operatorname* { m a x } \{ \log x , 0 \}$

Then, Lemma I.15 gives us that it is sufficient to have

$$
n \geq t _ { \mathrm { b u r n } } + 2 D L + \frac { 2 D } { \gamma _ { \pi } } \left[ \log \frac { 4 G ( \pi ) } { \delta _ { 1 } } + \log _ { + } \left( \frac { 2 D } { \gamma _ { \pi } } \right) \right] ,
$$

where $D$ and $\gamma _ { \pi }$ only depends on the POMDP parameters and the behavior policy. This concludes the proof of Proposition I.1.

Lemma I.15. If

$$
n \geq t _ { \mathrm { b u r n } } + 2 D L + \frac { 2 D } { \gamma _ { \pi } } \left[ \log \frac { 4 G ( \pi ) } { \delta _ { 1 } } + \log _ { + } \left( \frac { 2 D } { \gamma _ { \pi } } \right) \right] ,
$$

then

$$
n \geq t _ { \mathrm { b u r n } } + D q .
$$

Proof. Recall that

$$
q = L - 1 + \left\lceil \frac { 1 } { \gamma _ { \pi } } \log \frac { 2 ( n - t _ { \mathrm { b u r n } } + 1 ) G ( \pi ) } { \delta _ { 1 } } \right\rceil .
$$

Since $\lceil x \rceil \leq x + 1$ , we have

$$
q \leq L + \frac { 1 } { \gamma _ { \pi } } \log \frac { 2 ( n - t _ { \mathrm { b u r n } } + 1 ) G ( \pi ) } { \delta _ { 1 } } .
$$

Therefore,

$$
D q \le D L + \frac { D } { \gamma _ { \pi } } \log \frac { 2 ( n - t _ { \mathrm { b u r n } } + 1 ) G ( \pi ) } { \delta _ { 1 } } .
$$

It is enough to show that the right-hand side is at most $n - t _ { \mathrm { b u r n } }$

We first prove

$$
\log \frac { n - t _ { \mathrm { b u r n } } + 1 } { 2 } \leq \frac { \gamma _ { \pi } { \left( n - t _ { \mathrm { b u r n } } \right) } } { 2 D } + \log _ { + } \left( \frac { 2 D } { \gamma _ { \pi } } \right) .
$$

If $\begin{array} { r } { \frac { 2 D } { \gamma _ { \pi } } < 1 } \end{array}$ , then $\begin{array} { r } { \log _ { + } \left( \frac { 2 D } { \gamma _ { \pi } } \right) = 0 . } \end{array}$ and

$$
\log \frac { n - t _ { \mathrm { b u r n } } + 1 } { 2 } \leq \log ( n - t _ { \mathrm { b u r n } } + 1 ) \leq n - t _ { \mathrm { b u r n } } \leq \frac { \gamma _ { \pi } ( n - t _ { \mathrm { b u r n } } ) } { 2 D } ,
$$

where the last inequality uses $\begin{array} { r } { \frac { 2 D } { \gamma _ { \pi } } < 1 } \end{array}$

If $\begin{array} { r } { \frac { 2 D } { \gamma _ { \pi } } \geq 1 } \end{array}$ , then

$$
\begin{array} { r l } & { \log \frac { n - t _ { \mathrm { b u r n } } + 1 } { 2 } = \log \left( \displaystyle \frac { 2 D } { \gamma _ { \pi } } \right) + \log \left( \frac { \gamma _ { \pi } \left( n - t _ { \mathrm { b u r n } } + 1 \right) } { 4 D } \right) } \\ & { \qquad \leq \log \left( \displaystyle \frac { 2 D } { \gamma _ { \pi } } \right) + \log \left( 1 + \frac { \gamma _ { \pi } \left( n - t _ { \mathrm { b u r n } } \right) } { 2 D } \right) } \\ & { \qquad \leq \log \left( \displaystyle \frac { 2 D } { \gamma _ { \pi } } \right) + \frac { \gamma _ { \pi } \left( n - t _ { \mathrm { b u r n } } \right) } { 2 D } } \\ & { \qquad = \log _ { + } \left( \displaystyle \frac { 2 D } { \gamma _ { \pi } } \right) + \frac { \gamma _ { \pi } \left( n - t _ { \mathrm { b u r n } } \right) } { 2 D } . } \end{array}
$$

Thus, in all cases,

$$
\log \frac { n - t _ { \mathrm { b u r n } } + 1 } { 2 } \leq \frac { \gamma _ { \pi } { \left( n - t _ { \mathrm { b u r n } } \right) } } { 2 D } + \log _ { + } \left( \frac { 2 D } { \gamma _ { \pi } } \right) .
$$

Using this bound,

$$
\begin{array} { r l r } {  { \log \frac { 2 ( n - t _ { \mathrm { b u r n } } + 1 ) G ( \pi ) } { \delta _ { 1 } } = \log \frac { 4 G ( \pi ) } { \delta _ { 1 } } + \log \frac { n - t _ { \mathrm { b u r n } } + 1 } { 2 } } } \\ & { } & { \leq \log \frac { 4 G ( \pi ) } { \delta _ { 1 } } + \log _ { + } ( \frac { 2 D } { \gamma _ { \pi } } ) + \frac { \gamma _ { \pi } ( n - t _ { \mathrm { b u r n } } ) } { 2 D } . } \end{array}
$$

Multiplying by $D / \gamma _ { \pi }$ , we get

$$
\frac { D } { \gamma _ { \pi } } \log \frac { 2 ( n - t _ { \mathrm { b u r n } } + 1 ) G ( \pi ) } { \delta _ { 1 } } \leq \frac { D } { \gamma _ { \pi } } \left[ \log \frac { 4 G ( \pi ) } { \delta _ { 1 } } + \log _ { + } \left( \frac { 2 D } { \gamma _ { \pi } } \right) \right] + \frac { n - t _ { \mathrm { b u r n } } } { 2 } .
$$

Therefore,

$$
D q \leq D L + \frac { D } { \gamma _ { \pi } } \left[ \log \frac { 4 G ( \pi ) } { \delta _ { 1 } } + \log _ { + } \left( \frac { 2 D } { \gamma _ { \pi } } \right) \right] + \frac { n - t _ { \mathrm { b u r n } } } { 2 } .
$$

Then, assuming

$$
\frac { n - t _ { \mathrm { b u r n } } } { 2 } \geq D L + \frac { D } { \gamma _ { \pi } } \left[ \log \frac { 4 G ( \pi ) } { \delta _ { 1 } } + \log _ { + } \left( \frac { 2 D } { \gamma _ { \pi } } \right) \right]
$$

gives us

$$
D q \leq D L + \frac { D } { \gamma _ { \pi } } \left[ \log \frac { 4 G ( \pi ) } { \delta _ { 1 } } + \log _ { + } \left( \frac { 2 D } { \gamma _ { \pi } } \right) \right] + \frac { n - t _ { \mathrm { b u r n } } } { 2 } \leq \frac { n - t _ { \mathrm { b u r n } } } { 2 } + \frac { n - t _ { \mathrm { b u r n } } } { 2 } = n - t _ { \mathrm { b u r n } } .
$$

We first give the formal Lemma statement.

Lemma J.1 (Formal). Under Assumption 3.8, 3.9, 3.2, and 3.11, let

$$
t _ { \mathrm { b u r n } } = 1 + \left\lceil \frac { \log \bigl ( 8 G ( \pi ) / d _ { \mathrm { m i n } } ^ { \pi } \bigr ) } { \log \bigl ( 1 / \rho ( \pi ) \bigr ) } \right\rceil ,
$$

$$
L = \operatorname* { m i n } \left( \frac { C | \mathcal { S } | ^ { 2 } } { \sigma _ { \mathcal { O } } ^ { 4 } } \log \left( \frac { 1 0 2 4 \left( 1 + 1 / \delta _ { 2 } \right) | \mathcal { S } | } { 4 9 | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { \mathcal { O } } ^ { 2 } \left( d _ { \operatorname* { m i n } } ^ { \pi } \right) ^ { 2 } } \right) , \frac { C | \mathcal { S } | } { \sigma _ { \mathcal { O } } ^ { 2 } } \log ^ { 2 } \left( \frac { 1 0 2 4 \left( 1 + 1 / \delta _ { 2 } \right) \sqrt { | \mathcal { S } | \log | \mathcal { S } | } } { 4 9 | \mathcal { A } | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { \mathcal { O } } ^ { 2 } \left( d _ { \operatorname* { m i n } } ^ { \pi } \right) ^ { 2 } } \right) \right) ,
$$

and

$$
D = \frac { 1 4 \left( \log ( 8 / \delta _ { 1 } ) + | S | \log 9 \right) } { | \mathcal { A } | ^ { 2 } \pi _ { \operatorname* { m i n } } ^ { 2 } \sigma _ { T } ^ { 4 } \sigma _ { O } ^ { 4 } ( d _ { \operatorname* { m i n } } ^ { \pi } ) ^ { 4 } } .
$$

If

$$
n \geq 2 D L + \frac { 2 D } { \log ( \frac { 1 } { \rho ( \pi ) } ) } \left[ \log \left( \frac { 4 G ( \pi ) } { \delta _ { 1 } } \right) + \operatorname* { m a x } \left\{ \log \left( \frac { 2 D } { \log ( \frac { 1 } { \rho ( \pi ) } ) } \right) , 0 \right\} \right] + t _ { \mathrm { b u r n } } ,
$$

then, with probability at least $1 - \delta _ { 1 } - \delta _ { 2 } - \delta _ { 3 }$

$$
\sigma _ { \operatorname* { m i n } } \left( { \bar { B } } _ { a } \right) ^ { 2 } \geq { \frac { 1 } { 1 2 } } ( n - t _ { \mathrm { b u r n } } ) | A | \pi _ { \operatorname* { m i n } } ^ { 2 } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( d _ { \operatorname* { m i n } } ^ { \pi } \right) ^ { 2 } - \sqrt { 2 ( n - t _ { \mathrm { b u r n } } ) \log { \frac { 2 | { \mathcal { S } } | | A | } { \delta _ { 3 } } } } - { \frac { 2 } { 3 } } \log { \frac { 2 | { \mathcal { S } } | | A | } { \delta _ { 3 } } }
$$

holds simultaneouslyfor all $\iota \in { \mathcal { A } } .$

Taking $\delta _ { 1 } = \delta _ { 2 } = { \textstyle { \frac { 1 } { 2 } } } \delta$ gives the more compact representation.

It is easy to see that Lemma J.1satisfies the assumptions of Proposition I.1. Then, from Proposition I.1, we have that with probability at least $1 - \delta _ { 1 } - \delta _ { 2 } ^ { \top }$

$$
\sigma _ { \mathrm { m i n } } ( \bar { B } ) ^ { 2 } \geq \frac { 1 } { 1 2 } ( n - t _ { \mathrm { b u r n } } ) | A | \pi _ { \mathrm { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } ( d _ { \mathrm { m i n } } ^ { \pi } ) ^ { 2 } .
$$

In this appendix, we inherent the convention from Appendix I where we omit the $\theta ^ { * }$ and denote the true belief vectors as $b _ { t }$ and true predictive belief vectors as ${ { \bar { b } } _ { t } }$ . In addition, we use $\| \cdot \| _ { o p }$ to denote matrix spectral norms.

Fix $a \in A .$ . Let

$$
\mathcal { F } _ { t } : = \sigma ( o _ { 1 } , a _ { 1 } , \ldots , o _ { t } , a _ { t } ) .
$$

Since ${ { \bar { b } } _ { t } }$ is the predictive belief at time $t ,$ it is $\mathcal { F } _ { t - 1 }$ -measurable. Define

$$
p _ { t } ( a ) : = \mathbb { P } ( a _ { t } = a \mid { \mathcal { F } } _ { t - 1 } ) .
$$

Furthermore, since $\bar { b } _ { t } ( s ) = \mathbb { P } ( s _ { t } = s | \mathcal { F } _ { t - 1 } )$ , it is easy to see that

$$
p _ { t } ( a ) = \sum _ { s } \bar { b } _ { t } ( s ) \sum _ { o } O _ { \varphi ^ { * } } ( o | s ) \pi ( a | o ) \geq \pi _ { \operatorname* { m i n } } \sum _ { s } \bar { b } _ { t } ( s ) = \pi _ { \operatorname* { m i n } } ,
$$

where the inequality follows from Assumption 3.2.

Define $\widetilde { B }$ as the submatrix of $\bar { B }$ whose rows only contain those of $\bar { b } _ { t }$ with $t \geq t _ { \mathrm { b u r n } } + 1$ , and $\widetilde { B } _ { a }$ as the submatrix of ${ \bar { B } } _ { a }$ whose rows only contain those of ${ { \bar { b } } _ { t } }$ with $t \geq t _ { \mathrm { b u r n } } + 1$ . Then,

$$
\widetilde { B } _ { a } ^ { \top } \widetilde { B } _ { a } = \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \mathbf { 1 } \{ a _ { t } = a \} \bar { b } _ { t } \bar { b } _ { t } ^ { \top } .
$$

Therefore

$$
\widetilde { B } _ { a } ^ { \top } \widetilde { B } _ { a } = \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } p _ { t } ( a ) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } + \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \bigr ( { \bf 1 } \{ a _ { t } = a \} - p _ { t } ( a ) \bigr ) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } .
$$

Since $p _ { t } ( a ) \geq \pi _ { \operatorname* { m i n } }$ , this implies

$$
\begin{array} { l } { { \displaystyle \widetilde { B } _ { a } ^ { \top } \widetilde { B } _ { a } \succeq \pi _ { \operatorname* { m i n } } \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \bar { b } _ { t } \bar { b } _ { t } ^ { \top } + \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \big ( \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) \big ) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } } } \\ { { \displaystyle = \pi _ { \operatorname* { m i n } } \widetilde { B } ^ { \top } \widetilde { B } + \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \big ( \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) \big ) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } . } } \end{array}
$$

For symmetric matrices, Weyl’s inequality gives

$$
\lambda _ { \operatorname* { m i n } } \bigl ( \widetilde { B } _ { a } ^ { \top } \widetilde { B } _ { a } \bigr ) \geq \pi _ { \operatorname* { m i n } } \lambda _ { \operatorname* { m i n } } \bigl ( \widetilde { B } ^ { \top } \widetilde { B } \bigr ) + \lambda _ { \operatorname* { m i n } } \left( \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \bigl ( { \mathbf { 1 } \{ a _ { t } = a \} } - p _ { t } ( a ) \bigr ) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } \right) .\tag{34}
$$

Since

$$
\sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } ( \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) ) \bar { b } _ { t } \bar { b } _ { t } ^ { \top }
$$

is symmetric,

$$
\begin{array} { r l } & { \quad \left\| \displaystyle \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \big ( \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) \big ) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } \right\| _ { \mathrm { o p } } } \\ & { = \operatorname* { m a x } \left( \left| \lambda _ { \operatorname* { m i n } } \left( \displaystyle \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \big ( \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) \big ) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } \right) \right| , \right. } \\ & { \quad \quad \quad \left. \left| \lambda _ { \operatorname* { m a x } } \left( \displaystyle \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \big ( \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) \big ) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } \right) \right| \right) . } \end{array}
$$

Therefore,

$$
\lambda _ { \operatorname* { m i n } } \left( \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \left( \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) \right) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } \right) \geq - \left\| \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \left( \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) \right) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } \right\| _ { \mathrm { o p } } .\tag{35}
$$

Plugging (35) into (34) gives:

$$
\lambda _ { \operatorname* { m i n } } ( \widetilde { B } _ { a } ^ { \top } \widetilde { B } _ { a } ) \geq \pi _ { \operatorname* { m i n } } \lambda _ { \operatorname* { m i n } } ( \widetilde { B } ^ { \top } \widetilde { B } ) - \left\| \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } ( \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) ) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } \right\| _ { \mathrm { o p } } .\tag{36}
$$

Now define the self-adjoint matrix increments

$$
X _ { t } ( a ) : = \bigl ( \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) \bigr ) \bar { b } _ { t } \bar { b } _ { t } ^ { \top } , \qquad t = t _ { \mathrm { b u r n } } + 1 , \ldots , n .
$$

Since ${ { \bar { b } } _ { t } }$ is $\mathcal { F } _ { t - 1 }$ -measurable and

$$
\mathbb { E } [ \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) \ | \ \mathcal { F } _ { t - 1 } ] = 0 ,
$$

we have

$$
\mathbb { E } [ X _ { t } ( a ) \mid { \mathcal { F } } _ { t - 1 } ] = 0 .
$$

Hence the partial sums of $X _ { t } ( a )$ form a self-adjoint matrix martingale.

Because $\bar { b } _ { t }$ is a probability vector,

$$
\| X _ { t } ( a ) \| _ { \mathrm { o p } } \leq \| \bar { b } _ { t } \bar { b } _ { t } ^ { \top } \| _ { \mathrm { o p } } = \| \bar { b } _ { t } \| _ { 2 } ^ { 2 } \leq \| \bar { b } _ { t } \| _ { 1 } ^ { 2 } = 1 .
$$

Also,

$$
X _ { t } ( a ) ^ { 2 } = \big ( \mathbf { 1 } \{ a _ { t } = a \} - p _ { t } ( a ) \big ) ^ { 2 } ( \hat { b } _ { t } \hat { b } _ { t } ^ { \top } ) ^ { 2 } .
$$

Taking conditional expectations,

$$
\mathbb { E } \left[ X _ { t } ( a ) ^ { 2 } \mid \mathcal { F } _ { t - 1 } \right] = p _ { t } ( a ) \left( 1 - p _ { t } ( a ) \right) ( \bar { b } _ { t } \bar { b } _ { t } ^ { \top } ) ^ { 2 } \preceq ( \bar { b } _ { t } \bar { b } _ { t } ^ { \top } ) ^ { 2 } .
$$

Since

$$
( \bar { b } _ { t } \bar { b } _ { t } ^ { \top } ) ^ { 2 } = \| \bar { b } _ { t } \| _ { 2 } ^ { 2 } \bar { b } _ { t } \bar { b } _ { t } ^ { \top } \preceq \bar { b } _ { t } \bar { b } _ { t } ^ { \top } ,
$$

it follows that

$$
\mathbb { E } \big [ X _ { t } ( a ) ^ { 2 } \mid \mathcal { F } _ { t - 1 } \big ] \preceq \bar { b } _ { t } \bar { b } _ { t } ^ { \top } .
$$

Therefore

$$
\left\| \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \mathbb { E } \big [ X _ { t } ( a ) ^ { 2 } \mid \mathcal { F } _ { t - 1 } \big ] \right\| _ { \mathrm { o p } } \leq \left\| \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \bar { b } _ { t } \bar { b } _ { t } ^ { \top } \right\| _ { \mathrm { o p } } = \| \widetilde { B } ^ { \top } \widetilde { B } \| _ { \mathrm { o p } } .
$$

By triangle inequality of the operator norm,

$$
\lVert \widetilde { B } ^ { \top } \widetilde { B } \rVert _ { \mathrm { o p } } \leq \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } \lVert \bar { b } _ { t } \bar { b } _ { t } ^ { \top } \rVert _ { \mathrm { o p } } \leq n - t _ { \mathrm { b u r n } } .
$$

Now apply matrix Freedman’s inequality to the self-adjoint matrix martingale

$$
\sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } X _ { t } ( a ) .
$$

Since the matrix dimension is $| S |$ , the increment bound is 1, and the predictable quadratic variation bound is $n - t _ { \mathrm { b u r n } }$ , matrix Freedman’s inequality implies that for every $x > 0$

$$
\mathbb { P } \left( \left. \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } X _ { t } ( a ) \right. _ { \mathrm { o p } } \geq x \right) \leq 2 | S | \exp \left( - \frac { x ^ { 2 } } { 2 \left( \left( n - t _ { \mathrm { b u r n } } \right) + x / 3 \right) } \right) .
$$

We now choose x so that the right-hand side is at most $\delta / | { \mathcal { A } } |$ . It is enough to require

$$
2 | S | \exp \left( - \frac { x ^ { 2 } } { 2 \left( \left( n - t _ { \mathrm { b u r n } } \right) + x / 3 \right) } \right) \leq \frac { \delta } { | A | } .
$$

Taking logarithms, this is equivalent to

$$
\frac { x ^ { 2 } } { 2 \left( \left( n - t _ { \mathrm { b u r n } } \right) + x / 3 \right) } \geq \log \frac { 2 | \cal { S } | | \cal { A } | } { \delta } .
$$

Multiplying both sides by

$$
2 \left( { \left( n - t _ { \mathrm { b u r n } } \right) + x / 3 } \right)
$$

gives

$$
x ^ { 2 } \geq 2 ( n - t _ { \mathrm { b u r n } } ) \log \frac { 2 | \mathcal { S } | | \mathcal { A } | } { \delta } + \frac { 2 x } { 3 } \log \frac { 2 | \mathcal { S } | | \mathcal { A } | } { \delta } .
$$

Equivalently,

$$
x ^ { 2 } - \frac { 2 x } { 3 } \log \frac { 2 | \mathcal { S } | | \mathcal { A } | } { \delta } - 2 ( n - t _ { \mathrm { b u r n } } ) \log \frac { 2 | \mathcal { S } | | \mathcal { A } | } { \delta } \geq 0 .
$$

The corresponding quadratic equation is

$$
x ^ { 2 } - \frac { 2 x } { 3 } \log \frac { 2 | \mathcal { S } | | \mathcal { A } | } { \delta } - 2 ( n - t _ { \mathrm { b u r n } } ) \log \frac { 2 | \mathcal { S } | | \mathcal { A } | } { \delta } = 0 .
$$

Its positive root is

$$
x = \frac { 1 } { 3 } \log \frac { 2 | \mathcal { S } | | \mathcal { A } | } { \delta } + \sqrt { 2 ( n - t _ { \mathrm { b u r n } } ) \log \frac { 2 | \mathcal { S } | | \mathcal { A } | } { \delta } + \frac { 1 } { 9 } \left( \log \frac { 2 | \mathcal { S } | | \mathcal { A } | } { \delta } \right) ^ { 2 } } .
$$

Therefore, with probability at least $1 - \delta / | A |$

$$
\left\| \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } X _ { t } ( a ) \right\| _ { \mathrm { o p } } \leq \frac { 1 } { 3 } \log \frac { 2 | S | | A | } { \delta } + \sqrt { 2 ( n - t _ { \mathrm { b u r n } } ) \log \frac { 2 | S | | A | } { \delta } + \frac { 1 } { 9 } \left( \log \frac { 2 | S | | A | } { \delta } \right) ^ { 2 } } .
$$

Finally, using

$$
{ \sqrt { u + v } } \leq { \sqrt { u } } + { \sqrt { v } }
$$

with

$$
u = 2 ( n - t _ { \mathrm { b u r n } } ) \log \frac { 2 | \cal { S } | | { \cal { A } } | } { \delta } , \qquad v = \frac { 1 } { 9 } \left( \log \frac { 2 | \cal { S } | | { \cal { A } } | } { \delta } \right) ^ { 2 } ,
$$

we obtain the simpler bound

$$
\left\| \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } X _ { t } ( a ) \right\| _ { \mathrm { o p } } \le \sqrt { 2 ( n - t _ { \mathrm { b u r n } } ) \log \frac { 2 | S | | A | } { \delta } } + \frac { 2 } { 3 } \log \frac { 2 | S | | A | } { \delta }
$$

with probability at least $1 - \delta / | A |$

A union bound over all $a \in { \mathcal { A } }$ shows that, with probability at least $1 - \delta .$

$$
\left\| \sum _ { t = t _ { \mathrm { b u r n } } + 1 } ^ { n } X _ { t } ( a ) \right\| _ { \mathrm { o p } } \le \sqrt { 2 ( n - t _ { \mathrm { b u r n } } ) \log \frac { 2 | S | | A | } { \delta } } + \frac { 2 } { 3 } \log \frac { 2 | S | | A | } { \delta }
$$

simultaneously for all $a \in A .$

Plugging this into (36) gives that with probability at least $1 - \delta ,$ for every $a \in { \mathcal { A } } .$

$$
\lambda _ { \operatorname* { m i n } } \big ( \widetilde { B } _ { a } ^ { \top } \widetilde { B } _ { a } \big ) \geq \pi _ { \operatorname* { m i n } } \lambda _ { \operatorname* { m i n } } \big ( \widetilde { B } ^ { \top } \widetilde { B } \big ) - \sqrt { 2 \big ( n - t _ { \mathrm { b u r n } } \big ) \log \frac { 2 \big | S \big | \big | A \big | } { \delta } } - \frac { 2 } { 3 } \log \frac { 2 \big | S \big | \big | A \big | } { \delta } .
$$

Recall that Lemma J.1satisfies the assumptions of Proposition I.1, so the results of Appendix I apply directly. In particular, (33) yields with probability at least $1 - \delta _ { 1 } - \delta _ { 2 }$

$$
\lambda _ { \operatorname* { m i n } } ( \widetilde { B } ^ { \top } \widetilde { B } ) \geq \frac { 1 } { 1 2 } m | A | \pi _ { \operatorname* { m i n } } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } \left( d _ { \operatorname* { m i n } } ^ { \pi } \right) ^ { 2 } .
$$

Finally, setting δ to $\delta _ { 3 }$ , combining the two inequalities, and plugging in $\sigma _ { \operatorname* { m i n } } ( \bar { B } _ { a } ) ^ { 2 } \geq \sigma _ { \operatorname* { m i n } } ( \widetilde { B } _ { a } ) ^ { 2 } =$ $\lambda _ { \operatorname* { m i n } } ( \widetilde { B } _ { a } ^ { \top } \widetilde { B } _ { a } )$ proves the claim. Furthermore, setting $\begin{array} { r } { \delta _ { 1 } = \delta _ { 2 } = \delta _ { 3 } = \frac { \delta } { 3 } } \end{array}$ recovers the statement in Lemma 3.13.

## K

In this appendix, we present the proof for Theorem 3.14. We work on the intersection of the highprobability events in Lemma 3.13 and Corollary G.1, each invoked with confidence parameter $\delta / 2$ By a union bound, this event has probability at least $1 - \delta$

Fix any candidate $\hat { \theta } = ( \hat { \beta } , \hat { \varphi } )$ satisfying

$$
\ell _ { n } ( \theta ^ { * } ) - \ell _ { n } ( { \hat { \theta } } ) \leq G _ { n } .
$$

Apply Corollary G.1 with approximation radius $\varepsilon = 1 / n$ . There exists $\tilde { { \boldsymbol { \theta } } } = ( \tilde { \beta } , \tilde { \varphi } ) \in \Theta$ such that

$$
\| { \tilde { \theta } } - { \hat { \theta } } \| _ { \infty } \leq { \frac { 1 } { n } }
$$

and

$$
\sum _ { k = 1 } ^ { n } \left( \sum _ { o , o ^ { \prime } \in \Omega } \left| \sum _ { s \in S } \bar { b } _ { k , \tilde { \theta } } ( s ) \mathbb { P } _ { \tilde { \theta } } ( o , o ^ { \prime } \mid s , a _ { k } ) - \sum _ { s \in S } \bar { b } _ { k , \theta ^ { * } } ( s ) \mathbb { P } _ { \theta ^ { * } } ( o , o ^ { \prime } \mid s , a _ { k } ) \right| \right) ^ { 2 }\tag{37}
$$

$$
\leq \frac { 8 } { \pi _ { \mathrm { m i n } } } \left( G _ { n } + 2 d \log ( 2 | \Omega | n ) + 2 \log ( 4 / \delta ) + 2 \right) + 4 .
$$

Define

$$
H _ { n } ( \delta ) = G _ { n } + 2 d \log ( 2 | \Omega | n ) + 2 \log ( 4 / \delta ) + 2
$$

and

$$
\eta _ { n } ^ { 2 } = \frac { 1 } { n } \left( \frac { 8 H _ { n } ( \delta ) } { \pi _ { \mathrm { m i n } } } + 4 \right) .
$$

In particular,

$$
\eta _ { n } = O \left( \sqrt { \frac { G _ { n } + d \log ( 2 | \Omega | n ) + \log ( 1 / \delta ) + 1 } { n \pi _ { \operatorname* { m i n } } } } \right) .
$$

Step 1: Constructing the predictive tensors. Denote by $\bar { B } _ { a } ( \theta )$ the action-restricted predictivebelief matrix whose rows are $\bar { b } _ { k , \theta } ^ { \top } .$ , for $k \in \mathcal { T } _ { a }$

Define the forward observation matrix

$$
F _ { a } ( \theta ) = O _ { \varphi } T _ { \beta } ^ { a } \in \mathbb { R } ^ { | \Omega | \times | S | } , \qquad \theta = ( \beta , \varphi ) .
$$

For each $a \in { \mathcal { A } } ,$ introduce the tensor $\mathcal { X } _ { a } ( \theta ) \in \mathbb { R } ^ { | \mathcal { T } _ { a } | \times | \Omega | \times | \Omega | }$ defined by

$$
[ \mathcal { X } _ { a } ( \theta ) ] _ { k , o , o ^ { \prime } } = \frac { 1 } { \sqrt { n } } \sum _ { s \in S } \bar { b } _ { k , \theta } ( s ) \mathbb { P } _ { \theta } ( o , o ^ { \prime } \mid s , a ) , \qquad k \in \mathcal { Z } _ { a } .
$$

Since

$$
\mathbb { P } _ { \theta } \big ( o , o ^ { \prime } \mid s , a \big ) = O _ { \varphi } ( o \mid s ) \sum _ { s ^ { \prime } \in S } T _ { \beta } ^ { a } ( s ^ { \prime } \mid s ) O _ { \varphi } ( o ^ { \prime } \mid s ^ { \prime } ) ,
$$

the tensor admits the rank-S decomposition

$$
\mathcal X _ { a } ( \boldsymbol \theta ) = \sum _ { s = 1 } ^ { S } \left( \frac { \bar { B } _ { a } ( \boldsymbol \theta ) ( : , s ) } { \sqrt { n } } \right) \otimes O _ { \varphi } ( : , s ) \otimes F _ { a } ( \boldsymbol \theta ) ( : , s ) .\tag{38}
$$

The normalization by $\sqrt { n }$ will ensure that the factor matrices are uniformly bounded in operator norm.

For every action $^ { a , }$ the inequality $\begin{array} { r } { \sum _ { j } x _ { j } ^ { 2 } \le ( \sum _ { j } | x _ { j } | ) ^ { 2 } } \end{array}$ , together with (37), yields

$$
\Vert \mathcal { X } _ { a } ( \tilde { \theta } ) - \mathcal { X } _ { a } ( \theta ^ { * } ) \Vert _ { F } ^ { 2 } = \frac { 1 } { n } \sum _ { k \in \mathcal { T } _ { a } } \sum _ { o , o ^ { \prime } } \left| \sum _ { s } \bar { b } _ { k , \tilde { \theta } } ( s ) \mathbb { P } _ { \tilde { \theta } } ( o , o ^ { \prime } \mid s , a ) - \sum _ { s } \bar { b } _ { k , \theta ^ { * } } ( s ) \mathbb { P } _ { \theta ^ { * } } ( o , o ^ { \prime } \mid s , a ) \right| ^ { 2 }
$$

Thus, simultaneously for all $a \in { \mathcal { A } } .$

$$
\lVert \mathcal { X } _ { a } ( \widetilde { \theta } ) - \mathcal { X } _ { a } ( \theta ^ { * } ) \rVert _ { F } \leq \eta _ { n } .\tag{39}
$$

Notice that the same bound also controls the one-observation marginals. Indeed, summing over $o ^ { \prime }$ gives

$$
\sum _ { o ^ { \prime } } \mathbb { P } _ { \theta } ( o , o ^ { \prime } \mid s , a ) = O _ { \varphi } ( o \mid s ) ,
$$

and therefore

$$
\frac { 1 } { n } \sum _ { k \in \mathcal { T } _ { a } } \sum _ { o } \left. \sum _ { s } \bar { b } _ { k , \widetilde { \theta } } ( s ) O _ { \widetilde { \varphi } } ( o \mid s ) - \sum _ { s } \bar { b } _ { k , \theta ^ { * } } ( s ) O _ { \varphi ^ { * } } ( o \mid s ) \right. ^ { 2 } \leq \eta _ { n } ^ { 2 } .\tag{40}
$$

Hence Corollary G.1 supplies both the joint and marginal perturbation bounds using the same approximating parameter ${ \tilde { \theta } } .$

Step 2: Conditioning of the three factors. By Assumptions 3.2, 3.8, 3.9, and 3.11, the observation and transition kernels satisfy

$$
\sigma _ { \operatorname* { m i n } } ( O _ { \varphi ^ { * } } ) \geq \sigma _ { O } , \qquad \sigma _ { \operatorname* { m i n } } ( T _ { \beta ^ { * } } ^ { a } ) \geq \sigma _ { T } \quad \mathrm { f o r ~ e v e r y ~ } a \in \mathcal A .
$$

Consequently,

$$
\sigma _ { \operatorname* { m i n } } ( F _ { a } ( \theta ^ { * } ) ) = \sigma _ { \operatorname* { m i n } } ( O _ { \varphi ^ { * } } T _ { \beta ^ { * } } ^ { a } ) \geq \sigma _ { O } \sigma _ { T } .\tag{41}
$$

Furthermore, Lemma 3.13 guarantees that, simultaneously for all $a \in { \mathcal { A } }$

$$
\begin{array} { r } { \sigma _ { \operatorname* { m i n } } ( \bar { B } _ { a } ( \theta ^ { * } ) ) ^ { 2 } \geq c ( n - t _ { \mathrm { b u r n } } ) | A | \pi _ { \operatorname* { m i n } } ^ { 2 } \sigma _ { T } ^ { 2 } \sigma _ { O } ^ { 2 } ( d _ { \operatorname* { m i n } } ^ { \pi } ) ^ { 2 } . } \end{array}
$$

For $n \geq 2 t _ { \mathrm { b u r n } } .$ , this implies

$$
\sigma _ { \operatorname* { m i n } } \left( \frac { \bar { B } _ { a } ( \theta ^ { * } ) } { \sqrt { n } } \right) \geq \sqrt { \frac { c | A | } { 2 } } \pi _ { \operatorname* { m i n } } \sigma _ { T } \sigma _ { \mathcal O } d _ { \operatorname* { m i n } } ^ { \pi } = : \kappa _ { B } > 0 .\tag{42}
$$

Set

$$
\kappa = \operatorname* { m i n } \{ \kappa _ { B } , \sigma _ { O } , \sigma _ { O } \sigma _ { T } \} .
$$

Equations (41) and (42) show that all three factor matrices in (38) have full column rank, with smallest singular values at least κ.

Moreover, because predictive beliefs are probability vectors,

$$
\left. \frac { \bar { B } _ { a } ( \theta ) } { \sqrt { n } } \right. _ { 2 } \leq \left. \frac { \bar { B } _ { a } ( \theta ) } { \sqrt { n } } \right. _ { F } \leq 1
$$

for every $\theta \in \Theta$ . Similarly, since the columns of $O _ { \varphi }$ and $F _ { a } ( \theta )$ are probability vectors,

$$
\| O _ { \varphi } \| _ { 2 } \leq \sqrt { | S | } , \qquad \| F _ { a } ( \theta ) \| _ { 2 } \leq \sqrt { | S | } .
$$

Thus the three factors have quantitative full robust Kruskal rank $| S |$ . For $| S | \geq 2$ , their Kruskal rank satisfy $3 | S | \geq 2 | S | + 2 .$ , so the Kruskal uniqueness condition holds.

Step 3: Quantitative stability via robust Kruskal. We apply Theorem S.1 to the two tensor decompositions in (38).

Fix $a \in { \mathcal { A } } .$ , and abbreviate

$$
B _ { a } ^ { \mathrm { n o r m } } ( \theta ^ { * } ) = \frac { \bar { B } _ { a } ( \theta ^ { * } ) } { \sqrt { n } } , \qquad B _ { a } ^ { \mathrm { n o r m } } ( \tilde { \theta } ) = \frac { \bar { B } _ { a } ( \tilde { \theta } ) } { \sqrt { n } } ,
$$

$$
O ^ { * } = O _ { \varphi ^ { * } } , \qquad \widetilde { O } = O _ { \widetilde { \varphi } } , \qquad F _ { a } ^ { * } = F _ { a } ( \theta ^ { * } ) , \qquad \widetilde { F } _ { a } = F _ { a } ( \widetilde { \theta } ) .
$$

Thus

$$
\begin{array} { r } { \mathcal { X } _ { a } ( \theta ^ { * } ) = [ B _ { a } ^ { \mathrm { n o r m } } ( \theta ^ { * } ) , O ^ { * } , F _ { a } ^ { * } ] , \qquad \mathcal { X } _ { a } ( \widetilde { \theta } ) = [ B _ { a } ^ { \mathrm { n o r m } } ( \widetilde { \theta } ) , \widetilde { O } , \widetilde { F } _ { a } ] . } \end{array}
$$

By (41) and (42),

$$
\sigma _ { \operatorname* { m i n } } \bigl ( B _ { a } ^ { \mathrm { n o r m } } ( \theta ^ { * } ) \bigr ) \geq \kappa _ { B } , \qquad \sigma _ { \operatorname* { m i n } } ( O ^ { * } ) \geq \sigma _ { O } , \qquad \sigma _ { \operatorname* { m i n } } \bigl ( F _ { a } ^ { * } \bigr ) \geq \sigma _ { O } \sigma _ { T } .
$$

In particular, all three factors have full robust Kruskal rank $S ,$ with parameters

$$
\tau _ { B } = \kappa _ { B } ^ { - 1 } , \qquad \tau _ { O } = \sigma _ { O } ^ { - 1 } , \qquad \tau _ { F } = ( \sigma _ { O } \sigma _ { T } ) ^ { - 1 } .
$$

Indeed, the minimum singular value of every column submatrix is at least that of the full matrix. Consequently, for $| S | \geq 2 ,$

$$
\begin{array} { r } { \mathrm { K r a n k } _ { \tau _ { B } } ( B _ { a } ^ { \mathrm { n o r m } } ( \theta ^ { * } ) ) + \mathrm { K r a n k } _ { \tau _ { O } } ( O ^ { * } ) + \mathrm { K r a n k } _ { \tau _ { F } } ( F _ { a } ^ { * } ) = 3 | S | \geq 2 | S | + 2 . } \end{array}
$$

Furthermore, every column of $B _ { a } ^ { \mathrm { n o r m } } ( \theta ^ { * } )$ and $B _ { a } ^ { \mathrm { n o r m } } ( \tilde { \theta } )$ has Euclidean norm at most one, since the predictive beliefs are probability vectors and the matrices are normalized by $\sqrt { n }$ . Every column of $O ^ { * } , \widetilde { O } , F _ { a } ^ { * }$ , and ${ \widetilde { F } } _ { a }$ is also a probability vector and hence has Euclidean norm at most one. Thus both decompositions are (1, 1, 1)-bounded.

Let

$$
{ \cal D } _ { \mathrm { R K } } = \operatorname * { m a x } \left\{ 1 , | { \cal S } | ^ { c } \prod _ { L \in \{ B , { \cal O } , F \} } \vartheta _ { 5 } ( \tau _ { L } , 1 ) \right\} ,
$$

where c and $\vartheta _ { 5 }$ are the universal constant and polynomial in Theorem S.1. Importantly, D<sub>RK</sub> depends polynomially on

$$
| S | , \quad \kappa _ { B } ^ { - 1 } , \quad \sigma _ { O } ^ { - 1 } , \quad \sigma _ { T } ^ { - 1 } ,
$$

but not on the ambient dimension $| { \mathcal { T } } _ { a } |$ , and hence not on n.

Set

$$
\xi _ { n } = D _ { \mathrm { R K } } \eta _ { n } .
$$

Under the stated sample-size requirement, we may assume

$$
\xi _ { n } \leq \frac { 1 } { 4 \sqrt { | \Omega | } } < 1 .
$$

By (39) and Theorem S.1, there exist an $| S | \times | S |$ permutation matrix $P _ { a }$ and diagonal matrices $\Lambda _ { B , a } , \Lambda _ { O , a } , \Lambda _ { F , a }$ such that

$$
\begin{array} { r l r } & { } & { \| B _ { a } ^ { \mathrm { n o r m } } ( \widetilde { \theta } ) - B _ { a } ^ { \mathrm { n o r m } } ( \theta ^ { * } ) P _ { a } \Lambda _ { B , a } \| _ { F } \leq \xi _ { n } , } \\ & { } & { \| \widetilde { O } - O ^ { * } P _ { a } \Lambda _ { O , a } \| _ { F } \leq \xi _ { n } , } \\ & { } & { \| \widetilde { F } _ { a } - F _ { a } ^ { * } P _ { a } \Lambda _ { F , a } \| _ { F } \leq \xi _ { n } , } \end{array}\tag{43}
$$

and

$$
\| \Lambda _ { B , a } \Lambda _ { O , a } \Lambda _ { F , a } - I _ { | S | } \| _ { F } \leq \xi _ { n } .\tag{44}
$$

We next remove the diagonal scaling ambiguity. Write $\lambda _ { O , s }$ and $\lambda _ { F , s }$ for the s-th diagonal entries of $\Lambda _ { O , a }$ and $\Lambda _ { F , a }$ . Since every column of $\widetilde O$ and $O ^ { * } P _ { a }$ sums to one, (43) implies

$$
\begin{array} { r } { | 1 - \lambda _ {  Ḋ \mathcal Ḋ O Ḍ Ḍ , s } | = \left| \mathbf 1 _ { | \Omega | } ^ { \top } \big ( \widetilde { O } ( : , s ) - \lambda _ {  Ḋ \mathcal Ḋ O Ḍ Ḍ , s } (  { \mathcal Ḋ O Ḍ } ^ { * } P _ { a } ) ( : , s ) \big ) \right| \le \sqrt { | \Omega | } \xi _ { n } . } \end{array}
$$

Similarly, because the columns of ${ \widetilde { F } } _ { a }$ and $F _ { a } ^ { * } P _ { a }$ are probability vectors,

$$
\left| 1 - \lambda _ { F , s } \right| \leq \sqrt { \Omega } \xi _ { n } .
$$

Therefore,

$$
\begin{array} { r } { \| \Lambda _ { { \cal O } , a } - I _ { | { \cal S } | } \| _ { F } \leq \sqrt { | { \cal S } | | \Omega | } \xi _ { n } , \qquad \| \Lambda _ { { \cal F } , a } - I _ { | { \cal S } | } \| _ { F } \leq \sqrt { | { \cal S } | | \Omega | } \xi _ { n } . } \end{array}\tag{45}
$$

Since $\sqrt { | \Omega | } \xi _ { n } \leq 1 / 4$ , we have

$$
\begin{array} { l } { \displaystyle { | \lambda _ { O , s } \lambda _ { F , s } | \geq ( 1 - \sqrt { | \Omega | } \xi _ { n } ) ^ { 2 } \geq \frac { 1 } { 2 } . } } \end{array}
$$

Writing $\lambda _ { B , s }$ for the corresponding diagonal entry of $\Lambda _ { B , a }$ , we obtain from (44)

$$
\begin{array} { r l } & { | \lambda _ { B , s } - 1 | \leq \frac { \left| \lambda _ { B , s } \lambda _ { O , s } \lambda _ { F , s } - 1 \right| + \left| 1 - \lambda _ { O , s } \lambda _ { F , s } \right| } { \left| \lambda _ { O , s } \lambda _ { F , s } \right| } } \\ & { \qquad \leq 2 \left( \xi _ { n } + 2 \sqrt { \left| \Omega \right| } \xi _ { n } + \left| \Omega \right| \xi _ { n } ^ { 2 } \right) } \\ & { \qquad \leq 8 \sqrt { \left| \Omega \right| } \xi _ { n } . } \end{array}
$$

Consequently,

$$
\| \Lambda _ { B , a } - I _ { | \mathcal { S } | } \| _ { F } \leq 8 \sqrt { | \mathcal { S } | | \Omega | } \xi _ { n } .\tag{46}
$$

Every column of $B _ { a } ^ { * } , O ^ { * }$ , and $F _ { a } ^ { * }$ has Euclidean norm at most one. Combining (43), (45), and (46) with the triangle inequality yields

$$
\begin{array} { r l } & { \| B _ { a } ^ { \mathrm { n o r m } } ( \widetilde { \theta } ) - B _ { a } ^ { \mathrm { n o r m } } ( \theta ^ { * } ) P _ { a } \| _ { F } \leq ( 1 + 8 \sqrt { | { \cal S } | | \Omega | } ) \xi _ { n } , } \\ & { \qquad \| \widetilde { O } - { \cal O } ^ { * } P _ { a } \| _ { F } \leq ( 1 + \sqrt { | { \cal S } | | \Omega | } ) \xi _ { n } , } \\ & { \qquad \| \widetilde { F } _ { a } - F _ { a } ^ { * } P _ { a } \| _ { F } \leq ( 1 + \sqrt { | { \cal S } | | \Omega | } ) \xi _ { n } . } \end{array}
$$

Define

$$
C _ { \mathrm { R K } } = ( 1 + 8 \sqrt { | { \cal S } | | \Omega | } ) D _ { \mathrm { R K } } .
$$

We have established that, for every $a \in { \mathcal { A } } ,$ there exists a permutation matrix $P _ { a }$ satisfying

$$
\begin{array} { r l r } & { } & { \left\| \frac { \bar { B } _ { a } ( \tilde { \theta } ) } { \sqrt { n } } - \frac { \bar { B } _ { a } ( \theta ^ { * } ) } { \sqrt { n } } P _ { a } \right\| _ { F } \leq C _ { \mathrm { { R K } } } \eta _ { n } , } \\ & { } & { \| O _ { \tilde { \varphi } } - O _ { \varphi ^ { * } } P _ { a } \| _ { F } \leq C _ { \mathrm { R K } } \eta _ { n } , } \\ & { } & { \| F _ { a } ( \tilde { \theta } ) - F _ { a } ( \theta ^ { * } ) P _ { a } \| _ { F } \leq C _ { \mathrm { { R K } } } \eta _ { n } . } \end{array}\tag{47}
$$

In particular, the factor recovery error is linear in the tensor perturbation $\eta _ { n } .$ , and $C _ { \mathrm { R K } }$ depends polynomially on the problem parameters but not on n.

Step 4: Establishing a common permutation. Although (47) initially produces an actiondependent permutation $P _ { a }$ , we now show that all these permutations coincide.

For any $a , b \in A$ , the triangle inequality gives

$$
\begin{array} { r l } & { \| O _ { \varphi ^ { * } } ( P _ { a } - P _ { b } ) \| _ { F } \leq \| O _ { \varphi ^ { * } } P _ { a } - O _ { \tilde { \varphi } } \| _ { F } + \| O _ { \tilde { \varphi } } - O _ { \varphi ^ { * } } P _ { b } \| _ { F } } \\ & { \qquad \leq 2 C _ { \mathrm { { R K } } } \eta _ { n } . } \end{array}
$$

On the other hand, if $P _ { a } \neq P _ { b }$ , then

$$
\begin{array} { r l } & { \| O _ { \varphi ^ { * } } ( P _ { a } - P _ { b } ) \| _ { F } \geq \sigma _ { \operatorname* { m i n } } ( O _ { \varphi ^ { * } } ) \| P _ { a } - P _ { b } \| _ { F } } \\ & { \qquad \geq 2 \sigma _ { O } . } \end{array}
$$

Therefore, whenever $C _ { \mathrm { R K } } \eta _ { n } < \sigma _ { O }$ , which follows from the lower-bound on n in the theorem, we must have $P _ { a } = P _ { b }$

It follows that there exists a single permutation matrix $P$ such that, simultaneously for all $a \in { \mathcal { A } } .$

$$
\begin{array} { r l r } & { } & { \Vert O _ { \tilde { \varphi } } - O _ { \varphi ^ { * } } P \Vert _ { F } \leq C _ { \mathrm { R K } } \eta _ { n } , } \\ & { } & { \Vert O _ { \tilde { \varphi } } T _ { \tilde { \beta } } ^ { a } - O _ { \varphi ^ { * } } T _ { \beta ^ { * } } ^ { a } P \Vert _ { F } \leq C _ { \mathrm { R K } } \eta _ { n } . } \end{array}\tag{48}
$$

Step 5: Recovering the transition kernels. Recall that

$$
F _ { a } ( \theta ) = O _ { \varphi } T _ { \beta } ^ { a } .
$$

For each $a \in { \mathcal { A } } .$ , we have

$$
\begin{array} { r l } & { O _ { \varphi ^ { * } } P \left( T _ { \widetilde { \beta } } ^ { a } - P ^ { \top } T _ { \beta ^ { * } } ^ { a } P \right) } \\ & { \quad = \left( O _ { \widetilde { \varphi } } T _ { \widetilde { \beta } } ^ { a } - O _ { \varphi ^ { * } } T _ { \beta ^ { * } } ^ { a } P \right) - \left( O _ { \widetilde { \varphi } } - O _ { \varphi ^ { * } } P \right) T _ { \widetilde { \beta } } ^ { a } . } \end{array}
$$

Since $\sigma _ { \operatorname* { m i n } } ( O _ { \varphi _ { * } } P ) \geq \sigma _ { O }$ , it follows that

$$
\begin{array} { r l } & { \left\| T _ { \tilde { \beta } } ^ { a } - P ^ { \top } T _ { \beta ^ { * } } ^ { a } P \right\| _ { F } } \\ & { \quad \leq \frac { 1 } { \sigma _ { O } } \left( \| O _ { \tilde { \varphi } } T _ { \tilde { \beta } } ^ { a } - O _ { \varphi ^ { * } } T _ { \beta ^ { * } } ^ { a } P \| _ { F } + \| O _ { \tilde { \varphi } } - O _ { \varphi ^ { * } } P \| _ { F } \| T _ { \tilde { \beta } } ^ { a } \| _ { 2 } \right) . } \end{array}
$$

Because $T _ { \tilde { \beta } } ^ { a }$ is column stochastic, $\| T _ { \widetilde { \beta } } ^ { a } \| _ { 2 } \le \sqrt { | { \cal S } | }$ . Combining this with (48) gives

$$
\left. P ^ { \top } T _ { \beta ^ { * } } ^ { a } P - T _ { \widetilde { \beta } } ^ { a } \right. _ { F } \leq \frac { ( 1 + \sqrt { | S | } ) C _ { \mathrm { R K } } } { \sigma _ { O } } \eta _ { n } .\tag{49}
$$

This bound holds simultaneously for all $a \in { \mathcal { A } }$ , with the same permutation $P .$

Step 6: Returning to the candidate parameter and concluding. Recall that Corollary G.1 guarantees

$$
\| { \tilde { \theta } } - { \hat { \theta } } \| _ { \infty } \leq { \frac { 1 } { n } } .
$$

In particular,

$$
\| O _ { \widetilde { \varphi } } - O _ { \widehat { \varphi } } \| _ { 1 , \mathrm { e n t } } \leq \frac { | \Omega | | S | } { n } , \qquad \| T _ { \widetilde { \beta } } ^ { a } - T _ { \widehat { \beta } } ^ { a } \| _ { 1 , \mathrm { e n t } } \leq \frac { | S | ^ { 2 } } { n } .
$$

For any $r \times q$ matrix A,

$$
\| A \| _ { 1 , \mathrm { e n t } } \leq { \sqrt { r q } } \| A \| _ { F } .
$$

Consequently, by (48),

$$
\Vert O _ { \varphi ^ { * } } P - O _ { \hat { \varphi } } \Vert _ { 1 , \mathrm { e n t } } \leq \sqrt { | \Omega | | S | } C _ { \mathrm { R K } } \eta _ { n } + \frac { | \Omega | | S | } { n } .\tag{50}
$$

Similarly, by (49), for every $a \in { \mathcal { A } }$

$$
\Vert P ^ { \top } T _ { \beta ^ { * } } ^ { a } P - T _ { \hat { \beta } } ^ { a } \Vert _ { 1 , \mathrm { e n t } } \leq \frac { | S | ( 1 + \sqrt { | S | } ) C _ { \mathrm { R K } } } { \sigma _ { O } } \eta _ { n } + \frac { | S | ^ { 2 } } { n } .\tag{51}
$$

Recall from the definition of $\eta _ { n }$ in Step 1,

$$
\eta _ { n } = O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n \pi _ { \mathrm { m i n } } } } \right) .
$$

Moreover, by Lemma 3.13,

$$
\kappa _ { B } ^ { - 1 } = O \left( \frac { 1 } { \pi _ { \mathrm { m i n } } \sigma _ { T } \sigma _ { O } d _ { \mathrm { m i n } } ^ { \pi } } \right) ,
$$

up to polynomial factors in $| { \cal A } |$ . Therefore, $C _ { \mathrm { R K } }$ is bounded polynomially in

$$
| A | , \quad \frac { 1 } { \pi _ { \mathrm { m i n } } } , \quad \frac { 1 } { d _ { \mathrm { m i n } } ^ { \pi } } , \quad \frac { 1 } { \sigma _ { T } } , \quad \frac { 1 } { \sigma _ { O } } , \quad | S | , \quad | \Omega | .
$$

Under the stated sample-size requirement, n is sufficiently large that $n \geq 2 t _ { \mathrm { b u r n } }$ (since $t _ { \mathrm { b u r n } }$ is polynomial in all the expressions in (4)) and

$$
D _ { \mathrm { R K } } \eta _ { n } \leq \frac { 1 } { 4 \sqrt { | \Omega | } } , \qquad C _ { \mathrm { R K } } \eta _ { n } < \sigma _ { O } .
$$

The $n ^ { - 1 }$ approximation terms in (50) and (51) can then be absorbed into the $n ^ { - 1 / 2 }$ bounds.

Hence there exist problem-dependent constants $C _ { O } , C _ { T } > 0$ , polynomially bounded in the quantities above, such that

$$
\lVert O _ { \varphi ^ { * } } P - O _ { \hat { \varphi } } \rVert _ { 1 , \mathrm { e n t } } \leq C _ { O } \sqrt { \frac { H _ { n } ( \delta ) } { n } } .\tag{52}
$$

Furthermore, simultaneously for every $a \in { \mathcal { A } } .$

$$
\| P ^ { \top } T _ { \beta ^ { * } } ^ { a } P - T _ { \hat { \beta } } ^ { a } \| _ { 1 , \mathrm { e n t } } \leq C _ { T } \sqrt { \frac { H _ { n } ( \delta ) } { n } } .\tag{53}
$$

The high-probability events used in Steps 1–2 hold simultaneously with probability at least $1 - \delta .$ Since Corollary G.1 holds uniformly over all candidate parameters, the preceding argument applies to every <sup>ˆ</sup>θ satisfying $\ell _ { n } ( \theta ^ { * } ) - \ell _ { n } ( { \hat { \theta } } ) \leq G _ { n }$

This proves the claimed recovery rates.

## L

In this appendix, we explain how the proofs for the theoretical results in Section 3.1 can be modified to work when we replace Assumption 3.9 with aggregate transition conditioning:

$$
\sigma _ { \operatorname* { m i n } } ( T _ { \beta ^ { * } } ) \geq \bar { \sigma } _ { T } > 0 , \qquad T _ { \beta ^ { * } } : = [ T _ { \beta ^ { * } } ^ { a _ { 1 } } , \dots , T _ { \beta ^ { * } } ^ { a _ { | { \cal A } | } } ] ,
$$

and

$$
\sigma _ { \operatorname* { m i n } } ( \widetilde { T } _ { \beta ^ { * } } ) \geq \widetilde { \sigma } _ { T } > 0 , \qquad \widetilde { T } _ { \beta ^ { * } } : = [ ( T _ { \beta ^ { * } } ^ { a _ { 1 } } ) ^ { \top } , \dots , ( T _ { \beta ^ { * } } ^ { a _ { | A | } } ) ^ { \top } ] .
$$

In particular, we identify the two intermediate results whose proofs need to be modified: Proposition I.1 and Theorem 3.14.

## L.1

First, we explain how the proof of Proposition I.1 can be modified.

The only part of the proof that needs adjustment is Lemma I.3.

From (29):

$$
\mathbb { E } [ \bar { b } _ { t + 1 } \bar { b } _ { t + 1 } ^ { \top } ] \succeq \pi _ { \operatorname* { m i n } } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } \sum _ { a \in \mathcal { A } } T _ { \beta ^ { * } } ^ { a } ( T _ { \beta ^ { * } } ^ { a } ) ^ { \top } .
$$

Then, $\sigma _ { \operatorname* { m i n } } ( T _ { \beta ^ { * } } ) \geq \bar { \sigma } _ { T } > 0$ implies that

$$
\sum _ { a \in \mathcal { A } } T _ { \beta ^ { * } } ^ { a } ( T _ { \beta ^ { * } } ^ { a } ) ^ { \top } \succeq ( \bar { \sigma } _ { T } ) ^ { 2 } .
$$

Combining the two inequalities gives us

$$
\mathbb { E } [ \bar { b } _ { t + 1 } \bar { b } _ { t + 1 } ^ { \top } ] \succeq \pi _ { \operatorname* { m i n } } \left( \bar { \sigma } _ { T } \right) ^ { 2 } \sigma _ { O } ^ { 2 } \left( \frac { 7 d _ { \operatorname* { m i n } } ^ { \pi } } { 8 } \right) ^ { 2 } I ,
$$

which concludes the proof of Lemma I.3.

## L.2

Then, we explain how the proof of Theorem 3.14 can be modified.

Recall that the action-indexed tensors admit the decomposition

$$
\mathcal { X } _ { a } ( \theta ^ { * } ) = \sum _ { s = 1 } ^ { S } \frac { \bar { B } _ { a } ( \theta ^ { * } ) ( : , s ) } { \sqrt { n } } \otimes O _ { \varphi ^ { * } } ( : , s ) \otimes ( O _ { \varphi ^ { * } } T _ { \beta ^ { * } } ^ { a } ) ( : , s ) .
$$

When individual transition matrices are singular, robust Kruskal uniqueness need not apply separately to each $\mathcal { X } _ { a } ( \theta ^ { * } )$ . Instead, we exploit the shared observation factor across actions.

Marginalizing each tensor over its third mode gives $\bar { B } _ { a } ( \theta ^ { * } ) O _ { \varphi ^ { * } } ^ { \top } / \sqrt { n }$ . By the conditioning of $O _ { \varphi ^ { * } }$ and $\bar { B } _ { a } ( \theta ^ { * } ) / \sqrt { n }$ from Assumption 3.8 and Lemma 3.13, agreement of these marginals implies that the candidate and true factors are related, up to perturbation, by a single invertible change of basis H, common to all actions.

Recall that we define $D _ { o } = \mathrm { d i a g } ( O _ { \varphi ^ { * } } ( o \mid \cdot ) )$ and defining $\widehat { D } _ { o }$ analogously, comparison of the two-step tensor slices yields

$$
T _ { \beta ^ { * } } ^ { a } D _ { o } H \approx H T _ { \hat { \beta } } ^ { a } \widehat { D } _ { o } .
$$

Summing over o gives $T _ { \beta ^ { * } } ^ { a } H \approx H T _ { \hat { \beta } } ^ { a }$ , and hence

$$
T _ { \beta ^ { * } } ^ { a } ( D _ { o } H - H \widehat { D } _ { o } ) \approx 0 \qquad \mathrm { f o r ~ e v e r y ~ } a , o .
$$

The aggregate condition now implies

$$
\| D _ { o } H - H \widehat { D } _ { o } \| _ { F } \leq \frac { 1 } { \widetilde { \sigma } _ { T } } \left( \sum _ { a } \| T _ { \beta ^ { * } } ^ { a } ( D _ { o } H - H \widehat { D } _ { o } ) \| _ { F } ^ { 2 } \right) ^ { 1 / 2 } .
$$

Thus the approximate commutation relation holds without requiring any individual $T _ { \star } ^ { a }$ to be invertible. Since $O _ { \varphi }$ ∗ has full column rank, its columns are distinct. The approximate commutation relations, together with the nonsingularity of H and the column-stochastic normalization of the observation kernels, imply that H is close to a permutation matrix P. Consequently,

$$
\begin{array} { r } { O _ { \hat { \varphi } } \approx O _ { \varphi ^ { * } } P , \qquad T _ { \hat { \beta } } ^ { a } \approx P ^ { \top } T _ { \beta ^ { * } } ^ { a } P \quad \mathrm { f o r ~ a l l ~ } a . } \end{array}
$$

The quantitative perturbation argument follows the same steps as the original proof, with the aggregate conditioning parameter $\widetilde { \sigma } _ { T }$ replacing the individual transition-conditioning parameter in the joint alignment step.

## M

Since we assume that rewards are observed and R is finite, we can incorporate rewards into the observation space by defining the augmented observation kernels

$$
\overline { { { O } } } _ { \theta } ^ { a } ( ( o , r ) \mid s ) = O _ { \theta } ( o \mid s ) R _ { \theta } ^ { a } ( r \mid s ) .
$$

Here we use the conditional independence of the reward and observation given the state and action. Although ${ \overline { { O } } } _ { \theta } ^ { a }$ depends on $^ { a , }$ marginalizing over rewards recovers the original observation kernel:

$$
\sum _ { r \in \mathcal { R } } \overline { { { O } } } _ { \theta } ^ { a } ( ( o , r ) \mid s ) = O _ { \theta } ( o \mid s ) .
$$

In particular, quantitative full-column-rankness is preserved, since

$$
\sigma _ { \mathrm { m i n } } ( \overline { { O } } _ { \theta } ^ { a } ) \geq \frac { \sigma _ { \mathrm { m i n } } ( O _ { \theta } ) } { \sqrt { K _ { R } } } ,
$$

where $K _ { R }$ denotes the size of the reward alphabet.

The same argument of Theorem 3.14 therefore extends to these action-dependent augmented observation kernels. Any action-dependent permutations arising from the recovery can be aligned using the common observation kernel. Consequently, the augmented kernels are recovered under a common permutation $P$ at the rate of Theorem 3.14. Marginalizing over the original observations then recovers the reward kernels, and marginalization cannot increase the entrywise $\ell _ { 1 }$ error. Thi yields the claimed bound.

Remark M.1. Notice that our construction of $\overline { { O } } _ { \theta } ^ { a }$ implies that it is an action-dependent observation kernel, whereas Theorem 3.14 assumes the observation kernel is action-independent. However, the exact same argument of Theorem 3.14 applies if we use action-dependent observation kernels.

Remark M.2. The finite-alphabet restriction can be relaxed when only conditional mean rewards are of interest. For bounded continuous rewards, an auxiliary Bernoulli observation with success probability proportional to the realized reward reduces mean recovery to the finite-alphabet case. For conditionally sub-Gaussian rewards, one can instead estimate reward–observation moments using importance weighting and standard concentration arguments, then recover the conditional means using the full-column-rank observation kernel. We omit these extensions for brevity.

## N

## Example: Physics-constrained thermal control of a smart building.

POMDPs with state-based rewards have been used to model and optimize HVAC systems for energy cost and occupant comfort. This example is adapted from Murugesan et al. [2020].

Consider thermal control in a smart building. Let the continuous physical state be

$$
X _ { t } = \left( T _ { \mathrm { a i r } , t } , T _ { \mathrm { m a s s } , t } \right) ^ { \top } ,
$$

where $T _ { \mathrm { a i r } , t }$ is the indoor-air temperature and $T _ { \mathrm { m a s s } , t }$ , representing the effective temperature of walls and furniture, is not directly observed. A reduced-order stochastic thermal model may take the form

$$
d X _ { t } = A ( \beta ) X _ { t } d t + B ( \beta ) u _ { t } d t + g _ { \beta } ( T _ { \mathrm { o u t } , t } ) d t + G ( \beta ) d W _ { t } ,
$$

where $\beta$ contains physical parameters such as thermal resistances and capacitances, $u _ { t }$ is the HVAC control input, and $\dot { T } _ { \mathrm { o u t } , t }$ is the outdoor temperature. The matrices $A ( \beta )$ and $B ( \beta )$ describe the intrinsic thermal dynamics and the effect of HVAC control, respectively, $g _ { \beta } ( T _ { \mathrm { o u t } , t } )$ represents heat transfer due to the outdoor temperature, and $G ( \beta ) d W _ { t }$ models unobserved thermal disturbances.

Sensors provide noisy measurements

$$
Y _ { k } = C X _ { k \Delta } + \varepsilon _ { k } ,
$$

where $C$ specifies which components of the physical state are measured, $\varepsilon _ { k }$ is sensor noise, and $\Delta > 0$ is the sampling interval.

A POMDP approximation can use semantic thermal regimes such as cold, neutral, and hot, HVAC actions such as off/medium/high heating, and observation bins based on measured temperature. For state bins $B _ { i }$ and observation bins $C _ { o }$ , the induced approximate POMDP kernels are

$$
\begin{array} { r l r l } & { T _ { \beta } ^ { a } ( i , j ) = { \mathbb P } _ { \beta } \left( X _ { ( k + 1 ) \Delta } \in B _ { j } \mid X _ { k \Delta } \in B _ { i } , u _ { k } = a \right) , } & & { O _ { \varphi } ( o \mid i ) = { \mathbb P } _ { \varphi } \left( Y _ { k } \in C _ { o } \mid X _ { k \Delta } \in B _ { i } \right) . } \end{array}
$$

Thus, $T _ { \beta } ^ { a }$ is not an arbitrary matrix: all actions share the same low-dimensional physical parameter $\beta ,$ and the semantic states naturally support state-based comfort or safety rewards.

Our result does not require an unconstrained black-box MLE. One may optimize likelihood over the physics-constrained class

$$
\Theta _ { \mathrm { p h y s } } = \left\{ \left( \{ T _ { \beta } ^ { a } \} _ { a \in \mathcal { A } } , O _ { \varphi } \right) : \beta \in \mathcal { B } , \varphi \in \Phi \right\} ,
$$

where B encodes relevant domain knowledge and enforces model structures:

$$
( \hat { \beta } , \hat { \varphi } ) \in \arg \operatorname* { m a x } _ { ( \beta , \varphi ) \in \mathcal { B } \times \Phi } \ell _ { n } \left( \{ T _ { \beta } ^ { a } \} _ { a \in \mathcal { A } } , O _ { \varphi } \right) .
$$

Our theorem then applies to any exact or approximate optimizer whose candidate achieves the required likelihood gap. Hence, the PDE/SDE structure and semantic prior knowledge are retained rather than discarded; the result converts the likelihood accuracy of a structured estimator into recovery guarantees for the induced POMDP kernels.

## O

In this appendix, we show why Assumption 3.24 is necessary for achieving approximate POMDP kernel recovery up to latent-state relabeling. We emphasize that our claim with the following proposition is not that violating the assumption means every such POMDP is non-identifiable, but rather that it prevents a uniform permutation-recovery guarantee over general POMDPs.

Proposition O.1. Consider a finite POMDP with parameter

$$
\theta = { \bigl ( } O , \{ T ^ { a } \} _ { a \in { \mathcal { A } } } , b _ { 1 } { \bigr ) } ,
$$

where $b _ { 1 }$ is the initial belief, and suppose that there exist two distinct latent states $i , j \in S$ such that

$$
O ( \cdot \mid i ) = O ( \cdot \mid j ) .
$$

Assume further that the transition kernels are strictly positive and that, for some length- $( L - 1 )$ action string α, the corresponding L-step observation matrix $M _ { \alpha }$ hasfull column rank.

Then there exists another valid POMDP parameter

$$
{ \widetilde \theta } = ( O , \{ { \widetilde T } ^ { a } \} _ { a \in A } , b _ { 1 } )
$$

such that:

1. <sup>˜</sup>θ is not related to θ by any permutation of the latent states;

2. for every horizon H, every action sequence $a _ { 1 : H - 1 : }$ , and every observation sequence $O 1 ; H ,$

$$
\mathbb { P } _ { \theta } \left( o _ { 1 : H } \mid a _ { 1 : H - 1 } \right) = \mathbb { P } _ { \tilde { \theta } } \left( o _ { 1 : H } \mid a _ { 1 : H - 1 } \right) ;
$$

3. the transformed model remains L-step observable, and infact its L-step observation matrix satisfies

$$
{ \widetilde { M } } _ { \alpha } = M _ { \alpha } G
$$

for some invertible matrix G that is not a permutation matrix.

Consequently, ifa model class permits two identical columns ofO, then no estimator can be uniformly consistent for recovery of the POMDP parameters up to latent-state permutation. More precisely, there exists a constant $\Delta > 0$ such that,for every sample size n and every estimator $\hat { \theta } _ { n }$

$$
\operatorname* { m a x } \Big \{ \mathbb { E } _ { \theta } \big [ d _ { \mathrm { p e r m } } ( \widehat { \theta } _ { n } , \theta ) \big ] , \mathbb { E } _ { \widetilde { \theta } } \big [ d _ { \mathrm { p e r m } } ( \widehat { \theta } _ { n } , \widetilde { \theta } ) \big ] \Big \} \geq \frac { \Delta } { 2 } ,
$$

where $d _ { \mathrm { p e r m } }$ denotes any permutation-invariant parameter metric. Hence the estimation error cannot converge to zero, even with infinitely many trajectories.

In this sense, Assumption 3.24 is information-theoretically necessary for a general permutation-level kernel-recovery guarantee

Remark O.2. The entrywise strict positivity assumption on the transition kernels can be relaxed. In particular, it suffices to assume that rows i and j of each $T ^ { a }$ have identical support, i.e.,

$$
T ^ { a } ( i \mid s ) = 0 \quad \Longleftrightarrow \quad T ^ { a } ( j \mid s ) = 0 , \qquad \forall a \in { \mathcal A } , s \in { \mathcal S } .
$$

Under this weaker condition, the perturbation constructed in the proof still preserves the nonnegativity of the transition kernels, provided that the perturbation is sufficiently small.

Proof. Since

$$
O ( \cdot \mid i ) = O ( \cdot \mid j ) ,
$$

the two latent states i and $j$ induce exactly the same one-step observation distribution. For each observation o, define the diagonal observation operator

$$
D _ { o } : = \mathrm { d i a g } \big ( O ( o \mid 1 ) , \ldots , O ( o \mid | S | ) \big ) .
$$

The restriction of $D _ { o }$ to coordinates i and $j$ is therefore a multiple of the identity:

$$
( D _ { o } ) _ { i i } = ( D _ { o } ) _ { j j } \qquad \mathrm { f o r e v e r y } o .
$$

We construct an invertible, non-permutation matrix G that only mixes coordinates i and j. Let

$$
p = ( b _ { 1 } ) _ { i } , \qquad q = ( b _ { 1 } ) _ { j } ,
$$

and, on the (i, j) coordinates, define

$$
G _ { \varepsilon } ^ { ( i , j ) } : = \left( \begin{array} { c c } { 1 - \varepsilon q } & { \varepsilon p } \\ { \varepsilon q } & { 1 - \varepsilon p } \end{array} \right) ,
$$

while $G _ { \varepsilon }$ acts as the identity on all remaining coordinates. For sufficiently small $\varepsilon > 0 , G _ { \varepsilon }$ is invertible and is not a permutation matrix. Moreover,

$$
\mathbf { 1 } ^ { \top } G _ { \varepsilon } = \mathbf { 1 } ^ { \top } , \qquad G _ { \varepsilon } b _ { 1 } = b _ { 1 } .
$$

Because each $D _ { o }$ is scalar on the (i, j) block,

$$
\begin{array} { r } { D _ { o } G _ { \varepsilon } = G _ { \varepsilon } D _ { o } \qquad \mathrm { f o r e v e r y } o . } \end{array}
$$

For brevity, write $G : = G _ { \varepsilon }$

Now define

$$
{ \widetilde { \cal O } } : = { \cal O } , \qquad { \widetilde { \cal T } } ^ { a } : = { \cal G } ^ { - 1 } { \cal T } ^ { a } { \cal G } , \qquad a \in { \cal A } .
$$

Since $\mathbf { 1 } ^ { \top } G = \mathbf { 1 } ^ { \top }$ , we also have

$$
\mathbf { 1 } ^ { \top } G ^ { - 1 } = \mathbf { 1 } ^ { \top } .
$$

Hence

$$
\mathbf { 1 } ^ { \top } \widetilde { T } ^ { a } = \mathbf { 1 } ^ { \top } G ^ { - 1 } T ^ { a } G = \mathbf { 1 } ^ { \top } T ^ { a } G = \mathbf { 1 } ^ { \top } G = \mathbf { 1 } ^ { \top } G = \mathbf { 1 } ^ { \top } .
$$

Thus every ${ \widetilde { T } } ^ { a }$ has column sums equal to one. Since the original transition kernels are strictly positive and $G _ { \varepsilon } \to I { \mathrm { ~ a s ~ } } \varepsilon \to 0$ , continuity implies that, for all sufficiently small $\varepsilon > 0$ , every ${ \widetilde { T } } ^ { a }$ remains entrywise positive. By the same reasoning, for all sufficiently small $\varepsilon > 0 , \widetilde { T } ^ { a }$ has its minimum singular value bounded away from 0. Therefore $\tilde { \theta }$ is again a valid POMDP.

We next show that the two POMDPs induce exactly the same trajectory likelihood. Under our convention, for any observation sequence $O _ { 1 : H }$ and action sequence $a _ { 1 : H - 1 }$

$$
\mathbb { P } _ { \theta } \big ( o _ { 1 : H } \bigm | a _ { 1 : H - 1 } \big ) = \mathbf { 1 } ^ { \top } D _ { o _ { H } } T ^ { a _ { H - 1 } } D _ { o _ { H - 1 } } \cdot \cdot \cdot T ^ { a _ { 1 } } D _ { o _ { 1 } } b _ { 1 } .
$$

For the transformed model,

$$
\begin{array} { r l } & { \mathbb { P } _ { \tilde { \theta } } \big ( o _ { 1 : H } \mid a _ { 1 : H - 1 } \big ) } \\ & { \quad = \mathbf { 1 } ^ { \top } D _ { o _ { H } } G ^ { - 1 } T ^ { a _ { H - 1 } } G D _ { o _ { H - 1 } } \cdot \cdot \cdot G ^ { - 1 } T ^ { a _ { 1 } } G D _ { o _ { 1 } } b _ { 1 } . } \end{array}
$$

Using

$$
D _ { o } G = G D _ { o } , \qquad { \bf 1 } ^ { \top } G ^ { - 1 } = { \bf 1 } ^ { \top } , \qquad G b _ { 1 } = b _ { 1 } ,
$$

all intermediate copies of G and $G ^ { - 1 }$ cancel, yielding

$$
\mathbb { P } _ { \tilde { \theta } } ( o _ { 1 : H } \mid a _ { 1 : H - 1 } ) = \mathbb { P } _ { \theta } ( o _ { 1 : H } \mid a _ { 1 : H - 1 } ) .
$$

Thus the two models generate exactly the same conditioned trajectory likelihood for every horizon, every action sequence, and every observation sequence.

Importantly, this ambiguity persists even at the level of the full-rank L-step observation matrix. Fix an action string

$$
\alpha = ( a _ { 1 } , \ldots , a _ { L - 1 } ) ,
$$

and let $M _ { \alpha }$ denote the matrix whose row corresponding to $\zeta = \left( o _ { 1 } , \ldots , o _ { L } \right)$ is

$$
m _ { \alpha , \zeta } ^ { \top } = { \bf 1 } ^ { \top } D _ { o _ { L } } T ^ { a _ { L } - 1 } D _ { o _ { L - 1 } } \cdot \cdot \cdot T ^ { a _ { 1 } } D _ { o _ { 1 } } .
$$

For the transformed model,

$$
\begin{array} { r l } & { \widetilde { m } _ { \alpha , \zeta } ^ { \top } = \mathbf { 1 } ^ { \top } D _ { o _ { L } } G ^ { - 1 } T ^ { a _ { L - 1 } } G D _ { o _ { L - 1 } } \cdot \cdot \cdot G ^ { - 1 } T ^ { a _ { 1 } } G D _ { o _ { 1 } } } \\ & { \qquad = m _ { \alpha , \zeta } ^ { \top } G , } \end{array}
$$

again by commuting $G$ through the observation operators and cancelling the intermediate factors. Consequently,

$$
{ \widetilde { M } } _ { \alpha } = M _ { \alpha } G .
$$

Because G is invertible,

$$
\mathrm { r a n k } ( \widetilde { M } _ { \alpha } ) = \mathrm { r a n k } ( M _ { \alpha } ) = | S | ,
$$

so the transformed model remains L-step observable. In particular, full column rank of the L-step observation matrix does not eliminate this ambiguity.

We now verify that θ and $\tilde { \theta }$ are not equivalent up to a permutation of the latent states. Suppose, toward a contradiction, that there were a permutation matrix Π relating the two models. Then the corresponding L-step observation matrices would satisfy

$$
{ \widetilde { M } } _ { \alpha } = M _ { \alpha } \Pi .
$$

On the other hand, we have already shown that

$$
{ \widetilde { M } } _ { \alpha } = M _ { \alpha } G .
$$

Hence

$$
M _ { \alpha } ( G - \Pi ) = 0 .
$$

Since $M _ { \alpha }$ has full column rank,

$$
G = \Pi ,
$$

contradicting the construction of $G$ as a non-permutation matrix. Therefore θ and $\tilde { \theta }$ are genuinely distinct even after quotienting out latent-state permutations.

Let

$$
\Delta : = d _ { \mathrm { p e r m } } ( \theta , \tilde { \theta } ) .
$$

Since the two models are not permutation-equivalent,

$$
\Delta > 0 .
$$

Nevertheless, their trajectory distributions are exactly identical. Thus, for every sample size $n ,$ the law of any dataset consisting of n trajectories is the same under θ and <sup>˜</sup>θ. Consequently, any estimator $\hat { \theta } _ { n }$ has the same distribution under the two models.

By the triangle inequality for $d _ { \mathrm { p e r m } }$

$$
\Delta \leq d _ { \mathrm { p e r m } } ( \theta , \hat { \theta } _ { n } ) + d _ { \mathrm { p e r m } } ( \hat { \theta } _ { n } , \tilde { \theta } ) .
$$

Taking expectation with respect to the common data distribution gives

$$
\Delta \leq \mathbb { E } _ { \boldsymbol { \theta } } \left[ d _ { \mathrm { p e r m } } ( \hat { \boldsymbol { \theta } } _ { n } , \boldsymbol { \theta } ) \right] + \mathbb { E } _ { \boldsymbol { \tilde { \theta } } } \left[ d _ { \mathrm { p e r m } } ( \hat { \boldsymbol { \theta } } _ { n } , \boldsymbol { \tilde { \theta } } ) \right] .
$$

Therefore,

$$
\operatorname* { m a x } \left\{ \mathbb { E } _ { \theta } \left[ d _ { \mathrm { p e r m } } ( \hat { \theta } _ { n } , \theta ) \right] , \mathbb { E } _ { \tilde { \theta } } \left[ d _ { \mathrm { p e r m } } ( \hat { \theta } _ { n } , \tilde { \theta } ) \right] \right\} \geq \frac { \Delta } { 2 }
$$

for every n.

Hence the estimation error cannot converge to zero uniformly over a general POMDP class that permits identical columns of the one-step observation matrix. The obstruction is therefore genuinely information-theoretic: even infinitely many trajectories do not contain enough information to determine the latent-state realization up to permutation. □

## P

We prove the proposition by constructing a family of POMDPs satisfying Assumptions 3.9, 3.17, 3.18, 3.22, 3.24, for which kernel estimation requires exponentially many observations in $| S |$

Let $| S | = 2 r$ and $\mathcal { S } = \{ A _ { 1 } , \ldots , A _ { r } , B _ { 1 } , \ldots , B _ { r } \}$ . Take $\Omega = \{ o _ { 0 } , o _ { 1 } , b _ { 1 } , \ldots , b _ { r } \}$ and define

$$
p _ { j } = { \frac { 1 } { 4 } } + { \frac { j } { 4 ( r + 1 ) } } , \qquad O _ { \varphi ^ { * } } ( o _ { 1 } \mid A _ { j } ) = p _ { j } , \qquad O _ { \varphi ^ { * } } ( o _ { 0 } \mid A _ { j } ) = 1 - p _ { j } ,
$$

$$
O _ { \varphi ^ { * } } ( b _ { i } \mid B _ { j } ) = { \bf 1 } \{ i = j \} ,
$$

with all other emission probabilities zero.

Let

$$
Q = \left( \begin{array} { c c } { { 0 } } & { { r ^ { - 1 } { \bf 1 1 } ^ { \top } } } \\ { { I _ { r } } } & { { 0 } } \end{array} \right) , \qquad \Pi = \frac { 1 } { | S | } { \bf 1 1 } ^ { \top } .
$$

Let $v _ { A } \in \mathbb { R } ^ { r }$ be the unit vector with entries

$$
( v _ { A } ) _ { j } = { \frac { ( - 1 ) ^ { j - 1 } { \binom { r - 1 } { j - 1 } } } { \sqrt { { \binom { 2 r - 2 } { r - 1 } } } } } , \qquad j = 1 , \ldots , r ,
$$

and set

$$
\boldsymbol { v } = ( \boldsymbol { v } _ { A } ^ { \top } , 0 _ { r } ^ { \top } ) ^ { \top } , \qquad \boldsymbol { u } = \frac { \boldsymbol { e } _ { B _ { 1 } } - \boldsymbol { e } _ { B _ { 2 } } } { \sqrt { 2 } } , \qquad \boldsymbol { h } = \frac { 1 } { 1 2 8 r } , \qquad \zeta = 2 ^ { - r } .
$$

For $\nu \in \{ - 1 , + 1 \}$ , define $\theta _ { \nu } = ( \beta _ { \nu } , \varphi ^ { * } )$ by assigning, for every $a \in { \mathcal { A } } .$

$$
T _ { \beta _ { \nu } } ^ { a } = \left( 1 - \zeta \right) \left( \frac { 3 } { 4 } I _ { | \mathcal { S } | } + \frac { 1 } { 8 } Q + \frac { 1 } { 8 } \Pi + \nu h u v ^ { \top } \right) + \zeta Q ^ { \top } .
$$

We verify the five assumptions.

Assumption 3.9. Direct calculation gives, uniformly over $a \in { \mathcal { A } }$ and $\nu \in \{ - 1 , + 1 \}$

$$
\sigma _ { \operatorname* { m i n } } ( T _ { \beta _ { \nu } } ^ { a } ) \geq \frac { 1 } { 4 } , \qquad \operatorname* { m i n } _ { i , j } T _ { \beta _ { \nu } } ^ { a } ( i , j ) \geq \frac { 1 } { 3 2 S } .
$$

Thus Assumption 3.9 holds with $\sigma _ { T } = 1 / 4$

Assumption 3.17. Take $m = 2$ . The rows corresponding to observation pairs $( o _ { 0 } , b _ { i } )$ and $( b _ { i } , b _ { i } )$ form a block-diagonal submatrix of $\mathcal { M } _ { \theta _ { \nu } } ^ { ( 2 ) } [ \mathinner { w } ]$ , with each diagonal block uniformly well-conditioned. In particular,

$$
\sigma _ { \operatorname* { m i n } } \left( \mathcal { M } _ { \theta _ { \nu } } ^ { ( 2 ) } [ w ] \right) \geq \frac { 1 } { 3 2 } , \qquad \forall w \in \mathcal { A } .
$$

Hence Assumption 3.17 holds with $\sigma _ { M } = 1 / 3 2$ and $\mathcal { W } = \mathcal { A }$

Assumption 3.18. Let π be uniform over $\mathcal { A } \oplus \mathcal { A } \oplus \mathcal { W } \oplus \mathcal { A } = \mathcal { A } ^ { 4 }$ . Then, for every $a _ { 1 } , a _ { 2 } , w _ { \mathrm { ~ ~ } }$

$$
\sum _ { a ^ { \prime } \in \mathcal { A } } \pi ( a _ { 1 } \oplus a _ { 2 } \oplus w \oplus a ^ { \prime } ) = \frac { 1 } { | \mathcal { A } | ^ { 3 } } .
$$

Therefore Assumption 3.18 holds with $\pi _ { \mathrm { m i n } } ^ { \mathrm { b l k } } = | \mathcal { A } | ^ { - 3 }$

Assumption 3.22. Since $T _ { \beta _ { \nu } } ^ { a }$ is doubly stochastic for every action, the induced block-level transition kernel is $( T _ { \beta _ { \nu } } ^ { a } ) ^ { 4 }$ , with stationary distribution

$$
d _ { \mathrm { b l k } } ^ { \pi } = \frac { 1 } { | S | } \mathbf { 1 } , \qquad ( d _ { \mathrm { b l k } } ^ { \pi } ) _ { \mathrm { m i n } } = \frac { 1 } { | S | } .
$$

Moreover, the uniform lower bound $T _ { \beta _ { \nu } } ^ { a } ( i , j ) \geq 1 / ( 3 2 | S | )$ implies a dimension-independent Doeblin contraction. Thus the block-level chain is uniformly geometrically ergodic, verifying Assumption 3.22.

Assumption 3.24. The observation columns are distinct, with

$$
\Delta _ { \cal O } = \operatorname * { m i n } _ { s \neq s ^ { \prime } } \| { \cal O } _ { \varphi ^ { * } } ( \cdot \vert s ) - { \cal O } _ { \varphi ^ { * } } ( \cdot \vert s ^ { \prime } ) \| _ { 2 } = \frac { \sqrt { 2 } } { 4 ( r + 1 ) } = \Omega ( \vert S \vert ^ { - 1 } ) > 0 .
$$

Hence Assumption 3.24 holds.

Finally, the vector $v _ { A }$ satisfies

$$
\sum _ { j = 1 } ^ { r } ( v _ { A } ) _ { j } p _ { j } ^ { k } = 0 , \qquad k = 0 , \ldots , r - 2 .
$$

Consequently, under the kernels obtained by setting $\zeta = 0 ,$ , the two observable-trajectory laws agree until a run of $\dot { r } - 1$ consecutive observations from $\big \{ o _ { 0 } , o _ { 1 } \big \}$ occurs. Such a run has probability at most $\scriptstyle { \frac { n } { 2 } } ( 1 3 / 1 6 ) ^ { r - 2 }$ within n data points. Re-setting $\zeta = 2 ^ { - r }$ changes each observable-trajectory law by at most $n \zeta$ in total variation. Therefore,

$$
\mathrm { T V } \left( \mathbb { P } _ { \theta _ { + } } ^ { n } , \mathbb { P } _ { \theta _ { - } } ^ { n } \right) \leq \frac { n } { 2 } \left( \frac { 1 3 } { 1 6 } \right) ^ { r - 2 } + 2 n 2 ^ { - r } \leq n \left( \frac { 1 3 } { 1 6 } \right) ^ { r - 2 } .
$$

However, for every $a \in { \mathcal { A } } .$

$$
\left. T _ { \beta _ { + } } ^ { a } - T _ { \beta _ { - } } ^ { a } \right. _ { \mathrm { F } } = 2 ( 1 - \zeta ) h = \Theta ( | S | ^ { - 1 } ) .
$$

The distinct observation columns also prevent this separation from being eliminated by a state permutation.

Some calculation gives

$$
d _ { \mathrm { k e r } } ( \theta _ { + } , \theta _ { - } ) = 2 ( 1 - \zeta ) h \geq \frac { 1 } { 6 4 | S | } ,
$$

Le Cam’s two-point method (see, e.g., [Duchi, 2018, Section 5.2, Lemma 5.2.2]) yields

$$
\operatorname* { i n f } _ { \widehat { \theta } } \operatorname* { m a x } _ { \nu \in \{ - 1 , + 1 \} } \operatorname* { P r } _ { \nu } \left( d _ { \mathrm { k e r } } ( \widehat { \theta } , \theta _ { \nu } ) \geq \frac { 1 } { 2 5 6 | S | } \right) \geq \frac { 3 } { 8 }
$$

whenever $\begin{array} { r } { n \leq \frac { 1 } { 4 } ( 1 6 / 1 3 ) ^ { | S | / 2 - 2 } } \end{array}$ . Therefore, kernel estimation to accuracy $c / | S |$ for a sufficiently small universal constant $c > 0$ requires

$$
n = \Omega \left( \left( \frac { 1 6 } { 1 3 } \right) ^ { | S | / 2 } \right) = \exp ( \Omega ( | S | ) ) .
$$

Thus all five assumptions hold, yet polynomial sample complexity is impossible uniformly over this family.

## Q

Throughout the proof, the polynomial dependence asserted in Proposition 3.27 is understood with all relevant model parameters used in the assumptions mentioned in the Proposition bounded polynomially in $| S |$ . This includes everything in (5). The polynomial exponents are independent of |S|.

Write $H = m + 2 , d = d _ { \mathrm { b l k } } ^ { \pi }$ , and let

$$
P _ { \pi } : = \sum _ { \mathbf { a } } \pi ( \pmb { a } ) T _ { \beta ^ { * } } ^ { a _ { H } } \cdot \cdot \cdot T _ { \beta ^ { * } } ^ { a _ { 1 } }
$$

be the transition matrix of the latent chain at block boundaries. Let $X _ { j } = s _ { 1 + j H }$ , let ${ \mathcal { F } } _ { j }$ contain the observed records of the first j completed blocks, and recall that we define

$$
\bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } : = \mathbb { P } _ { \theta ^ { * } } ( X _ { j } = \cdot \mid \mathcal { F } _ { j } ) , \qquad \bar { B } ^ { \mathrm { b l k } } : = \left[ \overset { ( \bar { b } _ { 1 , \theta ^ { * } } ^ { \mathrm { b l k } } ) ^ { \top } } { \vdots } \right] .
$$

Extension to a general L. For simplicity, we present the argument when Assumption 3.26 holds with $L = H$

For $L \in H \mathbb { N }$ , regard $L / H$ consecutive original blocks as a single length-L analysis window, whose action-string distribution is $\pi _ { L } = \pi ^ { \otimes ( L / H ) }$ . Its latent transition matrix is $P _ { \pi } ^ { L \dot { / } H }$ , so its stationary boundary distribution remains d. In the stationary argument below, replace the current block’s record by the last L steps of observation and action history and consider $j \ge \bar { L } / H$ . The same Bayes identity and conditional Jensen argument then give $\mathbb { E } [ \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b i k } } ( \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } ) ^ { \top } ] \succeq \bar { \eta ^ { 2 } I }$ . For concentration, retain this length-L window after $r _ { \eta }$ intervening original blocks and set $h = r _ { \eta } + L / H$ . Conditional on ${ \mathcal { F } } _ { j - h } .$ the starting distribution of the retained window is $P _ { \pi } ^ { r _ { \eta } } \bar { b } _ { j - h , \theta ^ { * } } ^ { \mathrm { b l k } }$ . The same perturbation bound and conditional Jensen argument therefore yield

$$
\mathbb { E } [ \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } ( \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } ) ^ { \top } \mid \mathcal { F } _ { j - h } ] \succeq \frac { \eta ^ { 2 } } { 4 } I , \qquad j \geq h .
$$

The subsequent concentration argument applies with this value of $h ,$ omitting the first $h - 1$ positivesemidefinite summands. Since $L$ is polynomially bounded in |S|, this modification preserves polynomial sample complexity. The grouping is only for the analysis: the sampling policy and the rows of $\bar { B } ^ { \mathrm { b l k } }$ remain unchanged.

For L not divisible by $H _ { \ast }$ interpret $\Phi _ { L } ^ { \pi } ( x )$ as $\mathbb { E } _ { d } | x ^ { \top } c _ { L } |$ |, where $c _ { L }$ is the belief about a fixed terminal block-boundary state given the preceding L primitive-step records under the stationary block process. Set $L _ { + } = H \bar { \lceil { L } \ / { H } \rceil }$ and extend this window backward. The corresponding history sigma-fields satisfy $\mathcal { G } _ { L } \subseteq \mathcal { G } _ { L _ { + } }$ , so the tower property gives $c _ { L } = \mathbb { E } _ { d } [ c _ { L _ { + } } \mid { \mathcal { G } } _ { L } ]$ . Conditional Jensen’s inequality therefore yields

$$
\Phi _ { L _ { + } } ^ { \pi } ( x ) = \mathbb { E } _ { d } \vert x ^ { \top } c _ { L _ { + } } \vert \geq \mathbb { E } _ { d } \vert x ^ { \top } c _ { L } \vert = \Phi _ { L } ^ { \pi } ( x ) \geq \eta \Vert x \Vert _ { 2 } .
$$

Thus we may assume $L$ is divisible by H without weakening the excitation constant or changing the behavior policy. Since $L \leq L _ { + } < \mathbf { \bar {  { L } } } + H$ , this rounding preserves polynomial horizon bounds whenever L and H are polynomially bounded in |S|.

In the rest of this appendix, we assume $L = H$

We first show the first half (sufficiency) of Proposition 3.27: Suppose Assumptions 3.9, 3.17, 3.18, 3.22, 3.24, and 3.26 hold. Then the kernels can be estimated, up to a common permutation of the latent states, to accuracy ε with probability at least $1 - \delta$ using a trajectory of length pol $\mathrm { y } ( | S | , \varepsilon ^ { - 1 } , \log ( 1 / \delta ) )$ .

Sufficiency. Under stationary initialization, the belief c at the end of a block (also the start of the next block), conditional only on that block’s action and observation words, satisfies

$$
c = \frac { K _ { a , y } d } { \mathbf { 1 } ^ { \top } K _ { a , y } d }
$$

on every event of positive probability. Consequently,

$$
\mathbb { E } | x ^ { \top } c | = \Phi _ { H } ^ { \pi } ( x ) .
$$

Since the full-history belief uses at least this information, conditional Jensen’s inequality and Assumption 3.26 give

$$
\mathbb { E } [ \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } ( \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } ) ^ { \top } ] \succeq \mathbb { E } [ c c ^ { \top } ] \succeq \eta ^ { 2 } I .
$$

To justify the above, let ${ \mathcal { G } } _ { j } \subseteq { \mathcal { F } } _ { j }$ denote the sigma-fields generated by the current block’s actionobservation string and the full history, respectively. Since the blockwise action string is sampled independently of the block-start state, which has stationary distribution d, Bayes’ rule gives

$$
c = \mathbb { E } [ e _ { S _ { \mathrm { e n d } } } \mid \mathcal { G } _ { j } ] = \frac { K _ { a , y } d } { 1 ^ { \top } K _ { a , y } d } .
$$

Moreover, writing $b _ { j } = \mathbb { E } [ e _ { S _ { \mathrm { e n d } } } \ | \ \mathcal { F } _ { j } ]$ , the tower property yields $\mathbb { E } [ b _ { j } \ | \ { \mathcal { G } } _ { j } ] = c$ . Thus, for any $x \in \mathbb { R } ^ { | S | }$ <sup>|</sup>, conditional Jensen’s inequality and Assumption 3.26 imply

$$
\begin{array} { r } { \mathbb { E } [ ( { \boldsymbol { x } } ^ { \top } b _ { j } ) ^ { 2 } ] \geq \mathbb { E } [ ( { \boldsymbol { x } } ^ { \top } { \boldsymbol { c } } ) ^ { 2 } ] \geq \left( \mathbb { E } | { \boldsymbol { x } } ^ { \top } { \boldsymbol { c } } | \right) ^ { 2 } } \\ { = \left( \Phi _ { H } ^ { \pi } ( { \boldsymbol { x } } ) \right) ^ { 2 } \geq \eta ^ { 2 } \| { \boldsymbol { x } } \| _ { 2 } ^ { 2 } , } \end{array}
$$

where the equality follows by averaging over the block’s action-observation words and canceling the posterior normalizer against their probabilities.

To obtain a high-probability bound from an arbitrary initial distribution, define

$$
r _ { \eta } : = \operatorname* { m i n } \left\{ r \geq 0 : \operatorname* { s u p } _ { \mu \in \Delta _ { | S | } } \| P _ { \pi } ^ { r } \mu - d \| _ { 1 } \leq \frac { \eta } { 2 } \right\} , \qquad h : = r _ { \eta } + 1 .
$$

Assumption 3.22 controls $r _ { \eta }$ through log $\begin{array} { r } { \left( G _ { \mathrm { b l k } } ( \pi ) \right) \mathrm { a n d } \frac { 1 } { - \log \left( \rho _ { \mathrm { b l k } } ( \pi ) \right) } \colon } \end{array}$

$$
r _ { \eta } \le \operatorname* { m a x } \left\{ 0 , \left\lceil \frac { \log ( 2 G _ { \mathrm { b l k } } ( \pi ) / \eta ) } { - \log ( \rho _ { \mathrm { b l k } } ( \pi ) ) } \right\rceil \right\} ,
$$

which means

$$
h = O \left( 1 + \left\lceil \frac { \log ( G _ { \mathrm { b l k } } ( \pi ) / \eta ) } { - \log ( \rho _ { \mathrm { b l k } } ( \pi ) ) } \right\rceil \right) .
$$

Let $\Phi _ { H } ^ { \pi , \mu }$ denote the same functional as $\Phi _ { H } ^ { \pi }$ , with d replaced by $\mu .$ . The nonnegativity of the block operators and $\mathbf { 1 } ^ { \top } P _ { \pi } = \mathbf { 1 } ^ { \top }$ imply

$$
| \Phi _ { H } ^ { \pi , \mu } ( x ) - \Phi _ { H } ^ { \pi } ( x ) | \leq \| \mu - d \| _ { 1 } \| x \| _ { 2 } .
$$

For $j \geq h ,$ condition on $\mathcal { F } _ { j - h }$ and retain only the final block’s record in the subsequent history. After integrating out the intervening $r _ { \eta }$ blocks, the starting distribution of that final block is

$$
\mu _ { j } = P _ { \pi } ^ { r _ { \eta } } b _ { j - h } .
$$

Thus $\| \mu _ { j } - d \| _ { 1 } \leq \eta / 2$ , and conditional Jensen gives

$$
\mathbb { E } \big [ b _ { j } b _ { j } ^ { \top } \mid \mathcal { F } _ { j - h } \big ] \succeq \frac { \eta ^ { 2 } } { 4 } I .
$$

For each $r \in \{ 1 , \ldots , h \}$ , consider the residue class

$$
J _ { r } = \{ r + h , r + 2 h , \ldots \} \cap \{ 1 , \ldots , N \} , \qquad n _ { r } : = | J _ { r } | \geq { \frac { N } { h } } - 2 .
$$

The subsequence $\{ \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } ( \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } ) ^ { \top } \} _ { j \in J _ { \tau } }$ is adapted to the filtration $\{ { \mathcal { F } } _ { j } \} _ { j \in J _ { r } }$ , and its predictable sum satisfies

$$
\sum _ { j \in J _ { r } } \mathbb { E } [ \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } ( \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } ) ^ { \top } \mid \mathcal { F } _ { j - h } ] \succeq \frac { n _ { r } \eta ^ { 2 } } { 4 } I .
$$

Thus, Because $0 \preceq \bar { b } _ { i . \theta ^ { * } } ^ { \mathrm { b l k } } ( \bar { b } _ { i . \theta ^ { * } } ^ { \mathrm { b l k } } ) ^ { \top } \preceq I .$ , we can apply the adapted matrix Chernoff inequality [Tropp, 2011, Theorem 3.1], with $\dot { R } = 1$ and further choose $\delta = 1 / 2$ , which gives

$$
\mathbb { P } \left( \lambda _ { \operatorname* { m i n } } \left( \sum _ { j \in J _ { r } } \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } ( \bar { b } _ { j , \theta ^ { * } } ^ { \mathrm { b l k } } ) ^ { \top } \right) < \frac { n _ { r } \eta ^ { 2 } } { 8 } \right) \leq | S | \exp \left( - \frac { n _ { r } \eta ^ { 2 } } { 3 2 } \right) .
$$

If

$$
N \geq h \left( 2 + \frac { 3 2 } { \eta ^ { 2 } } \log \frac { | S | h } { \delta } \right) ,
$$

then the failure probability for each residue class is at most $\delta / h$ . Hence, by a union bound, with probability at least $1 - \delta .$

$$
\begin{array} { r l } & { \lambda _ { \operatorname* { m i n } } \bigl ( ( \bar { B } ^ { \mathrm { b l k } } ) ^ { \top } \bar { B } ^ { \mathrm { b l k } } \bigr ) = \lambda _ { \operatorname* { m i n } } \left( \displaystyle \sum _ { j = 1 } ^ { N } b _ { j } b _ { j } ^ { \top } \right) } \\ & { \qquad \quad \geq \displaystyle \sum _ { r = 1 } ^ { h } \lambda _ { \operatorname* { m i n } } \left( \displaystyle \sum _ { j \in J _ { r } } b _ { j } b _ { j } ^ { \top } \right) } \\ & { \qquad \quad \geq \displaystyle \frac { \eta ^ { 2 } } { 8 } \displaystyle \sum _ { r = 1 } ^ { h } n _ { r } = \frac { ( N - h ) \eta ^ { 2 } } { 8 } \geq \frac { N \eta ^ { 2 } } { 1 6 } , } \end{array}
$$

where the last inequality uses $N \geq 2 h$ . Consequently,

$$
\sigma _ { \mathrm { m i n } } \left( \frac { B _ { N } } { \sqrt { N } } \right) \ge \frac { \eta } { 4 } ,
$$

as claimed.

Hence inverse-polynomial η and polynomially controlled block mixing give inverse-polynomial belief conditioning after polynomially many blocks.

This completes the proof of Proposition R.1, which, together with Assumption 3.18, yields Lemma 3.28. These results, in turn, establish Theorem 3.29, providing a kernel recovery guarantee with polynomial sample complexity and thereby proving sufficiency.

Now, we show the second half of Proposition 3.27 (necessity): without Assumption 3.26, the remaining assumptions admit instances requiring $\exp ( \Omega ( | S | ) )$ observations for any estimator, even at accuracy $\varepsilon = \dot { \Theta } ( | S | ^ { - 1 } )$ and constant confidence.

Necessity for general kernel recovery. Proposition 3.25 supplies two model families satisfying all the remaining assumptions, with Assumption 3.17 satisfied with $m = 2$ . By Lemma Q.1, their backward excitation is exponentially small in |S| uniformly over all horizons. Thus both families violate Assumption 3.26 even when $L$ may be any polynomial in |S|. Their kernels are separated by order $\lvert S \rvert ^ { - 1 }$ modulo permutation, while their observation laws remain statistically indistinguishable for exponentially many observations. The resulting testing lower bound applies to every algorithm and estimator and therefore rules out a uniform polynomial kernel-recovery guarantee under the remaining assumptions alone.

Lemma Q.1 (Failure of backward excitation at every horizon). Let $| S | = 2 r ,$ , with $r \geq 3 ,$ and consider the two models $\theta _ { \nu } , \nu \in \{ - 1 , + 1 \}$ , constructed in Appendix P. Write $\Phi _ { L } ^ { \pi , \nu }$ for the backwardexcitation functional under $\theta _ { \nu }$ , initialized at $d = | S | ^ { - 1 } \mathbf { 1 }$ . For the unit vector v in the construction, every $L \geq 1$ satisfies

$$
\Phi _ { L } ^ { \pi , \nu } ( v ) \leq \frac { 1 } { 2 } \left( \frac { 1 3 } { 1 6 } \right) ^ { r - 2 } + ( r - 1 ) 2 ^ { - r } .
$$

Consequently,

$$
\operatorname* { s u p } _ { L \geq 1 } \operatorname* { i n f } _ { \| x \| _ { 2 } = 1 } \Phi _ { L } ^ { \pi , \nu } ( x ) \leq \frac { 1 } { 2 } \left( \frac { 1 3 } { 1 6 } \right) ^ { | S | / 2 - 2 } + \left( \frac { | S | } { 2 } - 1 \right) 2 ^ { - | S | / 2 } = \exp ( - \Omega ( | S | ) ) .
$$

Thus both modelfamilies violate Assumption 3.26for any $L$

Proof. Suppress the action, since every action has the same transition kernel, and write

$$
T _ { \nu } ^ { 0 } : = \frac 3 4 I _ { 2 r } + \frac 1 8 Q + \frac 1 8 \Pi + \nu h u v ^ { \top } , \qquad T _ { \nu } = ( 1 - \zeta ) T _ { \nu } ^ { 0 } + \zeta Q ^ { \top } , \qquad \zeta = 2 ^ { - r } .
$$

Both $T _ { \nu } ^ { 0 }$ and $Q ^ { \top }$ are doubly stochastic, so $d = | S | ^ { - 1 } \mathbf { 1 }$ is stationary. Moreover, the prescribed open-loop action records are independent of the latent and observation processes and may be omitted from the conditioning. Let $Y _ { t }$ denote the observation emitted from $s _ { t } ,$ , and set

$$
c _ { L } : = \mathbb { P } _ { \nu } ( s _ { L + 1 } = \cdot \mid Y _ { 1 : L } ) , \qquad s _ { 1 } \sim d .
$$

Then

$$
\Phi _ { L } ^ { \pi , \nu } ( v ) = \mathbb { E } _ { \nu } | v ^ { \top } c _ { L } | .
$$

Polynomial annihilation and resets. Since $p _ { j }$ is affine in $j ,$ , the definition of $v _ { A }$ and the finitedifference identity give

$$
\sum _ { j = 1 } ^ { r } ( v _ { A } ) _ { j } f ( p _ { j } ) = 0 \qquad \mathrm { ~ f o r ~ e v e r y ~ p o l y n o m i a l ~ } f \mathrm { ~ o f ~ d e g r e e ~ a t ~ m o s t ~ } r - 2 .
$$

Let $S _ { A } = \{ A _ { 1 } , . . . , A _ { r } \}$ and $\boldsymbol { S _ { B } } = \left\{ \boldsymbol { B } _ { 1 } , \ldots , \boldsymbol { B _ { r } } \right\}$ . The corresponding blocks of the unperturbed transition matrix are

$$
( T _ { \nu } ^ { 0 } ) _ { S _ { A } , S _ { A } } = { \frac { 3 } { 4 } } I _ { r } + { \frac { 1 } { 1 6 r } } { \bf 1 1 ^ { \top } } , \qquad ( T _ { \nu } ^ { 0 } ) _ { S _ { A } , S _ { B } } = { \frac { 3 } { 1 6 r } } { \bf 1 1 ^ { \top } } .
$$

In particular, observing $b _ { i }$ identifies the current state as $B _ { i }$ . If the following transition uses $T _ { \nu } ^ { 0 }$ , the S -component of the next predictive belief is the constant vector $( 3 / ( 1 6 r ) \bar { ) \mathbf { 1 } }$

Starting from this constant vector, suppose that the next k observations belong to $\big \{ o _ { 0 } , o _ { 1 } \big \}$ and that all their following transitions use $T _ { \nu } ^ { 0 }$ . The resulting $\mathcal { S } _ { A }$ -component has coordinates $f _ { k } ( p _ { 1 } ) , \ldots , f _ { k } ( p _ { r } )$ for a polynomial $f _ { k }$ of degree at most k. Indeed, an observation multiplies coordinate j by either $p _ { j }$ or $1 - p _ { j }$ , while the subsequent $s _ { A } , s _ { A }$ transition multiplies the vector by $3 / 4$ and adds a constant vector. Normalization only introduces a common scalar. Hence its projection onto $v _ { A }$ vanishes whenever $k \leq r - 2$ . The same argument applies starting from $d ,$ whose S -component is constant.

Accounting for the rare transitions. Realize each transition using an independent indicator $C _ { t } \sim$ Bernoulli(ζ): use $T _ { \nu } ^ { 0 }$ when $C _ { t } = 0$ and $Q ^ { \top }$ when $C _ { t } = 1$ . Define the more informed belief

$$
\widehat { c } _ { L } : = \mathbb { P } _ { \nu } ( s _ { L + 1 } = \cdot \mid Y _ { 1 : L } , C _ { 1 : L } ) .
$$

The tower property and conditional Jensen’s inequality imply

$$
c _ { L } = \mathbb { E } _ { \nu } [ \widehat { c } _ { L } \mid Y _ { 1 : L } ] , \qquad \Phi _ { L } ^ { \pi , \nu } ( v ) \leq \mathbb { E } _ { \nu } | v ^ { \top } \widehat { c } _ { L } | .
$$

First suppose $L \ge R : = r - 1$ , and define

$$
E _ { L } : = \left\{ C _ { t } = 1 \mathrm { f o r } \mathrm { s o m e } L - R + 1 \le t \le L \right\} \cup \left\{ Y _ { L - R + 1 : L } \in \left\{ o _ { 0 } , o _ { 1 } \right\} ^ { R } \right\} .
$$

On $E _ { L } ^ { c }$ , the last R observations contain a symbol $b _ { i } ,$ and all the last R transitions use $T _ { \nu } ^ { 0 }$ . After the most recent such $b _ { i } ,$ , at most $R - 1 = r - 2$ observations remain. The reset and polynomial-annihilation argument therefore gives

$$
v ^ { \top } \widehat { c } _ { L } = 0 \qquad \mathrm { o n } E _ { L } ^ { c } .
$$

Since $| v ^ { \top } \widehat { c } _ { L } | \leq 1$ , it follows that

$$
\Phi _ { L } ^ { \pi , \nu } ( v ) \leq \mathbb { P } _ { \nu } ( E _ { L } ) .
$$

The probability of a rare transition among the last R transitions is at most $R \zeta$ . Also, $Y _ { t } \in \{ o _ { 0 } , o _ { 1 } \}$ holds exactly when $s _ { t } \in S _ { A }$ . For every $s \in { S _ { A } }$ A

$$
\mathbb { P } _ { \nu } ( s _ { t + 1 } \in \mathcal { S } _ { A } \mid s _ { t } = s ) = ( 1 - \zeta ) \frac { 1 3 } { 1 6 } .
$$

Since $d ( S _ { A } ) = 1 / 2$ , stationarity gives

$$
\mathbb { P } _ { \nu } \left( Y _ { L - R + 1 : L } \in \{ o _ { 0 } , o _ { 1 } \} ^ { R } \right) = \frac { 1 } { 2 } \left( ( 1 - \zeta ) \frac { 1 3 } { 1 6 } \right) ^ { R - 1 } .
$$

Consequently,

$$
\Phi _ { L } ^ { \pi , \nu } ( v ) \leq ( r - 1 ) \zeta + \frac { 1 } { 2 } \left( \frac { 1 3 } { 1 6 } \right) ^ { r - 2 } , \qquad L \geq r - 1 .
$$

If $L < r - 1$ , then on $\{ C _ { 1 } = \cdot \cdot \cdot = C _ { L } = 0 \}$ the polynomial-annihilation argument applies either from the initial distribution or from the most recent $b _ { i }$ observation. The relevant polynomial has degree at most $L \leq r - 2 , \mathbf { s o } v ^ { \top } \widehat { c } _ { L } = 0$ . Therefore,

$$
\Phi _ { L } ^ { \pi , \nu } ( v ) \leq \mathbb { P } _ { \nu } ( \exists t \leq L : C _ { t } = 1 ) \leq L \zeta \leq ( r - 1 ) \zeta .
$$

Combining the two cases and substituting $\zeta = 2 ^ { - r }$ proves the asserted bound, uniformly over $L$ and ν.

Finally, for every fixed $c > 0$ independent of $| S |$ , the bound is smaller than $c | { \mathcal { S } } | ^ { - C }$ for all sufficiently large even |S|. Since $\| v \| _ { 2 } = 1$ , no horizon $L$ can then satisfy $\Phi _ { L } ^ { \pi , \nu } ( x ) \geq \dot { c } \vert \dot { S } \vert ^ { - C } \Vert x \Vert _ { 2 }$ for every x. This rules out Assumption 3.26 even with an existential polynomial-length horizon. □

Lemma Q.1 establishes the necessity of quantitative backward excitation for polynomial kernel recovery over general overcomplete POMDPs that satisfy Assumptions 3.9, 3.17, 3.18, 3.22, and 3.24. For completeness, we further establish that backward excitation over some polynomial-length history is necessary for polynomial conditioning of the empirical predictive-belief matrix underlying our recovery guarantee, which means Assumption 3.26 is essential to establishing Lemma 3.28 with polynomial complexity.

Existential necessity for polynomial belief conditioning. Work under stationary initialization. For each $\ell \in H \mathbb { N }$ , let $c _ { \ell }$ denote the belief at the terminal boundary conditional on the most recent length-ℓ window under the original behavior policy, and define

$$
M _ { \ell } : = \mathbb { E } _ { d } [ c _ { \ell } c _ { \ell } ^ { \top } ] .
$$

By definition,

$$
\Phi _ { \ell } ^ { \pi } ( x ) = \mathbb { E } _ { d } \vert x ^ { \top } c _ { \ell } \vert .
$$

Suppose that, for some polynomially bounded $N , \gamma > 0 .$ , and $\delta < 1$

$$
\mathbb { P } _ { d } \bigg [ \sigma _ { \operatorname* { m i n } } \bigg ( \frac { \bar { B } ^ { \mathrm { b l k } } } { \sqrt { N } } \bigg ) \geq \gamma \bigg ] \geq 1 - \delta .
$$

Set $L = N H$ . For every $j \leq N$ , stationarity identifies the distribution of the terminal belief based on the last $j H$ steps with that of $\bar { b } _ { j , \theta _ { \star } } ^ { \mathrm { b l k } }$ . Since the full length-L history contains these steps, conditional Jensen gives $M _ { j H } \preceq M _ { L }$ . Consequently,

$$
M _ { L } \succeq \frac { 1 } { N } \sum _ { j = 1 } ^ { N } M _ { j H } = \mathbb { E } _ { d } \bigg [ \frac { ( \bar { B } ^ { \mathrm { b l k } } ) ^ { \top } \bar { B } ^ { \mathrm { b l k } } } { N } \bigg ] \succeq ( 1 - \delta ) \gamma ^ { 2 } I .
$$

For every unit vector $x ,$ the bound $| x ^ { \top } c _ { L } | \leq 1$ implies

$$
\Phi _ { L } ^ { \pi } ( x ) = \mathbb { E } _ { d } \lvert x ^ { \top } c _ { L } \rvert \ge \mathbb { E } _ { d } [ ( x ^ { \top } c _ { L } ) ^ { 2 } ] = x ^ { \top } M _ { L } x \ge ( 1 - \delta ) \gamma ^ { 2 } .
$$

By homogeneity,

$$
\begin{array} { r } { \Phi _ { L } ^ { \pi } ( { \boldsymbol x } ) \geq ( 1 - \delta ) \gamma ^ { 2 } \Vert { \boldsymbol x } \Vert _ { 2 } , \qquad { \boldsymbol x } \in \mathbb { R } ^ { S } . } \end{array}
$$

Thus, if $\gamma ^ { - 1 }$ is polynomially bounded in |S|, and $1 - \delta$ is bounded below by a positive constant, quantitative backward excitation holds at some polynomially bounded horizon L. Together with the sufficiency argument, this establishes the equivalence between the polynomial minimum singular value of $\check { \bar { B } } ^ { \mathrm { b l k } }$ and Assumption 3.26.

## R

We first state the following proposition which is an overcomplete counterpart to Lemma 3.13.

Proposition R.1. Suppose Assumption 3.26 holdsfor the blockwise open-loop policy π, with parameters L and η. Fix a confidence level $\delta \in ( 0 , 1 )$ ). If n is bounded below by a universal polynomial with nonnegative powers in

$$
| S | , \quad \log ( 1 / \delta ) , \quad \log ( G _ { \mathrm { b l k } } ( \pi ) ) , \quad \frac { 1 } { - \log ( \rho _ { \mathrm { b l k } } ( \pi ) ) } , \quad L , \quad 1 / \eta ,
$$

then, with probability at least $1 - \delta ,$

$$
\sigma _ { \mathrm { m i n } } ^ { 2 } ( \bar { B } ^ { \mathrm { b l k } } ) \geq \frac { \eta ^ { 2 } } { 1 6 L } n .
$$

Proof. It is established in the Sufficiency part of Appendix Q.

Then, by combining Proposition R.1 and Assumption 3.18 and applying the same argument as in Lemma J.1, we obtain Lemma 3.28.

## S

We prove Theorem 3.29 using a modified version of the robust Kruskal’s theorem, specifically Theorem 5 from Bhaskara et al. [2014]. For concision and clarity, we track only the dependence on n, suppressing the polynomial dependence on $\begin{array} { r } { | A | , \ \frac { 1 } { \pi _ { \mathrm { m i n } } ^ { \mathrm { b l k } } } , \ \frac { 1 } { ( d _ { \mathrm { m i n } } ^ { \pi } ) ^ { \mathrm { b l k } } } , \ \frac { 1 } { \sigma _ { T } } , \ \frac { 1 } { \sigma _ { M } } , \ \bar { \Delta } _ { O } , \ | S | , | \Omega | } \end{array}$ , and $\textstyle { \frac { L } { \eta ^ { 2 } } }$ (the first nine quantities in (5)).

Robust Kruskal preliminaries. Throughout the proof, $\mathrm { K - r a n k } _ { \tau } ( \cdot )$ denotes the robust Kruskal rank of Bhaskara et al. [2014]: K-rank $( A ) \geq k$ means that every k-column submatrix of A has minimum singular value at least $1 / \tau$ . A matrix is ρ-bounded if the $\ell _ { 2 }$ norm of every column is upper-bounded by $\rho .$

Theorem S.1 (Dimension-free robust uniqueness of tensor decompositions). Let $\boldsymbol { T } = [ A B C ]$ be a rank-R tensor, where $A \in \mathbb { R } ^ { n _ { A } \times R } , B \in \mathbb { R } ^ { n _ { B } \times R }$ , and $C \in \mathbb { R } ^ { n c \times R }$ . Suppose T is $( \rho _ { A } , \rho _ { B } , \rho _ { C } ) .$ bounded, with

$$
\operatorname { K r a n k } _ { \tau _ { A } } ( A ) \geq k _ { A } , \quad \operatorname { K r a n k } _ { \tau _ { B } } ( B ) \geq k _ { B } , \quad \operatorname { K r a n k } _ { \tau _ { C } } ( C ) \geq k _ { C } ,
$$

satisfying $k _ { A } + k _ { B } + k _ { C } \geq 2 R + 2 .$

Then there exist a universal constant $c > 0$ and a polynomial ${ \widetilde { \vartheta } } _ { 5 } ,$ , independent of $n _ { A } , n _ { B } , n _ { C }$ , such thatfor every $0 < \varepsilon ^ { \prime } < 1$ , setting

$$
\varepsilon = \frac { \varepsilon ^ { \prime } } { R ^ { c } \prod _ { M \in \{ A , B , C \} } \widetilde { \vartheta } _ { 5 } ( \tau _ { M } , \rho _ { M } , \rho _ { M } ^ { \prime } ) } ,
$$

any other $( \rho _ { A } ^ { \prime } , \rho _ { B } ^ { \prime } , \rho _ { C } ^ { \prime } )$ -bounded rank-R decomposition $T ^ { \prime } = \left\lceil A ^ { \prime } B ^ { \prime } C ^ { \prime } \right\rceil$ satisfying $\Vert T - T ^ { \prime } \Vert _ { F } \leq \varepsilon$ admits an $R \times R$ permutation matrix Π and diagonal matrices $\Lambda _ { A } , \Lambda _ { B } , \Lambda _ { C }$ such that

$$
\| A ^ { \prime } - A \Pi \Lambda _ { A } \| _ { F } \leq \varepsilon ^ { \prime } , \qquad \| B ^ { \prime } - B \Pi \Lambda _ { B } \| _ { F } \leq \varepsilon ^ { \prime } ,
$$

$$
\| C ^ { \prime } - C \Pi \Lambda _ { C } \| _ { F } \le \varepsilon ^ { \prime } , \qquad \| \Lambda _ { A } \Lambda _ { B } \Lambda _ { C } - I _ { R } \| _ { F } \le \varepsilon ^ { \prime } .
$$

In particular, the robustness guarantee has no explicit dependence on the ambient dimensions $n _ { A } , n _ { B } , n _ { C }$

Proof sketch. We briefly explain how the proof of the original theorem in Bhaskara et al. [2014] can be adapted to prove our modified theorem.

First, in Lemma 30, instead of drawing the Gaussian vector in $\mathbb { R } ^ { d }$ , draw it in $V = \operatorname { s p a n } \{ u _ { 1 } , \ldots , u _ { t } \}$ Since $s = \dim ( V ) \leq t .$ the same Gaussian anti-concentration and norm-concentration arguments yield a unit vector $w \in V$ satisfying

$$
| \langle u _ { i } , w \rangle | > \frac { \eta } { 2 0 t \sqrt { s } } \geq \frac { \eta } { 2 0 t ^ { 3 / 2 } } , \qquad i \in [ t ] ,
$$

whenever $\| u _ { i } \| _ { 2 } \geq \eta > 0$ . Consequently, in the proof of Lemma 11, we may replace $\varepsilon _ { 1 } = n R \varepsilon$ with

$$
\varepsilon _ { 1 } = 2 0 R ^ { 3 / 2 } \varepsilon ,
$$

removing the dependence on n. In fact, a stronger dimension-free bound $| \langle u _ { i } , w \rangle | \geq \eta / t$ follows from a deterministic sign-maximization argument, which we omit here for brevity.

Second, in Lemma 12, the original proof already establishes

$$
\mathrm { d i s t } ( y , \operatorname { s p a n } ( X _ { A } ) ) \leq ( 2 q \tau \rho + 1 ) \varepsilon .
$$

Since $q \leq R$ and, without loss of generality, $\tau \geq 1$ , we obtain

$$
\mathrm { d i s t } ( y , \operatorname { s p a n } ( X _ { A } ) ) \leq 3 R \tau \operatorname* { m a x } \{ \rho , 1 \} \varepsilon .
$$

Thus, we may replace $\vartheta _ { 1 2 } = 4 n \tau \rho$ with

$$
\widetilde { \vartheta } _ { 1 2 } = 3 R \tau \operatorname* { m a x } \{ \rho , 1 \} .
$$

Substituting these two improvements into the induction proving Lemma 6 yields a robust permutation lemma whose error and parameters depend polynomially only on $R , \tau , \rho .$ . The factors needed to control the diagonal rescalings are likewise polynomial in these parameters, by the robust Kruskalrank condition.

The remaining tensor-contraction, rank-transfer, and permutation-alignment arguments proceed as in the original proof, with polynomial dependence on the robust conditioning and column-norm parameters. Combining these bounds gives the stated dimension-free polynomial ${ \widetilde { \vartheta } } _ { 5 }$ and exponent c.

An alternative argument for eliminating the ambient-dimension dependence is to restrict each mode to the subspace spanned by the columns of the two corresponding factor matrices. Specifically, for each $M \in { \mathsf { \bar { \{ A } } }  , B , C \}$ , define

$$
U _ { M } = \operatorname { s p a n } \bigl ( \operatorname { c o l } ( M ) \cup \operatorname { c o l } ( M ^ { \prime } ) \bigr ) .
$$

Since both M and $M ^ { \prime }$ have R columns, we have dim $( U _ { M } ) \leq 2 R$ . Representing both matrices in an orthonormal basis of $U _ { M }$ , with zero-padding if necessary, preserves their column norms, singular values, robust Kruskal ranks, and the Frobenius distance between the corresponding tensor decompositions. Consequently, we may apply the original Theorem 5 with $n _ { A } = n _ { B } = n _ { C } = 2 R$ and lift the resulting approximation back to the original ambient spaces. Since the original $\vartheta _ { 5 }$ is polynomial, substituting $n = 2 R$ yields a polynomial depending only on $R , \tau , \rho , \rho ^ { \prime }$ , thereby eliminating all dependence on $n _ { A } , n _ { B } , n _ { C }$ 口

We then proceed to prove Theorem 3.29 using Theorem S.1. For simplicity, we prove the pointwise version of the theorem using Lemma 3.21. The uniform statement displayed in Theorem $3 . { \overset { \cdot } { 2 9 } }$ can be shown identically using the uniform counterpart of Lemma 3.21, which can be proven by applying the same argument from Appendix E to Lemma 3.21.

1. Fix an action string. Fix $a _ { 1 } , a _ { 2 } \in { \mathcal { A } } ,$ denote $( a _ { 1 } , a _ { 2 } )$ as a˜, and fix

$$
w \in \mathcal { W } _ { \tilde { a } } ^ { \pi } .
$$

By Assumption 3.18, the blockwise behavior policy plays

$$
a _ { 1 } \oplus a _ { 2 } \oplus w \oplus a ^ { \prime }
$$

with probability $\Theta ( 1 )$ after marginalizing over the terminal action $a ^ { \prime }$ . Recall the definitions of $\mathcal { T } _ { \tilde { a } , w } ^ { \mathrm { b l k } }$ and $N _ { \tilde { a } , w } ^ { \mathrm { b l k } }$ . From Assumption 3.18, we can see that

$$
N _ { \tilde { a } , w } ^ { \mathrm { b l k } } = \Theta ( n )
$$

with high probability. We assume this high-probability event throughout the rest of the proof.

For notational simplicity, denote

$$
\begin{array} { r } { B : = \bar { B } _ { \bar { a } , w } ^ { \mathrm { b l k } } \in \mathbb { R } ^ { N \times | S | } , \qquad \widehat { B } : = \widehat { B } _ { \bar { a } , w } ^ { \mathrm { b l k } } \in \mathbb { R } ^ { N \times | S | } , \qquad N _ { \bar { a } , w } ^ { \mathrm { b l k } } : = N , \qquad \widehat { Z } _ { \bar { a } , w } ^ { \mathrm { b l k } } : = \mathcal { Z } . } \end{array}
$$

From Proposition R.1,

$$
\sigma _ { \operatorname* { m i n } } ( B ) = \Theta ( { \sqrt { N } } ) = \Theta ( { \sqrt { n } } ) .
$$

Thus, for some $\tau _ { B } = \Theta ( 1 )$ , where $\tau _ { B }$ depends polynomially on two of the first nine quantities in (5) (from Lemma 3.28),

$$
\mathrm { K - r a n k } _ { \tau _ { B } } \left( { \frac { B } { \sqrt { N } } } \right) = | S | .\tag{54}
$$

For notational simplicity, write

$$
M _ { w } : = \mathcal { M } _ { \theta ^ { * } } ^ { ( m ) } [ w ] , \qquad T _ { 1 } : = T _ { \beta ^ { * } } ^ { a _ { 1 } } , \qquad T _ { 2 } : = T _ { \beta ^ { * } } ^ { a _ { 2 } } ,
$$

and use hats for the corresponding estimated quantities.

Schematically, each block can be illustrated as below:

$$
S _ { 0 } \longrightarrow O _ { 0 } { \stackrel { a 1 } { \longrightarrow } } S _ { 1 } \longrightarrow O _ { 1 } { \stackrel { a 2 } { \longrightarrow } } S _ { 2 } { \stackrel { w } { \longrightarrow } } m \mathrm { - o b s e r v a t i o n \ b l o c k . }\tag{55}
$$

2. First tensor decomposition. Restrict the block indices to $i \in \mathcal { T }$ . Every such block has the form illustrated in (55). Let $Z \in \Omega ^ { m }$ denote the informative observation block generated after $a _ { 2 }$ Thus, for each $i \in \mathcal { Z }$ , the corresponding joint predictive law (i.e. $p _ { i , \theta ^ { * } }$ on the LHS of the bound in Lemma 3.21) can be factorized as

$$
\begin{array} { r l } & { p _ { i , \theta ^ { * } } ^ { ( m + 2 ) } : = { \mathbb P } _ { \theta ^ { * } } ( O _ { 0 } = o _ { 0 } , O _ { 1 } = o _ { 1 } , Z = z \mid \mathcal { H } _ { i } , a _ { 1 } , a _ { 2 } , w ) } \\ & { \qquad = \displaystyle \sum _ { s _ { 0 } , s _ { 1 } , s _ { 2 } } \bar { b } _ { i , \theta ^ { * } } ^ { \mathrm { b l k } } ( s _ { 0 } ) O _ { \varphi ^ { * } } ( o _ { 0 } \mid s _ { 0 } ) T _ { 1 } ( s _ { 1 } \mid s _ { 0 } ) O _ { \varphi ^ { * } } ( o _ { 1 } \mid s _ { 1 } ) T _ { 2 } ( s _ { 2 } \mid s _ { 1 } ) [ M _ { w } ] ( z , s _ { 2 } ) . } \end{array}
$$

Define

$$
A _ { 0 } : = M _ { 2 } T _ { 2 } T _ { 1 } .
$$

Marginalizing over $O _ { 1 }$ and using $\begin{array} { r } { \sum _ { o _ { 1 } } O _ { \varphi ^ { * } } ( o _ { 1 } \mid s _ { 1 } ) = 1 } \end{array}$ gives

$$
q _ { i , \theta ^ { * } } ( o _ { 0 } , z ) : = \mathbb { P } _ { \theta ^ { * } } ( O _ { 0 } = o _ { 0 } , Z = z \mid \mathcal { H } _ { i } , a _ { 1 } , a _ { 2 } , w ) = \sum _ { s _ { 0 } } \bar { b } _ { i , \theta ^ { * } } ^ { \mathrm { b l k } } ( s _ { 0 } ) O _ { \varphi ^ { * } } ( o _ { 0 } \mid s _ { 0 } ) [ A _ { 0 } ] ( z , s _ { 0 } ) .
$$

Stacking these marginalized laws over $i \in \mathcal { T }$ therefore yields the tensor

$$
\mathcal { T } _ { 0 } = [ A _ { 0 } , B , O _ { \varphi ^ { \ast } } ] ,
$$

and analogously

$$
\widehat { \mathcal { T } _ { 0 } } = [ \widehat { A } _ { 0 } , \widehat { B } , O _ { \hat { \varphi } } ] .
$$

To control the tensor error, we invoke Lemma 3.21.

By definition, for each $i \in \mathcal { Z } ,$

$$
q _ { i , \theta ^ { * } } \bigl ( o _ { 0 } , z \bigr ) = \sum _ { o _ { 1 } \in \Omega } p _ { t _ { i } , \theta ^ { * } } \bigl ( o _ { 0 } , o _ { 1 } , z \bigr ) .
$$

Furthermore, by construction, $q _ { i , \theta ^ { * } }$ and $q _ { i , \hat { \theta } }$ are precisely the ith slices of $\mathcal { T } _ { 0 }$ and ${ \widehat { \mathcal { T } } } _ { 0 } .$ , respectively. Since marginalization is an ℓ<sub>1</sub>-contraction,

$$
\lVert q _ { t _ { i } , \hat { \theta } } - q _ { i , \theta ^ { * } } \rVert _ { 1 } \leq \lVert p _ { t _ { i } , \hat { \theta } } - p _ { i , \theta ^ { * } } \rVert _ { 1 } .
$$

Moreover, reshaping $q _ { i , \theta }$ into a tensor slice does not change its entries, so

$$
\Vert \mathcal { T } _ { 0 } ( : , i , : ) - \widehat { \mathcal { T } _ { 0 } } ( : , i , : ) \Vert _ { F } \leq \Vert q _ { i , \hat { \theta } } - q _ { i , \theta ^ { * } } \Vert _ { 1 } .
$$

Therefore,

$$
\begin{array} { r l r } {  { \| \mathcal { T } _ { 0 } - \widehat { \mathcal { T } } _ { 0 } \| _ { F } ^ { 2 } = \sum _ { i \in \mathcal { Z } } \| \mathcal { T } _ { 0 } ( : , i , : ) - \widehat { \mathcal { T } } _ { 0 } ( : , i , : ) \| _ { F } ^ { 2 } } } \\ & { } & { \leq \displaystyle \sum _ { i \in \mathcal { Z } } \| p _ { t _ { i } , \hat { \theta } } - p _ { t _ { i } , \theta ^ { * } } \| _ { 1 } ^ { 2 } } \\ & { } & { \overset { \mathrm { \ L } n / ( m + 2 ) \big \vert } { \leq } \| p _ { t _ { i } , \hat { \theta } } - p _ { t _ { i } , \theta ^ { * } } \| _ { 1 } ^ { 2 } \leq \frac { 8 } { \pi _ { \operatorname* { m i n } } ^ { \mathrm { b l k } } } ( \ell _ { n } ( \theta ^ { * } ) - \ell _ { n } ( \hat { \theta } ) + 2 \log ( 2 / \delta ) ) + 4 , } \end{array}
$$

where the last step follows from Lemma 3.21. Since $N = | \mathcal { T } | = \Theta ( n )$ , normalizing the belief mode (B and $\widehat { B } )$ by $\sqrt { N }$ gives

$$
\varepsilon _ { n } : = \| \mathcal { T } _ { 0 , \mathrm { n o r m } } - \widehat { \mathcal { T } } _ { 0 , \mathrm { n o r m } } \| _ { F } = O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) ,\tag{56}
$$

where $\mathcal { T } _ { 0 , \mathrm { n o r m } }$ and $\widehat { \mathcal { T } } _ { 0 , \mathrm { n o r m } }$ are normalized counterparts of $\mathcal { T } _ { 0 }$ and $\widehat { \mathcal { T } _ { 0 } }$ and $\pi _ { \mathrm { m i n } } ^ { \mathrm { b l k } }$ is absorbed into the big O notation.

Note that we define $H _ { n } ( \delta ) = G _ { n } + 2 d \log ( 2 | \Omega | n ) + 2 \log ( 4 / \delta ) + 2$ . This makes the $\mathsf { b i g \mathrm { - } O }$ statement in (56) loose. It becomes tight when we replace Lemma 3.21 with its uniform counterpart to prove the uniform statement of Theorem 3.29.

Notice that, after this normalization, all factor matrices $( A _ { 0 } , B _ { : }$ , and $O _ { \varphi ^ { * } } )$ and their estimated counterparts satisfy the uniform column-norm boundedness required by Theorem S.1.

Then, note that Assumption 3.24 implies that every two-column submatrix of $O _ { \varphi ^ { * } }$ has minimum singular value bounded below by a positive constant. Hence, for some $\tau _ { O } = \Theta ( 1 )$ ,

$$
\mathrm { K } \mathrm { - r a n k } _ { \tau _ { O } } ( O _ { \varphi ^ { \ast } } ) \geq 2 .\tag{57}
$$

Furthermore, by Assumptions 3.9 and 3.17,

$$
\sigma _ { \operatorname* { m i n } } ( A _ { 0 } ) \geq \sigma _ { \operatorname* { m i n } } ( M _ { w } ) \sigma _ { \operatorname* { m i n } } ( T _ { 2 } ) \sigma _ { \operatorname* { m i n } } ( T _ { 1 } ) = \Theta ( 1 ) .
$$

Hence, for some $\tau _ { A _ { 0 } } = \Theta ( 1 )$ ,

$$
\operatorname { K } \mathrm { - r a n k } _ { \tau _ { A _ { 0 } } } ( A _ { 0 } ) = | \mathcal { S } | .
$$

Together with (57) and (54),

$$
\mathrm { K } \mathrm { - r a n k } _ { \tau _ { A _ { 0 } } } ( A _ { 0 } ) + \mathrm { K } \mathrm { - r a n k } _ { \tau _ { B } } \big ( \frac { B } { \sqrt { N } } \big ) + \mathrm { K } \mathrm { - r a n k } _ { \tau _ { O } } ( O _ { \varphi ^ { \bullet } } ) \geq | S | + | S | + 2 = 2 | S | + 2 .
$$

Thus Theorem S.1 applies. Furthermore, notice that $\tau _ { A _ { 0 } } , \tau _ { B }$ , and $\tau _ { O }$ are $\Theta ( 1 )$ and depend only polynomially on the fixed model parameters in (5) (namely, the first nine) and not on n. Consequently, for n that exceeds a threshold that is polynomial in the first nine quantities in (5), there exist a permutation matrix $\Pi _ { 0 }$ and diagonal matrices $\Lambda _ { A , 0 } , \Lambda _ { B , 0 }$ , and $\Lambda _ { O , 0 }$ such that

$$
\| \widehat { A } _ { 0 } - A _ { 0 } \Pi _ { 0 } \Lambda _ { A , 0 } \| _ { F } \leq C _ { K } \varepsilon _ { n } = O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) ,\tag{58}
$$

$$
\left\| \frac { \widehat { B } } { \sqrt { N } } - \frac { B } { \sqrt { N } } \Pi _ { 0 } \Lambda _ { B , 0 } \right\| _ { F } \leq C _ { K } \varepsilon _ { n } = O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) ,
$$

and

$$
\| O _ { \hat { \varphi } } - O _ { \varphi ^ { * } } \Pi _ { 0 } \Lambda _ { O , 0 } \| _ { F } \leq C _ { K } \varepsilon _ { n } = O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) ,\tag{59}
$$

with

$$
\| \Lambda _ { A , 0 } \Lambda _ { B , 0 } \Lambda _ { O , 0 } - I \| _ { F } \leq C _ { K } \varepsilon _ { n } = O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) ,\tag{60}
$$

where $C _ { K }$ depends polynomially on the first nine quantities in (5).

3. Remove the scaling ambiguity in the first tensor. Both $A _ { 0 }$ and $\widehat { A } _ { 0 }$ are column-stochastic, since $M _ { w }$ is an observation channel and $T _ { 1 } , T _ { 2 }$ are column-stochastic transition matrices. Hence

$$
\mathbf { 1 } ^ { \top } A _ { 0 } = \mathbf { 1 } ^ { \top } \widehat { A } _ { 0 } = \mathbf { 1 } ^ { \top } .
$$

Left-multiplying the LHS of (58) by $\mathbf { 1 } ^ { \top }$ gives

$$
\Lambda _ { A , 0 } = I + O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) .\tag{61}
$$

Note that left-multiplying by $\mathbf { 1 } ^ { \top }$ introduces an additional factor of $| \Omega | ^ { m / 2 }$ in the bound. Since m is fixed, this factor is polynomial in |Ω| and can be absorbed into the polynomial constant.

$$
\mathbf { 1 } ^ { \top } O _ { \varphi ^ { \ast } } = \mathbf { 1 } ^ { \top } O _ { \hat { \varphi } } = \mathbf { 1 } ^ { \top }
$$

implies from (59) that

$$
\Lambda _ { { \cal O } , 0 } = I + { \cal O } \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) .\tag{62}
$$

Combining (60), (61), and (62) gives us

$$
\Lambda _ { B , 0 } = I + O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) .
$$

Thus

$$
\widehat { A } _ { 0 } = A _ { 0 } \Pi _ { 0 } + O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) , \qquad O _ { \hat { \varphi } } = O _ { \varphi ^ { * } } \Pi _ { 0 } + O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) ,\tag{63}
$$

and

$$
\Vert \widehat { B } - B \Pi _ { 0 } \Vert _ { F } = O \left( \sqrt { H _ { n } ( \delta ) } \right) .
$$

Here and throughout, $O \left( { \sqrt { \frac { H _ { n } ( \delta ) } { n } } } \right)$ in matrix-valued expressions denotes a remainder whose Frobenius norm is $O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right)$ , and the same convention applies to $O ( \sqrt { H _ { n } ( \delta ) } )$ .

## 4. Construct a second set of matrices. For $o _ { 0 } \in \Omega .$ , define

$$
D _ { o _ { 0 } } : = \mathrm { d i a g } ( O _ { \varphi ^ { * } } ^ { \top } e _ { o _ { 0 } } ) , \qquad C _ { 1 , o _ { 0 } } : = T _ { 1 } D _ { o _ { 0 } } B ^ { \top } .
$$

Concatenate these matrices over o<sub>0</sub>:

$$
\begin{array} { r } { C _ { 1 } : = \left[ C _ { 1 , o _ { 0 } } \right] _ { o _ { 0 } \in \Omega } = \left[ T _ { 1 } D _ { o _ { 0 } } B ^ { \top } \right] _ { o _ { 0 } \in \Omega } . } \end{array}
$$

For every $x \in \mathbb { R } ^ { | S | }$

$$
\begin{array} { r l } & { \| C _ { 1 } ^ { \top } x \| _ { 2 } ^ { 2 } = \displaystyle \sum _ { o \in \Omega } \left\| B D _ { o } T _ { 1 } ^ { \top } x \right\| _ { 2 } ^ { 2 } } \\ & { \qquad \geq \displaystyle \frac { 1 } { | \Omega | } \left\| B \left( \displaystyle \sum _ { o \in \Omega } D _ { o } \right) T _ { 1 } ^ { \top } x \right\| _ { 2 } ^ { 2 } } \\ & { \qquad = \displaystyle \frac { 1 } { | \Omega | } \| B T _ { 1 } ^ { \top } x \| _ { 2 } ^ { 2 } , } \end{array}
$$

where the inequality is the vector-valued Cauchy–Schwarz inequality and

$$
\sum _ { o \in \Omega } D _ { o } = I .
$$

Consequently,

$$
\sigma _ { \mathrm { m i n } } ( C _ { 1 } ) \geq \frac { \sigma _ { \mathrm { m i n } } ( B ) \sigma _ { \mathrm { m i n } } ( T _ { 1 } ) } { \sqrt { | \Omega | } } = \Theta ( \sqrt { n } ) .
$$

Therefore, for some $\tau _ { C _ { 1 } } = \Theta ( 1 )$ that only depends polynomially on the first nine quantities in (5) plus $| \Omega |$ |,

$$
\mathrm { K } { \cdot } \mathrm { r a n k } _ { \tau _ { C _ { 1 } } } \left( \frac { C _ { 1 } ^ { \top } } { \sqrt N } \right) = | S | .\tag{64}
$$

## 5. Second tensor decomposition Define

$$
A _ { 1 } : = M _ { w } T _ { 2 } .
$$

For each $i \in \mathcal { T }$ and $o _ { 0 } , o _ { 1 } \in \Omega$ , consider the joint predictive law of the first two observations and the informative m-observation block $Z .$ . Define the tensor $\mathcal { T } _ { 1 }$ by

$$
\begin{array} { r } { [ \mathcal { T } _ { 1 } ] _ { z , ( i , o _ { 0 } ) , o _ { 1 } } : = \mathbb { P } _ { \theta ^ { * } } \left( O _ { 0 } = o _ { 0 } , O _ { 1 } = o _ { 1 } , Z = z \mid \mathcal { H } _ { i } , a _ { 1 } , a _ { 2 } , w \right) . } \end{array}
$$

Recall that

$$
[ C _ { 1 } ] _ { : , ( i , o _ { 0 } ) } = T _ { 1 } D _ { o _ { 0 } } \bar { b } _ { i , \theta ^ { * } } ^ { \mathrm { b l k } } ,
$$

which is the unnormalized latent-state distribution at $S _ { 1 }$ after observing $O _ { 0 } = o _ { 0 }$ and applying action $a _ { 1 }$

Conditioning on $S _ { 1 } = s$ therefore gives

$$
[ \mathcal { T } _ { 1 } ] _ { z , ( i , o _ { 0 } ) , o _ { 1 } } = \sum _ { s \in \mathcal { S } } [ A _ { 1 } ] _ { z , s } [ C _ { 1 } ] _ { s , ( i , o _ { 0 } ) } O _ { \varphi ^ { * } } ( o _ { 1 } \mid s ) .
$$

Hence $\mathcal { T } _ { 1 }$ has the rank-|S| CP decomposition

$$
\mathcal { T } _ { 1 } = [ A _ { 1 } , C _ { 1 } ^ { \top } , O _ { \varphi ^ { * } } ] .
$$

The estimated model analogously gives

$$
\widehat { \mathcal { T } _ { 1 } } = [ \widehat { A } _ { 1 } , \widehat { C } _ { 1 } ^ { \top } , O _ { \widehat { \varphi } } ] , \qquad \widehat { A } _ { 1 } : = \widehat { M } _ { w } \widehat { T } _ { 2 } .
$$

Using Lemma 3.21 again gives

$$
\left. \frac { \mathcal { T } _ { 1 } } { \sqrt { N } } - \frac { \widehat { \mathcal { T } _ { 1 } } } { \sqrt { N } } \right. _ { F } = O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) .
$$

Again, this normalization makes sure that all factor matrices and their estimated counterparts satisfy the uniform column-norm boundedness required by Theorem S.1.

Moreover,

$$
\sigma _ { \operatorname* { m i n } } ( A _ { 1 } ) \geq \sigma _ { \operatorname* { m i n } } ( M _ { w } ) \sigma _ { \operatorname* { m i n } } ( T _ { 2 } ) = \Theta ( 1 ) ,
$$

$\mathbf { s o } ,$ for some $\tau _ { A _ { 1 } } = \Theta ( 1 )$ that depends polynomially on the first five quantities in (5),

$$
\operatorname { K - r a n k } _ { \tau _ { A _ { 1 } } } ( A _ { 1 } ) = | { \cal S } | .
$$

Together with (57) and (64),

$$
\mathrm { K - r a n k } _ { \tau _ { A _ { 1 } } } ( A _ { 1 } ) + \mathrm { K - r a n k } _ { \tau _ { C _ { 1 } } } \left( \frac { C _ { 1 } ^ { \top } } { \sqrt { N } } \right) + \mathrm { K - r a n k } _ { \tau _ { O } } ( O _ { \varphi ^ { * } } ) \geq 2 | \cal S | + 2 .
$$

Hence Theorem S.1 applies again: There exist a permutation matrix $\Pi _ { 1 }$ and diagonal scalings such that, after removing the scaling ambiguity exactly as in the first tensor decomposition,

$$
\widehat { A } _ { 1 } = A _ { 1 } \Pi _ { 1 } + O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) ,
$$

$$
\widehat { C } _ { 1 } ^ { \top } = C _ { 1 } ^ { \top } \Pi _ { 1 } + O \left( \sqrt { H _ { n } ( \delta ) } \right) ,
$$

and

$$
O _ { \hat { \varphi } } = O _ { \varphi ^ { * } } \Pi _ { 1 } + O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) .\tag{65}
$$

Indeed, the first and third factors are column-stochastic, so their diagonal scalings are $I +$ $O \left( { \sqrt { \frac { H _ { n } ( \delta ) } { n } } } \right)$ , and the product-of-scalings condition in the theorem then forces the scaling of $C _ { 1 } ^ { \top }$ to be $\displaystyle I + O \left( { \sqrt { \frac { H _ { n } ( \delta ) } { n } } } \right)$ as well. Again, note that left-multiplying by $\mathbf { 1 } ^ { \top }$ introduces an additional factor of $| \Omega | ^ { m / 2 }$ in the upper bound of the first equation from the dimension of $A _ { 1 }$ . Since m is fixed, this factor is polynomial in |Ω| and can be absorbed into the polynomial constant.

6. Synchronize the two permutations. From (63) and (65),

$$
O _ { \varphi ^ { * } } \Pi _ { 0 } - O _ { \varphi ^ { * } } \Pi _ { 1 } = O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) .
$$

If $\Pi _ { 0 } \neq \Pi _ { 1 }$ , then for some column the left-hand side contains the difference of two distinct columns of $O _ { \varphi ^ { * } }$ , whose $\ell _ { \infty } .$ -distance is at least $\Delta _ { O } > 0$ . Hence, for sufficiently large n that exceeds a polynomial threshold in terms of the first nine quantities in (5),

$$
\Pi _ { 0 } = \Pi _ { 1 } = : \Pi .
$$

Define

$$
{ \boldsymbol { P } } : = \Pi ^ { \top } .
$$

Then

$$
O _ { \hat { \varphi } } P = O _ { \varphi ^ { * } } + O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) , \qquad \widehat { B } P = B + O \left( \sqrt { H _ { n } ( \delta ) } \right) ,\tag{66}
$$

and

$$
\widehat { A } _ { 1 } P = A _ { 1 } + O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) , \qquad \widehat { C } _ { 1 } ^ { \top } P = C _ { 1 } ^ { \top } + O \left( \sqrt { H _ { n } ( \delta ) } \right) .\tag{67}
$$

7. Recover the transition kernel. Transposing the second relation in (67) and left-multiplying by $P ,$

$$
\widehat { C } _ { 1 } = P C _ { 1 } + O \left( \sqrt { H _ { n } ( \delta ) } \right) .\tag{68}
$$

For each $o \in \Omega$ , the corresponding block of $C _ { 1 }$ is

$$
C _ { 1 , o } = T _ { 1 } D _ { o } B ^ { \intercal } ,
$$

whereas

$$
\widehat { C } _ { 1 , o } = \widehat { T } _ { 1 } \widehat { D } _ { o } \widehat { B } ^ { \intercal } .
$$

From (66) we get:

$$
\widehat { B } ^ { \intercal } = P B ^ { \intercal } + O \left( \sqrt { H _ { n } ( \delta ) } \right) ,\tag{69}
$$

and

$$
O _ { \hat { \varphi } } = O _ { \varphi ^ { * } } P ^ { \top } + O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) .\tag{70}
$$

(70) further implies

$$
\widehat { D } _ { o } = P D _ { o } P ^ { \top } + O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) .\tag{71}
$$

Thus, from (68),

$$
\begin{array} { r } { \widehat { T } _ { 1 } \widehat { D } _ { o } \widehat { B } ^ { \intercal } = P T _ { 1 } D _ { o } B ^ { \intercal } + O \left( \sqrt { H _ { n } ( \delta ) } \right) . } \end{array}
$$

Combining (69) and (71) gives us

$$
\widehat { D } _ { o } \widehat { B } ^ { \intercal } = P D _ { o } B ^ { \intercal } + O \left( \sqrt { H _ { n } ( \delta ) } \right) .
$$

Therefore,

$$
( \widehat { T } _ { 1 } P - P T _ { 1 } ) D _ { o } B ^ { \top } = O \left( \sqrt { H _ { n } ( \delta ) } \right) .
$$

Summing over o and using $\begin{array} { r } { \sum _ { o } D _ { o } = I , } \end{array}$

$$
\left( \widehat { T } _ { 1 } P - P T _ { 1 } \right) B ^ { \intercal } = O \left( \sqrt { H _ { n } ( \delta ) } \right) .
$$

Since $\sigma _ { \operatorname* { m i n } } ( B ) = \Omega ( { \sqrt { n } } )$ , right-inverting $B ^ { \top }$ and substituting back the original notations yield

$$
T _ { \hat { \beta } } ^ { a _ { 1 } } P - P T _ { \beta ^ { * } } ^ { a _ { 1 } } = O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) .
$$

Equivalently,

$$
P ^ { \top } T _ { \hat { \beta } } ^ { a _ { 1 } } P = T _ { \beta ^ { * } } ^ { a _ { 1 } } + O \left( \sqrt { \frac { H _ { n } ( \delta ) } { n } } \right) .
$$

By Assumption 3.18, for every $a _ { 1 }$ , there exists $a _ { 2 }$ and w such that $\bar { B } _ { \tilde { a } , w } ^ { \mathrm { b l k } }$ is $\Theta ( n )$ , where $\tilde { a } = ( a _ { 1 } , a _ { 2 } )$ Thus every action can be placed in the $a _ { 1 }$ -position. Repeating the argument for all such pairs recovers every transition matrix under the same permutation $\bar { P } .$ The permutation is common across all pairs because every application aligns the same observation kernel $O _ { \varphi ^ { * } }$ , and the separation $\Delta _ { O } > 0$ forces the permutations to coincide.

Note that the implicit constants in $O ( \sqrt { H _ { n } ( \delta ) / n } )$ depend polynomially on the first nine quantities in (5). The dependence on the first seventh arises from Theorem ${ \bf 5 . 1 }$ , while the dependence on the eighth, $| \Omega |$ , comes from converting Frobenius-norm bounds into entrywise $\ell _ { 1 }$ -norm bounds.

This complete the proof of Theorem 3.29.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state our main contribution: a finite-sample, candidate-agnostic translation from likelihood gap to parameter recovery error for both undercomplete and overcomplete POMDPs. This result is established in Section 3.1 for the undercomplete regime and in Section 3.2 for the overcomplete regime.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discussed the limitations of this work in Section 5. In particular, our current analysis imposes a full-rank assumption on the transition kernels in both the undercomplete and overcomplete regimes. Developing weaker, regime-specific assumptions would lead to a more refined theory. As noted in Remark 3.10, the current transition-kernel assumption can be relaxed.

We also acknowledge in Secion 5 and Remark 3.23 that Assumptions 3.18 and 3.22 are convenient sufficient conditions on the behavior policy in the overcomplete regime, chosen to enable a direct extension of the undercomplete analysis. Characterizing the minimal behavior-policy conditions under which likelihood-gap control still implies parameter recovery in the overcomplete regime remains an important direction for future work.

## Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Both Section 3.1 and Section 3.2 explicitly introduce the kernel identifiability assumptions, policy coverage assumptions, and the ergodicity assumptions needed for all the results.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [N/A]

Justification: The paper is theoretical and does not report experimental results. Its main claims and conclusions are established through formal statements and proofs rather than computational experiments, so there are no experimental results to reproduce.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [N/A]

Justification: The paper does not report computational experiments or empirical results. Consequently, there are no experimental datasets, code, or reproduction instructions to provide in the supplemental material.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [N/A]

Justification: The paper is theoretical and does not involve model training, test sets, data splits, hyperparameter tuning, or optimizer choices. The results are formal theorem statements and proofs, so these experimental details are not applicable.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

## Answer: [N/A]

Justification: The paper is theoretical and does not report computational experiments or empirical results. Therefore, error bars, statistical significance analyses, and other uncertainty estimates for experiments are not applicable.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [N/A]

Justification: The paper is theoretical and does not include computational experiments. Therefore, there are no compute resources, memory requirements, or execution times needed to reproduce experimental results.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The paper involves no human subjects or participants, no data collection, and no computational experiments. Since the work is purely theoretical, we are not aware of any specific ethical concerns raised by the paper.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: The paper is foundational and theoretical in nature, and is not focused on a specific application domain. We therefore do not identify direct societal impacts specific to this work.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper is theoretical and does not release data, trained models, pretrained models, or scraped datasets. Therefore, safeguards for the responsible release of high-risk artifacts are not applicable.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [N/A]

Justification: The paper is theoretical and does not use external assets such as code, datasets, or trained models. Prior scholarly work is cited in the usual way through the bibliography, but there are no external assets whose licenses or terms of use are applicable.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The paper does not introduce new assets such as datasets, code, trained models, or benchmarks. Therefore, there are no accompanying assets to document.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve human subjects or crowdsourcing, so there are no participant instructions or compensation details to report.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing or human-subjects research. Therefore, there are no study-participant risks, risk disclosures, or IRB approvals to report.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: LLMs are not used as part of the research methodology. The main results are theoretical, and the assumptions, theorem statements, and proofs were written and verified by the authors. Any use of LLM tools was limited to writing, editing, and formatting.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.