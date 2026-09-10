# Likelihood-free inference with nuisance parameters through normalizing flows

Phil Assheton<sup>1\*</sup>

<sup>1\*</sup>statsadvice.com, Berlin, Germany.

Corresponding author(s). E-mail(s): phil@statsadvice.com;

## Abstract

We present a simple decomposition of a neural-network-based normalizing flow that naturally uncovers a pivotal statistic (or something close) in the presence of nuisance parameters, based only on a sample generator from the distribution of interest. We show that the statistic is near-pivotal in the sense of minimum average KL-divergence of its p-values versus uniform and we argue that it can be expected to have good power when the dimension of the statistic equals the dimension of the parameter. It is able to incorporate prior knowledge about group invariances such as translation and scale. It can discover the one-sample t-test almost exactly, outperforms the Welch test in terms of worst-case size over a constrained variance-ratio range and achieves good calibration on partial biserial correlations, while showing higher power (and being much faster) on small-to-moderate samples than profile likelihood-ratio techniques.

Keywords: pivotal statistic, nuisance parameters, likelihood-free inference, simulator-based inference

## 1 Introduction

## 1.1 Context

Frequentist inference in the presence of nuisance parameters ideally seeks a “similar test”: one whose size is invariant across all possible configurations of the nuisance parameters. In many specific cases, clever tricks have been found to compute pivotal statistics from the sample, whose distribution does not vary with the nuisance parameter(s). A well-known example of this is the t-test, where the goal is to perform inference on the mean $\mu$ of a normal distribution, regardless of its variance, $\sigma ^ { 2 } ;$ a tstatistic is computed, which efectively cancels the efect of σ out, leaving a statistic that is pivotal and a test that is similar across all values of $\sigma .$

On the other hand, a number of more “general” approaches do exist, whose goal is to be able to perform inference across a broad range of parametric parent distributions. Owing to the complexity of the problem of finding a similar test across an entire space of nuisance parameters, these each rely on diferent constraints or approximations.

The profile likelihood ratio (LR) is the basis for many of these. The profile LR compares the likelihood at the maximum likelihood estimate (MLE) to that at a nullconstrained MLE, where the interest parameter is constrained to equal some null value. In the large sample limit, it can be shown that −2 log LR converges in distribution to a $\chi _ { 1 } ^ { 2 }$ distribution (Wilks 1938). That is, its limiting distribution is, to first order, independent of nuisance-parameter values.

An alternative general approach is the parametric bootstrap (Davison and Hinkley 1997), in which many samples are drawn from a fitted model, and used to approximate the sampling distribution. One may, for example, draw many datasets from a model using the MLE as its parameters, and map each dataset to a given scalar statistic; the spread of these bootstrap statistics can be used to estimate the standard error of that statistic. Alternatively, sampling from a model with parameter set to the null-constrained MLE can be used to generate a p-value against a given null hypothesis.

One popular statistic to bootstrap is the profile LR, since it is known to be highly informative about the interest parameter and also to have a limiting distribution that is pivotal. This efectively combines the bootstrap and LR approaches: using the powerful and asymptotically-pivotal LR, without relying on the large-sample assumption required for the $\chi ^ { 2 }$ approximation.

Nevertheless, the bootstrap is still only an approximation: the bootstrap distribution is drawn from the MLE, rather than the true population parameter. Higher order refinements and iterating bootstraps-within-bootstraps can improve accuracy in some settings, but at the cost of substantially increased computation (Hall and Martin 1988).

In recent years, there has been growing interest in applying machine learning techniques to this problem. A deep neural network (DNN) is an extremely flexible, diferentiable function approximator that can be quite eficiently trained to optimize some loss function. For example, Coccaro et al. (2020) use DNNs as surrogates for complex likelihood functions, so that they may be used in LR or Bayesian approaches, among other possibilities.

Cranmer et al. (2015) show that a neural net trained by cross-entropy to distinguish between samples generated under two diferent parameter values encodes (up to a transformation) the LR statistic, without needing to model the likelihood directly. Instead it is trained on samples drawn from the model of interest; in many real-world cases, such a simulator is available where an explicit likelihood function is not. In the presence of nuisance parameters, the same classifier-based ratio idea can be embedded in a profile-LR construction. Heinrich (2022) trains a model to distinguish data from any given null vs data from any of a mixture of alternatives, all at an equal Fishermetric distance from the given null. He notes that this should target the test statistic with best average power (for this mixture) and show that it recovers the profile LR in cases where that is known to be most powerful.

For the final step from statistic to p-value, Dalmasso et al. (2024) present a framework in which learned test statistics (such as their classifier-based LR-approximator called ACORE) are mapped to critical values (for confidence intervals/regions) by quantile regression and p-values by a probabilistic binary classifier. They refer to this framework as “Likelihood-free Frequentist Inference” (LF2I), again stressing the broader range of applications where no likelihood function is required; only a simulator. Al Kadhim et al. (2024) take this a step further to model the entire curve of critical values by training a binary classifier to model the CDF of the test statistic given the parameter. More specifically, they train the net by least squares to model the expectation of a binary indicator on whether samples are greater or less than a given threshold.

Particularly relevant to our work, however, is that of Louppe et al. (2017). They, like us, attempt to train directly a neural net that will, by its design, identify a pivotal statistic. They do this by adapting the concept of generative adversarial networks (GANs) from Goodfellow et al. (2014). A secondary “adversary” net attempts to predict, by maximum likelihood, the values of the nuisance parameters from the output of the pivot-learning “classifier” net; the latter, in turn, aims to minimize the same predictive log likelihood, while simultaneously maximizing the log likelihood of a target label given the observed data. In the context of frequentist statistical inference, this target label could encode the hypothesis or parameter value of interest. By optimizing both objectives, the net uncovers a statistic that is pivotal with respect to the nuisance parameters while remaining optimally predictive of the target label.

Training a neural network can be computationally expensive. However, in many cases, once fit, the same network can be reused over and over again very cheaply, efectively paying of the initial investment. For this reason, such approaches are often referred to as “amortized” (Zammit-Mangion et al. 2025).

## 1.2 Introducing NeuralCIs

In the following, we will introduce the first phase of a project called NeuralCIs, whose ultimate goal will be to use neural models to generate confidence intervals and regions. In this first phase, we focus on the generation of p-values in the presence of nuisance parameters, but these are expected to be quite readily converted into confidence intervals by a further training step (see Section 8.2.1) and we hope will be generalizable to simultaneous confidence intervals and confidence regions over vectorvalued interest parameter. The code for the project can be followed on GitHub at github.com/philassheton/neuralcis.

The idea is to train a single neural net that can solve an entire class of problems; once trained (this currently takes two to four hours on a modest RTX 3060 laptop GPU), the net can very quickly convert any dataset into a p-value (and ultimately confidence interval) against any of a range of values for the interest parameter; that is, the initial training cost is amortized.

Using a neural model in this way feels quite natural. To generate a single p-value one must anyway efectively compare the one dataset of interest to all datasets that are probable under one null hypothesis; to generate a confidence interval extends this search out to many “null hypotheses”. Whereas bootstrapping would start afresh for each new dataset, fitting a single neural net to model all possible outcomes at all of a wide range of parameters means that that broad “search” across many samples from many parameter values can be shared and reused each time a new dataset arrives that fits a given problem; that initial cost is amortized.

Furthermore, neural networks are quite eficiently diferentiable, which makes them great for optimisation; we will take advantage of this in Section 8.2.1 to propose a method to extend our p-values quite cleanly to confidence intervals.

Our work adapts the idea of the normalizing flow (Papamakarios et al. (2021)), which we will summarize in Section 2. The key novel contribution is a decomposition of the normalizing flow that automatically yields an on-average near-pivotal statistic (Section 3). We show that this statistic is near-pivotal in the sense that p-values derived from it have minimal average KL-divergence to uniform. We further argue that it can be expected to be reasonably powerful by an implicit bias argument that becomes more concrete for transformation models at one extreme and in the large sample local asymptotic normal (LAN) limit at the other. We assess both the calibration and power through simulation.

In Section 2, we introduce the normalizing flow, how we implement it in NeuralCIs (somewhat diferent than normal), and how we can flip its input and output roles to provide amortized sampling from an unnormalized target density. In Section 3, we detail the core decomposition that we will use to identify near-pivotal quantities, and how that can be used to generate p-values. In Section $^ { 4 , }$ we explore the formal statistical properties of this decomposition, in particular, its expected false-positive rate behaviour as well as an implicit bias argument that it might be expected in many practically useful contexts to achieve good true-positive rates. In Section 5, we outline further technical points central to our architecture (invariances, adding known values such as sample size, parameter sampling and neural network design). In Section 6, we present experimental methodology on one toy and three classical statistical problems, whose results we present in Section 7 and discuss in Section 8. A final conclusion is in Section 9.

## 2 Normalizing flows

Normalizing flows are highly flexible probability density models that can approximate almost any continuous density encountered in practice on $\mathbb { R } ^ { d }$ . The idea is very simple: use a neural network to model an invertible change of variables, such that the distribution in the new variables is a simple one, commonly a standard normal.

## 2.1 Core concept

We will abbreviate $\mathcal { N } ( \mathbf { z } ) : = \mathcal { N } ( \mathbf { z } ; \mathbf { 0 } , \mathbf { I } )$ and write a neural network mapping $\mathbf x \mapsto \mathbf z$ as $\mathbf { z } = f _ { \phi } ( \mathbf { x } )$ . Denote the true distribution of the inputs x as $p _ { \mathbf { x } } ( \mathbf { x } )$ and of the net outputs z as $p _ { \mathbf { z } } ( \mathbf { z } )$ . We optimize the weights $\phi$ of the network to bend samples from the true distribution $p _ { \mathbf { z } } ( \mathbf { z } )$ as close as possible in distribution to the theoretical target distribution $q _ { \mathbf { z } } ( \mathbf { z } ) : = \mathcal { N } ( \mathbf { z } )$ . This induces a theoretical approximate distribution $q _ { \mathbf { x } } ( \mathbf { x } )$

on x:

$$
q _ { \mathbf { x } } ( \mathbf { x } ) = q _ { \mathbf { z } } ( \mathbf { z } ) \left| \operatorname* { d e t } \frac { \partial \mathbf { z } } { \partial \mathbf { x } } \right| = \mathcal { N } ( f _ { \phi } ( \mathbf { x } ) ) \left| \operatorname* { d e t } \frac { \partial f _ { \phi } ( \mathbf { x } ) } { \partial \mathbf { x } } \right| ,
$$

whose fit we can optimize by minimizing the sum of the negative log likelihood across a large number of samples $\mathbf { x } \sim p _ { \mathbf { x } } ( \mathbf { x } )$

$$
- \log q _ { \mathbf { x } } ( \mathbf { x } ) = - \log \left( q _ { \mathbf { z } } ( \mathbf { z } ) \left| \operatorname* { d e t } \frac { \partial \mathbf { z } } { \partial \mathbf { x } } \right| \right) = - \log \mathcal { N } \left( f _ { \phi } ( \mathbf { x } ) \right) - \log \left| \operatorname* { d e t } \frac { \partial f _ { \phi } ( \mathbf { x } ) } { \partial \mathbf { x } } \right| .\tag{1}
$$

This maximization of $q _ { \mathbf { x } } ( \mathbf { x } )$ across samples from $p _ { \mathbf { x } } ( \mathbf { x } )$ is in the large-sample limit equivalent to minimizing the Kullback-Leibler (KL) divergence $D _ { \mathrm { K L } } ( p _ { \mathbf { x } } \Vert q _ { \mathbf { x } } ) \equiv$ $D _ { \mathrm { K L } } ( p _ { \mathbf { z } } \| \mathcal { N } )$ (Papamakarios et al. 2021).

## 2.2 Invertibility and tractability

In order to have tractable Jacobian determinants, a typical normalizing flow constrains each network layer to have a trivial Jacobian determinant (for example, by designing each layer to have a triangular Jacobian matrix), and to be invertible by construction.

We instead compute Jacobians by back-propagation (chain-rule backward through the net) and enforce approximate invertibility by punishing zero and negative Jacobian determinant. The latter is achieved by replacing the absolute value |j| of the Jacobian determinant j in (1) with max $( j , \epsilon \sigma ( j ) )$ , with $\begin{array} { r } { \sigma ( j ) : = \frac { 1 } { 1 + e ^ { - j } } } \end{array}$ and $\epsilon = 1 0 ^ { - 1 0 }$ chosen to be small, without being small enough to erase the gradients of σ below numerical precision.

This approach is quite computationally expensive $( \mathcal { O } ( n _ { \mathbf { x } } ^ { 3 } )$ in the number of statistics $n _ { \mathbf { x } } )$ but our flows still train in a few hours on a very modest laptop GPU (RTX 3060 mobile). We tolerate this extra cost because it afords us incredible flexibility in terms of introducing conditioning variables (Section 2.3) and decomposing our flow (Section 3), without the need to experiment with exotic constrained architectures. The key here is that the traditional normalizing flow (with those architectural constraints described above) is designed to work well at very high dimensions. Ours is instead (at present) designed for high flexibility with relatively low-dimensional problems.

We may in later iterations of the project experiment with higher dimensional problems. Apart from such architectural constraints, one particularly promising approach is ofered in recent work by Draxler et al. (2024). They propose an approximation method for fitting free-form flows that is instead $\mathcal { O } ( n _ { \mathbf { x } } )$ per data-point. Furthermore, they are able to target a global difeomorphism via an extra inverse flow whose role is to reconstruct the input. (Our approach can only at present target a local difeomorphism, which does leave scope for dysfunctional flows to be fitted, for example, by wrapping coordinates around a circular path back onto themselves.)

## 2.3 Conditional Normalizing Flows

With this more na¨ıve approach, we can very easily make our normalizing flow model a conditional probability density $p ( \mathbf { x } | \mathbf { \boldsymbol { \theta } } )$ , by simply adding θ as extra inputs to the network. This allows it to fit a diferent mapping at each value of $\theta ,$ each representing that particular conditional “slice” of distribution. Training proceeds as before, but with the extra θ inputs playing no role in the Jacobian $\overline { { \frac { \partial \mathbf { z } } { \partial \mathbf { x } } } }$ computed in the loss function (1).

## 2.4 “Denormalizing flows” for amortized sampling

During training, we will need to generate a huge number of θ samples, which we hope to draw approximately according to a Jefreys’ prior, $p ( \pmb \theta ) \propto \sqrt { \operatorname* { d e t } ( I ( \pmb \theta ) ) }$ (further details in Section 5.3). We can invert the logic of a traditional normalizing flow to train a model that, once fitted, can rapidly draw an unlimited supply of samples from such a diferentiable unnormalized target density (Rezende and Mohamed 2015; Papamakarios et al. 2021).

We will again use $p _ { \mathbf { z } }$ and $p _ { \mathbf { x } }$ to refer to the true distribution of our z and x samples and $q _ { \mathbf { z } } , \ q _ { \mathbf { x } }$ to mean some theoretical target distribution that we want to bend them toward. The key intuition here is that we reverse all the roles, compared with the core normalizing flow concept described in Section 2.1. The samples we feed into the net are standard normal $\mathbf { z } \sim p _ { \mathbf { z } } : = \mathcal { N } ( 0 , 1 )$ , and the outputs will aim to be distributed according to some distribution $\mathbf { x } \sim q _ { \mathbf { x } } : = c \bar { q } _ { \mathbf { x } }$ , where we only know $q _ { \mathbf { x } } ( \mathbf { x } )$ up to some multiplicative constant c. We aim then to learn the mapping $\mathbf z \mapsto \mathbf x$

We rewrite (1) as

$$
- \log q _ { \mathbf { z } } ( \mathbf { z } ) = - \log \left( c { \bar { q } } _ { \mathbf { x } } ( \mathbf { x } ) \left| \operatorname* { d e t } { \frac { \partial \mathbf { x } } { \partial \mathbf { z } } } \right| \right) = - \log { \bar { q } } _ { \mathbf { x } } ( f _ { \phi } ( \mathbf { z } ) ) - \log \left| \operatorname* { d e t } { \frac { \partial \mathbf { x } } { \partial \mathbf { z } } } \right| - \log c ,
$$

and we may discard the constant log c term without afecting the optimal value for $\phi .$ As with the regular normalizing flow, maximizing $q _ { \mathbf { z } } ( \mathbf { z } )$ across a large enough sample of $\mathbf { z } \sim p ( \mathbf { z } )$ is equivalent to minimizing $D _ { \mathrm { K L } } ( p _ { \mathbf { z } } , q _ { \mathbf { z } } ) \equiv D _ { \mathrm { K L } } ( p _ { \mathbf { x } } , q _ { \mathbf { x } } )$ . Intuitively, the determinant encourages the outputs x to spread out and explore the region of support of $q _ { \mathbf { x } } .$ , while the $\bar { q } _ { \bf x }$ component encourages them to cluster wherever $\bar { q } _ { \mathbf { x } }$ is high.

In the literature, this is still viewed as a normalizing flow, but $f i t$ by minimizing reverse KL divergence. As a shorthand in this paper, we will refer to this architecture as a “denormalizing flow”, to signify that it maps samples from a normal distribution to the target, rather than vice versa. This name distinguishes it from the form described in Section 2.1, which forms the basis of most of the work in this paper; the “denormalizing flow” is used only for parameter sampling during training.

## 3 Decomposing the normalizing flow

The key novel contribution of this paper is a very simple decomposition of the normalizing flow that naturally uncovers pivotal quantities and their distributions.

Suppose that the parameter θ has dimension $n _ { \theta }$ with scalar interest parameter $\psi ( \pmb \theta )$ ; all other dimensions of $\pmb { \theta }$ are to be considered nuisance parameters. We constrain ourselves here to the case where the dimension $n _ { \mathbf { x } }$ of our dataset summary x is the same as that of $\theta ;$ that is, $n _ { \bf x } = n _ { \theta }$ . We hope to explore the extension to $n _ { \mathbf { x } } \geq n _ { \theta }$ in future work (see Section 8.2). A pure normalizing flow to describe the distribution of x, given parameter θ is simply a neural net mapping $f _ { \phi } : \mathbb { R } ^ { n _ { \mathbf { x } } + n _ { \theta } } \mapsto \mathbb { R } ^ { n _ { \mathbf { \lambda } } }$ <sup>x</sup> , with the Jacobian determinant computed only versus the first $n _ { \mathbf { x } }$ inputs.

We propose to decompose this into two separate neural nets, a pivoting net $z _ { \mathrm { p } }$ and a nuisance net $\mathbf { z } _ { \mathrm { n } }$

$$
\begin{array} { r l } & { z _ { \mathrm { p } } ( \mathbf { x } ; \boldsymbol { \psi } ( \pmb { \theta } ) ) : \mathbb { R } ^ { n _ { \mathbf { x } } + 1 }  \mathbb { R } , } \\ & { \quad \quad \mathbf { z } _ { \mathrm { n } } ( \mathbf { x } ; \pmb { \theta } ) : \mathbb { R } ^ { n _ { \mathbf { x } } + n _ { \pmb { \theta } } }  \mathbb { R } ^ { n _ { \mathbf { x } } - 1 } . } \end{array}\tag{2}
$$

The key point here is that the first net $z _ { \mathrm { p } }$ is not aware of all parameters $\theta ,$ but only of the scalar interest function of them $\psi ( \pmb \theta )$ . The two outputs are concatenated,

$$
\mathbf { z } = { \binom { z _ { \mathrm { p } } } { \mathbf { z } _ { \mathrm { n } } } } \ ,
$$

so that $\mathbf { z } \in \mathbb { R } ^ { n _ { \mathbf { x } } }$

From here, the same normalizing flow training outlined in Section 2 is followed, to coerce z to follow a standard normal distribution, by maximizing the sum across (1). Under a few assumptions about the nature of $\psi$ and of the broader sampling distribution, $Z _ { \mathrm { p } } | \theta$ will be, on average across the training distribution of $\theta _ { \mathrm { { i } } }$ as close as possible to standard normal (in terms of KL-divergence, see Section 4), while the mapping $z _ { \mathrm { p } }$ required to generate it requires knowledge of θ only through $\psi ( \pmb \theta )$ .

## 3.1 One further constraint

We also add one further constraint,

$$
\frac { \partial z _ { \mathrm { p } } } { \partial \psi } < 0 .\tag{3}
$$

This constraint helps to prevent $z _ { \mathrm { p } }$ from flipping orientation and $. / \mathrm { o r }$ folding with respect to $\psi$ (which would otherwise be technically possible, as long as $\mathbf { z } _ { \mathrm { n } }$ compensates to keep the entire mapping a difeomorphism). It also helps to encourage a clean alignment of $z _ { \mathrm { p } }$ with the underlying nuisance-free component of the ψ-action on the data (see, for example, Appendix C.2.2), thereby improving our argument that our architecture should provide reasonable power.

This constraint is enforced in our architecture by adding to the loss

$$
\kappa \operatorname* { m a x } ( \partial z _ { \mathrm { p } } / \partial \psi , 0 ) ,
$$

with $\kappa = 1 0 0$ chosen large enough to have a large impact at small positive $\partial z _ { \mathrm { p } } / \partial \psi$ without causing numerical overflows.

## 3.2 Obtaining p-values from $z _ { \mathbf { p } }$

We convert $z _ { \mathrm { p } }$ to a one-tailed p-value v by the standard normal CDF, $v = \Phi ( z _ { \mathrm { p } } )$ . In Section 4.2, we show that (under certain assumptions) $z _ { \mathrm { p } }$ will be (on average across the training distribution of $\theta )$ as close to standard normal as the model permits (minimal in KL-divergence from normal). Equivalently, the average KL divergence between v and uniform is minimized. These are then converted to equal-tailed two-tailed p-values by $2 \operatorname* { m i n } ( v , 1 - v )$

## 3.3 An example: the one-sample t-test

As a simple example, we might want to reproduce the one-sample t-test. In this case, our parameters $\pmb \theta = ( \mu , \sigma )$ represent the population mean and standard deviation, with interest parameter $\boldsymbol { \psi } ( \pmb { \theta } ) = \boldsymbol { \mu }$ . Our statistics would be the sample mean and standard deviation $\mathbf { x } = ( m , s )$ . We would train simultaneously

$$
\begin{array} { r } { z _ { \mathrm { p } } ( m , s , \mu ) : \mathbb { R } ^ { 3 } \mapsto \mathbb { R } , } \\ { \mathbf { z } _ { \mathrm { n } } ( m , s , \mu , \sigma ) : \mathbb { R } ^ { 4 } \mapsto \mathbb { R } ^ { 1 } , } \end{array}
$$

and in our loss function, $\begin{array} { r } { \frac { \partial \mathbf { z } } { \partial \mathbf { x } } = \frac { \partial ( z _ { \mathrm { p } } , \mathbf { z } _ { \mathrm { n } } ) } { \partial ( m , s ) } } \end{array}$

We may then obtain a one-tailed p-value v for a given sample $( m , s )$ against null hypothesis $H _ { 0 } : \mu = \mu _ { 0 }$ , by $v = \Phi ( z _ { \mathrm { p } } ( m , s , \mu _ { 0 } ) )$ and a two-tailed p-value by 2 min $( v , 1 -$ v).

## 4 Formal properties

## 4.1 Assumptions

Let X|θ denote the random vector of statistics used by the method, with realized value $\mathbf { x } \in \mathbb { R } ^ { n _ { \mathbf { x } } }$ . We train our model over a large set of parameter values $\pmb { \theta } \in \Theta _ { \mathrm { o u t e r } } \subseteq \mathbb { R } ^ { n _ { \theta } }$ and make inferential claims on a smaller region $\pmb { \theta } \in \Theta _ { \mathrm { i n n e r } } \subseteq \Theta _ { \mathrm { o u t e r } }$ (see Section 5.3). All assumptions below are assumed to hold across $\Theta _ { \mathrm { o u t e r } } .$

## 4.1.1 Suitable models

We constrain our attention to a certain subset of models, for which we believe this architecture in its current form is most promising. We first assume that the statistic X has the same dimensionality as the parameter $\theta _ { \mathrm { { ; } } }$ , that is $n _ { \bf x } = n _ { \theta }$

We further assume that the interest parameter $\psi ( \pmb \theta )$ is a clean, unfolded coordinate in $\theta ;$ concretely that $\pmb \theta$ can be smoothly and one-to-one reparameterized as

$$
{ \pmb \theta }  ( { \pmb \psi } , { \pmb \lambda } ) , \qquad { \pmb \lambda } \in \mathbb { R } ^ { n _ { \pmb \theta } - 1 } ,
$$

and in this section we write $\pmb \theta = ( \psi , \lambda )$ , with $\psi = \psi ( \pmb \theta )$ .

In particular, we developed our architecture with models in mind of the form

$$
{ \bf X } = { \pmb \theta } + { \bf E } _ { \pmb \theta } , \qquad { \pmb \theta } = ( \psi , \pmb \lambda ) \in \mathbb { R } \times \mathbb { R } ^ { n _ { \pmb \theta } - 1 } ,
$$

for error $\mathbf { E } _ { \theta }$ that is relatively concentrated and varies smoothly, and slowly, in θ. We do not at present restrict ourselves exclusively to these models, but our arguments regarding power in Section 4.3 in particular will be more easily understood in this context.

In particular, we will assume not only that $n _ { \bf x } = n _ { \theta }$ , but also that the transport of the density $p ( \mathbf { x } | \mathbf { \boldsymbol { \theta } } )$ , induced by perturbations of θ, is full-rank. That is to say, at each θ, perturbing each dimension of θ transports the probability mass in a direction linearly independent of that induced by perturbing any other dimension. As shorthand, we will refer to such dimension-matched and full-rank-transported models as “full-rank models”.

## 4.1.2 Standard regularity assumptions

We assume that X|θ has a smooth density $p ( \mathbf { x } | \pmb { \theta } ) ~ \pmb { \theta } \in \Theta _ { \mathrm { o u t e r } }$ . We work on a single regular coordinate patch of the model, excluding regions where the model becomes locally singular, folds back on itself, or wraps around so that the same local model is represented by multiple parameter values.

Denoting the log likelihood by $\ell ( \pmb \theta ; \mathbf x ) = \log p ( \mathbf x | \pmb \theta )$ , we assume that the score $s ( \pmb \theta ; \mathbf x ) = \nabla _ { \pmb \theta } \ell ( \pmb \theta ; \mathbf x )$ , exists with finite variance. The Fisher information, $I ( \pmb \theta ) \ =$ $\operatorname { V a r } _ { \pmb { \theta } } s ( \pmb { \theta } ; \mathbf { X } )$ , is assumed to exist and have full rank throughout the region considered.

For the formal arguments below, we will additionally assume that the learned transformation is one-to-one on the relevant support (and defines a valid change of variables). Our present implementation encourages local invertibility through the Jacobian penalty, but does not guarantee global invertibility, as discussed in Section 2.2.

## 4.2 Near-uniformity (by KL) of p-values (type I error)

In the limit of infinite samples, maximizing the likelihood of our constrained normalizing flow has the efect of minimizing the Kullback-Leibler (KL) divergence between our z and the normal distribution ${ \mathcal { N } } _ { : }$ , averaged across the distribution used to sample θ during training.

Let X be the random variable that instantiates into individual samples x and

$$
\begin{array} { r } { Z _ { \mathrm { p } } = z _ { \mathrm { p } } ( \mathbf { X } ) , } \\ { \mathbf { Z } _ { \mathrm { n } } = \mathbf { z } _ { \mathrm { n } } ( \mathbf { X } ) } \end{array}
$$

and let $\mathcal { N } _ { d } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ represent a d-dimensional standard normal variable.

By the Kullback-Leibler chain rule, we have:

$$
\mathbb { E } _ { \pmb { \theta } } D _ { \mathrm { K L } } ( ( Z _ { \mathrm { p } } , \mathbf { Z } _ { \mathrm { n } } ) \| \mathcal { N } _ { \mathfrak { n } _ { \mathbf { x } } } ) = \mathbb { E } _ { \pmb { \theta } } D _ { \mathrm { K L } } ( Z _ { \mathrm { p } } \| \mathcal { N } _ { 1 } ) + \mathbb { E } _ { \pmb { \theta } , Z _ { \mathrm { p } } } D _ { \mathrm { K L } } ( \mathbf { Z } _ { \mathrm { n } } | Z _ { \mathrm { p } } \| \mathcal { N } _ { \mathfrak { n } _ { \mathbf { x } } - 1 } ) .
$$

Assuming that $\mathbf { z } _ { \mathrm { n } }$ is suficiently flexible to Gaussianize the conditional distribution $\mathbf { Z } _ { \mathrm { n } } | Z _ { \mathrm { p } }$ , the final term here can be made (approximately) zero. In this case, fitting our model, finds the model that minimizes the first term: $\mathbb { E } _ { \pmb { \theta } } D _ { \mathrm { K L } } ( Z _ { \mathrm { p } } \Vert \mathcal { N } _ { 1 } )$ . (Of course, our neural network is probably not suficiently flexible to make the final term exactly zero, but we assume it to be suficiently flexible to make it negligible.)

Furthermore, KL divergence is invariant under a bijective change of variables, such as conversion of our standard normal $Z _ { \mathrm { p } }$ into a one-tailed p-value, $V = \Phi ( Z _ { \mathrm { p } } )$ . This means that our p-values are optimized to have a minimum average KL divergence,

$$
\mathbb { E } _ { \pmb { \theta } } D _ { \mathrm { K L } } ( V \Vert U ) ,\tag{4}
$$

versus a uniform distribution, U. (Note that we are using the random variable V here to refer to a p-value to avoid confusion with our densities p and $q . \mathrm { ~ , ~ }$

## 4.2.1 Suitability of KL optimization

If there truly exists a perfect pivot, we expect our KL divergence to be minimized at the point where $z _ { \mathrm { p } }$ is indeed pivotal. However, in the case where one does not exist, the model must make trade-ofs. In this case KL is a somewhat unnatural loss function. Minimizing KL divergence does not focus on the sort of tail-specific calibration, or worst-case guarantees that we would ideally want for p-values.

Nevertheless, it has some nice properties in that direction. With a uniform target $U ,$ we have $\begin{array} { r } { D _ { \mathrm { K L } } ( V \| U ) = \int _ { 0 } ^ { 1 } p ( v ) \log p ( v ) \mathrm { d } v ; } \end{array}$ for V close to uniform, we can expand $p \log p$ around $p = 1$

$$
p \log p = ( p - 1 ) + \frac { ( p - 1 ) ^ { 2 } } { 2 } - \frac { ( p - 1 ) ^ { 3 } } { 6 } + \cdots ,
$$

and given that $\begin{array} { r } { \int _ { 0 } ^ { 1 } ( p ( v ) - 1 ) \mathrm { d } v = 0 } \end{array}$ when $V \in [ 0 , 1 ]$ , we have

$$
D _ { \mathrm { K L } } ( V \Vert U ) \approx \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \left( p ( v ) - 1 \right) ^ { 2 } \mathrm { d } v = \frac { 1 } { 2 } \Vert p ( v ) - 1 \Vert _ { 2 } ^ { 2 } .
$$

That is, when V is close to uniform, $D _ { \mathrm { K L } }$ will trade improvements at one part of the distribution for deviations at another according to the $L _ { 2 }$ distance between the true density and uniform.

When generating p-values, we are most interested in how the cumulative distributions compare; unfortunately, relatively small deviations in density can accumulate across v into relatively large discrepancies in cumulative distribution. On the other hand, our interest is particularly in comparing these cumulative distributions in the tails. In the tails, there is much less space for small discrepancies in density to accumulate; the only way to drastically inflate the error rate at $\alpha = 0 . 0 5$ is to squash a considerably increased density of p-values into the small space $V < 0 . 0 5$ , which would be detected by the Kullback-Leibler divergence.

So, while an ideal method might specifically target CDF uniformity in the tails, optimizing the Kullback-Leibler divergence should nonetheless give a result whose tail CDF is nudged into the right ballpark. Nonetheless, this is still on-average nearpivotal. It must be stressed that this leaves open a gap for severely non-pivotal behaviour at some parameter values to be traded of against consequently improved behaviour at others. This is particularly possible at low-probability parameter values, where very poor calibration may nonetheless have a very small impact on the loss. On the other hand, if a true pivot exists, we have good reason to expect that our model will find it.

## 4.3 Power (type II error)

Identifying a pivot does not in itself guarantee that that pivot contains any information whatsoever about the interest parameter ψ. Unlike the GAN approach from Louppe et al. (2017), our net makes no explicit attempt to retain ψ information in our pivot. Particularly in cases where $n _ { \mathbf { x } } > n _ { \theta }$ (but not limited to these), there may be many ancillary statistics that contain information neither about λ nor ψ. However, in the full-rank models (defined in Section 4.1.1) that we are focussed on, we believe that in general, identifying a pivot does recover a useful test statistic. We make this argument by analogy to the profile likelihood ratio (LR) and its gradient, the eficient score.

The profile LR achieves excellent power on nuisance parameter problems by starting with the most powerful simple-vs-simple test (the plain LR) and removing λ by choosing it at its optimal value. This has the efect of narrowing the LR for any parts of the data that both $\psi$ and λ could equally well explain. The result is that any contribution of $\psi$ that equally could have come from λ is absorbed away in the optimisation, leaving only the pure ψ-information behind. In our model, $\mathbf { z } _ { \mathrm { n } }$ has a similar role: it absorbs the nuisance directions, so that $z _ { \mathrm { p } }$ contains only pure ψ-information that cannot be mimicked by changing λ. In a “full-rank” model (Section 4.1.1), there are no redundant dimensions available for $z _ { \mathrm { p } }$ to inhabit other than this one.

We make this more concrete below by reference first to transformation models (in which this efect is quite clean) in Section 4.3.1, before attempting to extend this logic to a broad subset of non-transformation models, first somewhat heuristically in Section 4.3.2 and then more concretely based on asymptotics in Section 4.3.3.

## 4.3.1 Transformation models

We can make this link between $z _ { \mathrm { p } }$ and the profile LR quite tangible in transformation models, where the parameters $\pmb \theta$ and corresponding data X can be indexed over by the action of a group G with full rank (dim $G = n _ { \theta } )$ . This case covers a huge range of classical examples, such as inference on means and standard deviations of normal models, diferences of means and ratios of variances in normal models, linear regression, the ratio of two exponential rates, etc.

In Appendix C, we show how the most natural $z _ { \mathrm { p } }$ would also be (close to) profile LR suficient; that is, the profile LR would depend on X (almost) only via $z _ { \mathrm { p } } ( \mathbf { X } ; \psi )$ We argue this first by summarizing previous work showing (Appendix C.1) that in such models, the profile LR must be a function of the maximal invariant of $( { \bf x } , \psi )$ under the action of $g \in G$ . We further argue (Appendix C.2) that the most natural destination for $z _ { \mathrm { p } }$ is the same maximal invariant, with the only loss-minimizing alternative being highly globally organized deviations that

1. are neither encouraged nor discouraged by our loss function,

2. tend to average out to be very small across a random initialisation, and

3. occupy such a small volume in the space of possibilities, that random steps along these directions under stochastic gradient descent are expected to be negligible.

In Section 7.4, we present results of an experiment to demonstrate this efect. We also show in Appendix C.2.3, that small deviations of this kind would not degrade power significantly.

## 4.3.2 Near-transformation models

In general, non-transformation models have no maximal invariant to hang these arguments on. However, the core intuition behind the expected power performance in transformation models is: if $\mathbf { z } _ { \mathrm { n } }$ absorbs the nuisance directions, then what remains is the pure information about ψ, unpolluted by λ. We will argue here that the same intuition may be applied to a broad class of non-transformation models also.

At the infinitesimal level, the analogy to transformation models is strong: assuming (as we did with our transformation models) that each parameter transports the probability mass in a linearly independent direction, there is a quotient direction that contains the “pure ψ information”, after removal of any information which could have been accounted for by λ. And, since movement induced by λ produces, to first order, no displacement along this local quotient direction, it is locally pivotal with respect to λ and thereby, locally appealing to $z _ { \mathrm { p } }$

However, $g l o b a l l y , \ z _ { \mathrm { p } }$ must stitch together these local quotient directions. Transformation models are the special case where these directions are mutually consistent; each is a local view on the same global coordinate, the maximal invariant. In general models, the local quotient directions may be inconsistent with each other and the learned $z _ { \mathrm { p } }$ is there best viewed as a smooth global compromise, with its alignment to the local quotient directions depending on the regions of parameter space emphasized by the training distribution and likelihood. Formulating a condition under which these local invariants are “consistent enough” is a work in progress; such models we refer to here informally as “near-transformation models”.

As with type I error, this does again leave open the possibility for the architecture to trade poor power in one region of parameter space (particularly one with lower weighting) for improved power at another. There is no specific component to the loss that rewards an informative pivot and, as such, we will again rely on simulation of the resulting architecture, to understand its properties in this regard.

This argument can be made much more concrete as sample size increases:

## 4.3.3 The local and asymptotic case

Treating the n<sub>θ</sub>-dimensional x supplied to the architecture as the observed data, suppose that its distribution is locally asymptotically normal (LAN). This occurs, for example, when x is an eficient estimate of θ based on a growing underlying dataset. However the class of asymptotically normal estimators for which this holds is far broader.

As the underlying sample size increases, the local region of θ that is relevant for a given sample x shrinks, and on that shrinking local scale, the likelihood loses its higher-order shape, converging efectively to a Gaussian shift experiment. We explore such an experiment in Appendix C.1.1, and summarize the conclusions here, adapted as they relate to this local asymptotic normal (LAN) case:

1. Local to parameter $\pmb { \theta } _ { 0 } .$ , the likelihood function $\ell ( \pmb \theta )$ is approximately quadratic,

$$
\begin{array} { r } { - 2 ( \ell ( \pmb { \theta } ) - \hat { \ell } ) \approx ( \pmb { \theta } - \hat { \pmb { \theta } } ) ^ { \top } \mathbf { I } ( \pmb { \theta } - \hat { \pmb { \theta } } ) , \qquad \hat { \pmb { \theta } } = \pmb { \theta } _ { 0 } + \mathbf { I } ^ { - 1 } \mathbf { s } , } \end{array}
$$

where $\mathbf { s } = \mathbf { s } ( \pmb \theta _ { 0 } ; \mathbf x )$ and $\mathbf { I } = \mathbf { I } ( \theta _ { 0 } )$ are the score and Fisher information (respectively), evaluated at fixed $\pmb { \theta } _ { 0 }$ close to $\theta ,$ and $\hat { \ell } = \ell ( \hat { \pmb \theta } )$ is the constant likelihood at the peak of the quadratic.

2. Here $\hat { \pmb { \theta } } = \pmb { \theta } _ { 0 } + \mathbf { I } ( \pmb { \theta } _ { 0 } ) ^ { - 1 } \mathbf { s } ( \pmb { \theta } _ { 0 } ; \mathbf { x } )$ is the random variable in our local asymptotic Gaussian-shift experiment, $\hat { \pmb { \theta } } { \therefore \mathcal { N } ( \pmb { \theta } , \mathbf { I } ^ { - 1 } ) }$ . But it also plays the role of a simple conversion of x into a more intuitive form, which locates the θ at the turning point of the quadratic, so that it can be expressed as a perfect square. Computing $\hat { \pmb { \theta } }$ this way is also the only adaptation necessary to convert the Gaussian location experiment in Appendix C.1.1 into its LAN equivalent here.

3. Profiling this quadratic absorbs any information that can be explained by $\lambda ,$ to leave a one-dimensional quadratic profile likelihood $\ell _ { p }$ that has the same form as the marginal likelihood for ψ: it is centred at the same ψ<sup>ˆ</sup> as the ψ- dimension of the full likelihood ℓ, but with potentially reduced curvature, capturing information lost to λ. Subtracting the maximum, $\ell _ { p } ( \hat { \psi } )$ (a constant in $\psi )$ , gives the log profile LR Λ,

$$
\begin{array} { l } { { - 2 \log \Lambda ( \psi ) = } } \\ { { - 2 ( \ell _ { p } ( \psi ) - \hat { \ell } _ { p } ) \approx ( \psi - \hat { \psi } ) ^ { 2 } I _ { \psi \cdot \lambda } , } } \end{array}
$$

where $\hat { \psi }$ is the ψ- (first-) element of $\hat { \pmb \theta } _ { : }$

$$
\hat { \psi } = \hat { \pmb { \theta } } _ { \psi } = \psi _ { 0 } + s _ { \psi \cdot \pmb { \lambda } } / I _ { \psi \cdot \pmb { \lambda } } ,
$$

$s _ { \psi \cdot \lambda }$ is the eficient score, $I _ { \psi \cdot \lambda }$ is the eficient information

$$
I _ { \psi \cdot \lambda } ^ { - 1 } = ( { \bf I } ^ { - 1 } ) _ { \psi \psi } ,
$$

and $\hat { \ell } _ { p } = \ell _ { p } ( \hat { \psi } )$ is the constant profile likelihood value at the peak of the quadratic. 4. The maximal invariant of this local Gaussian location experiment is

$$
m _ { \psi } ^ { \mathrm { l o c a l } } = \hat { \psi } - \psi .
$$

We can see that the data enters the maximal invariant (the natural target for $z _ { \mathrm { p } } ,$ , see Appendix C.2) only through the eficient score: an object known to be highly informative about ψ. Furthermore (and as expected), the powerful profile LR is a function of this maximal invariant.

What is more, after a simple local rescale, this maximal invariant is known to converge in distribution, $\sqrt { I _ { \psi \cdot \lambda } } m _ { \psi } ^ { \mathrm { l o c a l } } \ \xrightarrow { d } \mathcal { N } ( 0 , 1 )$ , making it asymptotically pivotal and so an ideal target for $z _ { \mathrm { p } } .$ (The $\chi ^ { 2 }$ limiting distribution of −2 log Λ results directly from this.) This convergence in part reflects the shrinking local region over which the nuisance efects must be unified. The same benefit can be expected for $z _ { \mathrm { p } } \mathrm { . }$ consistency between the local maximal invariants is required over a progressively smaller neighbourhood, making the job of combining them into a global $z _ { \mathrm { p } }$ progressively easier.

The eficient score, profile LR and local maximal invariants all then embody precisely the intuition we are aiming at: that removal of λ-information distils the pure ψ-information. In the LAN limit, all three draw from the same local information about ψ, and we might expect $z _ { \mathrm { p } }$ to converge to their performance in terms of power.

## 5 Further considerations

In Section 5.1 we discuss how we incorporate invariances into our model and in Section 5.2 we further add extra “known values” (like sample size) to allow our model to work over a wider range of inputs. In Section 5.3, we discuss how to sample parameters so that we have confidence in our p-values. Finally in Section 5.4, we discuss small tweaks to the neural network design that we have found anecdotally to be helpful.

## 5.1 Incorporating invariances

In many cases, there are invariances in the structure of our model that we can exploit. In the one-sample t-test example (Section 3.3), we expect the same behaviour irrespective of location and scale. If we can incorporate these invariances into our neural network mappings, we can reduce their complexity (reducing their dimensionality), while simultaneously expanding the domain over which they can produce valid p-values (across all locations on a given orbit; in many cases this makes the domain infinite in one or more dimensions).

Suppose that the model we wish to fit is invariant (up to the corresponding Jacobian factor $| J _ { g } | )$ under the action of a transformation group $G ;$ specifically $\forall g \in G .$

$$
\begin{array} { c } { { p ( \mathbf { x } | \pmb { \theta } ) = p ( g \cdot \mathbf { x } | g \cdot \pmb { \theta } ) | J _ { g } | , } } \\ { { \psi ( g \cdot \pmb { \theta } ) = g \cdot \psi ( \pmb { \theta } ) . } } \end{array}
$$

where $| J _ { g } |$ represents the absolute Jacobian determinant of the mapping $^ { g , }$ and $G$ has $n _ { G }$ dimensions.

We introduce a canonicalization rule $g _ { \mathbf { x } } : = g _ { \mathbf { x } } ( \mathbf { x } ) : \mathcal { X } \mapsto G$ , which chooses a mapping, based on the statistics $\mathbf { x } \in \mathcal { X }$ , onto a single canonical representative from the orbit on which x lies. In doing so, it also reduces the dimensionality of $g _ { \bf x } \cdot { \bf x }$ to $n _ { \mathbf { x } } - n _ { G }$ . Since the canonicalization rule is chosen based on $\mathbf { x } .$ , no dimensions are lost from $\theta \mapsto g _ { \mathbf { x } } \cdot \theta$ and its dimensionality remains $n _ { \theta }$

Then we replace each of our normalizing flows by the canonicalization transform followed by a network with smaller input dimension:

$$
\begin{array} { r l } & { z _ { \mathrm { p } } ( g _ { \mathbf { x } } \cdot \mathbf { x } ; \boldsymbol { \psi } ( g _ { \mathbf { x } } \cdot \pmb { \theta } ) ) : \mathbb { R } ^ { ( n _ { \mathbf { x } } - n _ { G } ) + 1 }  \mathbb { R } , } \\ & { \quad \mathbf { z } _ { \mathrm { n } } ( g _ { \mathbf { x } } \cdot \mathbf { x } ; g _ { \mathbf { x } } \cdot \pmb { \theta } ) : \mathbb { R } ^ { ( n _ { \mathbf { x } } - n _ { G } ) + n _ { \pmb { \theta } } }  \mathbb { R } ^ { n _ { \mathbf { x } } - 1 } . } \end{array}\tag{5}
$$

Critically, when the Jacobian is computed for the loss in (1), it is still computed with respect to the full (pre-canonicalization) x, by chain rule through $g _ { \mathbf x }$

$$
\frac { \partial \mathbf { z } } { \partial \mathbf { x } } = \frac { \partial \mathbf { z } } { ( g _ { \mathbf { x } } \cdot \mathbf { x } , g _ { \mathbf { x } } \cdot \pmb { \theta } ) } \frac { \partial ( g _ { \mathbf { x } } \cdot \mathbf { x } , g _ { \mathbf { x } } \cdot \pmb { \theta } ) } { \partial \mathbf { x } } .
$$

One might wonder whether our Jacobian $\textstyle { \frac { \partial \mathbf { z } } { \partial \mathbf { x } } }$ can still be nonsingular, given that we first transform $\mathbf { x } \in \mathbb { R } ^ { n _ { \mathbf { x } } }$ to a lower dimensional $g _ { \mathbf x } \cdot \mathbf x \in \mathbb R ^ { n _ { \mathbf x } - n _ { G } }$ . The key intuition is that some of the gradient with respect to x is also transferred via $g _ { \mathbf x } \cdot \pmb \theta .$ , which now also depends on x. We must have $n _ { G } \leq n _ { \mathbf { x } }$ by definition and we have $n _ { \bf x } = n _ { \theta }$ by assumption (Section 3), so there will always be at least enough dimensions in our neural network for a full-rank Jacobian to be possible.

Again the t-test represents a simple example: we can capture scale invariance by $g _ { \mathbf { x } } \cdot \mathbf { x } = ( m / s )$ and $\begin{array} { r } { g _ { \mathbf { x } } \cdot \pmb { \theta } = \left( \frac { \mu } { s } , \frac { \sigma } { s } \right) } \end{array}$ . We can even reduce x to zero dimensions by further incorporating location invariance: $g _ { \bf x } \cdot { \bf x } = 0$ and $\begin{array} { r } { g _ { \mathbf { x } } \cdot \pmb { \theta } = \left( \frac { \mu - m } { s } , \frac { \sigma } { s } \right) } \end{array}$ . Even though there are no inputs to our net that directly represent x, the Jacobian with respect to $( m , s )$ is still transferred via $g _ { \mathbf x } \cdot \pmb \theta$ by

$$
\frac { \partial \mathbf { z } } { \partial \mathbf { x } } = \frac { \partial \mathbf { z } } { \partial ( \frac { \mu - m } { s } , \frac { \sigma } { s } ) } \frac { \partial ( \frac { \mu - m } { s } , \frac { \sigma } { s } ) } { \partial ( m , s ) } .
$$

We reproduce exactly this example in practice in Section 6.1 and present in Section 6.2 promising numerical results from an example where $n _ { \mathbf { x } } = n _ { \pmb { \theta } } = 3$ and $n _ { G } = 2$ , so that here also the Jacobian must flow via the parameters.

## 5.2 Conditioning on further known values

As described in Section 2.3, adding extra conditioning variables to our normalizing flows can be achieved simply by adding those conditioning variables as further inputs to the network. We use this to allow further “known values” ν (of dimension $n _ { \nu } )$ to be added to the model. Again referring to the t-test example, we might take the sample size $\nu = n$ as a known value, so that our net can model the t-test problem over a range of sample sizes.

Then our full normalizing flow decomposition (2) is expanded to

$$
\begin{array} { r l } & { z _ { \mathrm { p } } ( \mathbf { x } ; \boldsymbol { \psi } ( \pmb { \theta } ) , \nu ) : \mathbb { R } ^ { n _ { \mathbf { x } } + 1 + n _ { \nu } }  \mathbb { R } , } \\ & { \quad \quad z _ { \mathrm { n } } ( \mathbf { x } ; \pmb { \theta } , \nu ) : \mathbb { R } ^ { n _ { \mathbf { x } } + n _ { \pmb { \theta } } + n _ { \nu } }  \mathbb { R } ^ { n _ { \mathbf { x } } - 1 } } \end{array}\tag{6}
$$

and likewise for the G-invariant version in (5).

## 5.3 Sampling an appropriate range of parameters for training

If $z _ { \mathrm { p } }$ is to learn how to pivot, then every x that is likely to arise from a given set of parameters $\pmb { \theta } _ { 1 }$ must compete with the same x value arising from every other set of parameters $\pmb { \theta } _ { 2 }$ with the same $\psi ( \pmb { \theta } _ { 1 } ) = \psi ( \pmb { \theta } _ { 2 } )$ that are “likely” to give rise to it.

Therefore, the choice of parameter samples Θ to be included in training the net is rather important.

Define $\mathcal X _ { \mathrm { t a r g e t } } \subseteq \mathcal X$ to be the set of all x ∈ X for which we want to be able to generate valid p-values. Further, for any $\Theta _ { a } \subseteq \Theta$ , define $\mathcal { X } _ { \Theta _ { a } } \subseteq \mathcal { X }$ to be the set of all x that have non-negligible probability of arising from one or more $\pmb { \theta } \in \Theta _ { a }$ . Parameter samples θ for testing and training our net are then drawn respectively from:

1. $\Theta _ { \mathrm { i n n e r } } = \{ \pmb \theta : \pmb \chi _ { \{ \pmb \theta \} } \cap \pmb \chi _ { \mathrm { t a r g e t } } \neq \emptyset \}$ , an “inner” set of parameters, which represent all θ that might generate values in $\mathcal { X } _ { \mathrm { t a r g e t } }$ — we want our model to generate valid p-values for all $\pmb { \theta } \in \Theta _ { \mathrm { i n n e r } }$ and use this for testing;

2. $\Theta _ { \mathrm { o u t e r } } = \{ \pmb \theta : \pmb \chi _ { \{ \pmb \theta \} } \cap \pmb \chi _ { \Theta _ { \mathrm { i n n e r } } } \neq \emptyset \}$ , an “outer” set of parameters, which represent all θ that might generate x that coincide with any of those generated by Θ<sub>inner</sub> — we train our model on $\pmb { \theta } \in \Theta _ { \mathrm { o u t e r } }$

Within the respective sets $\Theta _ { \mathrm { i n n e r } }$ and $\Theta _ { \mathrm { o u t e r } }$ , we would like to sample our parameters according to the Jefreys prior $p ( \pmb \theta ) \propto \sqrt { \operatorname* { d e t } ( I ( \pmb \theta ) ) }$ . Since the Fisher information I is somewhat expensive to estimate in our setup, we instead note that for maximum likelihood estimators $\hat { \pmb { \theta } }$ of $\pmb \theta , \mathrm { C o v } ( \hat { \pmb \theta } | \pmb \theta ) \approx I ( \pmb \theta ) ^ { - 1 }$ and approximate I from the covariance matrix of a series of estimates computed from samples drawn from θ. We then use the square root determinant of these approximate I to train one denormalizing flow (Section 2.4) each to model $\Theta _ { \mathrm { i n n e r } }$ and $\Theta _ { \mathrm { o u t e r } }$ and allow rapid sampling from them.

We further use the same mapping from data to parameter estimates, $\mathbf { x } \mapsto { \hat { \pmb { \theta } } } .$ , to define $\mathcal { X } _ { \mathrm { t a r g e t } }$ as a bounding box B around these estimates. We refer to this bounding box as the “estimates box”. The procedure for achieving all of the above is currently a little ad hoc, and we hope to refine it in future work. It is described in Appendix A.

## 5.4 Neural Network Design

A neural network typically consists of simple diferentiable functions “stacked on top of each other”. The most common format for each “layer” $f _ { \phi ^ { ( i ) } } ^ { ( i ) }$ is the fully-connected layer, which looks rather like a series of parallel generalized linear models:

$$
f _ { \phi ^ { ( i ) } } ^ { ( i ) } = \sigma ( \mathbf { W } ^ { ( i ) } \mathbf { x } + \mathbf { b } ^ { ( i ) } ) ,\tag{7}
$$

for some non-linearity σ and with the matrix W of weights and vector b of biases representing the learnable parameters $\phi ^ { ( i ) }$ for that layer. Then the entire neural network is made of the composition of a series of these layers:

$$
f _ { \phi } = f _ { \phi ^ { ( n ) } } ^ { ( n ) } \circ f _ { \phi ^ { ( n - 1 ) } } ^ { ( n - 1 ) } \circ \cdot \cdot \cdot \circ f _ { \phi ^ { ( 1 ) } } ^ { ( 1 ) } .
$$

In NeuralCIs, our input and output layers are based on the classic fully connected layer, but with all intermediate (“hidden”) layers incorporating an extra multiplication and addition with a standardisation in-between:

$$
f _ { \phi ^ { ( i ) } } ^ { ( i ) } = \sigma \left( \mathbf { x } \odot ( \mathbf { W } ^ { ( i ) } \mathbf { x } + \mathbf { b } ^ { ( i ) } ) \right) + \mathbf { x } ,\tag{8}
$$

where ⊙ represents the Hadamard product. Our σ now represents a “layer-norm”, which standardizes the elements of its input vector by their combined mean and standard deviation.

The key intuition behind the incorporation of the extra Hadamard product is that it should allow each input more easily to moderate (“interact with”) the efect of another. In a classic fully-connected layer (7), a variable can moderate the efect of another only by means of cancellations of efects across diferent neurons. In our design (8), a given input can directly moderate the efect of any other input that is directed into its output path by W<sup>(i)</sup>.

Furthermore, the extra multiplication and layer-norm provide a clean, simple nonlinearity, thereby removing the need for an artificially chosen σ. The standardization helps to keep values from exploding or vanishing as they are propagated, as well as gradients as they are propagated backwards by chain rule. The final addition similarly helps to stop the gradients from dying out as they are propagated backwards, by providing a form of “skip connection”.

Anecdotally, this has performed somewhat better for our models. Removing the multiplication and instead adding an ELU activation in our tests in Section 6 provided slightly less accurate test sizes across the board (re-checked at various stages in the project). For Behrens-Fisher, standard deviation of test sizes at $\alpha \ : = \ : 0 . 0 5$ in our latest simulation (across 10000 parameter samples, each with one million p-values) was 0.0032 with multiplication vs 0.0035 with the ELU. For partial biserial correlation (the no-comparison run in Section 6.3.4, again 10000 samples each of one million p-values), these were 0.0033 (multiplication) vs 0.0038 (ELU). We might expect the efect to be more pronounced the larger the input dimension and the more complex the interaction structure between those inputs. But we leave a full analysis of this for a later paper.

Inputs to the nets are transformed as appropriate for the type of variable (log for SDs, atanh for correlations) and rescaled to focus their distribution very roughly onto [−1, 1]. Our model uses classic fully-connected layers at input and output to expand/shrink the dimension to/from a hidden dimension of 50. These layers have no non-linearity $( \sigma ( x ) = x )$ . Between these two layers are sandwiched 7 hidden layers following (8) with dimension 50. The full model design has not yet been fully optimized; we leave that also to future work.

Each net was optimized using Keras’ Nadam<sup>1</sup> optimizer, with an initial learning rate of 0.0025, decayed by a ratio of 0.9 after 8 epochs with no improvement in training loss, to a minimum learning rate of $1 0 ^ { - 6 }$ . The batch size was 1024; the $\left( z _ { \mathrm { p } } , \mathbf { z } _ { \mathrm { n } } \right)$ nets were (except the toy problem in Section 6.4) trained with 1000 epochs of 200 steps, and all other nets with 500 epochs of 100 steps.

## 6 Experimental Methodology

To evaluate the performance of NeuralCIs, we ran Monte Carlo simulations to compare to standard approaches on three classic statistics problems with nuisance parameters: normal mean with unknown variance, the Behrens-Fisher problem and partial biserial correlations. We further ran a toy simulation to illustrate the convergence of $z _ { \mathrm { p } }$ to the maximal invariant.

In each simulation, we ensure that the same random samples are used for each method (for example by using stateless random number generators) so that the data is truly paired. Power values are size-adjusted: that is, they are computed based on thresholding each p-value at an “idealized” value that corrects for the test being conservative or liberal. Without such a correction, a very liberal test can easily achieve higher power, without having detected anything more of substance.

This size adjustment is achieved by adding a further simulation that (i) draws samples from the alternative hypothesis instead of the null, (ii) computes a p-value for each such sample against a null hypothesis equal to this alternative, and (iii) finds the α quantile of these p-values and uses this as the threshold, rather than α. If the size of the particular method is correct, this should simply yield α as the threshold, but if the test is liberal, it will result in a lower threshold and correspondingly lower power, which represents the power the test would achieve, $i f$ the test were correctly calibrated and therefore right-sized.

## 6.1 t-Test

According to Section 4.3.1, we expect $z _ { \mathrm { p } }$ to align with the maximal invariant $m _ { \psi }$ of $( { \bf x } , \psi )$ in a transformation model. As an example, we fit the classical one-sample ttest, described in Section 3.3. To test this, we compare one-tailed p-values for a range of samples via the classical t-test with those generated via NeuralCIs. We fit two diferent NeuralCIs models for this. The first does not account for location and scale invariance (we will refer to this as the “non-canonicalized” model); the second (the “canonicalized” model) does, as described in Section 5.1.

In the non-canonicalized model, we draw sample sizes $n ~ \in ~ [ 3 , 1 0 0 ]$ and aim to be able to correctly generate $p { \cdot }$ -values for parameter estimates $\hat { \mu } = m \in [ - 3 , 3 ]$ and $\hat { \sigma } = s \in [ 0 . 3 3 3 , 3 ]$ . To be able to generate well-calibrated p-values for these samples, we might hope that any $( \mu , \sigma )$ that could generate such samples would generate uniform p-values across all samples, even those outside of the estimates box. This is the basis of our definition of $\Theta _ { \mathrm { i n n e r } }$ in Section 5.3. Our sampling scheme at present does not successfully reach to the very edge of $\Theta _ { \mathrm { i n n e r } }$ , see Appendix D.1, so in the main text of this first version of the paper, we present results for the reduced problem, where x are drawn from parameters drawn from the estimates box, $\mathbf { x } \sim p _ { \mathbf { x } } ( \mathbf { x } | \pmb \theta )$ with $\pmb { \theta } \in \mathcal { B }$ rather than the full problem $\pmb \theta \in \Theta _ { \mathrm { i n n e r } } .$

Specifically, we draw 50000 $\mu \in [ - 3 , 3 ]$ samples uniformly and 50000 $\sigma \in [ 0 . 3 3 3 , 3 ]$ and $n \in [ 3$ , 100] samples log-uniformly. We then draw one $( m , s )$ pair randomly at each parameter; that is, a total of 50000 samples. We hope to be able to expand this to the full $( \mu , \sigma ) \in \Theta _ { \mathrm { i n n e r } }$ in a subsequent version of the paper soon.

In the canonicalized model, we hope to obtain valid p-values with any $( m , s )$ drawn from any $( \mu , \sigma )$ and with $n \in [ 3 , 1 0 0 ]$ . We test this by drawing 50000 $\mu \in$ $[ - 1 0 0 , 1 0 0 ]$ and $\sigma \in [ 0 . 0 1 , 1 0 0 ]$ samples; we then draw a sample each from the sampling distributions at these parameters.

## 6.2 Behrens-Fisher

The Behrens-Fisher problem is a perfect example to test this on. Data for each of two groups are assumed to have been drawn from one normal distribution per group, each having its own mean $( \mu _ { 1 } , \mu _ { 2 } )$ and standard deviation $( \sigma _ { 1 } , \sigma _ { 2 } )$ . The goal is to infer the diference $\Delta = \mu _ { 2 } - \mu _ { 1 }$ between the means of the two normal distributions. Finding a test statistic for the Behrens-Fisher problem that is near-pivotal with a known reference distribution, while retaining good power, has long been an active area of research: Paul et al. (2019) review and simulate a number of alternatives and ultimately recommend the very widely used approximation developed by Welch (1947) in most small-to-moderate sample-size configurations.

We train our model on the parameters $\Delta , \sigma _ { 1 }$ and $\sigma _ { 2 }$ with statistics $\hat { \Delta } , \hat { \sigma } _ { 1 }$ and ${ \hat { \sigma } } _ { 2 }$ based on their unbiased and suficient statistics<sup>2</sup>. We further add the group sample sizes $n _ { 1 } , n _ { 2 } \in [ 3 , 1 0 0 ]$ as “known values” as per Section 5.2. We model location and scale invariance by subtracting $\hat { \Delta }$ from both $\Delta$ and $\hat { \Delta }$ and dividing all parameters and statistics by $\hat { \sigma } _ { 1 }$ . This reduces the set of statistics that will enter our net to just one: $\frac { \hat { \sigma } _ { 2 } } { \hat { \sigma } _ { 1 } }$ . We train our network to target valid p-values for any $\textstyle { \frac { { \widehat { \sigma } } _ { 2 } } { { \widehat { \sigma } } _ { 1 } } } \in [ 0 . 3 3 3 , 3 ]$

We aim to select a wide and representative sample of parameters to test the model. We do not use either of the denormalizing flows that are trained according to Section 5.3 and used in training the net, as these might conceal holes in the training data. Instead we sample a fresh set of 10,000 parameter combinations as described in Table 1, with which we would like to stress-test the invariance logic, by being far outside of the range of parameters encountered within training.

However, as with the t-test example in Section 6.1, we currently draw samples from parameters that have $\sigma _ { 2 } / \sigma _ { 1 } \in [ 0 . 3 3 3 , 3 ]$ rather than any parameter that can produce statistics in that range. Again, this relates to a slight weakness at the very edges in the current parameter sampling scheme, see $\mathrm { A }$ ppendix D.1, which we are working to fix. We hope to expand this to the full $\Theta _ { \mathrm { i n n e r } }$ in a subsequent revision soon.

<table><tr><td>Parameter</td><td>Distribution</td><td>Comment</td></tr><tr><td>Sample size group 1</td><td> $\overline { { n _ { 1 } \sim \mathrm { L o g U n i f } ( 3 , 1 0 1 ) } }$ </td><td>Floored (integer 3-100)</td></tr><tr><td>Sample size group 2</td><td> $n _ { 2 } \sim \mathrm { L o g U n i f } ( 3 , 1 0 1 )$ </td><td>Floored (integer 3-100)</td></tr><tr><td>SD group 1</td><td> $\sigma _ { 1 } \sim \mathrm { L o g U n i f } ( 0 . 0 1 , 1 0 0 )$ </td><td></td></tr><tr><td>SD group 2</td><td> $\sigma _ { 2 } \sim \mathrm { L o g U n i f } ( 0 . 3 3 3 \sigma _ { 1 } , 3 \sigma _ { 1 } )$ </td><td></td></tr><tr><td>Mean difference</td><td> $\mu _ { 2 } - \mu _ { 1 } \sim \mathrm { U n i f } ( - 1 0 0 \sigma _ { 1 } , 1 0 0 \sigma _ { 1 } )$ </td><td></td></tr></table>

Table 1 Sampling distributions of Behrens-Fisher parameters in our test.

For each parameter sample $\pmb \theta = ( \Delta , \sigma _ { 1 } , \sigma _ { 2 } )$ , we further compute a corresponding ∆<sup>power</sup> value that aims crudely to target a power drawn uniformly in the range $1 - \beta \in$ [0.05, 0.90], when applied to statistics sampled from the corresponding null parameters θ at a p-value threshold of $\alpha = 0 . 0 5$

$$
\begin{array} { r } { \Delta ^ { \mathrm { p o w e r } } = \Delta \pm \sigma \left( \Phi ^ { - 1 } ( 1 - \frac { \alpha } { 2 } ) + \Phi ^ { - 1 } ( 1 - \beta ) \right) , } \end{array}
$$

$$
\sigma = { \sqrt { { \frac { \sigma _ { 1 } ^ { 2 } } { n _ { 1 } } } + { \frac { \sigma _ { 2 } ^ { 2 } } { n _ { 2 } } } } } ,
$$

where $\Phi ^ { - 1 }$ is the standard normal quantile function; where ± is randomly assigned to addition or subtraction; and where $1 - \beta$ is a target power value uniformly sampled from [0.05, 0.90].

At each sampled $( \Delta , \Delta ^ { \mathrm { p o w e r } } , \sigma _ { 1 } , \sigma _ { 2 } , n _ { 1 } , n _ { 2 } )$ , we compute the following three sets of p-values:

1. At $( \Delta , \sigma _ { 1 } , \sigma _ { 2 } , n _ { 1 } , n _ { 2 } )$ , we draw $1 0 ^ { 6 }$ samples of statistics $( \hat { \Delta } , \hat { \sigma } _ { 1 } , \hat { \sigma } _ { 2 } )$ , and pass $( \hat { \Delta } , \hat { \sigma } _ { 1 } , \hat { \sigma } _ { 2 } , \Delta , n _ { 1 } , n _ { 2 } )$ into our model to generate $1 0 ^ { 6 }$ null-hypothesis $p { \vdash }$ values.

2. To assess power, we also pass the same $1 0 ^ { 6 }$ samples into our model as $( \hat { \Delta } , \hat { \sigma } _ { 1 } , \hat { \sigma } _ { 2 } , \Delta ^ { \mathrm { p o w e r } } , n _ { 1 } , n _ { 2 } )$

3. To further size-adjust this power, we repeat step 1 with $\Delta ^ { \mathrm { p o w e r } }$ in place of $\Delta .$ for a smaller sample of $1 0 ^ { 4 }$ size-adjust p-values.

We pass the same values across the three steps into the Welch test.

## 6.3 Partial biserial correlation

As a stress test for the architecture, we also chose to model the partial biserial correlation, where a latent partial correlation between two normal random variables $A \sim \mathcal N$ and $B \sim { \mathcal { N } }$ , conditioned on a third $C \sim \mathcal { N }$ , is masked by one of the former two variables only being observable through its binarization $A ^ { * }$ , where $A ^ { * } : = 1 \{ A > \tau \}$ for some threshold value τ.

We model the simplified case, where all normal variables are known to have zero mean and unit standard deviation. In that case, the parameters to the model are the correlations between the three normal variables $\rho _ { A B } , \rho _ { A C } , \rho _ { B C }$ as well as the binarization threshold $\tau .$ We find it more intuitive to model the threshold as a binarization probability $\pi _ { A ^ { * } } ~ = ~ \Phi ( - \tau )$ . Importantly, it is also cleaner to model the correlation $\rho _ { A B }$ by its partial correlation $\rho _ { A B | C }$ , so that each of the three correlation parameters can take any value $\rho \in ( - 1 , + 1 )$ without losing positive definiteness. As with our Behrens-Fisher model, we also add the sample size n as $\mathrm { a }$ “known value”.

We choose a set of cheap statistics: the observed (e.g. point biserial) correlations between the three observable variables $\hat { \rho } _ { A ^ { * } B } , \hat { \rho } _ { A ^ { * } C } , \hat { \rho } _ { B C }$ along with the observed prevalence in $A ^ { * } , { \hat { \pi } } _ { A ^ { * } }$ . Note that the statistics here are not direct estimates of the parameters, as the correlations with $A ^ { * }$ are point biserial correlations with binary $A ^ { * }$ They are also not known to be suficient statistics. However, under the latent Gaussian threshold model, the prevalence and point-biserial correlations are in one-to-one correspondence with the threshold and latent biserial correlations; see for example Olsson et al. (1982) for the relationship between biserial and point-biserial correlations.

This represents a stress test for two reasons in particular. Firstly, our statistics do not form a suficient statistic for the parameter vector of the model. Secondly, one of these statistics, ${ \hat { \pi } } _ { A ^ { * } }$ is discrete rather than continuous. Since the entire theory is based on difeomorphisms on continuous statistics, it raises the interesting question: can our model approximate in the discrete case?

We compare our approach with two widely used likelihood-ratio (LR) -based approximations: the $\chi ^ { 2 }$ likelihood-ratio test and the bootstrap likelihood-ratio test:

## 6.3.1 Likelihood ratio (LR) tests

A p-value simulation involving likelihood ratios must find the maximum likelihood estimate (MLE) of the parameters (once constrained to $\rho _ { A B | C } = \rho _ { A B | C } ^ { \mathrm { n u l l } } .$ once unconstrained) for every one of hundreds to millions of sampled datasets per parameter set that we wish to simulate from. Any dataset whose MLE lands at the boundary of our space (i.e. correlations close to ±1, prevalence close to 0 or 1) can be problematic, both for our optimizer and (particularly for directly-on-the-boundary samples) for our interpretation of the likelihood ratio. We aim to keep the number of boundary cases per parameter to a rate that is negligible, by selecting parameters θ to simulate that are suficiently far themselves from the boundary. We then track and report nonconvergence and boundary case rates, allowing each of these to remain in our p-value curve, rather than excluding them.

In Appendix B, we outline a sampling scheme for θ that aims to keep boundary correlation cases $\left( \left| \hat { \rho } \right| > 0 . 9 9 \right)$ and degenerate prevalence cases $( \hat { \pi } _ { A ^ { \ast } } \in \{ 0 , 1 \} )$ rare, to the order of approximately $1 0 ^ { - 4 }$ or less. For low n, this tends to mean quite a narrow range for our parameters (for $n = 2 0$ , we have $\pi _ { A ^ { * } } \in ( 0 . 3 9 , 0 . 6 1 )$ , $| \rho _ { A B | C } | < 0 . 0 7$ $| \rho _ { A C } | < 0 . 1 1 , | \rho _ { B C } | < 0 . 9 3 )$ , and this gradually fans out to a fairly broad range for higher n (for n = 100, $\pi _ { A ^ { \ast } } \in \left( 0 . 0 9 , 0 . 9 1 \right)$ ， $| \rho _ { A B | C } | < 0 . 7 2$ $| \rho _ { A C } | < 0 . 7 2$ , |ρ<sub>BC</sub>| < 0.98). We further pair each $\psi = \rho _ { A B | C }$ parameter sample with a corresponding $\rho _ { A B | C } ^ { \mathrm { p o w e r } }$ value, used to compute power against the same sampled statistics with just one extra likelihood optimization; this is sampled to very roughly target a uniform distribution for power $( 1 - \beta ) \sim \mathrm { U n i f } ( 0 . 0 5 , 0 . 9 0 )$ , but constrained to the same range.

Prevalences parameterizing the likelihood surface were mapped from (0, 1) to $( - 1 , 1 )$ and all parameters were then atanh-transformed. In this transformed space, there is quite a wide spread at the boundaries, leading occasionally to very slow convergence for extreme samples (delays, which are then multiplied across many parallel runs). To speed up convergence, the likelihood surface corresponding to extreme correlation values, $| \hat { \rho } | > 0 . 9 9$ was (in the optimization phase) swapped for a steep downward slope, so that the optimizer would not wander into this region. The same was applied to the transformed prevalences, i.e. a ramp was applied at $| 2 \hat { \pi } _ { A ^ { * } } - 1 | > 0 . 9 9$ . This ramp was removed when computing likelihood ratios, but had the efect of clipping extreme values around ±0.99. We explore the rates at which samples ended up caught on this fold (“boundary samples”) or with all-equal $A ^ { * }$ values (“degenerate samples”) in Appendix D.2.

## 6.3.2 Likelihood ratio $\chi ^ { 2 } \cdot \mathrm { t e s t }$

In the large sample limit, −2 log LR is asymptotically $\chi _ { 1 } ^ { 2 }$ distributed. This then allows for a relatively cheap test to be performed, in which two MLEs are evaluated and converted into an LR and directly into a p-value.

Nonetheless, optimizing to find the MLE twice in each case is far more computationally expensive than NeuralCIs, so we have to constrain ourselves to just

1000 parameter samples (compared with 10000 for Behrens Fisher, Section 6.2). As before, we generate one million sampled datasets at each parameter, to produce stable estimates of error rates and power. (This large number was achieved by running parallelized optimisations on an RTX 3060 Mobile GPU, using the Tensorflow-Probability BFGS optimizer.) The size-adjust is again based on a smaller sample of ten thousand.

## 6.3.3 Bootstrap likelihood ratio test

An alternative, and quite general approach to converting the profile LR to a p-value is to approximate its null distribution by parametric bootstrap. Samples are generated from our model, with $\rho _ { A B | C } = \rho _ { 0 }$ and nuisance parameters fixed at their constrained MLEs. (This is the same constrained MLE described in Section 6.3.1, optimized for the dataset in question.) For each sample, the profile LR is computed as described in Section 6.3.1, and the resulting bootstrap LR statistics are used as an approximation to the true null distribution. This is only an approximation, because we cannot guarantee (and often do not expect) that the null distribution of the LR will be the same across all relevant values of the nuisance parameters.

This is computationally several orders of magnitude more costly than the $\chi ^ { 2 }$ LR test, because each bootstrap sample costs as much as a single $\chi ^ { 2 }$ LR test. For this comparison, we ran 500 parameter samples, generating 500 p-values per parameter sample, each with 2000 bootstrap samples. We use the same number of samples and simulations again for the size-adjust.

## 6.3.4 Performance of NeuralCIs alone

Comparing NeuralCIs with bootstrap LR imposes a number of constraints, in particular a constrained parameter range (to avoid boundary statistics) and the small number of simulations that can be run (due to computational cost). Therefore, we will also present false positive rates of NeuralCIs without comparison to another method, with a much larger sample (as with Behrens-Fisher, one million p-values at each of 10,000 parameter samples), drawn much closer to the boundary.

To avoid overemphasizing the boundaries (and giving negligible probability mass to the important zero-correlation centre), we approximate drawing parameters at uniformly sampled angle and radius as follows. First a point is chosen uniformly from the 4D hypercube $[ - 1 , 1 ) ^ { 4 }$ and turned to a unit vector by dividing by its magnitude. This “angle” is then multiplied by a uniformly sampled magnitude in [0, 2). The fourth dimension is remapped $( - 1 , 1 ) \mapsto ( 0 , 1 )$ and the four dimensions are assigned to $\rho _ { A B | C } , \rho _ { B C } , \rho _ { A C }$ and π<sub>A</sub>∗ , respectively. Finally, any correlation greater in absolute value than 0.9 and any $\pi _ { A ^ { * } }$ ∗ not in the range $0 . 1 \le \pi _ { A ^ { * } } \le 0 . 9$ is excluded.

## 6.4 Toy problem: measuring deviation from maximal invariant

In Appendix C.2.1, we show that a model that is shift-uniform in the nuisance-orbit coordinate u would allow construction of a pivot that can be contaminated by a periodic dependency on u (that is, depend on x other than through the maximal invariant). We make an implicit-bias argument that such a solution is highly unlikely to be discovered when fitting $z _ { \mathrm { p } }$

To test this, we further fit a toy problem that has exactly this shift-uniform structure and monitor how closely $z _ { \mathrm { p } }$ depends on the maximal invariant $m _ { \psi }$ at each training epoch. We fit our architecture to the following two-dimensional model, with scalar parameters (ψ, λ):

$$
\begin{array} { l l } { { X _ { 1 } = \psi + X _ { 2 } + E _ { 1 } , \quad } } & { { X _ { 2 } = \lambda + E _ { 2 } , } } \\ { { E _ { 1 } \sim { \cal N } ( 0 , 1 ) , \quad } } & { { E _ { 2 } \sim \mathrm { U n i f } ( - 0 . 5 , 0 . 5 ) . } } \end{array}
$$

In this model, the maximal invariant $M _ { \psi }$ and the nuisance orbit coordinate $U$ are

$$
M _ { \psi } = X _ { 1 } - X _ { 2 } - \psi , \qquad U = X _ { 2 } .
$$

Before training, we construct a set of reference values of $m , \ e _ { 2 } , \ \psi$ and λ. For $m _ { \psi } = e _ { 1 }$ , we use 32 evenly spaced quantiles of the normal distribution, divided by their standard deviation to give them exact standard deviation of 1. For $e _ { 2 }$ , we use a uniformly distributed random sample of 64 values drawn between -0.5 and 0.5. For parameters we use 16 linearly spaced values each, for $\psi$ in [−2.2, 2.2] and for $\lambda$ in $[ - 1 , 1 ]$

Let $\mathbb { E } _ { m } , \mathrm { V a r } _ { \psi , \lambda }$ denote means and variances across the respective fixed grids $m , u , \psi , \lambda$ defined above. After each epoch, we compute $z _ { \mathrm { p } }$ and from these the following metrics.

1. A measure of the proportion of $Z _ { \mathrm { p } }$ -variance contributed by $U ,$ estimated by the proportion of $z _ { \mathrm { p } } -$ variance not accounted for by m:

$$
\mathrm { ~ U ~ p r o p o r t i o n } = \operatorname* { m a x } _ { \psi , \lambda } \frac { \mathbb { E } _ { m } [ \mathrm { V a r } ( z _ { \mathrm { p } } | m ) ] } { \mathrm { V a r } ( z _ { \mathrm { p } } ) }
$$

2. A measure of how similar the curve mapping M to $Z _ { \mathrm { p } }$ is across diferent $( \psi , \lambda )$

$$
\mathrm { M a p p i n g \ s t a b i l i t y } = \operatorname* { m a x } _ { m } \mathrm { V a r } _ { \psi , \lambda } ( \mathbb { E } [ z _ { \mathrm { p } } | m ] ) .
$$

3. A measure of the variance of $Z _ { \mathrm { p } }$

$$
\mathrm { O u t p u t \ v a r i a n c e } = \mathbb { E } _ { \psi , \lambda } \mathrm { V a r } ( z _ { \mathrm { p } } ) .
$$

4. An estimate of the Kolmogorov-Smirnov distance between the one-tailed p-values obtained from $Z _ { \mathrm { p } }$ and the uniform distribution:

$$
\mathrm { M a x ~ K S ~ o n e ~ t a i l e d } = \operatorname* { m a x } _ { \psi , \lambda } ~ \widehat { \mathrm { K S } } ( \Phi ( z _ { \mathrm { p } } ) , \mathrm { U n i f } ( 0 , 1 ) ) .
$$

We repeated this across 250 training runs, each with a diferent random initialisation, 500 epochs each of 100 steps. This is one quarter the training used in each of the other examples presented in this paper, where a full training run is 1000 epochs of 200 steps.

![](images/b1ec77905b2c001302128989e997a2e2046edea6d2fa08890ea49aaf84816e87.jpg)  
Fig. 1 Comparison of NeuralCIs to traditional t-test.

## 7 Results

In the following sections we present the results from these experiments: NeuralCIs for the one-sample mean vs the t-test in Section 7.1, for Behrens-Fisher vs Welch in Section 7.2, for partial biserial correlation vs profile likelihood ratio tests in Section 7.3 and finally the toy problem measuring deviation from the maximal invariant in Section 7.4.

## 7.1 t-Test

In Figure 1, we scatter the diference between the NeuralCIs one-tailed p-value and that of the t-test, against that baseline t-test p. Each plot is coloured according to sample size. Both models produce one-tailed p-values close to those generated by the t-test. For the non-canonicalized model, the absolute diferences between the two were less than 0.0020 in 99.5% of samples and less than 0.0033 in 100% of samples; for the canonicalized model they were less than 0.0012 in 99.5% and less than 0.0013 in 100% of samples.

## 7.2 Behrens-Fisher

Figure 2 compares false-positive rates and power between the NeuralCIs (top in each plot) and Welch (reflected below). Although Welch shows a very slightly narrower peak, its size error in the worst cases is considerably worse than that of NeuralCIs. The power diference between the two is negligible: less than one tenth of one percentage point in NeuralCIs’ favour.

## 7.3 Partial biserial correlation

Results for the likelihood ratio $\chi ^ { 2 }$ test, presented in Figure 3, show that, at these quite small sample sizes $( n \in [ 2 0 , 1 0 0 ] )$ , the $\chi ^ { 2 }$ test is consistently rather liberal, where NeuralCIs is quite neatly clustered around the target rate. Furthermore, NeuralCIs achieves on average 1.2 percentage points greater power.

Results for the bootstrap likelihood ratio are presented in Figure 4. Error rates are this time well centred around α for both tests (mean proportions below alpha are 0.010 and 0.049 for NeuralCIs and 0.010 and 0.050 for the bootstrap LR, at α = 0.01 and α = 0.05, respectively). Again, NeuralCIs has a size-adjusted power advantage of 1.2 percentage points.

![](images/7450b1e1f11ef4ad64a92259ced092f5d77c6d674608b146f3fcec2c5835c8de.jpg)  
Fig. 2 Comparison of NeuralCIs to traditional Welch test.

The error rates for the bootstrap LR appear to be quite widely spread, but with only 500 p-values per parameter sample, we already expect these error rates to be widely spread simply by chance variation. Table 2 shows that the spread (in terms of standard deviation) is close to the value expected from chance variation alone for both NeuralCIs and bootstrap LR; with only 500 p-values per parameter in the bootstrap comparison, modest variation in true test size would contribute very little to this spread<sup>3</sup> and therefore be hard to detect. However, combining our knowledge from Figures 3 and 4, we know that NeuralCIs has reasonable control of error rates and a power advantage over bootstrap LR. And importantly, it achieves this at speeds several orders of magnitude faster. Generating a half a million p-values for this run, spread across null and power p-values, took a total of 39 hours even highly parallelized on a GPU; generating ten billion p-values (20,000 times as many) for our stand-alone neural simulation (Section 6.3.4) took only fifteen minutes.

The comparison to the two profile LR approaches is much clearer when broken down by sample size. Figure 5 shows that the advantages of NeuralCIs in terms of power and error rates are strongest in small to moderate sample sizes (sample sizes 20 to 70 in particular) and progressively narrowed or even reversed as the sample size approaches 100. Between sample sizes 30-50, the power advantage appears to be around 3 percentage points on average. We must be careful in interpreting these results, because the parameter sampling region in our test also grows with sample size<sup>4</sup>, but the overall pattern is consistent with expectations: profile LR is asymptotically optimal in terms of local power and the $\chi ^ { 2 }$ approximation is asymptotically correctly sized, but may be liberal and/or underpowered in finite samples. The bootstrap can correct the size compared to the $\chi ^ { 2 }$ approximation, but with a very similar power disadvantage at smaller sample sizes (also to be expected, since the test statistic itself is the same).

![](images/b6c068bdc7188e90769672195a86ae9377267fd83fd3c83dc08c4e2ecb6c9fe5.jpg)

![](images/ea0a1fc2a5260f54abd70ec67603a9975d29c93fb875511bc3a9a1f4d858e087.jpg)

![](images/42e6ea108e6027f63dd61d9a9871ca21bc9c2e54b4131e3a11be52c45de36b49.jpg)

![](images/3439868762e18cae87d5254d928dca7a2c70af7846a10ebd6bfeccffbe7ac1df.jpg)

![](images/48d396c32418a1e67adbe12cd527d2c2c2b9f4c873e53f85dfd2f71022592e42.jpg)  
Power difference (Neural − χ<sup>2</sup> LR) at α = 0.05

![](images/fcbadc8732daff2cf57e5935efc102b273847611c2d2ad7651491a4f9df93cca.jpg)  
Fig. 3 Comparison of NeuralCIs to traditional $\chi ^ { 2 }$ likelihood-ratio test in partial biserial correlation problem. Power is size-adjusted, targeted at $\alpha = 0 . 0 5$

![](images/c198eab724d177dd1386ebd4e26306dce1bbc0e55ba26e33293bab9d249313c8.jpg)

![](images/13da651aabfcba2be820ac29cf4791eb0b1fc767deca8d5bd5ef505209bf330e.jpg)

Power difference (Neural − Bootstrap) at α = 0.05  
![](images/af9773892f4cb60b64d2617e424ded45b99c0bb6ff4c678d1ba719aa9ecb376e.jpg)  
Fig. 4 Comparison of NeuralCIs to the bootstrapped likelihood-ratio test in partial biserial correlation problem. Power is size-adjusted, targeted at $\alpha = 0 . 0 5$

<table><tr><td colspan="2">LR</td><td colspan="3">SD of error rate</td><td rowspan="2">Mean power difference</td></tr><tr><td>distribution</td><td>α</td><td>Expected</td><td>NeuralCIs</td><td>LR</td></tr><tr><td rowspan="2"> $\overline { { { \chi } ^ { 2 } } }$ </td><td>0.01</td><td>0.0001</td><td>0.0010</td><td>0.0022</td><td>0.011</td></tr><tr><td>0.05</td><td>0.0002</td><td>0.0028</td><td>0.0054</td><td>0.012</td></tr><tr><td rowspan="2">Bootstrap</td><td>0.01</td><td>0.0044</td><td>0.0043</td><td>0.0044</td><td>0.016</td></tr><tr><td>0.05</td><td>0.0097</td><td>0.0100</td><td>0.0099</td><td>0.012</td></tr><tr><td rowspan="2">None (only NeuralCIs)</td><td>0.01</td><td>0.0001</td><td>0.0011</td><td></td><td></td></tr><tr><td>0.05</td><td>0.0002</td><td>0.0033</td><td></td><td></td></tr></table>

Table 2 NeuralCIs and profile LR approaches compared. Expected standard deviation (SD) of error rates represents the standard deviation that would be expected, based on α and number $n _ { p }$ of p-values per null parameter, if the error rate were perfectly calibrated to α. This is just the standard deviation of a binomial distribution divided by $n _ { p }$ Expected $\begin{array} { r } { \mathrm { S D } = \sqrt { \frac { \alpha ( 1 - \alpha ) } { n _ { p } } } } \end{array}$ . Mean power diference is NeuralCIs minus Profile LR.

![](images/7b33952d4e043f97e4f7593a68059bde8cb70eb9f49d45670536194bb6356e94.jpg)  
Fig. 5 Results from Figures 3 and 4, broken down by sample size. This must be interpreted with caution, as the parameter sampling distribution is diferent at each sample size.

In Figure 6, we present false positive rates of NeuralCIs without comparison to another method. NeuralCIs does not become degenerate at the boundaries in the same way as likelihood approaches, so these results can capture a broad range of possible null parameters. The error rates are consistently quite close to the desired α. The ten billion p-values generated for Figure 6 took fifteen minutes to run.

![](images/82e4b75cd4e6f604a1f8a2095e01f98ab0125ca670c8f9234c6ed289edfc2c71.jpg)  
Fig. 6 NeuralCIs false positive rates at $\alpha = 0 . 0 1$ and $\alpha = 0 . 0 5$ and Kolmogorov-Smirnov distances vs uniform for partial biserial correlation example. Total of $1 0 ^ { 4 }$ null hypothesis parameter samples plotted, each computed from $1 0 ^ { 6 }$ datasets simulated at that null.

In Section D.2, we present rates of non-convergence and boundary and degenerate samples in the two profile-LR approaches. In the $\chi ^ { 2 }$ approximation, rates of nonconvergence are mostly well below one in a thousand (or slightly above in a few cases). Boundary samples are also almost entirely well below one in a thousand, except less than 1% of parameter samples which were slightly above that<sup>5</sup> and degenerate samples (prevalence of 0% or 100%) were almost all below one in ten thousand. For the bootstrap LR, we have far less ability to pre-plan avoiding boundary samples and most parameter samples have at least 1% of their $p \mathrm { - }$ values based on non-convergence rates between 0.001 and 0.03, with the worst cases being occasionally as high as ten percent non-converged. This applies only to the bootstrap distribution, since the same samples were used as the starting samples as in $\chi ^ { 2 }$

So, while the convergence rates on the $\chi ^ { 2 }$ approximation look very comfortable, we urge a little caution in interpreting the bootstrap LR results. Nonetheless, the pattern looks clean: both bootstrap LR and NeuralCIs are centred at the nominal $\alpha ,$ and the power advantage of NeuralCIs is consistent with that seen in the $\chi ^ { 2 }$ approach.

## 7.4 Deviation from the maximal invariant

Results of the u-dependence toy problem are shown in Figure 7. The proportion of variance not accounted for by the maximal invariant (the “U proportion”) rapidly becomes negligible and continues to become smaller as training progresses, and the mapping becomes very stable. All runs converge well to an output variance of 1 and a low KS-distance of p-values versus uniform. Note that there is a floor to the KSdistances, imposed by the small grid-sample of $m _ { \psi } = e _ { 1 } ;$ they cannot in this case be reduced below 0.02.

Of the 250 random initializations, 7 converged noticeably less cleanly than the others, all having final smoothed U-proportion above $1 0 ^ { - 5 }$ (coloured in Figure 7). While these higher final U-proportions are still efectively negligible, we trained these seven models for a further set of 500 epochs of 100 steps and present those results in Appendix D.3; all converge as expected after a further round of training.

![](images/281f987ae0c8966e4d012c422d2596c4c1c6fc8eb6a700498b0b3225f14fa732.jpg)

![](images/b5ef048526120e5051e3587837d3dbd8fe575cd7db92aa6efa9f01f9e3477e01.jpg)

![](images/38eea3529af5e69c876e40c5019546f628cc7727bdb6ea99ef344aa20d9d1eac.jpg)

![](images/9b030b8a0720022ac562661893915869ad4d050ee7e11c41df9b2c77ce2d6783.jpg)  
Fig. 7 Is $z _ { \mathrm { p } }$ a function of the maximal invariant alone? Each curve is smoothed using a uniform moving average with window size 10 epochs. All y-axes are log-axes. Those finishing with a smoothed $U .$ -proportion above $1 0 ^ { - 5 }$ are coloured.

## 8 Discussion

## 8.1 Interpretation of results

The Behrens-Fisher problem, with continuous suficient statistic of dimension $n _ { \bf x } = n _ { \theta }$ represents a perfect example of the sort of problem NeuralCIs aims at. Its performance is comparable to the Welch test, achieving considerably better worst-case false positive rates, with very similar power. This is, of course, within pre-specified bounds on $\hat { \sigma } _ { 2 } / \hat { \sigma } _ { 1 }$

On the other hand, our experiments with partial biserial correlation try to answer the question of whether this approach might be generalized to a cruder scenario, where “ball-park estimates” supplant suficient statistics and even one of those estimates is not continuous. In this particular case, NeuralCIs performs comparably with the bootstrap profile LR, but at a tiny fraction of the computational cost and with a small power advantage (1.2 percentage points on average), particularly at small sample sizes (a power advantage of around 3 percentage points for sample sizes from 30-50). Notably, this was at sample sizes where the traditional Chi-squared approximation to the profile LR was quite inaccurate. So, while the net must surely approximate (since one statistic is non-continuous), the approximation appears to be very efective.

Generating p-value distributions close to the boundary was a significant problem in the profile LR framework, so much so that we had to abandon attempts to do so. But in NeuralCIs, these can be generated quite readily and performed well.

Finally, the t-test example shows how close NeuralCIs can come to discovering the classical pivot for a transformation model, while the toy problem (“deviation from the maximal invariant”) shows that this occurs consistently, even when other pivots do exist.

## 8.2 Limitations and future work

We hope this to be the first of a line of steps in the NeuralCIs project, each step representing a current limitation of the project in its present form:

1. Fix the $\mathrm { e d g e - o f - } \Theta _ { \mathrm { i n n e r } }$ sampling issue highlighted in Appendix D.1.

2. The neural net dimensions have so far been chosen somewhat arbitrarily $/$ in an ad hoc fashion. Optimizing the hyper-parameters to the net, such as number of layers, size of layers, number of training epochs, and learning rate schedule is an important next step.

3. Invert our p-value net $z _ { \mathrm { p } }$ to train a further net that can generate confidence intervals for ψ. See Section 8.2.1.

4. Assess seed-to-seed variability in training.

5. Train and evaluate several models relating to interesting core statistical problems (as with partial biserial correlation) and wrap these up in an R package for true amortization of the training process.

6. Switch from explicitly diferentiating and punishing Jacobians to training via the free-form flows (FFF) approach of Draxler et al. (2024), to provide more scalable training and global difeomorphism targeting (see Section 2.2).

7. Currently only a single scalar interest parameter $\psi$ is supported. We would like to extend this to capturing simultaneous confidence intervals and/or confidence regions on a vector-valued interest parameter. One might, for example, make $z _ { \mathrm { p } }$ multidimensional, or model the combined distribution of several diferent NeuralCIs models, each trained on a diferent ψ.

8. The current architecture makes the severely limiting assumption that $n _ { \bf x } = n _ { \theta }$ When $n _ { \mathrm { { x } } } ~ > ~ n _ { \theta }$ , there may be multiple pivotal directions, some that are not informative about ψ. The subtractive argument made in Section 4.3 was that the single remaining dimension after subtraction of nuisance parameters is expected to be highly informative; this no longer holds. Expanding the framework to accept a higher-dimensional statistic would make it considerably more general.

9. A longer term extension would be to replace $z _ { \mathrm { p } }$ and $\mathbf { z } _ { \mathrm { n } }$ with recurrent neural networks (RNNs)<sup>6</sup>, that process individual i.i.d. observations $x _ { i }$ sequentially while updating an internal state representing the accumulated evidence for or against the null. This would depend on resolving point 8 above and would nonetheless require the fixed-dimensional learned state of the RNN to be able to hold suficient information from an arbitrarily large sample.

## 8.2.1 Confidence intervals

Generating confidence intervals from our current p-values approach will (we hope) be achieved by training a further net, using $z _ { \mathrm { p } }$ as a diferentiable loss function. The essence is to train a new net $\psi _ { \mathrm { C I } } ( \mathbf { x } , z )$ that maps statistics x, and a target $z \sim \mathcal { N } ( 0 , 1 )$ )， to a $\psi$ value that represents the boundary of a confidence interval. Specifically, if we feed the $\psi$ value output by ψ<sub>CI</sub> back into $z _ { \mathrm { p } }$ , the output from this,

$$
z _ { \mathrm { C I } } ( \mathbf { x } , z ) = z _ { \mathrm { p } } ( \mathbf { x } ; \psi _ { \mathrm { C I } } ( \mathbf { x } , z ) ) ,
$$

should aim to equal the input z value for all z and plausible $\mathbf { x } .$ . In doing so, it inverts the $p { \cdot }$ value implied by $z _ { \mathrm { p } }$ into a confidence interval made up of the $\psi$ values at which x would have the desired p-value. By plausible x, we mean specifically $\mathbf { x } \in \mathcal { X } _ { \Theta _ { \mathrm { i n n e r } } }$ 2 see Section 5.3.

We can achieve this by repeatedly

1. sampling parameter $\theta \in \Theta _ { \mathrm { i n n e r } }$ using the denormalizing flow trained in Section 5.3;

2. sampling each $\mathbf { x } \sim p _ { \mathbf { x } } ( \cdot ; \pmb \theta )$ from our simulator;

3. randomly sampling $z \sim \mathcal { N }$ values;

4. computing $z _ { C I } ( \mathbf { x } , z ) = z _ { \mathrm { p } } ( \mathbf { x } ; \psi _ { \mathrm { C I } } ( \mathbf { x } , z ) )$

5. adjusting the weights in the ψ<sub>CI</sub> net to minimize some discrepancy metric ${ \mathcal { L } } =$ $\Delta ( z _ { \mathrm { C I } } , z )$ between these two.

One may then obtain, say, a 95% CI for given x as

$$
[ \psi _ { \mathrm { C I } } ( \mathbf { x } , + 1 . 9 6 ) , \psi _ { \mathrm { C I } } ( \mathbf { x } , - 1 . 9 6 ) ] ,
$$

where the $\pm 1 . 9 6 = \Phi ^ { - 1 } ( ( 0 . 0 2 5 , 0 . 9 7 5 ) )$ values are the $z _ { \mathrm { p } }$ values corresponding to a two-tailed $p { \cdot }$ -value of 0.05.

## 9 Conclusion

We have seen that a simple decomposed normalizing flow-style neural network structure can extract near-pivotal statistics along with their distributions and that they can perform inference on three classical problems to an accuracy level that is competitive with classical approaches, and even with some advantages. The t-test is retrieved almost exactly, the Welch-test is bettered in terms of worst case error over a constrained variance-ratio range and the profile likelihood ratio for the partial biserial correlation is beaten in our simulation in terms of size-adjusted power.

The net can incorporate invariance information, making it more robust and giving it wider reach. Furthermore, once the expense of fitting the net has been invested, the fitted net can then be used extremely cheaply thereafter (much cheaper than bootstraps or likelihood surface optimizers). There are still considerable limitations, in particular the limitation that $n _ { \bf x } = n _ { \theta }$ (if we are to have reasonable power). Nevertheless, the work so far suggests that neural density methods based on normalizing flows could form a basis for a very general approach to frequentist inference.

## Acknowledgements

This paper is dedicated to the memory of my mother, Marion, who died during the final stages of its preparation.

## Declarations

Use of generative AI. The core contribution of this paper (the decomposition of the normalizing flow presented in Section 3) and the accompanying codebase were initially developed independently by the author. Anthropic’s Claude and OpenAI’s ChatGPT were used extensively throughout the theoretical development that followed, as a sounding board and as a critic: for brainstorming, for adversarial checking of the mathematical arguments, for directing the author towards relevant existing literature and theory (including, for instance, the classical theory of transformation models, which underpins the arguments in Appendix C), and for proofreading. A few short utility functions in the simulation code were also drafted with AI assistance. All arguments, experiments and conclusions have been verified by the author, who takes full responsibility for the content of this paper.

Code availability. The code for this project is available at github.com/ philassheton/neuralcis.

Funding. This work was carried out by the author independently and received no external funding.

Competing interests. The author declares no competing interests.

## References

Al Kadhim A, Prosper HB, Prosper OF (2024) Amortized simulation-based frequentist inference for tractable and intractable likelihoods. Machine Learning: Science and Technology 5(1):015020

Barndorf-Nielsen O, Jupp PE (1988) Diferential geometry, profile likelihood, l-suficiency and composite transformation models. The Annals of Statistics 16(3):1009–1043

Coccaro A, Pierini M, Silvestrini L, et al (2020) The DNNLikelihood: enhancing likelihood distribution with deep learning. The European Physical Journal C 80(7):664. https://doi.org/10.1140/epjc/s10052-020-8230-1

Cranmer K, Pavez J, Louppe G (2015) Approximating likelihood ratios with calibrated discriminative classifiers. URL https://arxiv.org/abs/1506.02169, arXiv:1506.02169

Dalmasso N, Masserano L, Zhao D, et al (2024) Likelihood-free frequentist inference: Bridging classical statistics and machine learning for reliable simulator-based inference. Electronic Journal of Statistics 18(2):5045–5090

Davison AC, Hinkley DV (1997) Bootstrap methods and their application. 1, Cambridge university press

Draxler F, Sorrenson P, Zimmermann L, et al (2024) Free-form flows: Make any architecture a normalizing flow. In: International Conference on Artificial Intelligence and Statistics, PMLR, pp 2197–2205

Goodfellow IJ, Pouget-Abadie J, Mirza M, et al (2014) Generative adversarial nets. Advances in neural information processing systems 27

Hall P, Martin MA (1988) On bootstrap resampling and iteration. Biometrika 75(4):661–671

Heinrich L (2022) Learning optimal test statistics in the presence of nuisance parameters. arXiv preprint URL https://arxiv.org/abs/2203.13079

Louppe G, Kagan M, Cranmer K (2017) Learning to pivot with adversarial networks. Advances in neural information processing systems 30

Olsson U, Drasgow F, Dorans NJ (1982) The polyserial correlation coeficient. Psychometrika 47(3):337–347

Papamakarios G, Nalisnick E, Rezende DJ, et al (2021) Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research 22(57):1–64

Paul S, Wang YG, Ullah I (2019) A review of the Behrens–Fisher problem and some of its analogs: Does the same size fit all? REVSTAT – Statistical Journal 17(4):563– 597. https://doi.org/10.57805/revstat.v17i4.281

Rezende D, Mohamed S (2015) Variational inference with normalizing flows. In: International conference on machine learning, PMLR, pp 1530–1538

Welch BL (1947) The generalization of ‘student’s’ problem when several diferent population variances are involved. Biometrika 34(1-2):28–35

Wilks SS (1938) The large-sample distribution of the likelihood ratio for testing composite hypotheses. The Annals of Mathematical Statistics 9(1):60–62. https: //doi.org/10.1214/aoms/1177732360

Zammit-Mangion A, Sainsbury-Dale M, Huser R (2025) Neural methods for amortized inference. Annual Review of Statistics and Its Application 12(1):311–335

## A Parameter sampling details

The following section is a rather terse “engineering decisions” section, and not necessary to a rough understanding of the model. Furthermore, we hope to improve the methodology below in future work. For the reader wanting only to understand the key innovations in this paper, this section could reasonably be skipped.

It should also be noted that this sampling scheme is designed for the ultimate goal to use NeuralCIs to generate confidence intervals, rather than directly p-values. With a confidence interval, one starts at the sample, and explores compatible $\psi$ values. On the other hand, with a p-value, one may choose a wildly inappropriate null $\psi _ { 0 }$ for a given sample. Our architecture does not currently employ efort to guard against such misadventure, since its goal is anyway a diferent destination.

To define $\mathcal { X } _ { \mathrm { t a r g e t } }$ in a compact way, we define a bounding box $B \subseteq \Theta$ over the parameter estimates $\hat { \pmb { \theta } }$ and define ${ \mathcal { X } } _ { \mathrm { t a r g e t } } = \{ \mathbf { x } \in { \mathcal { X } } | { \hat { \pmb { \theta } } } ( \mathbf { x } ) \in { \mathcal { B } } \}$ . In fact, we simplify each step by working in terms of θ<sup>ˆ</sup> (and which parameters produce $\hat { \pmb { \theta } }$ that overlap) rather than x.

To obtain a denormalizing flow (Section $2 . 4 )$ that can cheaply sample from $\Theta _ { \mathrm { o u t e r } }$ (as well as one that can sample cheaply from $\Theta _ { \mathrm { i n n e r } } )$ , a series of Metropolis-Hastingsstyle random walks, neural networks and denormalizing flows are fit to the sampling function:

1. Since $\hat { \pmb { \theta } }$ should be an estimator of θ, a number of MCMC chains are initialized starting at $\theta \in B$ . At each Metropolis-Hastings-style step, a large number of samples $ { \ddot { \pmb { \theta } } } (  { \mathbf { x } } )$ are drawn and an approximate bounding box, $B _ { \hat { \theta } ( \mathbf { x } ) }$ is computed as a Bonferroni-adjusted multiple of each estimate’s respective standard deviation (Bonferroni-adjusted to capture 99.5% of probability mass). For θ where $B _ { \hat { \pmb \theta } ( \mathbf { x } ) } \cap B \neq \emptyset$ , the covariance matrix of the $\hat { \pmb { \theta } } ( \mathbf { x } )$ is used to compute an approximate Fisher information, which will be used as the target distribution for the denormalizing flow in the following, while the proposal distribution for the following MCMC-jump is normal with the same covariance matrix.

2. A neural net is trained to model a surface that is a large negative number for inputs θ where $B _ { \hat { \pmb \theta } ( \mathbf { x } ) } \cap B = \emptyset$ in step 1, and to return the approximate square root Fisher information determinant from step 1 otherwise.

3. A denormalizing flow is fit to the surface from step 2. This can be used to draw unlimited samples from $\Theta _ { \mathrm { i n n e r } }$

4. Samples $\theta \in \Theta _ { \mathrm { i n n e r } }$ are drawn from the denormalizing flow fit in step 3, and $\hat { \pmb { \theta } } ( \mathbf { x } )$ values are sampled at each. A neural net is trained to produce higher values where these $\hat { \pmb { \theta } }$ values land and lower values where they do not. This is done by assigning target 1 to the x drawn from $\Theta _ { \mathrm { i n n e r } }$ and 0 to a set of samples drawn from a uniform distribution, and fitting a neural net with sum squared error<sup>7</sup>. This net then represents $\mathcal { X } _ { \Theta _ { \mathrm { i n n e r } } }$ by all $\bar { \hat { \theta } }$ at which the net returns a value higher than $^ 8 \ \epsilon = 0 . 0 1$

5. Steps 1 to 3 are repeated, except that generated $\hat { \pmb { \theta } }$ are compared not to $\boldsymbol { B }$ but instead to $\{ \pmb { \theta } ( \mathbf { x } ) : \mathbf { x } \in \mathcal { X } _ { \Theta _ { \mathrm { i n n e r } } } \}$ , according to the net in step 4. The resulting denormalizing flow is a very cheap sampler from $\Theta _ { \mathrm { o u t e r } }$

In practice, maximum likelihood estimates $\hat { \pmb { \theta } }$ can be rather expensive to compute. We allow the use of other estimators $\hat { \pmb { \theta } } ( \mathbf { x } )$ . Although these might not yield precisely the right volumes at each parameter, they should nonetheless scale in a “close-enough” way to provide “good enough” parameter sampling. The one condition that is critical, if invariances are to be incorporated (Section 5.1) is that the distribution of $\hat { \pmb { \theta } }$ is also transported by the group action.

## B Experimental sampling for Partial biserial correlation

The following are the details of the parameter sampling scheme used in the evaluation of NeuralCIs in comparison with the two likelihood-ratio techniques. Importantly, this is not the parameter sampling scheme used for training and it is rather narrow for smaller $n .$ . The reason for the narrow selection of parameters is to avoid those parameters at which the likelihood optimisation (in the baseline likelihood ratio methods) will hit a large number of boundary values.

The maximum and minimum values for each parameter are defined as follows. Firstly, defining a target rate of hitting any given “boundary” as $\epsilon = 5 \times 1 0 ^ { - 5 }$ , we compute the maximum prevalence in $\bar { A } ^ { * }$ by inverting the probability that all values in $A ^ { * }$ will equal 1:

$$
\begin{array} { r l } & { \pi _ { \operatorname* { m a x } } ( n ) = \epsilon ^ { 1 / n } , } \\ & { \pi _ { \operatorname* { m i n } } ( n ) = 1 - \pi _ { \operatorname* { m a x } } ( n ) . } \end{array}
$$

We further compute maximum and minimum values for the correlation $\rho _ { B C }$ according to the Fisher r-to-z transform:

$$
\rho _ { \mathrm { m a x } } ( n , r _ { \mathrm { m a x } } ) = \operatorname { t a n h } \left( ( z _ { \mathrm { m a x } } - z _ { \epsilon } ) \sigma _ { \mathrm { F i s h e r } } \right) ;\tag{9}
$$

$$
z _ { \mathrm { m a x } } = \frac { \mathrm { t a n h } ^ { - 1 } ( r _ { \mathrm { m a x } } ) } { \sigma _ { \mathrm { F i s h e r } } } ,\tag{10}
$$

$$
\begin{array} { r } { z _ { \epsilon } = \Phi ^ { - 1 } ( 1 - \frac { \epsilon } { 2 } ) , } \end{array}\tag{11}
$$

$$
\rho _ { \operatorname* { m i n } } ( n , r _ { \operatorname* { m a x } } ) = - \rho _ { \operatorname* { m a x } } ( n , r _ { \operatorname* { m a x } } ) ,\tag{12}
$$

$$
\sigma _ { \mathrm { F i s h e r } } = { \frac { 1 } { \sqrt { n - 3 } } } .\tag{13}
$$

where the division $\frac { \epsilon } { 2 }$ is to account for the fact that this is an approximation (even more so as it is used below).

Meanwhile, the maximum value for the biserial correlation $\rho _ { A ^ { * } C }$ is computed by treating $\rho _ { \mathrm { m a x } }$ from (9) as point-biserial and computing a conversion factor k based on

Olsson et al. (1982):

$$
\rho _ { \mathrm { m a x } } ^ { \mathrm { b i s e r i a l } } ( n , \pi _ { A ^ { \star } , r _ { \mathrm { m a x } } } ) = k \rho _ { \mathrm { m a x } } ( n , r _ { \mathrm { m a x } } / k ) ,\tag{14}
$$

$$
\begin{array} { r } { k = \frac { \sqrt { \pi _ { A ^ { * } } ( 1 - \pi _ { A ^ { * } } ) } } { \frac { e ^ { - 0 . 5 \Phi ^ { - 1 } ( \pi _ { A ^ { * } } ) ^ { 2 } } } { \sqrt { 2 \pi } } } . } \end{array}\tag{15}
$$

Finally, for the partial biserial correlation $\rho _ { A ^ { * } B }$ , we further subtract one from n:

$$
\rho _ { \mathrm { m a x } } ^ { \mathrm { p a r t i a l \ b i s e r i a l } } ( n , \pi _ { A ^ { \star } , r _ { \mathrm { m a x } } } ) = \rho _ { \mathrm { m a x } } ^ { \mathrm { b i s e r i a l } } ( n - 1 , \pi _ { A ^ { \star } , r _ { \mathrm { m a x } } } ) ,
$$

with minimum values again set as the negative of the maximum.

Since this sampling scheme tends anyway to bias samples towards the centre (particularly for low $n )$ , and also to cover a narrower region at lower $n ,$ we are satisfied to sample each parameter uniformly between its maximum and minimum values, including the sample size n.

## B.1 Targeting power

As with Behrens-Fisher problem, we also sample separate $\rho _ { A B | C } ^ { \mathrm { p o w e r } }$ values, which approximately target power $( 1 - \beta )$ uniformly sampled from $( 0 . 0 5 , 0 . { \dot { 9 } } 0 )$ , when used to assess the samples drawn under the null. Equivalently, we draw $\beta \sim \mathrm { U n i f } ( 0 . 1 0 , 0 . 9 5 )$ and:

$$
\rho _ { A B | C } ^ { \mathrm { p o w e r } } = k \operatorname { t a n h } ( z _ { \mathrm { p o w e r } } * \sigma _ { \mathrm { F i s h e r } } ) ;\tag{16}
$$

$$
z _ { \mathrm { p o w e r } } = z _ { \mathrm { n u l l } } \pm ( z _ { \alpha } + z _ { \beta } ) ,\tag{17}
$$

$$
z _ { \mathrm { n u l l } } = \frac { \mathrm { t a n h } ^ { - 1 } ( \rho _ { A B | C } ^ { \mathrm { n u l l } } / k ) } { \sigma _ { \mathrm { F i s h e r } } } ,\tag{18}
$$

$$
z _ { \alpha } = \Phi ^ { - 1 } ( 1 - \alpha / 2 ) ,\tag{19}
$$

$$
z _ { \beta } = \Phi ^ { - 1 } ( 1 - \beta ) ,\tag{20}
$$

$$
\sigma _ { \mathrm { F i s h e r } } = { \frac { 1 } { \sqrt { n - 1 - 3 } } } .\tag{21}
$$

where k is computed as in (15) and the ± in (17) is randomized (except where only one of the two is within valid range). Intuition note: dividing by σ<sub>Fisher</sub> here transforms our zs from Fisher r-to-z scale to “standard-deviation-equals-one” scale. This makes the intermediate calculations much more intuitive. Note also that in σ<sub>Fisher</sub> we use n − 1 − 3 rather than $n - 3$ , because our correlation is partial.

## C Dimension-matched transformation models and the learned scalar coordinate $z _ { \mathbf { p } }$

Let

$$
\mathbf { X } \in \mathbb { R } ^ { n _ { \theta } } , \qquad \theta = ( \psi , \lambda ) , \qquad \psi \in \mathbb { R } , \qquad \lambda \in \mathbb { R } ^ { n _ { \theta } - 1 } .
$$

The normalizing flow learns the mapping

$$
\mathbf { X } \mapsto \mathbf { Z } = ( Z _ { \mathrm { p } } , \mathbf { Z } _ { \mathrm { n } } ) ,
$$

with

$$
Z _ { \mathrm { p } } = z _ { \mathrm { p } } ( \mathbf { X } , \psi ) ,
$$

so that the scalar coordinate has no direct nuisance input. The remaining coordinate

$$
{ \bf Z } _ { \mathrm { n } } = { \bf z } _ { \mathrm { n } } ( { \bf X } , \theta )
$$

may depend on the full parameter and absorbs the remaining $( n _ { \theta } - 1 )$ directions.

In this section, we examine the link in regular, full-rank, dimension-matched $( n _ { \mathbf { x } } =$ $n _ { \pmb { \theta } } )$ transformation models, between $z _ { \mathrm { p } }$ and the profile likelihood ratio (LR). In Section $\mathrm { C . 1 }$ , we review the classical result that, in such transformation models, the profile likelihood ratio is a function of the maximal invariant. Then in Section C.2, we argue that $z _ { \mathrm { p } }$ can reasonably be expected to land very close to this same maximal invariant, thereby closely linking $z _ { \mathrm { p } }$ to the profile likelihood ratio in such models.

## C.1 Profile LR is a function of maximal invariant

The link between the profile LR and the maximal invariant in a transformation model is well established; see, for example, Barndorf-Nielsen and Jupp (1988). Here we present a simplified derivation, focussed on the particular case we have in mind for our architecture. The core intuition is that transforming our data and parameters by g multiplies both the numerator and denominator of the profile LR by the same Jacobian factor, which then cancels; so the profile LR itself is invariant under the group action and it must be a function of the maximal invariant.

Assume the model is a transformation model. An $n _ { \pmb { \theta } } .$ -dimensional Lie group $G$ acts smoothly on the data space and parameter space:

$$
\mathbf { x } \mapsto g \cdot \mathbf { x } , \qquad \pmb \theta \mapsto g \cdot \pmb \theta .
$$

Assume the interest parameter is equivariant:

$$
g \cdot \psi ( \pmb \theta ) : = \psi ( g \cdot \pmb \theta ) .
$$

Thus G acts jointly on pairs $( { \bf x } , \psi )$ by

$$
( \mathbf { x } , \psi ) \mapsto ( g \mathbf { x } , g \psi ) .\tag{22}
$$

Assume the density is invariant up to the Jacobian factor ${ j } _ { g } ( \mathbf { x } )$

$$
\begin{array} { r } { p _ { g \pmb { \theta } } ( g \mathbf { x } ) = j _ { g } ( \mathbf { x } ) p _ { \pmb { \theta } } ( \mathbf { x } ) , } \end{array}
$$

where ${ j _ { g } ( \mathbf { x } ) > 0 }$ is the Jacobian factor and does not depend on θ. Finally assume the joint action (22) on $( \mathbf { x } , \psi )$ is regular and full rank. Since

$$
\dim ( \mathbf { x } , \psi ) = n _ { \pmb { \theta } } + 1
$$

and

$$
\dim G = n _ { \theta } ,
$$

the quotient is locally one-dimensional. Let

$$
m _ { \psi } ( { \bf x } )
$$

denote a scalar maximal invariant of the joint action (22).

Define the profile likelihood

$$
L _ { p } ( \psi ; { \bf x } ) = \operatorname* { s u p } _ { \pmb { \theta } : \psi ( \pmb { \theta } ) = \psi } p \pmb { \theta } ( { \bf x } ) ,
$$

and the profile likelihood ratio

$$
\Lambda ( \psi ; { \bf x } ) = \frac { L _ { p } ( \psi ; { \bf x } ) } { \operatorname* { s u p } _ { \theta } p _ { \theta } ( { \bf x } ) } .
$$

Then

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { p } ( g \psi ; g { \bf x } ) = \operatorname* { s u p } _ { \theta ^ { \prime } \colon \psi ( \theta ^ { \prime } ) = g \psi } p _ { \theta ^ { \prime } } ( g { \bf x } ) } } \\ { ~ = \operatorname* { s u p } _ { \theta : \psi ( \theta ) = \psi } p _ { g \theta } ( g { \bf x } ) } \\ { ~ = \operatorname* { s u p } _ { \theta : \psi ( \theta ) = \psi } j _ { g } ( { \bf x } ) p _ { \theta } ( { \bf x } ) } \\ { ~ = j _ { g } ( { \bf x } ) L _ { p } ( \psi ; { \bf x } ) . } \end{array}
$$

Similarly,

$$
\operatorname* { s u p } _ { \theta ^ { \prime } } p _ { \theta ^ { \prime } } ( g \mathbf { x } ) = j _ { g } ( \mathbf { x } ) \operatorname* { s u p } _ { \theta } p _ { \theta } ( \mathbf { x } ) .
$$

(That is, for fixed $^ { g , }$ we find the supremum across all $\pmb \theta . )$

Therefore the Jacobian factor cancels in the likelihood ratio:

$$
\Lambda ( g \psi ; g { \bf x } ) = \Lambda ( \psi ; { \bf x } ) .
$$

So $\Lambda$ is invariant under the joint group action on $( { \bf x } , \psi )$ . Since $m _ { \psi } ( { \bf x } )$ is maximal invariant,

$$
\Lambda ( \psi ; { \bf x } ) = h ( m _ { \psi } ( { \bf x } ) ) ,
$$

for some scalar function h.

## C.1.1 Gaussian location and other examples

For the simple Gaussian location model, $\mathbf { X } \sim { \mathcal { N } } ( { \boldsymbol { \theta } } , { \boldsymbol { \Sigma } } )$ , with fixed covariance matrix ${ \pmb { \Sigma } } = { \bf I } ^ { - 1 }$ , the unrestricted MLE of θ is $\hat { \pmb { \theta } } = \mathbf { X }$

$$
\mathbf { X } = \hat { \pmb { \theta } } = \left( \hat { \psi } \atop \hat { \lambda } \right) \sim \mathcal { N } \left( \left( \begin{array} { l } { \psi } \\ { \lambda } \end{array} \right) , \left( \mathbf { I } _ { \psi \psi } \ \mathbf { I } _ { \psi \pmb { \lambda } } \right) ^ { - 1 } \right) .
$$

The group action is a translation

$$
\left( \begin{array} { c } { { \hat { \psi } } } \\ { { \hat { \Delta } } } \\ { { \psi } } \end{array} \right) \mapsto \left( \begin{array} { c } { { \hat { \psi } + a _ { \psi } } } \\ { { \hat { \lambda } + { \bf a } _ { \lambda } } } \\ { { \psi + a _ { \psi } } } \end{array} \right) ,
$$

with a simple maximal invariant

$$
m _ { \psi } = \hat { \psi } - \psi
$$

and quadratic log likelihood surface

$$
\begin{array} { r } { \ell ( \pmb \theta ; \hat { \pmb \theta } ) - \hat { \ell } = - \frac 1 2 \big ( \pmb \theta - \hat { \pmb \theta } \big ) ^ { \top } \mathbf { I } ( \pmb \theta - \hat { \pmb \theta } ) , } \end{array}
$$

where $\hat { \ell } = \ell (  { \hat { \theta } } ;  { \hat { \theta } } )$ is a constant (independent of θ) and represents the central (peak) log likelihood value at $\pmb \theta = \hat { \pmb \theta }$

Writing also $\mathbf { d } _ { \lambda } = \hat { \lambda } - \lambda$ , then the profile log likelihood $\ell _ { p } ,$ and thereby the log likelihood ratio log $\Lambda ( \psi )$ , is obtained by a simple optimization across this quadratic,

$$
\begin{array} { r l } { - 2 \log \Lambda ( \psi ) = } & { } \\ { - 2 \left( \ell _ { p } ( \psi ; \hat { \psi } ) - \hat { \ell } _ { p } \right) = \displaystyle \operatorname* { m i n } _ { \lambda } ~ ( \theta - \hat { \theta } ) ^ { \top } \mathbf { I } ( \theta - \hat { \theta } ) } \\ & { = \displaystyle \operatorname* { m i n } _ { \lambda } ~ \left( \frac { m _ { \psi } } { \mathbf { d } _ { \lambda } } \right) ^ { \top } \left( I _ { \lambda \upsilon } ~ \mathbf { I } _ { \lambda \lambda } \right) \left( \frac { m _ { \psi } } { \mathbf { d } _ { \lambda } } \right) } \\ & { = \displaystyle \operatorname* { m i n } _ { \mathbf { d } _ { \lambda } } ~ \left[ I _ { \psi \psi } m _ { \psi } ^ { 2 } + 2 m _ { \psi } \mathbf { I } _ { \psi \lambda } \mathbf { d } _ { \lambda } + \mathbf { d } _ { \lambda } ^ { \top } \mathbf { I } _ { \lambda \lambda } \mathbf { d } _ { \lambda } \right] } \\ & { = m _ { \psi } ^ { 2 } \left( I _ { \psi \psi } - \mathbf { I } _ { \psi \lambda } \mathbf { I } _ { \lambda \lambda } ^ { - 1 } \mathbf { I } _ { \lambda \psi } \right) } \\ & { = m _ { \psi } ^ { 2 } I _ { \psi \lambda } } \\ & { = ( \psi - \hat { \psi } ) ^ { 2 } I _ { \psi \lambda } . } \end{array}
$$

where $\hat { \ell } _ { p } = \ell _ { p } ( \hat { \psi } ; \hat { \psi } )$ is again a constant and $I _ { \psi \cdot \lambda }$ is the eficient information, which is constant across this quadratic log likelihood function.

This simple transformation model illustrates the profile log likelihood ratio as a function of the maximal invariant $m _ { \psi }$ of $( \hat { \pmb { \theta } } = { \bf x } , \psi )$

## C.2 Why we think $z _ { \mathbf { p } }$ will tend towards $m _ { \psi }$

A Gaussianization of the maximal invariant $m _ { \psi }$ is a very natural target for $z _ { \mathrm { p } } .$ . Define $G _ { n }$ to be the $( n _ { \theta } - 1 )$ )-dimensional nuisance subgroup of G that leaves $\psi$ stationary. Then $m _ { \psi }$ defines a foliation of X such that probability mass is only moved by $G _ { n }$ within each given leaf, and never between. The foliation provides the perfect structure to model with $\mathbf { z } _ { \mathrm { n } } ,$ leaving the distribution across the leaves $p ( m _ { \psi } )$ , which is a pivot independent of $\lambda$

In this section we will consider whether there are other viable targets for $z _ { \mathrm { p } }$ Transform X at fixed $\psi = \psi _ { 0 }$ as

$$
\mathbf { X } _ { ( \psi _ { 0 } , \lambda ) }  ( M , \mathbf { U } _ { \lambda } ) , \qquad M = m _ { \psi _ { 0 } } ( \mathbf { X } _ { ( \psi _ { 0 } , \lambda ) } ) , \qquad \mathbf { U } _ { \lambda } \in \mathbb { R } ^ { n _ { \mathbf { x } } - 1 } ,
$$

and correspondingly overload

$$
z _ { \mathrm { p } } ( { \bf x } ; { \psi } _ { 0 } )  z _ { \mathrm { p } } ( m , { \bf u } ) ; \qquad { \bf x }  ( m , { \bf u } ) .
$$

We want to know if a $z _ { \mathrm { p } } ( M , \mathbf { U } _ { \lambda } )$ that depends not only on $M$ , but also on $\mathbf { U } _ { \lambda }$ can still be standard normal for all λ.

The short answer is: yes, there are sometimes loss-optimal forms of $z _ { \mathrm { p } }$ that depend not just on m but also on u, but all require considerable coordination of cancellations across u-space that is unlikely to arise in any substantial form from the combination of a random initialisation and stochastic gradient descent optimisation.

In Section C.2.1, we start by assuming that $z _ { \mathrm { p } } ( m , \mathbf { u } )$ is monotone increasing in m. In this case, any dependence on u must be removed in expectation as λ slides $\mathbf { U } _ { \lambda }$ around space, with any u-dependence lost at the back end of the distribution, counterbalanced by a coordinated opposite dependence at the front end of the λ- induced movement of the probability mass.

We give a simple two-dimensional example (that is, with scalar $U _ { \lambda } ) \colon M \bot U _ { \lambda } , U _ { \lambda } \sim$ $\mathrm { U n i f } ( \lambda , \lambda + 1 )$ . In this case, a dependence on $U _ { \lambda }$ that is periodic (with period 1) will average out to a constant across the uniform $U _ { \lambda }$ at all λ. This clearly shows the cancellation concept: as the uniform distribution is moved along, the portion of $z _ { \mathrm { p } }$ that is no longer supported is replaced by exactly the same shape in a new region of u that previously had not been supported. The loss is blind to such a specific, coordinated u-dependence and will neither encourage nor discourage it: without the appearance of such a coherent, coordinated pattern in the random initialization, it can only be arrived at during training by random drift in that direction. But the component of any random drift in such an extremely specific direction could be expected to be negligible, as could the probability of it appearing in any significant proportion in the random initialization in the first place.

In Section C.2.2 we then explore the possibility that $z _ { \mathrm { p } }$ is non-monotone in $m .$ . This would make it possible for pairs of local u-dependencies to cancel each other, without needing the same level of global coordination. However, for $z _ { \mathrm { p } }$ to be non-monotone in m, satisfying our architectural constraint $\frac { \partial z _ { \mathrm { p } } } { \partial \psi } < 0$ in (3) while the full map still remains a difeomorphism, itself requires similarly complex, large-scale coordinated patterns. These are, similarly, unlikely to arise from our random initialization and training.

So, in summary, while it is technically possible for $z _ { \mathrm { p } }$ to stray far from the maximal invariant $m _ { \psi }$ , it would require accidental construction of such high levels of coordination in the random initialization, that we consider it highly unlikely, except perhaps for a very small component. Under the simple independent-noise benchmark in Section C.2.3, a small contamination like this should have a very mild efect on power (only second-order in the contamination amplitude).

## C.2.1 Dependence on U that disappears after marginalization

Assume that $z _ { \mathrm { p } } ( m , \mathbf { u } )$ is strictly increasing in m for each fixed u. Define the inverse of $z _ { \mathrm { p } }$ at each u as $\widetilde { m _ { z } } ( \mathbf { u } )$ according to

$$
z _ { \mathrm { p } } ( \widetilde { m _ { z } } ( \mathbf { u } ) , \mathbf { u } ) = z .
$$

We know that one excellent target for $z _ { \mathrm { p } } ( M , \mathbf { U } _ { \lambda } )$ would be a Gaussianization of M, call it $z _ { \mathrm { p } } ^ { * }$ . In this case the inverse is independent of u, call it $\overline { { m } } _ { z }$ :

$$
z _ { \mathrm { { p } } } ^ { * } ( \overline { { m } } _ { z } , { \bf { u } } ) = z ,
$$

for all u and z.

In order that $z _ { \mathrm { p } }$ be pivotal, we need that $z _ { \mathrm { p } } ( M , \mathbf { U } _ { \lambda } )$ have the same distribution as $z _ { \mathrm { p } } ^ { * } ( M , \mathbf { U } _ { \lambda } )$ for all λ. If we think of $\overline { { m } } _ { z }$ for any fixed z as a boundary that is straight in u, and $\widetilde { m _ { z } }$ as a boundary that is wiggly in u, they must cross each other so that $\widetilde { m _ { z } }$ can exchange increased probability mass in one region of u for decreased probability mass in another. We can represent these regions of gained or lost probability mass as regions of ±1 (with zero everywhere else),

$$
q _ { z } ( m , { \mathbf { u } } ) = { \mathbf { 1 } } \{ m < \widetilde { m _ { z } } ( { \mathbf { u } } ) \} - { \mathbf { 1 } } \{ m < \overline { { m } } _ { z } \} ,
$$

and pivotality requires that the same amount of probability is always exchanged between positive and negative regions,

$$
\begin{array} { r } { \mathbb { E } [ q _ { z } ( M , { \bf U } _ { \lambda } ) ] = 0 , } \end{array}
$$

for all z and λ.

The final bolded part, that this must hold for all z and λ is the critical part. Since our group acts in a full-rank way on the data, $\mathbf { U } _ { \lambda }$ is constantly moved around space by λ, and certain regions gain probability mass, while others lose it. These two must be constantly ofset and the result is that any dependence on u (if a feasible one should exist at all) must be highly globally structured across space.

This argument does not depend on the scale of the m-component, since m is defined only up to a monotone transformation. It may be made arbitrarily weak while still requiring the same cancellation in u-components as λ moves probability mass around space. In the limiting case where $z _ { \mathrm { p } }$ becomes independent of m altogether, the λ- induced motion of $\mathbf { U } _ { \lambda }$ in all directions would demand the same globally coordinated cancellations, to maintain pivotality at all λ, if such a construction can be made pivotal at all.

The argument is most easily visualized with an example. Suppose that $\mathbf { U } _ { \lambda } = U _ { \lambda }$ is scalar, M⊥U under each scalar λ and

$$
U _ { \lambda } \sim \mathrm { U n i f } ( \lambda , \lambda + 1 ) .
$$

If we choose a wiggly boundary $\widetilde { m _ { z } }$ such that, when fed through the CDF $F _ { M }$ of $M .$ it generates a sine wave in u,

$$
F _ { M } ( \widetilde { m _ { z } } ( u ) ) = F _ { M } ( \overline { { m } } _ { z } ) + a _ { z } \sin ( 2 \pi n u ) ,
$$

at any integer frequency $n \in \mathbb { Z } ^ { + }$ and any set of amplitudes $a _ { z } ,$ , such that the right-hand side remains in [0, 1] and increasing in z, then

$$
\begin{array} { r l } & { P ( z _ { \mathrm { p } } ( M , U _ { \lambda } ) < z ) = \mathbb { E } [ F _ { M } ( \widetilde { m _ { z } } ( U _ { \lambda } ) ) ] } \\ & { \quad \quad \quad = F _ { M } ( \overline { { m } } _ { z } ) + a _ { z } \mathbb { E } [ \sin ( 2 \pi n U _ { \lambda } ) ] } \\ & { \quad \quad \quad = F _ { M } ( \overline { { m } } _ { z } ) . } \end{array}
$$

This example is a very clean illustration of the broader point: as the uniform distribution is translated along u by λ, the wiggle that it leaves behind at the back must be compensated by another picked up at the front. In this case, that means that the dependence on u must be periodic (with period 1) across the whole of u.

As the example shows, such solutions may exist, but they also require global coordination of the dependence on u. If some such globally coordinated dependence is already present in the random initialisation, the KL loss may have no incentive to remove it. However, this is highly unlikely to form more than a very small part of the initial u-dependence. Most randomly initialized u-dependencies will fail to produce the required cancellations for all z and λ, and will therefore be ironed out during training.

The loss itself is ambivalent to such structures, so training may allow a random walk in that direction, but we could expect in most cases that the component of any random drift that is directed so specifically in that direction should be negligible. Furthermore, against the ambivalence of the loss, the architecture itself may well prefer a simpler solution (flat in u).

## C.2.2 Folding in M

If $z _ { \mathrm { p } } ( m , \mathbf { u } ; \psi )$ is not monotone in m, then, for fixed u, the set

$$
\{ m : z _ { \mathrm { p } } ( m , \mathbf { u } ; \psi ) \leq z \}
$$

may contain several disjoint intervals. Probability gained on one branch by a detour of the $z _ { \mathrm { p } } .$ -contour through u can then be ofset by probability lost on another, while $\mathbf { z } _ { \mathrm { n } }$ retains the information needed to keep the full transformation invertible.

Such a fold cannot stand alone. Since the full map

$$
( m , \mathbf { u } ) \mapsto ( z _ { \mathrm { p } } , \mathbf { z } _ { \mathrm { n } } )
$$

is a difeomorphism, the contours of $z _ { \mathrm { p } }$ are smooth, non-intersecting surfaces: they cannot terminate or cross, but rather form continuous “snakes” in space, enclosing volumes that encode how the model is distributed. A turning point $\begin{array} { r } { { \frac { \partial z _ { \mathrm { p } } } { \partial m } } \ = \ 0 } \end{array}$ can therefore only occur where the contour runs out through u, turns back and returns at a diferent value of m. A single fold has a ripple efect, requiring a coordinated family of “snaking” contours over at least a substantial neighbourhood, to maintain a difeomorphism that still captures the model densities.

Furthermore, our architectural constraint

$$
\frac { \partial z _ { \mathrm { p } } ( \mathbf { x } ; \psi ) } { \partial \psi } < 0
$$

from (3) is evaluated at fixed x: increasing ψ moves every point downward in $z _ { \mathrm { p } } .$ Write $Z _ { \mathrm { p } } = z _ { \mathrm { p } } ( \mathbf { X } ; \psi )$ . Then calibration $Z _ { \mathrm { p } } \sim \mathcal { N }$ must be preserved, while the data distribution itself moves under the infinitesimal interest action $\mathbf { V } _ { \psi } = \mathbf { v } _ { \psi } ( \mathbf { X } ; \psi , \pmb { \lambda } )$ Along this movement, we require that the flux (denoted A below) of probability density into and out of a given $Z _ { \mathrm { p } }$ level set, as ψ is tweaked, must be tracked/counterbalanced by a shift in the value of $Z _ { \mathrm { p } }$ (denoted B):

$$
\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } \psi } P _ { \psi , \lambda } ( Z _ { \mathrm { p } } \leq z ) = - \phi ( z ) \mathbb { E } _ { \psi , \lambda } \left[ \frac { \mathrm { d } } { \mathrm { d } \psi } Z _ { \mathrm { p } } \bigg | Z _ { \mathrm { p } } = z \right] \qquad } \\ { = - \phi ( z ) \mathbb { E } _ { \psi , \lambda } \left[ \underbrace { \nabla _ { \mathbf { x } } Z _ { \mathrm { p } } ^ { \top } \mathbf { V } _ { \psi } } _ { A } + \underbrace { \frac { \partial Z _ { \mathrm { p } } } { \partial \psi } } _ { B } \bigg | Z _ { \mathrm { p } } = z \right] = 0 } \end{array}\tag{23}
$$

for all $z , \lambda$ and $\psi .$

Because the interest transformation is not entirely contained within the nuisance orbits, its motion $\mathbf { V } _ { \psi }$ has a non-zero component in the m-direction; we choose the orientation of m so that

$$
\nabla _ { \mathbf x } m ^ { \top } \mathbf v _ { \psi } > 0 .
$$

In the case where $z _ { \mathrm { p } }$ is strictly increasing in m, there can be a natural pointwise cancellation between terms A and B in (23). On the other hand, wherever $z _ { \mathrm { p } }$ is decreasing in $m ,$ the m-component of term A reinforces term $B _ { : }$ which must then be cancelled by motion in u or by coordination with other m that are mapped to the same $z _ { \mathrm { p } }$

A folded solution is therefore not impossible and it can in principle hide udependencies from our loss, since gains and losses associated with diferent m sharing the same $z _ { \mathrm { p } }$ can be cancelled. But it must coordinate the geometry of an entire family of contours with compensating motion in u or on other branches, consistently across z, λ and $\psi .$ This is then (as with Section C.2.1) a highly organized construction, rather than a local perturbation, that random initialization and gradient training would naturally struggle to produce.

## C.2.3 Consequence of contamination for power

Our conclusion is that there may be a very small u-dependent component in $z _ { \mathrm { p } } { \mathrm { : } }$ which may be rather uninformative about ψ; indeed it may be pure noise. But there is unlikely to be a larger one. To illustrate the likely scale of this efect, we consider below the simple benchmark in which the contaminating component is independent standard normal noise carrying no ψ-signal. We show that noncentrality in this case is attenuated only quadratically with the amplitude of the mixed-in noise.

First, consider the canonical $Z _ { \mathrm { p } } ^ { * }$ , which depends only on M. Suppose that under the alternative hypothesis, $Z _ { \mathrm { p } } ^ { * }$ is translated by $\delta _ { \psi } \mathbf { : }$

$$
Z _ { \mathrm { p } } ^ { * } \sim \mathcal { N } ( \delta _ { \psi } , 1 ) ,
$$

so that $\delta _ { \psi }$ is the noncentrality of $Z _ { \mathbf { p } } ^ { * }$

Suppose we mix in some U-derived standard normal noise $W \sim { \mathcal { N } } ( 0 , 1 )$ , such that $W \bot Z _ { \mathrm { p } } ^ { \ast }$ and $W$ carries no ψ-signal at all, then

$$
Z _ { \mathrm { p } } = \frac { Z _ { \mathrm { p } } ^ { * } + \epsilon W } { \sqrt { 1 + \epsilon ^ { 2 } } } \sim \mathcal { N } \left( \frac { \delta _ { \psi } } { \sqrt { 1 + \epsilon ^ { 2 } } } , 1 \right) .
$$

So the non-centrality of $Z _ { \mathrm { p } }$ itself is divided by the denominator $\sqrt { 1 + \epsilon ^ { 2 } }$ . Or, expanding around $\epsilon = 0$ , its non-centrality is multiplied by

$$
\begin{array} { r } { 1 - \frac { 1 } { 2 } \epsilon ^ { 2 } + \mathcal { O } ( \epsilon ^ { 4 } ) . } \end{array}
$$

So the loss of noncentrality, and hence the resulting loss of power (which is a smooth function thereof), is only second-order in ϵ.

## D Further results

Here we present further results to support and further develop those presented in the main sections.

## D.1 t-Test: the edges of $\Theta _ { \mathrm { i n n e r } }$

In Figure 8, we compare the results for the t-test model with no canonicalization, across a number of diferent parameter testing regions. Top left reproduces Figure 1 from the main text, where parameters were drawn only from the estimates bounding box B. Of course, our goal is to generate valid p-values for all $\pmb { \theta } \in \Theta _ { \mathrm { i n n e r } } \supset B$

To test performance outside of the bounding box, we draw a Bonferroni-adjusted parameter sampling quadrilateral at each $n ,$ that extends beyond the estimates box. At a given $n ,$ consider the two-dimensional space $( \mu , \sigma )$ , on which the estimates box is drawn. At each point in this space, we compute a separate $p \mathrm { - }$ value in each dimension $( \mu$ and σ), taking the $( \mu , \sigma )$ coordinates at that point as the respective nulls, and the closest point on the bounding box as the estimates $( m , s ) = ( { \hat { \mu } } , { \hat { \sigma } } )$ . These $n _ { \theta } = 2$ p-values are then combined by Bonferroni. Those with a p-value greater than oneminus-confidence-level are allowed for inclusion in the plot, and the first 50000 in each case are retained for plotting.

![](images/d28ef71146ea57cb328c5f1347274aabf27d7324a537c916959f4ccb3e66f80f.jpg)  
Fig. 8 Comparison of Figure 1 (parameters drawn from the estimates bounding box) with results obtained when extending the parameter sampling distribution into the edges of $\Theta _ { \mathrm { i n n e r } } .$ Other plots draw parameters from a union of Bonferroni confidence regions computed around the boundary points of the bounding box; percentages in individual plot titles represent the confidence levels of these confidence regions.

It is striking that the larger sample sizes perform relatively well at all confidence levels, where the small n cases perform progressively worse towards the edge of the box. Our current hypothesis is that the current sampling scheme has two weak-points with regard to parameters with wider spread estimates (as is the case with low n in the t-test example):

1. Step 4 in Appendix A, which models the distribution of samples that can be drawn from $\Theta _ { \mathrm { i n n e r } } ,$ is based efectively on the absolute height of the probability density, which is much lower for a parameter with wider spread estimates (e.g. large σ at small n in the t-test). This leads to such parameters being sampled less broadly than those with a tighter sampling distribution. We might remedy this by making use of the net trained in step 2, which encodes the inverse of the volume occupied by the estimates and is a proxy for the height of the density.

2. The current sampling scheme, based on the Jefreys prior, only takes account of the volume of the sampling distribution at that parameter, but does not take account of the relative volume of space that it can fill. If either of B or Θ<sub>inner</sub> is large compared to the spread of the sampling distribution of the estimates, our parameter sampling will overweight parameters with more tightly spread estimates, potentially catastrophically. We might remedy this by weighting each parameter relative to probability of collision of the sampling distribution with B or $\mathcal { X } _ { \Theta _ { \mathrm { i n n e r } } }$ , rather than (as at present) the probability of the collision of the sampling distribution with a single point.

Point 2 is more easily understood with a concrete example. In this t-test example, we do need a higher probability of drawing small σ values, since their sampling distributions reach less far through space, and so a smaller volume of them overlap the bounding box; with equal sampling probability, they would then be underweight.

If the region we were aiming to overlap with were a single point, the Jefreys prior would correct for this exactly: we would pull the same proportion of small σ parameters as large σ. But as we extend the bounding box outwards, these up-weighted small-σ values start to dominate, as the same amount of space inside the bounding box can accommodate far more small-σ parameter samples than large-σ ones. This disproportionately impacts small sample sizes, because, particularly for $n = 3$ , the sample estimate s is quite noisy and so a comparatively huge range of $\sigma$ values must be modelled.

We are developing an improved sampling scheme, that takes account of the size of the estimates box B when weighting $\Theta _ { \mathrm { i n n e r } }$ , and of the size of $\Theta _ { \mathrm { i n n e r } }$ when weighting $\Theta _ { \mathrm { o u t e r } }$ , as well as attempting to address point 1 through the use of the modelled Cholesky determinants.

## D.2 Non-converged, boundary and degenerate cases (partial biserial correlation)

In Figures 9 and 10, we plot, across the one thousand parameter samples, convergence, boundary and degenerate sample rates, each computed on either one million main or ten thousand size-adjust simulations. Non-convergence (Figure 9) is broken down by each of the three types of likelihood fitted. The three boundary and degenerate sample plots in Figure 10 count cases of the following, respectively:

1. At least one correlation MLE lay suficiently close to the artificially imposed boundary at $| \hat { \rho } ^ { \mathrm { M L E } } | = 0 . 9 9$ that it may have been distorted. We define this as any sample with final $\lvert \hat { \rho } ^ { \mathrm { M L E } } \rvert > 0 . 9 8 8$

2. The prevalence MLE lay suficiently close to the artificially imposed boundaries at 0.005 and 0.995, that its likelihood may have been distorted. After accounting for the transform of prevalences onto the range (−1, 1), we define this similarly as any sample with final $| 2 \hat { \pi } _ { A ^ { * } } ^ { \mathrm { M L E } } - 1 | > 0 . 9 8 8$

3. All binary $A ^ { * }$ values were of the same value, zero or one. That is to say, ${ \hat { \pi } } _ { A ^ { * } } \in$ {0, 1}. In this case, the biserial correlation is undefined.

The cutof of 0.988 in step 1 was defined conservatively, based on Figure 11. Quite visible in this plot is a small build-up of samples just inside the 0.99 boundary. That is to say, not all boundary points converged precisely to 0.99. Choosing a threshold of 0.988, then, should be conservative, as this is quite far from the apparent build-up and also includes a number of samples that truly converged above 0.988, without ever testing the boundary.

![](images/f5f03b44c4c39bfd6b0a7322b8c6835d44d3575b4e8ad8075d7404980843c0e5.jpg)  
Fig. 9 Rates of convergence failures out of one million main samples and ten thousand size-adjust samples at each likelihood fit: respectively, nonconvergence rates in unconstrained MLE, MLE constrained to $H _ { 0 } : \rho _ { A B | C } = \rho _ { A B | C } ^ { \mathrm { n u l l } }$ and MLE constrained to $H _ { 1 } : \rho _ { A B | C } = \rho _ { A B | C } ^ { \mathrm { p o w e r } } .$

![](images/bb456f1e77544ac0d16922bdefe12ca455ff87065d4b3c6b685a389529a772e9.jpg)  
Fig. 10 Rates of boundary and degenerate samples out of one million main samples and ten thousand size-adjust samples per parameter sample: respectively, proportion of boundary samples for correlations and prevalences, followed by degenerate samples (those whose A<sup>∗</sup> has all the same values).

![](images/ebfea623059f26770835dd2cb499faea21f76dd92832cf359f9864d2f8f85b04.jpg)  
Fig. 11 Close-to-boundary absolute values of maximum likelihood estimates pooled across all correlations fitted by maximum likelihood on all runs. Note that this is very much zoomed in: the y-axis here is severely cropped (the maximum count in this plot is 200, vs 25340 if the y-limits were left free).

In Figure 12, we summarize the convergence rates across the bootstrap samples generated to convert them into p-values in the bootstrap LR. When bootstrapping, we have 500 parameter samples (index them by i), each made up of 500 p-values v, each in turn made up of 2000 bootstrap profile LR samples b. To summarize these, Figure 12 plots

$$
x _ { i } = \mathrm { S u m m a r y } _ { v } ( \mathbb { E } _ { b } [ 1 - \mathrm { c o n v e r g e d } _ { i v b } ] )
$$

![](images/d7fdf52c7213ebccc8a76cdaf7c269708338cb649133d7d89e1f31ea686134ef.jpg)  
Fig. 12 Summaries within each parameter sample across the p-values generated at each parameter sample of the non-convergence rates of the bootstrap samples within that p-value. For example, each datapoint on the ${ } ^ { \mathfrak { s } } \mathrm { { M e a n } } ^ { \mathfrak { s } }$ histogram is for a single parameter sample, the mean across all p-values generated for that parameter sample of the non-convergence rate across all bootstrap samples for the respective p-value.

for $i \in \{ 1 \dots 5 0 0 \}$ and Summary ∈ {Mean, 99th Percentile, Max}.

## D.3 Deviation from the maximal invariant

In Figure 13 we present the efect of further training the seven slower converging runs from Section 7.4 (that is, the coloured curves in Figure 7) for a fresh set of 500 epochs of 100 steps. Once the training runs find their way into the right direction, they converge as neatly as the other 243 runs did in the first set of training epochs. The further epochs (starting at epoch 500) start again from the initial higher learning rate.

![](images/86810411bce392c306f0d618ea83ea1c3a59f00b89b6e90b2b910d10d4421f83.jpg)

![](images/f0f486e6718c59c5776a5645f00e795e7865d9723a5e26083ebac473624bf974.jpg)

![](images/43f4ae14ad2d13c2aff2baa2e96670b3db149863ddfa71f0a9b5828254ca68f8.jpg)

![](images/617426ae18194557506d4c0bab30abd0b40b7283d3942382bd39e6da2502ddab.jpg)  
Fig. 13 Training the coloured runs from Figure 7 for a fresh set of 500 epochs of 100 steps.