# On the Dificulty of Sequence-Level Supervised Learning for Autoregressive Language Models

Mathieu Blondel<sup>1</sup>, Michaël E. Sander<sup>1</sup>, Germain Vivier-Ardisson<sup>1</sup>, Tianlin Liu<sup>1</sup> and Vincent Roulet<sup>1</sup> <sup>1</sup>Google DeepMind

During the supervised learning phase, autoregressive language models are typically trained by minimizing the negative log-likelihood of the training data, which decomposes as a sum of token-level cross-entropies via teacher forcing. This objective is simple and eficient but potentially sufers from so-called exposure bias: a mismatch between the way the model is trained and the way it is used at inference time. Can we instead design sequence-level loss functions that incorporate inference-time decoding procedures, such as greedy decoding or autoregressive sampling? We show that two natural attempts lead to loss functions that are fundamentally flawed: unbounded below. We further show that these two flawed loss functions are intimately connected: using autoregressive Gumbel tricks, the sampling based loss is the expected greedy loss under Gumbel perturbations. Since the pathology stems from the dependence of the decoded sequence on the model parameters, we analyze the update direction obtained by applying a stop gradient on the sampling procedure, leading to a moment matching direction. While it is not the gradient of any loss function, we prove that it defines a valid descent direction in function space for the autoregressive negative log-likelihood. Despite these theoretical insights, the moment matching direction sufers from high variance. We conclude that teacher forcing may well be “all you need” for supervised learning of autoregressive models.

## 1. Introduction

During the supervised learning phase, autoregressive (AR) language models are typically trained by minimizing the negative log-likelihood of the training data, which decomposes as a sum of token-level cross-entropies. During training, each token is predicted given the ground-truth prefix, a technique known as teacher forcing (Williams and Zipser, 1989).

While simple and computationally eficient (a single forward pass, no sampling), teacher forcing potentially sufers from exposure bias (Ranzato et al., 2015; Bengio et al., 2015): at training time, the model conditions on ground-truth prefixes, but at inference time it conditions on its own previous predictions, leading to error accumulation.

These limitations have motivated work on sequence-level training objectives. Existing eforts include policy-gradient (Ranzato et al., 2015; Rennie et al., 2017) and minimum risk training (Shen et al., 2016; Norouzi et al., 2016) methods. However, these methods all rely on external reward signals. In the structured prediction literature, designing loss functions for structured outputs has been studied extensively. The perceptron loss (Collins, 2002) requires an exact maximization oracle, while negative log-likelihood training of conditional random fields and more generally energy-based models (LeCun et al., 2006) requires a log-sum-exp (log-partition) oracle or a MCMC sampler. Unfortunately, such oracles or samplers are intractable over the space of sequences.

Can we instead design sequence-level loss functions that incorporate inference-time decoding procedures, such as greedy decoding or autoregressive sampling? In this paper, we make the following contributions.

• After reviewing some background (Section 2), we show that the sequence-level loss functions

associated with greedy decoding and autoregressive sampling are unbounded below (Section 3).   
These loss functions can therefore not be used for learning, as the loss would diverge to −∞.

• We connect greedy decoding and sampling through autoregressive Gumbel tricks, showing that the sampling based loss is the expected greedy loss under Gumbel perturbations (Section 4).

• Since the pathology of these loss functions stems from the dependence of the decoded sequence on the model parameters, we then analyze the moment matching direction obtained via a stop gradient on the sampling procedure: it is a surrogate gradient (not the gradient of any loss), yet a valid descent direction for the AR negative log-likelihood (Section 5) in function space. Unfortunately, such direction sufers from high variance.

• We conclude that teacher forcing may well be “all you need” for supervised learning of AR models.

Notation. We follow the same notation as (Blondel et al., 2025). We define the space of input sequences as � and the space of output sequences as �. For notational simplicity, we assume $\mathcal { V } : = \mathcal { V } ^ { T }$ where � is a fixed length and $\mathcal { V } : = \{ 1 , \ldots , V \}$ . For a detailed discussion on variable-length sequence handling, see (Blondel et al., 2025). Given a sequence $\pmb { y } = ( y _ { 1 } , \dots , y _ { T } ) , \pmb { y } _ { < t }$ denotes the sequence $\left( y _ { 1 } , \ldots , y _ { t - 1 } \right)$ and for convenience $\pmb { y } _ { < 1 }$ denotes the empty sequence. We use ⊕ to denote sequence concatenation. We denote random variables with capital letters, for example, �. We define the space of probability distributions over � as $\mathcal { P } ( \mathcal { V } )$ . We denote stop\_gradient the operation that keeps forward computations in automatic diferentiation unchanged while zeroing out the backpropagation of gradients in the backward pass, see e.g., Blondel and Roulet (2024) for more details.

## 2. Background

In this section, we review energy-based models (EBMs) and autoregressive (AR) models, emphasizing their shared structure and the loss functions associated with them.

## 2.1. Energy-based models

We are given a function $R ( { \pmb x } , { \pmb y } )$ , typically implemented as a bidirectional Transformer, which assigns a real-valued “afinity” score to a pair $( { \pmb x } , { \pmb y } )$ . We will also write $R _ { w }$ to indicate the dependence of � on some parameters $\pmb { w } \in \mathcal { W }$ and we suppose throughout this paper that $R _ { w }$ is diferentiable in �. We then define probabilistic energy-based models (EBMs) as

$$
p _ { R } ^ { \mathrm { E B M } } ( \pmb { y } | \pmb { x } ) : = \frac { \exp ( R ( \pmb { x } , \pmb { y } ) ) } { \sum _ { \pmb { y } ^ { \prime } \in \mathcal { V } } \exp ( R ( \pmb { x } , \pmb { y } ^ { \prime } ) ) } .
$$

We can see probabilistic EBMs as specifying a sequence-level Gibbs distribution (a.k.a. Boltzmann distribution) parameterized by �.

Sampling. Drawing i.i.d. sequence samples

$$
Y \sim p _ { R } ^ { \mathrm { E B M } } ( \cdot | \pmb { x } )
$$

typically requires Markov-chain Monte-Carlo (MCMC) methods, such as Gibbs sampling. These methods circumvent the intractable computation of the normalization factor in the probability definition. However, such samplers are typically inherently sequential and dificult to parallelize. Moreover, since they are never run to convergence in practice, they do not generate truly i.i.d. samples.

Sequence-level log-sum-exp and associated loss. The cumulant is the sequence-level log-sum-exp (a.k.a. log-partition function):

$$
A _ { R } ^ { \mathrm { L S E } } ( { \pmb x } ) : = \underset { { \pmb y } \in \mathscr { V } } { \mathrm { L S E } } R ( { \pmb x } , { \pmb y } ) : = \log \sum _ { { \pmb y } \in \mathscr { V } } \exp ( R ( { \pmb x } , { \pmb y } ) ) .
$$

The brute force computation cost is $O ( V ^ { T } )$

The associated loss is the negative log-likelihood:

$$
L _ { R } ^ { \mathrm { E B M } } ( { \pmb x } , { \pmb y } ) : = - \log p _ { R } ^ { \mathrm { E B M } } ( { \pmb y } | { \pmb x } ) = A _ { R } ^ { \mathrm { L S E } } ( { \pmb x } ) - R ( { \pmb x } , { \pmb y } ) .
$$

The loss is convex in � and satisfies the property

$$
L _ { R } ^ { \mathrm { E B M } } ( { \pmb x } , { \pmb y } ) = 0 \iff p _ { R } ^ { \mathrm { E B M } } ( { \pmb y } | { \pmb x } ) = 1 .
$$

The gradient w.r.t. parameters $\mathbf { \boldsymbol { w } } \in \mathcal { W }$ is (Song and Kingma, 2021)

$$
\begin{array} { r } { \nabla _ { w } L _ { R _ { w } } ^ { \mathrm { E B M } } ( { \pmb x } , { \pmb y } ) = \mathbb { E } _ { Y \sim p _ { R _ { w } } ^ { \mathrm { E B M } } ( \cdot \vert { \pmb x } ) } \left[ \nabla _ { w } R _ { w } ( { \pmb x } , Y ) \right] - \nabla _ { w } R _ { w } ( { \pmb x } , { \pmb y } ) . } \end{array}
$$

Unfortunately, this loss function requires access either to a log-sum-exp oracle, also known as marginal inference oracle (Wainwright et al., 2008), or being able to sample from $p _ { R w } ^ { \mathrm { E B M } }$ . Both are intractable in general over the space of sequences �.

Exact maximization oracle and associated loss. The mode of the distribution is

$$
{ \widehat { \pmb { y } } } _ { R } ^ { \mathrm { m a x } } ( { \pmb x } ) : = \underset { { \pmb y } \in \mathscr { Y } } { \mathrm { a r g m a x } } ~ p _ { R } ^ { \mathrm { E B M } } ( { \pmb y } | { \pmb x } ) = \underset { { \pmb y } \in \mathscr { Y } } { \mathrm { a r g m a x } } ~ R ( { \pmb x } , { \pmb y } ) \in \mathscr { V } .
$$

We can also define the value of the most plausible sequence:

$$
A _ { R } ^ { \operatorname* { m a x } } ( { \pmb x } ) : = \operatorname* { m a x } _ { { \pmb y } \in \mathcal { V } } R ( { \pmb x } , { \pmb y } ) \in \mathbb { R } .
$$

The associated loss function is

$$
L _ { R } ^ { \operatorname* { m a x } } ( { \pmb x } , { \pmb y } ) : = A _ { R } ^ { \operatorname* { m a x } } ( { \pmb x } ) - R ( { \pmb x } , { \pmb y } ) .\tag{1}
$$

The loss is again convex in � and satisfies the property

$$
L _ { R } ^ { \operatorname* { m a x } } ( { \pmb x } , { \pmb y } ) = 0 \iff { \pmb y } \in \widehat { { \pmb y } } _ { R } ^ { \operatorname* { m a x } } ( { \pmb x } ) .
$$

The loss is diferentiable almost everywhere with gradient

$$
\nabla _ { w } L _ { R _ { w } } ^ { \mathrm { m a x } } ( { \pmb x } , { \pmb y } ) = \nabla _ { w } R _ { w } ( { \pmb x } , \pmb { \widehat y } ) - \nabla _ { w } R _ { w } ( { \pmb x } , { \pmb y } ) ,\tag{2}
$$

where $\widehat { \pmb { y } } : =$ stop\_gradient $( \widehat { \pmb { y } } _ { R _ { w } } ^ { \operatorname* { m a x } } ( { \pmb x } ) )$ .

Unfortunately, this loss, known as the perceptron loss (Collins, 2002), requires an exact maximization oracle, which is intractable when � is the space of sequences. Indeed, the cost of brute force computation is again $O ( V ^ { T } )$ . More generally, Fenchel-Young losses (Blondel et al., 2022) provide an elegant way to construct a convex loss given access to a regularized decoding oracle. However, the theory again requires exact oracles, which are intractable over the space of sequences.

## 2.2. Autoregressive models

An autoregressive (AR) model factorizes the conditional distribution over sequences as a product of token-level distributions. We are given a scoring function $q ( s _ { t } , y _ { t } )$ , typically implemented using a causal Transformer, capturing the afinity between a prefix $\mathbf { \delta } _ { s t }$ and a token $y _ { t }$ . We will also write $q _ { w }$ to indicate the dependence of $q$ on some parameters $\mathbf { \pmb { w } } \in \mathcal { W }$ and we suppose throughout this paper that $q _ { w }$ is diferentiable in �. We can then define the next-token distribution as

$$
\pi _ { q } ( y _ { t } | \boldsymbol s _ { t } ) : = \frac { \exp ( q ( \boldsymbol s _ { t } , \boldsymbol y _ { t } ) ) } { \sum _ { j \in \mathcal { V } } \exp ( q ( \boldsymbol s _ { t } , j ) ) }
$$

and the sequence-level distribution as

$$
p _ { q } ^ { \mathrm { A R } } ( \pmb { y } | \pmb { x } ) : = \prod _ { t = 1 } ^ { T } \pi _ { q } ( y _ { t } | \pmb { x } \oplus \pmb { y } _ { < t } ) .\tag{3}
$$

Token-level log-sum-exp and associated loss. The token-level log-partition function is

$$
V _ { q } ( \pmb { s } _ { t } ) : = \operatorname { L S E } _ { y _ { t } \in \mathcal { V } } q ( \pmb { s } _ { t } , y _ { t } ) = \log \sum _ { y _ { t } \in \mathcal { V } } \exp ( q ( \pmb { s } _ { t } , y _ { t } ) ) .
$$

The negative log-likelihood decomposes as a sum of token-level losses:

$$
L _ { q } ^ { \mathrm { A R } } ( \pmb { x } , \pmb { y } ) : = - \log p _ { q } ^ { \mathrm { A R } } ( \pmb { y } | \pmb { x } ) = - \sum _ { t = 1 } ^ { T } \log \pi _ { q } ( y _ { t } | \pmb { s } _ { t } ) = \sum _ { t = 1 } ^ { T } ( V _ { q } ( \pmb { s } _ { t } ) - q ( \pmb { s } _ { t } , y _ { t } ) ) .
$$

The model is using the ground-truth tokens as prefix. This is known as teacher forcing (Williams and Zipser, 1989). In the imitation learning literature, teacher forcing corresponds to behavior cloning (Pomerleau, 1989; Bain and Sammut, 1995), where a policy is trained via supervised learning on expert state-action trajectories. At inference time, however, the model conditions on its own previous predictions, not on ground-truth prefixes. This train-inference mismatch is known as exposure bias (Bengio et al., 2015; Ranzato et al., 2015) and can lead to error accumulation.

Analogy with EBMs. It is instructive to draw an analogy with EBMs. We define the “log-partition” along the sequence $\pmb { y } = ( y _ { 1 } , \dots , y _ { T } )$ as

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { q } ^ { \mathrm { A R } } ( { \pmb x } , { \pmb y } ) : = V _ { q } ( s _ { 1 } ) + V _ { q } ( s _ { 2 } ) + \dots + V _ { q } ( s _ { T } ) } \ ~ } \\ { { \displaystyle ~ = \mathrm { L S E } ~ q ( s _ { 1 } , y _ { 1 } ^ { \prime } ) + \mathrm { L S E } ~ q ( s _ { 2 } , y _ { 2 } ^ { \prime } ) + \dots + \mathrm { L S E } ~ q ( s _ { T } , y _ { T } ^ { \prime } ) } \ ~ } \\ { { \displaystyle ~ = \log \sum _ { { \pmb y } ^ { \prime } \in \mathcal { V } } \exp ( q ( s _ { 1 } , y _ { 1 } ^ { \prime } ) + q ( s _ { 2 } , y _ { 2 } ^ { \prime } ) + \dots + q ( s _ { T } , y _ { T } ^ { \prime } ) ) , } } \end{array}
$$

where we used the shorthand $\pmb { s } _ { t } : = \pmb { x } \oplus \pmb { y } _ { < t }$ . We also define the cumulative score

$$
R _ { q } ( \pmb { x } , \pmb { y } ) : = \sum _ { t = 1 } ^ { T } q ( \pmb { x } \oplus \pmb { y } _ { < t } , y _ { t } ) = \sum _ { t = 1 } ^ { T } q ( \pmb { s } _ { t } , y _ { t } ) .\tag{4}
$$

Using these notations, we can equivalently write

$$
p _ { q } ^ { \mathrm { A R } } ( \pmb { y } | \pmb { x } ) = \exp ( R _ { q } ( \pmb { x } , \pmb { y } ) - A _ { q } ^ { \mathrm { A R } } ( \pmb { x } , \pmb { y } ) )\tag{5}
$$

$$
L _ { q } ^ { \mathrm { A R } } ( { \pmb x } , { \pmb y } ) = A _ { q } ^ { \mathrm { A R } } ( { \pmb x } , { \pmb y } ) - R _ { q } ( { \pmb x } , { \pmb y } )
$$

by analogy with

$$
\begin{array} { r l } & { p _ { R } ^ { \mathrm { E B M } } ( \pmb { y } | \pmb { x } ) = \exp ( R ( \pmb { x } , \pmb { y } ) - A _ { R } ^ { \mathrm { L S E } } ( \pmb { x } ) ) } \\ & { L _ { R } ^ { \mathrm { E B M } } ( \pmb { x } , \pmb { y } ) = A _ { R } ^ { \mathrm { L S E } } ( \pmb { x } ) - R ( \pmb { x } , \pmb { y } ) . } \end{array}
$$

## 3. Flawed decoding-aware sequence-level loss functions

To address exposure bias, a natural idea is to design sequence-level loss functions that incorporate the decoding procedure used at inference time. In this section, we describe two such attempts that appear natural on first sight, but turn out to be unusable for supervised learning.

## 3.1. Greedy decoding based loss function

In this section, we describe an attempt to define a loss function based on greedy decoding.

Intractability of exact maximization. Even if we assume that � decomposes as (4), the cost of exact maximization is still $O ( V ^ { T } )$ :

$$
\begin{array} { r l } & { A _ { R _ { q } } ^ { \operatorname* { m a x } } ( \pmb { x } ) : = \underset { y \in \mathscr { V } } { \operatorname* { m a x } } R _ { q } ( \pmb { x } , y ) } \\ & { \qquad = \underset { y \in \mathscr { V } } { \operatorname* { m a x } } q ( \pmb { x } , y _ { 1 } ) + q ( \pmb { x } \oplus y _ { 1 } , y _ { 2 } ) + \cdots + q ( \pmb { x } \oplus \pmb { y } _ { < T } , y _ { T } ) } \\ & { \qquad = \underset { y _ { 1 } \in \mathscr { V } } { \operatorname* { m a x } } q ( \pmb { x } , y _ { 1 } ) + \underset { y _ { 2 } \in \mathscr { V } } { \operatorname* { m a x } } q ( \pmb { x } \oplus y _ { 1 } , y _ { 2 } ) + \cdots + \underset { y _ { T } \in \mathscr { V } } { \operatorname* { m a x } } q ( \pmb { x } \oplus \pmb { y } _ { < T } , y _ { T } ) } \\ & { \qquad = \underset { y _ { 1 } \in \mathscr { V } } { \operatorname* { m a x } } q ( s _ { 1 } , y _ { 1 } ) + \underset { y _ { 2 } \in \mathscr { V } } { \operatorname* { m a x } } q ( s _ { 2 } , y _ { 2 } ) + \cdots + \underset { y _ { T } \in \mathscr { V } } { \operatorname* { m a x } } q ( s _ { T } , y _ { T } ) , } \end{array}\tag{6}
$$

where, with some abuse of notation, $\pmb { s } _ { t } : = \pmb { x } \oplus \pmb { y } _ { < t }$ depends on the variable � maximized over.   
Therefore, a frequently-used inference strategy in AR models is greedy decoding.

Greedy decoding. The key idea is to greedily decode one token at a time given the tokens decoded so far. Formally, we define the greedy decoding sequence as

$$
\widehat { \pmb { y } } _ { q } ^ { \mathrm { g r e e d y } } ( \pmb { x } ) : = ( \widetilde { \pmb { y } } _ { 1 } , \dots , \widetilde { \pmb { y } } _ { T } ) \in \mathcal { Y }
$$

where

$$
\begin{array} { r l } & { \widetilde { y } _ { 1 } : = \underset { y _ { 1 } \in \mathcal { V } } { \mathrm { a r g m a x } } \pi _ { q } ( y _ { 1 } | \widetilde { s } _ { 1 } ) = \underset { y _ { 1 } \in \mathcal { V } } { \mathrm { a r g m a x } } q ( \widetilde { s } _ { 1 } , y _ { 1 } ) \in \mathcal { V } } \\ & { \widetilde { y } _ { 2 } : = \underset { y _ { 2 } \in \mathcal { V } } { \mathrm { a r g m a x } } \pi _ { q } ( y _ { 2 } | \widetilde { s } _ { 2 } ) = \underset { y _ { 2 } \in \mathcal { V } } { \mathrm { a r g m a x } } q ( \widetilde { s } _ { 2 } , y _ { 2 } ) \in \mathcal { V } } \\ & { \quad \vdots } \\ & { \widetilde { y } _ { T } : = \underset { y _ { T } \in \mathcal { V } } { \mathrm { a r g m a x } } \pi _ { q } ( y _ { T } | \widetilde { s } _ { T } ) = \underset { y _ { T } \in \mathcal { V } } { \mathrm { a r g m a x } } q ( \widetilde { s } _ { T } , y _ { T } ) \in \mathcal { V } , } \end{array}
$$

and where we defined the prefix at step � by

$$
\begin{array} { r } { \tilde { \pmb { s } } _ { t } : = \pmb { x } \oplus \tilde { \pmb { y } } _ { < t } . } \end{array}
$$

Greedy maximization oracle. We define the corresponding greedy maximum value as

$$
\begin{array} { r l } & { A _ { q } ^ { \mathrm { g r e e d y } } ( \pmb { x } ) : = \underset { y _ { 1 } \in \mathcal { V } } { \operatorname* { m a x } } q ( \widetilde { s } _ { 1 } , y _ { 1 } ) + \underset { y _ { 2 } \in \mathcal { V } } { \operatorname* { m a x } } q ( \widetilde { s } _ { 2 } , y _ { 2 } ) + \cdot \cdot \cdot + \underset { y _ { T } \in \mathcal { V } } { \operatorname* { m a x } } q ( \widetilde { s } _ { T } , y _ { T } ) } \\ & { \qquad = q ( \widetilde { s } _ { 1 } , \widetilde { y } _ { 1 } ) + q ( \widetilde { s } _ { 2 } , \widetilde { y } _ { 2 } ) + \cdot \cdot \cdot + q ( \widetilde { s } _ { T } , \widetilde { y } _ { T } ) \in \mathbb { R } . } \end{array}
$$

The time complexity is now $O ( V T )$ , instead of $O ( V ^ { T } )$ for exact maximization.

Loss definition. As an approximation of the exact maximization based loss function in (1), it is a priori natural to define the greedy decoding loss function

$$
L _ { q } ^ { \mathrm { g r e e d y } } ( { \pmb x } , { \pmb y } ) : = A _ { q } ^ { \mathrm { g r e e d y } } ( { \pmb x } ) - R _ { q } ( { \pmb x } , { \pmb y } ) .
$$

Obviously, we have for all $\mathbf { \boldsymbol { x } } \in \mathcal { X }$

$$
A _ { R _ { q } } ^ { \mathrm { m a x } } ( { \pmb x } ) \geq A _ { q } ^ { \mathrm { g r e e d y } } ( { \pmb x } )\tag{7}
$$

and therefore for all $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$

$$
L _ { R _ { q } } ^ { \mathrm { m a x } } ( { \pmb x } , { \pmb y } ) \geq L _ { q } ^ { \mathrm { g r e e d y } } ( { \pmb x } , { \pmb y } ) .
$$

Gradient. The loss function has discontinuities due to the token-level argmaxes, but it is diferentiable almost everywhere.

Proposition 1 (Gradient of the greedy-decoding loss). Suppose $q _ { w }$ is diferentiable in �. The loss $L _ { q { \pmb w } } ^ { \mathrm { g r e e d y } } ( { \pmb x } , { \pmb y } )$ is diferentiable almost everywhere in �. Its gradient is

$$
\begin{array} { r } { \nabla _ { w } L _ { q _ { w } } ^ { \mathrm { g r e e d y } } ( { \pmb x } , { \pmb y } ) = \nabla _ { w } R _ { q _ { w } } ( { \pmb x } , \pmb { \widehat y } ) - \nabla _ { w } R _ { q _ { w } } ( { \pmb x } , { \pmb y } ) , } \end{array}
$$

where ${ \widehat { \pmb { y } } } : = \operatorname { s t o p } .$ \_gradient $( \widehat { \pmb { y } } _ { q w } ^ { \mathrm { g r e e d y } } ( { \pmb x } ) )$ .

A proof is given in Appendix A.1. The gradient has exactly the form we would expect: it perfectly mirrors the gradient in (2), with exact maximization replaced by greedy maximization.

Critical flaws. Unfortunately, $L _ { q } ^ { \mathrm { g r e e d y } }$ has critical flaws.

Proposition 2 (Flaws of the greedy decoding based loss). The lossfunction $L _ { q } ^ { \mathrm { g r e e d y } }$ is discontinuous, nonconvex in � and unbounded below.

## See Appendix A.2 for a proof.

This loss function can therefore not be used for learning, as the loss would diverge to −∞. Constraining the parameters to a bounded set (e.g., via $\ell _ { 2 }$ regularization) would prevent the loss from diverging, but would not fix the underlying issue: the constrained minimizer would sit at the boundary, dominated by the constraint rather than the loss, and the solution would depend on the arbitrary choice of regularization hyper-parameter.

## 3.2. Autoregressive sampling based loss function

In this section, we describe an attempt to define a loss function based on autoregressive sampling.

Autoregressive sampling. To draw an i.i.d. sequence $Y \sim p _ { q } ^ { \mathrm { A R } } ( \cdot | \pmb { x } )$ , from (3), we can use autoregressive sampling (a.k.a. ancestral sampling),

$$
\begin{array} { c } { Y _ { 1 } \sim \pi _ { q } ( \cdot ) } \\ { Y _ { 2 } \sim \pi _ { q } ( \cdot | S _ { 1 } ) } \\ { \vdots } \\ { Y _ { T } \sim \pi _ { q } ( \cdot | S _ { T } ) , } \end{array}
$$

where $S _ { t } : = \pmb { x } \oplus Y _ { < t }$ , with $Y _ { < t } : = ( Y _ { 1 } , \ldots , Y _ { t - 1 } )$ and $Y _ { < 1 }$ being an empty sequence.

Intractibility of sequence-level log-sum-exp. Using the associativity of the log-sum-exp (Mensch and Blondel, 2018; Blondel et al., 2025), we have that

$$
\begin{array} { r l } & { A _ { R _ { q } } ^ { \mathrm { L S E } } ( \pmb { x } ) = \underset { y \in \mathcal { Y } } { \mathrm { L S E } } R _ { q } ( \pmb { x } , \pmb { y } ) } \\ & { \qquad = \underset { y \in \mathcal { Y } } { \mathrm { L S E } } q ( \pmb { x } , y _ { 1 } ) + q ( \pmb { x } \oplus y _ { 1 } , y _ { 2 } ) + \cdots + q ( \pmb { x } \oplus \pmb { y } _ { < T } , y _ { T } ) } \\ & { \qquad = \underset { y _ { 1 } \in \mathcal { Y } } { \mathrm { L S E } } q ( \pmb { x } , y _ { 1 } ) + \underset { y _ { 2 } \in \mathcal { V } } { \mathrm { L S E } } q ( \pmb { x } \oplus y _ { 1 } , y _ { 2 } ) + \cdots + \underset { y _ { T } \in \mathcal { V } } { \mathrm { L S E } } q ( \pmb { x } \oplus \pmb { y } _ { < T } , y _ { T } ) } \\ & { \qquad = \underset { y _ { 1 } \in \mathcal { Y } } { \mathrm { L S E } } q ( s _ { 1 } , y _ { 1 } ) + \underset { y _ { 2 } \in \mathcal { V } } { \mathrm { L S E } } q ( s _ { 2 } , y _ { 2 } ) + \cdots + \underset { y _ { T } \in \mathcal { V } } { \mathrm { L S E } } q ( s _ { T } , y _ { T } ) , } \end{array}
$$

where, with some abuse of notation, $\pmb { s } _ { t } : = \pmb { x } \oplus \pmb { y } _ { < t }$ depends on the variable � being “log-sum-exped”. The expression above is the log-sum-exp counterpart of exact maximization in (6). Unfortunately, the complexity of brute force is still $O ( V ^ { T } )$

Sampling-based cumulant. Does there exist a counterpart of greedy maximization for the log-sumexp?

Recall that $\begin{array} { r } { A _ { q } ^ { \mathrm { A R } } ( { \pmb x } , { \pmb y } ) = \sum _ { t } V _ { q } ( { \pmb x } \oplus { \pmb y } _ { < t } ) = \sum _ { t } V _ { q } ( { \pmb s } _ { t } ) } \end{array}$ already decomposes the log-partition into token-level log-sum-exps, which are cheap to compute. However, unlike $A _ { R _ { a } } ^ { \mathrm { L S E } } ( { \pmb x } )$ , it depends on � through the ground-truth prefix $\pmb { s } _ { t } = \pmb { x } \oplus \pmb { y } _ { < t }$ , so it cannot serve as a cumulant on its own. A natural way to remove this dependency is to replace the ground-truth prefix with model-sampled prefixes and to average:

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { q } ^ { \mathrm { s a m p l i n g } } ( { \pmb x } ) : = \mathbb { E } _ { Y \sim p _ { q } ^ { \mathrm { A R } } ( \cdot \vert { \pmb x } ) } \left[ { \cal A } _ { q } ^ { \mathrm { A R } } ( { \pmb x } , Y ) \right] } \ ~ } \\ { { \displaystyle ~ = \mathbb { E } _ { Y \sim p _ { q } ^ { \mathrm { A R } } ( \cdot \vert { \pmb x } ) } \left[ \mathrm { L S E } q ( S _ { 1 } , y _ { 1 } ) + \mathrm { L S E } q ( S _ { 2 } , y _ { 2 } ) + \cdot \cdot \cdot + \mathrm { L S E } q ( S _ { T } , y _ { T } ) \right] } , } \end{array}\tag{8}
$$

where $S _ { t } : = \pmb { x } \oplus Y _ { < t }$

Combining (8) and (5), we can equivalently write

$$
A _ { q } ^ { \mathrm { s a m p l i n g } } ( { \pmb x } ) = \mathbb { E } _ { Y \sim p _ { q } ^ { \mathrm { A R } } ( \cdot \vert { \pmb x } ) } [ R _ { q } ( { \pmb x } , Y ) - \log p _ { q } ^ { \mathrm { A R } } ( Y \vert { \pmb x } ) ] .\tag{9}
$$

Due to the variational form

$$
\begin{array} { r l } {  { A _ { R } ^ { \mathrm { L S E } } ( \pmb { x } ) = \operatorname* { m a x } _ { p \in \mathcal { P } ( \mathcal { V } ) } \mathbb { E } _ { Y \sim p } [ R ( \pmb { x } , Y ) - \log p ( Y ) ] } } \\ & { = \mathbb { E } _ { Y \sim p _ { R } ^ { \mathrm { E B M } } ( \cdot \vert \pmb { x } ) } [ R ( \pmb { x } , Y ) - \log p _ { R } ^ { \mathrm { E B M } } ( Y \vert \pmb { x } ) ] , } \end{array}
$$

we immediately obtain for all $\mathbf { \boldsymbol { x } } \in \mathcal { X }$

$$
A _ { R _ { q } } ^ { \mathrm { L S E } } ( { \pmb x } ) \geq A _ { q } ^ { \mathrm { s a m p l i n g } } ( { \pmb x } ) ,
$$

which mirrors the inequality (7) we obtained for greedy maximization.

Loss definition. By analogy with previous sections, we define the sampling based loss function as

$$
L _ { q } ^ { \mathrm { s a m p l i n g } } ( { \pmb x } , { \pmb y } ) : = A _ { q } ^ { \mathrm { s a m p l i n g } } ( { \pmb x } ) - R _ { q } ( { \pmb x } , { \pmb y } ) .
$$

We then have for all $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$

$$
L _ { R _ { q } } ^ { \mathrm { E B M } } ( { \pmb x } , { \pmb y } ) \geq L _ { q } ^ { \mathrm { s a m p l i n g } } ( { \pmb x } , { \pmb y } ) .
$$

Critical flaws. Unfortunately, while the loss function is smooth, it is again not bounded below.

Proposition 3 (Flaws of the sampling-based loss). The sampling based loss $L _ { q } ^ { \mathrm { s a m p l i n g } }$ is nonconvex in � and unbounded below.

A proof is given in Appendix A.3.

This loss function can therefore not be used for learning either, as the loss would diverge to $- \infty$

## 4. Autoregressive Gumbel tricks

The classical Gumbel trick provides a powerful bridge between maximization and sampling: sampling from a Gibbs distribution is equivalent to maximizing a Gumbel perturbed score. This also connects exact maximization to log-sum-exp. In this section, we develop the autoregressive counterpart of the classical Gumbel trick, thereby connecting greedy decoding to autoregressive (ancestral) sampling and unifying the (flawed) loss functions from the previous section.

Shifted Gumbel noise. Following Blondel and Roulet (2024), we consider the shifted standard Gumbel distribution, whose probability density function (PDF) is defined by

$$
p ( x ) : = \exp ( - ( x + \gamma ) - \exp ( - ( x + \gamma ) ) ) ,
$$

where $\gamma \approx 0 . 5 7 7$ is Euler’s constant. Note that in some publications, the distribution is not shifted by �. Shifting the distribution as we do ensures that the mean is achieved at $x = 0$

Gumbel trick for categorical variables. Before deriving a Gumbel trick for autoregressive models, we review the standard Gumbel trick. Since $p _ { R } ^ { \mathrm { E B M } } ( \cdot | \pmb { x } ) = \mathrm { s o f t a r g m a x } ( R ( \pmb { x } , \cdot ) )$ , we can apply the standard Gumbel trick by seeing $\pmb { y } \in \mathcal { V }$ as a categorical variable of cardinality $| \mathcal { V } |$ . Let us define � such that each $Z ( { \pmb x } , { \pmb y } )$ is shifted Gumbel-distributed i.i.d. noise. Then, for any $R ,$ we have, see e.g. (Blondel and Roulet, 2024),

$$
\begin{array} { r l } & { Y = \widehat { \pmb { y } } _ { R + Z } ^ { \mathrm { m a x } } ( \pmb { x } ) \Longrightarrow Y \sim p _ { R } ^ { \mathrm { E B M } } ( \cdot | \pmb { x } ) } \\ & { \quad \quad \quad \quad \quad A _ { R } ^ { \mathrm { L S E } } ( \pmb { x } ) = \mathbb { E } _ { Z } [ A _ { R + Z } ^ { \mathrm { m a x } } ( \pmb { x } ) ] . } \end{array}
$$

Gumbel trick for sequential variables. We now derive analogous results, linking greedy decoding and autoregressive sampling.

Proposition 4 (Gumbel trick for autoregressive models). Let us define $Z$ such that each $Z ( \boldsymbol { s } , \boldsymbol { y } )$ is shifted Gumbel-distributed i.i.d. noise. For any $q ,$ we have

$$
Y = \widehat { \pmb { y } } _ { q + Z } ^ { \mathrm { g r e e d y } } ( { \pmb x } ) \implies Y \sim p _ { q } ^ { \mathrm { A R } } ( \cdot | { \pmb x } )
$$

$$
A _ { q } ^ { \mathrm { s a m p l i n g } } ( \pmb { x } ) = \mathbb { E } _ { Z } \left[ A _ { q + Z } ^ { \mathrm { g r e e d y } } ( \pmb { x } ) \right] .
$$

A proof is given in Appendix A.4.

Unifying the loss functions. Since

$$
\begin{array} { r l r } & { } & { L _ { q } ^ { \mathrm { s a m p l i n g } } ( { \pmb x } , { \pmb y } ) = A _ { q } ^ { \mathrm { s a m p l i n g } } ( { \pmb x } ) - R _ { q } ( { \pmb x } , { \pmb y } ) } \\ & { } & { L _ { q } ^ { \mathrm { g r e e d y } } ( { \pmb x } , { \pmb y } ) = A _ { q } ^ { \mathrm { g r e e d y } } ( { \pmb x } ) - R _ { q } ( { \pmb x } , { \pmb y } ) , } \end{array}
$$

Proposition 4 immediately implies

$$
L _ { q } ^ { \mathrm { s a m p l i n g } } ( \pmb { x } , \pmb { y } ) = \mathbb { E } _ { Z } \left[ L _ { q + Z } ^ { \mathrm { g r e e d y } } ( \pmb { x } , \pmb { y } ) \right] .
$$

That is, the sampling based loss is the expected greedy loss under Gumbel perturbations. This reveals that the two flawed loss functions from the previous section are in fact two faces of the same coin: greedy decoding is the deterministic limit, and ancestral sampling is the stochastic counterpart.

Related work. Gumbel noise was previously used with AR models (Kool et al., 2019) but the primary goal was to bridge deterministic beam search and sampling without replacement to create a practical, low-variance algorithm for generating diverse sequence outputs. In contrast, our goal is primarily to bridge greedy decoding and autoregressive sampling, as well as the resulting (flawed) loss functions. As we shall see in Section 5.4, the Gumbel trick also plays an important role in justifying reparameterization with a stop gradient on the greedy decoding.

## 5. Moment matching for autoregressive models

Since the pathology of the previous loss functions stems from the dependence of the decoded sequence on the model parameters, we now analyze what happens when this dependence is removed via a stop gradient.

## 5.1. Moment matching directions

As we already saw, gradients of the negative log-likelihood of an EBM are given by

$$
\begin{array} { r } { \nabla _ { w } L _ { R _ { w } } ^ { \mathrm { E B M } } ( { \pmb x } , { \pmb y } ) = \mathbb { E } _ { Y \sim p _ { R _ { w } } ^ { \mathrm { E B M } } ( \cdot \vert { \pmb x } ) } \left[ \nabla _ { w } R _ { w } ( { \pmb x } , Y ) \right] - \nabla _ { w } R _ { w } ( { \pmb x } , { \pmb y } ) . } \end{array}\tag{10}
$$

Such gradients encourage first-order moment matching. They are contrasting model samples to ground-truth samples at the sequence level.

Gradients of the negative log-likelihood of an AR model are given by

$$
\begin{array} { r l } & { \nabla _ { w } L _ { q _ { w } } ^ { \mathrm { A R } } ( \pmb { x } , \pmb { y } ) = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Y _ { t } \sim \pi _ { q _ { w } } ( \cdot \vert s _ { t } ) } \left[ \nabla _ { w } q _ { w } ( \pmb { s } _ { t } , Y _ { t } ) \right] - \nabla _ { w } R _ { q _ { w } } ( \pmb { x } , \pmb { y } ) } \\ & { \qquad = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Y _ { t } \sim \pi _ { q _ { w } } ( \cdot \vert s _ { t } ) } \left[ \nabla _ { w } q _ { w } ( \pmb { s } _ { t } , Y _ { t } ) \right] - \displaystyle \sum _ { t = 1 } ^ { T } \nabla _ { w } q _ { w } ( \pmb { s } _ { t } , y _ { t } ) , } \end{array}
$$

where $\mathbf { \pmb { s } } _ { t } : = \pmb { x } \oplus \pmb { y } _ { < t }$ . Notice that, in the left term, $q _ { w }$ is using ground-truth tokens as prefix! This is known as teacher forcing. Such gradients are contrasting model samples to ground-truth samples at the token-level.

As a middle ground, it is natural to consider the direction

$$
\begin{array} { r l } & { d _ { q _ { w } } ( \pmb { x } , \pmb { y } ) : = \mathbb { E } _ { Y \sim p _ { q _ { w } } ^ { \mathrm { A R } } ( \cdot \vert \pmb { x } ) } \left[ \nabla _ { w } R _ { q _ { w } } ( \pmb { x } , Y ) \right] - \nabla _ { w } R _ { q _ { w } } ( \pmb { x } , \pmb { y } ) } \\ & { \qquad = \mathbb { E } _ { Y \sim p _ { q _ { w } } ^ { \mathrm { A R } } ( \cdot \vert \pmb { x } ) } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \nabla _ { w } q _ { w } ( S _ { t } , Y _ { t } ) \right] - \displaystyle \sum _ { t = 1 } ^ { T } \nabla _ { w } q _ { w } ( s _ { t } , y _ { t } ) . } \end{array}
$$

where $S _ { t } : = \pmb { x } \oplus \pmb { Y } _ { < t }$ . Notice that, in the left term, $q _ { w }$ is now using the model generations as prefix, not the ground-truth tokens! However, since the model we use for sampling in the expectation is autoregressive, not energy-based, it is easy to sample from it.

The direction $\pmb { d } _ { q _ { w } } ( \pmb { x } , \pmb { y } )$ can be seen as diferentiating w.r.t. � the loss function

$$
\begin{array} { r } { L _ { q _ { w } } ^ { \mathrm { M M } } ( { \pmb x } , { \pmb y } ) : = \mathbb { E } _ { Y \sim p _ { q _ { w } } ^ { \mathrm { A R } } ( \cdot \vert { \pmb x } ) } \left[ R _ { q _ { w } } ( { \pmb x } , Y ) \right] - R _ { q _ { w } } ( { \pmb x } , { \pmb y } ) , } \end{array}
$$

while putting a stop gradient on the sampling inside the expectation.

Note that we see from (9) that $L _ { q } ^ { \mathrm { M M } }$ difers from $L _ { q } ^ { \mathrm { s a m p l i n g } }$ only by an entropy term, i.e.,

$$
L _ { q } ^ { \mathrm { s a m p l i n g } } ( \pmb { x } , \pmb { y } ) = L _ { q } ^ { \mathrm { M M } } ( \pmb { x } , \pmb { y } ) + H ( p _ { q } ^ { \mathrm { A R } } ( \cdot | \pmb { x } ) ) .
$$

This entropy term vanishes from the EBM gradient by the envelope theorem (stationarity of $p _ { R } ^ { \mathrm { E B M } } )$ ), but not in the AR case; see Section 5.4 for details.

In fact, we will show that the direction $\pmb { d } _ { q _ { w } } ( \pmb { x } , \pmb { y } )$ is a surrogate gradient: it is not the gradient of any function, but it defines, in function space, a descent direction for the negative log-likelihood of AR models.

Related work. It is informative to compare the direction $d _ { q _ { w } }$ to scheduled sampling (Bengio et al., 2015; Ranzato et al., 2015; Mihaylova and Martins, 2019), which was introduced to address exposure bias. In our notation, the direction used in scheduled sampling reads

$$
\left\{ \begin{array} { l l } { \mathbb { E } _ { Y \sim p _ { q _ { w } } ^ { \mathrm { A R } } ( \cdot \vert x ) } \left[ \sum _ { t = 1 } ^ { T } \nabla _ { w } q _ { w } ( S _ { t } , Y _ { t } ) - \sum _ { t = 1 } ^ { T } \nabla _ { w } q _ { w } ( S _ { t } , y _ { t } ) \right] } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } \alpha } \\ { \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Y \sim \pi _ { q _ { w } } ( \cdot \vert s _ { t } ) } \left[ \nabla _ { w } q _ { w } ( s _ { t } , Y _ { t } ) - \sum _ { t = 1 } ^ { T } \nabla _ { w } q _ { w } ( s _ { t } , y _ { t } ) \right] } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 1 - \alpha } \end{array} \right. ,
$$

where $\alpha \in [ 0 , 1 ]$ is a hyper-parameter. This leads to potential inconsistencies: the ground-truth token $y _ { t }$ is the correct next token for the ground-truth prefix $\mathbf { \delta } _ { \mathbf { \boldsymbol { s t } } }$ , but it is unlikely to be the correct next token for the prefix $S _ { t }$ sampled from the model. In contrast, the direction $d _ { q _ { w } }$ is a priori more principled, because it does not mix sampled prefix with ground-truth tokens.

## 5.2. Absence of a corresponding loss

In this section and the next, we use a function space representation (a.k.a. tabular setting in RL), where the function $q$ itself can be seen as the parameter we seek. In this representation, we write

$$
R _ { q } ( \pmb { x } , \pmb { y } ) = \sum _ { t = 1 } ^ { T } q ( \pmb { s } _ { t } , y _ { t } ) = \langle q , \phi ( \pmb { x } , \pmb { y } ) \rangle .
$$

Here, $\begin{array} { r } { \phi ( \pmb { x } , \pmb { y } ) : = \sum _ { t = 1 } ^ { T } \delta _ { ( \pmb { s } _ { t } , \pmb { y } _ { t } ) } } \end{array}$ is the feature map that places a unit mass on each state-action pair $( s _ { t } , y _ { t } )$ visited along the path defined by $( { \pmb x } , { \pmb y } )$ . The moment matching direction, in this function space, then simplifies to

$$
\begin{array} { r } { \pmb { d } _ { q } ( \pmb { x } , \pmb { y } ) = \mathbb { E } _ { Y \sim p _ { q } ^ { \mathrm { A R } } ( \cdot | \pmb { x } ) } \left[ \phi ( \pmb { x } , Y ) \right] - \phi ( \pmb { x } , \pmb { y } ) . } \end{array}\tag{11}
$$

It is then natural to ask whether there exists a loss function $L _ { q } ( \pmb { x } , \pmb { y } )$ whose functional gradient is $d _ { q } .$ It is well-known that moment matching gradients of the form (10) are a characteristic of the Gibbs distribution. Since $d _ { q }$ is obtained by replacing $p _ { R } ^ { \mathrm { E B M } }$ by $p _ { q } ^ { \mathrm { A R } }$ , which is not a Gibbs distribution for $T \geq 2$ , this suggests that $d _ { q }$ cannot be a gradient. This is confirmed by the following proposition.

Proposition 5 (The functional direction $d _ { q }$ is not the gradient of a loss). For $T \geq 2 ,$ , there does not exist a loss $L _ { q } ( \pmb { x } , \pmb { y } )$ whose gradient in � is the direction $ { d _ { q } } (  { \boldsymbol { { x } } } ,  { \boldsymbol { { y } } } )$

A proof is given in Appendix A.5.

## 5.3. Descent direction in function space

Although $d _ { q }$ is not the gradient of any loss, we now show that it is, in function space, a descent direction for the AR negative log-likelihood $L _ { q } ^ { \mathrm { A R } }$ . The key tool is the soft Bellman equation, which provides a mapping that converts an AR scoring function � into an equivalent EBM reward function �.

AR-to-EBM mapping. We define the mapping $r = \mathcal { M } ( q )$ as

$$
r ( s _ { t } , y _ { t } ) : = { \left\{ \begin{array} { l l } { q ( s _ { t } , y _ { t } ) } & { { \mathrm { i f ~ } } t = T } \\ { q ( s _ { t } , y _ { t } ) - V _ { q } ( s _ { t } \oplus y _ { t } ) } & { { \mathrm { i f ~ } } t < T } \end{array} \right. } .
$$

We denote by $\partial \mathcal { M } ( \boldsymbol { q } )$ the Jacobian of ℳ at � and by $\partial { \mathcal { M } } ( q ) ^ { * }$ its adjoint. As proved by Blondel et al. (2025), this mapping ensures that the EBM specified by the reward function $r = \mathcal { M } ( q )$ recovers the AR distribution exactly: for all $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$

$$
\begin{array} { r l } { p _ { R _ { r } } ^ { \mathrm { E B M } } ( { \pmb y } | { \pmb x } ) = p _ { q } ^ { \mathrm { A R } } ( { \pmb y } | { \pmb x } ) } & { } \\ { { \pmb A } _ { R _ { r } } ^ { \mathrm { E B M } } ( { \pmb x } ) = V _ { q } ( { \pmb x } ) . } \end{array}
$$

Preconditioned AR gradient. Using this mapping, we can relate the moment matching direction $d _ { q }$ to both the EBM and AR functional gradients.

Proposition 6 (Direction as preconditioned gradient). For any � and $( { \pmb x } , { \pmb y } )$ , we have

$$
\begin{array} { r } { \pmb { d } _ { q } ( \pmb { x } , \pmb { y } ) = \nabla _ { r } L _ { R _ { r } } ^ { \mathrm { E B M } } ( \pmb { x } , \pmb { y } ) \big | _ { r = \pmb { \mathcal { M } } ( q ) } } \end{array}
$$

and

$$
\begin{array} { r } { \partial \mathcal { M } ( q ) ^ { * } d _ { q } ( \pmb { x } , \pmb { y } ) = \nabla _ { q } L _ { q } ^ { \mathrm { A R } } ( \pmb { x } , \pmb { y } ) . } \end{array}
$$

A proof is given in Appendix A.6.

In other words, $d _ { q }$ is the EBM functional gradient in the reparameterized space, and it equals the AR functional gradient preconditioned by $( \partial \mathcal M ( q ) ^ { * } ) ^ { - 1 }$ . Since ℳ is invertible (Blondel et al., 2025), $d _ { q }$ and $\nabla _ { q } L _ { q } ^ { \mathrm { A R } }$ point into the same half-space, which gives us the descent property.

Proposition 7 (Descent direction). With $d _ { q }$ defined in (11), we have

$$
\begin{array} { r } { \langle \pmb { d } _ { q } ( \pmb { x } , \pmb { y } ) , \nabla _ { q } \pmb { L } _ { q } ^ { \mathrm { A R } } ( \pmb { x } , \pmb { y } ) \rangle < 0 . } \end{array}
$$

That $i s , - d _ { q }$ is a descent direction for $L _ { q } ^ { \mathrm { A R } }$

A proof is given in Appendix A.7.

## 5.4. Perspectives

## Reparameterization with stop gradient

The reparameterization trick (path gradient estimator) swaps diferentiation and expectation, but mathematically requires the integrand to be continuous and diferentiable almost everywhere in the parameters (Blondel and Roulet, 2024). Therefore, in principle, we cannot use the reparameterization trick for diferentiating $L _ { q _ { w } } ^ { \mathrm { s a m p l i n g } } ( \pmb { x } , \pmb { y } ) = \mathbb { E } _ { Z } [ L _ { q _ { w } + Z } ^ { \mathrm { g r e e d y } } ( \pmb { x } , \pmb { y } ) ]$ with respect to �. However, if we treat the argmax selections in greedy decoding as locally constant (using a stop gradient), reparameterization yields exactly $\pmb { d } _ { q _ { w } } ( \pmb { x } , \pmb { y } )$ . Using REINFORCE (score function estimator) would yield the mathematicallycorrect gradient of $L _ { q w } ^ { \mathrm { s a m p l i n g } } ( \pmb { x } , \pmb { y } )$ but we showed that this loss is unbounded below. In a sense, $\scriptstyle d _ { q _ { w } } ( { \pmb x } , { \pmb y } )$ is the mathematically-incorrect “gradient” of $\mathbb { E } _ { Z } [ L _ { q _ { w } + Z } ^ { \mathrm { g r e e d y } } ( \pmb { x } , \pmb { y } ) ]$

Remark 1. We point out that the reparameterization trick (with stop gradient) at the heart of the moment matching direction $d _ { q _ { w } }$ does not transfer to RL post-training. The direction $d _ { q _ { w } }$ relies on the fact that the scoring function $R _ { q _ { w } }$ shares parameters with the policy $p _ { q _ { w } } ^ { \mathrm { A R } }$ : applying a stop gradient on the sampling still leaves a nonzero gradient through $R _ { q _ { w } }$ . In RL, the reward � is usually a separate function, so stopping the gradient on the sampling zeroes out the signal entirely. For sequence-level objectives with external rewards, REINFORCE-based methods therefore remain unavoidable.

## The model as its own critic

We have the following variational inequality for any � and any �:

$$
\begin{array} { r l } & { L _ { R } ^ { \mathrm { E B M } } ( { \pmb x } , { \pmb y } ) = \underset { p \in \mathcal { P } ( \mathcal { V } ) } { \mathrm { m a x } } ~ \mathbb { E } _ { Y \sim p } [ R ( { \pmb x } , Y ) - \log p ( Y ) ] - R ( { \pmb x } , { \pmb y } ) } \\ & { \qquad = \mathbb { E } _ { Y \sim p _ { R } ^ { \mathrm { E B M } } ( \cdot \vert { \pmb x } ) } [ R ( { \pmb x } , Y ) - \log p _ { R } ^ { \mathrm { E B M } } ( Y \vert { \pmb x } ) ] - R ( { \pmb x } , { \pmb y } ) } \\ & { \qquad \geq \mathbb { E } _ { Y \sim p _ { q } ^ { \mathrm { A R } } ( \cdot \vert { \pmb x } ) } [ R ( { \pmb x } , Y ) - \log p _ { q } ^ { \mathrm { A R } } ( Y \vert { \pmb x } ) ] - R ( { \pmb x } , { \pmb y } ) . } \end{array}
$$

The last line can be seen as an adversarial loss where � plays the role of the discriminator and $p _ { q } ^ { \mathrm { A R } }$ plays the role of the generator. In the RL terminology, these correspond to the concepts of critic and actor, respectively.

In particular, by setting $R = R _ { q _ { w } }$ and $q = q _ { w } ,$ we obtain the loss function

$$
L _ { q _ { w } } ( \pmb { x } , \pmb { y } ) : = \mathbb { E } _ { Y \sim p _ { q _ { w } } ^ { \mathrm { A R } } ( \cdot | \pmb { x } ) } [ R _ { q _ { w } } ( \pmb { x } , Y ) - \log p _ { q _ { w } } ^ { \mathrm { A R } } ( Y | \pmb { x } ) ] - R _ { q _ { w } } ( \pmb { x } , \pmb { y } ) .
$$

If we put a stop gradient on $p _ { q _ { w } } ^ { \mathrm { A R } }$ , we immediately obtain that the gradient in � recovers the moment matching direction $d _ { q _ { w } }$ . In other words, $d _ { q _ { w } }$ corresponds to using the same underlying model for the discriminator (reward model) and for the generator, with a stop gradient on the generator!

## Non-stationarity gap between EBMs and AR models

In this section, we drop � for notational simplicity.

Let us define

$$
F ( p , R ) : = \langle p , R \rangle - \langle p , \log p \rangle = \mathbb { E } _ { Y \sim p } [ R ( Y ) - \log p ( Y ) ] .
$$

For any mapping $R \mapsto p _ { R } .$ we have by the chain rule

$$
\begin{array} { r l } & { \nabla _ { R } F ( p _ { R } , R ) = ( \partial _ { R } p _ { R } ) ^ { \top } \nabla _ { 1 } F ( p _ { R } , R ) + \nabla _ { 2 } F ( p _ { R } , R ) } \\ & { \qquad = ( \partial _ { R } p _ { R } ) ^ { * } \nabla _ { 1 } F ( p _ { R } , R ) + p _ { R } . } \end{array}
$$

Observe that

$$
A _ { R } ^ { \mathrm { L S E } } = F ( p _ { R } ^ { \mathrm { E B M } } , R ) .
$$

Since $p _ { R } ^ { \mathrm { E B M } }$ maximizes $F ( \cdot , R )$ , we have $\nabla _ { 1 } F ( p _ { R } ^ { \mathrm { E B M } } , R ) = 0$ and therefore

$$
\nabla _ { R } A _ { R } ^ { \mathrm { L S E } } = p _ { R } ^ { \mathrm { E B M } } .
$$

This is a consequence of the envelope theorem (Blondel and Roulet, 2024).

On the other hand, from (9),

$$
A _ { \boldsymbol { q } } ^ { \mathrm { s a m p l i n g } } = F ( p _ { \boldsymbol { q } } ^ { \mathrm { A R } } , R _ { \boldsymbol { q } } ) .
$$

Since $p _ { q } ^ { \mathrm { A R } }$ does not maximize $F ( \cdot , R _ { q } )$ over all distributions, $\nabla _ { 1 } F ( p _ { q } ^ { \mathrm { A R } } , R _ { q } ) \neq 0$ and

$$
\begin{array} { r } { \nabla _ { q } A _ { q } ^ { \mathrm { s a m p l i n g } } = \underbrace { ( \partial _ { q } p _ { q } ^ { \mathrm { A R } } ) ^ { * } \nabla _ { 1 } F ( p _ { q } ^ { \mathrm { A R } } , R _ { q } ) } _ { \mathrm { n o n - s t a t i o n a r i t y t e r m } } + p _ { q } ^ { \mathrm { A R } } , } \end{array}
$$

the direction

$$
d _ { q } ( \pmb { y } ) = \nabla _ { q } A _ { q } ^ { \mathrm { s a m p l i n g } } - \phi ( \pmb { y } ) = p _ { q } ^ { \mathrm { A R } } - \phi ( \pmb { y } )
$$

therefore corresponds to dropping the non-stationarity term, $\mathrm { i . e . , }$ treating $p _ { q } ^ { \mathrm { A R } }$ as if it satisfied the stationarity condition (as the EBM does). Here, $\phi ( \pmb { y } ) = \delta _ { \pmb { y } }$ is the Dirac distribution centered at the ground truth, so $d _ { q }$ matches the model distribution $p _ { q } ^ { \mathrm { A R } }$ to $\delta _ { y }$ . This is analogous to the stop-gradient trick: we diferentiate through $R _ { q }$ but treat the dependence of $p _ { q } ^ { \mathrm { A R } }$ on $q$ as fixed.

## 5.5. Limitations

## Function space vs. parameter space

We emphasize that Proposition 7 holds in function space. In parameter space with $q \ = \ q _ { w } ,$ , let $J ( \pmb { w } ) : = \partial q _ { \pmb { w } } / \partial \pmb { w }$ denote the Jacobian of the parameterization. By the chain rule, the parameterspace direction is ${ d _ { q } } _ { w } = J ( w ) ^ { * } d _ { q }$ and the parameter gradient is $\nabla _ { \mathbf { \boldsymbol { w } } } L _ { q _ { \boldsymbol { w } } } ^ { \mathrm { A R } } = J ( \mathbf { \boldsymbol { w } } ) ^ { * } \nabla _ { \boldsymbol { q } } \bar { L } _ { \boldsymbol { q } } ^ { \mathrm { A R } }$ . Their inner product equals $\langle \pmb { d } _ { q } , J ( \pmb { w } ) J ( \pmb { w } ) ^ { * } \nabla _ { q } L _ { q } ^ { \mathrm { A R } } \rangle$ . Because the Gram operator $J ( { \pmb w } ) J ( { \pmb w } ) ^ { * }$ deforms inner products, descent in function space does not imply descent in parameter space in general, even under a linear parameterization $q _ { w } = \Phi { w }$ . That being said, in deep learning, descent guarantees are routinely missing (Adam, momentum, etc.). The perspectives in Section 5.4 provide some motivation for the moment matching direction, beyond the descent property.

## Variance

Although $d _ { q _ { w } }$ avoids the pathologies of the flawed losses from Section 3, its practical use is limited by high variance. The teacher forcing gradient $\nabla _ { w } L _ { q w } ^ { \mathrm { A R } }$ decomposes as a sum of $T$ token-level expectations, each over � (of size � ), which can be computed exactly via a single forward pass (no sampling is needed). In contrast, the moment matching direction $d _ { q _ { w } }$ involves an expectation over $p _ { q w } ^ { \mathrm { A R } }$ , a distribution over $\mathcal { V } = \mathcal { V } ^ { T }$ , which can only be estimated by Monte Carlo sampling. Since sampled prefixes $S _ { t } = \pmb { x } \oplus \pmb { Y } _ { < t }$ are correlated across time steps, the variance of this estimator grows with the sequence length �. Since teacher forcing provides the exact gradient $\nabla _ { w } L _ { q _ { w } } ^ { \mathrm { A R } }$ cheaply (a single forward pass, no sampling) anyway, teacher forcing remains preferable in practice.

## 6. Conclusion

In this paper, we studied the possibility to design decoding-aware sequence-level loss functions for addressing exposure bias in supervised learning of autoregressive (AR) models.

We first studied $L _ { q } ^ { \mathrm { g r e e d y } }$ , a loss function based on greedy decoding. We showed that it is discontinuous and unbounded below. We then studied $L _ { q } ^ { \mathrm { s a m p l i n g } }$ , a loss function based on autoregressive sampling. While the loss function is smooth, it remains unbounded below. Interestingly, we showed that the two loss functions are related through $L _ { q } ^ { \mathrm { s a m p l i n g } } = \mathbb { E } _ { Z } [ L _ { q + Z } ^ { \mathrm { g r e e d y } } ]$ , where � is i.i.d. Gumbel noise. Unfortunately, neither loss function can be used for learning, as the loss would diverge to $- \infty$

Since the pathology of these loss functions stems from the dependence of the decoded sequence on the model parameters, we analyzed the update direction obtained by applying a stop gradient on the sampling procedure, leading to the moment matching direction $d _ { q }$ . We saw that this direction corresponds to dropping the problematic non-stationarity term in $\nabla _ { q } A _ { q } ^ { \mathrm { s a m p l i n g } }$

Nevertheless, while the moment matching direction is theoretically appealing, its practical use is limited by high variance. Since $- \pmb { d } _ { q }$ is merely a descent direction for $\bar { L } _ { q } ^ { \mathrm { \tiny \bar { A } R } }$ in function space, while teacher forcing provides the exact gradient $\nabla _ { q } L _ { q } ^ { \mathrm { A R } }$ cheaply anyway (a single forward pass, no sampling), teacher forcing remains preferable in practice.

We conclude that teacher forcing may well be “all you need” for the supervised learning phase of AR models from static datasets; see also (Foster et al., 2024; Blondel et al., 2025) for additional insights. While the exposure bias concern is real in principle, our paper suggests that it cannot be resolved through intrinsic sequence-level loss functions without introducing severe pathologies.

If one wishes to directly address exposure bias, it is safer to change the training data distribution rather than the loss function. For instance, online policy distillation (Agarwal et al., 2024) resolves the train-inference mismatch by querying a teacher model on prefixes generated by the student, but crucially, it still relies on the mathematically well-behaved token-level cross-entropy loss. Alternatively, if true sequence-level signals are desired, they are best incorporated via RL-based post-training. RL bypasses the aforementioned pathologies thanks to the score function estimator (REINFORCE), allowing the use of a task-specific fixed reward $R ( { \pmb x } , { \pmb y } )$ or a distance metric $R ( { \pmb x } , { \pmb y } ) = - D ( { \pmb y } , { \pmb y } ^ { \star } )$ to the ground-truth $y ^ { \star }$

## Appendix

## A. Proofs

## A.1. Greedy-decoding loss gradients (Proof of Proposition 1)

$$
\mathrm { R e c a l l ~ t h a t } \ L _ { \mathcal { G } _ { w } } ^ { \mathrm { g r e e d y } } ( x , y ) = A _ { \mathcal { G } _ { w } } ^ { \mathrm { g r e e d y } } ( x ) - R _ { q _ { w } } ( x , y ) , \mathrm { w h e r e } \ A _ { \mathcal { G } _ { w } } ^ { \mathrm { g r e e d y } } ( x ) = R _ { q _ { w } } ( x , \widehat { y } _ { \mathcal { G } _ { w } } ^ { \mathrm { g r e e d y } } ( x ) ) .
$$

Diferentiability almost everywhere. Let � be such that none of the step-wise greedy choices have ties, $\mathrm { i . e . , } q _ { w } ( \tilde { s } _ { t } , y _ { t } ^ { ( 1 ) } ) \neq q _ { w } ( \tilde { s } _ { t } , y _ { t } ^ { ( 2 ) } )$ for any step $t \in [ T ]$ and any distinct tokens $y _ { t } ^ { ( 1 ) } , y _ { t } ^ { ( 2 ) } \in \mathcal { V }$ , where $\tilde { \pmb { s } } _ { t } : = \pmb { x } \oplus \tilde { \pmb { y } } _ { < t }$ . There exists a neighborhood � of � such that any ${ \pmb w } ^ { \prime } \in U$ maintains the same strict inequalities at each step. Consequently, $\begin{array} { r } { \widehat { \pmb { y } } _ { { q } _ { w ^ { \prime } } } ^ { \mathrm { g r e e d y } } ( { \pmb x } ) = \widehat { \pmb { y } } _ { { q } _ { w } } ^ { \mathrm { g r e e d y } } ( { \pmb x } ) } \end{array}$ for all ${ \pmb w } ^ { \prime } \in U$ , so ${ \pmb w } \mapsto \widehat { { \pmb y } } _ { { \boldsymbol q } { \boldsymbol w } } ^ { \mathrm { g r e e d y } } ( { \pmb x } )$ is locally constant and its Jacobian is 0. The set of � where ties occur has Lebesgue measure zero, so the loss is diferentiable almost everywhere.

Gradient computation. At any point of diferentiability, $\widehat { \pmb { y } } : = \widehat { \pmb { y } } _ { \boldsymbol { q } _ { \boldsymbol { w } } } ^ { \mathrm { g r e e d y } } ( \pmb { x } )$ is locally constant in �. By the chain rule,

$$
\begin{array} { r l } & { \nabla _ { w } L _ { q _ { w } } ^ { \mathrm { g r e e d y } } ( { \pmb x } , { \pmb y } ) = \nabla _ { w } \left[ R _ { q _ { w } } ( { \pmb x } , \widehat { \pmb y } ) - R _ { q _ { w } } ( { \pmb x } , { \pmb y } ) \right] } \\ & { \qquad = \nabla _ { w } R _ { q _ { w } } ( { \pmb x } , \widehat { \pmb y } ) - \nabla _ { w } R _ { q _ { w } } ( { \pmb x } , { \pmb y } ) , } \end{array}
$$

where $\widehat { \pmb { y } }$ is treated as a constant (stop gradient).

## A.2. Discontinuity, non-convexity, and unboundedness of greedy decoding loss (Proof of Proposition 2)

Set $T = 2 , V = 2$ and $\mathcal { V } = \{ 1 , 2 \}$ for simplicity. Consider w.l.o.g. $y _ { 1 } = y _ { 2 } = 1$ . The greedy loss is

$$
\begin{array} { r l } & { L _ { q } ^ { \mathrm { g r e e d y } } ( \pmb { x } , \pmb { y } ) = A _ { q } ^ { \mathrm { g r e e d y } } ( \pmb { x } ) - R _ { q } ( \pmb { x } , \pmb { y } ) } \\ & { \qquad = \underset { y _ { 1 } ^ { \prime } } { \operatorname* { m a x } } q ( s _ { 1 } , y _ { 1 } ^ { \prime } ) - q ( s _ { 1 } , y _ { 1 } ) + \underset { y _ { 2 } ^ { \prime } } { \operatorname* { m a x } } q ( \pmb { x } \oplus \widetilde { y } _ { 1 } , y _ { 2 } ^ { \prime } ) - q ( s _ { 2 } , y _ { 2 } ) , } \end{array}
$$

where $\widetilde { y } _ { 1 } = \mathrm { a r g m a x } _ { y _ { 1 } ^ { \prime } } q ( \pmb { s } _ { 1 } , y _ { 1 } ^ { \prime } )$

Discontinuity and non-convexity. Define the family $q _ { \epsilon }$ by

$$
q _ { \epsilon } ( s _ { 1 } , 1 ) = 0 , \quad q _ { \epsilon } ( s _ { 1 } , 2 ) = \epsilon , \quad q _ { \epsilon } ( x \oplus 1 , \cdot ) = ( 0 , 1 ) , \quad q _ { \epsilon } ( x \oplus 2 , \cdot ) = ( 0 , 2 ) .
$$

Then lim $1 _ { \epsilon  0 ^ { - } } q _ { \epsilon } = \mathrm { l i m } _ { \epsilon  0 ^ { + } } q _ { \epsilon }$ , yet

$$
\operatorname* { l i m } _ { \epsilon \to 0 ^ { - } } L _ { q _ { \epsilon } } ^ { \mathrm { g r e e d y } } ( { \pmb x } , { \pmb y } ) = 1 \neq 2 = \operatorname* { l i m } _ { \epsilon \to 0 ^ { + } } L _ { q _ { \epsilon } } ^ { \mathrm { g r e e d y } } ( { \pmb x } , { \pmb y } ) ,
$$

since for $\epsilon < 0$ the greedy token is $\widetilde { y } _ { 1 } = 1$ (giving loss $0 + 1 = 1 )$ , while for $\epsilon > 0$ it switches to $\widetilde { y } _ { 1 } = 2$ (giving loss $\epsilon + 2  2 )$ . Hence $L _ { q } ^ { \mathrm { g r e e d y } }$ is not continuous in �. Any convex function is locally Lipschitz continuous so the function $L _ { q } ^ { \mathrm { g r e e d y } }$ cannot be convex.

Unboundedness. Define $q _ { \alpha }$ by

$$
q _ { \alpha } ( { \pmb s } _ { 1 } , 1 ) = 0 , \quad q _ { \alpha } ( { \pmb s } _ { 1 } , 2 ) = 1 , \quad q _ { \alpha } ( { \pmb x } \oplus 1 , \cdot ) = ( \alpha , 0 ) , \quad q _ { \alpha } ( { \pmb x } \oplus 2 , \cdot ) = ( 0 , 0 ) .
$$

Then $\widetilde { y } _ { 1 } = 2$ and

$$
L _ { q _ { \alpha } } ^ { \mathrm { g r e e d y } } ( { \pmb x } , { \pmb y } ) = 1 - \alpha \underset { \alpha  + \infty } { \longrightarrow } - \infty .
$$

Hence the greedy decoding loss is unbounded below. Note also that $\mathrm { i f }$ we set $\alpha = 2 | \sin ( \beta ) |$ for instance, we see that the loss has infinitely many zeros in $\beta .$

## A.3. Nonconvexity and unboundedness of sampling based loss (Proof of Proposition 3)

Non-convexity. We demonstrate that $L _ { q } ^ { \mathrm { s a m p l i n g } }$ is not convex in � by constructing a counterexample that violates Jensen’s inequality. Set $T = 2 , V = 2 .$ , and $\mathcal { V } = \left\{ 1 , 2 \right\}$ . Consider w.l. ${ . 0 . 8 \cdot y 1 } = y _ { 2 } = 1$ Because the reward term $R _ { q } ( { \pmb x } , { \pmb y } )$ is linear in $q ,$ the loss $L _ { q } ^ { \mathrm { s a m p l i n g } }$ is convex if and only if the expected greedy value $f ( q ) : = \mathbb { E } _ { Z } \left[ A _ { q + Z } ^ { \mathrm { g r e e d y } } ( { \pmb x } ) \right]$ is convex. Using Proposition $^ { 4 , }$ we have the closed-form expression:

$$
f ( q ) = \mathrm { L S E } _ { y _ { 1 } \in \mathcal { V } } q ( s _ { 1 } , y _ { 1 } ) + \sum _ { y _ { 1 } \in \mathcal { V } } \pi _ { q } ( y _ { 1 } | s _ { 1 } ) \mathrm { L S E } _ { y _ { 2 } \in \mathcal { V } } q ( \pmb { x } \oplus y _ { 1 } , y _ { 2 } ) ,
$$

where $\pi _ { \boldsymbol { q } } ( y _ { 1 } | \boldsymbol { s } _ { 1 } ) \propto \exp ( \boldsymbol { q } ( \boldsymbol { s } _ { 1 } , y _ { 1 } ) )$

Define two configurations $q _ { 1 }$ and $q _ { 2 }$ parameterized by $\alpha , \beta > 0 ;$

$$
\begin{array} { r l r l } & { q _ { 1 } ( s _ { 1 } , \cdot ) = ( \alpha , - \alpha ) , \qquad } & & { q _ { 1 } ( x \oplus 1 , \cdot ) = ( 0 , 0 ) , \qquad } & & { q _ { 1 } ( x \oplus 2 , \cdot ) = ( 0 , \beta ) } \\ & { q _ { 2 } ( s _ { 1 } , \cdot ) = ( - \alpha , \alpha ) , \qquad } & & { q _ { 2 } ( x \oplus 1 , \cdot ) = ( 0 , 0 ) , \qquad } & & { q _ { 2 } ( x \oplus 2 , \cdot ) = ( 0 , 0 ) . } \end{array}
$$

Their midpoint $q _ { M } = { \textstyle { \frac { 1 } { 2 } } } ( q _ { 1 } + q _ { 2 } )$ is given by:

$$
\begin{array} { r } { q _ { M } ( \pmb { s } _ { 1 } , \cdot ) = ( 0 , 0 ) , \quad q _ { M } ( \pmb { x } \oplus 1 , \cdot ) = ( 0 , 0 ) , \quad q _ { M } ( \pmb { x } \oplus 2 , \cdot ) = ( 0 , \frac { \beta } { 2 } ) . } \end{array}
$$

Evaluating � for $q _ { 1 }$ and $q _ { 2 } ,$ , we obtain:

$$
\begin{array} { r l } & { f ( q _ { 1 } ) = \log ( e ^ { \alpha } + e ^ { - \alpha } ) + \pi _ { q _ { 1 } } ( 1 | s _ { 1 } ) \log ( 2 ) + \pi _ { q _ { 1 } } ( 2 | s _ { 1 } ) \log ( 1 + e ^ { \beta } ) } \\ & { f ( q _ { 2 } ) = \log ( e ^ { - \alpha } + e ^ { \alpha } ) + \pi _ { q _ { 2 } } ( 1 | s _ { 1 } ) \log ( 2 ) + \pi _ { q _ { 2 } } ( 2 | s _ { 1 } ) \log ( 2 ) . } \end{array}
$$

As $\alpha  + \infty$ , the routing probabilities become deterministic: $\pi _ { q _ { 1 } } ( 1 | \pmb { s } _ { 1 } )  1$ and $\pi _ { q _ { 2 } } ( 2 | \pmb { s } _ { 1 } )  1$ Consequently, both $f ( q _ { 1 } )$ and $f ( q _ { 2 } )$ scale asymptotically as $\alpha + \log ( 2 )$ . Thus, their convex combination is bounded:

$$
\operatorname* { l i m } _ { \alpha \to + \infty } \left[ { \frac { 1 } { 2 } } f ( q _ { 1 } ) + { \frac { 1 } { 2 } } f ( q _ { 2 } ) - \alpha \right] = \log ( 2 ) .
$$

However, for the midpoint $q _ { M } ,$ , the first-step logits are zero, giving uniform routing probabilities $\begin{array} { r } { \pi _ { q _ { M } } ( \cdot | s _ { 1 } ) = \frac { 1 } { 2 } } \end{array}$ . Evaluating $f ( q _ { M } )$ yields:

$$
f ( q _ { M } ) = \log ( 2 ) + \frac { 1 } { 2 } \log ( 2 ) + \frac { 1 } { 2 } \log ( 1 + e ^ { \beta / 2 } ) .
$$

For any arbitrarily large $\alpha _ { z }$ , we can choose $\beta \gg \alpha$ such that $f ( q _ { M } ) > { \textstyle { \frac { 1 } { 2 } } } f ( q _ { 1 } ) + { \textstyle { \frac { 1 } { 2 } } } f ( q _ { 2 } )$ , which strictly violates the necessary condition for convexity. Therefore, $L _ { q } ^ { \mathrm { s a m p l i n g } }$ is not convex.

Unboundedness Set $T = 2 , V = 2$ and $\gamma = \{ 1 , 2 \}$ for simplicity. Consider w.l.o.g. $y _ { 1 } = y _ { 2 } = 1$ Using Proposition $^ { 4 , }$ we have

$$
L _ { q } ^ { \mathrm { s a m p l i n g } } ( \pmb { x } , \pmb { y } ) = \mathbb { E } _ { Z } \left[ A _ { q + Z } ^ { \mathrm { g r e e d y } } ( \pmb { x } ) \right] - R _ { q } ( \pmb { x } , \pmb { y } ) .
$$

Define $q _ { \alpha }$ by

$$
q _ { \alpha } ( { \pmb s } _ { 1 } , 1 ) = 0 , \quad q _ { \alpha } ( { \pmb s } _ { 1 } , 2 ) = 0 , \quad q _ { \alpha } ( { \pmb x } \oplus 1 , \cdot ) = ( \alpha , 0 ) , \quad q _ { \alpha } ( { \pmb x } \oplus 2 , \cdot ) = ( 0 , 0 ) .
$$

Then $R _ { q _ { \alpha } } ( { \pmb x } , { \pmb y } ) = q _ { \alpha } ( { \pmb s } _ { 1 } , 1 ) + q _ { \alpha } ( { \pmb x } \oplus 1 , 1 ) = \alpha$ . The greedy value under Gumbel noise � is

$$
A _ { q _ { \alpha } + Z } ^ { \mathrm { g r e e d y } } ( \pmb { x } ) = \operatorname* { m a x } _ { y _ { 1 } ^ { \prime } } Z ( \pmb { s } _ { 1 } , y _ { 1 } ^ { \prime } ) + \operatorname* { m a x } _ { y _ { 2 } ^ { \prime } } \big ( q _ { \alpha } ( \pmb { x } \oplus \widetilde { y } _ { 1 } , y _ { 2 } ^ { \prime } ) + Z ( \pmb { x } \oplus \widetilde { y } _ { 1 } , y _ { 2 } ^ { \prime } ) \big ) ,
$$

where $\widetilde { y } _ { 1 } = \mathrm { a r g m a x } _ { y _ { 1 } ^ { \prime } } Z ( \pmb { s } _ { 1 } , y _ { 1 } ^ { \prime } )$ . Taking expectations, the first term gives a constant $c : = \mathbb { E } _ { Z } [ \operatorname* { m a x } ( Z ( s _ { 1 } , 1 ) , Z ( s _ { 1 } , 2 ) ) ] .$ Since $q _ { \alpha } ( s _ { 1 } , 1 ) = q _ { \alpha } ( s _ { 1 } , 2 ) = 0$ , we have $\widetilde { y } _ { 1 } = 1$ or 2 with equal probability ${ \frac { 1 } { 2 } } .$ . By the law of total expectation,

$$
\begin{array} { r } { \mathbb { E } _ { Z } \left[ \underset { y _ { 2 } ^ { \prime } } { \operatorname* { m a x } } ( q _ { \alpha } + Z ) ( \boldsymbol { x } \oplus \widetilde { y } _ { 1 } , y _ { 2 } ^ { \prime } ) \right] = \frac { 1 } { 2 } \mathbb { E } _ { Z } [ \operatorname* { m a x } ( \alpha + Z ( \boldsymbol { x } \oplus 1 , 1 ) , Z ( \boldsymbol { x } \oplus 1 , 2 ) ) ] + \frac { 1 } { 2 } c . } \end{array}
$$

Without loss of generality, we assume standard Gumbel noise $( \mu = 0 , \beta = 1 )$ . By independence, the CDF of ma $\mathfrak { c } ( \alpha + Z ( \pmb { x } \oplus 1 , 1 ) , Z ( \pmb { x } \oplus 1 , 2 ) )$ is $F ( \zeta ) = \exp \bigl ( - e ^ { - \zeta } ( e ^ { \alpha } + 1 ) \bigr ) = \exp \bigl ( - e ^ { \log ( e ^ { \bar { \alpha } } + 1 ) - \zeta } \bigr )$ , which is Gumbel with location log $( 1 + e ^ { \alpha } )$ and scale 1. Its expectation is $\log ( 1 + e ^ { \alpha } ) + \gamma .$ , where $\gamma$ is the Euler–Mascheroni constant. Therefore,

$$
\begin{array} { r } { L _ { q _ { \alpha } } ^ { \mathrm { s a m p l i n g } } ( x , y ) = \frac { 3 } { 2 } c + \frac { 1 } { 2 } ( \log ( 1 + e ^ { \alpha } ) + \gamma ) - \alpha \underset { \alpha  + \infty } { \longrightarrow } - \infty . } \end{array}
$$

## A.4. Proof of autoregressive Gumbel tricks (Proposition 4)

Recall that $\begin{array} { r } { \widehat { \pmb { y } } _ { q } ^ { \mathrm { g r e e d y } } ( \pmb { x } ) = ( \widetilde { \pmb { y } } _ { 1 } , \dots , \widetilde { \pmb { y } } _ { T } ) } \end{array}$ is defined as

$$
\begin{array} { r l } & { \widetilde { y } _ { 1 } : = \underset { y _ { 1 } \in \mathcal { V } } { \mathrm { a r g m a x } } q ( \widetilde { s } _ { 1 } , y _ { 1 } ) \in \mathcal { V } } \\ & { \widetilde { y } _ { 2 } : = \underset { y _ { 2 } \in \mathcal { V } } { \mathrm { a r g m a x } } q ( \widetilde { s } _ { 2 } , y _ { 2 } ) \in \mathcal { V } , } \\ & { \quad \vdots } \\ & { \widetilde { y } _ { T } : = \underset { y _ { T } \in \mathcal { V } } { \mathrm { a r g m a x } } q ( \widetilde { s } _ { T } , y _ { T } ) \in \mathcal { V } , } \end{array}
$$

where $\tilde { \pmb { s } } _ { t } : = \pmb { x } \oplus \widetilde { \pmb { y } } _ { < t }$

Ancestral sampling. By successive applications of the Gumbel trick for categorical variables,

$$
\begin{array} { r l } & { Y _ { 1 } = \underset { y _ { 1 } \in \mathcal { V } } { \mathrm { a r g m a x } } ( q + Z ) ( S _ { 1 } , y _ { 1 } ) \sim \pi _ { q } ( \cdot | S _ { 1 } ) } \\ & { Y _ { 2 } = \underset { y _ { 2 } \in \mathcal { V } } { \mathrm { a r g m a x } } ( q + Z ) ( S _ { 2 } , y _ { 2 } ) \sim \pi _ { q } ( \cdot | S _ { 2 } ) } \\ & { \quad \quad \quad \quad \quad \quad \vdots } \\ & { Y _ { T } = \underset { y _ { T } \in \mathcal { V } } { \mathrm { a r g m a x } } ( q + Z ) ( S _ { T } , y _ { T } ) \sim \pi _ { q } ( \cdot | S _ { T } ) , } \end{array}
$$

where $S _ { t }$ now uses the perturbed greedy prefix.

Cumulant. We then have

$$
A _ { q + Z } ^ { \mathrm { g r e e d y } } ( \pmb { x } ) = \operatorname* { m a x } _ { y _ { 1 } \in \mathscr { V } } ( q + Z ) ( S _ { 1 } , y _ { 1 } ) + \operatorname* { m a x } _ { y _ { 2 } \in \mathscr { V } } ( q + Z ) ( S _ { 2 } , y _ { 2 } ) + . . . + \operatorname* { m a x } _ { y _ { T } \in \mathscr { V } } ( q + Z ) ( S _ { T } , y _ { T } ) .
$$

Therefore

$$
\begin{array} { r l } { A _ { \phi \phi } ^ { ( \mathbf { x } , \mathbf { u } ) , \mathbf { b } } ( \mathbf { x } _ { \phi } ) } & { = \mathbf { B } _ { z } [ \frac { \sin \phi } { \sin \phi } \mathbf { x } _ { \phi } ] ^ { \mathbf { A } } ( \mathbf { x } _ { \phi } ) } \\ & { = \mathbf { B } _ { z } [ \sin \phi - d ( \phi ( \mathbf { x } _ { \phi } , \mathbf { y } _ { \phi } ) )  } \\ & { \quad  - \mathbf { B } _ { z } [ \mathbf { B } _ { z } [ \mathbf { B } _ { z } [ \mathbf { B } _ { y } ( \mathbf { x } - \mathbf { Z } _ { z } ] ( \mathcal { B } _ { z } , \mathbf { y } _ { z } ) )    } \\ & { \quad    - \mathbf { B } _ { z } [ \mathbf { B } _ { y } [ \mathbf { B } _ { y } ( \mathbf { x } _ { \phi } ) ]     } \\ & { \quad     + \mathbf { B } _ { z } [ \mathbf { B } _ { z } [ \mathbf { B } _ { z } [ \mathbf { B } _ { y } ( \mathbf { x } _ { \phi } ) ]      } \\ & { \quad      - \mathbf { B } _ { z } [ \mathbf { B } _ { z } [ \mathbf { B } _ { z } [ \mathbf { B } _ { y } ( \mathbf { x } _ { \phi } ) ]      } \\ & { \quad     - \mathbf { B } _ { z } [ \mathbf { B } _ { z } [ \mathbf { B } _ { z } [ \mathbf { B } _ { y } ( \mathbf { B } _ { z } ) ]      } \\ &  \quad      - \mathbf { B } _ { z } [ \mathbf { B } _ { y } [ \mathbf { B } _ { z } [ \mathbf { B } _ { y } ( \mathbf { B } _  y \end{array}
$$

## A.5. The direction is not a gradient (Proof of Proposition 5)

If there existed a loss $L _ { q } ( \pmb { x } , \pmb { y } )$ such that $\nabla _ { q } L _ { q } ( { \pmb x } , { \pmb y } ) = { \pmb d } _ { q } ( { \pmb x } , { \pmb y } )$ , then the Jacobian $\partial d _ { q } ( { \pmb x } , { \pmb y } )$ would be self-adjoint. Since $d _ { q } ( x , y ) = \mu ( q , \pmb { x } ) - \phi ( \pmb { x } , \pmb { y } )$ , where $\mu ( q , \pmb { x } ) : = \mathbb { E } _ { Y \sim p _ { a } ^ { \mathrm { A R } } ( \cdot | \pmb { x } ) } [ \phi ( \pmb { x } , Y ) ]$ , the Jacobian $\partial _ { 1 } \mu ( q , { \pmb x } )$ would also be self-adjoint (that is, if we materialized the Jacobian as a matrix, it would be symmetric).

However, the step-1 component $[ \mu ( q , \pmb { x } ) ] ( \pmb { s } _ { 1 } , y _ { 1 } ) = \pi _ { q } ( y _ { 1 } | \pmb { s } _ { 1 } )$ depends only on $\{ q ( \pmb { \mathscr { s } } _ { 1 } , y ) : y \in \mathcal { V } \}$ , so

$$
[ \partial _ { 1 } \mu ( q , { \pmb x } ) \cdot h ] ( { \pmb s } _ { 1 } , y _ { 1 } ) = 0
$$

for any perturbation ℎ supported on steps $t \geq 2$ . On the other hand, for $t \ \geq \ 2$ , the component $[ \mu ( q , \pmb { x } ) ] ( \pmb { s } _ { t } , y _ { t } )$ involves $\pi _ { q } ( y _ { 1 } | \pmb { s } _ { 1 } ) \cdot \cdot \cdot \pi _ { q } ( y _ { t } | \pmb { s } _ { t } )$ and therefore depends on $q ( s _ { 1 } , \cdot )$ , so $[ \partial _ { 1 } \mu ( q , { \pmb x } )$ $g ] ( s _ { t } , y _ { t } ) \neq 0$ for generic perturbations � supported at step 1. The Jacobian is therefore not self-adjoint, a contradiction.

## A.6. Direction as preconditioned gradient (Proof of Proposition 6)

First claim. For any � and $( { \pmb x } , { \pmb y } )$ , we have

$$
\nabla _ { r } A _ { R _ { r } } ^ { \mathrm { E B M } } ( \pmb { x } ) = \mathbb { E } _ { Y \sim p _ { R _ { r } } ^ { \mathrm { E B M } } ( \cdot | \pmb { x } ) } [ \phi ( \pmb { x } , Y ) ] .
$$

Therefore,

$$
\begin{array} { r l } & { \nabla _ { r } L _ { R _ { r } } ^ { \mathrm { E B M } } ( { \pmb x } , { \pmb y } ) \big | _ { r = \mathcal { M } ( q ) } = \mathbb { E } _ { Y \sim p _ { R _ { r } } ^ { \mathrm { E B M } } ( \cdot | { \pmb x } ) } [ \phi ( { \pmb x } , Y ) ] - \phi ( { \pmb x } , { \pmb y } ) } \\ & { \qquad = \mathbb { E } _ { Y \sim p _ { q } ^ { \mathrm { A R } } ( \cdot | { \pmb x } ) } [ \phi ( { \pmb x } , Y ) ] - \phi ( { \pmb x } , { \pmb y } ) } \\ & { \qquad = d _ { q } ( { \pmb x } , { \pmb y } ) , } \end{array}
$$

where the second line uses $p _ { R _ { r } } ^ { \mathrm { E B M } } ( \cdot | \pmb { x } ) = p _ { q } ^ { \mathrm { A R } } ( \cdot | \pmb { x } )$ as proved by Blondel et al. (2025).

Second claim. Since $L _ { R _ { \mathcal { M } ( q ) } } ^ { \mathrm { E B M } } ( \pmb { x } , \pmb { y } ) = L _ { q } ^ { \mathrm { A R } } ( \pmb { x } , \pmb { y } )$ , we get by the chain rule

$$
\nabla _ { q } L _ { q } ^ { \mathrm { A R } } ( \pmb { x } , \pmb { y } ) = \partial \mathcal { M } ( q ) ^ { * } \left. \nabla _ { r } L _ { R _ { r } } ^ { \mathrm { E B M } } ( \pmb { x } , \pmb { y } ) \right| _ { r = \mathcal { M } ( q ) } = \partial \mathcal { M } ( q ) ^ { * } d _ { q } ( \pmb { x } , \pmb { y } ) .
$$

## A.7. Descent direction (Proof of Proposition 7)

We decompose any function ℎ on state-action pairs by step: $h \ : = \ : ( h _ { 1 } , \ldots , h _ { T } )$ , where $h _ { t }$ is the restriction of ℎ to step �.

Jacobian of ℳ. Since $[ \mathcal { M } ( \boldsymbol { q } ) ] ( \pmb { \mathscr { s } } _ { t } , \boldsymbol { y } _ { t } ) = \boldsymbol { q } ( \pmb { \mathscr { s } } _ { t } , \boldsymbol { y } _ { t } ) - V _ { \boldsymbol { q } } ( \pmb { \mathscr { s } } _ { t } \oplus \boldsymbol { y } _ { t } )$ for $t < T$ and $[ \mathcal { M } ( q ) ] ( s _ { T } , y _ { T } ) \stackrel { } { = }$ $q ( s _ { T } , y _ { T } )$ , the Jacobian of ℳ at $q$ acts on a perturbation ℎ as

$$
[ \partial \mathcal { M } ( q ) \cdot h ] ( s _ { t } , y _ { t } ) = \left\{ \begin{array} { l l } { h ( s _ { t } , y _ { t } ) - [ J _ { t + 1 } h _ { t + 1 } ] ( s _ { t } , y _ { t } ) } & { \mathrm { i f ~ } t < T } \\ { h ( s _ { t } , y _ { t } ) } & { \mathrm { i f ~ } t = T , } \end{array} \right.
$$

where the linear operator $J _ { t + 1 }$ is defined by

$$
[ J _ { t + 1 } h _ { t + 1 } ] ( s _ { t } , y _ { t } ) : = \mathbb { E } _ { Y _ { t + 1 } \sim \pi _ { q } ( \cdot | s _ { t + 1 } ) } [ h _ { t + 1 } ( s _ { t + 1 } , Y _ { t + 1 } ) ] , \quad \mathrm { w i t h ~ } s _ { t + 1 } : = s _ { t } \oplus y _ { t } .
$$

Indeed, this follows from a Taylor expansion of $V _ { q }$ . Since $\begin{array} { r } { V _ { q } ( \pmb { \mathscr { s } } _ { t + 1 } ) = \log \sum _ { y \in \mathcal { V } } \exp ( q ( \pmb { \mathscr { s } } _ { t + 1 } , y ) ) } \end{array}$ , we have

$$
\begin{array} { r l } & { V _ { q + \epsilon h } \big ( s _ { t + 1 } \big ) = \displaystyle \log \sum _ { y \in \mathcal { V } } \exp \big ( q ( s _ { t + 1 } , y ) + \epsilon h \big ( s _ { t + 1 } , y \big ) \big ) } \\ & { \quad \quad \quad = \displaystyle \log \sum _ { y \in \mathcal { V } } \exp \big ( q \big ( s _ { t + 1 } , y \big ) \big ) \big ( 1 + \epsilon h \big ( s _ { t + 1 } , y \big ) + o ( \epsilon ) \big ) } \\ & { \quad \quad \quad = \displaystyle \log \left[ \left( \sum _ { y \in \mathcal { V } } \exp \big ( q \big ( s _ { t + 1 } , y \big ) \big ) \right) \left( 1 + \epsilon \sum _ { y \in \mathcal { V } } \pi _ { q } \big ( y | s _ { t + 1 } \big ) h \big ( s _ { t + 1 } , y \big ) + o ( \epsilon ) \right) \right] } \\ & { \quad \quad \quad = V _ { q } \big ( s _ { t + 1 } \big ) + \displaystyle \log \big ( 1 + \epsilon \mathbb { E } _ { \mathcal { Y } + 1 \sim \pi \pi _ { q } ( \cdot \vert s _ { t + 1 } ) } \big [ h _ { t + 1 } \big ( s _ { t + 1 } , Y _ { t + 1 } \big ) \big ] + o ( \epsilon ) \big ) } \\ & { \quad \quad \quad = V _ { q } \big ( s _ { t + 1 } \big ) + \epsilon \mathbb { E } _ { \mathcal { Y } + 1 \sim \pi _ { q } ( \cdot \vert s _ { t + 1 } ) } \big [ h _ { t + 1 } \big ( s _ { t + 1 } , Y _ { t + 1 } \big ) \big ] + o ( \epsilon ) , } \end{array}
$$

where we used $\exp ( a + \epsilon b ) = \exp ( a ) ( 1 + \epsilon b + o ( \epsilon ) ) { \mathrm { ~ a n d ~ } } \log ( 1 + u ) = u + o ( u ) .$

Contraction property. We claim that $\| J _ { t + 1 } h _ { t + 1 } \| \leq \| h _ { t + 1 } \|$ . Indeed, by Jensen’s inequality (the square is convex),

$$
\begin{array} { r } { \| J _ { t + 1 } h _ { t + 1 } \| ^ { 2 } = \displaystyle \sum _ { s _ { t } , y _ { t } } \left( \mathbb E _ { Y _ { t + 1 } \sim \pi _ { q } ( \cdot | s _ { t + 1 } ) } [ h _ { t + 1 } ( s _ { t + 1 } , Y _ { t + 1 } ) ] \right) ^ { 2 } } \\ { \le \displaystyle \sum _ { s _ { t } , y _ { t } } \displaystyle \sum _ { y _ { t + 1 } } \pi _ { q } ( y _ { t + 1 } | s _ { t + 1 } ) h _ { t + 1 } ( s _ { t + 1 } , y _ { t + 1 } ) ^ { 2 } . } \end{array}
$$

Since the map $( \pmb { s } _ { t } , y _ { t } ) \mapsto \pmb { s } _ { t + 1 }$ is a bijection and each weight satisfies $\pi _ { q } ( y _ { t + 1 } | \boldsymbol { s } _ { t + 1 } ) \leq 1$ , we obtain $\begin{array} { r } { \| J _ { t + 1 } h _ { t + 1 } \| ^ { 2 } \leq \sum _ { s _ { t + 1 } , y _ { t + 1 } } h _ { t + 1 } ( s _ { t + 1 } , y _ { t + 1 } ) ^ { 2 } = \| h _ { t + 1 } \| ^ { 2 } } \end{array}$

Positive definiteness. We have

$$
\langle h , \partial { \mathcal { M } } ( q ) \cdot h \rangle = \sum _ { t = 1 } ^ { T } \| h _ { t } \| ^ { 2 } - \sum _ { t = 2 } ^ { T } \langle h _ { t - 1 } , J _ { t } h _ { t } \rangle .
$$

By Cauchy–Schwarz and the contraction property,

$$
\sum _ { t = 2 } ^ { T } \langle h _ { t - 1 } , J _ { t } h _ { t } \rangle \leq \sum _ { t = 2 } ^ { T } \| h _ { t - 1 } \| \| h _ { t } \| .
$$

Since for any real numbers $a , b$ we have $a b \leq \frac { 1 } { 2 } ( a ^ { 2 } + b ^ { 2 } )$

$$
\begin{array} { l } { \displaystyle \sum _ { t = 2 } ^ { T } \| h _ { t - 1 } \| \| h _ { t } \| \leq \displaystyle \frac { 1 } { 2 } \sum _ { t = 2 } ^ { T } ( \| h _ { t - 1 } \| ^ { 2 } + \| h _ { t } \| ^ { 2 } ) } \\ { \displaystyle \qquad = \frac { 1 } { 2 } \| h _ { 1 } \| ^ { 2 } + \displaystyle \sum _ { t = 2 } ^ { T - 1 } \| h _ { t } \| ^ { 2 } + \displaystyle \frac { 1 } { 2 } \| h _ { T } \| ^ { 2 } \leq \displaystyle \sum _ { t = 1 } ^ { T } \| h _ { t } \| ^ { 2 } . } \end{array}
$$

Moreover, $\begin{array} { r } { a b = \frac { 1 } { 2 } ( a ^ { 2 } + b ^ { 2 } ) } \end{array}$ holds if and only if $a = b$ . Thus, equality in the first step above holds if $\| h _ { t } \| = \alpha$ for all $t ,$ in which case $\begin{array} { r } { \sum _ { t = 2 } ^ { T } \| h _ { t - 1 } \| \| h _ { t } \| = ( T - 1 ) \alpha ^ { 2 } < T \alpha ^ { 2 } = \sum _ { t = 1 } ^ { T } \| h _ { t } \| ^ { 2 } } \end{array}$ whenever $\alpha \neq 0$ Therefore $\langle h , \partial \mathcal { M } ( q ) \cdot h \rangle > 0$ for all $h \neq 0$

Conclusion. Using Proposition 6:

$$
\langle - d _ { q } ( { \pmb x } , { \pmb y } ) , \nabla _ { q } L _ { q } ^ { \mathrm { A R } } ( { \pmb x } , { \pmb y } ) \rangle = - \langle d _ { q } ( { \pmb x } , { \pmb y } ) , \partial \mathcal { M } ( q ) ^ { * } d _ { q } ( { \pmb x } , { \pmb y } ) \rangle < 0 ,
$$

for any $q , \pmb { x } , \pmb { y }$ such that ${ d _ { q } } ( { \pmb x } , { \pmb y } ) \ne 0$

## References

R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The Twelfth International Conference on Learning Representations, 2024.

M. Bain and C. Sammut. A framework for behavioural cloning. In Machine Intelligence 15, pages 103–129. Oxford University Press, 1995.

S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.

M. Blondel and V. Roulet. The Elements of Diferentiable Programming. arXiv preprint arXiv:2403.14606, 2024.

M. Blondel, F. Llinares-López, R. Dadashi, L. Hussenot, and M. Geist. Learning energy networks with generalized Fenchel–young losses. Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 35:12516–12528, 2022.

M. Blondel, M. E. Sander, G. Vivier-Ardisson, T. Liu, and V. Roulet. Autoregressive language models are secretly energy-based models: Insights into the lookahead capabilities of next-token prediction. arXiv preprint arXiv:2512.15605, 2025.

M. Collins. Discriminative training methods for hidden markov models: Theory and experiments with perceptron algorithms. In Proceedings of the 2002 conference on empirical methods in natural language processing (EMNLP 2002), pages 1–8, 2002.

D. J. Foster, A. Block, and D. Misra. Is behavior cloning all you need? understanding horizon in imitation learning. Advances in Neural Information Processing Systems, 37:120602–120666, 2024.

W. Kool, H. Van Hoof, and M. Welling. Stochastic beams and where to find them: The gumbel-top-k trick for sampling sequences without replacement. In International conference on machine learning, pages 3499–3508. PMLR, 2019.

Y. LeCun, S. Chopra, R. Hadsell, M. Ranzato, F. Huang, et al. A tutorial on energy-based learning. Predicting structured data, 1(0), 2006.

A. Mensch and M. Blondel. Diferentiable dynamic programming for structured prediction and attention. In International Conference on Machine Learning, pages 3462–3471. PMLR, 2018.

T. Mihaylova and A. F. Martins. Scheduled sampling for transformers. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop, pages 351–356, 2019.

M. Norouzi, S. Bengio, N. Jaitly, M. Schuster, J. Shlens, D. Amodei, and N. Morozof. Rewardaugmented maximum likelihood for sequence fine-tuning. In Advances in Neural Information Processing Systems (NeurIPS), volume 29, 2016.

D. A. Pomerleau. ALVINN: An autonomous land vehicle in a neural network. In Advances in Neural Information Processing Systems, volume 1, pages 305–313, 1989.

M. Ranzato, S. Chopra, M. Auli, and W. Zaremba. Sequence level training with recurrent neural networks. arXiv preprint arXiv:1511.06732, 2015.

S. J. Rennie, E. Marcheret, Y. Mroueh, J. Ross, and V. Goel. Self-critical sequence training for image captioning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 7008–7024, 2017.

S. Shen, Y. Liu, Q. Liu, and H. Luan. Minimum risk training for neural machine translation. In Proceedings of the 54th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1683–1692, 2016.

Y. Song and D. P. Kingma. How to train your energy-based models. arXiv preprint arXiv:2101.03288, 2021.

M. J. Wainwright, M. I. Jordan, et al. Graphical models, exponential families, and variational inference. 2008.

R. J. Williams and D. Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural computation, 1(2):270–280, 1989.