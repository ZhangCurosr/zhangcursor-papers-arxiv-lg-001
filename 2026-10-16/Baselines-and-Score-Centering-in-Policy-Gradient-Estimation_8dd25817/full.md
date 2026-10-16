# Baselines and Score Centering in Policy Gradient Estimation

Shuo Han<sup>∗1</sup> and Ziqing Hu<sup>†2</sup>

<sup>1</sup>University of Illinois Chicago <sup>2</sup>Perplexity AI

October 15, 2026

## Abstract

Reinforcement learning has become a widely used approach to post-training large language models. In a typical implementation, learning is of-policy: The policy generating the responses difers from the policy being trained because of training–inference mismatch or policy staleness in asynchronous training. This discrepancy is commonly addressed by importance sampling, whose ratio is usually regularized in practice through truncation, masking, or clipping. Under regularized importance sampling, the importance-weighted policy gradient is no longer invariant to constant shifts in the reward. This invariance is recovered by using the covariance between the reward and the importance-weighted score as the estimand, which coincides with the policy gradient when the ratio is not regularized. This note develops this covariance perspective for both sequence-level and token-level importance sampling. The covariance perspective reveals the roles of the group-mean baseline and score centering in policy gradient estimation. Both perform centering in covariance estimation, but they difer in the mean being subtracted: The group-mean baseline uses a sample mean, whereas score centering uses a population mean. The two lead to distinct unbiased covariance estimators with diferent variances. Whether both forms of centering should be used simultaneously depends on whether importance sampling is applied at the sequence level or at the token level. With sequence-level importance sampling, score centering adds no benefit once the group-mean baseline is used; with token-level importance sampling, using the group-mean baseline without score centering generally introduces bias when the importance ratio is regularized.

## 1 Background

## 1.1 Post-training of language models

In post-training of language models, the goal is to make the language model generate a desired response for a given prompt. Denote by $x \in \mathcal { X }$ the prompt and $y \in \mathcal { V }$ the response. The language model can be viewed as a policy $\pi ,$ where $\pi ( y \mid x )$ denotes the probability of generating y given the prompt x. The response y consists of a sequence of tokens $\{ y _ { t } \} _ { t \ge 0 }$ , each of which is randomly generated based on past tokens and the prompt. The generation of one token is known as a decoding step. We use π both for the distribution over complete responses and, as in $\pi ( y _ { t } \mid x , y _ { < t } )$ , for the conditional distribution of a single token given the prompt x and the history $y _ { < t } \triangleq ( y _ { 0 } , \ldots , y _ { t - 1 } )$ when it is clear from the context. Thus, for any response $y ,$ it holds that

$$
\pi ( y \mid x ) = \prod _ { t = 0 } ^ { | y | - 1 } \pi ( y _ { t } \mid x , y _ { < t } ) .
$$

The quality of $y$ for a given $x$ is measured by a reward function $R \colon \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ as $R ( x , y )$ . Let D be the distribution of prompts. The goal is to find $\pi$ that maximizes

$$
\mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi ( \cdot | x ) } \left[ R ( x , y ) \right] .\tag{1}
$$

Throughout, we assume that generation terminates with probability one under every policy under consideration, so that $| y | < \infty$ almost surely and the expectation in (1) is well defined. In practice, this assumption is enforced by imposing a maximum response length.

## 1.2 Token generation as a Markov decision process

The token generation process can be modeled as an undiscounted indefinite-horizon Markov decision process (MDP). Let the state space $\boldsymbol { s }$ consist of sequences of tokens whose prefix is a valid prompt, and the action space $\mathcal { A }$ be the space of tokens. The idea is to use the state to store the prompt and all the tokens generated thus far, and to use the action to represent the next generated token:

$$
s _ { t } \triangleq ( x , y _ { < t } ) , \qquad a _ { t } \triangleq y _ { t } .
$$

The initial state is chosen as $s _ { 0 } ~ = ~ x$ . At time $t ,$ the action $a _ { t }$ is determined by the policy as $a _ { t } \sim \pi ( \cdot \mid s _ { t } )$ . The transition dynamics are deterministic: $s _ { t + 1 } = ( x , y _ { \leq t } )$ is the concatenation of $s _ { t }$ and $a _ { t }$ unless $a _ { t } = \mathrm { E O S }$ , the end-of-sequence token. If $a _ { t } = \mathrm { E O S }$ , the next state $s _ { t + 1 }$ becomes an absorbing state associated with no reward. The original reward function R induces a token-level reward $r \colon S \times \mathcal { A }  \mathbb { R }$ . During token generation, if $a _ { t } = \mathrm { E O S }$ , then $r ( s _ { t } , a _ { t } ) = R ( x , y _ { \leq t } )$ , where $y { \le } t$ denotes the completed response including the terminal EOS token, so that $| y | = t + 1 ;$ ; otherwise $r ( s _ { t } , a _ { t } ) = 0$ . In other words, the reward is sparse and assigned only at the last decoding step. Under the MDP framework, the training objective in (1) can be equivalently rewritten as

$$
\mathbb { E } _ { { x } \sim { D } , { y } \sim \pi ( \cdot \vert x ) } \left[ R ( x , y ) \right] = \mathbb { E } _ { s _ { 0 } \sim { D } } ^ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } r ( s _ { t } , a _ { t } ) \right] .
$$

Define the state value, state-action value, and advantage functions of a policy π by

$$
V ^ { \pi } ( s ) \triangleq \mathbb { E } ^ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } r ( s _ { t } , a _ { t } ) \Bigg | \ s _ { 0 } = s \right] , \qquad Q ^ { \pi } ( s , a ) \triangleq \mathbb { E } ^ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } r ( s _ { t } , a _ { t } ) \Bigg | \ s _ { 0 } = s , a _ { 0 } = a \right] ,
$$

$$
A ^ { \pi } ( s , a ) \triangleq Q ^ { \pi } ( s , a ) - V ^ { \pi } ( s ) .
$$

In what follows, we assume that the policy is parametrized by $\theta \in \Theta$ for some set $\Theta$ , and we denote the policy by $\pi _ { \theta }$ to highlight the parameter dependence. The training objective in (1) thus becomes a function of $\theta ,$ defined by

$$
J ( \theta ) \triangleq \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot | x ) } \left[ R ( x , y ) \right] .\tag{2}
$$

## 2 Importance Sampling in Policy Optimization

The training objective in (2) involves an expectation over the prompt distribution D and the distribution $\pi _ { \boldsymbol { \theta } } ( \cdot \mid x )$ over possible responses. Because $\mathcal { D }$ is unknown and the space of responses is prohibitively large to enumerate, it is intractable to compute the expectation exactly. Instead, the expectation is typically replaced by a sample-average approximation using sampled token sequences. A naive application of sample-average approximation to (2) requires these samples to be drawn from $\pi _ { \theta }$

However, modern post-training systems for reinforcement learning typically use separate engines with diferent design objectives. A training engine, built on $\mathrm { P y }$ Torch FSDP (Zhao et al., 2023) or Megatron-LM (Shoeybi et al., 2019), supports eficient gradient computation and stable updates to $\theta ,$ whereas an inference engine, e.g., vLLM (Kwon et al., 2023) and SGLang (Zheng et al., 2024), prioritizes generation throughput. Their distinct numerical implementations can yield diferent token probabilities even with identical policy parameters. This discrepancy is referred to as training– inference mismatch (Liu et al., 2025a). For a given parameter $\theta ,$ let $\pi _ { \theta }$ and $\hat { \pi }$ denote the policies implemented by the training and inference engines, respectively.

When samples are drawn from a diferent policy, importance sampling is a common technique for obtaining an unbiased estimate. It works by weighting the samples using an importance sampling ratio. Common formulations use two types of importance sampling ratios. One is sequence-level importance sampling, which uses

$$
{ \frac { \pi _ { \theta } ( y \mid x ) } { { \hat { \pi } } ( y \mid x ) } } = \prod _ { t = 0 } ^ { | y | - 1 } { \frac { \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) } { { \hat { \pi } } ( y _ { t } \mid x , y _ { < t } ) } } .
$$

The other is token-level importance sampling, which uses the individual ratios

$$
{ \frac { \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) } { { \hat { \pi } } ( y _ { t } \mid x , y _ { < t } ) } } , \quad t = 0 , \dots , | y | - 1
$$

rather than their product as a whole. As shown below, the two types of importance sampling ratios lead to diferent estimators, which are unbiased for diferent quantities.

While sampling a response y from $\pi _ { \theta }$ must be done sequentially, scoring $\pi _ { \theta } ( y \mid x )$ when y is given can be parallelized. Only one forward pass is required to compute all the $| y |$ quantities $\pi _ { \theta } ( y _ { 0 } \mid x )$ $\pi _ { \boldsymbol { \theta } } ( y _ { 1 } \mid x , y _ { 0 } ) , . . . .$ , and this is typically much faster than sampling. Both formulations require the inference engine to return the token probabilities $\hat { \boldsymbol { \tau } } ( y _ { t } \mid x , y _ { < t } )$ it used at decoding time.

## 2.1 Methods using sequence-level importance sampling

## 2.1.1 REINFORCE

The REINFORCE algorithm (Williams, 1992) updates θ by performing gradient ascent on the training objective J in (2). It is well known (see, $\mathrm { e . g . }$ , Sutton and Barto, 2018) that the gradient is given by

$$
\nabla J ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot \vert x ) } \left[ R ( x , y ) \nabla _ { \theta } \log \pi _ { \theta } ( y \mid x ) \right] .\tag{3}
$$

To use samples drawn from the inference policy $\hat { \pi }$ , we rewrite (3) as

$$
\nabla J ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \hat { \pi } ( \cdot \mid x ) } \left[ \frac { \pi _ { \theta } ( y \mid x ) } { \hat { \pi } ( y \mid x ) } R ( x , y ) \nabla _ { \theta } \log \pi _ { \theta } ( y \mid x ) \right] ,\tag{4}
$$

assuming that ${ \hat { \pi } } ( y \mid x ) > 0$ whenever $\pi _ { \theta } ( y \mid x ) > 0$ . For a given prompt x, draw samples $y _ { 1 } , \dotsc , y _ { G } \sim$ ${ \hat { \pi } } ( \cdot \mid x )$ . From the samples, $\nabla J ( \theta )$ can be approximated by a sample-average approximation as

$$
\nabla J ( \theta ) \approx \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { \pi _ { \theta } ( y _ { i } \mid x ) } { \hat { \pi } ( y _ { i } \mid x ) } R ( x , y _ { i } ) \nabla _ { \theta } \log { \pi _ { \theta } ( y _ { i } \mid x ) } .\tag{5}
$$

It can be shown that the right side of (5) is an unbiased estimator of $\nabla J ( \theta )$

Remark 1. Strictly speaking, for a fixed prompt x, the estimator in (5) is only unbiased for the inner expectation over $y \sim { \hat { \pi } } ( \cdot \mid x )$ in (4). It is unbiased for $\nabla J ( \theta )$ when the prompt x is drawn from D. This is what happens in practice, where a batch of prompts is sampled and the per-prompt estimators are averaged. For brevity, we will assume that x is drawn from $\mathcal { D }$ in all subsequent discussions unless mentioned otherwise.

## 2.1.2 Policy gradient with a baseline

To reduce the variance of sample-average approximation, one important technique is introducing a baseline (Greensmith et al., 2004) to the reward in $\nabla J ( \theta )$ . Let b be a baseline, which may be random and may depend on the prompt x. It turns out that $\nabla J ( \theta )$ remains unchanged after b is introduced to $R ( x , y )$ , i.e.,

$$
\nabla J ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \hat { \pi } ( \cdot \vert x ) } \left[ \frac { \pi _ { \theta } ( y \vert x ) } { \hat { \pi } ( y \vert x ) } \left( R ( x , y ) - b \right) \nabla _ { \theta } \log \pi _ { \theta } ( y \vert x ) \right]
$$

provided that b is conditionally independent of y given x. This can be shown by recalling the fact

$$
\mathbb { E } _ { y \sim \hat { \pi } ( \cdot \vert x ) } \left[ \frac { \pi _ { \theta } ( y \mid x ) } { \hat { \pi } ( y \mid x ) } \nabla _ { \theta } \log \pi _ { \theta } ( y \mid x ) \right] = \mathbb { E } _ { y \sim \pi _ { \theta } ( \cdot \vert x ) } \left[ \nabla _ { \theta } \log \pi _ { \theta } ( y \mid x ) \right] = 0 .
$$

If b and y are conditionally independent given x, the extra term contributed by the baseline becomes

$$
\mathbb { E } _ { y \sim \hat { \pi } ( \cdot \vert x ) } \left[ b \cdot \frac { \pi _ { \theta } ( y \mid x ) } { \hat { \pi } ( y \mid x ) } \nabla _ { \theta } \log \pi _ { \theta } ( y \mid x ) \right] = b \cdot \mathbb { E } _ { y \sim \hat { \pi } ( \cdot \vert x ) } \left[ \frac { \pi _ { \theta } ( y \mid x ) } { \hat { \pi } ( y \mid x ) } \nabla _ { \theta } \log \pi _ { \theta } ( y \mid x ) \right] = 0 .
$$

The use of a baseline can be extended to multiple samples. Let $y _ { 1 } , \dots , y _ { G } \sim { \hat { \pi } } ( \cdot \mid x )$ be drawn independently and let $b _ { 1 } , \dots , b _ { G }$ be baselines. If $b _ { i }$ is conditionally independent of y<sub>i</sub> given x for $i = 1 , \ldots , G$ , it can be shown that the sample average

$$
\frac { 1 } { G } \sum _ { i = 1 } ^ { G } \left[ \frac { \pi _ { \theta } ( y _ { i } \mid x ) } { \hat { \pi } ( y _ { i } \mid x ) } \left( R ( x , y _ { i } ) - b _ { i } \right) \nabla _ { \theta } \log \pi _ { \theta } ( y _ { i } \mid x ) \right]\tag{6}
$$

is an unbiased estimator of $\nabla J ( \theta )$ . Note that $b _ { i }$ is allowed to depend on the other samples $\{ y _ { j } \} _ { j \neq i }$ because those are independent of y given x.

One popular choice of the baseline, as in the REINFORCE leave-one-out (RLOO) algorithm (Kool et al., 2019), is setting

$$
b _ { i } = \frac { 1 } { G - 1 } \sum _ { j \neq i } R ( x , y _ { j } ) , \quad i = 1 , \ldots , G ,\tag{7}
$$

which is the empirical mean reward of the group with the ith sample removed. Define

$$
\bar { R } \triangleq \frac { 1 } { G } \sum _ { j = 1 } ^ { G } R ( x , y _ { j } ) .\tag{8}
$$

Using the fact

$$
\frac { 1 } { G } \left( R ( x , y _ { i } ) - b _ { i } \right) = \frac { 1 } { G - 1 } \left( R ( x , y _ { i } ) - \bar { R } \right) ,\tag{9}
$$

one can rewrite (6) as

$$
\frac { 1 } { G - 1 } \sum _ { i = 1 } ^ { G } \left[ \frac { \pi _ { \theta } ( y _ { i } \mid x ) } { \hat { \pi } ( y _ { i } \mid x ) } \left( R ( x , y _ { i } ) - \bar { R } \right) \nabla _ { \theta } \log \pi _ { \theta } ( y _ { i } \mid x ) \right] .\tag{10}
$$

Remark 2. Equation (9) shows that the leave-one-out baseline in (7) is closely related to the groupmean baseline, defined by $b _ { i } = \bar { R }$ for $i = 1 , \ldots , G$ . The group-mean baseline is used by the Dr. GRPO algorithm (Liu et al., 2025b), a variant of the group relative policy optimization algorithm (Shao et al., 2024), and leads to the following gradient estimator:

$$
\frac { 1 } { G } \sum _ { i = 1 } ^ { G } \left[ \frac { \pi _ { \theta } ( y _ { i } \mid x ) } { \hat { \pi } ( y _ { i } \mid x ) } \left( R ( x , y _ { i } ) - \bar { R } \right) \nabla _ { \theta } \log \pi _ { \theta } ( y _ { i } \mid x ) \right] .
$$

Because $b _ { i } = \bar { R }$ contains $R ( x , y _ { i } )$ , it is not conditionally independent of $y _ { i }$ . However, the resulting gradient estimator only difers from the RLOO gradient estimator by a deterministic multiplicative factor $( G - 1 ) / G$ , which is absorbed into the learning rate in practice. In other words, Dr. GRPO is practically equivalent to the RLOO algorithm.

## 2.2 Methods using token-level importance sampling

## 2.2.1 Trust-region policy optimization and proximal policy optimization

Methods that use token-level importance sampling consider a surrogate objective function (Kakade and Langford, 2002)

$$
\hat { J } ( \theta ) \triangleq \mathbb { E } _ { s \sim d _ { \hat { \pi } } , a \sim \pi _ { \theta } ( \cdot | s ) } \left[ A ^ { \hat { \pi } } ( s , a ) \right] ,
$$

where

$$
d _ { \hat { \pi } } ( s ) = \sum _ { t = 0 } ^ { \infty } \mathbb { P } ^ { \hat { \pi } } \left( s _ { t } = s \mid s _ { 0 } \sim \mathcal { D } \right)
$$

is the (unnormalized) state occupancy measure induced by $\hat { \pi }$ and the prompt distribution D. Because $d _ { \hat { \pi } }$ is unnormalized, the slightly abused notation $s \sim d _ { \hat { \pi } }$ denotes integration against this measure. We briefly recall why $\hat { J }$ is a useful surrogate function to optimize by examining the behavior of the training objective in the policy space; namely, we will examine how $J ( \theta )$ changes with the policy $\pi _ { \theta }$ rather than θ itself. Define

$$
V _ { 0 } ( \pi ) \triangleq \mathbb { E } _ { s _ { 0 } \sim \mathcal { D } } \left[ V ^ { \pi } ( s _ { 0 } ) \right] = \mathbb { E } _ { s _ { 0 } \sim \mathcal { D } } ^ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } r ( s _ { t } , a _ { t } ) \right] ,
$$

which is the value of a given policy. The value of a parametrized policy $\pi _ { \theta }$ recovers the training objective: $J ( \theta ) = V _ { 0 } ( \pi _ { \theta } )$ . Denote by $\hat { V } _ { 0 } ( \cdot ; \hat { \pi } )$ the linear approximation of $V _ { 0 }$ at πˆ:

$$
\hat { V } _ { 0 } ( \pi ; \hat { \pi } ) \triangleq V _ { 0 } ( \hat { \pi } ) + D V _ { 0 } ( \hat { \pi } ) \cdot ( \pi - \hat { \pi } ) ,
$$

where $D V _ { 0 } ( \hat { \pi } )$ denotes the derivative of $V _ { 0 }$ at πˆ. It can be shown that

$$
\hat { J } ( \theta ) = \hat { V } _ { 0 } ( \pi _ { \theta } ; \hat { \pi } ) - V _ { 0 } ( \hat { \pi } ) = D V _ { 0 } ( \hat { \pi } ) \cdot ( \pi _ { \theta } - \hat { \pi } ) .\tag{11}
$$

One can see from (11) that $\hat { J } ( \theta )$ is afine in $\pi _ { \theta }$ . Informally, $\hat { J }$ can be viewed as a linear approximation of $J$ in the policy space, up to an additive constant.

The relationship between $\hat { J }$ and $\hat { V } _ { 0 }$ reveals why $\hat { J }$ is a useful surrogate objective to optimize. Because $\hat { J } ( \theta )$ and $\hat { V } _ { 0 } ( \pi _ { \theta } ; \hat { \pi } )$ difer only by a constant independent of $\theta _ { ; }$ , maximizing $\hat { J } ( \theta )$ is equivalent to maximizing $\hat { V } _ { 0 } ( \pi _ { \theta } ; \hat { \pi } )$ . If π improves over $\hat { \pi }$ in $\hat { V } _ { 0 } ( \cdot ; \hat { \pi } )$ , i.e., $\hat { V } _ { 0 } ( \pi _ { \theta } ; \hat { \pi } ) > \hat { V } _ { 0 } ( \hat { \pi } ; \hat { \pi } ) = V _ { 0 } ( \hat { \pi } )$ , this implies $D V _ { 0 } ( \hat { \pi } ) \cdot ( \pi _ { \theta } - \hat { \pi } ) > 0$ , indicating that $\pi _ { \boldsymbol { \theta } } - { \hat { \pi } }$ is an ascent direction in the policy space. When $\pi _ { \theta }$ is suficiently close to ${ \hat { \pi } } ,$ it follows that $V _ { 0 } ( \pi _ { \theta } ) > V _ { 0 } ( \hat { \pi } )$ , i.e., $\pi _ { \theta }$ improves over $\hat { \pi }$ in the true objective. In the constrained version of trust-region policy optimization (TRPO) (Schulman et al., 2015), this condition is enforced by a trust-region constraint that restricts the divergence $D _ { \mathrm { K L } } ^ { \mathrm { m a x } } ( \pi _ { \theta } \parallel \hat { \pi } )$

To optimize $\hat { J }$ using a first-order method, one can obtain $\nabla \hat { J } ( \theta )$ as

$$
\begin{array} { r } { \nabla \hat { J } ( \theta ) = \mathbb { E } _ { s \sim d _ { \hat { \pi } } , a \sim \pi _ { \theta } ( \cdot | s ) } \left[ A ^ { \hat { \pi } } ( s , a ) \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) \right] . } \end{array}\tag{12}
$$

In comparison, it can be shown that $\nabla J ( \theta )$ can be rewritten as

$$
\nabla J ( \theta ) = \mathbb { E } _ { s \sim d _ { \pi _ { \theta } } , a \sim \pi _ { \theta } ( \cdot \vert s ) } \left[ A ^ { \pi _ { \theta } } ( s , a ) \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) \right]
$$

under mild conditions. This shows that generally $\nabla \hat { J } ( \theta ) \neq \nabla J ( \theta )$ , i.e., $\hat { J }$ is not a linear approximation of $J$ in the parameter space. To use samples from $\hat { \pi }$ , we rewrite (12) as

$$
\begin{array} { r l } & { \nabla \hat { J } ( \theta ) = \mathbb { E } _ { s \sim d _ { \hat { \pi } } , a \sim \hat { \pi } ( \cdot \vert s ) } \left[ \frac { \pi _ { \theta } ( a \mid s ) } { \hat { \pi } ( a \mid s ) } A ^ { \hat { \pi } } ( s , a ) \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { s _ { 0 } \sim \mathcal { D } } ^ { \hat { \pi } } \left[ \displaystyle \sum _ { t = 0 } ^ { \infty } \frac { \pi _ { \theta } ( a _ { t } \mid s _ { t } ) } { \hat { \pi } ( a _ { t } \mid s _ { t } ) } A ^ { \hat { \pi } } ( s _ { t } , a _ { t } ) \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } \mid s _ { t } ) \right] . } \end{array}
$$

The importance sampling ratio corrects the action distribution at each visited state, but the states are still distributed according to $d _ { \hat { \pi } }$ . Using the fact $\boldsymbol { s } _ { t } = \left( x , y _ { < t } \right)$ and $a _ { t } = y _ { t }$ , we can further write

$$
\nabla \hat { J } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \hat { \pi } ( \cdot | x ) } \left[ \sum _ { t = 0 } ^ { | y | - 1 } \frac { \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) } { \hat { \pi } ( y _ { t } \mid x , y _ { < t } ) } A ^ { \hat { \pi } } ( ( x , y _ { < t } ) , y _ { t } ) \nabla _ { \theta } \log { \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) } \right] ,\tag{13}
$$

where the sum terminates at $t = | y | - 1$ because the absorbing state reached after EOS has no contribution. For a given prompt $x _ { i }$ sample $y _ { 1 } , \dots , y _ { G } \sim { \hat { \pi } } ( \cdot \mid x )$ , with which $\nabla \hat { J } ( \theta )$ can be approximated as

$$
\frac { 1 } { G } \sum _ { i = 1 } ^ { G } \sum _ { t = 0 } ^ { | y _ { i } | - 1 } \frac { \pi _ { \theta } ( y _ { i , t } \mid x , y _ { i , < t } ) } { \hat { \pi } ( y _ { i , t } \mid x , y _ { i , < t } ) } A ^ { \hat { \pi } } ( ( x , y _ { i , < t } ) , y _ { i , t } ) \nabla _ { \theta } \log { \pi _ { \theta } ( y _ { i , t } \mid x , y _ { i , < t } ) } .
$$

The surrogate objective $\hat { J }$ is also used in the proximal policy optimization (PPO) algorithm (Schulman et al., 2017). Compared with TRPO, PPO replaces the trust-region constraint by a heuristic that involves clipping the (token-level) importance sampling ratio in $\hat { J }$ to a bounded interval:

$$
\mathbb { E } _ { s \sim d _ { \hat { \pi } } , a \sim \hat { \pi } ( \cdot | s ) } \left[ \operatorname* { m i n } \left( \frac { \pi _ { \theta } ( a \mid s ) } { \hat { \pi } ( a \mid s ) } A ^ { \hat { \pi } } ( s , a ) , \ \mathrm { c l i p } \left( \frac { \pi _ { \theta } ( a \mid s ) } { \hat { \pi } ( a \mid s ) } , 1 - \epsilon , 1 + \epsilon \right) A ^ { \hat { \pi } } ( s , a ) \right) \right]
$$

for some $\epsilon > 0$ . The heuristic aims to discourage the updated policy from leaving the trust region by limiting the amount of improvement made by the updated policy.

## 2.2.2 Token-level GRPO

In language model post-training, the value of $A ^ { \hat { \pi } } ( ( x , y _ { i , < t } ) , y _ { i , t } )$ is often dificult to evaluate or estimate, as it would require learning a value function over prefixes. To avoid evaluating $A ^ { \hat { \pi } }$ , one can use the fact

$$
\nabla \hat { J } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \hat { \pi } ( \cdot \vert x ) } \left[ R ( x , y ) \sum _ { t = 0 } ^ { \vert y \vert - 1 } \frac { \pi _ { \theta } ( y _ { t } \vert x , y _ { < t } ) } { \hat { \pi } ( y _ { t } \vert x , y _ { < t } ) } \nabla _ { \theta } \log { \pi _ { \theta } ( y _ { t } \vert x , y _ { < t } ) } \right]\tag{14}
$$

to use R in place of $A ^ { \hat { \pi } }$ (see, $\mathrm { e . g . }$ , Schulman et al. (2016)). A baseline can be incorporated similarly to what has been done in Section 2.1.2. For any baseline b that is conditionally independent of y given x, it can be shown that

$$
\nabla \hat { J } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \hat { \pi } ( \cdot \vert x ) } \left[ ( R ( x , y ) - b ) \sum _ { t = 0 } ^ { \vert y \vert - 1 } \frac { \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) } { \hat { \pi } ( y _ { t } \mid x , y _ { < t } ) } \nabla _ { \theta } \log { \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) } \right] .
$$

Suppose samples $y _ { 1 } , . . . , y _ { G } \sim \hat { \pi } ( \cdot \mid x )$ are available. Define $b _ { i }$ as in (7) for $i = 1 , \dots , G$ , and $\bar { R }$ as in (8). It can be shown that

$$
\frac { 1 } { G - 1 } \sum _ { i = 1 } ^ { G } \left[ \left( R ( x , y _ { i } ) - \bar { R } \right) \sum _ { t = 0 } ^ { | y _ { i } | - 1 } \frac { \pi _ { \theta } ( y _ { i , t } \mid x , y _ { i , < t } ) } { \bar { \pi } ( y _ { i , t } \mid x , y _ { i , < t } ) } \nabla _ { \theta } \log \pi _ { \theta } ( y _ { i , t } \mid x , y _ { i , < t } ) \right]\tag{15}
$$

is an unbiased estimator of $\nabla \hat { J } ( \theta )$

Remark 3. The estimator in (15) is the token-level counterpart of (10). In practice, a clipping heuristic is sometimes applied to the token-level importance ratio in (15) to further reduce variance at the price of introducing bias, so that the gradient estimator becomes

$$
\frac { 1 } { G - 1 } \sum _ { i = 1 } ^ { G } \left[ \left( R ( x , y _ { i } ) - \bar { R } \right) \sum _ { t = 0 } ^ { | y _ { i } | - 1 } \mathrm { c l i p } \left( \frac { \pi _ { \theta } \big ( y _ { i , t } \mid x , y _ { i , < t } \big ) } { \bar { \pi } \big ( y _ { i , t } \mid x , y _ { i , < t } \big ) } , 1 - \epsilon , 1 + \epsilon \right) \nabla _ { \theta } \log \pi _ { \theta } \big ( y _ { i , t } \mid x , y _ { i , < t } \big ) \right] .
$$

The bias of the estimator will be discussed in Section 3.5.

## 2.3 Comparison between sequence-level and token-level importance sampling

The results in Sections 2.1 and 2.2 show that the two importance sampling formulations yield unbiased gradient estimators for diferent objective functions: the sequence-level estimators target $\nabla J ( \theta )$ , whereas the token-level estimators target $\nabla \hat { J } ( \theta )$ . This diference, together with the diferent numerical behavior of importance sampling ratios, accounts for the trade-ofs discussed below.

The importance sampling ratio accounts for the main drawback of sequence-level importance sampling. Because $\pi _ { \theta } ( y \mid x ) / { \hat { \pi } } ( y \mid x )$ is a product of |y| token-level ratios, even when the pertoken discrepancy in the ratios is small, these discrepancies accumulate along the sequence. For long agentic traces, this accumulation can cause sequence-level ratios to span several orders of magnitude. As a result, computing the sequence-level ratio may lead to numerical instability. More importantly, the variance of the corresponding gradient estimator can become enormous. This is because the sample average may be dominated by a small number of samples with large importance sampling ratios, thus reducing the efective sample size of a batch (Kong, 1992; Liu, 2001).

In comparison, the token-level ratios are better behaved numerically. This is, however, at the cost that the gradient estimator only corresponds to a surrogate $\hat { J }$ at the current inference policy. At step k of policy optimization, the policy improvement guarantee only ensures that the updated training policy $\pi _ { \theta _ { k + 1 } }$ is better than the current inference policy $\hat { \pi } _ { k }$ , i.e., $V _ { 0 } ( \pi _ { \theta _ { k + 1 } } ) > V _ { 0 } ( \hat { \pi } _ { k } )$ . Due to the training–inference mismatch, $\hat { \pi } _ { k }$ generally difers from the current training policy $\pi _ { \boldsymbol { \theta } _ { k } }$ and can be worse, i.e., $V _ { 0 } ( \hat { \pi } _ { k } ) < V _ { 0 } ( \pi _ { \theta _ { k } } )$ . Consequently, it may occur that $V _ { 0 } ( \pi _ { \theta _ { k + 1 } } ) < V _ { 0 } ( \pi _ { \theta _ { k } } )$ , thus defeating the purpose of the policy update. One apparent remedy is to reduce the mismatch between $\hat { \pi } _ { k }$ and $\pi _ { \theta _ { k } }$ and consequently $| V _ { 0 } ( \hat { \pi } _ { k } ) - V _ { 0 } ( \pi _ { \theta _ { k } } ) |$ by choosing how the two engines are implemented and how the parameters are synchronized.

## 3 Policy Gradient as a Covariance

This section shows that both the policy gradient $\nabla J ( \theta )$ and the approximate policy gradient $\nabla \hat { J } ( \theta )$ can be viewed as a covariance. This interpretation of the policy gradient yields two main results. First, when the importance sampling ratio is regularized, e.g., by truncation, masking, or clipping (as in Remark 3), the covariance form recovers invariance to constant shifts in the reward, a key property enjoyed by standard importance sampling. Second, the covariance form admits two distinct unbiased estimators, which correspond to the group-relative methods, including RLOO and Dr. GRPO, in Section 2 and score centering (Marek and Ryabinin, 2026), respectively. Comparing the variances of these two estimators provides a guideline for choosing between the two algorithms for estimating the policy gradient. Throughout this section, the prompt x is fixed. As discussed in Remark 1, the per-prompt results extend to the full objective by drawing x from D.

## 3.1 Unbiased covariance estimation

To prepare for subsequent discussions on sample-average approximation of the policy gradient, this subsection briefly reviews the estimation of a covariance from samples. The results are applied in Sections 3.3 and 3.5.

Let $( X _ { 1 } , Y _ { 1 } ) , \dots , ( X _ { n } , Y _ { n } )$ be independent copies of a pair (X, Y ) with $X \in \mathbb { R }$ and $Y \in \mathbb { R } ^ { d }$ matching the setting in which X is a scalar reward and $Y$ a score. Write $\mu _ { X } \triangleq \mathbb { E } [ X ] , \mu _ { Y } \triangleq \mathbb { E } [ Y ]$ and

$$
\bar { X } \triangleq \frac 1 n \sum _ { i = 1 } ^ { n } X _ { i } , \qquad \bar { Y } \triangleq \frac 1 n \sum _ { i = 1 } ^ { n } Y _ { i } , \qquad \tilde { X } \triangleq X - \mu _ { X } , \qquad \tilde { Y } \triangleq Y - \mu _ { Y } .
$$

The estimand is the covariance vector $\sigma _ { X Y } \triangleq \operatorname { C o v } ( X , Y ) = \mathbb { E } [ \tilde { X } \tilde { Y } ] \in \mathbb { R } ^ { d }$ . Define the moments

$$
\mu _ { r 0 } \triangleq \mathbb { E } \left[ \tilde { X } ^ { r } \right] , \qquad \mu _ { r 1 } \triangleq \mathbb { E } \left[ \tilde { X } ^ { r } \tilde { Y } \right] \in \mathbb { R } ^ { d } , \qquad \mu _ { r 2 } \triangleq \mathbb { E } \left[ \tilde { X } ^ { r } \| \tilde { Y } \| ^ { 2 } \right] .\tag{16}
$$

In particular, we have $\mu _ { 1 1 } = \sigma _ { X Y } , \mu _ { 2 0 } = \operatorname { V a r } ( X )$ , and $\mu _ { 0 2 } = \operatorname { \mathbb { E } } \| { \tilde { Y } } \| ^ { 2 } = \operatorname { t r } \operatorname { C o v } ( Y )$ . For an estimator $\hat { \sigma }$ of $\sigma _ { X Y }$ , we write $\operatorname { V a r } ( \hat { \sigma } ) \triangleq \mathbb { E } \left\| \hat { \sigma } - \mathbb { E } \hat { \sigma } \right\| ^ { 2 }$ , the sum of the variances of individual coordinates, which for an unbiased estimator is its mean squared error.

Definition 4 (Sample-mean estimators). For $n \geq 2$ , define

$$
{ \hat { \sigma } } _ { 1 } \triangleq { \frac { 1 } { n - 1 } } \sum _ { i = 1 } ^ { n } \left( X _ { i } - { \bar { X } } \right) \left( Y _ { i } - { \bar { Y } } \right) ,
$$

$$
{ \hat { \sigma } } _ { 2 } \triangleq { \frac { 1 } { n - 1 } } \sum _ { i = 1 } ^ { n } \left( X _ { i } - { \bar { X } } \right) Y _ { i } ,
$$

$$
{ \hat { \sigma } } _ { 3 } \triangleq { \frac { 1 } { n - 1 } } \sum _ { i = 1 } ^ { n } X _ { i } \left( Y _ { i } - { \bar { Y } } \right) .
$$

The following shows that these three are the same unbiased estimator of $\sigma _ { X Y }$

Proposition 5 (Equivalence among sample-mean estimators). $\hat { \sigma } _ { 1 } = \hat { \sigma } _ { 2 } = \hat { \sigma } _ { 3 }$ for every realization of the sample. More generally, for any constants $\alpha \in \mathbb { R }$ and $\boldsymbol { \beta } \in \mathbb { R } ^ { d }$

$$
{ \frac { 1 } { n - 1 } } \sum _ { i = 1 } ^ { n } \left( X _ { i } - { \bar { X } } \right) \left( Y _ { i } - \beta \right) = { \frac { 1 } { n - 1 } } \sum _ { i = 1 } ^ { n } \left( X _ { i } - \alpha \right) \left( Y _ { i } - { \bar { Y } } \right) = { \hat { \sigma } } _ { 1 } ,\tag{17}
$$

and $\mathbb { E } \hat { \sigma } _ { 1 } = \sigma _ { X Y }$

Proof. For the first equality in (17),

$$
\sum _ { i } \left( X _ { i } - { \bar { X } } \right) \left( Y _ { i } - \beta \right) = \sum _ { i } \left( X _ { i } - { \bar { X } } \right) \left( Y _ { i } - { \bar { Y } } \right) + \left( \sum _ { i } \left( X _ { i } - { \bar { X } } \right) \right) \left( { \bar { Y } } - \beta \right) ,
$$

and $\begin{array} { r } { \sum _ { i } ( X _ { i } - \bar { X } ) = 0 } \end{array}$ . The second equality is symmetric, using $\begin{array} { r } { \sum _ { i } ( Y _ { i } - \bar { Y } ) = 0 } \end{array}$ . Setting $\beta = 0$ and $\alpha = 0$ recovers ${ \hat { \sigma } } _ { 2 }$ and ${ \hat { \sigma } } _ { 3 }$ , respectively. For unbiasedness, expand $\begin{array} { r } { \sum _ { i } ( X _ { i } - \bar { X } ) ( Y _ { i } - \bar { Y } ) = \sum _ { i } X _ { i } Y _ { i } - } \end{array}$ $n \bar { X } \bar { Y }$ and use $\mathbb { E } [ X _ { i } Y _ { i } ] = \sigma _ { X Y } + \mu _ { X } \mu _ { Y }$ together with $\mathbb { E } [ \bar { X } \bar { Y } ] = \operatorname { C o v } ( \bar { X } , \bar { Y } ) + \mu _ { X } \mu _ { Y } = \sigma _ { X Y } / n + \mu _ { X } \mu _ { Y }$ The expectation of the sum is thus $( n - 1 ) \sigma _ { X Y }$ □

Proposition 5 reveals the following two facts. First, it sufices to center either X or $Y ,$ and performing both has no further efect. Second, knowledge of the population mean $\mu _ { Y }$ is of no use when X<sup>¯</sup> is used for centering because setting $\beta = \mu _ { Y }$ in (17) leads to the same estimator.

In comparison, when X<sup>¯</sup> is not used, replacing Y<sup>¯</sup> in ${ \hat { \sigma } } _ { 3 }$ by $\mu _ { Y }$ yields a diferent estimator as defined below.

Definition 6 (Known-mean estimator). Suppose $\mu _ { Y }$ is known. For $n \geq 1$ , define

$$
{ \hat { \sigma } } _ { \mu _ { Y } } \triangleq { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } X _ { i } \left( Y _ { i } - \mu _ { Y } \right) .
$$

The sample-mean estimator and the known-mean estimator difer in how many samples they require. The known-mean estimator is defined at $n = 1$ , whereas the sample-mean estimator requires $n > 1$ . Similarly to what is done in (17), we consider a generalization of $\hat { \sigma } _ { \mu _ { Y } }$ by incorporating a centering constant b for X. The constant b is referred to as a baseline since it plays the role of a baseline reward in policy gradient.

Definition 7 (Known-mean estimator with a baseline). For $n \geq 1$ and any constant $b \in \mathbb { R }$ , define

$$
{ \hat { \sigma } } _ { \mu _ { Y } } \left( b \right) \triangleq { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \left( X _ { i } - b \right) \left( Y _ { i } - \mu _ { Y } \right) .\tag{18}
$$

It follows immediately that $\hat { \sigma } _ { \mu _ { Y } } = \hat { \sigma } _ { \mu _ { Y } } ( 0 )$ . The following shows that this estimator is unbiased for any b.

Proposition 8. E $\hat { \sigma } _ { \mu _ { Y } } ( b ) = \sigma _ { X Y }$ for every $n \geq 1$ and every constant b.

Proof. Writing $X _ { i } - b = \tilde { X } _ { i } + ( \mu _ { X } - b )$ and using $\mathbb { E } [ \tilde { Y } _ { i } ] = 0$

$$
\begin{array} { r } { \mathbb { E } \left[ \left( X _ { i } - b \right) \tilde { Y } _ { i } \right] = \mathbb { E } \left[ \tilde { X } _ { i } \tilde { Y } _ { i } \right] + \left( \mu _ { X } - b \right) \mathbb { E } \left[ \tilde { Y } _ { i } \right] = \sigma _ { X Y } . } \end{array}
$$

Although both $\hat { \sigma } _ { 1 }$ and ${ \hat { \sigma } } _ { \mu \gamma } ( b )$ are unbiased, their variances are diferent.

Proposition 9 (Variances of the estimators). Define the residual mean m $\triangleq \mu _ { X } - b$ and

$$
q ( m ) \triangleq m ^ { 2 } \mu _ { 0 2 } + 2 m \mu _ { 1 2 } .
$$

(i) For every $n \geq 1$

$$
\mathrm { V a r } ( \hat { \sigma } _ { \mu _ { Y } } ( b ) ) = \frac { \mu _ { 2 2 } - \| \mu _ { 1 1 } \| ^ { 2 } + q ( m ) } { n } .\tag{19}
$$

(ii) For every $n \geq 2$

$$
\mathrm { V a r } ( \hat { \sigma } _ { 1 } ) = \frac { \mu _ { 2 2 } - \| \mu _ { 1 1 } \| ^ { 2 } } { n } + \frac { \| \mu _ { 1 1 } \| ^ { 2 } + \mu _ { 2 0 } \mu _ { 0 2 } } { n ( n - 1 ) } .\tag{20}
$$

Proof. For (19), the summands of (18) are independent and identically distributed. This implies

$$
\mathrm { V a r } ( \hat { \sigma } _ { \mu _ { Y } } ( b ) ) = \frac { 1 } { n } \big ( \mathbb { E } \| ( X - b ) \tilde { Y } \| ^ { 2 } - \| \sigma _ { X Y } \| ^ { 2 } \big ) = \frac { 1 } { n } \big ( \mathbb { E } ( X - b ) ^ { 2 } \| \tilde { Y } \| ^ { 2 } - \| \sigma _ { X Y } \| ^ { 2 } \big ) .
$$

Expanding $( X - b ) ^ { 2 } = \tilde { X } ^ { 2 } + 2 m \tilde { X } + m ^ { 2 }$ and using (16), we obtain $\mathbb { E } [ ( X - b ) ^ { 2 } \| \tilde { Y } \| ^ { 2 } ] = \mu _ { 2 2 } +$ $2 m \mu _ { 1 2 } + m ^ { 2 } \mu _ { 0 2 }$ . For (20), notice that Var(·) is a sum over coordinates, and the kth entry of $\hat { \sigma } _ { 1 }$ is the ordinary scalar sample covariance of $( X _ { i } , Y _ { i k } )$ , whose exact variance is $\big ( \mu _ { 2 2 } ^ { ( k ) } - \sigma _ { X Y _ { k } } ^ { 2 } \big ) / n +$ $\big ( \sigma _ { X Y _ { k } } ^ { 2 } + \mu _ { 2 0 } \sigma _ { Y _ { k } } ^ { 2 } \big ) / ( n ( n - 1 ) )$ ) with $\mu _ { 2 2 } ^ { ( k ) } \triangleq \mathbb { E } [ \tilde { X } ^ { 2 } \tilde { Y } _ { k } ^ { 2 } ]$ ; summing over k gives (20). □

The baseline that minimizes $\operatorname { V a r } ( \hat { \sigma } _ { \mu _ { Y } } ( b ) )$ can be obtained by minimizing q in (19). The minimum of $q$ is attained at

$$
m ^ { \star } \triangleq - \frac { \mu _ { 1 2 } } { \mu _ { 0 2 } }
$$

with $q ( m ^ { \star } ) = - \mu _ { 1 2 } ^ { 2 } / \mu _ { 0 2 } \leq 0$ . Thus, the minimum-variance baseline is given by

$$
b ^ { \star } = \mu _ { X } - m ^ { \star } = \mu _ { X } + \frac { \mu _ { 1 2 } } { \mu _ { 0 2 } } = \frac { \operatorname { \mathbb { E } } \left[ X \left\| Y - \mu _ { Y } \right\| ^ { 2 } \right] } { \operatorname { \mathbb { E } } \left\| Y - \mu _ { Y } \right\| ^ { 2 } } .\tag{21}
$$

Proposition 9 shows that the choice of b determines which of the two unbiased estimators has the smaller variance. When the baseline is set to the population mean of $X , { \mathrm { i . e . , } } b = \mu _ { X }$ or equivalently $m = 0$ , the variance of the known-mean estimator is smaller by $\left( \left\| \mu _ { 1 1 } \right\| ^ { 2 } + \mu _ { 2 0 } \mu _ { 0 2 } \right) / \left( n ( n - 1 ) \right)$ , which is of order $O ( n ^ { - 2 } )$ . When the minimum-variance baseline $b ^ { \star }$ is used, the variance is further reduced by $\mu _ { 1 2 } ^ { 2 } / ( n \mu _ { 0 2 } )$ , which is of order $O ( n ^ { - 1 } )$ . More generally, ${ \hat { \sigma } } _ { \mu _ { Y } } ( b )$ has strictly smaller variance than $\hat { \sigma } _ { 1 }$ if and only if

$$
| m - m ^ { \star } | < \rho _ { n } \triangleq \sqrt { \frac { \| \mu _ { 1 1 } \| ^ { 2 } + \mu _ { 2 0 } \mu _ { 0 2 } } { ( n - 1 ) \mu _ { 0 2 } } + \frac { \mu _ { 1 2 } ^ { 2 } } { \mu _ { 0 2 } ^ { 2 } } } .
$$

Conversely, when $m$ is far from $m ^ { \star }$ , the variance of ${ \hat { \sigma } } _ { \mu _ { Y } } ( b )$ can be much larger than that of $\hat { \sigma } _ { 1 }$ , and the gap grows without bound as $| m |$ grows.

## 3.2 The sequence-level estimand

Denote by

$$
s _ { \theta } ( y \mid x ) \triangleq \nabla _ { \theta } \log \pi _ { \theta } ( y \mid x )
$$

the score. Recall that $\begin{array} { r } { \mathbb { E } _ { \boldsymbol { y } \sim \pi _ { \boldsymbol { \theta } } } [ s _ { \boldsymbol { \theta } } ( \boldsymbol { y } \mid \boldsymbol { x } ) ] = \sum _ { \boldsymbol { y } } \nabla _ { \boldsymbol { \theta } } \pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } \mid \boldsymbol { x } ) = \nabla _ { \boldsymbol { \theta } } 1 = 0 } \end{array}$ . This implies that the policy gradient $\nabla J ( \theta )$ in (3) can be viewed as a covariance:

$$
\nabla J ( \theta ) = \mathbb { E } _ { y \sim \pi _ { \theta } } \left[ R ( x , y ) s _ { \theta } ( y \mid x ) \right] = \operatorname { C o v } _ { \pi _ { \theta } } ( R ( x , y ) , s _ { \theta } ( y \mid x ) ) .
$$

Denote the (sequence-level) importance sampling ratio by

$$
w ( y \mid x ) \triangleq { \frac { \pi _ { \theta } ( y \mid x ) } { \hat { \pi } ( y \mid x ) } } .
$$

As discussed in Section 2.3, w is a product of $| y |$ token-level ratios, and the resulting sample average can be dominated by a few rollouts. The standard method is to regularize the importance sampling ratio by replacing w with $f ( w )$ for some $f \colon \mathbb { R } _ { + } \to \mathbb { R } _ { + }$ satisfying $f ( 1 ) = 1$ . Common choices are

$$
\begin{array} { l l } { f ( w ) = w } & { ( \mathrm { s t a n d a r d ~ i m p o r t a n c e ~ s a m p l i n g } ) , } \\ { f ( w ) = \operatorname* { m i n } ( w , C ) } & { ( \mathrm { t r u n c a t e d ~ i m p o r t a n c e ~ s a m p l i n g } ) , } \\ { f ( w ) = w \mathbf { 1 } [ w \in [ l , u ] ] } & { ( \mathrm { m a s k e d ~ i m p o r t a n c e ~ s a m p l i n g } ) , } \\ { f ( w ) = \mathrm { c l i p } ( w , 1 - \epsilon , 1 + \epsilon ) } & { ( \mathrm { r a t i o ~ c l i p p i n g } ) . } \end{array}
$$

The property that $f ( 1 ) = 1$ ensures that no regularization is imposed when the two policies agree. It is convenient to absorb the weight into the score by defining the weighted score

$$
{ \tilde { s } } ( y \mid x ) \triangleq f { \bigl ( } w ( y \mid x ) { \bigr ) } s _ { \theta } ( y \mid x ) .
$$

When importance sampling is used with regularization, the quantity in (4) can be written as $\mathbb { E } _ { \hat { \pi } } [ R \tilde { s } ]$ We take the covariance

$$
\sigma ^ { \mathrm { s e q } } \triangleq \operatorname { C o v } _ { \hat { \pi } } ( R ( x , y ) , \ : \tilde { s } ( y \mid x ) ) = \mathbb { E } _ { \hat { \pi } } \left[ R \ : \tilde { s } \right] - \mathbb { E } _ { \hat { \pi } } [ R ] \cdot \mathbb { E } _ { \hat { \pi } } \left[ \tilde { s } \right] ,\tag{22}
$$

as the sequence-level estimand instead of $\mathbb { E } _ { \hat { \pi } } [ R \tilde { s } ]$ for the following reasons. First, (22) reduces to $\nabla J ( \theta )$ if either $f$ is the identity mapping id (i.e., when standard importance sampling is used) or ${ \hat { \pi } } = \pi _ { \theta } \colon$ In the former case,

$$
\mathbb { E } _ { \hat { \pi } } \left[ \tilde { s } \right] = \sum _ { y } \hat { \pi } ( y \mid x ) \frac { \pi _ { \theta } ( y \mid x ) } { \hat { \pi } ( y \mid x ) } s _ { \theta } ( y \mid x ) = \mathbb { E } _ { \pi _ { \theta } } \left[ s _ { \theta } \right] = 0 .
$$

In the latter case, $w \equiv 1$ , and thus $\tilde { s } = f ( 1 ) s _ { \theta } = s _ { \theta }$ . Second, $\sigma ^ { \mathrm { s e q } }$ is unchanged if the reward is shifted by a constant, $R \mapsto R + c ,$ , whereas $\mathbb { E } _ { \hat { \pi } } [ R \tilde { s } ]$ is not.

## 3.3 Sequence-level covariance estimation

Since the sequence-level estimand $\sigma ^ { \mathrm { s e q } }$ in (22) is a covariance, the results of Section 3.1 apply directly. Fix a prompt x, draw $y _ { 1 } , \dots , y _ { G } \sim { \hat { \pi } } ( \cdot \mid x )$ independently, and set $X _ { i } = R ( x , y _ { i } )$ and $Y _ { i } = \tilde { s } ( y _ { i } \mid x )$ with $n = G$ . The following proposition shows that the sample-mean estimator of $\sigma ^ { \mathrm { s e q } }$ yields the RLOO algorithm and, up to a multiplicative factor, the Dr. GRPO algorithm.

Proposition 10 (RLOO and Dr. GRPO are the sample-mean estimator). Let $b _ { i }$ be the leave-one-out baseline of (7). Then

$$
{ \frac { 1 } { G } } \sum _ { i = 1 } ^ { G } \left( X _ { i } - b _ { i } \right) Y _ { i } = { \hat { \sigma } } _ { 1 } , \qquad { \frac { 1 } { G } } \sum _ { i = 1 } ^ { G } \left( X _ { i } - { \bar { R } } \right) Y _ { i } = { \frac { G - 1 } { G } } { \hat { \sigma } } _ { 1 } ,\tag{23}
$$

where R<sup>¯</sup> is the group mean (8).

Proof. By (9), $\begin{array} { r } { X _ { i } - b _ { i } = \frac { G } { G - 1 } ( X _ { i } - \bar { R } ) } \end{array}$ for each i. Thus the first sum in (23) becomes $\begin{array} { r } { \frac { 1 } { G - 1 } \sum _ { i } ( X _ { i } - } \end{array}$ $\bar { R } ) Y _ { i } = \hat { \sigma } _ { 2 }$ , which is $\hat { \sigma } _ { 1 }$ by Proposition 5. The second identity is immediate from the same expression. □

With standard importance sampling, i.e., $f = \mathrm { i d }$ , we have $\mu _ { Y } \triangleq \mathbb { E } _ { \hat { \pi } } [ \tilde { s } ] = 0 \mathrm { : }$ ; hence score centering is implicit. When $f \neq \operatorname { i d }$ , we have $\mu _ { Y } \neq 0$ in general, and one may ask whether score centering provides any benefit. Exact computation of $\mu _ { Y }$ is intractable since it involves a sum over all possible responses. More importantly, the following proposition shows that score centering with $\mu _ { Y }$ is unnecessary when the sample-mean reward is used as the baseline, even when $f \neq \operatorname { i d }$

Proposition 11 (Sequence-level centering with the group-mean baseline). With the group-mean baseline,

$$
{ \frac { 1 } { G - 1 } } \sum _ { i = 1 } ^ { G } \left( X _ { i } - { \bar { R } } \right) \left( Y _ { i } - \mu _ { Y } \right) = { \frac { 1 } { G - 1 } } \sum _ { i = 1 } ^ { G } \left( X _ { i } - { \bar { R } } \right) Y _ { i } = { \hat { \sigma } } _ { 1 } .\tag{24}
$$

Proof. The result immediately follows from Proposition 5 by setting $\beta = \mu _ { Y }$

Since both RLOO and Dr. GRPO can be viewed as the sample-mean estimator $\hat { \sigma } _ { 1 }$ up to a multiplicative factor, Proposition 11 implies that score centering is unnecessary for both algorithms.

Comparison with the variance-optimal baseline Suppose that standard importance sampling is used. Then $\mu _ { Y } = 0$ . From (21), the optimal baseline is given by $b ^ { \star } = \mathbb { E } [ X \| Y \| ^ { \bar { 2 } } ] / \mathbb { E } \| Y \| ^ { 2 }$ . This is the variance-optimal reward baseline of Greensmith et al. (2004, Theorem 11). The mean reward, which is the baseline used by the group-relative methods, is optimal only when $\mu _ { 1 2 } = 0$ , as also shown by Greensmith et al. (2004, Corollary 12). In language model post-training, $\mu _ { 1 2 } = \operatorname { C o v } ( R , \| s _ { \theta } \| ^ { 2 } )$ is not negligible, since $\| s _ { \theta } \| ^ { 2 }$ grows with response length, and length is empirically correlated with reward. This indicates that using the mean reward as the baseline may be far from optimal.

## 3.4 The token-level estimand

At the token level, the approximate policy gradient $\nabla \hat { J } ( \theta )$ is computed, and the importance sampling ratio is applied per token position as in (13) and (14). Write $h _ { t } \triangleq ( x , y _ { < t } )$ for the prefix,

$$
w ( y _ { t } \mid h _ { t } ) \triangleq \frac { \pi _ { \theta } ( y _ { t } \mid h _ { t } ) } { \hat { \pi } ( y _ { t } \mid h _ { t } ) } , \qquad \tilde { s } _ { t } \triangleq f ( w ( y _ { t } \mid h _ { t } ) ) \nabla _ { \theta } \log \pi _ { \theta } ( y _ { t } \mid h _ { t } ) ,
$$

and let

$$
\bar { s } ( h _ { t } ) \triangleq \mathbb { E } _ { \hat { \pi } } \left[ \tilde { s } _ { t } \mid y _ { < t } \right] = \sum _ { v \in \mathcal { V } } \hat { \pi } ( v \mid h _ { t } ) f ( w ( v \mid h _ { t } ) ) \nabla _ { \theta } \log \pi _ { \theta } ( v \mid h _ { t } )\tag{25}
$$

be the mean weighted score at a given prefix $h _ { t }$ , where V denotes the token vocabulary, i.e., the action space $\mathcal { A }$ in Section 1.2. Because the expectation in (25) is a finite sum over $\nu ,$ computing $\bar { s } ( h )$ is much more tractable than computing the mean score $\mu _ { Y }$ at the sequence level in Section 3.3.

As in the sequence-level case, $\bar { s } ( h ) = 0$ for any h when $f = \mathrm { i d }$ . We assume that the sampling policy terminates in finite expected time and that the weighted score has a bounded conditional mean:

$$
\mathbb { E } _ { \hat { \pi } } \left[ | y | \right] < \infty , \qquad \operatorname* { s u p } _ { h } \mathbb { E } _ { v \sim \hat { \pi } ( \cdot | h ) } \| f ( w ( v \mid h ) ) \nabla _ { \theta } \log \pi _ { \theta } ( v \mid h ) \| < \infty .\tag{26}
$$

To incorporate regularized importance sampling, we generalize $\nabla \hat { J } ( \theta )$ in (13) and define the token level estimand

$$
\sigma ^ { \mathrm { t o k } } \triangleq \mathbb { E } _ { \hat { \pi } } \left[ \sum _ { t = 0 } ^ { | y | - 1 } f ( w ( y _ { t } \mid h _ { t } ) ) A ^ { \hat { \pi } } ( h _ { t } , y _ { t } ) \nabla _ { \theta } \log \pi _ { \theta } ( y _ { t } \mid h _ { t } ) \right] .
$$

Write $Q _ { t } \triangleq \mathbb { E } _ { \hat { \pi } } [ R \mid y _ { \leq t } ]$ for the state-action value and $V ( h _ { t } ) \triangleq \mathbb { E } _ { \hat { \pi } } [ R \mid y _ { < t } ]$ for the state value under πˆ. The following lemma shows that $\sigma ^ { \mathrm { t o k } }$ can be expressed as a sum of conditional covariances.

Lemma 12. The token-level estimand

$$
\sigma ^ { \mathrm { t o k } } = \mathbb { E } _ { \hat { \pi } } \left[ \sum _ { { t = 0 } } ^ { \left| { y } \right| - 1 } \mathrm { C o v } _ { \hat { \pi } } \left( \tilde { s } _ { t } , Q _ { t } \mid y _ { < t } \right) \right] .\tag{27}
$$

Proof. For brevity, we assume that all the expectations and covariances are taken under πˆ and drop πˆ from all the subscripts. Using the facts $\mathbb { E } [ Q _ { t } \mid y _ { < t } ] = V ( h _ { t } )$ and $A ^ { \hat { \pi } } ( h _ { t } , y _ { t } ) = Q _ { t } - V ( h _ { t } )$ , it can be seen that

$$
\begin{array} { r l } & { \mathrm { C o v } \big ( \tilde { s } _ { t } , Q _ { t } \ \big | \ y _ { < t } \big ) = \mathbb { E } \left[ \tilde { s } _ { t } ( Q _ { t } - \mathbb { E } [ Q _ { t } \ | \ y _ { < t } ] ) \ \big | \ y _ { < t } \right] } \\ & { \qquad = \mathbb { E } \left[ \tilde { s } _ { t } ( Q _ { t } - V ( h _ { t } ) ) \ \big | \ y _ { < t } \right] = \mathbb { E } \left[ \tilde { s } _ { t } A ^ { \hat { \pi } } ( h _ { t } , y _ { t } ) \ \big | \ y _ { < t } \right] . } \end{array}\tag{28}
$$

Thus,

$$
\begin{array} { r l } {  { \mathbb { E } [  \sum _ { t = 0 } ^ { \infty } \mathrm { C o v } ( \tilde { s } _ { t } , Q _ { t } \mid y _ { < t } ) ] = \mathbb { E } [ \sum _ { t = 0 } ^ { \infty } \mathbf { 1 } \big [ t <  y  \big ] \mathbb { E } [ \tilde { s } _ { t } A ^ { \hat { \tau } } ( h _ { t } , y _ { t } ) \big \vert y _ { < t } ] ] } \ } \\ & { = \displaystyle \sum _ { t = 0 } ^ { \infty } \mathbb { E } [ \mathbf { 1 } [ t <  y  ] \mathbb { E } [ \tilde { s } _ { t } A ^ { \hat { \tau } } ( h _ { t } , y _ { t } ) \big \vert y _ { < t } ] ] = \sum _ { t = 0 } ^ { \infty } \mathbb { E } [ \mathbb { E } [ \mathbf { 1 } [ t <  y  ] \tilde { s } _ { t } A ^ { \hat { \tau } } ( h _ { t } , y _ { t } ) \big \vert y _ { < t } ] ] } \\ & { = \displaystyle \sum _ { t = 0 } ^ { \infty } \mathbb { E } [ \mathbf { 1 } [ t <  y  ] \tilde { s } _ { t } A ^ { \hat { \tau } } ( h _ { t } , y _ { t } ) ] = \mathbb { E } [ \sum _ { t = 0 } ^ {  y  - 1 } \tilde { s } _ { t } A ^ { \hat { \tau } } ( h _ { t } , y _ { t } ) ] = \sigma ^ { \mathrm { t o k } } . } \end{array}
$$

The interchange of the summation and the expectation is permitted under the conditions in (26). The third equality holds because the event $\{ | y | > t \}$ is a function of $y _ { < t }$ , as it is determined by whether EOS occurs among $y _ { 0 } , \ldots , y _ { t - 1 }$ □

The form in (27) does not depend on the score being centered since a covariance only requires one of its arguments to be centered, as achieved by $V ( h _ { t } ) = \mathbb { E } _ { \hat { \pi } } [ Q _ { t } \mid y _ { < t } ]$ in (28). Alternatively, one may center the score $\tilde { s } _ { t }$ by expressing

$$
\operatorname { C o v } _ { \hat { \pi } } \left( \tilde { s } _ { t } , Q _ { t } \mid y _ { < t } \right) = \operatorname { \mathbb { E } } _ { \hat { \pi } } \left[ \left( \tilde { s } _ { t } - \operatorname { \mathbb { E } } _ { \hat { \pi } } \left[ \tilde { s } _ { t } \mid y _ { < t } \right] \right) Q _ { t } \mid y _ { < t } \right] = \operatorname { \mathbb { E } } _ { \hat { \pi } } \left[ \left( \tilde { s } _ { t } - \bar { s } ( h _ { t } ) \right) Q _ { t } \mid y _ { < t } \right] .\tag{29}
$$

As will be shown in Proposition 14, score centering leads to an equivalent and more useful form. Define

$$
D \triangleq \sum _ { t = 0 } ^ { | y | - 1 } \left( \tilde { s } _ { t } - \bar { s } ( h _ { t } ) \right) .
$$

Lemma 13. $\mathbb { E } _ { \hat { \pi } } [ D ] = 0$

Proof. Let $M _ { t } \triangleq \mathbf { 1 } \big [ t < | y | \big ] \big ( \tilde { s } _ { t } - \bar { s } ( h _ { t } ) \big )$ . Both the prefix $h _ { t } = ( x , y _ { < t } )$ and the event $\{ | y | > t \}$ are functions of $y _ { < t }$ . By (25),

$$
\begin{array} { r } { \mathbb { E } \left[ M _ { t } \ \middle \vert \ y _ { < t } \right] = \mathbf { 1 } \Big [ t < \left. y \right. \Big ] \mathbb { E } \left[ \tilde { s } _ { t } - \bar { s } ( h _ { t } ) \ \middle \vert \ y _ { < t } \right] = \mathbf { 1 } \Big [ t < \left. y \right. \Big ] ( \mathbb { E } \left[ \tilde { s } _ { t } \ \middle \vert \ y _ { < t } \right] - \bar { s } ( h _ { t } ) ) = 0 . } \end{array}
$$

This shows that $\{ M _ { t } \}$ is a martingale diference sequence. Therefore, the partial sums of $\{ M _ { t } \}$ form a martingale, and $| y |$ is a stopping time. The conditions in (26) bound $\mathbb { E } [ \left| | \boldsymbol { M } _ { t } \right| | \ \boldsymbol { y } _ { < t } ]$ uniformly, so optional stopping applies and $\mathbb { E } [ D ] = 0$ □

The following proposition shows that $\sigma ^ { \mathrm { t o k } }$ can be expressed as a single covariance.

Proposition 14. The token-level estimand $\sigma ^ { \mathrm { t o k } } = \mathrm { C o v } _ { \hat { \pi } } ( R , D )$

Proof. Expanding the conditional covariance in (27) with score centering in (29) yields

$$
\begin{array} { l } { { \mathbb { E } [ \displaystyle \sum _ { t = 0 } ^ { \lfloor y \rfloor - 1 } \mathbb { E } [ ( \tilde { s } _ { t } - \tilde { s } ( h _ { t } ) ) Q _ { t } \mid y _ { < t } ] ] = \mathbb { E } [ \displaystyle \sum _ { t = 0 } ^ { \infty } \mathbf { 1 } [ l < | y | ] \mathbb { E } [ ( \tilde { s } _ { t } - \tilde { s } ( h _ { t } ) ) Q _ { t } \mid y _ { < t } ] ] } } \\ { { = \displaystyle \sum _ { t = 0 } ^ { \infty } \mathbb { E } [ \mathbf { 1 } [ t < | y | ] \mathbb { E } [ ( \tilde { s } _ { t } - \tilde { s } ( h _ { t } ) ) Q _ { t } \mid y _ { < t } ] ] = \displaystyle \sum _ { t = 0 } ^ { \infty } \mathbb { E } [ \mathbf { 1 } [ t < | y | ] \mathbb { E } [ ( \tilde { s } _ { t } - \tilde { s } ( h _ { t } ) ) \mathbb { E } [ R \mid y _ { \leq t } ] \mid y _ { < t } ] ] } } \\ { { = \displaystyle \sum _ { t = 0 } ^ { \infty } \mathbb { E } [ \mathbb { E } [ \mathbb { E } [ \mathbf { 1 } [ t < | y | ] ( \tilde { s } _ { t } - \tilde { s } ( h _ { t } ) ) R \mid y _ { \leq t } ] \mid y _ { < t } ] ] = \displaystyle \sum _ { t = 0 } ^ { \infty } \mathbb { E } [ \mathbf { 1 } [ t < | y | ] ( \tilde { s } _ { t } - \tilde { s } ( h _ { t } ) ) R ] } } \\ { { = \mathbb { E } [ \displaystyle \sum _ { t = 0 } ^ { \infty } \mathbf { 1 } [ t < | y | ] ( \tilde { s } _ { t } - \tilde { s } ( h _ { t } ) ) R ] = \mathbb { E } [ R D ] . } } \end{array}
$$

The interchange of the summation and the expectation is permitted under the conditions in (26). The fourth equality uses the fact that both $\tilde { s } _ { t } - \bar { s } ( h _ { t } )$ and the event $\{ | y | > t \}$ are functions of $y { \le } t$ □

## 3.5 Token-level covariance estimation

Since the token-level estimand in Proposition 14 is a single covariance, the results of Section 3.1 immediately apply with $X = R , Y = D$ , and $n = G$ . In addition, Lemma 13 implies $\mu _ { Y } = \operatorname { \mathbb { E } } [ D ] = 0$ Suppose that $y _ { 1 } , . . . , y _ { G } \sim { \hat { \pi } } ( \cdot \mid x )$ are drawn independently. The corresponding pairs $( R _ { i } , D _ { i } )$ $i = 1 , \ldots , G ,$ are independent and identically distributed. The estimator we mainly consider is the known-mean estimator in (18) with $\mu _ { Y } = 0$ , namely

$$
\hat { \sigma } _ { \mu _ { Y } } ( b ) = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } ( R _ { i } - b ) D _ { i } ,
$$

which is unbiased for every constant b and every $G \geq 1$ by Proposition $8 .$

In comparison, the sample-mean estimator analogous to ${ \hat { \sigma } } _ { 2 }$ in Definition 4 is given by

$$
\frac { 1 } { G - 1 } \sum _ { i = 1 } ^ { G } ( R _ { i } - \bar { R } ) D _ { i } .
$$

It also requires score centering because it involves $D _ { i }$ . Recall that, by Proposition 11, score centering for the sample-mean estimator in (23) is optional at the sequence level regardless of $f ,$ as long as

the group-mean baseline $b = \bar { R }$ is used. In contrast, the following proposition shows that score centering is no longer optional at the token level even when $b = { \bar { R } } ;$ removing score centering would generally yield a diferent estimator.

Proposition 15 (Token-level centering with the group-mean baseline). With the group-mean baseline, the uncentered estimator

$$
\frac { 1 } { G - 1 } \sum _ { i = 1 } ^ { G } ( R _ { i } - \bar { R } ) \sum _ { t = 0 } ^ { | y _ { i } | - 1 } \tilde { s } _ { i , t } ,\tag{30}
$$

obtained by applying f to (15), is unbiased for

$$
\sigma ^ { \mathrm { t o k } } + \mathrm { C o v } _ { \hat { \pi } } \Big ( R , \sum _ { t = 0 } ^ { | y | - 1 } \bar { s } ( h _ { t } ) \Big ) .
$$

Proof. Split the inner sum as $\begin{array} { r } { \sum _ { t } \tilde { s } _ { i , t } = D _ { i } + \sum _ { t } \bar { s } ( h _ { i , t } ) } \end{array}$ and rewrite (30) as

$$
\frac { 1 } { G - 1 } \sum _ { i = 1 } ^ { G } \bigl ( R _ { i } - \bar { R } \bigr ) D _ { i } + \frac { 1 } { G - 1 } \sum _ { i = 1 } ^ { G } \bigl ( R _ { i } - \bar { R } \bigr ) \sum _ { t } \bar { s } ( h _ { i , t } ) .
$$

Each of the two terms is an ordinary sample covariance with one-sided centering on $R .$ These are unbiased for $\mathrm { C o v } ( R , D ) = \sigma ^ { \mathrm { t o k } }$ and for Cov $( R , \textstyle \sum _ { t } { \bar { s } } ( h _ { t } ) )$ , respectively. □

Remark 16. The discrepancy vanishes when $f = \mathrm { i d }$ since $\bar { s } ( h ) = 0$ for any $h ,$ and is otherwise nonzero in general. The sequence-level cancellation fails here because the token-level mean score $\textstyle \sum _ { t } { \bar { s } } ( h _ { i , t } )$ varies across rollouts, whereas the sequence-level mean score $\mu _ { Y }$ in (24) is a single constant and can be annihilated by $\textstyle \sum _ { i } ( R _ { i } - { \bar { R } } )$

The optimal baseline can be obtained as

$$
b ^ { \star } = \frac { \mathbb { E } \big [ R \| D \| ^ { 2 } \big ] } { \mathbb { E } \| D \| ^ { 2 } }\tag{31}
$$

following (21). Recall $b ^ { \star } - \mu _ { X } = - m ^ { \star } = \mu _ { 1 2 } / \mu _ { 0 2 }$ . Since $\mu _ { 0 2 } > 0$ , the sign of $\mu _ { 1 2 }$ determines how $b ^ { \star }$ compares with the mean reward $\mu _ { X }$ . In the token-level setting, we have $\mu _ { 1 2 } = \mathbb { E } [ ( R - \mathbb { E } [ R ] ) \| D \| ^ { 2 } ]$ Since $\| D \| ^ { 2 }$ grows with response length, and a longer response empirically leads to higher reward, we expect $\mu _ { 1 2 } > 0$ to often hold in practice. This implies that it is preferable to set the baseline above the mean reward.

One should keep in mind that the expectations in (31) are taken over rollouts from a single prompt. Thus, $b ^ { \star }$ is prompt-dependent, and no constant baseline remains optimal across multiple prompts in general. Estimating $b ^ { \star }$ per prompt instead would still require multiple rollouts for the current prompt.

## 4 Summary

We have examined the roles of baselines and score centering in policy gradient estimation when responses are generated of-policy and the importance sampling ratio is regularized. Without regularization, sequence-level and token-level importance sampling yield unbiased gradient estimators for diferent objectives: the former for the true objective J and the latter for the surrogate ${ \hat { J } } .$ At either level, regularizing the ratio breaks the invariance of the importance-weighted gradient to constant shifts in the reward. Choosing the covariance between the reward and the weighted score as the estimand recovers this invariance, and the covariance reduces to the policy gradient when the ratio is not regularized. At the sequence level, the estimand is a single covariance by definition; at the token level, it is an expected sum of conditional covariances but can also be written as a single covariance (Proposition 14). Hence, the two classes of covariance estimators in Section 3.1 apply at both levels.

The covariance perspective clarifies the roles of baselines and score centering. Both perform centering in covariance estimation, but they difer in the choice of the mean used for centering. The group-mean baseline uses the sample mean and yields the sample-mean estimator, which coincides with RLOO and, up to a multiplicative factor, with Dr. GRPO (Proposition 10). With a sample mean, centering the reward, centering the score, and centering both are all equivalent (Proposition 5). Score centering uses the population mean instead and yields the known-mean estimator, which is unbiased for any constant baseline (Proposition 8) and is defined even for a single sample. The two estimators have diferent variances (Proposition 9). The known-mean estimator has strictly smaller variance if and only if its baseline is suficiently close to the variance-optimal baseline $b ^ { \star }$ and it can become arbitrarily worse as the baseline moves away from $b ^ { \star }$

Whether both forms of centering should be used simultaneously depends on whether importance sampling is applied at the sequence level or at the token level. At the sequence level, score centering is redundant once the group-mean baseline is used, regardless of whether the ratio is regularized (Proposition 11). Consequently, the population mean of the sequence-level weighted score, which is intractable to compute, is not needed. At the token level, while score centering is far more computationally tractable, it is no longer optional. Without score centering, the group-mean estimator is biased for the token-level estimand; its expectation contains an extra term (Proposition 15) that vanishes under standard importance sampling but is nonzero in general when the ratio is regularized.

These results suggest the following practice. With sequence-level importance sampling, the group-mean baseline alone sufices, and score centering should be omitted. With token-level importance sampling and a regularized ratio, score centering should be applied, either together with the group-mean baseline or with a constant baseline; the latter has smaller variance only if the baseline is suficiently close to $b ^ { \star }$ , which is prompt-dependent and must be estimated from multiple rollouts. Finally, because both families of estimators degrade as the training–inference mismatch grows, it remains useful to reduce the mismatch at its source, e.g., through matched kernels, batch-invariant computation, and consistent arithmetic precision across the training and inference engines.

## References

Greensmith, E., Bartlett, P. L., and Baxter, J. (2004). Variance reduction techniques for gradient estimates in reinforcement learning. Journal of Machine Learning Research, 5:1471–1530.

Kakade, S. and Langford, J. (2002). Approximately optimal approximate reinforcement learning. In International Conference on Machine Learning, pages 267–274.

Kong, A. (1992). A note on importance sampling using standardized weights. Technical Report 348, Department of Statistics, University of Chicago.

Kool, W., van Hoof, H., and Welling, M. (2019). Buy 4 REINFORCE samples, get a baseline for free! In Deep Reinforcement Learning Meets Structured Prediction, ICLR 2019 Workshop.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H.,

and Stoica, I. (2023). Eficient memory management for large language model serving with PagedAttention. In Symposium on Operating Systems Principles, pages 611–626.

Liu, J., Li, Y., Fu, Y., Wang, J., Liu, Q., and Shen, Y. (2025a). When speed kills stability: Demystifying RL collapse from the training–inference mismatch.

Liu, J. S. (2001). Monte Carlo Strategies in Scientific Computing. Springer Series in Statistics. Springer, New York.

Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., and Lin, M. (2025b). Understanding R1-Zero-like training: A critical perspective. In Conference on Language Modeling.

Marek, M. and Ryabinin, M. (2026). Score centering stabilizes of-policy reinforcement learning. arXiv preprint arXiv:2609.20807.

Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. (2015). Trust region policy optimization. In International Conference on Machine Learning, pages 1889–1897.

Schulman, J., Moritz, P., Levine, S., Jordan, M. I., and Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. In International Conference on Learning Representations.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., and Catanzaro, B. (2019). Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053.

Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, 2nd edition.

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4):229–256.

Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.-C., Xu, M., Wright, L., Shojanazeri, H., Ott, M., Shleifer, S., Desmaison, A., Balioglu, C., Damania, P., Nguyen, B., Chauhan, G., Hao, Y., Mathews, A., and Li, S. (2023). PyTorch FSDP: Experiences on scaling fully sharded data parallel. Proceedings of the VLDB Endowment, 16(12):3848–3860.

Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barrett, C., and Sheng, Y. (2024). SGLang: Eficient execution of structured language model programs. In Advances in Neural Information Processing Systems (NeurIPS), volume 37.