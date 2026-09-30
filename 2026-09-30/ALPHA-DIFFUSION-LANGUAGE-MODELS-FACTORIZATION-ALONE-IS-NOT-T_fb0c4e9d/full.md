# ALPHA DIFFUSION LANGUAGE MODELS:FACTORIZATION ALONE IS NOT THE PROBLEM

Nikita Gushchin<sup>1,2</sup> Dmitry Baranchuk<sup>3</sup> Alexander Korotin<sup>1,2</sup> <sup>1</sup>Applied AI Institute, Moscow, Russia <sup>2</sup>AXXX, Russia <sup>3</sup>Yandex Research

## ABSTRACT

Discrete diffusion language models can generate multiple tokens in parallel, but reducing the number of denoising steps can lead to inconsistent predictions. Standard cross-entropy training fits conditional token marginals, whereas parallel generation requires consistent joint predictions. We introduce Alpha Diffusion Language Models (AlphaDLM), trained with a sequence-level alpha loss that recovers cross-entropy in the limit of vanishing alpha and has a joint-mode optimum at alpha one. Our analysis characterizes how the objective and factorization jointly determine the fitted distribution. We identify conditions under which intermediate alpha preserves multiple valid completions while excluding invalid token combinations. Trained on TinyGSM, our method achieves 34.6% accuracy on GSM8K with only four model evaluations. We further scale the method to SDAR-1.7B and evaluate it on code and mathematics benchmarks. These results show that changing the training objective can improve the accuracy-computation trade-off of factorized diffusion language models.

![](images/88d48bdf0e549bdf3f7f0032ac1bee8de394305a4ff87810a07410e88430b775.jpg)  
AlphaDLM (ours) FMLM+ (Init) MDLM (ancestral) -FLM (exact, T= 0.1) MDLM (confidence) DUO (T= 0.1) -FLM (top-1) DBTM  
Figure 1: GSM8K accuracy with 1–16 model evaluations. A single AlphaDLM (ours) checkpoint with k = 16 reaches 34.6% in four evaluations. AlphaDLM and MDLM (confidence) use fixed-NFE confidence sampling with the token-reveal schedule of FMLM+. Other methods use their respective samplers. Sources and protocols: Table 1 and Appendix C.

## 1 INTRODUCTION

Discrete diffusion language models generate text by reconstructing corrupted sequences (Sahoo et al., 2024; Nie et al., 2025). A single evaluation predicts distributions at many positions, allowing several tokens to be committed in parallel. Independently predicted tokens can each fit the visible context yet form an invalid sequence when generated together (Wu et al., 2026).

Under cross-entropy training, the optimal factorized denoiser fits the conditional distribution of each token separately. When several completions are possible, independently combining these predictions can produce invalid sequences (Figure 2). One approach is to model token dependencies explicitly (Liu et al., 2025; Xu et al., 2025). However, a factorized distribution can already represent any single valid completion by assigning probability one to its tokens. Factorization therefore does not by itself prevent consistent parallel predictions. This suggests changing the training objective to favor token distributions that produce valid completions when combined.

We introduce Alpha Diffusion Language Models (AlphaDLM), trained with sequence-level alpha loss, a form of generalized cross-entropy (Zhang & Sabuncu, 2018). This loss includes cross-entropy as the limit $\alpha  0 ,$ , while at α = 1, predicting a joint mode minimizes the expected loss. These endpoints suggest a way to move from fitting token marginals toward fitting complete sequences. We analyze what happens between them when the denoiser is factorized. The optimum can preserve several valid completions while excluding invalid combinations, rather than selecting only one answer. We establish sufficient conditions for this behavior.

This change in the training objective is intended to improve generation when many tokens must be predicted in parallel within a small number of model evaluations. Recent approaches pursue this goal through trajectory supervision or self-distillation (Zhang et al., 2026; Chen et al., 2026; Kim et al., 2026). Our method instead trains directly on corrupted data, without teacher-generated trajectories, and uses the factorized denoiser. We evaluate whether this change improves accuracy at a fixed generation budget, using TinyGSM for comparisons between training objectives and SDAR-1.7B for experiments on code and mathematics. Figure 1 summarizes the low-NFE result on GSM8K. Appendix A further discusses related work.

Contributions. We combine this training approach with an analysis of its factorized optimum and experiments on parallel generation. Our main contributions are:

1. We introduce Alpha Diffusion Language Models: diffusion language models whose factorized denoisers are trained with sequence-level alpha loss. We characterize the distribution that minimizes this loss within the class of factorized models and explain how it differs from the optimum of token-wise alpha loss (Sections 3.1–3.3).

2. We establish conditions under which the optimal factorized denoiser trained with sequencelevel alpha loss assigns probability only to valid completions and can preserve several possible answers (Section 3.2, Appendix B.1).

3. We empirically demonstrate that our method achieves 34.6% GSM8K accuracy with four model evaluations and 7.66% with a single evaluation after training on TinyGSM. Sequence-level training also achieves higher accuracy than token-wise alpha loss at matched evaluation budgets (Section 4.1). We also demonstrate improved accuracy–TPF trade-offs with SDAR-1.7B on code and mathematics benchmarks (Section 4.2).

## 2 BACKGROUND

To understand how the training objective affects parallel generation, we first review how discrete diffusion language models are trained and decoded (Section 2.1). We then turn to generalized crossentropy, a family of losses that includes standard cross-entropy as a limiting case (Section 2.2).

## 2.1 DISCRETE DIFFUSION LANGUAGE MODELS

Discrete diffusion models learn to reverse a corruption process (Austin et al., 2021a; Lou et al., 2024). A denoiser predicts clean tokens from corrupted inputs, and a sampler uses these predictions to reconstruct the sequence.

![](images/3f9360a2dddae2bda1ecc95657baa0c1e677d1a6189c453618a8a4f7c8a6c803.jpg)  
Figure 2: Marginal fitting, token-mode fitting, and joint-mode fitting. Cross-entropy assigns probability 0.48 to ungrammatical completions. $\mathrm { A t } \ \alpha = 1$ , token-wise alpha loss predicts $\texttt { I } \texttt { i s } ,$ while sequence-level alpha loss fits the joint mode I am. Each fitted distribution minimizes its population loss within the same factorized family. Green and red mark grammatical and ungrammatical outcomes with positive probability.

Corruption process. Let $x \in \mathcal { V } ^ { L }$ be a clean sequence conditioned on a prompt or preceding blocks c, and let $z \sim q _ { t } ( \cdot \mid x )$ be its corruption at time $t \in [ 0 , 1 ]$ . Discrete corruption replaces tokens with vocabulary symbols or a mask. Masked diffusion independently keeps each token or replaces it with m $\notin \mathcal { V } \colon$

$$
q _ { t } ( z _ { i } \mid x _ { i } ) = { \bar { \alpha } } _ { t } \mathbf { 1 } \{ z _ { i } = x _ { i } \} + ( 1 - { \bar { \alpha } } _ { t } ) \mathbf { 1 } \{ z _ { i } = \mathbf { n } \} , \qquad q _ { t } ( z \mid x ) = \prod _ { i = 1 } ^ { L } q _ { t } ( z _ { i } \mid x _ { i } ) .\tag{1}
$$

The probability of keeping a token unchanged decreases from $\bar { \alpha } _ { 0 } = 1 \mathrm { t o } \bar { \alpha } _ { 1 } = 0$ . Despite independent corruption, the clean tokens can remain dependent conditional on z.

Factorized denoiser. For this masking process, the optimal clean-token denoiser under crossentropy training is independent of time t (Zheng et al., 2025, Proposition 3.2). We therefore write the denoiser without explicit time conditioning. Let $M = \{ i : \dot { z } _ { i } = \mathfrak { m } \}$ be the masked positions and $x _ { M } = ( x _ { i } ) _ { i \in M }$ their original tokens, in sequence order. Given the visible context, the denoiser predicts a categorical distribution at each position in M:

$$
p _ { \theta } ( x _ { M } \mid z , c ) = \prod _ { i \in M } p _ { \theta , i } ( x _ { i } \mid z , c ) .\tag{2}
$$

Factorization holds within each evaluation. Later predictions depend on previously revealed tokens, so the generated sequence can contain dependencies despite the factorized denoiser.

Training objective. The denoiser learns from corrupted inputs using weighted token cross-entropy (Sahoo et al., 2024; Shi et al., 2024):

$$
\mathcal { L } _ { \mathrm { C E } } ( \theta ) = \mathbb { E } _ { { x , c , t , z } } \left[ - w ( t ) \sum _ { i \in M } \log p _ { \theta , i } ( x _ { i } \mid z , c ) \right] ,\tag{3}
$$

The expectation is over data, training times, and corruptions $z \sim q _ { t } ( \cdot \mid x )$ . With uniform times and $\bar { \alpha } _ { t } = 1 - t$ , the continuous-time masked diffusion bound uses $w ( t ) = 1 / t$

For a token value $v \in \mathcal V$ , fixed $( z , c )$ , and unrestricted token predictors, the population cross-entropy is minimized by the conditional token marginals, $p _ { i } ^ { * } ( v \mid z , c ) = p _ { \mathrm { d a t a } } ( x _ { i } = v \mid z , c )$ . Their product can differ from the joint posterior over the missing tokens. In particular, independently selecting their modes can produce invalid token combinations.

Parallel decoding. Confidence sampling selects positions to reveal using the maximum predicted token probability at each position. In our confidence-sampling experiments, revealed tokens are chosen by argmax. Fixed-NFE confidence sampling uses a prescribed evaluation budget and ranks positions by confidence to meet a reveal schedule (Nie et al., 2025). The fixed-NFE sampler used for TinyGSM also reveals positions above a threshold (Section 4.1). Adaptive confidence sampling follows Fast-dLLM (Wu et al., 2026): it reveals positions above a threshold, or the most confident position if none qualifies. It runs until all positions are filled, so NFE varies across examples.

Blockwise diffusion. The same denoising procedure can operate within blocks generated autoregressively (Arriola et al., 2025; Cheng et al., 2025). Writing $x ^ { ( b ) }$ for block b among B blocks and $\bar { \boldsymbol x } ^ { ( < b ) }$ for all preceding blocks, the full sampling distribution is

$$
p _ { \theta } ^ { \mathrm { g e n } } ( x \mid c ) = \prod _ { b = 1 } ^ { B } p _ { \theta } ^ { \mathrm { g e n } } ( x ^ { ( b ) } \mid x ^ { ( < b ) } , c ) .\tag{4}
$$

Each block uses iterative calls to $p _ { \theta }$ with the preceding context cached.

## 2.2 GENERALIZED CROSS-ENTROPY

The cross-entropy used to train the diffusion denoiser is the standard loss for categorical prediction. For a target class y and predicted class probabilities $p ,$ it is $\ell _ { \mathrm { { C E } } } ( p , y ) = - \log p ( y )$ . This familiar loss is a limiting case of a broader family, generalized cross-entropy, also known as alpha loss (Zhang & Sabuncu, 2018):

$$
\ell _ { \alpha } ( p , y ) = { \frac { 1 - p ( y ) ^ { \alpha } } { \alpha } } , \qquad \operatorname* { l i m } _ { \alpha \to 0 } \ell _ { \alpha } ( p , y ) = - \log p ( y ) .\tag{5}
$$

The parameter α controls the penalty for low target probabilities. Cross-entropy diverges as $p ( y ) $ 0, whereas alpha loss is bounded by $1 / \alpha$ for $\alpha > 0 . \mathrm { A t } \alpha = 1$ , it reduces to $1 - p ( y )$ .

## 3 ALPHA DIFFUSION LANGUAGE MODELS

We first define our sequence-level objective by applying alpha loss to the joint probability of the masked tokens (Section 3.1). We then analyze how factorization changes the distribution that minimizes this loss, compare sequence-level and token-wise training, and establish conditions for excluding invalid token combinations (Section 3.2). Finally, we describe how we normalize the loss across different mask counts and train the denoiser for use with parallel and blockwise samplers (Section 3.3).

## 3.1 SEQUENCE-LEVEL DENOISING OBJECTIVE

We seek a training objective that improves joint predictions while using a factorized denoiser. Given a corrupted sequence $z ,$ the denoiser reconstructs the original tokens at the masked positions M. As in Section 2.1, we omit explicit time conditioning because the conditional distribution of the original tokens given z and c does not depend on t. The factorized denoiser assigns them the joint probability

$$
p _ { \theta } ( x _ { M } \mid z , c ) = \prod _ { i \in M } p _ { \theta , i } ( x _ { i } \mid z , c ) .
$$

Applying cross-entropy to this probability still gives a sum of token-level losses, because the logarithm turns the product into a sum. For positive α, generalized cross-entropy (Section 2.2) applied to this product no longer gives a sum of separate token losses. This yields the sequence-level alpha loss:

$$
\ell _ { \alpha } ^ { \mathrm { s e q } } ( \theta ; x _ { M } , z , c ) = \frac { 1 - p _ { \theta } ( x _ { M } \mid z , c ) ^ { \alpha } } { \alpha } = \frac { 1 - \left[ \prod _ { i \in M } p _ { \theta , i } ( x _ { i } \mid z , c ) \right] ^ { \alpha } } { \alpha } .\tag{6}
$$

This loss recovers the sum of token cross-entropies as $\alpha  0$ (Section 2.2).

## 3.2 FROM MARGINAL FITTING TO JOINT MODE FITTING

Fix a corrupted sequence z and context c. Let $q$ denote the conditional distribution of the original tokens at the masked positions:

$$
q ( y ) = p _ { \mathrm { d a t a } } ( x _ { M } = y \mid z , c ) .\tag{7}
$$

We compare the distribution minimizing the expected sequence-level alpha loss with and without factorization over the $m = | M |$ masked positions, indexed by $i \in M$ . Figure 3 compares these two cases with token-wise training at the same values of $\alpha .$

Without factorization. We first allow the predictor to represent any joint distribution. For $0 < \alpha <$ 1, we define the lower-temperature target

$$
q _ { T } ( y ) : = { \frac { q ( y ) ^ { 1 / T } } { \sum _ { y ^ { \prime } } q ( y ^ { \prime } ) ^ { 1 / T } } } , \qquad T = 1 - \alpha .\tag{8}
$$

The temperature $T$ changes the target distribution, not the sampling procedure. The expected sequence-level alpha loss is minimized by fitting this target (Sypherd et al., 2022, Proposition 1):

$$
p _ { \alpha } ^ { * , \mathrm { j o i n t } } ( y ) = q _ { T } ( y ) .\tag{9}
$$

At $\alpha = 1$ , any distribution supported on joint modes is optimal. For $\alpha > 1$ , only point masses on joint modes are optimal. Appendix B gives the parameter correspondence and the elementary argument for $\alpha \geq 1$

Without factorization, alpha loss fits a lower-temperature target. For $0 \textless \alpha \textless 1$ , more likely complete sequences receive a larger share of the probability (Figure 3, top row).

With factorization. Lowering the temperature does not remove token dependencies, so q generally cannot be represented by a factorized denoiser. The optimal denoiser instead fits the marginals of a joint distribution that balances closeness to q<sub>T</sub> against dependence between tokens. For $0 < \alpha < 1$ this distribution solves (Lapidoth & Pfister, 2019, Lemma 8, extended to m factors):

$$
\pi _ { \alpha } ^ { * } \in \arg \operatorname* { m i n } _ { \pi } \left\{ D _ { \mathrm { K L } } ( \pi \| q _ { T } ) + \frac { \alpha } { 1 - \alpha } \mathrm { T C } ( \pi ) \right\} , \quad p _ { \alpha } ^ { * , \mathrm { f a c t } } = \prod _ { i \in M } \pi _ { \alpha , i } ^ { * } .\tag{10}
$$

The KL term measures the departure from $q _ { T }$ . Total correlation, $\begin{array} { r } { \mathrm { T C } ( \pi ) = D _ { \mathrm { K L } } ( \pi \| \prod _ { i \in M } \pi _ { i } ) } \end{array}$ measures dependence between tokens and is zero when they are independent. Appendix B gives the derivation, including zero probabilities.

With factorization, the denoiser fits the marginals of $\pi _ { \alpha } ^ { * } .$ . The auxiliary distribution $\pi _ { \alpha } ^ { * }$ balances closeness to $q _ { T }$ against token dependence. The denoiser represents the product of its marginals (Figure 3, middle row).

Why sequence-level rather than token-wise? Applying alpha loss separately to each token probability gives

$$
\ell _ { \alpha } ^ { \mathrm { t o k } } = \sum _ { i \in M } { \frac { 1 - p _ { \theta , i } ( x _ { i } \mid z , c ) ^ { \alpha } } { \alpha } } .\tag{11}
$$

For $0 < \alpha < 1$ , minimizing its expectation gives

$$
p _ { \alpha , i } ^ { * , \mathrm { t o k } } ( y _ { i } ) = \frac { q _ { i } ( y _ { i } ) ^ { 1 / T } } { \sum _ { y _ { i } ^ { \prime } \in \mathcal { V } } q _ { i } ( y _ { i } ^ { \prime } ) ^ { 1 / T } } , \qquad T = 1 - \alpha .\tag{12}
$$

Here $q _ { i }$ is the marginal of $q$ at position $i ,$ and $y _ { i } \in \mathcal { V }$ is a token value. When q factorizes, the two objectives have the same optimum.

Sequence-level fitting uses the joint target. Token-wise loss sharpens each marginal of $q$ separately. Sequence-level loss instead fits the marginals of $\pi _ { \alpha } ^ { * } .$ , which depends on the joint distribution q. Figures 2 and 3 compare the endpoints and intermediate α, respectively.

$$
\begin{array} { r l r } { \sum _ { j = 1 } ^ { n } \sum _ { i = 1 } ^ { n } \mu _ { i + 1 } ^ { j } } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } \\ { = } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } \\ { = } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } \\ { = } & { \exp \left( \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( \frac { 1 } { n } \right) } \\ { = } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } \\ { = } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( \frac { 1 } { n } \right) } \\ { = } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } \\ { = } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } \\ { = } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } \\ { = } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } \\ { = } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } \\ { = } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } \\ { = } & { \exp \left( - \frac { 1 } { n } \right) } & { \exp \left( - \frac { 1 } { n } \right) } \\  \sum _ { j = 1 } ^ { n } \end{array}
$$

Figure 3: Interpolating from cross-entropy to mode fitting. Rows show globally optimal distributions for Figure $2 \mathrm { { : } } \mathrm { { s } }$ target: sequence-level loss without factorization, sequence-level loss with factorization, and token-wise loss with factorization. Columns use the same $\alpha ,$ with $\alpha = 0$ denoting cross-entropy. $\mathrm { A t } ~ \alpha = 0 . 5$ , the unrestricted optimum $q _ { T }$ supports all three original completions, while the sequence-level factorized optimum supports only he is and she is. Both sequencelevel rows reach I am at $\alpha = 1$ , whereas token-wise fitting reaches I is. Here $p = p ( x _ { 1 } , x _ { 2 } )$ and $p _ { i } = p _ { i } ( x _ { i } )$ . Green marks grammatical combinations, red ungrammatical ones. Entries are rounded.

Preserving multiple valid completions. Can a factorized predictor exclude invalid token combinations while assigning positive probability to several completions? Here, valid completions are those with positive probability under $q .$ The following result gives sufficient conditions for this behavior.

Proposition 1 (Multiple valid completions). Consider $m \geq 2$ masked tokens, and suppose the valid completions split into groups such that any within-group combination of tokens is valid, and different groups use disjoint token sets at every position. For $1 / m < \alpha < 1$ , every globally optimal factorized predictor selects one group and assigns positive probability to all completions in that group.

To illustrate the behavior described by Proposition 1, we consider the two-token example $( m = 2 )$ in Figure 3 at $\alpha = 1 / m = 1 / 2$ , the boundary of the proposition’s range. Without factorization, the optimal predictor still assigns positive probability to all three original completions. With factorization, the first position predicts either he or she, and the second always predicts is. Independent predictions then produce only he is and she is, both valid completions. This is a global optimum for this example, as verified separately at the boundary. Appendix B.1 gives the calculation for this example. As α increases further, the sequence-level optimum switches to I am, whereas token-wise fitting approaches the invalid combination I is.

Appendix B.1 gives an illustrated elementary proof and a counterexample showing why the support condition matters.

## 3.3 TRAINING AND DECODING

During training, examples have different numbers of masked tokens $m = | M |$ . Multiplying more token probabilities makes the joint probability smaller, even when the probability assigned to each target token stays the same. We therefore normalize the log-probability by m. For $m > 0$ , let $\textstyle s _ { \theta } { \overset { \cdot } { = } } { \frac { 1 } { m } } \sum _ { i \in M } { \mathrm { ~ l o g ~ } } p _ { \theta , i } ( x _ { i } \mid z , c )$ . With a fixed training parameter k, we use:

$$
\ell _ { k } ( s _ { \theta } ) = \frac { 1 - \exp ( k s _ { \theta } ) } { k } = \frac { 1 - p _ { \theta } ( x _ { M } \mid z , c ) ^ { \alpha _ { \mathrm { e f f } } } } { m \alpha _ { \mathrm { e f f } } } , \qquad \alpha _ { \mathrm { e f f } } = \frac { k } { m } .\tag{13}
$$

This is sequence-level alpha loss (Equation 6) with $\alpha _ { \mathrm { e f f } } = k / m$ , scaled by $1 / m$ . At a fixed corrupted context, this scaling leaves the optimum unchanged. $\mathbf { A s } \ k  0$ , the loss reduces to mean token cross-entropy.

For $k > 0$ , the gradient is the mean-token CE gradient multiplied by $\exp ( k s _ { \theta } )$ (Appendix B). Larger k gives less weight to examples whose target tokens receive low probabilities. We therefore initialize training from a pretrained denoiser.

To relate this training rule to Section 3.2, substitute $\alpha = k / m$ . Equation 10 applies when $0 ~ <$ $k < m$ . Under the assumptions of Proposition 1, $1 < k < m$ guarantees that the optimal predictor assigns positive probability to every completion in one group and none outside it. For $k \geq m$ predicting a single most likely completion is optimal (Appendix B).

At inference, we use the parallel or blockwise samplers described in Section 2.1.

## 4 EXPERIMENTS

We compare training objectives in the TinyGSM-to-GSM8K setting, a common evaluation setup in recent work on diffusion and flow language models (Deschenaux & Gulcehre, 2026; Agarwal et al., 2026; Li et al., 2026a) (Section 4.1). We then test transfer to SDAR-1.7B on code and mathematics (Section 4.2). TinyGSM results use task accuracy versus the number of model evaluations (NFE). SDAR results use accuracy versus tokens per forward (TPF), defined as returned response tokens divided by denoising forwards plus one prompt-prefill forward per example.

Table 1: GSM8K accuracy (%) ↑ at fixed NFE. Recomputed MDLM, DUO, and S-FLM results and published FMLM+ and DBTM values. IDLM values are reevaluated at 4–16 NFE and taken from the publication at 32–64 NFE. MDLM (confidence) and AlphaDLM use the same fixed-NFE confidence sampler. Other baselines use their original samplers. Bold values are the column maxima. Protocols are detailed in Appendix C.
<table><tr><td>Method / NFE</td><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td></tr><tr><td>MDLM (ancestral) (Sahoo et al., 2024)</td><td>1.7</td><td>5.3</td><td>13.0</td><td>22.1</td><td>28.2</td></tr><tr><td>DUO (T = 0.1) (Sahoo et al., 2025)</td><td>5.2</td><td>14.0</td><td>23.0</td><td>30.8</td><td>31.2</td></tr><tr><td>S-FLM (exact, T = 0.1) (Deschenaux &amp; Gulcehre, 2026)</td><td>7.6</td><td>12.6</td><td>15.1</td><td>16.9</td><td>16.9</td></tr><tr><td>S-FLM (top-1)</td><td>7.7</td><td>12.4</td><td>14.9</td><td>16.5</td><td>17.4</td></tr><tr><td>FMLM+ (Agarwal et al., 2026)</td><td>2.9</td><td>8.7</td><td>13.4</td><td>19.0</td><td>19.1</td></tr><tr><td>FMLM+ (Distill)</td><td>3.9</td><td>10.3</td><td>18.7</td><td>21.6</td><td>23.4</td></tr><tr><td>FMLM+ (Init)</td><td>5.1</td><td>15.1</td><td>26.1</td><td>31.8</td><td>33.6</td></tr><tr><td>DBTM (Tang &amp; Wang, 2026)</td><td>5.7</td><td>10.1</td><td>14.3</td><td>16.8</td><td>16.2</td></tr><tr><td>IDLM (MDLM) (Li et al., 2026a)</td><td>0.4</td><td>1.8</td><td>6.5</td><td>12.8</td><td>14.9</td></tr><tr><td>IDLM (DUO)</td><td>2.3</td><td>7.1</td><td>11.7</td><td>15.4</td><td>19.0</td></tr><tr><td>MDLM (confidence)</td><td>4.2</td><td>20.1</td><td>37.7</td><td>43.9</td><td>40.6</td></tr><tr><td>AlphaDLM (ours, k = 8)</td><td>28.9</td><td>44.0</td><td>49.2</td><td>49.9</td><td>50.0</td></tr><tr><td>AlphaDLM (ours, k = 16)</td><td>34.6</td><td>39.0</td><td>40.6</td><td>40.9</td><td>40.9</td></tr></table>

## 4.1 TINYGSM

Training setup. We train on TinyGSM (Liu et al., 2023) and evaluate Python solutions on all 1,319 GSM8K test problems (Cobbe et al., 2021), following the protocol of S-FLM and FMLM+ (Deschenaux & Gulcehre, 2026; Agarwal et al., 2026). Objective ablations start from a shared $k = 1$ checkpoint at 200k updates and continue for 50k updates with the same architecture and optimizer settings. We test sequence-level $k \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ and token-wise $\alpha \in \{ 0 . 2 5 , 0 . 3 7 5 , 0 . 5 , 0 . 7 5 \}$ with larger k tested for one-evaluation generation. We use $\alpha _ { \mathrm { e f f } } = k / m$ (Equation 13) and average $\ell _ { k }$ over data and corruptions without time weighting. Each recipe uses one training seed and EMA weights. Appendix C gives the protocols.

Sampling protocol. We use fixed-NFE confidence sampling (Section 2.1) with the FMLM+ token-reveal schedule (Agarwal et al., 2026). Each round reveals positions above confidence 0.999, adding highest-confidence positions as needed to distribute the remaining masks across the remaining rounds. Tokens are chosen by argmax. Regardless of the threshold, every example executes all prescribed forwards, even if all masks are filled earlier.

Comparison with existing methods. At four NFE, our AlphaDLM (k = 16) reaches 34.6% accuracy, compared with 4.2% for MDLM using the same sampler and at most 7.7% for the other baselines shown (Figure 1, Table 1).

To isolate the effect of applying alpha loss jointly rather than token-wise, we compare runs with the same initialization, training budget, and sampler (Figure 4 (a,b)). At four NFE, sequence-level k = 16 reaches 34.6%, compared with 8.1% for the $k = 1$ continuation and 19.9% for the best tested token-wise setting. At sixteen NFE, sequence-level k = 8 reaches 49.2%, compared with 44.4% for the best token-wise setting. The best token-wise parameter is selected separately at each budget.

With the same initialization, training budget, and sampler, sequence-level alpha loss gives higher accuracy than every tested token-wise setting at four and sixteen NFE.

(a) Sequence-level sweep  
![](images/aa2c116e3001c7dce8ff41c15ec675655a53e6e6b5e41e4b76485ac453087e18.jpg)

(b) Objective comparison  
![](images/c6834665a6058af423dd609ff8529faf96023fca2ccdbea7b48ac389a87c6880.jpg)  
Token α = 0.25 Token α = 0.75

(c) One-evaluation: sequence-levelFigure 4: How the objective changes parallel prediction.(d) One-evaluation: token-wise(a) Sequence-level sweep. (b) Token-) <sub>7.66</sub>wise comparison. All runs continue the same 200k-update $k = 1$ checkpoint to 250k using fixed-<sup>8</sup> <sup>(</sup> 6.75NFE confidence sampling. Values: Table 3. Panels (c,d) continue on page 9.

uChoosing k. The advantage of $k = 1 6$ at four NFE does not persist as we allow more evaluations. At 4a<sup>c</sup> <sub>2.73</sub>sixteen NFE, it reaches 40.6%, matching $k = 1$ but trailing $k = 8 .$ . Each curve uses one checkpoint, <sup>K</sup> 1.82 <sub>1.67</sub>so this reversal shows that the best tested k depends on the decoding budget.

To check whether the comparison depends on other experimental choices, we also evaluate adaptive20K confidence sampling (Appendix C.1) and CE initialization (Appendix E). With CE initialization,M sequence-level training still outperforms the tested token-wise settings at four and sixteen NFE.10<sub>G</sub>

One-evaluation generation. We next test whether this advantage persists when all tokens are predicted in a single model evaluation. Every masked position receives its argmax token, so a correct <sup>solution</sup> <sup>requires</sup> <sup>the</sup> <sup>independently</sup> <sup>predicted</sup> <sup>tokens</sup> <sup>to</sup> <sup>form</sup> <sup>a</sup> <sup>jointly</sup> <sup>correct</sup> <sup>answer</sup> <sup>(Figure</sup> <sup>4</sup> <sup>(c,d)).</sup>Model evaluations (NFE) Model evaluations (NFE) $\mathbf { A } \mathbf { t } k = 1 6 .$ , our method solves 83 of 1,319 problems (6.29%), compared with 28 (2.12%) for the best tested token-wise setting. Accuracy increases from 0.91% at<sup>k</sup>= 1 <sup>k</sup>= 8 $k = 1$ to 101/1,319 (7.66%) atce k = 8 Token α = 0 $k = 3 2 .$ then falls to $6 . 7 5 \%$ at $\bar { k = 6 4 }$ . These results remain below multi-step accuracy and do not identifyk = 16 Sequence k = 16 Token α = 0.5 the true conditional mode.

![](images/ffe59e82adf1c9636598ca636ae4d815b69dd255f61e52402b4a0bd8329cfd99.jpg)

![](images/f8d98f2c44d8a2a1cde5cf6b40a6d7e0f7fc6091c2c11ef1d7ee496847bdb400.jpg)  
Figure 4 (continued): One-evaluation generation. (c,d) Exact-argmax accuracy for sequence-level and token-wise sweeps from the same initialization.

Ancestral sampling. We also test whether the improvement depends on confidence-based position selection. In this sampler, reveal positions are random and independent of confidence (Figure 5, Table 4). At 32 NFE, increasing k from 1 to 16 raises accuracy from 8.2% to 32.5% with sampled tokens $( T = 1 )$ , and from 26.0% to 37.1% with argmax tokens $( T = 0 )$ . The gain at $T = 0$ also rules out scalar logit sharpening as the sole explanation, since sharpening changes neither argmax tokens nor reveal probabilities. Appendix F reports ancestral pass@K.

The gains persist when all tokens are predicted at once and when reveal positions are sampled independently of confidence.

![](images/17351760a434e6ce7a577523aa02c73093b27762f0a55a8746ce6d00e87db5ef.jpg)  
Figure 5: Ancestral sampling at 32 NFE. Reveal positions are random and independent of confidence. Tokens are selected by argmax $( T = 0 )$ or sampled $( T = 1 )$ . Values: Table 4.

## 4.2 TRANSFER TO SDAR ON MATHEMATICS AND CODE

We next test whether sequence-level alpha loss also improves blockwise generation with SDAR-1.7B-Chat (Cheng et al., 2025). We fine-tune on Nemotron-SFT-Math-v4 for mathematics and OpenCodeInstruct for code, and compare our AlphaDLM with the original model, CE fine-tuning, and entropy-regularized fine-tuning (CE+CAP) (Bie et al., 2025). We apply Equation 13 separately to each four-token response block. All methods use adaptive confidence sampling following FastdLLM (Wu et al., 2026) within four-token blocks (Section 2.1), with argmax tokens. We vary the threshold to compare accuracy and TPF. Figure 6 (a–d) uses k = 0.8 for mathematics and k = 0.4 for code. Panels (e,f) show the mathematics parameter sweep. Appendix D gives training and evaluation details, including block-loss aggregation.

Mathematics. At threshold 0.99, our method improves both accuracy and TPF over CE (Figure 6 (a,b)). Accuracy reaches 51.5% on MATH500 and 77.2% on GSM8K, compared with 50.6% and 76.3% for CE. TPF increases by 31.6% and 27.1%, respectively.

![](images/c30961b8bdc3fa0b1fbe895adaef2a25ac3c2b988510ae9a012dab9ffa657ebd.jpg)

Figure 6: Accuracy–TPF trade-offs with SDAR-1.7B. (a–d) Objective comparisons on mathematics and code. AlphaDLM (ours) uses $k = 0 . 8$ for mathematics and $k = 0 . 4$ for code. (e,f) Mathematics parameter sweep. All panels use adaptive confidence sampling with argmax tokens. Points average accuracy and TPF across runs at each confidence threshold. Rings in (a–d) mark threshold 0.99. Appendix D provides standard-deviation bands (Figures 12–13).

Code. At the same threshold, our method increases TPF over CE+CAP with similar pass@1 (Figure 6 (c,d)). On HumanEval-Instruct, both achieve 60.6% pass@1, while TPF increases from 1.74 to 1.94. On MBPP-Instruct, TPF increases from 1.51 to 1.68 with a 0.19-percentage-point drop in pass@1. This comparison concerns threshold 0.99. Across the full sweep, peak accuracy is highest for CE on HumanEval-Instruct and CE+CAP on MBPP-Instruct.

Choosing k. The mathematics sweep shows that larger k can increase TPF at the cost of peak accuracy (Figure 6 (e,f)). Among the tested values, $\bar { k } = 0 . 4$ and $k = 0 . 8$ give the highest peak accuracies. Increasing k to 1.6 or 2.0 increases TPF but reduces peak accuracy on both benchmarks.

## 5 DISCUSSION AND CONCLUSION

AlphaDLM applies alpha loss to the joint probability of masked tokens, interpolating between crossentropy and joint-mode fitting. Our analysis explains why this sequence-level objective behaves differently from applying the same loss to individual tokens. For $0 < \alpha < 1$ , alpha loss lowers the target temperature without factorization. With factorization, the optimal denoiser fits marginals of a joint distribution that balances closeness to this target against dependence between tokens. Under the conditions of Proposition 1, it can exclude invalid combinations while preserving several valid completions.

The experiments show that sequence-level alpha loss improves low-NFE accuracy while keeping the factorized architecture. The gains persist in fully parallel prediction and ancestral sampling, extending beyond confidence-based position selection. Together, these results show that the limitations of parallel prediction depend on the combination of factorization and the training objective. Alpha loss provides a way to improve this combination without explicitly modeling token dependencies within each denoising evaluation.

## REFERENCES

Manan Agarwal, Sheel Shah, Chanhyuk Lee, Jaehoon Yoo, Jerry Huang, Seunghoon Hong, Aditi Raghunathan, Jinwoo Kim, and Nicholas M. Boffi. Posterior Refinement: Fast Language Generation via Any-Order Flow Maps, 2026. URL https://arxiv.org/abs/2606.24773.

Wasi Uddin Ahmad, Aleksander Ficek, Mehrzad Samadi, Jocelyn Huang, Vahid Noroozi, Somshubra Majumdar, and Boris Ginsburg. OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs, 2025. URL https://arxiv.org/abs/2504.04030.

Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2503.09573.

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured Denoising Diffusion Models in Discrete State-Spaces. In Advances in Neural Information Processing Systems, volume 34, 2021a. URL https://arxiv.org/abs/2107.03006.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program Synthesis with Large Language Models, 2021b. URL https://arxiv.org/abs/2108.07732.

Parikshit Bansal and Sujay Sanghavi. Enabling approximate joint sampling in diffusion lms, 2025. URL https://arxiv.org/abs/2509.22738.

Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, Chengxi Li, Chongxuan Li, Jianguo Li, Zehuan Li, Huabin Liu, Lin Liu, Guoshan Lu, Xiaocheng Lu, Yuxin Ma, Jianfeng Tan, Lanning Wei, Ji-Rong Wen, Yipeng Xing, Xiaolu Zhang, Junbo Zhao, Da Zheng, Jun Zhou, Junlin Zhou, Zhanchao Zhou, Liwang Zhu, and Yihong Zhuang. LLaDA2.0: Scaling up diffusion language models to 100B, 2025. URL https://arxiv.org/abs/2512.15745.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL https:// arxiv.org/abs/2107.03374.

Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, and Xinchao Wang. dParallel: Learnable parallel decoding for dLLMs. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=hVOcstAURb.

Shuang Cheng, Yihan Bian, Dawei Liu, Linfeng Zhang, Qian Yao, Zhongbo Tian, Wenhai Wang, Qipeng Guo, Kai Chen, Biqing Qi, and Bowen Zhou. SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation, 2025. URL https://arxiv. org/abs/2510.06303.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training Verifiers to Solve Math Word Problems, 2021. URL https://arxiv. org/abs/2110.14168.

Justin Deschenaux and Caglar Gulcehre. Language Modeling with Hyperspherical Flows, 2026. URL https://arxiv.org/abs/2605.11125.

Jiatao Gu, James Bradbury, Caiming Xiong, Victor O. K. Li, and Richard Socher. Nonautoregressive neural machine translation. In International Conference on Learning Represen tations, 2018. URL https://arxiv.org/abs/1711.02281.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring Mathematical Problem Solving With the MATH Dataset. In NeurIPS Datasets and Benchmarks, 2021. URL https://arxiv.org/abs/2103.03874.

Fei Huang, Tianhua Tao, Hao Zhou, Lei Li, and Minlie Huang. On the Learning of Non-Autoregressive Transformers. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 9356–9376. PMLR, 2022. URL https://proceedings.mlr.press/v162/huang22k.html.

Byoungkwon Kim and Minhyuk Sung. Tensor-train joint modeling for few-step discrete diffusion, 2026. URL https://arxiv.org/abs/2607.03788.

Minseo Kim, Chenfeng Xu, Coleman Hooper, Harman Singh, Ben Athiwaratkun, Ce Zhang, Kurt Keutzer, and Amir Gholami. CDLM: Consistency diffusion language models for faster sampling. In Conference on Machine Learning and Systems, 2026. URL https://arxiv.org/abs/ 2511.19269.

Amos Lapidoth and Christoph Pfister. Two Measures of Dependence. Entropy, 21(8):778, 2019. doi: 10.3390/e21080778. URL https://www.isiweb.ee.ethz.ch/papers/docu/ alap-pfis-2019-2.pdf.

David Li, Nikita Gushchin, Dmitry Abulkhanov, Eric Moulines, Ivan Oseledets, Maxim Panov, and Alexander Korotin. IDLM: Inverse-distilled Diffusion Language Models. In International Conference on Machine Learning, 2026a. URL https://arxiv.org/abs/2602.19066.

Gaotang Li, Ruizhong Qiu, Xiusi Chen, Heng Ji, and Hanghang Tong. Beyond log likelihood: Probability-based objectives for supervised fine-tuning across the model capability continuum. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research. PMLR, 2026b. URL https://gaotangli.github.io/project\_page/Beyond-Log-Likelihood/ static/pdfs/beyond-log-likelihood.pdf.

Anji Liu, Oliver Broadrick, Mathias Niepert, and Guy Van den Broeck. Discrete copula diffusion. In International Conference on Learning Representations, 2025. URL https://arxiv.org/ abs/2410.01949.

Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and Yi Zhang. TinyGSM: Achieving > 80% on GSM8k with Small Language Models, 2023. URL https://arxiv.org/abs/2312.09241.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, 2024. URL https://arxiv.org/abs/2310.16834.

Thomas Minka. Divergence Measures and Message Passing. Technical Report MSR-TR-2005-173, Microsoft Research, 2005. URL https://tminka.github.io/papers/ message-passing/minka-divergence.pdf.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large Language Diffusion Models. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://arxiv.org/abs/2502. 09992.

Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and Effective Masked Diffusion Language Models. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://arxiv.org/abs/2406.07524.

Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, and Volodymyr Kuleshov. The Diffusion Duality. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 52584–52619, 2025. URL https://arxiv.org/abs/2506.10892.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. Simplified and generalized masked diffusion for discrete data. In Advances in Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/2406.04329.

Tyler Sypherd, Mario Diaz, John Kevin Cava, Gautam Dasarathy, Peter Kairouz, and Lalitha Sankar. A tunable loss function for robust classification: Calibration, landscape, and generalization. arXiv preprint arXiv:1906.02314, 2022. URL https://arxiv.org/abs/1906.02314v6. Version 6.

Sophia Tang and Shiyi Wang. Discrete Beckmann Transport Models for One-Step Language Modeling and Reasoning, 2026. URL https://arxiv.org/abs/2609.15903.

Martin J. Wainwright and Michael I. Jordan. Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning, 1(1–2):1–305, 2008. doi: 10.1561/ 2200000001. URL https://doi.org/10.1561/2200000001.

Chuling Wen, Weijie Liang, and Jian Lu. Conditional Total Correlation and the Serial Depth of Adaptive Parallel Sampling, 2026. URL https://arxiv.org/abs/2608.25505.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2505.22618.

Minkai Xu, Tomas Geffner, Karsten Kreis, Weili Nie, Yilun Xu, Jure Leskovec, Stefano Ermon, and Arash Vahdat. Energy-based diffusion language models for text generation. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2410. 21357.

Tunyu Zhang, Xinxi Zhang, Ligong Han, Haizhou Shi, Xiaoxiao He, Zhuowei Li, Hao Wang, Kai Xu, Akash Srivastava, Chengzhi Mao, Hao Wang, Vladimir Pavlovic, and Dimitris N. Metaxas. Few-step diffusion language models via trajectory self-distillation. arXiv preprint arXiv:2602.12262, 2026. URL https://arxiv.org/abs/2602.12262.

Zhilu Zhang and Mert R. Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. In Advances in Neural Information Processing Systems, volume 31, 2018. URL https://papers.neurips.cc/paper\_files/paper/2018/ hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html.

Yunxiao Zhao and Changxiao Cai. Adaptation to Intrinsic Dependence in Diffusion Language Models, 2026. URL https://arxiv.org/abs/2602.20126.

Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling. In International Conference on Learning Representations, 2025. URL https:// arxiv.org/abs/2409.02908.

## A RELATED WORK

Diffusion language models and decoding. Masked diffusion models learn token predictions under a corruption process (Sahoo et al., 2024; Shi et al., 2024; Nie et al., 2025). Confidence-based decoding controls how many tokens are revealed at each evaluation (Wu et al., 2026), while SDAR combines parallel prediction within blocks with autoregression across blocks (Arriola et al., 2025; Cheng et al., 2025). Our method, AlphaDLM, changes the training objective and can be used with either decoding structure.

Training for few-step generation. Several methods adapt diffusion language models for generation with fewer model evaluations. T3D (Zhang et al., 2026) combines trajectory self-distillation with a reverse-KL-inspired discriminative objective and analyzes how trajectory supervision reduces conditional token dependence. dParallel (Chen et al., 2026) combines trajectory supervision with entropy minimization on correctly predicted tokens to encourage earlier parallel commitments. CDLM (Kim et al., 2026) distills teacher trajectories into a block-causal student using teacher-distribution matching and consistency objectives. Our method shares the goal of improving generation under limited decoding budgets, but directly applies sequence-level alpha loss to training examples and their corruptions, without requiring teacher trajectories or distribution matching. Our analysis characterizes how this objective changes the joint target fitted by a factorized denoiser and establishes conditions under which its optimum preserves multiple valid completions while excluding invalid combinations.

Modeling token dependencies. Parallel prediction and its multimodality challenge predate diffusion language models, including non-autoregressive translation (Gu et al., 2018; Huang et al., 2022). For discrete diffusion, Discrete Copula Diffusion (Liu et al., 2025) supplements the denoiser with a generative model that supplies dependency information. EDLM (Xu et al., 2025) introduces a sequence-level energy correction. Kim & Sung (2026) represent the conditional clean distribution through low-rank tensor decompositions, while Bansal & Sanghavi (2025) train a lightweight sampler on top of a frozen diffusion model to approximate joint sampling. These approaches enrich the modeled dependencies or the sampling procedure. Our method uses the factorized denoiser and changes its training objective. Our analysis asks which joint target this restricted predictor fits and when independent predictions can exclude invalid combinations without concentrating on a single completion.

Evaluation and baselines. TinyGSM (Liu et al., 2023) is a shared setting for generating mathematical solutions as programs. S-FLM (Deschenaux & Gulcehre, 2026) compares hyperspherical flows with MDLM and DUO (Sahoo et al., 2025). FMLM+ (Agarwal et al., 2026) uses posterior refinement, DBTM (Tang & Wang, 2026) learns a discrete transport map, and IDLM (Li et al., 2026a) distills a pretrained diffusion teacher. Our comparison uses their respective generation procedures. Our AlphaDLM uses fixed-NFE confidence sampling with the token-reveal schedule of FMLM+, while its objective ablations hold the sampler fixed. Appendix C documents the baseline sources and sampling protocols.

Objectives and dependence. Generalized cross-entropy interpolates between cross-entropy and mean absolute error in classification (Zhang & Sabuncu, 2018). Li et al. (2026b) study probabilitybased objectives for autoregressive supervised fine-tuning, including the same alpha-loss family applied token-wise to individual token probabilities. They show that downweighting low-probability tokens can help when the pretrained model already has strong task-relevant capabilities, while cross-entropy performs better when these capabilities are weak. We apply alpha loss to the joint probability of masked tokens and analyze how factorization changes the distribution that minimizes this sequence-level objective. Huang et al. (2022) analyze dependence lost by non-autoregressive marginal fitting and interpret alternative objectives through proxy targets. Minka (2005) shows how the divergence shapes a factorized approximation, and Lapidoth & Pfister (2019) establish variational identities for product-distribution optimization. Our analysis connects this viewpoint to denoising objectives, characterizing the selected joint target, distinguishing sequence-level and token-wise fitting, and establishing conditions under which a factorized predictor assigns probability only to valid completions. Recent analyses relate unmasking schedules and parallel-sampling error to total correlation (Zhao & Cai, 2026; Wen et al., 2026). Our characterization concerns the joint target selected by training. Token-wise alpha loss and entropy-regularized SFT provide relevant empirical controls because both also change prediction confidence.

## B POPULATION OPTIMUM OF SEQUENCE-LEVEL ALPHA LOSS

We explain why alpha loss fits a lower-temperature target when the predictor is unrestricted, and what changes when it predicts tokens independently. The key step is to separate two questions: which joint distribution to fit, and how to predict its individual tokens. Appendix B.1 proves Proposition 1 directly from the expected alpha loss, using an elementary argument illustrated on the twotoken example.

Throughout, we fix the visible context. A sequence $y = ( y _ { 1 } , \dots , y _ { m } )$ fills the m masked positions, q is its true conditional distribution, and p is the predictor. The space of sequences is finite. We optimize over distributions directly, without neural-network constraints, and initially assume $0 ~ <$ $\alpha < 1$ . All logarithms are natural.

The expected loss depends on p only through the following weighted sum:

$$
\mathbb { E } _ { y \sim q } [ \ell _ { \alpha } ^ { \mathrm { s e q } } ( p ; y ) ] = \frac { 1 - F _ { \alpha } ( p ) } { \alpha } , \qquad F _ { \alpha } ( p ) = \sum _ { y } q ( y ) p ( y ) ^ { \alpha } .\tag{14}
$$

Thus minimizing the loss means maximizing $F _ { \alpha }$ . Taking its logarithm also leaves the maximizing predictor unchanged.

## Without factorization: the optimum is the lower-temperature target.

Derivation of Equation (9). We first allow p to be any joint distribution. At an optimum, moving a small amount of probability from one sequence to another cannot improve $F _ { \alpha }$ . The gain from adding probability must therefore be the same for every sequence with $q ( y ) > 0 ;$

$$
\frac { \partial F _ { \alpha } } { \partial p ( y ) } = \alpha q ( y ) p ( y ) ^ { \alpha - 1 } = \lambda \quad \Longrightarrow \quad p ( y ) \propto q ( y ) ^ { 1 / ( 1 - \alpha ) } .\tag{15}
$$

Here λ is the common gain. Normalizing the probabilities gives exactly $p = q _ { T }$ , with $T = 1 - \alpha$ as defined in Equation 8.

Two facts ensure that this calculation finds the unique global optimum. First, an optimum puts no probability where $q = 0$ , since moving that probability to a sequence with $q > 0$ increases $F _ { \alpha }$ . It puts positive probability on every sequence with $q > 0 ,$ , since the gain in Equation 15 grows without bound as its predicted probability approaches zero. Second, $x ^ { \alpha }$ is strictly concave for $0 < \alpha < 1$ The weighted sum $F _ { \alpha }$ is therefore strictly concave on this support, so the stationary solution is the unique maximum. □

This is the optimum in Sypherd et al. (2022, Proposition 1). Their parameter $\beta = 1 / ( 1 - \alpha )$ gives $1 - 1 / \beta = \bar { \alpha }$ and $\beta / ( \beta - \mathrm { \bar { 1 } } ) = 1 / \alpha$ , recovering both our loss and this optimum.

## With factorization: the optimum fits the marginals of a selected target.

Derivation ofEquation (10). Now $\begin{array} { r } { p ( y ) = \prod _ { i } p _ { i } ( y _ { i } ) } \end{array}$ . To fit the positions separately, we rewrite the objective using log-probabilities, which turn a product into a sum. An auxiliary joint distribution π will supply the weights for this fit. It is used only in the proof. The following three steps specialize the product-distribution identity of Lapidoth & Pfister (2019) to m positions.

1. The auxiliary distribution gives an equivalent objective. For a fixed predictor with $F _ { \alpha } ( p ) > 0$ normalize the terms in its objective to obtain a distribution:

$$
\pi _ { p } ( y ) = \frac { q ( y ) p ( y ) ^ { \alpha } } { F _ { \alpha } ( p ) } .
$$

For any joint distribution π supported where $q > 0$ , expanding the definition of KL divergence gives

$$
\alpha \mathbb { E } _ { y \sim \pi } \log p ( y ) - D _ { \mathrm { K L } } ( \pi \Vert \boldsymbol { q } ) = \log F _ { \alpha } ( p ) - D _ { \mathrm { K L } } ( \pi \Vert \pi _ { p } ) .\tag{16}
$$

KL divergence is nonnegative and is zero exactly when its two distributions agree. Thus the lefthand side is at most log $\bar { F } _ { \alpha } ( p )$ , with equality at $\pi = \pi _ { p }$ . Maximizing it over π recovers the original

objective exactly. This is the finite-distribution form of the Gibbs variational identity (Wainwright & Jordan, 2008, Theorem 3.4).

2. The best token predictions are the target marginals. We can now maximize over $p$ and π together. Fix π first. Factorization makes the log-probability a sum over positions:

$$
\mathbb { E } _ { y \sim \pi } \log p ( y ) = \sum _ { i } \sum _ { y _ { i } } \pi _ { i } ( y _ { i } ) \log p _ { i } ( y _ { i } ) = - \sum _ { i } H ( \pi _ { i } ) - \sum _ { i } D _ { \mathrm { K L } } ( \pi _ { i } | | p _ { i } ) .\tag{17}
$$

Here $\pi _ { i }$ is the token marginal and $\begin{array} { r } { H ( r ) = - \sum _ { x } r ( x ) } \end{array}$ log r(x) measures uncertainty in a distribution $^ { r } \cdot$ Each position is an ordinary cross-entropy fitting problem, solved by $p _ { i } = \pi _ { i }$ . Substituting these optimal token distributions leaves only the choice of π:

$$
\operatorname* { m a x } _ { p = \prod _ { i } p _ { i } } \log F _ { \alpha } ( p ) = - \operatorname* { m i n } _ { \pi } J ( \pi ) , \qquad J ( \pi ) : = D _ { \mathrm { K L } } ( \pi \| q ) + \alpha \sum _ { i } H ( \pi _ { i } ) .\tag{18}
$$

Thus the optimal denoiser fits the marginals of a distribution minimizing $J .$

3. The selected target balances fit and token dependence. The sum of token entropies equals the joint entropy plus total correlation:

$$
\sum _ { i } { H ( \pi _ { i } ) } = H ( \pi ) + \mathrm { T C } ( \pi ) .
$$

Total correlation is the dependence measure defined in Section 3.2. To express J using the lowertemperature target, write $\overset { \cdot } { q } _ { T } ( y ) = q ( y ) ^ { 1 / ( 1 - \alpha ) } / Z _ { T }$ , where $\begin{array} { r } { Z _ { T } = \sum _ { y } q ( y ) ^ { 1 / ( 1 - \alpha ) } } \end{array}$ . Using log $q =$ $( 1 - \alpha ) ( \log q _ { T } + \log Z _ { T } )$ and $D _ { \mathrm { K L } } ( \pi \lVert q ) = - H ( \pi ) - \mathbb { E } _ { \pi }$ log q gives

$$
J ( \pi ) = ( 1 - \alpha ) D _ { \mathrm { K L } } ( \pi \| q _ { T } ) + \alpha \mathrm { T C } ( \pi ) - ( 1 - \alpha ) \log Z _ { T } .\tag{19}
$$

The last term is constant in π. Removing it and dividing by $1 - \alpha > 0$ gives the claimed characterization:

$$
\pi _ { \alpha } ^ { * } \in \arg \operatorname* { m i n } _ { \pi } \left\{ D _ { \mathrm { K L } } ( \pi \| q _ { T } ) + \frac { \alpha } { 1 - \alpha } \mathrm { T C } ( \pi ) \right\} , \qquad p _ { \alpha } ^ { * , \mathrm { f a c t } } = \prod _ { i } \pi _ { \alpha , i } ^ { * } .
$$

Both optimization steps are exact. Every minimizing $\pi _ { \alpha } ^ { * }$ gives an optimal predictor through its marginals. Conversely, any optimal predictor $p ^ { * }$ paired with $\pi _ { p ^ { * } }$ attains the joint maximum in Steps 1–2. It must therefore satisfy $p _ { i } ^ { * } = \pi _ { p ^ { * } , i }$ , and $\pi _ { p ^ { * } }$ must minimize $J .$ . This proves the statement for all global optima. □

Zero probabilities. We use 0 log $0 = 0$ and set KL divergence to infinity when its first argument assigns probability outside the support of its second. In Equation $^ { 1 6 , }$ a distribution π assigning mass where $p = 0$ gives value −∞ on the left and cannot maximize it. $\mathbf { A }$ predictor with $F _ { \alpha } ( p ) = 0$ cannot be optimal, since predicting any sequence with $q > 0$ gives a positive value. Thus zeros require no positivity assumption on the target or the optimal predictor.

Token-wise loss and the endpoints. For token-wise loss, the expected objective is a sum of separate objectives for individual positions. Applying the unrestricted calculation to each marginal $q _ { i }$ gives Equation 12. If $q$ already factorizes, so does $q _ { T }$ . Taking $\pi = q _ { T }$ then makes both terms in Equation 10 zero, so sequence-level and token-wise loss have the same optimum.

At $\alpha = 0$ , both losses for a factorized predictor become the sum of token cross-entropies, whose optimal factors are the marginals of $q .$ For $\alpha \geq 1$ , use $p ( y ) ^ { \alpha } \leq p ( y )$ to obtain

$$
F _ { \alpha } ( p ) \leq \sum _ { y } q ( y ) p ( y ) \leq \operatorname* { m a x } _ { y } q ( y ) .
$$

Predicting a most likely sequence with probability one attains this bound and is possible with a factorized predictor. It is the unique optimum when the mode is unique. $\mathbf { A t } { \boldsymbol { \alpha } } = 1$ , any joint distribution supported on tied modes is also optimal if it belongs to the allowed predictor class. For $\alpha > 1$ , equality requires a point mass on one mode.

These statements optimize distributions at one fixed context. Taking a logarithm preserves this optimum, but does not generally preserve an objective averaged over contexts with shared model parameters. Appendix B.1 gives the additional support condition needed to exclude invalid combinations.

Numerical implementation. The mask-normalized sequence-level loss has a shared gradient weight determined by the joint probability of the original tokens at all masked positions:

$$
\nabla _ { \theta } \left( \frac { \ell _ { \alpha } ^ { \mathrm { s e q } } } { m } \right) = p _ { \theta } ( x _ { M } \mid z , c ) ^ { \alpha } \nabla _ { \theta } \ell _ { \mathrm { C E } } , \qquad \ell _ { \mathrm { C E } } = - \frac { 1 } { m } \log p _ { \theta } ( x _ { M } \mid z , c ) .\tag{20}
$$

All missing-token predictions share this weight, whereas token-wise alpha loss assigns a separate weight to each position. Under $\alpha _ { \mathrm { e f f } } = k / m$ , the shared weight is $\exp ( k s _ { \theta } )$ . Small target probabilities suppress this weight more strongly at large $k ,$ motivating a warm start. Averaging logprobabilities makes k comparable across mask counts, but does not remove this suppression.

The implemented loss evaluates $1 - \mathrm { e x p } ( k s _ { \theta } ) \mathrm { a s } - \mathrm { e x p m } 1 ( k s _ { \theta } )$ to avoid cancellation near zero.   
Examples without active positions contribute zero.

## B.1 GROUPS OF VALID COMPLETIONS AT INTERMEDIATE ALPHA

The example in Figure 3 has two groups of valid completions: $\{ \mathtt { I } \} \times \{ \mathtt { a m } \}$ and $\{ \mathrm { h e } , \mathrm { s h e } \} \times \{ \mathrm { i } \mathrm { s } \}$ Combining tokens within either group is safe, while mixing the groups can produce invalid answers. We prove directly that alpha loss can select one group while giving positive probability to every completion within it. The proof uses only probabilities, powers, and the arithmetic–geometric mean inequality.

Selection of a single group was established for two tokens with one completion per group by Minka (2005, Section 3.4) and Lapidoth & Pfister (2019, Lemma 32). Here each group may contain multiple completions, with arbitrary dependencies in their target probabilities.

The support condition. For position i and group $^ { g , }$ let $A _ { i g }$ be the allowed token set. The valid sequences form G nonempty groups

$$
\{ y : q ( y ) > 0 \} = \bigcup _ { g = 1 } ^ { G } S _ { g } , \qquad S _ { g } = A _ { 1 g } \times \cdot \cdot \cdot \times A _ { m g } , \qquad A _ { i g } \cap A _ { i g ^ { \prime } } = \emptyset \quad ( g \neq g ^ { \prime } ) .\tag{21}
$$

Thus every combination within a group is valid, and a token at any one position identifies its group.   
The index g refers to a group of completions, not a decoding block.

Proposition 1 (Multiple valid completions, restated). Consider $m \geq 2$ masked tokens, and suppose the valid completions split into groups such that any within-group combination of tokens is valid, and different groups use disjoint token sets at every position. For $1 / m < \alpha < 1$ , every globally optimal factorized predictor selects one group and assigns positive probability to all completions in that group.

Setting and running example. We fix the visible context and write $\begin{array} { r } { p ( y ) = \prod _ { i } p _ { i } ( y _ { i } ) } \end{array}$ . By Equation 14, minimizing expected alpha loss is equivalent to maximizing

$$
F _ { \alpha } ( p ) = \sum _ { y } q ( y ) p ( y ) ^ { \alpha } .
$$

The five steps below first show that an optimum selects one group, then that it covers the whole group. The figures use the same target as Figures 2 and 3, with $\alpha = 0 . 5 5$ strictly inside the proposition’s range for $m = 2$ . The example at the boundary $\alpha = 1 / 2$ is treated separately below.

Figure 7 starts from the cross-entropy optimum, whose factors are the marginals of $q .$ It assigns 0.48 probability to invalid completions. We will compare this predictor with the predictors obtained by restricting it to either valid group.

Conditioning on a group. Write $p _ { i } ( A _ { i g } )$ for the probability that position i chooses a token from group g. Independence gives

$$
p ( { \cal S } _ { g } ) = \prod _ { i = 1 } ^ { m } p _ { i } ( A _ { i g } ) , \qquad \sum _ { g } p _ { i } ( A _ { i g } ) \le 1 .
$$

![](images/3a83ba191ec2d60e794f09c4fd75bef55685900050d95c22d5c32d3b0936755f.jpg)  
Figure 7: Valid groups and a predictor that mixes them. The target is the two-token example of Figure 3. Rows are the first token and columns the second, as in that figure. Green cells are valid and red cells are invalid. In (b), row heights are $p _ { 1 }$ and column widths are $p _ { 2 }$ , so cell areas equal joint probabilities. The two valid groups receive total probability $0 . 1 6 + 0 . { \overset { } { 3 } } 6 = 0 . 5 2$ . The remaining 0.48 falls on invalid combinations.

The inequality allows tokens outside all valid groups. If $p ( S _ { g } ) > 0 .$ , conditioning on $S _ { g }$ gives another factorized predictor, denoted $p ^ { ( g ) }$ , with factors $p _ { i } ^ { ( g ) } = p _ { i } ( { \bf \cdot } | A _ { i g } )$ . It gives probability only to sequences in $S _ { g }$

Proof of Proposition 1. Step 1. Split the objective over groups. The target is zero outside the valid groups. Within a group of positive probability, $p ( y ) = p ( S _ { g } ) p ^ { ( g ) } ( y )$ . Substitution into the objective gives

$$
F _ { \alpha } ( p ) = \sum _ { g : p ( S _ { g } ) > 0 } p ( S _ { g } ) ^ { \alpha } F _ { \alpha } ( p ^ { ( g ) } ) .\tag{22}
$$

Groups with $p ( S _ { g } ) = 0$ contribute zero. Every displayed $F _ { \alpha } ( p ^ { ( g ) } )$ is positive, since $p ^ { ( g ) }$ gives positive probability to at least one valid completion. These values use the original q, without renormalizing it within a group.

Thus $F _ { \alpha } ( p )$ is a weighted sum of the values obtained by conditioning on individual groups. We next show that these weights sum to less than one unless p already selects one group.

Step 2. Splitting probability between groups reduces the total weight. For $\alpha > 1 / m$ and any group, raising a number in [0, 1] to a larger power makes it no larger. The arithmetic–geometric mean inequality then gives

$$
p ( S _ { g } ) ^ { \alpha } \leq p ( S _ { g } ) ^ { 1 / m } = \left( \prod _ { i = 1 } ^ { m } p _ { i } ( A _ { i g } ) \right) ^ { 1 / m } \leq \frac { 1 } { m } \sum _ { i = 1 } ^ { m } p _ { i } ( A _ { i g } ) .
$$

Summing over groups and using their disjoint token sets yields

$$
\sum _ { g } p ( S _ { g } ) ^ { \alpha } \leq \sum _ { g } p ( S _ { g } ) ^ { 1 / m } \leq { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } \sum _ { g } p _ { i } ( A _ { i g } ) \leq 1 .\tag{23}
$$

The first inequality is strict whenever any $p ( S _ { g } )$ lies strictly between zero and one. Consequently, the sum can equal one only when exactly one group has $p ( S _ { g } ) = 1$ . Figure 8 illustrates both the bound and the threshold $1 / m$

(a) Equal split  
![](images/b2c112ee9cff56d99b391ceb7e1673268523155ae05b8703fa0381d44c6f5fb7.jpg)

(b) Select group 2  
![](images/c9218f314086d1f1db9be0878564c4d4baf9e3f5fe422ccdeabe747b5b1966dd.jpg)

(c) Total group weight  
![](images/93bba5e1e810c31db3057c87f22c941a05c7c36a89b86e09000cc6455cd16384.jpg)  
Figure 8: Why the threshold is $1 / m .$ . For $m = 2$ , assigning group 1 probability s at both positions gives $\begin{array} { r } { W _ { \alpha } ( s ) \stackrel { \cdot } { = } \sum _ { a } p ( S _ { g } ) ^ { \alpha } = s ^ { \dot { 2 } \alpha } + ( 1 - s ) ^ { 2 \alpha } } \end{array}$ . This is the total weight in Equation 22, not the objective itself. It is below one for $0 < s < 1$ when α $> 1 / 2 .$ $\mathrm { A t } \alpha = 0 . 5 5$ , (a) gives W ≈ 0.933, while (b) gives $W = 1$

Step 3. Every optimum selects one group. Let p be an optimal factorized predictor. Its objective is positive, since predicting any valid completion with probability one gives $F _ { \alpha } = q ( y ) > 0$ . Therefore at least one group has positive probability. Let $F _ { \mathrm { m a x } }$ be the largest value of $F _ { \alpha } ( p ^ { ( g ) } )$ among these groups. Steps 1–2 imply

$$
F _ { \alpha } ( p ) \leq F _ { \operatorname* { m a x } } \sum _ { g } p ( S _ { g } ) ^ { \alpha } \leq F _ { \operatorname* { m a x } } .\tag{24}
$$

But $F _ { \mathrm { m a x } }$ is attained by one of the factorized predictors $p ^ { ( g ) }$ . Optimality of $p$ requires $F _ { \alpha } ( p ) \geq$ $F _ { \mathrm { m a x } } .$ , so equality must hold throughout. Since $\bar { F } _ { \mathrm { m a x } } > 0 .$ , Step 2 forces $p ( S _ { g } ) \stackrel { - } { = } 1$ for one group g. Every factor then satisfies $p _ { i } ( A _ { i g } ) = 1$ , because their product is one and none can exceed one.

This proves the first half of the proposition: the predictor assigns no probability outside one valid group. Figure 9 shows the improvement from conditioning a predictor that has not yet selected a group.

(a) Original p  
![](images/8b8922875d683ded3941f50214920c1f9afa0835280934c5e6cd65e44afbb9c7.jpg)  
F<sub>α</sub>(p) ≈ 0.382

(b) Condition on $S _ { 1 }$  
![](images/653b1511900bea3e8891b0dfc21e6a9fe6c0ade7f167113d7886a7f30482b9e4.jpg)  
F<sub>α</sub>(p<sup>(1)</sup>) = 0.400

(c) Condition on $S _ { 2 }$  
![](images/86201d0c41d79a30e5bc40a099742a8993a764e167de29795c528defef397f00.jpg)  
F<sub>α</sub>(p<sup>(2)</sup>) ≈ 0.415  
Figure 9: Conditioning improves the running predictor at $\alpha = 0 . 5 5$ . Both conditioned predictors outperform (a). The comparison in Step 3 gives $\begin{array} { r } { F _ { \alpha } ( p ) \approx 0 . 3 8 2 \leq F _ { \mathrm { m a x } } \sum _ { q } p ( S _ { g } ) ^ { \alpha } } \end{array}$ ≈ $0 . 3 8 8 <$ $F _ { \mathrm { m a x } } \approx 0 . 4 1 5$ . Conditioning preserves the relative token probabilities within each group. It does not yet optimize them.

Step 4. A token that can contribute cannot have probability zero. This step uses $0 < \alpha < 1$ and applies to any target $q .$ Hold all positions except i fixed and group the objective by the token value $v \in \mathcal V$ at that position:

$$
F _ { \alpha } ( p ) = \sum _ { v \in \mathcal { V } } a _ { i } ( v ) p _ { i } ( v ) ^ { \alpha } , \qquad a _ { i } ( v ) = \sum _ { y : y _ { i } = v } q ( y ) \prod _ { j \not = i } p _ { j } ( y _ { j } ) ^ { \alpha } .\tag{25}
$$

The coefficient $a _ { i } ( v )$ is positive exactly when this token forms a valid completion with some tokens that the other positions predict with positive probability.

Suppose $a _ { i } ( v ) > 0$ but $p _ { i } ( v ) = 0$ . Give this missing token probability ε, and multiply all existing probabilities at position i by $1 - \varepsilon$ , where $0 < \varepsilon < 1$ . The resulting predictor $\widetilde { p }$ is still factorized. Its objective changes by

$$
F _ { \alpha } ( \widetilde { p } ) - F _ { \alpha } ( p ) = a _ { i } ( v ) \varepsilon ^ { \alpha } - \left[ 1 - ( 1 - \varepsilon ) ^ { \alpha } \right] F _ { \alpha } ( p ) .
$$

For $\alpha < 1 , ( 1 - \varepsilon ) ^ { \alpha } \geq 1 - \varepsilon ,$ so the loss is at most $\varepsilon F _ { \alpha } ( p )$ . Hence

$$
F _ { \alpha } ( \widetilde { p } ) - F _ { \alpha } ( p ) \ge \varepsilon ^ { \alpha } \big [ a _ { i } ( v ) - \varepsilon ^ { 1 - \alpha } F _ { \alpha } ( p ) \big ] > 0\tag{26}
$$

for sufficiently small $\varepsilon ,$ because $\varepsilon ^ { 1 - \alpha }$ tends to zero. This contradicts optimality. Thus every token with $a _ { i } ( v ) > 0$ has $p _ { i } ( v ) > 0 \colon$ adding a missing valid possibility brings a gain of order $\varepsilon ^ { \alpha }$ , which outweighs a loss of order ε.

Step 5. Every completion in the selected group has positive probability. Take a token $v \in A _ { i g }$ in the group selected in Step 3. At each other position $j ,$ , choose a token $y _ { j } \in A _ { j g }$ with $p _ { j } ( y _ { j } ) > 0$ Such a token exists because $p _ { j } ( A _ { j g } ) = 1$ . With $y _ { i } = v _ { : }$ , these tokens form a valid completion by the support condition. Its contribution to $a _ { i } ( v )$ is positive, so Step 4 gives $p _ { i } ( v ) > 0$

This holds for every token at every position within the selected group. Their products are positive, so every completion in that group has positive probability. Together with Step 3, this proves the proposition. If the selected group contains several completions, all of them remain possible.

Figure 10 illustrates Steps 4–5. At $\alpha = 0 . 5 5$ , the best predictor within the second group gives probabilities approximately 0.679 and 0.321 to he is and she is, respectively. Its objective is approximately 0.417, above the value 0.400 for the only completion in the first group. By Step 3, it is therefore a global optimum.

(a) Add a missing completion  
![](images/7e49c3a1ee921095b1ad3e07b72289c32aa4b51159e0491aa8848a8d837234bc.jpg)

(b) Optimal predictor  
![](images/be249f66b763a0e165937421273bf68d001d88ec4f8db61540cdebeda228a7b9.jpg)  
F<sub>α</sub> ≈ 0.417 > 0.400  
Figure 10: Selecting a group preserves all of its completions. At $\alpha = 0 . 5 5 ,$ , start with a predictor that always generates $\mathrm { \Upsilon ~ h e ~ \ i \ : s }$ and give she probability ε at the first position. The objective becomes $0 . 3 5 ( \bar { 1 } - \bar { \varepsilon } ) ^ { 0 . 5 5 } + 0 . 2 5 \varepsilon ^ { 0 . 5 5 }$ , which increases for small positive ε. Its maximum assigns positive probability to both valid completions. This best predictor within $S _ { 2 }$ also outperforms the only predictor within $S _ { 1 }$

Where the assumptions enter. The Cartesian group structure makes conditioning preserve factorization. Disjoint token sets give the probability bound in Step 2. The condition $\alpha > 1 / $ m makes that bound strict whenever the predictor splits probability between groups. The condition $\alpha < 1$ makes a small addition to a missing token beneficial in Step 4. Finally, validity of every within-group combination makes every token eligible for that argument in Step 5. The requirement $m \geq 2$ makes the interval $1 / m < \alpha < 1$ nonempty.

What happens at the threshold? The strict lower bound on α is necessary for a guarantee covering all targets with this support. To compare groups, define their best achievable objective values

$$
C _ { g } ( \alpha ) = \operatorname* { m a x } _ { p = \prod _ { i } p _ { i } : p ( S _ { g } ) = 1 } F _ { \alpha } ( p ) , \qquad C ^ { \star } = \operatorname* { m a x } _ { g } C _ { g } ( \alpha ) .\tag{27}
$$

These maxima exist because the spaces of token probabilities are compact and $F _ { \alpha }$ is continuous.

At $\alpha = 1 / m$ . The arithmetic–geometric mean bound still gives $\begin{array} { r } { \sum _ { g } p ( S _ { g } ) ^ { 1 / m } \leq 1 } \end{array}$ . Equation 22 therefore implies

$$
F _ { \alpha } ( p ) \leq \sum _ { g } p ( S _ { g } ) ^ { 1 / m } C _ { g } ( \alpha ) \leq C ^ { \star } .
$$

A best single-group predictor attains $C ^ { \star }$ . If one group is uniquely best, positive probability on any other group makes the second inequality strict. Equality then forces $p ( \bar { S } _ { g } ) = 1$ for the best group, and Steps 4–5 ensure that all its completions have positive probability.

If two groups tie, take optimal predictors $p ^ { \prime }$ and $p ^ { \prime \prime }$ within them and mix their factors at every position: $p _ { i } \bar { = } ( 1 - s ) p _ { i } ^ { \prime } \bar { + } s p _ { i } ^ { \prime \prime }$ for $0 < s < 1$ . This is a product of mixed token distributions. Since $m \alpha = 1$ , its objective is

$$
F _ { \alpha } ( p ) = ( 1 - s ) ^ { m \alpha } C ^ { \star } + s ^ { m \alpha } C ^ { \star } = C ^ { \star } .
$$

It is optimal but also generates invalid cross-group combinations. For example, with $q ( 0 0 ) ~ =$ $q ( 1 1 ) \stackrel { - } { = } 1 / 2$ and $\alpha = 1 / 2$ , independent uniform factors are optimal while assigning half their probability to invalid pairs.

Below $1 / m$ . The group decomposition also gives the exact optimal allocation across groups. Let $\begin{array} { r } { s _ { g } = \frac { 1 } { m } \sum _ { i } p _ { i } ( A _ { i g } ) } \end{array}$ be the mean probability assigned to group g across positions. By arithmetic– geometric mean, $p ( S _ { g } ) \leq s _ { g } ^ { m }$ , so

$$
F _ { \alpha } ( p ) \leq \sum _ { g } C _ { g } ( \alpha ) s _ { g } ^ { m \alpha } , \qquad \sum _ { g } s _ { g } \leq 1 .
$$

Every allocation with $\textstyle \sum _ { q } s _ { g } = 1$ attains this bound if each position chooses group g with probability $s _ { g }$ and uses the corresponding factor of a best within-group predictor. All $C _ { g }$ are positive, so its maximum uses the full allocation $\textstyle \sum _ { q } s _ { g } = 1$ . It remains to maximize this weighted sum of powers. For $0 < m \alpha < 1$ , Equation 15 gives the unique optimal allocation

$$
p _ { i } ( A _ { i g } ) = s _ { g } ^ { * } = \frac { C _ { g } ( \alpha ) ^ { 1 / ( 1 - m \alpha ) } } { \sum _ { g ^ { \prime } } C _ { g ^ { \prime } } ( \alpha ) ^ { 1 / ( 1 - m \alpha ) } } > 0 \mathrm { f o r e v e r y p o s i t i o n } i .\tag{28}
$$

To attain the bound, equality in arithmetic–geometric mean requires all positions to use these same group probabilities. With at least two groups, every optimum therefore generates invalid cross-group combinations. At the threshold, ties can allow such combinations. Above it, Proposition 1 excludes them under its support condition.

The two-token example at $\alpha = 1 / 2$ . In Figure 3, the target probabilities of I am, he is, and she is are 0.4, 0.35, and 0.25. The first group has only one completion, so $C _ { 1 } = 0 . 4$ . In the second group, the second token is always is, leaving only the first token to fit. Equation 15 gives

$$
\begin{array} { r } { C _ { 2 } ( \alpha ) = \left( 0 . 3 5 ^ { 1 / ( 1 - \alpha ) } + 0 . 2 5 ^ { 1 / ( 1 - \alpha ) } \right) ^ { 1 - \alpha } . } \end{array}
$$

At $\alpha = 1 / 2 , C _ { 2 } = \sqrt { 0 . 1 8 5 } > 0 . 4 .$ The unique optimum therefore chooses the second group, with probabilities proportional $\tan { 0 . 3 5 ^ { 2 } }$ and $0 . 2 5 ^ { 2 }$ . These are $4 9 / 7 4$ for he is and $2 5 / 7 4$ for she is. This establishes the example at the boundary, which the strict inequality in Proposition 1 does not cover. The group values become equal at $\alpha _ { \star } \approx 0 . 6 1 6 2$ . Since $m \alpha _ { \star } > 1$ , either single-group predictor is then optimal, but mixtures between the groups are not.

Why the support condition matters. Consider instead two binary tokens with joint probabilities

$$
q = { \binom { 0 . 4 \quad 0 . 3 } { 0 . 3 \quad 0 } } , \qquad { \mathrm { r o w s ~ a n d ~ c o l u m n s ~ i n d e x e d ~ b y ~ } } 0 , 1 .
$$

The pairs 00, 01, and 10 are valid, while 11 is not. These valid pairs cannot be divided into the disjoint Cartesian groups required by the proposition.

For every $0 < \alpha < 1$ , an optimal factorized predictor assigns positive probability to all four pairs. To see this, use Step 4 of the proof above, which applies to any target. Token 0 has a positive coefficient at either position regardless of the other position’s distribution, because it forms a valid pair with both 0 and 1. Thus $p _ { 1 } ( 0 ) > 0$ and $p _ { 2 } ( 0 ) > 0$ . Token 1 then also has a positive coefficient at each position, since it forms a valid pair with the other position’s 0. Hence $p _ { 1 } ( 1 ) > 0$ and $p _ { 2 } ( 1 ) > 0$ giving positive probability to the invalid pair 11.

Factorization and increasing alpha alone therefore do not guarantee valid samples for $0 < \alpha < 1$ The support condition is essential. Proposition 1 concerns the population optimum at a fixed context, not convergence of neural-network training.

## C TINYGSM COMPARISON PROTOCOLS

We document the TinyGSM training and evaluation protocols and the sources of baseline results.   
Appendix C.1 then extends the fixed-budget comparisons to adaptive confidence sampling.

AlphaDLM (ours). The teaser uses one EMA checkpoint at 250k updates, with $k = 1 6$ after $k = 1$ pretraining to 200k. Evaluation uses 512-token sequences, the SmolLM-135M tokenizer, exact argmax token selection, and untempered confidence scores. Each round reveals the union of positions above confidence 0.999 and the highest-confidence $\lceil m / r \rceil$ remaining positions, where m is the current mask count and r is the number of rounds left. Revealed tokens remain fixed. This fixed-NFE confidence sampler adapts the commitment rule of FMLM+ (Agarwal et al., 2026) to an absorbing-mask denoiser with a fixed number of model evaluations.

Every example executes all R forwards, including rounds after all masks have been filled. Accuracy is the observed correct fraction over 1,319 test problems. These are single-checkpoint estimates, without repeated-training uncertainty.

Program verification. The shared GSM8K verifier extracts the first fenced code block when present, discards text before the first function definition, and attempts to remove trailing text that prevents parsing. It executes simple math problem() with a 5 s time limit and reads its numeric return value or, when unavailable, a numeric answer from its printed output. This value is compared with the number following #### in the reference answer. Integers must match exactly, while floating-point comparisons use absolute tolerance $1 0 ^ { - 3 }$ . Missing functions, parsing or execution failures, timeouts, and missing numeric answers count as incorrect.

Common-initialization objective comparisons. The $k = 1$ checkpoint at 200k updates initializes the sequence-level and token-wise continuations, which are evaluated at 250k. Table 2 specifies the shared training recipe. Sequence-level and token-wise objectives first normalize over the active positions of each example and then average over examples. Neither objective uses an additional $1 / t$ weight. Examples with no active positions contribute zero.

We tokenize each TinyGSM example as a beginning-of-sequence token, the question, the literal twocharacter separator $\backslash \mathrm { n } ,$ , the Python solution, and an end-of-sequence token. We exclude examples longer than 512 tokens and pad the remaining examples to this length without packing different examples together. The question and separator remain clean context. Response, end-of-sequence, and padding positions participate in corruption and training. A fixed 1% split is reserved for validation with split seed 42. The tokenizer is HuggingFaceTB/SmolLM-135M.

Training times are stratified using the implementation’s antithetic sampler for a uniform distribution on $[ \varepsilon , 1 ]$ , with $\varepsilon = 1 0 ^ { - 3 }$ . The probability of keeping a token unchanged follows the stabilized linear schedule $\bar { \alpha } _ { t } = \varepsilon + ( 1 - \varepsilon ) ( 1 - t )$ . It uses no adaptive schedule or importance sampling. The model does not receive a time-conditioning signal.

The one-evaluation token-wise control uses EMA weights, exact argmax tokens, and confidence threshold zero, revealing all missing tokens in a single forward. In the 32-evaluation ancestral comparison, $T = 0$ takes the argmax over clean-token predictions and still samples reveal positions from the ancestral transition. It is not an argmax over the complete reverse transition including the mask state.

The fixed-budget objective comparison evaluates each checkpoint at 1, 2, 4, 8, and 16 forwards using the same sampling protocol. The confidence threshold is 0.999, tokens are exact argmax predictions, and all models use four GPUs with batch size 16 per GPU, EMA weights, and seed 1. Every example executes the prescribed number of forwards. Table 3 reports all points.

Table 2: Shared TinyGSM training recipe. Sequence-level and token-wise continuations use the same architecture, data processing, and optimizer settings.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Denoiser</td><td>Bidirectional DiT with 12 blocks, 12 attention heads, hidden width 768, and conditioning width 128</td></tr><tr><td>Regularization</td><td>Dropout 0.1 and zero weight decay</td></tr><tr><td>Sequence length</td><td>512 tokens including prompt, response, and padding</td></tr><tr><td>Optimizer</td><td>AdamW, learning rate  $3 \times 1 0 ^ { - 4 } , ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 ) , \epsilon = 1 0 ^ { - 8 }$ </td></tr><tr><td>Learning-rate schedule</td><td>2,500 warmup updates followed by a constant rate</td></tr><tr><td>Optimization</td><td>Global batch size 512, gradient clipping 1.0, and training seed 1</td></tr><tr><td>Weight averaging</td><td>EMA decay 0.9999</td></tr><tr><td>Training duration</td><td>Common sequence-level k = 1 prefix through 200k updates, followed by 50k continuation updates</td></tr></table>

Table 3: GSM8K accuracy (%) ↑ with a shared initialization and fixed-NFE confidence sampling. All checkpoints use 250k total updates, including the common k = 1 prefix through 200k. Each value is an observed correct fraction over 1,319 examples.
<table><tr><td>Objective</td><td>1NFE</td><td>2NFE</td><td>4NFE</td><td>8NFE</td><td>16 NFE</td></tr><tr><td>Sequence k = 1</td><td>0.91</td><td>0.91</td><td>8.11</td><td>26.23</td><td>40.56</td></tr><tr><td>Sequence k = 2</td><td>0.99</td><td>1.29</td><td>10.61</td><td>28.58</td><td>43.06</td></tr><tr><td>Sequence k = 4</td><td>1.82</td><td>2.58</td><td>16.68</td><td>35.33</td><td>46.47</td></tr><tr><td>Sequence k = 8</td><td>2.73</td><td>7.73</td><td>28.89</td><td>43.97</td><td>49.20</td></tr><tr><td>Sequence k = 16</td><td>6.29</td><td>18.04</td><td>34.65</td><td>39.04</td><td>40.56</td></tr><tr><td>Token  $\alpha = 0 . 2 5$ </td><td>0.83</td><td>1.74</td><td>12.66</td><td>33.28</td><td>44.35</td></tr><tr><td>Token  $\alpha = 0 . 3 7 5$ </td><td>1.67</td><td>3.56</td><td>17.97</td><td>35.63</td><td>44.12</td></tr><tr><td>Token  $\alpha = 0 . 5$ </td><td>1.36</td><td>5.46</td><td>19.86</td><td>36.01</td><td>39.95</td></tr><tr><td>Token  $\alpha = 0 . 7 5$ </td><td>2.12</td><td>5.23</td><td>7.66</td><td>7.88</td><td>7.88</td></tr></table>

MDLM with fixed-NFE confidence sampling. We evaluate the released MDLM checkpoint from the S-FLM repository on the same 1,319 problems at $R \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 , 6 4 \}$ . We use the same fixed-NFE confidence sampling implementation and evaluation configurations as our method, including the tokenizer, confidence threshold, argmax token selection, and EMA weights. Each example’s recorded forward count is verified against R. The respective correct counts are 10, 9, 56, 265, 497, 579, and 536. This controls the sampler, while the training objectives and initialization histories differ.

Reevaluation of released checkpoints. We evaluate the released TinyGSM MDLM (Sahoo et al., 2024), DUO (Sahoo et al., 2025), and S-FLM (Deschenaux & Gulcehre, 2026) checkpoints with the authors’ sampling implementation. MDLM and DUO use ancestral sampling at $T \stackrel { - } { = } 0 . 1$ . Both S-FLM curves use the spherical backbone with the trained adaptive truncated schedule, with exact velocity at $T = 0 .$ 1 or top-1 velocity at $T = 1$ , followed by greedy final decoding. We use EMA weights, seed 1, four GPUs, and batch size 32 per GPU. The plot and table report integer correct counts divided by 1,319, rounded to one decimal in the table.

Published table values. FMLM+ uses greedy posterior refinement with threshold 0.999 from Table 8 of Agarwal et al. (2026). We show Init in the teaser and all three variants in Table 1. DBTM uses greedy refinement with threshold 0.8 from Table 8 of Tang & Wang (2026). For IDLM, we use the published values from Table 1 of Li et al. (2026a) at 32 and 64 NFE. We evaluate the released IDLM–MDLM and IDLM–DUO TinyGSM checkpoints at 4, 8, and 16 NFE using the S-FLM ancestral sampler at T = 1, EMA weights, seed 1, and batch size 32 on one GPU. We verify the exact forward count for all 1,319 examples. The correct counts are 5, 24, and 86 for IDLM–MDLM and 30, 93, and 154 for IDLM–DUO. As a reproduction check, our 32-NFE evaluations yield 12.51% and 17.06%, compared with the published 12.81% and 15.39%. These methods use their published training recipes. FMLM+ Init and the distillation variants incur teacher training in addition to their own training, so this comparison does not match total training compute.

For context, the autoregressive references in Deschenaux & Gulcehre (2026) reach 53.9% with sampling and 63.3% with greedy decoding.

Table 4: Ancestral accuracy (%) ↑ at 32 NFE. Checkpoints share the $k = 1$ initialization and 250k total updates. At $T = 0 ,$ clean-token predictions are argmax while reveal positions remain stochastic.
<table><tr><td>Training parameter</td><td>Ancestral  $( T = 1 )$ </td><td>Ancestral  $( T = 0 )$ </td></tr><tr><td>k=1</td><td>8.19</td><td>26.00</td></tr><tr><td>k=2</td><td>10.46</td><td>26.61</td></tr><tr><td>k=4</td><td>15.09</td><td>29.80</td></tr><tr><td>k=8</td><td>21.30</td><td>34.95</td></tr><tr><td>k=16</td><td>32.52</td><td>37.07</td></tr></table>

## C.1 ADAPTIVE CONFIDENCE SAMPLING

Here we use adaptive confidence sampling from Fast-dLLM (Wu et al., 2026) (Section 2.1) with argmax tokens and no prescribed NFE. We vary the confidence threshold, so each point reports the resulting average NFE (Figure 11, Table 5). At threshold 0.9, $k = 8$ reaches 49.43% at 17.83 NFE, while $k = 1$ reaches 43.06% at 57.59 NFE. The $k = 1 6$ run attains 40.03% at 5.70 NFE. Tokenwise $\alpha = 0 . 2 5$ reaches 47.31% at 32.00 NFE, so the sequence-level $k = 8$ point has 2.12 percentage points higher accuracy with 1.79 times fewer evaluations.

![](images/28e8457f60d5b1729cf7450971ece938957f2c51f8f62fa67c65fb2d8bbb60ec.jpg)

(b) Objective comparison  
![](images/8e53a660d52a0971bde66de31357f0706b528889de260cf815ba59ffa13a9b62.jpg)  
Figure 11: Accuracy–NFE trade-offs under adaptive confidence sampling. Left: AlphaDLM (ours) parameter sweep. Right: sequence-level and token-wise objectives. All checkpoints share the $k = 1$ initialization and 250k total updates. Points vary the confidence threshold. Higher accuracy and fewer evaluations are preferable.

Table 5: Adaptive confidence sampling from a common initialization. Each entry gives accuracy (%) ↑ / average NFE ↓ on all 1,319 examples. The checkpoints are evaluated at 250k updates after the common sequence-level $k = 1$ initialization at 200k.
<table><tr><td>Objective</td><td> $\tau = 0 . 6$ </td><td> $\tau = 0 . 7$ </td><td> $\tau = 0 . 8$ </td><td> $\tau = 0 . 9$ </td></tr><tr><td>Sequence k = 1</td><td>28.58 / 7.64</td><td>40.41 / 15.54</td><td>42.61 / 28.42</td><td>43.06 / 57.59</td></tr><tr><td>Sequence k = 2</td><td>26.76 / 6.94</td><td>40.41 / 13.09</td><td>44.43 / 24.12</td><td>45.87 / 49.54</td></tr><tr><td>Sequence k = 4</td><td>26.00 / 5.61</td><td>38.21 / 9.57</td><td>45.87 / 16.72</td><td>48.82 / 35.32</td></tr><tr><td>Sequence k = 8</td><td>24.72 / 4.06</td><td>35.33 / 5.69</td><td>42.15 / 8.83</td><td>49.43 / 17.83</td></tr><tr><td>Sequence k = 16</td><td>17.74 / 2.79</td><td>26.31 / 3.37</td><td>32.15 / 4.03</td><td>40.03 / 5.70</td></tr><tr><td>Token α = 0.25</td><td>18.20 / 5.21</td><td>32.52 / 7.82</td><td>42.30 / 12.86</td><td>47.31 / 32.00</td></tr><tr><td>Token α = 0.375</td><td>11.52 / 4.37</td><td>26.76 / 5.70</td><td>36.54 / 8.15</td><td>46.02 / 18.84</td></tr><tr><td>Token  $\alpha = 0 . 5$ </td><td>4.85 / 3.72</td><td>14.78 / 4.60</td><td>26.23 / 5.72</td><td>39.88 / 10.06</td></tr><tr><td>Token  $\alpha = 0 . 7 5$ </td><td>2.43 / 2.68</td><td>2.88 / 3.23</td><td>3.79 / 3.72</td><td>7.66 / 4.48</td></tr></table>

## D SDAR EXPERIMENTAL DETAILS

Data and training. We fine-tune JetLM/SDAR-1.7B-Chat on the same data for all objectives within each domain. For mathematics, we use 272,872 examples from the CoT subset of Nemotron-SFT-Math-v4.<sup>1</sup> We use assistant content, omit separate reasoning fields, remove lexical overlap candidates with GSM8K and MATH500, and keep complete examples with at most 256 prompt and 1,280 response tokens. For code, we use OpenCodeInstruct (Ahmad et al., 2025). For all setups, we use a batch size of 128 and a learning rate of $5 \times 1 0 ^ { - 6 }$ , and train for 4,000 steps. CE+CAP (Bie et al., 2025) uses a correctness-conditioned entropy regularizer of weight 0.5 to encourage confident correct predictions.

Block-level objective. Our method applies Equation 13 separately to each four-token response block. For the masked supervised positions $M _ { b }$ in block $b ,$ we use

$$
s _ { b } = \frac { 1 } { | M _ { b } | } \sum _ { j \in M _ { b } } \log p _ { \theta , j } ( y _ { j } \mid \mathrm { c o n t e x t } ) , \qquad \ell _ { b } ( k ) = \frac { 1 - \exp ( k s _ { b } ) } { k } .\tag{29}
$$

The $k = 0$ limit is mean cross-entropy within the block. A block with no active targets contributes zero. We sum block losses within each example, divide by a fixed dataset estimate of the mean number of valid response blocks, and average over examples. Thus, the nonlinear transformation couples targets within a block rather than across the entire response.

Confidence-aware parallel training. CAP adds an entropy penalty on correctly predicted masked targets. Let $C$ contain masked supervised positions whose logit argmax equals the target and let $r _ { j } = \mathrm { s o f t m a x } ( z _ { j } / 0 . 5 )$ , where $z _ { j }$ denotes the vocabulary logits. The objective is

$$
\mathcal { L } _ { \mathrm { C E + C A P } } = \mathcal { L } _ { \mathrm { C E } } + \frac { 0 . 5 } { | \boldsymbol { C } | } \sum _ { j \in \boldsymbol { C } } H ( r _ { j } ) .\tag{30}
$$

The entropy term is zero when $C$ is empty. The correctness gate is discrete, while gradients pass through the entropy of the selected distributions. The normalization uses the correct masked positions in each microbatch.

Evaluation. We evaluate accuracy on 1,319 GSM8K problems (Cobbe et al., 2021) and 500 MATH500 problems (Hendrycks et al., 2021), and pass@1 on 164 HumanEval-Instruct problems (Chen et al., 2021) and 427 MBPP-Instruct problems (Austin et al., 2021b). We use adaptive confidence sampling within four-token blocks, with confidence thresholds τ ∈ {0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99} for mathematics and $\tau \_ \in$ {0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99} for code.

![](images/c3c46acd89d891a6de1fcb176eea7321d0796d0031222185ba922deb8516a436.jpg)  
Figure 12: SDAR accuracy–TPF trade-offs. AlphaDLM (ours) uses $k = 0 . 8$ for mathematics and $k = 0 . 4$ for code. Points show means at each confidence threshold, with bands of ±1 sample standard deviation in accuracy across runs. Rings mark threshold 0.99. Axes match Figure 6 (a–d).

Candidate tokens are selected by argmax $( T = 0 )$ . At each iteration, the sampler accepts remaining positions whose confidence is strictly above the threshold. If none qualifies, it accepts the highestconfidence position. Accepted tokens are not remasked. Each block therefore requires at most four denoising evaluations. Generation uses prefix KV caching and response limits of 1,280 tokens for mathematics and 384 for code, stopping after completing the block containing EOS or the turn-end token.

Tokens per forward (TPF). For a benchmark with N examples, let $L _ { i }$ be the number of returned response tokens and $D _ { i }$ the number of denoising forwards used for example i. We compute

$$
\mathrm { T P F } = \frac { \sum _ { i = 1 } ^ { N } L _ { i } } { \sum _ { i = 1 } ^ { N } ( D _ { i } + 1 ) } ,\tag{31}
$$

where the additional forward accounts for prompt prefill once per example. The numerator excludes prompt tokens, the first EOS or stop token, and all subsequent tokens, and respects the responselength limit. Denoising forwards are counted separately for each active example, including all steps in the block containing its stop token. Additional forwards used only to update the KV cache are excluded. We aggregate token and forward counts over the benchmark before taking their ratio, then average the resulting TPF values across runs at each confidence threshold. With four-token blocks, TPF is at most four. Prompt prefill and partially filled final blocks reduce the measured value even when each block requires only one denoising step.

Aggregation. At each threshold, we average accuracy and TPF separately across 5 runs. Figure 12 shows the main-text means with ±1 sample standard deviation in accuracy.

Parameter sweep. Figure 13 compares $k \in \{ 0 , 0 . 4 , 0 . 8 , 1 . 2 , 1 . 6 , 2 . 0 \}$ on mathematics with a learning rate of $5 \times 1 0 ^ { - 6 }$ and a batch size of 128, using 4,000 training steps.

![](images/0359198e05519d2a40e5b0ae5da5ff19475f30002b97026d99ccbbc72992317e.jpg)  
Figure 13: SDAR parameter sweep for AlphaDLM (ours) and CE. Panels (e,f) use the same axes as Figure 6 (e,f). Bands show ±1 sample standard deviation in accuracy.

## E SENSITIVITY TO TRAINING INITIALIZATION

The main TinyGSM comparison starts from a sequence-level $k \ = \ 1$ checkpoint at 200k up dates. Here we repeat the objective comparison from a CE-trained checkpoint at the same update count. Both sets of continuations train for another 50k updates. We compare sequence-level $k \in \{ 2 , 4 , 8 , 1 6 \}$ and token-wise $\alpha \in \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 \}$ , using the same architecture, optimizer settings, and fixed-NFE confidence sampler. Each evaluation uses EMA weights, exact argmax token selection, confidence threshold 0.999, and exactly 1, 2, 4, 8, or 16 forwards on all $1 { , } 3 \bar { 1 } 9$ GSM8K problems. There is one training seed per recipe.

![](images/329ee86cb7ba818a365d5be26ffba860d0af57772ce660ac55872e19da991367.jpg)

![](images/99d4aecb1c07175737eb74226c7708b595a3b8a1b224ff0c251402972bf849b0.jpg)  
Figure 14: Objective comparisons under two initializations. Colors identify the continuation objective. Solid lines start from k = 1 and dashed lines from CE, both at 200k updates and evaluated at 250k. Only parameters available under both initializations are shown.

The qualitative advantage of sequence-level training persists under CE initialization (Figure 14). At four NFE, sequence-level $k = \bar { 1 } 6$ reaches 35.33%, compared with 19.86% for the strongest tested token-wise setting. At sixteen NFE, $k = 8$ reaches 47.61%, compared with 42.38% for the strongest tested token-wise setting. Among the tested sequence-level parameters, $k = 1 6$ is best at 1–4 NFE and $k = 8$ at 8–16 NFE under both initializations.

The $k = 1$ initialization generally yields higher accuracy for subsequent sequence-level $k = 2 , 4 , 8$ training. However, CE initialization improves $k = 1 6 \mathrm { a t } 4 \mathrm { - } 1 6 \mathrm { N F E }$ , by 0.68–1.22 percentage points. Token-wise differences also depend on the loss parameter and evaluation budget (Table 6). These single-seed comparisons support robustness of the objective comparison, rather than a uniform advantage of either initialization.

Table 6: GSM8K accuracy (%) ↑ under two initializations. Each paired row uses the same continuation objective and decoding protocol. The final row reports the CE training run continued to 250k, whose 200k checkpoint initializes the CE-start experiments. It is a reference, not a matched objective comparison with the $k = 1$ continuation.
<table><tr><td>Objective</td><td>Initialization</td><td>1 NFE</td><td>2</td><td>4</td><td>8</td><td>16</td></tr><tr><td>Sequence  $k = 2$ </td><td> $k = 1$ </td><td>0.99</td><td>1.29</td><td>10.61</td><td>28.58</td><td>43.06</td></tr><tr><td>Sequence k = 2</td><td>CE</td><td>0.99</td><td>1.14</td><td>9.40</td><td>25.93</td><td>41.17</td></tr><tr><td>Sequence k = 4</td><td> $k = 1$ </td><td>1.82</td><td>2.58</td><td>16.68</td><td>35.33</td><td>46.47</td></tr><tr><td>Sequence k = 4</td><td>CE</td><td>1.59</td><td>2.12</td><td>14.71</td><td>31.99</td><td>43.37</td></tr><tr><td>Sequence k = 8</td><td> $k = 1$ </td><td>2.73</td><td>7.73</td><td>28.89</td><td>43.97</td><td>49.20</td></tr><tr><td>Sequence k = 8</td><td>CE</td><td>2.27</td><td>5.91</td><td>25.02</td><td>40.49</td><td>47.61</td></tr><tr><td>Sequence k = 16</td><td> $k = 1$ </td><td>6.29</td><td>18.04</td><td>34.65</td><td>39.04</td><td>40.56</td></tr><tr><td>Sequence k = 16</td><td>CE</td><td>5.23</td><td>16.53</td><td>35.33</td><td>40.26</td><td>41.77</td></tr><tr><td>Token  $\alpha = 0 . 2 5$ </td><td> $k = 1$ </td><td>0.83</td><td>1.74</td><td>12.66</td><td>33.28</td><td>44.35</td></tr><tr><td>Token  $\alpha = 0 . 2 5$ </td><td>CE</td><td>0.61</td><td>1.14</td><td>10.99</td><td>29.26</td><td>42.38</td></tr><tr><td>Token α = 0.5</td><td> $k = 1$ </td><td>1.36</td><td>5.46</td><td>19.86</td><td>36.01</td><td>39.95</td></tr><tr><td>Token α = 0.5</td><td>CE</td><td>1.29</td><td>5.08</td><td>19.86</td><td>36.85</td><td>39.73</td></tr><tr><td>Token  $\alpha = 0 . 7 5$ </td><td> $k = 1$ </td><td>2.12</td><td>5.23</td><td>7.66</td><td>7.88</td><td>7.88</td></tr><tr><td>Token  $\alpha = 0 . 7 5$ </td><td>CE</td><td>1.52</td><td>4.85</td><td>7.13</td><td>7.35</td><td>7.35</td></tr><tr><td>CE</td><td>CE</td><td>0.45</td><td>0.53</td><td>5.69</td><td>18.50</td><td>36.32</td></tr></table>

This ablation covers fixed-budget decoding and the shared parameter grid only. The $k = 1$ -start results for token-wise $\alpha = 0 . 3 7 5$ , sequence-level $k = 3 2 , 6 4$ , ancestral decoding, and pass@K remain separate experiments.

## F MULTIPLE-ANSWER GENERATION

We study how the training objective affects the benefit of generating several answers to the same problem. We use the three sequence-level checkpoints with $k \in \{ 1 , 8 , 1 6 \}$ at 250k training updates, all continued from the common $k = 1$ checkpoint at 200k updates. For each of the 1,319 GSM8K test problems, we generate $n = 3 2$ answers with independently seeded ancestral sampling. We consider $R \in \{ 4 , 1 6 \}$ model evaluations per answer and temperatures $T \in \{ 0 , 1 \}$ , giving 12 configurations and 506,496 generated answers in total. $\mathbf { A } \mathbf { t } T = 0$ , the clean-token prediction is the argmax, while the ancestral reveal trajectory remains random. All runs use EMA weights, $\mathrm { t o p } { - } p = 1$ , the 512-token sequence limit, and the same program verifier as the single-answer evaluation.

Let $c _ { i }$ denote the number of correct answers among the 32 samples for problem i. We estimate the probability that at least one of $K$ answers is correct using the unbiased pass@K estimator of Chen et al. (2021),

$$
\widehat { \mathrm { p a s s @ { K } } } = \frac { 1 } { 1 3 1 9 } \sum _ { i = 1 } ^ { 1 3 1 9 } \left( 1 - \frac { { \binom { 3 2 - c _ { i } } { K } } } { { \binom { 3 2 } { K } } } \right) , \qquad K \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 \} ,\tag{32}
$$

where the numerator is zero when $K > 3 2 - c _ { i }$ . We include repeated answers in the estimator. This metric measures verifier-assisted coverage among K attempts, with a total generation budget of KR model evaluations per problem. The sampling repeats use one trained checkpoint for each value of k.

![](images/2db7929cc60f62670d40aad295ee78b8992e988f0bfbd6b789d376f81f1aacaa.jpg)

Figure 15 and Table 7 show that the preferred training parameter depends on the number of attempts. With $T = 0$ and $R = 1 6$ , increasing k from 1 to 16 improves pass@1 from 16.83% to 32.24%. At 32 attempts, $k = 8$ reaches 74.22%, compared with 72.71% for $k = 1 6$ and 68.39% for $k = 1$ . At $T = 1 , k = 1 6$ has the highest observed pass@K across the tested values of K at both per-answer budgets.

![](images/dc91d4a3a91242f0cb64c7a1f2c7ec3709b543c9149567d8483be08102456800.jpg)

(b) T = 1  
![](images/e7d7fd3425b397a0f559ece729ae7e36e5f06074c909758a5fd01f28a9f03665.jpg)  
Number of answers (K)  
(d) T = 1  
Figure 15: GSM8K pass@K with ancestral sampling. Top: coverage versus number of answers. Bottom: the same results versus total NFE, KR. Colors identify k, and line styles identify NFE per answer R. Estimates use 32 attempts per problem. Reveal positions remain random at $T = 0$

At each shared total budget (16, 32, 64, and 128 NFE), using sixteen evaluations per answer gives higher coverage than using four for every tested checkpoint and temperature.

Table 7: GSM8K pass@K (%) ↑ estimated from 32 answers per problem. T is the sampling temperature, R is NFE per answer, and k is the training parameter. Bold entries mark the highest value across training parameters for the same T, R, and K.
<table><tr><td>T</td><td>R</td><td>k</td><td>Pass@1↑</td><td>Pass@2↑</td><td>Pass@4↑</td><td>Pass@8↑</td><td>Pass@16↑</td><td>Pass@32↑</td></tr><tr><td>0</td><td>4</td><td>1</td><td>2.56</td><td>3.62</td><td>5.03</td><td>6.86</td><td>9.33</td><td>12.74</td></tr><tr><td>0</td><td>4</td><td>8</td><td>8.51</td><td>11.52</td><td>15.23</td><td>19.91</td><td>25.60</td><td>32.07</td></tr><tr><td>0</td><td>4</td><td>16</td><td>14.73</td><td>19.14</td><td>24.17</td><td>29.89</td><td>36.34</td><td>43.37</td></tr><tr><td>0</td><td>16</td><td>1</td><td>16.83</td><td>26.23</td><td>37.45</td><td>48.77</td><td>59.04</td><td>68.39</td></tr><tr><td></td><td>0 16</td><td>8</td><td>27.33</td><td>38.14</td><td>48.95</td><td>58.56</td><td>66.85</td><td>74.22</td></tr><tr><td>0</td><td>16</td><td>16</td><td>32.24</td><td>42.35</td><td>51.62</td><td>59.55</td><td>66.39</td><td>72.71</td></tr><tr><td>1</td><td>4</td><td>1</td><td>0.09</td><td>0.17</td><td>0.30</td><td>0.50</td><td>0.80</td><td>1.21</td></tr><tr><td>1</td><td>4</td><td>8</td><td>3.16</td><td>4.65</td><td>6.36</td><td>8.20</td><td>10.33</td><td>12.96</td></tr><tr><td>1</td><td>4</td><td>16</td><td>8.20</td><td>11.59</td><td>15.42</td><td>19.67</td><td>24.42</td><td>29.64</td></tr><tr><td>1</td><td>16</td><td>1</td><td>3.79</td><td>6.84</td><td>11.62</td><td>18.25</td><td>26.43</td><td>35.71</td></tr><tr><td>1</td><td>16</td><td>8</td><td>15.08</td><td>22.87</td><td>32.31</td><td>42.62</td><td>53.04</td><td>63.08</td></tr><tr><td>1</td><td>16</td><td>16</td><td>24.64</td><td>34.61</td><td>44.91</td><td>54.59</td><td>63.12</td><td>70.89</td></tr></table>