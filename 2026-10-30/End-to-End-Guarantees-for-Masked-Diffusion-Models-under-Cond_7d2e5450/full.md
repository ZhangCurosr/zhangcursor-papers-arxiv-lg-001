# End-to-End Guarantees for Masked Diffusion Models under Conditional Mixing

Fareed Sheriff Massachusetts Institute of Technology fareeds@mit.edu

## Abstract

Masked diffusion models train a denoiser on masked cross-entropy and generate by unmasking many positions in parallel. Existing MDM sampling bounds assume an exact denoiser or leave its error uncontrolled while training bounds cover only the masking distribution on which the model was trained. We present an end-to-end guarantee for a learned denoiser under conditional mixing (CM): stationary sources (e.g. floored order-r Markov and two-sided-Doeblin hidden-Markov sources) for which revealing distant visible tokens moves a masked token’s posterior on average exponentially little in the distance with correlation length τ<sub>c</sub>. We show the empirical risk minimizer trained under any masking distribution attains per-token excess $\varepsilon ^ { \prime }$ on that distribution from ε<sup>′−O(τc</sup> <sup>log</sup> <sup>|V|)</sup> training sequences, independent of sequence length N — one sample budget serves every decoding schedule when each schedule’s denoiser is trained under its own weighting. We further show that the exponent’s order $\Theta ( \tau _ { c } \log | \mathcal { V } | )$ is minimax-optimal up to constants; more efficient rates require added structure. With this, we bound the end-to-end KL as $\mathrm { K L } ( \pi \| p _ { \mathrm { a l g } } ) \le \bar { \varepsilon } + N \varepsilon ^ { \prime }$ under a dilated schedule of $T = \mathcal { O } ( \tau _ { c } \log ( N | \mathcal { V } | / \varepsilon ) )$ ) denoising steps versus Ω(N) steps for a fixed random order on an explicit CM source. Finally, we show that block-autoregressive samplers chain across blocks with no added error and that a Λ-looped order-r message-passing class on order-r sources achieves polynomial sample complexity, with approximation error improving per loop until saturation at $\Lambda \bar { r } \approx \tau _ { c } \log ( 1 / \bar { \varepsilon } ^ { \prime } )$ . Experiments on released and retrained models support our results’ qualitative predictions.

## 1 Introduction

Masked diffusion language models (MDMs) trade the autoregressive factorization for per-step conditional independence between tokens, which lets them generate many tokens in parallel. MDMs use an absorbing-state/masked forward process on tokens [Austin et al., 2021, Campbell et al., 2022, Shi et al., 2024, Sahoo et al., 2024] and generate by unmasking several positions per step. Discrete diffusion training theory and sampling theory have largely developed separately. On the sampling side, recent work bounds the convergence and step count of the reverse process [Zhang et al., 2025, Chen and Ying, 2025, Liang et al., 2025, 2026], gives information-theoretic accounts of the factorization error [Li and Cai, 2025, Lavenant and Zanella, 2025, Dmitriev et al., 2026], limits for parallel generation [Chen et al., 2026], and analyses of confidence-adaptive decoding [Cai and Li, 2026]. Every result assumes an exact denoiser or leaves the denoiser’s error at the masking patterns that decoding visits as a free parameter. On the training side, finite-sample bounds exist for generative masked language modeling [Li et al., 2024] and discrete diffusion score estimation [Srikanth et al., 2026, Cho and Wu, 2026]; these control the learned model at the masking distribution on which it was trained and compose it with a matched sampler (Gibbs-style chains or CTMC samplers). Without structural assumptions, their rates scale with the full state space $| \mathcal { V } | ^ { N }$ [Cho and Wu, 2026].

Composing training with the parallel-sampling decompositions above requires a bound covering the masking patterns of an arbitrary decoding strategy with a per-token sample budget independent of the strategy and $N ;$ no existing result provides one.

We provide such a bound for stationary data whose token-to-token influence on average decays exponentially with distance, a source family we call conditional mixing (Assumption 1), which covers finite-order Markov sources with a transition floor (e.g. smoothed n-grams) and hidden Markov sources whose hidden kernel satisfies a two-sided Doeblin condition. We show that under conditional mixing (CM),

• Training: the empirical risk minimizer trained under any masking distribution ω has pertoken excess $\varepsilon ^ { \prime }$ at $\omega$ with $( 1 / \varepsilon ^ { \prime } ) ^ { \mathcal { O } ( \tau _ { c } \log | \mathcal { V } | ) }$ training sequences, independent of sequence length. We further show this is minimax-optimal in the exponent’s order $\tau _ { c } \log | \nu |$ (correlation length times log-vocabulary size).

• End-to-end & dilated decoding: combining this with the training-sampling decomposition of Lavenant and Zanella [2025], we show $\mathrm { K L } ( \pi | | p _ { \mathrm { a l g } } ) \leq \mathcal { E } _ { \mathrm { f a c t } } ( \nu ) ^ { \bullet } + \breve { N } \varepsilon ^ { \prime }$ for any nonadaptive decoding strategy ν whose denoiser is trained under $\nu \mathrm { { s } }$ induced weighting. A dilated schedule with spacing $b = \mathcal { O } ( \tau _ { c } \log ( N | \mathcal { V } | / \varepsilon ) )$ ) has $\mathcal { E } _ { \mathrm { f a c t } } \leq \varepsilon$ in $T = b$ steps, giving $\varepsilon + N \varepsilon ^ { \prime } ;$ a fixed random order needs $\Omega ( N )$ steps on a two-state Markov chain.

• Architectures: block-autoregressive samplers chain across blocks with no added error, and on floored order-r sources a looped order-r message-passing class with loop count in $\mathcal { O } ( ( \tau _ { c } / r ) \log ( 1 / \varepsilon ^ { \prime } ) )$ ) achieves polynomial sample complexity $\tilde { O } ( | \mathcal { V } | ^ { r + 1 } / \varepsilon ^ { \prime 2 } )$

• Experiments: we probe the influence assumptions on a released MDM, block size on a synthetic source, and dilated versus random decoding on released and finetuned models (Section 7).

Proofs and experimental details are in the appendices.

## 2 Related work

Learning the denoiser. The learning results cited in Section 1 hold at the training masking distribution. Our bound holds at every masking distribution with one per-token budget that does not depend on N, provided the denoiser is trained under the decoding schedule’s weighting (Theorem 1). Its window argument extends short-memory prediction [Sharan et al., 2018] from one-sided to twosided conditionals under arbitrary masks. Its lower bound (Theorem 2) parallels minimax rates for predicting Markov chains [Hao et al., 2018, Han et al., 2023] and hidden Markov models [Han et al., 2024].

Composing learning with sampling. We use the decomposition of Lavenant and Zanella [2025], which splits the sampler’s KL into a learning term and the factorization error. Earlier end-to-end results pair the learned model with a sampler matched to its training distribution [Li et al., 2024, Cho and Wu, 2026]. Corollary 1 pairs it with any non-adaptive decoding strategy, including parallel ones, and with adaptive strategies under strong conditional mixing.

Parallel decoding. With exact conditionals, the KL of a deterministic adaptive decoding policy equals its accumulated conditional total correlation [Ben-Hamu et al., 2025], and order-r Markov chains admit exact schedules with $r ( \lceil \log _ { 2 } ( N / r ) \rceil + \bar { 1 } )$ steps [Wen et al., 2026]. For random walks on graphs, a bisection sampler is exact in logarithmically many steps [Bansal et al., 2026]. On order-r sources our dilated schedule uses $\mathcal { O } ( \tau _ { c } \log N )$ steps with $\tau _ { c } \leq r / ( 2 q _ { \mathrm { m i n } } ^ { r } )$ (Proposition 1). It needs only conditional mixing, which includes Doeblin hidden Markov sources, and it works with a learned denoiser. Under Dobrushin-type conditions, parallel Markov chain Monte Carlo samples Gibbs distributions in ${ \mathcal { O } } ( \log N )$ parallel rounds [Liu and Yin, 2022], and the LOCAL algorithm of Feng et al. [2017] needs an extra factor of the maximum degree.

Mixing conditions. Conditional mixing adapts strong spatial mixing [Weitz, 2006] to masked prediction and averages it over the source (Section 3). Our hidden Markov result (Proposition 2) extends filter stability under a mixing kernel [Le Gland and Oudjane, 2004, Atar and Zeitouni, 1997] to two-sided conditionals under arbitrary masks.

## 3 Masked diffusion and conditional mixing

Let V be a finite vocabulary, $\overline { { \mathcal { V } } } = \mathcal { V } \cup \{ [ \mathsf { m a s k } ] \}$ the vocabulary augmented with the mask token $[ \mathsf { m a s k } ] , z \subseteq [ N ]$ the set of visible (unmasked or previously predicted) positions, $M = [ N ] \backslash z$ the masked positions, and π a distribution on $\mathcal { V } ^ { N }$ from which training sequences are drawn i.i.d. We assume throughout this work that π is the length-N marginal of a stationary stochastic process.

Forward process and objective Masked or absorbing-state diffusion corrupts tokens independently and irreversibly: at time $i \in [ 0 , 1 ]$ each token survives with probability $\alpha _ { t }$ and is replaced by [mask] otherwise [Austin et al., 2021, Sahoo et al., 2024, Shi et al., 2024]. We analyze the unweighted masked cross-entropy

$$
\mathcal { L } _ { \mathrm { C E } } ( f ) = - \mathbb { E } _ { t \sim U [ 0 , 1 ] } \mathbb { E } _ { x , z } \left[ \frac { 1 } { \vert M \vert } \sum _ { i \in M } \log f ( x _ { i } \mid i , z , x _ { z } ) \right] ,\tag{1}
$$

where the denoiser $f$ maps each masked position and partially-masked sequence to a distribution over V, and the empty-mask event contributes zero. The denoiser needs no time input: its targets are the time-independent conditionals π $\cdot ( x _ { i } \in \cdot \mid x _ { z } )$ [Ou et al., 2025, Zheng et al., 2025].

Risk and weightings Denote the risk $R _ { \omega } ( f ) = \mathbb { E } _ { \omega } [ - \log f ( x _ { i } \mid i , z , x _ { z } ) ]$ ] where $( i , z , x )$ is drawn from a joint law ω (a weighting) with $x \sim \pi .$ , and excess risk $\mathcal { E } _ { \omega } ( f ) = R _ { \omega } ( \bar { f } ) - R _ { \omega } ( \pi )$ over the true conditionals. Call ω admissible if $x _ { i } \mid ( i , z , x _ { z } ) \sim \pi ( \cdot \mid x _ { z } )$ , so that

$$
\begin{array} { r } { \mathcal { E } _ { \omega } ( f ) = \mathbb { E } _ { \omega } \left[ \mathrm { K L } \left( \pi ( \cdot  { \left| \begin{array} { l } { x _ { z } } \end{array} \right| } | f ( \cdot  { \left| \begin{array} { l } { i , z , x _ { z } } \end{array} \right)  } \right] \geq 0 ; } \end{\right)array} \end{array}\tag{2}
$$

every masking distribution (where $( i , z )$ is independent of x) is admissible, and so is the weighting $\omega _ { \nu }$ induced by running any decoding strategy ν on $x \sim \pi$ under teacher forcing (Proposition 3).

Training as risk minimization Let the uniform masking distribution $\omega _ { \mathrm { u } }$ draw i uniformly and z as a uniform subset of $[ N ] \setminus \{ i \}$ of uniform size. Then with noising schedule $\alpha _ { t } = 1 - t$

$$
\mathcal { L } _ { \mathrm { C E } } ( f ) = \frac { N } { N + 1 } R _ { \omega _ { \mathrm { u } } } ( f )\tag{3}
$$

and training on masked CE [Sahoo et al., 2024] is empirical risk minimization under $\omega _ { { \mathrm { u } } } ,$ , the objectivelevel equivalence with any-order autoregression (Uria et al., 2014, Hoogeboom et al., 2022, Ou et al., 2025, Shi et al., 2024); other schedules reweight $\omega _ { \mathrm { u } }$

Sampler A monotone sampler (Algorithm 1 of Lavenant and Zanella, 2025) unmasks sets of positions over T steps, filling each commit set S conditionally independently given the visible tokens. The stepwise error splits exactly (cf. Ben-Hamu et al., 2025) as

$$
\mathrm { K L } \left( \pi ( x _ { S } \mid x _ { z } ) \Big | \Big | \prod _ { i \in S } f ( \cdot \mid i , z , x _ { z } ) \right) = \sum _ { i \in S } \mathrm { K L } \left( \pi ( x _ { i } \mid x _ { z } ) \| f ( \cdot \mid i , z , x _ { z } ) \right) + \mathrm { T C } _ { \pi } \left( x _ { S } \mid x _ { z } \right)\tag{4}
$$

where the second term is the total correlation of the true conditional, untouched by training. Existing theory bounds the total correlation summed over steps, thefactorization error, by global functionals (mutual-information couplings [Li and Cai, 2025], the information profile [Lavenant and Zanella, 2025], effective total correlation [Dmitriev et al., 2026]) that neither localize it nor address learning. Throughout, we bound the per-token excess risk by $\varepsilon ^ { \prime }$ and the factorization error by $\varepsilon ,$ so the sampler’s output law $p _ { \mathrm { a l g } }$ satisfies $\mathrm { K \bar { L } } ( \pi \| p _ { \mathrm { a l g } } ) \le \varepsilon + N \varepsilon ^ { \prime }$ [Lavenant and Zanella, 2025].

Conditional mixing To compose training and sampling guarantees, we need a source assumption that both limits effective model dimension (so the denoiser is learnable from finitely many samples) and bounds the total correlation among co-committed tokens (so the factorization error per sampling step is controlled). We adapt strong spatial mixing [Weitz, 2006], which Dobrushin’s uniqueness condition [Dobrushin, 1968] implies, to masked prediction. The averaged influence coefficient $\bar { \kappa } ( d )$ measures how far revealing visible tokens beyond distance d moves a masked token’s posterior on average over the source, and conditional mixing requires it to decay exponentially in d:

Assumption 1 (Conditional mixing). Define the averaged influence coefficient

$$
\bar { \kappa } ( d ) = \operatorname* { s u p } _ { \stackrel { i , z \not \ni i , } { A \subseteq z \backslash B ( i , d ) } } \mathbb { E } _ { x \sim \pi } \mathrm { T V } \left( \pi ( x _ { i } \in \cdot \mid x _ { z } ) , \pi ( x _ { i } \in \cdot \mid x _ { z \backslash A } ) \right)\tag{5}
$$

where $B ( i , d ) = \{ j : | j - i | \leq d \}$ . Distribution π satisfies conditional mixing (CM) with correlation length $\tau _ { c } > 0$ and prefactor $C _ { 0 } \geq 1$ if

$$
\bar { \kappa } ( d ) \leq C _ { 0 } e ^ { - d / \tau _ { c } } \mathrm { f o r } \mathrm { a l l } d \geq 0\tag{6}
$$

The worst-case form of $\bar { \kappa } ,$ the influence coefficient, compares pairs of visible configurations that agree near i:

$$
\kappa ( d ) = \operatorname* { s u p } _ { \stackrel { i , z \ni i , } { \varepsilon \sim _ { d } \xi ^ { \prime } } } \mathrm { T V } \left( \pi ( x _ { i } \in \cdot \mid x _ { z } = \xi ) , \pi ( x _ { i } \in \cdot \mid x _ { z } = \xi ^ { \prime } ) \right)\tag{7}
$$

where $\xi \sim _ { d } \xi ^ { \prime }$ means $( \xi , \xi ^ { \prime } ) \in \mathcal { V } ^ { z } \times \mathcal { V } ^ { z }$ agree on $z \cap B ( i , d )$ and both have positive probability under π.

Assumption 2 (Strong conditional mixing). π satisfies strong conditional mixing with correlation length $\tau _ { c } > 0$ and prefactor $C _ { 0 } \geq 1 \mathrm { i f } \kappa ( d ) \leq C _ { 0 } e ^ { - d / \tau _ { c } }$ for all $d \geq 0$

Conditional mixing is weaker than conditional independence: it permits dependence among tokens but bounds the sampler’s factorization error; tokens committed b apart carry zero factorization error whenever $\bar { \kappa } ( b - 1 ) = 0$ , by (17). Strong CM implies CM with the same constants, since $\bar { \kappa } \leq \kappa$ by convexity of total variation (10), but not conversely: CM tolerates configurations with large influence if they are rare under π. Strong CM (and thus CM) contains the classical autoregressive families with constants independent of sequence length (proofs in Section A.2):

Proposition 1 (Order-r Markov sources). Draw training sequencesfrom a stationary order-r Markov source on $\nu$ whose transition matrix Q has floor $q _ { \mathrm { m i n } } ~ = ~ \mathrm { m i n } _ { a \in { \mathcal V } , y \in { \mathcal V } ^ { r } } Q ( a ~ | ~ y ) ~ > ~ 0$ and let $\Delta _ { r } = 2 r \log ( 1 / q _ { \operatorname* { m i n } } )$ and $\vartheta _ { r } = \operatorname { t a n h } ( \Delta _ { r } / 4 ) = ( 1 - q _ { \mathrm { m i n } } ^ { r } ) / ( 1 + q _ { \mathrm { m i n } } ^ { \bar { r } } ) \big [$ . Then, the data distribution satisfies Assumption 2 with $\tau _ { c } = r / \log ( 1 / \vartheta _ { r } ) \leq r / ( 2 q _ { \operatorname* { m i n } } ^ { r } )$ and $C _ { 0 } = 2 \Delta _ { r } / \vartheta _ { r } ^ { 3 }$

Proposition 2 (Hidden Markov sources with a Doeblin kernel). Let $( h _ { t } ) _ { t \in \mathbb { Z } }$ be a stationary Markov chain onfinite set H whose kernel Γ satisfiesfor some $\gamma _ { \mathrm { D } } \in ( 0 , 1 )$ and some measure $\nu _ { \mathrm { D } }$ on H the two-sided Doeblin condition

$$
\gamma _ { \mathrm { { D } } } \nu _ { \mathrm { { D } } } ( h ^ { \prime } ) \leq \Gamma ( h , h ^ { \prime } ) \leq \gamma _ { \mathrm { { D } } } ^ { - 1 } \nu _ { \mathrm { { D } } } ( h ^ { \prime } ) \forall h , h ^ { \prime } \in \mathcal { H }\tag{8}
$$

and let the observations $x _ { t } \sim g ( \cdot \mid h _ { t } )$ be conditionally independent given $( h _ { t } ) _ { t \in \mathbb { Z } } f o r$ any emission kernel $g , \Delta _ { \Gamma } = 4 \log ( 1 / \gamma _ { \mathrm { D } } )$ , and $\vartheta _ { \Gamma } = \operatorname { t a n h } ( \Delta _ { \Gamma } / 4 ) \stackrel { \cdot } { = } ( 1 \stackrel { \cdot } { - } \gamma _ { \mathrm { { D } } } ^ { 2 } ) / ( \bar { 1 } + \gamma _ { \mathrm { { D } } } ^ { 2 } )$

Then, $( x _ { t } ) _ { t \in \mathbb { Z } }$ satisfies Assumption 2 with $\tau _ { c } = 1 / \log ( 1 / \vartheta _ { \Gamma } ) \leq 1 / ( 2 \gamma _ { \mathrm { D } } ^ { 2 } )$ and $C _ { 0 } = 2 \Delta _ { \Gamma } / \vartheta _ { \Gamma }$

Long-range parity constraints force $\kappa \equiv 1$ and $\bar { \kappa } \geq 1 / 2$ , as in known hard instances for parallel sampling and masked learning [Chen et al., 2026, Kim et al., 2025]. Conditional mixing is nonetheless a strong requirement: taking a single visible token in Assumption 1 bounds one token’s average influence on another with nothing between them visible so pairwise mutual information decays exponentially; in text, pairwise MI decays as a power law [Lin and Tegmark, 2017, Shen, 2019, Nakaishi et al., 2026], which CM excludes (Remark 1); with context visible, however, our probe of a trained MDM (Section 7) finds influence decaying exponentially on average but not in the worst case, which motivates CM’s averaged form over strong CM. We assume CM for most of our proofs and state where we assume strong CM.

## 4 Learning the denoiser

Training minimizes the masked cross-entropy (1) (risk at $\omega _ { \mathrm { u } } )$ , but decoding queries the denoiser at masking patterns a training-time analysis need not cover. The bound below is schedule-uniform in the sample budget: the same n suffices at every masking distribution $\omega ,$ with the ERM taken under that ω. It does not cover a model trained at one weighting and evaluated at another (the mask-training design question of Shih et al., 2022).

The denoising hypothesis class $\mathcal { F } _ { w }$ consists of denoisers that read only the w positions to each side of the masked target, outputting a distribution over V with every probability at least $p _ { \mathrm { m i n } }$ , keeping the log-loss below $\bar { \ell } _ { \mathrm { m a x } } \doteq \log ( \bar { 1 } / p _ { \mathrm { m i n } } )$ (Definition 1). Conditional mixing makes a window of width $w \asymp \tau _ { c } \log ( 1 / \varepsilon ^ { \prime } )$ sufficient $( \log | \overline { { \mathcal { V } } } |$ and log |V| agree to leading order).

Theorem 1 (Sample complexity of masked-diffusion training). Let π be stationary (Section 3), $N \geq 2 , \varepsilon ^ { \prime } \in \left( 0 , 1 \right]$ and $\delta \in ( 0 , 1 )$

(a) If Assumption 1 holds with correlation length $\tau _ { c }$ and prefactor $C _ { 0 } ,$ , then for $p _ { \operatorname* { m i n } } ~ =$ $\bar { \varepsilon } ^ { \prime } / ( 1 0 | \mathcal { V } | \log ( 1 0 | \mathcal { V } | / \varepsilon ^ { \prime } ) ) , w = \lceil \tau _ { c } \log ( 4 C _ { 0 } ( \bar { \ell } _ { \mathrm { m a x } } + 1 ) / \bar { \varepsilon } ^ { \prime } ) \rceil$ and every masking distribution $\omega ,$ , the ERM <sup>ˆ</sup>f over $\mathcal { F } _ { w }$ trained under ω satisfies $\mathcal { E } _ { \omega } ( \hat { f } ) \leq \varepsilon ^ { \prime } w . p . \geq 1 - \delta$ whenever

$$
n \geq \left( 1 / \varepsilon ^ { \prime } \right) ^ { 2 + 2 \tau _ { c } \log \vert \overline { { \mathcal { V } } } \vert + o ( 1 ) } \vert \mathcal { V } \vert \log \frac { 2 } { \delta } ,
$$

the $o ( 1 )$ taken as $\varepsilon ^ { \prime } \to 0$ at fixed $( \tau _ { c } , C _ { 0 } , | \mathcal { V } | )$

(b) IfAssumption 2 holds with $( \tau _ { c } , C _ { 0 } )$ , (a) holdsfor every admissible $\omega ,$ including the weightings induced by adaptive decoding strategies fixed before the training sample is drawn (Proposition 3).

Proof sketch. We can decompose the excess of $\hat { f }$ into an approximation error and estimation error. Specifically, the excess is at most the excess of the best denoiser in $\mathcal { F } _ { w }$ (the approximation error) plus the cost of choosing a denoiser from n samples (the estimation error). For the first, we use the true conditional given only visible tokens within distance w mixed with a uniform so every probability is at least $p _ { \mathrm { m i n } } ;$ then, conditional mixing (strong conditional mixing in (b)) bounds its expected TV from the true conditional by $C _ { 0 } e ^ { - w / \tau _ { c } }$ plus the mixing weight, and the floor $p _ { \mathrm { m i n } }$ turns a total variation t into a KL of at most $( \ell _ { \mathrm { m a x } } + 1 ) t$ . For the second, any element of $\mathcal { F } _ { w }$ is a distribution over V for each of the $| \overline { { \mathcal { V } } } | ^ { 2 w }$ possible window contents (a token or the mask at each of the 2w positions); Hoeffding with union bound over a grid of these $| \overline { { \mathcal { V } } } | ^ { 2 w } | \mathcal { V } |$ probabilities gives an error of order $\ell _ { \mathrm { m a x } } \sqrt { | \overline { { \mathcal { V } } } | ^ { 2 w } | \mathcal { V } | \log ( 1 / \varepsilon ^ { \prime } ) / n }$ . The stated w makes the first error at most $\varepsilon ^ { \prime } / 2$ and the stated n the second (Section A.4).

Window truncation is a common idea in one-sided prediction [Csiszár and Talata, 2006]; we extend it in Theorem 1 to two-sided conditionals under arbitrary masks.

Minimax optimality and complexity hierarchy The order of the exponent in Theorem 1 cannot be improved:

Theorem 2 (Minimax lower bound). Let $\tau _ { c } \geq 1 , C _ { 0 } \geq 1 6$ and $| \nu | \geq 2$ . Then, there is an explicit $\varepsilon _ { 0 } = \varepsilon _ { 0 } ( \tau _ { c } , C _ { 0 } ) > 0$ such that, for $\varepsilon ^ { \prime } \in \mathsf { \Gamma } ( 0 , \varepsilon _ { 0 } ]$ and d the largest integer that satisfies $1 6 0 d \exp ( 3 d / \tau _ { c } ) \sqrt { \varepsilon ^ { \prime } } \leq C _ { 0 } ,$ , there isfor every $N \geq 2 $ d afinitefamily Π ofstationary sources on $\mathcal { V } ^ { \mathbb { Z } }$ with uniform marginals each satisfying Assumption 2 with $( \bar { \tau _ { c } } , C _ { 0 } )$ such thatfor the weighting $\omega ^ { * }$ that masks thefinal token and reveals only the d tokens preceding it every map <sup>ˆ</sup>f from n unmasked training sequences to a predictor satisfies

$$
\operatorname* { m a x } _ { \pi \in \Pi } \mathbb { E } _ { \pi } \left[ \mathcal { E } _ { \omega ^ { * } } ( \hat { f } ) \right] \geq \varepsilon ^ { \prime } f o r \quad \quad n ( N - d ) < \frac { | \mathcal { V } | ^ { d - 1 } } { 3 0 0 \varepsilon ^ { \prime } } = \left( \frac { 1 } { \varepsilon ^ { \prime } } \right) ^ { 1 + ( 1 - o ( 1 ) ) \frac { \tau _ { c } } { 6 } \log | \mathcal { V } | } .
$$

Proof sketch. The sources are order-d Markov chains whose next-token distribution is uniform up to a perturbation of relative size $\gamma \asymp \sqrt { \varepsilon ^ { \prime } }$ with sign chosen separately for each of the $| \mathcal { V } | ^ { d - 1 }$ values of the preceding $d - 1$ tokens. This perturbation is sufficiently small that every such chain satisfies strong conditional mixing with $( \tau _ { c } , \bar { C } _ { 0 } )$ . Then, predicting the final token at $\omega ^ { * }$ with excess below $\varepsilon ^ { \prime }$ requires most of the signs and a sign can be learned only from tokens that follow its $d - 1$ preceding values all of which occurs $\mathbf { w . p . } \mid \mathcal { V } \mid ^ { - ( d - 1 ) }$ per token, and each sign needs about $1 / \gamma ^ { 2 } \asymp 1 \bar { / } \varepsilon ^ { \prime }$ such tokens; the sample therefore needs about $| \gamma | ^ { d - 1 } / \varepsilon ^ { \prime }$ tokens. For $N \in \Theta ( d )$ , this is $( 1 / \varepsilon ^ { \prime } ) ^ { 1 +  { \Omega } ( \tau _ { c } \log | \mathcal { V } | ) }$ training sequences, matching Theorem 1 in exponent order.

For text, the exponent is in the hundreds even at the local decay scale ${ \approx } 2 6$ with the GPT-2 vocabulary (log $| \overline { { \mathcal { V } } } | \approx 1 0 . 8 ) ;$ ; rates whose exponent does not grow with $\tau _ { c }$ log |V| need added structure. Under a conditional floor $q _ { c }$ on the masked conditionals (order-r sources give $q _ { c } \geq q _ { \operatorname* { m i n } } ^ { r + 1 } )$ ) and CM in mean square, the TV-to-KL conversion becomes quadratic and the exponent’s $\tau _ { c }$ term halves, yielding $2 + \tau _ { c } \log | \overline { { \mathcal { V } } } | + o ( 1 )$ (Corollary 5). For order-r Markov sources, the true conditional can be computed by iterating a transition kernel which gives $\tilde { O } ( | \mathcal { V } | ^ { r + 1 } / \varepsilon ^ { \prime 2 } )$ (Corollary 4).

Training objectives Unweighted masked CE and the ELBO both estimate the risk at $\omega _ { \mathrm { u } }$ (Section A.3); other noise schedules and time weightings change ω [Sahoo et al., 2026] and thus which decoding strategies are covered. Theorem 1 covers the per-sequence ERM on a single masked token and the unweighted masked CE minimizer, which averages over all masked positions of a sequence; both per-sequence losses are bounded by $\ell _ { \mathrm { m a x } }$

## 5 End-to-end guarantees

We compose Theorem 1 with the training-sampling decomposition of Lavenant and Zanella [2025]:

$$
\mathrm { K L } ( \pi \| p _ { \mathrm { a l g } } ) \leq { \mathcal E } _ { \mathrm { l e a r n } } + { \mathcal E } _ { \mathrm { f a c t } } , \qquad { \mathcal E } _ { \mathrm { f a c t } } = \mathbb E \sum _ { k \geq 1 } { \mathrm { T C } } _ { \pi } ( x _ { S _ { k } } \mid x _ { z _ { k } } )\tag{9}
$$

where $S _ { k }$ is the k-th commit set and $z _ { k } = S _ { 1 } \cup \dots \cup S _ { k - 1 }$ the visible set before it, $\mathcal { E } _ { \mathrm { l e a r n } }$ is the denoiser’s per-token excess at the weighting the strategy induces, accumulated over the N commits, and ${ \mathcal { E } } _ { \mathrm { f a c t } }$ is the information cost of committing tokens in parallel. For a strategy $\nu ,$ Theorem 1 applied to the ERM trained under $\omega _ { \nu }$ (Proposition 3) controls $\bar { \mathcal { E } _ { \mathrm { l e a r n } } } ;$ Corollary 1 states the composition and computes the constants for the dilated (residue-class) schedule, a constant-spacing relative of the coarse-to-fine dilated scheduler of Luxembourg et al. [2026]. Empirical schedule design pursues the same step-count reduction [Park et al., 2025, Azangulov et al., 2026], Peng et al. [2025] plan which tokens to update, and Zhao and Cai [2026] adapt step sizes to intrinsic dependence.

Corollary 1 (End-to-end guarantee). Let π be stationary and satisfy Assumption 1.

(i) Composition. Fix a non-adaptive decoding strategy ν (any ν under Assumption 2), chosen independently ofthe training sample, with induced weighting ω (Proposition 3). For <sup>ˆ</sup>f the ERM ofTheorem 1 trained under $\omega _ { \nu } ,$ running Algorithm 1 with $( \nu , \hat { f } )$ satisfies, w.p. at least $1 - \delta , \mathrm { K L } ( \pi \| p _ { \mathrm { a l g } } ) \leq { \mathcal { E } } _ { \mathrm { f a c t } } ( \nu ) + N \varepsilon ^ { \prime } .$

(ii) Dilated factorization. The dilated schedule that unmasks $\{ t , t + b , t + 2 b , \ldots \}$ at step t achieves $\mathcal { E } _ { \mathrm { f a c t } } \leq \varepsilon a t$ spacing $b = \mathcal { O } ( \tau _ { c } \log ( C _ { 0 } N | \mathcal { V } | / \varepsilon ) )$ in $T = b$ steps, so part (i) gives $\begin{array} { r } { \mathrm { K L } ( \pi \| p _ { \mathrm { a l g } } ) \le \varepsilon + N \varepsilon ^ { \prime } w . p . \ge 1 - \delta . } \end{array}$

Proofsketchfor (ii). Under a dilated schedule, tokens committed in the same step are at least b apart. By chain rule, the factorization error is a sum over the $N - b$ tokens not first in their step of the MI between the token and tokens committed before it in the same step, given those already visible. Each term is the average drop in the token’s conditional entropy when tokens at distance at least b are revealed — conditional mixing bounds this change of the conditional by $\bar { \kappa } ( b - 1 ) \leq C _ { 0 } e ^ { - ( b - 1 ) / \tau _ { c } }$ in expected total variation; applying an inequality between the difference in entropy and the total variation described in the full proof, we show the term is at most $\eta ( C _ { 0 } e ^ { - ( b - 1 ) / \tau _ { c } } )$ for some concave η we construct. The sum $( \bar { N } - b ) \eta ( C _ { 0 } e ^ { - ( b - 1 ) / \tau _ { c } } )$ is then at most ε for b ≈ $\tau _ { c } \log ( C _ { 0 } N | \mathcal { V } | / \varepsilon )$ (Section A.5).

Theorem 1(a) and Corollaries 1 and 3 use Assumption 1 only through two schedule-averaged quantities: the truncation error $\bar { \kappa } _ { \omega } ( w )$ at the training weighting (Lemma 3) and the influence $\bar { \kappa } _ { \nu } ( b - 1 )$ between tokens committed in the same dilated step (18). These results hold whenever the two averages obey the bound of Assumption 1, which is the averaged behavior our measurements probe (Section 7). Remark 1 (Limits of averaging). At spacing b the first dilated step commits $\{ 1 , 1 + b , \dotsc \}$ with nothing visible, so by the chain rule and stationarity $\mathcal { E } _ { \mathrm { f a c t } } \geq \left( \lceil N / b \rceil - \mathrm { \bar { 1 } } \right) I ( x _ { 0 } ; x _ { b } )$ . If $I ( x _ { 0 } ; x _ { b } ) \ge c b ^ { - \alpha }$ , as reported for text, then $\mathcal { E } _ { \mathrm { f a c t } } \leq \varepsilon$ forces $b ^ { 1 + \alpha } \geq c N / ( \ddot { 2 } \varepsilon )$ for $N \geq 2 b !$ polynomially many steps, and Assumption 1 fails however small the influence is with context.

Random-order separation Corollary 1(ii) needs $\mathcal { O } ( \tau _ { c } \log N )$ dilated steps; a fixed random order cannot match this on an explicit CM source:

Corollary 2 (Random-order separation). Let $\pi _ { q }$ be the stationary two-state Markov chain with flip probability $q \in ( 0 , \frac { 1 } { 2 } ) ( \tau _ { c } \leq 1 / ( 2 q )$ by Proposition 1). A random-order schedule draws its step sizes independently of x and commits positions in a uniformly random order $\sigma ;$ both are drawn once and then fixed, so each run is a deterministic schedule with error $\mathrm { K L } ( \pi \| p ^ { \sigma } )$ . With $c _ { q } = 8 q ^ { 4 } ( 1 - q ) ^ { 4 } ( 1 - 2 q ) ^ { 2 }$ , for all $N \geq 2$ and $\varepsilon > 0 ,$ , ifE KL $_ { i } ( \pi \| p ^ { \sigma } ) \leq \varepsilon$ (over σ and the sizes) with the exact conditionals, the expected number of steps satisfies

$$
\mathbb { E } [ T ] \geq \frac { c _ { q } N } { c _ { q } + 2 \varepsilon } .
$$

In particular E $\mathbf { \Omega } _ { \prime } [ T ] = \Omega ( N )$ at fixed ε, where dilated needs $\mathcal { O } ( \tau _ { c } \log ( N / \varepsilon ) )$ .

Proof sketch. With exact conditionals, the KL equals the factorization error. A step that commits g tokens contains $g ( g - 1 ) / 2$ pairs of tokens committed together; under uniform order, any pair are neighbors with probability $\Theta \bar { ( 1 / N ) }$ so each pair adds at least $c _ { q } / N$ (where $c _ { q }$ is the increase in KL due to a pair of neighbors). With step sizes $g _ { 1 } , \ldots , g _ { T }$ , the expected KL is at least $\begin{array} { r } { \frac { c _ { q } } { 2 N } \sum _ { k } g _ { k } ( g _ { k } - 1 ) \ge } \end{array}$ $\frac { c _ { q } } { 2 } \left( \frac { N } { T } - 1 \right)$ , where $\textstyle \sum _ { k } g _ { k } = N$ forces $\begin{array} { r } { \sum _ { k } g _ { k } ^ { 2 } \ge N ^ { 2 } / T } \end{array}$ (Section A.5).

Feng et al. [2025] earlier proved an $\Omega ( N )$ random-order lower bound for a sequence-error metric on a local copying source; Corollary 2 is a per-order KL version with explicit constants.

## 6 Architecture bounds

## 6.1 Block-autoregressive samplers

Block-autoregressive generation partitions a sequence into blocks of size B, generating blocks left to right autoregressively and running masked diffusion within each block given the generated prefix [Arriola et al., 2025]; recent works using block diffusion include LLaDA2.0 [Bie et al., 2025] and DiffusionGemma [DiffusionGemma Team et al., 2026], which uses uniform noise instead of masking. KL decomposes exactly over blocks, and the within-block weightings mix into one masking distribution, so Theorem 1 applies once.

Corollary 3 (Block-autoregressive samplers). Let π be stationary and satisfy Assumption 1, $N =$ mB, partition [N] into consecutive blocks of size B, and generate blocks left to right, each by Algorithm 1 on its block with the generated prefix revealed and an x-independent within-block strategy.

(i) Train the denoiser under this sampler’s induced weighting (constructed in Section A.5) at the same n as Theorem 1. Then w.p. $\geq 1 - \delta ,$

$$
\mathrm { K L } ( \pi \| p _ { \mathrm { a l g } } ) \leq \underbrace { \sum _ { k = 1 } ^ { m } \mathbb { E } _ { \boldsymbol \pi } \left[ \mathcal { E } _ { \mathrm { f a c t } } ^ { ( k ) } \right] } _ { \mathcal { E } _ { \mathrm { f a c t } } ^ { \mathrm { b l k } } } + N \varepsilon ^ { \prime } ,
$$

where $\mathcal { E } _ { \mathrm { f a c t } } ^ { ( k ) }$ is thefactorization error ofblock k’s run given its prefix.

(ii) The dilated schedule at spacing min $\{ b ^ { * } ( \varepsilon ) , B \}$ within each block achieves $\mathcal { E } _ { \mathrm { f a c t } } ^ { \mathrm { b l k } } \leq \varepsilon$ in

$$
T = \frac { N } { B } \cdot \mathrm { m i n } \{ b ^ { * } ( \varepsilon ) , B \} \ s t e p s ,
$$

where $b ^ { * } ( \varepsilon ) = \mathcal { O } ( \tau _ { c } \log ( C _ { 0 } N | \mathcal { V } | / \varepsilon ) )$ comes from Corollary 1(ii).

Above $b ^ { * } ( \varepsilon ) = \tilde { O } ( \tau _ { c } )$ , blocks decode with a $B / b ^ { * } ( \varepsilon ) \ – \mathrm { f o l d }$ parallel speedup within the ε budget, while below it the within-block schedule is sequential. At a fixed budget of T steps the within-block spacing is $B T / N$ , so the bound applies once $\tilde { B ^ { \prime } } \geq \left( N / T \right) b ^ { * } ( \varepsilon )$ . With the constants fitted in Section $\bar { 7 } , b ^ { * } ( \varepsilon )$ exceeds practical block sizes, so likelihood comparisons across B [Arriola et al., 2025] measure training error.

## 6.2 Looped denoisers

LoopMDM [Lee et al., 2026] loops a small shared block of early-middle layers in an MDM transformer and finds that moderate loop counts suffice for learning. We do not analyze transformer computation. Instead we bound an idealized looped class under an added source assumption: on order-r Markov data the true conditional is computed by iterating a shared width-r kernel, so each loop extends the receptive field by r positions at zero parameter cost. The looped class learns this shared order-r kernel, a correctly specified parametric model: its polynomial rate comes from the order-r structure, and Λ controls only the approximation error. Under CM alone the exponent’s order in Theorem 1 is optimal (Theorem 2).

Corollary 4 (Looped denoisers). Let π be a stationary order-r Markov source with transition floor q<sub>min</sub>, $( \tau _ { c } , C _ { 0 } )$ its constantsfrom Proposition $^ { l , }$ and $\mathcal { G } _ { r , q , \Lambda }$ the looped class ofDefinition 2: a shared order-r kernel with entries in $[ q , 1 ] , q \leq q _ { \mathrm { m i n } } ,$ , looped Λ times per side, output floor $p _ { \mathrm { m i n } }$ as in Theorem 1. Take

Table 1: Theory-motivated checks and results. Single runs; setups and estimators in Section B.
<table><tr><td>Prediction</td><td>Result</td></tr><tr><td colspan="2">Influence on MDLM&#x27;s conditionals</td></tr><tr><td>mean influence decays exponentially (Assumption 1)</td><td>exp. fit R2 0.989 vs. power law 0.966</td></tr><tr><td>worst-case influence stays large (Assumption 2)</td><td>95th pct. 0.72 at d = 64</td></tr><tr><td>dilated factorization term  $\hat { I } \approx 0 \ : \mathrm { b y } \ : b = 1 6$ </td><td>yes; random order ≠ 0 at 64</td></tr><tr><td colspan="2">Block size on a synthetic source (Corollary 3)</td></tr><tr><td>held-out excess small at every B</td><td>≤ 0.031 nats</td></tr><tr><td>no degradation for  $B \geq 4 b ^ { * } \dot { ( } 1 ) = 4 0$ </td><td>holds</td></tr><tr><td colspan="2">Decoding steps on MDLM and LLaDA-8B (Corollaries 1 and 2)</td></tr><tr><td>dilated saturates within [8, 32]</td><td>fwd. 24 (released), 32 (finetunes); gen. 9.7; LLaDA: none by 128</td></tr><tr><td>dilated saturates before random</td><td>fwd. 24 vs. 96, 32 vs. 64; gen. 9.7 vs. 44.7</td></tr><tr><td>saturation well below  $b ^ { * } ( 1 )$  ≈ 860</td><td>9.7–32 steps</td></tr></table>

$$
\Lambda = \left\lceil \frac { \tau _ { c } } { r } \log \frac { 6 C _ { 0 } ( \ell _ { \mathrm { { m a x } } } + 1 ) } { \varepsilon ^ { \prime } } \right\rceil .
$$

$\begin{array} { r } { I f n \geq \frac { 3 2 \ell _ { \mathrm { m a x } } ^ { 2 } } { \varepsilon ^ { \prime 2 } } \left[ | \mathcal { V } | ^ { r + 1 } \log \frac { 4 8 L _ { \mathrm { c o v } } } { \varepsilon ^ { \prime } } + \log \frac { 2 } { \delta } \right] } \end{array}$ , with $L _ { \mathrm { c o v } } \leq 4 r / q ^ { r + 1 }$ (exact constant in Lemma 4), then for every admissible ω the ERM <sup>ˆ</sup>f over $\mathcal { G } _ { r , q , \Lambda }$ trained under ω satisfies $\begin{array} { r } { \mathbb { P } ( \mathcal { E } _ { \omega } ( \hat { f } ) \leq \varepsilon ^ { \prime } ) \geq 1 - \delta . } \end{array}$

Proofsketch. We apply the same decomposition from Theorem 1 and split the excess into approxima tion and estimation errors. The class contains the looped predictor built from the source’s own kernel; after Λ loops it has used the visible tokens within Λr positions on each side, so its total variation with the true distribution is at most $\frac { 3 } { 2 } C _ { 0 } e ^ { - \Lambda r / \tau _ { c } }$ . Each pass of the recursion damps differences between messages so a small change of the kernel entries changes the output by a comparable amount regardless of the number of loops, and the estimation error depends only on the $| \mathcal { V } | ^ { r + 1 }$ kernel entries (Lemma 4). Each loop thus multiplies the approximation-error bound by $e ^ { - r / \tau _ { c } }$ until Λr reaches $\tau _ { c } \log ( 1 / \varepsilon ^ { \prime } )$ , after which further loops do not reduce the bound; with the dilated schedule, inference needs $T \Lambda = \tilde { \cal O } \left( \tau _ { c } \log N \cdot ( \tau _ { c } / r ) \log ( 1 / \varepsilon ^ { \prime } ) \right)$ ) loop iterations.

## 7 Experiments

![](images/5b34e0f9e2c72df2f9defbbad34b8a63b7d24e959a42e4f35ef25365b100ebec.jpg)

![](images/bf7e531aa164fa826886dac37df810e93587ac0b3a23c9ff7f915070806e93de.jpg)

![](images/9c65e0813daa5d9740d1044b6fb6d2a9b5f22264d9f840c507bf8e7be75fbcbd.jpg)  
Figure 1: Left: NLL under GPT-2-large (averaged over sequences) against decoding steps for dilated, random, and confidence-greedy schedules on MDLM; vertical line: the spacing at which the dilated <sup>ˆ</sup>I vanishes; gray band: half to twice that spacing; colored shading: bootstrap intervals. Middle: LLaDA-8B, dilated and random only, with MDLM’s threshold. Right: MAUVE on MDLM (512 reference windows). Single runs; colors as on the left.

Table 1 summarizes the first three experiments (details in Section B). The influence measurements use a trained model’s conditionals in place of the data’s, so we treat these results as qualitative evidence.

Influence decays exponentially on average and stays large in the worst case. On MDLM [Sahoo et al., 2024], we replace the visible tokens beyond distance d of a masked target with those of an unrelated held-out window and record the shift of the model’s conditional in TV over random masking patterns. Mean influence decays exponentially, with decay scale ≈26 at short distances and $\hat { \tau } _ { c } \approx 8 1$ in the tail, and the exponential fits better than a power law (Table 3). The 95th percentile is still 0.72 at $d = 6 4$ . On this model’s conditionals, the measurements are consistent with the averaged decay that Theorem 1(a) and Corollary 1 use and rule out Assumption 2 with useful constants. The dilated factorization term <sup>ˆ</sup>I is indistinguishable from zero by spacing 16, while the random-order estimate is still nonzero at the commit size of spacing 64.

Block size on an exactly computable source. Six block-autoregressive models, identical except for $B \in \{ 4 , \ldots , 1 2 8 \}$ , train on an order-3 synthetic source with exact conditionals. Decoding uses $T = N / \dot { 4 }$ steps, so the within-block spacing is $B / 4$ , and Corollary 3(ii) bounds the factorization error by 1 nat once $B \geq 4 b ^ { * } ( 1 ) = 4 0$ , with $\bar { b ^ { * } } ( 1 ) \stackrel { \cdot } { = } 1 0$ computed from influence constants fitted on the source (Section B.3). Held-out excess is at most 0.031 nats at every B, and generation KL shows a factorization penalty only at $B = 4$ and none above the threshold.

Dilated decoding saturates earlier than random order. We score held-out text by each schedule’s teacher-forced cross-entropy (forward sampling error; for random order, one fixed order per window, as in Corollary 2) and samples by NLL under GPT-2-large [Radford et al., 2019] and MAUVE [Pillutla et al., 2021] (Figure 1). By (23), dilated decoding saturates at $T = 2 4$ on the released model against 96 for random order (32 against 64 after dilated or uniform finetuning), and it lies below random order at every matched step count $T \geq 2$ The generation-NLL fit saturates at 9.7 steps (bootstrap interval [7.9, 18.2]) for dilated against 44.7 for random. Dilated decoding thus needs two to five times fewer steps than random order, and every dilated saturation point lies within twice the spacing 16 at which <sup>ˆ</sup>I vanishes, matching the ordering of Corollary 1(ii) and Corollary 2. At the fitted constants, $b ^ { * } ( 1 ) \approx 8 6 0$ is $2 7 { - } 8 9 \times$ larger than the observed points, so the theorem’s constants are loose while its predicted ordering holds. LLaDA-8B [Nie et al., 2025] is still improving at 128 steps under both orders. Past ∼64 steps, MDLM’s generation NLL keeps falling while MAUVE drops from its peak of 0.61 at $T = 2 4$ , which Zheng et al. [2025] attribute to reduced effective temperature in fp32 categorical sampling, the precision we use.

Loop count. In a looped variant of MDLM trained from scratch, the loop count at which held-out loss saturates moves right with training, $2 . 4 2  2 . 5 3  2 . 6 2$ at 1B, 2B and 4B tokens, the direction Corollary 4 suggests. This experiment is qualitative and incomplete (Section B.5).

## 8 Discussion

We present an end-to-end KL bound for masked diffusion with a learned denoiser under conditional mixing. The sample budget is the same for every decoding schedule and minimax-optimal in its exponent’s order, and we specialize the bound to dilated and block-autoregressive decoding and to a looped message-passing class on order-r sources. Dilated decoding reaches ${ \mathrm { K L } } \ \varepsilon + { \bar { N } } \varepsilon ^ { \prime }$ in $\mathcal { O } ( \tau _ { c } \log ( N | \mathcal { V } | / \varepsilon ) )$ steps, where a fixed random order needs $\Omega ( N )$ on an explicit source, and released and finetuned models show the same ordering. The guarantee holds for a denoiser trained under the decoding weighting, and its training term enters per token, so the sequence-level budget grows polynomially in N. Text with power-law pairwise mutual information lies outside conditional mixing, and for such sources dilated decoding needs polynomially many steps (Remark 1). Future work includes bounds for slower-mixing sources, schedules that avoid the context-free first step, tighter constants, influence probes on larger MDMs, and samplers that revise decoded tokens [Liu et al., 2025, Zhao et al., 2025, Wang et al., 2025].

## Acknowledgments and Disclosure of Funding

No third-party funding was received for this work; the A100 GPU time used in the experiments was rented. No competing interests are declared.

## References

M. Arriola, A. Gokaslan, J. T. Chiu, Z. Yang, Z. Qi, J. Han, S. S. Sahoo, and V. Kuleshov. Block diffusion: interpolating between autoregressive and diffusion language models. In International Conference on Learning Representations, pages 50726–50753, 2025. arXiv:2503.09573.

R. Atar and O. Zeitouni. Exponential stability for nonlinear filtering. Annales de l’Institut Henri Poincaré, Probabilités et Statistiques, 33(6):697–725, 1997. doi: 10.1016/S0246-0203(97) 80110-0.

K. M. R. Audenaert. A sharp continuity estimate for the von Neumann entropy. Journal of Physics A: Mathematical and Theoretical, 40(28):8127–8136, 2007. doi: 10.1088/1751-8113/40/28/S18.

J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, volume 34, pages 17981–17993, 2021.

I. Azangulov, T. Pandeva, N. Prasad, J. Zazo, and S. Karmalkar. Parallel sampling from masked diffusion models via conditional independence testing. In International Conference on Learning Representations, pages 140485–140521, 2026. arXiv:2510.21961.

V. Bansal, C. Cho, S. Kumar, S. Sanghavi, and P. Sarkar. Walk fast but be careful: understanding parallel sampling in masked diffusion. arXiv preprint arXiv:2606.22976, 2026.

H. Ben-Hamu, I. Gat, D. Severo, N. Nolte, and B. Karrer. Accelerated sampling from masked diffusion models via entropy bounded unmasking. In Advances in Neural Information Processing Systems, volume 38, pages 55981–56007, 2025.

T. Bie, M. Cao, K. Chen, L. Du, M. Gong, Z. Gong, Y. Gu, J. Hu, Z. Huang, Z. Lan, C. Li, C. Li, J. Li, Z. Li, H. Liu, L. Liu, G. Lu, X. Lu, Y. Ma, J. Tan, L. Wei, J.-R. Wen, Y. Xing, X. Zhang, J. Zhao, D. Zheng, J. Zhou, J. Zhou, Z. Zhou, L. Zhu, and Y. Zhuang. LLaDA2.0: scaling up diffusion language models to 100B. arXiv preprint arXiv:2512.15745, 2025.

G. Birkhoff. Extensions of Jentzsch’s theorem. Transactions of the American Mathematical Society, 85(1):219–227, 1957. doi: 10.1090/S0002-9947-1957-0087058-6.

P. J. Bushell. Hilbert’s metric and positive contraction mappings in a Banach space. Archive for Rational Mechanics and Analysis, 52(4):330–338, 1973. doi: 10.1007/BF00247467.

C. Cai and G. Li. Confidence-based decoding is provably efficient for diffusion language models. In Advances in Neural Information Processing Systems, 2026. To appear; arXiv:2603.22248.

A. Campbell, J. Benton, V. De Bortoli, T. Rainforth, G. Deligiannidis, and A. Doucet. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems, volume 35, pages 28266–28279, 2022.

H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman. MaskGIT: masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11315–11325, 2022.

H. Chen and L. Ying. Convergence analysis of discrete diffusion model: exact implementation through uniformization. Journal ofMachine Learning, 4(2):108–127, 2025. doi: 10.4208/jml.240812.

S. Chen, K. Cong, and J. Li. Optimal inference schedules for masked diffusion models. In Proceedings ofthe Thirty-Ninth Conference on Learning Theory (COLT), volume 336 of Proceedings ofMachine Learning Research, pages 1279–1311, 2026. arXiv:2511.04647.

C. Cho and Y. Wu. Minimax optimality of score-entropy discrete diffusion. arXiv preprint arXiv:2608.20635, 2026.

I. Csiszár and Z. Talata. Context tree estimation for not necessarily finite memory processes, via BIC and MDL. IEEE Transactions on Information Theory, 52(3):1007–1016, 2006. doi: 10.1109/TIT.2005.864431.

DiffusionGemma Team, A. A. Taïga, J. Assiene, D. Calandriello, R. Chaabouni, J. Gante, T. von Glehn, N. Keating, C. Knutsen, M. Kukla, T. Liu, I. Lobov, O. Nabati, J. G. Oliveira, N. Perez-Nieves, N. Prutianova, B. Shahriari, J. Tarbouriech, P. Tyletski, Ç. Ünlü, C. Wu, G. Cameron, J. Connor, S. Girgin, M. Grootendorst, A. Levkovitch, E. Nachmani, O. Sanseviero, P. Stanczyk, Q. Berthet, A. Campbell, C. Crepy, V. De Bortoli, A. Doucet, R. Elie, A. Galashov, K. Greff, A. Jacq, D. Ruhe, Y.-H. Wu, S. Flennerhag, B. O’Donoghue, G. Scrivener, and S. Thakoor. DiffusionGemma technical report. arXiv preprint arXiv:2608.00146, 2026.

D. Dmitriev, Z. Huang, and Y. Wei. Efficient sampling with discrete diffusion models: sharp and adaptive guarantees. In Proceedings of the Thirty-Ninth Conference on Learning Theory (COLT), volume 336 of Proceedings of Machine Learning Research, pages 2038–2104, 2026. arXiv:2602.15008.

R. L. Dobrushin. The description of a random field by means of conditional probabilities and conditions of its regularity. Theory ofProbability & Its Applications, 13(2):197–224, 1968. doi: 10.1137/1113026.

G. Feng, Y. Geng, J. Guan, W. Wu, L. Wang, and D. He. Theoretical benefit and limitation of diffusion language model. In Advances in Neural Information Processing Systems, volume 38, pages 24415–24459, 2025.

W. Feng, Y. Sun, and Y. Yin. What can be sampled locally? In Proceedings ofthe ACM Symposium on Principles of Distributed Computing (PODC), pages 121–130, 2017.

Y. Han, S. Jana, and Y. Wu. Optimal prediction of Markov chains with and without spectral gap. IEEE Transactions on Information Theory, 69(6):3920–3959, 2023. doi: 10.1109/TIT.2023.3239508.

Y. Han, T. Jiang, and Y. Wu. Prediction from compression for models with infinite memory, with applications to hidden Markov and renewal processes. In Proceedings of the Thirty-Seventh Conference on Learning Theory (COLT), volume 247 of Proceedings ofMachine Learning Research, pages 2270–2307, 2024.

Y. Hao, A. Orlitsky, and V. Pichapati. On learning Markov chains. In Advances in Neural Information Processing Systems, volume 31, 2018.

E. Hoogeboom, A. A. Gritsenko, J. Bastings, B. Poole, R. van den Berg, and T. Salimans. Autoregressive diffusion models. In International Conference on Learning Representations, 2022. arXiv:2110.02037.

J. Kim, K. Shah, V. Kontonis, S. M. Kakade, and S. Chen. Train for the worst, plan for the best: understanding token ordering in masked diffusions. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 30749–30768, 2025. arXiv:2502.06768.

H. Lavenant and G. Zanella. Error bounds and optimal schedules for masked diffusions with factorized approximations. arXiv preprint arXiv:2510.25544v2, 2025.

F. Le Gland and N. Oudjane. Stability and uniform approximation of nonlinear filters using the Hilbert metric and application to particle filters. Annals of Applied Probability, 14(1):144–187, 2004. doi: 10.1214/aoap/1075828050.

S. Lee, C. Hong, S. Kim, J. Lee, J. Park, and D. Park. Looped diffusion language models. arXiv preprint arXiv:2605.26106, 2026.

G. Li and C. Cai. Breaking AR’s sampling bottleneck: provable acceleration via diffusion language models. In Advances in Neural Information Processing Systems, volume 38, pages 11700–11725, 2025. arXiv:2505.21400.

Y. Li, A. Kirchmeyer, A. Mehta, Y. Qin, B. Dadachev, K. Papineni, S. Kumar, and A. Risteski. Promises and pitfalls of generative masked language modeling: theoretical framework and practical guidelines. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 27969–28017, 2024.

Y. Liang, Y. Liang, L. Lai, and N. Shroff. Discrete diffusion models: novel analysis and new sampler guarantees. In Advances in Neural Information Processing Systems, volume 38, pages 165511–165548, 2025. arXiv:2509.16756.

Y. Liang, Z. Tan, N. Shroff, and Y. Liang. Sharp convergence rates for masked diffusion models. arXiv preprint arXiv:2602.22505, 2026.

H. W. Lin and M. Tegmark. Critical behavior in physics and probabilistic formal languages. Entropy, 19(7):299, 2017. doi: 10.3390/e19070299.

H. Liu and Y. Yin. Simple parallel algorithms for single-site dynamics. In Proceedings of the 54th Annual ACM SIGACT Symposium on Theory ofComputing, pages 1431–1444, 2022.

S. Liu, J. Nam, A. Campbell, H. Stärk, Y. Xu, T. Jaakkola, and R. Gómez-Bombarelli. Think while you generate: discrete diffusion with planned denoising. In International Conference on Learning Representations, pages 81979–82026, 2025.

O. Luxembourg, H. H. Permuter, and E. Nachmani. Plan for speed: dilated scheduling for masked diffusion language models. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 83451–83474, 2026. arXiv:2506.19037.

K. Nakaishi, R. Yoshida, K. Kajikawa, K. Hukushima, and Y. Oseki. Rethinking the relationship between the power law and hierarchical structures. Transactions of the Association for Computational Linguistics, 14:1917–1935, 2026. doi: 10.1162/tacl.a.785. arXiv:2505.04984.

S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. Zhou, Y. Lin, J.-R. Wen, and C. Li. Large language diffusion models. In Advances in Neural Information Processing Systems, volume 38, pages 50608–50646, 2025. arXiv:2502.09992.

J. Ou, S. Nie, K. Xue, F. Zhu, J. Sun, Z. Li, and C. Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. In International Conference on Learning Representations, pages 64972–65009, 2025. arXiv:2406.03736.

Y.-H. Park, C.-H. Lai, S. Hayakawa, Y. Takida, and Y. Mitsufuji. Jump Your Steps: optimizing sampling schedule of discrete diffusion models. In International Conference on Learning Representations, pages 96272–96300, 2025.

F. Z. Peng, Z. Bezemek, S. Patel, J. Rector-Brooks, S. Yao, A. J. Bose, A. Tong, and P. Chatterjee. Path planning for masked diffusion model sampling. arXiv preprint arXiv:2502.03540, 2025.

K. Pillutla, S. Swayamdipta, R. Zellers, J. Thickstun, S. Welleck, Y. Choi, and Z. Harchaoui. MAUVE: measuring the gap between neural text and human text using divergence frontiers. In Advances in Neural Information Processing Systems, volume 34, pages 4816–4828, 2021.

A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. Technical report, OpenAI, 2019. URL https://cdn.openai.com/better-language-models/language\_models\_are\_ unsupervised\_multitask\_learners.pdf.

S. S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. T. Chiu, A. Rush, and V. Kuleshov. Simple and effective masked diffusion language models. In Advances in Neural Information Processing Systems, volume 37, pages 130136–130184, 2024.

S. S. Sahoo, J.-M. Lemercier, Z. Yang, J. Deschenaux, J. Liu, J. Thickstun, and A. Jukic. Scaling´ beyond masked diffusion language models. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 106997– 107014, 2026. arXiv:2602.15014.

E. Seneta. Non-negative Matrices and Markov Chains. Springer Series in Statistics. Springer, New York, 2nd edition, 2006. doi: 10.1007/0-387-32792-4. Revised printing of the 1981 second edition.

V. Sharan, S. Kakade, P. Liang, and G. Valiant. Prediction with a short memory. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing (STOC), pages 1074–1087, 2018.

H. Shen. Mutual information scaling and expressive power of sequence models. arXiv preprint arXiv:1905.04271, 2019.

J. Shi, K. Han, Z. Wang, A. Doucet, and M. K. Titsias. Simplified and generalized masked diffusion for discrete data. In Advances in Neural Information Processing Systems, volume 37, pages 103131–103167, 2024.

A. Shih, D. Sadigh, and S. Ermon. Training and inference on any-order autoregressive models the right way. In Advances in Neural Information Processing Systems, volume 35, pages 2762–2775, 2022.

A. Srikanth, M. Gaur, and V. Aggarwal. Discrete state diffusion models: a sample complexity perspective. In International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings ofMachine Learning Research, pages 3547–3555, 2026.

B. Uria, I. Murray, and H. Larochelle. A deep and tractable density estimator. In Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 467–475, 2014.

G. Wang, Y. Schiff, S. S. Sahoo, and V. Kuleshov. Remasking discrete diffusion models with inference-time scaling. In Advances in Neural Information Processing Systems, volume 38, pages 147282–147339, 2025. arXiv:2503.00307.

D. Weitz. Counting independent sets up to the tree threshold. In Proceedings of the 38th Annual ACM Symposium on Theory of Computing (STOC), pages 140–149, 2006.

C. Wen, W. Liang, J. Lu, and Z. Lin. Serial depth of adaptive parallel sampling: lower bounds and schedules for masked diffusion models. arXiv preprint arXiv:2608.25505, 2026.

B. Yu. Assouad, Fano, and Le Cam. In D. Pollard, E. Torgersen, and G. L. Yang, editors, Festschrift for Lucien Le Cam: Research Papers in Probability and Statistics, pages 423–435. Springer, New York, NY, 1997. doi: 10.1007/978-1-4612-1880-7\_29.

Z. Zhang, Z. Chen, and Q. Gu. Convergence of score-based discrete diffusion models: a discrete-time analysis. In International Conference on Learning Representations, pages 34747–34772, 2025.

Y. Zhao and C. Cai. Adaptation to intrinsic dependence in diffusion language models. arXiv preprint arXiv:2602.20126, 2026.

Y. Zhao, J. Shi, F. Chen, S. Druckmann, L. Mackey, and S. Linderman. Informed correctors for discrete diffusion models. In Advances in Neural Information Processing Systems, volume 38, pages 125510–125538, 2025.

K. Zheng, Y. Chen, H. Mao, M.-Y. Liu, J. Zhu, and Q. Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. In International Conference on Learning Representations, pages 63186–63227, 2025. arXiv:2409.02908.

## Appendix

## A Proofs

We prove the two theorems presented in the paper with the help of a few observations about how distant tokens affect Markov-structured data and motivates our conditional mixing assumption as well as how much information we lose by ignoring certain distant tokens.

We structure Appendix A by first proving in Section A.1 the lemmas and bounds that we use in our theorem proofs, analyzing in Section A.2 conditional mixing for finite-order and hidden Markov sources, connecting the sampler’s learning error to excess risk in Section A.3, and finally proving our main learning and decoding results in Sections A.4 and A.5.

## A.1 Bounding lemmas & insights

We prove two lemmas used in later proofs and discuss insights they provide into how Markov sources and conditional mixing relate:

• Lemma 1 shows that products of positive matrices (and thus the posteriors of Markov-type sources) forget distant inputs exponentially fast

• Lemma 2 bounds the estimation error of ERM

We first briefly consider how much revealing a subset of tokens refines a token posterior. Given $x _ { C } = \xi \mathrm { w i t h } \pi ( x _ { C } = \xi ) > 0$ and disjoint $C , { \bar { A } } \subseteq [ N ] \setminus \{ i \}$ , the posterior ${ \bar { p } } = \pi ( x _ { i } \in \cdot \mid x _ { C } = \xi )$ is the mixture $\mathbb { E } _ { \zeta } [ p _ { \zeta } ]$ of the refined posteriors $p _ { \zeta } = \pi ( x _ { i } \in \cdot \mid x _ { C } = \xi , x _ { A } = \zeta )$ over $\zeta \sim \pi ( x _ { A } \in \cdot |$ $x _ { C } = \xi )$ . Every measure of influence we present describes how spread out these components are.

We start by recalling that the worst-case coefficient κ bounds the distance between two components; by Jensen’s and convexity of total variation,

$$
\begin{array} { r } { \mathrm { T V } ( p _ { \zeta } , \bar { p } ) \leq \mathbb { E } _ { \zeta ^ { \prime } } \mathrm { T V } ( p _ { \zeta } , p _ { \zeta ^ { \prime } } ) \leq \kappa ( d ) \qquad \mathrm { w h e n e v e r ~ } A \cap \mathcal { B } ( i , d ) = \emptyset } \end{array}\tag{10}
$$

We also observe that strong conditional mixing implies conditional mixing $\bar { \kappa } \leq \kappa$ with the same constants: with $C = z \setminus A$ and $A \subseteq z \setminus B ( i , d )$ , (10) bounds $\mathrm { T V } ( \pi ( x _ { i } \in \cdot \bar { | } x _ { z } ) , \pi ( x _ { i } \in \cdot | x _ { z \backslash A } ) )$ by $\kappa ( d )$ pointwise. Consequently, truncating the visible context to within w tokens $( A = z \setminus B ( i , w ) )$ changes the posterior pointwise by at most $\kappa ( w )$ and by (5) likewise at most $\bar { \kappa } ( w )$ in expectation under any masking distribution.

While influence is controlled in total variation, the sampler’s error is an entropy and the learning error is a KL (excess risk). Consequently, we use Audenaert-Fannes to connect between entropy and total variation, allowing us to convert between all three: for distributions $p , q$ on V with $t = \mathrm { T } \bar { \mathrm { V } } ( p , q )$ and $h ( t ) \leq t \log e / t$ the binary entropy, the bound $| H ( p ) - H ( q ) | \leq t { \dot { \log ( | \mathcal { V } | - 1 ) } } + h ( t )$ of Audenaert [2007],

$$
| H ( p ) - H ( q ) | \leq \eta \left( \mathrm { T V } ( p , q ) \right) , \qquad \eta ( x ) = x \log ( e | \mathcal { V } | / x ) , \eta ( 0 ) = 0\tag{11}
$$

where $\eta$ is nondecreasing and concave on $[ 0 , 1 ] .$

For Markov-type sources the classical forward-backward recursion writes the posterior of a masked token as $p \propto R ( \alpha \odot D _ { 0 } \beta )$ for some matrix R with messages $\alpha , \beta$ that are products of positive transfer matrices and of diagonal matrices that encode the visible tokens. Sequences that agree near the token share the nearest factors, so bounding κ in (10) reduces to bounding how far such messages differ; we bound this difference in Lemma 1 through Hilbert’s projective metric $d _ { \mathrm { H } } ( u , v ) =$ log max $\dot { \mathbf { \rho } } _ { j } ( u _ { j } / v _ { j } ) \mathbf { \rho } +$ log ma $\mathrm { x } _ { j } ( v _ { j } / u _ { j } )$ , which compares positive vectors up to scale and satisfies the contraction tanh $\mathrm { \Omega } _ { \mathrm { l } } ( \bar { \Delta } _ { \mathrm { H } } ( M ) / \bar { 4 } ) < 1$ where $\begin{array} { r } { \Delta _ { \mathrm { H } } ( M ) \stackrel { - } { = } \operatorname* { s u p } _ { u , v \geq 0 } d _ { \mathrm { H } } ( M \stackrel { - } { u } , M v ) } \end{array}$ is its projective diameter [Birkhoff, 1957].

Lemma 1 (Exponential forgetting). Let $K _ { 1 } , K _ { 2 } , \ldots , K _ { m }$ be entrywise positive matrices with $\Delta _ { \mathrm { H } } ( K _ { j } ) \le \Delta , D _ { 1 } , D _ { 2 } , \dots , D _ { m }$ be nonnegative diagonal matrices, and $\vartheta = \operatorname { t a n h } ( \Delta / 4 )$ . Messages $\ddot { \alpha } = K _ { 1 } D _ { 1 } \cdot \cdot \cdot K _ { m } D _ { m } u$ and $\alpha ^ { \prime } = \breve { K } _ { 1 } D _ { 1 } \cdot \cdot \cdot \breve { K } _ { m } D _ { m } u ^ { \prime }$ that share these m factors satisfy $\begin{array} { r } { d _ { \mathrm { H } } ( \boldsymbol { \alpha } , \boldsymbol { \alpha } ^ { \prime } ) \leq \Delta \vartheta ^ { m - 1 } . } \end{array}$ for all nonnegative $u , u ^ { \prime } .$ . For entrywise positive $\alpha , \alpha ^ { \prime } , \beta , \beta ^ { \prime }$ , the posteriors $p \propto R ( \alpha \odot D _ { 0 } \beta )$ and $p ^ { \prime } \propto R ( \alpha ^ { \prime } \odot { \overline { { D _ { 0 } \beta ^ { \prime } } } } )$ where R is nonnegative and $D _ { 0 }$ is nonnegative diagonal satisfy $p _ { j } / p _ { j } ^ { \prime } \in [ e ^ { - h } , e ^ { h } ]$ for every j and $\mathrm { T V } ( p , p ^ { \prime } ) \leq h ,$ , where $h = d _ { \mathrm { H } } ( \alpha , \alpha ^ { \prime } ) + d _ { \mathrm { H } } ( \beta , \beta ^ { \prime } )$

Proof. We use standard facts about $d _ { \mathrm { H } }$ [Bushell, 1973, Seneta, 2006]: it is defined for nonnegative vectors with the same support, with the maxima over that support, and is $\infty$ otherwise; it vanishes exactly when $u \propto v$ and is invariant under rescaling either argument. For entrywise positive M the projective diameter is the maximal log cross-ratio,

$$
\Delta _ { \mathrm { H } } ( M ) = \operatorname* { m a x } _ { j , k , l , m } \log \frac { M _ { j l } M _ { k m } } { M _ { j m } M _ { k l } } ,\tag{12}
$$

so $\begin{array} { r } { \Delta _ { \mathrm { H } } ( M ) \leq 2 \log ( \operatorname* { m a x } _ { j k } M _ { j k } / \operatorname* { m i n } _ { j k } M _ { j k } ) } \end{array}$ and $\Delta _ { \mathrm { H } } ( M ^ { \top } ) = \Delta _ { \mathrm { H } } ( M )$ , and $\Delta _ { \mathrm { H } } ( M D ) \le \Delta _ { \mathrm { H } } ( M )$ for nonnegative diagonal $D ,$ since the image of MD lies in that of M. Birkhoff’s theorem [Birkhoff, 1957] states that for entrywise positive u, v with $M u , M v \ne 0$

$$
d _ { \mathrm { H } } ( M u , M v ) \leq \operatorname { t a n h } \left( \Delta _ { \mathrm { H } } ( M ) / 4 \right) d _ { \mathrm { H } } ( u , v ) .\tag{13}
$$

For the first claim, $K _ { m } D _ { m } u$ and $K _ { m } D _ { m } u ^ { \prime }$ lie in the image of $K _ { m } D _ { m }$ , whose diameter is at most $\Delta$ and each of the remaining $m - 1$ factors $K _ { j } D _ { j }$ has diameter at most $\Delta$ and contracts $d _ { \mathrm { H } }$ by ϑ by (13). For the second claim, by triangle inequality $\overset { \circ } { d } _ { \mathrm { H } } ( \alpha \odot D _ { 0 } \beta , \alpha ^ { \prime } \odot D _ { 0 } \beta ^ { \prime } ) \le h$ . Nonnegative matrices preserve entrywise order and are homogeneous and thus do not increase $d _ { \mathrm { H } }$ , so $d _ { \mathrm { H } } ( p , p ^ { \prime } ) \leq h$ and the ratio bound follows. Then $\begin{array} { r } { \mathrm { T V } ( p , p ^ { \prime } ) \stackrel { - } { = } \sum _ { j : p _ { j } > p _ { i } ^ { \prime } } ( p _ { j } - p _ { j } ^ { \prime } ) \leq 1 - e ^ { - h } \leq h } \end{array}$ □

Lemma 1 bounds how much distant tokens move a posterior, which governs both the sampler’s factorization error and the approximation error of a predictor that ignores distant tokens. To control the estimation error of ERM in the learning results, we next bound the estimation error of empirical risk minimization through the size of a finite cover of the predictor class; each training sequence contributes one bounded loss, so the bound depends only on the number of sequences n however long each sequence is.

Lemma 2 (Estimation over a finite cover). Let G be a class ofpredictors whose outputs are entrywise $\geq p _ { \mathrm { m i n } }$ with $\ell _ { \mathrm { m a x } } = \log ( 1 / p _ { \mathrm { m i n } } )$ . Suppose thatfor some $\rho \in ( 0 , \ell _ { \operatorname* { m a x } } ]$ a finite set $\Phi _ { \rho }$ offunctions with values $i n \left[ - \ell _ { \mathrm { m a x } } , 0 \right]$ satisfies mi $\mathbf { l } _ { \varphi \in \Phi _ { \rho } } \parallel$ log $f - \varphi \| _ { \infty } \leq \rho \dot { f } o r$ every $f \in { \mathcal { G } } .$ . Let ω be admissible and let $\hat { f }$ minimize the empirical risk over G on n independent drawsfrom ω. Then, $w . p . \geq 1 - \delta ,$

$$
\mathcal { E } _ { \omega } ( \hat { f } ) \leq \operatorname* { i n f } _ { f \in \mathcal { G } } \mathcal { E } _ { \omega } ( f ) + 2 \ell _ { \operatorname* { m a x } } \sqrt { \frac { \log ( 2 | \Phi _ { \rho } | / \delta ) } { 2 n } } + 4 \rho
$$

Ifinstead $\hat { f }$ minimizes the empirical masked cross-entropy (1), the same holds at $\omega = \omega _ { \mathrm { u } }$ with the last two terms multiplied by $( \bar { N } + 1 ) / N$

Proof. By Hoeffding and a union bound, with probability at least $1 - \delta$ the empirical risk of every $\varphi \in \Phi _ { \rho }$ (an average of n i.i.d. losses in $[ 0 , \ell _ { \mathrm { m a x } } ] )$ is within $s _ { n } = \ell _ { \mathrm { m a x } } \sqrt { \log ( 2 | \Phi _ { \rho } | / \delta ) / ( 2 n ) }$ of its expectation. Replacing a predictor by its nearest element in $\Phi _ { \rho }$ changes both its empirical and its population risk by at most $\rho ,$ so for every $f \in { \mathcal { G } }$ passing from $\hat { f } _ { \mathrm { ~ t o ~ } f }$ through their cover elements and the optimality of $\hat { f }$ yields $R _ { \omega } ( { \hat { f } } ) \leq R _ { \omega } ( f ) + 2 s _ { n } + 4 \rho$ . The masked cross-entropy follows by observing that each sequence’s average loss over its masked positions is again an i.i.d. loss in $[ \dot { 0 } , \ell _ { \mathrm { m a x } } ]$ with mean $\begin{array} { r } { \frac { N } { N + 1 } \dot { R } _ { \omega _ { \mathrm { u } } } } \end{array}$ by (3). □

## A.2 Influence decay in Markov sources

For Markov sources, visible tokens far from $x _ { i }$ can only affect its posterior through the chain of positions between them; for transitions with a probability floor $q _ { \mathrm { m i n } } > 0$ this influence thus contracts by a fixed factor, so the influence of visible tokens beyond distance d decays geometrically with d. This directly motivates conditional mixing: strong CM (Assumption 2) assumes this exponential decay of influence itself and CM (Assumption 1) assumes it holds only on average over the source; this decay is what lets a width- $\cdot \mathcal { O } ( \tau _ { c } \log ( 1 / \bar { \varepsilon ^ { \prime } } ) )$ window predictor approximate every conditional (Theorem 1) and makes tokens committed b apart nearly independent conditioned on context (Corollary 1). We formalize this in Propositions 1 and 2 for finite-order Markov sources with a transition floor and for hidden Markov sources with a two-sided Doeblin kernel, with correlation length set by the floor. Both proofs apply Lemma 1 to the forward-backward messages of the stationary process on Z, whose influence coefficient bounds that of every length-N marginal, so the constants do not depend on $N .$

ProofofProposition 1. The blocks $Y _ { t } = x _ { i + ( t - 1 ) r + 1 : i + t r } , t \in \mathbb { Z }$ form a stationary Markov chain with marginal $\mu _ { r }$ and transition matrix $\begin{array} { r } { Q ^ { ( r ) } ( y , y ^ { \prime } ) = \prod _ { s = 1 } ^ { r } Q ( y _ { s } ^ { \prime } \mid y _ { s : r } , y _ { 1 : s - 1 } ^ { \prime } ) \in [ q _ { \operatorname* { m i n } } ^ { r } , 1 ] ; } \end{array}$ ; by (12), $\Delta _ { \mathrm { H } } \big ( Q ^ { ( r ) } \big ) = \Delta _ { \mathrm { H } } \big ( Q ^ { ( r ) \top } \big ) \leq \Delta _ { r }$ . Conditioning on $x _ { z } = \xi$ multiplies block t by the diagonal indicator $D _ { t }$ of agreement with $\xi ,$ so the posterior of x is the normalized last-coordinate marginal of $\alpha \odot D _ { 0 } \beta$ with $\alpha = Q ^ { ( r ) \top } D _ { - 1 } \cdot \cdot \cdot Q ^ { ( r ) \top } D _ { - t } \mu _ { r }$ and $\beta = Q ^ { ( r ) } D _ { 1 } \cdot \cdot \cdot Q ^ { ( r ) } D _ { t } \mathbf { 1 }$ for any $t \ge m = \lfloor d / r \rfloor$ with $z \subseteq [ i - t r , i + t r ]$ . Configurations equal $\mathrm { o n } z \cap B ( i , d )$ give the same $D _ { s } \operatorname { f o r } - ( m - 1 ) \leq s \leq m$ so $\beta$ and α share their first m and $m - 1$ factors, respectively, and for $m \geq 2$ Lemma 1 gives

$$
\kappa ( d ) \leq \Delta _ { r } \vartheta _ { r } ^ { m - 1 } + \Delta _ { r } \vartheta _ { r } ^ { m - 2 } \leq 2 \Delta _ { r } \vartheta _ { r } ^ { \lfloor d / r \rfloor - 2 } \leq 2 \Delta _ { r } \vartheta _ { r } ^ { d / r - 3 } = C _ { 0 } e ^ { - d / \tau _ { c } } .\tag{14}
$$

For $d \ < \ 2 r$ the right side exceeds $2 \Delta _ { r } > 1 \geq \kappa ( d )$ since $q _ { \mathrm { m i n } } ~ \leq ~ \frac { 1 } { 2 }$ . Finally, $\log ( 1 / \vartheta _ { r } ) =$ 2 arctanh $( q _ { \mathrm { m i n } } ^ { r } ) \ge 2 q _ { \mathrm { m i n } } ^ { r }$ gives $\tau _ { c } \leq r / ( 2 q _ { \mathrm { m i n } } ^ { r } )$ □

ProofofProposition 2. By (8), the chain stays in supp $\nu _ { \mathrm { D } } ;$ there, Γ is entrywise positive with crossratios at most $\gamma _ { \mathrm { { D } } } ^ { - 4 }$ , so $\Delta _ { \mathrm { H } } ( \Gamma ) = \Delta _ { \mathrm { H } } ( \Gamma ^ { \top } ) \leq \Delta _ { \Gamma }$ by (12). With stationary distribution $\mu$ and emission matrices $G _ { s } = \mathrm { d i a g } ( g ( \xi _ { i + s } \mid \cdot ) )$ for visible positions $i + s \in z$ and $\dot { G } _ { s } = I$ otherwise, the posterior of $x _ { i }$ is proportional to $R ( \alpha \odot \beta )$ where $R _ { a h } = g ( a \mid h ) , \alpha = \Gamma ^ { \top } G _ { - 1 } \cdot \cdot \cdot \Gamma ^ { \top } G _ { - t } \mu$ , and $\beta = \Gamma G _ { 1 } \cdot \cdot \cdot \Gamma G _ { t } { \bf 1 }$ for any $t \geq d$ with $z \subseteq [ i - t , i + t ]$ ; these messages are entrywise positive when $\pi ( x _ { z } = \xi ) > 0$ . Configurations that agree on $z \cap B ( i , d )$ ) yield messages that share their first d factors on each side so for $d \geq 1$ , Lemma 1 gives $\kappa ( d ) \leq 2 \Delta _ { \Gamma } \vartheta _ { \Gamma } ^ { d - 1 } = C _ { 0 } e ^ { - d / \tau _ { c } }$ while $\kappa ( 0 ) \leq 1 \leq C _ { 0 }$ since $C _ { 0 } = 8 \log ( 1 / \gamma _ { \mathrm { { D } } } ) /$ tanh $( \log ( 1 / \gamma _ { \mathrm { { D } } } ) ) \geq 8 $ . Thus, $\log ( 1 / \vartheta _ { \Gamma } ) = 2$ arctanh $( \gamma _ { \mathrm { D } } ^ { 2 } ) \geq 2 \gamma _ { \mathrm { D } } ^ { 2 }$ gives $\tau _ { c } \leq 1 / ( 2 \gamma _ { \mathrm { D } } ^ { 2 } )$ ). □

## A.3 From decoding to denoising

A decoder only queries the denoiser at the pairs $( i , z )$ at which it commits tokens. We show that this pair distribution is admissible even for adaptive decoders and that the learning term $\mathcal { E } _ { \mathrm { l e a r n } }$ is N times the denoiser’s excess risk under this distribution. Thus, learning for a decoder is learning at its induced weighting, which is why Theorem 1 is stated for every masking distribution and under Assumption 2 for every admissible weighting.

Proposition 3 (Adaptive strategies are admissible). Let a decoding strategy ν choose at each step $k \geq 1$ a nonempty commit set $\bar { S } _ { k } \subseteq [ N ] \backslash z _ { k }$ , where the visible set is $z _ { k } = S _ { 1 } \cup \cdot \cdot \cdot \cup S _ { k - 1 }$ , as a function of $( z _ { k } , x _ { z _ { k } } )$ and ofa random variable $U \perp \perp x .$ . Run ν on $x \sim \pi$ under teacherforcing so that committed positions take their true values, draw $i \sim \operatorname { U n i f } ( [ N ] )$ independently of $( x , U )$ , and let $z = z _ { k } f o r k$ with $i \in S _ { k }$ . Then, the joint law $\omega _ { \nu } o f ( i , z , x )$ is admissible, andfor every predictor f the learning term $o f \left( 9 \right)$ [Lavenant and Zanella, 2025, Prop. 2] along the teacher-forced run ofν equals $\mathcal { E } _ { \mathrm { l e a r n } } ( f ) = \dot { N } \mathcal { E } _ { \omega _ { \nu } } ( f )$ ; if ν is non-adaptive (its selections depend on U alone), then $\omega _ { \nu }$ is a masking distribution.

Proof. A decoder decides to commit i when z is visible using only $x _ { z }$ and randomness independent of $x ,$ so conditioning on that decision reveals nothing about $x _ { i }$ beyond $x _ { z }$ . Visible sets only grow, so for $j \not \in Z$ the event $\mathbf { \bar { \{ } }  i = j , z = Z \}$ is decided by i and the selections made at visible sets inside $Z ,$ hence by $( i , x _ { Z } , U ) . { \mathrm { A s ~ } } ( i , U )$ ⊥⊥ x, the event is conditionally independent of $x _ { j }$ given $x z$ . For ν fixed before training, the per-sequence losses stay i.i.d. in $[ 0 , \ell _ { \mathrm { m a x } } ] .$ , so Lemma 2 applies unchanged. By (2), the learning term, the sum of the excess KLs over the N commits, is N times an average over a uniformly chosen position:

$$
\mathcal { E } _ { \mathrm { l e a r n } } ( f ) = \mathbb { E } \left[ \sum _ { k \geq 1 } \sum _ { j \in S _ { k } } \log \frac { \pi ( x _ { j } \mid x _ { z _ { k } } ) } { f ( x _ { j } \mid j , z _ { k } , x _ { z _ { k } } ) } \right] = N \mathbb { E } _ { \omega _ { v } } \left[ \log \frac { \pi ( x _ { i } \mid x _ { z } ) } { f ( x _ { i } \mid i , z , x _ { z } ) } \right] = N \mathcal { E } _ { \omega _ { v } } ( f ) .
$$

Under $\begin{array} { r } { \alpha _ { t } = 1 - t , \mathbb { P } ( M = S ) = \int _ { 0 } ^ { 1 } t ^ { | S | } ( 1 - t ) ^ { N - | S | } \mathrm { d } t = \left( ( N + 1 ) { \binom { N } { | S | } } \right) ^ { - 1 } } \end{array}$ , so M is uniform given its size, which is uniform on $\{ 0 , \ldots , N \}$ . Thus, $\mathbb { P } ( M \neq \dot { \varnothing } ) = N / ( N + \overset { \prime } { 1 } )$ ), and given $M \neq \emptyset$ a uniform $i \in M$ with $z = [ N ] \backslash M$ has law $\omega _ { \mathrm { u } }$ , which is (3). The $\mathsf { N E L B O } , ( 1 )$ with $\bar { 1 / | M | }$ | replaced by $1 / t$ , equals $N R _ { \omega _ { \mathrm { u } } } ( f ) \dot { : } \dot { 1 / t }$ cancels the probability t that a position i is masked, and the same integral over the other $N - 1$ positions gives z its law given i under $\omega _ { \mathrm { u } }$

## A.4 Learning the denoiser

By Proposition 3, it suffices to learn the true conditionals at a fixed weighting ω. Under conditional mixing, visible tokens beyond distance w change the posterior by $\mathcal { O } ( \exp ( - w \overline { { / } } \tau _ { c } ) )$ ; we thus consider predictors that read only a window of width w then bound their approximation error by the influence (Lemma 3) and their estimation error by a cover (Lemma 2), proving Theorem 1. Theorem 2 then shows that the linear exponent in the sample complexity is necessary under conditional mixing alone, and Corollaries 4 and 5 improve it under a floor or known parametric structure.

Definition 1 (Window class). For $w \ge 0 ,$ , the window context ct $\mathfrak { c } _ { w } ( i , z , x ) \in \overline { { \mathcal { V } } } ^ { 2 w }$ lists $x _ { i + o }$ for $o \in \{ \pm 1 , \ldots , \pm w \}$ , with [mask] at masked or out-of-range positions. The window class is

$$
\mathcal { F } _ { w } = \{ ( 1 - \lambda ) f + \lambda \mathrm { U n i f } ( \mathcal { V } ) : f : \overline { { \mathcal { V } } } ^ { 2 w }  \mathcal { P } ( \mathcal { V } ) \} , \qquad \lambda = | \mathcal { V } | p _ { \mathrm { m i n } } ,
$$

evaluated at the window context; its outputs are at least $p _ { \mathrm { m i n } }$ , so its log-losses lie in $[ 0 , \ell _ { \mathrm { m a x } } ]$

By stationarity, the window conditional $\pi _ { w } ( \cdot \mid c ) = \pi ( x _ { 0 } \in \cdot \mid x _ { o } = c _ { o } \forall o \mathrm { ~ s . t . ~ } c _ { o } \neq [ \mathsf { m a s k } ] )$ satisfies $\pi \big ( x _ { i } \in \cdot \mid x _ { z \cap \mathcal { B } ( i , w ) } \big ) = \pi _ { w } \big ( \cdot \mid \mathrm { c t x } _ { w } \big ( i , z , x \big ) \big )$ for every $N , i \in [ N ]$ , and $z \subseteq [ N ] \backslash \{ i \}$ Lemma 3 (Approximation by window truncation). For every admissible ω and every w $\begin{array} { r } { \ i \ge 0 , } \end{array}$

$$
\operatorname* { m i n } _ { f \in \mathcal { F } _ { w } } \mathcal { E } _ { \omega } ( f ) \le \left( \ell _ { \mathrm { m a x } } + 1 \right) \left( \bar { \kappa } _ { \omega } ( w ) + \lambda \right) ,
$$

where $\bar { \kappa } _ { \omega } ( w ) = \mathbb { E } _ { \omega } \left[ \mathrm { T V } \left( \pi ( x _ { i } \in \cdot \mid x _ { z } ) , \pi ( x _ { i } \in \cdot \mid x _ { z \cap \mathcal { B } ( i , w ) } ) \right) \right]$ is the averaged truncation error.

Proof. Take $f ^ { \star } = ( 1 - \lambda ) \pi _ { w } + \lambda \mathrm { U n i f } ( \mathcal { V } ) \in \mathcal { F } _ { w }$ , which is within $\bar { \kappa } _ { \omega } ( w ) + \lambda$ of the true conditional in expected total variation. For min $_ { \cdot a } q _ { a } \geq p _ { \mathrm { m i n } } ,$ using $\log ( p _ { a } / q _ { a } ) \leq \ell _ { \mathrm { m a x } }$ and log $s \leq s - 1$ ,

$$
\mathrm { K L } ( p \| q ) \leq \sum _ { a : p _ { a } > q _ { a } } \left[ \left( p _ { a } - q _ { a } \right) \log \frac { p _ { a } } { q _ { a } } + q _ { a } \log \frac { p _ { a } } { q _ { a } } \right] \leq \left( \ell _ { \mathrm { m a x } } + 1 \right) \mathrm { T V } ( p , q ) ,
$$

and (2) gives $\mathcal { E } _ { \omega } ( f ^ { \star } ) \leq ( \ell _ { \mathrm { m a x } } + 1 ) ( \bar { \kappa } _ { \omega } ( w ) + \lambda )$

ProofofTheorem 1. Fix $w \ge 0 , p _ { \mathrm { m i n } } \le 1 / ( 2 | \mathcal { V } | )$ and $\rho \in ( 0 , \ell _ { \mathrm { m a x } } ]$ . Rounding the $| \overline { { \mathcal { V } } } | ^ { 2 w } | \mathcal { V } |$ logprobabilities log $f ( a \mid c )$ where $c \in \overline { { \gamma } } ^ { 2 w } , a \in \mathcal { V }$ of each $f \in \mathcal { F } _ { w }$ to the grid $\rho \mathbb { Z } \cap [ - \ell _ { \mathrm { m a x } } , 0 ]$ yields a cover of size $\mathcal { N } _ { \rho }$ with log $\mathcal { N } _ { \rho } \leq | \overline { { \mathcal { V } } } | ^ { 2 w } | \mathcal { V } | \log ( 3 \ell _ { \operatorname* { m a x } } / \rho )$ , so Lemmas 2 and 3 give $\mathbf { w . p . } \geq 1 - \delta$ the oracle inequality

$$
\mathcal { E } _ { \omega } ( \hat { f } ) \leq \underbrace { \left( \ell _ { \mathrm { m a x } } + 1 \right) \left( \bar { \kappa } _ { \omega } ( w ) + \lambda \right) } _ { \mathrm { a p p r o x i m a t i o n } } + \underbrace { 2 \ell _ { \mathrm { m a x } } \sqrt { \frac { \log ( 2 \mathcal { N } _ { \rho } / \delta ) } { 2 n } } + 4 \rho } _ { \mathrm { e s t i m a t i o n } } .\tag{15}
$$

We next show that the ERM under ω and the masked cross-entropy minimizer (for which $\omega = \omega _ { \mathrm { u } } )$ both satisfy $\mathcal { E } _ { \omega } ( \hat { f } ) \leq \varepsilon ^ { \prime }$ with probability $\geq 1 - \delta$ whenever

$$
n \geq \frac { 7 2 \ell _ { \mathrm { m a x } } ^ { 2 } } { \varepsilon ^ { \prime 2 } } \left[ | \overline { { \mathcal { V } } } | ^ { 2 w } | \mathcal { V } | \log \frac { 7 2 \ell _ { \mathrm { m a x } } } { \varepsilon ^ { \prime } } + \log \frac { 2 } { \delta } \right] ,\tag{16}
$$

which gives the stated rate after substituting $w \leq \tau _ { c } \log ( 4 C _ { 0 } ( \ell _ { \mathrm { m a x } } + 1 ) / \varepsilon ^ { \prime } ) + 1$ into $| \overline { { \mathcal { V } } } | ^ { 2 w }$

Approximation. With $s _ { 0 } = \log ( 1 0 | \mathcal { V } | / \varepsilon ^ { \prime } )$ and $p _ { \operatorname* { m i n } } = \varepsilon ^ { \prime } / ( 1 0 | \mathcal { V } | s _ { 0 } ) \leq 1 / ( 2 | \mathcal { V } | ) , \ell _ { \operatorname* { m a x } } + 1 = s _ { 0 } +$ log $s _ { 0 } + 1 \leq 2 s _ { 0 } , \mathrm { s o } ( \ell _ { \mathrm { m a x } } + 1 ) \lambda \leq \varepsilon ^ { \prime } / 5$ . By (5) and $( 1 0 ) , \bar { \kappa } _ { \omega } ( w ) \leq \bar { \kappa } ( w )$ for masking distributions and $\bar { \kappa } _ { \omega } ( w ) \leq \kappa ( w )$ for admissible ω, so under Assumption 1 in part (a) and Assumption 2 in part (b), $w = \lvert \dot { \tau _ { c } } \log ( 4 C _ { 0 } ( \ell _ { \mathrm { m a x } } + 1 ) / \varepsilon ^ { \prime } ) \rvert$ gives $( \ell _ { \mathrm { m a x } } + 1 ) \bar { \kappa } _ { \omega } \hat { ( } w ) \leq \varepsilon ^ { \prime } \bar { / } 4$ and an approximation term below $\varepsilon ^ { \prime } / 2$

Estimation. For the ERM under $\omega , \rho = \varepsilon ^ { \prime } / 1 6$ and (16) bound the square-root term by $\varepsilon ^ { \prime } / 4 ,$ so $\mathcal { E } _ { \omega } ( \hat { f } ) \leq \varepsilon ^ { \prime } / 2 + \varepsilon ^ { \prime } / 4 + \varepsilon ^ { \prime } / 4 = \varepsilon ^ { \prime }$ . For the masked cross-entropy, $\rho = \varepsilon ^ { \prime } / 2 4$ bounds the estimation term by $\varepsilon ^ { \prime } / 3$ , and the factor $( N + 1 ) / N \le 3 / 2$ of Lemma 2 gives $\mathcal { E } _ { \omega _ { \mathrm { u } } } ( \hat { f } ) \leq \varepsilon ^ { \prime } / 2 + \varepsilon ^ { \prime } / 2 = \varepsilon ^ { \prime }$ □

The exponent of Theorem 1 grows linearly in $\tau _ { c } \log | \mathcal { V } |$ because the window class has $| \overline { { \mathcal { V } } } | ^ { 2 w }$ contexts and w $\mathbf { \bar { \rho } } \approx \tau _ { c } \log ( 1 / \varepsilon ^ { \prime } )$ ; we next show that no learner avoids this growth:

Proof of Theorem 2. Each context of the $d - 1$ most recent tokens hides one bit that sets the sign of a perturbation of size $\gamma \asymp \sqrt { \varepsilon ^ { \prime } } ;$ the perturbation is small enough to preserve strong conditional mixing, and a bit is observed only when its context occurs, so Assouad’s lemma forces about $| \gamma | ^ { d - 1 } / \varepsilon ^ { \prime }$ token observations.

Construction. Let $m = \operatorname* { m a x } \{ 1 , ( { \tau _ { c } } / { 2 } ) \log ( { C _ { 0 } } / { 2 } ) \} , \varepsilon _ { 0 } = \operatorname* { m i n } \{ 2 ^ { - 1 0 } , ( { C _ { 0 } } e ^ { - 3 m / \tau _ { c } } / ( 1 6 0 m ) ) ^ { 2 } \}$ and $\gamma = 8 \sqrt { \varepsilon ^ { \prime } }$ , so that d is the largest integer with $2 0 d \gamma e ^ { 3 d / \tau _ { c } } \leq C _ { 0 }$ and $d \geq \lfloor m \rfloor$ . Let $\chi , \psi : \mathcal { V } \to$ $\{ - 1 , 0 , 1 \}$ sum to zero and vanish at no more than one symbol each, and let $\Pi = \{ \pi _ { \theta } : \theta \in$ $\{ \pm 1 \} ^ { \nu ^ { d - 1 } } \}$ , where $\pi _ { \theta }$ is the stationary order-d chain with

$$
\pi _ { \theta } \left( x _ { t } = a \mid x _ { t - d } = \xi _ { 1 } , x _ { t - d + 1 : t - 1 } = y \right) = { \frac { 1 + \gamma \theta _ { y } \psi ( \xi _ { 1 } ) \chi ( a ) } { | \mathcal { V } | } } , \qquad \xi _ { 1 } \in \mathcal { V } , y \in \mathcal { V } ^ { d - 1 } .
$$

Strong conditional mixing. As $t \gamma e ^ { 3 t / \tau _ { c } } \leq C _ { 0 } / 2 0$ at $t = d$ and at $t = m$ and $e ^ { 2 m / \tau _ { c } } \geq C _ { 0 } / 2$ $d \gamma e ^ { d / \tau _ { c } } \leq 1 / 1 0 , \mathrm { s o } \gamma \leq \operatorname* { m i n } \{ 1 / ( 5 d ) , 1 / 4 \}$ and arctanh $\gamma \leq 1 . 1 \gamma$ . Whatever the visible tokens, $x _ { i }$ enters the joint law through at most $d + 1$ factors in $[ 1 - \dot { \gamma } , 1 + \gamma ] / | \mathcal { V } |$ , so any two posteriors of $x _ { i }$ satisfy

$$
\mathrm { T V } \le \sinh \left( 2 ( d + 1 ) \operatorname { a r c t a n h } \gamma \right) \le 9 d \gamma \le C _ { 0 } e ^ { - 3 d / \tau _ { c } } ,
$$

which gives $\kappa ( j ) \leq C _ { 0 } e ^ { - j / \tau _ { c } } \mathrm { f o r } j < 3 d$ . For $j \geq 3 d ,$ , the d-block kernel of the proof of Proposition 1 has entries in $[ ( 1 - \gamma ) ^ { d } , ( 1 + \gamma ) ^ { d } ] / | \mathcal { V } | ^ { d }$ , hence diameter $\Delta \leq 4 d \mathrm { a r c t a n h } \gamma \leq 5 d \gamma$ by (12) and $\bar { \vartheta } = \operatorname { t a n h } ( \Delta / 4 ) \leq 1 . 2 5 d \gamma \leq e ^ { - d / \tau _ { c } }$ , and (14) with $( d , \Delta , \bar { \vartheta } )$ ) in place of $( r , \Delta _ { r } , \vartheta _ { r } )$ gives $\kappa ( j ) \leq$ $1 0 d \gamma \bar { \vartheta } ^ { \lfloor j / d \rfloor - 2 } \le C _ { 0 } e ^ { - j / \tau _ { c } }$

Bounding the excess. As ψ sums to zero, the uniform law on $\mathcal { V } ^ { d }$ is stationary, so at $\omega ^ { * }$ the context $( \xi _ { 1 } , y )$ is uniform with conditional $p _ { \xi _ { 1 } } ^ { \pm } = ( 1 \pm \gamma \psi ( \xi _ { 1 } ) \chi ) / | \mathcal { V } |$ for $\theta _ { y } = \pm 1$ , and (2) separates the excess over the bits:

$$
\mathcal { E } _ { \omega ^ { * } } ( f ) = \sum _ { y } e _ { y } ( f , \theta _ { y } ) , \qquad e _ { y } ( f , \pm 1 ) = \frac { 1 } { | \mathscr { V } | ^ { d } } \sum _ { \xi _ { 1 } } \mathrm { K L } \left( p _ { \xi _ { 1 } } ^ { \pm } \| f ( \cdot \mid \xi _ { 1 } , y ) \right) .
$$

For $\psi ( \xi _ { 1 } ) \neq 0 , \mathrm { T V } ( p _ { \xi _ { 1 } } ^ { + } , p _ { \xi _ { 1 } } ^ { - } ) \geq \gamma / 2$ , so by Pinsker’s inequality $\mathrm { K L } ( p _ { \xi _ { 1 } } ^ { + } | | q ) + \mathrm { K L } ( p _ { \xi _ { 1 } } ^ { - } | | q ) \geq \gamma ^ { 2 } / 4$ for every q, and summing over the at least $| \nu | / 2$ such $\xi _ { 1 }$ gives $e _ { y } ( { \bar { f } } , + 1 ) + e _ { y } ( { \bar { f } } , - 1 ) \geq \delta =$ $\gamma ^ { 2 } / ( 8 | \dot { \mathcal { V } } | ^ { d - 1 } )$ . Flipping $\theta _ { y }$ changes only the rows at contexts with suffix $y ,$ which each of the $N - d$ later tokens follows with probability $| \nu | ^ { - ( d - 1 ) }$ , and KL $( p _ { \xi _ { 1 } } ^ { + } \| p _ { \xi _ { 1 } } ^ { - } ) \le 2 \gamma$ arctanh $\gamma \leq 2 . 2 \gamma ^ { 2 } ;$ ; so the two laws of the sample are within $\mathrm { K L 2 . 2 } n ( N - d ) \gamma ^ { 2 } | \mathcal { V } | ^ { - ( d - \bar { 1 } ) }$ , hence within total variation $\frac { 1 } { 2 }$ by Pinsker’s inequality when $n ( N - d ) < | \mathcal { V } | ^ { d - 1 } / ( 3 0 0 \varepsilon ^ { \prime } )$ , and Assouad’s lemma [Yu, 1997] gives

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } \left[ \mathcal { E } _ { \omega ^ { * } } ( \hat { f } ) \right] \geq \frac { 1 } { 2 } \sum _ { y } \frac { \delta } { 2 } \left( 1 - \frac { 1 } { 2 } \right) = \frac { | \mathcal { V } | ^ { d - 1 } \delta } { 8 } = \frac { \gamma ^ { 2 } } { 6 4 } = \varepsilon ^ { \prime } .
$$

The coefficient 2 of $\tau _ { c }$ in Theorem 1 comes from the linear conversion $\mathrm { K L } \ \leq \ ( \ell _ { \mathrm { m a x } } + 1 )$ TV. When the source’s own conditionals have a floor, the window conditional needs no smoothing, the conversion becomes quadratic, and the coefficient halves.

Corollary 5 (Conditional floor). Let π be stationary with $\pi ( x _ { i } = a \mid x _ { z } = \xi ) \geq q _ { c } > 0$ for all i, patterns $z \not \ni i , \xi$ with $\pi ( x _ { z } = \xi ) > 0 ;$ , and $a \in \mathcal V ,$ , and let π satisfy Assumption 1 in mean square:

$$
\begin{array} { r } { \mathbb { E } _ { x \sim \pi } \mathrm { T V } \left( \pi ( x _ { i } \in \cdot \mid x _ { z } ) , \pi ( x _ { i } \in \cdot \mid x _ { z \setminus A } ) \right) ^ { 2 } \leq C _ { 0 } ^ { 2 } e ^ { - 2 d / \tau _ { c } } } \end{array}
$$

for all $d ~ \geq ~ 0 , ~ i , ~ z ~ \not \supset ~ i$ and $A \ \subseteq \ z \ \backslash \ B ( i , d )$ . Then, for every masking distribution ω the proof of Theorem 1 applies with $p _ { \mathrm { m i n } } ~ \stackrel { \cdot } { = } ~ \mathrm { m i n } \{ q _ { c } , ( 2 | \mathcal { V } | ) ^ { - 1 } \}$ , so $\ell _ { \mathrm { m a x } }$ is constant in $\varepsilon ^ { \prime } ,$ with $\begin{array} { r c l } { w } & { = } & { \lceil \tau _ { c } \log ( 3 C _ { 0 } / \sqrt { q _ { c } \varepsilon ^ { \prime } } ) \rceil } \end{array}$ the sample complexity scales as $( 1 / \varepsilon ^ { \prime } ) ^ { 2 + \tau _ { c } \log | \overline { { \mathcal { V } } } | + o ( 1 ) }$ , so the $\tau _ { c ^ { - } }$ coefficient ofthe exponent halvesfrom 2 to 1.

Proof. The window conditional satisfies $\pi _ { w } \geq q _ { c } \geq p _ { \mathrm { m i n } }$ entrywise, so $\pi _ { w } \in \mathcal { F } _ { w }$ , and for min $\iota _ { a } q _ { a } \geq$ $q _ { c } ,$

$$
\mathrm { K L } ( p \| q ) \leq \log \left( 1 + \chi ^ { 2 } ( p \| q ) \right) \leq \sum _ { a } { \frac { ( p _ { a } - q _ { a } ) ^ { 2 } } { q _ { a } } } \leq { \frac { 4 \mathrm { T V } ( p , q ) ^ { 2 } } { q _ { c } } }
$$

As $( i , z ) \perp \perp \ x , \ ( 2 )$ and the mean-square condition with $\textit { A } = \textit { z } \backslash \textit { B } ( i , w )$ give $\mathcal { E } _ { \omega } ( \pi _ { w } ) \ \leq$ $4 C _ { 0 } ^ { 2 } e ^ { - 2 w / \tau _ { c } } / q _ { c } \leq 4 \varepsilon ^ { \prime } / 9 < \varepsilon ^ { \prime } / 2$ . The rest of the proof of Theorem 1 uses $p _ { \mathrm { m i n } }$ only through $\ell _ { \mathrm { m a x } } .$ , now constant, and substituting $w \leq \tau _ { c } \log ( 3 C _ { 0 } \dot { / } \sqrt { q _ { c } \varepsilon ^ { \prime } } ) + 1$ into (16) gives the rate. □

Mean-square conditional mixing is implied by Assumption 2 through (10) and implies Assumption 1 by Jensen’s inequality. Order-r sources have $q _ { c } \geq q _ { \operatorname* { m i n } } ^ { r + 1 } \colon \pi ( x _ { i } = a \mid x _ { - i } )$ is a product of $r + 1$ transition probabilities, each at least $q _ { \mathrm { m i n } }$ , over a probability at most 1, and $\pi ( x _ { i } = a \mid x _ { z } ) $ averages it.

The window class learns a separate distribution $f ( \cdot \mid c )$ for each of its $| \overline { { \mathcal { V } } } | ^ { 2 w }$ contexts $c .$ For an order-r source, the forward–backward recursion of Proposition 1 computes the true conditional from the $| \mathcal { V } | ^ { r + 1 }$ entries of one kernel, each pass reaching r positions further; learning the kernel and looping the recursion Λ times gives a cover that does not grow with Λ (Lemma 4) and an approximation error that decays in Λ, hence the polynomial rate of Corollary 4.

Definition 2 (Looped class). Let $q \in ( 0 , 1 / | \nu | ]$ . For an order-r kernel $Q _ { \theta }$ with entries in $[ q , 1 ]$ , the looped predictor is $f _ { \theta , \Lambda } = ( 1 - \lambda ) \bar { p } _ { \theta , \Lambda } + \lambda \mathrm { \dot { U } n i f } ( \mathcal { V } )$ with $\lambda = | \mathcal { V } | p _ { \mathrm { m i n } } ,$ , where $p _ { \theta , \Lambda }$ is the normalized last-coordinate marginal of α $\odot D _ { 0 } \beta$ for the messages $\alpha , \beta$ of the proof of Proposition 1, computed with the block kernel $Q _ { \theta } ^ { ( r ) }$ by Λ recursion steps each, from uniform initializations and with the visible tokens in $\mathbf { b l o c k s - N , \bar { \dots } , } \boldsymbol { \Lambda }$ entering as diagonal matrices. These predictors form the looped class $\mathcal { G } _ { r , q , \Lambda }$

Lemma 4 (Loop-count-free covering). Let $\Delta _ { r } ( q ) = 2 r \log ( 1 / q )$ and $\vartheta _ { r } ( q ) = \operatorname { t a n h } ( \Delta _ { r } ( q ) / 4 )$ Uniformly over $( i , z , \xi )$

$$
\left\| \log f _ { \theta , \Lambda } - \log f _ { \theta ^ { \prime } , \Lambda } \right\| _ { \infty } \le L _ { \mathrm { c o v } } \| Q _ { \theta } - Q _ { \theta ^ { \prime } } \| _ { \infty } , \qquad L _ { \mathrm { c o v } } = \frac { 4 r } { q ( 1 - \vartheta _ { r } ( q ) ) } \le \frac { 4 r } { q ^ { r + 1 } } .
$$

Hence, for $\rho \in ( 0 , 1 ]$ , a grid of mesh $\rho / L _ { \mathrm { c o v } }$ on the $| \mathcal { V } | ^ { r + 1 }$ kernel entries yields a cover of $\cdot { \mathcal { G } } _ { r , q , \Lambda }$ at log-scale $\rho ,$ in the sense ofLemma 2, with log-cardinality at most $\mathcal { \vert V \vert } ^ { r + 1 } \log ( 3 L _ { \mathrm { c o v } } / \rho )$

Proof. Each entry of $Q _ { \theta } ^ { ( r ) }$ is a product of r entries of $Q _ { \theta }$ in $[ q , 1 ] .$ , so ∥ log $\begin{array} { r } { Q _ { \theta } ^ { ( r ) } - \log Q _ { \theta ^ { \prime } } ^ { ( r ) } \| _ { \infty } \leq s = } \end{array}$ $r \| Q _ { \theta } - Q _ { \theta ^ { \prime } } \| _ { \infty } / { q }$ , and both kernels and their transposes have diameter at most $\Delta _ { r } ( q ) \stackrel { \mathrm { \scriptsize ~ b y ~ } } { \mathrm { \scriptsize ~ b y ~ } } ( 1 2 )$ . Write a message as $\dot { K } D _ { 1 } \cdot \cdot \cdot K D _ { \Lambda ^ { u } }$ , and let $P _ { k }$ use the kernel K of θ in its first k factors and the kernel $K ^ { \prime }$ of $\theta ^ { \prime }$ in the others. Then $P _ { k } u$ and $P _ { k - \cdot }$ <sub>1</sub>u are the images under $K D _ { 1 } \cdot \cdot \cdot K D _ { k - 1 }$ of $K D _ { k } w$ and $K ^ { \prime } D _ { k } w$ , which are within 2s in d , so by (13), whatever the visible tokens,

$$
d _ { \mathrm { H } } ( P _ { \Lambda } u , P _ { 0 } u ) \leq \sum _ { k = 1 } ^ { \Lambda } 2 s \vartheta _ { r } ( q ) ^ { k - 1 } \leq \frac { 2 s } { 1 - \vartheta _ { r } ( q ) } .
$$

By Lemma 1, every ratio $p _ { \theta , \Lambda } ( a ) / p _ { \theta ^ { \prime } , \Lambda } ( a )$ then lies in $[ e ^ { - h } , e ^ { h } ]$ with $h \leq 4 s / ( 1 - \vartheta _ { r } ( q ) ) =$ $L _ { \mathrm { c o v } } \| Q _ { \theta } - Q _ { \theta ^ { \prime } } \| _ { \infty } \mathrm { ; }$ mixing with $\lambda { \mathrm { U n i f } } ( \gamma )$ only moves the ratios toward 1, and $1 - \vartheta _ { r } ( q ) =$ $2 q ^ { r } / ( 1 + q ^ { r } ) \geq q ^ { r }$ . One kernel from each of the at most $( 3 L _ { \mathrm { c o v } } / \rho ) ^ { | \nu | ^ { r + 1 } }$ nonempty grid cells gives the cover. □

ProofofCorollary 4. As $q \leq q _ { \mathrm { m i n } } ,$ , the class contains $f _ { Q , \Lambda }$ for the source’s kernel $Q .$ . From the exact initializations, 1 for $\beta$ and $\mu _ { r }$ for α, its loops would compute $\pi ( x _ { i } \in \cdot \mid x _ { z \cap W } )$ for the union $W \supseteq B ( i , \Lambda r )$ of blocks $- \Lambda , \ldots , \Lambda ;$ the uniform start is exact up to scale for $\beta$ and shares Λ factors with the exact one for α. By (10) and Lemma 1, using $C _ { 0 } = 2 \Delta _ { r } / \vartheta _ { r } ^ { 3 }$ and $\vartheta _ { r } ^ { \Lambda } = e ^ { - \Lambda r / \tau _ { c } }$

$$
t _ { \Lambda } = \operatorname* { s u p } _ { ( i , z , \xi ) } \mathrm { T V } \left( p _ { Q , \Lambda } , \pi ( x _ { i } \in \cdot \mid x _ { z } ) \right) \le \kappa ( \Lambda r ) + \Delta _ { r } \vartheta _ { r } ^ { \Lambda - 1 } \le \frac 3 2 C _ { 0 } e ^ { - \Lambda r / \tau _ { c } } .
$$

As in Lemma 3, mi ${ \mathfrak { q } } _ { r , q , \Lambda } { \mathcal { E } } _ { \omega } \leq ( \ell _ { \operatorname* { m a x } } + 1 ) ( t _ { \Lambda } + \lambda )$ for every admissible $\omega ,$ and the stated Λ gives $( \ell _ { \mathrm { { m a x } } } + 1 ) t _ { \Lambda } \leq \varepsilon ^ { \prime } / 4$ ; the proof of Theorem 1 then applies with $t _ { \Lambda }$ in place of $\bar { \kappa } _ { \omega } ( w )$ and the cover of Lemma 4 at $\rho = \varepsilon ^ { \prime } / 1 6$ □

## A.5 End-to-end guarantees

The sampler’s KL splits into a learning term and the factorization error (9). Proposition 3 and Theorem 1 bound the learning term by $N \bar { \varepsilon ^ { \prime } }$ for a decoding strategy whose denoiser is trained under the strategy’s weighting, which is Corollary 1(i). The factorization error is the total correlation of the tokens committed in each step, and under conditional mixing tokens far apart carry little information about each other given the context, by (10) and (11); a schedule should therefore commit only distant tokens together, as the dilated schedule does, which gives Corollary 1(ii). A random order commits nearby tokens together and pays for each such pair, which gives the lower bound of Corollary 2, and block-autoregressive decoding inherits both parts of Corollary 1 block by block (Corollary 3).

Proof of Corollary $I ( i )$ . By (9), valid for adaptive strategies [Lavenant and Zanella, 2025, Prop. 2], and Proposition 3,

$$
\mathrm { K L } ( \pi \| p _ { \mathrm { a l g } } ) \leq { \mathcal { E } } _ { \mathrm { f a c t } } + N { \mathcal { E } } _ { \omega _ { \nu } } ( { \hat { f } } ) \leq { \mathcal { E } } _ { \mathrm { f a c t } } + N \varepsilon ^ { \prime }
$$

with probability at least $1 - \delta ,$ by Theorem 1(a) when ν is non-adaptive, so that $\omega _ { \nu }$ is a masking distribution, and by Theorem 1(b) under Assumption 2 otherwise. □

Proof of Corollary $I ( i i ) .$ . At step $t \in [ b ]$ the schedule commits the residue class $S _ { t } = \{ i _ { 1 } < \cdots <$ $i _ { | S _ { t } | } \}$ of t modulo b given the visible set $z _ { t } = S _ { 1 } \cup \cdot \cdot \cdot \cup S _ { t - 1 }$ , so by the chain rule, with $A =$ $\{ i _ { 1 } , \dotsc , i _ { m - 1 } \}$

$$
\mathcal { E } _ { \mathrm { f a c t } } = \sum _ { t = 1 } ^ { b } \operatorname { T C } _ { \pi } \left( x _ { S _ { t } } \mid x _ { z _ { t } } \right) = \sum _ { t = 1 } ^ { b } \sum _ { m = 2 } ^ { | S _ { t } | } I ( x _ { i _ { m } } ; x _ { A } \mid x _ { z _ { t } } ) .
$$

Each summand is the expected entropy change $\mathbb { E } [ H ( \bar { p } ) - H ( p _ { \zeta } ) ]$ , with $p _ { \zeta }$ and $\bar { p }$ the posteriors of $x _ { i _ { m } } { \mathrm { ~ g i v e n ~ } } x _ { z _ { t } \cup A }$ and given $x _ { z _ { t } }$ , so (11) and Jensen’s inequality for the concave η give

$$
\mathcal { E } _ { \mathrm { f a c t } } \leq \sum _ { t , m } \mathbb { E } \left[ \eta \left( \mathrm { T V } ( p _ { \zeta } , \bar { p } ) \right) \right] \leq ( N - b ) \eta \left( \bar { \kappa } _ { \nu } ( b - 1 ) \right) \leq \left( N - b \right) \eta \left( \bar { \kappa } ( b - 1 ) \right) ,\tag{17}
$$

where

$$
\bar { \kappa } _ { \nu } ( b - 1 ) = \frac { 1 } { N - b } \sum _ { t = 1 } ^ { b } \sum _ { m = 2 } ^ { | S _ { t } | } \mathbb { E } \left[ \mathrm { T V } ( p _ { \zeta } , \bar { p } ) \right] \leq \bar { \kappa } ( b - 1 )\tag{18}
$$

is the influence averaged over the pairs committed together, bounded by (5) with $z _ { t } \cup A$ as the visible set; since the choice of b below uses only that η is nondecreasing, Corollary 1(ii) and Corollary 3(ii) also hold with $\bar { \kappa } _ { \nu } ( b - 1 )$ in place of $\bar { \kappa } ( b - 1 )$ . For $\varepsilon \in ( 0$ , N log |V|] (for larger ε every b works), let

$$
L ^ { * } = \log ( e | \mathcal { V } | ) + \log \left( 1 + \frac { 2 C _ { 0 } N } { \varepsilon } \right) , \qquad b = b ^ { * } ( \varepsilon ) = 1 + \left\lceil \tau _ { c } \log \frac { 2 C _ { 0 } N L ^ { * } } { \varepsilon } \right\rceil .\tag{19}
$$

Then $\bar { \kappa } ( b - 1 ) \leq C _ { 0 } e ^ { - ( b - 1 ) / \tau _ { c } } \leq s = \varepsilon / ( 2 N L ^ { * } ) \leq 1$ and, as $\begin{array} { r } { C _ { 0 } \geq 1 , \log ( e | \mathcal { V } | ) + \log \frac { 2 N } { \varepsilon } \leq L ^ { * } } \end{array}$ , so

$$
\left( N - b \right) \eta \left( \bar { \kappa } ( b - 1 ) \right) \leq N s \log \frac { e | \mathcal { V } | } { s } = \frac { \varepsilon } { 2 L ^ { * } } \left( \log ( e | \mathcal { V } | ) + \log \frac { 2 N L ^ { * } } { \varepsilon } \right) \leq \frac { \varepsilon } { 2 L ^ { * } } \left( L ^ { * } + \log L ^ { * } \right) \leq \varepsilon .
$$

$[ \mathbf { f } \ b ^ { * } ( \varepsilon ) \geq N$ , take $b = N \colon$ every step commits a single token, so $\mathcal { E } _ { \mathrm { f a c t } } = 0$ . In both cases $\mathcal { E } _ { \mathrm { f a c t } } \leq \varepsilon$ in $T = b { \mathrm { ~ s t e p s } } ,$ and part (i) applies to this non-adaptive schedule. □

The dilated schedule reaches factorization error ε in $\mathcal { O } ( \tau _ { c } \log ( N / \varepsilon ) )$ steps. On the two-state chain $\pi _ { q }$ a fixed random order needs $\Omega ( N ) \colon$ : its expected KL is a weighted sum of increments of the information profile [Lavenant and Zanella, 2025, Lems. 6 and 8], and each increment is at least $c _ { q } / N$ because a newly revealed token is the target’s neighbor with probability $1 / N$ and neighbors share a constant amount of information.

ProofofCorollary 2. Fix the step sizes, with cumulative counts $0 = a _ { 0 } < a _ { 1 } < \cdots < a _ { T } = N$ gaps $g _ { k } = a _ { k } - a _ { k - 1 }$ and $r _ { a } ( i ) = \operatorname* { m i n } \{ a _ { k } : a _ { k } \geq i \}$ . Let $F ( j ) = \mathbb { E } [ I ( x _ { t } ; x _ { z _ { i } } ) ]$ be the information profile [cf. Lavenant and Zanella, 2025, Sec. 4.1], with $t \sim \mathrm { U n i f } ( [ N ] )$ and $z _ { j }$ an independent uniform j-subset of $[ N ] \ \backslash \ \{ t \}$ , and let $\Delta F ( i ) = F ( i ) - F ( i - 1 )$ ). For an order uniform given the sizes, Lavenant and Zanella [2025, Lems. 6 and 8] give

$$
\mathbb { E } _ { \sigma } \operatorname { K L } ( \pi \| p ^ { \sigma } ) = \sum _ { i = 1 } ^ { N - 1 } \Delta F ( i ) \left( r _ { a } ( i ) - i \right) ,
$$

so it suffices to bound the increments from below.

Each increment is at least $c _ { q } / N .$ . Let $z ^ { \prime }$ be a uniform i-subset of $[ N ] \setminus \{ t \} , U \sim \mathrm { U n i f } ( z ^ { \prime } )$ and $z = z ^ { \prime } \setminus \{ U \}$ . As z is a uniform $( i - 1 )$ -subset, the chain rule gives $\bar { \Delta { F } ( i ) } = \mathbb { E } [ I ( x _ { t } ; x _ { U } \mid x _ { z } ) ]$ and the event $G = \{ t \leq N - 1 , \dot { U } = t + 1 \}$ that the new token is the target’s right neighbor has probability $\begin{array} { r } { \frac { N - 1 } { N } \cdot \frac { \stackrel {  } { i } } { N - 1 } \cdot \frac { 1 } { i } = \frac { 1 } { N } } \end{array}$ . On $G ,$ fix $x _ { z } = \xi$ and let $s _ { - } < t$ and $s _ { + } > t + 1$ be the nearest positions of z, if any. By the Markov property,

$$
\begin{array} { r } { P ( u , v ) = { \mathbb P } ( x _ { t } = u , x _ { t + 1 } = v \mid x _ { z } = \xi ) = \varphi ( u ) Q ( u , v ) \psi ( v ) / Z , \qquad Q = \binom { 1 - q \_ q } { \ q \ 1 - q } , } \end{array}
$$

with $\varphi = Q ^ { t - s _ { - } } ( \xi _ { s _ { - } } , \cdot )$ and $\psi = Q ^ { s _ { + } - t - 1 } ( \cdot , \xi _ { s _ { + } } )$ , each replaced by $\left( { \frac { 1 } { 2 } } , { \frac { 1 } { 2 } } \right)$ if there is no such visible position, and $Z \leq 1$ as $\varphi ,$ ψ are probability vectors. The entries of $Q ^ { g }$ are $\bar { \frac { 1 } { 2 } } ( 1 \pm ( 1 - 2 q ) ^ { g } ) \in [ q , 1 - q ]$ for $g \geq 1$ , so $\varphi _ { 0 } \varphi _ { 1 } , \psi _ { 0 } \psi _ { 1 } \geq q ( 1 - q )$ and det $P = \varphi _ { 0 } \varphi _ { 1 } \psi _ { 0 } \psi _ { 1 }$ det $Q / Z ^ { \bar { 2 } } \geq q ^ { 2 } ( 1 - q ) ^ { 2 } ( 1 - 2 q )$ . As

P differs from the product of its marginals by det $P { \left( \begin{array} { l l } { 1 } & { - 1 } \\ { - 1 } & { 1 } \end{array} \right) }$ , at total variation 2 det P, Pinsker’s inequality gives $I ( x _ { t } ; x _ { t + 1 } \mid x _ { z } = \xi ) \geq 8 ( \operatorname * { d e t } P ) ^ { 2 } \geq c _ { q } { \mathrm { , } }$ so $\Delta F ( i ) \ge \mathbb { P } ( G ) c _ { q } = c _ { q } / N$ Summing over the steps. As $r _ { a } ( i ) - i$ runs through $0 , \ldots , g _ { k } - 1$ on the k-th gap and $\textstyle \sum _ { k } g _ { k } = N$ Cauchy-Schwarz gives

$$
\mathbb { E } _ { \sigma } \operatorname { K L } ( \pi \| p ^ { \sigma } ) \geq \frac { c _ { q } } { N } \sum _ { k = 1 } ^ { T } \frac { g _ { k } ( g _ { k } - 1 ) } { 2 } \geq \frac { c _ { q } } { 2 N } \left( \frac { N ^ { 2 } } { T } - N \right) = \frac { c _ { q } } { 2 } \left( \frac { N } { T } - 1 \right) .
$$

Averaging over the step sizes and applying Jensen’s inequality to the convex map $T \mapsto N / T$ gives $\begin{array} { r } { \varepsilon \geq \frac { c _ { q } } { 2 } \left( \frac { N } { \mathbb { E } [ T ] } - 1 \right) } \end{array}$ , that $\mathrm { i s } , \mathbb { E } [ T ] \geq c _ { q } N / ( c _ { q } + 2 \varepsilon )$ □

Block-autoregressive decoding generates blocks left to right and runs Algorithm 1 inside each block; the chain rule splits its KL exactly over blocks, so part (i) of Corollary 1 applies to the mixture of the within-block weightings and part (ii) applies inside each block.

ProofofCorollary 3. For (i), let $C _ { k }$ be the k-th block, $p _ { k } ( \cdot \ | \ x _ { C _ { < k } } )$ the law of the block that Algorithm 1 generates on it given the prefix, and $\omega _ { \nu _ { k } }$ the weighting of Proposition 3 for its withinblock strategy $\nu _ { k }$ . The induced weighting $\omega _ { B }$ draws $\ddot { J } \sim \operatorname { U n i f } ( [ m ] )$ and $( i , \bar { z } ^ { \prime } ) \sim \omega _ { \nu _ { J } }$ with $z ^ { \prime } \subseteq C _ { J }$ and sets $z = C _ { < J } \cup z ^ { \prime } ;$ it is a masking distribution because each $\nu _ { k }$ is x-independent. By the chain rule,

$$
{ \mathrm { K L } } ( { \boldsymbol \pi } \| p _ { \mathrm { a l g } } ) = \sum _ { k = 1 } ^ { m } \mathbb { E } _ { \boldsymbol { x } _ { C _ { < k } } \sim \boldsymbol { \pi } } [ { \mathrm { K L } } ( { \boldsymbol \pi } ( \boldsymbol { x } _ { C _ { k } } \in \cdot \mid \boldsymbol { x } _ { C _ { < k } } ) \| p _ { k } ( \cdot \mid \boldsymbol { x } _ { C _ { < k } } ) ) ] ,
$$

and by (9) and Proposition 3 for block $k ' s$ run the k-th term is at most $\mathbb { E } _ { \pi } [ \mathcal { E } _ { \mathrm { f a c t } } ^ { ( k ) } ] + B \mathbb { E } _ { \pi } [ \mathcal { E } ^ { ( k ) } ( \hat { f } ) ]$ with $\mathcal { E } ^ { ( k ) } ( \hat { f } )$ the excess of $\hat { f } \mathrm { ~ a t ~ } \omega _ { \nu _ { k } }$ given the prefix. By (2), $\begin{array} { r } { B \sum _ { k } \mathbb { E } _ { \pi } [ \mathcal { E } ^ { ( k ) } ( \hat { f } ) ] = B m \mathcal { E } _ { \omega _ { B } } ( \hat { f } ) = } \end{array}$ $N \mathcal { E } _ { \omega _ { B } } ( \hat { f } )$ , which Theorem 1(a) bounds by $N \varepsilon ^ { \prime }$ with probability at least $1 - \delta$

For (ii), let $\boldsymbol { b } = \mathrm { m i n } \{ b ^ { * } ( \varepsilon ) , \boldsymbol { B } \}$ . Applying (17) inside each block, with the prefix added to the visible set and the expectation taken jointly over it, block k contributes $B \bar { \ } - \ \boldsymbol { b }$ summands, so $\mathcal { E } _ { \mathrm { f a c t } } ^ { \mathrm { b l k } } \leq m ( B - b ) \eta ( \bar { \kappa } ( b - 1 ) ) \leq N \eta ( \bar { \kappa } ( b - \mathrm { \bar { 1 } } ) )$ . If $b ^ { * } ( \varepsilon ) \leq B$ , the chain following (19) bounds this by ε with $T = m b = \left( N / B \right) b ^ { \ast } ( \varepsilon )$ steps; otherwise $b = B$ , so $\mathcal { E } _ { \mathrm { f a c t } } ^ { \mathrm { b l k } } = 0$ and $T = m B = N$ □

## B Experimental details

## B.1 Setup

Benchmarks. All text experiments use the OpenWebText dataset with GPT-2 tokenizer. Training draws from the first 7.9M documents, and evaluation uses the last 100K documents. The block-size experiment (Section B.3) uses a synthetic order-3 Markov source over four symbols with exact conditionals, so held-out excess and generation KL are measured against the exact conditionals and entropy.

Models. We finetune MDLM [Sahoo et al., 2024], and train from scratch a looped variant of MDLM (Section B.5) and six small block-AR models on a synthetic source (Section B.3). Table 2 lists the training settings; each experiment’s subsection gives its architecture. All experiments ran on a single NVIDIA A100-SXM4-40GB; the full set of experiments took ∼ 130 GPU-hours.

## Metrics. We measure

• Influence coefficient: $\hat { \kappa } ( d )$ is the TV between a model’s conditional at a masked position under two contexts that only differ beyond distance d of the position

• Forward sampling error: the teacher-forced CE $\begin{array} { r l r l } { \frac { 1 } { N } \mathbb { E } _ { \pi } [ - \log p _ { \mathrm { a l g } } ( x ) ] } & { { } = } & { } \end{array}$ $\begin{array} { r } { \frac { 1 } { N } \left( H ( \pi ) + \mathrm { K L } ( \pi \| p _ { \mathrm { a l g } } ) \right) } \end{array}$ of a decoding schedule, whose KL term is the quantity we bound

• Generation quality: the mean per-sequence NLL of generated text, MAUVE [Pillutla et al., 2021] against held-out text, and masked NELBO

• Held-out excess and generation KL: the excess risk and $\mathrm { K L } ( \pi \| p _ { \mathrm { a l g } } )$ on the synthetic source

Table 2: Training settings of the models trained in this paper. Weight decay for the looped model applies to matrices only. The dilated finetune and the uniform control share the same drawn training windows and differ in the masking law and loss target alone.
<table><tr><td></td><td>dilated finetune</td><td>uniform control</td><td>looped model</td><td>block models</td></tr><tr><td>Initialization</td><td>released MDLM</td><td>released MDLM</td><td>scratch, seed 0</td><td>scratch, seed 0, shared across B</td></tr><tr><td>Data</td><td>OWT train, seed 0</td><td>same windows as finetune</td><td>OWT train</td><td>synthetic, 20,000</td></tr><tr><td>Context</td><td>1024</td><td>1024</td><td>1024</td><td>seqs 256</td></tr><tr><td>Tokens / steps</td><td>0.25B / 7630</td><td>0.25B / 7630</td><td>4B / 7630</td><td>plateau rule, 1500– 8000 steps</td></tr><tr><td>Batch (sequences) 32</td><td></td><td>32</td><td>512</td><td>256</td></tr><tr><td>Optimizer</td><td>AdamW (0.9,0.999)</td><td>AdamW (0.9,0.999)</td><td>AdamW (0.9, 0.95)</td><td>AdamW (0.9, 0.999)</td></tr><tr><td>Learning rate</td><td>10-5</td><td>10-5</td><td>3 × 10−4</td><td>3 × 10−4</td></tr><tr><td>Warmup, then</td><td>100, constant</td><td>100, constant</td><td>2500, constant</td><td>500, constant</td></tr><tr><td>Weight decay</td><td>0</td><td>0</td><td>0.1</td><td>0.01</td></tr><tr><td>Gradient clip</td><td>1.0</td><td>1.0</td><td>1.0</td><td>none</td></tr><tr><td>Dropout</td><td>0.1</td><td>0.1</td><td>0.1</td><td>none</td></tr><tr><td>Precision Masking law</td><td>bf16, fp32 loss dilated, b</td><td>bf16, fp32 loss ∈</td><td>bf16, fp32 head</td><td>fp32</td></tr><tr><td></td><td>{8, 16, 32, 64}</td><td></td><td>uniform t ∼ U(0, 1) uniform, loop Λ ∼ block-B prefix U{1..12}</td><td></td></tr><tr><td>Loss</td><td>k</td><td>unweighted CE, class unweighted CE, all ELBO 1/t masked</td><td></td><td>unweighted CE, block k</td></tr></table>

## B.2 Influence measurement

We measure how strongly visible tokens beyond distance d move a model’s conditional at a masked position, as an estimate of the influence coefficients of Assumptions 1 and 2, and the dilated factorization term <sup>ˆ</sup>I, which sets the threshold for the decoding experiments. We ask whether mean influence decays exponentially or as a power law, and whether the worst case decays.

Setting. We use the released MDLM checkpoint [Sahoo et al., 2024] (12 blocks, hidden size 768, 128-dimensional conditioning pathway, context 1024) over the GPT-2 vocabulary with a mask token. The checkpoint ignores its time input, so no time conditioning enters the measurement. Each measurement uses two distinct windows drawn uniformly from held-out OWT windows of length 1024. <sup>ˆ</sup>I is estimated on 192 windows of the eval stream.

Procedure. We draw two sequences x, y. For a masked target position i under a visibility pattern $z ,$ let $x ^ { \prime }$ equal x at every unmasked position within d of i and y beyond. The influence at distance d is the total variation between the model’s two conditionals at i,

$$
{ \hat { \kappa } } ( d ) = \mathrm { T V } \left( f ( \cdot \mid x _ { z } ) , f ( \cdot \mid x _ { z } ^ { \prime } ) \right)\tag{20}
$$

computed exactly from the token logits. We make the target i uniform on [64, 961) so that the window around it lies within the context; visibility is i.i.d. Bernoulli(v) for $v \in \{ \bar { 0 } . 2 5 , 0 . 5 , 0 . 9 \}$ and the target masked, and we estimate for distances $\dot { d } \in \{ 1 , 2 , 4 , 8 , 1 6 , 2 4 , 3 2 , 4 8 , 6 4 \}$ . Each (d, v) cell uses 1000 sampled configurations, and we record the mean and 95th percentile of Equation (20) per cell.

The decay scale $\hat { \tau } _ { c }$ and prefactor $\widehat { C _ { 0 } }$ come from an exponential least-squares fit of the per-distance means on the widest contiguous window of distances.

For a dilated step at spacing b with commit set S and previously revealed classes z, we estimate the per-summand factorization term by $\hat { I } ( b )$ , the mean of

$$
\log f \left( x _ { i } \mid x _ { z } , x _ { S \setminus \{ i \} } \right) - \log f \left( x _ { i } \mid x _ { z } \right)\tag{21}
$$

over the 192 held-out windows and over tokens $i \in S$ committed after other tokens of the same step; it estimates the per-summand mutual information $I ( x _ { i } ; x _ { S \setminus \{ i \} } \mid x _ { z } )$ that the proof of Corollary 1(ii) bounds. The random-order value uses commit sets of the same size. Its interval is a 95% percentile bootstrap interval from 1000 resamples of the windows; “indistinguishable from zero” means the interval contains zero.

Table 3: Influence measurement on MDLM’s conditionals.
<table><tr><td> $R ^ { 2 } ,$  mean decay exponential fit , mean decay power-law fit</td><td>0.989</td></tr><tr><td> $R ^ { 2 }$ </td><td>0.966</td></tr><tr><td> $( \widehat { \tau } _ { c } , \widehat { C _ { 0 } } )$  on the tail window; interval for  $\hat { \tau } _ { c }$ </td><td>(81.1, 1); [43, 95]</td></tr><tr><td>local decay scale over the first distances</td><td>≈26</td></tr><tr><td>95th percentile of  at  $d = 1 / d = 6 4$ </td><td>0.999 / 0.722</td></tr><tr><td> ${ \hat { I } } ( 1 6 )$  , dilated: point/upper interval edge (per summand)</td><td>0.0036 / 0.042</td></tr><tr><td>Î(64), random order (per summand)</td><td>0.087</td></tr></table>

Results. The exponential fits the mean decay better than the power law (Table 3), with decay scale 81.1 on the tail window and interval [43, 95]; the first few distances decay faster at a local scale near 26. The 95th percentile falls only from 0.999 at d = 1 to $0 . 7 2 2$ at $d = 6 4$ . The dilated $\hat { I }$ is indistinguishable from zero by spacing 16 while the random-order interval still excludes zero at the commit size of spacing 64. The spacing (19) of Corollary 1(ii) at these estimates is $b ^ { * } ( 1 ) \approx 8 6 0$ at $N = 1 0 2 4 .$

On the model’s conditionals, averaged influence decays fast and the worst case does not; Theorem 1(a) and Corollary 1 rely only on the averaged behavior. Against the observed dilated saturation points (9.7 to 32 steps), $b ^ { * } ( 1 ) \approx 8 6 0$ is larger by $2 7 { - } 8 9 \times$ . The exchange construction measures the released model’s conditionals, which stand in for the data’s conditionals that the theory concerns.

## B.3 Block size on an exactly computable source

We train block-AR models at six block sizes on a synthetic source with exact conditionals, then measure held-out excess and generation KL. Corollary 3(ii) predicts generation KL flat in the block size B above the threshold $4 \bar { b ^ { * } } ( 1 )$ .

Setting. Each model is a four-layer pre-norm bidirectional transformer of width 128, GELU feed-forward block of width 512, and learned position embeddings, over a five-symbol vocabulary (four source symbols and a mask token) at context 256. The six models differ only in $B \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 $ , 128} and have the same initialization.

The source is a stationary order-3 Markov chain over the four symbols whose conditional at each context is an equal mixture of uniform and Dir(2, 2, 2, 2). This yields $q _ { \mathrm { m i n } } = 0$ .136 and, over a restricted family of masks, influence 0.370, 0.148, 0.053, 0.020 for $j = 1 , \dotsc , 4$ and below $1 0 ^ { - 7 }$ by $j = 1 6$ , fit by $\tau _ { c } = 0 . 9 7 7$ $C _ { 0 } = 2 \AA$ , which gives $b ^ { * } ( 1 ) = 1 0$ and the threshold $4 b ^ { * } = 4 0$

Procedure. We draw a block index k uniformly on $\{ 0 , \ldots , N / B - 1 \}$ , reveal blocks before k, mask block k i.i.d. Bernoulli(t) with t uniform on [0, 1), and take the mean cross-entropy on the masked positions of block k. Decoding runs for $T = \dot { N } / \dot { 4 }$ steps at every B with held-out excess measured against the true conditionals and generation KL the mean of $- \log p _ { \mathrm { a l g } }$ over 4096 sequences drawn from the source π, minus the exact window entropy.

Results. Held-out excess is at most 0.031 nats at every B; generation KL is flat at about $N \hat { \varepsilon } ^ { \prime } .$ , with $\hat { \varepsilon } ^ { \prime }$ the held-out excess, from $B = 8$ up and shows a factorization penalty only at $B = 4 ;$ the observed saturation is near $B = 7$ , against the computed window [20, 80] of half to twice the threshold 40.

The comparison isolates factorization, and the computed window is conservative by about $5 \times$

## B.4 Decoding steps

The factorization error of dilated decoding at spacing b is at most N − b times the per-summand mutual information that ${ \hat { I } } ( b )$ estimates (Section B.2), so Corollary 1(ii) predicts that dilated decoding saturates within [8, 32], half to twice the spacing 16 at which <sup>ˆ</sup>I vanishes, and Corollary 2 predicts that random order saturates later. We test both predictions on the forward sampling error (Section B.4.1) and on generated samples (Section B.4.2).

## B.4.1 Forward sampling error

We score text under each decoding schedule, measuring the forward sampling error $\begin{array} { r } { \frac { 1 } { N } \mathbb { E } _ { \pi } [ - \log p _ { \mathrm { a l g } } ( x ) ] } \end{array}$ without generating samples. For random order we score one fixed random order per window, the per-order error of Corollary 2; the order-averaged generator has lower crossentropy by convexity.

Setting. We score MDLM and two finetunes of MDLM on 0.25B tokens from OWT (7630 steps of 32 sequences), one with dilated masking and the other with standard uniform masking, using the same data windows as in Section B.4.2 except all table cells share the same 512 windows.

Procedure. Let $S _ { 1 } , \cdots , S _ { T }$ be a schedule’s commit sets and $z _ { k } = S _ { 1 } \cup \cdot \cdot \cdot \cup S _ { k - 1 }$ the positions visible before step k. Then, the score of a window after T steps is

$$
\operatorname { C E } ( T ) = - { \frac { 1 } { N } } \sum _ { k = 1 } ^ { T } \sum _ { i \in S _ { k } } \log f \left( x _ { i } \mid x _ { z _ { k } } \right)\tag{22}
$$

where $f$ is the model’s masked conditional. We score for T ∈ {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128} with 95% bootstrap intervals over sequences from 1000 resamples. We calculate saturation as

$$
T _ { \delta } = \operatorname* { m i n } \left\{ T : \operatorname { C E } ( T ) - \operatorname { C E } ( 1 2 8 ) \leq \delta \right\}\tag{23}
$$

where the tolerance $\delta = 0 . 0 3$ is the median half-width of the cells’ intervals so that $T _ { \delta }$ is the first count at which a curve is within resolution of its value at the largest count.

Table 4: Forward sampling error at all step counts for the six model-schedule pairs.
<table><tr><td>T</td><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>24</td><td>32</td><td>48</td><td>64</td><td>96</td><td>128</td></tr><tr><td>released, dilated</td><td>7.556</td><td>4.654</td><td>3.703</td><td>3.381</td><td>3.260</td><td>3.241</td><td>3.230</td><td>3.235</td><td>3.219</td><td>3.229</td><td>3.223</td></tr><tr><td>released, random</td><td>7.556</td><td>5.210</td><td>4.192</td><td>3.708</td><td>3.482</td><td>3.409</td><td>3.376</td><td>3.339</td><td>3.324</td><td>3.292</td><td>3.286</td></tr><tr><td>dilated finetune, dilated</td><td>7.557</td><td>4.633</td><td>3.685</td><td>3.371</td><td>3.256</td><td>3.226</td><td>3.212</td><td>3.200</td><td>3.194</td><td>3.190</td><td>3.188</td></tr><tr><td>dilated finetune, random</td><td>7.557</td><td>5.176</td><td>4.146</td><td>3.666</td><td>3.439</td><td>3.368</td><td>3.333</td><td>3.296</td><td>3.280</td><td>3.263</td><td>3.257</td></tr><tr><td>uniform control, dilated</td><td>7.556</td><td>4.618</td><td>3.664</td><td>3.347</td><td>3.231</td><td>3.201</td><td>3.186</td><td>3.174</td><td>3.167</td><td>3.164</td><td>3.161</td></tr><tr><td>uniform control, random</td><td>7.556</td><td>5.162</td><td>4.130</td><td>3.644</td><td>3.415</td><td>3.344</td><td>3.309</td><td>3.272</td><td>3.256</td><td>3.236</td><td>3.230</td></tr></table>

Results. Dilated decoding lies below random order at every matched step count $T \geq 2$ (Table 4 and Figure 2), and below random order at twice the step count for $T \geq 4$ on the released model and $T \geq 8$ on the finetunes. The two schedules reach different floors of 3.223 vs. 3.286 at $T = 1 2 8$ Dilated decoding saturates at $T _ { \delta } = 2 4$ (interval [24, 32]) on the released model and at 32 on both finetunes, against 96 and 64 for random order: inside the predicted band $[ 8 , 3 2 ]$ , with two to four times fewer steps.  
![](images/410e30f457a0e4d0af44ec5d6d9bd18b3f624b61a87b16f2f4e97cc853faab7f.jpg)

![](images/ed982a1dfc989625262d15453ad7a4c2df5ef6cfc8e32bbd33957ef4ae02fcd2.jpg)  
Figure 2: Forward sampling error against decoding steps: dilated order on the released model and both finetunes, and random order on the released model; values from Table 4. Left: all step counts; right: zoom on $T \geq 8$

## B.4.2 Generation metrics

We generate text from MDLM at 18 step counts under three decoding orders and from LLaDA-8B at 8 step counts under the dilated and random orders at matched per-step commit sizes, score the generations, and fit the step count at which quality saturates.

Setting. We use the released MDLM checkpoint and LLaDA-8B-Base. We score generation with GPT-2-large; MAUVE also uses GPT-2-large features. Generation is unconditional at $N = 1 0 2 4$ The MAUVE reference set is the first 512 windows of the eval stream. Because LLaDA-8B uses a different tokenizer, its outputs are detokenized, retokenized to GPT-2, and truncated to 1024 tokens for scoring.

Procedure. At each step, we sample every committed position from the model’s categorical with no truncation, in fp32 precision. The per-step commit counts are the dilated schedule’s group sizes of $\lfloor N / T \rfloor$ at each order. We compare three decoding orders: dilated, random, and deterministic confidence-greedy [Chang et al., 2022]. For MDLM we evaluate the 14 step counts $T \le 1 2 8$ of its sweep plus {192, 256, 384, 512}. The quality is the mean per-sequence NLL under GPT-2-large with out-of-vocab tokens clamped to EOS; we measure generation entropy and MAUVE [Pillutla et al., 2021] (mauve-text 0.40). We use the saturation estimator

$$
\hat { T } ^ { * } = 2 ^ { ( a - \ell _ { \mathrm { f l o o r } } ) / c } , \qquad ( a , c , \ell _ { \mathrm { f l o o r } } ) = \arg \operatorname* { m i n } _ { T \leq 1 2 8 } \left( \ell ( T ) - \operatorname* { m a x } ( a - c \log _ { 2 } T , \ell _ { \mathrm { f l o o r } } ) \right) ^ { 2 }\tag{24}
$$

where the interval comes from resampling the points and refitting. LLaDA-8B generates at eight counts in $\{ 1 , \cdots , 1 2 8 \}$ .

Table 5: Generation quality on the released MDLM checkpoint at every step count of the sweep and the extended grid: mean per-sequence NLL under GPT-2-large (nats per token, 95% bootstrap intervals over sequences) and MAUVE against the held-out reference, for the three decoding orders at matched per-step commit sizes.
<table><tr><td rowspan=1 colspan=10>dilated                    random                  confidenceT        NLL      MAUVE      NLL      MAUVE      NLL      MAUVE</td></tr><tr><td rowspan=1 colspan=2>1   8.460</td><td rowspan=1 colspan=1>[8.45, 8.47]</td><td rowspan=1 colspan=3>0.005 8.461[8.45,8.47]</td><td rowspan=1 colspan=2>0.006 8.460 [8</td><td rowspan=1 colspan=1>.45, 8.47]</td><td rowspan=1 colspan=1>0.006</td></tr><tr><td rowspan=1 colspan=2>2  7.420[</td><td rowspan=1 colspan=1>7.41, 7.43]</td><td rowspan=1 colspan=2>0.022 8.132[</td><td rowspan=1 colspan=1>8.12, 8.14]</td><td rowspan=1 colspan=2>0.007 8.557 [8</td><td rowspan=1 colspan=1>.55, 8.57]</td><td rowspan=1 colspan=1>0.005</td></tr><tr><td rowspan=1 colspan=2>3  6.636</td><td rowspan=1 colspan=1>[6.62, 6.65]</td><td rowspan=1 colspan=2>0.068 7.841[</td><td rowspan=1 colspan=1>7.83, 7.85]</td><td rowspan=1 colspan=2>0.007 8.554 [8</td><td rowspan=1 colspan=1>.54, 8.56]</td><td rowspan=1 colspan=1>0.005</td></tr><tr><td rowspan=1 colspan=2>4  6.116[</td><td rowspan=1 colspan=1>6.10, 6.13]</td><td rowspan=1 colspan=2>0.158 7.583[</td><td rowspan=1 colspan=1>7.57, 7.60]</td><td rowspan=1 colspan=2>0.011 8.539[</td><td rowspan=1 colspan=1>8.53, 8.55]</td><td rowspan=1 colspan=1>0.006</td></tr><tr><td rowspan=1 colspan=2>6  5.652[</td><td rowspan=1 colspan=1>5.63, 5.67]</td><td rowspan=1 colspan=2>0.341 7.147[</td><td rowspan=1 colspan=1>7.13, 7.17]</td><td rowspan=1 colspan=2>0.007 8.512</td><td rowspan=1 colspan=1>[8.50, 8.53]</td><td rowspan=1 colspan=1>0.005</td></tr><tr><td rowspan=1 colspan=2>8  5.387[</td><td rowspan=1 colspan=1>5.37, 5.41]</td><td rowspan=1 colspan=1>0.443</td><td rowspan=1 colspan=1>6.698</td><td rowspan=1 colspan=1>[6.67, 6.72]</td><td rowspan=1 colspan=2>0.020 8.480 [8</td><td rowspan=1 colspan=1>.46, 8.50]</td><td rowspan=1 colspan=1>0.006</td></tr><tr><td rowspan=1 colspan=2>12 5.287[</td><td rowspan=1 colspan=1>5.26, 5.31]</td><td rowspan=1 colspan=1>0.580</td><td rowspan=1 colspan=1>6.123[</td><td rowspan=1 colspan=1>6.10, 6.15]</td><td rowspan=1 colspan=2>0.010 8.400[</td><td rowspan=1 colspan=1>8.38, 8.42]</td><td rowspan=1 colspan=1>0.006</td></tr><tr><td rowspan=1 colspan=2>16 5.014[</td><td rowspan=1 colspan=1>4.99, 5.04]</td><td rowspan=1 colspan=2>0.512 5.800[</td><td rowspan=1 colspan=1>5.77, 5.83]</td><td rowspan=1 colspan=2>0.015 8.316</td><td rowspan=1 colspan=1>[8.29, 8.34]</td><td rowspan=1 colspan=1>0.006</td></tr><tr><td rowspan=1 colspan=2>24 4.920[</td><td rowspan=1 colspan=1>4.89, 4.95]</td><td rowspan=1 colspan=2>0.615 5.468[</td><td rowspan=1 colspan=1>5.44, 5.49]</td><td rowspan=1 colspan=2>0.034 8.105</td><td rowspan=1 colspan=1>[8.07, 8.14]</td><td rowspan=1 colspan=1>0.004</td></tr><tr><td rowspan=1 colspan=2>32 4.883[</td><td rowspan=1 colspan=1>4.85, 4.92]</td><td rowspan=1 colspan=2>0.439 5.310[</td><td rowspan=1 colspan=1>5.29, 5.34]</td><td rowspan=1 colspan=2>0.016 7.920[</td><td rowspan=1 colspan=1>7.88, 7.96]</td><td rowspan=1 colspan=1>0.005</td></tr><tr><td rowspan=1 colspan=2>48 4.922[</td><td rowspan=1 colspan=1>4.89, 4.96]</td><td rowspan=1 colspan=1>0.506</td><td rowspan=1 colspan=1>5.081[</td><td rowspan=1 colspan=1>5.05, 5.11]</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>7.782[</td><td rowspan=1 colspan=1>7.74, 7.83]</td><td rowspan=1 colspan=1>0.006</td></tr><tr><td rowspan=1 colspan=2>64 4.835[</td><td rowspan=1 colspan=1>4.79, 4.88]</td><td rowspan=1 colspan=1>0.251</td><td rowspan=1 colspan=1>4.947[</td><td rowspan=1 colspan=1>4.92, 4.97]</td><td rowspan=1 colspan=1>0.016</td><td rowspan=1 colspan=1>7.626</td><td rowspan=1 colspan=1>[7.57, 7.68]</td><td rowspan=1 colspan=1>0.006</td></tr><tr><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>4.894[</td><td rowspan=1 colspan=1>4.84, 4.94]</td><td rowspan=1 colspan=1>0.257</td><td rowspan=1 colspan=1>4.906[</td><td rowspan=1 colspan=1>4.88, 4.93]</td><td rowspan=1 colspan=1>0.024</td><td rowspan=1 colspan=1>7.390</td><td rowspan=1 colspan=1>[7.32, 7.46]</td><td rowspan=1 colspan=1>0.005</td></tr><tr><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>4.785[</td><td rowspan=1 colspan=1>4.73, 4.83]</td><td rowspan=1 colspan=1>0.167</td><td rowspan=1 colspan=1>4.771[</td><td rowspan=1 colspan=1>4.74, 4.80]</td><td rowspan=1 colspan=1>0.013</td><td rowspan=1 colspan=1>7.096 [7</td><td rowspan=1 colspan=1>.00, 7.19]</td><td rowspan=1 colspan=1>0.008</td></tr><tr><td rowspan=1 colspan=1>192</td><td rowspan=1 colspan=1>4.801[</td><td rowspan=1 colspan=1>4.73, 4.87]</td><td rowspan=1 colspan=1>0.346</td><td rowspan=1 colspan=1>4.747[</td><td rowspan=1 colspan=1>4.72, 4.77]</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>6.675</td><td rowspan=1 colspan=1>[6.55, 6.80]</td><td rowspan=1 colspan=1>0.007</td></tr><tr><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>4.579[</td><td rowspan=1 colspan=1>4.51, 4.66]</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>4.734[</td><td rowspan=1 colspan=1>4.71, 4.76]</td><td rowspan=1 colspan=1>0.016</td><td rowspan=1 colspan=1>5.990</td><td rowspan=1 colspan=1>[5.83, 6.15]</td><td rowspan=1 colspan=1>0.009</td></tr><tr><td rowspan=1 colspan=1>384</td><td rowspan=1 colspan=1>4.435[</td><td rowspan=1 colspan=1>4.34, 4.54]</td><td rowspan=1 colspan=1>0.074</td><td rowspan=1 colspan=1>4.690</td><td rowspan=1 colspan=1>[4.67, 4.72]</td><td rowspan=1 colspan=2>0.014 5.306 [5</td><td rowspan=1 colspan=1>.12, 5.49]</td><td rowspan=1 colspan=1>0.009</td></tr><tr><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>4.079</td><td rowspan=1 colspan=1>[3.96, 4.20]</td><td rowspan=1 colspan=1>0.021</td><td rowspan=1 colspan=1>4.661 [4</td><td rowspan=1 colspan=1>.63, 4.69]</td><td rowspan=1 colspan=2>0.011 4.127 [3</td><td rowspan=1 colspan=1>.92, 4.33]</td><td rowspan=1 colspan=1>0.013</td></tr></table>

Results. Every order’s NLL reaches its minimum at 512 steps while dilated’s MAUVE peaks at 0.61 at 24 steps and falls through 0.25 at 64 and 0.17 at 128 to 0.02 at 512. The confidence order’s MAUVE never exceeds 0.013; generation entropy falls together with MAUVE, and at 512 steps dilated and the confidence order have overlapping NLL intervals (4.079 against 4.127). LLaDA-8B improves monotonically across its tested range without observed saturation. The dilated fit saturates at $\hat { T } ^ { * } = 9 . 7$ steps, inside the predicted band [8, 32] and before random order’s 44.7, so both generation-side predictions hold.

## B.5 Loop-count saturation

We train a looped variant of MDLM from scratch, evaluate held-out loss at every loop count at three training checkpoints, and fit the loop count at which loss saturates. This experiment is qualitative and incomplete: Corollary 4 concerns the idealized message-passing class of Section 6.2 and suggests this saturation point grows as training reduces the achievable excess.

Setting. We train a looped variant of MDLM with 12 stored blocks at hidden size 768 with a 128-dimensional conditioning pathway, of which block 0 runs once as the head, blocks 1 and 2 are shared and applied Λ times, and blocks 3 to 11 run once; the model has 170M parameters, context 1024, and the GPT-2 vocabulary with a mask token. We train on the OWT training region and evaluate on 1024-token windows from the evaluation stream.

Procedure. We trained for 4B tokens (7630 steps of 512 sequences) with checkpoints at 1, 2, and 4B, deliberately short of convergence to track behavior over training. The loop count was drawn uniformly on $\{ 1 , 2 , \cdots , 1 2 \}$ per microbatch and the loss is the ELBO’s 1/t-weighted masked CE with t clamped at $1 0 ^ { - 3 }$ . We used the same masks across choices of Λ. We fit the saturation point

$$
\hat { \Lambda } ^ { * } = \frac { a - \ell _ { \mathrm { H o o r } } } { c } , \qquad ( a , c , \ell _ { \mathrm { H o o r } } ) = \arg \operatorname* { m i n } \sum _ { \Lambda } \left( \ell ( \Lambda ) - \operatorname* { m a x } ( a - c \Lambda , \ell _ { \mathrm { H o o r } } ) \right) ^ { 2 }\tag{25}
$$

Results. Table 6 gives the fitted saturation points of $2 . 4 1 9  2 . 5 2 8  2 . 6 2 1$ . The extrapolated 4B $\hat { \Lambda } ^ { * }$ of 2.637 [2.60, 2.67] contains the measured 2.621 [2.608, 2.635]. Excluding $\Lambda = 1$ moves all three points right by close to 1 with ordering and spacing preserved; past the loop count minimizer for each checkpoint, the loss rises by 0.028. The saturation point moves right with training.

Table 6: Loop-count saturation at the three checkpoints: the fitted saturation point on the full grid, the refit excluding $\Lambda = 1$ , and the loop count at which held-out loss is lowest.
<table><tr><td>checkpoint</td><td> $\hat { \Lambda } ^ { * }$  (all counts)</td><td> $\hat { \Lambda } ^ { * } \left( \Lambda \geq 2 \right)$ </td><td>loss minimum at</td></tr><tr><td>1B</td><td>2.419</td><td>3.16</td><td> $\Lambda = 6$ </td></tr><tr><td>2B</td><td>2.528</td><td>3.42</td><td> $\Lambda = 6$ </td></tr><tr><td>4B</td><td>2.621 [2.608, 2.635]</td><td>3.61</td><td> $\Lambda = 7$ </td></tr></table>