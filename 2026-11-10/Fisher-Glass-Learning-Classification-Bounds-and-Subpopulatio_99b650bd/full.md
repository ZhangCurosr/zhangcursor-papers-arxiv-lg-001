# Fisher-Glass Learning: Classification Bounds and Subpopulation Audits

El Mustapha Mansouri School of Engineering, Institute of Science Tokyo Yokohama, Kanagawa 226-8501, Japan mansouri.e.2224@m.isct.ac.jp

## Abstract

Accurate classifiers can fail when background cues change without altering the correct labels. Predictive performance alone does not quantify how much task information remains after accounting for such nuisance variation. Here we extend the Fisher-glass framework to connect nuisance-adjusted information with local classification difficulty and subpopulation auditing. We derive lower bounds on local minimax excess classification risk under explicit distinguishability and decisionseparation conditions, alongside statistical audits accounting for estimation error and regularization. On Waterbirds, a feature-based diagnostic of task–background alignment selects an accurately classified group, with selection frozen before heldout evaluation. Across five training seeds, background replacements preserving the extracted bird foreground and labels yield 93.2% accuracy when background type is preserved and 40.6% when it changes, a controlled difference of 52.6 percentage points. Separately, search using task labels and a fixed nuisance direction identifies an independently evaluated population with 98.8% membership-weighted accuracy and low estimated projected predictive Fisher information. Training and acquisition experiments show that information gains and predictive gains can diverge. These findings support joint assessment of predictive performance, information under declared nuisance models and controlled sensitivity when diagnosing vulnerability and evaluating model improvements.

## 1 Introduction

A classifier can predict observed labels accurately while the data weakly distinguish a specified task effect from admissible nuisance variation. This ambiguity matters for classification when statistically similar alternatives prescribe different labels. Autonomous bird monitoring provides a practical motivation: species classifiers operate under variable illumination and occlusion [1]. We ask when it creates a learning obstacle, and how to audit the remaining task information on predefined and learned populations. Group-robust learning and error-based reweighting emphasize subpopulation performance [2, 3]; our objective is to establish which information claims the data support and assess their relevance through controlled nuisance interventions (Figure 1).

Efficient-score projection removes task variation that nuisance scores can reproduce [4, 5]. The Fisher-glass framework of Mansouri and Arai [6] connects this geometry to inverse-information tails and a population certificate. We develop its classification consequence: decision separation turns limited distinguishability into excess-risk bounds. A worked regression family shows that constant raw information at a reference parameter can conceal an effective sample size set by the mass near the decision boundary. Statistical corrections then determine which population information statements survive estimation error, regularization, and adaptive selection.

![](images/54957bd3572538149b1b802cf69cc9e97e1d4b3b74d362fc6ceff96829052192.jpg)  
Figure 1: Background sensitivity despite high control accuracy. (A,C) A matched Waterbirds image pair preserves the extracted bird foreground and label while changing background type. For the selected waterbird/water group, mean accuracy across five training seeds falls from 93.2% with same-type backgrounds to 40.6% with opposite-type backgrounds, a drop of 52.6 percentage points. (B) Schematic score geometry: projection removes the nuisance-aligned component of the task score, leaving a small residual.

On Waterbirds, a frozen feature-based rule selects a group with 92.0% source accuracy and a 52.6-point controlled drop between same-type and opposite-type background replacements. The information, accuracy, and negative log-likelihood (NLL) selectors identify the same eligible group, whose controlled accuracy drop quantifies background sensitivity. Within that group, nuisance-derived coordinates improve species-held-out prediction of intervention failures beyond a specified confidence baseline. A separate search discovers an accurate population with low predictive information; it sensitivity to interventions remains untested.

Our contributions are:

• A classification consequence. We derive local excess-risk bounds with explicit decision separation and radius, and exhibit a classification family with matching minimax rates and an interpretable loss of effective sample size.

• Population information audits. We account for estimation error, ridge bias, membership normalization, and support correction, and evaluate frozen learned slices independently. The audit distinguishes supported information floors from uncertain or adaptively selected estimates.

• Behavioral evaluation. Controlled interventions and confidence-baseline comparisons test the relevance of nuisance-derived measurements. Training and acquisition evaluate whether information gains correspond to predictive gains, which need not occur together.

## 2 Task–Nuisance Geometry and Information Collapse

Let e index an observed group or latent subpopulation, and let Π be the declared distribution over these environments. The audit fixes the task contrast, admissible nuisance variations, and common task-coordinate normalization. Information describes the task distinction remaining after nuisance adjustment.

## 2.1 Task Information After Nuisance Profiling

Consider a local model $P _ { \theta , \mu , e }$ with task parameter θ and nuisance parameter $\mu . \ \mathrm { ~ A ~ }$ likelihood score measures the local change in log likelihood as a parameter varies. At $P _ { e } = P _ { 0 , 0 , e } ,$ assume differentiability in quadratic mean, with centered, square-integrable task-score vector $s _ { T , e }$ and nuisance-score vector $s _ { N , e }$ . Let $\mathcal { T } _ { N , e }$ be the closed span of admissible nuisance scores in $L ^ { 2 } ( P _ { e } )$ , with inner product $\langle f , g \rangle _ { e } = \mathbb { E } _ { P _ { e } } [ f g ]$ , and let $\Pi _  T _ { N } , $ denote orthogonal projection onto that span. The efficient task score and its information are

$$
\begin{array} { r l r } {  { s _ { T , e } ^ { \mathrm { e f f } } = s _ { T , e } - \Pi _ { \mathcal { T } _ { N , e } } s _ { T , e } , } } \\ & { } & { F _ { e } : = F _ { T | N , e } ^ { \mathrm { e f f } } = \mathbb { E } _ { P _ { e } } [ s _ { T , e } ^ { \mathrm { e f f } } ( s _ { T , e } ^ { \mathrm { e f f } } ) ^ { \top } ] = J _ { T T , e } - J _ { T N , e } J _ { N N , e } ^ { - 1 } J _ { N T , e } . } \end{array}\tag{1}
$$

The projection acts componentwise. The Schur expression applies when finite nuisance scores span $\mathcal { T } _ { N , e }$ and $J _ { N N , e } \succ 0$ , with $J _ { A B , e } = \mathbb { E } _ { P _ { e } } [ s _ { A , e } s _ { B . e } ^ { \top } ] \left[ 4 , 5 , 7 \right]$ . Write $\lambda _ { e } = \lambda _ { \mathrm { m i n } } ( F _ { e } )$ for the weakest task information. A zero eigenvalue identifies a task direction contained in the nuisance span.

For scalar task and nuisance scores with positive raw information, their Fisher correlation $\rho _ { e } =$ $J _ { T N , e } / \sqrt { J _ { T T , e } J _ { N N , e } }$ gives

$$
F _ { e } = J _ { T T , e } ( 1 - \rho _ { e } ^ { 2 } ) , ~ \underbrace { F _ { e } / J _ { T T , e } } _ { \mathrm { s u r v i v i n g - i n f o r m a t i o n } ~ \mathrm { f r a c t i o n } } = 1 - \rho _ { e } ^ { 2 } .\tag{2}
$$

Absolute information measures remaining curvature in the declared coordinates. The fraction measures the proportion retained, lies in $[ 0 , 1 ]$ , and is invariant to nonzero scalar rescaling of either score. It compares alignment across slices with different raw information; the prospective selection statistic uses a ridge-regularized feature–Schur analogue with fixed coordinate scales. The Fisher-Glass certificate summarizes absolute information across environments.

## 2.2 Task–Nuisance Alignment and Loss–Information Separation

Let $b _ { e } , r _ { \epsilon }$ be centered, orthogonal, unit-norm scores, with nuisance span span $\{ b _ { e } \}$ . For $\delta _ { e } \geq 0$ define the task score $a _ { e } = b _ { e } + \delta _ { e } r _ { e }$ . Then

$$
a _ { e } = b _ { e } + \delta _ { e } r _ { e } , \quad J _ { T T , e } = 1 + \delta _ { e } ^ { 2 } , \quad F _ { e } = \delta _ { e } ^ { 2 } , \quad \frac { F _ { e } } { J _ { T T , e } } = \frac { \delta _ { e } ^ { 2 } } { 1 + \delta _ { e } ^ { 2 } } .\tag{3}
$$

Projection removes $b _ { e }$ , leaving $\delta _ { e } r _ { e } ;$ raw information stays at least one as efficient information vanishes. Section 5.2 illustrates low observed loss alongside the corresponding feature-moment geometry. A classification cost additionally requires different optimal decisions, which the next section makes explicit. The inherited tail mechanism is illustrated in Appendix Figure A.1.

## 3 Classification Difficulty and Population Information

We study the statistical consequences of nuisance projection at two scales. Within an environment, local classification bounds and a worked family connect weak task information to learning difficulty under explicit distinguishability and decision-separation conditions. Across environments, geometry and probability mass near Fisher-null sets govern inverse-information tails, motivating a certificate of population information coverage.

## 3.1 Efficient Information and Local Classification Difficulty

Parameter ambiguity need not imply classification difficulty: probabilities 0.90 and 0.95 prescribe the same label. A lower bound requires statistically similar alternatives whose optimal decisions disagree on enough probability mass. Define classification risk $R _ { P } ( g ) = P \{ g ( \dot { X } ) \neq Y \}$ , Bayes risk $\mathsf { \bar { \Pi } } _ { R _ { P } ^ { \star } } ^ { - } = \operatorname* { i n f } _ { g } \bar { R } _ { P } ( g )$ , and excess risk $\mathcal { E } _ { P } ( g ) = R _ { P } ( g ) - R _ { P } ^ { \star }$ . For a specified local family, let $\Re _ { e } ( n )$ denote minimax expected excess classification risk based on n independent observations from environment $e _ { \cdot }$ . Write $\operatorname { K L } ( P \| Q )$ ) for the Kullback–Leibler divergence.

Theorem 1 (Local classification difficulty). Fix $\lambda _ { e } > 0$ and a unit weakest efficient-information direction $u _ { e }$ . For every $0 < t \leq t _ { 0 , e }$ , suppose the local family contains $P _ { e , + t } , P _ { e , - t }$ with task coordinates ± $t u _ { e }$ and nuisance coordinates allowed to vary. Assume, for constants $K _ { e } , c _ { e } , t _ { 0 , e } > 0$ and $q > 0$

$$
\begin{array} { r } { \mathrm { K L } ( P _ { e , + t } \| P _ { e , - t } ) \leq K _ { e } t ^ { 2 } \lambda _ { e } , \qquad \underset { g } { \operatorname* { i n f } } \{ \mathcal { E } _ { P _ { e , + t } } ( g ) + \mathcal { E } _ { P _ { e , - t } } ( g ) \} \geq c _ { e } t ^ { q } . } \end{array}\tag{4}
$$

Then,for a universal constant $c _ { 0 } > 0 ,$

$$
\Re _ { e } ( n ) \geq c _ { 0 } c _ { e } \operatorname* { m i n } \{ t _ { 0 , e } ^ { q } , ( n K _ { e } \lambda _ { e } ) ^ { - q / 2 } \} .\tag{5}
$$

The KL bound limits distinguishability; decision separation supplies the classification cost. Both must hold over radius $t _ { 0 , \epsilon }$ . With quadratic separation, the interior rate is $c _ { e } / ( n K _ { e } \lambda _ { e } )$ , capped by the local decision scale. Appendix $\bar { \mathbf { A . } 3 }$ proves the testing reduction; the next example calculates both conditions and a matching upper bound.

## 3.2 A Worked Classification Mechanism

Let $X \sim \mathrm { U n i f } [ - 1 , 1 ] , 0 < \delta \leq 1 / 2 , | \theta | , | \mu | \leq \delta / 8 .$ , and

$$
\begin{array} { r } { \eta _ { \theta , \mu } ( x ) : = P _ { \theta , \mu } ( Y = 1 \mid X = x ) = \frac { 1 } { 2 } + \frac { x } { 4 } + \theta + \mu \mathbf { 1 } \{ | x | > \delta \} . } \end{array}\tag{6}
$$

Here $\delta$ is boundary-region half-width and probability mass. The nuisance reproduces a task shift outside that region; the Bayes rule is $g _ { \theta } ^ { \star } ( x ) = \mathbf { 1 } \{ x \ge - 4 \theta \}$ throughout the family. At $( \theta , \mu ) = ( 0 , 0 )$

$$
J _ { T T } = 4 \log 3 , \qquad F _ { \delta } = 4 \log \frac { 2 + \delta } { 2 - \delta } \asymp \delta .\tag{7}
$$

The alternatives $( t , - t )$ and $\left( - t , t \right)$ agree outside $| X | \leq \delta ,$ , have decision separation $4 t ^ { 2 }$ , and satisfy $\mathrm { K L } ( P _ { + t } | | P _ { - t } ) \leq 5 F _ { \delta } \dot { t } ^ { 2 }$ for $0 < t \leq \delta / 8$ . Theorem 1 and a matching estimator give minimax risk over this parameter rectangle, uniformly in $n \geq 1$ and δ,

$$
\Re _ { \delta } ( n ) \asymp \operatorname* { m i n } \{ \delta ^ { 2 } , ( n \delta ) ^ { - 1 } \} .\tag{8}
$$

Approximately nδ observations distinguish the task effect from nuisance variation where the decision boundary can move. The $\delta ^ { 2 }$ cap retains the shrinking local decision scale. This statistical nuisance preserves Bayes labels at fixed θ; it does not model the physical background-swap operation. Appendix A.4 gives the calculation and a finite-sample upper bound.

## 3.3 Information Tails Across Populations

Population aggregation describes how often weak task information occurs; classification still retains the radius and separation factors above. The inherited geometric law is:

Theorem 2 (Geometric tail law [6]). Under the uniform regularity and localization assumptions in Appendix A.2, let the Fisher-null set have codimension c, information vanish to order 2m, and nearby density have order $\nu > - c$ . Then, for some $C > 0 ;$

$$
\operatorname* { P r } ( \lambda ( z ) ^ { - 1 } > x ) \sim C x ^ { - \beta } , \qquad \beta = \frac { c + \nu } { 2 m } .\tag{9}
$$

Quadratic collapse with positive density gives $\beta = c / 2 ;$ ; the inverse mean is finite exactly when $\beta > 1$ . Appendix ${ \tt A . 2 }$ contains the calculation and Figure ${ \mathrm { A . 1 } }$

## 3.4 The Fisher-Glass Certificate

Upper-tail conditional value-at-risk, $\mathrm { C V a R } _ { \alpha }$ , averages the largest $1 - \alpha$ fraction of a quantity, with fractional weighting at atoms [8, 9]. We apply the Fisher-glass certificate [6] to classification environments $e \sim \Pi$ , using the weakest task information $\lambda _ { e }$ and stabilization ε:

$$
L _ { e } ^ { ( \varepsilon ) } = ( \lambda _ { e } + \varepsilon ) ^ { - 1 } , \quad \quad I _ { \alpha } ^ { \mathrm { F G } } = \big [ \mathrm { C V a R } _ { \alpha , e \sim \Pi } ( L _ { e } ^ { ( \varepsilon ) } ) \big ] ^ { - 1 } , \quad 0 < \alpha < 1 .\tag{10}
$$

A larger certificate means a smaller inverse-information tail, expressed in information units. Stabilization $\varepsilon > 0$ caps inverse information at $1 / \varepsilon ;$ it does not remove the underlying null geometry. Under Theorem $2 \mathrm { { : } } \mathrm { { s } }$ power-law regime, the limit $\varepsilon \downarrow 0$ is positive for $\beta > 1$ and vanishes for $\beta \leq 1$ Proposition 1 (Information coverage). For $\rho > \varepsilon$

$$
I _ { \alpha } ^ { \mathrm { F G } } \geq \rho \quad \Longrightarrow \quad \operatorname* { P r } _ { e \sim \Pi } ( \lambda _ { e } \geq \rho - \varepsilon ) \geq \alpha .\tag{11}
$$

The certificate establishes an absolute information floor on an α-fraction of the declared environment distribution; classification consequences retain the local radius and decision separation. Appendix A.5 gives the proof. With four equally weighted groups and $\alpha = 0 . 9 5$ , it reduces to the minimum stabilized information, so that real-data audit does not isolate a benefit from tail aggregation.

## 4 Computing and Auditing Fisher-Glass

The audit establishes which population information statements are supported after accounting for how the matrix and population were constructed. It fixes the task contrast, nuisance hypothesis, matrix, normalization, and audit distribution before independent evaluation. Ridge can raise an information estimate, adaptive selection can exaggerate extremes, and weak support can destabilize weighted populations. The corrections below address these distinct sources of distortion; fitted nuisance probes and temperatures are frozen before final evaluation.

## 4.1 Operational Scores and Nuisance Coordinates

For a fitted binary predictive probability $p _ { w } ( x )$ , including any fixed temperature calibration, introduce task and nuisance directions $z ( x )$ and $b ( x )$ through logit $\begin{array} { r } { \dot { p } _ { \theta , \mu } ( x ) = \dot { \mathrm { l o g i t } } p _ { w } ( x ) + \theta z ( x ) + \mu b ( x ) } \end{array}$ At $\theta = \mu = 0$ , the score is $( Y _ { i } - p _ { i } ) v _ { i }$ , with $\boldsymbol { v } _ { i } = ( z _ { i } , b _ { i } ) ^ { \top }$ and $p _ { i } = p _ { w } ( x _ { i } )$ . Feature moments use $M _ { i } = v _ { i } v _ { i } ^ { \top }$ ; predictive Fisher uses $M _ { i } = p _ { i } ( 1 - p _ { i } ) v _ { i } v _ { i } ^ { \top }$ . The prospective Waterbirds audit uses the uncentered feature–Schur fraction of task-logit and background-probe coordinates. Hidden-slice discovery uses temperature-calibrated predictive Fisher, mixing 90% of the gated matrix with 10% of the current split’s same-class mean matrix before projection, during both search and held-out evaluation. This corresponds to shrinkage $\eta = 0 . 1 0 ;$ feature moments supply a comparison. Appendix B.1 specifies these constructions, the observed-label score alternative, operational subspaces, and computational cost.

For the scalar, unregularized, unshrunk predictive target with positive nuisance information,

$$
F _ { \mathrm { p r e d } } = \operatorname* { m i n } _ { a } \mathbb { E } [ p ( X ) ( 1 - p ( X ) ) \{ z ( X ) - a b ( X ) \} ^ { 2 } ] .\tag{12}
$$

Small information can therefore reflect task–nuisance alignment or confident probabilities with small $p ( 1 - p )$ . Neither mechanism alone establishes intervention sensitivity. Nuisance coordinates come from metadata, fitted annotation probes, or a fixed semantic direction; without annotations, that direction is the hypothesis being audited. Appendices C and D.5 report stability and specification checks.

## 4.2 Ridge Estimation and Lower Information Bounds

Let $M _ { i }$ be the per-example task–nuisance matrix under the chosen construction, with task and nuisance blocks indexed by T and N. For a predefined slice g of size $n _ { g }$ , form $\begin{array} { r } { \widehat { J } _ { g } = n _ { g } ^ { - 1 } \sum _ { i \in g } M _ { i } } \end{array}$ and compute the ridge Schur estimate

$$
\begin{array} { r } { \widehat { F } _ { g } ^ { \mathrm { r i d } } = \widehat { J } _ { T T , g } - \widehat { J } _ { T N , g } ( \widehat { J } _ { N N , g } + \lambda _ { \mathrm { r i d } } I ) ^ { - 1 } \widehat { J } _ { N T , g } . } \end{array}\tag{13}
$$

Real-data computations use $\lambda _ { \mathrm { r i d } } = 1 0 ^ { - 3 }$ . Ridge stabilizes residualization but raises information relative to the unregularized target. Let $a _ { g }$ bound estimation error and $d _ { g }$ bound the ridge-tounregularized gap on a simultaneous event over the audited slices. For declared weights $\Pi _ { G }$ , use

$$
\begin{array} { r l } & { \quad \underline { { \lambda } } _ { g } = [ \lambda _ { \operatorname* { m i n } } ( \widehat { F } _ { g } ^ { \mathrm { r i d } } ) - a _ { g } - d _ { g } ] _ { + } , } \\ & { \underline { { I } } _ { \alpha , G } = \left[ \mathrm { C V a R } _ { \alpha , g \sim \Pi _ { G } } \left( ( \underline { { \lambda } } _ { g } + \varepsilon ) ^ { - 1 } \right) \right] ^ { - 1 } . } \end{array}\tag{14}
$$

Here $[ x ] _ { + } = \operatorname* { m a x } \{ x , 0 \}$ . On that event, this lower-bounds the unregularized population certificate. A large lower bound supports a positive information floor; a small lower bound alone does not establish small true information. Certification also requires a positive lower confidence bound for $\lambda _ { \operatorname* { m i n } } ( J _ { N N , g } )$ . The real-data audits use Bonferroni-adjusted Gaussian/Wald moment simulation at nominal 95% confidence, conditional on fitted models, slices, and normalization. These implemented intervals differ from the concentration guarantees under Appendix B’s assumptions.

## 4.3 Hidden-Slice Discovery and Held-Out Certification

A gate $h _ { \phi } ( x , y ) \in [ 0 , 1 ]$ assigns membership weights. Discovery maximizes inverse projected information subject to effective-sample-size, maximum-weight, and class-mass constraints. We normalize memberships within class, cap excessive weights, and mix with uniform weights as needed to satisfy the effective-sample-size constraint. Five validation folds assess transfer and stability; a final gate is fitted on all discovery data, frozen, and evaluated on an untouched split. The support correction is recomputed on evaluation data under the frozen gate. Its unshrunk target is the weighted ratio $J _ { \phi } = \mathbb { E } [ h _ { \phi } \hat { M } ] / \mathbb { E } [ h _ { \phi } ]$ , with positive membership mass. Uncertainty must account for the numerator, membership normalization, and adjustments used to enforce the weight constraints; the Waterbirds target retains the stated shrinkage. Appendix C specifies the constraints, objective, and held-out target.

Algorithm 1 Fisher-Glass audit   
1: Specify: predictor, matrix, coordinates, normalization, calibration, and audit weights.   
2: Fix predefined slices, or fit and freeze a constrained gate on discovery data.   
3: Audit independently: check support and conditioning; compute ridge estimates, unregularized   
estimates, and lower confidence bounds.   
4: Report: raw and projected information, surviving fraction, slice support, tail certificate, and   
uncertainty.

## 5 Evaluating Information Audits

The experiments test distinct claims: sensitivity of a frozen predefined group, incremental failure prediction beyond confidence measures, the inherited geometric tail law, and held-out discovery of a declared information target. The predefined-group and learned-slice studies use different models and matrices.

## 5.1 Prospective Diagnosis Under a Nuisance Intervention

On Waterbirds [2], the four groups cross bird class with background type and are written class/background. We fully fine-tune ImageNet-initialized MobileNetV3-small models [10] with empirical risk minimization (ERM). Validation data fit the temperature and cross-fitted background probe. Eligible slices have accuracy at least the overall validation accuracy and negative log-likelihood (NLL) at most the median across slices. Among them, we select the smallest surviving fraction, $\lambda _ { \operatorname* { m i n } } ( \widehat { F } _ { g } ^ { \mathrm { r i d } } ) / \widehat { J } _ { T T , g }$ . After pilot experiments, this rule is fixed for five additional training seeds, 22–26. Each checkpoint, temperature, probe, selected slice, and audit constant is frozen before intervention scoring.

The rule selects waterbird/water in all five seeds: its validation surviving fraction is 24.8%, compared with 59.0% for landbird/land, the other eligible slice. Held-out interventions recompose birds onto same-type or opposite-type backgrounds using the same extracted foreground layer, matte, and task labels. Same-type recomposition controls for the image-construction procedure. Selected-slice accuracy is 92.0% on original images, 93.2% under the control, and 40.6% under opposite-type replacement: a $5 2 . 6 \pm 1 . 4$ percentage-point controlled drop, with a nominal 95% normal-approximation interval across seeds.

Minimum validation accuracy and maximum validation NLL select the same eligible group in all five seeds (Figure 2). Among eligible slices, minimum raw and absolute projected information select landbird/land, with a 26.7-point drop. Centering switches selection in all five seeds, identifying slice means as a ranking contributor; predictive Fisher retains waterbird/water in all five. Appendix D.2 reports matched selectors, matrix and aggregation comparisons, and individual flip prediction.

Predicting failures beyond confidence. Within the selected group, a retained auxiliary analysis predicts opposite-type failures among originally correct birds using five-fold cross-fitting by species. The baseline includes source NLL, calibrated true-label confidence, absolute raw logit, and standardized task logit z. Adding the background component $\beta _ { g } b$ and residual magnitude $| z - \beta _ { g } b |$ improves average precision by 0.0257 (nominal 95% paired interval [0.0211, 0.0302]), area under the receiver operating characteristic curve (AUROC) by 0.0150, and mean log likelihood by 0.0128 nats per donor row. All gains are positive in all five seeds; intervals use 1.96 standard errors of paired seed differences (Table D.7). This establishes incremental value of the added coordinates. It does not isolate the residual’s contribution from the background score or test the scalar population certificate as an individual predictor.

![](images/ef87e135493a86fe9ef93c779db41e8955e3835a39e0e20923cbf06a8a9ff10a.jpg)

## B Held-out outcome

![](images/c76566762c3f40f5006b9990972905b047b130eef5968af425594e9dedfc2041.jpg)  
Figure 2: A 52.6-point controlled accuracy drop in the selected group. (A) The uncentered feature–Schur fraction selects waterbird/water from two eligible groups; matched accuracy and NLL selectors agree. (B) Accuracy falls from 93.2% under same-type background recomposition to 40.6% under opposite-type recomposition, with bird foregrounds and labels preserved. Points show means and bars nominal 95% normal-approximation intervals across five training seeds.

## 5.2 Synthetic Classification and Tail Checks

Draw independent $s , r \sim N ( 0 , 1 )$ , set $z = s + \delta _ { e } r$ and $Y = \mathbf { 1 } \{ z > 0 \}$ , and evaluate the nuisanceonly logit 4s. Environments use $\delta _ { e } = 0 . 3 5 U _ { e } ^ { 1 / \gamma } , U _ { e } \sim$ Uniform(0, 1), with $\gamma = 1$ in the main experiment. Five-seed mean accuracy is $9 4 . 5 4 \%$ and $\mathrm { C V a R _ { 0 . 9 5 } }$ of environment-mean NLL is 0.262. The information audit uses exact population feature moments of $( z , s )$ , giving $J _ { T T , e } = 1 + \delta _ { e } ^ { 2 }$ and $\lambda _ { e } = \delta _ { e } ^ { 2 }$ . Its certificate is $I _ { 0 . 9 5 } ^ { \mathrm { F G } } = ( 1 . 2 2 \pm 0 . 5 3 ) \times 1 0 ^ { - 8 }$ , with $\varepsilon = 1 0 ^ { - 1 2 }$

The inverse-information exponent is $0 . 4 9 8 \pm 0 . 0 0 3$ , matching $\beta = \gamma / 2 = 0 . 5$ in the geometric tail law of Theorem 2 (Figure A.1). Log-survival regression uses the largest 5% of inverse-information observations. Additional experiments check $\bar { \beta \ } = K / 2$ for K independent information branches (Appendix A.2) and the information floor obtained by allocating observations to an informative source (Appendix E.1). Uncertainties are nominal 95% normal-approximation intervals across five seeds. These checks exhibit favorable prediction alongside weak projected information and the predicted tail behavior (Appendix D.3).

## 5.3 Discovering Low-Information Subpopulations

In a separate experiment, we fit a linear task head on frozen ImageNet MobileNetV3-small representations. Gate discovery uses task labels and a nuisance score obtained by projecting CLIP image embeddings onto a fixed water-minus-land text direction [11]. Background and group annotations remain sealed through test evaluation. We search each task class separately with a class-specific logistic gate and select the class–gate pair with highest discovery utility. Gates target 10% within-class mass under Section 4.3’s support constraints. The validation split supplies discovery data: five folds assess transfer, after which the gate is refitted on the full split, frozen, and evaluated once on untouched test data.

The discovered slice has 98.8% membership-weighted accuracy and lower estimated projected predictive Fisher information than high-loss and random selectors targeting the same mass in every seed (Figure 3). Its calibrated predictive information is $0 . 0 0 5 \pm 0 . 0 \bar { 0 } 1$ , and its uncalibrated NLL is $0 . 0 3 1 \pm 0 . 0 1 7$ , with 95% Student-t intervals across five paired seeds. Accuracy and NLL use support-corrected membership weights without information pooling (Appendix C.1). The Fisher and high-loss gates’ top 10% raw-membership sets within their selected class have mean Jaccard overlap 0.001, indicating largely distinct selections.

The random-direction control attains similar predictive information, with the paired CLIP-minusrandom interval spanning zero. Feature-moment comparisons with high-loss selection also have paired intervals containing zero. Predictive Fisher includes the confidence factor $p ( 1 - p )$ , whereas displayed NLL is uncalibrated; these results do not separate confidence from nuisance alignment.

![](images/cd253a9b9313bb771ec24069500550d8f64108b3add443fe7b3e360ebbab955d.jpg)  
Figure 3: Discovery finds accurate slices with low projected predictive information. (A) CLIPguided search finds low projected predictive Fisher information and 98.8% membership-weighted accuracy; the random-direction control also attains low information. (B) Both searches yield low uncalibrated NLL (log scale). Selectors target 10% within-class mass. Points show means and bars 95% Student-t intervals across five paired seeds, with gates frozen before testing.

They establish discovery of a low-information population, not its unusual intervention sensitivity or superiority to confidence-matched selection. Appendix D.4 gives paired intervals, support diagnostics, and composition.

## 6 Information-Guided Interventions

We evaluate whether information changes improve the intended predictive outcome. An informative observation source can guarantee an information floor under the Fisher-reserve design [6] (Appendix E.1); training and acquisition require separate behavioral evaluation. For predictor parameters w, empirical task-risk objective $\widehat { R } _ { \mathrm { b a s e } }$ (ERM or group distributionally robust optimization, Group DRO), and penalty λ<sub>FG</sub>, training uses equally weighted groups:

$$
\operatorname* { m i n } _ { w } \widehat { R } _ { \mathrm { b a s e } } ( w ) + \lambda _ { \mathrm { F G } } \mathrm { C V a R } _ { \alpha , g } \Big [ \big \{ \lambda _ { \mathrm { m i n } } \big ( \widehat { F } _ { g } ^ { \mathrm { r i d } } ( w ) \big ) + \varepsilon \big \} ^ { - 1 } \Big ] .\tag{15}
$$

Coordinates follow Section 4. Among the frozen-feature models whose validation worst-group accuracy is within two percentage points of the best, we select the coefficient with the largest validation certificate. This selection uses validation group labels.

Table 1: Frozen-feature Waterbirds training. Means with nominal 95% normal-approximation intervals (±1.96 standard errors) across five seeds; uncentered feature–Schur ridge certificates use $\alpha = 0 . 9 5$ . Validation selection follows Section 6.
<table><tr><td>Objective</td><td>Average accuracy (%)</td><td>Worst-group accuracy (%)</td><td>Fisher-Glass certificate</td></tr><tr><td>ERM</td><td> $7 7 . 2 \pm 0 . 1$ </td><td> $5 5 . 7 \pm 0 . 3$ </td><td> $0 . 1 9 8 \pm 0 . 0 0 1$ </td></tr><tr><td>ERM + Fisher-Glass</td><td> $8 0 . 4 \pm 0 . 2$ </td><td> $5 6 . 3 \pm 0 . 3$ </td><td> $0 . 3 9 9 \pm 0 . 0 0 6$ </td></tr><tr><td>Group DRO</td><td> $8 0 . 9 \pm 0 . 3$ </td><td> $6 7 . 3 \pm 0 . 3$ </td><td> $0 . 2 3 3 \pm 0 . 0 0 3$ </td></tr><tr><td>Group DRO + Fisher-Glass</td><td> $8 0 . 6 \pm 0 . 9$ </td><td> $6 9 . 4 \pm 1 . 3$ </td><td> $0 . 6 1 4 \pm 0 . 0 5 9$ </td></tr></table>

Fisher-Glass approximately doubles ERM’s certificate; with Group DRO it raises both the certificate and mean worst-group accuracy (Table 1). Stronger penalties can reduce accuracy (Appendix E.3). These source-performance gains do not establish a repair of background sensitivity.

In prospective fine-tuning, the validation certificate increases from 0.289 to 0.330, while controlled swap drops remain 52.9 points for FG and 52.6 for ERM. A separate Fisher-based frozen-slice repair under validation-accuracy constraints selects the unchanged model in all five seeds. On another cohort, post hoc selection raises rediscovered-slice information but reduces worst-group accuracy by 12.6 points; selected coefficients match validation-NLL minimization in every seed (Appendix E.3). Thus information improvement and the intended behavioral improvement are separate outcomes.

For candidate a with modified feature/input $x _ { a } ^ { \mathrm { c f } }$ and label $y _ { a } .$ , tail-transverse influence $\mathrm { T T I } ( a ) =$ $G _ { \alpha } ( a ) \ell _ { w } ( x _ { a } ^ { \mathrm { c f } } , y _ { a } )$ weights current predictive loss by the first-order inverse-information CVaR decrease $G _ { \alpha } ( a )$ along a’s allocation path. This differentiates the Schur complement along its weakest task direction with tail-sensitive weights (Appendix E.4). Waterbirds ranks candidates by $\widetilde { \mathrm { T T I } } _ { \alpha } ( a ) = \widetilde { G } _ { \alpha } ( a ) \ell _ { w } ( x _ { a } ^ { \mathrm { c f } } , y _ { a } )$ , using the tail-weighted squared-residual surrogate $\widetilde { G } _ { \alpha }$ defined in Appendix E.5.

Waterbirds candidates translate features by the within-class difference of background-group means, scaled by $\gamma _ { \mathrm { c f } }$ . Training data fit means and the nuisance probe; validation supplies candidates for retraining, and evaluation is held out. Budgets are fractions of original training size. At $\gamma _ { \mathrm { c f } } = 1$ and a 1% budget, $\widetilde { \mathrm { T T I } _ { \alpha } }$ yields a mean worst-group accuracy gain of 4.52 points versus 3.27 for high-loss selection; high loss produces the larger mean certificate gain. At this strength, $\widetilde { \mathrm { T T I } _ { \alpha } }$ has the larger mean worst-group accuracy gain at budgets of 1%, 2%, and 5%; selection by $\widetilde { G } _ { \alpha }$ alone yields smaller improvements (Appendix E.5).

## 7 Related Work

Efficient information and classification. Efficient-score theory defines task information after projection away from nuisance tangent spaces [4, 5, 12]. Fisher-glass theory connects nuisanceadjusted information to inverse-information tails in metrology [6]. Our analysis develops classification bounds and statistical guarantees for population audits. The classification argument uses local testing reductions [13, 14] under an explicit Bayes-decision separation condition. The operational analysis addresses Schur estimation, ridge effects, and independent evaluation of learned population gates.

Tail risk, robust learning, and gradient geometry. CVaR concentrates objectives on adverse outcomes [8, 9]; Group DRO applies robust optimization to predictive loss across groups [2]. Invariant Risk Minimization seeks shared optimal classifiers [15], while Fishr regularizes domain-level gradient variances [16]. We use inverse projected-information tail risk as a training penalty alongside empirical risk or Group DRO, and evaluate information and predictive performance together.

Hidden stratification and slice discovery. Hidden-stratification research motivates finer population evaluation [17]. GEORGE clusters representations to infer subclasses [18], JTT upweights initially misclassified examples [3], and AGRO learns adversarial group assignments using prediction loss [19]. Our search targets low predictive projected information given task labels and a fixed nuisance direction. Support constraints and an independently evaluated frozen gate make accurately classified, low-loss populations eligible audit targets.

Information-based design and acquisition. E-optimal design emphasizes the smallest information eigenvalue [20]; BAIT uses Fisher information for neural active learning [21]. TTI measures local information gain along a weak nuisance-adjusted task direction, weights environments by tail sensitivity, and multiplies by current task loss. Acquisition experiments assess realized information and accuracy changes after retraining.

## 8 Limitations

Specification and interpretation. Audits quantify information for a declared task contrast, nuisance space, representation, and population; admissible nuisance changes must preserve the task. Unregularized nuisance projection of predictive Fisher gives efficient information for the declared predictive model, while feature–Schur quantities describe residual feature geometry. Their relation to population efficient information depends on the score model. Coordinate scaling and matrix construction affect the audit; standardization does not establish representation invariance. The incremental contribution of semantic nuisance projection remains unresolved (Section 5). A general procedure for admitting nuisance specifications through relevance, task preservation, stability, and conditioning checks remains to be evaluated (Appendix B.1).

Theory and uncertainty. The local classification bound requires distinguishability and quantitative Bayes-decision separation throughout its radius. The worked family verifies these conditions and a matching rate, but its statistical nuisance is not a model of physical background swaps. Coverage concerns the declared environment distribution and frozen population rules. Small mass and weak nuisance conditioning amplify uncertainty; ridge stabilization can inflate information, so positive ridge values alone do not establish identifiability (Section 4.2; Appendix B.2). Implemented Gaussian/Wald coverage is nominal and conditional on fitted models, evaluated families, and normalization; it omits uncertainty from evaluation-fitted normalization. Repeated use of certification data requires additional control (Section 4; Appendix B).

Empirical scope and interventions. Evidence centers on Waterbirds, supplementary CivilComments diagnostics, and synthetic constructions. The prospective rule uses fresh seeds of the development benchmark; transfer requires further evaluation. The auxiliary failure-prediction gain combines background and residual coordinates, without a background-only ablation. No retained experiment tests whether the learned low-information slice is more sensitive than a confidence-matched population. Information gains and coefficient-selection policies require separate behavioral validation (Section 6). Reserve guarantees assume an informative anchor; acquisition uses group-informed feature shifts whose transfer to realizable task-preserving inputs remains open (Appendix E).

## 9 Conclusion

Task–nuisance ambiguity has a classification cost when statistically similar alternatives require different decisions. Our local bound makes that condition explicit, and a worked family shows how nuisance confounding reduces the effective sample size near a decision boundary despite constant raw information at the reference parameter. A matching upper bound identifies the learning rate and the cap imposed by the local decision scale. Population audits then account for estimation error, ridge bias, adaptive membership, and support, establishing which information statements are supported by data.

The retained Waterbirds experiments establish a 52.6-point controlled accuracy drop in a frozen predefined group and incremental species-held-out failure-prediction value from nuisance-derived coordinates beyond a specified confidence baseline. Independent evaluation validates a discovered low predictive-information population. Training and acquisition show that information gains and predictive gains can diverge.

The classification result, population audit, and behavioral evaluations have complementary roles. They connect ambiguity to learning difficulty, quantify a declared information target with uncertainty, and test its relationship to observed behavior. This supports evaluating information objectives together with predictive performance and controlled nuisance sensitivity when assessing a classifier or a proposed improvement.

## Acknowledgments and Disclosure of Funding

This work received no funding. The author declares no competing interests.

## References

[1] El Mustapha Mansouri. Autonomous AI bird feeder for backyard biodiversity monitoring, 2025. URL https://arxiv.org/abs/2508.09398.

[2] Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In International Conference on Learning Representations, 2020. URL https: //openreview.net/forum?id=ryxGuJrFvS.

[3] Evan Z. Liu, Behzad Haghgoo, Annie S. Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, and Chelsea Finn. Just train twice: Improving group robustness without training group information. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 6781–6792. PMLR, 2021. URL https://proceedings.mlr.press/v139/liu21f.html.

[4] Peter J. Bickel, Chris A. J. Klaassen, Ya’acov Ritov, and Jon A. Wellner. Efficient and Adaptive Estimationfor Semiparametric Models. Johns Hopkins University Press, 1993.

[5] Anastasios A. Tsiatis. Semiparametric Theory and Missing Data. Springer Series in Statistics. Springer, 2006. doi: 10.1007/0-387-37345-4.

[6] El Mustapha Mansouri and Keigo Arai. Fisher Glasses: Tail-certified quantum metrology in quenched environments. arXiv preprint arXiv:2607.01085v1, 2026. URL https://arxiv.or g/abs/2607.01085v1.

[7] D. R. Cox and N. Reid. Parameter orthogonality and approximate conditional inference. Journal of the Royal Statistical Society: Series B (Methodological), 49(1):1–18, 1987. doi: 10.1111/j.2517-6161.1987.tb01422.x.

[8] R. Tyrrell Rockafellar and Stanislav Uryasev. Optimization of conditional value-at-risk. The Journal ofRisk, 2(3):21–41, 2000. doi: 10.21314/JOR.2000.038.

[9] R. Tyrrell Rockafellar and Stanislav Uryasev. Conditional value-at-risk for general loss distributions. Journal ofBanking & Finance, 26(7):1443–1471, 2002. doi: 10.1016/S0378-4266(02)0 0271-6.

[10] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, and Hartwig Adam. Searching for MobileNetV3. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1314–1324, 2019. doi: 10.1109/ICCV.2019.00140. URL https://openaccess.thecvf.com/content\_ICCV\_2019/html/Howard\_Searching\_ for\_MobileNetV3\_ICCV\_2019\_paper.html.

[11] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021. URL https://proceedings.mlr.press/v139/radford21a.html.

[12] Whitney K. Newey. Semiparametric efficiency bounds. Journal of Applied Econometrics, 5(2): 99–135, 1990. doi: 10.1002/jae.3950050202.

[13] Lucien Le Cam. Asymptotic Methods in Statistical Decision Theory. Springer Series in Statistics. Springer, 1986. doi: 10.1007/978-1-4612-4946-7.

[14] Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer, 2009. doi: 10.1007/b13794.

[15] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019. doi: 10.48550/arXiv.1907.02893.

[16] Alexandre Rame, Corentin Dancette, and Matthieu Cord. Fishr: Invariant gradient variances for out-of-distribution generalization. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 18347– 18377. PMLR, 2022. URL https://proceedings.mlr.press/v162/rame22a.html.

[17] Luke Oakden-Rayner, Jared Dunnmon, Gustavo Carneiro, and Christopher Ré. Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. In Proceedings ofthe ACM Conference on Health, Inference, and Learning, pages 151–159. ACM, 2020. doi: 10.1145/3368555.3384468.

[18] Nimit S. Sohoni, Jared A. Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Ré. No subclass left behind: Fine-grained robustness in coarse-grained classification problems. In Advances in Neural Information Processing Systems, volume 33, pages 19339–19352. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/hash/e 0688d13958a19e087e123148555e4b4-Abstract.html.

[19] Bhargavi Paranjape, Pradeep Dasigi, Vivek Srikumar, Luke Zettlemoyer, and Hannaneh Hajishirzi. AGRO: Adversarial discovery of error-prone groups for robust optimization. In International Conference on Learning Representations, 2023. URL https://openreview.n et/forum?id=IrzkT99fDJH.

[20] Friedrich Pukelsheim. Optimal Design of Experiments, volume 50 of Classics in Applied Mathematics. SIAM, 2006. doi: 10.1137/1.9780898719109.

[21] Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Sham M. Kakade. Gone fishing: Neural active learning with Fisher embeddings. In Advances in Neural Information Processing Systems, volume 34, pages 8927–8939. Curran Associates, Inc., 2021. URL https://proceedings.ne urips.cc/paper/2021/hash/4afe044911ed2c247005912512ace23b-Abstract.html.

[22] Jean-Yves Audibert and Alexandre B. Tsybakov. Fast learning rates for plug-in classifiers. The Annals ofStatistics, 35(2):608–633, 2007. doi: 10.1214/009053606000001217.

[23] Joel A. Tropp. User-friendly tail bounds for sums of random matrices. Foundations of Computational Mathematics, 12(4):389–434, 2012. doi: 10.1007/s10208-011-9099-z.

[24] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet large scale visual recognition challenge. International Journal ofComputer Vision, 115:211–252, 2015. doi: 10.1007/s11263-015-0816-y.

[25] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR, 2017. URL https://proceedings.mlr.press/v70/guo17a.html.

[26] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The Caltech-UCSD Birds-200-2011 dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. URL https://www.vision.caltech.edu/datasets/cub\_200\_2011 /.

[27] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(6):1452–1464, 2018. doi: 10.1109/TPAMI.2017.2723009. URL https://doi.org/10.1109/TPAMI.2017.2723009.

[28] Alexandru Telea. An image inpainting technique based on the fast marching method. Journal of Graphics Tools, 9(1):25–36, 2004. doi: 10.1080/10867651.2004.10487596. URL https://doi.org/10.1080/10867651.2004.10487596.

[29] Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. Nuanced metrics for measuring unintended bias with real data for text classification. In Companion Proceedings of The 2019 World Wide Web Conference, pages 491–500. ACM, 2019. doi: 10.1145/3308560.3317593.

[30] Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton Earnshaw, Imran Haque, Sara M. Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. WILDS: A benchmark of in-the-wild distribution shifts. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 5637–5664. PMLR, 2021. URL https://proceedings.mlr.press/v139/koh21a. html.

## A Proofs of the Core Results

Throughout, task coordinates use the common normalization of Section $^ { 2 , }$ and environments follow the declared distribution Π.

## A.1 Projection, Normalization, and Task–Nuisance Tangency

Use the centered, square-integrable scores and blocks $J _ { A B , e }$ of Section 2.1. Suppose the d<sub>N</sub> coordinates of the finite nuisance-score vector span $\mathcal { T } _ { N , \epsilon }$ and $J _ { N N , e } \succ 0$ . With $\Gamma _ { e } = J _ { T N , e } J _ { N N , e } ^ { - 1 } ,$

$$
\begin{array} { r l } & { \mathbb { E } _ { P _ { e } } [ ( s _ { T , e } - \Gamma _ { e } s _ { N , e } ) s _ { N , e } ^ { \top } ] = J _ { T N , e } - \Gamma _ { e } J _ { N N , e } = 0 , } \\ & { \quad \quad F _ { e } = \mathbb { E } _ { P _ { e } } [ s _ { T , e } ^ { \mathrm { e f f } } ( s _ { T , e } ^ { \mathrm { e f f } } ) ^ { \top } ] = J _ { T T , e } - J _ { T N , e } J _ { N N , e } ^ { - 1 } J _ { N T , e } . } \end{array}\tag{A.1}
$$

Because $\Gamma _ { e } s _ { N , \epsilon }$ lies componentwise in the nuisance span, orthogonality identifies the residual as the efficient score in Equation (1) [4, 5]. Equivalently,

$$
\boldsymbol { u } ^ { \intercal } \boldsymbol { F } _ { e } \boldsymbol { u } = \operatorname* { m i n } _ { \boldsymbol { v } \in \mathbb { R } ^ { d _ { N } } } \mathbb { E } _ { P _ { e } } [ ( \boldsymbol { u } ^ { \intercal } \boldsymbol { s } _ { T , e } - \boldsymbol { v } ^ { \intercal } \boldsymbol { s } _ { N , e } ) ^ { 2 } ] , \qquad \boldsymbol { v } ^ { \star } = J _ { N N , e } ^ { - 1 } J _ { N T , e } \boldsymbol { u } .\tag{A.2}
$$

Hence $F _ { e } \succeq 0 .$ , with $u ^ { \top } F _ { e } u = 0$ exactly when $u ^ { \top } s _ { T , e }$ belongs to the nuisance span. The projection definition extends directly to the closed span of an infinite-dimensional nuisance tangent space.

For separate invertible transformations $\tilde { s } _ { T } = U s _ { T }$ and $\widetilde { s } _ { N } = V s _ { N }$ , the nuisance span is unchanged and

$$
\widetilde { s } _ { T } ^ { \mathrm { e f f } } = U s _ { T } ^ { \mathrm { e f f } } , \qquad \widetilde { F } = U F U ^ { \top } ; \qquad \frac { F } { J _ { T T } } = 1 - \rho ^ { 2 } \quad \mathrm { i n ~ t h e ~ s c a l a r ~ c a s e } .\tag{A.3}
$$

Here $\rho = J _ { T N } / \sqrt { J _ { T T } J _ { N N } }$ requires positive raw task and nuisance information. Exact information is invariant to nuisance-basis changes; task congruence preserves rank but generally changes eigenvalues. The scalar surviving fraction is invariant to nonzero task rescaling and invertible nuisance-basis changes. Nuisance ridge preserves the latter invariance only if $\lambda _ { \mathrm { r i d } } I$ becomes $\lambda _ { \mathrm { { r i d } } } V V ^ { \top }$ under $s _ { N } \mapsto V s _ { N }$ (Appendix B).

For the tangency construction in Section 2.2, let $b _ { e } , r _ { e } \in L _ { 0 } ^ { 2 } ( P _ { e } )$ be orthogonal unit vectors, $\mathcal { T } _ { N , e } =$ span $\{ b _ { e } \}$ , and $a _ { e } = b _ { e } + \delta _ { e } r ,$ with $\delta _ { e } \geq 0$ . The projection coefficient is $\langle a _ { e } , b _ { e } \rangle _ { e } / \| b _ { e } \| _ { e } ^ { 2 } = 1$ , so

$$
a _ { e } ^ { \mathrm { e f f } } = \delta _ { e } r _ { e } , \qquad F _ { e } = \| \delta _ { e } r _ { e } \| _ { e } ^ { 2 } = \delta _ { e } ^ { 2 } , \qquad J _ { T T , e } = 1 + \delta _ { e } ^ { 2 } , \qquad \frac { F _ { e } } { J _ { T T , e } } = \frac { \delta _ { e } ^ { 2 } } { 1 + \delta _ { e } ^ { 2 } } .\tag{A.4}
$$

This establishes Equation (3); coexistence with low prediction risk additionally requires a successful nuisance-based predictor.

## A.2 Normal Coordinates and the Inverse-Information Tail

The full assumptions of Theorem 2 are as follows. The zero set $\mathcal { Z } = \{ \lambda = 0 \}$ is a nonempty compact $C ^ { 2 }$ embedded manifold without boundary, of codimension $c \in \{ 1 , \ldots , d \}$ . Write $\bar { U _ { r } ( \mathcal { Z } ) } = \bar { \{ } z  :$ dis $\operatorname { t } ( z , { \mathcal { Z } } ) < r \}$ for its tube, $N _ { s } \mathcal { Z }$ for the normal space at $s , S ( N _ { s } \mathcal { \tilde { Z } } )$ for its unit sphere, and vol<sub>Z</sub> for induced volume. The nonnegative λ is continuous on $\overline { { U } } _ { r _ { \star } } ( \mathcal { Z } )$ and bounded away from zero outside it. In coordinates $z = s + r \omega$ , assume uniformly over ω $\dot { \in { \cal S } ( N _ { s } \mathcal { Z } ) }$ and $s \in { \mathcal { Z } }$

$$
\lambda ( s + r \omega ) = a ( s , \omega ) r ^ { 2 m } ( 1 + o ( 1 ) ) , \qquad p ( s + r \omega ) = b ( s , \omega ) r ^ { \nu } ( 1 + o ( 1 ) ) ,\tag{A.5}
$$

where $p$ is the Lebesgue density, $m > 0 , \nu > - c$ and continuous $a ,$ b have positive uniform lower and upper bounds. Put

$$
\beta = \frac { c + \nu } { 2 m } , \qquad C _ { Z } = \frac { 1 } { c + \nu } \int _ { \mathcal { Z } } \int _ { S ( N _ { s } \mathcal { Z } ) } b ( s , \omega ) a ( s , \omega ) ^ { - \beta } d \sigma _ { s } ( \omega ) d \mathrm { v o l } _ { \mathcal { Z } } ( s ) .\tag{A.6}
$$

The sphere measure $\sigma _ { s }$ is unnormalized; for $c = 1$ , each of the two normal directions has mass one.

ProofofTheorem 2. For any $0 < r _ { 0 } < r _ { \star }$ , continuity and absence of zeros on the compact annulus $r _ { 0 } \le \mathrm { d i s t } ( z , \mathcal { Z } ) \le r _ { \star }$ give a positive minimum of $\lambda .$ . Together with the bound outside the outer tube, this localizes $\{ \lambda < t \}$ to $U _ { r _ { 0 } } ( \mathcal { Z } )$ for all sufficiently small t. The tubular volume element is

$d z = j ( s , r \omega ) r ^ { c - 1 }$ dr dσ d vol , where the coordinate Jacobian satisfies $j = 1 + O ( r )$ uniformly. Thus, for any $0 < \zeta < 1$ , uniformity permits $r _ { 0 }$ small enough that

$$
\begin{array} { r l r l } & { ( 1 - \zeta ) a r ^ { 2 m } \le \lambda ( s + r \omega ) \le ( 1 + \zeta ) a r ^ { 2 m } , } & & { } \\ & { ( 1 - \zeta ) b r ^ { \nu } \le p ( s + r \omega ) j ( s , r \omega ) \le ( 1 + \zeta ) b r ^ { \nu } } & & { ( 0 < r < r _ { 0 } ) . } \end{array}\tag{A.7}
$$

For small t, all radial thresholds lie inside this tube. The event contains $r < ( t / ( ( 1 + \zeta ) a ) ) ^ { 1 / ( 2 m ) }$ and is contained in $r < ( t / ( ( 1 - \zeta ) a ) ) ^ { 1 / ( 2 m ) }$ . Integrating uses $\begin{array} { r } { \int _ { 0 } ^ { ( t / a ) ^ { 1 / ( 2 m ) } } b r ^ { c + \nu - 1 } d r = b a ^ { - \beta } t ^ { \beta } / ( c + \nu ) } \end{array}$ and gives

$$
( 1 - \zeta ) ( 1 + \zeta ) ^ { - \beta } C _ { \mathcal { Z } } \leq \frac { \operatorname* { P r } ( \lambda < t ) } { t ^ { \beta } } \leq ( 1 + \zeta ) ( 1 - \zeta ) ^ { - \beta } C _ { \mathcal { Z } } .\tag{A.8}
$$

Let $t \downarrow 0$ and then $\zeta \downarrow 0$ . Compactness and the coefficient bounds ensure $0 < C _ { \mathcal { Z } } < \infty .$ , so $\operatorname* { P r } ( \lambda < t ) \sim C _ { \mathcal { Z } } t ^ { \beta }$ . Since $c \geq 1$ , Z has zero Lebesgue and environment measure; substituting $t = x ^ { - 1 }$ proves the inverse tail in the next display. □

The tail-integral identity then yields, for every $p _ { 0 } > 0$

$$
\begin{array} { r c l } { { } } & { { } } & { { \operatorname* { P r } ( \lambda ^ { - 1 } > x ) \sim C _ { \mathcal Z } x ^ { - \beta } , } } \\ { { } } & { { } } & { { \operatorname * { \mathbb { E } } [ \lambda ^ { - p _ { 0 } } ] = p _ { 0 } \displaystyle \int _ { 0 } ^ { \infty } x ^ { p _ { 0 } - 1 } \operatorname* { P r } ( \lambda ^ { - 1 } > x ) d x < \infty \quad \Longleftrightarrow \quad p _ { 0 } < \beta . } } \end{array}\tag{A.9}
$$

The bounded-x part is finite; the tail has the convergence criterion of $\int ^ { \infty } x ^ { p _ { 0 } - \beta - 1 }$ dx. In particular, the inverse mean is finite exactly for $\beta > 1$ , with logarithmic divergence at $\beta = 1$

Scalar branches. For $\lambda = \delta ^ { 2 } , \operatorname* { P r } ( \delta < t ) \sim C _ { \delta } t ^ { \gamma }$ gives exponent $\gamma / 2$ . More generally, let $\begin{array} { r } { \lambda _ { K } = \sum _ { k = 1 } ^ { K } w _ { k } \delta _ { k } ^ { 2 } , w _ { k } > 0 , } \end{array}$ , where $\delta \in [ 0 , \infty ) ^ { K }$ has joint density $f$ continuous and positive at the origin from within the orthant. Changing variables gives the scalar portfolio law [6]:

$$
\begin{array} { r l } & { \operatorname* { P r } ( \lambda _ { K } < t ) = \frac { t ^ { K / 2 } } { \sqrt { \prod _ { k } w _ { k } } } \int _ { B _ { K } ^ { + } } f \big ( \sqrt { t / w _ { 1 } } u _ { 1 } , \dots , \sqrt { t / w _ { K } } u _ { K } \big ) d u } \\ & { \qquad \sim \frac { f ( 0 ) v _ { K } } { 2 ^ { K } \sqrt { \prod _ { k } w _ { k } } } t ^ { K / 2 } . } \end{array}\tag{A.10}
$$

Here $B _ { K } ^ { + }$ is the positive-orthant unit ball and $v _ { K }$ the full ball’s volume. Independence gives $f ( 0 ) =$ $\textstyle \prod _ { k } f _ { k } ( { \vec { 0 } } + )$ for branch marginal densities $f _ { k }$ ; the inverse mean is finite for $K \geq 3$ . This calculation concerns additive scalar information or a fixed task direction. A minimum-eigenvalue conclusion instead requires applying Theorem 2 to that eigenvalue.

## A.3 Local Classification: Testing Reduction and Tail Aggregation

Let $\mathcal { E } _ { P } ( g ) = R _ { P } ( g ) - R _ { F } ^ { \star }$ . Fix a local family $\mathcal { P } _ { e } ^ { \mathrm { l o c } }$ independently of sample size, and define

$$
\Re _ { e } ( n ) = \operatorname* { i n f } _ { { \widehat { g } } _ { n } } \ \operatorname* { s u p } _ { P \in \mathcal { P } _ { e } ^ { \mathrm { l o c } } } \mathbb { E } _ { S _ { n } \sim P ^ { \otimes n } } [ \mathcal { E } _ { P } ( \widehat { g } _ { n } ( S _ { n } ) ) ] ,\tag{A.11}
$$

where learning rules are measurable and n counts independent observations from environment $e ;$ independent learner randomization is included in the expectation. Assume $\lambda _ { e } > 0$ and a unit weakest efficient-information direction $u _ { e }$ . For every $0 < t \leq t _ { 0 , e }$ , the fixed family contains $P _ { e , \pm t }$ with task coordinates $\pm t u _ { \epsilon }$ and potentially varying nuisance coordinates. For fixed $K _ { e } , c _ { e } , t _ { 0 , e } > 0$ and $q > 0$ assume both bounds in Equation (4) hold throughout that radius.

ProofofTheorem 1. Fix an admissible t and abbreviate $P _ { \pm } = P _ { e , \pm t }$ . For any learning rule, the decision-separation condition $\mathcal { E } _ { P _ { + } } ( g ) + \mathcal { E } _ { P _ { - } } ( g ) \geq c _ { e } t ^ { q }$ implies that choosing the alternative with smaller excess risk makes any testing error cost at least $c _ { e } t ^ { q } / 2$ under the true alternative. Explicitly,

$$
\begin{array} { r l r } & { } & { \psi _ { t } ( S _ { n } ) = + \iff \mathcal { E } _ { P _ { + } } ( \widehat { g } _ { n } ( S _ { n } ) ) \leq \mathcal { E } _ { P _ { - } } ( \widehat { g } _ { n } ( S _ { n } ) ) , } \\ & { } & { \underset { \sigma \in \{ + , - \} } { \operatorname* { m a x } } \mathbb { E } _ { P _ { \sigma } ^ { \otimes n } } [ \mathcal { E } _ { P _ { \sigma } } ( \widehat { g } _ { n } ) ] \geq \frac { c _ { e } t ^ { q } } { 4 } \{ P _ { + } ^ { \otimes n } ( \psi _ { t } = - ) + P _ { - } ^ { \otimes n } ( \psi _ { t } = + ) \} . } \end{array}\tag{A.12}
$$

A Score geometry  
![](images/8449b761f50159d637db6d1e885c4ce15e251039de899f1a8b73dbda6987cdae.jpg)

B Information collapse  
![](images/95ff1091caf760114a3b3863c4f55dba82b619d794bae4186b3a16954701d521.jpg)  
C Inverse-information tail

![](images/cf67aa7dec33031614945b2087e46bbc0b8d2745f6fa0c9612d564664da66a48.jpg)  
Figure A.1: Near-alignment produces a heavy inverse-information tail. (A) Projection leaves a residual score of norm δ, giving information $\lambda \stackrel { . } { = } \delta ^ { 2 }$ . (B) Raw information $1 + \delta ^ { 2 }$ remains near one as λ approaches zero. (C) Empirical survival probabilities (blue) follow the $x ^ { - 1 / 2 }$ reference (dashed). Panels B–C use 300,000 environments with δ ∼ Uniform(0, 0.35) and $L = ( \delta ^ { 2 } + 1 0 ^ { - 1 2 } ) ^ { - 1 }$

The standard exponential two-point testing inequality [14, Lemma 2.6], together with product additivity of KL and the assumed KL bound, gives

$$
\begin{array} { r l } & { P _ { + } ^ { \otimes n } ( \psi _ { t } = - ) + P _ { - } ^ { \otimes n } ( \psi _ { t } = + ) \geq \frac { 1 } { 2 } \exp [ - n \operatorname { K L } ( P _ { + } \| P _ { - } ) ] \geq \frac { 1 } { 2 } e ^ { - n K _ { c } t ^ { 2 } \lambda _ { c } } , } \\ & { \qquad \Re _ { e } ( n ) \geq \frac { c _ { e } } { 8 } t ^ { q } e ^ { - n K _ { e } t ^ { 2 } \lambda _ { e } } . } \end{array}\tag{A.13}
$$

The second line takes the supremum over the fixed family containing both alternatives and then the infimum over learning rules. Choose

$$
\begin{array} { c c c } { { t _ { n } = \operatorname* { m i n } \{ t _ { 0 , e } , ( n K _ { e } \lambda _ { e } ) ^ { - 1 / 2 } \} , } } & { { n K _ { e } \lambda _ { e } t _ { n } ^ { 2 } \leq 1 , } } \\ { { } } & { { } } \\ { { \mathfrak { R } _ { e } ( n ) \geq c _ { 0 } c _ { e } \operatorname* { m i n } \{ t _ { 0 , e } ^ { q } , ( n K _ { e } \lambda _ { e } ) ^ { - q / 2 } \} , } } & { { c _ { 0 } = \displaystyle \frac { e ^ { - 1 } } { 8 } . } } \end{array}\tag{A.14}
$$

Independent learner randomization has the same law under both alternatives and leaves their KL divergence unchanged, so the argument also covers randomized rules. □

Quadratic decision separation. Suppose the binary alternatives share a feature marginal $Q _ { e } ,$ with regression functions $\bar { \eta _ { \pm , t } } ( x ) = P _ { e , \pm t } \bar { ( } Y = 1 \mid X \stackrel { . } { = } x )$ and Bayes rules $g _ { \pm , t } ^ { \star } ( x ) = \mathbf { \bar { 1 } } \{ \eta _ { \pm , t } ( x ) \ge$ $1 / 2 \}$ . Let $D _ { e , t } = \{ g _ { + , t } ^ { \star } \neq g _ { - , t } ^ { \star } \}$ be their disagreement set. Applying the binary excess-risk identity [22, Eq. (1.1)] gives

$$
\begin{array} { l l } { \displaystyle \operatorname* { i n f } _ { g } \{ \mathcal { E } _ { P _ { e , + t } } ( g ) + \mathcal { E } _ { P _ { e , - t } } ( g ) \} } \\ { \displaystyle \quad \quad = \int _ { D _ { e , t } } \operatorname* { m i n } \{ | 2 \eta _ { + , t } ( x ) - 1 | , | 2 \eta _ { - , t } ( x ) - 1 | \} d Q _ { e } ( x ) . } \end{array}\tag{A.15}
$$

Pointwise, a classifier must disagree with at least one Bayes rule on $D _ { e , t } ;$ choosing the smaller margin attains the integral for unrestricted measurable binary classifiers, while it remains a lower bound for a restricted class. If, for every $0 < t \leq t _ { 0 , e } ,$ a measurable $A _ { e , t } \subseteq D _ { e , i }$ <sub>t</sub> has $Q _ { e } ( A _ { e , t } ) \geq c _ { \mathrm { m a s s } , e } t$ and both margins at least $c _ { \mathrm { m a r g i n } , e } t .$ , with positive constants independent of t, the integral is at least $c _ { \mathrm { m a s s } , e } c _ { \mathrm { m a r g i n } , e } t ^ { 2 }$ . Thus $q = 2$ and $c _ { e } = c _ { \mathrm { m a s s } , e } c _ { \mathrm { m a r g i n } , e }$ satisfy decision separation; the KL condition is still required separately.

Environmental aggregation. If these assumptions hold for Π-almost every environment and all quantities are measurable, monotonicity and positive homogeneity of CVaR give

$$
\begin{array} { r } { \mathcal { B } _ { e } ( n ) = c _ { e } \operatorname* { m i n } \{ t _ { 0 , e } ^ { q } , ( n K _ { e } \lambda _ { e } ) ^ { - q / 2 } \} , } \\ { \mathrm { C V a R } _ { \alpha , e \sim \Pi } ( \mathfrak { R } _ { e } ( n ) ) \geq c _ { 0 } \mathrm { C V a R } _ { \alpha , e \sim \Pi } ( \mathcal { B } _ { e } ( n ) ) . \qquad } \end{array}\tag{A.16}
$$

Separation $c _ { e } ,$ distinguishability $K _ { e }$ , and radius $t _ { 0 , e }$ remain inside the aggregation. For $q = 2$ and $n K _ { e } \lambda _ { e } \ge t _ { 0 , e } ^ { - 2 } , \mathcal { B } _ { e } ( n ) = c _ { e } / ( n K _ { e } \lambda _ { e } )$ ; outside that interior regime the radius truncates the bound.

## A.4 A Boundary-Region Classification Family

Fix a known $0 < \delta \le 1 / 2$ and let $\mathcal { P } _ { \delta }$ be the family in Equation (6), with $X \sim \mathrm { U n i f } [ - 1 , 1 ]$ and parameter rectangle $| \theta | , | \dot { \mu } | \leq \delta / 8$ . Define

$$
\mathfrak { R } _ { \delta } ( n ) = \operatorname* { i n f } _ { \widehat { g } _ { n } } \operatorname* { s u p } _ { | \theta | , | \mu | \leq \delta / 8 } \mathbb { E } _ { \theta , \mu } [ R _ { \theta , \mu } ( \widehat { g } _ { n } ) - R _ { \theta , \mu } ^ { \star } ] ,
$$

where the infimum ranges over measurable, possibly randomized learning rules based on n independent observations.

Proposition A.1 (Nuisance confounding near a classification boundary). At the reference parameter (0, 0), the raw and efficient task information satisfy

$$
J _ { T T } = 4 \log 3 , \qquad F _ { \delta } = 4 \log \frac { 2 + \delta } { 2 - \delta } , \qquad 4 \delta \leq F _ { \delta } \leq \frac { 6 4 } { 1 5 } \delta .
$$

For every $n \geq 1$

$$
\frac { 1 } { 1 2 8 e } \operatorname* { m i n } \{ \delta ^ { 2 } , ( n \delta ) ^ { - 1 } \} \leq \Re _ { \delta } ( n ) \leq \operatorname* { m i n } \left\{ \frac { \delta ^ { 2 } } { 3 2 } , \frac { 1 - ( 1 - \delta ) ^ { n + 1 } } { 2 ( n + 1 ) \delta } \right\} .\tag{A.17}
$$

Consequently, the rate in Equation (8) holds with constants independent of n and δ.

Proof. Write $b _ { \delta } ( x ) = \mathbf { 1 } \{ | x | > \delta \}$ and $\eta _ { 0 } ( x ) = 1 / 2 + x / 4$ . Throughout the rectangle, $1 / 8 \leq \eta _ { \theta , \mu } \leq$ $7 / 8 ,$ so the binary likelihood is regular in $( \theta , \mu )$ for each fixed $\delta > 0$ . Inside $| x | \leq \delta .$ , the regression crosses $1 / 2 { \mathrm { ~ a t ~ } } - { \dot { 4 } } \theta \in [ - \delta / 2 , \delta / 2 ]$ . Outside, $| \dot { \theta } + \mu | \leq \delta / 4 < | x | / 4$ , so its sign relative to $1 / 2$ is the sign of x. Thus

$$
g _ { \theta , \mu } ^ { \star } ( x ) = { \bf 1 } \{ x \geq - 4 \theta \} , \quad \quad \mathcal { E } _ { \theta , \mu } ( g _ { a } ) = 2 ( a - \theta ) ^ { 2 } \quad \mathrm { f o r } \ g _ { a } ( x ) = { \bf 1 } \{ x \geq - 4 a \} , | a | \leq \delta / 8 .\tag{A.18}
$$

The second identity integrates the margin over the interval between the two thresholds, which lies entirely inside $[ - \delta , \delta ]$

Information. Let $v _ { 0 } ( x ) = \eta _ { 0 } ( x ) ( 1 - \eta _ { 0 } ( x ) ) = ( 4 - x ^ { 2 } ) / 1 6$ . The reference scores are

$$
s _ { T } = \frac { Y - \eta _ { 0 } ( X ) } { v _ { 0 } ( X ) } , \qquad s _ { N } = b _ { \delta } ( X ) s _ { T } .
$$

Since $b _ { \delta } ^ { 2 } = b _ { \delta } , J _ { T N } = J _ { N N } > 0$ , so the nuisance-projection coefficient is one and $s _ { T } ^ { \mathrm { e f f } } = \mathbf { 1 } \{ | X | \leq$ $\delta \} _ { { s _ { T } } }$ . Hence

$$
J _ { T T } = \int _ { - 1 } ^ { 1 } \frac { 8 } { 4 - x ^ { 2 } } d x = 4 \log 3 , \qquad F _ { \delta } = \int _ { - \delta } ^ { \delta } \frac { 8 } { 4 - x ^ { 2 } } d x = 4 \log \frac { 2 + \delta } { 2 - \delta } .\tag{A.19}
$$

On $[ - \delta , \delta ] , 2 \le 8 / ( 4 - x ^ { 2 } ) \le 3 2 / 1 5$ , giving the asserted bounds on $F _ { \delta }$

Decision separation and distinguishability. For $0 < t \leq \delta / 8$ , take $P _ { + t } = P _ { t , - t }$ and $P _ { - t } = P _ { - t , t }$ Their regression functions are $\eta _ { 0 } \pm t { \bf 1 } \{ | x | \le \delta \}$ and their Bayes rules disagree on $( - 4 t , 4 t )$ . The binary excess-risk identity therefore gives the exact separation

$$
\operatorname* { i n f } _ { g } \{ { \mathcal E } _ { + t } ( g ) + { \mathcal E } _ { - t } ( g ) \} = \frac 1 2 \int _ { - 4 t } ^ { 4 t } \left( 2 t - \frac { | x | } 2 \right) d x = 4 t ^ { 2 } .\tag{A.20}
$$

For Bernoulli parameters $p , q , { \mathrm { K L } } ( \mathrm { B e r } ( p ) \| \mathrm { B e r } ( q ) ) \leq ( p - q ) ^ { 2 } / [ q ( 1 - q ) ]$ . Here the alternatives agree outside the central region; inside, $p - q = 2 t$ and $q \in [ 5 / 1 6 , 1 1 / 1 6 ]$ . Since $P ( | X | \leq \delta ) = \delta$ and $F _ { \delta } \geq 4 \delta$

$$
\mathrm { K L } ( P _ { + t } \| P _ { - t } ) \leq \frac { 1 0 2 4 } { 5 5 } \delta t ^ { 2 } \leq \frac { 2 5 6 } { 5 5 } F _ { \delta } t ^ { 2 } < 5 F _ { \delta } t ^ { 2 } .\tag{A.21}
$$

Theorem 1 thus applies with $q = 2 , c _ { e } = 4 , K _ { e } = 5 , t _ { 0 , e } = \delta / 8 .$ , and $\lambda _ { e } = F _ { \delta }$ . Its proof gives

$$
\Re _ { \delta } ( n ) \geq \frac { 1 } { 2 e } \operatorname* { m i n } \left\{ \frac { \delta ^ { 2 } } { 6 4 } , \frac { 1 } { 5 n F _ { \delta } } \right\} \geq \frac { 1 } { 1 2 8 e } \operatorname* { m i n } \{ \delta ^ { 2 } , ( n \delta ) ^ { - 1 } \} ,
$$

using $F _ { \delta } \leq 6 4 \delta / 1 5$ in the last step.

Matching upper bound, including empty central samples. Let $\begin{array} { r } { I _ { i } = \mathbf { 1 } \{ | X _ { i } | \leq \delta \} , N = \sum _ { i } I _ { i } \sim } \end{array}$ $\mathrm { B i n } ( n , \delta )$ , and define

$$
{ \widehat \theta } = \mathrm { c l i p } _ { [ - \delta / 8 , \delta / 8 ] } \left( \frac { \sum _ { i = 1 } ^ { n } I _ { i } ( Y _ { i } - 1 / 2 - X _ { i } / 4 ) } { N + 1 } \right) .\tag{A.22}
$$

This estimator is zero when $N = 0$ . Conditional on the feature observations, each selected residual has mean $\theta ,$ variance at most $1 / 4 .$ , and independent noise. Before clipping, its mean-squared error is at most $( N / 4 + \theta ^ { 2 } ) / ( N + 1 ) ^ { 2 ^ { \prime } } \leq 1 / [ 4 ( N ^ { \prime } + 1 ) ^ { \prime }$ ]; clipping cannot increase squared error because the interval contains θ. The identity

$$
\mathbb { E } \frac { 1 } { N + 1 } = \frac { 1 - ( 1 - \delta ) ^ { n + 1 } } { ( n + 1 ) \delta }
$$

and Equation $\left( \mathrm { A . 1 8 } \right)$ give the second upper bound in Equation (A.17). The fixed rule $g _ { 0 }$ gives the first, since $\dot { 2 } \theta ^ { 2 } \leq \delta ^ { 2 } / 3 2$ . Taking the better of these two known rules proves the stated upper bound.

Only the central observations distinguish the least-favorable task alternatives: their expected number is nδ. The inverse-information rate is active once $n \delta ^ { 3 }$ is sufficiently large; otherwise the local decision scale $\delta ^ { 2 }$ caps the risk. The nuisance parameter here changes conditional probabilities outside the boundary region while preserving the Bayes rule at fixed θ. This is a statistical learning mechanism, not a mathematical model of the image-background intervention.

## A.5 CVaR, Information Coverage, and Stabilization

For $0 \textless \alpha \textless 1$ and nonnegative $Z ,$ the CVaR minimization formula holds for integrable $Z \ [ 9 ,$ Theorem 10]; the threshold-weight argument below gives its dual:

$$
\begin{array} { r l r } & { } & { \mathrm { C V a R } _ { \alpha } ( Z ) = \underset { \tau \in \mathbb { R } } { \operatorname* { i n f } } \left\{ \tau + \frac { \mathbb { E } [ ( Z - \tau ) _ { + } ] } { 1 - \alpha } \right\} = \underset { \xi \in \mathcal { W } _ { \alpha } } { \operatorname* { s u p } } \mathbb { E } [ \xi Z ] , } \\ & { } & { \mathcal { W } _ { \alpha } = \{ \xi : 0 \leq \xi \leq ( 1 - \alpha ) ^ { - 1 } , \mathbb { E } \xi = 1 \} . } \end{array}\tag{A.23}
$$

Both values are +∞ if $\mathbb { E } Z = + \infty$ . For integrable $Z ,$ a maximizing weight equals $( 1 - \alpha ) ^ { - 1 }$ above $v _ { \alpha } = \operatorname* { i n f } \{ v : \operatorname* { P r } ( Z \leq v ) \geq \alpha \}$ , zero below it, and fills the remaining mass at the threshold. Such fractional weighting is possible because $\operatorname* { P r } ( Z > v _ { \alpha } ) \leq 1 - \alpha \leq \bar { \operatorname* { P r } } ( Z \geq v _ { \alpha } )$ . The dual gives monotonicity, positive homogeneity, and $v _ { \alpha } \leq \mathrm { C V a R } _ { \alpha } ( Z )$ . For the finite audit distribution $\Pi _ { G }$ , let $\textstyle \pi _ { g } = \Pi _ { G } ( \{ \bar { g } \} ) \geq 0 , \sum _ { q } \pi _ { g } { \overset { . } { = } } 1$ . Then

$$
\operatorname { C V a R } _ { \alpha , \pi } ( Z _ { g } ) = \operatorname* { m a x } _ { \sum _ { g } \omega _ { g } = 1 \atop 0 \leq \omega _ { g } \leq \pi _ { g } / ( 1 - \alpha ) } \sum _ { g } \omega _ { g } Z _ { g } .\tag{A.24}
$$

These normalized tail weights, including threshold atoms, also govern Appendix E.4.

ProofofProposition 1. Let $\varepsilon > 0$ and $\lambda _ { e } \in [ 0 , \infty )$ be finite almost surely. Write $L _ { e } ^ { ( \varepsilon ) } = ( \lambda _ { e } + \varepsilon ) ^ { - 1 }$ and $\check { I } _ { \alpha } ^ { \mathrm { F G } } ( \varepsilon )$ for the certificate in Equation (10). If $I _ { \alpha } ^ { \mathrm { F G } } ( \varepsilon ) \geq \rho > \varepsilon .$ , then

$$
\begin{array} { c } { v _ { \alpha } ( L ^ { ( \varepsilon ) } ) \leq \mathrm { C V a R } _ { \alpha } ( L ^ { ( \varepsilon ) } ) = [ I _ { \alpha } ^ { \mathrm { F G } } ( \varepsilon ) ] ^ { - 1 } \leq \rho ^ { - 1 } , } \\ { \mathrm { P r } _ { \mathrm { \scriptsize { e } } ^ { \mathrm { \scriptsize { } } } } ( \lambda _ { e } \geq \rho - \varepsilon ) = \mathrm { \textstyle { P r } } _ { e \mathrm { \scriptsize { \sim } } \Pi } ( L _ { e } ^ { ( \varepsilon ) } \leq \rho ^ { - 1 } ) \geq \alpha . } \end{array}\tag{A.25}
$$

The last step is the quantile definition and includes equality at the information threshold.

Since $0 < L ^ { ( \varepsilon ) } \leq \varepsilon ^ { - 1 }$ , always $I _ { \alpha } ^ { \mathrm { F G } } ( \varepsilon ) \geq \varepsilon ;$ only $\rho > \varepsilon$ certifies a strictly positive floor. With four equally weighted groups and $\alpha = 0 . 9 5$ , one group already contains the entire upper-tail mass, so

$$
\mathrm { C V a R } _ { 0 . 9 5 } ( L ^ { ( \varepsilon ) } ) = \operatorname* { m a x } _ { g } L _ { g } ^ { ( \varepsilon ) } , \qquad I _ { 0 . 9 5 } ^ { \mathrm { F G } } ( \varepsilon ) = \operatorname* { m i n } _ { g } \lambda _ { g } + \varepsilon .\tag{A.26}
$$

Exact limit and positivity. Set $L ^ { ( 0 ) } = \lambda ^ { - 1 }$ , with $1 / 0 = + \infty$ ${ \mathrm { A s ~ } } \varepsilon { \mathrm { ~ \downarrow ~ 0 , ~ } } L ^ { ( \varepsilon ) } { \mathrm { ~ \uparrow ~ } } L ^ { ( 0 ) }$ , and monotone convergence applies to $\mathbb { E } [ \xi L ^ { ( \varepsilon ) } ]$ for every feasible ξ. The supremum over $\xi$ commutes with the increasing supremum along any sequence $\varepsilon _ { k } \downarrow 0 .$ , giving

$$
\operatorname { C V a R } _ { \alpha } ( L ^ { ( \varepsilon ) } ) \uparrow \operatorname { C V a R } _ { \alpha } ( \lambda ^ { - 1 } ) ,
$$

$$
I _ { \alpha } ^ { \mathrm { F G } } ( \varepsilon ) \downarrow I _ { \alpha } ^ { \mathrm { F G } } ( 0 ) : = \big [ \mathrm { C V a R } _ { \alpha } ( \lambda ^ { - 1 } ) \big ] ^ { - 1 } , \qquad 1 / ( + \infty ) = 0 .\tag{A.27}
$$

For nonnegative $Z ,$ the feasible weight $\xi = 1$ and variational choice $\tau = 0$ imply

$$
\mathbb { E } Z \le \mathrm { C V a R } _ { \alpha } ( Z ) \le \frac { \mathbb { E } Z } { 1 - \alpha } , \qquad I _ { \alpha } ^ { \mathrm { F G } } ( 0 ) > 0 \iff \mathbb { E } [ \lambda ^ { - 1 } ] < \infty .\tag{A.28}
$$

Under the geometric tail assumptions, this is equivalent to $\beta = ( c + \nu ) / ( 2 m ) > 1$ ; the exact certificate vanishes for $\beta \leq 1$ . Information coverage is a population statement under Π, separate from the confidence level of an estimated lower certificate.

## B Estimation and Certification

The bounds condition on a predictor, coordinates, calibration, and slices fixed independently of evaluation data. Evaluation-fitted normalization requires additional error control. Implementation details and numerical records appear in the artifact’s docs/extended\_appendix\_details/ (see the artifact availability statement).

## B.1 Operational Models and Matrix Targets

For evaluation distribution $P _ { g }$ and calibrated K-class logits $\ell _ { w }$ , maps $\Phi _ { T } ( x ) \in \mathbb { R } ^ { K \times d _ { T } }$ and $\Phi _ { N } ( x ) \in$ $\mathbb { R } ^ { K \times d _ { N } }$ convert task and nuisance parameters into logit changes:

$$
\ell _ { \theta , \mu } ( x ) = \ell _ { w } ( x ) + \Phi _ { T } ( x ) \theta + \Phi _ { N } ( x ) \mu , \qquad D = ( \Phi _ { T } \Phi _ { N } ) , \quad d = d _ { T } + d _ { N } .\tag{B.1}
$$

With p = softmax $\langle \ell _ { w } \rangle , H = \mathrm { d i a g } ( p ) - p p ^ { \intercal }$ , one-hot label vector $\mathbf { e } _ { Y }$ , and normalized feature coordinates $v = ( v _ { T } ^ { \top } v _ { N } ^ { \top } ) ^ { \top }$ , distinguish

$$
\begin{array} { r l } & { M ^ { \mathrm { p r e d } } = D ^ { \top } H D , \qquad M ^ { \mathrm { o b s } } = s s ^ { \top } , \quad s = D ^ { \top } ( \mathbf { e } _ { Y } - p ) , } \\ & { M ^ { \mathrm { f e a t } } = v v ^ { \top } , \qquad J _ { g } = \mathbb { E } _ { P _ { g } } [ M ] . } \end{array}\tag{B.2}
$$

For a binary scalar logit $\ell _ { \theta , \mu } = \ell _ { w } + \theta z + \mu b$ , this gives $M ^ { \mathrm { p r e d } } = p ( 1 - p ) v v ^ { \top }$ , with $p = \sigma ( \ell _ { w } )$ and $\boldsymbol { v } = ( z , b ) ^ { \top }$ . Predictive scores have conditional mean zero under the fitted label distribution even when features do not; their Schur complement is efficient information for this operational model. Observed-label moments equal predictive Fisher under conditional correct specification, $P _ { g } ( Y = \cdot \mid X ) = p ( X )$ ; otherwise their centered covariance subtracts $\mathbb { E } [ s ] \mathbb { E } [ s ] ^ { \top }$ . Feature geometry satisfies $J _ { g } = \Sigma _ { g } + \mu _ { g } \mu _ { q } ^ { \top }$ , with $\mu _ { g } = \mathbb { E } [ v ]$ and $\Sigma _ { g } = \operatorname { C o v } ( v )$ . The prospective centered comparison uses within-slice covariance $\Sigma _ { g }$ with divisor $n _ { g }$ and unchanged global scales, removing the mean contribution without rescaling.

Prospective selection uses uncentered feature–Schur fractions; Appendix D.2 compares centered and predictive targets. Within-group normalization of $p ( 1 - p )$ weights changes the effective nuisance ridge and differs from the predictive moment. Hidden discovery leaves this factor unnormalized and shrinks toward the same-class pooled matrix, with $\eta = 0 . 1 0$ . Removing ridge leaves the declared matrix, including shrinkage, unchanged. Normalization conventions and nuisance-specification checks appear in Appendices D and D.5.

Operational subspaces and cost. A task coordinate specifies the predictive contrast to be audited; nuisance coordinates specify variations expected to preserve it. Candidate constructions include an answer-logit contrast with controlled prompt-template variation for a transformer, a region-level class contrast with validated appearance changes for segmentation, and a conditional denoising-score contrast for a diffusion model. Logits, probes, intermediate representations, and low-rank parameter perturbations provide finite-dimensional coordinates. For total operational dimension $d = d _ { T } + d _ { N }$ dense moment storage costs $O ( d ^ { 2 } )$ , and direct factorization or eigendecomposition costs $O ( d ^ { 3 } )$ per slice. Minibatch accumulation and iterative eigensolvers support larger operational subspaces; feature or score extraction remains dependent on the backbone and input size. These examples are candidate extensions for future evaluation.

Admission of new nuisance specifications. A proposed admission protocol prespecifies independent checks of nuisance relevance, task preservation, split stability, and nuisance-block conditioning. An identifiability conclusion requires all four checks to pass; a failed prerequisite is recorded as an unvalidated specification. For a finite admitted family, report the minimum certificate with uncertainty accounting for the family. The present experiments evaluate prespecified nuisance hypotheses, with implemented checks and sensitivity analyses in Appendices B.4, C, and D.5. The general admission procedure remains to be evaluated.

## B.2 Ridge Regularization and the Unregularized Target

For $\begin{array} { r } { J = \left( \begin{array} { c c } { A } & { B } \\ { B ^ { \top } } & { C } \end{array} \right) \succeq 0 } \end{array}$ , with A = J<sub>TT</sub>, B = J<sub>TN</sub>, C = J<sub>NN</sub> ≻ 0, and $\lambda _ { \mathrm { r i d } } > 0$ , define

$$
S _ { 0 } ( J ) = A - B C ^ { - 1 } B ^ { \top } , \qquad S _ { \mathrm { r i d } } ( J ) = A - B ( C + \lambda _ { \mathrm { r i d } } I ) ^ { - 1 } B ^ { \top } .\tag{B.3}
$$

Both are positive semidefinite. The resolvent identity, whose inverse factors commute, gives

$$
\begin{array} { r l } & { ~ \Delta _ { \mathrm { r i d } } ( J ) : = S _ { \mathrm { r i d } } ( J ) - S _ { 0 } ( J ) = \lambda _ { \mathrm { r i d } } B C ^ { - 1 } ( C + \lambda _ { \mathrm { r i d } } I ) ^ { - 1 } B ^ { \top } \succeq 0 , } \\ & { \| \Delta _ { \mathrm { r i d } } ( J ) \| _ { \mathrm { o p } } \leq \frac { \lambda _ { \mathrm { r i d } } M ^ { 2 } } { \kappa ( \kappa + \lambda _ { \mathrm { r i d } } ) } \quad \mathrm { i f } C \succeq \kappa I , \kappa > 0 , \| B \| _ { \mathrm { o p } } \leq M . } \end{array}\tag{B.4}
$$

Conditioning is essential: scalar blocks $A = 1 , B = { \sqrt { c } } , C = c > 0$ give a positive-semidefinite joint matrix with $S _ { 0 } = 0$ and $S _ { \mathrm { r i d } } = { \lambda _ { \mathrm { r i d } } } / { ( c + \lambda _ { \mathrm { r i d } } ) } { \stackrel { . } {  } } 1$ as $c / \lambda _ { \mathrm { r i d } } \downarrow 0$ . Comparisons therefore retain the declared standardized nuisance coordinates and ridge value.

## B.3 Error Allowances and Lower Certificates

Proposition B.1 (Simultaneous lower certificate). Fix G groups and weights $\pi _ { g } > 0$ summing to one. Within each group, let $M _ { g i }$ be i.i.d. with mean $J _ { g } , 0 \preceq M _ { g i } \preceq R _ { g } I _ { d }$ , for specified finite $R _ { g } ,$ , and $\begin{array} { r } { \widehat { J } _ { g } = n _ { g } ^ { - 1 } \sum _ { i } M _ { g i } . \ F o r \ 0 < \delta < 1 _ { } } \end{array}$ , set

$$
\begin{array} { r l } & { u _ { \delta } = \log ( 2 d G / \delta ) , \qquad t _ { g } = R _ { g } \left( \sqrt { 2 u _ { \delta } / n _ { g } } + 2 u _ { \delta } / ( 3 n _ { g } ) \right) , } \\ & { \mathcal { E } _ { \delta } = \overset { , } { \underset { g } { \bigcap } } \{ \| \widehat { J } _ { g } - J _ { g } \| _ { \mathrm { o p } } \leq t _ { g } \} , \qquad \operatorname* { P r } ( \mathcal { E } _ { \delta } ) \geq 1 - \delta . } \end{array}\tag{B.5}
$$

Assume each $C _ { g } \succ 0$ . Define observable bounds and allowances by

$$
\begin{array} { r } { \overline { { M } } _ { g } = \| \widehat { B } _ { g } \| _ { \mathrm { o p } } + t _ { g } , \qquad \quad \frac { \kappa _ { g } = \lambda _ { \mathrm { m i n } } ( \widehat { C } _ { g } ) - t _ { g } , } { h _ { \mathrm { r i d } } ( t , M ) } } \\ { h _ { \mathrm { r i d } } ( t , M ) = t + ( 2 M t + t ^ { 2 } ) / \lambda _ { \mathrm { r i d } } + M ^ { 2 } t / \lambda _ { \mathrm { r i d } } ^ { 2 } , \quad a _ { g } = h _ { \mathrm { r i d } } ( t _ { g } , \overline { { M } } _ { g } ) , } \\ { d _ { g } = \frac { \lambda _ { \mathrm { r i d } } \overline { { M } } _ { g } ^ { 2 } } { \underline { { \kappa } } _ { g } \left( \underline { { \kappa } } _ { g } + \lambda _ { \mathrm { r i d } } \right) } \qquad \quad w h e n a l l \underline { { \kappa } } _ { g } > 0 . } \end{array}\tag{B.6}
$$

Report the lower certificate in (14) with audit distribution π only when these conditioning margins are positive; otherwise record conditioning failure. Its probability of exceeding the unregularized population certificate is at most $\delta .$ On $\mathcal { E } _ { \delta } ,$ , a reported level $\underline { { { I } } } _ { \alpha , \pi } ~ \geq ~ \rho ~ > ~ \varepsilon$ also implies $\begin{array} { r } { \sum _ { g : \lambda _ { \operatorname* { m i n } } ( S _ { 0 } ( J _ { g } ) ) \geq \rho - \varepsilon } \pi _ { g } \geq \alpha . } \end{array}$

Proof. For $X _ { g i } = M _ { g i } - J _ { g } , \| X _ { g i } \| _ { \mathrm { o p } } \leq R _ { g }$ and $\mathbb { E } [ X _ { g i } ^ { 2 } ] = \mathbb { E } [ M _ { g i } ^ { 2 } ] - J _ { g } ^ { 2 } \preceq R _ { g } ^ { 2 } I$ . Matrix Bernstein [23, Theorem $1 . 4 ]$ with the displayed $t _ { g }$ gives failure at most $\delta / G$ per group; the union bound allows dependence between group estimates. On this event, block extraction and Weyl give $\| B _ { g } \| _ { \mathrm { o p } } \leq \overline { { M } } _ { g }$ and $C _ { g } \succeq \underline { { \kappa } } _ { g } I$ , so (B.4) bounds the ridge gap by $d _ { g }$

For the estimation error, write $Q = ( C + \lambda _ { \mathrm { { r i d } } } I ) ^ { - 1 } , \widehat { Q } = ( \widehat { C } + \lambda _ { \mathrm { { r i d } } } I ) ^ { - 1 }$ , and $\Delta B = \widehat { B } - B$ Positive semidefiniteness and the resolvent identity give $\lVert Q \rVert , \lVert \widehat { Q } \rVert \leq \lambda _ { \mathrm { r i d } } ^ { - 1 }$ and $\lVert \widehat { Q } - Q \rVert \leq t / \lambda _ { \mathrm { r i d } } ^ { 2 }$ Decompose the quadratic-block difference as $\Delta B \widehat { Q } \widehat { B } ^ { \intercal } + B \widehat { Q } \Delta B ^ { \intercal } + B ( \widehat { Q } - Q ) B ^ { \intercal }$ . If ∥B∥ ≤ M, its three norms are bounded by $t ( M + t ) / \lambda _ { \mathrm { r i d } } , M t / \lambda _ { \mathrm { r i d } }$ , and $\dot { \boldsymbol { M } } ^ { 2 } t / \lambda _ { \mathrm { r i d } } ^ { 2 }$ . Adding the task-block error t proves the bound $h _ { \mathrm { r i d } } ( t , M )$ and hence $a _ { g }$

Writing $\lambda _ { g } ^ { 0 } = \lambda _ { \operatorname* { m i n } } ( S _ { 0 } ( J _ { g } ) )$ , Weyl now yields

$$
\begin{array} { r } { \lambda _ { g } ^ { 0 } \geq \lambda _ { \operatorname* { m i n } } \big ( S _ { \mathrm { r i d } } ( J _ { g } ) \big ) - d _ { g } \geq \lambda _ { \operatorname* { m i n } } \big ( S _ { \mathrm { r i d } } ( \widehat { J } _ { g } ) \big ) - a _ { g } - d _ { g } . } \end{array}\tag{B.7}
$$

Since $S _ { 0 } ( J _ { g } ) \succeq 0$ , clipping in (14) preserves $\underline { { \lambda } } _ { g } \le \lambda _ { g } ^ { 0 }$ . Thus monotonicity of CVaR and reciprocation give

$$
\underline { { I } } _ { \alpha , \pi } \leq I _ { \alpha , \pi } ^ { 0 } : = \left[ \mathrm { C V a R } _ { \alpha , g \sim \pi } ( ( \lambda _ { g } ^ { 0 } + \varepsilon ) ^ { - 1 } ) \right] ^ { - 1 } .\tag{B.8}
$$

The information-coverage conclusion follows from Appendix A.5.

□

Confidence concerns held-out observations; coverage concerns the fixed weights π. A finite predeclared nuisance family can be included in the union bound over group–specification pairs; the infimum of certificates preserves coverage separately for each specification. Appendix C.4 supplies matrix allowances for frozen soft gates, including random membership normalization, pooling, and support correction, to which the same deterministic argument applies.

## B.4 Implemented Audit and Numerical Comparison

Tables B.1–B.2 audit uncentered feature moments of normalized scalar task and nuisance coordinates $z _ { i } , b _ { i }$ , with uniform within-group weights. Separately from the finite-sample proposition, Gaussian/Wald simulation uses $1 \mathrm { { 0 ^ { 5 } } }$ moment draws per group at nominal 95% confidence. For general normalized weights $w _ { i }$ and $u _ { i } = ( z _ { i } ^ { 2 } , z _ { i } b _ { i } , b _ { i } ^ { 2 } ) ^ { \top }$ , draw $m ^ { * } = ( A ^ { * } , B ^ { * } , C ^ { * } ) ^ { \top } \sim N ( \widehat { m } , \widehat { \Sigma } _ { \widehat { m } } )$ , where

$$
\widehat { m } = \sum _ { i } w _ { i } u _ { i } , \qquad \widehat { \Sigma } _ { \widehat { m } } = \frac { \sum _ { i } w _ { i } ^ { 2 } ( u _ { i } - \widehat { m } ) ( u _ { i } - \widehat { m } ) ^ { \top } } { 1 - \sum _ { i } w _ { i } ^ { 2 } } .\tag{B.9}
$$

Uniform weights give sample covariance divided by n. Simulation fixes realized coordinatestandardization constants and weights, omitting evaluation-fitted standardization error. Bonferroni tail $\delta _ { \mathrm { W } } = 0 . 0 5 / ( 3 G )$ covers nuisance conditioning, ridge lower information, and an upper ridge gap within a run $( \dot { G } = 4$ Waterbirds; 18 CivilComments).

Let $J ^ { * }$ have blocks $( A ^ { * } , B ^ { * } , C ^ { * } )$ , and write $\widehat { F } _ { g } ^ { 0 } = S _ { 0 } ( \widehat { J } _ { g } ) , \widehat { F } _ { g } ^ { \mathrm { r i d } } = S _ { \mathrm { r i d } } ( \widehat { J } _ { g } )$ . The $\delta _ { \mathrm { W } }$ -quantiles $C _ { g } ^ { L } , L _ { g } ^ { \mathrm { r i d } } , L _ { g } ^ { 0 }$ of $C ^ { * } , [ S _ { \mathrm { r i d } } ( J ^ { * } ) ] _ { + } , [ S _ { 0 } ( J ^ { * } ) ] _ { + }$ are clipped to $[ 0 , \widehat { C } _ { g } ] , [ 0 , \widehat { F } _ { g } ^ { \mathrm { r i d } } ] , [ 0 , \widehat { F } _ { g } ^ { 0 } ]$ , respectively. The $( 1 - \delta _ { \mathrm { { W } } } )$ -quantile $U _ { g } ^ { \Delta }$ of the closed-form ridge gap is enlarged to at least its plug-in value. Lower and upper quantiles use lower and higher order statistics, respectively. The implemented lower information is

$$
\begin{array} { r } { \underline { { \lambda } } _ { g } ^ { \mathrm { W } } = \operatorname* { m i n } \{ [ L _ { g } ^ { \mathrm { r i d } } - U _ { g } ^ { \Delta } ] _ { + } , L _ { g } ^ { 0 } , \widehat { F } _ { g } ^ { 0 } \} . } \end{array}\tag{B.10}
$$

A denominator $C ^ { * }$ or $C ^ { * } + \lambda _ { \mathrm { r i d } }$ at or below $1 0 ^ { - 1 2 }$ gives zero for its corresponding Schur value and an infinite gap; otherwise (B.4) is used even for non-PSD draws. An infinite upper gap or $C _ { q } ^ { L } \leq 1 0 ^ { \cdot }$ −12 forces zero and an uncertifiable status; support, empirical-PSD, and identity checks can also prevent certification. With $\lambda _ { \mathrm { r i d } } = 1 0 ^ { - 3 }$ , certificates with superscripts 0, rid, and W aggregate $\widehat { F } _ { g } ^ { 0 } , \widehat { F } _ { g } ^ { \mathrm { r i d } }$ , and ${ \underline { { \lambda } } } _ { g } ^ { \mathrm { W } }$ , respectively, using uniform environment weights, $\alpha = 0 . 9 5 , \varepsilon = 1 0 ^ { - 8 }$ , and fractional CVaR threshold-atom mass.

Table B.1: Same-cohort summaries of 75 default-ridge held-out audits. Conditioning/gap parentheses are within-run nominal Wald lower/upper values; entries are extrema across runs, not joint confidence statements across seeds or methods. Inflation is ridge minus exact plug-in certificate.
<table><tr><td></td><td colspan="3">Conditioning and ridge gap</td><td colspan="3">Certificates</td></tr><tr><td>Scope</td><td>Audits</td><td>Minimum  $J _ { N N }$ </td><td>Maximum group gap</td><td>Maximum inflation</td><td>Minimum exact</td><td>Minimum Wald lower</td></tr><tr><td>CivilComments diagnostics</td><td>5</td><td>1.456 (1.417)</td><td>0.00116 (0.00133)</td><td>0.00018</td><td>0.4970</td><td>0.4618</td></tr><tr><td>Waterbirds diagnostics</td><td>5</td><td>0.211 (0.178)</td><td>0.00125 (0.00166)</td><td>0.00071</td><td>0.1639</td><td>0.1504</td></tr><tr><td>Frozen-feature training</td><td>30</td><td>0.227 (0.192)</td><td>0.00144 (0.00199)</td><td>0.00062</td><td>0.1972</td><td>0.1814</td></tr><tr><td>End-to-end training</td><td>35</td><td>0.349 (0.349)</td><td>0.00252 (0.00300)</td><td>0.00140</td><td>0.2185</td><td>0.2004</td></tr><tr><td>All audits</td><td>75</td><td>0.211 (0.178)</td><td>0.00252 (0.00300)</td><td>0.00140</td><td></td><td>0.1504</td></tr></table>

Every run passed conditioning and satisfied $\underline { { I } } _ { 0 . 9 5 } ^ { \mathrm { W } } \le \widehat { I } _ { 0 . 9 5 } ^ { 0 } \le \widehat { I } _ { 0 . 9 5 } ^ { \mathrm { r i d } } ;$ ; Waterbirds diagnostic values are 0.1504, 0.1639, and 0.1646. Across the five CivilComments diagnostic seeds, the mean ridge and unregularized certificates are 0.4993 and 0.4991, respectively. The inequality $\underline { { I } } _ { 0 . 9 5 } ^ { \mathrm { W } } \le \widehat { I } _ { 0 . 9 5 } ^ { 0 }$ is enforced by the caps and does not establish confidence coverage. Training audits use fresh models: original training runs lacked the retained environment blocks needed for checkpoint-specific auditing.

For the separate audits in Table B.2, frozen Group DRO + FG maximizes validation certificate within two percentage points of the best validation worst-group accuracy among {0.1, 0.3}; seeds 17–21 select (0.3, 0.1, 0.3, 0.3, 0.3). Repeated frozen-feature training/audit runs at ridges $1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 }$ give maximum group gaps 0.0001443, 0.0014374, 0.0138286 and Wald upper values 0.0001996, 0.0019880, 0.0191165.

Table B.2: Separate fresh training audits: five-seed means at the stated coefficients. Table B.1 instead includes frozen models selected over the full grid {0.01, 0.03, 0.1, 0.3}.
<table><tr><td>Training setting</td><td>Ridge plug-in</td><td>Exact plug-in</td><td>Wald lower</td></tr><tr><td>Frozen  $\mathrm { F G } , \lambda _ { \mathrm { F G } } = 0 . 0 3$ </td><td>0.4002</td><td>0.3999</td><td>0.3663</td></tr><tr><td>Frozen Group  $\mathrm { D R O } + \mathrm { F G } ,$  selected from {0.1, 0.3}</td><td>0.6254</td><td>0.6252</td><td>0.5664</td></tr><tr><td>End-to-end Ġroup  $\mathrm { D R O } + \mathrm { F G } , \lambda _ { \mathrm { F G } } = 1 0 ^ { - 4 }$ </td><td>0.5447</td><td>0.5442</td><td>0.4890</td></tr></table>

## C Hidden-Slice Discovery

Appendix D.4 reports this protocol’s outcomes. Frozen-gate evaluation recomputes membership normalization and support correction. Prompts, model revision and execution details are in the artifact.

## C.1 Gate Family and Support Constraints

For the frozen standardized 576-dimensional representation $\psi ( x )$ and task class c, define a logistic gate with parameters $\phi = \left( a _ { c } , b _ { c } \right)$

$$
\begin{array} { r l } & { h _ { \phi , c } ( x ) = \sigma ( a _ { c } ^ { \top } \psi ( x ) + b _ { c } ) , \qquad q _ { i } ^ { 0 } = \cfrac { h _ { \phi , c } ( x _ { i } ) } { \sum _ { j \in \mathcal { D } _ { c } } h _ { \phi , c } ( x _ { j } ) } , \quad \sigma ( t ) = ( 1 + e ^ { - t } ) ^ { - 1 } , } \\ & { m _ { \mathrm { n o m } } = \tau _ { \mathrm { s l i c e } } n _ { c } , \qquad m _ { \mathrm { E S S } } = \operatorname* { m i n } \{ n _ { c } , \operatorname* { m a x } \{ 1 , 0 . 8 m _ { \mathrm { n o m } } \} \} , \qquad b _ { \mathrm { m a x } } = \operatorname* { m i n } \{ 1 , 2 / m _ { \mathrm { n o m } } \} . } \end{array}
$$

Here $\mathcal { D } _ { c }$ contains the current sample’s examples with task label $c , n _ { c } = | \mathcal { D } _ { c } |$ , and $\tau _ { \mathrm { s l i c e } } = 0 . 1 0$ is within-class mass. Discovery calibrates $b _ { c }$ to this mean membership; evaluation freezes the selected intercept, without re-centering or mass recalibration. The cap denominator is nominal slice size, not the ESS floor.

Support correction uses Euclidean projection of $q ^ { 0 }$ onto the capped simplex only if its maximum exceeds $b _ { \mathrm { m a x } } .$ Writing the result as $q ^ { \mathrm { { \bar { c a p } } } }$ and $u _ { i } = 1 / n _ { c } ,$ it takes $\widetilde { q } = \dot { u } + t ( q ^ { \mathrm { { c a p } } } - u )$ with the largest $t \in [ 0 , 1 ]$ ] satisfying $\textstyle \sum _ { i } { \tilde { q } } _ { i } ^ { 2 } \leq 1 / m _ { \mathrm { E S S } }$ . Thus $\begin{array} { r } { \widetilde { q } \ge 0 , \sum _ { i } \widetilde { q } _ { i } = 1 } \end{array}$ , max $\tilde { q } _ { i } \ \leq \ b _ { \mathrm { m a x } }$ , and $\begin{array} { r } { n _ { \mathrm { e f f } } = ( \sum _ { i } \widetilde { q } _ { i } ^ { 2 } ) ^ { - 1 } \ge m _ { \mathrm { E S S } } } \end{array}$ . These constraints do not imply nuisance conditioning. Each sample uses its own $n _ { c }$ , making $\widetilde { q }$ batch-dependent. Reported mass averages raw selected-class membership; ESS uses ${ \widetilde { q } } .$

Metrics and controls. Fisher and high-loss gates report accuracy, uncalibrated binary NLL (temperature one), and post-hoc composition averaged under ${ \widetilde { q } } ,$ without information pooling. Predicted labels use a zero-logit threshold; calibrated NLL is stored separately. Random controls weight $k = \lceil \tau _ { \mathrm { s l i c e } } n _ { c } \rceil$ sampled rows of the Fisher-selected class uniformly; the constant gate uses $1 / n _ { c }$ . Controls using feature moments and centered covariances reevaluate the predictive-Fisher and high-los gates without rediscovery.

## C.2 Discovery Objective and Settings

With matrix $M _ { i }$ and same-class pool $\begin{array} { r } { \widehat { J } _ { \mathrm { p o o l } , c } \ = n _ { c } ^ { - 1 } \sum _ { i \in \mathcal { D } _ { c } } M _ { i } } \end{array}$ , the ideal constrained objective searches over class–gate pairs:

$$
\begin{array} { r l } { \widehat { J } _ { \phi , c } ^ { ( \eta ) } = ( 1 - \eta ) \displaystyle \sum _ { i } \widetilde { q } _ { i } ( \phi ) M _ { i } + \eta \widehat { J } _ { \mathrm { p o o l } , c } , } & { } \\ { ( c ^ { \star } , \phi ^ { \star } ) \in \displaystyle \arg \operatorname* { m a x } _ { c , \phi } \Big \{ \lambda _ { \mathrm { m i n } } \Big ( S _ { \mathrm { r i d } } \big ( \widehat { J } _ { \phi , c } ^ { ( \eta ) } \big ) \Big ) + \varepsilon \Big \} ^ { - 1 } , \qquad 0 \le \eta < 1 . } \end{array}\tag{C.1}
$$

Table C.1: Fixed hidden-discovery settings; sensitivity grids are declared before metadata reveal.
<table><tr><td>Component</td><td>Specification</td></tr><tr><td>Representation/nuisance</td><td>Gate: frozen ImageNet MobileNetV3-small; nuisance: CLIP ViT-B/32 image projection onto a fixed water-minus-land text direction.</td></tr><tr><td>Head and calibration</td><td>Binary linear head: 1,200 full-batch AdamW steps, learning rate 0.01, weight decay  $1 0 ^ { - 4 } ;$  temperature minimizes calibration NLL over 400 log-spaced values in [0.05, 20].</td></tr><tr><td>Gate optimization</td><td>Ten restarts; 1,000 AdamW steps, learning rate 0.03, weight decay  $1 0 ^ { - 4 } ;$  Gaussian initial weights with standard deviation 0.02.</td></tr><tr><td>Matrix and stabilization</td><td> $M _ { i } = p _ { i } ( 1 - p _ { i } ) v _ { i } v _ { i } ^ { \top }$  on two task/nuisance coordinates;  $\eta = 0 . 1 0 ,$   $\lambda _ { \mathrm { r i d } } = 1 0 ^ { - 3 } , \varepsilon = 1 0 ^ { - 8 } .$ </td></tr><tr><td>Replication/sensitivity</td><td>Seeds  $1 7 - 2 1$  ; five validation folds; mass and shrinkage grids each {0.05, 0.10, 0.20}, headline 0.10.</td></tr></table>

For the scalar task/nuisance blocks, write $\begin{array} { r c l } { \widehat { F } ^ { \mathrm { r i d } } } & { = } & { S _ { \mathrm { r i d } } ( \widehat { J } _ { \phi , c } ^ { ( \eta ) } ) } \end{array}$ Optimization and checkpoint/restart/class selection use $- \log ( \widehat { F } ^ { \mathrm { r i d } } + \varepsilon ) - 1 0 ( \bar { h } / \tau _ { \mathrm { s l i c e } } - 1 ) ^ { 2 }$ , where $\begin{array} { r } { \bar { h } = n _ { c } ^ { - 1 } \sum _ { i } h _ { \phi , c } ( x _ { i } ) } \end{array}$ The intercept is recalibrated before discovery evaluation; only mass-feasible finite checkpoints are retained, including a constant-gate baseline. High-loss search uses the same gate, mass and support rules, replacing negative log information with weighted, uncalibrated binary NLL. Information uses $\boldsymbol { v } _ { i } = ( z _ { i } , b _ { i } ) ^ { \top } \colon z _ { i }$ is the raw head logit $f _ { w } ( x _ { i } )$ standardized by fitting-subset moments, $b _ { i }$ the standardized CLIP nuisance score, and $p _ { i } = \sigma ( f _ { w } ( x _ { i } ) / T )$ ). Calibration changes the unnormalized predictive factor. Shrinkage pools complete matrices within the current sample’s class before taking the Schur complement. Feature controls omit $p _ { i } ( 1 - p _ { i } )$ but retain gate weights. The centered control forms gated and uniform same-class covariances separately, then pools them with weights $1 - \eta$ and η before taking the Schur complement. For CLIP, normalize five prompt embeddings per concept, normalize their concept means, then normalize the water-minus-land difference. Its dot product with normalized image embeddings gives the raw nuisance score, without background/group annotations.

## C.3 Fitting and Independent Evaluation

Table C.2 uses task labels and official splits throughout; background/group metadata remain sealed until all seeds finish held-out evaluation. Jaccard overlap uses the $k = \operatorname* { m a x } \{ 1 , \lceil \tau _ { \mathrm { s l i c e } } n _ { c } \rceil \}$ largest raw memberships in each selected class, intersected on original row identities; different classes give zero. None of the ten Fisher/high-loss gates has a cutoff tie. Fold stability applies this rule to the official training data’s 15% calibration subset: average same-class fold-pair overlaps, then weight these class averages by selected-fold counts normalized over classes with at least two gates. Neither comparison uses positive-membership supports or corrected-weight ranks.

## C.4 Held-Out Targets and Uncertainty

Conditional on fitting data and fitted quantities, let $P _ { c }$ be the selected-class distribution and $n = n _ { c }$ its evaluation sample size. For frozen gate h and matrix map M,

$$
J _ { h } = \frac { A } { p } , \quad A = \mathbb { E } _ { P _ { c } } [ h M ] , \quad p = \mathbb { E } _ { P _ { c } } [ h ] > 0 ; \qquad \widehat { J } _ { h } = \frac { \widehat { A } } { \widehat { p } } , \quad ( \widehat { A } , \widehat { p } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( h _ { i } M _ { i } , h _ { i } ) .\tag{C.2}
$$

Table C.2: Fitting and evaluation sequence under the frozen protocol.
<table><tr><td>Stage</td><td>Fitted or computed</td><td>Carried forward</td></tr><tr><td>Training/calibration</td><td>Split official training data 85/15 by task label: fit head and coordinate normalizers on  $8 5 \%$  ; temperature on 15%. CLIP scores use official-training, then fit-subset normalization.</td><td>Fixed predictor, coordinates, nuisance direction, temperature.</td></tr><tr><td>Validation folds</td><td>Five shared task-stratified assignments across selectors; Protocol assessment; fold discover on complementary folds, evaluate on held-out gates are not ensembled. folds; assess stability on the 15% training-calibration subset.</td><td></td></tr><tr><td>Full validation</td><td>Refit the gate under the fixed protocol; select class, gate, Freeze gate weights, and restart by discovery utility.</td><td>calibrated intercept, selected class, support settings.</td></tr><tr><td>Test</td><td>Apply frozen memberships; recompute admissible weights, same-class pool, and metrics before metadata reveal.</td><td>Held-out results, then post-hoc interpretation.</td></tr></table>

Support correction leaves all five primary Fisher gates’ test weights unchanged; for high-loss seed 19 it raises ESS from 94.48 to 102.72. Frozen memberships need not give fixed numerical weights or inactive bootstrap correction.

Proposition C.1 (Frozen-gate ratio bound). For i.i.d. evaluation observations from $P _ { c } ,$ assume $0 \leq h \leq 1$ and $0 \preceq M \preceq M _ { \operatorname* { m a x } } I _ { d } ,$ , with finite $M _ { \mathrm { m a x } }$ specified independently of evaluation data. Given $0 < \delta < 1$ , let

$$
u = \log ( 4 d / \delta ) , r _ { A } = M _ { \operatorname* { m a x } } \left( \sqrt { 2 u / n } + 2 u / ( 3 n ) \right) , r _ { p } = \sqrt { \log ( 4 / \delta ) / ( 2 n ) } .
$$

With conditional probability at least $1 - \delta ,$ , simultaneously $\| \widehat { A } - A \| _ { \mathrm { o p } } \leq r _ { A }$ and $| \widehat { p } - p | \le r _ { p }$ Whenever $\underline { { p } } = \widehat { p } - r _ { p } > 0$ on this event,

$$
\| \widehat { J } _ { h } - J _ { h } \| _ { \mathrm { o p } } \leq r _ { J } : = \frac { r _ { A } + \| \widehat { J } _ { h } \| _ { \mathrm { o p } } r _ { p } } { \widehat { p } - r _ { p } } .\tag{C.3}
$$

Proof. The centered matrices $h _ { i } M _ { i } - A$ have norm at most $M _ { \mathrm { m a x } }$ and variance bounded by $M _ { \mathrm { m a x } } ^ { 2 } I _ { d }$ Matrix Bernstein [23, Theorem 1.4] gives the numerator event with failure at most $\delta / 2 ;$ Hoeffding gives the denominator event with failure at most $\delta / 2 .$ . Their union bound and

$$
\widehat { J } _ { h } - J _ { h } = \frac { ( \widehat { A } - A ) - \widehat { J } _ { h } ( \widehat { p } - p ) } { p } , \qquad p \geq \widehat { p } - r _ { p } > 0 ,
$$

prove the observable bound. The shared random denominator is therefore explicitly controlled.

Pooling and support correction. Set $J _ { \mathrm { p o o l } , c } = \mathbb { E } _ { P _ { c } } [ M ]$ and $J _ { h } ^ { ( \eta ) } = ( 1 - \eta ) J _ { h } + \eta J _ { \mathrm { p o o l } , c } .$ Allocate $\delta _ { \mathrm { r a t i o } } + \delta _ { \mathrm { p o o l } } = \delta .$ using the former in Proposition C.1. For the evaluation mean ${ \widehat J } _ { \mathrm { p o o l } , c } = n ^ { - 1 } \sum _ { i } M _ { i }$ matrix Bernstein supplies

$$
r _ { \mathrm { p o o l } } = M _ { \mathrm { m a x } } \left[ \sqrt { \frac { 2 \log ( 2 d / \delta _ { \mathrm { p o o l } } ) } { n } } + \frac { 2 \log ( 2 d / \delta _ { \mathrm { p o o l } } ) } { 3 n } \right] .
$$

Writing $\begin{array} { r } { \widetilde { J } _ { h } = \sum _ { i } \widetilde { q } _ { i } M _ { i } } \end{array}$ , the triangle inequality gives $\begin{array} { r } { \| \widetilde { J } _ { h } - \widehat { J } _ { h } \| _ { \mathrm { o p } } \leq M _ { \operatorname* { m a x } } \sum _ { i } | \widetilde { q } _ { i } - q _ { i } ^ { 0 } | = : r _ { \mathrm { p r o j } } } \end{array}$ Hence, on the simultaneous event and when $\underline { { p } } > 0$

$$
\| ( 1 - \eta ) \widetilde { J } _ { h } + \eta \widehat { J } _ { \mathrm { p o o l } , c } - J _ { h } ^ { ( \eta ) } \| _ { \mathrm { o p } } \leq ( 1 - \eta ) ( r _ { J } + r _ { \mathrm { p r o j } } ) + \eta r _ { \mathrm { p o o l } } .\tag{C.4}
$$

The union bound permits dependence between pool and ratio estimates; $r _ { \mathrm { p r o j } }$ covers batch-dependent support correction. If the pool is fixed from fitting data, it belongs to the conditional target and needs no evaluation-error term. Appendix B.3 converts the matrix allowance into a Schur/eigenvalue lower certificate, requiring positive lower membership mass and nuisance-conditioning margins. The target retains shrinkage; evaluation-estimated centering additionally requires first-moment control.

Reported uncertainty. Each of 2,000 within-class bootstrap replicates resamples examples with membership, coordinates, logit and label together, recomputing support-corrected weights and the same-class pool. Gate coefficients/intercept, normalizers and temperature stay frozen. The 95% interval uses the $2 . 5 \mathrm { t h } / 9 7$ .5th percentiles of the pooled predictive matrix’s ridge Schur value, retaining $\eta = 0 . 1 0$ and $\lambda _ { \mathrm { r i d } } = 1 0 ^ { - 3 }$ , with linear quantile interpolation; ε enters inverse information separately. For any reported metric T, form paired differences $\dot { \Delta } _ { s } = T _ { s } ^ { \mathrm { g a t e } } - T _ { s } ^ { \mathrm { c o n t r o l } }$ across five seeds, then report $\overline { { \Delta } } \pm t _ { 0 . 9 7 5 , 4 } s _ { \Delta } / \sqrt { 5 }$ , with sample standard deviation $s \Delta \}$ single-procedure intervals use its five values. Bootstrap intervals condition on fitted quantities; seed intervals reflect shared-benchmark fitting variation. Neither validates the finite-sample radii above.

## D Experimental Protocols and Additional Results

## D.1 Experimental Quantities and Aggregation

Table D.1 identifies the operational targets; Appendix B.1 defines the matrix constructions. Real-data audits use nuisance ridge $1 0 ^ { - 3 }$ and stabilization $1 0 ^ { - 8 } ;$ ; synthetic audits use $\varepsilon = 1 0 ^ { - 1 2 }$ . Certificates use $\alpha = 0 . 9 5$ . Primary predefined-group coordinates use training moments; prospective normalization is specified below. Reported evaluation Loss-CVaR averages group-mean NLLs at or above their empirical 95th percentile, weighting groups equally; synthetic experiments use environment-mean NLLs. The Loss-CVaR training baseline averages the largest ⌈0.1n⌉ individual-example NLLs.

Table D.1: Diagnostic targets and replication units.
<table><tr><td>Experiment</td><td>Representation and reported quantity</td><td>Seeds</td></tr><tr><td>Prospective Waterbirds</td><td>Fine-tuned MobileNetV3-small; background-probe surviving frac- 22–26 tion</td><td></td></tr><tr><td>Synthetic</td><td>Scalar task  $z = s + \delta _ { e } r$  and nuisance s; exact population moments, 11–15 inverse-information tail and certificate</td><td></td></tr><tr><td>Hidden Waterbirds slices</td><td>Frozen MobileNetV3-small and fixed CLIP direction; shrunken pre- 17–21 dictive information</td><td></td></tr><tr><td>CivilComments groups</td><td>TF-IDF logistic classifier; feature-Schur certificate</td><td>13-17</td></tr><tr><td>Waterbirds groups</td><td>Frozen MobileNetV3-small and logistic fitting; feature-Schur certifi- 17–21 cate</td><td></td></tr></table>

Uncertainty. Prospective, synthetic, predefined-group, and frozen-feature training summaries report nominal 95% normal-approximation half-widths $1 . 9 6 s / { \sqrt { 5 } } ,$ with sample standard deviation s. Hidden-slice and group-free comparisons use five-seed Student-t intervals; contrasts are paired within seed. The end-to-end coefficient sweep reports standard deviations, and the conditional Gaussian/Wald audit is separate (Appendix B.4). Hidden-gate bootstrap intervals condition on the frozen gate and recompute evaluation weights (Appendix C). Acquisition retains means without recovered per-seed intervals.

Seeds describe fitting/partition or control variation on fixed benchmark splits, not independent datasets. For the predefined-group Waterbirds diagnostic in Appendix D.5, the classifier is deterministic after feature extraction and logistic fitting; only its random controls and reserve anchor vary across seeds. Prospective intervals use training seeds: 23,176 composites share 5,794 birds, with two donors per bird and arm. Accuracy changes are percentage points; information retains its original units.

## D.2 Prospective Nuisance Intervention

Training and frozen selection. ImageNet-initialized MobileNetV3-small [10, 24] is fully finetuned on Waterbirds [2] for seeds 22–26: 50 epochs, batch size 32, learning rate 10<sup>−4</sup>, and bf16 on an H100. A validation-fitted temperature and class-balanced logistic background probe use the pooled representation. Five probe folds are stratified by validation group (3,000 maximum iterations); features use validation standardization. Validation nuisance scores are out of fold and use validation moments; test scores use the full-validation probe and test moments. Task logits use training moments. Test normalization never selects the slice. The rule is

$$
\begin{array} { r l } & { \mathcal { G } _ { \mathrm { e l i g i b l e } } = \{ g : \mathrm { A c c } _ { v } ( g ) \geq \mathrm { A c c } _ { v } , \quad \mathrm { N L L } _ { v } ( g ) \leq \mathrm { m e d i a n } _ { h } \mathrm { N L L } _ { v } ( h ) \} , } \\ & { \qquad g ^ { \star } \in \underset { g \in \mathcal { G } _ { \mathrm { e l i g i b l e } } } { \arg \operatorname* { m i n } } \lambda _ { \operatorname* { m i n } } ( \widehat { F } _ { g } ^ { \mathrm { r i d } } ) / \widehat { J } _ { T T , g } . } \end{array}\tag{D.1}
$$

Here v denotes validation and eligibility uses uncalibrated NLL. Temperature scaling [25] minimizes full-validation binary NLL over 400 log-spaced temperatures in [0.05, 20], choosing the smaller temperature at ties. The checkpoint, temperature, probe, slice and audit constants are frozen before scoring; checkpoint mismatches abort evaluation. Table D.2 shows that LL (landbird/land) and WW (waterbird/water) qualify; WW is selected in all five seeds.

Foreground-preserving construction. Waterbirds combines CUB birds [26] with Places backgrounds [27] under soft CUB mattes. For each bird, we retain background pixels verbatim outside the

Table D.2: Validation selection. Means ± nominal 95% normal-approximation half-widths across seeds 22–26; accuracy is a fraction. Information uses uncentered feature moments.  
(a) Performance and eligibility
<table><tr><td>Group</td><td>n</td><td>Accuracy</td><td>NLL</td><td>Eligible</td></tr><tr><td>Landbird/land</td><td>467</td><td> $0 . 9 9 0 \pm 0 . 0 0 5$ </td><td> $0 . 0 3 8 \pm 0 . 0 1 6$ </td><td>Yes</td></tr><tr><td>Landbird/water</td><td>466</td><td> $0 . 7 0 3 \pm 0 . 0 1 9$ </td><td> $1 . 6 7 0 \pm 0 . 1 5 6$ </td><td>No</td></tr><tr><td>Waterbird/land</td><td>133</td><td> $0 . 3 9 0 \pm 0 . 0 2 1$ </td><td> $7 . 7 1 8 \pm 0 . 8 0 0$ </td><td>No</td></tr><tr><td>Waterbird/water</td><td>133</td><td> $0 . 9 2 0 \pm 0 . 0 1 3$ </td><td> $0 . 6 3 8 \pm 0 . 0 3 8$ </td><td>Yes</td></tr></table>

$$
0 . 7 3 7 \pm 0 . 0 4 6
$$

$$
0 . 7 9 4 \pm 0 . 0 3 2
$$

$$
0 . 4 3 6 \pm 0 . 0 4 5
$$

$$
0 . 5 9 0 \pm 0 . 0 2 7
$$

$$
0 . 2 8 9 \pm 0 . 0 1 4
$$

$$
0 . 9 7 3 \pm 0 . 0 3 3
$$

$$
0 . 3 6 5 \pm 0 . 0 2 6
$$

$$
2 . 0 6 3 \pm 0 . 0 5 4
$$

$$
0 . 7 6 4 \pm 0 . 0 3 1
$$

$$
0 . 5 1 1 \pm 0 . 0 2 1
$$

$$
0 . 7 8 6 \pm 0 . 0 3 6
$$

$$
0 . 2 4 8 \pm 0 . 0 1 5
$$

dilated foreground and fill the covered region using Telea inpainting [28], with a 5-pixel inpainting radius and $\mathrm { 4 5 \times 5 }$ elliptical dilation kernel. Its extracted bird layer is recomposed onto training-image donors of the same or opposite land/water background type. Donors have foreground fraction at most 0.08; a deterministic sequence tries four, taking the first with visible inpainted area at most 0.03, or the least-inpainted of four. The same-type arm undergoes identical recomposition.

Both arms use the same foreground layer, matte, label and source birds. Opaque foreground pixels are unchanged before JPEG encoding (quality 95, no chroma subsampling); soft-edge mixtures vary with background. Two draws per arm yield 23,176 composites. Visible inpainted area has mean 4.13%, p95 8.45%, maximum 20.37%, and arm means 4.13%/4.12%. Correctness is averaged across donors within bird, then across birds.

Outcomes and confident flips. Table D.3 reports all groups. The confident half uses the withinslice median calibrated true-label probability among originally correct birds, including ties. In WW, opposite-type/control flip rates are 56.4%/3.3% overall and 41.8%/0.7% in that half; the fixed threshold $p \geq 0 . 9$ selects no WW birds.

Table D.3: Held-out interventions. Accuracies and flip rates are seed means in percent; controlled drops are same-type minus opposite-type percentage points with nominal 95% normal-approximation half-widths. Panel (b) counts are mean confident-half counts.  
(a) Paired intervention arms
<table><tr><td>Group</td><td>Birds</td><td>Original</td><td></td><td></td><td>Same type Opposite type Controlled drop</td></tr><tr><td>Landbird/land</td><td>2,255</td><td>98.9</td><td>99.3</td><td>72.6</td><td> $2 6 . 7 \pm 1 . 8$ </td></tr><tr><td>Landbird/water</td><td>2,255</td><td>71.3</td><td>72.3</td><td>99.2</td><td> $- 2 6 . 9 \pm 2 . 0$ </td></tr><tr><td>Waterbird/land</td><td>642</td><td>42.3</td><td>40.6</td><td>92.3</td><td> $- 5 1 . 7 \pm 2 . 0$ </td></tr><tr><td>Waterbird/water</td><td>642</td><td>92.0</td><td>93.2</td><td>40.6</td><td> ${ \bf 5 2 . 6 \pm 1 . 4 }$ </td></tr></table>

(b) Opposite-typeflips among originally correct predictions
<table><tr><td>Group</td><td>All originally correct</td><td>half</td><td>Confident Confident-half count</td></tr><tr><td>Landbird/land</td><td>26.8</td><td>14.8</td><td>1,122</td></tr><tr><td>Landbird/water</td><td>0.2</td><td>0.1</td><td>805</td></tr><tr><td>Waterbird/land</td><td>1.2</td><td>0.3</td><td>136</td></tr><tr><td>Waterbird/water</td><td>56.4</td><td>41.8</td><td>299</td></tr></table>

Matched selection comparisons. Reapplying eligibility within each seed leaves LL and WW. Minimum validation accuracy and maximum uncalibrated NLL, evaluated separately on this identical set, agree with the surviving-fraction selector in all five seeds. All-group comparisons answer a different question; landbird/water determines the model-level certificate (Table D.4).

Centering and predictive Fisher. The post hoc check fixes the checkpoints, temperatures, validation coordinates $\mathbf { \bar { \theta } } ^ { \bullet } = ( z , b ) ^ { \top }$ , original scales, eligible sets and intervention outcomes. No model/probe is refitted. With $\langle \cdot \rangle _ { g }$ the divisor-n<sub>g</sub> slice mean, compare

$$
\begin{array} { r } { J _ { g } = \langle v v ^ { \top } \rangle _ { g } = \Sigma _ { g } + \mu _ { g } \mu _ { g } ^ { \top } , \quad \Sigma _ { g } = \langle ( v - \mu _ { g } ) ( v - \mu _ { g } ) ^ { \top } \rangle _ { g } , \quad \mu _ { g } = \langle v \rangle _ { g } . } \end{array}
$$

Centering does not rescale. The local model logit $p _ { \theta , \xi } ( x ) = f _ { w } ( x ) / \widehat { T } + \theta z ( x ) + \xi b ( x )$ at zero has score $( Y ^ { - } - p ) \boldsymbol { \imath }$ with conditional mean zero even when $\mu _ { g } \neq 0$ . Its predictive Fisher and the common fraction are

$$
I _ { g } = \langle p ( 1 - p ) v v ^ { \top } \rangle _ { g } , \qquad r _ { \rho } ( A ) = 1 - \frac { A _ { z b } ^ { 2 } } { A _ { z z } ( A _ { b b } + \rho ) } , \quad \rho = 1 0 ^ { - 3 } .
$$

Perturbations follow calibration and profile only the declared background coordinate. The normalizedweight comparator is $K _ { g } = I _ { g } / \bar { q } _ { g } , \bar { q } _ { g } = \langle p ( 1 - p ) \rangle _ { g } , \mathrm { s o } r _ { \rho } ( K _ { g } ) = r _ { \bar { q } _ { g } \rho } ( I _ { g } ) ;$ : normalization preserves the unregularized fraction but changes effective ridge. Table D.4 therefore reports it separately.

Table D.4: Selection and matrix sensitivity, evaluated within each seed. Panel (a) distinguishes original eligible and all-group sets; drops are mean percentage points. Panel (b) fixes the original eligible LL/WW set and interventions; fractions/drops are means ± nominal 95% normal-approximation half-widths.  
(a) Matched selectors
<table><tr><td>Statistic</td><td colspan="2">Candidates</td><td colspan="2">Selected group, seeds</td><td>Drop</td></tr><tr><td>Minimum raw task information</td><td colspan="2">Eligible</td><td colspan="2">Landbird/land, 5/5</td><td>26.7</td></tr><tr><td>Minimum absolute projected information</td><td colspan="2">Eligible</td><td colspan="2">Landbird/land, 5/5</td><td>26.7</td></tr><tr><td>Minimum surviving fraction</td><td colspan="2">Eligible</td><td colspan="2">Waterbird/water, 5/5</td><td>52.6</td></tr><tr><td>Minimum validation accuracy or maximum Eligible validation NLL</td><td colspan="2"></td><td colspan="2">Waterbird/water, 5/5</td><td>52.6</td></tr><tr><td>Minimum normalized-weight fraction</td><td colspan="2">Eligible</td><td colspan="2">Waterbird/water, 5/5</td><td>52.6</td></tr><tr><td>Maximum NLL or minimum accuracy</td><td colspan="2">All four</td><td colspan="2">Waterbird/land, 5/5</td><td>-51.7</td></tr><tr><td>Group determining  $I _ { 0 . 9 5 } ^ { \mathrm { F G } }$ </td><td colspan="2">All four</td><td colspan="2">Landbird/water, 5/5</td><td>-26.9</td></tr><tr><td colspan="6">(b) Frozen-coordinate constructions</td></tr><tr><td>Construction</td><td colspan="2">LL fraction</td><td colspan="2">WW fraction Selected, seeds</td><td>Drop</td></tr><tr><td>Uncentered  $J _ { g }$ </td><td colspan="2"> $0 . 5 9 0 \pm 0 . 0 2 7$ </td><td colspan="2"> $0 . 2 4 8 \pm 0 . 0 1 5$  WW,5/5</td><td> $5 2 . 6 \pm 1 . 4$ </td></tr><tr><td>Centered  $\Sigma _ { g }$ </td><td colspan="2"> $0 . 7 7 9 \pm 0 . 0 2 8$ </td><td colspan="2"> $0 . 8 5 5 \pm 0 . 0 3 2$  LL, 5/5</td><td> $2 6 . 7 \pm 1 . 8$ </td></tr><tr><td>Predictive Fisher  $I _ { g }$ </td><td colspan="2"> $0 . 9 8 7 \pm 0 . 0 1 4$   $0 . 2 7 3 \pm 0 . 0 1 6$ </td><td colspan="2">WW, 5/5</td><td> $5 2 . 6 \pm 1 . 4$ </td></tr><tr><td>Normalized weights  $K _ { g }$ </td><td colspan="2"> $0 . 9 8 7 \pm 0 . 0 1 4$ </td><td colspan="2"> $0 . 2 6 9 \pm 0 . 0 1 6$  WW, 5/5</td><td> $5 2 . 6 \pm 1 . 4$ </td></tr></table>

Centering selects LL in all seeds, whereas predictive Fisher retains WW. Slice means therefore contribute to the original ranking: for WW in seed 22, $\mu _ { z } ^ { 2 } = 2 . 0 9 4$ versus $\Sigma _ { z z } = 0 . 0 6 3 6$ . Table D.5 preserves the per-seed reversal; full means, covariance entries, weights, matrices and hashes are retained in the artifact.

Table D.5: Per-seed fractions on the original global scales. Each cell is LL / WW; the smaller value determines selection. Centering reverses the ranking in every seed.
<table><tr><td>Seed</td><td>Uncentered  $r _ { \rho } ( J _ { g } )$ </td><td>Centered  $r _ { \rho } ( \Sigma _ { g } )$ </td><td>Predictive  $r _ { \rho } ( I _ { g } )$ </td></tr><tr><td>22</td><td>0.5570 / 0.2291</td><td>0.7430 / 0.8542</td><td>0.9817 / 0.2587</td></tr><tr><td>23</td><td>0.6048 / 0.2494</td><td>0.8098 / 0.8481</td><td>0.9920 / 0.2729</td></tr><tr><td>24</td><td>0.6356 / 0.2755</td><td>0.8132 / 0.8359</td><td>0.9620 / 0.3032</td></tr><tr><td>25</td><td>0.5803 / 0.2392</td><td>0.7514 / 0.9156</td><td>0.9998 / 0.2571</td></tr><tr><td>26</td><td>0.5716 / 0.2460</td><td>0.7779 / 0.8189</td><td>0.9998 / 0.2732</td></tr></table>

Development chronology. The surviving-fraction rule followed an absolute-information pilot on seeds 17–21 and was frozen for newly trained seeds 22–26 on the same benchmark. Primarycohort entries use the documented rerun; the original-versus-rerun ledger is retained in the artifact. Appendix E.3 gives the Fisher-Glass training comparison under this same intervention.

Raw, projected, and tail summaries. We compare four summaries of retained validation geometry: mean raw task second moment A, arithmetic mean projected information $P ,$ reciprocal mean inverse information H, and inverse-CVaR certificate C. For the four equally weighted groups, let $F _ { g } = \widehat { F } _ { g } ^ { \mathrm { r i d } }$ and $L _ { g } = ( F _ { g } + \varepsilon ) ^ { - 1 }$ . Then

$$
\begin{array} { l l } { { \displaystyle { \cal A } = \frac { 1 } { 4 } \sum _ { g } \widehat { J } _ { T T , g } , } } & { { \displaystyle { \cal P } = \frac { 1 } { 4 } \sum _ { g } { \cal F } _ { g } , } } \\ { { } } & { { } } \\ { { \displaystyle { \cal H } = \left( \frac { 1 } { 4 } \sum _ { g } L _ { g } \right) ^ { - 1 } , } } & { { \displaystyle { \cal C } = [ { \mathrm { C V a R } } _ { 0 . 9 5 , \pi } ( L _ { g } ) ] ^ { - 1 } , \qquad \pi _ { g } = \frac { 1 } { 4 } . } } \end{array}
$$

Thus H is the harmonic mean of the stabilized projected information values $F _ { g } + \varepsilon .$ . The four summaries use uncentered feature moments; projection uses nuisance ridge $1 0 ^ { - 3 }$ , and H and C use reciprocal stabilization $\varepsilon = 1 0 ^ { - 8 }$ . This post hoc analysis joins validation summaries to held-out interventions in the primary (seeds 22–26) and development (17–21) cohorts. It compares summaries of existing models, not training objectives. “Selected $\mathrm { \bar { F } G ^ { \prime } }$ denotes the Fisher-Glass models selected within each cohort, including the accuracy-budget fallbacks listed in the caption; Appendix E.3 details the primary training comparison.

Table D.6: Four validation summaries and their intervention associations. Panels (a)–(b): means ± nominal 95% normal-approximation half-widths, $1 . 9 6 s / { \sqrt { 5 } } ;$ changes are paired within seed and drops are signed percentage points. Selected FG includes accuracy-budget fallbacks (primary seeds 22, 24; development seed 20). Panel (c): descriptive Spearman coefficients for ten models arranged in five ERM/FG seed pairs per cohort, with average ranks for tied outcomes.  
(a) Model summaries and paired changes
<table><tr><td>Cohort</td><td>Model/change</td><td>A: raw</td><td>P: projected</td><td></td><td>H: harmonic C: inverse-CVaR</td></tr><tr><td>Primary</td><td>ERM</td><td> $1 . 1 4 2 \pm 0 . 0 1 2$ </td><td> $0 . 5 0 0 \pm 0 . 0 1 0$ </td><td> $0 . 4 4 2 \pm 0 . 0 1 1$ </td><td> $0 . 2 8 9 \pm 0 . 0 1 4$ </td></tr><tr><td></td><td>Selected FG</td><td> $0 . 7 3 0 \pm 0 . 0 8 6$ </td><td> $0 . 4 8 4 \pm 0 . 0 4 7$ </td><td> $0 . 4 3 5 \pm 0 . 0 4 3$ </td><td> $0 . 3 3 0 \pm 0 . 0 4 3$ </td></tr><tr><td></td><td>FG – ERM</td><td> $- 0 . 4 1 2 \pm 0 . 0 7 9$ </td><td> $- 0 . 0 1 6 \pm 0 . 0 4 6$ </td><td> $- 0 . 0 0 7 \pm 0 . 0 4 1$ </td><td> $+ 0 . 0 4 1 \pm 0 . 0 4 6$ </td></tr><tr><td>Development ERM</td><td></td><td> $1 . 1 5 2 \pm 0 . 0 0 5$ </td><td> $0 . 4 8 9 \pm 0 . 0 1 1$ </td><td> $0 . 4 1 3 \pm 0 . 0 1 4$ </td><td> $0 . 2 5 5 \pm 0 . 0 1 8$ </td></tr><tr><td></td><td>Selected FG</td><td> $0 . 7 5 4 \pm 0 . 1 1 8$ </td><td> $0 . 5 1 9 \pm 0 . 0 6 5$ </td><td> $0 . 4 3 3 \pm 0 . 0 5 7$ </td><td> $0 . 3 0 3 \pm 0 . 0 8 2$ </td></tr><tr><td></td><td>FG - ERM</td><td> $- 0 . 3 9 8 \pm 0 . 1 1 8$ </td><td> $+ 0 . 0 3 0 \pm 0 . 0 6 3$ </td><td> $+ 0 . 0 2 0 \pm 0 . 0 6 6$ </td><td> $+ 0 . 0 4 8 \pm 0 . 0 9 5$ </td></tr></table>

(b) ERM group-selection rules and controlled drops
<table><tr><td>Rule</td><td></td><td>Candidates Selected, each cohort</td><td></td><td>Primary Development</td></tr><tr><td>Minimum raw second moment</td><td>All four</td><td>LL 4/5; LW 1/5</td><td> $+ 1 5 . 8 \pm 2 1 . 3$ </td><td> $+ 1 6 . 9 \pm 2 3 . 2$ </td></tr><tr><td>Minimum projected / largest inverse contribution / CVaR-active group</td><td>All four</td><td>LW 5/5</td><td> $- 2 6 . 9 \pm 2 . 0$ </td><td> $- 2 9 . 7 \pm 2 . 2$ </td></tr><tr><td>Largest inverse contribution</td><td>Eligible</td><td>LL 5/5</td><td> $+ 2 6 . 7 \pm 1 . 8$ </td><td> $+ 2 9 . 2 \pm 2 . 8$ </td></tr></table>

(c) Model-level rank associations with held-out outcomes
<table><tr><td></td><td colspan="2">Primary</td><td colspan="2">Development</td></tr><tr><td>Summary</td><td> $D _ { \mathrm { m a x } }$ </td><td> $W _ { \mathrm { o p p } }$ </td><td> $D _ { \mathrm { m a x } } \mathrm { ~ \ r ~ { ~ W _ { o p p } } ~ }$ </td><td></td></tr><tr><td>A: mean raw</td><td>-0.10 +0.26 +0.26 -0.13</td><td></td><td></td><td></td></tr><tr><td>P: mean projected</td><td>-0.09 +0.21 -0.03 -0.43</td><td></td><td></td><td></td></tr><tr><td>H: reciprocal mean inverse -0.15 +0.21 -0.12 -0.40</td><td></td><td></td><td></td><td></td></tr><tr><td>C: inverse-CVaR</td><td></td><td>+0.15 -0.24 -0.03 -0.48</td><td></td><td></td></tr></table>

For Table D.6(c), $D _ { \mathrm { m a x } } = 1 0 0 \mathrm { m a x } _ { g } ( \mathrm { A c c } _ { \mathrm { s a m e } , g } - \mathrm { A c c } _ { \mathrm { o p p o s i t e } , g } )$ and $W _ { \mathrm { o p p } } = \mathrm { m i n } _ { g } \mathrm { A c c } _ { \mathrm { o p p o s i t e } , g } ,$ both over all four groups. Integer correct counts recover exact outcome ties: primary ERM seeds 22 and 25 share drop 348/642, despite different floating-point subtractions. LL and LW denote landbird/land and landbird/water. Negative drops mean opposite-type recomposition improves accuracy. The group with the largest inverse-information contribution accounts for 38.3% and 40.6% of summed inverse information on average in the primary and development cohorts, respectively. The original headline rules differed: surviving fraction in the primary cohort, absolute projected information in development; this comparison consistently uses all four groups for model summaries and correlations.

Projection and aggregation give different model summaries, with rank associations varying across cohorts. For four equally weighted groups and $\begin{array} { r } { \alpha = 0 . 9 5 , C = \operatorname* { m i n } _ { g } F _ { g } + \varepsilon ; } \end{array}$ its active group also minimizes projected information and maximizes inverse contribution. Here that group is LW, whose opposite-type recomposition improves accuracy. The prospective WW diagnosis instead uses the surviving fraction within the eligible set. This comparison does not establish superior swap-damage prediction by inverse-CVaR; paired changes in P, H, C all have intervals containing zero. The artifact provides per-model and per-seed values, exact-count outcomes, input hashes, and replay code.

Auxiliary flip prediction. Each originally correct source bird contributes two opposite-type donor rows. All predictor covariates are evaluated on the original, pre-intervention image. Baseline predictors are uncalibrated ${ \mathrm { N L L } } ,$ calibrated true-label confidence, absolute raw logit and standardized task logit z. Augmentation adds $\beta _ { g } b$ and $| z - \beta _ { g } b |$ , with $\beta _ { g } = \widehat { J } _ { T N , g } / ( \widehat { J } _ { N N , g } + 1 0 ^ { - 3 } )$ frozen from the validation block. Both models use identical five-fold, nonshuffled CUB-species GroupKFold splits, keeping species and donor pairs together. Logistic regression uses an intercept, $\ell _ { 2 }$ penalty, $\bar { C } = 1 , \mathrm { L } \mathbf { - } \bar { \mathbf { B } } \mathrm { F } \bar { \mathbf { G } } \mathrm { S }$ , tolerance $1 0 ^ { - 4 }$ , maximum 5,000 iterations, seed 0 and no class weighting or tuning. All columns, including group indicators, are centered and scaled using training-fold means and population standard deviations; standard deviations below $1 0 ^ { - 9 }$ are replaced by one.

Pool out-of-fold predictions within seed before computing average precision (reported as AUPRC), area under the receiver operating characteristic curve (AUROC), and mean log likelihood. For seed $s , N _ { s }$ counts all retained donor rows in the analyzed population, including flips and nonflips. Let $y _ { s i } ^ { \mathrm { { f i p } } } = 1$ indicate an incorrect prediction on the donor composite and $y _ { s i } ^ { \mathrm { { f i p } } } = 0$ a correct prediction. The probabilities $q _ { s i } ^ { A }$ and $q _ { s i } ^ { B }$ are the augmented and baseline out-of-fold flip predictions. For clipped probabilities $q \in [ \tilde { 1 0 } ^ { - 1 2 } , \tilde { 1 } ^ { - } 1 0 ^ { - 1 2 } ]$

$$
G _ { s } = \frac { 1 } { N _ { s } } \sum _ { i = 1 } ^ { N _ { s } } \Bigl [ \mathrm { L L } \bigl ( y _ { s i } ^ { \mathrm { f i p } } , q _ { s i } ^ { A } \bigr ) - \mathrm { L L } \bigl ( y _ { s i } ^ { \mathrm { f i p } } , q _ { s i } ^ { B } \bigr ) \Bigr ] , \quad \mathrm { L L } ( y , q ) = y \log q + ( 1 - y ) \log ( 1 - q ) .\tag{D.2}
$$

$G _ { s }$ is augmented-minus-baseline gain in nats per donor row. Table D.7 averages paired changes equally across seeds 22–26. The coordinates improve prediction without group indicators and within WW; with indicators, AUPRC decreases. This tests their incremental value, not unique attribution to the scalar information statistic. Original inputs, exact-source refits and paired records are in the artifact.

Table D.7: Incremental flip prediction. Counts and rates are seed means; parentheses count seeds with improvement. Changes are paired within seed.
<table><tr><td></td><td colspan="5">Flip Log-likelihood</td></tr><tr><td>Analysis</td><td>Rows</td><td>rate</td><td>gain</td><td>∆AUPRC</td><td>∆AUROC</td></tr><tr><td>All groups, no indicators</td><td>9,398</td><td>0.199</td><td>+0.016</td><td>+0.035 (5/5)</td><td>+0.096 (5/5)</td></tr><tr><td>All groups, with indicators</td><td>9,398</td><td>0.199</td><td>+0.001</td><td>-0.007 (0/5) +0.003 (5/5)</td><td></td></tr><tr><td>Selected slice</td><td>1,181</td><td>0.564</td><td>+0.013</td><td>+0.026 (5/5) +0.015 (5/5)</td><td></td></tr></table>

## D.3 Synthetic Classification and Tail Checks

Draw $U _ { e } \sim$ Uniform(0, 1), set $\delta _ { e } = 0 . 3 5 U _ { e } ^ { 1 / \gamma }$ , and draw independent standard normals $s , r$ for each observation. The task is $\begin{array} { r } { z = s + \delta _ { e } r , Y = \mathbf { 1 } \{ z > 0 \} } \end{array}$ , scored by nuisance-only logit 4s. Thus

$$
\operatorname* { P r } ( \delta _ { e } < t ) = ( t / 0 . 3 5 ) ^ { \gamma } ( 0 \leq t \leq 0 . 3 5 ) , \qquad J _ { T T , e } = 1 + \delta _ { e } ^ { 2 } , \quad F _ { e } ^ { 0 } = \delta _ { e } ^ { 2 } .\tag{D.3}
$$

Population feature moments are exact; loss/accuracy use 256 observations per environment. The $\gamma = 1$ experiment uses 300,000 environments and seeds 11–15. Fit the log-survival slope over the largest 5% of inverse information; empirical CVaR averages values at or above the empirical α-quantile. Figure A.1 regenerates seed 11 with $L _ { e } = ( \delta _ { e } ^ { 2 } + 1 0 ^ { - 1 2 } ) ^ { - 1 }$ , rather than recovering missing original raw output; its survival curve and the five-seed summaries have different replication levels.

Table D.8 checks Theorem 2 and the scalar portfolio law in Appendix A.2. For K independent draws, information $K ^ { - 1 } \textstyle \sum _ { j } \delta _ { e , j } ^ { 2 }$ has exponent $K \dot { / } 2 \mathrm { a t } \gamma = 1$ ; weights affect its constant. The mass sweep

Table D.8: Synthetic checks. Panel (a) and the branch rows are five-seed means $\pm$ nominal 95% normal-approximation half-widths; zero widths reflect rounding. Mass-sweep rows are reported single-seed values.  
(a) Prediction and information
<table><tr><td colspan="2">Quantity Estimate</td></tr><tr><td>Mean accuracy Fifth-percentile environment accuracy</td><td> $9 4 . 5 4 \pm 0 . 0 0 \%$   $8 8 . 6 7 \pm 0 . 0 0 \%$ </td></tr><tr><td> $\mathrm { C V a R _ { 0 . 9 5 } }$  of environment-mean NLL  $I _ { 0 . 9 5 } ^ { \mathrm { F G } }$ </td><td> $0 . 2 6 2 \pm 0 . 0 0 0$   $( 1 . 2 2 \pm 0 . 5 3 ) \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Inverse-information exponent  $\widehat { \beta }$ </td><td> $0 . 4 9 8 \pm 0 . 0 0 3$ </td></tr><tr><td colspan="2">(b) Inverse-information tail exponents</td></tr><tr><td>Construction Setting Predicted</td><td> $\beta$  Estimated  $\beta$ </td></tr><tr><td>Mass near the null  $\gamma = 1$ </td><td>0.5 0.509</td></tr><tr><td>Mass near the null  $\gamma = 2$ </td><td>1 0.985</td></tr><tr><td>Mass near the null  $\gamma = 4$ </td><td>2  $2 . 0 0 1$ </td></tr><tr><td>Independent branches  $\dot { K } = 1$ </td><td> $0 . 5$   $0 . 4 9 9 \pm 0 . 0 0 3$ </td></tr><tr><td>Independent branches  $K = 2$ </td><td>1  $1 . 0 0 0 \pm 0 . 0 0 9$ </td></tr><tr><td>Independent branches  $K = 3$ </td><td>1.5  $1 . 5 0 8 \pm 0 . 0 1 5$ </td></tr></table>

predicts $\gamma / 2 .$ An anchor with $\delta _ { 0 } = 0 . 5$ and reserve mass 0.05 yields information $0 . 9 5 \delta _ { e } ^ { 2 } + 0 . 0 5 ( 0 . 2 5 )$ floor 0.0125, and measured certificate $0 . 0 1 2 6 0 \pm 0 . 0 0 0 0 0 1$ (five seeds; Appendix E.1).

## D.4 Hidden-Slice Discovery Results

Appendix C gives the blind protocol: task labels, frozen MobileNet features and a fixed CLIP direction are available; background/group metadata remain sealed through held-out evaluation. Results use the primary hidden-slice cohort, seeds 17–21, with full-validation gates frozen before test. Support correction may depend on the test sample but leaves all five primary CLIP gates’ test weights unchanged. Information uses same-class pooling; accuracy, uncalibrated NLL and composition use admissible within-class weights without pooling. The artifact identifies this cohort as v2 and records original hashes; the older uncorrected cohort is excluded.

All selectors share classifiers with average accuracy $7 6 . 3 \pm 0 . 7 \%$ and post hoc worst-group accuracy $5 7 . 3 { \pm } 1 . 3 \%$ . CLIP search finds 98.8% membership-weighted slice accuracy. Table D.9 includes every matched-mass comparator and paired contrast. The random-direction interval and the uncentered and centered feature-moment intervals against high-loss selection contain zero. Feature control reevaluate gates selected using predictive Fisher without rediscovery.

Table D.9: Hidden-slice comparisons. Panel (a): $\mathrm { m e a n s } \pm 9 5 \%$ Student-t half-widths across five paired seeds; predictive information retains calibration and shrinkage $\eta = 0 . 1 0$ . Panel (b): CLIPsearch minus comparator, with paired 95% Student-t intervals; negative differences indicate lower information.

(a) Held-out selectors
<table><tr><td>Selector</td><td>Predictive projected information</td><td>Uncalibrated slice NLL</td></tr><tr><td>Fisher search, fixed CLIP direction</td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td> $0 . 0 3 1 \pm 0 . 0 1 7$ </td></tr><tr><td>High loss, same target within-class mass</td><td> $0 . 0 1 9 \pm 0 . 0 0 6$ </td><td> $5 . 7 7 9 \pm 0 . 4 9 6$ </td></tr><tr><td>Random, same target within-class mass</td><td> $0 . 0 1 3 \pm 0 . 0 0 1$ </td><td> $1 . 0 2 5 \pm 0 . 0 5 9$ </td></tr><tr><td>Constant class-conditional gate</td><td> $0 . 0 1 3 \pm 0 . 0 0 1$ </td><td> $1 . 0 2 7 \pm 0 . 0 4 9$ </td></tr><tr><td>Fisher search, random direction</td><td> $0 . 0 0 6 \pm 0 . 0 0 1$ </td><td> $0 . 0 3 1 \pm 0 . 0 1 5$ </td></tr></table>

(b) Paired information contrasts
<table><tr><td>Information</td><td>Comparison</td><td>Difference</td><td>95% interval</td></tr><tr><td>Predictive</td><td>High-loss slice</td><td>-0.014</td><td>[-0.020, -0.008]</td></tr><tr><td>Predictive</td><td>Random slice</td><td>-0.008</td><td>[-0.009, -0.007]</td></tr><tr><td>Predictive</td><td>Constant gate</td><td>-0.008</td><td>[-0.009, -0.007]</td></tr><tr><td>Predictive</td><td>Random direction</td><td>-0.0004</td><td>[-0.0015,0.0007]</td></tr><tr><td>Feature moments</td><td>High-loss slice</td><td>-0.034</td><td>[-0.106, 0.039]</td></tr><tr><td>Centered covariance High-loss slice</td><td></td><td>-0.071</td><td>[−0.166, 0.024]</td></tr></table>

Class selection and overlap. CLIP, random-direction, random-slice and constant-gate selectors choose landbirds in all five seeds; high loss chooses landbirds in seeds 17,18,21 and waterbirds in 19,20. The two different-class Jaccards are zero; same-class values are 0.00222, 0.00111, and 0. Their mean 0.000666, reported as 0.001, combines between-class and within-class differences under Appendix C.3’s top-count rule.

Support and sensitivity. On same-class rows from the 15% training-calibration subset, crossfold gate-score correlation is 0.419 [0.353, 0.485] and overlap 0.258 [0.225, 0.291]. Test ESS is 414 [383, 445] using admissible weights. Raw membership mass is 0.083 [0.077, 0.089] within the selected class (target 0.10), not the full test set; landbirds contain 4,510 test examples. Validation crossfitting, separate from final testing, gives information 0.004–0.005 over masses {0.05, 0.10, 0.20} versus high-loss 0.012–0.027, and 0.004–0.006 over shrinkages {0.05, 0.10, 0.20}.

Composition after held-out evaluation is 97.0% landbird/land for CLIP, 94.4% for random direction, and 58.4% landbird/water for high loss. The hidden slice has lower projected predictive information than the weakest predefined group in all five seeds: mean paired difference −0.0029, with 95% Student-t interval [−0.0034, −0.0024], taking the predefined-group minimum separately within each seed. Both sides use the same model, CLIP coordinate, calibrated predictive moments, pooling $\eta = 0 . 1 0$ , and ridge $1 0 ^ { - 3 }$ . A metadata-trained oracle probe used only after blind freeze gives information $0 . 0 0 3 \pm 0 . 0 0 0$ , NLL $0 . 0 0 9 \pm 0 . 0 0 4$ , and 99.1% landbird/land composition (zero width is rounding). Repair outcomes are in Appendix E.3.

## D.5 Predefined-Group Diagnostics

Fitting. CivilComments-WILDS [29, 30] predicts comment toxicity, with $y = 0$ denoting nontoxic comments. Identity-any indicates that at least one of the 24 original identity annotation scores is at least 0.5. Training-fitted unigram/bigram TF-IDF (50,000 features, minimum document frequency 5, sublinear TF) feeds class-balanced SGD logistic task/identity-any probes $( \ell _ { 2 } = 1 0 ^ { - 5 }$ , 50 passes, tolerance $1 0 ^ { - 3 } )$ . Identity-by-label groups require 100 examples; tail regression uses the largest 50% of group inverse information, requiring eight groups and four tail points. Seed 13 gives the fixed diagnostic; seeds 13–17 give aggregates.

Waterbirds uses frozen ImageNet MobileNetV3-small features (extraction batch 64) and balanced L BFGS logistic task/background models (1,000 maximum iterations). Four groups are reported directly, without fitting a tail exponent; only controls/anchors vary over seeds 17–21. Primary task/probe and metadata coordinates use training moments. These original runs used an i9-11900F/32 GB workstation, about one minute for CivilComments and three minutes for first Waterbirds extraction/fitting, then cached features.

Table D.10: Original predefined-group diagnostics, distinct from fresh audits in Appendix B.4. Means ± nominal 95% normal-approximation half-widths; Waterbirds zeros reflect fixed-model fitting.
<table><tr><td>Dataset</td><td>Average accuracy (%)</td><td>Worst-group accuracy (%)</td><td> ${ \mathrm { L o s s } } -$   $\mathrm { C V a R _ { 0 . 9 5 } }$ </td><td> $I _ { 0 . 9 5 } ^ { \mathrm { F G } }$ </td></tr><tr><td></td><td> $8 6 . 1 0 \pm 0 . 2 2$ </td><td> $4 3 . 7 5 \pm 0 . 6 4$ </td><td> $0 . 9 9 8 \pm 0 . 0 1 2$ </td><td> $0 . 5 0 0 \pm 0 . 0 0 1$ </td></tr><tr><td>CivilComments Waterbirds</td><td> $7 7 . 1 1 \pm 0 . 0 0$ </td><td> $5 6 . 5 4 \pm 0 . 0 0$ </td><td> $1 . 7 4 1 \pm 0 . 0 0 0$ </td><td> $0 . 1 6 5 \pm 0 . 0 0 0$ </td></tr></table>

For fixed CivilComments, the weakest-information group is muslim/y = 0 (accuracy 60.1%); lowest accuracy is $\mathtt { b l a c k } / y = 0 ( 4 4 . 0 \% )$ . Waterbirds’ weakest-information group is landbird/land (97.3% accuracy), whereas waterbird/land has lowest accuracy (56.5%); inverse-information $\mathsf { p } 9 5$ is 5.90.

Specification and stress controls. The prespecified semantic family is $\{ b _ { \mathrm { m e t a } } , b _ { \mathrm { p r o b e } } \}$ : standardized identity-any/background annotation and its training-fitted probe, without cross-fitting. Semantic relevance follows those annotations; no held-out relevance threshold, anchor task-preservation tolerance, fold-stability admission screen, or nuisance-conditioning threshold was applied. The ridge is $1 0 ^ { - 3 } \colon$ ; family minima are 0.499 and 0.165. Reserve anchors (identity-present label-balanced CivilComments; four-group-balanced Waterbirds) are construction inputs, not admission samples. Fresh exact/ridge audits and hidden-gate stability are separate analyses.

Table D.11: Nuisance sensitivity of fixed-model diagnostics. Entries are $I _ { 0 . 9 5 } ^ { \mathrm { F G } }$ ; worst-group accuracy remains 44.0%/56.5% for CivilComments/Waterbirds.
<table><tr><td>Specification</td><td>CivilComments</td><td>Waterbirds</td><td>Role</td></tr><tr><td>Metadata</td><td>0.587</td><td>0.266</td><td>Semantic-family member</td></tr><tr><td>Learned probe</td><td>0.499</td><td>0.165</td><td>Semantic-family minimum</td></tr><tr><td>Random</td><td>0.588</td><td>0.482</td><td>Negative control</td></tr><tr><td>Under-projected</td><td>0.588</td><td>0.482</td><td>Under-projection control</td></tr><tr><td>Over-projected</td><td>0.001</td><td>0.012</td><td>Task-erasure control</td></tr><tr><td>Hidden-loss</td><td>0.060</td><td>0.177</td><td>Loss-coordinate stress control</td></tr></table>

Random nuisance is independent standard normal; under-projection sets it to zero. Over-projection uses standardized task score plus Gaussian noise of standard deviation 0.01 and deliberately erases task information. “Hidden-loss” is observed test loss with original groups fixed. The random, overprojected, and observed-loss coordinates use evaluated-array normalization and are not independently learned nuisance scores. The metadata/random contrast is 0.587/0.588 on CivilComments, versus probe/random 0.165/0.482 on Waterbirds, illustrating specification dependence.

The versioned artifact (see the artifact availability statement) retains execution commands, fullprecision matrices and overlaps, run-history ledgers and expanded source tables, with original cohort provenance. Those records supplement the scientific definitions and controls above.

## E Information-Changing Interventions

This appendix gives the reserve proof, training and acquisition comparisons, and the influence derivative behind Section 6. Distinct cohorts and information targets remain separate. Expanded protocols and numerical breakdowns are retained in the versioned artifact’s docs/extended\_ append $\mathrm { i } \mathrm { x } \_ \mathrm { d e t a i l } \mathrm { s } /$ , with executable reproduction guides and provenance (see the artifact availability statement).

## E.1 Information Reserves under Observed-Source Allocation

Fix the predictor, coordinates, and normalization. Let $J _ { e } , J _ { 0 } \succeq 0$ have positive-definite nuisance blocks. The anchor is an additional observation source with $F _ { 0 } ^ { 0 } : = S _ { 0 } \bar { ( } J _ { 0 } ) \succeq \rho _ { 0 } I , \rho _ { 0 } > 0$ . For the allocation in Section $^ { 6 , }$ each observation’s source is observed and allocation probabilities are parameter-independent. Sampling the anchor with probability $r \in ( 0 , 1 ]$ therefore gives

$$
\begin{array} { r l r } & { } & { J _ { e } ^ { \mathrm { r e s } } = ( 1 - r ) J _ { e } + r J _ { 0 } , } \\ & { } & { S _ { 0 } ( J _ { e } ^ { \mathrm { r e s } } ) \succeq ( 1 - r ) S _ { 0 } ( J _ { e } ) + r S _ { 0 } ( J _ { 0 } ) \succeq r \rho _ { 0 } I . } \end{array}\tag{E.1}
$$

The same statement applies to affine allocation of a fixed operational matrix map; it is not an assertion about an unobserved mixture likelihood.

Proof. For $\boldsymbol { J } = \big ( \mathbf { \Pi } _ { B ^ { \intercal } } ^ { A } \mathbf { \Pi } _ { C } ^ { B } \big ) , C \succ 0$ , completing the square gives

$$
\begin{array} { r } { u ^ { \top } S _ { 0 } ( J ) u = \underset { v } { \operatorname* { m i n } } q _ { J } ( u , v ) , \qquad q _ { J } ( u , v ) = \left( \begin{array} { l } { u } \\ { - v } \end{array} \right) ^ { \top } J \left( \begin{array} { l } { u } \\ { - v } \end{array} \right) , \quad v _ { * } = C ^ { - 1 } B ^ { \top } u . } \end{array}\tag{E.2}
$$

Linearity in J implies

$$
\begin{array} { r l } & { \underset { v } { \operatorname* { m i n } } \{ ( 1 - r ) q _ { J _ { e } } ( u , v ) + r q _ { J _ { 0 } } ( u , v ) \} \geq ( 1 - r ) \underset { v } { \operatorname* { m i n } } q _ { J _ { e } } ( u , v ) + r \underset { v } { \operatorname* { m i n } } q _ { J _ { 0 } } ( u , v ) } \\ & { \qquad \geq r \rho _ { 0 } \| u \| ^ { 2 } . } \end{array}\tag{E.3}
$$

The last inequality uses $S _ { 0 } ( J _ { e } ) \succeq 0$ and the anchor bound; holding for every $u ,$ it proves (E.1).

For any environment distribution, $0 < \alpha < 1$ and $\varepsilon \geq 0$ , CVaR monotonicity gives

$$
L _ { e } ^ { \mathrm { r e s } , ( \varepsilon ) } \leq ( r \rho _ { 0 } + \varepsilon ) ^ { - 1 } , \qquad I _ { \alpha } ^ { \mathrm { F G , r e s } } ( \varepsilon ) \geq r \rho _ { 0 } + \varepsilon \geq r \rho _ { 0 } .\tag{E.4}
$$

Here $r = m / ( n + m )$ for m anchor and n base observations. An estimated anchor’s positive lower bound from Appendix B.3 yields the same guarantee on its confidence event. The claim concerns the fixed predictor; retraining requires a new audit. Appendix D.3 evaluates $r = . 0 5 , \rho _ { 0 } = . 2 5$ , and the resulting floor .0125.

## E.2 Regularized Training

For a declared group matrix $\widehat { J } _ { g } ( w )$ , put $\widehat { L } _ { g } ( w ) = [ \lambda _ { \mathrm { m i n } } ( S _ { \mathrm { r i d } } ( \widehat { J } _ { g } ( w ) ) ) + \varepsilon ] ^ { - 1 }$ . With fixed $\pi _ { g } \geq 0 _ { : }$ $\textstyle \sum _ { g } \pi _ { g } = 1$ , training solves

$$
\operatorname* { m i n } _ { w , \tau } \widehat { R } _ { \mathrm { b a s e } } ( w ) + \lambda _ { \mathrm { F G } } \left[ \tau + \frac { 1 } { 1 - \alpha } \sum _ { g } \pi _ { g } ( \widehat { L } _ { g } ( w ) - \tau ) _ { + } \right] ,\tag{E.5}
$$

Here w denotes predictor parameters, $\widehat { R } _ { \mathrm { b a s e } }$ is the ERM or Group DRO task-risk objective, and $\lambda _ { \mathrm { F G } }$ its information-penalty coefficient. Frozen-feature Waterbirds uses uncentered second moments of standardized task logits and fixed background-probe scores, ridge $1 0 ^ { - 3 }$ and $\varepsilon = 1 0 ^ { - 8 }$ . Training $\alpha = . 7 5$ and audit $\alpha = . 9 5$ both give the maximum inverse information over four equally weighted groups.

Protocol and selection. ImageNet MobileNetV3-small features and background-probe scores use training-moment standardization. A linear head receives 1,200 full-batch AdamW updates $( \mathrm { l r } = . 0 1$ weight decay $1 0 ^ { - 4 }$ , seeds 17–21), Gaussian $N ( 0 , . 0 1 ^ { 2 } )$ initial weights and zero intercept. Current logits use full-training population standard deviation plus $1 0 ^ { - 6 } ;$ ; training projected information is floored at ε before reciprocal stabilization. Evaluation uses fitted training-logit moments. Group DRO minimizes the maximum group mean cross-entropy; Loss-CVaR uses the largest ⌈.1n⌉ training losses, and JTT refits with weight 20 on initial ERM mistakes.

For each FG family and seed, search $\Lambda _ { \mathrm { F G } } = \{ . 0 1 , . 0 3 , . 1 , . 3 \}$ . Among candidates whose validation worst-group accuracy is within .02 of the family’s best, maximize the validation certificate, breaking ties by higher validation worst-group accuracy. This uses validation group annotations. Historical choices are .03 for $\mathrm { E R M } + \mathrm { F G }$ in all seeds and .1 or .3 for Group DRO + FG. Table E.1 retains all six methods; official test data enter only final evaluation.

Table E.1: Original frozen-feature training cohort, seeds 17–21. Both panels give means ± nominal 95% normal-approximation half-widths; accuracy is percent and the uncentered feature–Schur ridge certificate uses $\alpha = . 9 5 .$
<table><tr><td>Method</td><td> $\mathbf { A v g . a c c . } \left( \% \right)$ </td><td> $\mathrm { W o r s t a c c . } \left( \% \right)$ </td><td> $\mathrm { L o s s \mathrm { - } C V a R \mathrm { _ { . 9 5 } } }$ </td></tr><tr><td>ERM</td><td> $7 7 . 2 \pm 0 . 1$ </td><td> $5 5 . 7 \pm 0 . 3$ </td><td> $1 . 9 6 9 \pm 0 . 0 0 7$ </td></tr><tr><td>Group DRO</td><td> $8 0 . 9 \pm 0 . 3$ </td><td> $6 7 . 3 \pm 0 . 3$ </td><td> $1 . 1 4 3 \pm 0 . 0 2 1$ </td></tr><tr><td>JTT</td><td> $7 7 . 4 \pm 0 . 1$ </td><td> $5 4 . 1 \pm 0 . 2$ </td><td> $2 . 1 6 7 \pm 0 . 0 0 4$ </td></tr><tr><td>Loss-CVaR</td><td> $7 8 . 2 \pm 0 . 1$ </td><td> $5 2 . 1 \pm 0 . 3$ </td><td> $2 . 4 2 7 \pm 0 . 0 4 1$ </td></tr><tr><td>ERM + Fisher-Glass</td><td> $8 0 . 4 \pm 0 . 2$ </td><td> $5 6 . 3 \pm 0 . 3$ </td><td> $1 . 8 6 6 \pm 0 . 0 2 5$ </td></tr><tr><td>Group DRO + FG</td><td> $8 0 . 6 \pm 0 . 9$ </td><td> $6 9 . 4 \pm 1 . 3$ </td><td> $0 . 8 5 5 \pm 0 . 0 8 2$ </td></tr><tr><td>Method</td><td>Ridge certificate</td><td> $\scriptstyle { \mathrm { I n v e r s e ~ p 9 5 } }$ </td><td></td></tr><tr><td>ERM</td><td> $0 . 1 9 8 \pm 0 . 0 0 1$ </td><td> $4 . 8 6 \pm 0 . 0 2$ </td><td></td></tr><tr><td>Group DRO</td><td> $0 . 2 3 3 \pm 0 . 0 0 3$ </td><td> $4 . 1 0 \pm 0 . 0 6$ </td><td></td></tr><tr><td>JTT</td><td> $0 . 2 3 4 \pm 0 . 0 0 1$ </td><td> $4 . 1 3 \pm 0 . 0 1$ </td><td></td></tr><tr><td>Loss-CVaR</td><td> $0 . 2 3 3 \pm 0 . 0 0 2$ </td><td> $4 . 1 7 \pm 0 . 0 3$ </td><td></td></tr><tr><td> $\mathrm { E R M } + \mathrm { F i s h e r - G l a s s }$ </td><td> $0 . 3 9 9 \pm 0 . 0 0 6$ </td><td> $2 . 4 6 \pm 0 . 0 4$ </td><td></td></tr><tr><td> $\mathrm { G r o u p ~ D R O + F G }$ </td><td> $0 . 6 1 4 \pm 0 . 0 5 9$ </td><td> $1 . 6 3 \pm 0 . 1 7$ </td><td></td></tr></table>

FG doubles ERM’s certificate; adding it to Group DRO raises the certificate from .233 to .614 and mean worst-group accuracy from 67.3% to 69.4%. These historical summaries differ from Appendix B’s fresh ridge/exact audit cohort, which does not retroactively certify the historical checkpoints. Their original per-seed outputs were not recovered; retained reports, complete settings and independent rerun commands are in the artifact (see the artifact availability statement).

## E.3 Extended Training and Selection Results

End-to-end Group DRO + Fisher-Glass coefficient sweep. MobileNetV3-small is fine-tuned for 10 epochs (batch 128, evaluation batch 512, seeds 17–21) with AdamW learning rate and weight decay $1 0 ^ { - 4 }$ ; the final epoch is evaluated. The reported H100 configuration uses bf16 and TF32. The nuisance is training-standardized background annotation; task logits use minibatch population standard deviation plus $1 0 ^ { - 6 }$ . Groups with fewer than two observations are omitted. Ridge is $1 0 ^ { - 3 }$ $\alpha = . 9 5$ , and projected information is floored at $1 0 ^ { - 8 }$ before another $1 0 ^ { - 8 }$ enters the reciprocal. Table E.2 retains the complete coefficient sweep: stronger penalties can increase information while degrading prediction.

Table E.2: End-to-end Waterbirds Group DRO + Fisher-Glass coefficient sweep. FG entries are means ± sample standard deviations over five seeds, not confidence intervals. The Group DRO reference is the baseline from the $\lambda _ { \mathrm { F G } } = 1 0 ^ { - 4 }$ sweep; each sweep reran its baseline, so this is not a shared paired reference for every coefficient.
<table><tr><td>Objective / λFG</td><td>Avg. acc. (%)</td><td>) Worst acc. (%)</td><td> $\mathrm { L o s s \mathrm { - } C V a R \mathrm { _ { . 9 5 } } }$ </td></tr><tr><td>Group DRO reference</td><td>85.4</td><td>53.2</td><td>0.925</td></tr><tr><td> $5 \times \mathrm { { 1 0 ^ { - 5 } } }$ </td><td> $7 9 . 7 \pm 6 . 2$ </td><td> $5 6 . 6 \pm 5 . 8$ </td><td> $0 . 7 3 3 \pm 0 . 0 6 1$ </td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td> $7 9 . 5 \pm 1 0 . 2$ </td><td> $5 7 . 7 \pm 8 . 3$ </td><td> $0 . 7 3 3 \pm 0 . 1 1 0$ </td></tr><tr><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $6 9 . 9 \pm 9 . 7$ </td><td> $5 8 . 8 \pm 8 . 7$ </td><td> $0 . 6 5 3 \pm 0 . 0 6 4$ </td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td> $6 2 . 7 \pm 7 . 9$ </td><td> $4 9 . 9 \pm 1 . 8$ </td><td> $0 . 7 1 0 \pm 0 . 0 1 2$ </td></tr><tr><td> $3 \times 1 0 ^ { - 3 }$ </td><td> $5 7 . 7 \pm 4 . 5$ </td><td> $4 8 . 6 \pm 6 . 6$ </td><td> $0 . 7 1 7 \pm 0 . 0 3 6$ </td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td> $4 7 . 9 \pm 1 6 . 5$ </td><td> $2 2 . 7 \pm 1 6 . 7$ </td><td> $1 . 0 3 1 \pm 0 . 4 1 6$ </td></tr><tr><td>Objective / λFG</td><td>Ridge certificate</td><td> $\scriptstyle { \mathrm { I n v e r s e ~ p 9 5 } }$ </td><td></td></tr><tr><td>Group DRO reference</td><td>0.259</td><td>3.68</td><td></td></tr><tr><td> $5 \times \mathrm { i 0 ^ { - 5 } }$ </td><td> $0 . 3 9 3 \pm 0 . 0 5 9$ </td><td> $2 . 4 9 \pm 0 . 3 4$ </td><td></td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td> $0 . 5 0 2 \pm 0 . 1 6 6$ </td><td> $2 . 1 0 \pm 0 . 5 8$ </td><td></td></tr><tr><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $0 . 6 1 4 \pm 0 . 1 8 9$ </td><td> $1 . 7 1 \pm 0 . 4 6$ </td><td></td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td> $0 . 7 8 6 \pm 0 . 1 2 4$ </td><td> $1 . 2 9 \pm 0 . 2 0$ </td><td></td></tr><tr><td> $3 \times 1 0 ^ { - 3 }$ </td><td> $0 . 8 1 7 \pm 0 . 0 6 8$ </td><td> $1 . 2 2 \pm 0 . 1 0$ </td><td></td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td> $0 . 8 0 9 \pm 0 . 0 7 6$ </td><td> $1 . 2 3 \pm 0 . 1 2$ </td><td></td></tr></table>

At $\lambda _ { \mathrm { F G } } = 1 0 ^ { - 4 }$ the certificate rises from .259 to .502 and inverse p95 falls from 3.68 to 2.10; at $\lambda _ { \mathrm { F G } } = 1 0 ^ { - 2 }$ worst-group accuracy is only 22.7%. Historical report summaries support this table; Appendix B’s fresh audits are a separate cohort.

Prospective information training. Use Appendix D.2’s shared model, 50-epoch schedule and seeds 22–26. Backbone fine-tuning uses AdamW with learning rate and weight decay $1 0 ^ { - 4 }$ . FG adds coefficients {.003, .01, .03} and a background probe trained on detached features (AdamW learning rate .001, default weight decay .01), whose scores are detached in the penalty. After one penalty-free epoch, training uses $\alpha = . 7 5$ , ridge .001, batch standard-deviation guard $1 0 ^ { - 6 }$ , groups of at least six observations with nuisance standard deviation at least .05, and a .05 information floor before direct inversion. This floor differs from audit stabilization.

For this prospective 50-epoch cohort, select the final model with largest validation certificate among candidates whose overall validation accuracy is at least ERM minus .01. If none qualify, maximize the certificate over all FG candidates for that seed and record the fallback. Choices are [.01, .03, .01, .003, .01], with fallbacks in seeds 22 and 24. All audit quantities are frozen before intervention scoring. In the prospective replication, FG raises the validation certificate from .289 to .330, but the controlled drop is 52.9 versus 52.6 percentage points for ERM and the originally-correct flip rate is 57.7% versus 56.4%. The accuracy constraint fails in two seeds and background sensitivity persists.

Constrained hidden-slice repair. The primary hidden-slice cohort of Appendix D.4 compares loss, Fisher and combined repair without opening group metadata. Each arm adds its repair penalties to ERM, includes the unchanged model, and must preserve overall, balanced and each-class validation accuracy within .02 of the base. Loss coefficients are {.1, .3, 1}, Fisher coefficients {.01, .03, .1}, and combined candidates use their Cartesian product. The same warm start and fresh AdamW optimizer (lr = .01, weight decay $1 0 ^ { - 4 }$ , gradient clip 10) receive at most 1,200 steps. Calibration NLL at temperature one is checked every 25 steps, with patience 12 and improvement threshold $1 0 ^ { - 5 } ;$ the best checkpoint is restored, with the initial model eligible for selection. The matched ERM continuation sets both penalties to zero. Base temperature remains fixed for information training, selection and subsequent audits.

For the frozen training slice, the dimensionless Fisher penalty is

$$
P _ { \mathrm { F G } } ( w ) = \frac { \mathrm { m a x } \{ F _ { \mathrm { i n i t } } , \varepsilon \} } { F ( w ) + \varepsilon } , \qquad \varepsilon = 1 0 ^ { - 8 } ,\tag{E.6}
$$

Here $F ( w )$ is the frozen slice’s projected predictive information and $F _ { \mathrm { i n i t } }$ its value at the unchanged model, with shrinkage .1, ridge .001, and negative numerical information clipped to zero. Within the accuracy envelope, loss/Fisher repair minimizes frozen-slice loss/maximizes information. Combined selection uses the loss–information Pareto frontier, maximizing the smaller relative improvement; ties favor mean relative improvement, then the smaller coefficient sum normalized by grid maxima, $\lambda _ { \mathrm { l o s s } } + 1 0 \lambda _ { \mathrm { F i s h e r } }$ , then earlier index. Fresh rediscovery uses matched gate/fold seeds, so the reported population can change.  
Table E.3: Constrained repair on the primary hidden-slice cohort, seeds 17–21. Accuracy, NLL and information are means ± 95% Student-t half-widths; composition is a mean percentage. Panel two audits each arm’s rediscovered CLIP-defined slice. The Group DRO oracle uses group annotations.
<table><tr><td>Arm</td><td>Avg. acc. (%)</td><td>Worst acc. (%)</td><td>Unchanged</td></tr><tr><td>Matched ERM continuation</td><td> $7 6 . 3 \pm 0 . 7$ </td><td> $5 7 . 3 \pm 1 . 3$ </td><td></td></tr><tr><td>Loss repair</td><td> $7 6 . 3 \pm 0 . 6$ </td><td> $5 6 . 9 \pm 1 . 3$ </td><td>3/5</td></tr><tr><td>Fisher repair</td><td> $7 6 . 3 \pm 0 . 7$ </td><td> $5 7 . 3 \pm 1 . 3$ </td><td>5/5</td></tr><tr><td>Combined repair</td><td> $7 6 . 4 \pm 0 . 9$ </td><td> $5 6 . 7 \pm 1 . 6$ </td><td>4/5</td></tr><tr><td>Oracle Group DRO</td><td> $8 0 . 0 \pm 0 . 3$ </td><td> $6 4 . 8 \pm 0 . 9$ </td><td></td></tr><tr><td>Arm</td><td>Slice NLL</td><td>Projected info.</td><td>LL mass (%)</td></tr><tr><td>Matched ERM continuation</td><td> $0 . 0 3 1 \pm 0 . 0 1 7$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td>97.0</td></tr><tr><td>Loss repair</td><td> $0 . 0 2 3 \pm 0 . 0 1 7$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 2$ </td><td>97.8</td></tr><tr><td>Fisher repair</td><td> $0 . 0 3 1 \pm 0 . 0 1 7$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td>97.0</td></tr><tr><td>Combined repair</td><td> $0 . 0 3 1 \pm 0 . 0 1 7$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td>97.2</td></tr><tr><td>Oracle Group DRO</td><td> $0 . 0 3 3 \pm 0 . 0 1 8$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 2$ </td><td>95.9</td></tr></table>

Fisher repair chooses the unchanged model in every seed. Paired worst-group changes versus matched continuation are −.4, .0 and −.6 points for loss, Fisher and combined repair, with intervals containing zero. The group-supervised Group DRO oracle reaches 64.8% worst-group accuracy while rediscovered information remains .005. LL denotes landbird/land.

Post hoc group-free coefficient selection. This analysis was designed after earlier group/test outcomes were inspected. Its grid $\{ 0 , . 0 1 , . 0 3 , . 1 , . 3 \}$ uses (E.6), not the known-group objective’s scale. Training is partitioned $7 0 / 1 5 \mathrm { \dot { / } 1 5 \% }$ for fitting/calibration/gate discovery; a frozen train-only slice supplies the penalty. All candidates share the warm start, training seed, optimizer, 1,200-step budget, checkpoint rule and base temperature above. Official validation enters only after heads are fixed; group metadata stays sealed through blind test evaluation.

Let $\widehat { R } _ { \lambda }$ be mean fixed-temperature validation NLL, $s _ { \lambda } = \mathrm { s d } _ { i } ( \ell _ { \lambda i } ) / \sqrt { 1 1 9 9 }$ its within-run standard error, and $\lambda _ { R } = \arg \operatorname* { m i n } _ { \lambda } ( \widehat { R } _ { \lambda } , \lambda )$ lexicographically. Here $\ell _ { \lambda i }$ is the fixed-temperature validation NLL of example i, and sd is the sample standard deviation with divisor 1198. Five task-stratified validation folds rediscover candidate-specific slices; $V _ { k , \lambda }$ is held-out inverse predictive information and $^ { s } V , \lambda$ its sample standard deviation. Selection uses

$$
\begin{array} { r l } & { \Lambda _ { \mathrm { r i s k } } = \{ \lambda : \widehat { R } _ { \lambda } \leq \widehat { R } _ { \lambda _ { R } } + s _ { \lambda _ { R } } \} , \quad U _ { \lambda } ^ { \mathrm { a d j } } = \overline { { V } } _ { \lambda } + t _ { 4 , 9 9 } s _ { V , \lambda } / \sqrt { 5 } , } \\ & { \widehat { \lambda } _ { \mathrm { h i d d e n } } = \operatorname* { m i n } \arg \operatorname* { m a x } _ { \lambda \in \Lambda _ { \mathrm { r i s k } } } ( U _ { \lambda } ^ { \mathrm { a d j } } ) ^ { - 1 } , \quad t _ { 4 , 9 9 } = 3 . 7 4 6 9 5 . } \end{array}\tag{E.7}
$$

The risk allowance uses the minimizer’s SE for all candidates. Bonferroni adjustment covers five coefficients, but overlapping gate-training folds make this a ranking score without independentfold coverage. Accuracy screens and paired inverse-information tests are diagnostics, not selection constraints. The group-supervised comparator maximizes validation worst-group accuracy over the same $\Lambda _ { \mathrm { r i s k } }$ , breaking ties toward smaller λ.

Matched zero continuation restores step zero in all five seeds, explaining equality with frozen ERM despite intervening optimization. Group-free selection equals plain NLL minimization in every seed; the information criterion does not change any choice. Versus matched continuation, paired mean changes are +2.1 average-accuracy points $( [ + . 2 , + 4 . 0 ] ) , + . 0 0 6$ information $( [ + . 0 0 1 , + . { \overline { { . } } } 0 1 0 ] )$ and −12.6 worst-group points $( [ - 2 1 . 6 , - 3 . 7 ] )$ . The group-supervised comparator opens validation annotations only after blind decisions and records are fixed and is not an upper bound. The post hoc coefficient-selection results and primary repair cohort have separate artifact identifiers (see the artifact availability statement), alongside extended settings and full tables.

Table E.4: Post hoc coefficient selection, five paired seeds. Accuracies are test results; information is projected predictive information in test slices defined by validation-rediscovered, frozen gates. Brackets are 95% Student-t intervals. Policies selecting identical models across these seeds share rows; the text explains both coincidences. Coefficient counts are λ : number of seeds.
<table><tr><td>Policy</td><td>Average accuracy (%)</td><td>Worst accuracy (%)</td></tr><tr><td>Frozen ERM / matched zero</td><td>75.5 [74.3, 76.6]</td><td>55.7 [53.3, 58.2]</td></tr><tr><td>Fixed  $\lambda = . 0 3$ </td><td>76.6 [76.1, 77.2]</td><td>51.5 [47.1, 55.8]</td></tr><tr><td>Risk only / group-free</td><td>77.5 [76.5, 78.6]</td><td>43.1 [33.1, 53.1]</td></tr><tr><td>Group-supervised comparator</td><td>77.7 [76.6, 78.8]</td><td>44.2 [34.2, 54.3]</td></tr><tr><td>Policy</td><td>Coefficient counts</td><td>Slice information</td></tr><tr><td>Frozen ERM / matched zero</td><td>-/0:5</td><td>0.007 [0.005, 0.009]</td></tr><tr><td>Fixed  $\lambda = . 0 3$ </td><td>.03 : 5</td><td>0.007 [0.004, 0.010]</td></tr><tr><td>Risk only / group-free</td><td> $0 : 1 , \ . 3 : 4$ </td><td>0.012 [0.006, 0.019]</td></tr><tr><td>Group-supervised comparator</td><td> $0 : 1 , . 1 : 1 , . 3 : 3$ </td><td>0.011 [0.005, 0.018]</td></tr></table>

Operational choice of the penalty. For group-free selection, we propose prespecifying an information requirement and a validation-loss budget, then choosing the smallest coefficient satisfying both. If no candidate qualifies, report the requirement as unmet. With neither a specified information requirement nor a group-labeled audit set, retain $\lambda _ { \mathrm { F G } } = 0 .$ A worst-group-accuracy objective requires group labels or a justified group proxy. This threshold-based policy requires prospective validation; the reported experiment maximizes information within an NLL eligibility set.

## E.4 Tail-Transverse Influence

Fix the predictor, coordinates, calibration, slice rules and finite audit weights $\begin{array} { r } { \pi _ { g } \geq 0 , \sum _ { q } \pi _ { g } = 1 } \end{array}$ . An action specifies a positive-semidefinite path $J _ { g } ( t , a ) = J _ { g } + t H _ { g , a } + o ( t )$ as $t \downarrow 0 ,$ , with symmetric $H _ { g , a }$ and an operator-norm remainder. Allocation, normalization and retained pooling belong in this path; audit weights stay fixed.

One differential for exact and ridge targets. Write $J _ { g } = \left( \begin{array} { c c } { { A _ { g } } } & { { B _ { g } } } \\ { { B _ { q } ^ { \top } } } & { { C _ { g } } } \end{array} \right) $ . Choose a fixed $\rho \geq 0$ with $C _ { g } + \rho I \succ 0 \colon \rho = 0$ gives exact Schur information and $\rho = \lambda _ { \mathrm { r i d } }$ gives the ridge target. Set $\overset { \triangledown } { \boldsymbol { Q _ { g } } } = ( \boldsymbol { C _ { g } } + \rho \boldsymbol { I } ) ^ { - 1 } , \boldsymbol { S _ { \rho } } ( \boldsymbol { J _ { g } } ) = \boldsymbol { A _ { g } } - B _ { g } \boldsymbol { Q _ { g } } \boldsymbol { B } _ { g } ^ { \intercal }$ , and let $\lambda _ { g }$ be its smallest eigenvalue. Since $D Q _ { g } [ H _ { N N } ] = - Q _ { g } H _ { N N } Q _ { g } ,$

$$
\begin{array} { r l } & { D S _ { \rho } ( J _ { g } ) [ H ] = H _ { T T } - H _ { T N } Q _ { g } B _ { g } ^ { \top } - B _ { g } Q _ { g } H _ { T N } ^ { \top } } \\ & { \phantom { D G _ { \rho } ( J _ { g } ) [ H ] = H _ { T T } - } + B _ { g } Q _ { g } H _ { N N } Q _ { g } B _ { g } ^ { \top } , } \\ & { p _ { g } ( u ) = \left( - \begin{array} { l l } { u } & { u ^ { \top } D S _ { \rho } ( J _ { g } ) [ H ] u = p _ { g } ( u ) ^ { \top } H p _ { g } ( u ) . } \end{array} \right. } \end{array}\tag{E.8}
$$

For a simple minimum eigenvalue with unit eigenvector $u _ { g } .$ , differentiating the eigenvalue equation therefore gives

$$
h _ { g , a } : = D _ { a } \lambda _ { g } = p _ { g } ^ { \top } H _ { g , a } p _ { g } , \qquad p _ { g } = p _ { g } ( u _ { g } ) .\tag{E.9}
$$

Let $s _ { a }$ be the candidate’s score vector and $M _ { g , a }$ its matrix contribution to group $g .$ For allocation rate $\eta _ { g } \geq 0$ , additive and normalized contributions have different derivatives:

$$
\begin{array} { c } { H _ { g , a } = \eta _ { g } s _ { a } s _ { a } ^ { \top } \implies h _ { g , a } = \eta _ { g } ( p _ { g } ^ { \top } s _ { a } ) ^ { 2 } , } \\ { J _ { g } ( t , a ) = \frac { J _ { g } + t \eta _ { g } M _ { g , a } } { 1 + t \eta _ { g } } \implies H _ { g , a } = \eta _ { g } ( M _ { g , a } - J _ { g } ) , } \end{array}\tag{E.10}
$$

$$
h _ { g , a } = \eta _ { g } \big [ p _ { g } ^ { \top } M _ { g , a } p _ { g } - \lambda _ { g } + \rho \| Q _ { g } B _ { g } ^ { \top } u _ { g } \| ^ { 2 } \big ] .
$$

Indeed $\begin{array} { r } { Q _ { g } C _ { g } Q _ { g } = Q _ { g } - \rho Q _ { g } ^ { 2 } , \mathrm { s o } p _ { g } ^ { \top } J _ { g } p _ { g } = \lambda _ { g } - \rho \| Q _ { g } B _ { g } ^ { \top } u _ { g } \| ^ { 2 } ; } \end{array}$ ; the correction vanishes at $\rho = 0$ Per unit allocation, the additive rank-one contribution is the squared nuisance-adjusted residual $[ u _ { q } ^ { \top } ( s _ { T , a } - B _ { g } Q _ { g } s _ { N , a } ) ] ^ { 2 } ; \mathrm { a t } \rho = 0$ , this is the efficient-score residual. Replacement uses the allocated new-minus-removed contribution. For one observation appended to group g, its indicator and original sample-count scaling enter $\eta _ { g } ;$ they cannot be omitted when claiming a normalized-augmentation derivative.

Tail sensitivity, including nonsmooth cases. Fix $\varepsilon > 0$ and $0 < \alpha < 1$ , and put $L _ { g } = ( \lambda _ { g } + \varepsilon ) ^ { - 1 }$ Appendix $\mathbf { A . } \dot { 5 } \mathbf { \dot { s } }$ finite CVaR dual uses $\begin{array} { r } { \mathcal { W } _ { \alpha , \pi } = \{ \omega \ge 0 : \sum _ { g } \omega _ { g } = 1 , \omega _ { g } \le \pi _ { g } ^ { - } / ( 1 - \bar { \alpha } ) \} } \end{array}$ . Let $\Omega _ { \alpha } ^ { \star } ( L )$ maximize $\Sigma _ { g } \omega _ { g } L _ { g }$ over this set. If the maximizer $\omega _ { \alpha }$ is unique and eigenvalues are simple,

$$
G _ { \alpha } ( a ) : = - D _ { a } { \mathrm { C V a R } } _ { \alpha , \pi } ( L ) = \sum _ { g } \omega _ { \alpha , g } { \frac { h _ { g , a } } { ( \lambda _ { g } + \varepsilon ) ^ { 2 } } } , \qquad D _ { a } I _ { \alpha , \pi } = I _ { \alpha , \pi } ^ { 2 } G _ { \alpha } ( a ) .\tag{E.11}
$$

Thus $G _ { \alpha } ( a )$ is the first-order decrease in inverse-information CVaR along action a’s specified allocation path. It follows from $D _ { a } L _ { g } = - h _ { g , a } / ( \lambda _ { g } + \varepsilon ) ^ { 2 }$ ; the dual weights include fractional threshold mass and upper-tail normalization.

For a repeated minimum eigenvalue with eigenspace $\mathcal { E } _ { g } ,$ symmetric eigenvalue perturbation gives the first formula below. At a CVaR tie, the right derivative of a maximum is the maximum of its active directional derivatives; negation gives the second:

$$
\begin{array} { r l r } & { } & { h _ { g , a } = \displaystyle \operatorname* { m i n } _ { u \in \mathcal { E } _ { g } } p _ { g } ( u ) ^ { \top } H _ { g , a } p _ { g } ( u ) , } \\ & { } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { } & { G _ { \alpha } ^ { + } ( a ) : = \displaystyle - D _ { a } ^ { + } \mathrm { ~ C V a R } _ { \alpha , \pi } ( L ) = \displaystyle \operatorname* { m i n } _ { \omega \in \Omega _ { \alpha } ^ { \star } ( L ) } \sum _ { g } \omega _ { g } \frac { h _ { g , a } } { ( \lambda _ { g } + \varepsilon ) ^ { 2 } } . } \end{array}\tag{E.12}
$$

These formulas cover either or both sources of nondifferentiability; the reciprocal-certificate identity holds with right derivatives. An arbitrary fixed maximizing tail vector is only a score unless it attains this direction-dependent minimum.

For assigned label $y _ { a }$ , tail-transverse influence multiplies the directional information gain by current prediction loss:

$$
\mathrm { T T I } ( a ) = G _ { \alpha } ( a ) \ell _ { w } ( x _ { a } ^ { \mathrm { c f } } , y _ { a } ) , \qquad \ell _ { w } ( x , y ) = - \log p _ { w } ( y \mid x ) ,\tag{E.13}
$$

using $G _ { \alpha } ^ { + }$ at nonsmooth points. This ranks actions at the fixed predictor; finite augmentation, refitting and changed normalization require a new audit. The implemented residual-square surrogate $\widetilde { G } _ { \alpha }$ below omits allocation/baseline terms and is distinguished from this derivative.

## E.5 Feature-Space Acquisition Experiments

Candidates, budgets and refitting. Using training-standardized frozen MobileNetV3-small representations, let $\bar { v } _ { y , b }$ denote the official-training mean for task label $y$ and background $b \in \{ 0 , 1 \}$ . Each official-validation candidate becomes

$$
\begin{array} { r } { v ^ { \mathrm { c f } } = v + \gamma _ { \mathrm { c f } } \big ( \bar { v } _ { y , 1 - b } - \bar { v } _ { y , b } \big ) , \qquad y ^ { \mathrm { c f } } = y , \quad b ^ { \mathrm { c f } } = 1 - b . } \end{array}\tag{E.14}
$$

This preserves the assigned label, not necessarily semantic or pixel-level content. The reported unit-strength run uses label-conditional means for every candidate. A balanced logistic background probe, fitted and standardized on training data, stays fixed and scores the translated features. Test data supply final evaluation only.

Budgets $b _ { \mathrm { a c q } } = . 0 1 , . 0 2$ , .05 use $k = \mathrm { m i n } \{ \mathrm { m a x } ( 1$ , round $( b _ { \mathrm { a c q } } n _ { \mathrm { t r a i n } } ) ) , n _ { \mathrm { c a n d } } \}$ , giving 48, 96, 240 candidates from the 1,199-example validation pool relative to 4,795 original training examples. These are formula-implied counts; historical selected-index records were not recovered. Acquired features and labels augment training, and a balanced logistic task head is fitted from scratch, without an FG penalty or special acquisition weights. Both probe and head use L-BFGS, at most 1,000 iterations, and library defaults $( \bar { C } = 1$ , tolerance $1 0 ^ { - 4 }$ , fitted intercept). Class weights are recomputed on the augmented set. New task logits are normalized by augmented-training moments, while nuisance normalization stays fixed. Acquired validation examples are no longer an independent evaluation sample. Seeds are $^ { 1 7 - 2 1 ; }$ the historical scikit-learn version was not pinned.

Implemented score and its scope. Let $( z , q )$ be the task logit and nuisance score standardized on original training data. Original validation-group moments are unweighted second moments, without $p ( 1 - p ) ;$ ; put $\beta _ { g } = \bar { J _ { T N , g } } / ( J _ { N N , g } + \bar { . 0 0 1 } ) , \lambda _ { g } = J _ { T T , g } - \bar { J _ { T N , g } ^ { 2 } } / ( J _ { N N , g } + . 0 0 1 )$ , and $L _ { g } = ( \lambda _ { g } + \varepsilon ) ^ { - 1 }$ with $\varepsilon = 1 0 ^ { - 8 }$ . Numerical negative information is floored at zero. For the threshold-inclusive tail $T = \{ g : L _ { g } \geq$ ≥ quantile $_ { . 9 5 } ( \breve { L } _ { 1 } , \dots , L _ { 4 } ) \}$ , the code ranks by

$$
\widetilde { G } _ { \alpha } ( a ) = \frac { \sum _ { g \in { \cal T } } ( \lambda _ { g } + \varepsilon ) ^ { - 2 } ( z _ { a } ^ { \mathrm { c f } } - \beta _ { g } q _ { a } ^ { \mathrm { c f } } ) ^ { 2 } } { \operatorname* { m a x } \{ \sum _ { g \in { \cal T } } ( \lambda _ { g } + \varepsilon ) ^ { - 2 } , 1 0 ^ { - 1 2 } \} } , \qquad \widetilde { \mathrm { T T } } _ { \alpha } ( a ) = \widetilde { G } _ { \alpha } ( a ) \ell _ { w } ( v _ { a } ^ { \mathrm { c f } } , y _ { a } ) .\tag{E.15}
$$

Every tail projector is applied to every candidate, irrespective of its assigned group; for a unique worst group the inverse-square weight cancels. The score lacks group allocation, sample-count scaling, old information subtraction and replacement terms. Its nonnegative residual square is an unnormalized rank-one contribution at fixed coordinates, not the derivative of normalized augmentation or of the post-refit certificate. Threshold-inclusive ties also differ from the fractional/direction-dependent CVaR convention in Appendix E.4.

High loss uses $\ell _ { w } ;$ pure tail selects by $\widetilde { G } _ { \alpha }$ alone. Rank product maps ascending argsort positions for each score evenly from 0 to 1, then multiplies their rank scores; equal values receive separate ranks. Uncertainty product multiplies $\widetilde { G } _ { \alpha }$ by $p ( 1 - p )$ . The acquisition baseline has 78.2% average and 52.5% worst-group accuracy, certificate .209, and weakest-information group landbird/land; it is distinct from the predefined-diagnostic cohort.

Table E.5: Feature-space acquisition, means over seeds 17–21. ∆a is worst-group accuracy gain in percentage points and $\Delta I$ the absolute ridge-certificate gain. Panel (a) gives the complete strengthby-budget comparison; panel (b) retains pure-tail and alternative-product controls at unit strength. Only aggregate reports survive, so historical seed-level intervals are unavailable.
<table><tr><td colspan="7">(a) Strength and budget: implemented geometric score × loss versus high loss</td></tr><tr><td colspan="2"></td><td colspan="2"> $\widetilde { \mathrm { T T I } _ { \alpha } }$ </td><td colspan="2">High loss</td></tr><tr><td>γcf 0.25</td><td>Budget</td><td> $\Delta a$ </td><td> $\Delta I$ </td><td> $\Delta a$ </td><td>∆I</td></tr><tr><td></td><td>1%</td><td>+4.05 +7.17</td><td>+0.0308</td><td>+5.76</td><td>+0.0411</td></tr><tr><td rowspan="3">0.5</td><td>2% 5%</td><td></td><td>+0.0366</td><td>+7.63</td><td>+0.0477</td></tr><tr><td>1%</td><td>+8.88</td><td>+0.0521</td><td>+10.12</td><td>+0.0590</td></tr><tr><td>2%</td><td>+6.39</td><td>+0.0205</td><td>+4.36</td><td>+0.0328</td></tr><tr><td rowspan="3">1</td><td>5%</td><td>+5.61</td><td>+0.0300</td><td>+4.83</td><td>+0.0446</td></tr><tr><td>1%</td><td>+6.54 +4.52</td><td>+0.0417</td><td>+8.57 +3.27</td><td>+0.0437</td></tr><tr><td>2%</td><td>+6.85</td><td>+0.0317 +0.0452</td><td>+5.76</td><td>+0.0603</td></tr><tr><td rowspan="5">1.5</td><td>5%</td><td>+7.94</td><td>+0.0606</td><td>+7.01</td><td>+0.0581 +0.0630</td></tr><tr><td>1%</td><td>+7.48</td><td>+0.0662</td><td>+7.63</td><td></td></tr><tr><td>2%</td><td>+8.72</td><td>+0.0826</td><td>+9.81</td><td>+0.1018</td></tr><tr><td>5%</td><td></td><td></td><td>+12.31</td><td>+0.1006</td></tr><tr><td></td><td>+9.97</td><td>+0.1159</td><td></td><td>+0.1137</td></tr></table>

<table><tr><td colspan="3">(b) Unit-strength controls Pure tail</td><td colspan="2">Rank product</td><td colspan="2">Score ×p(1 − p)</td></tr><tr><td>Budget</td><td> $\Delta a$ </td><td>∆I</td><td> $\Delta a$ </td><td>∆I</td><td>∆a</td><td>∆I</td></tr><tr><td>1%</td><td>+0.62</td><td>+0.0045</td><td>+5.45</td><td>+0.0287</td><td>+1.56</td><td>+0.0012</td></tr><tr><td>2%</td><td>+2.80</td><td>+0.0098</td><td> $+ 6 . 5 4$ </td><td>+0.0326</td><td>+1.71</td><td>+0.0073</td></tr><tr><td>5%</td><td>+3.58</td><td>+0.0340</td><td>+6.07</td><td>+0.0516</td><td>+3.12</td><td>+0.0230</td></tr></table>

At $\gamma _ { \mathrm { c f } } = 1$ , the loss product exceeds high loss in mean worst-group accuracy at all budgets, while high loss gives larger certificate gains. This advantage is strength-dependent: high loss wins accuracy at every budget for .25 and 1.5, and at the 5% budget for .5. The rank product also exceeds the chosen loss product at 1% (5.45 versus 4.52 points); the chosen product is not uniformly best. Information gains and predictive gains are distinct outcomes.

Table E.5 reproduces historical reports, not fresh runs. Missing per-seed JSON/CSV, selected indices and pinned versions preclude uncertainty reconstruction. The artifact supplies rerun code, aggregate candidate score/loss summaries, allocation mechanics and protocols. Five seeds on fixed splits do not constitute five independent datasets.

## Artifact Availability

Code and supporting artifacts. The public repository and versioned release camera-ready-v22 are available at

https://github.com/E-zClap/fisher-glass-learning/releases/tag/cam era-ready-v22.

The release provides experiment code, dependencies, derived numerical inputs, per-seed records, checksums, and commands for reproducing the retained analyses and independently rerunning the experiments. Its reproduction guides specify data preparation, environments, and experiment settings. Its artifact manifest maps manuscript tables to the supporting files and identifies the corrected hiddenslice cohort. Expanded appendix sources retain displaced execution details and numerical records; the manuscript map records consolidated table labels. The prospective matrix and auxiliary flip-prediction reanalyses can be replayed from the included frozen numerical inputs. Selected image examples used in Figure 1 are included with source provenance. Full benchmark datasets and pretrained weights are not redistributed; dataset sources and access terms are documented separately. Historical synthetic and frozen-training summaries and the acquisition results retain their stated provenance: where original per-seed outputs are unavailable, the release includes the supporting aggregate reports and executable rerun workflows; fresh experiment replication does not imply exact replay of missing historical outputs.