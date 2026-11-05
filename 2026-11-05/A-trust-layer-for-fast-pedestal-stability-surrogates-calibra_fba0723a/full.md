# A trust layer for fast pedestal-stability surrogates: calibrated error bars, out-of-distribution alarms and few-shot recalibration

Amin Vakhshouri<sup>a)</sup>

dataquanta LLC, Delaware, USA

(Dated: 5 November 2026)

Machine-learning surrogates of linear peeling-ballooning stability now run in milliseconds, but they are mostly deployed as point estimators: they do not say how wrong they may be, or when a query lies outside what they were trained on. For a fusion company designing a new device, every equilibrium of interest is outside the training data of existing surrogates, so this is the case that matters. We describe a model-agnostic trust layer for stability surrogates, made of conformal error bars, out-of-distribution alarms that route doubtful cases to the physics code, and recalibration from a few simulations of a new design, and we test it with pre-registered hypotheses. On a large public dataset of JET-like equilibria with ideal-MHD growth rates, the error bars hold their nominal coverage in distribution and fall short under a shift in plasma shape; a few dozen simulations of the new shape, or routing the cases the alarms flag, restore it. On resistive BOUT++ surrogates, the bars stay calibrated on held-out runs, the alarm flags tokamaks absent from training, and a scenario scan is screened in seconds instead of hours. Across equilibria from the SPARC library and the DIII-D coil set, a surrogate trained on our own graded BOUT++ labels beats the baseline on unseen equilibria, while a surrogate trained elsewhere ranks the same cases backwards and its alarm declines them. Building our own neural-operator surrogate, the harness exposed two input defects whose fix closed its gap to a simple network. For every surrogate tested, the trust layer keeps coverage near target, with a bar width set by that surrogate’s own error. Not every pre-registered hypothesis held, and we report those results and the limits of all of them.

## I. INTRODUCTION

Linear stability of the edge pedestal to peelingballooning modes sets the pedestal height in predictive models such as EPED<sup>1</sup> and constrains the operating space of present and planned tokamaks. Codes that compute it (ELITE, MISHKA, BOUT++, PB3D) need minutes to hours per equilibrium<sup>2,3</sup>, so scenario scans and between-shot analysis are rationed by compute. Machine-learning surrogates remove that bottleneck: EPED1-NN<sup>4</sup>, EuroPED-NN<sup>5</sup>, MISHKA-NN<sup>6</sup>, KARHU<sup>7</sup> and, most recently, ELMO<sup>8</sup> predict pedestal or stability quantities in milliseconds.

Speed alone does not make a surrogate usable for decisions. A physicist or an engineer will act on a growth rate only if it comes with an error bar that means what it says, and with a warning when the input is unlike the training data. Existing pedestal surrogates either re turn point estimates (KARHU, MISHKA-NN, EPED1- NN) or Bayesian uncertainties whose calibration is not guaranteed: ELMO’s authors report that their Gaussianprocess intervals cover 62% of test cases at a nominal 68% and 86% at a nominal 95%<sup>8</sup>. The problem is sharpest for fusion companies designing new devices, whose equilibria are by construction outside the training distribution of every existing surrogate.

This paper treats the missing piece, a trust layer, as the object of study. The layer wraps any surrogate and supplies (i) distribution-free conformal intervals with a finitesample coverage guarantee under exchangeability<sup>9,10</sup>, (ii)

out-of-distribution (OOD) alarms that route unfamiliar cases to the physics code, and (iii) cheap recalibration on a handful of labelled cases from a new device. Conformal prediction has been applied to fusion field surrogates by Gopakumar et al.<sup>11,12</sup>; to our knowledge this is its first systematic test on a stability surrogate, together with shift, alarms and recalibration.

Contributions:

1. A pre-registered evaluation of the trust layer on the public KARHU dataset (38,879 equilibria)<sup>13</sup>, under a controlled plasma-shape shift, with all outcomes reported.

2. Two practical onboarding paths for a new device: recalibration on ∼30 labelled cases, or zero-label routing of flagged cases to the physics code.

3. A resistive BOUT++ case study with real-machine equilibria: calibrated intervals, an ensemble alarm that separates seen from unseen tokamaks, and a decision-level scenario scan.

4. The same harness applied to our own architecture, which exposed two input defects (a radial grid too coarse at the pedestal and unscaled profiles) and confirmed their fix on a pre-registered shifted test.

5. A pre-registered onboarding test (how many simulations of a new design, chosen how) and a path from a G-EQDSK file to a calibrated answer or a reasoned refusal.

6. An account of what went wrong along the way (Sec. VI), including a reported number we could not reproduce.

## II. TRUST LAYER

Let $\hat { \gamma } ( x )$ be any surrogate of the growth rate $\gamma$ for input x.

Split conformal intervals. Given n labelled calibration cases not used for training, with scores $s _ { i } = | \gamma _ { i } - \rrangle$ $\hat { \gamma } ( x _ { i } ) |$ , the interval $\hat { \gamma } ( x ) \pm q$ with q the $\lceil ( n + 1 ) ( 1 - \alpha ) \rceil { \mathrm { - t h } }$ smallest score covers a new exchangeable case with prob ability at least $1 - \alpha ^ { 1 0 }$ . We use $\alpha = 0 . 1$ . The guarantee is marginal, not conditional on x.

Mondrian (per-range) calibration. Computing q separately within bins of the predicted growth rate gives coverage per bin<sup>9</sup>, which matters because errors are not uniform across stable, marginal and strongly unstable cases.

Weighted conformal under covariate shift. When the test distribution difers from calibration by a known or estimable likelihood ratio w(x), weighting the calibration scores restores coverage<sup>14</sup>. We estimate w with a logistic domain classifier on shape features. Weighted conformal requires overlapping support: where the calibration data contain no comparable cases it returns infinite intervals, which is the correct answer.

Few-shot recalibration. With m labelled cases from a new device, split conformal on those m cases alone gives a valid interval for $m \geq 9$ at $\alpha = 0 . 1$ ; the open question is how its width and its draw-to-draw variability depend on m.

OOD alarms and routing. We use two alarms: the spread $\sigma ( x )$ of a $K = 5$ deep ensemble<sup>15</sup>, and a Mahalanobis distance of the input from the training data<sup>16</sup>. Each threshold is set from in-distribution calibration data only (99th percentile, or the maximum when n is small), so the false-alarm rate is fixed before any shifted data are seen. Flagged cases are sent to the physics code; the coverage that matters is then the coverage of the cases kept.

## III. DATA AND SURROGATES

KARHU (ideal MHD). The public KARHU training set<sup>13</sup> contains 38,880 JET-ILW-like equilibria with MISHKA growth rates for toroidal mode numbers 3–50, pressure, safety-factor and $R B _ { \phi }$ profiles on 64 flux points, a 128-point boundary, and global scalars. The target is the maximum growth rate $\gamma _ { \mathrm { { m a x } } } ~ \in ~ [ 0 , 0 . 5 9 ] ~ ( 2 7 . 4 \%$ stable); one case without a label is dropped. The boundary is normalised and up-down symmetric; elongation barely varies (interquartile range within 1.67), triangularity δ spans 0.02–0.59, so δ is the shift variable.

$B O U T + +$ (resistive MHD). Linear growth rates of the BOUT++ elm pb model<sup>2</sup> on the CBM18 equilibrium, varying resistivity η and viscosity ν: 50 training runs on an η×ν grid $( \eta \in [ \dot { 1 0 ^ { - 6 } } , 2 \times 1 0 ^ { - 4 } ] , \nu \in [ 3 \times 1 0 ^ { - 7 } , \stackrel { \sim } { 3 } \times 1 0 ^ { - 4 } ] )$ and 42 held-out runs at geometric midpoints of that grid, generated afterwards. Four real-machine equilibria (three SPARC, one DIII-D) are used in fine-tuning;

TABLE I. Pre-registered outcomes on KARHU (MLP ensemble).
<table><tr><td>Result</td><td></td><td>Verdict</td></tr><tr><td>H1</td><td>89.9% at 90%,  $q = \pm 0 . 0 1 6 .$   $R ^ { 2 } = { }$  0.933 1</td><td>pass</td></tr><tr><td>H2</td><td>hard shift: 85.3% (criterion &lt; 85%)</td><td>not met</td></tr><tr><td>H3</td><td>soft shift: 89.0% → 91.1% weighted</td><td>pass (weak)</td></tr><tr><td>H4</td><td> $m = 1 0 / 3 0 / 1 0 0 \colon 9 1 . 5 / 9 0 . 3 / 9 0 . 4 \% ,$   $\pm 0 . 0 5 5 / \dot { 0 } . 0 \dot { 2 } 6 / 0 . 0 2 4$ </td><td>knee at 30</td></tr><tr><td>H5</td><td>AUROC: spread 0.63, distance 0.73</td><td>mixed</td></tr><tr><td>H6</td><td> $\gamma ~ > ~ 0 . 2 \colon ~ 7 8 . 4 \% ~  ~ 8 8 . 4 \%$  (Mon- drian)</td><td>pass</td></tr></table>

JET 79692, NSTX 132588 and ITER 15 MA are never seen.

Surrogates. MLP: a 4-layer multilayer perceptron on the 333 KARHU inputs, used to test the trust layer independently of architecture. plasmafast: a flux-surface attention network (0.80 M-parameter encoder) that maps an equilibrium and $( \eta , \nu )$ to γ. Its input is built from the $q ( \psi )$ and $p ( \psi )$ profiles on a 64 × 64 flux grid, with pressure rescaled to a fixed axis value; it does not see plasma shape or the pressure magnitude. For BOUT++ we feed $\log _ { 1 0 } \eta$ and $\log _ { 1 0 } \nu$ directly to the growth-rate head (Sec. VI). For KARHU $\left( { } ^ { 6 6 } \mathrm { v } 2 ^ { 3 7 } \right)$ we feed 15 global quantities $( \beta _ { N } , \ \beta _ { p } ,$ , q<sub>95</sub>, q<sub>0</sub>, κ, δ, inverse aspect ratio, field, current, axis pressure and others) to the head and train from scratch.

## IV. RESULTS ON KARHU (PRE-REGISTERED)

The hypotheses, pass criteria and one design amendment were committed before any result (repository file docs/research/karhu trust layer plan.md). Splits: S0 random 60/20/20; S1 hard shift, training and calibration on $\delta \leq 0 . 3 5 6$ (70th percentile), test on the rest $( n = 1 1 , 6 6 4 ) ;$ S2 soft shift, a test set drawn with probability $\propto \ e ^ { 2 z _ { \delta } }$ , overlapping the source. Figure 1 summarises.

In distribution (H1). Coverage is 89.9% at a 90% target with a half-width of 0.016.

Shape shift (H2, H3). Training on lower triangularity and testing on higher lowers coverage from 90.1% on heldout source cases to 85.3%, and raises the mean absolute error from 0.0083 to 0.0106. We had predicted a larger drop; within this JET-like family the shift is mild. On the soft shift, standard calibration already gives 89.0% and weighting gives 91.1%, with 7.7% infinite intervals; on the hard shift weighting returns infinite intervals for 99.7% of cases, as expected without support overlap.

Onboarding a new shape $( H \% )$ . Recalibrating on m labelled cases of the new shape gives mean coverage $9 1 . 5 / 9 0 . 3 / 9 0 . 4 \%$ for $m \ : = \ : 1 0 / 3 0 / 1 0 0$ , with half-widths 0.055/0.026/0.024. Thirty cases bring the interval close to its in-distribution width; one hundred add little. Coverage varies between draws of the m cases: the worst 5% of 30-case draws reach 82%.

![](images/f2ff7b0dd153a033082772f02cf0f06b0baffb52ea0e0d835a90bcb473a8d861.jpg)

(b) Recalibrating on a new shape: how many labels?  
![](images/321bd585fec11a6ab149afe11d148f72ae89e1cca15e992b4b38211490d8247f.jpg)

![](images/bc2210ca03de0f67b1d1df7327048d1fdb61f7aeb422a72ab0594ac5e1ebc2d3.jpg)

![](images/35df28c01a738a94be61619715c463d3452cadfc91c3f2eb5b0a1fefece4a60b.jpg)  
Data: Bruncrona, Kit, Järvinen (VTT), Zenodo 10.5281/zenodo.18629219, CC BY 4.0. Shape shift = triangularity δ. Plan pre-registered: docs/research/karhu\_trust\_layer\_plan.md  
FIG. 1. Trust layer on KARHU (MLP ensemble, $K = 5 )$ . (a) Coverage in distribution and under a soft and a hard triangularity shift, with standard and adjusted calibration. (b) Few-shot recalibration on the new shape: mean and worst-5% coverage over 200 draws, and interval half-width. (c) Exploratory: coverage of the cases kept after routing what each alarm flags. (d) Coverage by true growth rate, one interval for all vs. per predicted-γ range.

Alarms (H5) and routing (exploratory). As detectors of the shifted shape, the ensemble spread is weak (AUROC 0.63) and the Mahalanobis distance moderate (0.73), each at about 1% false alarms. In an analysis we had not pre-registered we asked the question a user needs answered: does the alarm fire on the cases the surrogate gets wrong? The spread ranks misses well (AUROC 0.86; 92% of the cases it flags lie outside the interval) but fires on only 2% of shifted cases; the distance flags 22%. Routing the union (22.7% of shifted cases, 1.9% false alarms) to the physics code leaves 92.5% coverage on the cases kept, without any new simulation.

Coverage by growth rate (H6). One interval for all gives 94.7% for stable cases and 95.0% near marginal stability but only 78.4% for $\gamma > 0 . 2$ . Calibrating per predicted range gives 88–92% in every range, with halfwidths from ±0.007 near marginal stability to ±0.023 for

strongly unstable cases.

## The deployed architecture on KARHU

We then trained plasmafast itself on KARHU and passed it through the same harness, in three preregistered stages (Parts 2–4 of the plan) plus diagnostics that we label exploratory. Table II and Fig. 2 compare all surrogates.

v2: global inputs, original grid. Feeding the 15 global quantities to the growth-rate head matters: a single model without them reaches $R ^ { 2 } = 0 . 7 4 0$ , with them 0.875 (pre-registered V2, passed). The five-member ensemble reaches $R ^ { 2 } = 0 . 8 8 0$ in distribution, short of the MLP’s 0.933 (V1, failed). The trust layer behaves as on the MLP (coverage 90.2%, weighted coverage under the soft shift 90.4%, 30-case recalibration 90.6%), with intervals about twice as wide because the errors are.

TABLE II. Same trust-layer harness, four surrogates on KARHU (ensembles of five; S0 in distribution, S1 hard and S2 soft triangularity shift; p5: worst 5% of 200 draws of the 30 recalibration cases).
<table><tr><td></td><td>MLP</td><td> $\mathrm { v 2 }$ </td><td> $\mathrm { v 3 }$ </td><td> $\mathbf { v 4 }$ </td></tr><tr><td> $R ^ { 2 }$  (S0)</td><td>0.9333</td><td>0.8802</td><td>0.8836</td><td>0.9325</td></tr><tr><td>MAE (SO)</td><td>0.0078</td><td>0.0165</td><td>0.0159</td><td>0.0116</td></tr><tr><td>Coverage (S0)</td><td>89.9%</td><td>90.2%</td><td>90.3%</td><td>90.3%</td></tr><tr><td>Half-width (S0)</td><td>0.016</td><td>0.037</td><td>0.036</td><td>0.025</td></tr><tr><td>MAE, S1 target</td><td>0.0106</td><td>0.0176</td><td>0.0180</td><td>0.0139</td></tr><tr><td>Coverage, S1 target</td><td>85.3%</td><td>88.0%</td><td>87.0%</td><td>86.1%</td></tr><tr><td>Weighted coverage, S2</td><td>91.1%</td><td>90.4%</td><td>91.3%</td><td>90.7%</td></tr><tr><td>30-case recal. (p5)</td><td>90.3 (82)</td><td>90.6 (81)</td><td>90.4 (81)</td><td>90.6 (79)</td></tr><tr><td> $\gamma > 0 . 2 , \mathrm { p e r ~ r a n g e }$ </td><td>88.4%</td><td>83.1%</td><td>84.1%</td><td>88.4%</td></tr></table>

Why plasmafast trailed (exploratory). An MLP given exactly plasmafast’s information (the q and p profiles and the 15 scalars) reaches $R ^ { 2 } = 0 . 9 \dot { 2 } - 0 . \bar { 9 } 4 .$ , so the information is present. Two losses explain the gap. First, the grid: resampling the profiles onto plasmafast’s 64-point uniform-ρ grid raises the $\mathrm { M L P ^ { \prime } s }$ error by 16% (three seeds), because the pedestal falls between grid points (Sec. VI); float16 storage costs nothing. Second, scaling: plasmafast reads the q and p channels raw, while the MLP standardizes every input. In a two-by-two test of single networks (same seed), standardizing q and p per radial point lowers the error by 18% on the original grid $( R ^ { 2 } 0 . { \bar { 8 } } 7 5$ to 0.900); the edge-packed grid below adds nothing on raw inputs but a further 11% once inputs are standardized $( R ^ { 2 } \ 0 . 9 2 4 )$

v3: edge-packed grid. Regenerating the frames on a 96-point grid with 64 points in $\psi _ { N } \in [ 0 . 8 , 1 ]$ (edge spacing 0.0032) alone changes little for the network: $R ^ { 2 } = 0 . 8 8 4 .$ mean error 4% below v2, against a pre-registered 10% (W1 and W2 failed).

v4: edge-packed grid and standardized profiles. Because standardization was chosen after seeing one model’s in-distribution test error, we pre-registered the shifted split S1, which played no part in that choice, as the confirmatory test. On S1’s new shapes the v4 ensemble’s mean error is 23% below v3’s (0.0139 against 0.0180; criterion 10%, passed). In distribution it reaches $R ^ { 2 } \ : = \ : 0 . 9 3 2 5$ level with the MLP (0.933) but 0.0005 short of the bar we had set (failed); its mean error, 0.0116, is 30% below v2’s yet still above the MLP’s 0.0078. Its intervals narrow accordingly (±0.025 against ±0.037 for v2), and calibration per predicted range covers strongly unstable cases at 88.4%, as for the MLP.

What carries over. For every surrogate, from 0.880 to 0.933 in $R ^ { 2 }$ , the conformal layer keeps coverage at 90–91% in distribution and restores it under shift; what changes is the interval width, which tracks the surrogate’s own error. A better surrogate buys a tighter interval; a worse one never buys a false guarantee.

## Onboarding a new design

The customer’s question is how many simulations of a new design a surrogate needs, and whether letting the trust layer choose which ones to run saves any. We preregistered a test (Part 5 of the plan): the v4 ensemble trained on the old family (S1 source) meets the hightriangularity family; a fixed 40% of that family is the test set and the other 6,999 equilibria form a pool whose labels are revealed on request. In every run 30 pool cases drawn at random are used only to calibrate the bar, because cases chosen by the model are not exchangeable with the test set and cannot calibrate it. The rest of the budget is chosen at random or guided by the ensemble (highest spread, then spread out in parameter space), and the members are fine-tuned after each round.

Within this family the model transfers well: with the 30 calibration runs alone its error on the new shapes is 0.0138, against 0.0122 when all 6,999 pool cases are simulated (Fig. 3). To close half of that gap, guided selection needs 107 simulations and random 119, a 10% saving against the 30% we had set as the bar (not met). Guided selection is never worse than random (passed), and the bar stays honest throughout, on the conserva tive side: mean coverage is 93–94% for random and 90– 93% for guided picks, above the 88–92% band we had set (not met), with one guided repeat falling to 81%. In an analysis we had not pre-registered, 1,000 guided simulations match the model trained on all 6,999 (mean error 0.0122 for both; $R ^ { 2 }$ 0.876 against 0.864), and the guided picks cover the sparse corners of the new family that random picks miss. For a mild shift the practical answer is therefore about 30 simulations for an honest bar; guided selection pays when the budget is larger.

## From an equilibrium file to an answer

A tool must accept the file a user has. We built one path from a G-EQDSK to plasmafast’s inputs: the boundary is taken at $\psi _ { N } = 0 . 9 9$ (KARHU equilibria have a fixed boundary), q and the normalized pressure are sampled on KARHU’s flux grid, and the 15 global in puts are computed by their standard definitions, checked against KARHU’s own data where it allows (edge q exactly; normalized current and axis field within 3% and 2%). The frames come from the same function that built the training frames, which regenerates the training set byte for byte, so training and inference cannot drift apart (Sec. VI). The deployable ensemble (v4 with a floor on the per-point spread used for standardization, Sec. VI) reaches $\bar { R ^ { 2 } } = 0 . 9 3 3$ and 90% coverage at ±0.026 on KARHU, and reproduces that accuracy through this path.

We then asked it about five real equilibria (Fig. 4). It answers for none of them. SPARC, ITER and NSTX lie outside the training range in field, current, size or shape, which a range check on the global inputs already shows.

(a) Prediction error: training range vs new plasma shapes  
![](images/e94f506c72f71e33a1ad73be0caf4935156fe201d40d9d96b7070d403d0f7249.jpg)

(b) Error bar: coverage holds, width tracks the model  
![](images/522eb8c3f92972c96e1df0ff560fde53b5ffdac94792206de1db148f04349fcb.jpg)  
Public KARHU data (Bruncrona, Kit, Järvinen, VTT; Zenodo 10.5281/zenodo.18629219, CC BY 4.0). Ensembles of five; pre-registered: docs/research/karhu trust layer plan.md, Parts 2-4

FIG. 2. The deployed architecture through the same trust layer on KARHU. (a) Mean absolute error in distribution (S0) and on the shifted shapes (S1 target). (b) Calibrated half-width and in-distribution coverage at a 90% target.  
(a) Error on the new design vs simulations run  
![](images/0c4fb19a26faf5bf13497a3630c4ce11dcfd139642495cd848b6cb2d3fe98bf4.jpg)  
(b) Error-bar coverage

![](images/b4cb7152e1a7e897c294c9b392a7dfb00f760a4b5e47551da4644b547c683d81.jpg)

(c) Error-bar half-width  
![](images/7a9990f3c64cab4f32c1b32a8a82625c56bcef5f3085340a308ec8187e3f28c9.jpg)  
KARHU (VTT, CC BY 4.0). Old family δ ≤ 0.356, new design above; 3 repeats (bands = range); 30 random simulations per run used only to calibrate the bar. Pre-registered: docs/research/karhu\_trust\_layer\_plan.md, Part 5.  
FIG. 3. Onboarding the high-triangularity family (pre-registered; three repeats, bands show the range). (a) Error on the new family’s test set against the number of new simulations, random against guided selection; diamonds mark where each reaches the target halfway between the base model and the model trained on the whole pool. (b) Coverage of the conformal bar calibrated on 30 random simulations. (c) Its half-width.

JET and DIII-D are subtler: DIII-D’s global inputs all lie inside the training range, yet the five members disagree (spread 0.053 against a threshold of 0.030). The reason is visible in the profiles: every real pressure profile is more peaked than KARHU’s training set from mid-radius outward $( p / p _ { 0 } = 0 . 3 2$ at $\psi _ { N } = 0 . 6$ for DIII-D and 0.17 for JET, against 0.59 for KARHU’s 5th percentile). The ensemble caught a shift that a check on scalar inputs would have missed, and the tool reports it as a reason, not just a flag. A surrogate that is to answer for these machines needs training equilibria with realistic profile shapes.

## V. RESULTS ON BOUT++

Calibrated intervals. On the 42 held-out runs, the surrogate’s mean absolute error is 0.028 and split conformal gives 92.3% coverage at a 90% target with a halfwidth of 0.052–0.054. On the CBM18 reference case $( \eta = 1 0 ^ { - 4 } , \nu = 3 \times 1 0 ^ { - 7 } , \mathrm { B O U T + + } \gamma = 0 . 4 2 3 )$ the prediction is 0.413 (−2.5%). As a code-to-code check of BOUT++ itself, the near-ideal run $( \eta = 1 0 ^ { - 6 } )$ gives 0.294 against PB3D’s ideal-MHD 0.245 on the same equilibrium (+20%).

![](images/482d5b800d0554791d858d54065502a79e04094733ebef6c52bf8b1f57b10660.jpg)  
FIG. 4. The deployable surrogate on five real G-EQDSK equilibria. Columns: plasma boundary used; pressure profile against the KARHU training envelope (5–95%), with the model’s radial grid as ticks; global inputs against their training range (dark: 5–95%, light: 0.5–99.5%); verdict and reasons.

Alarm on real machines. Table III lists the spread of a five-member head ensemble for six real equilibria, relative to the largest spread on the in-distribution held-out set. All six inputs are built with the inference pipeline (Sec. VI). All three unseen tokamaks are flagged, including JET, where the surrogate is badly wrong (0.11 vs. BOUT++ 1.81); none of the three fine-tuning equilibria are. With six cases this is a demonstration, not a rate. An input-distance alarm, useful on KARHU, flags all six here, because the training data contain essentially one

geometry.

Many equilibria. The results above rest on one equilibrium family (CBM18 and four fine-tuning cases). To test the recipe across equilibria, we graded BOUT++ elm pb tor growth rates on 96 equilibria from the public SPARC library and from FreeGS solutions on the DIII-D coil set (127 runs), keeping only runs whose seeded mode is resolved (radial width ≥ 5 cells), fitted with $R ^ { 2 } \ge 0 . 8 ,$ and not dominated by a numerical mode at the radial boundary (Sec. VI). In a pre-registered test with whole equilibria held out (grouped five-fold cross validation, five fold assignments), gradient-boosted trees on the 15 global inputs, $\log _ { 1 0 } \eta$ and three pedestal metrics reach a mean absolute error of 0.087 on SPARC equilibria against 0.200 for the median (ratio 0.44; preregistered $\mathrm { p a s s } \leq 0 . 7 0 )$ Group-wise CV+ 90% intervals cover $9 0 . 4 \%$ of all held-out labels and 86.4% of SPARC labels, with a median half-width of 0.20 (Fig. 5). The KARHU-trained surrogate ranks the same SPARC growth rates backwards (Spearman −0.52): SPARC lies outside KARHU’s range, and its alarm declines these cases (Sec. IV). All DIII-D labels so far are stable, so the signal is SPARC’s; the stable cases have pedestal pressure gradients $\alpha _ { \mathrm { m a x } } \lesssim 8$ , the unstable SPARC cases $\alpha _ { \mathrm { m a x } } \geq 1 5$

TABLE III. Ensemble-spread alarm on real equilibria (BOUT++ surrogate). Threshold: largest spread on 42 indistribution held-out runs.
<table><tr><td>Equilibrium</td><td>In training</td><td>Spread / mean</td><td>Alarm</td></tr><tr><td>SPARC 1358</td><td>yes</td><td>1.0×</td><td>一</td></tr><tr><td>SPARC 1514</td><td>yes</td><td>1.1×</td><td>一</td></tr><tr><td>DIII-D 145419</td><td>yes</td><td>0.6×</td><td>一</td></tr><tr><td>JET 79692</td><td>no</td><td>5.9×</td><td>yes</td></tr><tr><td>NSTX 132588</td><td>no</td><td>9.9×</td><td>yes</td></tr><tr><td>ITER 15 MA</td><td>no</td><td>4.8×</td><td>yes</td></tr></table>

Scenario scan. Figure 6 screens the 42 held-out operating points against a growth-rate budget: a point is accepted if its whole interval is below the budget, rejected if above, and sent to BOUT++ otherwise (intervals fitted leave-one-out). The scan takes about 2 s; BOUT++ took a median 16.7 min per point on x86 CPUs, about 12 h serially. Across budgets from 0.05 to 0.45, 289 confident decisions contain no error; 43–98% of the scan is settled without BOUT++, least when the budget sits inside the dense high-γ band.

## VI. LESSONS AND CORRECTIONS

We report the problems found while preparing this work, because each would have misled a user.

• An irreproducible number. An earlier version of this work (our conference abstract) reported that ensemble spread rank-correlates with error at 0.69. The ensemble had been scored before one member finished training; with all members trained the correlation is about zero in distribution. Indistribution errors are nearly uniform, so a fixed conformal interval is already the right tool, and we use the spread only as an OOD alarm.

• Inputs the network could not see. Resistivity and viscosity entered as raw constant channels of order

$1 0 ^ { - 6 } – 1 0 ^ { - 4 }$ next to order-one channels; the growthrate head saturated at 0.33 for every case above $0 . 2 5 ,$ on training data too. Feeding $\log _ { 1 0 } \eta$ and $\log _ { 1 0 } \nu$ to the head halved the held-out error (0.053 to 0.028) and the interval width.

• A validation gate on the wrong case. A check against PB3D read the input of a diferent run because of an indexing error; fixed, it also showed that comparing a resistive surrogate with an ideal code is the wrong test, so we judge the surrogate against BOUT++ and report BOUT++ against PB3D separately.

• Missing global inputs. The deployed input keeps only profile shapes; plasma shape and pressure magnitude are not seen. This is a main reason the BOUT++ surrogate fails on unseen machines, and the motivation for the v2 model.

• A grid too coarse at the edge. The deployed input samples q and p on 64 points uniform in $\rho _ { \mathrm { p o l } }$ , an edge spacing of $\Delta \psi _ { N } = 0 . 0 3 1$ ; the steepest pressure gradient in KARHU sits at $\psi _ { N } \approx 0 . 9 7 5  – 0 . 9 9$ , so the pedestal gets one or two points. The resampling alone costs a simple MLP 16% in mean error (three seeds). An edge-packed grid (32 points on ψ<sub>N</sub> ∈ [0, 0.8), 64 on [0.8, 1]) recovers it (Sec. IV).

• The same scaling mistake, twice. After the η, ν lesson above, the q and p profiles still entered raw. Standardizing them per radial point cut a single network’s error by 18%, more than the grid fix; the two together made the KARHU surrogate competitive.

• Two input conventions. The BOUT++ training frames, built from simulation output, and the frames built from a G-EQDSK at inference difer: the radial coordinate spans the simulation annulus $\left( \psi _ { N } ~ 0 . 4 0  – 1 . 1 9 \right)$ in one and axis to separatrix in the other, the safety-factor channel is empty in training, and pressure and poloidal angle are normalized diferently. Real-machine predictions therefore rest on the four fine-tuning frames, which is consistent with the large error on JET. The held-out and scenario-scan results use the training convention and are unafected. One frame builder for training and inference is the fix; the KARHU surrogates already use one, and the shared builder regenerates their training frames byte for byte.

• Which growth rates to trust. Our first quality rule for BOUT++ labels rejected 46 of 64 SPARC runs whose toroidal harmonics grew at one rate. Read on the properly assembled grid, the runs seed only $n \ = \ 1 5 ;$ the other harmonics start at round-of $( 1 0 ^ { - 1 6 }$ of it) and copy its growth, so the label is the seeded mode’s rate and the copies are ignored. The rates of 1–3 that worried us came from a numerical mode at the inner radial boundary, mostly

Surrogate + trust layer on BOUT++ labels: 127 labels, 96 equilibria, grouped by equilibrium (pre-registered)

![](images/6d517a81b053aacdb66d9fb81e8a709ca2f13107a506867597b63c5b2449dcd0.jpg)

![](images/8f7ebb3ca080e6620b7608e9fbd3e2c7a21ab45dcdcd8280c49b683c8e30d6f6.jpg)  
FIG. 5. Surrogate and trust layer on BOUT++ labels, equilibria held out (pre-registered). Left: held-out predictions with the 90% CV+ interval. Centre: error on held-out SPARC equilibria by model. Right: the KARHU-trained surrogate’s output on the same SPARC cases, outside its range.

plasmafast scenario scan — 42 operating points in 2.1s (BOUT++ ≈ 12 h serial → 20,325× faster)  
![](images/4b46fd48029562aa90355d6659fc9bc6908c7d510c56660d252331c745476ccb.jpg)

![](images/c147aa025e1a8edb24307e6df81b758481f86f079b3f11275a774229945893b9.jpg)  
FIG. 6. Scenario scan on 42 held-out BOUT++ operating points at a growth-rate budget of 0.25. Left: decision map in $( \eta , \nu )$ Right: prediction ± calibrated interval, sorted by prediction, with BOUT++ values as dashes.

at the lowest resistivity; such runs are now excluded rather than labelled stable. The trainable SPARC set grew from 14 to 51 labels from the same runs, and an equilibrium’s growth rate agrees across resistivities to a median factor of 1.2.

• Dividing by a near-zero spread. Standardizing each radial point by the training spread failed on real equilibria: near the axis KARHU’s pressure profiles are almost identical, so a small real diference became |z| ≈ 100 and the ensemble members diverged. A floor on the spread (10% of the channel mean) removed that without changing accuracy on KARHU.

## VII. DISCUSSION AND LIMITATIONS

For a group designing a new device, the KARHU results translate into two ways to start trusting a surrogate on their equilibria: about thirty simulations of the new design to recalibrate, or none, at the price of send ing roughly one case in four to the physics code. Every routed case also produces a label where the surrogate is weakest, so routing and recalibration combine into an active-learning loop<sup>17</sup>.

The guarantees are marginal and assume exchangeability between calibration and use; no distribution-free method can certify an individual prediction, which is why routing to the physics code remains part of the design. The KARHU test covers one machine family, one shift variable and ideal MHD. The BOUT++ surrogate is trained on essentially one equilibrium plus four real cases, so it is in efect a model of $\gamma ( \eta , \nu )$ on CBM18, and its real-machine alarm result rests on six equilibria; crossmachine skill needs many equilibria, as in the KARHU surrogates. The input fixes found on KARHU (edgepacked grid, standardized profiles, one frame builder) are not yet applied to the BOUT++ surrogate; the v4 in-distribution result is selection-biased, and only its shifted-split result is confirmatory. On real machines we have no MISHKA labels, so the real-equilibrium result is about refusing correctly, not about accuracy. Nothing here is validated against experiment; anchoring to observed ELM onset is the next step. The approach is model-agnostic and could wrap other surrogates, ELMO<sup>8</sup> and KARHU<sup>7</sup> included.

## ACKNOWLEDGMENTS

We thank A. Bruncrona, A. Kit and A. J¨arvinen (VTT) for making the KARHU training data public under CC BY 4.0. This work was supported by dataquanta LLC.

## DATA AND CODE AVAILABILITY

KARHU data: Zenodo, doi:10.5281/zenodo.18629219. Analysis code, pre-registration and results for the KARHU experiments are available from the author on request (amin@dataquanta.io).

<sup>1</sup>P. B. Snyder, R. J. Groebner, A. W. Leonard, T. H. Osborne, and H. R. Wilson, “Development and validation of a predictive model for the pedestal height,” Phys. Plasmas 16, 056118 (2009).

<sup>2</sup>B. D. Dudson, M. V. Umansky, X. Q. Xu, P. B. Snyder, and H. R. Wilson, “BOUT++: A framework for parallel plasma fluid simulations,” Comput. Phys. Commun. 180, 1467–1480 (2009).

<sup>3</sup>T. Weyens, R. S´anchez, G. Huijsmans, A. Loarte, and L. Garc´ıa, “PB3D: A new code for edge 3-D ideal linear peeling-ballooning stability,” J. Comput. Phys. 330, 997–1009 (2017).

<sup>4</sup>O. Meneghini, S. P. Smith, P. B. Snyder, G. M. Staebler, J. Candy, E. Belli, et al., “Self-consistent core-pedestal transport simulations with neural network accelerated models,” Nucl. Fusion 57, 086034 (2017).

<sup>5</sup>A. Panera Alvarez, A. Ho, A. J¨arvinen, S. Saarelma, and S. Wiesen, “EuroPED-NN: uncertainty aware surrogate model,” Plasma Phys. Control. Fusion 66, 095012 (2024).

<sup>6</sup>C. Heo, B. Kim, O. Kwon, S. K. Kim, and Y.-S. Na, “Development of a neural network model for peeling–ballooning stability analysis in the KSTAR tokamak pedestals,” Nucl. Fusion 64, 076031 (2024).

<sup>7</sup>A. M. Bruncrona, A. Kit, A. E. J¨arvinen, S. Saarelma, and L. Frassinetti, “Machine learning surrogate model for ideal peeling–ballooning pedestal MHD stability,” Phys. Plasmas 32, 092501 (2025).

<sup>8</sup>N. Li, X. Q. Xu, T. Osborne, E. Suchyta, Y. C. Fu, N. Podhorszki, H. Wang, and Z. Li, “ELMO: An uncertainty-aware simulation-to-surrogate workflow for fast pedestal linear-stability prediction,” arXiv preprint arXiv:2608.28763 (2026).

<sup>9</sup>V. Vovk, A. Gammerman, and G. Shafer, Algorithmic Learning in a Random World (Springer, 2005).

<sup>10</sup>J. Lei, M. G’Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman, “Distribution-free predictive inference for regression,” J. Am. Stat. Assoc. 113, 1094–1111 (2018).

<sup>11</sup>V. Gopakumar, A. Gray, J. Oskarsson, L. Zanisi, D. Giles, M. J. Kusner, S. Pamela, and M. P. Deisenroth, “Uncertainty quantification of surrogate models using conformal prediction,” Mach. Learn.: Sci. Technol. 7, 015025 (2026).

<sup>12</sup>V. Gopakumar, A. Gray, L. Zanisi, T. Nunn, D. Giles, M. J. Kusner, S. Pamela, and M. P. Deisenroth, “Calibrated physicsinformed uncertainty quantification,” in Proc. 42nd Int. Conf. Machine Learning (ICML), PMLR, Vol. 267 (2025) pp. 20103– 20141.

<sup>13</sup>A. Bruncrona, A. Kit, and A. J¨arvinen, “KARHU – MHD stability surrogate model – training dataset,” Zenodo (2026), CC BY 4.0.

<sup>14</sup>R. J. Tibshirani, R. Foygel Barber, E. J. Cand\`es, and A. Ramdas, “Conformal prediction under covariate shift,” in Advances in Neural Information Processing Systems 32 (NeurIPS) (2019).

<sup>15</sup>B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” in Advances in Neural Information Processing Systems 30 (NIPS) (2017).

<sup>16</sup>K. Lee, K. Lee, H. Lee, and J. Shin, “A simple unified framework for detecting out-of-distribution samples and adversarial attacks,” in Advances in Neural Information Processing Systems 31 (NeurIPS) (2018).

<sup>17</sup>L. Zanisi, A. Ho, J. Barr, T. Madula, J. Citrin, S. Pamela, et al., “Eficient training sets for surrogate models of tokamak turbulence with Active Deep Ensembles,” Nucl. Fusion 64, 036022 (2024).