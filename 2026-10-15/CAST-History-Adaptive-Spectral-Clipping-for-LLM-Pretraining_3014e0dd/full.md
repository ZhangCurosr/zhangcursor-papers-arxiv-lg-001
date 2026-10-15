# CAST: History-Adaptive Spectral Clipping for LLM Pretraining

Jiacheng Li, Jianchao Tan<sup>∗</sup>, Kefeng Zhang, Yuchen Xie, Xunliang Cai Meituan, Beijing, China {lijiacheng14, tanjianchao02}@meituan.com

## Abstract

Soft spectral clipping (SSC) reduces gradient singular values, but its efect depends on the gradient scale and clipping threshold. We introduce CAST (Clipping with Adaptive Scales and Thresholds), a history-adaptive extension of SSC. AdamW’s previous second moment sets row and column scales. An exponential moving average of the clipped output’s estimated spectral norm sets each matrix’s threshold. CAST restores the original coordinates before the AdamW update. We train a mixture-ofexperts (MoE) model with 19.08B total parameters for 83.89B tokens. CAST achieves lower training and mean validation losses than nine gradient-processing baselines, with the fewest relative gradient spikes. It reaches the best baseline’s final smoothed training loss in 7.2% fewer steps.

## 1 Introduction

Gradient clipping(Zhang et al., 2020) controls large gradients during language model(Team et al., 2025b; Brown et al., 2020; Liu et al., 2024; Yang et al., 2025; DeepSeek-AI et al., 2026; Team et al., 2025a, 2026) pretraining (Pascanu et al., 2013). When applied before AdamW, it also changes the moment estimates used by later updates (Loshchilov and Hutter, 2019). The clipping rule therefore afects both the current update and the optimizer’s state over time.

Global norm clipping rescales the whole gradient by a single factor. This limits the influence of an unusually large gradient while preserving its direction. A matrix gradient can be large mainly along a few directions. Rescaling the entire matrix then also weakens other directions, even when their magnitudes are already moderate. Spectral clipping controls the gradient magnitude in each singular direction. SPECTRA(Jiang et al., 2026) develops a smooth spectral transformation for application before or after a base optimizer(Loshchilov and Hutter, 2019; Kingma and Ba, 2017; Jordan et al., 2024). We use pre-optimizer SSC (SSC-Pre) because CAST also acts on gradients before AdamW. We label its constant and cosine threshold schedules SSC-constant and SSC-cosine.

![](images/88e84f9afe145e1729c377c45a8095d3a7cd1548010c6e74c39ac9b14795440d.jpg)  
Figure 1: Validation loss after 5,000 steps: unweighted mean over 20 subsets (one run per method; seed 48). GS: GradientStabilizer. The vertical axis is truncated. Full comparison: Table 1.

A clipping threshold is meaningful only relative to the gradient scale. Persistent diferences in row and column magnitudes afect the spectrum on which clipping acts. An unusually large value in one part of a matrix may be routine in another, and the overall scale can change over training. A fixed threshold may therefore apply diferent clipping strengths as the gradient scale changes. SPEC-TRA’s decaying pre-clipping threshold shows the value of adjusting this threshold during training. Such a schedule depends on the training step. A history-based rule can instead respond to changes in each matrix. We therefore use each matrix’s history to set its gradient scale and clipping threshold.

We introduce CAST (Clipping with Adaptive

![](images/01b6e3b0799a6513f44014bf96158984b95cfb2a6729492ed114a668da1a8489.jpg)  
Figure 2: CAST adds historical scaling (HS) and output feedback around SSC (Algorithm 1). AdamW uses restored $G _ { \mathrm { o u t } } ;$ its V becomes the next $V _ { \mathrm { o l d } }$ , bias-corrected within S. The EMA hold $h _ { \mathrm { o l d } }$ tracks the estimated $\| Z _ { \mathrm { o u t } } \| _ { 2 }$ . Dashed histories advance only after a successful update.

Scales and Thresholds), which uses two complementary histories (Figure 2). Historical scaling (HS) uses row and column summaries of AdamW’s previous second moment to set the scale for each gradient matrix. SSC acts in these standardized coordinates, and the output returns to its original scale before updating AdamW. AdamW updates both moments from this restored gradient. The updated second moment supplies the scale used at the next step.

We also maintain an exponential moving average (EMA) of the clipped output’s estimated spectral norm, computed in the same standardized coordinates. This estimate measures the largest singular value after clipping. The previous EMA sets the current threshold before clipping begins. CAST retains SSC’s soft map and Newton–Schulz routine. The two histories control the input coordinates and clipping threshold.

Our contributions are threefold:

• Historical scaling. We use row and column summaries of the previous second moment to define the coordinates of spectral clipping. We then restore the gradient’s original scale.

• Output feedback. We adapt each matrix’s threshold to its clipped output history, so the threshold follows changes in the clipped spectrum.

• Pretraining results. CAST achieves the lowest training and mean validation losses among nine baselines in our MoE study (Figure 1). We examine its component efects, gradient dynamics, and training cost.

## 2 Related Work

## 2.1 Spectral Clipping

SPECTRA is the direct starting point for this work (Jiang et al., 2026). It smoothly reduces singular values through a matrix inverse square root, computed with Newton–Schulz iterations(Liu et al., 2025). Its framework combines spectral clipping of optimizer updates with optional gradient preprocessing. This gradient preprocessing forms the SSC baselines in our study.

SPECTRA also studies how the clipping threshold changes during training. Its pre-clipping study compares a fixed threshold of 0.1 with cosine decay from 10 to 0.01. The decaying schedule gives better results. We use the previous second moment to set the clipping scale and the complete clipped output to update the threshold. We retain full-matrix soft clipping throughout. Section 5.1 describes our SSC baselines.

## 2.2 Gradient Transformations

Adafactor represents second moments through row and column statistics (Shazeer and Stern, 2018). Our scale uses the same factored structure, with positive floors, to define the coordinates of spectral clipping. AdamW keeps its full moment states and uses the gradient after we restore its original scale.

AdaGC adapts per-tensor thresholds from previous clipped norms (Wang et al., 2026a). Our spectral history follows this output-based principle for each matrix. It uses an estimated spectral norm in standardized coordinates. Parameters outside the spectral branch use an output-norm controller. ZClip uses running global norm statistics to detect spikes (Kumar et al., 2025). GradientStabilizer transforms global magnitude using two scalar moving averages (Huang et al., 2026). These histories track diferent quantities from AdamW’s coordinatewise second moment.

Other transformations use current parameter or gradient magnitudes. AGC bounds unitwise gradient norms relative to parameter norms (Brock et al., 2021); coordinate clipping bounds individual entries. GradPower applies a signed elementwise power before the optimizer (Wang et al., 2026b). These methods cover global, tensor, coordinate, and spectral gradient processing.

Algorithm 1 Soft spectral clipping (SSC) (Jiang   
et al., 2026)   
Require: $B \in \mathbb { R } ^ { q \times d } , q \leq d ;$ threshold $\tau > 0 ;$   
iterations K   
Ensure: Output $B _ { \mathrm { o u t } }$ and clipping flag a   
1: $C \gets B B ^ { \top }$   
2: u ← min $\{ \| C \| _ { \mathrm { F } } , \quad$ max<sub>i</sub> $\textstyle \sum _ { j } | C _ { i j } | \}$   
3: if $u \leq \tau ^ { 2 }$ then   
4: return (B, false)   
5: end if   
6: $\alpha  \tau ^ { 2 } + u ; A _ { 0 }  ( \tau ^ { 2 } I + C ) / \alpha$   
7: $Y \gets A _ { 0 } ; R \gets I$   
8: for $j = 1 , \ldots , K$ do   
9: $T \gets ( 3 I - R Y ) / 2$   
10: $Y  Y T ; R  T R$   
11: end for   
12: return $( ( \tau / \sqrt { \alpha } ) R B , \mathrm { t r u e } )$

## 3 Soft Spectral Clipping

For a singular value $\sigma \geq 0$ and threshold $\tau > 0$ the SPECTRA soft map is

$$
\phi _ { \tau } ( \sigma ) = \frac { \sigma } { \sqrt { 1 + ( \sigma / \tau ) ^ { 2 } } } .\tag{1}
$$

It slightly reduces small singular values and applies a stronger reduction to larger ones. For $\boldsymbol { B } \in \mathbb { R } ^ { q \times a }$ with $q \leq d ,$ , its matrix form is

$$
H _ { \tau } ( B ) = \tau ( \tau ^ { 2 } I + B B ^ { \top } ) ^ { - 1 / 2 } B .\tag{2}
$$

The inverse square root acts on the full $q \ \times \ q$ Gram matrix. Newton–Schulz iterations approximate this matrix function without computing an explicit singular value decomposition (Jiang et al., 2026).

Algorithm 1 restates the SSC routine from SPECTRA in our notation. It first bounds the largest squared singular value using the full Gram matrix. If that bound is at or below the squared threshold, it returns the input unchanged. Otherwise, it approximates $H _ { \tau }$ with K Newton–Schulz iterations. We also return a Boolean a that records whether clipping was applied. This flag lets the caller preserve the original gradient exactly when SSC is bypassed.

The soft map in Equation 2 describes the active clipping branch. Figure 3 illustrates its shape and the efect of changing the threshold. CAST and both SSC baselines share the gate and the Newton– Schulz approximation. Appendix A gives the input rescaling used to evaluate Algorithm 1 safely in FP32.

## 4 Method: CAST

Let $G ~ \in ~ \mathbb { R } ^ { m \times n }$ be a gradient after accumulation, distributed reduction, and mixed-precision unscaling. We transform it into $G _ { \mathrm { o u t } }$ before either AdamW moment is updated:

$$
\begin{array} { r } { M _ { t } = \beta _ { 1 } M _ { t - 1 } + ( 1 - \beta _ { 1 } ) G _ { \mathrm { o u t } } , } \\ { V _ { t } = \beta _ { 2 } V _ { t - 1 } + ( 1 - \beta _ { 2 } ) G _ { \mathrm { o u t } } ^ { \odot 2 } . } \end{array}\tag{3}
$$

AdamW then applies its usual bias correction and decoupled weight decay. We write  and $\oslash$ for elementwise multiplication and division, and k · k<sub>F</sub> for the Frobenius norm.

Our method makes two changes around Algorithm 1: it standardizes the input and derives the threshold from output history. SSC acts on the full matrix of each selected parameter. We select two-dimensional tensors in the dense optimizer group, excluding embeddings and output matrices by name. Other tensors, including parameters assigned to the expert optimizer, use the output-norm controller described below. Appendix B.1 gives the selection rule for MoE projections and routers.

Algorithm 2 separates these additions from the inherited SSC routine. It shows matrices that have completed calibration and have a gradient at every step. Appendix B covers initialization and missing gradients. Both histories are read before the current AdamW update and committed only after it succeeds.

## 4.1 Historical Scaling

We use the previous second moment to measure each gradient relative to its historical coordinate scales. After $k \geq 1$ moment updates, let $\widehat { V } \ =$ $V / ( 1 - \beta _ { 2 } ^ { k } )$ bdenote the bias-corrected previous second moment. Its row, column, and overall means are

$$
\begin{array} { l } { { r _ { i } = \displaystyle \frac { 1 } { n } \sum _ { j } \widehat { V } _ { i j } , \quad c _ { j } = \displaystyle \frac { 1 } { m } \sum _ { i } \widehat { V } _ { i j } , } } \\ { { b _ { 0 } = \displaystyle \frac { 1 } { m n } \sum _ { i , j } \widehat { V } _ { i j } . } } \end{array}\tag{4}
$$

![](images/6acee61dc08bb17275df422b3e2031071196c9da44f3e44bcb90dfb7826c0fc5.jpg)

![](images/04a3d342c5e302fed27b935d7804a8bc24a7e5821d557327d343663aff93e998.jpg)  
Figure 3: The soft map and CAST’s threshold choice. (a) Hard and soft clipping at τ = 1, following the functional view of SSC in Jiang et al. (2026). (b) The same soft map at $\tau = 1 . 3 h$ for three illustrative values of the previous output history h. These analytical curves describe the active branch in standardized coordinates; the bypass gate and finite Newton–Schulz approximation are separate steps in Algorithm 1.

Algorithm 2 CAST gradient preprocessing for   
AdamW   
Require: Gradients G, old corrected moments $\widehat { V }$   
histories h   
1: for each selected matrix after calibration do   
1. HS   
2: S ← Scale(V) ▷ Equation 5   
3: $Z  G \oslash S ; B \gets \mathrm { O r i e n t } ( Z )$   
2. Threshold from output history   
4: τ ← 1.3h   
5: $( B _ { \mathrm { o u t } } , a ) \gets \mathrm { S S C } ( B , \tau , 5 )$ ▷ Algorithm 1   
6: Z<sub>out</sub> ← Unorient(B<sub>out</sub>)   
7: $G _ { \mathrm { o u t } }  G$   
8: if a = true then   
9: $G _ { \mathrm { o u t } }  S \odot Z _ { \mathrm { o u t } }$   
10: end if   
11: s ← PowerTop(Z<sub>out</sub>, 3)   
12: if $s > 0$ then   
13: Stage $h ^ { \prime } \gets 0 . 9 5 h + 0 . 0 5 s$   
14: end if   
15: end for   
16: Clip other tensors with the historical norm rule   
17: Apply AdamW to all transformed gradients   
18: Commit staged state after a successful update

$$
\begin{array} { r l } & { \mathrm { W i t h } \ \rho = 0 . 0 1 \ \mathrm { a n d } \ \epsilon _ { s } = 1 0 ^ { - 8 } , \mathrm { d e f i n e } } \\ & { \qquad b = \operatorname* { m a x } \{ b _ { 0 } , ( \epsilon _ { s } / \rho ) ^ { 2 } \} , } \\ & { \qquad S _ { i j } = \sqrt { \frac { \operatorname* { m a x } \left( r _ { i } , \rho b \right) \operatorname* { m a x } \left( c _ { j } , \rho b \right) } { b } } , } \\ & { \qquad Z = G \oslash S . } \end{array}\tag{5}
$$

The floors provide a positive reference for sparsely active coordinates. The row and column factors account for persistent diferences in gradient magnitudes. Each coordinate uses one row factor and one column factor. After clipping, we restore the original coordinates with $G _ { \mathrm { o u t } } = S \odot Z _ { \mathrm { o u t } }$

## 4.2 Adaptive Thresholds

We adapt the clipping threshold separately for each selected gradient matrix. Each matrix maintains a positive scalar history h of its clipped output’s estimated spectral norm. The current threshold uses its previous value:

$$
\tau _ { t } = \lambda h _ { t - 1 } , ~ \lambda = 1 . 3 .\tag{6}
$$

We orient Z so that B has the smaller dimension as its row count, then call Algorithm 1 with $K = 5$ When the algorithm returns $a =$ false, we keep the original G directly. Otherwise, we restore the clipped matrix to $Z ' \mathrm { s }$ orientation and set $G _ { \mathrm { o u t } } ~ =$ $S \odot Z _ { \mathrm { o u t } }$

The controller observes the complete standardized output through three power iterations:

$$
\begin{array} { r l } & { s _ { t } = \mathrm { P o w e r T o p } ( Z _ { \mathrm { o u t } , t } , 3 ) , } \\ & { h _ { t } = \beta _ { h } h _ { t - 1 } + ( 1 - \beta _ { h } ) s _ { t } , \quad \beta _ { h } = 0 . 9 5 . } \end{array}\tag{7}
$$

After calibration, positive observations update the history; zero outputs leave it unchanged. Each observation uses the same scale as the current transform. The updated history sets the next threshold.

## 4.3 Implementation

The first 40 successful optimizer updates use global clipping at norm 1 to initialize moments and histories. A selected matrix enters the spectral branch when its previous second moment, history, and participation count are ready. Other tensors use a threshold $\lambda f ,$ where f tracks their clipped output norms. These tensors include the MoE experts and embeddings, so the output-norm controller covers a large part of the model. $\mathsf { A p - }$ pendix B specifies initialization, intermittent participation, and checkpoint restoration.

The distributed implementation all-gathers gradient fragments into complete logical matrices, applies the transformation, and writes each rank’s local optimizer slice. Stored row and column summaries recover the previous second-moment scale without repeatedly gathering the full moment tensor. The transform uses FP32 within BF16 training(Micikevicius et al., 2018). For $\begin{array} { r l r } { q } & { { } = } & { \operatorname* { m i n } ( m , n ) } \end{array}$ , its matrix arithmetic costs $O ( \operatorname* { m a x } ( m , n ) q ^ { 2 } + K q ^ { 3 } + p m n )$ , with $K = 5$ and $p = 3$ . Section 5.5 reports the time per step observed during training.

## 5 Experiments

We evaluate pretraining quality, component efects, and training cost over 5,000 steps (83.89B nominal tokens).

## 5.1 Experimental Setup

Model and recipe. We train a shortcut MoE model(Cai et al., 2025) with 10 blocks, hidden size 1536, and 16 attention heads(Vaswani et al., 2017). Each block contains two multi-head latent attention modules(DeepSeek-AI et al., 2024), two dense SwiGLU modules, and one MoE module(Shazeer et al., 2017; Fedus et al., 2022; Jiang et al., 2024). The model structure gives approximately 19.08B total parameters. Its target activated backbone has 647.6M parameters, excluding input and output embeddings. Actual expert activation depends on routing. The full calculation appears in Appendix C.

The runs share the recorded data and tokenizer configurations. We use sequence length 8192, global batch size 2048, microbatch size 1, and training seed 48. AdamW uses $( \beta _ { 1 } , \beta _ { 2 } )$ = $( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 1 6 }$ , and weight decay 0.1. The learning rate reaches $8 \times 1 0 ^ { - 4 }$ after 500 warmup iterations and remains constant through the comparison window. Training uses BF16 and expert parallelism of eight. The model, optimizer fields, and recorded learning-rate trajectory agree across the conditions.

Baselines. We compare with GlobalGC, AdaGC, SSC-constant, SSC-cosine, GradientStabilizer, GradPower, ClipByValue, AGC, and ZClip. Appendix C.2 gives each baseline’s processing order and hyperparameters.

Both SSC baselines use five Newton–Schulz iterations on the same selected matrices in raw coordinates. SSC-constant uses $\tau = 0 . 1$ . SSC-cosine decays τ from 10 to 0.1 over 5,000 steps. The component study adds three variants of CAST.

Measurements. Training loss is the mean language modeling loss over iterations 4801–5000. We average validation loss over the same 20 subsets at step 5000. These include domain corpora, question–answer text, and code-formatted text. We use the recorded token-loss definition for each subset. The 100- and 500-step training windows provide a sensitivity check. Each condition has one run with seed 48; subsets and time points describe within-run variation.

## 5.2 Pretraining Results

CAST achieved the lowest training and mean validation losses among the nine baselines at step 5000 (Table 1). CAST’s mean validation loss was 1.26× $1 0 ^ { - 2 }$ below SSC-constant and $2 . 1 0 \times 1 0 ^ { - 2 }$ below SSC-cosine. CAST had lower loss on 20 and 19 subsets, respectively. SSC-cosine had higher training and mean validation losses than SSC-constant. AGC had the closest mean validation loss, with a gap of 1 $. . 1 8 \times 1 0 ^ { - 2 }$ . CAST’s training loss remained lowest across the 100-, 200-, and 500-step tail windows.

Figure 4 shows the loss curves. Appendix D reports the losses for all 20 validation subsets.

## 5.3 Ablation Study

Table 2 compares CAST with three component variants. The variant without HS sets $S = 1$ , retaining the spectral EMA and AdamW’s moments. The other variants keep S and use fixed $\tau =$ 0.1 or cosine $\tau$ in standardized coordinates. All three keep CAST’s calibration and output-norm controller for unselected tensors.

The variant without HS increased training and mean validation losses by $2 . 4 2 \times 1 0 ^ { - 3 }$ and $8 . 9 6 \times$

<table><tr><td>Method</td><td>Train loss ↓</td><td>Validation loss ↓</td><td>CAST lower</td></tr><tr><td>CAST</td><td>1.6516</td><td>1.8738</td><td>一</td></tr><tr><td>GlobalGC</td><td>1.6718</td><td>1.8962</td><td>19/20</td></tr><tr><td>AdaGC</td><td>1.6625</td><td>1.8872</td><td>20/20</td></tr><tr><td>SSC-constant</td><td>1.6638</td><td>1.8864</td><td>20/20</td></tr><tr><td>SSC-cosine</td><td>1.6700</td><td>1.8947</td><td>19/20</td></tr><tr><td>GradientStabilizer</td><td>1.6828</td><td>1.9034</td><td>18/20</td></tr><tr><td>GradPower</td><td>1.6823</td><td>1.9139</td><td>20/20</td></tr><tr><td>ClipByValue</td><td>1.6789</td><td>1.9022</td><td>20/20</td></tr><tr><td>AGC</td><td>1.6641</td><td>1.8856</td><td>19/20</td></tr><tr><td>ZClip</td><td>1.6711</td><td>1.8937</td><td>19/20</td></tr></table>

Table 1: Pretraining at 5,000 steps (83.89B nominal tokens). Train: mean over steps 4801–5000. Validation: unweighted mean over 20 subsets at step 5000. The final column counts subsets where CAST has lower loss. All runs use seed 48.

![](images/8d090c1275d0b1672bae1ee031808cf8007bee0847dad1263f782233d6b985e6.jpg)

![](images/bcac00c8aa674ea09f9a0fd206f30c833070239b1e49261a814cb5b607de614f.jpg)

![](images/386eafa930a8726b20cac41578cbdef5df6500d53c9e455157e7be11f3bf50dd.jpg)  
Figure 4: Quality trajectories for all ten primary conditions. Training curves show trailing 100-step means from iteration 2000 onward; validation curves average the same 20 subsets at each complete checkpoint. Appendix E includes the training trajectory from initialization.

<table><tr><td>Condition</td><td>Train loss</td><td>Validation loss</td><td>Gate fraction</td></tr><tr><td>CAST</td><td>1.6516</td><td>1.8738</td><td>0.884</td></tr><tr><td>w/o HS</td><td>1.6541</td><td>1.8747</td><td>0.403</td></tr><tr><td>w/ τ = 0.1</td><td>1.6937</td><td>1.9215</td><td>1.000</td></tr><tr><td>w/ cosine τ</td><td>1.6767</td><td>1.9034</td><td>1.000</td></tr></table>

Table 2: CAST component study (HS: historical scaling). Loss windows follow Table 1; bold marks the lowest values. Gate fraction is the mean share of eligible matrices entering SSC over steps 4801–5000.

$1 0 ^ { - 4 }$ , respectively. CAST had lower loss on 18 validation subsets.

Replacing the fixed threshold with cosine decay lowered training and mean validation losses by

$1 . 7 0 \times 1 0 ^ { - 2 } \mathrm { a n d } 1 . 8 2 \times 1 0 ^ { - 2 }$ . CAST’s mean validation loss was $2 . 9 6 \times 1 0 ^ { - 2 }$ lower than the cosinethreshold variant’s. Both threshold variants activated SSC for every eligible matrix in the tail window.

## 5.4 Training Stability

Input-gradient spikes mark abrupt increases before clipping. Figure 6 shows these norms after learning-rate warmup. We count norms above 1 and relative spikes. A relative spike exceeds twice the preceding 100-step median, including pre-window points when needed. Both counts include consecutive exceedances. Relative spikes track sudden increases against each run’s recent gradient scale. The absolute count applies the same norm threshold across runs.

![](images/ccac1ab64a55955c7a6797fc90c06acf9e6d9ff0b40c4b7fc37869df738423c9.jpg)  
Figure 5: Training trajectories for CAST and its three component variants (HS: historical scaling). Curves show trailing 100-step means from iteration 2000 onward; Table 2 reports their final training and validation losses.

![](images/6e58ea39ab1f32b2fba1b289ed95d09e289445b533e5c6ebf8276338ae52c93c.jpg)  
Figure 6: Raw input-gradient norms over iterations 501–5000 for ten primary conditions and three CAST ablations. The final row removes HS (historical scaling) or replaces the spectral EMA with fixed or cosine thresholds. All panels share a logarithmic vertical scale. Titles report relative spike counts; the dashed line marks norm 1. Table 3 gives the statistics.

CAST recorded one relative spike and no input norm above 1 during iterations 501–5000. SSCconstant and SSC-cosine recorded 22 and 21 relative spikes, respectively (Table 3). CAST and SSCconstant had similar median norms, while CAST had a smaller upper tail.

Among the baselines, GradPower had the lowest median norm but 422 relative spikes. ZClip had the largest isolated norm, at 918.79, and 75 relative spikes. AdaGC had six relative spikes and a lower 99th-percentile norm than CAST. CAST combines the fewest relative spikes with the lowest mean validation loss.

Cosine decay reduced the fixed-threshold variant’s relative spikes from six to two and norms

<table><tr><td>Method</td><td>Median norm</td><td>P99 norm</td><td>Maximum norm</td><td>Steps with  $\mathrm { n o r m } > 1$ </td><td>Relative spikes</td></tr><tr><td>CAST</td><td>0.1510</td><td>0.3702</td><td>0.5476</td><td>0</td><td>1</td></tr><tr><td>GlobalGC</td><td>0.1541</td><td>0.4574</td><td>15.1610</td><td>10</td><td>38</td></tr><tr><td>AdaGC</td><td>0.1343</td><td>0.3485</td><td>1.0541</td><td>1</td><td>6</td></tr><tr><td>SSC-constant</td><td>0.1591</td><td>0.4723</td><td>1.1829</td><td>1</td><td>22</td></tr><tr><td>SSC-cosine</td><td>0.1653</td><td>0.4996</td><td>1.1922</td><td>2</td><td>21</td></tr><tr><td>GradientStabilizer</td><td>0.1701</td><td>0.6413</td><td>4.1736</td><td>5</td><td>31</td></tr><tr><td>GradPower</td><td>0.1142</td><td>0.6987</td><td>4.9144</td><td>25</td><td>422</td></tr><tr><td>ClipByValue</td><td>0.1645</td><td>0.6814</td><td>5.0252</td><td>22</td><td>58</td></tr><tr><td>AGC</td><td>0.1425</td><td>0.4701</td><td>3.9416</td><td>25</td><td>52</td></tr><tr><td>ZClip</td><td>0.1540</td><td>1.2761</td><td>918.7862</td><td>52</td><td>75</td></tr><tr><td>CAST w/o HS</td><td>0.1463</td><td>0.3382</td><td>0.4548</td><td>0</td><td>3</td></tr><tr><td>CAST  $\mathrm { w } / \tau = 0 . 1$ </td><td>0.1207</td><td>2.7853</td><td>4.0557</td><td>194</td><td>6</td></tr><tr><td>CAST w/ cosine τ</td><td>0.1083</td><td>0.8505</td><td>1.6856</td><td>21</td><td>2</td></tr></table>

Table 3: Input-gradient norms before clipping, steps 501–5000. P99: 99th percentile. Relative spikes exceed twice the preceding 100-step median, counting consecutive steps. Bold/underline: lowest/second-lowest counts, including ties. Last three rows: CAST ablations (HS: historical scaling).  
above 1 from 194 to 21.

## 5.5 Training Eficiency

Table 4 compares 64-device runs on the same hardware type. Each device has about 60% of an NVIDIA H100’s compute capacity. CAST takes 2.0% longer per step than SSC-constant and 1.9% longer than ClipByValue. Timings include execution and logging and vary with cluster load. AdaGC has the lowest final smoothed loss among the baselines. CAST first reaches this loss at step 4,639, using 7.2% fewer steps (Figure 7). At the mean step times in Table 4, reaching AdaGC’s loss target takes an estimated 5.5% less time than 5,000 ClipByValue steps.

<table><tr><td>Measure</td><td>CAST</td><td>SSC-constant</td><td>ClipByValue</td></tr><tr><td>Mean (s)</td><td>44.408</td><td>43.533</td><td>43.580</td></tr><tr><td>Median (s)</td><td>44.206</td><td>43.329</td><td>43.375</td></tr><tr><td>P95 (s)</td><td>44.876</td><td>43.960</td><td>44.119</td></tr><tr><td>Tokens/s/ device</td><td>5903</td><td>6022</td><td>6015</td></tr></table>

Table 4: Iteration times over steps 4801–5000 (200 steps per method, 64 devices). Mean is the arithmetic average; Median and P95 are the 50th and 95th percentiles. Throughput is 2048 × 8192/(64t<sup>¯</sup>), where t<sup>¯</sup> is the mean iteration time in seconds. Timings include execution and logging.

![](images/884d97a8471e458e09aacd33481481c3a90e8f4bf90f8d263ec4dc9245baa029.jpg)  
Figure 7: Final 500 steps for CAST and nine baselines (100-step means). CAST reaches AdaGC’s final loss (dashed) at step 4,639. Colors follow Figure 9.

## 6 Conclusion

CAST uses two histories to guide spectral clipping in pretraining. AdamW’s previous second moment sets the gradient scale. An EMA of the clipped output’s estimated spectral norm sets each matrix’s threshold. In 5,000-step MoE pretraining, CAST achieved lower training and mean validation losses than nine baselines and three component variants. It also had the fewest relative gradient spikes. CAST reached AdaGC’s final smoothed loss in 7.2% fewer steps. Including per-step overhead, the estimated time saving is 5.5% relative to 5,000 ClipByValue steps. These results support using both histories to improve pretraining quality and eficiency.

## Limitations

Full-matrix computation and communication times depend on tensor shape. The reported timings also reflect the cluster environment. Changes in hardware or training settings can also change the exact speedup. The timing comparison uses the same model, training settings, hardware type, and device count. This shared setup supports a fair comparison, so the estimated relative speedup remains a useful reference.

## References

Andy Brock, Soham De, Samuel L Smith, and Karen Simonyan. 2021. High-performance large-scale image recognition without normalization. In International conference on machine learning, pages 1059–1071. PMLR.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jefrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.

Weilin Cai, Juyong Jiang, Le Qin, Junwei Cui, Sunghun Kim, and Jiayi Huang. 2025. Shortcut-connected expert parallelism for accelerating mixture-of-experts. Preprint, arXiv:2404.05019.

DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, and 138 others. 2024. Deepseek-v2: A strong, economical, and eficient mixture-of-experts language model. Preprint, arXiv:2405.04434.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chengyu Hou, Chenhao Xu, Chenze Shao, Chong Ruan, Conner Sun, and 300 others. 2026. Deepseekv4: Towards highly eficient million-token context intelligence. Preprint, arXiv:2606.19348.

William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal ofMachine Learning Research, 23(120):1–39.

Tianjin Huang, Zhangyang Wang, Haotian Hu, Zhenyu Zhang, Gaojie Jin, Xiang Li, Li Shen, Jiaxing Shang, Tianlong Chen, Ke Li, Lu Liu, Qingsong Wen, and

Shiwei Liu. 2026. Gradientstabilizer:fix the norm, not the gradient. Preprint, arXiv:2502.17055.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, and 7 others. 2024. Mixtral of experts. Preprint, arXiv:2401.04088.

Xiaowen Jiang, Andrei Semenov, and Sebastian U. Stich. 2026. Enhancing llm training via spectral clipping. Preprint, arXiv:2603.14315.

Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. 2024. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan. github. io/posts/muon, 6(3):4.

Diederik P. Kingma and Jimmy Ba. 2017. Adam: A method for stochastic optimization. Preprint, arXiv:1412.6980.

Abhay Kumar, Louis Owen, Nilabhra Roy Chowdhury, and Fabian Güra. 2025. Zclip: Adaptive spike mitigation for llm pre-training. Preprint, arXiv:2504.02507.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, and 9 others. 2025. Muon is scalable for llm training. Preprint, arXiv:2502.16982.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. Preprint, arXiv:1711.05101.

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. 2018. Mixed precision training. Preprint, arXiv:1710.03740.

Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. 2013. On the dificulty of training recurrent neural networks. In International conference on machine learning, pages 1310–1318. Pmlr.

Herbert Robbins and Sutton Monro. 1951. A stochastic approximation method. The annals of mathematical statistics, pages 400–407.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. Preprint, arXiv:1701.06538.

Noam Shazeer and Mitchell Stern. 2018. Adafactor: Adaptive learning rates with sublinear memory cost. In International conference on machine learning, pages 4596–4604. PMLR.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, M. C., Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y. Charles, H. S. Che, Guanduo Chen, Guangyu Chen, Guanzheng Chen, Huarong Chen, Jia Chen, Jianlong Chen, Jun Chen, and 383 others. 2026. Kimi k3: Open frontier intelligence. Preprint, arXiv:2607.24653.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, and 150 others. 2025a. Kimi k2: Open agentic intelligence. Preprint, arXiv:2507.20534.

Meituan LongCat Team, Bayan, Bei Li, Bingye Lei, Bo Wang, Bolin Rong, Chao Wang, Chao Zhang, Chen Gao, Chen Zhang, Cheng Sun, Chengcheng Han, Chenguang Xi, Chi Zhang, Chong Peng, Chuan Qin, Chuyu Zhang, Cong Chen, Congkui Wang, and 163 others. 2025b. Longcat-flash technical report. Preprint, arXiv:2509.01322.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Guoxia Wang, Shuai Li, Congliang Chen, Jinle Zeng, Jiabin Yang, Dianhai Yu, Yanjun Ma, and Li Shen. 2026a. Adagc: Enhancing llm pretraining stability via adaptive gradient clipping. Preprint, arXiv:2502.11034.

Jinbo Wang, Mingze Wang, Jiaqi Zhang, Wei Wang, Peng Pei, Xunliang Cai, Weinan E, and Lei Wu. 2026b. Gradpower: Powering gradients for faster language model pre-training. Preprint, arXiv:2505.24275.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Jingzhao Zhang, Tianxing He, Suvrit Sra, and Ali Jadbabaie. 2020. Why gradient clipping accelerates training: A theoretical justification for adaptivity. Preprint, arXiv:1905.11881.

## A Numerical Details

## A.1 Orientation and Normalization

For a standardized gradient $Z \in \mathbb { R } ^ { m \times n }$ , the implementation forms a full matrix B with $q \quad =$ min $( m , n )$ rows. It uses $B = Z$ when $m \ < \ n ,$ and $B = Z ^ { \top }$ otherwise, including square matrices. The output is transposed back before restoring S. This convention makes the left Gram $B B ^ { \top }$ the smaller Gram.

The Newton–Schulz routine scales its inputs before forming the Gram matrix. Let

$$
\begin{array} { r l } & { d _ { * } = \operatorname* { m a x } \{ \tau , \underset { i , j } { \operatorname* { m a x } } | B _ { i j } | \} , } \\ & { B _ { 0 } = B / d _ { * } , \qquad t _ { 0 } = \tau / d _ { * } , } \\ & { C _ { 0 } = B _ { 0 } B _ { 0 } ^ { \top } , \qquad u _ { 0 } = \operatorname* { m i n } \{ \| C _ { 0 } \| _ { \mathrm { F } } , \| C _ { 0 } \| _ { \infty } \} , } \end{array}\tag{8}
$$

where $\| C _ { 0 } \| _ { \infty }$ is the maximum absolute row sum. The gate returns an unchanged gradient copy when $u _ { 0 } \leq t _ { 0 } ^ { 2 }$ . In the active branch, define

$$
\alpha = t _ { 0 } ^ { 2 } + u _ { 0 } , \qquad A _ { 0 } = ( t _ { 0 } ^ { 2 } I + C _ { 0 } ) / \alpha .\tag{9}
$$

Starting with $( Y _ { 0 } , R _ { 0 } ) = ( A _ { 0 } , I )$ , the coupled iteration is

$$
\begin{array} { c } { { T _ { j } = \frac { 1 } { 2 } ( 3 I - R _ { j } Y _ { j } ) , } } \\ { { Y _ { j + 1 } = Y _ { j } T _ { j } , \qquad R _ { j + 1 } = T _ { j } R _ { j } . } } \end{array}\tag{10}
$$

The transformed matrix is

$$
\widetilde { B } = ( t _ { 0 } / \sqrt { \alpha } ) R _ { 5 } B .\tag{11}
$$

The rightmost factor is the original full matrix B. An unrepresentable positive normalized threshold or a nonfinite intermediate raises a numerical error before state commit.

## A.2 Spectral Norm Estimation

For a nonzero output X, PowerTop(X, 3) starts from a normalized Gaussian vector x. Each round normalizes $X x$ and then $X ^ { \top } ( X x / \| X x \| _ { 2 } )$ . After three rounds, the observation is $\| X x \| _ { 2 }$ . The vector is seeded by the logical parameter identity, successful update index, and probe purpose, using base seed 1234. A degenerate normalization triggers one retry with a diferent seeded probe. A zero matrix supplies no positive observation.

## B State and Communication

Table 5 lists the controller settings.

## B.1 Parameter Selection

Selection uses each complete logical parameter before optimizer sharding is restored. The spectral branch selects two-dimensional logical parameters in the dense optimizer group. It excludes parameters whose lowercase names contain an embedding/output exclusion marker. Those markers include embedding, lm\_head, and output\_layer; position embeddings are also excluded. The expert-matrix and embedding-matrix inclusion switches are both false. MoE projections and routers follow the same selection rule. Routing tensors assigned to the expert optimizer follow the norm branch even when their logical shape is twodimensional. All other shapes also use the norm branch.

## B.2 Calibration and State Updates

The first 40 successful global updates apply GlobalGC with norm cap 1. Each logical tensor separately counts its successful participating updates. Spectral observations begin when a valid previous second moment is available. The first positive observation initializes h directly. Subsequent calibration observations use the ordinary EMA. The spectral branch becomes active once its history and previous second moment are ready and calibration is complete. A late-participating tensor uses norm cap 1 until ready.

For a tensor $T$ outside the spectral branch, the active transformation is

$$
T _ { \mathrm { o u t } } = T \operatorname* { m i n } \{ 1 , \lambda f / \| T \| _ { \mathrm { F } } \} ,\tag{12}
$$

with zero output for zero input. Positive output norms update f with the same initialization and decay rules as h. The histories have floor $1 0 ^ { - 1 2 }$ There is no additional global cap after the active branches.

If an active tensor returns after a gap of $\Delta$ successful global updates, its history uses decay $\beta _ { h } ^ { \Delta }$ A missing gradient leaves state unchanged. A genuine zero gradient can advance participation without providing a positive history observation. Moment update counts, participation counts, and observation indices remain distinct. The optimizer stages transformed gradients and candidate controller states, then calls AdamW once. It commits the controller states only when that update succeeds. Checkpoints preserve these histories, counts, factored moment summaries, and a configuration fingerprint.

<table><tr><td>Controller setting</td><td>Value</td></tr><tr><td>Selected spectral space</td><td>Full matrix</td></tr><tr><td>Newton-Schulz iterations</td><td>5</td></tr><tr><td>Output power iterations</td><td>3</td></tr><tr><td>Multiplier λ</td><td>1.3</td></tr><tr><td>History decay  $\beta _ { h }$ </td><td>0.95</td></tr><tr><td>Calibration updates / cap</td><td>40 /1</td></tr><tr><td>Scale floor  $\rho$ </td><td>0.01</td></tr><tr><td>Scale epsilon  $\epsilon _ { s }$ </td><td>10-8</td></tr><tr><td>History floors</td><td> $1 0 ^ { - 1 2 }$ </td></tr><tr><td>Probe seed</td><td>1234</td></tr><tr><td>Additional global clipping</td><td>None</td></tr></table>

Table 5: Main-method controller settings. The training seed is separately fixed at 48.

Gradient fragments are packed into buckets and all-gathered within the relevant data-parallel or expert-data-parallel group. Each rank reconstructs and transforms the same complete logical matrix, then writes its local optimizer slice. Full-matrix temporaries and communication therefore remain part of the implementation cost. The norm branch obtains its magnitudes through scalar reductions.

The row summary of the uncorrected moment obeys

$$
\bar { r } _ { i , t } = \beta _ { 2 } \bar { r } _ { i , t - 1 } + ( 1 - \beta _ { 2 } ) \frac { 1 } { n } \sum _ { j } ( G _ { \mathrm { o u t } , t } ) _ { i j } ^ { 2 } .\tag{13}
$$

Column and global summaries follow the same update rule. Linearity makes these the corresponding means of AdamW’s second moment in exact arithmetic. Bias correction recovers the scale inputs in Equation 5. The optimizer still stores and uses its full moments. The summaries support clipping and avoid repeatedly gathering the full previous second moment.

## C Model and Baselines

## C.1 Parameter Counts

Each of the 10 shortcut blocks contains two MLA attention modules, two dense SwiGLU modules, and one MoE module. The hidden width is 1536 and the dense feed-forward width is 3072. Each MLA has 16 heads, query low-rank width 384, and key/value low-rank width 128. Its non-RoPE query/key width is 128 per head, RoPE width is 64, and value width is 128. These dimensions give 5,734,912 parameters per MLA, including its lowrank normalization parameters.

<table><tr><td>Parameter component</td><td>Count across the model</td></tr><tr><td>20 MLA modules, including low-rank normalization</td><td>114,698,240</td></tr><tr><td>20 dense SwiGLU modules</td><td>283,115,520</td></tr><tr><td>Learned experts</td><td>18,119,393,280</td></tr><tr><td>MoE input/output projections Routers</td><td>47,185,920</td></tr><tr><td>Block and MoE normalization</td><td>13,762,560</td></tr><tr><td>Input embedding and output projection</td><td>76,800</td></tr><tr><td>Final normalization</td><td>503,316,480</td></tr><tr><td></td><td>1,536</td></tr><tr><td>Total</td><td>19,081,550,336</td></tr><tr><td>Target activated backbone, excluding embedding/output</td><td>647,584,256</td></tr></table>

Table 6: Parameter counts from the recorded architecture and model source. Replicated parameters are counted once. The activated backbone uses the target of 32 learned experts per MoE module; the actual count varies across tokens. Selecting 48 learned experts per MoE module gives an upper bound of approximately 0.742B activated backbone parameters.

![](images/f4b374df83482c49bbce4263d2275bc637ca67e70cada4e630878588a0b8d930.jpg)  
Figure 8: Every baseline-minus-CAST validation loss diference. Positive values favor CAST; outlined negative cells favor the baseline. Subsets correspond to Table 7.

Each MoE module contains 3,072 learned experts and 512 identity experts without feedforward weights. Learned experts operate on 384- dimensional inputs with feed-forward width 512. Routing selects 48 candidates in total and targets 32 learned-expert activations per token; the actual learned-expert count varies with routing. The input embedding and output projection are untied, with vocabulary size 163840. Table 6 summarizes the parameter counts.

## C.2 Baseline Settings

All transformations precede both AdamW moment updates. GlobalGC uses a global norm cap of 1. AdaGC uses per-tensor multiplier 1.04, EMA decay 0.99, and 100 calibration steps. These values come from AdaGC’s run configuration.

SSC-constant applies Equation 2 to the selected dense matrices in raw coordinates, with threshold 0.1 and five Newton–Schulz iterations. It starts immediately and passes all unselected tensors through unchanged. For square gradients, it forms the Gram matrix from the untransposed input. SSC-cosine replaces the fixed threshold with

![](images/8a653a5d8e684c66f9245840c1660a09e3b8b4aed88f147bc9caf375577225ec.jpg)  
Figure 9: Training from the first complete 100-step averaging window through iteration 5000. All ten primary conditions and three CAST ablations (HS: historical scaling) use trailing 100-step means. The main plot uses a logarithmic vertical axis; the box and connected inset show the final 500 steps on a linear scale.

$$
\begin{array} { l } { \tau _ { k } = 0 . 1 + 4 . 9 5 \big [ 1 + \cos ( \pi u _ { k } ) \big ] , } \\ { u _ { k } = \operatorname* { m i n } \{ ( k - 1 ) / 4 9 9 9 , 1 \} . } \end{array}\tag{14}
$$

Here k is the one-based training iteration, including skipped updates. The threshold starts at 10 and reaches 0.1 at iteration 5000. It stays at 0.1 afterward.

GradientStabilizer uses global norm and squared-norm EMA decays 0.6 and 0.999, without bias correction or an extra global cap. GradPower applies GlobalGC(1), then the signed elementwise power with exponent 1.2. ClipByValue clamps each coordinate to [−0.1, 0.1] without another clipping step. AGC uses multiplier 0.01, weight-norm floor $1 0 ^ { - 3 }$ , and gradient-norm floor $1 0 ^ { - 6 }$ , including embedding and output tensors. ZClip uses decay 0.97, z-score threshold 2.5, 25 calibration steps, and epsilon 10−<sup>6</sup>. It applies an additional norm cap of 1 without skipping spike steps.

The three component variants preserve CAST’s calibration and output-norm controller for unselected tensors. The variant without HS removes the clipping scale S while retaining AdamW’s second moment. The fixed-threshold variant sets the selected spectral threshold to 0.1 while retaining HS. The cosine-threshold variant uses Equation 14 in the same standardized coordinates. Its schedule follows the training iteration throughout calibration and checkpoint resumes.

## D Validation Results

Table 7 includes every baseline and validation label. Question–answer and benchmark-formatted rows measure the configured token loss for those texts. Figure 8 shows the loss diferences for each baseline and validation subset.

## E Full Training Curves

The full training overview is shown in Figure 9. The main-text Figure 6 and Table 3 contain the input-gradient traces and statistics for these conditions.

<table><tr><td>Label</td><td>CAST</td><td>GlobalGC</td><td>AdaGC</td><td>SSC-constant</td><td>SSC-cosine</td></tr><tr><td>01 Academic / vocational</td><td>2.2466</td><td>2.2718</td><td>2.2569</td><td>2.2601</td><td>2.2693</td></tr><tr><td>02 Clinical / health</td><td>1.9951</td><td>2.0179</td><td>2.0054</td><td>2.0078</td><td>2.0168</td></tr><tr><td>03 Consumer / leisure</td><td>2.1824</td><td>2.2031</td><td>2.1924</td><td>2.1935</td><td>2.2026</td></tr><tr><td>04 Economy / services</td><td>2.0175</td><td>2.0434</td><td>2.0310</td><td>2.0309</td><td>2.0399</td></tr><tr><td>05 IT / programming</td><td>1.8519</td><td>1.8727</td><td>1.8631</td><td>1.8646</td><td>1.8713</td></tr><tr><td>06 Periodicals</td><td>2.7619</td><td>2.7855</td><td>2.7760</td><td>2.7744</td><td>2.7822</td></tr><tr><td>07 Regional culture</td><td>2.1047</td><td>2.1263</td><td>2.1155</td><td>2.1164</td><td>2.1243</td></tr><tr><td>08 Science / technology</td><td>1.8652</td><td>1.8862</td><td>1.8752</td><td>1.8782</td><td>1.8852</td></tr><tr><td>09 Sports / vehicles</td><td>2.2699</td><td>2.2944</td><td>2.2844</td><td>2.2838</td><td>2.2921</td></tr><tr><td>10 CMMLU text</td><td>1.4037</td><td>1.4355</td><td>1.4218</td><td>1.4363</td><td>1.4444</td></tr><tr><td>11 Open-source code A</td><td>0.9368</td><td>0.9520</td><td>0.9450</td><td>0.9433</td><td>0.9495</td></tr><tr><td>12 Open-source code B</td><td>0.9367</td><td>0.9518</td><td>0.9449</td><td>0.9432</td><td>0.9493</td></tr><tr><td>13 English books</td><td>2.9816</td><td>3.0043</td><td>2.9952</td><td>2.9944</td><td>2.9987</td></tr><tr><td>14 FullStackBench text</td><td>1.1398</td><td>1.1726</td><td>1.1725</td><td>1.1415</td><td>1.1682</td></tr><tr><td>15 GSM8K text</td><td>0.7507</td><td>0.7656</td><td>0.7526</td><td>0.7604</td><td>0.7587</td></tr><tr><td>16 HumanEval text</td><td>1.1460</td><td>1.1416</td><td>1.1612</td><td>1.1466</td><td>1.1357</td></tr><tr><td>17 MMLU text</td><td>1.4045</td><td>1.4435</td><td>1.4355</td><td>1.4263</td><td>1.4555</td></tr><tr><td>18 English web</td><td>2.7558</td><td>2.7810</td><td>2.7697</td><td>2.7709</td><td>2.7797</td></tr><tr><td>19 Chinese essays</td><td>3.9568</td><td>3.9867</td><td>3.9759</td><td>3.9789</td><td>3.9880</td></tr><tr><td>20 Software engineering</td><td>0.7679</td><td>0.7873</td><td>0.7703</td><td>0.7767</td><td>0.7831</td></tr></table>

<table><tr><td>Label</td><td>GS</td><td>GradPower</td><td>Value</td><td>AGC</td><td>ZClip</td></tr><tr><td>01 Academic / vocational</td><td>2.2849</td><td>2.2943</td><td>2.2801</td><td>2.2603</td><td>2.2716</td></tr><tr><td>02 Clinical / health</td><td>2.0304</td><td>2.0401</td><td>2.0255</td><td>2.0082</td><td>2.0181</td></tr><tr><td>03 Consumer / leisure</td><td>2.2195</td><td>2.2273</td><td>2.2124</td><td>2.1938</td><td>2.2050</td></tr><tr><td>04 Economy / services</td><td>2.0575</td><td>2.0645</td><td>2.0512</td><td>2.0329</td><td>2.0422</td></tr><tr><td>05 IT / programming</td><td>1.8864</td><td>1.8941</td><td>1.8807</td><td>1.8634</td><td>1.8758</td></tr><tr><td>06 Periodicals</td><td>2.7960</td><td>2.8002</td><td>2.7914</td><td>2.7766</td><td>2.7865</td></tr><tr><td>07 Regional culture</td><td>2.1391</td><td>2.1479</td><td>2.1330</td><td>2.1168</td><td>2.1267</td></tr><tr><td>08 Science / technology</td><td>1.8992</td><td>1.9072</td><td>1.8944</td><td>1.8773</td><td>1.8881</td></tr><tr><td>09 Sports / vehicles</td><td>2.3096</td><td>2.3162</td><td>2.3025</td><td>2.2840</td><td>2.2947</td></tr><tr><td>10 CMMLU text</td><td>1.4010</td><td>1.4414</td><td>1.4344</td><td>1.4142</td><td>1.4178</td></tr><tr><td>11 Open-source code A</td><td>0.9586</td><td>0.9604</td><td>0.9525</td><td>0.9427</td><td>0.9534</td></tr><tr><td>12 Open-source code B</td><td>0.9585</td><td>0.9603</td><td>0.9524</td><td>0.9426</td><td>0.9532</td></tr><tr><td>13 English books</td><td>3.0211</td><td>3.0242</td><td>3.0098</td><td>2.9957</td><td>3.0059</td></tr><tr><td>14 FullStackBench text</td><td>1.1778</td><td>1.1702</td><td>1.1694</td><td>1.1526</td><td>1.1626</td></tr><tr><td>15 GSM8K text</td><td>0.7760</td><td>0.7770</td><td>0.7680</td><td>0.7510</td><td>0.7664</td></tr><tr><td>16 HumanEval text</td><td>1.1651</td><td>1.1835</td><td>1.1718</td><td>1.1351</td><td>1.1394</td></tr><tr><td>17 MMLU text</td><td>1.3934</td><td>1.4597</td><td>1.4359</td><td>1.4423</td><td>1.4108</td></tr><tr><td>18 English web</td><td>2.7981</td><td>2.8058</td><td>2.7888</td><td>2.7716</td><td>2.7816</td></tr><tr><td>19 Chinese essays</td><td>4.0075</td><td>4.0102</td><td>3.9984</td><td>3.9766</td><td>3.9908</td></tr><tr><td>20 Software engineering</td><td>0.7882</td><td>0.7939</td><td>0.7918</td><td>0.7737</td><td>0.7830</td></tr></table>

Table 7: Token LM losses on all 20 validation subsets at iteration 5000. Bold and underline mark the best and second-best values across all ten conditions, spanning both panels. GS is GradientStabilizer; Value is ClipByValue.