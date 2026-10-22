# Local and Global Recursive Ridge Regression for Online Residual Correction of Frozen Multivariate Forecasters under Label Delay

Takumi Fujimoto<sup>1</sup> Hiroaki Nishi<sup>2</sup>

<sup>1</sup>School of Science for Open and Environmental Systems,

Graduate School of Science and Technology, Keio University, Japan

takmin@keio.jp

<sup>2</sup>Department of System Design, Faculty of Science and Technology,

Keio University, Japan

west@keio.jp

## Abstract

A frozen multi-horizon forecaster can be corrected online from its own residuals. However, under label delay the target of a block arrives only after the whole horizon has passed, and the corrector’s retained state grows with the number of channels. Aiming to combine a small retained state with a large error reduction, we propose DRiC (DCT Ridge Corrector), which fits a ridge regression of the leading discrete cosine transform (DCT) coeficients of the next residual block on those of the frozen forecast and of the latest input block, with suficient statistics updated recursively. Its local variant, DRiC-L, keeps one such regression per channel; its global variant, DRiC-G, pools all channels into one regression, whose regression statistics do not depend on the channel count. A guard tracks a running scale of the frozen forecast’s residual for each channel; it skips updates on blocks far outside that scale and clips the issued correction. We evaluate on seven widely used datasets, ETT (ETTh1, ETTh2, ETTm1, ETTm2), Weather (with its missing-value codes filled), Electricity, and Trafic, with DLinear, PatchTST, and iTransformer backbones, against 10 naive, gradient-trained, and closed-form methods. DRiC-L with eight DCT coeficients (DRiC-L8) reduces the mean squared error (MSE) by 14.55% on average over 126 evaluation conditions, against 12.98% for ELF, the most accurate comparator, while retaining 15–17 times less state, although ELF is more accurate on Trafic. DRiC-G retains 14.6 kB on the 321-channel Electricity data and 18.9 kB on the 862-channel Trafic data and is the most accurate method at or below its state on these two datasets. However, 5 to 7 of the 10 comparators, all with at least 10 times its state, are more accurate there. On the Weather file with its missing-value codes, DRiC-L8 without the guard has on average 8.0 times the frozen forecast’s MSE, whereas with the guard it reduces the MSE by 7.94%. Code and numerical records are available at https://github.com/keiotakmin/dric.

Keywords: Online regression, recursive least squares, residual correction, test-time adaptation, time series forecasting.

## 1 Introduction

A deployed forecaster is often frozen: retraining it is expensive, its weights may be shared by many sites, or it is a pretrained model that the operator cannot modify. Its errors, however, are observed as the stream continues, and test-time adaptation methods for time series forecasting use them to correct the frozen forecast, either by small gradient-trained adapters around the model [1–4] or by closed-form regressors on its output [5–7]. A recent audit showed that the order of information matters: the target of a multi-horizon forecast is complete only H steps after the forecast is issued, and methods that update on partially observed targets gain from information a deployed system would not yet have [6]. Under this strict delay, closed-form correctors are strong reference methods [6].

The cost of a corrector is not only its computation but also the state it must keep between forecasts. Closed-form correctors keep suficient statistics or stored samples per channel, and often per horizon step. For the 862-channel Trafic data, ELF [5] retains tens of megabytes and the RLS-Bank of the audit [6] about ten megabytes in our implementation. When a corrector runs next to the forecaster on a gateway or embedded controller, or when one server corrects many streams, this state limits where the correction can be deployed.

This paper aims to combine a small retained state with a large error reduction, that is, to match or exceed the accuracy of the most accurate comparator with an order of magnitude less retained state. We propose DRiC (DCT Ridge Corrector), a closed-form corrector based on the discrete cosine transform (DCT), with a local and a global variant that share a guard. DRiC-L targets this aim; DRiC-G trades accuracy for statistics that do not grow with the channel count. The design rests on three decisions, each of which we test in Section 5.2:

1. Low-dimensional input and output. The corrector predicts only the leading K DCT coeficients of the next residual block, from the leading K coeficients of the frozen forecast and of the latest input block of H samples. Its per-channel state is a $( 2 K + 2 ) \times ( 2 K + 2 )$ Gram matrix and its right-hand side, whose size depends on neither the lookback, the horizon, nor the length of the stream.

2. A guard referred to the frozen forecast’s residual. A recursive estimator cannot reweight past samples after they have entered its statistics, so it must decide at update time whether a block is admissible. The guard skips the update when a block’s residual power is far above a running scale of that channel’s past residuals, and bounds the issued correction by the same scale. Because the reference is the frozen forecast’s residual rather than the corrector’s own error, the decision does not depend on what the corrector has learned.

3. Local and global suficient statistics. The local variant, DRiC-L, keeps one regression per channel; the global variant, DRiC-G, pools all channels into one Gram matrix, following the globalmodel view of forecasting groups of series [8]. The local statistics grow with the channel count; the global statistics do not, and only the guard’s scale, one value per channel, grows with it.

We evaluate DRiC under the strict delay on seven widely used datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather with its missing-value codes filled (Weather\*), Electricity, and Trafic) with three trained backbones (DLinear [9], PatchTST [10], and iTransformer [11]), against naive forecasts, gradient trained adapters (TAFAS, PETSA, COSA, FAC, ORCA), and closed-form correctors (RLS-Bank, ELF, EPOC). DRiC-L with K = 8 (DRiC-L8) reduces the mean squared error (MSE) by 14.55% on average over 126 conditions, against 12.98% for ELF, the most accurate comparator, with 15–17 times less retained state; ELF has the higher mean reduction only on Trafic. DRiC-G retains 18.9 kB on Trafic and is the most accurate method at or below its state on Electricity and Trafic. However, 5 to 7 of the 10 comparators, all with at least 10 times its state, are more accurate there. Section 2 positions the correctors among existing methods, Section 3 defines them, Sections 4 and 5 describe the experiments and results, and Section 6 discusses their limits.

## 2 Related Work

Test-time adaptation of frozen forecasters. TAFAS [1] and PETSA [2] place gated calibration modules before and after a frozen forecaster and update them by gradient steps on partially observed targets; COSA [3] adapts a residual adapter on the output; FAC [4] calibrates input and output in the frequency domain; DynaTTA [12] makes the adaptation shift-aware. ORCA [13] trains a small residual network on the input context and the frozen forecast from a replay bufer. These methods keep optimizer state and, for ORCA, a bufer of past windows. We compare with all of them except DynaTTA under a common protocol in which every method updates only after a complete target block.

Correctors fitted before deployment. Other recent methods correct a frozen forecaster with a module that is fitted before deployment on training or validation data and then fixed. Post-training corrections [14] select, by a bandit search on validation error, a sequence of simple transformations of the forecast, such as scaling and trend adjustments. UEC-STD [15] trains a network on the validation split to correct the errors that accumulate when a forecaster’s outputs are fed back as inputs, after decomposing the forecast into trend and seasonal parts. CRC [16] combines a ridge regression with a nonlinear refinement and decides on validation data, for each variable and horizon step, whether to deploy the correction, so as not to degrade the frozen forecast where the residual has little structure. These correctors use no labels after deployment. DRiC is updated from the delayed labels of the stream, and its guard decides during deployment, for each block and channel, whether the block enters the statistics.

Closed-form correctors. ELF [5] fits a linear forecaster on the Fourier coeficients of the whole lookback by recursive ridge regression and combines it with the frozen forecast; its statistics grow with the square of the retained frequency count for every channel. The delay-aware audit [6] proposes an RLS-Bank with one recursive least-squares estimator per channel, horizon step, and forgetting factor on four time-domain features, aggregated by a median. EPOC [7] regresses each retained DCT coeficient of the next residual on the previous residual block’s coeficients, its final value, and the current forecast’s coeficient. Two concurrent preprints are close to the present correctors. ECHO’s recent-error expert [17] maps the spectra of the whole context and of the forecast to the residual spectrum by a Huber-weighted ridge regression that is refitted on a growing set of stored records. EFC [18] combines a per-variable ridge regression on the diagonal of verified past errors, accumulated without forgetting, with a recurrent corrector trained with a Huber loss, and uses the errors of partially elapsed horizons as they arrive. The inputs of DRiC-L are close to a subset of those of ECHO’s expert: the latest of the four input blocks instead of the whole context (the variant with all four blocks in Table 3 is the nearest to ECHO’s choice). DRiC-L difers from both preprints in its estimator, fixed-size recursive statistics instead of stored records or a trained network, and in its guard, which screens blocks before they enter these statistics; it also uses only complete target blocks.

Robust recursive estimation. Thresholding updates by a running estimate of the error scale is standard in robust adaptive filtering; the recursive least M-estimate algorithm [19] down-weights samples whose a priori error exceeds a threshold derived from a robust variance estimate. The guard applies a threshold of this kind to the frozen forecast’s residual rather than to the estimator’s own error, uses it both to skip updates and to bound the output, and keeps one scale per channel.

Local and global models. A model fitted to all series of a group (global) can match or exceed separate per-series (local) models [8]. Regressing observations on the output of a numerical model is also the basis of model output statistics in weather forecasting [20]. DRiC-G brings this choice to online residual correction: its regression statistics are shared by all channels and do not grow with the channel count.

## 3 DRiC: DCT Ridge Corrector

## 3.1 Setting and feedback order

Forecast origins advance by H samples and index consecutive blocks $t = 1 , 2 , \ldots \mathrm { A t }$ the origin of block t, a frozen forecaster maps the input window $w _ { t } \in \mathbb { R } ^ { L \times C }$ to the forecast $f _ { t } \in \mathbb { R } ^ { H \times C }$ for C channels. A corrector issues $\hat { y } _ { t } = f _ { t } + \alpha _ { t } g _ { t }$ , where $g _ { t }$ is its correction and $\alpha _ { t } \in [ 0 , 1 ]$ a blending weight. The target y<sub>t</sub> is complete only after the block has passed; then the base residual

$$
e _ { t } = y _ { t } - f _ { t }\tag{1}
$$

Issue block t  
![](images/3cfb81cdd2576d8f3191177535745d5c0c2d0852bb4868e2daa3d926bc23123d.jpg)

After the target of block t is complete (H samples later)  
![](images/6f87f2e87be8fa56cc74421a100a876ca658b797702eb553bde9287bffaec065.jpg)  
Figure 1: One block of DRiC-L8 on ETTh1 (PatchTST, refit, seed 0): the block and channel whose MSE reduction is the median over all blocks and channels of this condition. Top: at the origin of block t, the corrector maps the leading DCT coeficients of the newest input block and of the frozen forecast through the ridge map, clips the predicted residual coeficients to the guard scale, and issues the corrected forecast. Bottom: once the target is complete, the residual’s coeficients update the statistics if the guard admits the block. Curves, coeficient bars, the clipping bound, and the guard ratio are those of this block; in this typical block neither the clipping nor the guard is active. The clipping bound and the guard use the same running scale s of the channel. Dashed arrows: the update, which is used from the next issue on. The heatmaps are schematic.

is known and the corrector updates its state. Every quantity used for block t is therefore computed from blocks $1 , \ldots , t - 1$ (label delay $d = H \ [ 6 ] )$ . The residual is measured against the frozen forecast, so it does not depend on the correction issued. We use $L = 9 6$ and $H = 2 4$ , so the input window consists of $M = L / H = 4$ sub-blocks of H samples. Figure 1 summarizes the information flow.

## 3.2 Residual representation

Let $U _ { J } \in \mathbb { R } ^ { H \times J }$ hold the first J columns of the orthonormal DCT-II basis [21],

$$
( U _ { J } ) _ { h , j } = \left\{ \begin{array} { l l } { H ^ { - 1 / 2 } , } & { j = 1 , } \\ { \sqrt { 2 / H } \cos \left[ \pi ( h - \frac { 1 } { 2 } ) ( j - 1 ) / H \right] , } & { j > 1 , } \end{array} \right.\tag{2}
$$

and let $U _ { K }$ be its first $K \leq J$ columns. We use $J = 8$ . For channel c, the corrector reads the coeficients of the forecast and of the newest input sub-block $w _ { t , c } ^ { ( 1 ) } \in \mathbb { R } ^ { H }$ , and the last input value $\ell _ { t , c } \colon$

$$
\begin{array} { r } { a _ { t , c } = U _ { K } ^ { \top } f _ { t , c } , \qquad x _ { t , c } = U _ { K } ^ { \top } w _ { t , c } ^ { ( 1 ) } , \qquad \phi _ { t , c } = \left[ 1 , \ a _ { t , c } ^ { \top } , \ x _ { t , c } ^ { \top } , \ \sqrt { H } \ell _ { t , c } \right] ^ { \top } \in \mathbb { R } ^ { 2 K + 2 } . } \end{array}\tag{3}
$$

The factor $\sqrt { H }$ puts a constant level on the scale of the first DCT coeficient. The regression target is $\boldsymbol { z } _ { t , c } = \boldsymbol { U } _ { K } ^ { \top } \boldsymbol { e } _ { t , c } \in \mathbb { R } ^ { K }$ , and a predicted coeficient vector $\hat { z } _ { t , c }$ is mapped back to the horizon by $U _ { K } \hat { z } _ { t , c }$

## 3.3 Recursive ridge regression

Each regression is an exponentially weighted ridge regression with forgetting factor $\rho = 2 ^ { - 1 / n _ { 1 / 2 } }$ for a half-life of $n _ { 1 / 2 }$ blocks and a penalty λ that also applies to the intercept [22, 23]. It keeps the upper

![](images/acee25b0408dc0de56eb9f1e7c2e71ec12a6d58a80a887f1a458354d28d08ed5.jpg)  
Figure 2: Statistics of the two variants. DRiC-L keeps one $( 2 K + 2 ) \times ( 2 K + 2 )$ Gram matrix and right-hand side per channel $^ { c , }$ fitted on that channel’s blocks only, with the K leading coeficients of the forecast and of the newest input block and the last input value in $\phi _ { t , c }$ . DRiC-G pools the blocks of all C channels into one $q _ { \mathrm { g } } \times q _ { \mathrm { g } }$ Gram matrix and right-hand side $\left( q _ { \mathrm { g } } = 4 2 \right)$ , with $J = 8$ coeficients of the forecast and of all four input blocks and the last input value in $\phi _ { t , c } ^ { \mathrm { g } } .$ Sizes are in double precision; both variants also keep one guard scale per channel (8 bytes). The heatmaps are schematic.

triangle of a Gram matrix G and a right-hand side $R ,$ initialized to λI and 0, and predicts $\hat { z } = \hat { B } ^ { \top } \phi$ with ${ \hat { B } } = G ^ { - 1 } R$ . After a complete block, an admitted sample $\bigl ( \phi , z \bigr )$ updates

$$
G \gets \rho G + \phi \phi ^ { \mathsf { T } } + ( 1 - \rho ) \lambda I , \qquad R \gets \rho R + \phi z ^ { \mathsf { T } } ,\tag{4}
$$

which keeps the penalty at λ and minimizes $\begin{array} { r } { \lambda \| B \| _ { F } ^ { 2 } + \sum _ { s } \rho ^ { n - s } \| z _ { s } - B ^ { \mathsf { T } } \phi _ { s } \| ^ { 2 } } \end{array}$ over the n admitted samples.

Local variant (DRiC-L). Each channel has its own $( G _ { c } , R _ { c } )$ with $\phi _ { t , c }$ of (3). A channel whose block is not admitted by the guard (Section 3.4) leaves its statistics unchanged, including the decay.

Global variant (DRiC-G). One (G, R) is shared by all channels. Its feature vector uses J coeficients and all M input sub-blocks,

$$
\boldsymbol { \phi } _ { t , c } ^ { \mathrm { g } } = \left[ 1 , \ ( \boldsymbol { U } _ { J } ^ { \top } { f } _ { t , c } ) ^ { \top } , \ ( \boldsymbol { U } _ { J } ^ { \top } \boldsymbol { w } _ { t , c } ^ { ( 1 ) } ) ^ { \top } , \ldots , ( \boldsymbol { U } _ { J } ^ { \top } \boldsymbol { w } _ { t , c } ^ { ( M ) } ) ^ { \top } , \ { \sqrt { H } } \ell _ { t , c } \right] ^ { \top } \in \mathbb { R } ^ { 2 + J + M J } ,\tag{5}
$$

and its target is $U _ { J } ^ { \mathsf { T } } e _ { t , c }$ . After a block in which at least one channel is admitted, the statistics decay once and add the admitted channels’ samples; after a block in which no channel is admitted, they are unchanged. Because the data are standardized per channel (Section 4), the shared map acts on comparable scales. Figure 2 contrasts the two correctors.

## 3.4 Guard

For each channel the guard keeps a running scale $s _ { c }$ of the frozen forecast’s residual power over the J leading coeficients, $p _ { t , c } = \| U _ { J } ^ { \mathsf { T } } e _ { t , c } \| ^ { 2 } / J$ . The first block sets $s _ { c } = p _ { 1 , c } .$ . Afterwards a block is admitted if $p _ { t , c } \leq \tau ^ { 2 } s _ { c } ;$ then $s _ { c } \gets \rho _ { s } s _ { c } + ( 1 - \rho _ { s } ) p _ { t , c }$ , and the regressions are updated. The scale has its own forgetting factor $\rho _ { s } = 2 ^ { - 1 / 1 2 8 }$ , which we keep fixed when the regressions’ half-life $n _ { 1 / 2 }$ is varied. A block that is not admitted changes neither the scale nor the regressions. Before issuing, the coeficient vector $\hat { z } _ { t , c } ,$ padded with zeros to J entries, is scaled down to norm at most $\tau \sqrt { J s _ { c } }$ . The guard depends only on observed residuals of the frozen forecast, so its decisions are the same whatever the corrector has learned. Figure 3 shows its decisions around the missing-value codes of the standard Weather file: it skips the blocks whose target contains the codes and most of the later blocks whose frozen forecast is distorted because the codes have entered the input window. One such block falls just below the threshold and enters the statistics.

![](images/d98bd8e95bbc093d97b76085c2def60ac6ae044d272cc24495557e96b8f506cd.jpg)

Figure 3: Issue and update events for the OT column of the standard Weather file around its missingvalue codes, with the frozen DLinear forecast (refit, seed 0) and the guard decisions of DRiC-L8 for this channel. Labels name where the codes are; the admitted block with codes in its input window lies just below the threshold. The forecast of a block is issued at its origin; its update happens one horizon later. Arrows mark forecasts that leave the plotted range.  
Algorithm 1 One block of DRiC-L or DRiC-G   
1: Issue: form $\phi _ { t , c }$ (local) or $\phi _ { t , c } ^ { \mathrm { g } }$ (global) from $f _ { t }$ and $w _ { t } ;$ solve $\hat { B } = G ^ { - 1 } R ; \hat { z } _ { t , c } = \hat { B } ^ { \mathsf { T } } \phi _ { t , c }$   
2: clip $\hat { z } _ { t , c }$ to norm $\tau \sqrt { J s _ { c } } ; g _ { t } = U _ { J } \hat { z } _ { t } ;$ issue $\hat { y } _ { t } = f _ { t } + \alpha _ { t } g _ { t }$   
3: After $y _ { t }$ is complete: $e _ { t } = y _ { t } - f _ { t } ;$ update the blending sums with $( g _ { t } , e _ { t } )$   
4: $p _ { t , c } = \| \boldsymbol { U } _ { J } ^ { \mathsf { T } } \boldsymbol { e } _ { t , c } \| ^ { 2 } / J ;$ admitted channels: $p _ { t , c } \leq \tau ^ { 2 } s _ { c }$ (all channels at $t = 1 )$   
5: for admitted channels: update $s _ { c }$ and the regression statistics by (4) (global: one decay if any channel   
is admitted)   
3.5 Blending and retained state   
Following EPOC [7], the issued correction is $g _ { t } = U _ { J } ;$ zˆ scaled by one weight shared by all channels,   
$\alpha _ { t } = \mathrm { c l i p } _ { [ 0 , 1 ] } \frac { \sum _ { s = t - 3 2 } ^ { t - 1 } \langle g _ { s } , e _ { s } \rangle } { \sum _ { s = t - 3 2 } ^ { t - 1 } \lVert g _ { s } \rVert ^ { 2 } } ,$ (6)   
with $\alpha _ { t } = 0$ while the denominator is zero. The same rule is applied to every comparator.   
The retained state is the set of arrays kept between forecasts, excluding the frozen forecaster and the   
shared input window. With $q = 2 K + 2$ , DRiC-L keeps $C \left[ q ( q + 1 ) / 2 + q K \right]$ values; the global regression   
keeps $q _ { \mathrm { g } } ( q _ { \mathrm { g } } + 1 ) / 2 + q _ { \mathrm { g } } J$ values with $q _ { \mathrm { g } } = 2 + J + M J$ , independent of C. Both correctors keep the   
basis $U _ { J }$ , the component indices, the C guard scales, and 66 values for the blending weight. In double   
precision, a local channel takes 760 bytes for $K = 4$ and 2,520 bytes for $K = 8 ,$ , and the global regression   
9,912 bytes; with the guard scales, 8 bytes per channel, DRiC-G retains 12.1 kB on ETT, 14.6 kB on   
Electricity, and 18.9 kB on Trafic. Algorithm 1 lists one block.

Table 1: Evaluated datasets. Evaluation blocks are the complete 24-sample targets scored per condition.
<table><tr><td>Dataset</td><td>Channels</td><td>Length</td><td>Sampling</td><td>Evaluation blocks</td></tr><tr><td>ETTh1</td><td>7</td><td>17,420</td><td>1 h</td><td>362</td></tr><tr><td>ETTh2</td><td>7</td><td>17,420</td><td>1 h</td><td>362</td></tr><tr><td>ETTm1</td><td>7</td><td>69,680</td><td>15 min</td><td>1,451</td></tr><tr><td>ETTm2</td><td>7</td><td>69,680</td><td>15 min</td><td>1,451</td></tr><tr><td>Weather</td><td>21</td><td>52,696</td><td>10 min</td><td>1,097</td></tr><tr><td>Electricity</td><td>321</td><td>26,304</td><td>1 h</td><td>548</td></tr><tr><td>Traffic</td><td>862</td><td>17,544</td><td>1 h</td><td>365</td></tr></table>

## 4 Experimental Setup

## 4.1 Data and protocol

We use the four ETT series [24] and the Weather, Electricity, and Trafic files distributed with Autoformer [25] (Table 1); all numerical columns are forecast. For every series, $L = 9 6 , H = 2 4$ , and forecast origins are H samples apart. With series length T, $n = \lfloor T / 2 \rfloor$ , and $a = \lfloor 0 . 8 n \rfloor$ , a legacy base is trained on [0, a), [a, n) serves for selection, and a refit base is retrained on [0, n) with the selected settings. Evaluation origins are $n , n + H , . . . ,$ , and only complete target blocks are scored. Each channel is standardized with the mean and standard deviation of [0, n), and errors are reported on this scale. This split difers from the common 70/10/20 split, so the frozen forecasts’ errors (Table 6) are not comparable with published values.

The standard Weather file contains 81 cells with the missing-value code −9999: 1 in the column wv, 30 in max. PAR, and 50 in OT. 45 of them lie in the range [0, n) used for standardization (1, 10, and 34 in these columns) and inflate the standard deviations used for standardization; the codes therefore become about −162, −18, and −28 after standardization. The other 36 fall in evaluation targets (20 in max. PAR and 16 in OT). We report the standard file separately (18 conditions) and use, in the main view, a copy in which these cells are forward-filled, with the bases retrained on it (Weather\*). We omit the Exchange data: the compact PatchTST base, which has no instance normalization, fails on its evaluation period (its MSE is 13 times that of DLinear), so a correction there mostly repairs a failed base.

## 4.2 Frozen forecasters

We use DLinear [9], a compact channel-independent PatchTST [10] (patch length 16, stride 8, width 64, four heads, two layers, feed-forward width 128) without instance normalization, and iTransformer [11] of the same size with its per-window normalization. Each is trained with Adam at $1 0 ^ { - 3 }$ , batch size 32, and the number of updates chosen from {200, 500, 1000, 2000, 4000, 8000, 20000} by validation MSE, for seeds 0, 1, and 2. The legacy and refit bases of the seven datasets give $7 \times 3 \times 3 \times 2 = 1 2 6$ conditions. Every method receives the same forecasts, targets, and origins in a condition.

## 4.3 Comparators

All methods issue a block before its target arrives and update only after the target is complete; none uses partially observed targets. Every correction passes through the shared blending weight of Section 3.

• Naive forecasts. Persistence and a seasonal naive forecast with the daily period (144 samples for Weather), blended with the frozen forecast.

• Gradient-trained adapters. TAFAS [1] and PETSA [2] with their oficial calibration modules, COSA [3] with its oficial adapter, and FAC [4] reconstructed from its equations. Each takes one

Adam step after every complete block; TAFAS and PETSA therefore run without their periodicityaware adaptation schedule, which updates on partially observed targets. Learning rates are selected per dataset, backbone, and seed by chronological online MSE on the legacy validation range, on a ladder from $1 0 ^ { - 6 }$ to 1 that is extended from an initial grid until the best value is interior or reaches the end of the ladder; 4 of the 252 selections stay at 10<sup>−6</sup> (FAC 1, PETSA 1, TAFAS 2). The gate initializations of TAFAS and PETSA are selected from $\{ 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 3 , 1 \}$ ; 45 of 126 selections are at an end of this set. ORCA [13] uses its oficial network, routing, and published configuration, with its time constants converted from samples to blocks. It issues no correction until its replay bufer is full, which takes the first 125 blocks (9–35% of the evaluation blocks, depending on the dataset).

• Closed-form correctors. The RLS-Bank of the delay-aware audit [6] (oficial code; six forgetting factors and a pass-through, median per channel and horizon step), ELF [5] as configured in [7], and EPOC [7] with its default configuration. Our ELF is its linear forecaster: a ridge regression, accumulated without forgetting, from 86 real Fourier features of the mean-removed lookback to the leading ten frequencies of the horizon, with one Gram matrix per channel. It departs from the original in two respects: its statistics are updated after every complete block instead of periodically, and its forecast is combined with the frozen forecast by the shared blending weight instead of ELF’s weighting module. One Gram matrix shared by all channels would make $\mathrm { E L F } \mathrm { s }$ state independent of the channel count; we did not evaluate this variant.

## 4.4 Correctors and reporting

The correctors use λ = 1, a half-life of $n _ { 1 / 2 } = 1 2 8$ blocks, and τ = 10; DRiC-L uses K = 4 (DRiC-L4) or ${ \cal K } = 8 ~ ( \mathrm { D R i C \mathrm { - } L 8 } )$ , and $\scriptstyle \mathrm { { D R i C - G } } \ K = J = 8$ . For each method we report the MSE reduction relative to the frozen forecast, 100 $( 1 - \mathrm { M S E / M S E _ { f r o z e n } } )$ , and the retained state in bytes as implemented. We average the reduction over the 126 conditions, and also give its mean over the 12 dataset–backbone panels (the four ETT series form one dataset) and its median over the conditions, because 72 of the 126 conditions are ETT. For methods whose state depends on the backbone, the largest value over the three backbones is shown. The 126 conditions are not independent datasets, and we make no significance claims from counts of paired wins.

## 5 Results

## 5.1 Accuracy and retained state

Table 2 reports the MSE reduction and the retained state of every method. DRiC-L8 has the highest mean reduction over the 126 conditions, 14.55%, and also the highest mean over the 12 panels, 13.40%, and the highest median, 12.51%. ELF, the most accurate comparator, reaches 12.98%, 11.72%, and 10.94%. DRiC-L8 has lower MSE than ELF in 99 of the 126 conditions, and its mean over the 42 conditions of each seed ranges from 14.29% to 14.88%, against 12.84–13.20% for ELF. On ETT, Weather\*, and Electricity, DRiC-L8 has a higher mean reduction than every comparator. The exception is Trafic: there ELF has lower MSE than DRiC-L8 in 16 of the 18 conditions (15.93% against 13.00%), and with PatchTST, FAC, whose state (2.57 MB) is close to that of DRiC-L8, is also more accurate (15.03% against 10.57%). DRiC-L4 is above ELF in the mean over conditions (13.17%) and in the median (12.01%) but below it in the mean over panels (11.46%); we regard the two as comparable in accuracy. DRiC-L8 retains 19.8 kB on ETT and 2.18 MB on Trafic, and DRiC-L4 7.50 kB and 664 kB, against 302 kB and 37.1 MB for ELF.

The rank of DRiC-G depends on the weighting. Its mean over conditions, 9.17%, is above every comparator except ELF. However, its mean over panels, 6.74%, is below those of FAC (8.73%), RLS-Bank (7.94%), TAFAS (7.91%), and PETSA (7.67%), and it has lower MSE than EPOC, FAC, and RLS-Bank in only 62, 64, and 58 of the 126 conditions. Its reduction is largest on ETT and smallest on

Table 2: MSE reduction relative to the frozen forecast (%) and retained state. All 126: conditions (7 datasets × 3 backbones × 3 seeds × legacy/refit) with the filled Weather file (Weather\*); Mean: over the conditions; Panels: over the 12 dataset–backbone panels, with the four ETT series as one dataset; Median: over the conditions. Per-dataset columns: means over 72 ETT or 18 conditions. Weather is the standard file with its missing-value codes and is not part of All 126. State: largest over the three backbones. Bold marks the best value in each accuracy column.
<table><tr><td></td><td colspan="8">MSE reduction vs. frozen forecast (%)</td><td colspan="2">Retained state</td></tr><tr><td></td><td colspan="3">All 126</td><td colspan="3">Mean per dataset</td><td></td><td></td><td></td><td></td></tr><tr><td>Method</td><td>Mean</td><td>Panels</td><td>Median</td><td>ETT</td><td>Weather*</td><td>Elec.</td><td>Traffic</td><td>Weather</td><td>ETT</td><td>Traffic</td></tr><tr><td>Persistence</td><td>1.31</td><td>0.84</td><td>0.16</td><td>1.93</td><td>1.21</td><td>0.00</td><td>0.24</td><td>-0.61</td><td>528 B</td><td>528 B</td></tr><tr><td>Seasonal naive</td><td>3.73</td><td>2.34</td><td>2.13</td><td>5.58</td><td>0.85</td><td>0.24</td><td>2.71</td><td>-2.33</td><td>528 B</td><td>528 B</td></tr><tr><td>TAFAS</td><td>7.35</td><td>7.91</td><td>5.52</td><td>6.60</td><td>7.10</td><td>7.95</td><td>9.99</td><td>2.27</td><td>833 kB</td><td>103 MB</td></tr><tr><td>PETSA</td><td>7.64</td><td>7.67</td><td>6.50</td><td>7.60</td><td>6.41</td><td>7.62</td><td>9.04</td><td>5.65</td><td>195 kB</td><td>21.1 MB</td></tr><tr><td>COSA</td><td>5.90</td><td>5.07</td><td>3.36</td><td>7.01</td><td>3.58</td><td>2.02</td><td>7.65</td><td>3.19</td><td>71.4kB</td><td>8.71 MB</td></tr><tr><td>ORCA</td><td>6.14</td><td>4.05</td><td>4.11</td><td>8.92</td><td>1.43</td><td>2.57</td><td>3.30</td><td>1.12</td><td>754kB</td><td>79.7MB</td></tr><tr><td>FAC</td><td>8.52</td><td>8.73</td><td>7.55</td><td>8.23</td><td>8.20</td><td>8.01</td><td>10.49</td><td>6.99</td><td>21.4kB</td><td>2.57 MB</td></tr><tr><td>RLS-Bank</td><td>8.75</td><td>7.94</td><td>7.77</td><td>9.82</td><td>7.64</td><td>3.93</td><td>10.38</td><td>-77.85</td><td>83.9 kB</td><td>10.3 MB</td></tr><tr><td>ELF</td><td>12.98</td><td>11.72</td><td>10.94</td><td>14.67</td><td>6.38</td><td>9.91</td><td>15.93</td><td>3.51</td><td>302 kB</td><td>37.1 MB</td></tr><tr><td>EPOC</td><td>8.45</td><td>6.49</td><td>7.71</td><td>11.06</td><td>5.43</td><td>6.99</td><td>2.48</td><td>-644.36</td><td>4.53 kB</td><td>394 kB</td></tr><tr><td>DRiC-G</td><td>9.17</td><td>6.74</td><td>6.99</td><td>12.41</td><td>4.18</td><td>2.56</td><td>7.81</td><td>3.29</td><td>12.1 kB</td><td>18.9kB</td></tr><tr><td>DRiC-L4</td><td>13.17</td><td>11.46</td><td>12.01</td><td>15.46</td><td>10.69</td><td>11.28</td><td>8.41</td><td>8.57</td><td>7.50 kB</td><td>664kB</td></tr><tr><td>DRiC-L8</td><td>14.55</td><td>13.40</td><td>12.51</td><td>16.08</td><td>10.10</td><td>14.40</td><td>13.00</td><td>7.94</td><td>19.8 kB</td><td>2.18 MB</td></tr></table>

Electricity, where FAC and EPOC have lower MSE in all 18 conditions; on Trafic, FAC has lower MSE in all 18 conditions as well. What DRiC-G ofers is its state: 18.9 kB on Trafic, 21 and 136 times less than EPOC and FAC.

Figure 4 places every method by its retained state and accuracy for each dataset and backbone. A method lies on the dashed frontier if no method with the same or smaller state is more accurate. DRiC-L4 or DRiC-L8 lies on the frontier in 11 of the 12 panels. DRiC-G lies on it in 8 panels, including all six Electricity and Trafic panels, where it is the most accurate method at or below its state. In these six panels, the only methods with no more state are the naive forecasts and, with PatchTST, TAFAS and PETSA; the next smallest is EPOC, with 10 (Electricity) and 21 (Trafic) times the state of DRiC-G. DRiC-G lies on the frontier in none of the three ETT panels, where DRiC-L4 is more accurate with less state. The one panel without DRiC-L4 or DRiC-L8 on the frontier is Trafic with PatchTST: there DRiC-G (10.59%) is above DRiC-L8 (10.57%) with less state, and the seasonal naive forecast (7.55%) is above DRiC-L4 (7.54%); both diferences are a few hundredths of a point. The panel means are available in the code repository.

## 5.2 Components

Table 3 adds or removes one component at a time.

Local versus global. Adding the global regression before the local one, with the local regression fitted to what the global one leaves, changes the mean reduction by −0.01 points for K = 4 and −0.82 points for K = 8, and adds about 10 kB. The local regression carries the accuracy; adding it to DRiC-G raises the reduction but makes the state grow with the channel count. The two correctors are therefore complementary operating points rather than parts of one model.

Input. Reading only the forecast, without the latest input block and the last input value, changes the mean reduction by −3.36 (K = 8) and −4.04 (K = 4) points, the largest efect of any component. Reading

![](images/a4541047b9c2491233f5428e816abe1c9171c0cf03cf4141f872d48f40739eaa.jpg)  
Figure 4: Mean MSE reduction against retained state for each dataset (rows) and backbone (columns). Each point averages 24 ETT conditions or 6 conditions of the other datasets. Stars: the correctors of this paper. The dashed line is the best reduction achievable at or below each state among the methods shown. Weather\* is the filled Weather file. The horizontal axis is logarithmic.

Table 3: Components of the correctors. All: mean MSE reduction over the 126 conditions (%); ∆: mean paired diference from the first row of each group (percentage points); Weather: standard file. Without the input block and the last value, the local regression reads only the forecast’s coeficients.
<table><tr><td>Configuration</td><td>All (%)</td><td>∆</td><td>Weather (%)</td><td>State, ETT</td><td>State, Traffic</td></tr><tr><td>DRiC-L8</td><td>14.55</td><td></td><td>7.94</td><td>19.8 kB</td><td>2.18 MB</td></tr><tr><td>+ global regression</td><td>13.73</td><td>-0.82</td><td>7.03</td><td>29.7kB</td><td>2.19MB</td></tr><tr><td>- guard</td><td>12.80</td><td>-1.75</td><td>-700.38</td><td>19.8 kB</td><td>2.17MB</td></tr><tr><td>- input block and last value</td><td>11.19</td><td>-3.36</td><td>6.02</td><td>8.74kB</td><td>816 kB</td></tr><tr><td>all four input blocks</td><td>13.53</td><td>-1.02</td><td>7.44</td><td>71.6 kB</td><td>8.55 MB</td></tr><tr><td>K = 4</td><td>13.17</td><td>-1.37</td><td>8.57</td><td>7.50 kB</td><td>664kB</td></tr><tr><td>DRiC-L4</td><td>13.17</td><td></td><td>8.57</td><td>7.50kB</td><td>664kB</td></tr><tr><td>+ global regression</td><td>13.16</td><td>-0.01</td><td>7.53</td><td>17.4kB</td><td>674kB</td></tr><tr><td>- guard</td><td>11.80</td><td>-1.37</td><td>-678.54</td><td>7.45 kB</td><td>657kB</td></tr><tr><td>- input block and last value</td><td>9.13</td><td>-4.04</td><td>5.83</td><td>4.14kB</td><td>250 kB</td></tr><tr><td>all four input blocks</td><td>13.20</td><td>+0.02</td><td>8.78</td><td>21.3 kB</td><td>2.36 MB</td></tr><tr><td>DRiC-G</td><td>9.17</td><td></td><td>3.29</td><td>12.1 kB</td><td>18.9kB</td></tr><tr><td>- guard</td><td>8.83</td><td>-0.34</td><td>0.93</td><td>12.0 kB</td><td>12.0 kB</td></tr></table>

all four input blocks instead of the latest one changes it by −1.02 and +0.02 points while multiplying the Trafic state to 8.55 MB and 2.36 MB. In these data, the older blocks add state without adding accuracy.

Number of components. Going from K = 8 to K = 4 changes the reduction by −1.37 points and the state by a factor of about three; K trades accuracy for state.

Guard. Removing the guard changes the mean reduction by −1.75 (K = 8), −1.37 (K = 4), and −0.34 (global) points. Table 4 lists by dataset how many channel updates the guard skips and how it changes accuracy; since its decisions depend only on the frozen forecast’s residuals, every guarded corrector skips the same updates. On ETT it skips at most 3 updates per condition. On Weather\*, Electricity, and Trafic it skips 162, 124, and 309 updates per condition on average, below 1% of the channel updates, and these few blocks matter: on Weather\*, from which the missing-value codes have been removed, the guard raises the reduction by +9.61 (K = 8), +3.08 (K = 4), and +4.13 (global) points. On the standard Weather file it skips 176 updates per condition, only slightly more than on the filled file, and DRiC-L without the guard reaches reductions of −700.38% and −678.54%, i.e., about 8.0 and 7.8 times the frozen forecast’s MSE, while the guarded correctors reduce the error by 7.94% and 8.57%. The guard’s cost is concentrated on Trafic with PatchTST, where it changes the reduction by −2.08 points for K = 8 and −5.08 points for DRiC-G. For DRiC-L4, in contrast, the guard changes it by +10.37 points.

## 5.3 Computation

On one thread of an Intel Xeon Silver 4310 CPU, issuing and updating one block takes 0.26 and 2.59 ms for DRiC-L4 on ETT and Trafic, 0.31 and 6.38 ms for DRiC-L8, and 0.35 and 4.04 ms for DRiC-G, in a NumPy implementation that solves every channel’s system at each block. The timing uses synthetic inputs with the datasets’ channel counts, since the cost does not depend on the values; we did not time the comparators. A block spans 24 samples, i.e., 4 to 24 hours of the evaluated data.

Table 4: The guard by dataset. Channel updates: channels × evaluation blocks per condition. Skipped updates: mean and range over the 18 conditions. Efect: mean paired diference in MSE reduction between each corrector and its version without the guard (percentage points). Weather: standard file.
<table><tr><td colspan="2"></td><td colspan="2">Skipped updates</td><td colspan="3">Effect of the guard (points)</td></tr><tr><td>Dataset</td><td>Channel updates</td><td>Mean</td><td>Range</td><td>DRiC-L8</td><td>DRiC-L4</td><td>DRiC-G</td></tr><tr><td>ETTh1</td><td>2,534</td><td>0.0</td><td>0-0</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>ETTh2</td><td>2,534</td><td>0.0</td><td>0-0</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>ETTm1</td><td>10,157</td><td>0.3</td><td>0-2</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>ETTm2</td><td>10,157</td><td>0.8</td><td>0-3</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Weather*</td><td>23,037</td><td>161.7</td><td>36-330</td><td>+9.61</td><td>+3.08</td><td>+4.13</td></tr><tr><td>Electricity</td><td>175,908</td><td>123.8</td><td>37-273</td><td>+1.15</td><td>+0.78</td><td>0.00</td></tr><tr><td>Traffic</td><td>314,630</td><td>308.8</td><td>53-599</td><td>+1.49</td><td>+5.76</td><td>-1.77</td></tr><tr><td>Weather</td><td>23,037</td><td>176.1</td><td>41-360</td><td>+708.32</td><td>+687.11</td><td>+2.36</td></tr></table>

## 6 Discussion and Limitations

Why might the latest block sufice? One explanation, which we have not tested, is that the residual of a frozen forecaster at the next block depends mostly on how the newest observations depart from what the forecaster assumed. The latest input block and the forecast’s own leading coeficients would then describe this departure, and older blocks would add regressors whose coeficients must be estimated from the same number of blocks. The results are consistent with this explanation but do not establish it: adding the older blocks hardly changed accuracy with K = 4 and lowered it with K = 8.

The guard. The guard is a screen for blocks far outside a channel’s residual scale, whatever their cause. Missing-value codes are one such cause. However, most skipped updates occur on data without them: on Weather\*, Electricity, and Trafic (Table 4). It trades a loss on some panels for this protection. On Trafic with PatchTST, the guarded DRiC-L8 and the guarded DRiC-G are less accurate than their unguarded versions, by −2.08 and −5.08 points; larger τ reduces this cost but weakens the protection on the standard Weather file (Appendix A). The guard matters for a recursive estimator because blocks that have entered its statistics cannot be removed or reweighted afterwards.

Frozen forecasters. The compact PatchTST has no instance normalization, and its frozen MSE on ETT is above that of DLinear (Table 6); part of the correction on this backbone may compensate level shifts that instance normalization would remove. DRiC-L4 and DRiC-L8 reduce the error most with PatchTST (17.70% for K = 8, 16.61% for K = 4) and least with iTransformer, which normalizes each window (11.17% and 10.17%), so the gains should not be read as independent of the backbone.

Scope. We did not evaluate DRiC fitted once on the validation range and then frozen, nor the correctors fitted before deployment [14–16]; whether online updating is needed for the accuracy reported here is therefore not tested. We evaluate one horizon (H = 24), three trained backbones of moderate size, and seven datasets. Foundation-model forecasters, longer horizons, a channel-shared ELF, and the concurrent correctors ECHO and EFC were not evaluated. Retained state is counted as implemented arrays in double precision; a deployment could store them in lower precision.

## 7 Conclusion

We asked whether an online residual corrector for a frozen multi-horizon forecaster can combine a small retained state with a large error reduction under label delay. Restricting the regression to the leading

DCT coeficients of the forecast and of the latest input block, updating fixed-size recursive statistics, and guarding them with a residual scale of the frozen forecast gives DRiC-L8, which reduces the MSE more than ELF on average with 15–17 times less retained state, although ELF is more accurate on Trafic. The guard protects these recursive statistics from blocks they cannot discard: on the standard Weather file, whose missing-value codes drive the unguarded DRiC-L8 to on average 8.0 times the frozen forecast’s MSE, the guarded DRiC-L8 reduces the error by 7.94%; its cost is a lower reduction on Trafic with PatchTST. Pooling the regression over channels gives DRiC-G, which gives up accuracy for statistics that do not grow with the channel count; on Electricity and Trafic it is the most accurate method at or below its state. However, 5 to 7 of the 10 comparators, all with at least 10 times its state, are more accurate there.

## References

[1] HyunGi Kim, Siwon Kim, Jisoo Mok, and Sungroh Yoon. Battling the non-stationarity in time series forecasting via test-time adaptation. Proceedings of the AAAI Conference on Artificial Intelligence, 39(17):17868–17876, 2025.

[2] Heitor R. Medeiros, Hossein Sharifi-Noghabi, Gabriel L. Oliveira, and Saghar Irandoust. Accurate parameter-eficient test-time adaptation for time series forecasting. arXiv:2506.23424, version 1, 2025. Second Workshop on Test-Time Adaptation, ICML 2025.

[3] Jeonghwan Im and Hyuk-Yoon Kwon. COSA: Context-aware output-space adapter for test-time adaptation in time series forecasting. In International Conference on Learning Representations, 2026.

[4] Haochun Wang, Ruichen Xu, Georgios Kementzidis, Karen Cho, Sebastian Ramirez Villarreal, and Yuefan Deng. Towards principled test-time adaptation for time series forecasting. Transactions on Machine Learning Research, 2026.

[5] Thomas L. Lee, William Toner, Rajkarn Singh, Artjom Joosen, and Martin Asenov. Lightweight online adaption for time series foundation model forecasts. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 33736–33764, 2025.

[6] Mohamed Readh Fentazi, Mazene Ameur, and Adlen Ksentini. Training on the future: A delay-aware audit of test-time adaptation for time-series forecasting. arXiv:2610.12232, version 1, 2026.

[7] Takumi Fujimoto and Hiroaki Nishi. EPOC: Endpoint-preserving online correction with compressed residual state for multi-horizon time series forecasting. arXiv:2609.30929, version 2, 2026.

[8] Pablo Montero-Manso and Rob J. Hyndman. Principles and algorithms for forecasting groups of time series: Locality and globality. International Journal of Forecasting, 37(4):1632–1653, 2021.

[9] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers efective for time series forecasting? In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 11121–11128, 2023.

[10] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations, 2023.

[11] Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. iTransformer: Inverted transformers are efective for time series forecasting. In International Con ference on Learning Representations, 2024.

[12] Shivam Grover and Ali Etemad. Shift-aware test-time adaptation and benchmarking for time-series forecasting. Second Workshop on Test-Time Adaptation: Putting Updates to the Test, at the International Conference on Machine Learning, 2025.

[13] Xilin Dai, Yiding Liu, Hongjie Xia, Yifan Hu, Zewei Dong, Jiang-Ming Yang, and Qiang Xu. Learning the context of errors: Black-box online adaptation of time series foundation models. In Advances in Neural Information Processing Systems, 2026. arXiv:2606.14222.

[14] Hamza Cherkaoui, Malik Tiomoko, Giuseppe Paolo, Yili Zhang, Yu Meng, Zhang Keli, and Hafiz Tiomoko Ali. Post-training corrections for improved time-series forecasting. arXiv:2505.15354, version 3, 2026.

[15] Minh Hoang Nguyen, Dai Do, Huu Hiep Nguyen, Dung Nguyen, Kien Do, and Hung Le. Reviving error correction in modern deep time-series forecasting. In Proceedings of the International Conference on Machine Learning, 2026. arXiv:2605.21088.

[16] Jianxiang Xie, Yuncheng Hua, Mingyue Cheng, Flora D. Salim, and Hao Xue. Robust residual correction via selective deployment for time series forecasting. In Advances in Neural Information Processing Systems, 2026. arXiv:2512.22428.

[17] Boya Zhang, Xilin Dai, Yilin Wang, Qiang Xu, Luyao Dong, Xing He, and Wen Wu. ECHO: Correcting time series foundation model forecasts with historical analogues and errors. arXiv:2610.16433, version 1, 2026.

[18] Haonan Yang, Yulin He, and Zhuo Li. Look back, correct forward: Error feedback calibration for time series forecasting. arXiv:2610.22872, version 1, 2026.

[19] S. C. Chan and Y. X. Zou. A recursive least M-estimate algorithm for robust adaptive filtering in impulsive noise: Fast algorithm and convergence performance analysis. IEEE Transactions on Signal Processing, 52(4):975–991, 2004.

[20] Harry R. Glahn and Dale A. Lowry. The use of model output statistics (MOS) in objective weather forecasting. Journal of Applied Meteorology, 11(8):1203–1211, 1972.

[21] N. Ahmed, T. Natarajan, and K. R. Rao. Discrete cosine transform. IEEE Transactions on Computers, C-23(1):90–93, 1974.

[22] Arthur E. Hoerl and Robert W. Kennard. Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1):55–67, 1970.

[23] Ankit Goel, Adam L. Bruce, and Dennis S. Bernstein. Recursive least squares with variable-direction forgetting: Compensating for the loss of persistency. IEEE Control Systems Magazine, 40(4):80–102, 2020.

[24] Haoyi Zhou. ETDataset: Electricity transformer temperature dataset. Dataset repository. Accessed: Sep. 13, 2026.

[25] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In Advances in Neural Information Processing Systems, volume 34, pages 22419–22430, 2021.

## A Sensitivity to the settings

Table 5 varies one setting at a time around the defaults. Apart from λ = 100 and $\tau \geq 8 0$ for DRiC-L4 and DRiC-L8 and $\lambda \ge 1 0 0 0$ for DRiC-G, the mean reduction stays within 12.71–13.45% $( K = 4 )$ , 13.97– 14.79% $( K = 8 )$ , and 8.78–9.38% (global). On the 126 conditions, DRiC-L8 and DRiC-L4 have slightly higher mean reductions with $\tau = 2 0$ and $\tau = 4 0$ , respectively. However, loosening the guard to $\tau = 4 0$ lowers the reduction on the standard Weather file for DRiC-L8 and for DRiC-G, and we keep $\tau = 1 0$

Table 5: One-at-a-time sensitivity: mean MSE reduction over the 126 conditions (%), with the standard Weather file in parentheses. $n _ { 1 / 2 } \colon$ half-life of the regressions in blocks.
<table><tr><td>Setting</td><td>DRiC-L4</td><td>DRiC-L8</td><td>DRiC-G</td></tr><tr><td>default  $( \tau = 1 0 , n _ { 1 / 2 } = 1 2 8 , \lambda = 1 )$ </td><td>13.17 (8.57)</td><td>14.55 (7.94)</td><td>9.17 (3.29)</td></tr><tr><td> $\tau = 5$ </td><td>12.94 (8.58)</td><td>14.29 (8.25)</td><td>8.95 (3.84)</td></tr><tr><td> $\tau = 2 0$ </td><td>13.35 (8.82)</td><td>14.79 (7.40)</td><td>9.10 (2.51)</td></tr><tr><td> $\tau = 4 0$ </td><td>13.45 (9.66)</td><td>13.97 (3.04)</td><td>8.78 (1.29)</td></tr><tr><td> $\tau = 8 0$ </td><td>12.52 (5.87)</td><td></td><td></td></tr><tr><td> $\tau = 1 6 0$ </td><td>11.57 (0.79)</td><td></td><td></td></tr><tr><td> $n _ { 1 / 2 } = 6 4$ </td><td>13.08 (8.32)</td><td>14.27 (7.69)</td><td>8.88 (3.32)</td></tr><tr><td> $n _ { 1 / 2 } = 2 5 6$ </td><td>12.96 (8.42)</td><td>14.41 (7.74)</td><td>9.20 (3.08)</td></tr><tr><td> $n _ { 1 / 2 } = 5 1 2$ </td><td>12.71 (8.26)</td><td>14.18 (7.52)</td><td>9.13 (2.92)</td></tr><tr><td> $\lambda = 0 . 1$ </td><td>12.98 (8.51)</td><td>14.03 (7.87)</td><td>9.06 (3.28)</td></tr><tr><td> $\lambda = 1 0$ </td><td>12.73 (8.23)</td><td>14.17 (7.70)</td><td>9.32 (3.24)</td></tr><tr><td> $\lambda = 1 0 0$ </td><td>10.38 (7.14)</td><td>11.60 (6.85)</td><td>9.38 (3.21)</td></tr><tr><td> $\lambda = 1 0 0 0$ </td><td></td><td></td><td>8.03 (3.12)</td></tr><tr><td> $\lambda = 1 0 ^ { 4 }$ </td><td></td><td></td><td>5.35 (2.55)</td></tr></table>

## B Frozen forecasts

Table 6 gives the MSE of the frozen forecasts on the evaluation blocks, on the standardized scale.

Table 6: MSE of the frozen forecasts (standardized scale), mean over the three seeds and the legacy and refit bases. Weather: standard file.
<table><tr><td>Dataset</td><td>DLinear</td><td>PatchTST</td><td>iTransformer</td></tr><tr><td>ETTh1</td><td>0.396</td><td>0.438</td><td>0.404</td></tr><tr><td>ETTh2</td><td>0.145</td><td>0.181</td><td>0.157</td></tr><tr><td>ETTm1</td><td>0.263</td><td>0.317</td><td>0.288</td></tr><tr><td>ETTm2</td><td>0.100</td><td>0.117</td><td>0.096</td></tr><tr><td>Weather*</td><td>0.289</td><td>0.251</td><td>0.240</td></tr><tr><td>Electricity</td><td>0.193</td><td>0.152</td><td>0.131</td></tr><tr><td>Traffic</td><td>0.578</td><td>0.463</td><td>0.351</td></tr><tr><td>Weather</td><td>0.324</td><td>0.261</td><td>0.264</td></tr></table>