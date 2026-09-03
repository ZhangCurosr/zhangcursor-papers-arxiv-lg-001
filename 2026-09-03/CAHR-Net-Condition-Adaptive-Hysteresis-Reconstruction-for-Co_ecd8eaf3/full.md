# CAHR-Net: Condition-Adaptive Hysteresis Reconstruction for Compact and Interpretable Magnetic Core Loss Modeling

Chunye Gong<sup>†</sup> and Cong Yao<sup>†</sup>

Abstract—Magnetic core loss originates in the hysteresis loop: the energy dissipated per excitation cycle equals the loop area, and frequency, temperature, and waveform shape set the loss by reshaping the loop geometry. Most existing models, however, let these operating conditions act only on a terminal scalar— empirical equations fold them into fitted exponents, while datadriven predictors append them to encoded features—so no intermediate hysteresis representation remains for the conditions to reshape. This paper proposes a condition-adaptive hysteresis reconstruction network (CAHR-Net) that injects the operating conditions where they physically act. The proposed method preserves the physically interpretable chain from flux density waveform to magnetic field reconstruction, hysteresis-loop area integration, and power loss estimation. Different from scalar concatenation or terminal correction, CAHR-Net uses feature-wise linear modulation to inject frequency, temperature, and waveform statistics into the intermediate reconstruction representation. A matched large-batch training protocol based on AdamW, cosine learningrate scheduling, and a staged reconstruction-loss-to-power-loss objective is further reported, because the modulation pathway takes effect only within it. Experiments on the MagNet final A– E material protocol show that CAHR-Net attains an average p95 relative error of 6.89% with only 1874 parameters—the lowest average p95 among all compared methods, together with a lower worst-material p95 than the strongest black-box solution at about 48× fewer parameters—and reduces the average p95 of the physical reconstruction backbone from 7.47% to 6.89% and the p95 of material D, the most difficult material, from 16.40% to 14.87%. Ablation and condition-slice analyses indicate that the improvement comes from the coupling of physical loop reconstruction, structured condition modulation, and the matched optimization trajectory.

Index Terms—Condition modulation, core loss, feature-wise linear modulation, hysteresis reconstruction, MagNet, power magnetics.

## I. INTRODUCTION

AGNETIC core loss is not an abstract scalar response: core equals the area of the B–H hysteresis loop, and this loss largely dictates the thermal design, power density, and efficiency of high-frequency converters. Operating conditions govern the loss through the geometry of this loop—frequency broadens it, temperature rescales its amplitude and local slope, and the flux-density waveform sets the trajectory along which it is traversed. Every core loss model is therefore, implicitly, an answer to the question of where the operating conditions should enter. The classical Steinmetz equation [1] answers it by collapsing the loop into a compact power law whose fitted exponents absorb the conditions, and successive refinements have extended this answer to nonsinusoidal excitation by adding parameters to the same scalar law [2]–[6]. The collapse, however, is lossy: fixed functional forms cannot track how arbitrary waveforms, wide temperature ranges, and materialspecific nonlinearities jointly deform the loop.

This difficulty is fundamental rather than parametric. Maxwell’s equations and finite-element field solvers already describe the linear behavior of conductors and the geometric and thermal aspects of a component with high fidelity; what resists compact modeling is the strongly nonlinear, historydependent magnetization that generates the loop itself, compounded by the dispersion that material composition and manufacturing introduce at the component level. Hysteresis formalisms such as the Preisach model [7] and the Jiles– Atherton model [8] keep the loop as the modeling object, but their semi-empirical parameters are difficult to identify consistently across wide frequency, temperature, and waveform ranges. Core loss modeling has therefore long faced a dilemma: preserve the loop and struggle to fit it, or collapse the loop and lose the very geometry through which the operating conditions act.

Public datasets and modeling challenges [9]–[12] have recently made a third route practical: learning the waveformto-loss map directly from large-scale measurements. Neural predictors built on waveform encoding, multimodal fusion, and physics-inspired structures [13]–[16], together with knowledge-aware networks [17], history-dependent hysteresis operators [18], adversarial generation [19], and cross-material transfer [20], have pushed accuracy well beyond the empirical family. Yet most of these models achieve accuracy by abandoning the loop altogether, and they inherit a subtler form of the same misplacement: frequency, temperature, and flux-density statistics are appended to a feature vector or concatenated at the regression head, so the model can observe which condition produced a sample, while no intermediate hysteresis representation remains for the conditions to reshape.

Gray-box reconstruction restores the missing object. The HARDCORE framework [21] first reconstructs the magnetic field waveform and then integrates the BH loop area, showing that keeping the loop inside the model yields a favorable accuracy–complexity tradeoff under arbitrary waveforms. Even here, however, the operating conditions still enter as appended scalars: the object is right, but the conditioning pathway is not. We observe that the leading effects of frequency and temperature on the loop—broadening, tilting, and amplitude scaling—are naturally expressed as a scaling and shifting of the representation from which the loop is generated. This suggests encoding the operating conditions into featurewise scale and shift parameters that act directly on the intermediate reconstruction representation. CAHR-Net, the conditionadaptive hysteresis reconstruction network proposed in this paper, realizes this principle: a dilated temporal-convolutional encoder [22] extracts the waveform skeleton, feature-wise linear modulation (FiLM) [23] lets the conditions reshape the hidden representation before the magnetic field waveform is reconstructed, and a staged reconstruction-to-loss objective with matched large-batch optimization allows the modulation branch to leave the near-identity regime and become effective.

The contributions of this paper follow from this single design question—where should the operating conditions enter the model—and are summarized as follows.

• The object: a physically interpretable reconstruction backbone. A lightweight architecture keeps the hysteresis loop as the modeling object for arbitrary-waveform core loss estimation, following the interpretable path $B ( t ) $ $\hat { H } ( t )  B H  \hat { P } _ { \mathrm { v } }$ rather than direct terminal regression.

• The pathway: condition-adaptive FiLM modulation. Frequency, temperature, and waveform statistics are encoded into feature-wise scaling and shifting parameters that act directly on the intermediate hysteresis reconstruction representation. This pathway admits an electromagnetic reading—scaling matches loop broadening and amplitude rescaling, shifting matches bias and remanence drift— that scalar concatenation and terminal correction do not provide.

• The evidence: a favorable accuracy–parameter– interpretability tradeoff. Under the unified MagNet final A–E protocol, CAHR-Net attains an average p95 relative error of 6.89% with only 1874 parameters, the lowest average p95 among all compared methods—and a lower worst-material $p 9 5$ than the strongest black-box entry at about 48× fewer parameters—and it reduces the average p95 of the physical reconstruction backbone from 7.47% to 6.89% and the p95 of material D, the most difficult material, from 16.40% to 14.87%.

The rest of this paper is organized as follows. Section II develops CAHR-Net, from the problem formulation to the loop reconstruction chain, the condition modulation pathway, and the deployment-oriented inference optimization. Section III describes the experimental setup, including the network configuration and training protocol, Section IV presents the results and analyses, and Section V concludes the paper.

## II. METHODOLOGY

## A. Problem Formulation and Method Overview

Given a single-period flux density waveform

$$
{ \bf B } = \{ B _ { t } \} _ { t = 1 } ^ { N } , \quad N = 1 0 2 4 ,\tag{1}
$$

and a scalar operating vector s containing frequency, temperature, and waveform-level statistics, the goal is to predict the volumetric core loss $P _ { \mathrm { { v } } }$ . A direct map

$$
\hat { P } _ { \mathrm { v } } = f _ { \boldsymbol \theta } ( { \bf B } , { \bf s } )\tag{2}
$$

treats the loss as an isolated scalar output and leaves no intermediate representation for the operating conditions to act on. CAHR-Net instead uses a physically structured intermediate target that keeps such a pathway open: it reconstructs the magnetic field waveform ${ \hat { H } } ( t )$ , computes a loop-areabased initial estimate, and then applies a lightweight residual correction in the logarithmic loss domain.

Fig. 1 shows the resulting architecture. CAHR-Net contains a temporal convolutional branch for waveform encoding, a scalar multilayer perceptron for condition encoding, a FiLM block [23] that couples the two, a magnetic-field reconstruction head with BH area integration, and a residual power-loss correction head.

## B. Physical Loop Reconstruction Chain

The temporal encoder first extracts a waveform skeleton from the normalized 1024-point flux density sequence,

$$
\mathbf { F } _ { 0 } = \mathcal { E } _ { B } ( B ( t ) ) ,\tag{3}
$$

where $\mathcal { E } _ { B } ( \cdot )$ denotes a stack of dilated temporal convolution blocks [22]. The scalar branch maps the operating vector to modulation parameters. After modulation, the reconstruction head generates

$$
\begin{array} { r } { \hat { H } ( t ) = \mathcal G ( B ( t ) , \mathbf s ; \gamma , \beta ) . } \end{array}\tag{4}
$$

The reconstructed field waveform is converted to an initial power-loss estimate through discrete loop-area integration,

$$
\hat { P } _ { B H } = f b _ { \mathrm { l i m } } h _ { \mathrm { l i m } } \mathrm { t r a p z } \big ( \hat { H } , B \big ) ,\tag{5}
$$

where $f$ is the excitation frequency, $b _ { \mathrm { l i m } }$ and $h _ { \mathrm { { l i m } } }$ are the scale factors used to restore physical magnitudes, and trapz(·) denotes trapezoidal integration.

The final prediction is obtained in the logarithmic domain:

$$
\begin{array} { r } { \hat { y } _ { P } = \log \left( \hat { P } _ { B H } \right) + \Delta \left( [ \mathbf { s } , \log ( \hat { P } _ { B H } ) ] \right) , } \end{array}\tag{6}
$$

$$
\hat { P } _ { \mathrm { v } } = \exp ( \hat { y } _ { P } ) .\tag{7}
$$

In implementation, the residual head uses a standardized form of $\log ( \hat { P } _ { B H } )$ for numerical stability. ( 5 to 7 keep the prediction chain anchored to the physical relation between the hysteresis loop and core loss while allowing a small learned correction for scale and integration bias.

Both the reconstruction target and the scale factors in ( 5 are anchored in measured data. For every record, the MagNet database provides the measured magnetic field waveform $H ( t )$ , obtained from the sensed excitation current through Ampere’s law, alongside the flux density waveform obtained\` from the induced voltage [9]. The reconstruction loss defined in Section III-D therefore supervises ${ \hat { H } } ( t )$ with measured waveforms rather than with model-generated references, following the same protocol as HARDCORE [21]. The factor $b _ { \mathrm { l i m } }$ is the maximum absolute flux density over the training records of the material, and $h _ { \mathrm { { l i m } } }$ is the corresponding field normalization bound, capped at 150 $\mathrm { A } / \mathrm { m }$ and rescaled per record by the record’s relative flux amplitude, so that ( 5 restores physical magnitudes from normalized waveforms.

![](images/df7c5c34cfd1995a20f20bb7b1a6f22e9204e68ae91eb539f57bb5844fb64213.jpg)  
Fig. 1. Overall architecture of CAHR-Net. The operating vector s is encoded into the FiLM pair $[ \gamma , \beta ] .$ which rescales (⊙, by 1 + αγ) and then shifts (+, by β) eleven of the twelve encoder channels before the magnetic-field reconstruction head; one raw channel bypasses modulation. The reconstructed field $\check { H } ( t )$ drives the BH-area integration, and a residual multilayer perceptron corrects the loop-area estimate in the logarithmic loss domain to produce $\hat { P } _ { \bf v }$

## C. Condition-Adaptive FiLM Modulation

The key difference between CAHR-Net and scalarconcatenation models is the location and form of condition injection. Let $\mathcal { E } _ { s } ( \cdot )$ be the scalar encoder. The modulation parameters are generated by

$$
[ \gamma , \beta ] = \Psi ( { \mathcal E } _ { s } ( { \bf s } ) ) ,\tag{8}
$$

where $\Psi ( \cdot )$ projects the condition embedding into the feature modulation space. For an intermediate representation $\mathbf { F } _ { : }$ CAHR-Net applies

$$
\widetilde { \mathbf F } = \left( 1 + \alpha \gamma \right) \boldsymbol { \odot } \mathbf F + \boldsymbol { \beta } ,\tag{9}
$$

where ⊙ is channel-wise multiplication and α controls modulation strength. The default setting uses $\alpha = 0 . 1$

The form of ( 9 mirrors the two dominant ways in which operating conditions deform the measured hysteresis loop. As frequency rises, eddy-current and excess contributions widen the loop and rescale its local slope, which is a multiplicative deformation of the trajectory amplitude; temperature shifts the permeability and coercivity operating point, which acts closer to a translation of the loop. The FiLM pair reproduces these two degrees of freedom on the hidden representation: γ applies channel-wise rescaling, matching loop broadening and amplitude scaling, while $\beta$ applies channel-wise shifting, matching bias and remanence drift. The same waveform pattern can therefore be amplified, suppressed, or translated under different operating conditions before ${ \hat { H } } ( t )$ is reconstructed. Additive bias injection, as used in the backbone, covers only the translational degree of freedom, and input-level concatenation provides neither in a structured form. We do not claim a one-to-one identification between individual FiLM parameters and specific loop features; the correspondence holds at the level of transformation families, and it is consistent with the ablation in Fig. 6(a), where replacing FiLM with biasonly injection removes most of the gain under the matched

optimization protocol.

## D. Deployment-Oriented Inference Optimization

The 1874-parameter budget suggests that CAHR-Net is inexpensive to deploy, but parameter count alone does not determine the latency realized in practice. In the intended deployment scenarios—loss evaluation inside iterative converter design loops, large design-space sweeps, and online estimation on general-purpose processors without an accelerator—the model is queried one operating point at a time on a CPU, and for a network of this size the per-query cost is dominated by framework dispatch rather than by arithmetic. CAHR-Net therefore admits an accuracy-neutral inference optimization that lowers this per-query cost by changing only the execution backend, without modifying weights, architecture, or numerical precision.

The optimization exports the trained model to the ONNX format and serves it with ONNX Runtime 1.18.1 [24]. The export is not a push-button step: the loop-area integration of ( 5 relies on a trapezoidal-integration operator that has no ONNX equivalent. A thin export wrapper therefore rewrites the integration as an explicit trapezoidal sum built from elementary slicing, addition, multiplication, and reduction operations, and bakes the per-material normalization constants into the graph, so that the exported graph is self-contained. Two numerical gates guard the export. First, the wrapper is verified to be bit-identical to the original TorchScript model, with a maximum absolute output difference of exactly zero. Second, the exported model under ONNX Runtime is compared against the original over all 7651 material-A test samples: the maximum deviation is |∆ log $\hat { P } _ { \mathrm { v } } | < 8 \times 1 0 ^ { - 6 }$ and the test-set p95 relative error remains 5.86%, unchanged to two decimals. The optimization is thus accuracy-neutral both by construction and by measurement; its latency benefit is quantified in Section IV-F.

## III. EXPERIMENTAL SETUP

## A. Dataset, Protocol, and Metrics

Experiments use the MagNet final-stage A–E material protocol [9], [12]. Each material is trained and tested independently on its corresponding final split. The input waveform contains 1024 samples per period, and the scalar vector contains frequency, temperature, and waveform statistics as specified in Section $\mathrm { I I I - C }$

Prediction quality is measured by the relative error

$$
{ \mathrm { R e l . ~ E r r . } } = \left| { \frac { { \hat { P } } _ { \mathrm { v } } - P _ { \mathrm { v } } } { P _ { \mathrm { v } } } } \right| \times 1 0 0 \%\tag{10}
$$

and reported as the average relative error, the average $p 9 5$ and $p 9 9$ percentiles across the five materials, and the worstmaterial $p 9 5$ [25]. In all experiments reported below, the worst-material p95 occurs on material D, so “worst $p 9 5 '$ and “material-D $p 9 5 '$ denote the same quantity. The $p 9 5 -$ oriented view is adopted because thermal design is usually more sensitive to high-risk tail errors than to mean behavior; an average $p 9 5$ relative error below 10% is generally regarded as strong under the MagNet evaluation protocol [12]. All reported results follow the single-model reporting convention of the MagNet Challenge and HARDCORE [12], [21], using one trained model per method under the released protocol.

## B. Compared Methods

Three groups of baselines are considered. First, empirical models, including SE, MSE, GSE, and iGSE, are re-fitted under the same A–E protocol. SE is fitted by logarithmic least squares, whereas MSE, GSE, and iGSE use bounded nonlinear least squares with five multi-start restarts. Second, representative public challenge methods are summarized from the MagNet Challenge report [12]. Third, internal models are evaluated under the same experimental chain, including the HARDCORE physical reconstruction backbone [21], the same backbone trained with AdamW and cosine scheduling (HARDCORE + AWC), SE-TCN, FiLM-TCN, CAHR-Net, and width or combination variants.

## C. Network Configuration

Table I lists the complete layer-by-layer configuration of CAHR-Net; the parameter subtotals add up to the 1874 parameters reported throughout the paper. The waveform encoder receives five input channels—the globally normalized flux density waveform, a per-record normalized copy, its first and second time derivatives, and a saturation-emphasizing tangent transform [21]. The operating vector $\textbf { s } \in \mathbb { R } ^ { 1 1 }$ collects the logarithmic frequency, the temperature, four waveform-shape indicator variables, the peak-to-peak flux density and the mean absolute flux slew rate together with their logarithms, and the fundamental period. All convolutions use circular padding, consistent with the periodic single-period input, and the three temporal blocks use kernel size 9 with dilation rates 4, 8, and 16. FiLM acts at the interface between the waveform encoder and the reconstruction head: eleven of the twelve encoder channels are modulated by $\gamma , \beta \in \mathbb { R } ^ { 1 1 }$ , while the first channel passes through unmodulated so that a raw waveform pathway is always preserved.

CAHR-Net is fully specified by Table I and the training protocol below; HARDCORE is included in Table III only as a reconstruction baseline under the same A–E evaluation protocol.

TABLE I  
LAYER-BY-LAYER CONFIGURATION OF CAHR-NET
<table><tr><td>Stage</td><td>Configuration</td><td>Params</td></tr><tr><td>Waveform encoder  $\mathcal { E } _ { B }$ </td><td>Conv1d 5→12, k=9, d=4, tanh</td><td>552</td></tr><tr><td>Condition encoder  $\mathcal { E } _ { s } , \Psi$ </td><td>Linear 11→22, tanh, [γ, β]</td><td>264</td></tr><tr><td>FiLM modulation</td><td>(9, 11 of 12 channels</td><td>0</td></tr><tr><td>Reconstruction head  $\mathcal { G }$ </td><td>Conv1d 12→8, k=9, d=8, tanh</td><td>872</td></tr><tr><td></td><td>Conv1d 8→1, k=9, d=16</td><td>73</td></tr><tr><td>BH integration</td><td>Trapezoidal,(5</td><td>0</td></tr><tr><td>Residual head  $\Delta$ </td><td>Linear 12→8, tanh; Linear 8→1</td><td>113</td></tr><tr><td>Total</td><td></td><td>1874</td></tr></table>

## D. Training Protocol

Introducing FiLM alone does not guarantee a stable gain under large-batch training. CAHR-Net therefore uses a matched optimization protocol. The magnetic-field reconstruction loss and logarithmic power-loss loss are

$$
\begin{array} { r } { \mathcal { L } _ { H } = \mathrm { M S E } \Big ( \hat { H } , H \Big ) , \quad \mathcal { L } _ { P } = \mathrm { M S E } ( \hat { y } _ { P } , \log P _ { \mathrm { v } } ) . } \end{array}\tag{11}
$$

At epoch e of $E$ total epochs, the total loss is

$$
\mathcal { L } ( e ) = \left( 1 - \frac { e } { E } \right) \mathcal { L } _ { H } + \frac { e } { E } \mathcal { L } _ { P } .\tag{12}
$$

This schedule prioritizes stable loop reconstruction at the beginning and gradually shifts the emphasis toward loss prediction. The ordering matters for interpretability: if the powerloss objective dominated from the start, the residual head could compensate for a physically meaningless loop, and the reconstruction chain would lose its electromagnetic reading.

The optimizer is AdamW [26] with cosine learning-rate scheduling [27], learning rate $2 \times 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ batch size 512, and 10000 epochs; the modulation strength $\alpha = 0 . 1$ in ( 9 keeps the FiLM branch close to identity at initialization. These settings are stated in full because of an empirical finding documented in Fig. 6(a): under the original NAdam optimizer with step decay, FiLM behaves close to bias injection, whereas AdamW with cosine scheduling and the enlarged learning rate allows the modulation branch to leave the near-identity regime and take effect. We regard this coupling between the conditioning structure and the optimization trajectory as an empirical finding rather than a standalone contribution. Fig. 2 summarizes the resulting training-to-inference protocol.

## IV. RESULTS AND DISCUSSION

## A. Comparison: Empirical Equations, Challenge Entries, and the Accuracy–Parameter Frontier

Table II consolidates the external comparison into three views. Empirical equations. Even when SE, MSE, GSE, and iGSE are fitted independently for each material, they remain in the 56.84%–78.30% average-p95 band; their limitation is therefore structural rather than a matter of parameter fitting, because a terminal scalar law cannot represent arbitrarywaveform, wide-condition effects. MagNet Challenge entries. The public field spans a wide range, from the strongest blackbox solution, Bristol, at 7.78% average $p 9 5$ with 90,653 parameters, to gray-box entries that stay above 13%; these numbers were obtained under different implementations and possibly different evaluation protocols, so they serve as an external reference rather than a controlled ranking. Accuracy– parameter frontier. Isolating the four most competitive operating points shows where the proposed method leads: Bristol, which reports the best accuracy yet placed third officially; Fuzhou, the official runner-up; HARDCORE, the method of the official winner Paderborn, evaluated here as the reconstruction backbone; and CAHR-Net. Note that the official final ranking weighted model size jointly with accuracy [12], which is why the 8914-parameter Fuzhou entry outranked Bristol despite Bristol’s lower error. CAHR-Net attains the lowest average $p 9 5$ in the whole table, 6.89% against Bristol’s 7.78% and Fuzhou’s 7.94%, together with the lowest worst-material p95, 14.87% against Bristol’s 15.90%, yet reaches this with only 1874 parameters—about 1/48 of the Bristol budget and 1/4.8 of Fuzhou—while the gray-box models near its size stay above 13% average p95. CAHR-Net thus occupies a region of the accuracy–parameter plane that neither the blackbox nor the existing gray-box group covers, while keeping the prediction chain physically readable. Fig. 3 visualizes this relationship.

## B. Ablation Study

Table III isolates the contribution of each design choice under a shared data pipeline and evaluation script, with rows marked AWC further sharing the optimization protocol of Section III-D, moving from the HARDCORE backbone toward CAHR-Net one axis at a time. Three axes are examined. Optimization protocol. Applying the matched large-batch protocol (AWC) to the unchanged backbone lowers the average p95 from 7.47% to 7.27% and the worst-material p95 from 16.40% to 15.64%, so a better-conditioned training region already helps before any architectural change. Conditioninjection module. Compared under the same protocol, replacing the injection with SE or a plain FiLM branch reaches 7.14% and 7.85% average $p 9 5 .$ , whereas the condition-adaptive injection of CAHR-Net attains 6.89% together with the lowest average p99 of 11.82% and the lowest worst-material $p 9 5$ of 14.87%; the gap to FiLM-TCN trained at the original learning rate is consistent with the finding of Section III-D that the modulation pathway requires the matched optimization region to become fully effective. Capacity. Widening the network through CAHR-W12, SE+CAHR-W12, and SE+CAHR-W16 raises the parameter count from 1874 to as many as 3057 yet degrades the average p95 to the 7.69%–8.47% range and pushes the worst-material p95 above 20%, indicating that the improvement stems from how condition information is injected and trained rather than from added capacity. CAHR-Net therefore gives the best accuracy of the chain at a nearminimal parameter budget.

## C. Material-Level Tail Error

Fig. 4(a) shows that the main gain is concentrated on material D, the most difficult material. CAHR-Net lowers the material-D p95 from 16.40% to 14.87%, while materials $\mathbf { A } ,$ B, C, and E do not show obvious degradation. This behavior is desirable because it improves the high-risk tail without transferring error to easier materials.

Fig. 4(b) further verifies that the error distribution on material D shifts left and contracts in the high-error region. Therefore, the proposed method does not merely reduce the mean error; it directly compresses samples that would otherwise dominate design risk.

## D. Condition-Slice Analysis

To locate where the improvement appears, sample errors are pooled across the five materials and stratified by temperature, frequency, peak flux density, and true loss magnitude under shared quartile boundaries. Fig. 5 summarizes these four condition-slice views. The gains are broad but modest: CAHR-Net lowers the $p 9 5$ in most slices by a few tenths to about two percentage points, and holds the temperature-slice p95 within

![](images/79b8b3c7f3146660a0e4be1cc279aab74704f4c71f8d672d848f2f9bc9ea060c.jpg)  
Fig. 2. Training-to-inference protocol of CAHR-Net. Each material A–E is trained independently. During training, the forward pass of Fig. 1 produces H<sup>ˆ</sup> and $\hat { P } _ { \bf v }$ , which are scored by the staged loss $\mathcal { L } ( e )$ of ( 12 and optimized with AdamW and cosine scheduling; the curriculum schedule shifts the emphasis from the magnetic-field reconstruction loss $\mathcal { L } _ { H }$ early in training to the logarithmic core-loss objective $\mathcal { L } _ { P }$ late in training. The trained parameters $\theta ^ { \star }$ are then deployed on held-out inputs, and the average, p<sub>95</sub>, and p<sub>99</sub> relative-error metrics are computed per material.

TABLE II  
UNIFIED COMPARISON: EMPIRICAL EQUATIONS, MAGNET CHALLENGE ENTRIES, AND THE ACCURACY–PARAMETER FRONTIER
<table><tr><td>Method</td><td>Core Expression / Category</td><td>Params</td><td>Avg p95 (%)</td><td>Worst p95 (%)</td></tr><tr><td colspan="5">(a) Empirical equations (per-material fit, unified A-E protocol)</td></tr><tr><td>SE</td><td> $k f ^ { \alpha } B _ { \mathrm { p k } } ^ { \beta }$ </td><td></td><td>78.30</td><td>85.84</td></tr><tr><td>MSE</td><td> $k f ^ { \alpha } B _ { \mathrm { p k } } ^ { \beta } f _ { \mathrm { e q } } ^ { \alpha - 1 }$ </td><td></td><td>56.84</td><td>62.31</td></tr><tr><td>GSE</td><td> $k f ^ { \alpha } \Delta \dot { B } ^ { \beta - \alpha } \int | \dot { B } | ^ { \alpha }$ </td><td></td><td>57.66</td><td>64.52</td></tr><tr><td>iGSE</td><td> $k f ^ { \alpha } \sum _ { i } \Delta B _ { i } ^ { \beta } \Delta t _ { i } ^ { 1 - \alpha }$ </td><td></td><td>61.23</td><td>72.42</td></tr><tr><td colspan="5">(b) MagNet Challenge entries (as reported [12])</td></tr><tr><td>Bristol</td><td>Black-box</td><td>90653</td><td>7.78</td><td>15.90</td></tr><tr><td>Fuzhou</td><td>Black-box</td><td>8914</td><td>7.94</td><td>20.70</td></tr><tr><td>Tsinghua</td><td>Black-box</td><td>116061</td><td>16.88</td><td>29.90</td></tr><tr><td>XJTU</td><td>Black-box</td><td>17342</td><td>14.20</td><td>30.00</td></tr><tr><td>MMINN [15]</td><td>Gray-box</td><td>1084</td><td>13.86</td><td>30.70</td></tr><tr><td>PI-MFF-CN [13]</td><td>Gray-box</td><td>139938</td><td>30.56</td><td>79.10</td></tr><tr><td colspan="5">(c) Accuracy-parameter frontier</td></tr><tr><td>Bristol (best reported accu- Black-box racy)</td><td></td><td>90653 (48×)</td><td>7.78 (+12.9%)</td><td>15.90 (+6.9%)</td></tr><tr><td>Fuzhou (official 2nd place)</td><td>Black-box</td><td>8914 (4.8×)</td><td>7.94 (+15.2%)</td><td>20.70 (+39.2%)</td></tr><tr><td>HARDCORE (official 1st place; backbone)</td><td>Gray-box</td><td>1742 (0.93×)</td><td>7.47 (+8.4%)</td><td>16.40 (+10.3%)</td></tr><tr><td>CAHR-Net Gray-box</td><td></td><td>1874</td><td>6.89</td><td>14.87</td></tr></table>

Blocks (a) and (c, backbone/CAHR-Net rows) are evaluated under the unified protocol of this work (Section III); block (b) and the Bristol/Fuzhou rows of (c) are as reported in the original sources [12], so the cross-group comparison is an external reference rather than a controlled ranking. Official placements refer to the challenge’s final ranking, which jointly weighted accuracy and model size. Parenthesized values in block (c) are relative to CAHR-Net: the parameter count as a multiple of 1874, and the two p95 columns as the relative increase over 6.89% and 14.87%, respectively. The Bristol entry used 90653-parameter models for materials A–C and 16449-parameter transfer-learned models for D–E, so its worst-material p95 (15.90, material D) was obtained with the 16449-parameter model. Empirical models are analytic per-material fits, whose coefficient count is not comparable to a network parameter budget (“—”).

7.7%–10.7% versus 8.5%–10.8% for the backbone. The clearest improvements fall on the lowest peak-flux-density quartile, from 10.91% to 8.53%, and on the highest-error, lowest-lossmagnitude quartile, from 12.62% to 11.00%, where relative error is intrinsically harder to control; a small number of slices, such as the second flux-density quartile, are essentially unchanged.

TABLE III  
ABLATION STUDY UNDER THE UNIFIED EXPERIMENTAL CHAIN.
<table><tr><td>Method</td><td></td><td>Parameters Avg. Rel. Err. (%) Avg. p95 (%)</td><td></td><td>Avg. p99 (%)</td><td></td><td>Worst p95 (%) Rel. ∆p95 vs. CAHR-Net (%)</td></tr><tr><td>HARDCORE</td><td>1742</td><td>2.63</td><td>7.47</td><td>12.53</td><td>16.40</td><td>+8.4</td></tr><tr><td> $\mathrm { H A R D C O R E } + \mathrm { A W C }$ </td><td>1742</td><td>2.62</td><td>7.27</td><td>11.95</td><td>15.64</td><td>+5.5</td></tr><tr><td> $\mathrm { S E { - } T C N ~ + ~ A W C ~ + ~ 2 ~ \times ~ }$   $1 0 ^ { - 3 }$ </td><td>1875</td><td>2.48</td><td>7.14</td><td>11.83</td><td>16.23</td><td>+3.6</td></tr><tr><td> $\mathrm { F i L M { - } T C N } \quad + \quad \mathrm { A W C } \quad + \quad$   $1 0 ^ { - 3 }$ </td><td>1874</td><td>2.70</td><td>7.85</td><td>13.19</td><td>17.61</td><td>+13.9</td></tr><tr><td>CAHR-Net</td><td>1874</td><td>2.42</td><td>6.89</td><td>11.82</td><td>14.87</td><td>0.0</td></tr><tr><td>CAHR-W12</td><td>2346</td><td>2.50</td><td>7.74</td><td>13.24</td><td>20.68</td><td>+12.3</td></tr><tr><td> $\mathbf { S E } + \mathbf { C A H R - W } 1 2$ </td><td>2524</td><td>2.46</td><td>7.69</td><td>12.94</td><td>20.04</td><td>+11.6</td></tr><tr><td> $\mathbf { S E } + \mathbf { C A H R } { \mathbf { - W } } 1 6$ </td><td>3057</td><td>2.70</td><td>8.47</td><td>15.86</td><td>24.41</td><td>+22.9</td></tr></table>

![](images/4c76d7de9b36dbbaa6e6752b934728aaa9e882e93725e63270ab69df2821db93.jpg)  
Fig. 3. Parameter-count and average-p95 relation between public representative methods and CAHR-Net.

E. Injection–Optimization Interaction and Parameter Efficiency

Fig. 6(a) presents a two-dimensional sweep over condition injection and optimization strategy. Under NAdam with step decay, bias, SE, and FiLM injection are close. After switching to AdamW with cosine scheduling, FiLM starts to show a stable advantage. Increasing the learning rate to $2 \times 1 0 ^ { - 3 }$ moves FiLM into the best region, corresponding to the CAHR-Net configuration reported in Table III.

Fig. 6(b) shows that widening the model or adding SEstyle recalibration does not reliably outperform CAHR-Net under a similar parameter budget. The dominant improvement therefore comes from how condition information enters the reconstruction process and how this structure is trained, rather than from a simple increase in capacity.

## F. Inference Latency Optimization

We now quantify the latency benefit of the accuracy-neutral deployment optimization of Section II-D. Table IV reports the per-sample latency of four execution backends on a single thread of a server-class Xeon Platinum 8360Y CPU, using the material-A test set and the trained checkpoint; each entry is the median over repeated timed runs, with 300 repetitions at batch size 1. Three observations follow. First, at batch size 1—the regime that matters for per-operating-point queries—switching from TorchScript to ONNX Runtime reduces the latency from

![](images/39dd26d7aad194160c495b9918c406b24558ada80daccce24c942107272140eb.jpg)

![](images/0c97634f9b4acc9bd33bc8fc19ea1bd5d41194eb73265d7e2d4d02bdfa7e2031.jpg)  
Fig. 4. Material-level tail analysis. (a) Material-level p95 comparison between HARDCORE and CAHR-Net. (b) Empirical cumulative distribution of relative error on material D; the inset magnifies the tail region around the 95th percentile, where CAHR-Net shifts the p95 from 16.40% to 14.87%.

![](images/94525ccdecb1939ec1c077ccd75ae074fdc64006e2895db4bab7722423fea34f.jpg)

![](images/72a9b488a69ebfda53de3f977da62639b83b468950cda57cb738123dbeeb8af9.jpg)

![](images/b21d221005da26ce3a4a9bca53903f2cd049bd54736bb316391badcbc39ef069.jpg)

![](images/bc0834b282acf5a07a6e1f3334ceb91ad1dd61b7152869242398a4ecf17c93a8.jpg)  
Fig. 5. Condition-slice analysis of p95 relative error, shown from top to bottom for temperature, frequency, peak flux density $B _ { \mathrm { p k } }$ , and true loss magnitude quartiles.

0.82 ms to 0.19 ms per sample, a 4.4× speedup, and 5.6× over eager execution, at zero accuracy cost. Second, the table itself diagnoses where the gain comes from: at batch size 256 all four backends converge to approximately $1 5 0 \mu \mathrm { s }$ per sample, which is the arithmetic floor of the model on this core; consequently, about 82% of the batch-1 TorchScript time is per-invocation framework overhead rather than computation, and the runtime switch removes precisely this overhead instead of approximating the model. Third, at batch size 2048 the PyTorch-based backends degrade to roughly $3 7 0 \mu \mathrm { s }$ per sample whereas ONNX Runtime remains at $1 7 4 \mu \mathrm { s } ,$ , indicating better operator fusion and memory locality at large batches.

![](images/8c47a7d592d8dec0837a430884f695fce527d30790b4bc46a33780ea9a64a2e1.jpg)

![](images/b8ef5717b57bcc4a9706a14b3ff43b247f3b0b5b0e5ca9c2a3d9049eb1645dcd.jpg)  
Fig. 6. Injection–optimization interaction and parameter efficiency. (a) Interaction between condition injection strategy and optimization protocol. (b) Parameter-count and average-p95 Pareto distribution of internal model variants.

To assess whether these gains are tied to the proposed architecture, the identical export-and-verification procedure is applied to the bias-injection backbone HARDCORE, and Table V contrasts the two models before and after the backend switch. The backbone exhibits the same 4.4× reduction in single-sample latency, from 798 to $1 8 2 \mu \mathrm { s } .$ , which indicates that the optimization exploits a property shared by this class of compact loop-reconstruction models—their inference time is dominated by framework dispatch rather than by computation—and hence generalizes beyond CAHR-Net itself. Table V also shows that the latency of the two models remains within 5% of each other at every batch size under the optimized backend—186 versus $1 8 2 \mu \mathrm { s }$ at batch size 1— an absolute difference of a few microseconds per sample. Condition-adaptive modulation thus incurs no practically relevant serving cost: the accuracy improvements of Table III are obtained at essentially the latency of the unmodulated backbone.

From a deployment perspective, 0.19 ms per sample on one CPU thread corresponds to more than 5000 loss evaluations per second per core, which is sufficient to embed the model directly in interactive design iteration or in the inner loop of a circuit-level optimization sweep over thousands of operating points, with no GPU in the serving path. The exported ONNX graph, together with the roughly 7.5 KB of FP32 weights, is also the standard entry format for embedded inference toolchains, so it constitutes a ready artifact for microcontrollerclass deployment. Two caveats bound this result: the reported latencies are measured on a server-class CPU core rather than on a microcontroller, and further reductions through integer quantization or sequence downsampling are left as future work.

TABLE IV  
SINGLE-THREAD CPU INFERENCE LATENCY OF CAHR-NET ACROS EXECUTION BACKENDS (µS PER SAMPLE, MATERIAL-A TEST SET)
<table><tr><td></td><td></td><td></td><td>Batch Size PyTorch Eager TorchScript TorchScript (Opt.) ONNX Runtime</td><td></td></tr><tr><td>1</td><td>1051</td><td>822</td><td>788</td><td>186</td></tr><tr><td>16</td><td>200</td><td>185</td><td>169</td><td>149</td></tr><tr><td>256</td><td>161</td><td>154</td><td>152</td><td>152</td></tr><tr><td>2048</td><td>366</td><td>372</td><td>372</td><td>174</td></tr></table>

TABLE V

MODEL-LEVEL LATENCY BEFORE AND AFTER BACKEND OPTIMIZATION(µS PER SAMPLE, SINGLE CPU THREAD)
<table><tr><td></td><td>Batch Size HARDCORE (TorchScript) CAHR-Net Before Opt. (TorchScript) CAHR-Net After Opt. (ONNX Runtime)</td><td></td><td></td></tr><tr><td>1</td><td>798</td><td>822</td><td>186</td></tr><tr><td>16</td><td>167</td><td>185</td><td>149</td></tr><tr><td>256</td><td>147</td><td>154</td><td>152</td></tr><tr><td>2048</td><td>354</td><td>372</td><td>174</td></tr></table>

## V. CONCLUSION

This paper returns to a fundamental question: where, in a magnetic core loss model, should operating conditions enter? CAHR-Net answers by placing them in the hysteresis-loop reconstruction process rather than at the model output. The network predicts along the physical $B ( t )  \hat { H } ( t )  B H $ $\hat { P } _ { \mathrm { v } }$ chain, keeping the hysteresis loop as an explicit modeling object throughout. On this basis, FiLM converts frequency, temperature, and waveform statistics into channel-wise scales and shifts that act directly on the intermediate representation from which the magnetic-field waveform is reconstructed— the point at which these conditions physically act—instead of merely correcting a scalar output at the end of the model. The training protocol is presented together with the network architecture because the two are inseparable: under the original NAdam and step-decay configuration, the modulation pathway remains in a near-identity regime; only under the matched protocol combining AdamW, cosine scheduling, and a larger learning rate does it leave that regime and become effective. CAHR-Net therefore derives its value not from any single dimension, but from the coupling of physical loop reconstruction, structured condition modulation, and matched optimization: within a model budget on the order of $1 0 ^ { 3 }$ parameters, it retains intermediate quantities that engineers can compare directly with measured hysteresis loops while achieving the tail accuracy attainable by black-box models.

Under the MagNet A–E final protocol, CAHR-Net attains the lowest average p95 among the compared methods—6.89% with 1874 parameters, versus 7.78% for the strongest blackbox entry at about 48× the parameter count, which it also surpasses on worst-material p95—and offers an operating point that neither black-box nor existing gray-box models cover: a material-D p95 compressed from 16.40% to 14.87% relative to the reconstruction backbone, and a prediction chain whose intermediate quantities remain electromagnetically readable. Ablations attribute this gain to the synergy of loop reconstruction, structured condition modulation, and matched optimization. The accuracy–interpretability combination is also cheap to serve: after an accuracy-neutral export to a dedicated inference runtime, single-sample CPU latency drops from 0.82 ms to 0.19 ms on one thread, so the model can run inside design loops without an accelerator.

This study remains limited to per-material training on the five MagNet ferrites and single-period steady-state excitation without DC bias. Future work will address these issues together with material recommendation, uncertainty-aware design margins, and deployment-oriented model compression.

## REFERENCES

[1] C. P. Steinmetz, “On the law of hysteresis,” Transactions of the American Institute of Electrical Engineers, vol. 9, pp. 1–64, 1892.

[2] J. Reinert, A. Brockmeyer, and R. W. A. A. De Doncker, “Calculation of losses in ferro- and ferrimagnetic materials based on the modified Steinmetz equation,” IEEE Transactions on Industry Applications, vol. 37, no. 4, pp. 1055–1061, 2001.

[3] K. Venkatachalam, C. R. Sullivan, T. Abdallah, and H. Tacca, “Accurate prediction of ferrite core loss with nonsinusoidal waveforms using only Steinmetz parameters,” in 2002 IEEE Workshop on Computers in Power Electronics, 2002, pp. 36–41.

[4] J. Muhlethaler, J. Biela, J. W. Kolar, and A. Ecklebe, “Improved core-¨ loss calculation for magnetic components employed in power electronic systems,” IEEE Transactions on Power Electronics, vol. 27, no. 2, pp. 964–973, 2012.

[5] T. Guillod, J. S. Lee, H. Li, S. Wang, M. Chen, and C. R. Sullivan, “Calculation of ferrite core losses with arbitrary waveforms using the composite waveform hypothesis,” in 2023 IEEE Applied Power Electronics Conference and Exposition, 2023, pp. 1586–1593.

[6] A. Arruti, J. Anzola, F. J. Perez-Cebolla, I. Aizpuru, and M. Mazuela, “The composite improved generalized Steinmetz equation (ciGSE): An accurate model combining the composite waveform hypothesis with classical approaches,” IEEE Transactions on Power Electronics, vol. 39, no. 1, pp. 1162–1173, 2024.

[7] F. Preisach, “Uber die magnetische nachwirkung,” <sup>¨</sup> Zeitschrift fur Physik ¨ , vol. 94, no. 5–6, pp. 277–302, 1935.

[8] D. C. Jiles and D. L. Atherton, “Theory of ferromagnetic hysteresis,” Journal of Magnetism and Magnetic Materials, vol. 61, no. 1–2, pp. 48–60, 1986.

[9] H. Li, D. Serrano, T. Guillod, S. Wang, E. Dogariu, A. Nadler, M. Luo, V. Bansal, N. K. Jha, Y. Chen, C. R. Sullivan, and M. Chen, “How MagNet: Machine learning framework for modeling power magnetic material characteristics,” IEEE Transactions on Power Electronics, vol. 38, no. 12, pp. 15 829–15 853, 2023.

[10] H. Li, D. Serrano, S. Wang, and M. Chen, “MagNet-AI: Neural network as datasheet for magnetics modeling and material recommendation,” IEEE Transactions on Power Electronics, vol. 38, no. 12, pp. 15 854– 15 869, 2023.

[11] D. Serrano, H. Li, S. Wang, T. Guillod, M. Luo, V. Bansal, N. K. Jha, Y. Chen, C. R. Sullivan, and M. Chen, “Why MagNet: Quantifying the complexity of modeling power magnetic material characteristics,” IEEE Transactions on Power Electronics, vol. 38, no. 11, pp. 14 292–14 316, 2023.

[12] M. Chen, H. Li, S. Wang, T. Guillod, D. Serrano, N. Forster, W. Kirchgassner, T. Piepenbrock, O. Schweins, O. Wallscheid, Q. Huang, Y. Li, X. Shen, H. Wouters, W. Martinez et al., “MagNet challenge for data-driven power magnetics modeling,” IEEE Open Journal of Power Electronics, vol. 6, pp. 883–898, 2025.

[13] Y. Hu, J. Xu, J. Wang, and W. Xu, “Physics-inspired multimodal feature fusion cascaded networks for data-driven magnetic core loss modeling,” IEEE Transactions on Power Electronics, vol. 39, no. 9, pp. 11 356– 11 367, 2024.

[14] N. Rajput, H. B. Sandhibigraha, N. Agrawal, and V. M. Iyer, “An empirical model informed neural network core loss predictor for soft magnetic materials,” IEEE Transactions on Power Electronics, vol. 40, no. 8, pp. 11 257–11 267, 2025.

[15] Q. Huang, Y. Li, J. Zhu, and S. Li, “Magnetization mechanism-inspired neural networks for core loss estimation,” IEEE Transactions on Power Electronics, vol. 39, no. 12, pp. 16 382–16 390, 2024.

[16] Y. Xiao, C. Li, and Z. Zheng, “A magnetic core loss model based on physics-informed neural network with cross-attention,” IEEE Transactions on Power Electronics, vol. 41, no. 1, pp. 92–96, 2026.

[17] J. Deng, W. Wang, Z. Ning, P. Venugopal, J. Popovic, and G. Rietveld, “High-frequency core loss modeling based on knowledge-aware artificial neural network,” IEEE Transactions on Power Electronics, vol. 39, no. 2, pp. 1968–1973, 2024.

[18] Q. Huang, Y. Li, Y. Dou, Y. Li, J. Zhu, and S. Li, “History-dependent Prandtl-Ishlinskii neural network for quasi-static core loss prediction under arbitrary excitation waveforms,” IEEE Transactions on Power Electronics, 2025, early access.

[19] X. Shen, Y. Zuo, and W. Martinez, “Conditional generative adversarial network aided iron loss prediction for high-frequency magnetic components,” IEEE Transactions on Power Electronics, vol. 39, no. 8, pp. 9953–9964, 2024.

[20] J. Chen, Y. Liang, X. Li, X. Liu, H. Jiang, X. Miao, and L. Zhang, “A magnetic core loss modeling method based on domain adaptation using multiple kernel maximum mean discrepancy,” IEEE Transactions on Power Electronics, 2025, early access.

[21] W. Kirchgassner, N. Forster, T. Piepenbrock, O. Schweins, and O. Wallscheid, “HARDCORE: H-field and power loss estimation for arbitrary waveforms with residual, dilated convolutional neural networks in ferrite cores,” IEEE Transactions on Power Electronics, vol. 40, no. 2, pp. 3326–3335, 2025.

[22] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv preprint arXiv:1803.01271, 2018.

[23] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.

[24] ONNX Runtime developers, “ONNX Runtime,” https://onnxruntime.ai, 2021, version 1.18.1.

[25] C. Tofallis, “A better measure of relative prediction accuracy for model selection and model estimation,” Journal of the Operational Research Society, vol. 66, no. 8, pp. 1352–1362, 2015.

[26] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019.

[27] ——, “SGDR: Stochastic gradient descent with warm restarts,” in International Conference on Learning Representations, 2017.