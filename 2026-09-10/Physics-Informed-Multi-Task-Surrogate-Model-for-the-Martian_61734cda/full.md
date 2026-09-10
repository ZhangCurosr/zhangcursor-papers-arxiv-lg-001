# Physics-Informed Multi-Task Surrogate Model for the Martian Nightside Thermosphere

Sergey Nikiforov<sup>∗1</sup>

<sup>1</sup>Center for Astrophysics and Space Science (CASS), New York University Abu Dhabi, Abu Dhabi, United Arab Emirates

Modeling the Martian nightside thermosphere remains challenging due to sparse in situ sampling and strong coupling among transport, magnetic, and seasonal processes. Purely data-driven models can produce non-physical artifacts, such as density inversions, in poorly sampled altitude regimes.

We present a multi-task physics-informed neural network that simultaneously predicts the base-10 logarithmic densities of four neutral species (O, CO<sub>2</sub>, N<sub>2</sub>, and Ar) using more than a decade of MAVEN/NGIMS observations (MY 32–38, 2014–2025). A shared backbone learns a common representation of the nightside thermospheric state and branches into species-specific output heads.

A weak monotonicity prior is incorporated via automatic diferentiation by penalizing positive vertical gradients in logarithmic density. Experiments using an orbit-disjoint train/validation/test split show that physics-informed regularization substantially reduces non-physical inversions while preserving predictive skill and slightly improving it in the best-performing configuration, as measured by RMSE, MAE, and $R ^ { 2 }$

The resulting model provides a computationally eficient surrogate for nightside thermospheric reconstruction with improved vertical consistency.

## 1 Introduction

Modeling the Martian nightside thermosphere is complicated by the lack of direct local solar forcing and the dominance of complex transport processes, compounded by incomplete observational coverage. Global Circulation Models (GCMs), such as the LMD-GCM [1] and the NASA Ames GCM [2], provide a robust framework for large-scale dynamics. However, they are computationally intensive and not always practical for rapid evaluation. The resolution of complex hydrodynamics and photochemistry across a global grid requires substantial high-performance computing power. Engineering references like Mars-GRAM [3] ofer standardized, computationally eficient average profiles. However, capturing localized, small-scale variability due to short-lived space weather efects remains a significant challenge.

More than a decade of in situ measurements from the MAVEN mission [4] provides valuable insight into Martian upper-atmosphere variability. However, nightside coverage remains inherently irregular, as the mission was not optimized for systematic sampling of these regions. Such sparsity can cause purely data-driven models to extrapolate poorly, producing non-physical artifacts such as density inversions. To address this limitation, we introduce a multitask physics-informed neural network (MT-PINN) [5] that augments MAVEN observations with a weak monotonicity constraint on the vertical density gradient.

The model is intended as an observationally constrained complement to existing thermospheric frameworks. It is designed to improve representation in regions where global simulations [6] and empirical climatologies [7] can deviate from in situ measurements [8], particularly under sparsely sampled nightside conditions.

MT-PINN is not intended to replace physics-based GCMs. It provides a fast observational surrogate that preserves a simple constraint on vertical structure and can be queried in regimes where data coverage is limited.

A fast nightside density surrogate can also support mission-oriented workflows, including profile reconstruction along arbitrary trajectories, sensitivity studies across seasons and space weather conditions, and model–data comparison without repeated GCM runs.

MT-PINN therefore provides a direct link between irregular in situ observations and physics-based model comparison. Its diferentiable output also allows profile-by-profile comparison with GCM results under matched environmental conditions.

## 2 Data and Parameter Formulation

We analyze MAVEN Key Parameters (KP) [9] from late MY 32 to early MY 38, spanning a complete solar cycle and the MY 34 global dust storm [10, 11]. The broader dataset combines in situ measurements from NGIMS [12] (neutral densities), MAG [13] (magnetic fields), SWEA [14] and SEP [15] (particle fluxes), along with geometry from SPICE [16]. Atmospheric dust loading is parameterized using reconstructed climatology maps [17, 18].

The input space X encompasses spatial, temporal, and geophysical drivers. In addition to standard kinematic data (altitude, latitude, longitude, solar zenith angle, local solar time) and seasonal parameters $\left( \boldsymbol { L _ { s } } \right)$ , we explicitly include external energy inputs that influence nightside thermospheric dynamics. When direct solar extreme ultraviolet insolation is absent at the local nightside, the thermospheric state is shaped by global day-to-night circulation and localized heating from particle precipitation. To represent upstream solar wind conditions when direct local measurements are unavailable, we use density, velocity, temperature, and dynamic pressure estimates from an external Gaussian-process solar wind model [19]. Nightside conditions are identified by solar zenith angle exceeding 90<sup>◦</sup>.

The model is trained jointly for four neutral species (O, $\mathrm { C O } _ { 2 } , \mathrm { N } _ { 2 }$ , and Ar) using a multi-task learning framework. Measurements with non-positive density or reported density error above the species-specific 99th percentile are excluded from the corresponding target mask. To account for varying data fidelity, the remaining NGIMS measurements are weighted according to their reported density error values, as described in Section 3.2.

## 3 Multi-Task Physics-Informed Neural Network

## 3.1 Neural Network Architecture

We adopt a multi-task learning framework [20, 21] combined with a physics-informed neural network formulation [5, 22]. The model simultaneously predicts the base-10 logarithms of neutral number densities for four thermospheric species: $\mathrm { O , C O _ { 2 } , N _ { 2 } }$ , and Ar.

For each species i, we define the target variable as

$$
y _ { i } = \log _ { 1 0 } ( \rho _ { i } )\tag{1}
$$

where $\rho _ { i }$ denotes the measured neutral number density. The network produces corresponding predictions

$$
\hat { y } _ { i } = \log _ { 1 0 } ( \hat { \rho } _ { i } )\tag{2}
$$

with $\hat { \rho } _ { i }$ representing the model-estimated density. All regression is therefore performed in log-density space.

Joint prediction is motivated by the fact that these species share a common background neutral temperature and large-scale wind fields, which govern their transport and difusive separation under external drivers such as solar wind forcing and crustal magnetic topology.

The architecture consists of a shared feature extractor (backbone) followed by species-specific linear output heads. The shared backbone maps the inputs to a latent representation that captures common responses to altitude, solar zenith angle, $L _ { s }$ , magnetic structure, and particle precipitation.

The backbone is implemented as a three-layer multilayer perceptron (Table 1):

Each species-specific head produces

$$
\hat { y } _ { i } = W _ { i } h + b _ { i } , \quad i \in \{ \mathrm { O } , \mathrm { C O } _ { 2 } , \mathrm { N } _ { 2 } , \mathrm { A r } \} ,\tag{3}
$$

Table 1: Multi-Task PINN architecture.
<table><tr><td>Backbone</td><td> $\overline { { d _ { i n } \to 2 5 6 \to 1 2 8 \to 6 4 } }$ </td></tr><tr><td>Heads</td><td>BN + ReLU (+DO for first two layers)  $4 \times \mathrm { L i n e a r } ( 6 4 , 1 )$ </td></tr></table>

BN: batch normalization; ReLU: rectified linear unit; DO: dropout

where $\textit { h } \in \mathbb { R } ^ { 6 4 }$ denotes the shared latent feature vector. The final prediction vector is formed by concatenation:

$$
\hat { \mathbf { y } } = \mathrm { c o n c a t } ( \hat { y } _ { \mathrm { O } } , \hat { y } _ { \mathrm { C O _ { 2 } } } , \hat { y } _ { \mathrm { N _ { 2 } } } , \hat { y } _ { \mathrm { A r } } )\tag{4}
$$

Modeling densities in logarithmic space improves numerical stability and efectively captures the approximately exponential decrease of density with altitude. It also reduces variability between species, whose absolute densities difer by several orders of magnitude in the 140–350 km regime.

## 3.2 Loss Function

To train the MT-PINN, we formulate a composite objective that balances fidelity to in situ observations with a physically motivated structural prior. The total loss is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { d a t a } } + \lambda \mathcal { L } _ { \mathrm { p h y s } } , } \end{array}\tag{5}
$$

where ${ \mathcal { L } } _ { \mathrm { d a t a } }$ enforces agreement with measurements, and $\mathcal { L } _ { \mathrm { p h y s } }$ introduces a weak monotonicity constraint in the vertical direction. Here, physics-informed regularization refers to a structural constraint rather than direct enforcement of a governing partial diferential equation.

Thermospheric measurements exhibit varying signalto-noise ratios across altitude, species, and environmental conditions. To account for this heterogeneity, we employ an uncertainty-weighted mean squared error:

$$
\mathcal { L } _ { \mathrm { d a t a } } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \frac { \sum _ { i } m _ { j , i } w _ { j , i } ( \hat { y } _ { j , i } - y _ { j , i } ) ^ { 2 } } { \sum _ { i } m _ { j , i } + \epsilon }\tag{6}
$$

where N is the batch size, $j$ indexes the individual samples within a batch, and i indexes the gas species. The variable $m _ { j , i }$ denotes a validity mask (1 for a valid NGIMS measurement and 0 otherwise), and $y _ { j , i } \left( \hat { y } _ { j , i } \right)$ represents the true (predicted) log density. The measurement weights are defined as

$$
\tilde { w } _ { j , i } = \frac { 1 } { u _ { j , i } + 0 . 0 5 } , \qquad w _ { j , i } = \frac { \tilde { w } _ { j , i } } { \langle \tilde { w } _ { i } \rangle } ,\tag{7}
$$

where $u _ { j , i }$ denotes the reported NGIMS density error quantity and $\left. \tilde { w } _ { i } \right.$ is the mean weight for valid measurements of species i. The validity mask follows the species-specific filtering described in Section 2. A small constant ϵ is added to the denominator for numerical stability. The per-sample normalization prevents samples containing multiple valid species from contributing disproportionately to the batch loss.

Under approximately hydrostatic conditions, neutral densities decrease roughly exponentially with altitude. Purely data-driven regressors may produce non-physical positive vertical gradients (density inversions), particularly in sparsely sampled regimes.

To mitigate this behavior, we penalize positive vertical gradients in the logarithmic density:

$$
\mathcal { L } _ { \mathrm { p h y s } } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \left( \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \left[ \operatorname { R e L U } \left( \frac { \partial \hat { y } _ { j , i } } { \partial z _ { j } } \right) \right] ^ { 2 } \right)\tag{8}
$$

where $T = 4$ is the number of species, N is the batch size, and $z _ { j }$ denotes the altitude corresponding to sample j. The ReLU operator selectively penalizes only positive gradients, leaving negative gradients unconstrained.

Gradients are computed via automatic diferentiation with respect to the normalized altitude feature. To recover dimensional consistency, we rescale the gradients by the altitude standard deviation used during input normalization.

This soft regularization does not enforce strict hydrostatic balance. Instead, it reduces large-scale non-physical inversions while allowing localized deviations due to transient heating or measurement noise. The hyperparameter λ controls the trade-of between observational accuracy and physical consistency.

## 3.3 Training and Data Splitting Strategy

To ensure a realistic evaluation of model generalization, we perform data partitioning at the orbit level rather than at the individual sample level. All measurements belonging to a single MAVEN orbit are treated as a coherent block and assigned entirely to either the training, validation, or test subset.

Successive measurements along a single orbit are strongly correlated in altitude, latitude, magnetic topology, and environmental conditions. Random sample-level shufling would therefore introduce data leakage, allowing the network to interpolate between nearly identical adjacent points rather than learning the underlying relationships between geophysical drivers and thermospheric structure. Such leakage typically results in artificially inflated performance metrics and overestimates of predictive skill.

We adopt a strict orbit-disjoint three-way split. Approximately 80% of the data are assigned to training, while the remaining 20% are divided evenly into validation and test subsets. The validation subset is used for early stopping, learning-rate scheduling, and model checkpoint selection. The held-out test subset is used for final performance reporting.

The resulting split tests generalization to independent nightside trajectories with orbital conditions not used during training, including variations in crustal magnetic field configuration, season $( L _ { s } ) _ { : }$ , and upstream solar wind forcing.

## 4 Results

We evaluate MT-PINN on two criteria: generalization to unseen orbital passes and consistency of the predicted vertical structure. Unless stated otherwise, all reported performance metrics refer to the held-out test set.

## 4.1 Calibration of Physical Regularization

Rather than treating the physics weight λ as a fixed hyperparameter, we interpret it as a calibration parameter controlling the balance between statistical fidelity and physical structure. We therefore explore a range:

$$
\lambda \in \{ 0 , 0 . 0 1 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 0 . 5 0 , 0 . 7 5 , 1 . 0 0 \} .\tag{9}
$$

Predictive skill is evaluated in base-10 log-density space using RMSE, MAE, and $R ^ { 2 } { \mathrm { , } }$ , computed only for valid MAVEN/NGIMS measurements of each species. Physical consistency is quantified via the inversion rate, defined as the fraction of held-out test inputs exhibiting a locally positive vertical gradient:

$$
\frac { \partial \hat { y } _ { i } } { \partial z } > 0 .\tag{10}
$$

Because the surrogate predicts all four species at every input state, inversion rates are evaluated over all held-out test inputs.

Figure 1 illustrates the accuracy–consistency trade-of across the tested regularization weights. Among the tested values, $\lambda = 0 . 5 0$ gives both the lowest macro-averaged RMSE and the lowest mean inversion rate across species.

Table 2 summarizes the resulting behavior. Purely datadriven training $( \lambda = 0 )$ already provides strong predictive skill but exhibits residual non-physical vertical gradients. The response to regularization is not strictly monotonic across the tested range or across individual species. At $\lambda = 0 . 5 0$ , the model achieves the lowest macro-averaged RMSE and MAE and the highest $R ^ { 2 } { \mathrm { ; } }$ , while substantially reducing inversion rates for O and $\Nu _ { 2 }$ and maintaining a very low inversion rate for Ar. Increasing λ to 0.75 or 1.00 worsens predictive performance and increases inversion rates relative to $\lambda = 0 . 5 0$

An important practical advantage of MT-PINN is that it produces continuous, diferentiable vertical profiles. This enables direct diagnostics of physically interpretable quantities such as $\partial \hat { y } _ { i } / \bar { \partial } ;$ and supports integration with physicsbased workflows. In this sense, λ controls not only statistical fit, but also the reliability of gradient-based interpretation in sparsely sampled regimes.

![](images/c8d1bd2877596a8f24fc1abb2f876cd2750d6fb6a7792433ce5d0831e68ddba6.jpg)  
Figure 1: Accuracy–consistency trade-of on the held-out test set. Macro-averaged RMSE across $\mathrm { O , C O _ { 2 } , N _ { 2 } , }$ , and Ar is shown against the mean inversion rate across the four species. Marker color denotes the tested physics regularization weight λ. Selected values are annotated for reference, and the star highlights $\lambda =$ 0.50.

Table 2: Held-out test-set performance as a function of the physics regularization weight. Predictive metrics are macroaveraged across O, $\mathrm { C O } _ { 2 } , \mathrm { N } _ { 2 }$ , and Ar. Inversion rates are evaluated over all held-out test inputs and are reported in percent. Best values in each column are shown in bold.
<table><tr><td> $\lambda$ </td><td>RMSE</td><td colspan="2">MAE</td><td colspan="2"> $R ^ { 2 }$ </td></tr><tr><td>0.00</td><td>0.2896</td><td colspan="2">0.2179</td><td colspan="2">0.9231</td></tr><tr><td>0.01</td><td colspan="2">0.2859</td><td colspan="2">0.2152</td><td colspan="2">0.9255</td></tr><tr><td>0.05</td><td colspan="2">0.2857</td><td colspan="2">0.2140</td><td colspan="2">0.9254</td></tr><tr><td>0.10</td><td colspan="2">0.2932</td><td colspan="2">0.2224</td><td colspan="2">0.9213</td></tr><tr><td>0.20</td><td colspan="2">0.2840</td><td colspan="2">0.2134</td><td colspan="2">0.9262</td></tr><tr><td>0.50</td><td colspan="2">0.2833</td><td colspan="2">0.2124</td><td colspan="2">0.9268</td></tr><tr><td>0.75 1.00</td><td colspan="2">0.2868 0.2864</td><td colspan="2">0.2154</td><td colspan="2">0.9249</td></tr><tr><td></td><td colspan="2"></td><td colspan="2">0.2157</td><td colspan="2">0.9248</td></tr><tr><td>λ</td><td colspan="2">O Inv. (%)</td><td colspan="2">CO2 Inv. (%)</td><td colspan="2">N2 Inv. (%)</td></tr><tr><td>0.00</td><td>2.226</td><td colspan="2">0.533</td><td colspan="2">0.456</td><td colspan="2">0.0036</td></tr><tr><td>0.01</td><td>2.918</td><td colspan="2">0.716</td><td colspan="2">0.630</td><td colspan="2">0.0143</td></tr><tr><td>0.05</td><td>2.039</td><td colspan="2">0.753</td><td colspan="2">0.613</td><td colspan="2">0.0152</td></tr><tr><td>0.10</td><td>4.122</td><td colspan="2">1.122</td><td colspan="2">0.876</td><td colspan="2">0.0411</td></tr><tr><td>0.20</td><td>1.433</td><td colspan="2">0.950</td><td colspan="2">0.182</td><td colspan="2">0.0006</td></tr><tr><td>0.50</td><td>0.608</td><td colspan="2">0.586</td><td colspan="2">0.115</td><td colspan="2">0.0009</td></tr><tr><td>0.75</td><td>2.499</td><td colspan="2">0.610</td><td colspan="2">0.393</td><td colspan="2">0.0062</td></tr><tr><td>1.00</td><td>2.806</td><td colspan="2">0.858</td><td colspan="2">0.867</td><td colspan="2">0.0214</td></tr></table>

## 4.2 Contextual Baseline Comparison

As a non-neural reference point, we trained an XGBoost regressor to predict log density using a compact feature set with engineered cyclical variables and an orbit-disjoint split (every fifth orbit assigned to test). For O, the tuned model achieves $\mathrm { R M S E } \approx 0 . 2 0 7$ and $R ^ { 2 } \approx 0 . 9 1 1$ in logarithmic density space. Because this baseline uses a reduced feature set and a diferent split protocol than the MT-PINN experiments, we treat it as contextual benchmarking rather than a strictly controlled head-to-head comparison.

While tree-based ensembles can yield strong pointwise predictive metrics, they do not naturally provide smooth, diferentiable vertical structure. As a result, gradient-based diagnostics $( \partial \hat { y } / \partial z )$ are either undefined or dominated by piecewise constant behavior. For atmospheric applications where vertical gradients carry physical meaning and are used in downstream coupling and consistency checks, this motivates the use of diferentiable models such as MT-PINN.

## 5 Discussion and Outlook

## 5.1 Limitations and Scope

The model is restricted to nightside neutral densities within the 140–350 km altitude range and is not a globally selfconsistent thermospheric model. The weak monotonicity constraint improves vertical plausibility but does not enforce hydrostatic balance, energy conservation, or selfconsistent momentum coupling.

In a multi-task setting, shared latent representations may propagate systematic biases between species. Their specific contribution to multi-species prediction requires further controlled investigation.

Localized departures from monotonicity may also arise from gravity waves, transient heating, or measurement noise. The regularization is therefore intentionally soft, reducing large-scale non-physical inversions without enforcing strictly monotonic profiles.

## 5.2 Computational Eficiency and Model Intercomparison

A systematic quantitative intercomparison with classical thermospheric models remains outside the scope of the present study. Once trained, the network provides rapid pointwise evaluation without the computational cost of running a full three-dimensional circulation model.

The weak structural constraint adds physical guidance without changing the model into a full dynamical solver. This is useful when rapid evaluation is more important than a globally self-consistent atmospheric solution.

## 5.3 Implications

For the Martian nightside, sparse and irregular sampling makes purely empirical reconstruction dificult. The present results show that even a weak structural prior can reduce non-physical vertical gradients without reducing predictive skill.

In this model, a shared multi-species representation can be combined with weak physical regularization while retaining accurate density predictions and improving vertical consistency.

## 5.4 Future Roadmap

A systematic intercomparison with GCM outputs across seasons, crustal magnetic configurations, and extreme solar events is ongoing. Future work will test temperaturedependent hydrostatic constraints based on species-specific scale heights and extend the model to ionospheric constituents.

## 6 Acknowledgments

This material is based upon work supported by Tamkeen under the NYU Abu Dhabi Research Institute grant CASS.

## References

1. Forget, F. et al. Improved general circulation models of the Martian atmosphere from the surface to above 80 km. Journal ofGeophysical Research: Planets 104, 24155–24175 (Oct. 1999).

2. Haberle, R. M. et al. Documentation of the NASA/Ames Legacy Mars Global Climate Model: Simulations of the Present Seasonal Water Cycle. Icarus 333, 130–164 (2019).

3. Justus, C. G. & Johnson, D. L. Mars Global Reference Atmospheric Model 2001 Version (Mars-GRAM 2001): User’s Guide tech. rep. (NASA George C. Marshall Space Flight Center, Marshall Space Flight Center, Alabama, 2001).

4. Jakosky, B. M. et al. The Mars Atmosphere and Volatile Evolution (MAVEN) Mission. Space Science Reviews 195, 3–48 (2015).

5. Raissi, M. et al. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal ofComputational Physics 378, 686–707 (2019).

6. Lewis, S. R. et al. A Climate Database for Mars. Journal ofGeophysical Research: Planets 104, 24177–24194 (1999).

7. Millour, E. et al. The Mars Climate Database (Version 5.3) in Scientific Workshop: ’From Mars Express to ExoMars’ (Madrid, Spain, Feb. 2018), 68.

8. Bougher, S. W. et al. Early MAVEN Deep Dip campaign reveals thermosphere and ionosphere variability. Science 350 (Nov. 2015).

9. Dunn, P. A. (NASA Planetary Data System, 2023). https://pdsppi.igpp.ucla.edu/collection/urn:nasa:pds:maven. insitu.calibrated:data.kp.

10. Sánchez-Lavega, A. et al. in Zonal Jets: Phenomenology, Genesis, and Physics (eds Galperin, B. & Read, P. L.) 72–103 (Cambridge University Press, 2019).

11. Stone, S. et al. Neutral Composition and Horizontal Variations of the Martian Upper Atmosphere From MAVEN NGIMS. Journal of Geophysical Research: Planets (2022).

12. Mahafy, P. R. et al. The neutral gas and ion mass spectrometer on the Mars atmosphere and volatile evolution mission. Space Science Reviews 195, 49–73 (2015).

13. Connerney, J. et al. The MAVEN magnetic field investigation. Space Science Reviews 195, 257–291 (Dec. 2015).

14. Mitchell, D. L. et al. The MAVEN Solar Wind Electron Analyzer. Space Science Reviews 200, 495–528 (2016).

15. Larson, D. E. et al. The MAVEN Solar Energetic Particle Investigation. Space Science Reviews 195, 153–172 (2015).

16. Acton, C. et al. A look towards the future in the handling of space science mission geometry. Planetary and Space Science 150, 9–12 (2018).

17. Montabone, L. et al. Eight-year Climatology of Dust Optical Depth on Mars. Icarus 251, 65–95 (2015).

18. Montabone, L. et al. Martian Year 34 Column Dust Climatology from Mars Climate Sounder Observations: Reconstructed Maps and Model Simulations. Journal ofGeophysical Research: Planets (2020).

19. Azari, A. R. et al. A Virtual Solar Wind Monitor at Mars With Uncertainty Quantification Using Gaussian Processes. Journal ofGeophysical Research: Machine Learning and Computation 1, e2024JH000155 (2024).

20. Caruana, R. Multitask learning. Machine learning 28, 41–75 (1997).

21. Crawshaw, M. Multi-task learning with deep neural networks: A survey. arXiv preprint arXiv:2009.09796 (2020).

22. Karniadakis, G. E. et al. Physics-informed machine learning. Nature Reviews Physics 3, 422–440 (2021).