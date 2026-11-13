# HybridCT: ML Accelerated Tomography for Climate Relevant 3D Cloud Observations

Linda Forster Jet Propulsion Laboratory California Institute of Technology Pasadena, CA, USA.

Marcin J. Kurowski Jet Propulsion Laboratory California Institute of Technology Pasadena, CA, USA.

Bernhard Mayer Ludwig-Maximilians-Universität Meteorologisches Institut, Fakultät für Physik Munich, Germany.

Nicholas J. LaHaye Jet Propulsion Laboratory California Institute of Technology Pasadena, CA, USA.

Anna Jungbluth Imperial College London London, UK

Anthony B. Davis Jet Propulsion Laboratory California Institute of Technology Pasadena, CA, USA.

## Abstract

Low-level clouds are one of the largest sources of uncertainty in climate predictions, partly because we lack detailed three-dimensional (3D) measurements of their internal structure. Here, we present HybridCT, a cloud tomographic reconstruction scheme accelerated by machine learning (ML) that recovers 3D cloud extinction directly from passive multi-angle observations. An ML emulator maps radiances from multiple viewing angles to cloud optical thickness (COT), producing pathintegrated projections, and a Simultaneous Algebraic Reconstruction Technique (SART) explicitly inverts these projections to retrieve 3D extinction. Training observations are generated from synthetic 3D cloud fields using radiative transfer simulations for nine viewing angles between ±70<sup>◦</sup>. HybridCT reconstructs extinction with RMSE 13.3 km<sup>−1</sup> in 3.5 s, compared to 22.7 km<sup>−1</sup> in approx. 30 min for the state-of-the-art physics-based tomography framework AT3D, a speedup exceeding 500×. HybridCT thus provides a scalable route toward large-scale observations of 3D cloud structure relevant to radiative-budget interpretation and evaluation of high-resolution cloud simulations.

## 1 Introduction

Low-level clouds strongly influence Earth’s radiation budget and remain one of the largest sources of uncertainty in climate feedbacks and future warming projections [1–5]. Many processes governing cloud evolution and radiative effects are inherently 3D, yet satellite observations involve a fundamental tradeoff: Vertically resolved measurements provide detailed profiles but only along limited sampling tracks, while passive imagers provide broad spatial and temporal coverage but mostly of columnintegrated or cloud-top properties. This limits our ability to observe cloud depth, internal structure, and organization over large areas. Recent studies suggest that such 3D information could improve understanding of cloud processes, radiative effects, and the evaluation of high-resolution atmospheric models [6, 7]. Multi-angle tomography can address this observational gap. Multiple views of the same cloud provide complementary geometric information for reconstructing 3D structure [8–10]. The main obstacle is computational: physics-based tomography requires repeated 3D radiative-transfer calculations that are expensive for realistic cloud scenes [11, 12]. This limits the number and size of cloud fields that can be processed, and consequently the use of tomography for large observational datasets. While machine learning (ML) is increasingly used to derive 3D cloud structure from 2D observations, we distinguish tomography from other 3D reconstruction techniques, like radiance mapping/extrapolation of single view points [13–15]. Tomography requires multi-angle observations that constrain 3D information geometrically and assumes the cloud field remains static across viewing angles. Prior works using multi-angle observations include 3DeepCT [16] and VIPCT [17] for extinction, and NeMF [18] for microphysical properties. However, these approaches do not explicitly constrain geometry, but learn to extrapolate 3D structure through supervised learning with simulated ground truth cloud volumes. For real observations, ground truth volumes are not available. ProbCT [19] partially addresses this through a 3D radiative transfer solver that compares real imagery against renderings from predicted volumes. However, this is again limited by the computational constraint of 3D radiative transfer calculations. Neural radiance fields [20] instead constrain volumes via differentiable rendering, but are optimized per scene and require higher angular sampling and spatial resolutions than spaceborne multi-angle instruments currently provide. HybridCT addresses these constraints with a hybrid ML and tomographic approach:

![](images/214cbd44c740194722b3ae80c7052e85d397fc8c45e2a15f652bea795d42d5f3.jpg)  
Figure 1: Overview of HybridCT. (1) Multi-angle passive radiances are first mapped to cloud optical thickness projections using a U-Net. (2) The resulting projections are then inverted with an explicit SART tomography solver to reconstruct the three-dimensional cloud extinction field.

$$
\begin{array} { r } { { \bf L } _ { 1 : N } \xrightarrow { \mathrm { U - N e t } } \tau _ { 1 : N } \xrightarrow { \mathrm { S A R T } } \beta _ { \mathrm { e x t } } ( x , y , z ) . } \end{array}\tag{1}
$$

As illustrated in Figure 1, an ML emulator U-Net first converts passive multi-angle radiances L into cloud optical thickness (COT) τ projections for N viewing angles. An explicit algebraic tomography step then reconstructs the 3D extinction field $\beta _ { \mathrm { e x t } } ( x , y , z )$ from those projections. The neural network therefore solves only the nonlinear radiance-to-COT problem, while the vertical structure remains constrained by the multi-angle observation geometry. Our contributions are threefold: 1) we introduce a hybrid framework in which ML retrieves physically interpretable COT projections and an explicit tomographic solver reconstructs 3D cloud volumes, 2) vertical structure is constrained by passive multi-angle geometry rather than learned from 3D labels or extrapolated from vertical profiles, 3) we show that this decomposition reduces reconstruction time by > 500× relative to iterative radiance-space tomography while retaining comparable accuracy. This provides a step toward scalable, climate-relevant 3D cloud observations that support studies of cloud processes, 3D radiative effects, and evaluation of high-resolution atmospheric models.

## 2 Data & Methodology

Synthetic cloud scene and multi-view radiance. We demonstrate HybridCT using synthetic observations from large-eddy simulations (LES) performed with the System for Atmospheric Modeling [21] for marine subtropical clouds [22, 23]. Each of the six LES scenes spans 80 km×80 km at 40 m horizontal resolution and contains realistic 3D fields of shallow cumulus with varying degrees of convective organization and complexity. Synthetic radiances are generated using the 3D Monte Carlo radiative-transfer solver MYSTIC [24–26] which is part of libRadtran [27, 28], using $1 0 ^ { 3 }$ photons per pixel. The simulated cloud radiance field accounts for a realistic atmospheric background in a maritime environment [29–31] and ocean-surface reflection assuming a wind speed of 10 m/s [32, 33]. The illumination geometry assumes a sun-synchronous satellite orbit with an equator crossing time in the local AM and the sun in the South-East. This translates to a solar azimuth angle of $3 2 0 ^ { \circ }$ and a solar zenith angle (SZA) ranging from $2 0 ^ { \circ }$ to 50<sup>◦</sup>. We simulate the nine viewing directions of the Multiangle Imaging Spectroradiometer (MISR) [34] between $\pm 7 0 ^ { \circ }$ at 670 nm, retaining the 40 m model-grid resolution. All data used in this work are publicly available on Zenodo [35]. The HybridCT codebase is available at https://github.com/lforster/hybridCT and [36]. Version v4.1.2 of AT3D is preserved in [37] and is publicly available at https://github.com/CloudTomography/AT3D.

![](images/90030fc5e9070e6d382eb7237b9379a7c37787d028950ef50d02821602c0b03e.jpg)

![](images/41abaf10a8ea7c56f7fc8f5b625078e7628ea0f3bf63923c1625014845a947e8.jpg)

![](images/9853fdf48b3f56f7f2d9d6843cb5ccec4f4eb395c06388a35dae6d7e374e7e2b.jpg)

![](images/e72ff255316d9e74f5ee28abb1bdff968ee412bddf1815cc951a661231dd0f09.jpg)  
Figure 2: (a) True Cloud optical thickness (COT) field, and true minus predicted COT for (b) U-Net, (c) 1D look-up table (LUT), and (d) Random Forest regression for the nadir view $\left( \mathrm { V Z A } = 0 ^ { \circ } \right)$ . Scatter plots show correlation coefficient and linear fit slope.

Radiance-to-COT retrieval. For each viewing angle, a convolutional U-Net [38] emulates the mapping of 2D radiance to COT fields. We train a separate emulator per viewing angle for flexibility when some or different perspectives are available. Using spatial context and directional information, the network captures 3D radiative effects that 1D look-up tables (LUTs) cannot represent, resulting in higher correlations, reduced bias, and more consistent performance [39]. For training, we split the large cloud and radiance scenes into overlapping square tiles. Hyperparameter tuning with Python’s ray module [40] resulted in a 16-pixel tile size, 2-pixel tile stride, learning rate of 0.01, and MinMax scaler. With a train-test split of 70% to 30%, final training over 96,605 samples × 9 views took about four hours on one GPU with Adam [41]. One of the six LES scenes was held out from training and tuning for final evaluation, and all results reported here are based on this held-out scene. Figure 2 compares predicted with true COT for the U-Net (b), 1D LUT (c), and Random Forest (d). Random Forest regression captures the nonlinear radiance-COT relationship but lacks spatial context and remains strongly underestimated at high COT (R = 0.96 and slope = 0.30). The 1D LUT, used operationally [42, 43], assumes horizontally homogeneous clouds and neglects 3D radiative effects, resulting in low correlation $( R = 0 . 3 7 )$ and strong underestimation (slope = 0.38) for broken clouds [44]. The U-Net provides the best overall agreement, combining high correlation $( R = 0 . 8 5 )$ with a slope closer to unity (0.70). The learned task is deliberately restricted to retrieving 2D path-integrated COT fields, which then serve as projection data for tomography.

Multi-view tomography. Optical thickness along ray i is defined as $\begin{array} { r } { \pmb { \tau } _ { i } = \int _ { \mathrm { L O S } , i } \beta _ { \mathrm { e x t } } ( \mathbf { r } ) d s } \end{array}$ , with the cloud extinction coefficient $\beta _ { \mathrm { e x t } }$ , the spatial position r , and the measurement-ray index i. After discretization, the forward problem becomes ${ \boldsymbol { \tau } } = \mathbf { F } { \boldsymbol { \beta } } _ { \mathrm { e x t } } ,$ , where $\tau$ is the vector of observed optical thicknesses, $\beta _ { \mathrm { e x t } }$ is the voxelized 3D extinction field, and F represents the multi-angle ray-voxel geometry. HybridCT solves this inverse problem using the Simultaneous Algebraic Reconstruction Technique (SART) [45], iteratively updating $\beta _ { \mathrm { e x t } } ^ { ( k + 1 ) } = \beta _ { \mathrm { e x t } } ^ { ( k ) } + \lambda \mathbf { B } [ \tau - \mathbf { F } \beta _ { \mathrm { e x t } } ^ { ( k ) } ]$ , while enforcing positivity and a prescribed cloud mask, which here is taken from the ground-truth cloud field. Here, k denotes the iteration index, F and B denote the forward-projection and normalized back-projection operators, respectively, and the relaxation parameter λ is set to 0.15. Thus, ML estimates the projection quantities, while the volumetric structure is constrained geometrically rather than learned.

## 3 Results & Conclusions

We first evaluate HybridCT for a shallow cloud approx. 0.9 km across and 1.2 km deep within a larger LES cloud field and compare the result with the physics-based baseline model AT3D (Figure 3). This model reconstructs 3D directly from multi-angle radiances using a 3D radiative-transfer forward operator [11, 46]. We also tried to compare our results to VIPCT as the closest ML baseline, but their trained checkpoint did not generalize to the viewing and rendering geometries of our data. HybridCT reconstructs extinction with RMSE = 13.29 km<sup>−1</sup>, $R = 0 . 7 7 ,$ a 0.74 slope, while retaining second-scale runtime. AT3D reaches $\mathrm { R M S E } = \mathrm { 2 2 . 6 9 k m ^ { - 1 } }$ and $R \approx 0 . 5 8$ with a 0.63 slope after 30 min. Figure 4 compares HybridCT and AT3D over the full cloud field. HybridCT (a) reconstructs more than 6300 clouds in under 6 h with $R = 0 . 4 0$ and slope = 0.37, whereas AT3D (c) reconstructs 20 randomly selected clouds in 18 h with lower correlation $( R = 0 . 2 6 )$ and $\mathrm { s l o p e } = 0 . 1 6$ . Figure 4b shows results for HybridCT in the theoretical case with error-free COT prediction. A similar experiment with the cloud in Figure 3 reduced the RMSE from 13.3 to $6 . 2 \mathrm { k m ^ { - 1 } }$ while isolating the target cloud from neighboring clouds further reduced it to $3 . 7 6 \mathrm { k m ^ { - 1 } }$ (not shown here). This highlights two main error sources for both HybridCT and AT3D: (1) the conversion from radiance to COT and (2) contributions from neighboring clouds that intersect with oblique viewing rays and contaminate individual projections.

![](images/6afdda8df2f5c8b65fff5c8216fc48a3699466c5ade37d19eaf7c4005f9abdc6.jpg)  
Figure 3: HybridCT (a–f) vs. AT3D (g–l) tomographic reconstruction of cloud extinction.

In summary, HybridCT differs from fully learned multi-view tomography [19, 47, 48] by breaking the problem into two components: The neural network predicts only COT projections, while the 3D extinction field must satisfy an explicit multi-view geometric projection operator. This restricted role for machine learning makes the source of vertical information physically transparent, and ensures ML is only used where it is most useful. Moreover, since HybridCT’s tomographic reconstruction is geometrically constrained rather than learned, it transfers directly to other datasets (e.g., the LES clouds from VIPCT [48]) without retraining or distribution-shift issues that might arise from applying models to unseen data. The climate relevance follows two main pathways. First, cloud geometry influences how radiation propagates through and around heterogeneous cloud fields. Observations and 3D radiative-transfer studies show that neglecting this structure can produce substantial radiative errors, particularly for broken shallow clouds [49–51]. Retrieved 3D extinction and geometry therefore provide direct observational constraints for studying and evaluating these effects. Second, high-resolution atmospheric models increasingly resolve cloud structures and processes that remain parameterized in coarser models, creating a need for observations against which those simulations are evaluated [52, 53].

While multi-angle cloud tomography offers a route toward volumetric cloud observations, the computational cost of repeated 3D radiative-transfer calculations has so far prevented its application to large, realistic cloud scenes [8, 9, 11, 12]. HybridCT directly targets this computational barrier by moving the expensive radiative inversion into a learned preprocessing step while retaining an explicit geometric reconstruction. The latter runs easily on the CPU of a personal computer and is orders of magnitude faster than comparable physics-based models. Important limitations remain. The present study uses synthetic observations from one cloud regime, the U-Net is trained for MISR-like viewing geometry and a limited range of environmental conditions, and COT does not preserve all information contained in the original radiances. Real-data application will require broader training, observation-based 3D cloud masking, and independent validation with active and in situ observations. The static-scene assumption generally requires near-simultaneous multi-angle acquisition which will be achieved by next-generation satellite constellations such as CloudCT [54]. HybridCT demonstrates that learned radiance inversion can be combined with explicit passive tomography to reduce reconstruction from hours to seconds while retaining useful 3D fidelity. This opens a pathway toward volumetric retrievals at regional to global scales, providing observations of internal extinction structure and cloud organization for studying 3D radiative effects and evaluating cloud-resolving and climate models.

Reconstructed Extinction Volumes, Cloud Scene  
![](images/932ff84da0b0d72e283d492276899359cf736400e6b23a95eb2e3e25b42ba771.jpg)

![](images/34a7cb866a60260529c97bddac72177ac2ae73241b759865f836a6ef88803fc4.jpg)

![](images/4c77df4035112ca25bf6d9277701bf114d659d4602a1336dd31860897d3ea2b6.jpg)  
Figure 4: Reconstruction of entire cloud scene showing true vs. reconstructed extinction with correlation coefficient R and slope of the median prediction in each bin (solid gray curve) with 25th–75th percentile range (shaded) for HybridCT with (a) predicted, (b) true COT, and (c) AT3D.

## Acknowledgments and Disclosure of Funding

The research was carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration (80NM0018D0004). The High Performance Computing resources used in this investigation were provided by funding from the JPL Information and Technology Solutions Directorate. The authors acknowledge the Texas Advanced Computing Center (TACC) at The University of Texas at Austin for providing computational resources that have contributed to the research results reported within this paper https://www.tacc.utexas.edu. We thank Aviad Levis, Katherine L. Bouman, and Steffen Mauceri for fruitful discussions.

## References

[1] Piers Forster, Trude Storelvmo, Kyle Armour, William Collins, Jean-Louis Dufresne, David Frame, Daniel J. Lunt, Thorsten Mauritsen, Matthew D. Palmer, Masahiro Watanabe, Martin Wild, and Hua Zhang. The Earth’s Energy Budget, Climate Feedbacks, and Climate Sensitivity. In Valérie Masson-Delmotte, Panmao Zhai, Anna Pirani, Sarah L. Connors, Clotilde Péan, Sophie Berger, Nada Caud, Yang Chen, Leah Goldfarb, Melissa I. Gomis, Mengtian Huang, Katherine Leitzell, Elisabeth Lonnoy, J. B. Robin Matthews, Tom K. Maycock, Tim Waterfield, Özge Yelekçi, Rong Yu, and Baiquan Zhou, editors, Climate Change 2021: The Physical Science Basis. Contribution ofWorking Group I to the Sixth Assessment Report ofthe Intergovernmental Panel on Climate Change, pages 923–1054. Cambridge University Press, Cambridge, United Kingdom and New York, NY, USA, 2021.

[2] Dennis L. Hartmann, Maureen E. Ockert-Bell, and Marc L. Michelsen. The effect of cloud type on Earth’s energy balance: Global analysis. Journal ofClimate, 5(11):1281 – 1304, 1992.

[3] Louise Nuijens and A. Pier Siebesma. Boundary layer clouds and convection over subtropical oceans in our current and in a warmer climate. Current Climate Change Reports, 5(2):80–94, 2019.

[4] Mark J. Webb, Adrian P. Lock, and Tomoo Ogura. What are the main causes of positive subtropical low cloud feedbacks in climate models? Journal of Advances in Modeling Earth Systems, 16(1):e2023MS003716, 2024.

[5] S. C. Sherwood, M. J. Webb, J. D. Annan, K. C. Armour, P. M. Forster, J. C. Hargreaves, G. Hegerl, S. A. Klein, K. D. Marvel, E. J. Rohling, M. Watanabe, T. Andrews, P. Braconnot, C. S. Bretherton, G. L. Foster, Z. Hausfather, A. S. von der Heydt, R. Knutti, T. Mauritsen, J. R. Norris, C. Proistosescu, M. Rugenstein, G. A. Schmidt, K. B. Tokarska, and M. D. Zelinka. An assessment of earth’s climate sensitivity using multiple lines of evidence. Reviews of Geophysics, 58(4):e2019RG000678, 2020. e2019RG000678 2019RG000678.

[6] Marcin J. Kurowski, Joao Teixeira, Chi Ao, Shannon Brown, Anthony B. Davis, Linda Forster, Kuo-Nung Wang, Matthew Lebsock, Mary Morris, Vivienne Payne, Mark T. Richardson, Richard Roy, David R. Thompson, and Robert C. Wilson. Synthetic observations of the planetary boundary layer from space: A retrieval observing system simulation experiment framework. Bulletin ofthe American Meteorological Society, 104(11):E1999 – E2022, 2023.

[7] Linda Forster, Mark T. Richardson, Anthony B. Davis, Mauricio A. Hess-Flores, Igor Yanovsky, Jacqueline Ryan, Matthew D. Lebsock, Derek J. Posselt, Felix C. Seidel, Chi O. Ao, Shannon T. Brown, David J. Diner, Evan F. Fishbein, Larry Di Girolamo, Marcin J. Kurowski, Lukas Mandrake, Charles E. Miller, Nicholas C. Parazoo, Graeme L. Stephens, Joao Teixeira, Jason J. Hyon, and Michael S. Seablom. Scientific applications of multiangle measurements to reconstruct earth’s atmosphere through tomography. Bulletin ofthe American Meteorological Society, 107(8):E1723 – E1745, 2026.

[8] Aviad Levis, Yoav Y Schechner, Amit Aides, and Anthony B Davis. Airborne three-dimensional cloud tomography. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 3379–3387, 2015.

[9] Aviad Levis, Yoav Y Schechner, and Anthony B Davis. Multiple-scattering microphysics tomography. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6740–6749, 2017.

[10] Aviad Levis, Yoav Y Schechner, Anthony B Davis, and Jesse Loveridge. Multi-view polarimetric scattering cloud tomography and retrieval of droplet size. Remote Sensing, 12(17):2831–, 2020.

[11] Jesse Loveridge, Aviad Levis, Larry Di Girolamo, Vadim Holodovsky, Linda Forster, Anthony B Davis, and Yoav Y Schechner. Retrieving 3D distributions of atmospheric particles using Atmospheric Tomography with 3D Radiative Transfer–Part 1: Model description and jacobian calculation. Atmospheric Measurement Techniques, 16(7):1803–1847, 2023.

[12] Jesse Loveridge, Aviad Levis, Larry Di Girolamo, Vadim Holodovsky, Linda Forster, Anthony B Davis, and Yoav Y Schechner. Retrieving 3D distributions of atmospheric particles using Atmospheric Tomography with 3D Radiative Transfer–Part 2: Local optimization. Atmospheric Measurement Techniques, 16(16):3931–3957, 2023.

[13] Sarah Brüning, Stefan Niebler, and Holger Tost. Artificial intelligence (ai)-derived 3d cloud tomography from geostationary 2d satellite data. Atmospheric Measurement Techniques, 17:961–978, 2024.

[14] Kai Jeggle, Mikolaj Czerkawski, Federico Serva, Bertrand Le Saux, David Neubauer, and Ulrike Lohmann. IceCloudNet: 3D reconstruction of cloud ice from Meteosat SEVIRI. 2024.

[15] Stella Girtsou, Emiliano Diaz Salas-Porras, Lilli Freischem, Joppe Massant, Kyriaki-Margarita Bintsi, Guiseppe Castiglione, William Jones, Michael Eisinger, Emmanuel Johnson, and Anna Jungbluth. 3D cloud reconstruction through geospatially-aware masked autoencoders, 2025.

[16] Yael Sde-Chen, Yoav Y Schechner, Vadim Holodovsky, and Eshkol Eytan. 3D scattering tomography by deep learning with architecture tailored to cloud fields. December 2020.

[17] Roi Ronen, Vadim Holodovsky, and Yoav Y Schechner. Variable imaging projection cloud scattering tomography. IEEE Trans. Pattern Anal. Mach. Intell., 47(8):6097–6109, August 2025.

[18] Inbal Kom-Betzer, Roi Ronen, Vadim Holodovsky, Yoav Y Schechner, and Ilan Koren. NeMF: Neural microphysics fields. IEEE Trans. Pattern Anal. Mach. Intell., 47(9):7218–7230, September 2025.

[19] Roi Ronen, Ilan Koren, Aviad Levis, Eshkol Eytan, Vadim Holodovsky, and Yoav Y Schechner. Learned 3D volumetric recovery of clouds and its uncertainty for climate analysis. arXiv preprint arXiv:2403.05932, 2024.

[20] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. 2020.

[21] Marat F. Khairoutdinov and David A. Randall. Cloud resolving modeling of the ARM Summer 1997 IOP: Model formulation, results, uncertainties, and sensitivities. Journal ofthe Atmospheric Sciences, 60(4):607 – 625, 2003.

[22] Robert M. Rauber, Bjorn Stevens, Harry T. Ochs, Charles Knight, B. A. Albrecht, A. M. Blyth, C. W. Fairall, J. B. Jensen, S. G. Lasher-Trapp, O. L. Mayol-Bracero, G. Vali, J. R. Anderson, B. A. Baker, A. R. Bandy, E. Burnet, J.-L. Brenguier, W. A. Brewer, P. R. A. Brown, R Chuang, W. R. Cotton, L. Di Girolamo, B. Geerts, H. Gerber, S. Göke, L. Gomes, B. G. Heikes, J. G. Hudson, P. Kollias, R. R Lawson, S. K. Krueger, D. H. Lenschow, L. Nuijens, D. W. O’Sullivan, R. A. Rilling, D. C. Rogers, A. P. Siebesma, E. Snodgrass, J. L. Stith, D. C. Thornton, S. Tucker, C. H. Twohy, and P. Zuidema. Rain in shallow cumulus over the ocean: The RICO campaign. Bulletin of the American Meteorological Society, 88(12):1912 – 1928, 2007.

[23] Margreet C van Zanten, Bjorn Stevens, Louise Nuijens, A Pier Siebesma, A. S. Ackerman, F. Burnet, A. Cheng, F. Couvreux, H. Jiang, M. Khairoutdinov, Y. Kogan, D. C. Lewellen, D. Mechem, K. Nakamura, A. Noda, B. J. Shipway, J. Slawinska, S. Wang, and A. Wyszogrodzki. Controls on precipitation and cloudiness in simulations of trade-wind cumulus as observed during RICO. Journal of Advances in Modeling Earth Systems, 3(2), 2011.

[24] B. Mayer. Radiative transfer in the cloudy atmosphere. European Physical Journal Conferences, 1:75–99, 2009.

[25] R. Buras and B. Mayer. Efficient unbiased variance reduction techniques for Monte Carlo simulations of radiative transfer in cloudy atmospheres: The solution. J. Quant. Spectrosc. Radiat. Transfer, 112(3):434– 447, 2011.

[26] C. Emde, R. Buras, and B. Mayer. ALIS: An efficient method to compute high spectral resolution polarized solar radiances using the Monte Carlo approach. J. Quant. Spectrosc. Radiat. Transfer, 112(10):1622–1631, 2011.

[27] B. Mayer and A. Kylling. Technical note: The libradtran software package for radiative transfer calculations - description and examples of use. Atmospheric Chemistry and Physics, 5(7):1855–1877, 2005.

[28] C. Emde, R. Buras-Schnell, A. Kylling, B. Mayer, J. Gasteiger, U. Hamann, J. Kylling, B. Richter, C. Pause, T. Dowling, and L. Bugliaro. The libradtran software package for radiative transfer calculations (version 2.0.1). Geoscientific Model Development, 9(5):1647–1672, 2016.

[29] G.P. Anderson, S.A. Clough, F.X. Kneizys, J.H. Chetwynd, and E.P. Shettle. AFGL Atmospheric Constituent Profiles (0–120 km). Technical Report AFGL-TR-86-0110, AFGL (OPI), Hanscom AFB, MA 01736, 1986.

[30] J. Gasteiger, C. Emde, B. Mayer, R. Buras, S.A. Buehler, and O. Lemke. Representative wavelengths absorption parameterization applied to satellite channels and spectral bands. Journal of Quantitative Spectroscopy and Radiative Transfer, 148:99–115, 2014.

[31] M. Hess, P. Koepke, and I. Schult. Optical properties of aerosols and clouds: the software package OPAC. Bull. Amer. Meteor. Soc., 79(5):831–844, 1998.

[32] C. Cox and W. Munk. Measurement of the roughness of the sea surface from photographs of the sun’s glitter. Journal ofthe Optical Society ofAmerica, 44(11):838–850, 1954.

[33] C. Cox and W. Munk. Statistics of the sea surface derived from sun glitter. Journal ofMarine Research, 13:198–227, 1954.

[34] D. J. Diner, J. C. Beckert, T. H. Reilly, C. J. Bruegge, J. E. Conel, R. A. Kahn, J. V. Martonchik, T. P. Ackerman, R. Davies, S. A. W. Gerstl, H. R. Gordon, J. Muller, R. B. Myneni, P. J. Sellers, B. Pinty, and M. M. Verstraete. Multi-angle Imaging SpectroRadiometer (MISR) instrument description and experiment overview. IEEE Transactions on Geoscience and Remote Sensing, 36:1072–1087, July 1998.

[35] Linda Forster, Nicholas LaHaye, Marcin Kurowski, and Anthony B. Davis. Hybridct dataset – ground truth, training, predicted data, and model parameters, 2025.

[36] Linda Forster and Nicholas LaHaye. HybridCT, December 2025.

[37] Jesse Loveridge, Aviad Levis, Amit Aides, Linda Forster, and Vadim Holodovsky. Atmospheric tomography with 3d radiative transfer, August 2022.

[38] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation, 2015.

[39] V. Nataraja, S. Schmidt, H. Chen, T. Yamaguchi, J. Kazil, G. Feingold, K. Wolf, and H. Iwabuchi. Segmentation-based multi-pixel cloud optical thickness retrieval using a convolutional neural network. Atmospheric Measurement Techniques, 15(17):5181–5205, 2022.

[40] Richard Liaw, Eric Liang, Robert Nishihara, Philipp Moritz, Joseph E Gonzalez, and Ion Stoica. Tune: A research platform for distributed model selection and training. arXiv preprint arXiv:1807.05118, 2018.

[41] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014.

[42] T. Nakajima and M.D. King. Determination of the optical thickness and effective particle radius of clouds from reflected solar radiation measurements. Part I: Theory. J. Atmos. Sci., 47:1878–1893, 1990.

[43] S. Platnick, M.D. King, S.A. Ackerman, W.P. Menzel, B.A. Baum, J.C. Riedi, and R.A. Frey. The MODIS cloud products: algorithms and examples from Terra. IEEE Transactions on Geoscience and Remote Sensing, 41(2):459–473, 2003.

[44] H.W. Barker and D. Liu. Inferring optical depth of broken clouds from Landsat data. Journal ofClimate, 8:2620–2630, 1995.

[45] Anders H Andersen and Avinash C Kak. Simultaneous Algebraic Reconstruction Technique (SART): A superior implementation of the ART algorithm. Ultrasonic Imaging, 6(1):81–94, 1984.

[46] K Franklin Evans. The spherical harmonics discrete ordinate method for three-dimensional atmospheric radiative transfer. Journal ofthe Atmospheric Sciences, 55(3):429–446, 1998.

[47] Yael Sde-Chen, Yoav Y. Schechner, Vadim Holodovsky, and Eshkol Eytan. 3DeepCT: Learning volumetric scattering tomography of clouds. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 5651–5662, 2021.

[48] Roi Ronen, Vadim Holodovsky, and Yoav Y Schechner. Variable imaging projection cloud scattering tomography. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–12, 2022.

[49] Jake J. Gristey, Graham Feingold, Ian B. Glenn, K. Sebastian Schmidt, and Hong Chen. Surface solar irradiance in continental shallow cumulus fields: Observations and large-eddy simulation. Journal ofthe Atmospheric Sciences, 77(3):1065–1088, 2020.

[50] Najda Villefranque and Robin J. Hogan. Evidence for the 3D radiative effects of boundary-layer clouds from observations of direct and diffuse surface solar fluxes. Geophysical Research Letters, 48(14):e2021GL093369, 2021.

[51] Clare E. Singer, Ignacio Lopez-Gomez, Xiyue Zhang, and Tapio Schneider. Top-of-atmosphere albedo bias from neglecting three-dimensional cloud radiative effects. Journal of the Atmospheric Sciences, 78(12):4053–4069, 2021.

[52] Tapio Schneider, João Teixeira, Christopher S. Bretherton, Florent Brient, Kyle G. Pressel, Christoph Schär, and A. Pier Siebesma. Climate goals and computing the future of clouds. Nature Climate Change, 7(1):3–5, 2017.

[53] Tapio Schneider, Shiwei Lan, Andrew Stuart, and João Teixeira. Earth system modeling 2.0: A blueprint for models that learn from observations and targeted high-resolution simulations. Geophysical Research Letters, 44(24):12396–12417, 2017.

[54] Klaus Schilling, Yoav Y. Schechner, and Ilan Koren. CloudCT—computed tomography of clouds by a small satellite formation. In Proceedings of the 12th IAA Symposium on Small Satellites for Earth Observation, volume 6, Berlin, Germany, 2019. International Academy of Astronautics.