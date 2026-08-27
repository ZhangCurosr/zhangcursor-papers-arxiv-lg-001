Graphical Abstract

Energy Yield and Lifetime Climate Classification via Machine Learning for Optimizing Photovoltaic Module Design and Materials

Youri Blom, Sofia Dutto, Alexandru Costache, Rowan Richie, Ruben Pelsser, Wesley Berger, Jing Sun, Rudi Santbergen, Olindo Isabella, Malte Ruben Vogt

![](images/9c1eb6d9fc0b0226a5aa6cad1b31499a68afe3b410daa5d8c0b6781514034a03.jpg)

Classification  
![](images/6d2edb950905e7fc81eb885c332a264a599c1d7b9bda1da4199f44bda5826da8.jpg)

## Highlights

Energy Yield and Lifetime Climate Classification via Machine Learning for Optimizing Photovoltaic Module Design and Materials

Youri Blom, Sofia Dutto, Alexandru Costache, Rowan Richie, Ruben Pelsser, Wesley Berger, Jing Sun, Rudi Santbergen, Olindo Isabella, Malte Ruben Vogt

• A climate classification is created covering both energy yield and lifetime aspects

• Annual GHI and mean ambient temperature are the most important features

• Six main clusters and fifteen subclusters are found to be optimal

• The low-temperature continental climate has the highest lifetime energy yield

# Energy Yield and Lifetime Climate Classification via Machine Learning for Optimizing Photovoltaic Module Design and Materials

Youri Blom<sup>a,∗</sup>, Sofia Dutto<sup>b</sup>, Alexandru Costache<sup>a</sup>, Rowan Richie<sup>a</sup>, Ruben Pelsser<sup>a</sup>, Wesley Berger<sup>a</sup>, Jing Sun<sup>a</sup>, Rudi Santbergen<sup>a</sup>, Olindo Isabella<sup>a</sup>, Malte Ruben Vogt<sup>a,∗∗</sup>

<sup>a</sup>Delft University of Technology, Mekelweg 4, Delft, 2628CD, The Netherlands <sup>b</sup>University of Naples Federico II, Via Claudio, 21, Naples, 80125, Italy

## Abstract

To resiliently and sustainably meet our future energy demand, photovoltaic (PV) modules must be deployed across a broad and diverse range of geographical regions with varying operating conditions. As these conditions strongly afect both performance and optimal system design, a dedicated PV-specific climate classification can be of great use. In this work, we develop a climate classification framework tailored to PV applications using a variety of machine learning (ML) techniques. Building on previous studies, our approach incorporates both energy yield, and for the first time, also the module lifetime with climate dependent degradation. We generate an interpolated dataset containing twelve input features and two target variables (i.e. energy yield and module lifetime). Feature importance analysis shows that annual global horizontal irradiation and ambient temperature are the most influential predictors. The most accurate regression model achieves root mean square errors (RMSE) of 0.007 MW h for energy yield and 1.5 years for lifetime prediction. The calculated feature importance scores are then integrated into a hierarchical clustering framework, resulting in 6 primary climate clusters—Tropical, Desert, Continental, Temperate, Boreal, and Polar—and 15 corresponding subclusters. Our analysis shows that the low-temperature continental climate ofers the highest discounted lifetime energy yield. These results can

support a wide range of applications, including PV module optimization, system siting decisions, and comparative performance studies.

Keywords: PV Climate classification, Machine Learning, Energy yield, Lifetime

## 1. Introduction

To meet future energy demand in a resilient and sustainable way, the global photovoltaic (PV) capacity must grow at an annual rate of 25% over the next decade [1]. As a result, PV modules will need to be deployed across an increasingly diverse set of geographical regions, each characterized by widely varying operating conditions. These varying operating conditions strongly influence the performance and behavior of PV modules, afecting both their operating eficiency [2, 3] and degradation rates [4, 5]. Consequently, PV manufacturers have already started customizing the bill of materials (BoM) for modules intended for diferent environmental conditions [6, 7]. It has also been demonstrated that the optimal cell design—such as wafer thickness in crystalline silicon (c-Si) cells [8] or the bandgap energy of the perovskite top cell in perovskite/silicon tandems [9]—varies across locations and climate types. Similarly, the International Energy Agency PV Power Systems program has published specific guidelines for operation and maintenance practices tailored to diferent climate zones [10].

To simplify the task of customizing solar cell and BoM designs for specific locations, a climate classification can be developed that groups regions with similar operating conditions. Such a classification would reduce the problem from potentially designing a PV module for each location to only a limited number of module types, each optimized for a particular climate zone. Several PV-specific climate classifications have already been proposed in the literature. Dash et al. [11] divided India into 7 climatic zones based on annual daytime irradiance and ambient temperature. The most widely used global climate classification is the Koppen–Geiger (KG) system, originally proposed¨ by Kottek et al. [12], which divides the world into 6 main climate types and 14 sub-climates. Because the KG classification is based solely on ambient temperature and precipitation, several studies have extended it to include solar irradiance. Ascencio-Vásquez et al. [13] developed the Koppen–Geiger Pho-¨ tovoltaic (KGPV) classification, resulting in 12 PV-relevant climate zones. Similarly, Skandalos et al. [14] adapted the KG system for building-integrated

PV applications.

Although these extended climate classifications represent meaningful improvements over the original KG system, several parameters relevant to PV performance—such as wind speed and ultraviolet irradiance—are still not included. Moreover, because solar energy was not the focus of the original KG classification, the derived PV-specific adaptations may not be fully optimal. To address these limitations, Triana de las Heras et al. [15] developed a new PV climate classification map using a range of Machine Learning (ML) techniques. The use of ML in PV-related research has increased rapidly in recent decades [16, 17], largely due to its strong capability for handling large datasets and identifying complex patterns [18]. In those works, a regressionbased ML model is first employed to determine the most influential environmental variables (features) for predicting energy yield at global scale. These key features are then used in a clustering algorithm to group locations with similar characteristics. While this approach is promising, the study relies on relatively simple ML methods—such as linear regression and k-means clustering. In addition, the resulting climate classification focuses solely on energy yield and does not incorporate lifetime performance or degradation behavior.

This work builds on the approach of Triana de las Heras et al. by developing a ML-based climate classification that focuses on both the energy yield and lifetime performance of c-Si modules. By combining ML methods of varying complexity, the most relevant environmental parameters are identified and subsequently used to cluster locations with similar characteristics. The paper is structured as follows: Section 2 describes the dataset used to construct the climate classification; Section 3 outlines the methodology of this study, covering both the physical models employed to generate the dataset and the ML techniques used in the analysis; Section 4 presents the obtained results, beginning with the accuracy of the ML methods and the resulting feature importance, followed by the final clustering outcomes; lastly, Section 5 summarizes the main findings and ofers concluding remarks.

## 2. Dataset

To develop the climate classification, a global dataset is required that includes both environmental conditions and PV system performance across the world. The Meteonorm database [19] is used to extract hourly environmental data from all available weather stations worldwide. Based on this data, 12 environmental characteristics (features) and two performance metrics (targets) are defined. Note that this database is used instead of the available dataset from Triana de las Heras et al. [15], as hourly data is needed for an accurate calculation of both targets. Because the initial sampling of locations is not uniformly distributed across the Earth, spatial interpolation is applied to correct for geographical imbalances. Finally, the correlations between all features and targets are evaluated.

## 2.1. Features

To describe the hourly environmental conditions, features are defined that are potentially relevant to both energy yield and lifetime performance. Table 1 lists all features considered in this study. The selection is primarily based on the features used by Triana de las Heras et al. [15], supplemented with additional features that have a physical relationship with the chosen targets. The mathematical definitions of these features, as well as their global spatial distributions, are provided in Appendix A. All features are normalized to have zero mean and unit standard deviation to facilitate a better comparison.

Table 1: The features that are considered for the climate classification.
<table><tr><td rowspan=1 colspan=6>Feature</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { T } _ { \mathrm { m e a n } } [</td><td rowspan=1 colspan=2>^ { \circ } \mathrm { C } ] } }$ </td><td rowspan=1 colspan=3></td><td></td></tr><tr><td rowspan=1 colspan=6> $\mathrm { T } _ { \mathrm { m i n } } [ ^ { \circ } \mathrm { C } ]$ </td><td></td></tr><tr><td rowspan=1 colspan=6> $\mathrm { T } _ { \mathrm { m a x } } [ ^ { \circ } \mathrm { C } ]$ </td><td></td></tr><tr><td rowspan=1 colspan=6></td><td></td></tr><tr><td rowspan=1 colspan=6> $\mathrm { T _ { d i f f } } \mathrm { \Pi } ^ { \circ } \mathrm { C } ]$ </td><td></td></tr><tr><td rowspan=1 colspan=6> $\mathrm { G H I _ { a n n } [ M W h m ^ { - 2 } ] }$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=6> $\mathrm { G H I _ { m i n } [ k W h m ^ { - 2 } ] }$ </td><td></td></tr><tr><td rowspan=1 colspan=6> $\mathrm { G H I _ { m a x } [ k W h m ^ { - 2 } ] }$ </td><td></td></tr><tr><td rowspan=1 colspan=6> $\mathrm { D N I _ { s h a r e } [ \% ] }$ </td><td></td></tr><tr><td rowspan=1 colspan=6> $\mathrm { W S } _ { \mathrm { m e a n } } \mathrm { [ m s ^ { - 1 } ] }$ </td><td></td></tr><tr><td rowspan=1 colspan=6> $\mathrm { A M _ { \mathrm { e f f } } [ - ] }$ </td><td></td></tr><tr><td rowspan=1 colspan=2> $\mathrm { R H } _ { \mathrm { m e a</td><td rowspan=1 colspan=2>n } } [ \% ]$ </td><td rowspan=1 colspan=2></td><td></td></tr><tr><td rowspan=1 colspan=4> $\mathrm { U V _ { a n n } [ k W h m ^ { - 2 } ] }$ </td><td rowspan=1 colspan=2></td><td></td></tr></table>

## 2.2. Model targets

The two targets used to evaluate PV module performance are the annual energy yield and module lifetime. Since, to the best of the authors knowledge, no global dataset of measured energy yield and module lifetime exists, physical models are used to generate these quantities. We employ the PVMD Toolbox [20, 21] to simulate the energy yield and lifetime of c-Si modules installed all around the world.

The PVMD Toolbox is a modeling framework for simulating the outdoor performance of PV systems. It consists of sequential steps that simulate diferent aspects of the PV system, resulting in the hourly electricity production [21]. In recent work [22], the PVMD Toolbox has been extended for degradation analysis, allowing for the calculation of the module lifetime. Here, module lifetime is defined as the time at which the module performance decreases to 80% of its initial performance due to degradation efects. As some values for the module lifetime exceeds the commonly expected range of 30-40 years [23], we have performed a validation in Appendix B. This shows that the simulated degradation rates are consistent with field measurements.

These physical models are used to calculate the annual energy yield and module lifetime of c-Si modules at diferent locations across the world, as shown in Figure 1. The module layout and the Standard Test Conditions (STC) characteristics of the considered PV module can be found in the Appendix C. To accelerate the computational time of these simulations, the calculations are carried out on the Delft High Performance Computing Centre [24].

## 2.3. Spatial interpolation

As shown in Figure 1a, the simulated locations are unevenly distributed across the globe. This non-uniformity may introduce bias into the climate classification, as regions with a higher density of data points could disproportionately influence the results. To mitigate this efect, a nearest-neighbor interpolation approach [25] (using three neighbors) is applied to the targets and all features, resulting in a more spatially uniform dataset. Figure 1 presents the geographical datasets after spatial interpolation for the targets. Both the original and interpolated datasets are provided as supplementary material.

## 2.4. Correlation

To obtain an initial indication of feature relevance, Pearson correlation coeficients [26] (R) are computed for all features. Figure 2 displays the correlations between each feature and the two targets: energy yield and module lifetime. For energy yield, $\mathrm { G H I _ { a n n } }$ exhibits the highest correlation, which is expected since it directly represents the total annual solar irradiation incident on the module. For module lifetime, $\mathrm { T } _ { \mathrm { m e a n } }$ shows the strongest correlation, reflecting the strong influence of temperature on degradation rates. Furthermore, $\mathrm { A M _ { e f f } }$ demonstrates a correlation of similar magnitude but opposite sign compared to $\mathrm { T } _ { \mathrm { m e a n } } ,$ which is consistent with the strong inverse relationship between these two features, as shown in Figure 1c.

![](images/8af74cc2a279d698b7e4348c27ad2b66e456d2914b7d9eab0ec7963bc267c2e7.jpg)  
<sub>graphical</sub> <sub>distribution</sub> <sub>of</sub> <sub>the</sub> <sub>simulated</sub> <sub>energy</sub> <sub>yield</sub> <sub>and</sub> m<sup>odule</sup> <sup>lifetime</sup> <sup>at</sup> <sup>the</sup> <sup>loca</sup> <sub>The</sub> <sub>interpolated</sub> <sub>values</sub> <sub>of</sub> <sub>energy</sub> <sub>yield</sub> <sub>and</sub> <sub>module</sub> <sup>lifetime</sup> <sup>for</sup> <sup>all</sup> <sup>coordinates.</sup> <sup>c)</sup> <sup>Th</sup> <sub>all</sub> <sub>fe</sub>a<sup>tur</sup>

![](images/bc328b3bac8c09cbbe31a0fd0d33d21ab6b3fa92eb17ffc7bf460bf969173f45.jpg)

b)  
![](images/fd035960bd67671fb7113bda209d0f218ad6dc9c4496d3b441256ac3d68c8de4.jpg)  
Figure 2: The correlation coeficients between all features and the energy yield (a) and module lifetime (b). Their values are on the right-hand side of both diagrams. The background colors indicate regions of no significant correlation (red), some significant correlation (orange), and strong significant correlation (green).

Although the correlation coeficient ofers direct insight into linear relationships between features and targets, relying solely on correlation for feature relevance has several well-known limitations, such as only accounting for linear trends and the influence of the range of observations [27]. Therefore, more advanced ML-based techniques are employed to determine the feature importance, as described in the next section.

## 3. Methodology

The climate classification is constructed through several methodological steps. An overview of the complete workflow is shown in Figure 3. Following the pre-processing described in the previous section, a feature importance analysis is performed using multiple ML regression methods. These feature importance scores are then used as inputs to a clustering algorithm, which groups locations based on similarities in the most relevant environmental characteristics.

![](images/c383d7854d53d1ea4264ccaf97559c240226cc8c5ccd23e407c403b73c7e568d.jpg)  
Figure 3: An overview of the methodology used for the climate classification. Features and targets are obtained from the operating conditions and the physical models. Feature importance is then determined using ML regression models and subsequently used as input for the clustering procedure.

## 3.1. Calculation of feature importance

To quantify the importance of all features, a brute-force approach is employed, similar to the method proposed by Meinshausen et al. [28]. This approach incorporates five diferent ML regression methods, which are dis cussed later in this section. Using multiple ML methods allows for a more robust assessment of feature relevance, reducing dependence on any single model.

Instead of directly using all available features, the number of included features $( N _ { i } )$ is varied from 1 to the total number of features $( N _ { \mathrm { f e a t } } )$ . For each value of $N _ { i }$ , all possible feature combinations $\left( \binom { N _ { \mathrm { f e a t } } } { N _ { i } } \right)$ in total) are evaluated, and the combination with the lowest error is selected. These selected sets represent the feature combinations that contain the most information for a given value of $N _ { i }$

For each feature $j ,$ , an importance score $S _ { j }$ is defined as

$$
S _ { j } = \sum _ { N _ { i } } ^ { N _ { f e a t } } \sum _ { k } ^ { N _ { m o d e l s } } \frac { \epsilon _ { m a x } } { N _ { i } } \left( \frac { 1 } { \epsilon _ { k } ( N _ { i } ) } - \frac { 1 } { \epsilon _ { k } ( N _ { i } - 1 ) } \right) \cdot K ( j , N _ { i } , k ) ,\tag{1}
$$

where k denotes the ML method, $N _ { m o d e l s }$ is the total number of ML methods, $\epsilon _ { k } ( N _ { i } )$ is the model error of method k when using $N _ { i }$ features, $\epsilon _ { m a x }$ is the maximum error observed for a given target across all models. Including $\epsilon _ { m a x }$ is important as it allows to fairly add the importance score of the diferent targets. The function $K ( j , N _ { i } , k )$ is binary, equaling 1 if feature $j$ is selected by method k at iteration $N _ { i }$ , and 0 otherwise.

Equation 1 can be interpreted as assigning a performance-based boost to each selected feature at iteration $N _ { i }$ for all models. The magnitude of this boost depends on the improvement in model performance relative to the previous iteration $( N _ { i } - 1 )$ and is distributed evenly across all selected features. For completeness, $\epsilon _ { k } ( 0 )$ is defined to be infinite.

## 3.2. Machine learning regression models

Table 2 provides an overview of the ML regression models utilized in this work. The selected models span a range of regression types, including linear, kernel-based, spline-based, probabilistic, and neural network approaches. This diversity enables a robust and model-independent assessment of feature importance.

Table 2: An overview of the utilized ML regression methods that are used in this work.
<table><tr><td>ML method</td><td>Reference</td><td>Modifications compared to default MATLAB set- tings</td></tr><tr><td>Linear regression</td><td>[29]</td><td rowspan="3"></td></tr><tr><td>Support Vector Machine</td><td>[30, 31]</td></tr><tr><td>Multivariate adaptive re-</td><td>[32, 33]</td></tr><tr><td>gression spline (MARS) Neural network</td><td>[34, 35]</td><td>Changed activation func-</td></tr><tr><td>Gaussian process regression</td><td>[31]</td><td>tion to sigmoid [36]</td></tr></table>

For each model, the dataset is split into 85% training data and 15% test data, and the Root Mean Square Error (RMSE) is used as the performance metric. Given the large number of feature combinations $( 2 ^ { 1 2 }$ in total), exhaustive hyperparameter optimization is computationally impractical. Instead, fixed hyperparameter settings are used, with minor adjustments as listed in Table 2.

## 3.3. Clustering technique

The final step in the climate classification is clustering, for which agglomerative hierarchical clustering is employed. Hierarchical clustering is chosen over alternative methods, such as k-means clustering [37], because it enables the construction of nested sub-classifications within larger clusters in a systematic and transparent manner. This hierarchical structure allows the level of classification detail to be easily adjusted, making the approach flexible for diferent applications.

As described by Murtagh et al. [38], pairwise distances between all samples are first computed in the $N _ { \mathrm { f e a t } } { \mathrm { - d i m e n s i o n a l } }$ feature space using a predefined distance metric (here, Euclidean distance). These distances are then used to iteratively merge samples into clusters, resulting in a dendrogram that represents the hierarchical clustering structure. A final classification is obtained by selecting an appropriate cut-of point in this dendrogram.

In hierarchical clustering, diferent linkage methods define how distances between clusters are computed. Common linkage methods include single, complete, average, centroid, median, weighted, and Ward linkage. In this work, the Ward linkage method [39] is employed, as it produced the most balanced cluster structure among the tested linkage methods. Ward linkage defines inter-cluster distances based on the increase in within-cluster variance upon merging clusters and can be expressed in terms of centroid distances under Euclidean assumptions. A comparison of alternative linkage methods is provided in Appendix D. To ensure that diferences in the most influential features contribute more strongly to the clustering, the feature values are weighted by their importance scores $S _ { j }$ from Equation 1 before constructing the dendrogram.

## 4. Results

The methods and dataset described in the previous sections are now applied to construct the climate classification. We begin by examining the accuracy of the ML regression models, followed by the calculation of the feature importance scores. Finally, the clustering results and resulting climate classifications are presented.

## 4.1. Accuracy of the regression models

All methods listed in Table 2 are employed in the brute-force feature-selection approach. Figure 4 shows the accuracy of each method for diferent number of features $N _ { i }$ , along with the selected features at each stage for both targets. As expected, model performance improves as more features become available. For both targets, the RMSE decreases rapidly at low values of $N _ { i }$ and begins to plateau around $N _ { i } = 6$ , indicating that six features capture most of the relevant information. For certain models, the RMSE increases again at $N _ { i } = 1 2$ , suggesting potential overfitting when all features are included. Linear regression and Support Vector Machines yield the highest RMSE values, which is consistent with their limited ability to capture non-linear relationships. The remaining models, which can model non-linear trends, achieve considerably higher accuracy, with Gaussian Process Regression showing the best overall performance, reaching RMSE values of 0.007 MW h and 1.5 year for the energy yield and lifetime target, respectively.

![](images/f2cb1b0a2347461d7d0d88b1831e7da83252d756466412c3969d6d842bafa604.jpg)

![](images/20a0498dc3e344f94629db67e3516822eff01ef5765d54035a4aec250817cfdf.jpg)

c)  
![](images/ed5083791721b7b7adf3930ce1fb09713fcacd065076a1ffcccbf92877a5f3d7.jpg)

d)  
![](images/408958760ad8fd8d945c1d2c62eda4c4d82166cd128be4332157414f74090e33.jpg)  
Figure 4: a) The highest accuracy of the diferent methods with number of selected features $N _ { i }$ using the energy yield as target. b) The selected features by the diferent models for the energy yield as target. A colored box indicates that the method has selected a certain feature. c) The highest accuracy of the diferent methods with number of selected features $N _ { i }$ using the lifetime as target. d) The selected features by the diferent models for the lifetime as target.

Figure 4b) and d) show that some features are utilized more often than others. For the energy yield target, the $\mathrm { G H I _ { a n n } }$ and $\mathrm { A M _ { e f f } }$ are most often selected, suggesting that these features contain the highest information. Similarly, $\mathrm { T } _ { \mathrm { m e a n } }$ and $\mathrm { R H } _ { \mathrm { m e a n } }$ are most frequent selected when the lifetime is used as target. These results will be used to quantify the importance of each feature.

## 4.2. Feature importance

The results from Figure 4 together with Equation 1 are used to compute the importance scores $S _ { j }$ for all features for both targets. Figures 5a and 5b show the resulting feature importance scores for the energy-yield and lifetime targets, respectively. For both targets, the features with the highest scores correspond well with the most frequently selected features shown previously in Figure 4.

![](images/a18f65fe84358000f1aaf2599f3c94ce185f3f308a059f155b34fcd7e09b9164.jpg)

![](images/1cb56ca60fe9e290608c4a70cf4e79eb84a6c2907065a980291d19e0fd8a045a.jpg)

![](images/f3551dec14f6fb3d24b9cab88add0a2cd171edee0147ad5d0c235a5de4805564.jpg)  
Figure 5: The importance score of all features considering the energy yield (a) or lifetime (b) as target. The total feature importance score (c) is obtained by taking the average of the two.

To obtain a unified importance measure that reflects both energy-yield and lifetime considerations, the average of the two target-specific scores is taken, as shown in Figure $5 \mathrm { c } .$ The features $\mathrm { G H I _ { a n n } }$ and $\mathrm { T } _ { \mathrm { m e a n } }$ emerge as the two most influential parameters, each topping the importance ranking for the energy yield and lifetime targets, respectively. These combined scores (Figure 5c) are used as input for the clustering procedure.

## 4.3. Clustering & Classification

The method described in Section 3.3 is applied to construct the climate classification using the feature importance scores from Figure ${ \mathrm { 5 c } } ,$ thereby incorporating both energy-yield and lifetime aspects. It should be noted that using the feature importance scores from either Figure 5a or 5b would allow for classifications focused solely on energy yield or lifetime, respectively.

Corresponding classifications and analyses for these individual targets are provided in Appendix E.

Figure 6a shows the dendrogram constructed from the weighted distance map. While the figure displays only the first 25 clusters for readability, the hierarchical structure continues until each data point forms its own cluster. To derive a practical climate classification, a cut-of threshold must be chosen to determine the number of clusters $( N _ { \mathrm { c l u s t e r } } )$ . However, selecting the optimal number of clusters is not straightforward, as no universally accepted metric exists for this purpose [40].

a)  
![](images/5097f87964ea1135aa2c44cedc1604b504ddb070ff5601d129d687c1c01090a7.jpg)

![](images/99f8691af2ac06a5d9b5fcd3907e469eea51e7437423353d5a63cfdddb7f1f2d.jpg)  
Figure 6: a) The dendrogram that is created based on the distance map and the corresponding cluster labels. b) The relative distance to the next cut-of is plotted for each value of $N _ { c l u s t e r s }$ . The values of $N _ { c l u s t e r s } = 2$ and $N _ { c l u s t e r s } = 3$ (0.51 and 0.23, respectively) are not visible, since the image is cropped to clearly show the meaningful part of the graph. This figure can be used to determine where to cut-of the dendrogram.

In this work, we use the distances in the dendrogram to determine the optimal value of $N _ { c l u s t e r s }$ . Figure 6b shows the relative distance between successive cut-of points for diferent values of $N _ { c l u s t e r s }$ . Several local maxima can be observed; these represent points in the dendrogram where the distance between adjacent cut-ofs is relatively large, indicating potentially meaningful locations at which to divide the hierarchy. Although the edge case of $N _ { \mathrm { c l u s t e r s } } = 2$ exhibits the largest relative distance (0.51), this option is disregarded, as a classification with only two clusters would not provide useful diferentiation.

The first local maximum occurs at $N _ { \mathrm { c l u s t e r s } } = 6$ , which is therefore selected for defining the 6 main climate clusters $( N _ { \mathrm { m a i n } } )$ . To obtain finer granularity, subclusters $( N _ { \mathrm { s u b } } )$ are also introduced. We choose $N _ { \mathrm { s u b } } = 1 5$ , corresponding to the first local maximum for which $N _ { \mathrm { c l u s t e r s } } > 2 \cdot N _ { \mathrm { m a i n } }$ . The resulting two cut-of levels used for classification are indicated by dashed lines in Figure 6. Also, the corresponding cluster labels are plotted.

One thing that should be realized is that the created dendrogram is sensitive to small changes in the computed distances. This means that the number of optimal clusters can change for small changes in the feature importance score, as illustrated in Appendix F. Future research efor can be spent on analyzing the impact of small changes to the classification.

Figure 7 presents the final climate classification using $N _ { \mathrm { m a i n } } = 6$ and $N _ { \mathrm { s u b } } = 1 5$ . The six main climates are labeled Tropical, Desert, Continental, Temperate, Boreal, and Polar, closely aligning with the terminology used in the KGPV classification [13]. The subclusters within each main climate are further distinguished by levels of irradiance (I), temperature (T), or wind speed (W), denoted as low (L) or high (H).

As an example, the Desert climate is subdivided into three zones: low temperature (LT), high temperature (HT), and high irradiance (HI). It should be noted that these subcluster labels express the relative feature values within each main climate. A detailed explanation of the subcluster labeling procedure is provided in Appendix G. Furthermore, an overview comparing this climate classification with previously developed classifications is presented in Appendix H.

To compare the performance of PV systems across the diferent clusters, Figure 8 presents box plots of the discounted Lifetime Energy Yield (LEY) for each climate zone. The discounted LEY assumes a discount rate of 7% [41], and its mathematical definition is given in Appendix I.

The results show that the highest LEY values occur in the temperate-LT climate, where high energy yields coincide with long module lifetimes. Additionally, across all main climates, the LT subclusters consistently outperform their HT counterparts. This trend is expected, as lower temperatures reduce degradation rates and thereby extend module lifetime.

The climate classification in Figure 7, together with the LEY distributions in Figure 8, enables several practical applications. First, the classification can guide optimized cell- or BoM-design strategies, allowing PV manufacturers to tailor module designs to specific climate zones. Second, site-allocation decisions for PV systems can benefit from the classification, as it highlights the regions ofering the highest lifetime performance. Lastly, the classification facilitates fair comparisons between diferent PV technologies. By selecting a representative set of locations, one for each climate zone, global performance comparisons can be conducted without the need to evaluate a large number of sites. To facilitate this comparison, Appendix J provides an overview of representative locations for each climate zone.

![](images/e43ff79fb7cb78c5986a247715045cb32890197ffd4ef988942b38a4cfa69fbd.jpg)  
<sub>imate</sub> <sub>classification</sub> <sub>usingNmain=</sub> <sub>6and</sub>N<sup>sub=</sup> <sup>15.</sup> <sup>The</sup> <sup>main</sup> <sup>climates</sup> <sup>are</sup> <sup>labelle</sup> <sub>ate,</sub> <sub>Boreal,</sub> <sub>and</sub> <sub>Polar.</sub> <sub>Subcluster</sub> <sub>are</sub> <sub>further</sub> <sub>distinguished</sub> <sub>by</sub> <sub>irradiance</sub> <sub>(I),</sub> <sub>te</sub>m<sup>p</sup> <sub>)</sub> <sub>levels,</sub> <sub>denoted</sub> <sub>as</sub> <sub>lo</sub>w <sup>(L)</sup> <sup>and</sup> <sup>h</sup>

![](images/b08fa36cbea7330b66b48b70e365a150712ac0b83366ad309f74a55e4ebe4a81.jpg)  
Figure 8: The box plot distribution of the discounted lifetime energy yield for the diferent climate zones.

## 5. Conclusion

This work presents a climate classification that utilizes multiple ML methods and incorporates both energy-yield and lifetime aspects. First, a dataset is constructed containing twelve environmental features and two performance targets. To correct for the non-uniform geographical distribution of the raw data, spatial interpolation is applied to obtain a uniformly distributed global dataset. A brute-force approach using five diferent ML regression methods is then employed to quantify the importance of each feature, and the resulting feature importance scores are subsequently used as inputs for hierarchical clustering.

The analysis shows that linear regression and Support Vector Machines yield the lowest performance, as both methods are limited to modeling predominantly linear relationships. The gaussian process reaches the highest accuracy, obtaining RMSE values of 0.007 MW h and 1.5 years for the energy yield and lifetime, respectively. Furthermore, the performance of all ML methods saturates once more than six features are included, indicating that the majority of relevant information is captured within the top six features. The most influential features for the energy-yield and lifetime targets are

$\mathrm { G H I _ { a n n } }$ and $\mathrm { T } _ { \mathrm { m e a n } } .$ , respectively.

Based on the dendrogram structure, six main climate clusters and fifteen subclusters are defined, corresponding to the largest relative gaps in the hierarchy. The main climates are labeled Tropical, Desert, Continental, Temperate, Boreal, and Polar, and are further diferentiated by relatively high or low values of temperature, irradiance, or wind speed. Finally, the discounted lifetime energy yield (LEY) is evaluated across all clusters, with the highest values observed in the Temperate-LT climate zone. These results enable various applications, including optimized PV module and BoM design, improved PV system siting decisions, and more eficient comparative studies across PV technologies.

## Declaration of Generative AI and AI-assisted technologies in the writing process

During the preparation of this work the author used CoPilot in order to paraphrase sentences and improve the language and readability. After using this tool/service, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication.

## Supporting Information

The dataset and the software of the methods can be accessed on a 4TU dataset via this link: https://doi.org/10.4121/c7e04a2d-123a-40a9-a 4e7-f260f8f23e5c

## Appendix A. Definition of features

Table 1 provides an overview of all features considered in this work. These features are derived from the hourly weather data obtained from Meteonorm [19], which includes ambient temperature, global horizontal irradiance (GHI), direct normal irradiance (DNI), difuse horizontal irradiance (DHI), sun position, wind speed (WS), and relative humidity (RH).

Most features $( \mathrm { T _ { m e a n } , \mathrm { \Delta T _ { m i n } , \mathrm { \Delta T _ { m a x } , \mathrm { G H I _ { a n n } , \mathrm { G H I _ { m i n } , \mathrm { G H I _ { m a x } , \mathrm { W S _ { m e a n } } } } } } } }$ , and $\mathrm { R H } _ { \mathrm { m e a n } } )$ are obtained simply by taking the annual mean, monthly minimum, monthly maximum, or sum of the corresponding hourly values. However, several features require more complicated definitions.

The feature $\mathrm { T _ { d i f f } }$ represents the average daily temperature fluctuation and is defined as

$$
\mathrm { T _ { d i f f } } = \frac { \sum _ { i } ^ { N _ { d a y s } } \left( \operatorname* { m a x } ( T _ { a m b } ( t \in h o u r s _ { i } ) ) - \operatorname* { m i n } ( T _ { a m b } ( t \in h o u r s _ { i } ) ) \right) } { N _ { d a y s } } ,\tag{A.1}
$$

where $N _ { d a y s }$ are the number of days in the year, $T _ { a m b } ( t )$ is the ambient temperature at hour t, hours<sub>i</sub> represent all hours within day i.

$\mathrm { D N I _ { s h a r e } }$ is defined as the share of direct irradiance contributing to the total irradiance, which is defined as

$$
\mathrm { D N I } _ { \mathrm { s h a r e } } = \frac { \sum _ { t } \left( D N I ( t ) \cdot \cos ( \phi _ { s u n } ( t ) ) \right) } { \sum _ { t } G H I ( t ) } ,\tag{A.2}
$$

where $D N I ( t ) , G H I ( t )$ and $\phi _ { s u n } ( t )$ are direct normal irradiance, global horizontal irradiance, and sun altitude, respectively, at time t.

$\mathrm { A M _ { \mathrm { e f f } } }$ is the efective air mass at which the irradiance is received, which is defined as

$$
{ \mathrm { A M } } _ { \mathrm { e f f } } = { \frac { \sum _ { t } { \left( { \frac { 1 } { \cos ( \phi _ { s u n } ( t ) ) } } \cdot G H I ( t ) \right) } } { \sum _ { t } G H I ( t ) } }\tag{A.3}
$$

Finally, $\mathrm { U V } _ { \mathrm { a n n } }$ denotes the annual received ultraviolet irradiance. This quantity is computed directly by the PVMD Toolbox and corresponds to all incident irradiance with wavelengths below 400 nm.

## Appendix B. Validation lifetime calculation

Figure 1 shows the geographical distribution of the simulated energy yield and module lifetime. In some locations, the simulated lifetime exceeds the commonly expected module lifetime of 30/40 years. These high values arise from the definition used in this study, where lifetime corresponds to the time required for the module performance to decrease to 80% of its initial value [22]. Because temperature is the dominant stress factor influencing degradation, regions with consistently low temperatures, such as Nordic countries, naturally exhibit lower degradation rates and therefore longer lifetimes. Psimopoulos et al. [42] report that PV modules in Sweden retain more than 90% of their initial performance after 30 years. Linearly extrapolating these results suggests that reaching the 80% threshold could indeed require around 60 years.

To obtain a broader validation, reported degradation rates from the global compendium compiled by Jordan et al. [5] are used for comparison. Figure B.1 shows the relative distribution of degradation rates predicted by the degradation model alongside those collected in the compendium.

![](images/8396525d4c482e7057803aecdb0222ac46d416ec2b0ed3839f40731e9e94c0f1.jpg)  
Figure B.1: The distribution of degradation rates obtained from the degradation model and collected by the compendium of degradation rates [5].

Although there are noticeable diferences between the simulated and reported values, the overall range of degradation rates is comparable. This indicates that the simulated lifetimes are not unrealistic. The remaining discrepancies can be attributed to diferences in module technologies, installation characteristics, and geographical sampling between the model and the reported datasets.

## Appendix C. Considered module

The energy yield and module lifetime targets from the dataset are simulated with a module also used in previous work [22]. It is a monofacial silicon heterojunction module, consisting of 144 half-cut cells with a G12 wafer size, and is equipped with an EVA encapsulant and PET backsheet. Figure C.1 shows the optical and electrical module performance. The full details (and the considered parameters of the lifetime simulations) can be found in the previous work [22].

![](images/27635d98f223abfb93b0ec08d866659077f4acb500c692d2d297ca7b09867d66.jpg)

![](images/ac5b4c82fe5e960e9b0a415a01bc8a29ba927f2015681a9329f684234118eb09.jpg)  
Figure C.1: The optical performance (a) and electrical performance (b) of the considered module at standard test conditions.

For each location, the module tilt and azimuth are chosen that maximize the received irradiance. These values can be found in the supplementary information.

## Appendix D. Comparison distance metrics

To construct the dendrogram in Figure 6a), both a distance metric between samples and a linkage method defining distances between clusters must be specified. In this work, Euclidean distance is used as the underlying distance metric, while diferent linkage methods are evaluated.

The MATLAB environment [43] provides several linkage methods, which are summarized in Table D.3. For all definitions, $d ( r , s )$ denotes the distance between clusters r and $s , N _ { r }$ and $N _ { s }$ are the number of samples in clusters r and s, respectively, and $\left| \left| x - y \right| \right| _ { 2 }$ denotes the Euclidean distance between two samples. It should be noted that the median and weighted linkage methods are defined recursively, where cluster r is formed by merging clusters p and q.

The choice of linkage method strongly influences the structure of the dendrogram and therefore afects the resulting climate classification. The top plots of Figure D.1 illustrate how diferent distance metrics alter the dendrogram, demonstrating that its shape is highly sensitive to the metric used. In this work, our objective is to identify a linkage method that produces a dendrogram in which data points are distributed as evenly as possible across clusters. Such a metric is desirable because it avoids classifications in which some climate zones correspond to only very small regions.

Table D.3: Linkage methods used in hierarchical clustering and their corresponding definitions. For all equations, $d ( r , s )$ is the distance between clusters r and $s , N _ { r }$ and $N _ { s }$ are the number of points in cluster r and s, and $\vert \vert x - y \vert \vert _ { 2 }$ is the Euclidean distance between two points. It should be noted that the median and weighted metric are recursive metrics, where cluster r is created by combining cluster p and cluster $q .$
<table><tr><td>Metric</td><td>Equation</td></tr><tr><td>average</td><td> $\begin{array} { r } { \overline { { d ( \boldsymbol { r } , \boldsymbol { s } ) = \frac { 1 } { N _ { r } \cdot N _ { s } } \sum _ { i = 1 } ^ { N _ { r } } \sum _ { j = 1 } ^ { N _ { s } } | | \boldsymbol { x } _ { r i } - \boldsymbol { x } _ { s j } | | _ { 2 } } } } \end{array}$ </td></tr><tr><td>centroid</td><td> $d ( r , s ) = | | \bar { x } _ { r } - \bar { x } _ { s } | | _ { 2 }$ </td></tr><tr><td>complete</td><td> $d ( r , s ) = \operatorname* { m a x } ( | | x _ { r i } - x _ { s j } | | _ { 2 } ) , i \in ( 1 , . . . , N _ { r } ) , j \in ( 1 , . . . , N _ { s } )$ </td></tr><tr><td>median</td><td> $d ( r , s ) = | | \tilde { x } _ { r } - \tilde { x } _ { s } | | _ { 2 } , \tilde { x _ { r } } = \textstyle { \frac { 1 } { 2 } } ( \tilde { x } _ { p } + \tilde { x } _ { q } )$ </td></tr><tr><td>single</td><td> $d ( r , s ) = \operatorname* { m i n } ( | | x _ { r i } - x _ { s j } | | _ { 2 } ) { \overline { { , } } } i \in ( 1 , . . . , N _ { r } ) , j \in ( 1 , . . . , N _ { s } )$ </td></tr><tr><td>ward</td><td> $\begin{array} { r } { d ( r , s ) = \sqrt { \frac { 2 \cdot N _ { r } \cdot N _ { S } } { N _ { r } + N _ { s } } | | \bar { x } _ { r } - \bar { x } _ { s } | | _ { 2 } } } \end{array}$ </td></tr><tr><td>weighted</td><td> $\begin{array} { r } { d ( r , s ) = \frac { { \dot { d } } ( p , s ) + { d ( q , s ) } } { 2 } } \end{array}$ </td></tr></table>

To quantify the uniformity of the distribution, we compute the standard deviation of the number of points per cluster normalized by the mean number of points per cluster across all levels of the dendrogram, written as $\frac { { \mathrm { s t d } } ( P _ { i } ) } { \mathrm { m e a n } ( P _ { i } ) }$ where $P _ { i }$ denotes the number of points in cluster i. A lower value of this ratio indicates a more even distribution of data points across clusters. The bottom panels of Figure D.1 show the values of $\mathbf { \dot { \Pi } } _ { \operatorname* { m e a n } ( P _ { i } ) } ^ { \mathrm { s t d } ( P _ { i } ) }$ for all considered linkage methods. As observed, the Ward linkage consistently yields the lowest ratio across the entire dendrogram. Therefore, Ward linkage is selected for this work, as it produces the most uniform distribution of data points among clusters.

## Appendix E. Classifications for separate targets

The clustering and classification procedure can also be applied when considering only the energy-yield or lifetime target. In this case, the feature values are weighted using the importance scores shown in Figures 5a and 5b, respectively. This weighting alters the resulting dendrogram and therefore the climate classification. Figures E.1a and E.1c present the dendrograms obtained when considering only the energy yield or lifetime feature importance scores, respectively. The corresponding relative distances between successive cut-of points in the dendrogram, used to determine the optimal number of clusters, are shown in Figures E.1b and E.1d.

Applying the same selection criteria as in the main analysis yields 4 main clusters and 10 subclusters for the energy-yield-only classification, and 6 main clusters and 14 subclusters for the lifetime-only classification. The resulting classifications are shown in Figures E.2 and E.3, respectively.

![](images/9c6926bc3e31880f71b5098f1b2427490eaf7692a0d16c3e33664b3cdb5fa0bf.jpg)  
Figure D.1: Comparison of hierarchical clustering results using diferent linkage methods. The top figures show the resulting dendrograms. The figures below show the standard deviation of points in the cluster over the average points in the cluster, indicating the spread of all clusters.

For clarity, the main clusters in these two alternative classifications are labeled numerically (1, 2, 3, . . . ), and subclusters alphabetically (a, b, c, . . . ). Using similar climate-zone names as in the main study could lead to confusion, as the resulting regions difer substantially. When only an energy-yield-based or lifetime-based classification is required for specific applications, future work may focus on assigning meaningful descriptive labels to these clusters.

## Appendix F. Sensitivity to changes in feature importance score

The dendrogram obtained from hierarchical clustering is sensitive to small perturbations in the computed distances. To illustrate this sensitivity, we reproduce Figure 6b using slightly modified feature importance scores. Specifically, a random value drawn uniformly from the interval [−0.5, 0.5] is added to each feature importance score.

![](images/b6df32d5838b6d663373031f4755683061ecd59c7813d8ce4105b1fe92e1270d.jpg)

![](images/13427200e075935da2b67697c236c85dc45ff801c05c7220cbf982df481cd47e.jpg)

![](images/2c1154adfceceb27df574c48cab4da407dcef0f422df505f9f4ccfc7137b341a.jpg)

d)  
![](images/593b003da840a22249bbb13b5ee48b0acc16d0ade900cc3404dc688b1991bbe9.jpg)

Figure E.1: The diferent dendrogram and cluster selection when only considering the energy yield or lifetime. (a) and (b) show the dendrogram and relative distances for the energy yield target, finding local maxima for 4 and 10 clusters.(c) and (d) show the dendrogram and relative distances for the lifetime target, finding local maxima for 6 and 14 clusters.  
![](images/1a32e1cc9777cbf5b1b808b40cb55ad033a0e9440eea02e56ceaf92b52f4f909.jpg)  
Figure E.2: The climate classification when considering the feature scores for the energy yield targets. Main clusters are indicated with 1, 2, 3, and 4. Subclusters are indicated with a, b, and c to avoid confusion with the presented classification.

![](images/f6fa64c08e18baffd8ce3d77b34a0e64a510460c3f7ffda7631d3be2bce4cdcc.jpg)  
Figure E.3: The climate classification when considering the feature scores for the lifetime targets. Main clusters are indicated with 1, 2, 3, 4, 5, and 6. Subclusters are indicated with a, b, c, and d to avoid confusion with the presented classification.

Figure F.1 shows the resulting optimal numbers of clusters and subclusters for three independent realizations of this perturbation. Although these small changes do not alter the relative ranking of the three most important features, they can still afect the structure of the dendrogram and, consequently, the identified optimal number of clusters. This highlights the sensitivity of the clustering outcome to minor variations in the feature importance scores.

## Appendix G. Labeling of subclusters

As explained in the main text, the subclusters are denoted with relative high (H) or low (L) values in temperature (T), irradiance (I), or wind speed (W). To justify the labeling of the subclusters, Figure G.1 contains scatter plots in selected features of all main clusters. For each main cluster, the scatter plot between two features has been selected that show a clear separation between the subclusters.

These scatter plots are used to assign meaningful names to the subclusters. As already stated in the main text, it should be realized that the indicators high and low are relative to their main clusters.

![](images/e516c898fb8dd0ef39caed404b3b51ea075f90031ae8432532e6cee1ac88753a.jpg)  
Figure F.1: Sensitivity analysis for determining the optimal number of clusters. The top panels show the feature importance scores for the reference case (left) and three perturbed cases. The bottom panels show the corresponding dendrogram distances and the resulting optimal numbers of main clusters and subclusters.

## Appendix H. Overlap with existing classifications

To investigate the similarity between the created climate classification and the already existing ones, Figure H.1 presents the overlap the map in Figure 7 and the one generated by Ascencio-Vásquez et al. [13] and Triana de las Heras et al. [15]. It can be seen that is a significant overlap between the climate zones, showing $\mathrm { a > } 7 0 \%$ overlap for multiple clusters. Still, there are also significant diferences, highlighting the need for the classification developed in this work.

## Appendix I. Definition discounted lifetime energy yield

Figure 8 in the main text shows the boxplot of the discounted lifetime energy yield $\left( L E Y _ { d i s } \right)$ for all climate zones. $L E Y _ { d i s }$ for each location is cal-

![](images/b05c1244d3cc6b8f4696f1c43370225d90d3367ef53c3f4eae8768139544c6d4.jpg)

Figure G.1: Scatter plots of the relevant features of the diferent subclusters in each main cluster. This is used to create meaningful annotations for the subclusters. Please note that diferent colors have been selected to more clearly indicate the diferent subclusters.  
![](images/ee979e8ebd9580a055d8bcafc4779d53f53f7037810e098be7672daec050c0cd.jpg)  
Figure H.1: The overlap between the generated classification and the already existing ones. The percentages indicate which fraction of the climate zone from this work is also in the climate zone from [13] or [15]. Therefore, the values in all rows equal to 100%.

culated with

$$
L E Y _ { d i s } = \sum _ { t = 1 } ^ { L T } \frac { E Y \cdot \left( 1 - 0 . 2 \cdot \frac { t } { L T } \right) } { ( 1 + r _ { d } ) ^ { t } } ,\tag{I.1}
$$

where $L T$ is the module lifetime, $E Y$ is the annual energy yield, $r _ { d }$ is the discount rate (taken as $7 \% [ 4 1 ] )$ , and t represents the years from 1 till LT.

The term $1 - 0 . 2 \frac { t } { L T }$ is used to account for a degradation-induced reduction in energy yield throughout the lifetime of the module. It assumes a linear degradation rate, needed to obtain an 80% module performance at the end of the module lifetime.

## Appendix J. Representative locations for all climate zones

To enable consistent global comparisons, a single representative location is identified for each climate zone. For a given climate zone $k ,$ , the mean weighted value of feature $j ,$ , denoted by $\bar { x } _ { j }$ , is computed as

$$
\bar { x } _ { j } = \frac { \sum _ { i \in k } x _ { i , j } \cdot S _ { j } } { N _ { k } } ,\tag{J.1}
$$

where $x _ { i , j }$ represents the value of feature $j$ at location $i , S _ { j }$ is the feature importance score, and $N _ { k }$ denotes the total number of locations within climate zone k. The representative location for each climate zone is then selected as the location whose feature values are closest to the corresponding mean weighted feature values. Table J.1 summarizes the geographic coordinates of the representative locations identified for each climate zone.

## References

[1] N. M. Haegel, P. Verlinden, M. Victoria, P. Altermatt, H. Atwater, T. Barnes, C. Breyer, C. Case, S. D. Wolf, C. Deline, M. Dharmrin, B. Dimmler, M. Gloeckler, J. C. Goldschmidt, B. Hallam, S. Haussener, B. Holder, U. Jaeger, A. Jaeger-Waldau, I. Kaizuka, H. Kikusato, B. Kroposki, S. Kurtz, K. Matsubara, S. Nowak, K. Ogimoto, C. Peter, I. M. Peters, S. Philipps, M. Powalla, U. Rau, T. Reindl, M. Roumpani, K. Sakurai, C. Schorn, P. Schossig, R. Schlatmann, R. Sinton, A. Slaoui, B. L. Smith, P. Schneidewind, B. Stanbery, M. Topic, W. Tumas, J. Vasi, M. Vetter, E. Weber, A. W. Weeber, A. Weidlich, D. Weiss, A. W. Bett, Photovoltaics at multiterawatt scale: Waiting is not an option, Science 380 (6640) (2023) 39– 42. arXiv:https://www.science.org/doi/pdf/10.1126/science.adf6957, doi:10.1126/science.adf6957. URL https://www.science.org/doi/abs/10.1126/science.adf695 7

Table J.1: The coordinates of the representative locations for all climate zones.
<table><tr><td>Climate zone</td><td>Latitude</td><td>Longitude</td><td>Country</td></tr><tr><td>Desert-LT</td><td>-19</td><td>30</td><td>Zimbabwe</td></tr><tr><td>Desert-HI</td><td>-21</td><td>129</td><td>Australia</td></tr><tr><td>Desert-HT</td><td>8</td><td>42</td><td>Ethiopia</td></tr><tr><td>Tropical</td><td>22</td><td>95</td><td>Myanmar</td></tr><tr><td>Continental-HT</td><td>29</td><td>117</td><td>China</td></tr><tr><td>Continental-LT</td><td>45</td><td>-89</td><td>United States of America</td></tr><tr><td>Temperate-HT</td><td>-32</td><td>27</td><td>South Africa</td></tr><tr><td>Temperate-LI</td><td>39</td><td>32</td><td>Türkiye</td></tr><tr><td>Temperate-LT</td><td>39</td><td>101</td><td>China</td></tr><tr><td>Boreal-HI</td><td>54</td><td>-98</td><td>Canada</td></tr><tr><td>Boreal-HT</td><td>54</td><td>33</td><td>Russia</td></tr><tr><td>Boreal-LT</td><td>59</td><td>100</td><td>Russia</td></tr><tr><td>Boreal-LI</td><td>62</td><td>9</td><td>Norway</td></tr><tr><td>Polar-HW</td><td>67</td><td>-109</td><td>Canada</td></tr><tr><td>Polar-LW</td><td>67</td><td>166</td><td>Russia</td></tr></table>

[2] M. Rahman, M. Hasanuzzaman, N. Rahim, Efects of various parameters on pv-module power and eficiency, Energy Conversion and Management 103 (2015) 348–358. doi:https://doi.org/10.1016/j.enconman.2015.06.067. URL https://www.sciencedirect.com/science/article/pii/S019 6890415006159

[3] K. Hasan, S. B. Yousuf, M. S. H. K. Tushar, B. K. Das, P. Das, M. S. Islam, Efects of diferent environmental and operational factors on the pv performance: A comprehensive review, Energy Science & Engineering 10 (2) (2022) 656–675. doi:https://doi.org/10.1002/ese3.1043. URL https://scijournals.onlinelibrary.wiley.com/doi/abs/10 .1002/ese3.1043

[4] C. Barretta, A. E. Macher, M. Köntges, J. Ascencio-Vásquez, M. Topič, G. Oreski, Efect of encapsulant degradation on photovoltaic modules performances installed in diferent climates, IEEE Journal of Photovoltaics 15 (2) (2025) 290–296. doi:10.1109/JPHOTOV.2024.3523546.

[5] D. C. Jordan, S. R. Kurtz, K. VanSant, J. Newmiller, Compendium of

photovoltaic degradation rates, Progress in Photovoltaics: Research and Applications 24 (7) (2016) 978–989. doi:10.1002/pip.2744.

[6] Shield: Trinasolar’s extreme climate solution for asia pacific, accessed on 2026-02-24 (Apr 2025). URL https://taiyangnews.info/technology/trinasolar-shiel d-extreme-climate-solution-topcon-tracker

[7] China switches on 1 gw of solar at 4,600 m above sea level, accessed on 2026-02-24 (Jan 2026). URL https://www.pv-magazine.com/2026/01/26/china-switche s-on-1-gw-of-solar-at-4600-m-above-sea-level/

[8] H. Ziar, A global statistical assessment of designing silicon-based solar cells for geographical markets, Joule 8 (6) (2024) 1667–1690. doi:10.1016/j.joule.2024.02.023. URL https://doi.org/10.1016/j.joule.2024.02.023

[9] Y. Blom, M. R. Vogt, O. Isabella, R. Santbergen, Optimization of the perovskite cell in a bifacial two-terminal perovskite/silicon tandem module, Solar Energy Materials and Solar Cells 282 (2025) 113431. doi:https://doi.org/10.1016/j.solmat.2025.113431. URL https://www.sciencedirect.com/science/article/pii/S092 7024825000327

[10] International Energy Agency Photovoltaic Power Systems Programma, Guidelines for operation and maintenance of photovoltaic power plants in diferent climates (2022). URL https://iea-pvps.org/key-topics/guidelines-for-operati on-and-maintenance-of-photovoltaic-power-plants-in-differe nt-climates/

[11] P. Dash, N. Gupta, R. Rawat, P. Pant, A novel climate classification criterion based on the performance of solar photovoltaic technologies, Solar Energy 144 (2017) 392–398. doi:https://doi.org/10.1016/j.solener.2017.01.046. URL https://www.sciencedirect.com/science/article/pii/S003 8092X17300658

[12] M. Kottek, J. Grieser, C. Beck, B. Rudolf, F. Rubel, World Map of the Köppen-Geiger climate classification updated, Meteorologische Zeitschrif 15 (3) (2006) 259–263.

[13] J. Ascencio-Vásquez, K. Brecl, M. Topič, Methodology of Köppen-Geiger-Photovoltaic climate classification and implications to worldwide mapping of PV system performance, Solar Energy 191 (2019) 672–685. doi:10.1016/j.solener.2019.08.072.

[14] N. Skandalos, M. Wang, V. Kapsalis, D. D’Agostino, D. Parker, S. S. Bhuvad, Udayraj, J. Peng, D. Karamanis, Building pv integration according to regional climate conditions: Bipv regional adaptability extending köppen-geiger climate classification against urban and climate-related temperature increases, Renewable and Sustainable Energy Reviews 169 (2022) 112950. doi:https://doi.org/10.1016/j.rser.2022.112950. URL https://www.sciencedirect.com/science/article/pii/S136 4032122008310

[15] F. J. T. de las Heras, O. Isabella, M. R. Vogt, A machine learning approach to pv-climate classification, Renewable Energy (2025) 123685doi:https://doi.org/10.1016/j.renene.2025.123685. URL https://www.sciencedirect.com/science/article/pii/S096 0148125013473

[16] A. Alcañiz, D. Grzebyk, H. Ziar, O. Isabella, Trends and gaps in pho tovoltaic power forecasting with machine learning, Energy Reports 9 (2023) 447–471. doi:https://doi.org/10.1016/j.egyr.2022.11.208. URL https://www.sciencedirect.com/science/article/pii/S235 2484722025975

[17] G. M. Tina, C. Ventura, S. Ferlito, S. De Vito, A state-of-art-review on machine-learning based methods for pv, Applied Sciences 11 (16) (2021). doi:10.3390/app11167550. URL https://www.mdpi.com/2076-3417/11/16/7550

[18] N. Dahiya, S. Gupta, S. Singh, A review paper on machine learning applications, advantages, and techniques, ECS Transactions 107 (1) (2022) 6137. doi:10.1149/10701.6137ecst. URL https://doi.org/10.1149/10701.6137ecst

[19] J. Remund, S. Müller, M. Schmutz, P. Graf, Meteonorm version 8, ME-TEOTEST (www. meteotest. com) (2020).

[20] M. Vogt, C. R. Tobon, A. Alcañiz, P. Procel, Y. Blom, A. N. El Din, T. Stark, Z. Wang, E. G. Goma, J. Etxebarria, H. Ziar, M. Zeman, R. Santbergen, O. Isabella, Introducing a comprehensive physics-based modelling framework for tandem and other PV systems, Solar Energy Materials and Solar Cells 247 (2022) 111944. doi:10.1016/j.solmat.2022.111944.

[21] Y. Blom, M. R. Vogt, O. Isabella, R. Santbergen, Improving the comprehensive modeling framework for various novel photovoltaic systems, Submittd to Advanced Theory and Simulations (2026).

[22] Y. Blom, R. Santbergen, O. Isabella, M. R. Vogt, Combining physicaland scenario-based modeling to identify tolerable degradation rates of perovskite in monolithic two-terminal perovskite/silicon tandem modules, Solar Energy Materials and Solar Cells 299 (2026) 114169. doi:https://doi.org/10.1016/j.solmat.2026.114169. URL https://www.sciencedirect.com/science/article/pii/S092 7024826000103

[23] VDMA, International technology roadmap for photovoltaics (itrpv), 2024 results (2025).

[24] Delft High Performance Computing Centre (DHPC), DelftBlue Supercomputer (Phase 2), https://www.tudelft.nl/dhpc/ark:/44463/D elftBluePhase2 (2024).

[25] D. Shepard, A two-dimensional interpolation function for irregularlyspaced data, in: Proceedings of the 1968 23rd ACM National Conference, ACM ’68, Association for Computing Machinery, New York, NY, USA, 1968, p. 517–524. doi:10.1145/800186.810616. URL https://doi.org/10.1145/800186.810616

[26] J. Benesty, J. Chen, Y. Huang, I. Cohen, Pearson correlation coeficient, in: Noise reduction in speech processing, Springer, 2009, pp. 1–4.

[27] R. J. Janse, T. Hoekstra, K. J. Jager, C. Zoccali, G. Tripepi, F. W. Dekker, M. van Diepen, Conducting correlation analysis: important limitations and pitfalls, Clinical Kidney Journal

14 (11) (2021) 2332–2337. arXiv:https://academic.oup.com/ckj/articlepdf/14/11/2332/41100015/sfab085.pdf, doi:10.1093/ckj/sfab085. URL https://doi.org/10.1093/ckj/sfab085

[28] N. Meinshausen, P. Bühlmann, Stability selection, Journal of the Royal Statistical Society Series B: Statistical Methodology 72 (4) (2010) 417– 473. doi:10.1111/j.1467-9868.2010.00740.x. URL https://doi.org/10.1111/j.1467-9868.2010.00740.x

[29] S. Chatterjee, A. S. Hadi, Influential observations, high leverage points, and outliers in linear regression, Statistical Science 1 (3) (1986) 379–393. URL http://www.jstor.org/stable/2245477

[30] R.-E. Fan, P.-H. Chen, C.-J. Lin, et al., Working set selection using second order information for training support vector machines., Journal of machine learning research 6 (12) (2005).

[31] W. J. Nash, T. L. Sellers, S. R. Talbot, A. J. Cawthorn, W. B. Ford, The population biology of abalone (haliotis species) in tasmania. i. blacklip abalone (h. rubra) from the north coast and islands of bass strait, Sea Fisheries Division, Technical Report 48 (1994) p411.

[32] G. Jekabsons, Adaptive regression splines toolbox for matlab/octave, Version 1 (2016) 72.

[33] J. H. Friedman, Multivariate Adaptive Regression Splines, The Annals of Statistics 19 (1) (1991) 1 – 67. doi:10.1214/aos/1176347963. URL https://doi.org/10.1214/aos/1176347963

[34] X. Glorot, Y. Bengio, Understanding the dificulty of training deep feedforward neural networks, in: Y. W. Teh, M. Titterington (Eds.), Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, Vol. 9 of Proceedings of Machine Learning Research, PMLR, Chia Laguna Resort, Sardinia, Italy, 2010, pp. 249–256. URL https://proceedings.mlr.press/v9/glorot10a.html

[35] K. He, X. Zhang, S. Ren, J. Sun, Delving deep into rectifiers: Surpassing human-level performance on imagenet classification, in: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015, pp. 1026–1034.

[36] S. Narayan, The generalized sigmoid activation function: Competi tive supervised learning, Information Sciences 99 (1) (1997) 69–82. doi:https://doi.org/10.1016/S0020-0255(96)00200-9. URL https://www.sciencedirect.com/science/article/pii/S002 0025596002009

[37] J. A. Hartigan, M. A. Wong, Algorithm as 136: A k-means clustering algorithm, Journal of the Royal Statistical Society. Series C (Applied Statistics) 28 (1) (1979) 100–108. URL http://www.jstor.org/stable/2346830

[38] F. Murtagh, P. Contreras, Algorithms for hierarchical clustering: an overview, WIREs Data Mining and Knowledge Discovery 2 (1) (2012) 86–97. doi:https://doi.org/10.1002/widm.53. URL https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/ widm.53

[39] J. H. W. Jr., Hierarchical grouping to optimize an objective function, Journal of the American Statistical Association 58 (301) (1963) 236–244. doi:10.1080/01621459.1963.10500845. URL https://www.tandfonline.com/doi/abs/10.1080/01621459.1 963.10500845

[40] B. Mirkin, Choosing the number of clusters, WIREs Data Mining and Knowledge Discovery 1 (3) (2011) 252–260. arXiv:https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/widm.15, doi:https://doi.org/10.1002/widm.15. URL https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/ widm.15

[41] International Energy Agency, Nuclear Energy Agency, Projected costs of generating electricity (2020). URL https://www.iea.org/reports/projected-costs-of-generat ing-electricity-2020

[42] E. Psimopoulos, J. Plautz, F. Fiedler, A. Augusto, Performance of a pv system operating for 30-years in scandinavia, in: 2024 IEEE 52nd Photovoltaic Specialist Conference (PVSC), 2024, pp. 0353–0355. doi:10.1109/PVSC57443.2024.10748925.

[43] Matlab help center: Linkage, accessed on 2026-03-17. URL https://nl.mathworks.com/help/stats/linkage.html\#d126 e777688