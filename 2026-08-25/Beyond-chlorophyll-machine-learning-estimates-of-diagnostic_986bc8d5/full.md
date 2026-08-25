# Beyond chlorophyll: machine learning estimates of diagnostic phytoplankton pigments from multispectral ocean colour data

David Moffat<sup>1,2,†∗</sup>, Angus Laurenson<sup>1,2,†</sup>, Victor Martinez-Vicente<sup>1</sup>, Gemma Kulk<sup>1,2</sup>, Xuerong Sun<sup>3</sup>, Robert J. W. Brewin<sup>3</sup>, and Shubha Sathyendranath<sup>1,2</sup>

<sup>1</sup>Plymouth Marine Laboratory, Plymouth, UK

<sup>2</sup>National Center for Earth Observation, Plymouth Marine Laboratory, UK <sup>3</sup>Department ofEarth and Environmental Sciences, Centre for Geography and Environmental Science, University ofExeter, Cornwall, United Kingdom

Correspondence\*: David Moffat dmof@pml.ac.uk

## ABSTRACT

Phytoplankton play a central role in marine ecosystems and the global carbon cycle, with their functions in ocean biogeochemical cycles differing types of phytoplankton. Though standard techniques exist for monitoring the concentration of phytoplankton from ocean colour data, their community composition remains difficult to observe at large scales. The main phytoplankton pigment chlorophyll-a, readily available from satellite-based ocean-colour data, is widely used as a measure of phytoplankton biomass but provides limited information on taxonomic composition, in and by itself. Accessory pigments, some of which are considered to be diagnostic of some important phytoplankton groups, offer additional information on phytoplankton communities, but their retrieval from ocean-colour remote-sensing data is challenging due to the limited spectral bands traditionally available from satellite data and due to their strong covariance with chlorophyll-a. In this study, we evaluate the capability of machine learning methods to estimate diagnostic pigment concentrations from multispectral satellite observations. Using a global dataset of 33,640 High Performance Liquid Chromatography (HPLC) measurements matched with ESA Ocean Colour Climate Change Initiative (OC-CCI) ocean-colour reflectance data, we compare Random Forest and TabPFN models trained on multispectral reflectance with baseline models using chlorophyll-a alone. A temporally stratified validation scheme is employed to reduce the effects of autocorrelation. Results show that multispectral models consistently outperform approaches using only satellite derived chlorophyll, demonstrating that ocean colour reflectance contains additional information relevant to pigment discrimination. Improvements vary by pigment, with pigments strongly correlating with chlorophyll-a showing limited gains, while others exhibit substantial improvement. These findings highlight the potential of machine learning to extract ecologically relevant information from satellite data beyond conventional chlorophyll-based methods.

Keywords: machine learning, diagnostic pigment, phytoplankton, ocean colour, remote sensing, phytoplankton community composition, explainable AI

## PREPRINT

20 August 2026

This manuscript is a non-peer-reviewed preprint submitted to ArXiv. The manuscript has also been submitted for peer review to Frontiers in Marine Science. The content of this preprint has not yet undergone formal peer review and should therefore be considered preliminary. Subsequent versions of this manuscript may differ from the version presented here following peer review and publication.

Citation: Moffat, D. et al. (2026). Beyond chlorophyll: machine learning estimates of diagnostic phytoplankton pigments from multispectral ocean colour data. ArXiv.

## 1 INTRODUCTION

Phytoplankton are a diverse group of microscopic organisms that live in the sunlit portions of aquatic ecosystems, both freshwater and marine. Through photosynthesis, they are responsible for around 50% of global primary production of organic carbon (Longhurst et al., 1995; Field et al., 1998). They form the basis of the food web upon which marine life depends (Falkowski, 2012). Understanding how phytoplankton respond to climate change and what effects this will have on the environment are important questions that require long-term monitoring, on a global scale (Sathyendranath et al., 2017). Oceancolour satellite observations provide an important tool for such monitoring, based on the observation of the pigment chlorophyll-a (Gordon et al., 1980; Sathyendranath et al., 2019). All phytoplankton contain chlorophyll-a, representing the sum of monovinyl chlorophyll-a, divinyl chlorophyll-a, and chlorophyllide-a. Chlorophyll-a is essential for photosynthesis. In the open-ocean, Case-1 waters, it is generally assumed that chlorophyll-a concentration can be treated as the single, independent variable that determines the reflectance signal (all other substances that contribute to the signal are assumed to covary in some fashion with chlorophyll-a); and chlorophyll-a can be accurately measured from remote sensing. It is a measure of phytoplankton biomass and informs models of marine primary production (Platt and Sathyendranath, 1988; Behrenfeld and Falkowski, 1997; Kulk et al., 2020; Baxter et al., 2025; Silsbe et al., 2025; Sathyendranath et al., 2020). The Case-1 assumption that chlorophyll-a concentration can be an indicator of the underlying ecosystem structure has been exploited to infer community structure from chlorophyll-a (Sun et al., 2025; Brewin et al., 2015) with remarkable success. However, we recognise that deviation from the Case-1 assumption can occur even in the open ocean. Furthermore, estimates of phytoplankton biomass, productivity, and community structure become more uncertain towards the coast where other constituents in the water can dominate the optical signal.

Quantifying the contributions from individual phytoplankton types to total phytoplankton biomass is not a trivial problem: Phytoplankton are a highly diverse community, with estimates suggesting between 5,000 and 150,000 species worldwide (Sournia et al., 1991; Armbrust and Palumbi, 2015) spanning multiple phyla (Tett and Barton, 1995). Several methods exist for identifying individual phytoplankton species in situ, including DNA sequencing and microscopy (Bracher et al., 2017; IOCCG, 2014). However, these techniques are expensive and typically have limited geographical coverage. Simplifications are therefore necessary and phytoplankton are often grouped by their function in the ecosystem into Phytoplankton Functional Types (PFTs; Nair et al., 2008; IOCCG, 2014) and by their physical size into Phytoplankton Size Classes (PSCs) (Brewin et al., 2014). A method that is commonly used to identify different groups of phytoplankton in situ, is High Performance Liquid Chromatography (HPLC). This method measures phytoplankton pigments that can then be used to estimate PFTs with established methods such as Diagnostic Pigment Analysis (DPA; Vidussi et al., 2001), ChemTax (Mackey et al., 1996), and more recently PhytoClass (Hayward et al., 2023). Yet, the relationship between pigments and phytoplankton types is ambiguous, as several phytoplankton groups can synthesise the same pigments, and phytoplankton are able to modify the relative concentrations of their pigment complement according to physiological state or environmental conditions (Nair et al., 2008; Table 1). Accessory pigments help phytoplankton adapt to underwater light conditions by altering the spectrum of light they can absorb for photosynthesis (Majchrowski and Ostrowska, 2000; Dubinsky and Schofield, 2010). The varied morphology and ecological niches of phytoplankton groups have led to the evolution of different accessory pigments (Reynolds, 2006). These broadly correspond to PFTs; for example, the pigment peridinin is associated with dinoflagellates, whilst Prochlorococcus uniquely uses divinyl-chlorophylls (Table 1). But, the pigment fucoxanthin, often associated with diatoms, can also be synthesised by other phytoplankton groups (Table 1). It has been demonstrated that four broad groups of phytoplankton can be discriminated from diagnostic pigments at the global scale: cyanobacteria, diatoms/dinoflagellates, haptophytes, and green algae (Kramer and Siegel, 2019).

Table 1. Overview of pigments estimated by the machine learning models and their taxonomic affiliation with phytoplankton groups. Information taken from Nair et al. (2008), with the taxonomic affiliation identified for those phytoplankton groups in which the specific pigment is always or often present. Some pigments are unique markers to certain phytoplankton groups (i.e., unambiguous) and others appear across multiple phytoplankton groups (i.e., ambiguous).
<table><tr><td>Pigment</td><td>Marker type</td><td>Taxonomic affiliation</td></tr><tr><td>19&#x27;-butanoyloxyfucoxanthin</td><td>Ambiguous</td><td>Haptophytes, Pelagophytes</td></tr><tr><td>19&#x27;-hexanoyloxyfucoxanthin</td><td>Ambiguous</td><td>Chrysophytes, Haptophytes</td></tr><tr><td>Alloxanthin</td><td>Unambiguous</td><td>Cryptophytes</td></tr><tr><td>Monovinyl Chlorophyll-a</td><td>Ambiguous</td><td>All phytoplankton, except Prochlorococcus</td></tr><tr><td>Divinyl chlorophyll-a</td><td>Unambiguous</td><td>Prochlorococcus</td></tr><tr><td>Chlorophyll-b</td><td>Ambiguous</td><td>Chlorophytes, Euglenophytes, Prasinophytes</td></tr><tr><td>Divinyl chlorophyll-b Fucoxanthin</td><td>Ambiguous</td><td>Prochlorococcus</td></tr><tr><td></td><td>Ambiguous</td><td>Chrysophytes, Diatoms, Haptophytes, Pelagophytes, Raphidophytes</td></tr><tr><td>Peridinin</td><td>Unambiguous</td><td>Dinoflagellates</td></tr><tr><td>Zeaxanthin</td><td>Ambiguous</td><td>Cyanobacteria, Chlorophytes, Chrysophytes, Euglenophytes, Prasinophytes, Prochlorococcus, Raphidophytes</td></tr></table>

Several studies have explored machine learning approaches for predicting phytoplankton diversity indicators, such as pigments, phytoplankton types, and size classes, from satellite observations and environmental data. Early work by Brewin et al. (2011) compared nine approaches for estimating PSCs and found that abundance-based methods, which rely solely on chlorophyll-a, performed comparably to approaches incorporating spectral or environmental variables such as sea surface temperature. This result raises questions about the extent to which additional predictors provide independent information beyond chlorophyll-a. Subsequent studies have attempted to leverage spectral information more explicitly. Bracher et al. (2015) developed an empirical model based on linear regression of truncated principal components derived from hyperspectral reflectance, trained using coincident HPLC pigment measurements. When applied to multispectral observations of the Medium Resolution Imaging Spectrometer (MERIS), the model achieved good performance. Similarly, El Hourany et al. (2019) applied self-organising maps to predict pigments at the global scale and reported strong performance. However, the use of random data splitting for validation in these studies does not account for spatial and temporal autocorrelation, and may therefore lead to overly optimistic estimates of predictive skill, as the training and validation data may not be statistically independent. Stock and Subramaniam (2020) conducted a systematic comparison of machine learning approaches for predicting diagnostic pigment concentrations normalised by total chlorophyll-a. They demonstrated that the choice of crossvalidation strategy has a strong influence on estimated model performance, emphasising the importance of accounting for the spatial and temporal structure of oceanographic data. More recently, Foundation Models were developed for remote sensing products, but the models were validated only for the prediction of chlorophyll-a (Dawson et al., 2025).

In this study, we developed machine learning models to predict phytoplankton pigments from ocean colour observations of the Ocean Colour Climate Change Initiative (OC-CCI; Sathyendranath et al., 2019). A key unresolved question is whether multispectral ocean colour observations contain information on phytoplankton pigment composition beyond that provided by chlorophyll-a. Because many accessory pigments are strongly correlated with chlorophyll-a, apparent predictive skill may arise simply from covariance with phytoplankton biomass rather than from independent spectral information. Our hypothesis is that multispectral reflectance observations contain information that can discriminate accessory pigments beyond their covariance with chlorophyll-a. To test this hypothesis, we train two machine learning models using remote sensing reflectances $( R _ { r s } )$ at six OC-CCI wavebands, together with $K _ { d } ( 4 9 0 )$ and chlorophyll-a. As a control, we also train a baseline model using remote sensing derived chlorophyll-a as the sole predictor. This baseline represents abundance based approaches and provides a benchmark against which the additional information contained in multispectral observations can be evaluated. We validate all models using temporally independent in situ observations selected to minimise correlation with the training dataset and apply the resulting models at the global scale to investigate whether ecologically meaningful patterns in phytoplankton pigment composition can be recovered from long-term ocean colour data records.

## 2 MATERIALS & METHODS

## 2.1 In situ data

In situ data are required to train and evaluate model performance. We use a dataset of 33,640 High Performance Liquid Chromatography (HPLC) measurements, collated from multiple sources between 1991 and 2021. Full details of the in situ dataset are provided in Sun et al. (2026). In the purposes of the rest of the paper we refer to total chlorophyll-a(Tchla), i.e. both monovinyl and divinyl. Quality assurance was applied to the in situ data by removing values below 0.005 mg m<sup>−3</sup> where data were only reported to three decimal places. This filtering was applied where measurements approached the sensor detection limit, in order to remove quantisation artefacts.

It is generally expected that the proportions of accessory pigments are highly correlated and not truly independent (Bricaud et al., 2004). A correlation plot of the pigments in the in situ dataset is shown in Figure 1. It is shown that most pigments in the in situ dataset exhibited a strong correlation with chlorophyll-a. The highest correlation was observed for fucoxanthin, which was nearly perfectly correlated with chlorophyll-a, with a Pearson correlation coefficient of r = 0.95. Zeaxanthin was the only exception, showing little correlation with chlorophyll-a (r = 0.10).

## 2.2 Earth Observation data

The HPLC dataset spans several decades, exceeding the lifespan of individual ocean colour satellite missions. Therefore, data from multiple sensors must be merged to form a consistent dataset. Here, we use the European Space Agency (ESA) Ocean Colour Climate Change Initiative (OC-CCI) version 6.0 dataset at 4-km spatial and daily temporal resolution (Sathyendranath et al., 2019) for match-up with in situ HPLC observations. OC-CCI data were also used for application of the developed pigment models at the global scale, for which a 4-km, monthly image of May 2010 was used, as an month, with 2010 representing a year with good data coverage in OC-CCI, which includes multiple sensors contributing to the merged data product.

HPLC measurements were matched with coincident remote sensing reflectances $( R _ { r s } )$ , the attenuation coefficient of light at 490 nm $( K _ { d } ( 4 9 0 ) )$ and chlorophyll-a concentrations from OC-CCI to create a combined in situ and EO match-up dataset. Observations were matched when they occurred on the same day and the in situ location fell within the satellite pixel. Only in situ samples collected within the upper 5 m of the water column were considered. Where multiple samples existed at the same location and day but at different depths within the top 5 m, pigment concentrations were averaged. Quality control was applied to the match-up dataset using a $3 \times 3$ pixel window centred on each in situ observation. Match-ups were retained only when at least four of the nine satellite pixels contained valid observations and the coefficient of variation within the window was less than 0.15, thereby reducing the influence of sub-pixel spatial heterogeneity and residual retrieval errors (Jordan et al., 2023).

## 2.3 Data Partitioning

In situ measurements are often spatially and temporally correlated. Research cruises collect sequences of observations along transects, resulting in strong correlations due to geographic proximity and seasonal sampling. Furthermore, sampling is often biased towards particular regions (e.g., the Atlantic Ocean) and seasons (e.g., summer). To account for this, we partitioned the match-up dataset into training and validation subsets by withholding four years of data (2002, 2007, 2012, and 2017) for validation. The remaining 20 years were used for training. This resulted in a total of 6,625 data points available for training and 1,701 data points for validation. We note that the number of match-up data points available for each individual pigment varies from these total numbers.

![](images/216732a35611b1231e101f6b293fbc09bb74bb2c3a15d4fe5955c233084cf00c.jpg)

![](images/d1fec5578707f9b2dc06e3b8526bfa7642638b3aab95c499f066ce52bfb995bd.jpg)

![](images/715cf20e5579fa41464084be9e7c380cca00422ff1191b9036c2e46ecf45b8c9.jpg)

![](images/894de32298792dd3c5fb22a5616409f00268767e597c8eb334e220007663718a.jpg)

![](images/9fdc7c65770632b2b644a5a179ecf046815f0380541f5866cf1ce379cbc865f8.jpg)

![](images/d7802a751f96b27fb1d7e8836dd06e8f9a4e7e432af67c51acd8468e608192d1.jpg)

![](images/6edb8891196087617a5b4ebbd22f4a4d36ff0e689f7dab707798e9258e88175b.jpg)

![](images/5ff3868096c9f84644848401c3c1bd773e013957feb6f4521c861024cf10c909.jpg)  
Figure 1. Relationships between chlorophyll-a and other pigments in the in situ dataset, including the Pearson correlation in log space. The bottom right figure shows the Pearson correlation in log space of each pigment compared with each other pigment.

![](images/9c8b696111d4dbec07a385e6b6d12dea3985f73858c16130538a46b0c1ce55b3.jpg)  
Figure 2. Global map of match-up dataset. Blue points show locations of training data, whilst red points show data held back for validation, from years 2002, 2007, 2012, and 2017

Partitioning the data by year enables assessment of model generalisation to unseen temporal conditions and partially mitigates the effects of temporal autocorrelation. However, it may introduce distributional differences between training and validation datasets, particularly due to geographic variability in cruise sampling. Figure 2 shows the spatial distribution of the match-up dataset and its partitioning between training and validation sets. The training data cover several regions not included in the validation set, such as the vicinity of Madagascar and the western Pacific Ocean. Conversely, some regions are present only in the validation dataset, such as the Gulf of Mexico, as such we need to consider how this method will extrapolate to these locations.

## 2.4 Machine learning

We compare two machine learning approaches for predicting pigment concentrations from spectral information. The first approach is a foundation model for tabular data, TabPFN (Hollmann et al., 2023). TabPFN is a transformer-based foundation model trained on large numbers of synthetic tabular datasets and uses Bayesian posterior estimation to make predictions without explicit model retraining. TabPFN has been shown to perform well on small tabular datasets and avoids the need for extensive modelspecific training. The second approach is a Random Forest (RF) model (Breiman, 2001). RF models can capture non-linear relationships and are relatively robust to overfitting and noise, making them suitable for small and noisy datasets (Opitz and Maclin, 1999). RF models have also performed well in related ocean colour applications (Toming et al., 2026; Laine et al., 2024). To assess the contribution of spectral information (i.e., Rrs, $K _ { d } ( 4 9 0 ) )$ , we also trained a RF baseline model using only chlorophyll-a as input. This provides a benchmark for comparison with abundance-based approaches, which relate the PFT or PSC to chlorophyll-a reported in literature.

All models were trained to predict the concentrations of eight pigments: chlorophyll-a, chlorophyllb, alloxanthin, 19’-butanoyloxyfucoxanthin, 19’-hexanoyloxyfucoxanthin, fucoxanthin, peridinin, and zeaxanthin (Table 1). All target variables were log-transformed to better represent both low and high concentration ranges. The predictor variables used for both TabPFN and RF models included $R _ { r s } ( \lambda )$ at 412, 443, 490, 510, 560 and 665 nm, attenuation coefficient at 490 nm $( K _ { d } ( 4 9 0 ) )$ ) and chlorophyll-a (i.e., all from OC-CCI). Informal testing was performed using different subsets of predictor features, eg., Spectral $R _ { r s }$ only, and it was decided to include $( K _ { d } ( 4 9 0 ) )$ and chlorophyll-a as this provided improvements to the results. Note that the RF baseline model was trained using chlorophyll-a only.

Table 2. Overview of the different machine learning models developed to predict eight phytoplankton pigments using ocean colour satellite observations. Details of the models are provided in Section 2.4. The maximum tota number of observations (N) used for training and validation of the models is also provided, but specific numbers vary per pigment, due to not all pigments being captured for all samples.
<table><tr><td>Model</td><td>Method</td><td>Predictor variables</td><td>Max N (Training)</td><td>Max N (Validation)</td></tr><tr><td>TabPFN</td><td>Tabular Foundation Model</td><td> $\overline { { R _ { r s } , K _ { d } ( 4 9 0 ) } }$  , Chl-a</td><td>6,625</td><td>1,701</td></tr><tr><td>RF</td><td>Random Forest model</td><td> $R _ { r s } , K _ { d } ( 4 9 0 )$  , Chl-a</td><td>6,625</td><td>1,701</td></tr><tr><td>Baseline</td><td>Random Forest model</td><td>Chl-a</td><td>6,625</td><td>1,701</td></tr></table>

Model hyperparameters were optimised separately for each pigment. For TabPFN, the variant with automated hyperparameter optimisation (HPO) was used, with 50 optimisation iterations per pigment model. For the Random Forest models, hyperparameters including the number of trees, number of features, number of training samples, and maximum tree depth were optimised using a grid search.

Model performance was evaluated using the coefficient of determination $( R ^ { 2 } ;$ Ozer 1985). The $R ^ { 2 }$ statistic quantifies the proportion of variance in the observed pigment concentrations that is explained by the model predictions, with a value of 1 indicating perfect agreement, a value of 0 indicating performance equivalent to predicting the mean of the observations, and negative values indicating performance worse than the mean predictor. As pigment concentrations were modelled in log-transformed space, all reported $R ^ { 2 }$ values were calculated using log-transformed observations and predictions.

## 2.5 Explainable AI analysis

To interpret the machine learning models, we employed SHapley Additive exPlanations (SHAP) (Lundberg and Lee, 2017; Lundberg et al., 2020), a model-agnostic method based on cooperative game theory that quantifies the contribution of each input feature to individual model predictions. SHAP values represent the marginal contribution of each feature relative to a baseline prediction, enabling consistent and locally accurate attribution of model outputs. SHAP analysis was applied to the trained RF models for each pigment to assess the relative importance of predictor variables, including $R _ { r s } ( \lambda )$ chlorophyll-a, and $K _ { d } ( 4 9 0 )$ . SHAP violin plots were used to visualise the distribution of feature contributions across the dataset, showing both the magnitude and direction of influence on pigment predictions. Such plots allow identification of which spectral bands contribute most strongly to each pigment, and whether their effects are consistent across observations. This provides insight into the extent to which the models rely on chlorophyll-a versus independent spectral information, and helps to characterise the nature of the pigment signals captured by the machine learning models.

## 3 RESULTS

## 3.1 Comparison of pigment models performance

The performance of each pigment prediction model is summarised in Figures 3 and 4, with results shown for the training and validation datasets for the TabPFN foundation model and the RF model using spectral information and the baseline model using chlorophyll-a as input only. All models showed relatively high performance during training, with $R ^ { 2 }$ ranging between 0.89 – 0.97 for the TabPFN model,

Table 3. Comparison of phytoplankton pigment retrieval performance for the foundation model (predFM), chlorophyll-only random forest (predRF chl), and full random forest (predRF). Statistics shown are sample size (n), Pearson correlation (r), bias, root mean square difference (RMSD), Centre-pattern root mean square difference (CP-RMSD), and coefficient of determination $( \mathbf { R } ^ { 2 } )$
<table><tr><td>Pigment</td><td>Model</td><td>n</td><td>r</td><td>bias</td><td>RMSD</td><td>CP-RMSD</td><td> $\mathbb { R } ^ { 2 }$ </td></tr><tr><td>TChlb TChlb</td><td>predFM predRF chl</td><td>6656 6656</td><td>0.76 0.71</td><td>3.22 3.21</td><td>3.41 3.41</td><td>1.14 1.15</td><td>0.52 0.13</td></tr><tr><td>TChlb</td><td>predRF</td><td>6656</td><td>0.76</td><td>3.21</td><td>3.41</td><td>1.15</td><td>0.47</td></tr><tr><td>Allo</td><td>predFM</td><td>4631 4631</td><td>0.65 0.58</td><td>3.68 3.68</td><td>3.86</td><td>1.18</td><td>0.36</td></tr><tr><td>Allo</td><td>predRF chl</td><td>4631</td><td>0.62</td><td>3.68</td><td>3.86</td><td>1.18</td><td>-0.25</td></tr><tr><td>Allo</td><td>predRF</td><td>6580</td><td>0.73</td><td>4.00</td><td>3.86 4.12</td><td>1.18</td><td>0.29</td></tr><tr><td>But</td><td>predFM</td><td>6580</td><td>0.62</td><td>4.00</td><td>4.12</td><td>0.97 0.98</td><td>0.41</td></tr><tr><td>But</td><td>predRF chl</td><td>6580</td><td>0.70</td><td>4.00</td><td>4.12</td><td>0.97</td><td>-0.31</td></tr><tr><td>But</td><td>predRF predFM</td><td>7409</td><td>0.60</td><td>2.79</td><td>3.22</td><td>1.61</td><td>0.36</td></tr><tr><td>Fuco</td><td>predRF chl</td><td>7409</td><td>0.57</td><td>2.76</td><td>3.22</td><td>1.66</td><td>0.74</td></tr><tr><td>Fuco</td><td>predRF</td><td>7409</td><td>0.59</td><td>2.77</td><td>3.22</td><td>1.63</td><td>0.48 0.72</td></tr><tr><td>Fuco</td><td>predFM</td><td>7775</td><td>0.45</td><td>3.12</td><td>3.28</td><td>0.99</td><td></td></tr><tr><td>Hex</td><td>predRF chl</td><td>7775</td><td>0.35</td><td>3.12</td><td>3.28</td><td>1.01</td><td>0.51</td></tr><tr><td>Hex</td><td></td><td>7775</td><td>0.41</td><td>3.12</td><td>3.27</td><td></td><td>-0.19</td></tr><tr><td>Hex</td><td>predRF</td><td>4883</td><td>0.65</td><td>3.76</td><td>4.01</td><td>1.00</td><td>0.46</td></tr><tr><td>Peri</td><td>predFM</td><td>4883</td><td></td><td>3.76</td><td>4.02</td><td>1.41</td><td>0.45</td></tr><tr><td>Peri</td><td>predRF chl</td><td>4883</td><td>0.56</td><td>3.76</td><td></td><td>1.42</td><td>0.08</td></tr><tr><td>Peri</td><td>predRF</td><td>7399</td><td>0.61 0.73</td><td>3.45</td><td>4.01</td><td>1.41</td><td>0.41</td></tr><tr><td>Zea</td><td>predFM</td><td></td><td>0.64</td><td>3.45</td><td>3.58</td><td>0.94</td><td>0.23</td></tr><tr><td>Zea</td><td>predRF chl</td><td>7399</td><td></td><td>3.45</td><td>3.58</td><td>0.95</td><td>-0.38</td></tr><tr><td>Zea</td><td>predRF</td><td>7399</td><td>0.71</td><td></td><td>3.58</td><td>0.95</td><td>0.13</td></tr></table>

0.87 – 0.95 for the RF model and 0.83 – 0.95 for the baseline model across the eight different pigments. When pigment prediction was compared with independent validation data, performance was generally lower compared to that of the training dataset with $R ^ { 2 }$ ranging between 0.23 – 0.74 for the TabPFN model, 0.13 – 0.72 for the RF model and 0.075 – 0.48 for the baseline model. The TabPFN model for each pigment showed highest performance, closely followed by the RF model that also incorporated spectral information (Figure 5a), and these two models also showed similar bias and root-mean-squaredifference (Table 3). The models for chlorophyll-a and fucoxanthin showed highest performance and those for zeaxanthin showed lowest performance across all model types (Figure 5a) Further metrics were calculated, beyond coefficient of determination, however there was very little difference between the different models using these metrics, as shown in Table 3.

The baseline model, using chlorophyll-a as input, consistently underperformed relative to those models incorporating multispectral reflectance (i.e., TabPFN and RF models; Figure 5a). As shown previously in Figure 1, several pigments exhibited strong correlations with chlorophyll-a. Therefore, the improvement observed when including reflectance bands can indicate the extent to which additional spectral information contributes beyond these correlations, which is shown in Figure 5b. While the models for fucoxanthin showed high performance $( R ^ { 2 } > 0 . 7 2 )$ , the TabPFN and RF models exhibited the smallest improvement over the baseline model for this pigment. This is consistent with Figure 1, which shows that fucoxanthin was highly correlated with chlorophyll-a and we might therefore expect that this pigment can largely be predicted from abundance alone. The 19’-butanoyloxyfucoxanthin and 19’-hexanoyloxyfucoxanthin pigment models showed intermediate performance $( R ^ { 2 }$ between 0.36 – 0.51), but exhibited the largest improvement relative to the baseline model (Figure 5b). This was primarily due to the poor performance of the chlorophyll-a only models $( R ^ { 2 }$ between -0.3 – -0.19), resulting in an improvement in $R ^ { 2 }$ of approximately 0.65 – 0.7. The relative poor performance of zeaxanthin models across all model types was consistent with the weak correlation between zeaxanthin and chlorophyll-a (Figure 1) and suggested that this pigment was not strongly expressed in the spectral information captured by the available reflectance bands.

![](images/df8b33601d1ef25d2cccb60ed3af3f79e3bb0e9f8f04b155c37b46c98878258d.jpg)

![](images/214d54ef8db9e21daf95ea99aac80e0da35db5bb7bfa17fc93a27469e8dd978a.jpg)  
(a) 19’-butanoyloxyfucoxanthin

![](images/d44bc54dda8d946c94f380de1ecc4a131ceb6d0f9a564e6c37326f4d90ded1f2.jpg)

![](images/a0e753b9fd3ea76495160396b3e059d79b39920beedff13cf3b8ad64c9355f4e.jpg)

![](images/8022deae74911e632b132ba2d7f10778daad58dcbe9aac245819ba75cf06d3e5.jpg)  
(b) 19’-hexanoyloxyfucoxanthin

RandomForest Chl only Hex Concentration (mg m⁻³)  
![](images/4858df01bdec269b0e217be456854d2d732ab2979b98b451011fcff7c93f69a8.jpg)

![](images/d4ef4d2cfff4718778f25d8a3714a66fb2ce5972cc057f960b75872d68d4db3b.jpg)

![](images/dfcaf314705f80f22286c1b97398df343f79109bb342471d52d77eeaa909d2aa.jpg)

RandomForest Chl only Allo Concentration (mg m⁻³)  
![](images/5f37cdb1d86475fb7865d492ecdb8f4d499b271b362382e504aecb578e7dbaef.jpg)  
(c) Alloxanthin  
Figure 3. Scatter plots of predicted versus measured pigment concentrations for $1 9 ^ { \circ }$ -butanoyloxyfucoxanthin (a), 19’-butanoyloxyfucoxanthin (b), alloxanthin (c). The x-axis shows HPLC-derived measurements and the y-axis shows model predictions of each pigment in mg m<sup>−3</sup>. Results for the training (blue) and validation (orange) datasets are presented for each type of model, i.e., the TabPFN $\mathbf { \Gamma } ( \mathbf { a } , \mathbf { d } , \mathbf { g } )$ and RF $^ { ( \mathrm { b , e , h } ) }$ models that incorporate spectral information and the baseline model (c,f,i) that is based on chlorophyll-a as a predictor.

Overall, the pigment models that incorporate spectral information performed best, with the TabPFN model providing modest performance improvements over the RF model across all eight pigments. However, the better performance of the TabPFN model (i.e., higher $R ^ { 2 }$ , but similar bias and variance)

RandomForest Chl only Zea Concentration (mg m⁻³)  
FoundationModel Fuco Concentration (mg m⁻³)  
![](images/12dfd1d9a6a153fc386fa83219f728024516cd21e18ed5f03ad1c65722408303.jpg)

![](images/a57a470c35e79532b05b86e1227badef07fd896243f5fe8925a17e43d8c6e7cc.jpg)  
(a) Chlorophyll-b

RandomForest Chl only TChlb Concentration (mg m⁻³)  
![](images/f2c749b9051de2e82f7e3f69ec46e9c405b0a0831fd3a6b29dcc10a66cae6d49.jpg)

![](images/bb7d8155c0a3e3821b70f6348d69f1c044bb857bd65597d66ed9a5709b650fc7.jpg)

![](images/1b6c69d6c66cc5255f35260c2b91c963a274bc1b8f084ae2a3408ed0040f1011.jpg)  
(b) Fucoxanthin

RandomForest Chl only Fuco Concentration (mg m⁻³)  
![](images/684174135f03487941c87669eadbcff3ddb9badb3b011939f4e6760349cec022.jpg)

![](images/5471dfd3c54dd553c71f61209b5f4415a14ca2126728870fb0a644f1124b233a.jpg)

![](images/1731feaa20f9a6ef79ca84b06a86a6b60c4e72bf80f2a02b67ea3b0705317646.jpg)

![](images/b3fdc306298a89ffb53dc51382aac17c6b355ae9c2377a704099bc2694248b02.jpg)

(c) Peridinin  
![](images/2063f46b15a7c79812bf1a934f8f5687151008df50006be133161293c598c652.jpg)

![](images/c84622b84402cde2b72e545e09cc4eea5b21e49bcaff9ea51d2f6ee5d6ec1c1e.jpg)  
(d) Zeaxanthin

![](images/e69cf778799707b38e917cd129dfd51f0c88e5ca1dce7bf33a396b774d6e75f5.jpg)  
Figure 4. As in Figure 3, for chlorophyll-b (a), fucoxanthin (b), peridinin (c), and zeaxanthin (d).

was relatively small considering the computational cost of this type of model. For the remainder of this analysis, we therefore focus on the RF model that incorporates spectral information.

![](images/cb00d5a27200d6ca5b5b7bc4cc0f1c0b87b3e648dff15475ecba1c4aefa79a1a.jpg)  
(a) Validation $R ^ { 2 }$ scores

![](images/5cca7f2adf2136120a4ff2d8d29718ef467d49db19cf6fcc581da9767a88089a.jpg)  
(b) Improvement relative to chlorophyll-only RF model  
Figure 5. Comparison of model performance on the validation dataset. Panel (a) shows the validation $R ^ { 2 }$ values obtained for each pigment and model configuration. Panel (b) shows the improvement in $R ^ { 2 }$ achieved by the Random Forest model using multispectral reflectance relative to the chlorophyll-only Random Forest baseline, illustrating the additional predictive information contained within the reflectance observations.

## 3.2 Interpretation of pigment models

The SHAP analysis revealed substantial differences in the predictor variables utilised by the RF pigment models that incorporated spectral information (Figure 6). Fucoxanthin and peridinin predictions were strongly influenced by chlorophyll-a, consistent with the strong correlations observed between these pigments and chlorophyll-a in the match-up dataset (Figure 1). For both pigments, chlorophyll-a exhibited the largest SHAP values and therefore contributed most to the model predictions (Figure 6). In contrast, the alloxanthin, 19’-butanoyloxyfucoxanthin and $1 9 ^ { \prime }$ -hexanoyloxyfucoxanthin models relied primarily on combinations of $R _ { r s }$ bands and $K _ { d } ( 4 9 0 )$ , with chlorophyll-a contributing relatively little. The dominance of spectral variables in these models is consistent with the substantial improvement in predictive performance relative to the baseline models that only include chlorophyll-a, indicating that additional information relevant to pigment composition is present within the multispectral reflectance observations. Zeaxanthin exhibited the most distinct SHAP structure. Despite its comparatively poor predictive performance, the model relied predominantly on blue-green reflectance bands and $K _ { d } ( 4 9 0 )$ rather than chlorophyll-a. As chlorophyll-a is typically calculated as a ratio of blue to green reflectance bands, then perhaps the Zeaxanthin pigment is related more to the magnitude of the $R _ { r s }$ bands rather than simply the ratio between the bands. These results suggest that the model is attempting to exploit optical signatures associated with zeaxanthin rather than relying on covariance with chlorophyll-a. These results indicated that the machine learning models utilised varying combinations of spectral information depending on the pigment being predicted.

## 3.3 Global maps of pigments

One of the objectives of this work was to assess whether the machine learning tools tested here could bring in more information about pigment composition than what could be achieved solely on the basis of correlation between chlorophyll-a and the accessory pigments. Figure 5 provides a quantitative measure of the improvement of the RF model, relative to approaches that rely solely on chlorophylla, when considering the validation data. Additional insights into the algorithm performance can be garnered by studying the global maps of the accessory pigments, in absolute units, and in concentrations relative to chlorophyll-a (Figure 7 and 8). The left-hand panels of these two figures show coherent large-scale oceanographic structures, reproducing known contrasts between productive high-latitude and coastal waters and the oligotrophic subtropical gyres. The predicted distributions were therefore consistent with established patterns of marine biogeography and suggest that the machine learning models recovered ecologically meaningful information from multispectral ocean colour observations. While that is reassuring, and is essential for any successful set of algorithms, it is difficult to discern from these panels whether the machine learning models developed here successfully distinguishes each pigment from all the other pigments, and importantly, whether they capture regional differences in the distribution of the accessory pigments.

![](images/259e5771f530a591e1216a0e350b49351c1866e0b59ac09720ad2a10aac51921.jpg)  
(a) 19’-butanoyloxyfucoxanthin

![](images/e0fd1840f015cb526bb44e1f112fe4b772eac8f70f355efdeae657fddf4e60a5.jpg)  
(b) 19’-hexanoyloxyfucoxanthin

![](images/a71d6dddf6adc59f70c409213f65f746865c141e31b590e1f0528acf00fcab13.jpg)  
(c) Alloxanthin

![](images/3f0024180d9933de50f21993db144473512521cc72c676dbe1cbc447c7fd77ca.jpg)  
(d) Chlorophyll-b

![](images/188d022a778bcf4ee7530d530085bbf1545d4c13fa3b17ef04661b3acc15b93f.jpg)  
(e) Fucoxanthin

![](images/6be894f6c74a3edc3abf73516614a766d597e59e5f4b109d619fd738e442dd47.jpg)  
(f) Peridinin

![](images/17ea754a50402d08cb20924e8e6ed87fb80235c561c9e14c4fc602a4e5b65f6f.jpg)  
(g) Zeaxanthin  
Figure 6. SHAP Explainability plots showing the impact of each input variable on pigment concentrations. The x-axis represents the change in predicted pigment concentrations on a logarithmic scale, while the y-axis lists the input features used in the RF models. Features are ordered by importance, with the top feature having the greatest influence. The colours indicate the distribution of feature values and their contribution to the predicted pigment concentrations. Panels (a–g) correspond to the different pigments.

![](images/923882811ee702edd3c7594b770063832f329c7d200eb5d10e47c86445c27a64.jpg)  
(a) 19’-butanoyloxyfucoxanthin concentration

![](images/7bbc074a5c244c8e3fb28c5a5976a082ae4a4803dbf442791fdee28bc69d0a93.jpg)  
(b) 19’-butanoyloxyfucoxanthin:Chl-a ratio

![](images/3a8689b3eedbe20a2decf0ae80310916e2724d1cfe1ffeec532aee6c816b79f4.jpg)  
(c) 19’-hexanoyloxyfucoxanthin concentration

![](images/24d6e1909d200527662c0ea5eecc1eed11d5163e6227828d74d6eb3684ba4bed.jpg)  
(d) 19’-hexanoyloxyfucoxanthin:Chl-a ratio

![](images/b22ef98e8d59b500096e4937e4e4eeff4fd7e67f72632a745566e631c69bcb47.jpg)  
(e) Alloxanthin concentration

![](images/38e4324638c1e58b2d67cf3ca58309104c99805ed6ba8dde36ea63ad43c36863.jpg)  
(f) Alloxanthin:Chl-a ratio

![](images/6d0f076b40a16aecc2d531ec6d492385b4c5b9be7c7371b05d31a3397a0c5ffc.jpg)  
(g) Total chlorophyll-b concentration

![](images/d5be7142c5f989bba16e6c7fe058822ca61156f7cfdd6faf7d808fc4c9a51ffd.jpg)  
(h) Total chlorophyll-b:Chl-a ratio  
Figure 7. Global maps of predicted pigment concentrations (left column) and pigment-to-chlorophyll-a ratios (right column) for May 2010 generated using the Random Forest model that incorporates spectral information. Panels (a–h) show 19’-butanoyloxyfucoxanthin, 19’-hexanoyloxyfucoxanthin, alloxanthin and total chlorophyll-b.

![](images/42046cd1d8f48a53ea08a8c59c965fc60c96cf236f531612577bfbe9481f5f7a.jpg)  
(a) Fucoxanthin concentration

![](images/f433564800cfe463b4a81f49e80214f9db8c0b3314a727b766910a10172f78ab.jpg)  
(b) Fucoxanthin:Chl-a ratio

![](images/7e125a82e31ee09d4e66fb6c037c61917bbf26f52e73662d0d2a3da70d3dedc4.jpg)  
(c) Peridinin concentration

![](images/6cea0ed10fe350b29d4459b5d369a795ebe070be732212c4b7608145ad63de27.jpg)  
(d) Peridinin:Chl-a ratio

![](images/8255bed5ba54e7d996e16d140c2689fdf890a20dc28d84216ee7b5be14a69628.jpg)  
(e) Zeaxanthin concentration

![](images/6c6566a3b4d7e5b18b8dbfca80e3dcc09e5d3a4f00a6255aabd6c07410b3816b.jpg)  
(f) Zeaxanthin:Chl-a ratio  
Figure 8. Global maps of predicted pigment concentrations (left column) and pigment-to-chlorophyll-a ratios (right column) for May 2010 generated using the Random Forest model. Panels (a–f) show fucoxanthin, peridinin and zeaxanthin.

This aspect of the investigation was brought to the fore in the right-hand panels of Figure 7 and Figure 8, in which each accessory pigment is plotted relative to the chlorophyll concentration. These figures clearly demonstrate that the models were able to capture faithfully the shifting community structure of phytoplankton at the global scale, as brought into evidence by the accessory pigments, with, admittedly, some ambiguities when it comes to relating pigments to phytoplankton types. To highlight some notable features, the relative concentration of fucoxanthin has a global distribution that mirrors that of the absolute concentration of fucoxanthin (Figure 8a and b), consistent with the increase in diatoms with increasing chlorophyll-a concentrations. But this result is not a true test of independence from chlorophyll-a variability, since we know that fucoxanthin is highly correlated with chlorophyll-a (Figure 1). From this perspective, it is more interesting to see the relative distribution of total chlorophyllb (Figure 7h). As expected based on the known distribution of Prochlorococcus, whose concentrations are highest in the subtropical gyres, and whose pigment complement includes divinyl chlorophyll-b, the results captured this feature very well. The pigment dataset that we are using combines divinyl chlorophyll-b and regular chlorophyll-b into a total chlorophyll-b, such that some of the chlorophyll-b we see outside of the subtropical gyres may be associated with chlorophytes rather than Prochlorococcus. Similarly, relative concentrations of 19’-butanoyloxyfucocanthin (Figure 7b) considered to be marker pigments of haptophytes, revealed their predominance in the Southern Hemisphere, where as and 19- hexanoyloxyfucoxanthin(Figure 7d), also considered to be marker pigments of haptophytes tends to be present across both hemispheres, which is supported by existing literature (Liu et al., 2009). The performance of the zeaxanthin retrieval was relatively poor when compared with the test data (Figure 4d), which showed that the machine learning models do not capture much more than the mean of its concentration. This quality was reflected in Figure 7e, where zeaxanthin concentration appeared to be relatively uniformly distributed over the whole globe. Nevertheless, the distribution of the relative concentration of zeaxanthin (Figure 8f) revealed a more realistic pattern. With the exception of zeaxanthin, the predicted distributions were therefore consistent with established patterns of marine biogeography and suggest that the machine learning models recovered ecologically meaningful information from multispectral ocean colour observations.

Several pigments exhibited spatial patterns that closely resembled those of total chlorophyll-a, which is shown in the chlorophyll-a to pigment ratio plots (Figure 7b,d,f,h and Figure 7b,d,f). Fucoxanthin and peridinin showed strong enhancement in productive regions, including the North Atlantic, North Pacific, Southern Ocean and major coastal upwelling systems, while concentrations were substantially reduced within the subtropical gyres. These distributions are consistent with the ecology of larger phytoplankton groups, including diatoms and dinoflagellates, that characteristically dominate nutrientrich waters. The similarity between the fucoxanthin, peridinin and chlorophyll-a maps is also consistent with the strong chlorophyll dependence identified by the SHAP analysis and the comparatively limited improvement over chlorophyll-only baseline models. A second group of pigments, including alloxanthin, 19’-butanoyloxyfucoxanthin and total chlorophyll-b, displayed intermediate behaviour. These pigments remained associated with productive waters but exhibited weaker large-scale gradients and increased regional variability relative to fucoxanthin and peridinin. Elevated concentrations were observed at high latitudes and along equatorial regions, whereas reduced concentrations persisted in the subtropical gyres. These pigments are commonly associated with cryptophytes, chlorophytes, prasinophytes, haptophytes and pelagophytes, suggesting that the models are capturing variability associated with phytoplankton groups occupying a broader range of ecological niches. In contrast, 19’-hexanoyloxyfucoxanthin and zeaxanthin exhibited substantially broader open-ocean distributions. For 19’-hexanoyloxyfucoxanthin, elevated concentrations remained evident at high latitudes but the reduction in oligotrophic gyres was less pronounced than the other accessory pigments. The resulting distribution was more spatially continuous across ocean basins, consistent with the widespread occurrence of haptophyte communities throughout the global ocean (Liu et al., 2009). Zeaxanthin displayed the most distinctive pattern of all pigments, with comparatively weak large-scale gradients and relatively high concentrations persisting throughout tropical and subtropical waters. This pattern contrasts strongly with that of chlorophyll-a and is consistent with the global distribution of picophytoplankton communities, including Prochlorococcus and Synechococcus, which are adapted to stratified oligotrophic environments.

In comparison with total chlorophyll-a, if pigment concentrations were being estimated solely from their covariance with chlorophyll-a, all maps would be expected to exhibit similar spatial patterns. Instead, substantial differences are observed among pigments, particularly between fucoxanthin, peridinin and chlorophyll-a on one hand, and 19’-hexanoyloxyfucoxanthin and zeaxanthin on the other. These contrasting distributions suggest that the models are exploiting information beyond chlorophyll-a and are capable of recovering broad-scale variability in phytoplankton community composition from multispectral ocean colour observations.

## 4 DISCUSSION

The primary aim of this study was to evaluate whether multispectral ocean colour observations contain information that can improve estimates of diagnostic pigments beyond that provided by chlorophyll-a alone. Across all pigments examined, machine learning models trained using multispectral reflectance consistently outperformed models trained solely on chlorophyll-a. This demonstrates that ocean colour reflectance contains additional information relevant to phytoplankton pigment composition, although the magnitude of this information varies substantially between pigments.

The strongest predictive performance was obtained for fucoxanthin. However, fucoxanthin also exhibited the smallest improvement relative to the baseline model, which only had chlorophyll-a as an input. This result is consistent with both the correlation analysis and the SHAP interpretation. Fucoxanthin showed the strongest correlation with chlorophyll-a $( r \ : = \ : 0 . 9 5 )$ , and the SHAP analysis revealed that chlorophyll-a dominates model predictions. Taken together, these results suggest that a large proportion of fucoxanthin predictability arises from its covariance with total chlorophyll-a concentration.

In contrast, models of alloxanthin, 19’-butanoyloxyfucoxanthin and 19’-hexanoyloxyfucoxanthin that incorporated spectral information showed substantially larger improvements relative to the baseline models. For these pigments, the SHAP analysis demonstrated that multispectral reflectance bands and $K _ { d } ( 4 9 0 )$ were generally more influential than chlorophyll-a. The consistency between the SHAP rankings and model performance provides strong evidence that predictive skill for these pigments derives from information contained within the spectral reflectance observations rather than from chlorophyll-a alone. This supports the hypothesis that multispectral ocean colour observations contain information related to phytoplankton community composition that is not fully represented by chlorophyll-a.

The results for zeaxanthin are particularly informative. Zeaxanthin was the least accurately predicted pigment and exhibited only weak correlation with chlorophyll-a. Nevertheless, the SHAP analysis revealed that model predictions depend primarily on blue-green reflectance bands and $K _ { d } ( 4 9 0 )$ rather than chlorophyll-a. Furthermore, the global map of zeaxanthin differed markedly from those of other pigments, exhibiting a broader distribution across tropical and subtropical waters as might be expected for smaller phytoplankton species that contain this pigment. Although the overall predictive accuracy remained modest, these results suggest that the model is identifying an ecological signal distinct from total phytoplankton biomass. This finding is encouraging because it implies that ocean colour observations may contain information about cyanobacteria-associated pigments that is not captured by chlorophyll-abased approaches. It has previously been shown that satellite based sea surface temperature can be used to improve the prediction of Zea pigment (Pan et al., 2013). This could be considered for future work.

The global maps provided an additional line of evidence supporting this interpretation. The predicted distributions display coherent large scale biogeographical patterns that are consistent with known contrasts between productive high-latitude waters and oligotrophic subtropical gyres. Pigments such as fucoxanthin, peridinin and alloxanthin are typically found in larger phytoplankton cells, including diatoms, dinoflagellates and haptophytes, which have a higher nutrient demand and are therefore found in regions where nutrient concentrations are higher, i.e., coastal, upwelling and seasonally stratified regions (Kramer and Siegel, 2019). Pigments such as zeaxanthin are typically found in smaller phytoplankton cells, including Prochlorococcus, that can benefit from relatively low nutrient concentrations in open oceans. The spatial distribution of the pigments is therefore consistent with contrasting ecological niches occupied by the underlying phytoplankton communities. Importantly, the maps differ substantially between pigments. If the models were simply reproducing chlorophyll-a variability and scaling pigments according to their covariance with biomass, more similar spatial patterns would be expected. Instead, the observed differences suggest that the machine learning models are extracting information related to phytoplankton community structure and ecological niche variability.

Our results are broadly consistent with previous studies that have demonstrated limited but measurable skill in predicting pigments from remotely sensed observations (Bracher et al., 2015; El Hourany et al., 2019; Stock and Subramaniam, 2020). Earlier work has often reported stronger predictive performance than observed here, particularly when random train and test splitting strategies were employed. However, oceanographic datasets exhibit substantial spatial and temporal autocorrelation, meaning that nearby samples are frequently more similar than independent observations. Consequently, random splitting can lead to information leakage between training and validation datasets and produce overly optimistic performance estimates (Stock and Subramaniam, 2020). By withholding entire years from model training, our validation strategy provides a more realistic assessment of model generalisation to unseen observations. The reduced performance reported here may therefore better represent the predictive capability achievable in operational applications.

The SHAP analysis further demonstrates the value of explainable AI techniques for ocean colour applications. Rather than treating machine learning models as black boxes, SHAP provides insight into the physical variables driving model predictions and allows comparison between pigments. The distinction between chlorophyll-a dominated pigments, such as fucoxanthin and peridinin, and the pigments where the $\boldsymbol { R _ { r s } }$ bands are considered more important, such as alloxanthin, $1 9 ^ { \prime }$ -butanoyloxyfucoxanthin and 19’-hexanoyloxyfucoxanthin, is particularly informative. These results suggest that spectral-anomalies, around the chlorophyll-a to $R _ { r s }$ relationships, are better associated with certain pigments (Alvain et al., 2005)

Overall, the results indicate that machine learning methods can extract ecologically meaningful information on phytoplankton pigment composition from multispectral ocean colour observations. While chlorophyll-a remains a dominant predictor for some pigments, the inclusion of spectral reflectance data provides measurable improvements in predictive skill and yields physically plausible global distributions. These findings support the use of machine learning approaches as a complementary tool for deriving information on phytoplankton community composition from long-term ocean colour records and provide a foundation for future exploitation of hyperspectral satellite observations.

Our work has only been possible through the collection of in situ HPLC pigment measurements. The algorithms developed here are based on relationships between ocean colour and pigment data collected over past decades, which may change under future ocean conditions (Sathyendranath et al., 2017). Continued collection of these in situ measurements is therefore essential to improve algorithm performance and detect climate-driven changes in model parameterisations (Satterthwaite et al., 2025).

## 5 CONCLUSIONS

This study evaluated the ability of machine learning models to estimate diagnostic phytoplankton pigments from multispectral ocean colour observations. Models incorporating multispectral reflectance consistently outperformed chlorophyll-only approaches, demonstrating that ocean colour observations contain information on phytoplankton pigment composition beyond that provided by chlorophyll-a alone.

Predictive skill varied considerably between pigments. Fucoxanthin achieved the highest overall accuracy but was also the pigment most strongly associated with chlorophyll-a, indicating that much of its predictability derives from covariance with phytoplankton biomass. In contrast, alloxanthin, butfucoxanthin and hex-butfucoxanthin exhibited substantial improvements relative to chlorophyll only models, demonstrating that additional information contained within multispectral reflectance observations can be exploited to improve pigment estimation. Although zeaxanthin remained challenging to predict, its distinct SHAP signatures and global distribution suggest that the model captures ecologically meaningful variability.

The SHAP analysis and global pigment maps provided complementary evidence that the models do not simply reproduce chlorophyll-a variability. Instead, different pigments were associated with distinct combinations of spectral and optical predictors, resulting in spatial distributions that are consistent with known patterns of phytoplankton biogeography.

Overall, these results demonstrate that multispectral ocean colour observations contain measurable information on diagnostic pigments beyond chlorophyll-a. Machine learning approaches provide a practical framework for exploiting this information and offer a pathway towards improved satellite monitoring of phytoplankton community composition in the era of hyperspectral Earth Observation.

## FUNDING

This research was supported by the European Space Agency (ESA) through the Biodiversity in the Open Ocean: Mapping, Monitoring and Modelling (BOOMS, grant number 4000137125/22/I-DT ) project and the Biodiversity of the Coastal Ocean: Monitoring with Earth Observation (BiCOME, grant number 4000135756/21/I-EF ) project. Additional support was provided by the Simons Foundation through the Computational Biogeochemical Modeling of Marine Ecosystems (CBIOMES; grant 549947 to S.S.) project and the ESA PHYTOplankton biomass and diversity Climate Change Initiative (PHYTO-CCI; contract number 4000147645/25/I-LR). XS and RB were supported by a UKRI FLF grant (MR/V022792/1). This work is a contribution to the activities of the National Centre of Earth Observation (NCEO) of the UK.

## REFERENCES

Alvain, S., Moulin, C., Dandonneau, Y., and Breon, F.-M. (2005). Remote sensing of phytoplankton groups in case 1 waters from global seawifs imagery. Deep Sea Research Part I: Oceanographic Research Papers 52, 1989–2004

Armbrust, E. V. and Palumbi, S. R. (2015). Uncovering hidden worlds of ocean biodiversity. Science 348, 865–867

Baxter, I., Ding, Q., Ballinger, T., Wang, H., Holland, M., Wang, H., et al. (2025). Water sources and land capacitor effects stimulate observed summer arctic moistening and warming. Communications Earth & Environment

Behrenfeld, M. J. and Falkowski, P. G. (1997). Photosynthetic rates derived from satellite-based chlorophyll concentration. Limnology and Oceanography 42, 1–20. doi:10.4319/lo.1997.42.1.0001

Bracher, A., Bouman, H. A., Brewin, R. J. W., Bricaud, A., Brotas, V., Ciotti, A. M., et al. (2017). Obtaining phytoplankton diversity from ocean color: A scientific roadmap for future development. Frontiers in Marine Science 4

Bracher, A., Taylor, M. H., Taylor, B., Dinter, T., Rottgers, R., and Steinmetz, F. (2015). Using empirical¨ orthogonal functions derived from remote-sensing reflectance for the prediction of phytoplankton pigment concentrations. Ocean Science 11, 139–158. doi:10.5194/os-11-139-2015

Breiman, L. (2001). Random forests. Machine learning 45, 5–32

Brewin, R. J., Sathyendranath, S., Jackson, T., Barlow, R., Brotas, V., Airs, R., et al. (2015). Influence of light in the mixed-layer on the parameters of a three-component model of phytoplankton size class. Remote Sensing ofEnvironment 168, 437–450

Brewin, R. J., Sathyendranath, S., Tilstone, G., Lange, P. K., and Platt, T. (2014). A multicomponent model of phytoplankton size structure. Journal ofGeophysical Research: Oceans 119, 3478–3496

Brewin, R. J. W., Hardman-Mountford, N. J., Lavender, S. J., Raitsos, D. E., Hirata, T., Uitz, J., et al. (2011). An intercomparison of bio-optical techniques for detecting dominant phytoplankton size class from satellite remote sensing. Remote Sensing of Environment 115, 325–339. doi:10.1016/j.rse.2010. 09.004

Bricaud, A., Claustre, H., Ras, J., and Oubelkheir, K. (2004). Natural variability of phytoplanktonic absorption in oceanic waters: Influence of the size structure of algal populations. Journal of Geophysical Research: Oceans 109

Dawson, G., Vandaele, R., Taylor, A., Moffat, D., Tamura-Wicks, H., Jackson, S., et al. (2025). A sentinel-3 foundation model for ocean colour. arXiv preprint arXiv:2509.21273

Dubinsky, Z. and Schofield, O. (2010). From the light to the darkness: thriving at the light extremes in the oceans. Hydrobiologia 639, 153–171

El Hourany, R., Abboud-abi Saab, M., Faour, G., Aumont, O., Crepon, M., and Thiria, S. (2019).´ Estimation of secondary phytoplankton pigments from satellite observations using self-organizing maps (soms). Journal of Geophysical Research: Oceans 124, 1357–1378

Falkowski, P. (2012). Ocean science: the power of plankton. Nature 483, S17–S20

Field, C. B., Behrenfeld, M. J., Randerson, J. T., and Falkowski, P. (1998). Primary production of the biosphere: integrating terrestrial and oceanic components. Science 281, 237–240. doi:10.1126/science. 281.5374.237

Gordon, H. R., Clark, D. K., Mueller, J. L., and Hovis, W. A. (1980). Phytoplankton pigments from the nimbus-7 coastal zone color scanner: comparisons with surface measurements. Science 210, 63–66

Hayward, A., Pinkerton, M. H., and Gutierrez-Rodriguez, A. (2023). Phytoclass: A pigmentbased chemotaxonomic method to determine the biomass of phytoplankton classes. Limnology and Oceanography: Methods 21, 220–241. doi:10.1002/lom3.10541. Publisher: Wiley

Hollmann, N., Muller, S., Eggensperger, K., and Hutter, F. (2023). TabPFN: A transformer that ¨ solves small tabular classification problems in a second. In International Conference on Learning Representations 2023

IOCCG (2014). Phytoplankton Functional Types from Space (Reports of the International Ocean-Colour Coordinating Group, No. 15, IOCCG, Dartmouth, Canada)

Jordan, T. M., Simis, S. G., Selmes, N., Sent, G., Ienna, F., and Martinez-Vicente, V. (2023). Spatial structure of in situ reflectance in coastal and inland waters: implications for satellite validation. Frontiers in Remote Sensing 4, 1249521

Kramer, S. J. and Siegel, D. A. (2019). How can phytoplankton pigments be best used to characterize surface ocean phytoplankton groups for ocean color remote sensing algorithms? Journal ofGeophysical Research: Oceans 124, 7557–7574. doi:10.1029/2019jc015604. Publisher: American Geophysical Union (AGU)

Kulk, G., Platt, T., Dingle, J., Jackson, T., Jonsson, B., Bouman, H., et al. (2020). Primary production, an ¨ index of climate change in the ocean: Satellite-based estimates over two decades. Remote Sensing 12, 826. doi:10.3390/rs12050826

Laine, M., Kulk, G., Jonsson, B. F., and Sathyendranath, S. (2024). A machine learning model-based¨ satellite data record of dissolved organic carbon concentration in surface waters of the global open ocean. Frontiers in Marine Science 11, 1305050

Liu, H., Probert, I., Uitz, J., Claustre, H., Aris-Brosou, S., Frada, M., et al. (2009). Extreme diversity in noncalcifying haptophytes explains a major pigment paradox in open oceans. Proceedings of the national academy ofsciences 106, 12803–12808

Longhurst, A., Sathyendranath, S., Platt, T., and Caverhill, C. (1995). An estimate of global primary production in the ocean from satellite radiometer data. Journal of plankton Research 17, 1245–1271

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., et al. (2020). From local explanations to global understanding with explainable ai for trees. Nature machine intelligence 2, 56–67

Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems 30

Mackey, M., Mackey, D., Higgins, H., and Wright, S. (1996). CHEMTAX - a program for estimating class abundances from chemical markers:application to HPLC measurements of phytoplankton. Marine Ecology Progress Series 144, 265–283. doi:10.3354/meps144265

Majchrowski, R. and Ostrowska, M. (2000). Influence of photo-and chromatic acclimation on pigment composition in the sea. Oceanologia 42

Nair, A., Sathyendranath, S., Platt, T., Morales, J., Stuart, V., Forget, M.-H., et al. (2008). Remote sensing of phytoplankton functional types. Remote Sensing ofEnvironment 112, 3366–3375

Opitz, D. and Maclin, R. (1999). Popular ensemble methods: An empirical study. Journal of artificial intelligence research 11, 169–198

Ozer, D. J. (1985). Correlation and the coefficient of determination. Psychological bulletin 97, 307

Pan, X., Wong, G. T., Ho, T.-Y., Shiah, F.-K., and Liu, H. (2013). Remote sensing of picophytoplankton distribution in the northern south china sea. Remote Sensing of Environment 128, 162–175

Platt, T. and Sathyendranath, S. (1988). Oceanic Primary Production: estimation by remote sensing at local and regional scales. Science 241, 1613–1620. doi:10.1126/science.241.4873.1613

Reynolds, C. S. (2006). The ecology of phytoplankton (Cambridge University Press)

Sathyendranath, S., Brewin, R. J., Brockmann, C., Brotas, V., Calton, B., Chuprin, A., et al. (2019). An ocean-colour time series for use in climate studies: the experience of the ocean-colour climate change initiative (oc-cci). Sensors 19, 4285

Sathyendranath, S., Brewin, R. J., Jackson, T., Melin, F., and Platt, T. (2017). Ocean-colour products´ for climate-change studies: What are their ideal characteristics? Remote Sensing of Environment 203, 125–138

Sathyendranath, S., Platt, T., Kovac,ˇ Z., Dingle, J., Jackson, T., Brewin, R. J., et al. (2020). Reconciling <sup>ˇ</sup> models of primary production and photoacclimation. Applied Optics 59, C100–c114

Satterthwaite, E. V., Field, J. C., Fassbender, A. J., Aceves-Medina, G., Bograd, S. J., Hazen, E. L., et al. (2025). The essential role of large research vessels in marine ecosystem observations and ocean sustainability. Limnology and Oceanography 70, 2767–2792

Silsbe, G. M., Fox, J., Westberry, T. K., and Halsey, K. H. (2025). Global declines in net primary production in the ocean color era. Nature Communications 16, 5821

Sournia, A., Chrdtiennot-Dinet, M.-J., and Ricard, M. (1991). Marine phytoplankton: how many species in the world ocean? Journal ofPlankton Research 13, 1093–1099

Stock, A. and Subramaniam, A. (2020). Accuracy of Empirical Satellite Algorithms for Mapping Phytoplankton Diagnostic Pigments in the Open Ocean: A Supervised Learning Perspective. Frontiers in Marine Science 7. doi:10.3389/fmars.2020.00599. Publisher: Frontiers Media SA

Sun, X., Brewin, R. J., Sathyendranath, S., Dall’Olmo, G., Antoine, D., Barlow, R., et al. (2025). Coupling ecological concepts with an ocean-colour model: Parameterisation and forward modelling. Remote Sensing ofEnvironment 316, 114487

Sun, X., Brewin, R. J., Sathyendranath, S., Dall’Olmo, G., Antoine, D., Barlow, R., et al. (2026). Coupling ecological concepts with an ocean-colour model: inversion modelling. Frontiers in Remote Sensing 6, 1692306

Tett, P. and Barton, E. (1995). Why are there about 5000 species of phytoplankton in the sea? Journal of Plankton Research 17, 1693–1704. doi:10.1093/plankt/17.8.1693

Toming, K., Kulk, G., Quast, R., Shevchuk, R., and Kutser, T. (2026). Towards a global assessment of coastal dissolved organic carbon. Remote Sensing ofEnvironment 339, 115388

Vidussi, F., Claustre, H., Manca, B. B., Luchetta, A., and Marty, J. (2001). Phytoplankton pigment distribution in relation to upper thermocline circulation in the eastern Mediterranean Sea during winter. Journal ofGeophysical Research Atmospheres 106, 19939–19956. doi:10.1029/1999jc000308