# Explainable Prediction of High-Speed Solar Wind Streams driven by Coronal Holes via Topological Data Analysis

Francesco Marchetti<sup>1,2</sup> ·

Sabrina Guastavino<sup>2,3</sup> · Daniel Collin<sup>4,5</sup>

© The author(s) ••••

Abstract High-speed solar wind streams (HSSs) originating from coronal holes (CHs) are an important driver of recurrent geomagnetic activity. Since HSSs are associated to coronal structures visible in extreme-ultraviolet observations, this provides a natural basis for forecasting several days in advance. In this work, we investigate whether the topology of coronal structures observed in extreme ultraviolet (EUV) images contains predictive information on the subsequent occurrence of CH-driven HSSs at Earth. To this end, we consider Topological Data Analysis (TDA), and in particular persistent homology, to classify EUV solar images according to the occurrence of HSSs at Earth four days later. Persistent homology is used to extract multiscale topological information from EUV images at wavelength 193 Å, which is subsequently encoded through diferent vector representations and used as input to machine-learning classifiers. The target is a binary classification of HSS occurrence at a fourday lead time, defined using an HSS interval catalog and solar wind speed thresholds. Several combinations of topological representations and classifiers

are evaluated under a common cross-validation framework. The best-performing configurations, based on persistence-image representations and linear classifiers, achieve an area under the ROC curve (AUC) of approximately 0.82 and a True Skill Statistic (TSS) of approximately 0.43. These results indicate that the topology of EUV coronal structures alone contains predictive information on subsequent CH-driven HSS occurrence, despite the absence of explicit CH geometric descriptors in the model input. To investigate the origin of this predictive information, we develop a model-level explainability framework based on cross-validation-aggregated feature weights and spatial evidence maps. A fully traced example connects persistence diagrams and persistence-image features to the classifier output and to their spatial localization in the original EUV observation. The proposed framework provides a transparent and computationally lightweight approach for investigating the relationship between coronal topology and subsequent high-speed solar wind streams.

Keywords: Solar wind; Coronal holes; Corona; Machine learning; Topological Data Analysis; High-speed solar wind streams

## 1. Introduction

Near-Earth space weather disturbances are primarily associated with coronal mass ejections (CMEs; Webb and Howard (2012); Georgoulis et al. (2024)) and high-speed solar wind streams (HSSs), both of which can perturb the near-Earth environment and afect satellites and technological infrastructure (Schwenn 2006; Temmer 2021; Richardson 2018). While CMEs are responsible for many of the most intense geomagnetic storms, recurrent HSSs represent an important source of sustained geomagnetic activity, particularly during the declining and minimum phases of the solar cycle (Richardson, Cliver, and Cane 2000; Tsurutani et al. 2006; Telloni et al. 2020; Guastavino et al. 2024). HSSs are commonly associated with coronal holes (CHs), regions of reduced extreme-ultraviolet (EUV) and X-ray emission characterized by predominantly open magnetic-field configurations that allow plasma to escape into the heliosphere. The link between coronal holes and fast solar wind has a long observational history: the seminal identification of a CH as source of a high-velocity stream dates back to the Skylab era (Krieger, Timothy, and Roelof 1973), and subsequent studies have investigated the relation between CH properties, high-speed streams, and recurrent geomagnetic activity (Nolte et al. 1977; Sheeley, Harvey, and Feldman 1976; Cranmer 2002).

The association between CHs and HSSs provides a natural basis for forecasting solar wind conditions at Earth. Since their solar sources can be observed before the associated streams reach 1 AU, EUV observations of the solar disk can provide predictive information several days in advance, with typical Sun-Earth propagation times of the order of a few days. Nevertheless, establishing a quantitative relationship between solar observations and subsequent in-situ solar wind conditions remains challenging. The solar wind speed measured at Earth does not depend on a single property of the source region (Hofmeister et al.

2018), and the propagation and evolution of streams through the heliosphere can modify their signatures before they reach 1 AU (Richardson 2018; Hofmeister et al. 2022). Moreover, approaches based on explicit CH identification introduce additional dependence on segmentation procedures and boundary definitions (Reiss et al. 2021, 2024), as well as on the association between individual CHs and the corresponding in-situ streams.

Several studies have therefore investigated the relationship between observable properties of CHs and the characteristics of the associated HSSs. Empirical approaches have related solar wind speed to quantities such as CH area and position on the solar disk (Vršnak, Temmer, and Veronig 2007; Rotter et al. 2012), while more recent studies have exploited image-derived features and machine-learning techniques to capture more complex relationships between coronal observations and the solar wind (Bu et al. 2019; Samara et al. 2022; Collin et al. 2025). Physics-based models provide a complementary route by modeling the magnetic configuration of the corona and the propagation of the solar wind through the heliosphere. More recently, physically informed and compact machine-learning representations have also shown that useful predictive skill can be obtained without necessarily resorting to increasingly complex architectures (Collin et al. 2025).

However, a key open issue is that predictive performance does not automatically imply interpretability. In many image-based models, the link between learned internal representation and physically meaningful solar structures remains opaque.

Topological Data Analysis (TDA) ofers a complementary approach for characterizing the structure of solar images through their topological properties. In particular, persistent homology characterizes how topological structures evolve as a dataset is examined across a sequence of filtration levels. When applied to images, it allows connected components and loops to be followed across intensity levels rather than describing structures at a single, predefined threshold. Their appearance and disappearance are summarized through persistence diagrams, providing a multiscale representation of the topology of the image. Persistent homology has well-established mathematical foundations, including algorithmic formulations and stability results (Edelsbrunner, Letscher, and Zomorodian 2002; Zomorodian and Carlsson 2005; Cohen-Steiner, Edelsbrunner, and Harer 2007; De Marchi et al. 2022), and its use in data analysis has grown substantially through the development of vectorized representations suitable for statistical and machine-learning methods (Otter et al. 2017; Chazal, Michel, and Rinaldo 2021). From the perspective of solar EUV observations, this provides an appealing way of retaining information on the organization and persistence of image structures without reducing the input a priori to a small set of predefined CH descriptors.

In this work, we investigate whether topological information extracted from full-disk 193 Å EUV solar images contains predictive information on the occurrence of HSSs at Earth four days later. Persistent homology is used to characterize the images, and diferent vectorizations of the resulting persistence diagrams are evaluated in combination with machine-learning classifiers under a common cross-validation framework. Importantly, the models are based exclusively on topological representations: no explicit CH segmentation, area, location, or other geometrical CH descriptors are provided as input. Our objective is therefore not to introduce persistent homology as a new mathematical methodology, nor to propose a state-of-the-art operational HSS forecasting system, but rather to assess how much predictive information can be extracted from the topology of EUV observations alone.

A second objective is to investigate where the predictive information identified by the classifiers is located in the original solar observations. To this end, we complement the classification analysis with an explainability framework based on fold-aggregated model weights and spatial evidence maps. For a representative example, we trace the complete chain from the EUV image and its persistence diagram to the vectorized topological representation, standardized classifier input, linear model output, and spatially localized positive and negative evidence. This provides a direct connection between the mathematical topological representation, the resulting classification, and the solar structures contributing to the model decision.

The manuscript is structured as follows. Section 2 introduces persistent homology with emphasis on the concepts needed in this work. Section 3 details the end-to-end pipeline and implementation choices. Section 4 presents results and interpretability analysis, including a fully traced worked example. Conclusions are ofered in Section 5.

## 2. On persistent homology

Topological Data Analysis (TDA) provides a mathematical framework for characterizing the structure of data through their topological properties. In this work, we use persistent homology, a TDA technique that describes how topological structures appear, evolve, and disappear as a scalar field is examined across multiple levels (Edelsbrunner, Letscher, and Zomorodian 2002; Zomorodian and Carlsson 2005; Otter et al. 2017). When applied to an image, persistent homology therefore provides a multiscale description of its intensity structure without requiring the selection of a single threshold.

## 2.1. From an image to a persistence diagram

Let $I ( u , v ) \in \mathbb { R } , \ ( u , v ) \in \mathbb { N } \times $ N denote the intensity associated with pixel $( u , v )$ in a two-dimensional image I. The pixel grid can be analyzed through a cubical complex, in which pixels and their adjacencies define the underlying combinatorial structure (Otter et al. 2017; Chazal, Michel, and Rinaldo 2021).

A simple way of investigating the structure of an image is to select an intensity threshold α and consider only pixels whose intensities lie above or below that threshold. For example, two standard choices consist in taking sublevel or superlevel set defined as

$$
K _ { \alpha } ^ { \mathrm { s u b } } = K ( \{ ( u , v ) \mid I ( u , v ) \leq \alpha \} ) , \qquad K _ { \alpha } ^ { \mathrm { s u p } } = K ( \{ ( u , v ) \mid I ( u , v ) \geq \alpha \} ) ,\tag{1}
$$

where $\kappa ( \cdot )$ denotes the induced complex. In practice, both conventions are equivalent up to monotone transformation (e.g., replacing I by $\operatorname* { m a x } ( I ) - I )$ . In the following introduction, we refer to the superlevel approach because it provides a direct visualization of the thresholding process with the considered examples, which are $\mathrm { ~ a ~ } ^ { 6 } 0 ^ { 9 }$ and $^ { 6 6 } 8 ^ { 9 }$ digit from the well known MNIST handwritten-digit dataset (LeCun et al. 1998) (see Figure 1).

![](images/d1e9742bfaf20712e3a6e362feba7037464daa694ff533caa57bf4d4b2ec53ad.jpg)  
(a) $\mathrm { ~ A ~ } ^ { 6 } 9 ^ { 9 }$ from MNIST.

![](images/26d8926fff405e6281676af0ed8ca963e1c0c43440359723023c4a54e273c800.jpg)  
(b) An $^ { 6 6 } 8 ^ { 3 }$ from MNIST.  
Figure 1.: MNIST images considered.

At each filtration level, homology characterizes topological structures of diferent dimensions. For the two-dimensional images considered here, we focus on $H _ { 0 }$ which describes connected components, and $H _ { 1 }$ which describes one-dimensional holes or loops. Formally, the h-th homology group identifies h-dimensional cycles that are not boundaries; a detailed algebraic treatment can be found, for example, in (Edelsbrunner, Letscher, and Zomorodian 2002; Zomorodian and Carlsson 2005; Chazal, Michel, and Rinaldo 2021). For the purposes of the present work, the relevant point is that $H _ { 0 }$ and $H _ { 1 }$ provide complementary descriptions of the connectivity and hole structure of image-intensity patterns.

As α varies, the topology of $K _ { \alpha } ^ { \mathrm { s u b } }$ or $K _ { \alpha } ^ { \mathrm { s u p } }$ changes: connected regions can appear or merge, and holes can form or disappear. The dependence of these structures on the selected threshold can be illustrated using a simple example. Figure 2 shows two superlevel sets obtained at two diferent intensity thresholds from the $^ { 6 6 } 8 ^ { 9 }$ digit. The topology of the active set $\{ ( u , v ) \mid I ( u , v ) > \alpha \}$ changes as the threshold varies: structures that are visible at one level may merge or disappear at another level. In particular, one snapshot has one hole, while another has two holes. Consequently, describing the image at a single threshold would retain only a particular snapshot of its topology.

![](images/da2d4f3a6034ee06ad4f840bbb6026904e7cd03b9c4187242e083f8de16e234c.jpg)  
(a) MNIST 8 original image.

![](images/6727c04e0c8da45f160c8a09ce7ac06561d288511dd7b23e3ae27bb139d3fe19.jpg)  
(b) Superlevel set at $\alpha =$ 0.166.

![](images/1aa6e81407191fe9043994c7327ce063c912985ab0e90549f914c0536a5e1027.jpg)  
(c) Superlevel set at $\alpha =$ 0.519.  
Figure 2.: MNIST-8 example.

Rather than selecting a single threshold, persistent homology considers the entire sequence of thresholded sets, referred to as a filtration. Sublevel sets grow with α $( K _ { \alpha } ^ { \mathrm { s u b } } \subseteq K _ { \beta } ^ { \mathrm { s u b } }$ for $\alpha \leq \beta )$ , while superlevel sets shrink. Therefore, each topological structure is tracked throughout the filtration: A structure is assigned a birth value $b ,$ corresponding to the filtration level at which it appears, and a death value $d ,$ corresponding to the level at which it disappears or merges into an older structure, building a persistence pair $( b , d ) , b < d ,$ with persistence $p = d - b$ . The collection of persistence pairs in homology dimension h defines the persistence diagram $D [ h ] = \{ ( b _ { i } , d _ { i } ) \} _ { i = 1 } ^ { N _ { h } }$ , where $N _ { h }$ denotes the number of topological generators in homology dimension h. Points farther from the diagonal $b \ = \ d$ correspond to structures that persist over a larger range of filtration values, whereas points close to the diagonal represent short-lived topological structures and are generally more sensitive to perturbations. Persistence itself, however, does not determine whether a structure is relevant for a particular prediction problem: short- and long-lived generators may carry diferent amounts of information depending on the data and task.

To visualize this construction, we compute the persistence diagrams of the two MNIST samples in Figure 3. We observe that the digit “0” yields one dominant $H _ { 0 }$ generator (one connected component) and one dominant $H _ { 1 }$ generator (one principal hole), while the digit “8” yields one dominant $H _ { 0 }$ generator and two dominant $H _ { 1 }$ generators (two principal holes).

![](images/c23c691b57bf25d020877da604760523f761fde63a297af8fe5964eebd09ba08.jpg)

![](images/909b5179224e39d13b2446f651e2ac3e3544053ec030a3453af29742dae1bd88.jpg)  
(a) Persistence diagram (b) Persistence diagram  
extracted from Figure 1a. extracted from Figure 1b.  
Figure 3.: Persistence diagrams constructed from Figure 1. Both $H _ { 0 } / H _ { 1 }$ homologies are shown. Birth and death coordinates refer to the x- and y-axis, respectively. We find important generators far from the diagonal, whereas points close to the diagonal yield unstable, noisy generators.

An important mathematical property of persistence diagrams is their stability with respect to perturbations of the underlying scalar field (Cohen-Steiner, Edelsbrunner, and Harer 2007), that is, controlled perturbations of image intensities induce controlled changes in their persistence diagrams. This property provides an important motivation for using persistent homology as a representation of image structure, being suitable in case of noisy data, although its practical behavior also depends on the adopted preprocessing and filtration.

## 2.2. Vectorizing persistence diagrams for machine learning

Even if persistence diagrams provide a compact representation of topological structures, they are multisets of birth–death pairs whose cardinality generally varies from image to image. Therefore, a persistence diagram cannot be directly used as a fixed-dimensional input to standard machine-learning algorithms. To address this issue, a vectorization maps $D [ h ]$ into a fixed-dimensional representation $x [ h ] \in \mathbb { R } ^ { d }$ that can subsequently be used as input to a machine-learning model (Otter et al. 2017; Chazal, Michel, and Rinaldo 2021).

In this work, we consider four commonly used persistence-based representations: Betti curves, persistence landscapes, silhouettes, and persistence images. These representations retain diferent aspects of the information contained in the persistence diagram and therefore allow us to investigate whether the predictive information depends on how the underlying topology is encoded.

Betti curves. The Betti curve counts how many classes are alive at each filtration level t:

$$
\beta _ { k } ( t ) = \sum _ { i } { \bf 1 } _ { \{ b _ { i } \leq t < d _ { i } \} } .
$$

Sampling $\beta _ { k } ( t )$ on a fixed grid yields a finite-dimensional vector. Intuitively, this captures how much topology is active at each scale, without tracking individual generators (Otter et al. 2017; Chazal, Michel, and Rinaldo 2021). In the context of EUV solar images, the Betti curve describes how the number of topological structures changes as diferent intensity levels are explored. For $H _ { 0 }$ , it summarizes the number of disconnected intensity regions present at each filtration level, whereas for $H _ { 1 }$ it describes the number of holes enclosed by such regions. Therefore, it provides a global description of how the connectivity of the EUV intensity pattern evolves across the filtration, without retaining the birth and death coordinates of individual generators.

Persistence landscapes. For each point $( b _ { i } , d _ { i } )$ , define the triangular (tent) function

$$
\lambda _ { i } ( t ) = \operatorname* { m a x } \bigl ( 0 , \operatorname* { m i n } ( t - b _ { i } , d _ { i } - t ) \bigr ) .
$$

This is zero outside $[ b _ { i } , d _ { i } ]$ , peaks at $( b _ { i } + d _ { i } ) / 2$ , and has height $( d _ { i } - b _ { i } ) / 2$ . A landscape layer $\Lambda _ { m } ( t )$ is the m-th largest value among $\{ \lambda _ { i } ( t ) \}$ <sub>i</sub> at each t. For an EUV image, persistence landscapes describe at which intensity-related filtration levels the most persistent topological structures are present and how long they remain across the filtration. The first landscape layers (low m) emphasize the dominant persistent structures, while higher layers (large m) retain information about additional structures occurring over similar filtration ranges. They therefore provide a representation of the hierarchy of persistent connectivity and hole patterns present in the solar image (Bubenik 2015).

Silhouettes. A silhouette is a weighted average of landscape functions:

$$
\mathrm { S i l } [ h ] ( t ) = \frac { \sum _ { i } \omega _ { i } \lambda _ { i } ( t ) } { \sum _ { i } \omega _ { i } } , \qquad \omega _ { i } = ( d _ { i } - b _ { i } ) ^ { p } ( p \geq 0 ) .
$$

Compared with full landscapes, silhouettes compress information into one curve while emphasizing persistent generators through $w _ { i }$ (Chazal, Michel, and Rinaldo 2021). In an EUV solar image, the silhouette provides a global persistenceweighted profile of the topological structures encountered across the intensity filtration. Rather than retaining separate information for individual generators or landscape layers, it summarizes where in the filtration the overall topological activity is concentrated. The parameter $p$ controls the emphasis placed on longlived structures: increasing $p$ gives progressively greater influence to structures that persist across a wider range of intensity-related filtration levels.

Persistence images. First map points to birth-persistence coordinates $( b _ { i } , p _ { i } )$ with $p _ { i } = d _ { i } - b _ { i }$ . Then place weighted Gaussian kernels and integrate on a fixed grid:

$$
\rho [ h ] ( x , y ) = \sum _ { i } w ( p _ { i } ) \exp \left( - \frac { ( x - b _ { i } ) ^ { 2 } + ( y - p _ { i } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{2}
$$

followed by pixel-wise integration. Flattening the resulting grid gives a vector of fixed length. This representation is thus stable and naturally compatible with linear models (Adams et al. 2017). For an EUV solar image, diferent regions of the persistence image therefore correspond to topological structures emerging at diferent intensity-related levels and characterized by diferent lifetimes. In particular, the representation distinguishes short-lived structures from structures that remain topologically identifiable over a broad range of the filtration. By retaining this two-dimensional organization, persistence images make it possible to identify which ranges of birth and persistence are most relevant to the classifier and, subsequently, to relate these contributions back to structures in the original solar image

The four representations therefore summarize persistence information in different ways: Betti curves encode the number of simultaneously active generators, landscapes retain ordered persistence profiles, silhouettes provide a weighted functional summary, and persistence images preserve a two-dimensional representation of the distribution of generators in birth–persistence space. Their practical behavior depends on choices such as filtration, preprocessing, weighting, smoothing, and discretization resolution (Cohen-Steiner, Edelsbrunner, and Harer 2007; Fasy et al. 2014).

In the next Section 3, we describe how this persistent homology pipeline is constructed from the 193 Å EUV observations and then used as input to the classification models. For the persistence-image representation, the connection between the original solar image, persistence diagram, vectorized features, classifier coeficients, and spatial evidence is further examined in Section 4.2.

## 3. Data and methodology

The proposed framework investigates whether the topology of full-disk EUV solar images contains information about the occurrence of high-speed solar wind streams (HSSs) at Earth four days later. The analysis consists of four main steps: i) definition of the image-HSS pairs and of the corresponding binary target; ii) preprocessing of the EUV observations and extraction of persistent-homology features; iii) classification using diferent vector representations and machinelearning models; and iv) interpretation of the learned topological features and their localization in the original solar images.

## 3.1. Data and label definition

Following the observational setting adopted in (Collin et al. 2025), we combine three synchronized data sources. Full-disk extreme-ultraviolet (EUV) observations are obtained from the 193 Å channel of the Atmospheric Imaging Assembly (AIA; (Lemen et al. 2012)) on board the Solar Dynamics Observatory (SDO; (Pesnell and Chamberlin 2012)), with an hourly cadence. The images are corrected for position, rotation, and scaling of the solar disk, light scattering, and instrument degradation as described in (Collin et al. 2026). Near-Earth solar wind speed measurements are obtained from the hourly OMNI dataset (Papitashvili and King 2020), while HSS intervals are identified using the catalog constructed through the peak-filtering procedure described in (Collin et al. 2025, 2026).

The AIA 193 Å channel is particularly suitable for the present analysis because coronal holes (CHs), which constitute the main source regions of HSSs, appear as regions of reduced EUV emission relative to the surrounding corona.

Importantly, however, no explicit CH segmentation is performed in the proposed approach. The entire full-disk EUV image is used as the scalar field from which the topological representation is extracted.

For each EUV image acquired at time t, the classification target is defined according to the solar wind conditions observed at t + 4 days. The four-day lead time is motivated by the typical transit time of CH-driven high-speed solar wind from the Sun to Earth and is consistent with the forecasting setup adopted in (Collin et al. 2025, 2026). The binary labels are defined as

$$
y = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ t h e ~ t a r g e t ~ t i m e ~ b e l o n g s ~ t o ~ a n ~ H S S ~ i n t e r v a l } } } \\ { 1 , } & { { \mathrm { w i t h ~ } } { \tilde { V } } ( t + 4 \mathrm { d } ) > 5 0 0 { \mathrm { ~ k m s } } ^ { - 1 } , } \\ { ~ } & { { \mathrm { i f ~ t h e ~ t a r g e t ~ t i m e ~ d o e s ~ n o t ~ b e l o n g ~ t o ~ a n ~ H S S ~ i n t e r v a l } } } \\ { 0 , } & { { \mathrm { a n d ~ } } { \tilde { V } } ( t + 4 \mathrm { d } ) < 4 0 0 { \mathrm { ~ k m } } { \mathrm { s } } ^ { - 1 } . } \end{array} \right. }
$$

Samples that do not satisfy either condition are excluded from the analysis. Here, $\tilde { V } ( t + 4 \mathrm { d } )$ denotes the OMNI hourly solar wind speed at the timestamp nearest to t + 4 days (accepted only if the nearest sample is within 12 hours). Therefore, the threshold is applied to a single target-time speed value, not to a maximum over a time interval.

This definition deliberately introduces a separation between the positive and negative classes. The intermediate cases are excluded to reduce ambiguity in the association between the solar observation and the subsequent solar wind regime.

The resulting dataset contains 71,976 labeled EUV images acquired between 2010-05-18 and 2024-06-30, of which 13,968 are assigned to the positive $( y = 1 )$ HSS class and 58,008 to the negative $( y = 0 )$ non-HSS class.

We point out that an important issue in the evaluation of HSS prediction models is the strong temporal dependence of the solar wind time series, which exhibits an auto-correlation for up to 4 days and recurring auto-correlation peaks every multiple of 27 days. The latter are associated with long-lived coronal holes that can produce recurrent HSSs reaching Earth over successive solar rotations. A random CV split would cause data leakage if either temporally close data points or recurring coronal holes appear in both training and test data. To avoid this, we discard 90 days of data between training, validation, and test datasets.

Furthermore, the state of the solar corona and correspondingly the solar wind parameters vary over the solar cycle (Richardson, Cliver, and Cane 2000; Tsurutani et al. 2006). Therefore, we ensure that data subsets are distributed across the solar cycle. To form the five subsets, we divide the dataset into 20 continuous chunks of approximately nine months each, and assign to each subset four of those chunks, while ensuring that each data point appears exactly once as test data. The resulting CV splits can be seen in Figure 4. The overall performance of a tested model is assessed by computing the evaluation metrics on the aggregated set of all test data predictions.

![](images/810497a6362f2e1ba5a588c17ff0600c02e62c174bf7e9284d197c1d3dbd1e3f.jpg)  
Figure 4.: Chronological 5-fold split used in this work. Colors indicate train/test allocation by fold over time; each test fold is a contiguous temporal block to reduce recurrence-driven leakage.

## 3.2. Preprocessing and topology extraction assumptions

Since the images have values spanning diferent orders of magnitude, we normalize them as suggested by (Jarolim et al. 2021). First, we clip the values to an upper threshold of 7757.31. Then, we rescale the values to [0, 1], apply an inverse hyperbolic sine function stretch to the image I as

$$
I _ { \mathrm { n o r m } } = \frac { \mathrm { a s i n h } ( I / a ) } { \mathrm { a s i n h } ( 1 / a ) } , \qquad a = 0 . 0 0 5 ,
$$

and rescale to [−1, 1]. Here, $a > 0$ controls contrast in low-intensity regions. These steps enhance coronal-hole–specific features, reduce strong variations near active regions, and ensure a consistent numerical range.

Further, we invert the normalized images, using

$$
I _ { \mathrm { p r o c } } = \operatorname* { m a x } ( I _ { \mathrm { n o r m } } ) - I _ { \mathrm { n o r m } } .
$$

In the pipeline, unlike the examples in the previous section, we then use sublevel filtration on the processed image $I _ { \mathrm { p r o c } } .$ . The inversion has a practical role: CH regions are dark in raw EUV intensity, and inversion turns them into peaks in $I _ { \mathrm { p r o c } } ,$ , so CH-like structures are entered early and consistently in the sweep. Moreover, inversion changes which homology classes carry the physically relevant signal. Without inversion, CH darkness is represented indirectly through complements/background, and $H _ { 0 }$ fragment statistics become sensitive to weak contrast fluctuations. With inversion, CH boundaries are more naturally represented by loop-like structures, and robust $H _ { 1 }$ generators associated with CH morphology are separated earlier from near-diagonal noise. Importantly, we do not assume an a priori correspondence between coronal holes and a specific homology dimension. The $H _ { 0 }$ and $H _ { 1 }$ persistence diagrams, respectively, characterize the evolution of connected components and hole-like structures across the filtration. Their relevance to HSS classification, and their spatial association with structures in the original EUV images, are subsequently investigated through the classification and explainability analyses.

## 3.3. Classification models and evaluation

As previously described, a persistence diagram $D [ h ]$ built on a preprocessed EUV image $I _ { \mathrm { p r o c } }$ is vectorized in $\mathbb { R } ^ { d }$ as $x [ h ] , h \in \{ 0 , 1 \}$ . The contribution of both homologies can be stacked in a vector

$$
x = ( x [ 0 ] , x [ 1 ] ) \in \mathbb { R } ^ { 2 d } .
$$

Let $x = ( x _ { 1 } , \ldots , x _ { 2 d } )$ denote the components of this resulting topological features vector. First, we standardize each feature as

$$
x _ { j } ^ { \mathrm { s t d } } = \frac { x _ { j } - \mu _ { j } } { \sigma _ { j } } , \qquad j = 1 , \dotsc , 2 d ,
$$

where $\mu _ { j }$ and $\sigma _ { j }$ are the mean and standard deviation of feature j computed on the training subset of the corresponding fold. This distinction is important for interpretation: x retains the original topological amplitudes, whereas $x ^ { \mathrm { s t d } }$ is the representation on which the classifier will be fitted and evaluated.

Since diferent classifiers may exploit the information encoded in the topological features in diferent ways, we consider a set of models spanning linear, tree-based, boosting, and kernel-based approaches. This allows us to investigate not only whether the topological representations contain predictive information, but also whether such information can be captured through relatively simple relationships or requires more complex nonlinear decision rules.

Logistic Regression and LinearSVM. We first consider two linear classifiers, Logistic Regression and a linear Support Vector Machine (LinearSVM) (Cox 1958; Cortes and Vapnik 1995). Logistic Regression models the probability of HSS occurrence as

$$
p ( \boldsymbol { y } = 1 \mid \boldsymbol { x } ) = \mathrm { s i g m o i d } ( \boldsymbol { w } ^ { \top } \boldsymbol { x } + \delta ) , \qquad \mathrm { s i g m o i d } ( z ) = \frac { 1 } { 1 + e ^ { - z } } ,\tag{3}
$$

where w contains the coeficients associated with the topological features and δ is the intercept. LinearSVM similarly separates the two classes through a linear decision boundary, but determines this boundary by maximizing the separation margin between the classes. These models provide a useful baseline for assessing whether HSS and non-HSS observations can already be distinguished through linear combinations of the topological features. Moreover, their signed coeficients provide a direct connection between individual features and the model decision, making linear models particularly suitable for the explainability analysis presented in Section 3.4.

Random Forest and Extra Trees. Random Forest and Extra Trees are ensemble methods that combine the predictions of multiple decision trees (Breiman 2001; Geurts, Ernst, and Wehenkel 2006). Unlike the linear models, they can capture nonlinear relationships and interactions among the topological features without requiring these relationships to be specified in advance. Random Forest introduces variability among the individual trees through resampling of the training data and random feature selection, whereas Extra Trees further increases randomization when constructing the decision boundaries within individual trees. Their inclusion therefore allows us to assess whether nonlinear combinations of topological features provide additional information for distinguishing images associated with subsequent HSS occurrence.

XGBoost. XGBoost is a gradient-boosting approach in which decision trees are constructed sequentially, with each new tree aimed at improving the errors of the current ensemble (Chen and Guestrin 2016). This iterative construction makes it possible to represent complex nonlinear relationships and feature interactions while controlling model complexity through regularization. In the present analysis, XGBoost provides a complementary test of whether combinations of topological features that are not captured by linear decision boundaries improve the classification of HSS and non-HSS observations.

GreedyMatern. Finally, we consider a kernel-based classifier employing a Matérn kernel together with a greedy selection strategy. Kernel methods implicitly represent the observations in a feature space in which nonlinear relationships between the original topological descriptors can be captured through pairwise similarities. The Matérn kernel provides a flexible notion of similarity controlled by its smoothness and length-scale parameters, while the greedy procedure selects a reduced set of representative basis elements. This provides a nonlinear alternative to the tree-based approaches while retaining a comparatively compact model representation (Camattari et al. 2024; Rasmussen and Williams 2006; Wenzel, Santin, and Haasdonk 2023).

Taken together, these models cover substantially diferent assumptions on the relationship between the topological representation and the HSS target. The linear classifiers test whether the relevant information can be recovered through additive linear contributions of the topological features, whereas the tree-based, boosting, and kernel approaches allow progressively more flexible nonlinear relationships. The comparison is therefore intended primarily to assess how consistently predictive information can be extracted from the diferent topological representations, rather than to identify an optimized operational forecasting architecture.

For each combination of topological vectorization and classifier, model selection and performance evaluation are carried out independently within each of the five temporal cross-validation folds described in Section 3.1. Model hyperparameters are selected using only the training and validation subsets of the corresponding fold via a grid search. Once the model configuration has been selected, its performance is evaluated on the test subset, which is not involved in either model fitting or hyperparameter selection. Since the five test subsets are mutually exclusive, the predictions obtained across the five folds can be concatenated to provide one out-of-sample prediction for every observation included in the evaluation.

Model performance is evaluated through F1 score, Area Under Curve (AUC), and TSS, which are defined as

$$
\mathrm { F 1 } = \frac { \mathrm { 2 T P } } { \mathrm { 2 T P } + \mathrm { F P } + \mathrm { F N } } ,
$$

$$
\mathrm { A U C } = \int _ { 0 } ^ { 1 } \mathrm { T P R } ( \mathrm { F P R } ) d ( \mathrm { F P R } ) ,
$$

where

$$
\mathrm { T P R } = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } , \qquad \mathrm { F P R } = \frac { \mathrm { F P } } { \mathrm { F P } + \mathrm { T N } } ,
$$

and

$$
\mathrm { T S S } = \mathrm { T P R } - \mathrm { F P R } ,
$$

given true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN). To contextualize these scores under class imbalance, we also report a random baseline driven by fold prevalence. In fold $k ,$ let $q _ { k }$ denote the proportion of training samples belonging to the positive (HSS) class. The random classifier then assigns each sample to the HSS class with probability $q _ { k } ,$ and to the non-HSS class with probability $1 - q _ { k }$ . This yields an expected baseline with $\mathrm { A U C } = 0 . 5$ and $\mathrm { T S S } = 0$ , while the other metrics depend on prevalence mismatch between train and test subsets.

Because the objective of this work is to investigate the predictive information contained in the topological representation rather than to optimize an operational forecasting system, the comparison across classifiers presented in Section 4.1 is primarily used to assess robustness of the information carried by the diferent topological vectorizations.

## 3.4. Topological attribution and spatial evidence maps

The classification analysis described above establishes whether the topological representations contain information that can discriminate between images associated with HSS and non-HSS conditions. However, classification scores alone do not indicate which topological features drive the model decisions or where the corresponding structures are located in the original EUV observations. We therefore complement the classification analysis with an explainability procedure that connects the vectorized persistence features used by the classifier to their spatial localization in the original EUV image. A complete example illustrating all the steps of this procedure is presented in Section 4.2.

## 3.4.1. From model weights to topological relevance.

For each model-vectorization pair, the cross-validation procedure produces K independently trained models, one for each fold. Each model receives as input the 2d standardized topological features introduced above. After training, we extract from the model fitted in each fold a measure of the relevance assigned to each input feature. These 2d feature-relevance values describe, at the model level, which components of the topological representation are used to discriminate between the HSS and non-HSS classes. We have therefore the following weights

$$
w ^ { ( k ) } = ( w ^ { ( k ) } [ 0 ] , w ^ { ( k ) } [ 1 ] ) = \left( w _ { 1 } ^ { ( k ) } , \ldots , w _ { 2 d } ^ { ( k ) } \right) , \qquad k = 1 , \ldots , K ,
$$

where $K = 5$ in the present analysis. For linear classifiers, $w _ { j } ^ { ( k ) }$ corresponds directly to the coeficient associated with feature $j .$ Its sign indicates the direction of the contribution to the decision function: positive coeficients are associated with the HSS class and negative coeficients with the non-HSS class. For treebased models, $w _ { j } ^ { ( k ) }$ is the model feature importance (Breiman 2001; Geurts, Ernst, and Wehenkel 2006; Chen and Guestrin 2016); for GreedyMatern, it is permutation importance (Breiman 2001; Fisher, Rudin, and Dominici 2019).

The resulting vector

$$
\bar { w } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } w ^ { ( k ) }\tag{4}
$$

provides a global, cross-validation-aggregated description of feature relevance. Features with large absolute values of $\bar { w } _ { j }$ are those receiving a consistently strong contribution across folds, while, for signed relevance measures such as linearmodel coeficients, the sign distinguishes contributions associated with the HSS and non-HSS classes.

## 3.4.2. Projection onto the EUV image.

The final step consists of relating the relevant topological features to structures in the original solar observation. Persistence features are defined in filtration space rather than directly in the spatial coordinates of the image. Their birth and death values, however, are associated with levels of the scalar field used to construct the filtration. This correspondence provides the link between the topological representation and the EUV image.

Due to its better performance with respect to the other vectorization strategies, which are reported in Table 1, in the following we restrict the methodology to the persistence image vectorization scheme. Consequently, with a slight abuse of notation, it is more comfortable to reshape both $x [ h ] ^ { \mathrm { s t d } }$ and $\bar { w } [ h ]$ from vectors in $\mathbb { R } ^ { d }$ to images in $\mathbb { R } ^ { \sqrt { d } } \times \mathbb { R } ^ { \sqrt { d } }$

For a selected observation, we then build homology-specific contribution maps in persistence-image coordinates:

$$
c [ h ] ( i , j ) = x [ h ] ^ { \mathrm { s t d } } ( i , j ) \bar { w } [ h ] ( i , j ) , \qquad h \in \{ 0 , 1 \} ,
$$

where i indexes persistence bins and $j$ indexes birth bins. We then collapse along persistence bins to obtain birth-axis profiles

$$
s [ h ] ( j ) = \sum _ { i } C [ h ] ( i , j ) , \qquad s ( j ) = s [ 0 ] ( j ) + s [ 1 ] ( j ) .
$$

Let $I _ { \mathrm { p r o c } } ( u , v )$ be the normalized/inverted scalar field used as the starting point for the TDA. If $N _ { h }$ is the number of birth bins, we define the birth-bin index of each pixel as the quantization

$$
j [ h ] ( u , v ) = \left\lfloor \frac { I _ { \mathrm { p r o c } } ( u , v ) - \operatorname* { m i n } ( I _ { \mathrm { p r o c } } ( u , v ) ) } { \operatorname* { m a x } ( I _ { \mathrm { p r o c } } ( u , v ) ) - \operatorname* { m i n } ( I _ { \mathrm { p r o c } } ( u , v ) ) } \left( N _ { h } - 1 \right) \right\rfloor , \qquad h \in \{ 0 , 1 \} ,
$$

where ⌊·⌋ indicates the integer part or floor function. This is a linear quantization of the pixel intensity into the same birth-axis discretization used by the persistence-image representation: each pixel is assigned to the birth-intensity band with which it is most naturally associated under the filtration (not to a unique topological generator). The projected fields are then obtained by:

$$
E [ h ] ( u , v ) = s [ h ] \big ( j [ h ] ( u , v ) \big ) .
$$

In this way, each pixel receives the signed contribution value of its birth bin, and feature-level information is transformed into a spatial signed evidence field over the solar disk. Note that E and $I _ { \mathrm { p r o c } }$ are images of the same dimensions, which is crucial for the interpretation of the topological analysis on the original signal.

Positive evidence in $E [ h ]$ denotes image regions whose associated topological features support classification as HSS, whereas negative evidence denotes regions whose associated features support the non-HSS class.

Operationally, we first compute the signed projected field $E [ h ]$ on disk pixels, then smooth it with a Gaussian filter (default $\sigma = 2$ pixels) to reduce discretization artifacts. To identify the regions displaying suficiently strong evidence, we define separate thresholds for the positive and negative components

$$
\tau [ h ] _ { + } = Q _ { \xi } \left( \{ E [ h ] _ { + } ( u , v ) : ( u , v ) \in \Omega \} \right) ,
$$

$$
\tau [ h ] _ { - } = Q _ { \xi } \left( \{ E [ h ] _ { - } ( u , v ) : ( u , v ) \in \Omega \} \right) ,
$$

where Ω denotes the set of pixels belonging to the solar disk and $Q _ { \xi }$ is the empirical quantile operator of order $\xi \in [ 0 , 1 ]$ . In this work, we set $\xi = 0 . 5 5$ corresponding to the 55th percentile of the respective evidence distributions. Then we compute

$$
\begin{array} { r } { R [ h ] ( u , v ) = \displaystyle \frac { \operatorname* { m a x } ( E [ h ] _ { + } ( u , v ) - \tau [ h ] _ { + } , 0 ) } { \operatorname* { m a x } ( E [ h ] _ { + } ) - \tau [ h ] _ { + } + \varepsilon } \mathrm { , } } \\ { B [ h ] ( u , v ) = \displaystyle \frac { \operatorname* { m a x } ( E [ h ] _ { - } ( u , v ) - \tau [ h ] _ { - } , 0 ) } { \operatorname* { m a x } ( E [ h ] _ { - } ) - \tau [ h ] _ { - } + \varepsilon } \mathrm { , } } \end{array}
$$

with small $\varepsilon > 0$ . This produces graded red (R) and blue (B) overlays on the EUV image instead of binary masks. Thanks to the thresholding produced by the quantile, pixels become active only when they exceed the corresponding onset, and are then displayed with continuous intensity proportional to exceedance. Finally, we regularize the supports with morphology (opening/closing), keep connected components larger than a minimum area (default 120 pixels, with largest-component fallback), and apply one dilation step for visual continuity.

These operations suppress isolated speckles and preserve coherent structures while retaining the sign and relative strength of the evidence.

The resulting maps should not be interpreted as segmentations of CHs, nor do they impose an a priori association between a given homology class and a particular solar structure. Rather, they provide a spatial representation of the regions of the EUV observation associated with topological features contributing to the classifier decision. This allows us to investigate, a posteriori, whether the predictive topological evidence is spatially associated with CHs or with other structures present in the solar corona.

## 4. Results and Discussion

## 4.1. Global performance of classification methods

In Table 1, we show the top 10 model/vectorization configurations according to the mean performance achieved in the test folds considered. Moreover, in Figure 5 we show the global performance of all methods.

Table 1. Performance comparison of the ten best-performing model–vectorization combinations for HSS prediction, ranked by the F1 score. Performance is evaluated in terms of F1 score, AUC, and TSS. Bold values indicate the highest score achieved for each metric among the reported combinations.
<table><tr><td>Model</td><td>Vectorization</td><td>F1</td><td>AUC</td><td>TSS</td></tr><tr><td>LogisticRegression</td><td>image</td><td>0.5093</td><td>0.8204</td><td>0.4238</td></tr><tr><td>LinearSVM</td><td>image</td><td>0.5090</td><td>0.8203</td><td>0.4288</td></tr><tr><td>LogisticRegression</td><td>silhouette</td><td>0.4924</td><td>0.8057</td><td>0.3979</td></tr><tr><td>LinearSVM</td><td>silhouette</td><td>0.4875</td><td>0.8003</td><td>0.3937</td></tr><tr><td>XGBoost</td><td>image</td><td>0.4714</td><td>0.8156</td><td>0.3864</td></tr><tr><td>LogisticRegression</td><td>landscape</td><td>0.4829</td><td>0.7700</td><td>0.3848</td></tr><tr><td>XGBoost</td><td>betti</td><td>0.4771</td><td>0.8064</td><td>0.3816</td></tr><tr><td>LogisticRegression</td><td>betti</td><td>0.4786</td><td>0.7948</td><td>0.3794</td></tr><tr><td>ExtraTrees</td><td>betti</td><td>0.4581</td><td>0.8034</td><td>0.3692</td></tr><tr><td>RandomForest</td><td>image</td><td>0.4666</td><td>0.8110</td><td>0.3670</td></tr></table>

For the same folds and labels, the prevalence-driven random baseline is computed as follows. In each fold k, we estimate the prevalence of the HSS class in the training subset as follows

$$
q _ { k } ^ { \mathrm { t r a i n } } = \frac { N _ { \mathrm { H S S , t r a i n } } ^ { ( k ) } } { N _ { \mathrm { t r a i n } } ^ { ( k ) } } ,
$$

where $N _ { \mathrm { H S S , t r a i n } } ^ { ( k ) }$ denotes the number of class 1 and $N _ { \mathrm { t r a i n } } ^ { ( k ) }$ the number of samples in the training subset of fold $k ,$ respectively, then, for each test sample in that fold, we draw a random prediction from Bernoulli $( q _ { k } ^ { \mathrm { t r a i n } } )$ . This uses only training information to set the random predictor and applies it on the test subset.

![](images/8ab70080724eb404f0d57c83fa55d68368c33542cc6590777b21c8ad4d48401f.jpg)

(a) F1.  
![](images/592a6870edac840398aa743cbc034583dda97fc3176273cb2de6bcf48ba9545d.jpg)

(b) AUC.  
![](images/8c5000be699a47a7f637036c53955540ba7b1a2a407aa27ff1bce5373017eab8.jpg)  
(c) TSS.  
Figure 5.: Performance comparison across the diferent model–vectorization combinations for HSS prediction. The panels show the average performance in terms of (a) F1 score, (b) AUC, and (c) TSS.

Let $q _ { k } ^ { \mathrm { t e s t } }$ be test prevalence in fold k. Under this baseline, expected fold metrics are

$$
\mathrm { F } 1 _ { k } = \frac { 2 q _ { k } ^ { \mathrm { t e s t } } q _ { k } ^ { \mathrm { t r a i n } } } { q _ { k } ^ { \mathrm { t e s t } } + q _ { k } ^ { \mathrm { t r a i n } } } , \qquad \mathrm { A U C } _ { k } = 0 . 5 , \qquad \mathrm { T S S } _ { k } = 0 .
$$

The resulting mean baseline is $\mathrm { F 1 } = 0 . 1 9 5 6 , \mathrm { A U C } = 0 . 5 .$ and $\mathrm { T S S } = 0 ,$ substantially lower than all reported TDA-based models in F1, AUC, and TSS. A complete overview for all folds is displayed in Table 2.

Table 2. Fold-wise class distribution and prevalence-driven random baseline. In fold k, random predictions use $q _ { k } ^ { \mathrm { t r a i n } }$ . The last column reports $\Delta \mathrm { F } 1 _ { k } = \mathrm { F } 1 _ { k } ^ { \mathrm { r a n d } } - \mathrm { F } 1 _ { k } ^ { \mathrm { L o g R e g + i m a g e } }$ . The second and third columns represent the number of positive samples and total number of samples in the test set associated to k fold.
<table><tr><td>Fold</td><td># Test HSS</td><td># Test non-HSS</td><td> $q _ { k } ^ { \mathrm { t r a i n } }$ </td><td>F1</td><td>∆F1 vs LogReg+image</td></tr><tr><td>0</td><td>2496</td><td>10704</td><td>0.1937</td><td>0.1913</td><td>-0.2665</td></tr><tr><td>1</td><td>2928</td><td>11016</td><td>0.2008</td><td>0.2053</td><td>-0.1218</td></tr><tr><td>2</td><td>2592</td><td>11448</td><td>0.1983</td><td>0.1912</td><td>-0.3286</td></tr><tr><td>3</td><td>3216</td><td>12480</td><td>0.1888</td><td>0.1965</td><td>-0.3924</td></tr><tr><td>4</td><td>2760</td><td>12336</td><td>0.2054</td><td>0.1935</td><td>-0.3276</td></tr></table>

The results reported show that both linear and non-linear approaches provide comparable performance, with no vectorization method consistently outperforming the others. This may suggest that the underlying data might not contain substantially more information that can be exploited to further improve performance. Although LinearSVM+image is first by TSS, LogisticRegression+image is nearly tied and provides cleaner signed linear explainability, so we adopt it as primary interpretation model for our subsequent XAI. Therefore, in Figure 6 we display the global weights (see equation (4)) corresponding to the image vectorization strategy coupled with the trained logistic classifier, both for $H _ { 0 }$ and $H _ { 1 }$ extracted features. Note that the dimension of the persistence image is $1 4 \times 1 4$ , which turned out to be a satisfactory trade-of between data representation and model complexity in our tests. The mixed-sign profile confirms that both HSS-supporting and quiet-supporting regions exist in feature space.

## 4.2. Worked example: full chain from diagram to decision

To clarify our XAI methodology, in the following we track a correctly classified positive sample through the entire pipeline.

Before discussing the topological pipeline itself, it is useful to place this sample in its temporal forecasting context. The left panel of Figure 7 shows the original AIA 193 Å image at time $t _ { 0 } = 2 0 1 0 { - } 0 8 { - } 2 3 0 6 { : } 0 0$ , while the right panel reports the hourly solar wind speed time series around the forecast target time t + 4 days. The plotted thresholds (400 and $5 0 0 \mathrm { k m / s } )$ are the same operational cutofs used for non-HSS/HSS labeling in this work, and highlighted HSS intervals come from the considered catalog. This links the image-level morphology to the downstream event definition before entering the persistent-homology steps.

![](images/fee19d0dc674c114bd76699540b2c008754ca41770294fa5f6c3024005bb8198.jpg)  
(a) $H _ { 0 }$ global weights.

![](images/f2ae6096146834553ab2834f3733fab07c765630fd1334de2fa40b013aa0c785.jpg)  
(b) $H _ { 1 }$ global weights.

Figure 6.: Global-weight explanation outputs for LogisticRegression+image, shown separately for $H _ { 0 }$ and $H _ { 1 }$ to improve readability of axes and colorbar labels.  
![](images/92438a35457febed9d8f772210b4c9906e958b43c27bc4bede2edcd88fd06771.jpg)

![](images/7ba7259a8165af1e1c1db33b738ad6a34dcc741ef344ac224826e60444b7c3b3.jpg)  
Figure 7.: Worked-example context. Left: original AIA 193 $\mathrm { \AA }$ image at $t _ { 0 } =$ 2010-08-23 06:00. Right: hourly solar wind speed around the corresponding forecast window, with non-HSS/HSS thresholds and temporal intervals used for labeling.

## Step 1: persistence diagram

The first step consists of processing the original AIA image (normalization and inversion) and then computing the persistence diagram for both $H _ { 0 }$ and $H _ { 1 }$ homologies out of the processed image (Figure 8). In light of the MNIST examples in Section 2, the contrast is evident: instead of one or two clearly isolated dominant loops (as for the digits $^ { 6 6 } 0 ^ { 9 }$ and $^ { 6 6 } 8 ^ { 3 3 } )$ , the solar diagram contains a much denser population of generators across scales. This means that the coronal morphology around this event is topologically richer than the synthetic handwritten-digit cases. After normalization and inversion, CH-related boundaries and other coronal structures produce persistent loop classes $\left( H _ { 1 } \right)$ that appear farther from the diagonal, while the many points close to the diagonal correspond to short-lived, fine-scale fluctuations. In other words, the solar case combines robust mesoscale structures with a broad background of transient small-scale topology.

![](images/0a086384a9baff22c5efb31d300cb4c9d6d76145f5d7876a8a6ed9bea832c7c6.jpg)  
(a) $H _ { 0 }$ persistence diagram.

![](images/b5ab7f07f2210dc9b062b6807eaae2aac5516a5120121159b677e00a0293048d.jpg)  
(b) $H _ { 1 }$ persistence diagram.  
Figure 8.: Persistence diagrams of the selected sample shown separately for $H _ { 0 }$ and $H _ { 1 }$ (left and right panels, respectively). The maximum finite birth/death coordinate is 0.7366 (max $H _ { 1 }$ death), so there are no finite generators above that value and the axis cap at 0.75 is data-driven.

## Step 2: persistence-image vectorization

The second step consists of vectorizing the persistence diagrams separately. This leads to two persistence images (Figure 9). The visually sparse appearance is expected: most generators are concentrated at low persistence and around specific birth ranges, while bins receiving contributions from near-diagonal generators remain non-zero but comparatively small. Because color scaling is linear and dominated by the densest bins, these low-amplitude bins appear close to black although they are not exactly zero.

## Step 3: feature contribution maps in persistence-image space

Then, images are standardized and, according to the logistic regression model, they are then multiplied times the vector of weights as shown in Figure 10.

The transformation from persistence-image (PI) contributions to annotated solar images follows the procedure defined in Section 3.4: construction of $C [ h ] ( i , j )$ , birth-axis aggregation $s [ h ] ( j )$ , projection through $j [ h ] ( u , v )$ , and quantile-thresholded continuous red/blue overlays after smoothing. Thus, Figure 10 is the PI-space decomposition, while Figure 11 is its deterministic image-space projections.

## 4.3. Analysis of topologically annotated samples

In addition to the HSS example presented in the previous subsection, here we include in the discussion the topological annotation linked to a correctly classified non-HSS sample in Figure 12, a false negative sample in Figure 13 and a false positive one in Figure 14.

![](images/e3f90318c2b9cab10f0907ce5412da7bbe83bed80c089c3f2e98ddad73fb93a1.jpg)  
(a) $H _ { 0 }$ persistence image.

![](images/58d49b0abb981716b1c7e076e9e9e6136a5e47b4a2ba6115c4c583db19760908.jpg)  
(b) $H _ { 1 }$ persistence image.  
Figure 9.: Persistence-image vectorization (14 × 14) shown separately for $H _ { 0 }$ and H<sub>1</sub> (left and right panels, respectively). Image values are in logarithmic scale. Bins having zero values are colored in black.

Across the four examples, a first qualitative point is that HSS-supporting (red) evidence is present in all cases. This is expected, and it also explains why a purely visual “red area vs blue area” reading is not suficient. The classifier decision is made in persistence-image feature space (see equation (3)), not in the displayed overlay space. Therefore, what changes the class is the sign of the signed weighted sum over all bins, including many weak contributions that are not visually salient in the final maps. The red/blue overlays are a post-hoc projection of those signed contributions (Section 3.4), further processed for readability by persistence-axis collapse, smoothing, quantile onset $( q = 5 5 \% )$ , and separate red/blue normalization. Consequently, they are faithful as qualitative localization maps, but not as a literal visual encoding of the exact decision process. For reference, the stored LogisticRegression+image probabilities for these examples are: p ≈ 1.0 for the HSS true positive (Figure 11), $p = 9 . 4 8 \times 1 0 ^ { - 1 0 }$ for the non-HSS true negative (Figure 12), $p = 0 . 5 7 9$ for the selected false positive (Figure 14), and $p = 0 . 4 5 9$ for the selected false negative (Figure 13).

We highlight that our model efectively associates HSS-supporting regions in $H _ { 1 }$ to CHs, strongly indicating that the topological analysis was able to extract physically relevant information. Moreover, to test whether the slight HSS-supporting contribution near active-region boundaries is recurrent, we performed an exploratory post-hoc analysis of the 30 EUV images for which Logistic Regression models based on persistence images had been used to generate spatial evidence maps for annotation. These images represent a selected subset rather than the entire test set. For each image t, we defined a positive-evidence mask $P _ { t }$ as the union of red pixels in the $H _ { 0 } / H _ { 1 }$ evidence maps. We also defined an approximate AR-boundary mask $A _ { t }$ from the corresponding normalized AIA image. Specifically, we identified the brightest regions within the solar disk and applied morphological operations to extract their boundaries. This mask therefore provides a brightness-based proxy for AR boundaries rather than an explicit segmentation of individual active regions. To quantify the spatial association between positive evidence and these boundaries, we computed

![](images/d1fb94b9adbb2ea3ecf153fd8371591390ce328c22f57b70ea900b36ea9cc521.jpg)  
(a) $H _ { 0 } \colon x _ { 0 } ^ { \mathrm { s t d } }$

![](images/2c07e2e4922e5d2d98971e72107ae0fdb0b644761165bd5de8b1537c820b7c90.jpg)  
(b) $H _ { 0 } \colon \bar { w } _ { 0 }$

![](images/8a562e07e8cd20194af7d4007ef15c3eac891a45df647806d5cbf63ff53d1dc4.jpg)  
(c) $H _ { 0 } \colon x _ { 0 } ^ { \mathrm { s t d } }$ ⊙ w¯<sub>0</sub>.

![](images/8aa6c8a1facbb1d47b2218743f9f141119ddd0ed9167fec63fbbb6ac94358b36.jpg)  
(d) $H _ { 1 } \colon x _ { 1 } ^ { \mathrm { s t d } }$

![](images/d33c06d922dd807e97bb12676d7942ef82b1f1906f36db62a17bd599b3c5b31d.jpg)  
(e) $H _ { 1 } \colon \bar { w } _ { 1 }$

![](images/a3559ac7ec49afe3cfb453f69ef2989a8f941590d11c3ae4b5b4791e9827d6d7.jpg)  
(f) $H _ { 1 } \colon x _ { 1 } ^ { \mathrm { s t d } }$ ⊙ w¯<sub>1</sub>.

Figure 10.: Step-3 decomposition with single-panel assets: standardized feature maps, global weight maps, and element-wise contributions for $H _ { 0 }$ and $H _ { 1 }$  
![](images/912c8c6a0d36b8e2529d1aa9eeca92c4cb57df18b8bede4510dc8f1d60deef3e.jpg)  
(a) Processed AIA.

![](images/263269843e57340f41bf9c01f2f548e6114ca703611eb3dec9025beb9eaae309.jpg)  
(b) $H _ { 0 }$ evidence.

![](images/6ab3d218728ce74f3e2c64fa751d50768a598fb0e3f9c2628dc07726d396ad1e.jpg)  
(c) $H _ { 1 }$ evidence.  
Figure 11.: HSS example (2010-08-23 06:00): preprocessed AIA view with corresponding $H _ { 0 }$ and $H _ { 1 }$ projected evidence maps.

$$
r _ { t } = \frac { | P _ { t } \cap A _ { t } | } { | P _ { t } | } , \qquad a _ { t } = \frac { | A _ { t } | } { | D _ { t } | } , \qquad e _ { t } = \frac { r _ { t } } { a _ { t } } ,
$$

![](images/aa81055224c7baf994f533febbafd4334ed1c5fa01260af8eee59f5731504c4e.jpg)  
(a) Processed AIA.

![](images/d728d17808dd1412b1ca6e852a8b9b45622a8700091c8139d62bd3acb11d5a6a.jpg)  
(b) $H _ { 0 }$ evidence.

![](images/4b166fe16e66542e26f44e8c2a853d9c77a34db197d9a08d060ec5763c0852a0.jpg)  
(c) $H _ { 1 }$ evidence.

Figure 12.: Non-HSS example (2014-02-25 01:00): processed AIA view with corresponding $H _ { 0 }$ and $H _ { 1 }$ projected evidence maps. Red indicates HSS-supporting projected evidence, blue indicates quiet-supporting projected evidence.  
![](images/d95606899c4c256a09019432b0af7ff1a904ec8db1b7b0cd9cb58885229c26ae.jpg)  
(a) Processed AIA.

![](images/0c133599aba258eca2e64b862d52007c49806e411a0675498c1b9b5b777010c5.jpg)  
(b) $H _ { 0 }$ evidence.

![](images/39ba93ee0e54c616b1c204a5daa7b2303df779ac095fc69894289c912f385c77.jpg)  
(c) $H _ { 1 }$ evidence.  
Figure 13.: False negative example (2022-02-16 21:00): processed AIA view with corresponding $H _ { 0 }$ and $H _ { 1 }$ projected evidence maps. Red indicates HSSsupporting projected evidence, blue indicates non-HSS-supporting projected evidence.

where $D _ { t }$ is the disk mask. Here, $r _ { t }$ is the fraction of positive-evidence pixels overlapping the approximate AR boundaries, while $a _ { t }$ is the fraction of the solar disk occupied by those boundaries. Their ratio, $\boldsymbol { e } _ { t } ,$ measures the spatial enrichment of positive evidence near AR boundaries relative to a uniform distribution over the solar disk. Values $e _ { t } > 1$ indicate greater overlap than expected under this reference distribution. We find $e _ { t } > 1$ in 24/30 targets (80%), with median enrichment factor 1.46 (interquartile range 1.05–1.81). However, the spatial overlap was generally limited: the median fraction of positive-evidence pixels overlapping the AR-boundary proxy was 5.9%, exceeding 10% in only 9 of the 30 images. These findings suggest a recurrent but relatively weak spatial association between positive topological evidence and bright-region boundaries in the selected images.

This spatial association should be interpreted with caution, as it may reflect both properties of the underlying solar structures and characteristics of the explainability procedure. From a physical perspective, the complex intensity patterns surrounding active regions may contribute to the topological information extracted from EUV images. However, a methodological explanation is also possible. In constructing the spatial evidence maps, feature contributions in the persistence-image representation are aggregated along the persistence dimension to obtain a profile as a function of birth level. Consequently, multiple low-persistence topological features with similar birth values may collectively produce a noticeable contribution, even when their individual contributions are small. The positive evidence observed near bright-region boundaries may therefore partly reflect the accumulation of weak, short-lived topological features rather than the contribution of highly persistent structures. Overall, these findings suggest a recurrent but secondary spatial association, which should not be interpreted as evidence that active-region boundaries independently predict HSS occurrence.

![](images/1d7346c0a70d96ade5c41500b0d9ddb19a6cfbb37e0c37f04115e83be7a771d4.jpg)  
(a) Processed AIA.

![](images/1423946dfddac2a28f73a21eb6140d7b379a6f51fbcd67308020cb7cc7b44a31.jpg)  
(b) $H _ { 0 }$ evidence.

![](images/fafd07ffefbfc3b81436d94c9828dfbe294431a6cac85de62103d4577071ace3.jpg)  
(c) $H _ { 1 }$ evidence.  
Figure 14.: False positive example (2020-08-14 09:00): processed AIA view with corresponding $H _ { 0 }$ and $H _ { 1 }$ projected evidence maps. Red indicates HSSsupporting projected evidence, blue indicates non-HSS-supporting projected evidence.

## 4.4. Discussion of limitations

Our approach prioritizes interpretability and reproducibility, but several limitations should be kept explicit.

First, the explanation is global-model based: projected maps are derived from fold-aggregated weights and sample features, not from a local surrogate fitted around a single image. This is useful for identifying recurrent discriminative patterns, but it does not by itself establish local causal mechanisms for one specific event.

Second, label construction is intentionally strict $( > 5 0 0 \mathrm { k m / s }$ for positive samples, i.e HSS samples, < 400 km/s for negative samples, i.e. non-HSS samples). This improves class separability and interpretability of decision boundaries, but removes intermediate regimes that may still be physically meaningful. The reported metrics therefore describe performance under this strict operational definition, not under a fully continuous wind-speed target.

Third, although the temporal cross-validation strategy reduces the risk of information leakage associated with the recurrence of solar structures over successive rotations, model performance varies across the diferent temporal folds. Nevertheless, the results consistently indicate that linear classifiers combined with persistence images provide promising predictive performance. Small differences in performance between individual model configurations should therefore be interpreted with caution, particularly when they are comparable to the variability observed across folds.

Fourth, the image projection is not unique. We adopt birth-axis aggregation because it provides a deterministic and interpretable mapping from feature space to image space: each pixel has a direct local scalar value, hence a natural birth-bin assignment, while death/persistence coordinates generally do not admit a unique local pixel assignment without additional cycle-tracking assumptions. This collapse therefore improves interpretability and reproducibility of the projection operator. By contrast, avoiding the aggregation and projecting full birth–persistence contributions would require extra modeling assumptions (e.g., local allocation rules for persistence/death, generator tracking, or kernel redistribution). Those alternatives can preserve more 2D persistence information, but they are less direct, more sensitive to implementation choices, and harder to compare across samples.

Finally, the pipeline remains topology-only by design. This is a strength for transparency, but it may miss complementary geometric descriptors (e.g., explicit CH area/elongation) that could improve calibration in edge cases. Particularly, previous studies (see, e.g. Rotter et al. (2012); Reiss et al. (2016); Collin et al. (2025)) have found that the main CH properties that contribute to the solar wind speed are the CH area and the location of the CH. These properties are not encoded by the topological features. A controlled hybrid benchmark is thus a natural next step rather than a contradiction of the present framework.

## 5. Conclusions

In this work, we investigated whether topological information extracted from full-disk EUV solar images contains predictive information on the occurrence of coronal-hole-driven high-speed solar wind streams at Earth four days later. By combining persistent homology with diferent vectorization techniques and machine-learning classifiers, we showed that the topology of EUV intensity patterns provides measurable predictive information, even without explicitly identifying or segmenting individual coronal holes or introducing predefined geometrical descriptors.

Among the investigated approaches, persistence images combined with linear classifiers provided particularly promising results, achieving an AUC of approximately 0.82 and a TSS of approximately 0.42–0.43. These findings suggest that the connectivity and hole structure of EUV intensity patterns, characterized across multiple filtration levels, contain information relevant to the subsequent occurrence of HSSs at Earth.

Beyond classification performance, we developed an explainability framework that connects the topological representations to the classifier decisions and their spatial interpretation in the original EUV observations. By examining both global feature relevance and image-specific contributions, the proposed approach allows us to identify which regions of the topological representation influence the classification and to explore how the corresponding evidence is distributed across the solar disk. The worked example illustrates the complete connection between persistence diagrams, vectorized features, model coeficients, and spatial evidence maps, providing a means to investigate the relationship between topological information and observed coronal structures.

The present study is intended as an assessment of the predictive information contained in EUV image topology rather than as an operational HSS forecasting system. Its conclusions are limited by the adopted binary event definition, the fixed four-day prediction horizon, and the dependence of the spatial evidence maps on the projection procedure. Future work will investigate the sensitivity of the results to diferent prediction horizons and HSS definitions, assess the robustness of the spatial attribution maps, and compare topology-based representations with approaches combining topological and geometrical information under the same temporal validation framework.

Author Contribution F.M. developed and implemented the methodology and performed the analysis. D.C. contributed to dataset preparation. S.G. and D.C. contributed to the experimental design and contributed to the interpretation of the results. F.M. wrote the main manuscript text. All authors reviewed and approved the manuscript.

Funding AIA data used here are available by courtesy of the AIA and HMI consortia. We acknowledge use of NASA/GSFC’s Space Physics Data Facility’s OMNIWeb service. F.M. and S.G. acknowledge financial support from the Istituto Nazionale di Astrofisica (INAF) through the grant “Investigating coronal holes’ shapes and their relationship with high-speed solar wind streams via topological data analysis”. F.M. and S.G. are also grateful to the Gruppo Nazionale per il Calcolo Scientifico - Istituto Nazionale di Alta Matematica (GNCS - INdAM). D.C. acknowledges the support of the Helmholtz International Berlin Research School in Data Science (HEIBRiDS).

Data Availability The observational data and HSS list used in this study are the same as those described in (Collin et al. 2025) and are publicly available through https://doi.org/ 10.5880/gfz.2.7.2024.001 (Collin et al. 2024). Software for downloading and preprocessing of solar images was developed in (Collin et al. 2026) and is available at https://github.com/ DanielCollin96/solar\_image\_processing (Collin and Hofmeister 2026).

## References

Adams, H., Emerson, T., Kirby, M., Neville, R., Peterson, C., Shipman, P., Chepushtanova, S., Hanson, E., Motta, F., Ziegelmeier, L.: 2017, Persistence images: A stable vector representation of persistent homology. Journal of Machine Learning Research 18, 1.

Breiman, L.: 2001, Random forests. Machine Learning 45, 5. DOI.

Bu, X., Li, X., Wan, W., Liang, J.: 2019, Forecasting high-speed solar wind streams based on solar extreme ultraviolet images. Space Weather 17, 1482. DOI.

Bubenik, P.: 2015, Statistical topological data analysis using persistence landscapes. Journal of Machine Learning Research 16, 77.

Camattari, F., Guastavino, S., Marchetti, F., Piana, M., Perracchione, E.: 2024, Classifierdependent feature selection via greedy methods. Statistics and Computing 34, 151. DOI.

Chazal, F., Michel, B., Rinaldo, A.: 2021, An introduction to topological data analysis: Fundamental and practical aspects for data scientists. Frontiers in Artificial Intelligence 4, 667963. DOI.

Chen, T., Guestrin, C.: 2016, XGBoost: A scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785. DOI.

Cohen-Steiner, D., Edelsbrunner, H., Harer, J.: 2007, Stability of persistence diagrams. Discrete & Computational Geometry 37, 103. DOI.

Collin, D., Hofmeister, S.J.: 2026, Solar Image Processing, Zenodo. DOI. URL.

Collin, D., Shprits, Y., Hofmeister, S.J., Bianco, S., Gallego, G.: 2024, Solar Wind Speed Prediction from Coronal Holes, GFZ Data Services. DOI. URL.

Collin, D., Shprits, Y., Hofmeister, S.J., Bianco, S., Gallego, G.: 2025, Forecasting high-speed solar wind streams from solar images. Space Weather 23, e2024SW004125. DOI.

Collin, D., Shprits, Y., Chiarabini, L., Hofmeister, S.J., Klein, N., Gallego, G.: 2026, PROSWIN: Probabilistic Solar Wind Speed Forecasting Using Deep Distributional Regression From Solar Images. DOI. URL.

Cortes, C., Vapnik, V.: 1995, Support-vector networks. Machine Learning 20, 273. DOI.

Cox, D.R.: 1958, The regression analysis of binary sequences. Journal of the Royal Statistical Society: Series B 20, 215.

Cranmer, S.R.: 2002, Coronal holes and the high-speed solar wind. Space Science Reviews 101, 229. DOI.

De Marchi, S., Lot, F., Marchetti, F., Poggiali, D.: 2022, Variably Scaled Persistence Kernels (VSPKs) for persistent homology applications. Journal of Computational Mathematics and Data Science 4, 100050. DOI. URL.

Edelsbrunner, H., Letscher, D., Zomorodian, A.: 2002, Topological persistence and simplification. Discrete & Computational Geometry 28, 511. DOI.

Fasy, B.T., Lecci, F., Rinaldo, A., Wasserman, L., Balakrishnan, S., Singh, A.: 2014, Confidence sets for persistence diagrams. The Annals of Statistics 42, 2301. DOI.

Fisher, A., Rudin, C., Dominici, F.: 2019, All models are wrong, but many are useful: Learning a variable’s importance by studying an entire class of prediction models simultaneously. Journal of Machine Learning Research 20, 1.

Georgoulis, M.K., Yardley, S.L., Guerra, J.A., Murray, S.A., Ahmadzadeh, A., Anastasiadis, A., Angryk, R., Aydin, B., Banerjee, D., Barnes, G., et al.: 2024, Prediction of solar energetic events impacting space weather conditions. Advances in Space Research.

Geurts, P., Ernst, D., Wehenkel, L.: 2006, Extremely randomized trees. Machine Learning 63, 3. DOI.

Guastavino, S., Bahamazava, K., Perracchione, E., Camattari, F., Audone, G., Telloni, D., Susino, R., Nicolini, G., Fineschi, S., Piana, M., et al.: 2024, Forecasting geofective events from solar wind data and evaluating the most predictive features through machine learning approaches. The Astrophysical Journal 971, 94.

Hofmeister, S.J., Veronig, A., Temmer, M., Vennerstrom, S., Heber, B., Vršnak, B.: 2018, The dependence of the peak velocity of high-speed solar wind streams as measured in the ecliptic by ACE and the STEREO satellites on the area and co-latitude of their solar source coronal holes. Journal of Geophysical Research: Space Physics 123, 1738.

Hofmeister, S.J., Asvestari, E., Guo, J., Heidrich-Meisner, V., Heinemann, S.G., Magdalenic, J., Poedts, S., Samara, E., Temmer, M., Vennerstrom, S., et al.: 2022, How the area of solar coronal holes afects the properties of high-speed solar wind streams near Earth: An analytical model. Astronomy & Astrophysics 659, A190.

Jarolim, R., Veronig, A.M., Hofmeister, S., Heinemann, S.G., Temmer, M., Podladchikova, T., Dissauer, K.: 2021, Multi-channel coronal hole detection with convolutional neural networks. A&A 652, A13. DOI.

Krieger, A.S., Timothy, A.F., Roelof, E.C.: 1973, A coronal hole and its identification as the source of a high velocity solar wind stream. Solar Physics 29, 505. DOI.

LeCun, Y., Bottou, L., Bengio, Y., Hafner, P.: 1998, Gradient-based learning applied to document recognition. Proceedings of the IEEE 86, 2278. DOI.

Lemen, J.R., Title, A.M., Akin, D.J., Boerner, P.F., Chou, C., Drake, J.F., Duncan, D.W., Edwards, C.G., Friedlaender, F.M., Heyman, G.F., Hurlburt, N.E., Katz, N.L., Kushner, G.D., Levay, M., Lindgren, R.W., Mathur, D.P., McFeaters, E.L., Mitchell, S., Rehse, R.A., Schrijver, C.J., Springer, L.A., Stern, R.A., Tarbell, T.D., Wuelser, J.-P., Wolfson, C.J., Yanari, C., Bookbinder, J.A., Cheimets, P.N., Caldwell, D., Deluca, E.E., Gates, R., Golub, L., Park, S., Podgorski, W.A., Bush, R.I., Scherrer, P.H., Gummin, M.A., Smith, P., Auker, G., Jerram, P., Pool, P., Soufli, R., Windt, D.L., Beardsley, S., Clapp, M., Lang, J., Waltham, N.: 2012, The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO). Solar Physics 275, 17. DOI.

Nolte, J.T., Krieger, A.S., Timothy, A.F., Gold, R.E., Roelof, E.C., Vaiana, G., et al.: 1977, High coronal structure of high velocity solar wind stream sources. Solar Physics 46, 303. DOI.

Otter, N., Porter, M.A., Tillmann, U., Grindrod, P., Harrington, H.A.: 2017, A roadmap for the computation of persistent homology. EPJ Data Science 6, 17. DOI.

Papitashvili, N.E., King, J.H.: 2020, OMNI Hourly Data [Dataset], NASA Space Physics Data Facility. DOI.

Pesnell, W.D., Chamberlin, P.: 2012, The solar dynamics observatory (SDO). In: The solar dynamics observatory, Springer, 3.

Rasmussen, C.E., Williams, C.K.I.: 2006, Gaussian Processes for Machine Learning, MIT Press.

Reiss, M.A., Temmer, M., Veronig, A.M., Nikolic, L., Vennerstrom, S., Sch"ongassner, F., Hofmeister, S.J.: 2016, Verification of high-speed solar wind stream forecasts using operational solar wind models. Space Weather 14, 495. DOI.

Reiss, M.A., Muglach, K., Möstl, C., Arge, C.N., Bailey, R., Delouille, V., Garton, T.M., Hamada, A., Hofmeister, S., Illarionov, E., et al.: 2021, The observational uncertainty of coronal hole boundaries in automated detection schemes. The Astrophysical Journal 913, 28.

Reiss, M.A., Muglach, K., Mason, E., Davies, E.E., Chakraborty, S., Delouille, V., Downs, C., Garton, T.M., Grajeda, J.A., Hamada, A., et al.: 2024, A community data set for comparing automated coronal hole detection schemes. The astrophysical journal supplement series 271, 6.

Richardson, I.G.: 2018, Solar wind stream interaction regions throughout the heliosphere. Living Reviews in Solar Physics 15, 1. DOI.

Richardson, I.G., Cliver, E.W., Cane, H.V.: 2000, Sources of geomagnetic activity over the solar cycle. Journal of Geophysical Research: Space Physics 105, 18203.

Rotter, T., Veronig, A.M., Temmer, M., Vršnak, B.: 2012, Relation between coronal hole areas on the Sun and the solar wind parameters at 1 AU. Solar Physics 281, 793. DOI.

Samara, E., Heinemann, S.G., Temmer, M., Reiss, M.A., Hofmeister, S.J.: 2022, Influence of coronal hole morphology on the solar wind speed at Earth. Astronomy & Astrophysics 657, A176. DOI.

Schwenn, R.: 2006, Space weather: The solar perspective. Living Reviews in Solar Physics 3, 2. DOI.

Sheeley, N.R., Harvey, J.W., Feldman, W.C.: 1976, Coronal holes, solar wind streams, and recurrent geomagnetic disturbances: 1973–1976. Solar Physics 49, 271. DOI.

Telloni, D., Carbone, F., Antonucci, E., Bruno, R., Grimani, C., Villante, U., Giordano, S., Mancuso, S., Zangrilli, L.: 2020, Study of the influence of the solar wind energy on the geomagnetic activity for space weather science. The Astrophysical Journal 896, 149.

Temmer, M.: 2021, Space weather: The solar perspective. Living Reviews in Solar Physics 18, 4. DOI.

Tsurutani, B.T., Gonzalez, W.D., Gonzalez, A.L.C., Guarnieri, F.L., Gopalswamy, N., Grande, M., et al.: 2006, Corotating solar wind streams and recurrent geomagnetic activity: A review. Journal of Geophysical Research: Space Physics 111, A07S01.

Vršnak, B., Temmer, M., Veronig, A.M.: 2007, Coronal holes and solar wind high-speed streams: I. Forecasting the solar wind parameters. Solar Physics 240, 315. DOI.

Webb, D.F., Howard, T.A.: 2012, Coronal mass ejections: Observations. Living Reviews in Solar Physics 9, 3. DOI.

Wenzel, T., Santin, G., Haasdonk, B.: 2023, Analysis of target data-dependent greedy kernel algorithms: Convergence rates for f-, f · P- and f /P-greedy. Constructive Approximation 57, 45. DOI.

Zomorodian, A., Carlsson, G.: 2005, Computing persistent homology. Discrete & Computational Geometry 33, 249. DOI.