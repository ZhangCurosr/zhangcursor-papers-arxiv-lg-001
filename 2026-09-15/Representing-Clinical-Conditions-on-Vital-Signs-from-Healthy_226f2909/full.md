# Representing Clinical Conditions on Vital Signs from Healthy Individuals using Latent Modeling

Rafael Pina   
Institute for Digital Technologies   
Loughborough University London   
London, United Kingdom   
r.m.pina@lboro.ac.uk   
Varuna De Silva   
Institute for Digital Technologies   
Loughborough University London   
London, United Kingdom   
v.d.de-silva@lboro.ac.uk   
Mindula Illeperuma   
Institute for Digital Technologies   
Loughborough University London   
London, United Kingdom   
k.m.illeperuma@lboro.ac.uk

Abstract—Machine learning can be crucial to help scale complex signal processing applications in scenarios such as healthcare. However, these machine learning models need rich datasets to be trained and there are often cases where it is not possible to access representative datasets. In this paper, we propose a deep generative model based on conditional variational autoencoders with the objective of augmenting the vital signs of healthy individuals in a way that mimics the patterns of a certain clinical condition. More specifically, we use a publicly available ICU (Intensive Care Unit) dataset to train our model and then evaluate it using the vital data that we have collected from healthy individuals. Our results demonstrate that the proposed model can not only learn the underlying dynamics of the ICU data but, more importantly, can reshape our collected data from healthy individuals in a way that is aligned with the vital signs of a certain clinical condition. We propose a distance metric that shows how our model can generate samples that are more aligned with the intended clinical labels when compared to the tested baselines.

Index Terms—Vital signs data, Variational autoencoders, Clinical data generation, Deep generative modeling

## I. INTRODUCTION

Healthcare systems are one of the major beneficiaries of digital signal processing [1]–[3]. Signal processing techniques for vital sign monitoring and evaluation have been well studied area, and provide the foundation for many smart healthcare applications such as monitoring elderly, or heart conditions through the integration of machine learning techniques [4]– [8]. Monitoring vital signs such as heart and breathing rates or ECG, is a key technique used in Intensive Care Units (ICUs) to assist physicians, and surgeons in surgical theaters [9].

There are also many critical healthcare applications in which vital sign monitoring and real-time processing and machine learning based predictions can be very useful. Consider the triaging of wounded citizens in a disaster zone, or wounded soldiers in a battlefield. Real-time signal analysis from wearable sensors can be extremely useful in saving lives in these situations [10], [11]. However, to utilise machine learning techniques to achieve predictive applications such as triaging in such settings is extremely difficult due to the difficulty of collecting representative data.

![](images/a45d0a76be543e875ba1a438c5a43aff539ba01091a30b4391e21338a9706473.jpg)  
Fig. 1. Diagram depicting our approach combining an ICU dataset and our collected data using Zephyr BioHarness modules [15] to simulate clinical conditions from healthy vital data.

Generative Artificial Intelligence (GenAI) has currently received significant attention from the research communities including the signal processing community [12]–[14]. However, most of the GenAI applications have focused around text and image data, which are relatively easier to collect in massive amounts compared with vital sign signals pertaining to rare events. Sourcing large amounts of relevant data for healthcare applications is a massive challenge. On the other hand, data from healthcare applications is inherently multimodal, i.e., vital signs can come from various different modalities such as heart rate, breathing patterns, ECG, visual, radio frequency. In this paper, we work towards addressing this gap for a branch of signal processing applications of vital signs.

Funded by the Engineering and Physical Sciences Research Council of UK, within the ATRACT project we consider the application of triaging wounded soldiers in a battlefield. Soldiers often wear multiple vital sign sensors. For triaging using machine learning, under different acute conditions we need to have data pertaining to that setting. However, collecting such data is prohibitive due to security, ethical and practical considerations. To solve that issue, we collect data from healthy individuals while having their movements, and then seek to augment the vital sign signals with the patterns from ICU patients. We propose a novel conditional variational autoencoder (CVAE) architecture to generate vital sign signals. The contributions of this paper are as follows:

• We formulate a CVAE architecture that is capable of augmenting vital sign signals with acute medical conditions seen in ICU patients. The proposed model is conditioned on the required medical condition to simulate, and regularised for stable generation of signals.

• We propose an objective method to evaluate the quality of signals that are generated from the deep learning architecture, for benchmarking purposes.

The rest of this paper is organised as follows: we present the details of the data collection process, and the ICU dataset used for this work, in section II. The proposed variational architecture along with the evaluation method is presented in section III, followed by experimental results and discussion in section IV. We then conclude the paper in section V.

## II. PRELIMINARIES

## A. Data Collection

We collected data from healthy individuals during normal physical activity that does not require intensive effort such as crawling and walking. To capture the vital signs of the participants, we used the Zephyr Technology BioHarness 3.0 BioModule [15] (Fig. 1). This module can be attached to a strap placed around the chest of the participants, in direct contact with their skin allowing to record vital signals such as heart rate, breathing rate, acceleration, position and posture.

We connected all the participants wearing the BioModules to a central laptop with the Zephyr OmniSense software running. With this software and the modules connected, it is possible to live-track the vital signs and performance of the participants. Although the modules only provide a short range, by using an ECHO gateway repeater we can extend the range of the signal significantly. The gateway used allows to establish a radio network following 2.4GHz 802.115.4, allowing a range of 300 yards that is covered by our radio network to which the BioModules can be connected. This scheme can be found in Fig. 1 (similar to a PSM configuration from [16]).

## B. The MIMIC Database

To train the model proposed in this paper, we have selected a dataset from the MIMIC database [17], [18]. We opted to use the MIMIC-I database since that is readily available out of the box (in the future, we intend to extend to the more recent versions). This dataset is formed by records of over 90 ICU patients with periodic measurements of their vital signs obtained from a bedside monitor. These measurements were taken over several hours of observation and the records contain metrics such as heart and breathing rates, ECG signals, blood pressure, SpO2, etc, and each patient is labeled with a clinical class. The clinical conditions available in this dataset are (for conciseness, ahead we refer to them by simply using the letter within brackets): (A) Angina, (B) Bleed, (C) Brain Injury, (D) Congestive Pulmonary Failure/Pulmonary Edema, (E) Cardiac

![](images/aba07d87621696e645f405d832e7833363c9cea934bc24df3bd9710215bc101a.jpg)  
Fig. 2. CVAE-inspired architecture of the proposed model in this paper to reshape time series samples from healthy individuals in such a way that mimics vital signs of unhealthy individuals.

Arrest, (F) Cardiogenic Shock, (G) Post-OP Coronary Artery Bypass Graft (Post-OP CABG), (H) Post-OP Valve, (I) Renal Failure and (J) Respiratory Failure.

1) Data Preparation: In our experiments, we intend to explore how we can augment and reshape the data collected from the healthy participants in such a way that it can represent a certain clinical class from the MIMIC-I dataset. Considering that the Zephyr BioModules used in the data collection are non-invasive, we can only collect signals such as heart rate and breathing rate with them, and not blood-related measurements. In this sense, after analysing the MIMIC-I dataset, we selected the maximum number of subjects that contain the Heart Rate (HR) and Breathing Rate (BR) together with other metrics that are consistent. This led us to a total of 50 different subjects that have recordings for the HR, BR, systolic ABP (Arterial Blood Pressure), mean ABP, diastolic ABP, Pulse and SpO2.

Each subject was tracked over several hours, with most of them over 20 hours. To increase the sample size and make the time series suitable for our model, we sliced them into slices of 60 seconds (longer series would make the task more difficult to the model) and each was labeled with the respective clinical class of the subject (the slices of each subject were assigned with that subject’s class). While other techniques such as Multiple Instance Learning could have been used for the label assignments and alleviate the model, we opted to label the sliced blocks individually and shuffle them for learning. This led us to over 70k sub-samples, each with a label out of the 10 mentioned above. The data was also normalised.

## III. METHODOLOGY

## A. Proposed Architecture - CVVitAE

In this section, we propose Conditional Variational Vital Autoencoder (CVVitAE) (Fig. 2), a CVAE [19] model to reshape and augment the data collected from our healthy participants so that we can simulate as if they were being affected by a clinical condition present in the ICU database.

After cleaning the MIMIC-I data as described in section II, we feed it to our model that is composed of an encoder and decoder networks, as in standard VAEs [20]. However, the objective is to reshape data in a way that approximates a given clinical condition, meaning that the model needs to learn the relationships between the vital signs and the corresponding clinical labels. Hence, we follow the principles of CVAEs instead of normal VAEs. This means that the vital signs will be conditioned in the clinical labels before being fed to the model, i.e., $z = q _ { \phi } ( z | x , y )$ , considering an encoder network $q _ { \phi }$ that receives the inputs x conditioned on the true labels y. The encoder will transform the inputs into the latent space $z ,$ from which new data points can be sampled and given to a decoder $p _ { \theta }$ that will attempt to reconstruct the original input data. However, to control the values that are sampled from the latent space based on a given clinical class, the decoder can be represented following the probability distribution $p _ { \theta } ( x | z , y )$ where y is a certain clinical label. Our network can be trained by minimising the ELBO loss as described in

![](images/66ec49ec3098dac2588cf01ede2f8c723b39527575be844694de66fd42e56180.jpg)  
(a) Angina (A)

![](images/01a10b4079da66ee1306a78a9b91211ec2186017cb4074e110b8c12b8db2cdab.jpg)  
(b) Bleed (B)

![](images/ed717088e35b94cef7393d7211fa75863b54967560258354f5ed61c144c70a43.jpg)  
(c) MI Cardiogenic Shock (F)

![](images/051239cc226074a0495a0aab1f7eabfaf9e6879c7ea464b0ede658d5b9bca974.jpg)  
(d) Renal Failure (I)

Fig. 3. Distances for the unhealthy samples generated by feeding our collected data from healthy individuals to the trained vanilla CVAE model. Darker colours represent smaller values. The letter within brackets represents the ID of the clinical class, as described in section II-B.  
![](images/3d6143f0f1e7ef780cdcda081f0cafe996700545ff2fd007e81b2164e312db17.jpg)  
(a) Angina (A)

![](images/38d4a60978b5aa2a438caf46f09c154562ffd02a0f56feb6e23b5aa5c99af9ea.jpg)  
(b) Bleed (B)

![](images/ba25b7729f94d8ff74a46e696d0d219d76e3d4021dd3a4400b49330cd74a0f2e.jpg)  
(c) MI Cardiogenic Shock (F)

![](images/005c7f8884a69892e4076f2b75087af62deb22795318a53afd0f6e21c1cbee41.jpg)  
(d) Renal Failure (I)  
Fig. 4. Distances for the unhealthy samples generated by feeding our collected data from healthy individuals to the trained CVVitAE model. Darker colours represent smaller values. The letter within brackets represents the ID of the clinical class, as described in section II-B.

$$
\mathcal { L } _ { \mathrm { A } } = \mathbb { E } _ { q _ { \phi ( z | x , y ) } } [ \log p _ { \theta } ( x | z , y ) ] - D _ { \mathrm { K L } } [ q _ { \phi } ( z | x , y ) | | p _ { \theta } ( z | x ) ] .\tag{1}
$$

We approximate the reconstruction term with the MSE of the reconstructions from the network to the original inputs, which can be written as $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } ( y _ { t } ^ { \prime } - y _ { t } ) ^ { 2 } } \end{array}$ , and the KL-Divergence to a normal distribution $\mathcal { N } ( 0 , I )$ for the regularisation term.

This describes the architecture of the CVAE applied to our problem with vital signs time series. However, simply using this model may not be enough to ensure that the latent space z can accurately consider the clinical labels as a condition to the generated samples. To make sure that the latent space is aligned with the clinical labels, we include an additional component in our architecture that works as a regularizer unit for the latent space. We use an LSTM-based layer that receives the time series sampled from the latent space and will match them with the respective clinical label. In essence, this layer works as a latent space regularizer that predicts the labels from latent samples, resulting in a second objective of minimising the cross-entropy between the predicted labels by this classifier and the true labels. This loss can be represented as

$$
\mathcal { L } _ { \mathrm { { B } } } = - \sum _ { i = 1 } ^ { C } y _ { i } \ l o g \ p ( y _ { i } ) ,\tag{2}
$$

where $C$ is the number of clinical labels, and $p$ is the distribution for the predicted labels. This second loss function can then be integrated in our first loss described in Eq. (1), resulting in the overall loss

$$
\mathcal { L } = \alpha \times \mathcal { L } _ { \mathrm { A } } + \mathcal { L } _ { \mathrm { B } } ,\tag{3}
$$

where $\alpha$ is a weighting factor that controls the weight given to the losses when training the model. All the components of the proposed architecture are depicted in Fig. 2.

## B. Evaluating the Proposed Model

After training our model, the goal is to use the collected data and feed it to the model described in the previous section to be augmented. Normally, the trained decoder would be enough for the generation stage, but in this case, we are using real collected data and not just sampled noise. Hence, the data needs to go through the encoder in order to be consistent with the model. Since some of the features of the collected data are missing when compared to the MIMIC-I dataset, we fill the missing values with noise that is normalised in accordance with the values of the collected data and the dataset. Then this can be fed to the decoder, together with the clinical label that we intend to mimic by reshaping the original data.

Importantly, the vital signs of clinical labels of different classes can be close to each other since the variations in the vital signs over time can be very small. Hence, it might be non-trivial to evaluate the quality of the samples generated. For that purpose, we define a metric of proximity from the generated samples to the real values of the input features of each clinical class. The idea is to compare the proximity of the values generated for each feature to the same feature of the samples with the same clinical label in the original MIMIC-I dataset. To achieve this, we calculate the average values of each feature in the original set and build a map containing the distance from this average to the average of the values in the features of the augmented samples. This procedure is done for each clinical class that we augment data for, resulting in a value $m _ { j , c }$ that illustrates the proximity between a generated sample from our model and the real samples of the MIMIC-I dataset for a given class c, and for each vital sign feature j. Formally, this can be written

$$
m _ { j , c } = \frac { \left( \{ x _ { j } ^ { \prime } \} _ { t = 1 } ^ { T } - \overline { { x } } _ { j } \right) } { T } ,\tag{4}
$$

where $j$ corresponds to the index of each feature in the vital signs time series, T to the number of timesteps in the series, and $c \in \{ 1 , \ldots , C \}$ to the index of a certain clinical label.

## IV. RESULTS

The main objective of this work is to create a generative model that aligns samples of data collected from healthy individuals with a target clinical condition. We evaluate our model (after training convergence) using samples from healthy individuals and attempt to reshape them in a way that mimics the vital signs of clinical conditions. This can be seen in the plots in Fig. 3 and 4, showing the distances of each feature for four generated classes to the features of each class in the real data. In Fig. 4, we observe that, for each generated class with CVVitAE, the values of the features are always closer to the feature of that intended class in the initial dataset. This observation is consistent across all clinical classes that we have generated, meaning that our model is capable of augmenting samples of healthy individuals in a way that is aligned with an intended clinical condition. Instead, when using a vanilla CVAE the features are much more distant, as seen in Fig. 3.

One of the key components of our architecture that potentiates the generation of aligned samples is the use of the latent space classifier that aligns the learned latent space with the intended clinical labels, as it can be seen in Table I, where the architecture that doesn’t use this classifier (Vanilla CVAE, similar to as used in [21], [22]) generates samples that are more distant from the real data. The confusion matrix of the classifier layer from our architecture that matches the number of predicted with true labels can also be seen in Table II. We can see that the classification layer can accurately align the latent space of our CVVitAE with the intended labels. The use of LSTM layers in our architecture is also an important component for the success of our model, as it can be seen in Table I, where the samples generated from the same model that uses the classifier but with linear layers instead of LSTM layers in the encoder and decoder (Linear CVVitAE) generates samples that are more distant from the real ICU data.

TABLE I  
DISTANCES TO THE CLINICAL LABELS FROM OUR COLLECTED DATA SAMPLES AUGMENTED WITH THE TRAINED MODELS.
<table><tr><td></td><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td><td>G</td><td>H</td><td>I</td><td>7</td></tr><tr><td>Vanilla CVAE</td><td>99.10</td><td>15.39</td><td>48.81</td><td>98.12</td><td>86.75</td><td>102.37</td><td>42.38</td><td>88.24</td><td>31.49</td><td>49.79</td></tr><tr><td>Linear CVAE</td><td>9.49</td><td>10.47</td><td>8.05</td><td>9.99</td><td>16.86</td><td>9.92</td><td>28.65</td><td>14.90</td><td>8.90</td><td>13.72</td></tr><tr><td>CVVitAE</td><td>6.89</td><td>7.34</td><td>13.61</td><td>9.98</td><td>7.21</td><td>11.99</td><td>9.46</td><td>18.99</td><td>11.31</td><td>12.09</td></tr></table>

TABLE II

CONFUSION MATRIX OUTPUTTED BY THE CLASSIFIER LAYER IN OUR MODEL AT THE END OF TRAINING WITH THE TEST SET (15811 SAMPLES).
<table><tr><td rowspan=2 colspan=1>TrueLabel</td><td rowspan=1 colspan=10>Predicted Label</td></tr><tr><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>J</td></tr><tr><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>1551</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2233</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>788</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3241</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>101</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1840</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>503</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>962</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>I</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>505</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>3804</td></tr></table>

Overall, our results demonstrate that the proposed model can successfully learn the dynamics of the used dataset and further process real data that we collected from healthy individuals and reshape it in a way that mimics as if they were suffering from some clinical condition. Our model also proves to be stronger when compared to the baselines.

## V. CONCLUSION AND FURTHER WORK

It can be challenging to find representative data in applications such as healthcare in extreme environments as in battlefields, natural disasters or sports, which are needed to train machine learning models. In this paper, we proposed a CVAE architecture that can process ICU data and generate new time series of sensor signals that relate to a targeted clinical condition. In this sense, we can use the vital signs data collected from healthy individuals and augment these with patterns associated with targeted clinical conditions. Leveraging generative AI to help process and augment this type of signals can be crucial to ensure feature consistency across despaired data sources. By learning latent signal representations it is possible to mitigate these challenges.

Overall, the results showed that the proposed generative architecture is capable of aligning samples with the intended clinical condition with 98% accuracy. Furthermore, the proposed method proved to be successful in aligning vital signs from healthy individuals with the targeted clinical conditions, as evaluated by our metric of distance to the real samples. In the future, we intend to include more modalities in the proposed method such as visual signals and localisation, helping to scale it to more complex scenarios.

[1] F. Khan, A. Ghaffar, N. Khan, and S. H. Cho, “An overview of signal processing techniques for remote health monitoring using impulse radio uwb transceiver,” Sensors, vol. 20, p. 2479, 04 2020.

[2] N. S. Amer and S. B. Belhaouari, “Eeg signal processing for medical diagnosis, healthcare, and monitoring: A comprehensive review,” IEEE Access, vol. 11, pp. 143 116–143 142, 2023.

[3] A. Perez, “Reshaping engineered clinical decision support systems: Biomedical signal processing and artificial intelligence in health care,” IEEE Potentials, vol. 42, no. 4, pp. 17–23, 2023.

[4] S. Liu, Q. Qi, H. Cheng, J. Zhang, W. Xian, T. Ma, Y. Wang, Y. Liu, D. Li, and J. Chai, “An intelligent signal processing method for motional vital signs detection system based on deep learning,” IEEE Access, vol. 10, pp. 106 463–106 481, 2022.

[5] M. G. Amin, Y. D. Zhang, F. Ahmad, and K. D. Ho, “Radar signal processing for elderly fall detection: The future for in-home monitoring,” IEEE Signal Processing Magazine, vol. 33, no. 2, pp. 71–80, 2016.

[6] H. Mou, C. Li, H. Zhou, D. Zhang, W. Wang, J. Yu, and J. Tian, “Using data augmentation to improve the accuracy of blood pressure measurement based on photoplethysmography,” Electronics, vol. 13, no. 8, 2024. [Online]. Available: https://www.mdpi.com/2079- 9292/13/8/1599

[7] C. El-Hajj and P. Kyriacou, “Cuffless blood pressure estimation from ppg signals and its derivatives using deep learning models,” Biomedical Signal Processing and Control, vol. 70, p. 102984, 2021. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1746809421005814

[8] I. Komorska and A. Puchalski, “Condition monitoring using a latent space of variational autoencoder trained only on a healthy machine,” Sensors, vol. 24, no. 21, 2024. [Online]. Available: https://www.mdpi.com/1424-8220/24/21/6825

[9] D. Evans, B. Hodgkinson, and J. Berry, “Vital signs in hospital patients: a systematic review,” International Journal of Nursing Studies, vol. 38, no. 6, pp. 643–650, 2001. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S002074890000119X

[10] B. M. Carius, J. F. Naylor, M. D. April, A. D. Fisher, I. L. Hudson, P. J. Stednick, J. K. Maddry, E. K. Weitzel, V. A. Convertino, and S. G. Schauer, “Battlefield vital sign monitoring in role 1 military treatment facilities: A thematic analysis of after-action reviews from the prehospital trauma registry,” Military medicine, vol. 187, no. 1-2, pp. e28–e33, 2022.

[11] W. Zhang, F. Sun, Z. Lu, S. Fan, Z. Huang, Y. Hao, Z. Pan, L. Chen, Y. Lou, and J. Liu, “A wearable medical sensors system for in-situ monitoring vital signs of patients with hemorrhagic shock in big disaster scenes,” Sensors and Actuators B: Chemical, vol. 395, p. 134448, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0925400523011632

[12] L. Wang, C. Zhang, Q. Zhao, H. Zou, S. Lasaulce, G. Valenzise, Z. He, and M. Debbah, “Generative ai for rf sensing in iot systems,” IEEE Internet of Things Magazine, vol. 8, no. 2, pp. 112–120, 2025.

[13] N. Van Huynh, J. Wang, H. Du, D. T. Hoang, D. Niyato, D. N. Nguyen, D. I. Kim, and K. B. Letaief, “Generative ai for physical layer communications: A survey,” IEEE Transactions on Cognitive Communications and Networking, vol. 10, no. 3, pp. 706–728, 2024.

[14] J. Wang, H. Du, D. Niyato, Z. Xiong, J. Kang, B. Ai, Z. Han, and D. In Kim, “Generative artificial intelligence assisted wireless sensing: Human flow detection in practical communication environments,” IEEE Journal on Selected Areas in Communications, vol. 42, no. 10, pp. 2737– 2753, 2024.

[15] Z. Technology, “Bioharness 3.0 user manual,” 2012. [Online]. Available: https://www.zephyranywhere.com/media/download/bioharness3- user-manual.pdf

[16] ——, “Psm training echo user guide,” 2014. [Online]. Available: https://www.zephyranywhere.com/media/download/psm-traininguser-guide.pdf

[17] G. Moody and R. Mark, “A database to support development and evaluation of intelligent intensive care monitoring,” in Computers in Cardiology 1996, 1996, pp. 657–660.

[18] A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, “Physiobank, physiotoolkit, and physionet,” Circulation, vol. 101, 06 2000.

[19] K. Sohn, H. Lee, and X. Yan, “Learning structured output representation using deep conditional generative models,” in Advances in Neural Information Processing Systems, C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, Eds., vol. 28. Curran Associates, Inc., 2015.

[20] D. P. Kingma, M. Welling et al., “Auto-encoding variational bayes,” 2013.

[21] Y. Xia, W. Wang, and K. Wang, “Ecg signal generation based on conditional generative models,” Biomedical Signal Processing and Control, vol. 82, p. 104587, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1746809423000204

[22] C. I. Wong, M. H. Jaward, V. M. Baskaran, C. H. Chee, and M.-L. Sim, “Joint channel estimation and signal detection using latent space representations in vae,” in 2022 International Symposium on Intelligent Signal Processing and Communication Systems (ISPACS), 2022, pp. 1–4.