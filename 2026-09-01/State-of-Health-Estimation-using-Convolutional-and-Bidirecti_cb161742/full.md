# State of Health Estimation using Convolutional and Bidirectional LSTM Neural Networks tuned by Bayesian Optimization

Panagiotis Eleftheriadis Foivos Georgios Kyrgios Sonia Leva

Department of Energy, Politecnico di Milano, via Lambruschini 4a, 20156 Milano, Italy

panagiotis.eleftheriadis@polimi.it

## Abstract

In this research, a novel framework is proposed for the SOH estimation, which employs a hybrid deep learning architecture of a concatenation of a Convolution Neural Network (CNN) and a Bidirectional Long Short-Term Memory (BiLSTM) Neural Network (NN) with the integration of Bayesian Optimization-based hyperparameter tuning for the network. Three different deep learning architectures are being evaluated: standalone recurrent models, CNN-RNN architectures and CNN-RNN combinations enhanced with intermediate Fully Connected (FC) layers. Among the three, the model with the intermediate FC layers demonstrated the highest predictive accuracy. A comprehensive feature engineering approach combines capacity (Q), voltage (V), Incremental Capacity Analysis (ICA), and Differential Voltage Analysis (DVA), with systematic evaluation of multiple combinations to identify the optimal input representation. To validate the proposed method, three publicly available datasets were utilized, ensuring reproducibility of the results, two from external sources and one developed by the author of this study using a unique experimental setup. The comparison study was performed using the Mean Absolute Error (MAE), the Root Mean Squared Error (RMSE) and the FLoating-point OPerations (FLOPs) as evaluation metrics.

Keywords: Bayesian Optimization, Lithium-Ion Battery, Machine Learning, State of Health, CNN, RNN.

## 1 Introduction

Given the recent policy directions [1], the widespread electrification across all sectors, particularly in transportation with the increasing penetration of Electric Vehicles (EVs) into the market, alongside the increasing renewable energy integration, has intesified demand for Energy Storage Systems (ESS). Driven by current trends and advancements in battery materials, Battery Energy Storage Systems (BESS) are gaining increasing prominence across various applications [2]. Effective battery management is fundumental to ensuring safe, dependable, and optimal performance of BESS. A key element of any Battery Management System (BMS) is the accurate estimation of the State of Health (SOH), which delivers essential insights into the battery capacity [3]. Reliable SOH estimation enables effective scheduling, planning, and overall system optimization. Multiple approaches have been proposed for the SOH estimation and can be categorized as either model-based or data-driven methods. Given the complex internal dynamics of batteries and the growing capabilities of artificial intelligence, deep learning methods, within the data-driven category, have garnered significant attention and exhibited remarkable performance in recent studies [3]. Among data-driven SOH estimation techniques, the methods that have garnered greater interest include RNN as a model type, as wel as hybrid approaches. Hybrid approaches encompass the fusion of different deep learning architectures or the incorporation of optimization algorithms alongside data-driven methods, as detailed in [3]. RNNs have the ability to leverage information from prior predictions as well as measurements to enhance estimation ac curacy. Due to the remarkable capabilities of current neural networks, deep-learning methods have obtained the ability to efficiently process time-series data, showcasing remarkable performance in estimation and prediction applications. Some of the more established deep-learning SOH estimation methods are RNNs, LSTMs and CNNs, among others. It is worth noting that, LSTMs’ ability to capture temporal dependencies in extensive time-series datasets makes it one of the most promising alternatives. In particular, in [4], the authors proposed an SOH estimation method utilizing an LSTM network and employed Bayesian Optimiza tion (BO) for tuning the hyperparameters of the model. Through this approach, they effectively captured the patterns of the Li-ion battery ageing procedure and significantly improved the accuracy of predictions. Additionally, Bharath et al. [5] introduced a cascaded LSTM-RNN model which demonstrated improved efficiency compared to existing state-of-the-art benchmarks. In [6], a BiLSTM model was proposed and coupled with Particle Swarm Optimization (PSO) for hyperparameter tuning. By implementing this ap proach, MAE and RMSE were limited to under 2.6%. Lastly, researchers in [7] developed a CNN-LSTM hybrid framework, without altering the model architecture and parameters, aiming to create a simplified SOH estimation approach. They managed to maintain the model’s robustness while ensuring its adaptabil ity to different battery conditions. The aforementioned studies clearly demonstrate the effectiveness and popularity of deep-learning approaches for SOH estimation. However, despite these advancements, chal lenges related to computational complexity, real-world implementation, and data scarcity still remain. To overcome these gaps, this paper proposes an automated hyperparameter optimization approach specifically designed for battery SOH estimation. This research presents an integrated framework based on the combi nation of a CNN and a BiLSTM incorporating hyperparameter tuning through BO coupled with Gaussian Process (BOGP). The overarching objective is to create a comprehensive model that can reliably forecast the SOH under partial discharge data. A novel set of hyperparameters was introduced while maintaining a compact network architecture, expanding the hyperparameter space to include BiLSTM layer count, units per BiLSTM layer, Fully Connected (FC) layers count, neurons per FC layer, and their activation functions. CNN filter counts across both convolutional layers were also searchable. The Adam optimizer learning rate was included as an additional tunable parameter. This comprehensive hyperparameter search space allows discovery of near-optimal configurations without requiring manual intervention. Nevertheless, network per formance depends heavily on the hyperparameter range and selection, representing a potential limitation of this research. To assess the effectiveness of the approach, the proposed framework was applied to three publicly available datasets were utilized, ensuring reproducibility of the results, two from external sources and one developed by the author of this study using a unique experimental setup, while conducting a performance comparison with existing methods. For each dataset, an exhaustive analysis of input features was conducted to identify the combination that produced the most effective predictive outcomes for each solution. The primary contributions of this work are:

• Proposing a structure combining a CNN for the extraction of spatial features, while using in cascade connection a RNN capturing the temporal variations of the battery dynamics.

• Development of an innovative hyperparameter space allowing for efficient exploration utilizing the Bayesian algorithm to attain optimal performance.

• Three battery datasets analyzed and used to validate the proposed model.

• An exhaustive input feature analysis between five combinations identifying the most promising combination.

This work is structured as follows: Section II describes the datasets employed and feature engineering approach, Section III details the proposed SOH estimation methodology and model training, Section IV reports the experimental results, Section V provides discussion of findings, and Section VI concludes the research.

Table 1: Datasets Main Features
<table><tr><td>Features</td><td>Oxford</td><td>NASA</td><td>PoliMi</td></tr><tr><td>Battery Type</td><td>8740mAh 18650 - NMC lpb</td><td>4 2Ah Li-Ion</td><td>6 LG 2.5Ah 18650 NMC</td></tr><tr><td>Testing Equipment</td><td>Bio-Logic MPG-205</td><td></td><td>Neware battery tester</td></tr><tr><td>Temperature Range</td><td>40℃</td><td>23℃</td><td>35℃</td></tr><tr><td>Stressing Cycles</td><td>Artemis</td><td>Constant</td><td>UDDS, US06, LA92, Mixed1-Mixed6</td></tr><tr><td>Sampling rate</td><td>1 second</td><td>5 seconds</td><td>1 second</td></tr></table>

## 2 Datasets

As previously mentioned, three datasets were utilized in this study. The first is the PoliMi dataset, a private dataset obtained through a collaboration between TUB and Politecnico di Milano. The other two datasets, Oxford and NASA, are publicly available. The key characteristics of these datasets are summarized in Table 1. Detailed information about the two public datasets can be found in [8], while details about the private dataset are provided in [9]. For all datasets, the features are information related to the Voltage, the Capacity, the ICA curve, and the DVA curve, given by Equation 1 and Equation 2, respectively. The Incremental Capacity (IC) is defined as:

$$
\mathrm { I C } = { \frac { \mathrm { d } Q } { \mathrm { d } V } }\tag{1}
$$

where the dQ is the change in capacity and dV is the corresponding change in voltage. Similarly, the Differential Voltage (DV) is calculated as:

$$
\mathrm { D V } = { \frac { \mathrm { d } V } { \mathrm { d } Q } }\tag{2}
$$

These metrics are plotted against voltage in order to produce the ICA and DVA curves, subsequently processed using Savitzky-Golay filtering, eliminating noise and preparing the data for feature extraction.

## 2.1 Oxford dataset

For this study, the Oxford University dataset has been utilized. Particularly, eight NMC lithium-ion batteries, connected in parallel, underwent cycling ageing. The testing process was conducted at a constant temperature of $4 0 ~ ^ { \circ } \mathbf { C } .$ The upper and lower voltage limits of the batteries were 4.2 V and 2.7 V respectively, while the nominal value was 3.7 V. Additionally, the nominal capacity was 740 mAh. The cycling protocol initiated with a constant-current, constant-voltage charging profile. Subsequently, the batteries were discharged based on the urban Artemis profile. Further information on characterization measurements, charging and discharging rates, and other specifications are provided in the dataset’s supplementary file. Figure 1 depicts all the voltage profiles of the dataset’s cells, providing a clear view of the ageing phenomenon through the reduction of discharging time as cycling progresses. The ageing mechanism can also be observed in Figure 2 where the maximum charge of every cell gradually drops as the number of cycles increases. For the purposes of this study, only the discharge phase data have been utilized, as these are more practically relevant.

![](images/d23f4cd6babaa9040a89f15554c86bf480a35e14da0c1b7b115a57a82d4dded0.jpg)  
Figure 1: Oxford dataset cell voltage profiles during cycling. Red lines demarcate the voltage window employed for subsequent analysis.

Nevertheless, the implementation of this framework could also be based on the charging stage, yielding similar results. In particular, features are extracted from discharge cycles conducted at 1C for the Oxford University dataset, with nominal capacity values referenced against pseudo-OCV measurements. To minimize the processing time required for each battery, instead of considering the full voltage range of 2.7-4.2 V, this method narrows the window of interest to 3.4-4.1 V. This range incorporates the highest peak of the ICA plot and captures most of the useful information [10]. This approach provides an optimal framework, minimizing computational costs without compromising data quality.

## 2.2 NASA dataset

Additionally, the NASA dataset [11] was utilized. Four lithium-ion battery cells were used for this dataset. The upper voltage limit was 4.2 V while the lower voltage limit varied from 2.2 V to 2.7 V among the cells. Figure 3 depicts the voltage profiles of the NASA cells. Unlike the Oxford dataset, the NASA voltage profiles exhibit greater variability across cycles [12]. Figure 4 illustrates discharge capacity per cycle, demonstrating consistent degradation across all cells.

## 2.3 PoliMi

There are limited publicly accessible datasets available for researchers to validate their algorithms for SOC and SOH estimation. While there are sufficient datasets for SOC estimation, the availability is notably less for SOH, and most existing datasets do not include aging under dynamic driving cycles to mimic real-life conditions. For this reason, a series of experiments were designed and conducted in the battery laboratory of T.U. Berlin, with the collaboration of the Politecnico di Milano. The data associatted with the dataset can be found in Mendeley Data [13]. This dataset comprises six NMC lithium-ion batteries. Cycling was performed at a controlled temperature of $3 5 ^ { \circ } \mathrm { C } .$ , with discharge profiles based on three driving simulation cycles. Figure 6 clearly shows the degradation mechanism as a function of cycle number.

![](images/946e7d7e3b68198702e54add3609d8b2e741006ad45a62edd4233de51bc70b38.jpg)  
Figure 2: Oxford dataset discharge capacity per cycle. Cells 4 and 6 were disconnected during testing.

![](images/ac189c42ef8ad7422501d8350a1e87d3cafc6dc5cfff766868633c779ce72f13.jpg)

![](images/8edfc05c0aef6fa2eab9c566395f57c2474318cb1fe343ce1646f7d06af6c518.jpg)

![](images/f0d52607990cb315d29bb2a47af2bf49ea015220c4bbe415281cfac9ec513120.jpg)

![](images/61c12e2842f927b7c7b1339952eb84b67784f975c224aab764fc5bd4c209e528.jpg)  
Time (s)  
Figure 3: NASA dataset cell voltage profiles during cycling. Red lines demarcate the voltage window employed for subsequent analysis.

## 2.4 Input Feature Selection

Unlike method described in [12], which extracted only peak ICA values, corresponding voltage points, and curve integration, this research employs four parameters as input features for the network: ICA, DVA, V, and Q from the discharging cycles. The complete dataset for these parameters will be leveraged to fully harness the capabilities of the RNN. The ICA and DVA values are presented for the PoliMi dataset as a refernce in Figure 5. For the other two datasets a similar behavior is observed. Therefore, the input features will be the time-series representations of these parameters, creating time-series data for the entire ICA, DVA, V, and Q curves during partial discharging represented as $[ I C A , D V A , V , Q ] _ { W S }$ , where WS denotes the window length. Figure 7 depicts an example of the time-series input created for SOH estimation at each cycle. For the Oxford and NASA datasets, the partial discharging voltage range will remain the same, between 3.4 to 4.1 V. In contrast, for the TUB dataset, the voltage range is configured between 3.1 and 3.7 V to capture all the information of the ICA peaks. Furthermore, to optimize the network’s robustness, the sampling interval for the Oxford and PoliMi datasets was manually adjusted to a 10-second resolution, with no significant impact on the predictive results observed. Consequently, the WS of the input time series for the NASA, PoliMi, and Oxford datasets was set at 151, 117, and 238 time steps, respectively. For model training, cells, 1,2 and 4 were used for NASA, cells 1-8 for Oxford, and cell 1 for PoliMi. Model evaluation employed cells 3,7 and 8 from the PoliMi, NASA, and Oxford University datasets, respectively. In the data preprocessing phase, the input dataset undergoes min-max scaling [14]. Through this transformation the data values range from 0 to 1, facilitating the neural network’s ability to handle diverse input features effectively [15]. The NN implementation is incorporated through the Keras Library [16]. Additionally, the Bayesian Optimization framework, responsible for fine-tuning the model’s hyperparameters, is skillfully integrated through the use of the Keras Tuner Library [17]. The Adam optimizer [18] was used along with the searchable hyper parameter of the learning rate, while the Huber loss was chosen [19]. The network underwent training for 1000 epochs, incorporating an early stopping strategy if there was no improvement in validation loss after 50 epochs. Figure 8 illustrates the overall structure of the proposed solution, which is segmented into three main areas: dataset preprocessing, hyperparameter optimization, and model validation. A batch size of 128 was employed, and a total of 50 trials were conducted to determine a set of hyperparameters that were close to optimal.

![](images/87c39811d445a7ba388609b0491bfb6c59303cd055c801eccecbeb9e4fbf9d80.jpg)  
Figure 4: NASA dataset discharge capacity per cycle. Spikes indicate cell rest periods between successive cycles.

![](images/897b43f317f6cf32414a8f4c917669bd2bb13629bff09add9e9b2247c1f69029.jpg)

![](images/eb2b4c7abdd838dd6c31823a7fafd282eee566455258ea2a94a38c8cd439fc0b.jpg)  
Figure 5: The ICA and DVA vs Voltage for all cycles for cell 1 of the PoliMi dataset. As the cell degrades, there is a noticeable trend of decreasing ICA/DVA values and changes in voltage.

![](images/90538d78321ac5bbb236c25d5ba88f583bc179a1d76c71def5557e73dce42a23.jpg)  
Figure 6: The evolution of the SOH of the NMC battery cells.

![](images/19fa8fd18ad0851b5b7061732ced8c4de0a07e9acc10f9a6a2bcd3bb23059391.jpg)  
Figure 7: Time-series input shape and output estimation of each partial discharge cycle. At each time step t, an estimation is made, with the total number of steps being equal to the End of Life (EoL) of the cell, which occurs when the SOH reaches 70%.

## 3 Proposed Method

## 3.1 Convolutional Neural Network

The CNN is one of the most prominent networks in the field of deep learning [20]. The CNN is a type of Feed Forward Neural Network (FFNN) designed to extract features from data using convolution structures.

![](images/df571053fb6ab091f1905c5b139c334bbdab3e96c9b0b7fd653de7152e03fe5a.jpg)  
Figure 8: General framework of the proposed solution.

Key features of the CNN include the establishment of local connections where each neuron is linked to only a few preceding neurons instead of all. Additionally, weight sharing in CNNs reduces the number of parameters needed. Finally, dimension reduction is achieved through pooling techniques, further decreasing the required number of parameters. In constructing a CNN, four essential components are needed. The first is convolution, crucial for extracting features, resulting in feature maps. To address information loss at the borders during convolution, padding is used, which involves enlarging the input with zeros to indirectly modify its size. The stride, determining the convolution’s density, is also important; a larger stride results in lower density. Finally, to prevent overfitting due to the abundance of features in the feature maps, pooling (or downsampling) is used to reduce redundancy. Pooling can be of two types: max pooling and average pooling. The structure of a typical CNN is presented in Figure 9. The most typical types of convolutional

![](images/57b2595a8a32d032702e3e2d63b293c1c45c8562eb298c68d11317c3fb96d2ad.jpg)  
Figure 9: Convolutional Neural Network architecture [20].

layers are the ‘Conv1D’, ‘Conv2D’, and ‘Conv3D’, each one dealing with data of different shapes. The ‘Conv1D’ typically used for time-series data and sequential data applies a 1D convolutional operation with the data having a 2D or 3D dimension, with shape [‘number of samples’, ‘sequence length’, ‘number of input features’]. Furthermore, the ‘Conv2D’ typically used for image data of data with spatial relationships applies a 2D convolutional operation with the data having usually 4D dimensions with shape [‘number of samples’, ‘height’, ‘width’, ‘channels’]. Finally, the ‘Conv3D’ is used for 3D data as volumetric data or videos and applies a 3D convolutional operation with the data having a 5D dimension with shape [‘number of samples’, ‘depth’, ‘height’, ‘width’, ‘channels’].

## 3.2 Bidirectional Long Short-Term Memory Neural Network

This research also employs LSTM neural networks. This type of RNN overcomes the vanishing gradients challenge by using gating mechanisms to control information flow. These gates include the input, output and forget gates. Analytical information about the aforementioned gates and their operation can be found in [21]. As a result, they capture long-term dependencies effectively across data. The LSTM cell architecture is depicted in Figure 10. In summary, LSTMs’ ability to retain relevant information while filtering noise makes them well-suited for sequential data processing. Building on this, BiLSTMs process information bidirectionally, capturing information in both temporal directions. This results in more comprehensive feature representations compared to standard LSTM networks [21]. Figure 11 depicts the BiLSTM cell

![](images/55cff401b4f090b29825850d8b03dbee0c2aa2c124204a344a875437a1b4db09.jpg)  
Figure 10: LSTM cell architecture [21]

![](images/ab6a4ddc67004e3e67b932517a0b2723525d14215f2927476a1b425995dda3c7.jpg)  
Figure 11: BiLSTM cell architecture [22]

architecture. Information flows bidirectionally through forward and backward layers, enabling the network to capture temporal information from both past and future contexts [21]. LSTM networks with stacked layers excel at modeling battery behavior by capturing temporal dynamics and long-term dependencies in time-series measurements. In [23], the LSTM showcases remarkable performance in predicting SOH for a single time step, demonstrating its ability to process sequential data including voltage, current, and temperature measurements. Similarly, researchers in [24] implement a backpropagation LSTM for battery SOH prediction, validating the applicability and efficiency of such networks.

## 3.3 Optimization Algorithm

The main focus of this work lies on the use of Bayesian optimization in order to achieve optimal model hyperparameters. Compared to other exhaustive approaches such as grid and random search, BO demonstrates superiority in both efficiency and accuracy in high-dimensional space navigation [25]. The representation of the objective function through stochastic models makes it possible to update its beliefs, thereby allowing it to focus on promising regions. Bayesian optimization (BO) starts by establishing a prior distribution over the hyperparameters. In this way, it leverages all available prior information and evaluations regarding their behavior. At each iteration, the algorithm selects promising configurations taking into account previous performance results. Through this refinement process, BO navigates efficiently through the hyperparameter space, focusing on promising candidates and discarding unlikely solutions [26]. This results to the reduction of the computational costs of hyperparameter tuning.

![](images/24863187819b57c61b02f2ed6aa3abfdd297be0861228eccb71bd7fa9327f371.jpg)  
Figure 12: General Stacked CNN-BiLSTM architecture

## 3.4 Training Procedure

The configuration of the proposed network includes a number of key parameters, meticulously fine-tuned to improve its performance. A unique hyperspace has been developed for each hyperparameter to aid in the identification of the most effective, or near-optimal, hyperparameter settings. The following parameters were made searchable for the CNN-RNN network’s hyperparameter search space:

• Learning rate of the optimizer between the values of 0.001, 0.0001 and 0.00001.

• Two CNN layers with 1 to 15 filters, individually searchable, while the rest of the hyperparameters are fixed with a kernel size of 3, activation function being the Relu.

• One to three initial RNN layers with 1 to 15 units, (1-unit increments), returning the full sequence.

• One RNN layer within the range of 1 to 15 units, in increments of 1 unit, without the whole sequence being returned.

• A single RNN layer with 1-15 units (1-unit increments), without sequence return.

• One to three additional FC layers.

• FC layer units: 1-15 neurons per layer (1-neuron increments).

The goal of defining the search space for hyperparameters is to facilitate a comprehensive probabilistic exploration through Bayesian optimization, ensuring that no possible configuration is missed. This detailed strategy allows for the precise adjustment of the proposed network, enhancing its predictive accuracy and robustness, thereby aligning with the study’s objectives for more precise and reliable predictions. To enhance accuracy and robustness, multiple stacked layers were added to every stage of the final algorithm, namely the CNN, RNN and FC layers. Figure 12 depicts the general framework of the proposed solution. Specifically, x represents the sequential data input previously discussed, and y<sub>t</sub> denotes the SOH estimation output. The figure illustrates the n stacked layers at each intermediate stage of the model, which substantially improve performance.

## 4 Results

The prediction performance of the proposed model was assessed using the RMSE and MAE metrics. The definitions of these two error metrics are given by Equation 3 and Equation 4, respectively. As the SOH is a

percentage, the corresponding error metric will be reported in percentage (%).

$$
R M S E = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { ( S O H _ { i } - S \hat { O } H _ { i } ) ^ { 2 } } }\tag{3}
$$

$$
M A E = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \lvert S O H _ { i } - S \hat { O } H _ { i } \rvert\tag{4}
$$

Where $S O H _ { i }$ and $S \hat { O } H _ { i }$ refer to the battery’s real and predicted SOH value, respectively. Additionally, to assess the computational burden required for the SOH estimation, the total number of FLOPs is considered for every candidate solution. FLOPs represent the number of floating-point operations needed to run a single instance of the given model, the calculation of which was performed using the Keras library [27]. For each dataset, the BOGP algorithm was applied to fine-tune a total of thirteen different architectures within the designated hyperspace for each of the five input feature combinations. The evaluation metrics were then used for comparison. These thirteen architectures are detailed in Table 2. The first is the baseline network, a simple FFNN. The second category comprises straightforward stacked RNNS, the LSTM, the BiLSTM, the GRU, and the BiGRU NNs depicted in Figure 13. The next category involves combinations of stacked CNN and stacked RNN with the previously mentioned layers, presented in Figure 14. The last category is similar to the third, but with the addition of FC layers between the CNN and RNN layers, presented in Figure 15, representing the proposed method.

![](images/2896f09df1664ca4aa084a6f9db3bc469436cafba7ead4c5aced2e5bf2fe40a5.jpg)  
Figure 13: Second category of stacked RNN layers.

![](images/a7bf5f34d689157054f1ba34c3b7f770eca4e9aaac64127bf77505a535e431a4.jpg)  
Figure 14: Third category of the combination of CNN and RNN layers.

![](images/b6b5bedde809dabfbfe849ffb26fe22e14dea7b04b2d059fdb2465c5e58966c0.jpg)  
Figure 15: Fourth category of the combination of CNN, FC, and RNN layers.

Table 2: Models employed in the comparison to find the one yielding the best results.
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=4>Models</td></tr><tr><td rowspan=1 colspan=1>FNN</td><td rowspan=1 colspan=4>MLP</td></tr><tr><td rowspan=1 colspan=1>RNN</td><td rowspan=1 colspan=1>LSTM</td><td rowspan=1 colspan=1>BiLSTM</td><td rowspan=1 colspan=1>GRU</td><td rowspan=1 colspan=1>BiGRU</td></tr><tr><td rowspan=1 colspan=1>CNN-RNN</td><td rowspan=1 colspan=1>CNN-LSTM</td><td rowspan=1 colspan=1>CNN-BiLSTM</td><td rowspan=1 colspan=1>CNN-GRU</td><td rowspan=1 colspan=1>CNN-BiGRU</td></tr><tr><td rowspan=1 colspan=1>CNN-FC-RNN</td><td rowspan=1 colspan=1>CNN-FC-LSTM</td><td rowspan=1 colspan=1>CNN-FC-BiLSTM</td><td rowspan=1 colspan=1>CNN-FC-GRU</td><td rowspan=1 colspan=1>CNN-FC-BiGRU</td></tr></table>

## 4.1 Results on PoliMi dataset

This section evaluates the performance of the proposed model on the PoliMi dataset. The results of the best three models are presented in Table 3. The highest predictive performance was achieved with a model of the last category, where the CNN was integrated in a cascading manner with FC layers, followed by RNN layers. In this setup, the most effective RNN layer proved to be the BiLSTM layer. This configuration, particularly when using ICA and voltage as inputs, yielded a MAE of 0.12% and a RMSE of 0.14%. The comparison between the estimated values and the actual values can be viewed in Figure 16. The rest of the results of the exhaustive analysis can be found in Appendix A. This solution entails a high number of FLOPS, making it

![](images/b38a6c4e91ac0bcd88da00f760a9e940a9273db75096adfc49f24a5bd12bf4c5.jpg)  
Figure 16: Estimated values of SOH for the PoliMi dataset of the CNN-FC-BiLSTM model with inputs the ICA, DVA and the voltage.

Table 3: The top three models for the PoliMi dataset achieving the lowest errors with the corresponding input features, MAE, RMSE of the estimated SOH and the FLOPS
<table><tr><td>Model</td><td>CNN+FC+BiLSTM</td><td>CNN+FC+LSTM</td><td>BiGRU</td></tr><tr><td>Input</td><td>ICA, DVA, V</td><td>ICA</td><td>ICA, V, Q</td></tr><tr><td>MAE</td><td>0.12</td><td>0.12</td><td>0.16</td></tr><tr><td>RMSE</td><td>0.14</td><td>0.15</td><td>0.18</td></tr><tr><td>FLOPS</td><td>6424192</td><td>8605504</td><td>60544</td></tr></table>

computationally intensive. It would be beneficial to be implemented in a cloud-based environment to avoid the need for additional computational resources in the BMS of the EV. If computational constraints limit the application, the results based on FLOPS with the smallest size of the developed model are presented in Table 4. Combining the smallest size with the best predictive capabilities among the three cases presented in Table 4, the first case is the most favourable of the GRU as a model with FLOPS of 2176, achieving a MAE of 0.17% and a RMSE of 0.20%. The minimal computational cost of this configuration, based on the minimum FLOPS, coupled with its acceptable accuracy, offer a practical approach for online applications, implementing the model into a microcontroller of a BMS.

Table 4: The top three models for the PoliMi dataset achieving the minimum size with the corresponding input features, MAE, RMSE of the estimated SOH and the FLOPS
<table><tr><td>Model</td><td>GRU</td><td>LSTM</td><td>GRU</td></tr><tr><td>Input</td><td>ICA, V</td><td>ICA, DVA, V, Q</td><td>ICA, DVA, V, Q</td></tr><tr><td>MAE</td><td>0.17</td><td>0.20</td><td>0.31</td></tr><tr><td>RMSE</td><td>0.20</td><td>0.23</td><td>0.40</td></tr><tr><td>FLOPS</td><td>2176</td><td>2176</td><td>2176</td></tr></table>

## 4.2 Results on Oxford dataset

This section focuses on analyzing the performance of the proposed model using the Oxford dataset. The outcomes from the top three models are detailed in Table 5. Once more, the model from the last category, integrating a CNN in a cascading sequence with FC layers, followed by RNN layers, achieved the best predictive performance. Within this configuration, the BiLSTM layer emerged as the most effective RNN layer. This configuration, particularly when using ICA, DVA and voltage as inputs, yielded a MAE of 0.16% and a RMSE of 0.19%. The comparison between the estimated values and the actual values can be viewed in Figure 17. The rest of the results of the exhaustive analysis can be found in Appendix A. Similar to the pre-

![](images/33b34886aabae43ae4019df87c615f96b4c0f5779fc7c65171841cd4c7530512.jpg)  
Figure 17: Estimated values of SOH for the Oxford dataset of the CNN-FC-BiLSTM model with inputs the ICA and the voltage.

Table 5: The top three models for the Oxford dataset achieving the lowest errors with the corresponding input features, MAE, RMSE of the estimated SOH and the FLOPS
<table><tr><td>Model</td><td>CNN+FC+BiLSTM</td><td>CNN+FC+LSTM</td><td>CNN+BiGRU</td></tr><tr><td>Input</td><td>ICA, V</td><td>ICA, DVA, V, Q</td><td>ICA, V</td></tr><tr><td>MAE</td><td>0.16</td><td>0.16</td><td>0.17</td></tr><tr><td>RMSE</td><td>0.19</td><td>0.20</td><td>0.23</td></tr><tr><td>FLOPS</td><td>15076288</td><td>9379712</td><td>1630080</td></tr></table>

vious dataset, this solution requires a substantial number of FLOPS, leading to high computational demands. Implementing it in a cloud-based setting would be advantageous, while if computational limitations are once again a concern, the outcomes based on FLOPS for the most compact version of the developed model are shown in Table 6. Among the three scenarios presented there, the most favorable is the final case with the LSTM as model. It strikes a balance between size and performance, with FLOPS of 2305 and achieving an MAE of 0.22% and an RMSE of 0.26%. The reduced computational burden of this approach based on

Table 6: The top three models for the Oxford dataset achieving the minimum size with the corresponding input features, MAE, RMSE of the estimated SOH and the FLOPS
<table><tr><td>Model</td><td>LSTM</td><td>LSTM</td><td>LSTM</td></tr><tr><td>Input</td><td>ICA, DVA, V, Q</td><td>ICA, V</td><td>ICA, DVA, V</td></tr><tr><td>MAE</td><td>0.25</td><td>0.44</td><td>0.22</td></tr><tr><td>RMSE</td><td>0.29</td><td>0.54</td><td>0.26</td></tr><tr><td>FLOPS</td><td>2176</td><td>2176</td><td>2304</td></tr></table>

the minimum FLOPS, coupled with its acceptable accuracy, demonstrats the feasibility of implementing the model into a microcontroller of a BMS.

## 4.3 Results on NASA dataset

This final section is dedicated to evaluating the performance of the proposed model with the NASA dataset. Detailed results of the top three models are presented in Table 7. This time, the model from the third category, which combines a CNN in a cascading connection with the RNN layers, without the intermediate connection of FC layers, delivered the highest predictive accuracy. The BiGRU layer within this setup, this time, proved to be the most effective RNN layer. When utilizing solely the ICA as input, this configuration achieved a MAE of 0.45% and a RMSE of 0.71%. The comparison between estimated and actual values is available in Figure 18. Comprehensive results from the extensive analysis are available in Appendix A.

![](images/bb93748660dc72f78660ee707e501bc7eba95e726300ba1b3c155c153452aa4e.jpg)  
Figure 18: Estimated values of SOH for the NASA dataset of the CNN-BiGRU model with input the ICA

As with the previous dataset, this solution also involves a high number of FLOPS, necessitating significant computational resources. Opting for a cloud-based implementation would be beneficial in this context. However, if computational constraints are a concern, the results for the most compact model version, based on FLOPS, are displayed in Table 8. Of the three scenarios outlined there, the most advantageous is the third case, which uses the BiGRU model while being the second-best model overall based on the error metrics as it can be seen in Table 7. This option offers a good compromise between size and efficacy, with FLOPS of 4096 and achieving a MAE of 0.53% and a RMSE of 0.72%. The reduced computational burden of this approach, based on the minimum FLOPS, coupled with its acceptable accuracy, provide a viable approach for real-time operational scenarios, implementing the model into a microcontroller of a BMS.

Table 7: The top three models for the NASA dataset achieving the lowest errors with the corresponding input features, MAE, RMSE of the estimated SOH and the FLOPS
<table><tr><td>Model</td><td>CNN+BiGRU BiGRU</td><td>FFNN</td></tr><tr><td>Input</td><td>ICA ICA</td><td>ICA</td></tr><tr><td>MAE</td><td>0.45 0.53</td><td>0.56</td></tr><tr><td>RMSE</td><td>0.71 0.72</td><td>0.81</td></tr><tr><td>FLOPS</td><td>5799424 4096</td><td>327104</td></tr></table>

Table 8: The top three models for the NASA dataset achieving the minimum size with the corresponding input features, MAE, RMSE of the estimated SOH and the FLOPS
<table><tr><td>Model</td><td>LSTM</td><td>LSTM</td><td>LSTM</td></tr><tr><td>Input</td><td>ICA, V</td><td>ICA, V, Q</td><td>ICA</td></tr><tr><td>MAE</td><td>1.76</td><td>1.01</td><td>0.53</td></tr><tr><td>RMSE</td><td>2.23</td><td>1.17</td><td>0.72</td></tr><tr><td>FLOPS</td><td>512</td><td>640</td><td>4096</td></tr></table>

## 5 Discussion

The experimental results confirm the effectiveness of the proposed hybrid deep learning framework of combination of CNN and RNN moddels for accurate SOH estimation across three publicly available datasets. Specifically, the CNN-FC-BiLSTM model yielded the best performance on the Oxford and Polimi datasets, while the CNN-BiGRU model achieved superior results on the NASA dataset. For the Oxford and Polimi datasets, the CNN-FC-BiLSTM model successfully leveraged the strengths of convolutional layers for spatial feature extraction, fully connected layers for nonlinear transformation, and BiLSTM layers for capturing bidirectional temporal dependencies in the degradation patterns. The consistent performance across different battery cells within these datasets demonstrates the model’s ability to generalize well under relatively stable and homogeneous operating conditions. In contrast, the NASA dataset—characterized by higher noise levels and more aggressive degradation—was better handled by the CNN-BiGRU model. GRU units, known for their simpler structure compared to LSTMs, are often more efficient and robust in the presence of noisy or sparse data [28]. The superior performance of the CNN-BiGRU model in this context suggests that GRUs may offer advantages in real-world BMS applications where data quality is not always ideal. These results also highlight the importance of model selection and architecture design in battery health diagnostics. A one-size-fits-all model may not be optimal across diverse datasets, and hybrid architectures that integrate convolutional and recurrent elements offer a flexible foundation for adapting to different battery types and usage profiles. An additional observation is the benefit of combining multiple input features, such as voltage, current, capacity, and more complex post-processing of those to enrich the model’s understanding of battery behavior. This supports the use of multi-modal time-series data in SOH estimation, encouraging further research into data fusion techniques and sensor integration. Nevertheless, challenges remain. The current models assume fixed sequence lengths. Future work could investigate dynamic input handling with different input sequences for greater deployment flexibility. In conclusion, the presented results validate the hybrid CNN-based architectures as powerful tools for SOH estimation. These insights can inform the development of next-generation battery management systems that are both accurate and adaptable.

## 6 Conclusions

This research presents a novel approach for battery SOH estimation combining cascaded CNN and RNN layers. For PoliMi and Oxford datasets, BiLSTM layers with intermediate FC connections provided optimal performance, while the NASA dataset benefited from BiGRU layers without FC layers. Bayesian Optimization enabled systematic hyperparameter tuning, reducing prediction errors and improving generalization. Comparative analysis demonstrated superior accuracy relative to existing machine learning techniques. Performance validation across multiple datasets confirms the method’s effectiveness in capturing battery degradation dynamics. These results highlight the practical value of this framework for real-time battery management systems requiring accurate SOH assessment. Model architecture selection proves critical in optimizing SOH estimation accuracy across heterogeneous datasets. CNN-FC-BiLSTM demonstrated superior performance on the PoliMi and Oxford datasets, while CNN-BiGRU excels under the diverse conditions of the NASA dataset. This underscores that architectural configuration selection must be informed by dataset-specific degradation patterns. Bayesian hyperparameter tuning and integration of multi-modal features enhanced the robustness and generalizability of the framework, rendering it suitable for deployment across various operational environments. Future research could focus on exploring the advantages of adding attention mechanisms or transformer-based architectures to enhance the model’s ability to capture long-term dependencies in battery behavior. Additionally, investigating transfer learning approaches to utilize insights from similar battery datasets for refining the models for specific scenarios or battery chemistries is a promising direction. Conducting comprehensive experiments with a variety of battery datasets to assess the model’s ability to generalize across different battery types and conditions is also essential. Lastly, for practical implementation and to reduce computational expenses, methods such as model pruning and weight quantization could be considered to reduce the model’s size.

## References

[1] European Council, “Fit for 55 - the EU’s plan for a green transition,” https://www.consilium.europa. eu/en/policies/green-deal/fit-for-55-the-eu-plan-for-a-green-transition/, 2022, accessed 10 June 2023.

[2] P. Eleftheriadis, S. Giazitzis, S. Leva, and E. Ogliari, “Data-driven methods for the state of charge estimation of lithium-ion batteries: An overview,” Forecasting, vol. 5, no. 3, pp. 576–599, 2023. [Online]. Available: https://www.mdpi.com/2571-9394/5/3/32

[3] P. Eleftheriadis, S. Giazitzis, C. Ozmalatyalilar, S. Leva, and R. Zich, “An overview of data-driven methods for the state of health estimation,” in 2023 IEEE International Conference on Environment and Electrical Engineering and 2023 IEEE Industrial and Commercial Power Systems Europe (EEEIC / ICPS Europe), 2023, pp. 1–6.

[4] H. Meng, M. Geng, and T. Han, “Long short-term memory network with bayesian optimization for health prognostics of lithium-ion batteries based on partial incremental capacity analysis,” Reliability Engineering and System Safety, vol. 236, p. 109288, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S095183202300203X

[5] Y. Bharath, V. Anandu, U. Vinatha, and S. Sudeep, “Long-term estimation of soh using cascaded lstmrnn for lithium batteries subjected to aging and accelerated degradation,” Energy Storage, vol. 6, no. 8, p. e70066, 2024.

[6] S. Li, Z. Jiang, Z. Zhu, W. Jiang, Y. Ma, X. Sang, and S. Yang, “A framework of joint soc and soh estimation for lithium-ion batteries: Using bilstm as a battery

model,” Journal of Power Sources, vol. 635, p. 236342, 2025. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S0378775325001788

[7] F. Xiao, W. Yang, Y. Ding, X. Li, K. Zhang, and J. Liu, “A cnn-lstm method based on voltage deviation for predicting the state of health of lithium-ion batteries,” Battery Energy, vol. 4, no. 3, p. e20240036, 2025. [Online]. Available: https://onlinelibrary.wiley.com/doi/abs/10.1002/bte2.20240036

[8] G. dos Reis, C. Strange, M. Yadav, and S. Li, “Lithium-ion battery data and where to find it,” Energy and AI, vol. 5, p. 100081, 2021.

[9] Eleftheriadis, “PoliMi-TUB dataset - LG 18650HE4 li-ion battery,” 2024. [Online]. Available: https://data.mendeley.com/datasets/6hyhsjbwkb/1

[10] C. Lin, J. Xu, M. Shi, and X. Mei, “Constant current charging time based fast state-of-health estimation for lithium-ion batteries,” Energy, vol. 247, p. 123556, 2022.

[11] B. Saha and K. Goebel, “Uncertainty management for diagnostics and prognostics of batteries using bayesian techniques,” in 2008 IEEE Aerospace Conference, 2008, pp. 1–8.

[12] P. Eleftheriadis, M. Gangi, S. Leva, A. V. Rey, E. Groppo, and L. Grande, “Comparative study of machine learning techniques for the state of health estimation of li-ion batteries,” Electric Power Systems Research, vol. 235, p. 110889, 2024. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S0378779624007752

[13] P. Eleftheriadis, “PoliMi-TUB dataset - LG 18650HE4 Li-Ion battery,” 2024.

[14] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg et al., “Scikit-learn: Machine learning in python,” the Journal of machine Learning research, vol. 12, pp. 2825–2830, 2011.

[15] P. Eleftheriadis, S. Giazitzis, S. Leva, and E. Ogliari, “State of health estimation of lithium-ion battery via bidirectional long short-term memory neural network with bayesian hyperparameter tuning,” in 2023 IEEE International Conference on Energy Technologiesfor Future Grids (ETFG), 2023.

[16] C. Keras-Team, “Keras-team/keras: Deep learning for humans.” [Online]. Available: https: //github.com/fchollet/keras

[17] T. O’Malley, E. Bursztein, J. Long, F. Chollet, H. Jin, L. Invernizzi et al., “KerasTuner,” https://github. com/keras-team/keras-tuner, 2019, accessed 10 June 2023.

[18] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[19] P. J. Huber, “Robust estimation of a location parameter,” in Breakthroughs in statistics: Methodology and distribution. Springer, 1992, pp. 492–518.

[20] Z. Li, F. Liu, W. Yang, S. Peng, and J. Zhou, “A survey of convolutional neural networks: Analysis, applications, and prospects,” IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 12, pp. 6999–7019, 2022.

[21] P. Eleftheriadis, S. Leva, and E. Ogliari, “Bayesian hyperparameter optimization of stacked bidirectional long short-term memory neural network for the state of charge estimation,” Sustainable Energy, Grids and Networks, vol. 36, p. 101160, 2023.

[22] P. Eleftheriadis, M. Hegde, H. S. Sohal, and S. Leva, “Hyperband optimization of stacked bidirectional long short-term memory neural network for the state of charge estimation,” in 2023 IEEE International Conference on Environment and Electrical Engineering and 2023 IEEE Industrial and Commercial Power Systems Europe (EEEIC / ICPS Europe), 2023, pp. 1–6.

[23] G.-W. You, S. Park, and D. Oh, “Diagnosis of electric vehicle batteries using recurrent neural networks,” IEEE Transactions on Industrial Electronics, vol. 64, no. 6, pp. 4885–4893, 2017.

[24] G. Cheng, X. Wang, and Y. He, “Remaining useful life and state of health prediction for lithium batteries based on empirical mode decomposition and a long and short memory neural network,” Energy, vol. 232, p. 121022, 2021. [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S0360544221012706

[25] P. I. Frazier, “A tutorial on bayesian optimization,” arXiv preprint arXiv:1807.02811, 2018.

[26] L. Yang and A. Shami, “On hyperparameter optimization of machine learning algorithms: Theory and practice,” Neurocomputing, vol. 415, pp. 295–316, 2020.

[27] Keras, “Keras-FLOPs,” https://pypi.org/project/keras-flops/, 2020, accessed 10 June 2023.

[28] R. Józefowicz, W. Zaremba, and I. Sutskever, “An empirical exploration of recurrent network architectures,” in International Conference on Machine Learning, 2015. [Online]. Available: https://api.semanticscholar.org/CorpusID:9668607

## A Tables

Table 9: Errors and FLOPS for the first category of FFNN
<table><tr><td>Dataset</td><td colspan="3">Oxford</td><td colspan="3">NASA</td><td colspan="3">PoliMi</td></tr><tr><td>Features</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPs (k)</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPS (k)</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPS (k)</td></tr><tr><td>ICA</td><td>0.18</td><td>0.22</td><td>70016</td><td>0.56</td><td>0.81</td><td>327104</td><td>0.58</td><td>0.71</td><td>357248</td></tr><tr><td>ICA, V</td><td>0.23</td><td>0.30</td><td>155136</td><td>2.09</td><td>2.21</td><td>456320</td><td>0.30</td><td>0.40</td><td>582784</td></tr><tr><td>ICA, V, Q</td><td>0.33</td><td>0.38</td><td>126080</td><td>2.25</td><td>2.31</td><td>795904</td><td>0.26</td><td>0.32</td><td>991744</td></tr><tr><td>ICA, DVA, V</td><td>0.21</td><td>0.27</td><td>1306240</td><td>3.02</td><td>3.08</td><td>737024</td><td>0.27</td><td>0.40</td><td>872704</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.25</td><td>0.30</td><td>1922496</td><td>0.88</td><td>1.15</td><td>60160</td><td>0.49</td><td>0.52</td><td>1248832</td></tr></table>

Table 10: Errors and FLOPS for the second category of RNNs
<table><tr><td colspan="3">Dataset</td><td colspan="2">Oxford</td><td colspan="3">NASA</td><td colspan="3">PoliMi</td></tr><tr><td>Layer</td><td>Features</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPs (k)</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPS (k)</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPS (k)</td></tr><tr><td rowspan="5">LSTM</td><td>ICA</td><td>0.49</td><td>0.60</td><td>64.4</td><td>0.63</td><td>0.94</td><td>48.1</td><td>0.23</td><td>0.26</td><td>91.3</td></tr><tr><td>ICA, V</td><td>0.44</td><td>0.54</td><td>2.2</td><td>0.93</td><td>1.30</td><td>6.8</td><td>0.32</td><td>0.42</td><td>64.4</td></tr><tr><td>ICA, V, Q</td><td>0.24</td><td>0.31</td><td>4.9</td><td>1.01</td><td>1.17</td><td>0.6</td><td>0.24</td><td>0.27</td><td>9.0</td></tr><tr><td>ICA, DVA, V</td><td>0.22</td><td>0.26</td><td>2.3</td><td>0.85</td><td>1.09</td><td>65.9</td><td>0.55</td><td>0.61</td><td>31.7</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.25</td><td>0.29</td><td>2.2</td><td>2.89</td><td>3.07</td><td>36.6</td><td>0.20</td><td>0.23</td><td>2.2</td></tr><tr><td rowspan="5">BiLSTM</td><td>ICA</td><td>0.34</td><td>0.38</td><td>60.5</td><td>0.88</td><td>1.25</td><td>7.7</td><td>0.44</td><td>0.52</td><td>60.5</td></tr><tr><td>ICA, V</td><td>0.35</td><td>0.39</td><td>4.5</td><td>1.76</td><td>2.23</td><td>0.5</td><td>0.20</td><td>0.25</td><td>60.5</td></tr><tr><td>ICA, V, Q</td><td>0.61</td><td>0.72</td><td>7.7</td><td>1.86</td><td>2.05</td><td>38.5</td><td>0.21</td><td>0.24</td><td>45.2</td></tr><tr><td>ICA, DVA, V</td><td>0.65</td><td>0.76</td><td>120.1</td><td>1.01</td><td>1.30</td><td>11.0</td><td>0.26</td><td>0.31</td><td>9.0</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.26</td><td>0.34</td><td>4.5</td><td>1.42</td><td>1.72</td><td>60.9</td><td>0.19</td><td>0.26</td><td>59.0</td></tr><tr><td rowspan="5">GRU</td><td>ICA</td><td>0.89</td><td>1.08</td><td>2.6</td><td>2.81</td><td>2.95</td><td>6.8</td><td>0.45</td><td>0.56</td><td>7.0</td></tr><tr><td>ICA, V</td><td>0.80</td><td>0.96</td><td>4.6</td><td>1.56</td><td>1.80</td><td>6.4</td><td>0.17</td><td>0.20</td><td>2.2</td></tr><tr><td>ICA, V, Q</td><td>0.49</td><td>0.56</td><td>61.7</td><td>0.71</td><td>0.93</td><td>31.7</td><td>0.21</td><td>0.27</td><td>36.6</td></tr><tr><td>ICA, DVA, V</td><td>0.49</td><td>0.58</td><td>91.3</td><td>1.97</td><td>2.28</td><td>9.0</td><td>0.41</td><td>0.43</td><td>31.7</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.41</td><td>0.46</td><td>9.3</td><td>2.05</td><td>2.39</td><td>61.5</td><td>0.31</td><td>0.40</td><td>2.2</td></tr><tr><td rowspan="5">BiGRU</td><td>ICA</td><td>0.34</td><td>0.45</td><td>4.1</td><td>0.53</td><td>0.72</td><td>4.1</td><td>0.32</td><td>0.41</td><td>6.8</td></tr><tr><td>ICA, V</td><td>0.50</td><td>0.58</td><td>66.3</td><td>1.38</td><td>1.56</td><td>4.1</td><td>0.69</td><td>0.77</td><td>36.5</td></tr><tr><td>ICA, V, Q</td><td>0.56</td><td>0.66</td><td>4.5</td><td>0.91</td><td>1.10</td><td>60.5</td><td>0.16</td><td>0.18</td><td>60.5</td></tr><tr><td>ICA, DVA, V</td><td>0.58</td><td>0.64</td><td>9.2</td><td>1.90</td><td>2.11</td><td>68.2</td><td>0.26</td><td>0.34</td><td>5.9</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.91</td><td>1.05</td><td>11.6</td><td>0.65</td><td>0.91</td><td>54.4</td><td>0.84</td><td>0.88</td><td>8.8</td></tr></table>

Table 11: Errors and FLOPS for the third category of the combination of CNN and RNNs
<table><tr><td colspan="3">Dataset</td><td colspan="2">Oxford</td><td colspan="3">NASA</td><td colspan="3">PoliMi</td></tr><tr><td>Layer</td><td>Features</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPs (k)</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPS (k)</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPS (k)</td></tr><tr><td rowspan="5">CNN-LSTM</td><td>ICA</td><td>0.18</td><td>0.24</td><td>12111.5</td><td>0.77</td><td>0.98</td><td>5795.0</td><td>0.40</td><td>0.45</td><td>7025.0</td></tr><tr><td>ICA, V</td><td>0.33</td><td>0.37</td><td>13419.5</td><td>1.61</td><td>1.80</td><td>5281.3</td><td>0.21</td><td>0.26</td><td>1507.1</td></tr><tr><td>ICA, V, Q</td><td>0.19</td><td>0.22</td><td>14778.5</td><td>1.51</td><td>1.81</td><td>182.0</td><td>0.35</td><td>0.41</td><td>9223.7</td></tr><tr><td>ICA, DVA, V</td><td>0.26</td><td>0.33</td><td>14778.5</td><td>0.81</td><td>0.97</td><td>471.0</td><td>0.18</td><td>0.22</td><td>1984.0</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.26</td><td>0.31</td><td>14089.0</td><td>1.43</td><td>1.52</td><td>5142.8</td><td>0.33</td><td>0.41</td><td>10113.7</td></tr><tr><td rowspan="5">CNN-BiLSTM</td><td>ICA</td><td>0.19</td><td>0.24</td><td>2499.6</td><td>0.72</td><td>0.97</td><td>613.2</td><td>0.25</td><td>0.29</td><td>3267.5</td></tr><tr><td>ICA, V</td><td>0.17</td><td>0.23</td><td>1630.1</td><td>0.91</td><td>1.23</td><td>2280.4</td><td>0.22</td><td>0.24</td><td>8421.2</td></tr><tr><td>ICA, V, Q</td><td>0.21</td><td>0.24</td><td>7074.9</td><td>1.45</td><td>1.68</td><td>7130.0</td><td>0.21</td><td>0.24</td><td>5856.8</td></tr><tr><td>ICA, DVA, V</td><td>0.23</td><td>0.27</td><td>14785.3</td><td>0.96</td><td>1.25</td><td>7074.3</td><td>0.30</td><td>0.37</td><td>5840.9</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.31</td><td>0.36</td><td>16169.5</td><td>0.99</td><td>1.28</td><td>7755.9</td><td>0.65</td><td>0.78</td><td>10087.7</td></tr><tr><td rowspan="5">CNN-GRU</td><td>ICA</td><td>0.39</td><td>0.48</td><td>2586.8</td><td>0.70</td><td>1.03</td><td>5741.1</td><td>0.23</td><td>0.28</td><td>7534.1</td></tr><tr><td>ICA, V</td><td>0.40</td><td>0.49</td><td>8645.6</td><td>0.85</td><td>1.10</td><td>441.7</td><td>0.23</td><td>0.27</td><td>1749.2</td></tr><tr><td>ICA, V, Q</td><td>0.19</td><td>0.22</td><td>1987.2</td><td>1.09</td><td>1.22</td><td>7068.5</td><td>0.39</td><td>0.45</td><td>4407.6</td></tr><tr><td>ICA, DVA, V</td><td>0.32</td><td>0.41</td><td>1987.2</td><td>1.90</td><td>2.02</td><td>7100.3</td><td>0.31</td><td>0.44</td><td>9250.6</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.26</td><td>0.29</td><td>16145.4</td><td>1.56</td><td>1.89</td><td>7730.9</td><td>0.16</td><td>0.19</td><td>10108.9</td></tr><tr><td rowspan="5">CNN-BiGRU</td><td>ICA</td><td>0.80</td><td>0.94</td><td>12064.4</td><td>0.45</td><td>0.71</td><td>5799.4</td><td>0.27</td><td>0.32</td><td>7509.2</td></tr><tr><td>ICA, V</td><td>0.47</td><td>0.53</td><td>13453.7</td><td>1.82</td><td>2.09</td><td>6521.3</td><td>0.19</td><td>0.22</td><td>8364.7</td></tr><tr><td>ICA, V, Q</td><td>0.51</td><td>0.65</td><td>1258.4</td><td>1.00</td><td>1.12</td><td>2559.1</td><td>0.19</td><td>0.27</td><td>3809.0</td></tr><tr><td>ICA, DVA, V</td><td>0.79</td><td>0.89</td><td>14866.8</td><td>0.86</td><td>1.25</td><td>7183.7</td><td>0.20</td><td>0.23</td><td>9264.1</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.55</td><td>0.63</td><td>16142.5</td><td>0.83</td><td>1.05</td><td>3202.2</td><td>0.39</td><td>0.43</td><td>10156.4</td></tr></table>

Table 12: Errors and FLOPS for the fourth category of the combination of CNN, FC and RNNs
<table><tr><td colspan="3">Dataset</td><td colspan="2">Oxford</td><td colspan="3">NASA</td><td colspan="3">PoliMi</td></tr><tr><td>Layer</td><td>Features</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPs (k)</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPS (k)</td><td>MAE (%)</td><td>RMSE (%)</td><td>FLOPS (k)</td></tr><tr><td rowspan="5">CNN-D-LSTM</td><td>ICA</td><td>0.25</td><td>0.32</td><td>12174.85</td><td>1.04</td><td>1.33</td><td>3982.912</td><td>0.12</td><td>0.15</td><td>8605.504</td></tr><tr><td>ICA, V</td><td>0.19</td><td>0.23</td><td>2211.712</td><td>2.04</td><td>2.17</td><td>7425.152</td><td>0.24</td><td>0.28</td><td>9600.32</td></tr><tr><td>ICA, V, Q</td><td>0.27</td><td>0.30</td><td>10179.2</td><td>0.94</td><td>1.14</td><td>6478.528</td><td>0.33</td><td>0.47</td><td>7300.736</td></tr><tr><td>ICA, DVA, V</td><td>0.23</td><td>0.30</td><td>2980.608</td><td>0.77</td><td>1.27</td><td>8272.32</td><td>0.25</td><td>0.32</td><td>6586.688</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.16</td><td>0.20</td><td>9379.712</td><td>1.38</td><td>1.57</td><td>2098.496</td><td>0.19</td><td>0.24</td><td>4149.952</td></tr><tr><td rowspan="5">CNN-D-biLSTM CNN-D-GRU</td><td>ICA</td><td>0.46</td><td>0.52</td><td>4258.048</td><td>1.58</td><td>1.98</td><td>8269.504</td><td>0.28</td><td>0.32</td><td>6424.448</td></tr><tr><td>ICA, V</td><td>0.16</td><td>0.19</td><td>15076.29</td><td>0.89</td><td>1.18</td><td>7374.656</td><td>0.47</td><td>0.51</td><td>9672.512</td></tr><tr><td>ICA, V, Q</td><td>0.17</td><td>0.21</td><td>4866.304</td><td>1.65</td><td>1.79</td><td>7927.744</td><td>0.47</td><td>0.56</td><td>5159.104</td></tr><tr><td>ICA, DVA, V</td><td>0.45</td><td>0.51</td><td>15347.07</td><td>1.34</td><td>1.46</td><td>7582.144</td><td>0.12</td><td>0.14</td><td>6424.192</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.25</td><td>0.31</td><td>6664.64</td><td>0.78</td><td>1.04</td><td>2570.816</td><td>0.37</td><td>0.44</td><td>1184.96</td></tr><tr><td rowspan="5"></td><td>ICA</td><td>0.97</td><td>1.14</td><td>1181.824</td><td>1.12</td><td>1.26</td><td>6710.336</td><td>0.34</td><td>0.39</td><td>9778.24</td></tr><tr><td>ICA, V</td><td>0.71</td><td>0.86</td><td>2833.28</td><td>0.97</td><td>1.19</td><td>5116.544</td><td>0.38</td><td>0.43</td><td>1561.664</td></tr><tr><td>ICA, V, Q</td><td>0.36</td><td>0.43</td><td>9384.832</td><td>0.68</td><td>1.14</td><td>3202.624</td><td>0.18</td><td>0.23</td><td>414.784</td></tr><tr><td>ICA, DVA, V</td><td>0.35</td><td>0.45</td><td>5039.488</td><td>0.82</td><td>1.00</td><td>9565.504</td><td>0.24</td><td>0.28</td><td>5277.568</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.65</td><td>0.73</td><td>3039.488</td><td>0.62</td><td>0.86</td><td>1374.144</td><td>0.18</td><td>0.22</td><td>6828.736</td></tr><tr><td rowspan="5">CNN-D-biGRU</td><td>ICA</td><td>1.19</td><td>1.31</td><td>12463.74</td><td>1.99</td><td>2.41</td><td>6549.184</td><td>0.53</td><td>0.61</td><td>3895.936</td></tr><tr><td>ICA, V</td><td>0.59</td><td>0.66</td><td>13823.1</td><td>1.27</td><td>1.56</td><td>5555.52</td><td>0.42</td><td>0.45</td><td>9438.784</td></tr><tr><td>ICA, V, Q</td><td>0.73</td><td>0.84</td><td>6489.024</td><td>2.83</td><td>3.01</td><td>7927.744</td><td>0.65</td><td>0.71</td><td>3857.152</td></tr><tr><td>ICA, DVA, V</td><td>0.28</td><td>0.31</td><td>16562.94</td><td>1.26</td><td>1.37</td><td>1425.984</td><td>0.48</td><td>0.53</td><td>7059.52</td></tr><tr><td>ICA, DVA, V, Q</td><td>0.59</td><td>0.73</td><td>17868.54</td><td>1.18</td><td>1.43</td><td>1304.384</td><td>0.47</td><td>0.60</td><td>10587.01</td></tr></table>