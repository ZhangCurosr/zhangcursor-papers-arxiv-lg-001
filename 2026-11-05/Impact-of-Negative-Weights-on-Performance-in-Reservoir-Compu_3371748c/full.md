# Impact of Negative Weights on Performance in Reservoir Computing

Yeonsu Jeong<sup>∗</sup> and Seung-Woo Son<sup>†</sup>

Department of Applied Physics, Hanyang University, Ansan 15588, Korea

Seonho Nam<sup>∗</sup> and Dahan Choi<sup>∗</sup>

Department of Applied Physics, Hanyang University, Ansan 15588, Korea and R&D Center, Green Star Incorporated, Anyang 14056, Korea

Mi Jin Lee<sup>‡</sup> Department of Physics, Pusan National University, Busan 46241, Korea

Reservoir computing provides an eficient framework for time-series prediction because only the readout layer is trained while the recurrent reservoir remains fixed. In this study, we examine how the negative sign ratio of a signed Erdős–Rényi reservoir afects the internal dynamics and forecasting performance of an echo state network on the Mackey–Glass benchmark. To characterize the reservoir responses, we use two Shannon-entropy-based observables, namely node-state entropy and node frequency entropy, as empirical proxies for the diversity of reservoir states and temporal activation patterns. We find that intermediate negative sign ratios are associated with richer internal dynamics and substantially lower prediction error than the sign-homogeneous extremes for the task considered here. These results suggest that link-sign composition can be a useful structural parameter in reservoir design for chaotic forecasting. The present findings should be interpreted within the scope of the current benchmark and reservoir setting, and broader validation will be required to establish their generality across other tasks, topologies, and controlled spectral conditions.

Keywords: Reservoir computing, Echo state network, Chaotic systems

## I. INTRODUCTION

Artificial neural networks (ANNs) have become increasingly ubiquitous across diverse domains of scientific and technological applications, fundamentally transforming fields ranging from autonomous vehicle navigation to computational creativity in image synthesis algorithms [1–4]. Their computational capabilities extend to natural language processing and environmental modeling, where they contribute significantly to the accuracy and reliability of prediction systems [5–8]. Recent advances in representation learning have further enabled eficient retrieval and similarity search in large-scale document corpora, such as patents, by embedding textual data into latent spaces using pretrained language models and contrastive learning techniques [9]. The exponential growth in computational resources and data availability has facilitated the evolution of increasingly sophisticated ANN architectures [6, 10, 11]. This technological advancement has enabled the development of deep learning models capable of extracting intricate patterns and generating predictions with unprecedented accuracy, including representation learning frameworks that embed complex user characteristics into latent spaces using metric learning approaches such as triplet loss [12]. Nevertheless, the substantial computational demands and extensive training time required by state-of-the-art ANN architectures remain significant challenges, particularly in resource-constrained computational environments.

Within this context, reservoir computing (RC) has emerged as a compelling computational paradigm ofering distinct advantages. Characterized by reduced computational overhead and eficient training protocols, RC presents a robust alternative to conventional ANN architectures [4, 13–15]. The defining feature of RC, a fixed reservoir network structure maintained throughout the learning phase, enables a streamlined training procedure that circumvents the need for extensive iterative weight optimization characteristic of deep learning approaches [13, 16]. Furthermore, the architectural stability inherent to RC proves particularly advantageous for physical neural network implementations, where hardware constraints impose limitations on system flexibility and adaptability. This fixed architecture substantially simplifies the physical realization of ANNs, establishing RC as a promising methodology for developing computationally eficient artificial intelligence systems deployable across diverse applications, from edge computing devices to large-scale industrial systems [1, 5–8, 16].

Among the major RC paradigms, echo state networks (ESNs) are particularly attractive for chaotic time-series prediction because only the readout layer is trained while the recurrent reservoir remains fixed [13, 14]. This separation between nonlinear dynamical transformation and lightweight readout learning makes RC highly efective for tasks in which temporal memory and nonlinear mixing are both essential [17–19]. Figure 1 summarizes the workflow adopted in this study: the target signal is injected during training, multi-step forecasting is performed by autonomous feedback during prediction, and the reservoir itself is systematically converted into a signed network by controlling the fraction of negative links. Although random reservoirs are often treated as generic dynamical substrates, their detailed sign structure can strongly influence the resulting internal dynamics. In signed directed networks, positive and negative connections encode qualitatively diferent modes of interaction, and changing their ratio alters the balance between amplification and suppression in the network response [20–23]. In the context of RC, such structural changes are expected to modify the reservoir’s eigenvalue distribution, state diversity, temporal complexity, and ultimately its predictive utility [24, 25]. However, the role of the negative sign ratio in reservoir design has not been systematically clarified.

![](images/08fc3fc3d83ba69932947cc695869a8ad1bbf999266a32ca2cca64b4368150df.jpg)  
FIG. 1: Schematic architecture of the RC model. (a) Training mode, in which the input time series is fed into the reservoir and the readout weights $\mathbf { W _ { \mathrm { o u t } } }$ are optimized by comparing the output with the ground truth. (b) Prediction mode, in which the model operates autonomously by feeding the previous output back as the next input for iterative multi-step forecasting. (c) Construction of signed reservoir networks with controlled negative sign ratios η by systematically reassigning the signs of links in an initial Erdős–Rényi (ER) reservoir.

In this work, we investigate how the negative sign ratio of an ER reservoir afects the dynamics and predictive performance of an ESN on the Mackey–Glass benchmark. We combine dynamical inspection of the reservoir states with two complementary Shannon-entropy measures, namely node-state entropy and node-frequency entropy, to quantify the information capacity induced by diferent sign compositions. We then relate these entropy measures to forecasting accuracy. Our central finding is that an intermediate negative sign ratio generates richer internal dynamics and yields substantially lower prediction error than either sign-homogeneous extreme. This result provides a simple but efective structural principle for improving reservoir design.

## II. EXPERIMENTAL METHODOLOGY

Echo-state network - Reservoir computing is commonly implemented using two major paradigms: ESNs and liquid state machines. In this work, we focus on the ESN framework because it enables eficient training with a fixed recurrent reservoir, while only the readout parameters are learned [13, 14]. The ESN architecture consists of three functionally distinct layers: an input layer, a recurrent reservoir, and an output layer. As illustrated in Fig. 1(a), the input signal $\mathbf { u } ( t ) \in \mathbb { R } ^ { \tilde { N } _ { u } }$ is injected into the reservoir, generating the internal state $\mathbf { x } ( t ) \in \mathbb { R } ^ { N }$ . In the present work, the reservoir dynamics are governed by

$$
\begin{array} { r } { \mathbf { x } ( t ) = ( 1 - \alpha ) \mathbf { x } ( t - 1 ) \mathbf { \Omega } } \\ { + \alpha \operatorname { t a n h } ( \mathbf { W } _ { \mathrm { i n } } \mathbf { u } ( t ) + \mathbf { W } \mathbf { x } ( t - 1 ) + \mathbf { b } _ { \mathrm { r e s } } ) , } \end{array}\tag{1}
$$

![](images/315d97e5b5b60ffe34832b1f426c626387f4a5c979ac232cf5b9fbfb4e58ac69.jpg)

![](images/5cd8cc6f845bc7a2fe8e083aa22f9946c031ddc71d8aabd9a4c7ea0531eeea12.jpg)

![](images/b4900016bc69389f2ac7bff96bc7932b7806f80df62aad5c503592dee8e23def.jpg)

(d)  
![](images/5a61b8cbe967f3890094deca5a4fa3c5d6b45ec49034455384931bbd4a84db97.jpg)

![](images/6a6eb41810138fe2c7130cd258cfb78ca95717c837527180e48684deda195e4f.jpg)

![](images/cca7c4db4d086ec9f77addee582ccca98265f8fff17d60ac827988abbe2d3300.jpg)  
FIG. 2: Efect of the negative sign ratio on the internal dynamics of the reservoir. (a)-(c) Eigenvalue distributions of the reservoir weight matrix in the complex plane for representative negative sign ratios (0.0, 0.5, and 1.0), showing how the spectral structure changes with link-sign composition. (d)-(f) Representative node-state trajectories for the corresponding reservoirs, illustrating that intermediate sign ratios generate richer and more heterogeneous temporal responses than the extreme cases.

where α is the leaking rate, N is the number of reservoir nodes, $\mathbf { W } _ { \mathrm { i n } } \in \mathbb { R } ^ { \breve { N } \times N _ { u } }$ is the input weight matrix, $\mathbf { W } \in \mathbb { R } ^ { N \times N }$ is the recurrent reservoir weight matrix, and $\mathbf { b } _ { \mathrm { r e s } } \in \mathbb { R } ^ { N }$ is the reservoir bias vector.

The one-step-ahead prediction is obtained from an augmented feature vector consisting of the current input and reservoir state,

$$
\hat { \mathbf { y } } ( t + 1 ) = \widetilde { \mathbf { W } } _ { \mathrm { o u t } } \widetilde { \mathbf { x } } ( t ) \mathrm { ~ , ~ }\tag{2}
$$

where the augmented readout matrix and feature vector are defined as

$$
\begin{array} { r } { \widetilde { \mathbf { W } } _ { \mathrm { o u t } } = \left[ \mathbf { b } _ { \mathrm { o u t } } \ \mathbf { W } _ { u } \ \mathbf { W } _ { \mathrm { o u t } } \right] \in \mathbb { R } ^ { N _ { y } \times ( 1 + N _ { u } + N ) } , } \end{array}
$$

$$
\widetilde \mathbf { x } ( t ) = \left[ \mathbf { u } ( t ) \right] \in \mathbb { R } ^ { 1 + N _ { u } + N } .\tag{3}
$$

Here, $\mathbf { b _ { \mathrm { o u t } } } \in \mathbb { R } ^ { N _ { y } }$ is the readout bias, $\mathbf { W } _ { u } \in \mathbb { R } ^ { N _ { y } \times N _ { u } }$ represents the direct contribution of the input to the output, and $\mathbf { W _ { \mathrm { o u t } } } \in \mathbb { R } ^ { N _ { y } \times N }$ denotes the readout weights associated with the reservoir state. In this study, both the input and output are scalar variables $( N _ { u } = N _ { y } = 1 )$ ， since the Mackey–Glass benchmark is represented by a univariate time series.

The training and prediction modes difer in how the reservoir is driven. During training, the target time series is provided, and the readout parameters are determined from the augmented features constructed from the input and reservoir states, as shown in Fig. 1(a). Unlike conventional recurrent neural-network training based on backpropagation, in which network parameters are iteratively updated, the readout parameters of the ESN are obtained directly through ridge regression without iterative gradient-based optimization. During prediction, the model operates autonomously: the predicted output at each time step is fed back as the input for the next reservoir update, $\mathbf u ( t + 1 ) = \hat { \mathbf y } ( t + 1 )$ , as shown in Fig. 1(b). This closed-loop configuration enables iterative multistep forecasting and provides a direct test of whether the reservoir has captured the underlying temporal dynamics of the chaotic signal.

For the readout layer, we employ ridge regression, which provides a simple and numerically stable approach for fitting the linear readout while mitigating overfitting [15, 19, 26]. During training, the input sequence is sequentially fed into the reservoir to generate the corresponding reservoir states. After discarding the initial washout period, the retained input and reservoir state at each time step are combined into the augmented feature vector xe(t). These feature vectors are collected columnwise into the augmented feature matrix

![](images/a6c33d17073da10964fd2e270ebdd6a23829a26eff12bd8243f9d6d2813a1b9f.jpg)  
FIG. 3: Temporal evolution of the Mackey–Glass time series and the corresponding ESN predictions for diferent negative sign ratios: (a) $\eta _ { - } = 0 ,$ , (b) $\eta _ { - } = 0 . 5$ , and (c) $\eta _ { - } = 1$ . The shaded region indicates the washout period, while the black, blue, and red dashed lines denote the training data, ground truth, and autonomous prediction, respectively.

$$
\widetilde { \mathbf { X } } = \left[ \widetilde { \mathbf { x } } ( t _ { 1 } ) \widetilde { \mathbf { x } } ( t _ { 2 } ) \cdots \widetilde { \mathbf { x } } ( t _ { T _ { \mathrm { t r } } } ) \right] \in \mathbb { R } ^ { ( 1 + N _ { u } + N ) \times T _ { \mathrm { t r } } } ,\tag{4}
$$

where $T _ { \mathrm { t r } }$ denotes the number of training samples retained after the washout period. Since the readout is trained for one-step-ahead prediction, the corresponding target values are collected as

$$
\mathbf { Y } _ { \mathrm { t a r g e t } } = \left[ \mathbf { y } _ { \mathrm { t a r g e t } } ( t _ { k } + 1 ) \right] _ { k = 1 } ^ { T _ { \mathrm { t r } } } \in \mathbb { R } ^ { N _ { y } \times T _ { \mathrm { t r } } } ,\tag{5}
$$

where $t _ { k }$ denotes the k-th retained time index. The augmented readout matrix is then obtained by ridge regression as

$$
\widetilde { \mathbf { W } } _ { \mathrm { o u t } } = \mathbf { Y } _ { \mathrm { t a r g e t } } \widetilde { \mathbf { X } } ^ { \mathrm { T } } \left( \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \mathrm { T } } + \beta \mathbf { I } \right) ^ { - 1 } ,\tag{6}
$$

where $\beta$ is the regularization parameter and ${ \textbf { I } } \in$ $\mathbb { R } ^ { ( 1 + N _ { u } + N ) \times ( 1 + N _ { u } + N ) }$ is the identity matrix. This readout-only training strategy leaves the reservoir parameters fixed while allowing the output mapping to be determined eficiently by linear regression.

Signed reservoir - The reservoir network in our RC model is constructed from an ER random graph with $N = 5 0 0$ nodes and connection probability $p = 0 . 1 \ [ 1 3$ 15, 19, 27]. While the classical ER model specifies only the existence of links, the reservoir implementation additionally includes directionality and link weights [28– 30]. In conventional reservoir construction, the reservoir weight matrix W is commonly rescaled by its spectral radius, defined as the largest absolute eigenvalue. In the present study, however, W is normalized using the magnitude of the second-largest eigenvalue. Normalization by the largest eigenvalue, i.e., the spectral radius, produces slowly varying reservoir states and a correspondingly long transient relaxation in the present setting. The use of the second-largest eigenvalue suppresses this slow transient behavior. The signs of a prescribed fraction of links are subsequently reversed, with no additional normalization after the sign reassignment. The spectrum is therefore allowed to vary with the sign composition.

We define the negative sign ratio as

$$
\eta _ { - } = \frac { N _ { - } } { E } ,\tag{7}
$$

where $N _ { - }$ is the number of negative-sign links and E is the total number of links in the reservoir network. Figure 1(c) schematically illustrates this construction procedure. The sign ratio is varied from 0 to 1 in increments of 0.05, with $\eta _ { - } ~ = ~ 0$ corresponding to an all-positive reservoir and $\eta _ { - } ~ = ~ 1$ corresponding to an all-negative reservoir. Because the sign structure directly changes the spectrum of the recurrent weight matrix, it also changes the balance between stability, diversity, and responsiveness in the reservoir dynamics [24, 25]. The aim of the present work is to quantify how these changes propagate to entropy and prediction performance.

Chaotic benchmark - The Mackey–Glass system, introduced in 1977 by Mackey and Glass, is a nonlinear timedelay dynamical system originally formulated to model hematopoietic regulation in biological systems [31]. In the present study, the resulting scalar time series is used as both the input to the reservoir during training and the target for prediction. The governing equation is

$$
\frac { d z } { d t } = \frac { 0 . 2 z ( t - \tau ) } { 1 + z ( t - \tau ) ^ { 1 0 } } - 0 . 1 z ( t ) ~ ,\tag{8}
$$

where the delay parameter is fixed $\mathrm { a t } \ \tau = 1 7 .$ , for which the system exhibits chaotic behavior [32], and the resulting time series is illustrated in blue lines in Fig. 3(ac). The signal displays irregular oscillations with nonrepeating amplitudes and intervals, making it a demanding benchmark for recurrent predictors. This task is therefore well suited for testing whether a given reservoir design can simultaneously retain short-term memory and perform nonlinear transformation of past inputs.

Entropy measures and performance metric - To quantify the efect of the signed reservoir structure on its internal dynamics, we analyze two complementary measures based on Shannon entropy [33]: node-state entropy and node-frequency entropy. For a discrete probability distribution $\{ p _ { i } \}$ obtained by binning an observable, the Shannon entropy is defined as

$$
H = - \sum _ { i } p _ { i } \log p _ { i } \ .\tag{9}
$$

The two entropy measures considered here difer in the observable used to construct the probability distribution and in their corresponding binning procedures. The node-state entropy characterizes the instantaneous diversity of reservoir states across nodes [34]. At the observation time $T _ { : }$ , the state values $\{ x _ { i } ( T ) \} _ { i = 1 } ^ { N }$ of all reservoir nodes are divided into $B _ { \mathrm { s t a t e } } = 4 0$ equally spaced bins over the interval $[ - 1 , 1 ]$ For the j-th bin, the probability is defined as $p _ { j } ^ { \mathrm { s t a t \bar { e } } } = n _ { j } ^ { \mathrm { s t a t e } } / \dot { N }$ , where $n _ { j } ^ { \mathrm { s t a t e } }$ is the number of nodes whose state falls within that bin. The node-state entropy is then given by

$$
H _ { \mathrm { s t a t e } } = - \sum _ { j = 1 } ^ { B _ { \mathrm { s t a t e } } } p _ { j } ^ { \mathrm { s t a t e } } \log p _ { j } ^ { \mathrm { s t a t e } } .\tag{10}
$$

A larger $H _ { \mathrm { s t a t e } }$ therefore indicates that the reservoir nodes occupy a broader range of state values at the observation time.

The node-frequency entropy characterizes the diversity of dominant temporal frequencies across reservoir nodes. For each node i, the power spectrum is obtained from its temporal state trajectory, and the peak frequency $f _ { i } ^ { \mathrm { p e a k } }$ is identified as the frequency at which the spectral power is maximal. The absolute peak frequencies $\{ | f _ { i } ^ { \mathrm { p e a k } } | \} _ { i = 1 } ^ { N }$ are divided into $B _ { \mathrm { f r e q } } = 2 0$ equally spaced bins spanning the minimum and maximum values of the corresponding peak-frequency distribution. For the j-th frequency bin, the probability is defined as $p _ { j } ^ { \mathrm { f r e q } } = n _ { j } ^ { \mathrm { f r e q } } / N$ , where $n _ { j } ^ { \mathrm { f r e q } }$ is the number of nodes whose peak frequency falls within that bin. The node-frequency entropy is then given by

$$
H _ { \mathrm { f r e q } } = - \sum _ { j = 1 } ^ { B _ { \mathrm { f r e q } } } p _ { j } ^ { \mathrm { f r e q } } \log p _ { j } ^ { \mathrm { f r e q } } .\tag{11}
$$

Thus, $H _ { \mathrm { s t a t e } }$ describes the instantaneous heterogeneity of node states, whereas $H _ { \mathrm { f r e q } }$ describes the heterogeneity of dominant temporal frequencies across the reservoir.

The predictive performance of the ESN is evaluated by the mean squared error (MSE) between the target signal and the autonomous multi-step prediction. By comparing the MSE with the entropy measures across sign ratios, we assess whether a richer internal dynamical repertoire indeed translates into better forecasting of chaotic $\mathrm { { d y } \mathrm { { - } } }$ namics.

## III. RESULTS AND DISCUSSION

Signed reservoirs reshape spectral and state dynamics - We first examine how the sign composition modifies the intrinsic dynamics of the reservoir. Figure 2(a) shows representative eigenvalue distributions of the reservoir weight matrix for $\eta _ { - } ~ = ~ 0 . 0 , ~ 0 . 5$ , and 1.0. When the reservoir is dominated by one sign, the eigenvalue cloud is strongly elongated and accompanied by isolated real outliers, indicating biased dynamical modes. In contrast, the mixed-sign case $\eta _ { - } = 0 . 5$ produces a more balanced and dispersed spectrum in the complex plane. This more isotropic spectral structure implies that no single dynamical direction dominates the reservoir response.

The dynamical consequence of this spectral reorganization is evident in Fig. 2(b). For $\eta _ { - } = 0 . 0$ many node trajectories occupy similar narrow bands, indicating limited heterogeneity across nodes. For $\eta _ { - } ~ = 1 . 0$ , several node states are driven toward nearly fixed or weakly varying values, suggesting partial saturation of the reservoir response. The intermediate sign ratio, however, produces node activations with clearly diferent amplitudes, phases, and temporal patterns. In other words, mixing positive and negative links promotes a broader set of internal dynamical modes. These observations show that changes in the link-sign composition are accompanied by substantial changes in both the spectral structure of the reservoir and its dynamical responses.

Prediction performance exhibits an intermediate-sign optimum - To evaluate the forecasting performance of the ESN as a function of the negative sign ratio, we calculate the MSE, as shown in Fig. 4(a). For $\eta _ { - } \lesssim 0 . 4 2 5$ the MSE remains at a comparable level with no pronounced dependence on $\eta _ { - } .$ In the intermediate range $( 0 . 4 2 5 \lesssim \eta _ { - } \lesssim 0 . 8 2 5 )$ , the MSE decreases sharply by several orders of magnitude, indicating substantially improved forecasting performance. For $\eta _ { - } ~ \gtrsim ~ 0 . 8 2 5$ the MSE increases again and returns to values comparable to those observed at $\eta _ { - } \lesssim 0 . 4 2 5$ Thus, the lowest prediction errors are obtained in the mixed-sign regime rather than at either sign-homogeneous limit. This nonmonotonic dependence indicates that the link-sign composition, together with the accompanying changes in reservoir dynamics, is strongly associated with forecasting performance in the present setting. The pronounced reduction in MSE over the intermediate range therefore motivates a closer examination of the reservoir dynamics associated with this low-error regime.

Entropy characterizes the low-error dynamical regime - To characterize the dynamical properties of this lowerror regime, Fig. 4(b) shows the node-state and nodefrequency entropies as functions of $\eta _ { - } .$ The node-state entropy $H _ { \mathrm { s t a t e } }$ remains low at small η<sub>−</sub> and increases sharply as negative links are introduced. For $\eta _ { - } \gtrsim 0 . 4 2 5$ $H _ { \mathrm { s t a t e } }$ remains high over a broad mixed-sign range, followed by a slight decrease for $\eta _ { - } ~ \gtrsim ~ 0 . 8 2 5$ This behavior indicates that individual reservoir nodes explore a broader range of state values over time as the sign composition becomes mixed. The node-frequency entropy exhibits a stronger dependence on the negative sign ratio. It is relatively large over an intermediate range of η<sub>−</sub> but decreases markedly for $\eta _ { - } \gtrsim 0 . 8 2 5$ Thus, the reservoir can maintain a relatively broad of visited state values even when the diversity of dominant frequencies across nodes becomes substantially reduced. The two entropy measures therefore characterize complementary aspects of the reservoir dynamics: the diversity of state values explored by individual nodes and the heterogeneity of their dominant temporal frequencies.

![](images/519b5f7785ef2c29863ccb95c4676bd146399db2252632e453d436e634bcda6d.jpg)

![](images/27b15db3c72fde1b73e5b7cc82b452ea7e89c212b47b57ba78b3866dabe2b451.jpg)  
FIG. 4: Prediction performance and entropy-based measures versus η<sub>−</sub>. (a) Circles indicate the mean MSE, and error bars denote the standard deviation. (b) The colored symbols denote node-state entropy $H _ { \mathrm { s t a t e } }$ (blue) and $H _ { \mathrm { f r e q } }$ (red), and error bars denote the standard deviation. Shaded regions are included as visual guides to distinguish diferent ranges of η<sub>−</sub>.

Comparison with the prediction results in Fig. 4(a) shows that the low-error region broadly coincides with high node-state entropy and appreciable node-frequency entropy. For $\eta _ { - } \gtrsim 0 . 8 2 5$ , however, the node-state entropy remains relatively high while the node-frequency entropy decreases and the prediction error increases. This observation indicates that broad exploration of node-state values alone is not suficient to characterize the regime of accurate forecasting. Within the present setting, low prediction error is associated with the coexistence of broad state exploration and heterogeneous dominant temporal frequencies.

These results suggest that entropy-based measures can provide useful diagnostics of changes in reservoir dynamics induced by the link-sign composition. For the Mackey–Glass task considered here, the two entropy measures help characterize the dynamical regime associated with improved forecasting performance. They should, however, be regarded as empirical indicators of state and frequency diversity rather than as general measures of computational capacity.

## IV. CONCLUSION

In this study, we investigated how the negative sign ratio of a signed ER reservoir influences the internal dynamics and prediction performance of an echo state network on the Mackey–Glass benchmark. Within the present setting, reservoirs with intermediate negative sign ratios exhibited more heterogeneous node dynamics, larger entropy-based diversity measures, and lower prediction error than the sign-homogeneous extremes. These observations indicate that link-sign composition is closely associated with the dynamical and predictive behavior of the reservoir in the present setting.

At the same time, the present work should be interpreted as a proof of concept rather than as a general design law for RC. The analysis was performed on a single benchmark task with a specific reservoir construction, and the entropy-based measures used here are best understood as empirical proxies for internal state and dominant-frequency diversity rather than formal measures of computational capacity. Therefore, our results support the practical relevance of link-sign composition in the current setting, but they do not yet establish universal optimal sign ratios for reservoir design.

Several directions for future studies follow naturally from this work. First, it will be important to test whether the intermediate-sign optimum persists when the spectral radius is held fixed across diferent sign ratios, thereby partially separating the efect of sign composition from variations in spectral radius. Second, the robustness of the present result should be further examined across diferent reservoir sizes, sparsities, and input scalings. Third, the same framework should be evaluated on additional benchmark tasks, including other chaotic systems and real-world temporal datasets, to determine how task-dependent the optimal sign regime is. Fourth, entropy-based indicators should be compared with more formal measures such as memory capacity and related computational-capacity metrics [35, 36]. Finally, extending the present framework to alternative reservoir topologies and hardware-oriented physical implementations may clarify when signed interactions provide a robust practical advantage.

Overall, the present study suggests that the negative sign ratio is a promising structural parameter for shaping reservoir dynamics in chaotic prediction, and motivates broader investigations of signed reservoirs across reservoir-computing tasks and architectures.

[1] G. Diraco, A. Leone, and P. Siciliano, Sensors 19, 3549 (2019).

[2] X. Cai, H. Ning, S. Dhelim, R. Zhou, T. Zhang, Y. Xu, and Y. Wan, Digit. Commun. Netw. 7, 505 (2021).

[3] P. P. Angelov, E. A. Soares, R. Jiang, N. I. Arnold, and P. M. Atkinson, Wiley Interdiscip. Rev. Data Min. Knowl. Discov. 11, e1424 (2021).

[4] G. Tanaka, T. Yamane, J. B. Héroux, R. Nakane, N. Kanazawa, S. Takeda, H. Numata, D. Nakano, and A. Hirose, Neural Netw. 115, 100 (2019).

[5] C. Wingfield, C. Zhang, B. Devereux, E. Fonteneau, A. Thwaites, X. Liu, P. Woodland, W. Marslen-Wilson, and L. Su, Front. Comput. Neurosci. 16, 1057439 (2022).

[6] A. Kalhor, in Hardware Architectures for Deep Learning (Institution of Engineering and Technology, 2020), pp. 3–26.

[7] S. Huang, W. Klein, and H. Gould, Phys. Rev. E 103, 033305 (2021).

[8] J. Degrave, F. Felici, J. Buchli, M. Neunert, B. Tracey, F. Carpanese, T. Ewalds, R. Hafner, A. Abdolmaleki, D. de Las Casas, et al., Nature 602, 414 (2022).

[9] S. H. Jo, S. Park, Y. Kwon, M. J. Lee, and S.-W. Son, New Phys.: Sae Mulli 73, 478 (2023).

[10] I. Lorencin, N. Anđelić, J. Španjol, and Z. Car, Artif. Intell. Med. 102, 101746 (2020).

[11] J. Lyu, H. Shi, J. Zhang, and J. Norvilitis, Front. Neuroinform. 16, 961588 (2022).

[12] S. Koo, S. Park, Y. Kwon, M. J. Lee, and S.-W. Son, New Phys.: Sae Mulli 73, 385 (2023).

[13] M. Lukoševičius and H. Jaeger, Comput. Sci. Rev. 3, 127 (2009).

[14] H. Jaeger, Bonn, Germany: German national research center for information technology gmd technical report 148, 13 (2001).

[15] Y. Kawai, J. Park, and M. Asada, Neural Netw. 112, 15 (2019).

[16] Y. Zhang, H. Zhang, J. Zhang, L. Li, and Z. Zheng, in 2021 International Symposium on Electrical, Electronics and Information Engineering (2021), pp. 344–349.

[17] D. J. Gauthier, E. Bollt, A. Grifith, and W. A. Barbosa,

## Acknowledgments

This work was supported by the National Research Foundation (NRF) of Korea through the Grant Nos. RS-2024-00341317 (M.J.L.) and RS-2026-25488703 (S.- W.S.). We also acknowledge the hospitality at APCTP, where part of this work was done. We thank APCTP, Pohang, Korea, for their hospitality during the Topical Research Program [APCTP-2025-T04], from which this work greatly benefited.

Nat. Commun. 12, 5564 (2021).

[18] P.-R. Vlachas, J. Pathak, B. R. Hunt, T. P. Sapsis, M. Girvan, E. Ott, and P. Koumoutsakos, Neural Netw. 126, 191 (2020).

[19] J. Pathak, B. Hunt, M. Girvan, Z. Lu, and E. Ott, Phys. Rev. Lett. 120, 024102 (2018).

[20] T. Zaslavsky, Discrete Appl. Math. 4, 47 (1982).

[21] J. Kunegis, S. Schmidt, A. Lommatzsch, J. Lerner, E. W. De Luca, and S. Albayrak, in Proceedings of the 2010 SIAM international conference on data mining (SIAM, 2010), pp. 559–570.

[22] Y. Guan, S. Ren, and A. Li, Automatica 147, 110694 (2023).

[23] Z. Du and B. Zhou, Oper. Matrices 17, 57 (2023).

[24] M. Brunetti and Z. Stanić, Comput. Appl. Math. 41, 118 (2022).

[25] D. Verstraeten and B. Schrauwen, in International Conference on Artificial Neural Networks (Springer, 2009), pp. 985–994.

[26] F. Wyfels, B. Schrauwen, and D. Stroobandt, in International conference on artificial neural networks (Springer, 2008), pp. 808–817.

[27] F. Wyfels and B. Schrauwen, Neurocomputing 73, 1958 (2010).

[28] P. Erdős, A. Rényi, et al., Publ. Math. Inst. Hungar. Acad. Sci 5, 17 (1960).

[29] S.-W. Son, B. J. Kim, H. Hong, and H. Jeong, Phys. Rev. Lett. 103, 228702 (2009).

[30] A. Zeng, S.-W. Son, C. H. Yeung, Y. Fan, and Z. Di, Phys. Rev. E 83, 045101 (2011).

[31] M. C. Mackey and L. Glass, Science 197, 287 (1977).

[32] L. Junges and J. A. Gallas, Phys. Lett. A 376, 2109 (2012).

[33] C. E. Shannon, Bell Syst. Tech. J. 27, 379 (1948).

[34] S. Mandal and M. D. Shrimali, Chaos 31, 031101 (2021).

[35] H. Jaeger, Scholarpedia 2, 2330 (2007).

[36] J. Dambre, D. Verstraeten, B. Schrauwen, and S. Massar, Sci. Rep. 2, 514 (2012).