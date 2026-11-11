# MouldIt: Data Efficiency of Multi-Task Temporal Representation Learning in Injection Moulding

Abdullah Ayad

Abstract—Smart industrial solutions that can address multiple tasks require models that understand the underlying processes. We examine representation learning for modeling the injection moulding process as an example of such solutions. We consider its data efficiency and multi-modal benefit compared to classical machine learning alternatives. To this end, we synthesize a processinformed dataset based on a real reference cycle of an Erlanger Träger part. We then introduce MouldIt, a temporal convolution and attention based variational autoencoder that reconstructs in-mould signals, while classifying parts as acceptable or as one of our synthetic failure modes. We evaluate MouldIt on two tasks, known-class classification and unknown-class detection, across nine dataset sizes in single-modal and multi-modal settings. We compare it with an accept-all baseline and the classical methods k-nearest neighbors, logistic regression, random forests, and principal component analysis, which we adapt to support both tasks. MouldIt outperforms classical methods across both tasks from 4000 cycles in the single-modal setting and from 2000 cycles in the multi-modal setting. It achieves part sorting losses of 6.0 ± 0.7 % and 1.0 ± 0.3 % in the single-modal and multimodal settings respectively, ranking first across all baselines and a combination of principal component analysis and random forests.

Note to Practitioners—Sorting injection moulded parts automatically requires a system that recognizes common defects and also flags defects it has never seen. Defect data are expensive to collect, so engineers need to know how much data such a system requires and whether additional sensors are worth installing. Better sorting raises quality by shipping fewer faulty parts and reduces waste by rejecting fewer good ones, and one model that handles both recognition and flagging avoids maintaining two. We studied these questions on synthetic cycles generated from one recorded cycle of a real part. With only a cavity pressure sensor, random forests with a separate check for unknown defects sort parts best when data are scarce, and our neural network overtakes them only from about 4000 cycles. With a temperature sensor and two ultrasound sensors added, the network sorts best at every dataset size we tested, although the random forests match it at 16000 cycles. These results can guide a first choice of sensors and methods, but do not replace testing on a real line: all cycles derive from one part and one material, the synthetic defect sizes are modeling choices, and about 5% false alarms is high for production. Before deployment, the system should be validated on real defects, and its alarm threshold set by weighing the cost of shipping a faulty part against that of rejecting a good one. The learned representations of the process could also be used for control, not only for sorting.

Index Terms—Data efficiency, fault detection, injection moulding, multi-modal sensing, part sorting, representation learning, synthetic data, variational autoencoders

## I. INTRODUCTION

NDUSTRIAL production has traditionally relied on custom pre-programmed solutions that operate in known and optimized environments with reliability, productivity, and cost reduction as their objectives. Meanwhile, in other domains like language, general-purpose solutions have offered us a different problem-solving perspective. Instead of targeting a single objective, language models aim to understand their data in a way that allows them to address multiple downstream tasks. Inspired by this progress, we expect similar benefits from learning representations of industrial processes.

Learning such representations has to account for the nature of industrial process data, which are generally temporal and usually tabular. Temporal neural network architectures with suitable inductive biases can offer more than the prediction strengths of classical methods like random forests or the encoding strengths of methods like PCA.

Injection moulding is a suitable candidate for evaluating such models, as it is a critical process for most industries, and it typically employs multi-modal data. The process is characterized by static machine setting parameters (MSPs) and dynamic process parameters (PPs), typically recorded on the producing machines. These process parameters are commonly used for tasks such as monitoring for anomaly detection, process optimization, and process control. For finegrained dynamic information, in-mould sensors are usually installed, such as pressure, temperature, and ultrasound sensors. Ultrasound, for example, can track gap development between the part and the cavity wall, which pressure cannot monitor [1]. The process outputs usually have a number of quality metrics, like weight, size, or specific part deformation labels.

A natural application in this process is automated part sorting, where a decision can go wrong in either direction: good parts are rejected, or faulty parts are shipped. Sorting builds on two tasks, classifying known faults and detecting unknown ones, since a classifier trained on known faults alone can ship a new fault as a good part. A single model that performs both tasks would also avoid maintaining separate models. Automated sorting thus addresses three concerns of production: quality, by keeping faulty parts from shipping; efficiency, by not rejecting good ones; and reliability, by flagging unseen faults.

Naturally, the wide adoption of such models in industry faces barriers. These models are unpredictable and not well understood; therefore, works on explainability, on necessary conditions for those models to excel, and on production metrics may facilitate their adoption.

Data efficiency is among these necessary conditions, since representation learning typically requires large amounts of data to learn the underlying process. This raises four questions: how much data is needed for model performance to saturate; whether incorporating additional modalities can reduce this requirement; whether, at such a saturation point, representation-learning models improve over classical alternatives on productionrelevant metrics; and whether their learned representations generalize to unknown failure modes.

![](images/8ea4caff940fb7bd68ebe4dc1f749df641863b6e51cabba5b861522ab0dff83b.jpg)  
Fig. 1. Overview of this work. (a) We start from one real cycle of an Erlanger Träger part. (b) From it, we synthesize OK cycles and process-informed NOK variants, in datasets of 50 to 16000 cycles. (c) We train MouldIt, a temporal VAE with convolution and attention layers, and compare it with random forests, logistic regression, PCA, PCA with random forests, kNN, and an accept-all reference. (d) MouldIt leads on both known-class classification and unknown class detection from 4000 cycles with pressure only, and from 2000 cycles with multi-modal input. (e) A latent analysis shows whether each unknown cycle is detected by the latent term, the reconstruction term, or both, here for secondary pressure rise. (f) At 16000 cycles, its part sorting loss is 6 ± 0.7% with pressure only and 1 ± 0.3% with multi-modal input.

Answering these questions requires data that cover many failure modes. However, since representation learning is not yet widely used in this industry, foundational injection moulding datasets are not yet available. Additionally, access to industrygrade simulation software is often proprietary. Synthetic data is thus a practical way to explain and benchmark such models. We therefore address them on synthetic injection moulding data (Fig. 1), with the following contributions:

• an injection moulding data synthesis method requiring a single real sample, inspired by the success of [2] with synthetic training.

• a novel multi-objective temporal representation learning model for injection moulding in-mould data (pressure, temperature, and ultrasound).

• an analysis of the effect of dataset size and input modalities on model performance across known-class classification and unknown-class detection.

• a latent analysis of how MouldIt detects unknown faults.

• an evaluation of part sorting losses across the same dataset sizes and input modalities.

## II. RELATED WORKS

Our work focuses on representation learning for industrial applications, specifically injection moulding. We therefore review each topic first across industrial processes, then narrow down to injection moulding where such work exists.

## A. Synthetic Data and Injection Moulding Datasets

Synthetic data generation has been explored in manufacturing to address the scarcity of defective samples, including methods that generate visual defects for industrial inspection [3]–[5].

In injection moulding (IM), public datasets are scarce and heterogeneous. Some provide cycle-level scalar features [6] or machine-side time series such as injection pressure and flow curves [7], [8]. The dataset [9] additionally contains full-cycle cavity pressure (in-mould), with manual OK/NOK and defect-type labels. Further in-mould pressure data are either limited to a few labeled curves [10] or simulated [11]. Synthetic generation of IM data from causal models has also been explored [12], though only for cycle-level scalar features rather than full-cycle pressure curves. We consider an Erlanger Träger part, for which a public dataset provides one cycle of in-mould pressure, temperature and ultrasound data [13]. We therefore synthesize a larger dataset from this cycle.

## B. Representation Learning for Industrial Processes

Representation learning has gained traction for modeling industrial processes. In fault diagnosis, [14] compares singlemodal and multi-modal inputs and explores how to fuse them. Autoencoders have been used to detect anomalies in semiconductor manufacturing [15] and, through their latent space, faults in air handling units [16], while multi-modal fusion has supported soft sensing in sintering [17].

In injection moulding, classical machine learning models, specifically random forests, have been effective on tabular data [6]. More recently, representation learning, which learns latent process representations alongside task-specific predictions, has been used for process monitoring and anomaly detection [18]– [20], quality prediction [21], and process control [22]. We investigate it through two new lenses: data efficiency and multi-modal benefit. Unlike [18], which uses a single modality, we design a novel multi-objective VAE that learns from multimodal synthetic data, and we study both lenses on two tasks: classification and unknown-class detection.

## C. Data Efficiency of Industrial Models

Labeled IM data require production trials and part inspection, which makes the amount of training data a central constraint. Many works therefore reduce the data a model needs. In manufacturing, a model can learn from the quality estimates of physics-based models when measured labels are costly [23]. Transfer learning has also been used to generate scarce fault data [24]. In IM, transfer learning of supervised models on scalar process parameters is a common approach [25]–[27].

A separate question is how much data a model needs in the first place, which shows in how its performance scales with the training-set size. For cavity pressure data, a decision tree has reached over 90% accuracy with 1–3 training samples per class [28]. For learned representations, this scaling has not been the focus of prior IM studies, neither for classification accuracy nor for unknown-class detection. We measure this scaling for MouldIt and classical methods.

## D. Unknown Fault Detection in Industrial Processes

Production can encounter faults that are missing from the training data. [29] address open-set fault detection with a contrastive conditional adversarial autoencoder, and [30] diagnose unseen faults from fault attributes. [31] unify fault detection and classification, which are conventionally treated as separate tasks, in a framework that handles new faults without full retraining. We consider these tasks in injection moulding, with a single model that classifies known faults and detects unknown ones.

## E. Explainability of Industrial Models

Interpretable fault diagnosis has built on process knowledge [32] and on the dynamic dependencies between process variables [33]. In IM, [34] interpret a neural network to analyze how features of the cavity pressure profile affect part weight, [19] interpret the root causes of detected anomalies, and [18] derive interpretable indices of the deviations caused by machine parameters. We analyze the latent space to explain how MouldIt detects unknown faults.

## III. METHODS

## A. Synthetic Data Generation

We generate synthetic single-modal and multi-modal datasets and label each sample as OK or as one of several processinformed NOK variants. We first preprocess the reference cycle by aligning its modalities on a common time grid (Section III-A2). We then locate phase markers (fill start, pressure peak, holding end) in the normalized cavity pressure channel, as well as the temperature peak and the ultrasonic minima (Section III-A3); most failure modes act relative to these points. Next, we initialize each sample as an OK copy of the reference cycle and obtain each NOK variant by applying one of the failure modes of the respective setting (Sections III-A4 and III-A5). Finally, we apply OK disturbances, i.e., benign process variations that leave the label unchanged, to all samples.

![](images/cbd754306806bf0bcdb14ba664e714a59d5919d934034f23e323cbd19fe887a7.jpg)  
Fig. 2. Real Erlanger Träger reference cycle from which all cycles are synthesized, on the common time grid and with each channel min-max normalized. The single-modal input is (a) the cavity pressure of Kistler sensor P1, and the multi-modal input adds (b) the cavity temperature of Kistler sensor T3 and (c), (d) the RMS amplitudes of ultrasound sensors US1 and US4. Dotted lines mark the fill start, pressure peak, and holding end at 3.3, 3.7, and 15.2 s, and shaded regions the filling and holding phases. The temperature peaks at 4.4 s, US1 has its minimum at the pressure peak, and US4 has its minimum in the cooling phase at 19.3 s.

1) Input Modalities: Our reference cycle is the first cycle of a public dataset of an Erlanger Träger part [13]. In the single-modal setting, we learn from Kistler in-mould pressure sensors (Fig. 2(a)), as in [18]. In the multi-modal setting, we use temperature, pressure, and two ultrasound sensors (Fig. 2).

2) Temporal Alignment: Each sensing channel (modality) provides samples $r _ { i } ^ { ( c ) }$ at timestamps $\tau _ { i } ^ { ( c ) }$ , where $c \in$ {P, T, US1, US4}. The pressure and temperature channels share irregular timestamps at, on average, ≈ 7.5 Hz, since the acquisition software stores more samples during the filling phase, while the ultrasound channels are exported as normalized RMS values at a regular ≈ 47.5Hz. We therefore align all channels on a uniform time grid matched to the slower pressure/temperature recording. The grid is

$$
t _ { k } = t _ { \mathrm { s t a r t } } + k \Delta t , \qquad \Delta t = \frac { t _ { \mathrm { e n d } } - t _ { \mathrm { s t a r t } } } { T - 1 } ,\tag{1}
$$

for $k = 0 , \ldots , T - 1$ , where $\left[ t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } \right]$ is the recording window covered by all sensors,

$$
t _ { \mathrm { s t a r t } } = \operatorname* { m a x } _ { c } \operatorname* { m i n } _ { i } \tau _ { i } ^ { ( c ) } , \qquad t _ { \mathrm { e n d } } = \operatorname* { m i n } _ { c } \operatorname* { m a x } _ { i } \tau _ { i } ^ { ( c ) } ,\tag{2}
$$

$$
T = \big | \{ i : t _ { \mathrm { s t a r t } } \leq \tau _ { i } ^ { ( \mathrm { P , T } ) } \leq t _ { \mathrm { e n d } } \} \big | ,\tag{3}
$$

where $\tau _ { i } ^ { ( \mathrm { P , T } ) }$ are the timestamps shared by the pressure and temperature channels. The pressure/temperature recording spans 0.09–57.2s and the ultrasound recording 0.11–45.4s, so the pressure/temperature signals are truncated from 403 to $T = 3 4 0$ samples, dropping one sample before $t _ { \mathrm { s t a r t } }$ and the 62 samples after $t _ { \mathrm { e n d } }$ , which lie in the residual cooling phase. This gives $t _ { \mathrm { e n d } } - t _ { \mathrm { s t a r t } } \approx 4 5 . 3 \mathrm { s }$ , i.e. $\Delta t \approx 0 . 1 3 4 \mathrm { s }$ . Both settings use this window and thus share the sequence length T.

At each grid point $t _ { k } .$ , a channel’s value is linearly interpolated between its two neighboring samples $\tau _ { i } ^ { ( c ) } \leq t _ { k } \leq \tau _ { i + 1 } ^ { ( c ) }$

$$
\bar { x } _ { k } ^ { ( c ) } = \left( 1 - \nu _ { k } \right) r _ { i } ^ { ( c ) } + \nu _ { k } r _ { i + 1 } ^ { ( c ) } , \quad \nu _ { k } = \frac { t _ { k } - \tau _ { i } ^ { ( c ) } } { \tau _ { i + 1 } ^ { ( c ) } - \tau _ { i } ^ { ( c ) } } ,\tag{4}
$$

and each channel is then min-max normalized to $[ 0 , 1 ] .$

$$
x _ { k } ^ { ( c ) } = \frac { \bar { x } _ { k } ^ { ( c ) } - \operatorname* { m i n } _ { j } \bar { x } _ { j } ^ { ( c ) } } { \operatorname* { m a x } _ { j } \bar { x } _ { j } ^ { ( c ) } - \operatorname* { m i n } _ { j } \bar { x } _ { j } ^ { ( c ) } } .\tag{5}
$$

3) Phase Extraction: We extract three phase markers from the normalized pressure $p _ { k } = x _ { k } ^ { ( \mathsf { P } ) }$ , namely the onset of the filling phase, the pressure maximum, and the end of the holding phase. These markers are shared by all channels, since all channels lie on the same grid (Fig. 2). With a fixed threshold $\theta = 0 . 1$ , they are

$$
k _ { \mathrm { p e a k } } = \arg \operatorname* { m a x } _ { k } p _ { k } ,\tag{6}
$$

$$
k _ { \mathrm { f i l l } } = \operatorname* { m i n } \{ k < k _ { \mathrm { p e a k } } : p _ { k } > \theta \} ,\tag{7}
$$

$$
k _ { \mathrm { h o l d } } = \mathrm { m i n } \{ k \geq k _ { \mathrm { p e a k } } : p _ { k } < \theta \} .\tag{8}
$$

The markers define three phases: the filling phase ${ \mathcal { F } } =$ $[ k _ { \mathrm { f i l l } } , k _ { \mathrm { p e a k } } ) ;$ the holding phase $\mathcal { H } = [ k _ { \mathrm { p e a k } } , k _ { \mathrm { h o l d } } ) ;$ ; and the cooling phase $\mathbf { \dot { \mathcal { C } } } = [ k _ { \mathrm { h o l d } } , T )$

For the multi-modal failure modes, we additionally extract modality-specific indices, namely the temperature peak and the ultrasonic minima,

$$
k _ { \mathrm { m a x } } ^ { \mathrm { ( T ) } } = \arg \operatorname* { m a x } _ { k } x _ { k } ^ { \mathrm { ( T ) } } ,\tag{9}
$$

$$
k _ { \mathrm { m i n } } ^ { ( c ) } = \arg \operatorname* { m i n } _ { k } x _ { k } ^ { ( c ) } , \qquad c \in \{ \mathrm { U S } 1 , \mathrm { U S } 4 \} .\tag{10}
$$

For the reference cycle, the markers take the following values, with times measured from $t _ { \mathrm { s t a r t } } .$

<table><tr><td></td><td> $k _ { \mathrm { f i l l } }$ </td><td> $k _ { \mathrm { p e a k } }$ </td><td> $k _ { \mathrm { h o l d } }$ </td><td> $k _ { \mathrm { m a x } } ^ { \mathrm { ( T ) } }$ </td><td> $k _ { \mathrm { m i n } } ^ { \mathrm { ( U S 1 ) } }$ </td><td> $k _ { \mathrm { m i n } } ^ { \mathrm { ( U S 4 ) } }$ </td></tr><tr><td>Index</td><td>25</td><td>28</td><td>114</td><td>33</td><td>28</td><td>144</td></tr><tr><td>Time [s]</td><td>3.3</td><td>3.7</td><td>15.2</td><td>4.4</td><td>3.7</td><td>19.3</td></tr></table>

4) Single-Modal Data: Part quality, and thus the decision to accept a part, depends strongly on the process parameters, among them the cavity pressure and its evolution throughout the cycle [34], [35].

OK disturbances (Fig. 3(a)). For each cycle, we apply one to three distinct types from Table I, chosen at random.

Time shift (Fig. 3(b)). We delay or advance the pressure profile by warping the normalized time axis $u _ { k } = k / ( T - 1 )$ while the overall signal shape stays intact. We move each sample from $u _ { k }$ to

$$
\nu _ { k } = u _ { k } + \alpha \sin ( \pi u _ { k } ) ,\tag{11}
$$

TABLE I  
OK DISTURBANCES
<table><tr><td>Type</td><td>Operation</td><td>Range</td></tr><tr><td>Gaussian noise</td><td> $x _ { k } ^ { ( c ) } + \mathcal { N } ( 0 , \sigma _ { \mathrm { n } } ^ { 2 } )$ </td><td> $\sigma _ { \mathrm { n } } \in \left[ 0 . 0 0 2 , 0 . 0 1 \right]$ </td></tr><tr><td>Lead or lag</td><td> $x _ { k - \ell } ^ { ( c ) }$ </td><td> $| \ell | \leq 2$ </td></tr><tr><td>Gain</td><td> $f x _ { k } ^ { ( c ) }$ </td><td> $f \in [ 0 . 9 8 , 1 . 0 2 ]$ </td></tr><tr><td>Linear trend</td><td> $x _ { k } ^ { ( c ) } \pm m T k / ( T - 1 )$ </td><td> $m \in [ 2 \cdot 1 0 ^ { - 5 } , 1 0 ^ { - 4 } ]$ </td></tr><tr><td>Moving average</td><td> $\begin{array} { r l } { ~ } & { { } \frac { 1 } { w } \sum _ { j = 0 } ^ { w - 1 } x _ { k - 1 + j } ^ { ( c ) } } \end{array}$ </td><td> $w \in \{ 2 , 3 \}$ </td></tr></table>

Benign process variations applied to every channel $x ^ { ( c ) }$ of every synthetic cycle. One to three types are drawn per cycle, with parameters from the given ranges. Lead or lag and moving average pad with the edge value.

with shift fraction $| \alpha | \in [ 0 . 0 3 , 0 . 0 7 ]$ of random sign, and linearly interpolate the points $( \nu _ { k } , p _ { k } )$ back onto the grid $\left\{ u _ { k } \right\}$ . The warp keeps the cycle start and end fixed; $\alpha > 0$ delays events and $\alpha < 0$ advances them. The timing of the process state points, especially the end of the packing and cooling stages, is known to strongly influence part weight [34], [36].

Underpacking/overpacking (Fig. 3(c)). We scale the pressure over the filling and holding phases,

$$
\tilde { p } _ { k } = \phi p _ { k } , \qquad k \in \mathcal { F } \cup \mathcal { H } ,\tag{12}
$$

with φ drawn from [0.80,0.92] (underpacking) with probability 0.6 and from [1.08,1.20] (overpacking) otherwise. Such variations in the packing pressure profile are known to drive defects like short shots or flash [36], [37].

Fast gate freeze (Fig. 3(d)). We decay the pressure exponentially over the holding phase,

$$
\tilde { p } _ { k } = p _ { k } \exp \left( - \delta \frac { k - k _ { \mathrm { p e a k } } } { | \mathcal { H } | } \right) , \qquad k \in \mathcal { H } ,\tag{13}
$$

with decay rate $\delta \in [ 1 , 2 . 5 ]$ and $| \mathcal { H } | = k _ { \mathrm { h o l d } } - k _ { \mathrm { p e a k } }$ . A fast pressure decay after the peak mimics premature gate freeze, since holding pressure can no longer push material into the cavity once the gate has frozen [38].

Secondary pressure rise (Fig. 3(e)). We add a Gaussianshaped pressure bump after an onset sample $k _ { 0 } = \lfloor \xi T \rfloor$ with $\xi \in [ 0 . 4 0 , 0 . 6 5 ]$

$$
\tilde { p } _ { k } = p _ { k } + A \exp \left( - \frac { ( k - k _ { 0 } - 0 . 3 T _ { \mathrm { r } } ) ^ { 2 } } { 2 ( 0 . 1 5 T _ { \mathrm { r } } ) ^ { 2 } } \right) , \qquad k \geq k _ { 0 } ,\tag{14}
$$

where $T _ { \mathrm { r } } = T - k _ { 0 }$ is the number of remaining samples and $A \in [ 0 . 0 3 , 0 . 1 0 ]$ is the amplitude.

Machine drift (Fig. 3(f)). We model drift with a drift fraction $d \in [ 0 . 1 0 , 0 . 3 5 ]$ that reduces the holding pressure,

$$
\tilde { p } _ { k } = \left( 1 - 0 . 4 d \right) p _ { k } , \qquad k \in \mathcal { H } ,\tag{15}
$$

and adds broadband noise $\pmb { \varepsilon } _ { k } \sim \mathcal { N } \big ( 0 , ( 0 . 0 3 d ) ^ { 2 } \big )$ to every sample. Ambient conditions and machine wear are known to cause such a drift of the process data during production [18].

5) Multi-Modal Data: For multi-modal failure modes, we keep the underpacking/overpacking, fast gate freeze, and secondary pressure rise of the single-modal setting (Section III-A4) unchanged (Fig. 4(c)–(e)), adapt the time shift and machine drift modes, and introduce three modes that change only the additional modalities. The OK disturbances act on every channel $x ^ { ( c ) }$ (Table I).

![](images/345a500b240698b93ae982f69c2afe6918a73a7969dcbc86744768c3b18ee323.jpg)  
Fig. 3. Synthetic single-modal data, one panel per class: (a) OK; (b) time shift, which moves the events later or earlier; (c) underpacking/overpacking, which scales the pressure over the filling and holding phases; (d) fast gate freeze, which decays the pressure over the holding phase; (e) secondary pressure rise, which adds a bump in the cooling phase; and (f) machine drift, which lowers the holding pressure and adds noise. Each panel shows five cycles of its class, OK in blue and failure modes in orange, over the real reference in black.

Time shift (Fig. 4(b)). We apply the warp (11) with the same α to every channel $x ^ { ( c ) }$ . Since this failure mode changes the phase timing of the process, all modalities reflect it.

Machine drift (Fig. 4(i)). In addition to the pressure drift (15), we raise the temperature baseline, $\tilde { x } _ { k } ^ { ( \mathrm { T } ) } = \dot { x } _ { k } ^ { ( \mathrm { T } ) } + 0 . 1 5 d .$ , and the broadband noise $\varepsilon _ { k }$ now also perturbs the temperature and ultrasonic channels. The two perturbations are causally independent, and the temperature rise is a modeling choice. Elevated melt temperature is known to promote material degradation and excessive post-cooling shrinkage [38].

Inter-sensor inconsistency $( \mathrm { F i g . 4 ( g ) } )$ . We rescale both ultrasonic RMS channels around their pre-fill level $x _ { 0 } ^ { ( c ) }$

$$
\tilde { x } _ { k } ^ { ( c ) } = x _ { 0 } ^ { ( c ) } + \gamma _ { c } \big ( x _ { k } ^ { ( c ) } - x _ { 0 } ^ { ( c ) } \big ) , \qquad c \in \{ \mathrm { U S } 1 , \mathrm { U S } 4 \} ,\tag{16}
$$

where one factor is drawn from [0.75, 0.95] and the other from [1.05,1.25], assigned to the sensors at random, so that one sensor is attenuated while the other is amplified over the entire cycle. We use this mode to model a difference in contact between the two sensor locations, since ultrasonic amplitude reflects the contact between the cavity wall and the polymer [1]. Such a difference can indicate uneven shrinkage, which is known to cause warpage [38].

Distorted ultrasonic response (gap-formation hypothesis; Fig. 4(h)). We rescale each ultrasonic channel in two periods, split at its minimum $k _ { \mathrm { m i n } } ^ { ( c ) }$ from (10), with a dip scale $\rho _ { \mathrm { d } }$ and a recovery scale $\rho _ { \mathrm { r } }$ , both drawn from [0.60,0.90]:

$$
\begin{array} { r } { \tilde { x } _ { k } ^ { ( c ) } = \left\{ \begin{array} { l l } { x _ { 0 } ^ { ( c ) } + \rho _ { \mathrm { d } } \bigl ( x _ { k } ^ { ( c ) } - x _ { 0 } ^ { ( c ) } \bigr ) , } & { k _ { \mathrm { p e a k } } \leq k < k _ { \mathrm { m i n } } ^ { ( c ) } , } \\ { x _ { k _ { \mathrm { m i n } } ^ { ( c ) } } ^ { ( c ) } + \rho _ { \mathrm { r } } \bigl ( x _ { k } ^ { ( c ) } - x _ { k _ { \mathrm { m i n } } ^ { ( c ) } } ^ { ( c ) } \bigr ) , } & { k \geq k _ { \mathrm { m i n } } ^ { ( c ) } . } \end{array} \right. } \end{array}\tag{17}
$$

Between the pressure peak and the minimum, we pull the signal towards its pre-fill level, which reduces its drop before the minimum. From the minimum onward, we pull it towards the minimum value, which suppresses the recovery. The minimum value itself stays unchanged. For $x ^ { ( \mathrm { U S 1 } ) }$ <sup>)</sup>, the minimum coincides with the pressure peak, so only the recovery changes. For $x ^ { ( \mathrm { U S 4 } ) }$ it lies in the cooling phase, so the first period spans the holding phase and the beginning of the cooling phase. Unlike intersensor inconsistency, this mode changes both sensors in the same direction, from the pressure peak onward. We use it to model a gap forming at both sensor locations, since changes in ultrasonic amplitude and in the reflection and transmission at the wall reflect gap formation and part detachment [1].

Temperature dynamics (Fig. 4(f)). First, we shift the whole temperature channel, $\check { x } _ { k } ^ { ( \mathrm { T } ) } = \check { x } _ { k } ^ { ( \mathrm { T } ) } + b ,$ , with $| b | \in [ 0 . 0 2 , 0 . 0 6 ]$ of random sign, to emulate a changed bulk melt temperature [39], where $b > 0$ raises the risk of material degradation and shrinkage [38]. Second, we warp the cooling segment from the temperature peak $k _ { \mathrm { m a x } } ^ { \mathrm { ( T ) } }$ from (9) to the end of the cycle. With $T _ { \mathrm { c } } = T - k _ { \mathrm { m a x } } ^ { \mathrm { ( T ) } }$ and $u _ { j } = j / ( T _ { \mathrm { c } } - 1 ) , \ j = 0 , \ldots , T _ { \mathrm { c } } - 1$ , the segment becomes

$$
\begin{array} { r } { \tilde { x } _ { k _ { \mathrm { m a x } } ^ { ( \mathrm { T } ) } + j } ^ { ( \mathrm { T } ) } = \check { x } ^ { ( \mathrm { T } ) } \big ( u _ { j } ^ { 1 / \kappa } \big ) , } \end{array}\tag{18}
$$

where $\check { x } ^ { ( \mathrm { T } ) } ( u )$ is the linear interpolation of the shifted segment at normalized time $u \in [ 0 , 1 ]$ . The peak and the last value stay in place, and only the temperature between them changes: κ is drawn with equal probability from [1.1,1.4], where the temperature falls earlier (faster cooling), or from [0.7, 0.9], where it falls later (slower cooling). Cooling rate is known to influence crystallization kinetics and crystal size in injectionmoulded polymers [40].

## B. MouldIt Model

We formulate MouldIt as a variational autoencoder, with Q additional prediction heads, one per known class, projected directly from the encoder embedding (Fig. 5). The encoder comprises two temporal convolution blocks to capture local features, followed by a temporal attention block to model long-range dependencies between the captured features. We flatten the result and pass it through a linear layer followed by two linear heads to predict the $\mu$ and $\log \sigma ^ { 2 }$ of the latent distribution. After resampling, we pass the latent sample in parallel to two different network blocks. First, a block of Q linear heads projects the latent sample to the probabilities of OK and each NOK variant. Second, a decoder mirrors the encoder architecture in reverse, with a small variation in the channel widths of the temporal convolution blocks.

MouldIt optimizes a loss composed of three components (19): a reconstruction MSE between the input signal and the reconstructed signal, a KLD regularizer between the approximate posterior and a standard normal prior, and a BCE between the ground-truth part labels and their predicted probabilities.

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { r e c } } \left( \mathcal { L } _ { \mathrm { M S E } } + \beta ( e ) \mathcal { L } _ { \mathrm { K L D } } \right) + \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { B C E } } , } \end{array}\tag{19}
$$

where e is the training epoch. The first two loss terms together are weighted against the BCE $( \lambda _ { \mathrm { { r e c } } } = 1$ and $\lambda _ { \mathrm { c l s } } = 0 . 1 )$ .

For a batch of B signals $\mathbf { x } _ { i } \in \mathbb { R } ^ { C \times T }$ with $C \in \{ 1 , 4 \}$ channels and sequence length T (Section III-A2), and their reconstructions $\hat { \mathbf { x } } _ { i } ,$ the reconstruction MSE is

$$
\mathcal { L } _ { \mathrm { M S E } } = \frac { 1 } { B C T } \sum _ { i = 1 } ^ { B } \sum _ { c = 1 } ^ { C } \sum _ { t = 1 } ^ { T } \left( x _ { i , c , t } - \hat { x } _ { i , c , t } \right) ^ { 2 } .\tag{20}
$$

![](images/517f2b78fdca25f63b5e4288dbd2615dabc1fb526d8e241ee8aa8b7790578cce.jpg)  
Time [s] (each channel min-max normalised)  
P1, T3: Kistler pressure sensor 1 and temperature sensor 3; US1, US4: ultrasound sensors 1 and 4

Real reference Synthetic OK (5 samples per panel) Synthetic failure mode (5 samples per panel) Unchanged channel

Fig. 4. Synthetic multi-modal data, one column per class: (a) OK, (b) time shift, (c) underpacking/overpacking, (d) fast gate freeze, (e) secondary pressure rise, (f) temperature dynamics, (g) inter-sensor inconsistency, (h) distorted ultrasonic response, and (i) machine drift. Rows show the pressure P1, temperature T3, and ultrasound US1 and US4 channels. Time shift and machine drift act on all channels, modes (c)–(e) only on the pressure, (f) only on the temperature, and (g) and (h) only on the ultrasound. Each panel shows five synthetic cycles, OK in blue and failure modes in orange, over the real reference in black. Hatched, faded panels mark channels that the failure mode leaves unchanged; these differ from the reference only by the OK disturbances applied to every cycle.

The approximate posterior is a diagonal Gaussian $q ( \mathbf { z } \mid \mathbf { x } _ { i } ) =$ ${ \mathcal { N } } \big ( \mu _ { i } , \mathrm { d i a g } ( \sigma _ { i } ^ { 2 } ) \big )$ over $D = 3 2$ dimensions, where $\mu _ { i }$ and log $\sigma _ { i } ^ { 2 }$ are predicted by the encoder, and its KLD against the standard normal prior $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ is

$$
\mathcal { L } _ { \mathrm { K L D } } = - \frac { 1 } { 2 B } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { D } \left( 1 + \log \sigma _ { i , j } ^ { 2 } - \mu _ { i , j } ^ { 2 } - \sigma _ { i , j } ^ { 2 } \right) .\tag{21}
$$

The KLD is weighted by β(e): $\beta ( e )$

$$
\beta ( e ) = \beta \operatorname * { m i n } \biggl ( 1 , \frac { e } { 1 0 0 } \biggr ) , \quad e = 0 , 1 , 2 , . . .\tag{22}
$$

We set $\beta = 2 \sigma _ { \mathrm { O K } } ^ { 2 } / ( C T )$ , where CT is the number of values in one cycle and $\sigma _ { \mathrm { O K } }$ is the root mean square deviation of synthetic OK cycles from the reference cycle, estimated once on 300 OK cycles. With this choice, the first two loss terms match, up to a constant factor, those of a VAE whose Gaussian decoder assumes the noise variance $\sigma _ { \mathrm { O K } } ^ { 2 }$ . With $\sigma _ { \mathrm { O K } } = 0 . 0 3 0 4$ and 0.0253, this gives $\beta = 5 . 4 \times 1 0 ^ { - 6 }$ in the single-modal and $\beta = 9 . 4 \times 1 0 ^ { - 7 }$ in the multi-modal setting, which we fix for all seeds and dataset sizes.

Each of the Q heads predicts one class in a one-vs-rest manner. The BCE between the one-hot labels $y _ { i q } \in \{ 0 , 1 \}$ and the predicted class probabilities $\hat { y } _ { i q } \in ( 0 , 1 )$ is

$$
\mathcal { L } _ { \mathrm { B C E } } = - \frac { 1 } { B Q } \sum _ { i = 1 } ^ { B } \sum _ { q = 1 } ^ { Q } \left[ y _ { i q } \log \hat { y } _ { i q } + \left( 1 - y _ { i q } \right) \log \left( 1 - \hat { y } _ { i q } \right) \right] ,\tag{23}
$$

and the predicted class is the one with the highest probability. The part quality prediction problem thus maps $\mathbb { R } ^ { \hat { C } \times T } \to \mathbb { R } ^ { \bar { Q } }$ while the reconstruction returns the same input space.

## C. Novelty Score

To detect unknown faults, we compute a novelty score from the outputs of the trained MouldIt model, without changing the network. For every cycle x, it combines a latent term, from its position in the latent space, and a reconstruction term, from its reconstruction error. For the latent term, we fit one Gaussian per known class $q$ to the training cycles in the latent space, each represented by its predicted mean $\mu ,$ using Ledoit–Wolf shrinkage [41]. With $M _ { q } ( \mathbf { x } )$ the squared Mahalanobis distance of $\mu$ to class q, and $\eta _ { q }$ its 95th percentile on the validation cycles of that class, the latent term is the distance to the nearest class in units of its 95% boundary,

$$
L ( \mathbf { x } ) = \operatorname* { m i n } _ { q } \frac { M _ { q } ( \mathbf { x } ) } { \eta _ { q } } .\tag{24}
$$

For the reconstruction term, we use the summed squared error $R ( \mathbf { x } ) = \| \mathbf { x } - \hat { \mathbf { x } } \| ^ { 2 }$ , with $\eta _ { R }$ its 95th percentile on all known validation cycles. The novelty score is

$$
N ( \mathbf { x } ) = \operatorname* { m a x } \left( L ( \mathbf { x } ) , \ { \frac { R ( \mathbf { x } ) } { \eta _ { R } } } \right) .\tag{25}
$$

At inference, we use $\mu$ instead of a sampled $\mathbf { z } ,$ so the score is deterministic. PCA uses the same score with 32 components, the size of the MouldIt latent space: L is computed on the PCA scores (Hotelling’s $T ^ { 2 }$ per class) and R is the squared prediction error (SPE). PCA predicts the class with the smallest $M _ { q } / \eta _ { q }$ . The remaining baselines have a single score: the mean distance to the five nearest standardized cycles for kNN, and one minus the highest class probability for logistic regression and random forests (200 trees). Every method flags a cycle as unknown when its score exceeds Θ, the 95th percentile of the score on the known validation cycles.

## IV. EXPERIMENTS

We compare MouldIt with classical methods on two tasks: classifying cycles into OK or one of the known failure modes, and detecting unknown failure modes that were not part of training. We are particularly interested in how performance depends on the amount of training data. For both the singlemodal (pressure) and the multi-modal (pressure, temperature, ultrasound) setting, we therefore generate datasets of 50, 100, 200, 500, 1000, 2000, 4000, 8000, and 16000 cycles from the real reference cycle (Section III-A). All classes are equally represented, with six classes in the single-modal and nine in the multi-modal setting, so each size is rounded down to a multiple of the number of classes, for example to 15996 and 15993 cycles at 16000. We split each dataset into 60% training, 20% validation, and 20% test data, balanced by class.

![](images/718212dcb4301b4e8279ac9bcc6ccd655b7e8e5e22f7e2ca68633776bdca1e2e.jpg)  
Fig. 5. MouldIt architecture. The input x has C channels of length $T = 3 4 0 ,$ with $C = 1$ in the single-modal and $C = 4$ in the multi-modal setting. The encoder consists of two TCN blocks, a temporal attention block, and a flatten and a linear layer, and predicts the mean µ and log variance log $\sigma ^ { 2 }$ of the 32-dimensional latent distribution $q ( \mathbf { z } \mid \mathbf { x } ) . \mathrm { A }$ latent sample $\mathbf { z } = \boldsymbol { \mu } + \boldsymbol { \sigma } \odot \boldsymbol { \varepsilon }$ , with $\varepsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , feeds two branches: Q linear heads predict the part label yˆ (OK or a NOK variant), and a decoder of two linear layers, a temporal attention block, and two TCN blocks reconstructs the signal xˆ. Dashed lines mark the three loss terms: the MSE between x and xˆ, the β-weighted KLD between q and the prior p(z), and the BCE between y and ˆy. At inference, µ replaces z.

MouldIt (Section III-B) is trained for 1000 epochs with cosine annealing from a learning rate of 0.001. As baselines, we use k-nearest neighbors (kNN), logistic regression, random forests, principal component analysis (PCA), and a trivial reference that always predicts OK. We run all of them on our data, including methods used in prior IM studies, such as random forests [6] and kNN [28]. We do not compare numerically with the published IM models of Section II: they report results on other parts, sensors, and failure modes, and, to our knowledge, the closest one [18] has no public implementation. Every run is repeated with three seeds, and we report the mean ± standard deviation.

Classification. Here, all classes are seen during training, and we report the balanced accuracy over all classes.

Unknown detection. Setup: We hold out one failure mode at a time from training and validation. All of its cycles, the unknown faults, are added to the test set.

Novelty score: Each method assigns a novelty score to every cycle (Section III-C). The threshold Θ corresponds to a 5% false positive rate (FPR) on the known validation cycles.

Metrics: We report the unknown AUROC of the novelty score and the unknown detection at 5% FPR, i.e., the share of unknown faults that are flagged. Both are averaged over the held-out failure modes.

Sorting losses. A part is shipped only if it is not flagged and is predicted OK. This decision can go wrong in four ways: good parts rejected, known faults shipped, unknown faults shipped, and unknown faults rejected as a known fault. We call the mean of these four rates the sorting loss, report it over dataset size, and rank the methods at 16000 cycles. Besides the baselines above, we evaluate random forests with the PCA unknown flag, which ships a part only if random forests predict OK and PCA does not flag it.

## A. Single-Modal Experiments

1) Classification: With little training data, random forests are clearly the most accurate classifier, reaching a balanced accuracy of $8 6 . 6 \pm 2 . 1 \%$ at a size of 100 cycles (Fig. 6(a)). MouldIt’s balanced accuracy becomes consistent across seeds from a size of 1000, and from 4000 onward it is higher than that of random forests. At 16000, MouldIt reaches $9 9 . 1 \pm$ 0.5% compared with $9 7 . 9 \pm 0 . 3 \%$ for random forests. None of the other baselines exceeds 88%.

2) Unknown Detection: When training data is scarce, PCA is the best detector. MouldIt overtakes it in unknown AUROC from a size of 1000 (Fig. 6(b)). In unknown detection, it ties with PCA at 2000 and leads from 4000 onward (Fig. 6(c)). At 16000, MouldIt reaches an unknown AUROC of $0 . 9 3 7 \pm 0 . 0 1 6$ and detects $8 0 . 6 \pm 4 . 2 \%$ of the unknown faults at 5% FPR. PCA follows with $0 . 9 2 1 \pm 0 . 0 0 4$ and $7 3 . 5 \pm 0 . 4 \%$ . Machine drift is the hardest failure mode for MouldIt, with an unknown AUROC of 0.823, since it deviates from the reference only slightly more than normal OK cycles do. For PCA, the hardest failure mode is underpacking/overpacking, with 0.766.

3) Sorting Losses: With little training data, random forests with the PCA unknown flag sort parts best, with a sorting loss of $1 2 . 5 \pm 3 . 2 \%$ at a size of 200 (Fig. 7). MouldIt catches up at 2000 and has the lowest sorting loss from 4000 onward. At 16000, its sorting loss is 6.0 ± 0.7% (Fig. 7(e)), followed by random forests with the PCA unknown flag with $7 . 5 \pm$ 0.1% and kNN with $1 0 . 4 \pm 0 . 3 \%$ . The remaining methods have sorting losses between 18.8% and 36.2%, and the reference has 50.0%. Most of MouldIt’s sorting loss comes from unknown faults rejected as a known fault $( 1 7 . 7 \pm 4 . 2 \%$ Fig. 7(d)); its other three rates stay below 4% (Fig. 7(a)–(c)). PCA rejects $6 7 . 1 \pm 1 1 . 1 \%$ of the good parts (Fig. 7(a)), and random forests ship $2 7 . 3 \pm 2 . 0 \%$ of the unknown faults (Fig. 7(c)).

![](images/d61183dd80339303f20bb0564fc82b3376bb7271cd5393ad98f3c6afa36d4f0f.jpg)

![](images/cc52e2fbab3fa63e36f755f0816eafdf2b3908d6b0b23f7caf97ab8c8a47ec54.jpg)

![](images/8ebc805363f9928c06a48b2d04fb9c8fd0d357dcc7a7e07b0250759c1d9f1960.jpg)  
Fig. 6. Single-modal results over dataset size: (a) balanced accuracy over all six classes, and (b) unknown AUROC and (c) unknown detection at a 5% false positive rate, both for failure modes held out from training and averaged over the five modes. Lines show the mean over three seeds and the band the standard deviation of MouldIt; the dotted line and value mark MouldIt at 16000 cycles. The dashed grey line is the reference that accepts every part, with a chance-level accuracy of 1/6. Random forests with the PCA unknown flag are not shown, since they have no classifier or score of their own. Dashed circles mark the size from which MouldIt stays stable across seeds (standard deviation below 0.05), solid circles the size from which it is also the best method, and the shaded region the sizes where it is stable and the best in all three metrics, here from 4000 cycles.

## B. Multi-Modal Experiments

1) Classification: With multi-modal data, MouldIt reaches a balanced accuracy at least as high as every baseline at all dataset sizes (Fig. 8(a)). From a size of 2000 onward, it classifies every test cycle correctly in all seeds. At 16000, random forests and kNN come close with $9 9 . 9 \pm 0 . 1 \%$ and $9 9 . 4 \pm 0 . 3 \%$ , whereas PCA and logistic regression stay below 94%.

2) Unknown Detection: MouldIt has the highest unknown detection at every dataset size (Fig. 8(c)). Already at a size of 200, it flags $9 0 . 7 \pm 7 . 4 \%$ of the unknown faults, compared with $7 2 . 3 \pm 5 . 1 \%$ for PCA. From 8000 onward, it flags all unknown faults in every seed. At 16000, PCA comes closest with an unknown AUROC of 0.995 ± 0.000 (Fig. 8(b)) and an unknown detection of $9 7 . 8 ~ \pm ~ 0 . 2 \%$ . None of the other baselines detects more than 80%.

3) Sorting Losses: MouldIt has the lowest sorting loss at every dataset size (Fig. 9). At a size of 200, its sorting loss is $4 . 3 \pm 1 . 9 \%$ , roughly half that of random forests with the PCA unknown flag $( 8 . 1 \pm 1 . 3 \% )$ . At 16000, MouldIt’s sorting loss of $1 . 0 \pm 0 . 3 \%$ (Fig. 9(e)) is on par with that of random forests with the PCA unknown flag $( 1 . 1 \pm 0 . 1 \% )$ . The other baselines have higher sorting losses: $5 . 5 \pm 0 . 3 \%$ for kNN, $1 0 . 4 \pm 0 . 4 \%$ for random forests, $2 3 . 8 \pm 2 . 0 \%$ for PCA, and $4 2 . 5 \pm 0 . 3 \%$ for logistic regression. MouldIt’s only errors are good parts rejected $( 3 . 9 \pm 1 . 2 \%$ ; Fig. 9(a)).

## V. DISCUSSION

We discuss the data efficiency and multi-modal benefit of MouldIt, how it detects unknown faults, how well it reconstructs them, and the limitations of our study.

## A. Data Efficiency and Multi-Modal Benefit

Comparing the two settings shows that additional modalities reduce the data MouldIt needs. With pressure alone (Figs. 6 and 7), it has not fully saturated even at 16000 cycles: between 4000 and 16000 cycles, its unknown detection still rises by 4.4 percentage points and its sorting loss falls by 1.2. With all four channels (Fig. 9), its sorting loss no longer improves by more than the seed variation from 4000 cycles. The benefit is largest with little data: at 200 cycles, the multi-modal sorting loss is 4.5 times lower than the single-modal one.

The two settings are not the same problem with more sensors. The multi-modal setting has more classes, and three of its failure modes exist only in the added channels (Section III-A5). The classical methods also profit from these channels (Table II), so the benefit is not specific to MouldIt. What sets MouldIt apart is that, with all four channels, it needs less data for a low sorting loss: it falls below 5% at 200 cycles, whereas random forests with the PCA unknown flag need 1000 (Fig. 9). In addition, one jointly trained model does the work of two separate ones. MouldIt does not lead on every sorting error, but it has the lowest overall sorting loss (Table II).

## B. Latent Analysis

To understand these results, we split the novelty score (25) of every fold into its latent and reconstruction terms, for MouldIt and PCA (Table III, Fig. 10).

![](images/b2127793b2a3d93ab282b3f30677f279dcad78d19546b73295eec4994751b979.jpg)

![](images/90668c556f1efb2fa67266c6b26852eef07f84435d2349d802912c6cc367233a.jpg)  
(d) Unknown faults rejected as a known fault ↓

(c) Unknown faults shipped ↓  
![](images/5234ff88529b3346734baf01bb8c3d6768cbd524990d44d239311e53529824e3.jpg)

![](images/df3f6bb58c989885499d79489a86ad8c834208770f44503a2fc31a79cc891ee8.jpg)

![](images/db1bfed7ce530eb0dddd0d146edc88bd45e814b9afd072c4cc8baa7362871b39.jpg)  
$: \Sigma ^ { \prime }$ MouldIt is stable from here on (seed std < 5 %) MouldIt is stable and is the best from here on  
Fig. 7. Single-modal sorting errors over dataset size: (a) good parts rejected, (b) known faults shipped, (c) unknown faults shipped, and (d) unknown faults rejected as a known fault, where a part is shipped only if it is predicted OK and not flagged as unknown. Lines show the mean over three seeds and the band the standard deviation of MouldIt. The reference accepts every part, so it rejects no good part and ships every fault. Dashed circles mark the size from which MouldIt stays stable across seeds (standard deviation below 5%), and solid circles the size from which it is also the best method. Panel (e) ranks the methods by their sorting loss, the mean of the four rates, at 16000 cycles.

TABLE II  
RESULTS AT 16000 CYCLES
<table><tr><td rowspan="2">Method</td><td colspan="3">Higher is better</td><td colspan="5">Lower is better</td></tr><tr><td>Balanced accuracy</td><td>Unknown AUROC</td><td>Unknown detection</td><td>Good parts rejected</td><td>Known faults shipped</td><td>Unknown faults shipped</td><td>Unknown rejected as known</td><td>Sorting loss</td></tr><tr><td colspan="9">Single-modal (pressure; 6 classes)</td></tr><tr><td>MouldIt</td><td>99.1 ± 0.5</td><td> ${ \bf 0 . 9 3 7 \pm 0 . 0 1 6 }$ </td><td> ${ \bf 8 0 . 6 \pm 4 . 2 }$ </td><td> $3 . 4 \pm 1 . 3$ </td><td> $1 . 2 \pm 2 . 0$ </td><td> $1 . 7 \pm 0 . 2$ </td><td> $1 7 . 7 \pm 4 . 2$ </td><td> ${ \bf 6 . 0 \pm 0 . 7 }$ </td></tr><tr><td>RF + PCA flag</td><td></td><td></td><td></td><td> $3 . 2 \pm 0 . 7$ </td><td> $0 . 1 \pm 0 . 1$ </td><td> $5 . 5 \pm 0 . 7$ </td><td> $2 1 . 0 \pm 0 . 4$ </td><td> $7 . 5 \pm 0 . 1$ </td></tr><tr><td>kNN</td><td> $8 5 . 1 \pm 0 . 1$ </td><td> $0 . 8 9 7 \pm 0 . 0 0 1$ </td><td> $6 4 . 1 \pm 0 . 6$ </td><td> $2 . 2 \pm 1 . 3$ </td><td> $3 . 4 \pm 0 . 2$ </td><td> $1 3 . 1 \pm 0 . 4$ </td><td> $2 2 . 8 \pm 0 . 2$ </td><td> $1 0 . 4 \pm 0 . 3$ </td></tr><tr><td>Random forest</td><td> $9 7 . 9 \pm 0 . 3$ </td><td> $0 . 7 9 4 \pm 0 . 0 1 1$ </td><td> $3 0 . 1 \pm 1 . 3$ </td><td> $5 . 3 \pm 0 . 3$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> $2 7 . 3 \pm 2 . 0$ </td><td> $4 2 . 6 \pm 0 . 8$ </td><td> $1 8 . 8 \pm 0 . 3$ </td></tr><tr><td>PCA</td><td> $8 7 . 0 \pm 1 . 6$ </td><td> $0 . 9 2 1 \pm 0 . 0 0 4$ </td><td> $7 3 . 5 \pm 0 . 4$ </td><td> $6 7 . 1 \pm 1 1 . 1$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 0 . 4 } \pm { \bf 0 . 3 }$ </td><td> $2 6 . 2 \pm 0 . 7$ </td><td> $2 3 . 4 \pm 2 . 9$ </td></tr><tr><td>Logistic regression</td><td> $8 3 . 8 \pm 0 . 9$ </td><td> $0 . 3 4 8 \pm 0 . 0 1 0$ </td><td> $1 . 5 \pm 0 . 4$ </td><td> $3 7 . 3 \pm 2 . 2$ </td><td> $8 . 7 \pm 0 . 2$ </td><td> $2 5 . 7 \pm 4 . 5$ </td><td> $7 2 . 9 \pm 4 . 1$ </td><td> $3 6 . 2 \pm 0 . 5$ </td></tr><tr><td>Reference (always OK)</td><td> $1 6 . 7 \pm 0 . 0$ </td><td> $0 . 5 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td></tr><tr><td colspan="9">Multi-modal (pressure, temperature, two ultrasound channels; 9 classes)</td></tr><tr><td>MouldIt</td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 1 . 0 0 0 \ : \pm { \ : 0 . 0 0 0 } }$ </td><td> ${ \bf 1 0 0 . 0 \ : \pm { \ : 0 . 0 } }$ </td><td> $3 . 9 \pm 1 . 2$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 1 . 0 \pm 0 . 3 }$ </td></tr><tr><td>RF + PCA flag</td><td></td><td></td><td></td><td> $1 . 9 \pm 0 . 4$ </td><td> $0 . 1 \pm 0 . 1$ </td><td> $2 . 2 \pm 0 . 2$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> $1 . 1 \pm 0 . 1$ </td></tr><tr><td>kNN</td><td> $9 9 . 4 \pm 0 . 3$ </td><td> $0 . 9 3 8 \pm 0 . 0 0 2$ </td><td> $7 9 . 9 \pm 0 . 4$ </td><td> ${ \bf 1 . 1 \pm 0 . 7 }$ </td><td> $0 . 7 \pm 0 . 2$ </td><td> $1 5 . 1 \pm 0 . 2$ </td><td> $5 . 0 \pm 0 . 5$ </td><td> $5 . 5 \pm 0 . 3$ </td></tr><tr><td>Random forest</td><td> $9 9 . 9 \pm 0 . 1$ </td><td> $0 . 9 2 0 \pm 0 . 0 0 8$ </td><td> $7 2 . 2 \pm 1 . 1$ </td><td> $1 3 . 6 \pm 0 . 7$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> $1 5 . 5 \pm 0 . 5$ </td><td> $1 2 . 3 \pm 0 . 6$ </td><td> $1 0 . 4 \pm 0 . 4$ </td></tr><tr><td>PCA</td><td> $9 3 . 0 \pm 4 . 8$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 0$ </td><td> $9 7 . 8 \pm 0 . 2$ </td><td> $9 3 . 2 \pm 8 . 3$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> $2 . 2 \pm 0 . 2$ </td><td> $2 3 . 8 \pm 2 . 0$ </td></tr><tr><td>Logistic regression</td><td> $8 0 . 8 \pm 0 . 8$ </td><td> $0 . 5 4 4 \pm 0 . 0 5 7$ </td><td> $0 . 1 \pm 0 . 0$ </td><td> $6 2 . 5 \pm 2 . 2$ </td><td> $7 . 5 \pm 0 . 9$ </td><td> $1 9 . 9 \pm 4 . 9$ </td><td> $8 0 . 0 \pm 4 . 9$ </td><td> $4 2 . 5 \pm 0 . 3$ </td></tr><tr><td>Reference (always OK)</td><td> $1 1 . 1 \pm 0 . 0$ </td><td> $0 . 5 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $5 0 . 0 \pm 0 . 0$ </td></tr></table>

Values are in percent, except for the AUROC, and are given as mean ± standard deviation over three seeds. Unknown metrics are first averaged over the held-out failure modes. Unknown detection is at a 5% false positive rate. The sorting loss is the mean of the four sorting errors. RF + PCA flag is random forests with the PCA unknown flag; it has no classifier or score of its own, so it only has sorting errors. Bold marks the best method per column, excluding the reference.

![](images/77426838565bc4eabc14ffec725cd0b3a4e1ba499d80c0e7740a8a36c4da7f07.jpg)

![](images/fe417a25d590d4a0a655e9a8c6849de4ef483856274b5f97640b85b98bfd8514.jpg)  
() MouldIt is stable from here on (seed std < 0.05) MouldIt is stable and is the best from here on

![](images/a4e08d7be41a8bd01a40e6a2772983317f1fd4b04b49dfe69009dbbb0d8030f9.jpg)  
MouldIt is stable and is the best in accuracy, AUROC and detection

Fig. 8. Multi-modal results over dataset size: (a) balanced accuracy over all nine classes, and (b) unknown AUROC and (c) unknown detection at a 5% false positive rate, both for failure modes held out from training and averaged over the eight modes. Lines show the mean over three seeds and the band the standard deviation of MouldIt; the dotted line and value mark MouldIt at 16000 cycles. The dashed grey line is the reference that accepts every part, with a chance-level accuracy of 1/9. Random forests with the PCA unknown flag are not shown, since they have no classifier or score of their own. Dashed circles mark the size from which MouldIt stays stable across seeds (standard deviation below 0.05), solid circles the size from which it is also the best method, and the shaded region the sizes where it is stable and the best in all three metrics, here from 2000 cycles.  
TABLE III  
DETECTION OF UNKNOWN FAULTS AT 16000 CYCLES
<table><tr><td></td><td colspan="2">Detected</td><td colspan="2">Latent only</td><td colspan="2">Recon. only</td><td colspan="2">Both</td></tr><tr><td>Held-out mode</td><td>M</td><td>P</td><td>M</td><td>P</td><td>M</td><td>P</td><td>M</td><td>P</td></tr><tr><td>Single-modal</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Machine drift</td><td>42</td><td>39</td><td>28</td><td>19</td><td>10</td><td>3</td><td>4</td><td>17</td></tr><tr><td>Secondary pressure rise</td><td>92</td><td>95</td><td>0</td><td>0</td><td>64</td><td>68</td><td>27</td><td>27</td></tr><tr><td>Underpacking/overpacking</td><td>69</td><td>33</td><td>29</td><td>33</td><td>1</td><td>0</td><td>40</td><td>0</td></tr><tr><td>Time shift</td><td>100</td><td>100</td><td>0</td><td>0</td><td>3</td><td>0</td><td>97</td><td>100</td></tr><tr><td>Fast gate freeze</td><td>100</td><td>100</td><td>0</td><td>0</td><td>11</td><td>0</td><td>89</td><td>100</td></tr><tr><td>Multi-modal</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Secondary pressure rise</td><td>100</td><td>83</td><td>28</td><td>24</td><td>0</td><td>1</td><td>72</td><td>58</td></tr><tr><td>Underpacking/overpacking</td><td>100</td><td>100</td><td>6</td><td>100</td><td>0</td><td>0</td><td>94</td><td>0</td></tr><tr><td>Machine drift</td><td>100</td><td>100</td><td>8</td><td>83</td><td>0</td><td>0</td><td>92</td><td>17</td></tr><tr><td>Temperature dynamics</td><td>100</td><td>99</td><td>1</td><td>18</td><td>0</td><td>0</td><td>99</td><td>81</td></tr><tr><td>Inter-sensor inconsistency</td><td>100</td><td>100</td><td>0</td><td>0</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Fast gate freeze</td><td>100</td><td>100</td><td>0</td><td>0</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Distorted US response</td><td>100</td><td>100</td><td>0</td><td>0</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Time shift</td><td>100</td><td>100</td><td>0</td><td>0</td><td>0</td><td>0</td><td>100</td><td>100</td></tr></table>

Mean percentage of each held-out mode’s cycles over three seeds, hardest first. M: MouldIt, P: PCA. Each detected cycle is flagged by the latent term of the novelty score only, its reconstruction term only, or both, so the three sum to Detected, up to rounding.

1) OK and Known Faults: At 16000 cycles, only 1.0% (single-modal) and 0% (multi-modal) of the OK test cycles lie inside the 95% boundary of a known fault in the MouldIt latent space (Fig. 10(a), (c)). In the PCA space, 96.0% and 97.6% do (Fig. 10(b), (d)). In units of these boundaries, the median distance of an OK cycle to its nearest known fault is 10.1 and 25.3 in MouldIt, but only 0.17 and 0.15 in PCA. PCA components follow the largest variance in the data, whether or not it separates the classes, while MouldIt is also trained to classify from its latent space. Since PCA predicts the nearest class (Section III-C), this explains why it rejects most good parts (Figs. 7(a) and 9(a)). The separation needs data: at 200 single-modal cycles, 60.6% of the OK cycles still lie inside the boundary of a known fault in MouldIt.

2) Unknown Faults: With more data, the MouldIt latent space places unknown faults outside the known classes. The share of unknown cycles inside the boundary of a known class falls from 88.3% at 200 cycles to 33.1% at 16000 cycles with pressure alone, and from 14.4% to 0% with all channels. With less data, MouldIt therefore relies more on reconstruction error: at 1000 single-modal cycles, 54.7% of the unknown cycles are flagged by the reconstruction term alone, and at 16000 only 17.8%, as most are then flagged by both terms.

3) Single-Modal Limits: With pressure alone, underpacking/overpacking, secondary pressure rise, and machine drift stay below full detection (Table III). Underpacking/overpacking scales the pressure (Section III-A4), a change that lies almost entirely in the linear span of the PCA components. PCA therefore reconstructs this unknown fault even better than the known cycles: its median reconstruction term is 0.007, and detecting the fault from this term alone gives an unknown AUROC of 0.410. For MouldIt, the same values are 1.01 and 0.806, which makes this failure mode the main source of its single-modal lead in unknown detection. MouldIt places 66.8 ± 26.8% of the secondary pressure rise cycles inside the OK boundary (Fig. 10(a)). These cycles are caught almost only by the reconstruction term, and those below its threshold are missed (Fig. 10(e)). Machine drift stays close to the known classes, and neither term separates it well for either method.

4) Multi-Modal Gain: With all channels, MouldIt fully detects underpacking/overpacking, secondary pressure rise, and machine drift. For machine drift, the added channels carry direct evidence of the fault, since our generator raises the temperature baseline and adds noise there (Section III-A5). Secondary pressure rise and underpacking/overpacking, however, leave the added channels unchanged (Fig. 4(c), (e)). Yet the latent space now moves secondary pressure rise away from OK (Fig. 10(c), (g)). The gain on these two modes therefore cannot come from direct evidence of the fault in the added channels. The richer set of known classes or the coupling between channels may shape the latent space, but we have not tested either.

![](images/93a4314f54160f272d3948b1d71d9ea54300e806102b98ec6e07e23c5faeddac.jpg)  
(c) Unknown faults shipped ↓

![](images/599ffdb241842fbdc4a1d4cfa262f822d73093e3d594b2d214a74ff9ecad0142.jpg)  
(d) Unknown faults rejected as a known fault ↓

![](images/2773ce214c79db62fd4c7df900abadc122d9aeddd1d3d402337a1983f58232a7.jpg)

![](images/1722459b4ce3761cf72f1793df074a00a2258edade35740aa88ce37ffd24f245.jpg)

![](images/0ef914a9b4b1f041dd8b6aee27ee6ddc20ef7691c69b2af6ace60a49f07ce436.jpg)  
MouldIt is stable from here on (seed std < 5 %)  
Fig. 9. Multi-modal sorting errors over dataset size: (a) good parts rejected, (b) known faults shipped, (c) unknown faults shipped, and (d) unknown faults rejected as a known fault, where a part is shipped only if it is predicted OK and not flagged as unknown. Lines show the mean over three seeds and the band the standard deviation of MouldIt. The reference accepts every part, so it rejects no good part and ships every fault. Dashed circles mark the size from which MouldIt stays stable across seeds (standard deviation below 5%). Panel (e) ranks the methods by their sorting loss, the mean of the four rates, at 16000 cycles.

TABLE IV  
RECONSTRUCTION FIDELITY AT 16000 CYCLES
<table><tr><td></td><td colspan="2">Single-modal</td><td colspan="2">Multi-modal</td></tr><tr><td>Metric</td><td>PCA</td><td>MouldIt</td><td>PCA</td><td>MouldIt</td></tr><tr><td>Known faults</td><td></td><td></td><td></td><td></td></tr><tr><td>Deviation  $R ^ { 2 }$ </td><td> $\mathbf { 0 . 9 9 7 \mathop { \pm } { 0 . 0 0 0 } }$ </td><td> $0 . 9 9 4 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 9 5 \ : \pm 0 . 0 0 0 }$ </td><td> $0 . 9 9 4 \pm 0 . 0 0 1$ </td></tr><tr><td>DTW shape fidelity</td><td> $\mathbf { 0 . 9 8 4 } \pm \mathbf { 0 . 0 0 0 }$ </td><td> $0 . 9 7 6 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 9 8 9 } \pm \mathbf { 0 . 0 0 0 }$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 1$ </td></tr><tr><td>TDI</td><td> $\mathbf { 0 . 0 1 6 \ : \pm { \ : 0 . 0 0 0 } }$ </td><td> $0 . 0 4 9 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 0 0 1 \ : \pm { \ : 0 . 0 0 0 } }$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td></tr><tr><td>Unknown fault</td><td></td><td></td><td></td><td></td></tr><tr><td>Deviation  $R ^ { 2 }$ </td><td>0.850 ± 0.001</td><td>0.485 ± 0.180</td><td>0.779 ± 0.001</td><td> $0 . 4 3 3 \pm 0 . 0 4 3$ </td></tr><tr><td>DTW shape fidelity</td><td>0.22 ± 0.06</td><td>0.26 ± 0.23</td><td>-0.20 ± 0.02</td><td> $- 2 . 5 8 \pm 1 . 4 4$ </td></tr><tr><td>TDI</td><td>0.048 ± 0.001</td><td>0.083 ± 0.003</td><td>0.019 ± 0.000</td><td> $0 . 0 3 3 \pm 0 . 0 0 2$ </td></tr></table>

Mean ± standard deviation over three seeds, on faulty test cycles; for the unknown fault, averaged over the held-out failure modes. Deviation R<sup>2</sup>: share of the deviation from the reference cycle that the reconstruction recovers. DTW shape fidelity: the same under dynamic time warping. TDI: temporal distortion index. Bold: better method per setting. Higher is better, except for the TDI.

## C. Reconstruction Fidelity

We measure how well MouldIt and PCA reconstruct the deviation that a fault causes (Table IV). The deviation $R ^ { 2 } =$ $1 - \Sigma ( \hat { x } - x ) ^ { 2 } / \Sigma ( x - x _ { \mathrm { r e f } } ) ^ { 2 }$ is the share of this deviation from the reference cycle that the reconstruction recovers, where 0 means no better than copying the reference. The DTW shape fidelity is the same measure under dynamic time warping (DTW), following the shape and timing split of [42], and the temporal distortion index (TDI) [43] measures the remaining timing error, where 0 means none. The averages of the unknown DTW shape fidelity are dominated by the time shift mode, which is negative for both methods.

PCA is clearly ahead on unknown faults. Our data is close to linear around one reference cycle, PCA directly minimizes the squared error, and the MouldIt latent space also serves classification. High fidelity on unknown faults is not a goal in itself, since their poorer reconstruction is what makes the MouldIt reconstruction term useful for unknown detection.

## D. Limitations

All our cycles derive from one real cycle of one part and one material, and the magnitudes of our failure modes are modeling choices rather than calibrated effects (Section III-A). Our results therefore show what MouldIt can learn from process-informed synthetic data, not how it performs on real failures.

The sorting loss (Section IV) weighs its four rates equally, and our classes are balanced, whereas real production has few faults and unequal costs. Our operating point of 5% FPR is high for a production line.

At small sizes, our results are noisy. At 50 cycles, a class has at most five training cycles and one or two validation cycles, and the novelty thresholds are set from these few cycles. We selected the KLD and classification weights on one seed at a size of 1000, on folds that are also part of our final sweep, which is a mild selection on test data. Our novelty score (Section III-C) fits one Gaussian per class, although some classes are bimodal by design, for example underpacking and overpacking. Such a Gaussian also covers the gap between the two modes, so an unknown fault landing there looks familiar. Finally, we chose the network size and the latent dimension empirically. We also did not record a formal ablation, so we cannot yet attribute the latent separation to the BCE rather than to the MSE and KLD, which train the same encoder.

![](images/6e6bd0e74c6a43627de3890bd9f52ca72ae4e6f858b055c5d05123bba40f0a97.jpg)

![](images/a46a41f57ef3e366c35f08275eb16cdc02a7c510fbf2e5ac44292ecd3abfdeb7.jpg)

![](images/6aa0def1d628cdbdb66865228c96374676c23ce08ee63265ecc7c3d9abeea969.jpg)

(d) PCA, multi-modal  
![](images/79248e624fe5ecbfafb6f5508f85d0c1cd47c7c3b492fa881ba0a1b34fd7e4ee.jpg)  
● OK Known faults Detected by latent term Not detected by latent term

![](images/9dd88c2aec6f7d69cb60e1988a232eeee399b9f39caef31828027107563d72f8.jpg)

![](images/5e2a28c6963e0c67188cc2ad5a56e6f0887573eb2c8c726f5dc8da6b933b943d.jpg)

![](images/649fa9730f27e635406db2bfe74cb066b9923998bc740bace8728537d079b8e4.jpg)

(h) PCA, multi-modal  
![](images/8e76c824c598630e1a668366620ad237557c8441e87116b538e2251100fc843f.jpg)  
OK Known faults Detected by reconstruction only Detected by latent only Detected by both Missed  
Fig. 10. Novelty score of MouldIt and PCA with secondary pressure rise held out as the unknown fault, at 16000 cycles and seed 42. Distances are measured in the 32-dimensional latent space of MouldIt and on the 32 components of PCA. Panels (a)–(d) show the distance of each test cycle to the OK class and to its nearest known fault, each in units of that class’s 95% boundary. Panels (e)–(h) show the two terms of the novelty score, the latent term L and the reconstruction term $R / \eta _ { R }$ . Dashed lines mark the threshold Θ in all panels: the latent term detects a cycle only past both lines in (a)–(d), and a cycle is missed only in the lower left box in (e)–(h). Unknown cycles are colored by whether the latent term detects them in (a)–(d), and by whether the reconstruction term only, the latent term only, or both detect them in (e)–(h). The percentages give the share of unknown cycles flagged in this fold. We draw about half of the cycles for each label in the legend.

Future work should run MouldIt on real cycles with real failures, and could use its learned representations for process control. Future work should also train it without the BCE term, to test whether the BCE causes the latent separation, and with four channels on only the single-modal failure modes, to test whether the multi-modal gain comes from the richer set of known classes or from the coupling between channels.

## VI. CONCLUSION

We introduced MouldIt, a multi-objective temporal VAE for in-mould signals, and studied how its performance depends on the amount of training data and the input modalities, using synthetic datasets generated from a single real cycle. A learned representation outperforms the classical methods once there is enough data or more than one modality, and a single model can then classify parts, detect unknown faults, and sort parts. At 16000 cycles, the part sorting loss of MouldIt is the lowest of all compared methods, at $6 . 0 \pm 0 . 7 \%$ with pressure alone and $1 . 0 \pm 0 . 3 \%$ with all four channels; in the latter setting, random forests with the PCA unknown flag are on par. Whether this holds in real production remains the main open question.

## ACKNOWLEDGMENT

This work was carried out in part within the research project FORinFPRO (Project No. AZ-1596-23), funded by the Bayerische Transformations- und Forschungsstiftung. The author thanks R. Kini and L. Schweizer for the dataset used in this work. The author conceptualized this work, and generative AI was not used to formulate its research questions. Claude, Perplexity, and ChatGPT were used for literature search and brainstorming. Under the author’s direction, Claude was also used to write code, following the author’s code patterns, to prepare the figures, to co-create the Discussion, Conclusion, figure captions, and novelty score description from the author’s results, insights, post-results abstract, and instructions, and to edit the wording and formatting of the remaining text. The author reviewed and revised all AI-assisted work over many iterations and takes responsibility for the content.

## REFERENCES

[1] M. Kariminejad, D. Tormey, S. Huq, J. Morrison, and M. McAfee, “Ultrasound sensors for process monitoring in injection moulding,” Sensors, vol. 21, no. 15, 2021, Art. no. 5193, doi: 10.3390/s21155193.

[2] N. Hollmann, S. Müller, K. Eggensperger, and F. Hutter, “TabPFN: A transformer that solves small tabular classification problems in a second,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023, doi: 10.48550/arXiv. 2207.01848.

[3] S. Mou, R. Vemulapalli, S. Li, Y. Liu, C. Thomas, M. Cao, H. Bai, O. Tuzel, P. Huang, J. Shan, and J. Shi, “Synth4Seg—learning defect data synthesis for defect segmentation using bi-level optimization,” IEEE Trans. Autom. Sci. Eng., vol. 22, pp. 18 231–18 242, 2025, doi: 10.1109/ TASE.2025.3583713.

[4] W. Li, C. Gu, J. Chen, C. Ma, X. Zhang, B. Chen, and S. Wan, “DLS-GAN: Generative adversarial nets for defect location sensitive data augmentation,” IEEE Trans. Autom. Sci. Eng., vol. 21, no. 4, pp. 5173– 5189, 2024, doi: 10.1109/TASE.2023.3309629.

[5] Y. Tai, K. Yang, T. Peng, Z. Huang, and Z. Zhang, “Defect image sample generation with diffusion prior for steel surface defect recognition,” IEEE Trans. Autom. Sci. Eng., vol. 22, pp. 8239–8251, 2025, doi: 10.1109/ TASE.2024.3482362.

[6] A. Polenta, S. Tomassini, N. Falcionelli, P. Contardo, A. F. Dragoni, and P. Sernani, “A comparison of machine learning techniques for the quality classification of molded products,” Information, vol. 13, no. 6, 2022, Art. no. 272, doi: 10.3390/info13060272.

[7] L. Bogedale, S. Doerfel, A. Schrodt, and H.-P. Heim, “Online prediction of molded part quality in the injection molding process using highresolution time series,” Polymers, vol. 15, no. 4, 2023, Art. no. 978, doi: 10.3390/polym15040978.

[8] N. West and M. Klute, “Cross-process-chain dataset archive: Combined data collection from injection molding and screw driving,” Zenodo, 2025, doi: 10.5281/zenodo.17240390.

[9] J. Lambers, J. Schüder, and J. Krauß, “Injection-molding production data with quality labels,” B2SHARE, 2022, doi: 10.23728/b2share. 03133fb279294389a15baefd55e4257a.

[10] Z. Ahmed Shaikh, “Molding machine free access data,” Zenodo, 2024, doi: 10.5281/zenodo.10844048.

[11] S. Michiels, C. De Schryver, L. Houthuys, F. Vogeler, and F. Desplentere, “Machine learning for automated quality control in injection moulding manufacturing,” in Proc. Eur. Symp. Artif. Neural Netw. (ESANN), 2022, pp. 127–132, doi: 10.14428/esann/2022.es2022-48.

[12] C. Hennebold, J. Krauß, and M. F. Huber, “Synthetic data generation using causal models for injection molding processes,” Procedia CIRP, vol. 134, pp. 831–836, 2025, doi: 10.1016/j.procir.2025.02.202.

[13] R. Kini and L. Schweizer, “FORinFPRO-HIMD: Multimodal sensor dataset for hybrid injection molding of continuous fiber-reinforced polypropylene composites,” Zenodo, 2026, doi: 10.5281/zenodo. 20744054.

[14] D. Wang, X. Pan, Y. Gao, F. Tsung, and F. Li, “Adaptive multimodal industrial fault diagnosis with attention-driven fusion boosting unimodal performance,” IEEE Trans. Autom. Sci. Eng., vol. 23, pp. 4454–4476, 2026, doi: 10.1109/TASE.2026.3658782.

[15] M. Maggipinto, A. Beghi, and G. A. Susto, “A deep convolutional autoencoder-based approach for anomaly detection with industrial, non images, 2-dimensional data: A semiconductor manufacturing case study,” IEEE Trans. Autom. Sci. Eng., vol. 19, no. 3, pp. 1477–1490, 2022, doi: 10.1109/TASE.2022.3141186.

[16] V. Tra, M. Amayri, and N. Bouguila, “Latent code description for unsupervised AHU fault detection using adaptive adversarial autoencoder,” IEEE Trans. Autom. Sci. Eng., vol. 23, pp. 3241–3254, 2026, doi: 10.1109/TASE.2024.3481211.

[17] H. Zhang, H. Bai, C. Yang, and L. Li, “Multi-level fusion transformer coupled with mechanistic model and multimodal data for soft-sensor modeling in sintering process,” IEEE Trans. Autom. Sci. Eng., vol. 23, pp. 6437–6447, 2026, doi: 10.1109/TASE.2026.3672271.

[18] P. Yan, A. Abdulkadir, G. Aguzzi, G. A. Schatte, B. F. Grewe, and T. Stadelmann, “Automated process monitoring in injection molding via representation learning and setpoint regression,” in Proc. 11th IEEE Swiss Conf. Data Sci. (SDS), 2024, pp. 138–145, doi: 10.1109/SDS60720. 2024.00027.

[19] V. Ketonen and J. O. Blech, “Anomaly detection for injection molding using probabilistic deep learning,” in Proc. 4th IEEE Int. Conf. Ind. Cyber-Phys. Syst. (ICPS), 2021, pp. 70–77, doi: 10.1109/ICPS49255. 2021.9468190.

[20] G. Kim, J. G. Choi, M. Ku, H. Cho, and S. Lim, “A multimodal deep learning-based fault detection model for a plastic injection molding process,” IEEE Access, vol. 9, pp. 132 455–132 467, 2021, doi: 10.1109 ACCESS.2021.3115665.

[21] M. Muaz, H. Yu, W. L. Sung, C. Liu, and B. Drescher, “A multitask encoder–decoder model for quality prediction in injection moulding,” J.

Manuf. Processes, vol. 103, pp. 238–247, 2023, doi: 10.1016/j.jmapro. 2023.08.030.

[22] P. Yan, A. Abdulkadir, G. A. Schatte, G. Aguzzi, J. Gha, N. Pascher, M. Rosenthal, Y. Gao, B. F. Grewe, and T. Stadelmann, “Learning actionable world models for industrial process control,” in Proc. IEEE Swiss Conf. Data Sci. (SDS), 2025, pp. 111–118, doi: 10.1109/SDS66131. 2025.00022.

[23] D. F. Alenezi, M. Biehler, J. Shi, and J. Li, “Physics-informed weaklysupervised learning for quality prediction of manufacturing processes,” IEEE Trans. Autom. Sci. Eng., vol. 22, pp. 2006–2019, 2025, doi: 10. 1109/TASE.2024.3374098.

[24] X. Jiang, J. Zheng, Z. Chen, Z. Ge, Z. Song, and X. Ma, “Leveraging transfer learning for data augmentation in fault diagnosis of imbalanced time-frequency images,” IEEE Trans. Autom. Sci. Eng., vol. 22, pp. 6762–6772, 2025, doi: 10.1109/TASE.2024.3454418.

[25] H. Tercan, A. Guajardo, J. Heinisch, T. Thiele, C. Hopmann, and T. Meisen, “Transfer-learning: Bridging the gap between real and simulation data for machine learning in injection molding,” Procedia CIRP, vol. 72, pp. 185–190, 2018, doi: 10.1016/j.procir.2018.03.087.

[26] Y. Lockner and C. Hopmann, “Induced network-based transfer learning in injection molding for process modelling and optimization with artificial neural networks,” Int. J. Adv. Manuf. Technol., vol. 112, no. 11–12, pp. 3501–3513, 2021, doi: 10.1007/s00170-020-06511-3.

[27] Y. Lockner, C. Hopmann, and W. Zhao, “Transfer learning with artificial neural networks between injection molding processes and different polymer materials,” J. Manuf. Processes, vol. 73, pp. 395–408, 2022, doi: 10.1016/j.jmapro.2021.11.014.

[28] R. D. Párizs, D. Török, T. Ageyeva, and J. G. Kovács, “Machine learning in injection molding: An Industry 4.0 method of quality prediction,” Sensors, vol. 22, no. 7, 2022, Art. no. 2704, doi: 10.3390/s22072704.

[29] Z. Zuo, T. Meng, Z. Li, H. Zhang, T. Liu, Z. Chen, and Z. Pang, “Contrastive conditional adversarial autoencoder with class-specific forces for imbalanced open-set fault detection,” IEEE Trans. Autom. Sci. Eng., vol. 22, pp. 15 757–15 767, 2025, doi: 10.1109/TASE.2025.3570077.

[30] Z. Liu, G. Cai, and H. Han, “A knowledge evaluation-based multiattribute zero-shot fault diagnosis for industrial process,” IEEE Trans. Autom. Sci. Eng., vol. 23, pp. 16 874–16 885, 2026, doi: 10.1109/TASE.2026.3735080.

[31] L. Cai, H. Yin, and J. Lin, “A unified framework with incremental learning capacity for industrial fault detection and classification,” IEEE Trans. Autom. Sci. Eng., vol. 22, pp. 765–775, 2025, doi: 10.1109/TASE. 2024.3353535.

[32] W. He, J. Mao, Y. Wang, Z. Li, X. Wang, and S. Wang, “MQKIN: Manufacturing quality knowledge-driven interpretable fault diagnosis network for robotic grinding equipment,” IEEE Trans. Autom. Sci. Eng., vol. 22, pp. 17 430–17 441, 2025, doi: 10.1109/TASE.2024.3418972.

[33] L. Ma, Q. Yang, and K. Peng, “Dynamic dependencies of fault-related variables-based interpretable fault diagnosis framework for industrial processes,” IEEE Trans. Autom. Sci. Eng., vol. 23, pp. 11 183–11 195, 2026, doi: 10.1109/TASE.2026.3703004.

[34] J. Gim and B. Rhee, “Novel analysis methodology of cavity pressure profiles in injection-molding processes using interpretation of machine learning model,” Polymers, vol. 13, no. 19, 2021, Art. no. 3297, doi: 10. 3390/polym13193297.

[35] P. Zhao, J. Zhang, Z. Dong, J. Huang, H. Zhou, J. Fu, and L.-S. Turng, “Intelligent injection molding on sensing, optimization, and control,” Adv. Polym. Technol., vol. 2020, 2020, Art. no. 7023616, doi: 10.1155/2020/ 7023616.

[36] M. Myers, R. Mulyana, J. M. Castro, and B. Hoffman, “Experimental development of an injection molding process window,” Polymers, vol. 15, no. 15, 2023, Art. no. 3207, doi: 10.3390/polym15153207.

[37] A. Mourya, A. Nanda, K. Parashar, Sushant, and R. Kumar, “An explanatory study on defects in plastic molding parts caused by machine parameters in injection molding process,” Mater. Today, Proc., vol. 78, pp. 656–661, 2023, doi: 10.1016/j.matpr.2022.12.070.

[38] N.-Y. Zhao, J.-Y. Lian, P.-F. Wang, and Z.-B. Xu, “Recent progress in minimizing the warpage and shrinkage deformations by the optimization of process parameters in plastic injection molding: A review,” Int. J. Adv. Manuf. Technol., vol. 120, no. 1–2, pp. 85–101, 2022, doi: 10.1007/ s00170-022-08859-0.

[39] M. Baum, D. Anders, and T. Reinicke, “Approaches for numerical modeling and simulation of the filling phase in injection molding: A review,” Polymers, vol. 15, no. 21, 2023, Art. no. 4220, doi: 10.3390/ polym15214220.

[40] F. J. Alzahrani, A. A. Duhduh, P. Gao, and J. P. Coulter, “The influence of cooling rate and mold temperature on polymers crystallization kinetics in injection molding,” in Proc. ASME Int. Mech. Eng. Congr. Expo., vol. 2B, 2021, Art. no. V02BT02A039, doi: 10.1115/IMECE2021-73665.

[41] O. Ledoit and M. Wolf, “A well-conditioned estimator for largedimensional covariance matrices,” J. Multivariate Anal., vol. 88, no. 2, pp. 365–411, 2004, doi: 10.1016/S0047-259X(03)00096-4.

[42] V. Le Guen and N. Thome, “Shape and time distortion loss for training deep time series forecasting models,” in Proc. Adv. Neural Inf. Process. Syst., vol. 32, 2019, doi: 10.48550/arXiv.1909.09020.

[43] L. Frías-Paredes, F. Mallor, M. Gastón-Romeo, and T. León, “Assessing energy forecasting inaccuracy by simultaneously considering temporal and absolute errors,” Energy Convers. Manage., vol. 142, pp. 533–546, 2017, doi: 10.1016/j.enconman.2017.03.056.