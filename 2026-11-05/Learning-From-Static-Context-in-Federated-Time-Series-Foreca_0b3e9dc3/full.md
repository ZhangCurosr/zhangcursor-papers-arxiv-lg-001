# Learning From Static Context in Federated Time Series Forecasting

Viktor Walter

Faculty of Management Science and Engineering

Karlsruhe University of Applied Sciences (HKA)

Karlsruhe, Germany

viktor.walter@h-ka.de

Fabian Stricker

Institute of Data-Centric Software Systems (IDSS)

Karlsruhe University of Applied Sciences (HKA)

Karlsruhe, Germany

fabian.stricker@h-ka.de

Andreas Wagner

Faculty of Management Science and Engineering

Karlsruhe University of Applied Sciences (HKA)

Karlsruhe, Germany

Fraunhofer Institute for Industrial Mathematics (ITWM)

Kaiserslautern, Germany

andreas.wagner@h-ka.de

Abstract—Time series forecasting is shifting from locally trained, per-series models to global models trained across many heterogeneous series, which requires learning what sets the series apart. When the raw data cannot be centralized, such a global model is trained with federated learning (FL), one client per data owner. State-of-the-art forecasters such as the Temporal Fusion Transformer (TFT) tell series apart through a static input that describes each series and does not change over time. In the special case of a federated setting where each client holds a single series, this static input is constant within each client and varies only across clients. We ask whether a federated TFT can learn from it. Per-batch FedAdam, which aggregates gradients after every batch, uses the static input almost as well as central training, whereas communication-efficient schemes with local epochs (FedAvg, FedAdam+Prox) extract far less from it and gain nothing significant on either application dataset. At the same local computation, an Adam-compatible SCAFFOLD keeps the static pathway responsive on wind power and gains significantly there. On building energy its error still drops, but the static pathway falls silent, so drift and communication frequency cannot be fully separated as the cause. Only per-batch FedAdam matches or beats the per-client local models across all datasets. We confirm the effect on building-energy meter data, a wind-power dataset, and a controlled synthetic dataset. The loss is not specific to the TFT: on the synthetic and the wind data two further architectures with different static-conditioning mechanisms show it as well.

Index Terms—Federated Learning, Time Series Forecasting, Temporal Fusion Transformer, Static Features, Client Drift, Heterogeneity

## I. INTRODUCTION

Time series forecasting increasingly relies on large, deep models trained across many heterogeneous series, rather than one specialized model per series. The data come from diverse sources whose technical characteristics shape each series. In renewable energy forecasting, for example, technical differences determine the shape of a time series of electricity production and may have to be learned along with it [1]. Another domain common in federated forecasting is electricity demand forecasting [2], where office buildings, hospitals, schools, and homes follow very different patterns. A model exposed to many series must therefore learn what sets them apart. Training one global model across many series instead of one model per series is a recent shift [3], and a federated forecaster is such a global model.

A state-of-the-art forecasting architecture is the Temporal Fusion Transformer (TFT) [4]. It uses a dedicated static input to tell entities apart. A static feature is time-invariant and describes the series, in the simplest case as a categorical identifier, or continuously as hub height for wind turbines, floor area for buildings, or tilt angle for photovoltaic systems.

Pooling the raw data is not an option in these domains. Load profiles are personal data, because non-intrusive load monitoring infers occupancy and appliance use from them [5]. Wind and photovoltaic operators treat power curves and supervisory control data as trade secrets, and marketers may not pass on the data of the plants they market [1]. Federated learning (FL) trains one model across the owners without moving their data, one client per owner.

In the special case where each client holds a single entity, the client boundary is the entity boundary and the static feature is constant within a client. This case raises the problems we investigate in this paper: a constant feature has no variance, so a client cannot learn from it on its own, and its information lies entirely in how it varies across clients, which only federated training can access.

We study this with a federated TFT trained under several aggregation schemes, a probe that measures static-input usage directly, two diagnostics that localize the loss, and three datasets where the role of the static input is known or controllable. The answers to the three questions below are the contributions of this paper.

RQ1 Can a federated TFT learn from a client-constant static input? Fully only under per-batch FedAdam, which uses it almost as well as central training. Schemes with local epochs extract a fraction at most. RQ2 Is the limit the communication frequency or the client drift? Not conclusively. Fewer local steps help, but they raise the communication frequency and lower the drift together. Drift correction at a fixed frequency, the one experiment that separates the two, keeps part of the static pathway on wind power and gains significantly there, but recovers little on the synthetic data and nothing on building energy. RQ3 Where does the failure sit, and what does it cost to avoid? Not in the static encoder alone. On building energy, the loss grows with the number oflocal steps under a proximal term and is complete under FedAvg, and coupling only the static encoder per batch leaves the static pathway silent. Avoiding it costs one communication round per gradient step.

## II. RELATED WORK

Federated learning trains a shared model across clients without sharing raw data [6]. It is widely used for energy time series forecasting, where data are private and siloed [1, 2], and its recurring obstacle is client heterogeneity. Domain-informed features are engineered to reduce input heterogeneity across clients, so that a single shared model fits them more easily [7]. Another mitigation is personalization, which keeps part of the model local to each client [8]. We exclude personalization and keep a single global model. Personalization holds the last layers local and aggregates the rest, whereas the static input enters at the start of the network, and personalizing the static pathway would defeat its purpose, because its input is constant within a client.

Deep forecasting models separate time-varying inputs from static inputs that describe each series. The Temporal Fusion Transformer encodes static covariates into context vectors that initialize the states of its long short-term memory (LSTM) encoder and enrich the temporal features [4]. Global models trained over many series rely on static entity embeddings to tell series apart [9]. All of this is centralized, where the static input varies across the dataset. None addresses the federated case, where it is constant within a client. Recent work deploys the federated TFT for battery state-of-health prediction [10] and cross-silo psychiatric risk forecasting [11], but does not ask whether the static input is learned under aggregation.

With heterogeneous data, local updates move toward local optima, a phenomenon known as client drift that grows with the number of local steps [12]. The theory of local versus minibatch stochastic gradient descent (SGD) suggests that, in sufficiently heterogeneous regimes, averaging locally updated weights can be worse than aggregating gradients [13]. Several methods counteract drift, through control variates [12, 14], proximal terms [15], or adaptive server optimization [16]. These methods concern the convergence of the model as a whole. Layer-wise studies measure which layers diverge most, on image classification [17, 18]. None localizes the failure to a functional input pathway such as a static covariate encoder, or uses selective aggregation as a diagnostic. Bao et al. prove that local steps help to learn client-specific features that are discriminative within a client [19]. A client-constant static input is the opposite case: locally it is indistinguishable from a bias, and only the aggregation across clients can identify it.

![](images/7c36c6da0db75eb0a1d8da3235cbc7473c624081cb0a6923f9f73ab6538684cd.jpg)  
Fig. 1. TFT with its temporal pathway (blue) and static pathway (orange). Encoder (observed past) and decoder (known future) inputs are separate. The static context vectors condition both at three points: variable selection, LSTM state initialization, and GRN enrichment. Both GRNs feed the shared self attention

## III. METHODOLOGY

## A. Temporal Fusion Transformer

The TFT is an attention-based model for multi-horizon forecasting [4]. It separates two kinds of inputs. Time-varying inputs are the observed past and the known future covariates, each passed through a variable selection network (VSN), an LSTM encoder–decoder, and a self-attention layer. A VSN weights the input variables at each time step and keeps the informative ones. A gated residual network (GRN) is the recurring building block: a gated nonlinear transform with a skip connection.

A static covariate encoder maps the static input to four context vectors [4]. As shown in Figure 1, they condition three points of the temporal pathway: they enter the variable selection, initialize the hidden and cell states of the LSTM, and enrich the features through a GRN (static enrichment).

## B. Federated Aggregation Schemes

We train the TFT across K clients that exchange only model updates. We compare four aggregation schemes that differ in what they aggregate and how they handle client drift (Table I).

Per-batch FedAdam aggregates gradients rather than weights. Clients compute gradients on a shared model after every batch, the server averages them and applies an Adam step, so coupling is maximal: one aggregation per gradient step. This is FedAdam [16] in the single-local-step regime, where the client update reduces to the minibatch gradient [6].

TABLE I  
FEDERATED AGGREGATION SCHEMES. b IS THE NUMBER OF BATCHES IN ONE LOCAL EPOCH: 214 TO 314 ON BUILDING ENERGY, 45 ON WIND POWER, 7 ON THE SYNTHETIC DATA. CV: CONTROL VARIATES
<table><tr><td>Scheme</td><td>Local steps per round</td><td>Aggregated</td><td>Server opt.</td></tr><tr><td>Per-batch FedAdam</td><td>1</td><td>gradients</td><td>Adam</td></tr><tr><td>FedAvg</td><td>one epoch (b)</td><td>weights</td><td>none</td></tr><tr><td>FedAdam+Prox</td><td>one epoch (b)</td><td>deltas</td><td>Adam</td></tr><tr><td>SCAFFOLD</td><td>one epoch (b)</td><td>deltas + CV</td><td>none</td></tr></table>

FedAvg aggregates weights. Each client runs one local epoch, and the server averages the resulting weights, weighted by each client’s training sample size [6].

FedAdam+Prox is a hybrid of two state-of-the-art schemes. Clients run local epochs with a proximal term that penalizes drift from the global model (FedProx [15]). The server treats the averaged weight deltas as a pseudo-gradient and applies an adaptive Adam update (FedAdam [16]). We use this hybrid as a strong representative of communication-efficient FL, since it combines the two dominant drift remedies [20].

SCAFFOLD adds client- and server-side control variates to FedAvg to correct client drift [12]. We include it as the canonical drift-correction scheme, to test whether explicit drift correction restores the static pathway under local epochs. Canonical SCAFFOLD assumes SGD client optimizers. Transformer-based models such as our TFT, however, do not train stably under SGD and diverge [21], so we train all schemes with Adam. We therefore use an Adam-compatible variant in the style of FAdamGC [22]. The drift correction is added to the raw gradient before Adam’s moment estimation, because applying it afterwards fails through the nonlinear coupling of the two moments. The control variates use the optimizer-agnostic mean of the raw local gradients rather than the SGD-specific update rule. Mime [14] instead carries server-side Adam statistics into every local step. We keep the structure of SCAFFOLD, whose only change against FedAvg is the drift correction. This restores stable training, with errors in the range of FedAvg and FedAdam+Prox. All three schemes with local epochs run one local epoch per round.

FedAdam+Prox uses $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9$ on the server, an adaptivity of $\tau = 1 0 ^ { - 3 }$ , and a proximal weight of $\lambda =$ $1 0 ^ { - 2 }$

A static feature needs one more federated step before training. Because it is constant within a client, standardizing it locally would map it to zero, as it has no withinclient variance. We therefore standardize the static features with statistics aggregated globally across clients [23], which preserves their cross-client variation. For each static feature $v ,$ each client reports its sample count and the sums of v and $v ^ { 2 } .$ , the server returns the global mean $\mu$ and variance $\sigma ^ { 2 }$ , and each client applies $( v - \mu ) / \sigma$ . This needs one data pass and one communication round.

This step carries a privacy cost. Because the feature is constant within a client, the reported sum and count already determine its value, so the server learns each client’s static feature exactly. Aggregating these statistics relaxes the federated privacy guarantee for the very input we study. We accept this as a limitation here, since our question is whether the static input is learned, not how to protect it, and we standardize identically across all schemes to keep the comparison fair. Domain knowledge could set these constants without the exchange, since typical hub heights, rotor diameters, or floor areas are not secret (Section VI).

## C. Probing Static-Input Usage

Error metrics do not show whether a model uses the static input, so we probe it directly at test time, without retraining. Let $f _ { \theta } ( \mathbf { x } _ { i } , \mathbf { u } _ { i } , s _ { i } )$ be the prediction for test sample i with target $y _ { i }$ , from observed inputs $\mathbf { x } _ { i }$ , known inputs ${ \bf u } _ { i } ,$ and static input $s _ { i } .$ . Without the index, x, u, s, and y collect all test samples.

We measure static-input sensitivity as the relative increase in error when the static input is randomly permuted across the test set, the permutation feature importance of Breiman [24]. We pool the test samples of all clients, evaluate the root mean square error (RMSE) with the true static input, then permute the static input across the pooled set and re-evaluate:

$$
S = \frac { \mathrm { R M S E } \big ( f _ { \theta } ( \mathbf { x } , \mathbf { u } , \pi ( s ) ) , y \big ) } { \mathrm { R M S E } \big ( f _ { \theta } ( \mathbf { x } , \mathbf { u } , s ) , y \big ) } - 1 ,\tag{1}
$$

where $\pi ( s )$ is a random permutation of the static inputs over the pooled test set. S is zero for a model that ignores the static input and grows with its reliance on it, since shuffling an unused input cannot change the error. Dividing by the unpermuted error makes $S$ comparable across schemes whose error levels differ. We report every sensitivity as $\tilde { S } = S / S _ { \mathrm { c e n t r a l } } ,$ relative to the central TFT on the same dataset, so the scales are comparable across datasets.

## D. Two Diagnostics on the Static Pathway

The probe shows whether a model responds to the static input, not what removes the response. Two controlled variations address that.

The first varies the amount of local computation. We sweep the number of local epochs per round from $E = 0 . 1$ to $E = 2$ for FedAdam+Prox and from $E = 0 . 2 5 \mathrm { t o } E = 2$ for FedAvg, and hold everything else fixed. If local training removes the static input, its sensitivity S<sup>˜</sup> should fall with $E$ rather than vanish at a threshold.

The second isolates the static pathway. We aggregate the parameters of the static covariate encoder after every batch while the temporal pathway keeps its local epochs, and call this static-path coupling. The static encoder holds 22.3 % of the parameters.

## E. Statistical Significance

For every client k we average the test RMSE over seeds, with and without the static input, and write the two means $e _ { k }$ and $e _ { k } ^ { 0 } .$ . The client is the unit we generalize over, and averaging over seeds first keeps the runs of one seed from counting as independent observations. We test whether the static input lowers the error with a one-sided Wilcoxon signed-rank test on the K paired differences $e _ { k } ^ { 0 } - e _ { k }$ and report the p-value, with $p < 0 . 0 5$ as the threshold.

## IV. EXPERIMENTAL SETUP

We study the question on two application domains, building energy and wind power, and on controlled synthetic data, where the role of the static feature is known.

## A. Building Energy Data

We use the Building Data Genome 2 (BDG2) corpus of hourly electricity meter readings [25]. We treat $K \ = \ 3 2$ buildings as clients. Each building is described by its primary use, floor area, year built, and floor count as static features, and by past load, local weather, and calendar features as timevarying inputs (Table II). The primary use covers categories such as education, residential, office, and healthcare. It is labelencoded and embedded linearly like the other static features. We use a moving window that advances one time step, with the lookback and horizon of Table III.

We drop any window that contains a missing value rather than impute. Up to 20% of the values per client are missing, mostly in contiguous stretches of several days, so the share of dropped windows stays close to the share of missing values.

## B. Wind Power Data

As a second domain we consider wind power forecasting [26], where each of the K = 32 wind turbines is one client. The power output is physically simulated and validated against real wind turbines [27], whereas the weather inputs are real numerical weather prediction (NWP) output. The static features capture the physical configuration that shapes the power output (Table II). The manufacturer’s power curve enters through its cut-in, rated, and cut-out wind speeds. The time-varying inputs are the lagged power and NWP wind speed. We draw the NWP wind speed from two systems of different spatial resolution, the 2.2 km ICON-D2 of the German Weather Service (DWD) and the high-resolution forecast (HRES) of the European Centre for Medium-Range Weather Forecasts (ECMWF), which we use on a 0.25<sup>◦</sup> grid (about 28 km, natively $0 . 1 ^ { \circ }$ or 9 km). ICON-D2 reaches 48 hours ahead. The series run from August 2023 to October 2024. We use the same moving-window protocol as for the load setting.

## C. Synthetic Data

The synthetic generator isolates the static pathway. We generate $K = 1 2$ clients, one per entity. Each entity has a univariate series that shares a common temporal shape but differs in level $\ell _ { k } .$ The shape is the sum of a daily and a weekly sinusoid. We add to each entity a static feature, a standardized scalar that is constant over time and unique to the entity, and set the level so that it can be recovered only from this feature. We mean-center each observed window, so the level cannot be read off the recent history and must come from the static feature, whereas the forecast target keeps its absolute level. This is a deliberate worst case: real static descriptors such as hub height vary smoothly across entities and generalize, whereas here the static input is used purely as an entity cue. We add Gaussian observation noise with a standard deviation of 0.35, against sinusoid amplitudes of 2.0 and 1.0, with an independent realization per entity. We use the same moving-window protocol as for the application domains, with the lookback and horizon of Table III. To mimic the data loss of the real setting, we randomly drop up to 20% of the windows per client, comparable to BDG2.

TABLE II  
INPUT VARIABLES PER SETUP
<table><tr><td>Setup</td><td>Observed</td><td>Known</td><td>Static</td></tr><tr><td>Synthetic</td><td>target lagged</td><td>sin/cos 24h, sin/cos 168h</td><td>entity feature</td></tr><tr><td>Building energy</td><td>load lagged, air temp., wind speed</td><td>hour, day of week (sin/cos), weekend flag</td><td>primary use floor area year built floor count</td></tr><tr><td>Wind power</td><td>power lagged</td><td>wind speed ICON-D2: 78m, 127m, 184m ECMWF HRES: 100m</td><td>turbine age power curve hub height rotor diam. altitude</td></tr></table>

TABLE III  
MODEL ARCHITECTURE AND WINDOWING PER SETUP
<table><tr><td></td><td>Synthetic</td><td>Building energy</td><td>Wind power</td></tr><tr><td> $d _ { \mathrm { m o d e l } }$ </td><td>16</td><td>64</td><td>64</td></tr><tr><td>Attention heads</td><td>2</td><td>4</td><td>4</td></tr><tr><td>Lookback</td><td>24</td><td>72</td><td>48</td></tr><tr><td>Horizon</td><td>24</td><td>24</td><td>48</td></tr><tr><td>Dropout</td><td>0</td><td>0.1</td><td>0.1</td></tr></table>

## D. Models and Baselines

We use the TFT of Section III-A and train it with the four federated schemes of Section III-B. Its architecture and the windowing per setup are summarized in Table III. All federated schemes use a batch size of 32 and a client learning rate of $1 0 ^ { - 3 }$ , and the server optimizers a learning rate of 10<sup>−2</sup>. Central training uses a batch size of 256 on building energy, 64 on wind power and 32 on the synthetic data. All models are trained with the pinball loss at the median $( q = 0 . 5 )$ .

We add two baselines. Central pools all data and trains a single model, with and without the static input. Local trains a separate model per client without the static input, which is meaningless on a single client.

## E. Protocol

We split the building-energy and synthetic series $6 0 / 4 0$ in time into a training and a test fold. The wind series use their last three months, August to October 2024, as the test fold after twelve months of training, so the wind test fold covers one season. In all cases the test data are strictly later than the training data. To set the number of training epochs we hold out the last 20% of the training fold as a validation set and pick the epoch that minimizes the validation RMSE. In federated training this validation RMSE is aggregated across clients, and training stops after 10, 20 and 15 rounds without improvement on building energy, wind power and the synthetic data. For per-batch FedAdam a round is a block of at most 50 gradient steps. We then retrain on the full training fold for that many epochs and evaluate once on the test fold. This rules out comparing under-trained models while keeping the test fold untouched during training. We repeat every configuration over five random seeds and report the mean. We evaluate with RMSE, mean absolute error (MAE), and $R ^ { 2 }$ , the static-input probe of Section III-C, and the significance test of Section III-E. Errors are reported on the log-transformed load standardized per building, on power normalized by rated capacity, and on the raw synthetic scale.

TABLE IV  
SYNTHETIC DATA. MEAN OVER SEEDS AND CLIENTS. p: ONE-SIDED WILCOXON SIGNED-RANK TEST OVER PER-CLIENT RMSE PAIRS. \* AND SHADING: p < 0.05
<table><tr><td>Scheme</td><td>Static</td><td>RMSE</td><td>MAE</td><td> $R ^ { 2 }$ </td><td>p(static↓)</td></tr><tr><td>Central</td><td>√</td><td>0.36</td><td>0.28</td><td>0.95</td><td>&lt;.001*</td></tr><tr><td>Central</td><td>1</td><td>1.42</td><td>1.25</td><td>0.22</td><td></td></tr><tr><td>Per-batch FedAdam</td><td>√</td><td>0.35</td><td>0.28</td><td>0.95</td><td>&lt;.001*</td></tr><tr><td>Per-batch FedAdam</td><td></td><td>1.43</td><td>1.26</td><td>0.22</td><td></td></tr><tr><td>FedAdam+Prox</td><td>√</td><td>1.00</td><td>0.75</td><td>0.62</td><td>0.026*</td></tr><tr><td>FedAdam+Prox</td><td>一</td><td>1.43</td><td>1.25</td><td>0.21</td><td></td></tr><tr><td>FedAvg</td><td>√</td><td>1.11</td><td>0.87</td><td>0.53</td><td>0.039*</td></tr><tr><td>FedAvg</td><td>一</td><td>1.45</td><td>1.24</td><td>0.19</td><td></td></tr><tr><td>SCAFFOLD</td><td>√</td><td>1.02</td><td>0.86</td><td>0.60</td><td>0.039*</td></tr><tr><td>SCAFFOLD</td><td>一</td><td>1.44</td><td>1.26</td><td>0.20</td><td></td></tr><tr><td>Local</td><td></td><td>0.36</td><td>0.29</td><td>0.95</td><td></td></tr></table>

## V. RESULTS

## A. Static-Input Usage Under Federated Aggregation

Table IV reports the synthetic results. By construction, the static input is the only feature from which the model can recover the level. Central training and per-batch FedAdam use it almost fully, raising $R ^ { 2 }$ from 0.22 to 0.95, on par with the local baseline. The local-epoch schemes fall far short. All three local-epoch schemes gain significantly but only partially $( R ^ { 2 } \ 0 . 5 3$ to 0.62 against 0.95, $p ~ \leq ~ 0 . 0 3 9 )$ . SCAFFOLD lies between FedAvg and FedAdam+Prox, so its drift correction adds little on this dataset. The static input is highly informative, yet only frequent coupling lets the model exploit it fully.

## B. Probing the Static Pathway

The static-input sensitivity of Section III-C shows whether the input is used (Figure 2). Central training and per-batch FedAdam are by far the most sensitive to the static input on all three datasets. On building energy and wind power, FedAvg and FedAdam+Prox are not: permuting the static input barely changes their error. On the synthetic data all three local-epoch schemes still respond, but far below central training (0.11 for FedAvg, 0.16 for SCAFFOLD and 0.18 for FedAdam+Prox), which matches the partial benefit they extract there. SCAFFOLD stands out from FedAvg and FedAdam+Prox only on wind power, where it retains part of its sensitivity and scatters strongly across seeds.

![](images/2972dbf8537ba9f754a65769f51cbf9bed2a35902c46715466b2497c25f788c9.jpg)  
Fig. 2. Static-input sensitivity relative to central per dataset. Central and perbatch FedAdam stay sensitive on all three datasets. The schemes with local epochs keep a small share on the synthetic data and drop to near zero on the application datasets, except SCAFFOLD on wind power

## C. Other Conditioning Mechanisms

The static conditioning of the TFT runs through context vectors. We repeat the experiment with two further architectures that condition differently, a multilayer perceptron (MLP) that receives the static input in the first layer of its shared representation, and a gated recurrent unit (GRU) whose encoder state is initialized and whose decoder state is shifted by it. Table V reports the static-input sensitivity of all three under per-batch FedAdam and under FedAvg, on the scale of Figure 2.

On the synthetic data all three architectures lose almost all of their sensitivity under FedAvg, and on wind power the TFT and the GRU lose it entirely while the MLP keeps 0.08 of 0.23. The loss is therefore not a property of the context vectors of the TFT.

On building energy, where the load history already identifies the entity, the TFT loses its response entirely, the MLP has almost none to lose, and the GRU keeps a third of its sensitivity, 0.37 of 1.21. Both use the hidden size and training setup of the TFT and have 7 to 18 times fewer parameters, so this comparison supports whether the loss occurs, not which architecture forecasts better.

## D. Building Energy

Table VI shows a similar, though weaker, pattern on BDG2. Central training and per-batch FedAdam still gain a significant benefit from the static input, but a small one. FedAvg and

TABLE V  
STATIC-INPUT SENSITIVITY PER ARCHITECTURE UNDER PER-BATCH FEDADAM (PB) AND UNDER FEDAVG (FA), RELATIVE TO THE CENTRAL TFT ON THE SAME DATASET AS IN FIGURE 2. A DROP TO NEAR ZERO MEANS THE ARCHITECTURE STOPS RESPONDING TO THE STATIC INPUT
<table><tr><td rowspan="2"></td><td colspan="2">Synthetic</td><td colspan="2">Building energy</td><td colspan="2">Wind power</td></tr><tr><td>PB</td><td>FA</td><td>PB</td><td>FA</td><td>PB</td><td>FA</td></tr><tr><td>TFT</td><td>1.01</td><td>0.11</td><td>1.01</td><td>0.00</td><td>0.96</td><td>0.00</td></tr><tr><td>MLP</td><td>0.17</td><td>0.01</td><td>0.09</td><td>0.00</td><td>0.23</td><td>0.08</td></tr><tr><td>GRU</td><td>0.68</td><td>0.01</td><td>1.21</td><td>0.37</td><td>0.75</td><td>0.00</td></tr></table>

TABLE VI

BDG2 BUILDING-ENERGY DATA. MEAN OVER SEEDS, p AS IN TABLE IV. A SIGNIFICANT p SAYS THE ERROR DROPS, FIGURE 2 SAYS WHETHER THE STATIC INPUT IS USED
<table><tr><td>Scheme</td><td>Static</td><td>RMSE</td><td>MAE</td><td> $R ^ { 2 }$ </td><td>p(static↓)</td></tr><tr><td>Central</td><td>√</td><td>0.382</td><td>0.222</td><td>0.883</td><td>一  $< . 0 0 1 ^ { \ast }$  –</td></tr><tr><td>Central</td><td>一</td><td>0.394</td><td>0.228</td><td>0.876</td><td></td></tr><tr><td>Per-batch FedAdam</td><td>√</td><td>0.382</td><td>0.220</td><td>0.880</td><td>0.007*</td></tr><tr><td>Per-batch FedAdam</td><td>一</td><td>0.390</td><td>0.225</td><td>0.875</td><td></td></tr><tr><td>FedAdam+Prox</td><td>√</td><td>0.505</td><td>0.348</td><td>0.796</td><td>0.640</td></tr><tr><td>FedAdam+Prox</td><td>一</td><td>0.511</td><td>0.349</td><td>0.797</td><td></td></tr><tr><td>FedAvg</td><td>√</td><td>0.521</td><td>0.361</td><td>0.791</td><td>0.865</td></tr><tr><td>FedAvg</td><td>一</td><td>0.519</td><td>0.356</td><td>0.796</td><td></td></tr><tr><td>SCAFFOLD</td><td>√</td><td>0.502</td><td>0.347</td><td>0.810</td><td>0.006*</td></tr><tr><td>SCAFFOLD</td><td></td><td>0.509</td><td>0.353</td><td>0.805</td><td></td></tr><tr><td>Local</td><td></td><td>0.513</td><td>0.298</td><td>0.802</td><td></td></tr></table>

FedAdam+Prox gain nothing significant, and SCAFFOLD only appears to.

SCAFFOLD splits the two measurements apart. It reaches RMSE 0.502 with the static input against 0.509 without, a significant contrast $( p = 0 . 0 0 6 )$ , yet its sensitivity is 0.002, as silent as FedAvg. The model without the static input has no static encoder at all, so the significance test compares two architectures rather than two information states. A constant static input keeps the architecture and removes only the information. In a separate series of three seeds under FedAdam+Prox with one local epoch, the constant matches the real input (0.519 against 0.516, $p = 0 . 7 0 )$ , whereas under per-batch FedAdam the real input wins (0.375 against 0.388, $p < 0 . 0 0 1$ , two seeds). Where the probe is silent, the error difference comes from the additional parameters of the static encoder, which act as learned constants, and not from the entity information. At equal local computation the drift correction recovers nothing here. Shortening the local run keeps only a trace: FedAdam+Prox at $E = 0 . 1$ retains a sensitivity of 0.073 and gains nothing significant (RMSE 0.461, $p = 0 . 3 5 3 )$ . On wind power the same correction keeps the static pathway responsive and gains significantly (Section V-E).

Figure 3 breaks this down per building. Under per-batch FedAdam the static features move four buildings from worse to better (7 to 3 of 32 worse) and lift the median from 0.046 to 0.058. Under FedAvg the distribution does not move, and 26 of 32 buildings stay worse than local in both cases.

![](images/af678e68895256427702b03840c9a61f675dd6af17a5cfc1a585e4763bf0eb6d.jpg)  
Fig. 3. Per-building RMSE improvement over the local baseline on building energy, without and with static features. Boxes mark the quartiles and the median. Values above zero beat the local model

TABLE VII  
WIND POWER DATA. MEAN OVER SEEDS, p AS IN TABLE IV
<table><tr><td>Scheme</td><td>Static</td><td>RMSE</td><td>MAE</td><td> $R ^ { 2 }$ </td><td>p(static↓)</td></tr><tr><td>Central</td><td>√</td><td>0.094</td><td>0.055</td><td>0.761</td><td>&lt;.001*</td></tr><tr><td>Central</td><td>一</td><td>0.100</td><td>0.059</td><td>0.725</td><td></td></tr><tr><td>Per-batch FedAdam</td><td>√</td><td>0.095</td><td>0.056</td><td>0.761</td><td>&lt;.001*</td></tr><tr><td>Per-batch FedAdam</td><td>一</td><td>0.100</td><td>0.059</td><td>0.728</td><td></td></tr><tr><td>FedAdam+Prox</td><td>√</td><td>0.113</td><td>0.069</td><td>0.598</td><td>0.789</td></tr><tr><td>FedAdam+Prox</td><td>一</td><td>0.113</td><td>0.068</td><td>0.601</td><td></td></tr><tr><td>FedAvg</td><td>√</td><td>0.112</td><td>0.067</td><td>0.627</td><td>0.610</td></tr><tr><td>FedAvg</td><td>一</td><td>0.112</td><td>0.067</td><td>0.624</td><td></td></tr><tr><td>SCAFFOLD</td><td>√</td><td>0.107</td><td>0.065</td><td>0.672</td><td>0.003*</td></tr><tr><td>SCAFFOLD</td><td></td><td>0.110</td><td>0.066</td><td>0.628</td><td>一</td></tr><tr><td>Local</td><td></td><td>0.097</td><td>0.059</td><td>0.755</td><td></td></tr></table>

## E. Wind Power

Table VII shows the same pattern on wind power, with one informative exception. Central and per-batch FedAdam gain significantly from the static input, raising $R ^ { 2 }$ from about 0.73 to 0.76. FedAvg and FedAdam+Prox do not $( p > 0 . 1 )$ , and they also reach a lower $R ^ { 2 }$ overall. SCAFFOLD is the only local-epoch scheme that gains significantly from the static input here $( R ^ { 2 } 0 . 6 3 ~  ~ 0 . 6 7 , ~ p ~ = ~ 0 . 0 0 3 )$ , and unlike on building energy the probe agrees: it keeps a sensitivity of 0.18 (Figure 2). It takes about as many local steps per round as FedAvg, so the difference is its drift correction, which partly recovers the static pathway, though it still trails central, perbatch, and the local baseline. The local baseline is strong here $( R ^ { 2 } 0 . 7 5 )$ : FedAvg and FedAdam+Prox fall below it, central and per-batch FedAdam exceed it by using the static input.

## F. Localizing the Failure

Both diagnostics of Section III-D are measured on building energy. Over the four seeds common to all sweep runs, the RMSE rises with E for both schemes, from 0.486 to 0.575 for FedAvg and from 0.465 to 0.599 for FedAdam+Prox between

TABLE VIII  
COMMUNICATION COST PER SCHEME AND DATASET: DATA SENT AND RECEIVED PER CLIENT (GB/CL.) AND TOTAL (GB) OVER K CLIENTS (K=12 SYNTHETIC, K=32 APPLICATIONS)
<table><tr><td rowspan="2"></td><td colspan="2">Synthetic</td><td colspan="2">Building energy</td><td colspan="2">Wind power</td></tr><tr><td>GB/cl.</td><td>GB</td><td>GB/cl.</td><td>GB</td><td>GB/cl.</td><td>GB</td></tr><tr><td>Train data</td><td>0.0002</td><td>0.002</td><td>0.037</td><td>1.18</td><td>0.0028</td><td>0.090</td></tr><tr><td>Per-batch FedAdam</td><td>0.109</td><td>1.31</td><td>9.46</td><td>302.6</td><td>29.39</td><td>940.6</td></tr><tr><td>FedAvg</td><td>0.031</td><td>0.38</td><td>0.301</td><td>9.6</td><td>0.842</td><td>27.0</td></tr><tr><td>FedAdam+Prox</td><td>0.026</td><td>0.32</td><td>0.295</td><td>9.4</td><td>0.693</td><td>22.2</td></tr><tr><td>SCAFFOLD</td><td>0.038</td><td>0.45</td><td>0.360</td><td>11.5</td><td>1.375</td><td>44.0</td></tr></table>

$E = 0 . 2 5$ and $E \ = \ 2 .$ That is the ordinary cost of local steps. The static-input sensitivity separates the two schemes. FedAdam+Prox retains 0.073 at $E = 0 . 1$ and 0.030 at $E =$ 0.25, and falls below 0.01 from $E = 1 ~ \mathrm { o n }$ . FedAvg is already at 0.001 at $E = 0 . 2 5$ . Under FedAdam+Prox the sensitivity thus falls steadily with $E ,$ , without a threshold. FedAvg is silent after a quarter of an epoch, so any threshold there lies below it.

Static-path coupling does not restore the use of the static input. With the temporal pathway running full local epochs under FedAdam+Prox, the model reaches RMSE 0.539 with the static input against 0.556 without, a significant contrast $( p = 0 . 0 1 3 )$ , yet its sensitivity is 0.015 of the central value, as silent as under FedAvg. As for SCAFFOLD in Section V-D, the contrast comes from the additional parameters of the static encoder and not from the entity information. Synchronizing the encoder keeps its weights global, yet the static pathway stays silent, so the loss does not sit in the encoder alone. A variant that also aggregates the temporal pathway every 50 batches reaches 0.453 $( p = 0 . 0 2 4 )$ , yet FedAdam+Prox at $E = 0 . 1$ reaches 0.461 without any coupling, so that gain comes from the more frequent temporal aggregation rather than from the encoder. Coupling the encoder alone is not cheap either, at 186 % of the per-batch volume, because it synchronizes the encoder about 13 800 times over 44 epochs, whereas per-batch FedAdam stops after about 1 700 steps. We report static-path coupling as a diagnostic and not as a scheme to adopt.

## G. Communication Cost

The federated scheme that uses the static input fully is also the most expensive to communicate. We quantify this cost as the volume of data transmitted per client, since latency and bandwidth are deployment-specific. Table VIII reports the volume sent and received per client and in total for the static-input configuration, with the local training-set size as reference. Per-batch FedAdam synchronizes after every batch rather than once per round, so it transmits 31 times as much per client as FedAvg on building energy, 35 times as much on wind power, and 3.5 times as much on the small synthetic dataset. Wind power costs more than building energy because per-batch FedAdam trains longer there, about 6 500 against 1 700 steps.

## H. Discussion

Two observations cut across the three datasets. First, the heterogeneity across clients is severe: no scheme with local epochs clearly beats the per-client local baseline. The largest margin is SCAFFOLD on building energy, $R ^ { 2 } ~ 0 . 8 1 0$ against 0.802 and RMSE 0.502 against 0.513 (Tables IV, VI and VII). For these schemes, federating does not improve on local training.

Second, on the synthetic and the wind data, pooling more data raises accuracy only when the static input travels with it. This is clearest on the synthetic data, where the windows are mean-centered. Without the static input the pooled model collapses to $R ^ { 2 } 0 . 2 2 ,$ , far below the local baseline at 0.95, and the static input restores it. Wind power shows the same pattern, where the pooled static-free model fails to beat local training and only the static input pushes it past. Building energy is the exception, because the recent load history already identifies the entity and pooling helps even without the static input. Where the static input is the entity cue, the gains from more data are realized through the static pathway, which is exactly the pathway that FedAvg and FedAdam+Prox leave silent.

## VI. CONCLUSION

We asked whether a federated Temporal Fusion Transformer can learn from static, client-constant features. The answer depends on the aggregation scheme. Per-batch FedAdam uses the static input almost as well as central training. Communication-efficient schemes with local epochs, FedAvg and FedAdam+Prox, extract far less from it and gain nothing significant on either application dataset. Whether client drift or communication frequency causes the loss cannot be settled here: at the same local computation a drift-corrected SCAFFOLD keeps the static pathway responsive on wind power and gains significantly there, while on building energy it recovers nothing. Full recovery still needs per-batch FedAdam.

The effect holds across a synthetic setup and two domains, building energy and wind power, and it is not tied to the context vectors of the TFT: on the synthetic and the wind data two further architectures with different conditioning mechanisms lose most or all of their static-input sensitivity as well, and on building energy, where the static input improves the error only slightly, the GRU keeps 0.37 of 1.21. Two diagnostics, both measured on building energy, narrow it down further. The loss is graded in the number of local steps under a proximal term and complete under FedAvg, and coupling only the static encoder per batch leaves the static pathway silent. Per-batch FedAdam is therefore preferable whenever the static input carries entity information, at the cost of one synchronization per gradient step (Table VIII).

Two directions follow. First, on wind power drift correction was the only lever besides communication frequency that kept the static pathway responsive, and carrying server statistics into every local step, as Mime does [14], is the next test. Second, where the distribution of the static features is known, its standardization could be fixed from domain knowledge instead of aggregating client statistics. We leave this to future work, along with an evaluation under realistic latency and privacy constraints. More broadly, client-constant inputs deserve dedicated treatment in federated training.

## REFERENCES

[1] V. Walter, F. Stricker, A. Wagner, and C. Zirpins, “Towards Federated Learning-Based Forecasting of Renewable Energy Production,” in 2024 2nd International Conference on Federated Learning Technologies and Applications (FLTA). IEEE, 2024, pp. 128–135.

[2] S. Bose and K. Kim, “Federated short-term load forecasting with personalization layers for heterogeneous clients,” arXiv preprint arXiv:2309.13194, 2023.

[3] P. Montero-Manso and R. J. Hyndman, “Principles and algorithms for forecasting groups of time series: Locality and globality,” International Journal of Forecasting, vol. 37, no. 4, pp. 1632–1653, 2021.

[4] B. Lim, S. O. Arık, N. Loeff, and T. Pfister, “Temporal<sup>¨</sup> fusion transformers for interpretable multi-horizon time series forecasting,” International Journal of Forecasting, vol. 37, no. 4, pp. 1748–1764, 2021.

[5] G. W. Hart, “Nonintrusive appliance load monitoring,” Proceedings ofthe IEEE, vol. 80, no. 12, pp. 1870–1891, 1992.

[6] B. McMahan et al., “Communication-efficient learning of deep networks from decentralized data,” in Artificial intelligence and statistics. PMLR, 2017, pp. 1273–1282.

[7] V. Walter and A. Wagner, “Strategies to improve federated forecasts of day-ahead pv power production,” in 2025 3rd International Conference on Federated Learning Technologies and Applications (FLTA). IEEE, 2025, pp. 48–55.

[8] M. G. Arivazhagan, V. Aggarwal, A. K. Singh, and S. Choudhary, “Federated learning with personalization layers,” arXiv preprint arXiv:1912.00818, 2019.

[9] H. Wang and B. Ma, “QKCV attention: Enhancing time series forecasting with static categorical embeddings for both lightweight and pre-trained foundation models,” arXiv preprint arXiv:2510.20222, 2025.

[10] V. Lella et al., “Federated learning with temporal fusion transformer for interpretable soh prediction in ev batteries,” in 2025 IEEE 2nd International Conference on Green Industrial Electronics and Sustainable Technologies (GIEST). IEEE, 2025, pp. 1–7.

[11] A. Ahamed, R.-R. Kang, and K. Lee, “Fedtft: Federated temporal fusion transformer for interpretable multihorizon psychiatric risk prediction across cross-silo hospitals,” IEEE Journal of Biomedical and Health Informatics, 2026.

[12] S. P. Karimireddy et al., “SCAFFOLD: Stochastic controlled averaging for federated learning,” in International Conference on Machine Learning (ICML). PMLR, 2020, pp. 5132–5143.

[13] B. E. Woodworth, K. K. Patel, and N. Srebro, “Minibatch vs local sgd for heterogeneous distributed learning,”

Advances in Neural Information Processing Systems, vol. 33, pp. 6281–6292, 2020.

[14] S. P. Karimireddy et al., “Breaking the centralized barrier for cross-device federated learning,” Advances in Neural Information Processing Systems, vol. 34, pp. 28 663– 28 676, 2021.

[15] T. Li et al., “Federated optimization in heterogeneous networks,” in Proceedings of Machine Learning and Systems (MLSys), vol. 2, 2020, pp. 429–450.

[16] S. Reddi et al., “Adaptive federated optimization,” arXiv preprint arXiv:2003.00295, 2020.

[17] Y. Zhao et al., “Federated learning with non-IID data,” arXiv preprint arXiv:1806.00582, 2018.

[18] S. Lee, T. Zhang, and A. S. Avestimehr, “Layer-wise adaptive model aggregation for scalable federated learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 7, 2023, pp. 8491–8499.

[19] Y. Bao, M. Crawshaw, and M. Liu, “Provable benefits of local steps in heterogeneous federated learning for neural networks: A feature learning perspective,” in International Conference on Machine Learning (ICML), vol. 235. PMLR, 2024, pp. 2857–2902.

[20] H. Nguyen, H. Warrier, and Y. Gupta, “A novel approach for federated learning with non-iid data,” in 2022 9th International Conference on Soft Computing & Machine Intelligence (ISCMI). IEEE, 2022, pp. 62–67.

[21] J. Zhang et al., “Why are adaptive methods good for attention models?” Advances in Neural Information Processing Systems, vol. 33, pp. 15 383–15 393, 2020.

[22] E. Chen et al., “Gradient correction in federated learning with adaptive optimization,” arXiv preprint arXiv:2502.02727, 2025.

[23] X. Xu and G. Cormode, “FedPS: Federated data preprocessing via aggregated statistics,” arXiv preprint arXiv:2602.10870, 2026.

[24] L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.

[25] C. Miller et al., “The building data genome project 2, energy meter data from the ASHRAE great energy predictor III competition,” Scientific Data, vol. 7, no. 1, p. 368, 2020.

[26] V. Walter, “Synthetic simulation data of wind power time series in Germany,” https://data.mendeley.com/datasets/ xv3s82rvd2/3, 2026.

[27] V. Walter and A. Wagner, “A measurement- and ERA5- driven simulation framework for synthetic wind power time series with turbine aging,” SSRN 6346310, 2026.