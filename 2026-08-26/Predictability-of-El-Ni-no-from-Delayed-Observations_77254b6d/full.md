# Predictability of El Ni˜no from Delayed Observations

Francisco J. Beron-Vera Department of Atmospheric Sciences Rosenstiel School of Marine, Atmospheric & Earth Science University of Miami Miami, FL 33149, USA fberon@miami.edu

Started: August 6, 2026. This version: August 26, 2026.

## Abstract

Using monthly Ni˜no-3.4 anomalies through July 2026, we investigate how much predictive information is contained in delayed observations of the index. Ridge regression identifies informative delays, while multilayer perceptron and sparse identification of nonlinear dynamics (SINDy) models test whether nonlinear complexity provides additional direct forecast skill; gated recurrent unit (GRU) and long short-term memory (LSTM) networks provide a complementary test in which the temporal representation is learned internally. Delayed observations substantially improve forecasts over persistence and climatology at leads of up to six months, but increasing model complexity provides no systematic improvement. Historical recursive experiments favor a simple explicit SINDy recurrence and select shallow recurrent architectures, with no appreciable gain from learning the temporal representation internally. These results support a compact predictive representation of Ni˜no-3.4 evolution in which the representation of past information is more consequential than model complexity. As a prospective application, the selected models are used to forecast the developing 2026 event beyond the last available observation and to compare its predicted evolution with completed historical El Ni˜no events.

## Plain Language Summary

El Ni˜no prediction often draws on many oceanic and atmospheric variables. This study examines how much predictive information is already contained in the past history of the Ni˜no-3.4 index. A small number of delayed observations substantially improves forecasts, while more complex nonlinear and recurrent models provide little additional skill. The results indicate that representing the recent history of Ni˜no-3.4 is more important for prediction than increasing model complexity.

## Key Points

• A small number of delayed Ni˜no-3.4 observations provides substantial forecast skill relative to persistence and climatology.

• Nonlinear MLP and SINDy models provide little improvement over linear ridge regression once informative delayed coordinates are included.

• GRU and LSTM networks that learn temporal representations internally provide no gain over a simple delayed-coordinate recurrence.

## 1 Introduction

The El Ni˜no–Southern Oscillation (ENSO) is the dominant mode of interannual tropical climate variability and a major source of predictability for climate anomalies worldwide (Dijkstra and Burgers, 2002; Clarke, 2014). Its evolution reflects coupled ocean–atmosphere feedbacks operating over a range of time scales. In particular, conceptual models of ENSO have long emphasized memory and delayed negative feedback. The delayed-oscillator picture represents the influence of basin-scale equatorial-wave adjustment through an explicit time delay (Suarez and Schopf, 1988; Battisti and Hirst, 1989), whereas recharge–discharge descriptions emphasize the slower evolution of equatorial upper-ocean heat content (Jin, 1997). Despite their diferent physical interpretations, both imply that the present seasurface-temperature anomaly alone need not contain all the information relevant to its subsequent evolution.

The role of memory in ENSO motivates an examination of the predictive information contained in past values of the observed Ni˜no-3.4 sea-surface-temperature anomaly. Delayed observations of a scalar variable can, under appropriate conditions, provide coordinates for an underlying dynamical state (Takens, 1981). We do not assume that a finite set of Ni˜no-3.4 delays reconstructs the ENSO state, nor do we identify individual delays with particular oceanic adjustment processes. Our aim is to determine empirically whether a small collection of past Ni˜no-3.4 anomalies provides useful predictive coordinates, what memory ranges emerge, and whether forecast skill saturates as additional delayed observations are included.

Data-driven methods have become increasingly prominent in ENSO prediction (Dijkstra et al., 2019; Wang et al., 2023). Much of this work seeks long-lead forecast skill by exploiting spatially distributed oceanic and atmospheric information. Convolutional neural networks trained on tropical Pacific sea-surface-temperature fields, supplemented by climate-model simulations, have demonstrated skill at long leads (Ham et al., 2019), while more recent work has used deep learning to identify the oceanic information underlying such predictability (Chen et al., 2025). We use only the monthly Ni˜no-3.4 index to examine how much of its predictability can be recovered from its own delayed history.

Following the delayed-coordinate forecasting framework introduced by Beron-Vera et al. (Beron-Vera et al., 2026), ridge regression (Hoerl and Kennard, 1970) is used to identify forecast-relevant delayed coordinates, while multilayer perceptron (MLP) (Rumelhart et al., 1986) and sparse identification of nonlinear dynamics (SINDy) (Brunton et al., 2016) models test whether nonlinear maps extract additional information from those co ordinates. Gated recurrent unit (GRU) (Cho et al., 2014) and long short-term memory (LSTM) (Hochreiter and Schmidhuber, 1997) networks broaden the comparison by learning the temporal representation internally from a consecutive sequence rather than using explicitly selected delayed coordinates.

This comparison separates two sources of forecasting complexity: the representation of the state and the map acting on that representation. It also allows an explicit evolution law to be sought with SINDy rather than treating forecasting solely as a black-box prediction problem. Using monthly Ni˜no-3.4 through July 2026, we find that forecast skill saturates with a small number of delayed coordinates and that nonlinear and recurrent complexity provides no systematic improvement. Historical El Ni˜no experiments favor a simple explicit SINDy recurrence, while pseudo-real-time hindcasts select shallow recurrent architectures. The selected models are then applied prospectively to the developing 2026 event.

## 2 Methods

We use the monthly Ni˜no-3.4 sea-surface-temperature anomaly from the NOAA Climate Prediction Center, based on ERSSTv6 (Huang et al., 2025a,b), denoted by N(t), from December 1949 through July 2026. The index is the sea-surface-temperature anomaly averaged over $5 ^ { \circ } \mathrm { S } - 5 ^ { \circ } \mathrm { N } , 1 7 0 ^ { \circ } \mathrm { W } - 1 2 0 ^ { \circ } \mathrm { W }$ , with centered climatological base periods. We work directly with this monthly index, which provides the monthly input to the Oceanic Ni˜no Index (ONI), rather than ONI itself, whose overlapping three-month averages introduce temporal smoothing before delayed observations are considered. For a forecast lead $\Delta .$ , we seek an empirical predictive map

$$
\begin{array} { r } { \widehat { N } ( t + \Delta ) = F _ { \Delta } \big ( \mathbf { z } ( t ; T ) ; \theta \big ) , \qquad \mathbf { z } ( t ; T ) = \big ( N ( t - \tau _ { 1 } ) , \dots , N ( t - \tau _ { m } ) \big ) , } \end{array}\tag{1}
$$

where $\mathcal { T } = \{ \tau _ { 1 } , . . . , \tau _ { m } \}$ , with $\tau _ { 1 } = 0$ , specifies the delayed coordinates and θ denotes the model parameters. The use of delayed coordinates is motivated by state-reconstruction ideas, but the existence of a predictive map for the particular finite lag sets considered here is not implied by Takens’ embedding theorem (Takens, 1981). Whether these delayed coordinates provide useful predictive information is instead assessed empirically.

We first use ridge regression both as a forecasting model and to identify informative delayed coordinates. Candidate delays are drawn from {0, 1, 2, 3, 4, 6, 9, 12, 15} months, with the instantaneous coordinate always included. For each number m of coordinates, all admissible combinations of the remaining delays are considered, and the best-performing lag set is retained. We then use the ridge-selected coordinates to test whether nonlinear dependence provides additional predictive information with MLP and SINDy. The corresponding realizations of (1) may be summarized as

$$
\begin{array} { r l } & { F _ { \Delta } ^ { \mathrm { r i d g e } } ( \mathbf { z } ) = \alpha _ { 0 } + \alpha \cdot \mathbf { z } , } \\ & { F _ { \Delta } ^ { \mathrm { M L P } } ( \mathbf { z } ) = \mathbf { w } _ { 2 } \cdot \sigma ( \mathbf { W } _ { 1 } \mathbf { z } + \mathbf { b } _ { 1 } ) + b _ { 2 } , } \\ & { F _ { \Delta } ^ { \mathrm { S I N D y } } ( \mathbf { z } ) = \Theta ( \mathbf { z } ) \pmb { \xi } , } \end{array}\tag{2}
$$

where the ridge coeficients are obtained by L<sub>2</sub>-regularized least squares; $\mathbf { W } _ { 1 } , \mathbf { w } _ { 2 }$ and $\mathbf { b } _ { 1 } , b _ { 2 }$ are the MLP weights and biases, respectively, and σ is a nonlinear activation function; and Θ is a library of candidate polynomial functions whose coeficient vector $\boldsymbol { \xi }$ is constrained to be sparse. The MLP expression in (2) illustrates a single hidden layer, with deeper networks obtained by composing additional layers. Polynomial SINDy libraries through degree three are considered. Thus these three approaches act on the same explicitly prescribed delayed representation while ranging from linear to flexible and sparse nonlinear forecast maps.

We also consider GRU and LSTM networks, which instead learn an internal representation of temporal dependence from a consecutive sequence of past observations. Writing

$$
\mathbf { x } _ { t } = \mathbf { z } ( t ; \{ 0 , \delta , \ldots , ( q - 1 ) \delta \} ) = ( N ( t ) , N ( t - \delta ) , \ldots , N ( t - ( q - 1 ) \delta ) ) ,\tag{3}
$$

with $\delta = 1$ month, for an input sequence of q consecutive observations, their forecasts may be represented schematically as

$$
\mathbf { h } _ { t } ^ { R } = \mathcal { R } _ { R } ( \mathbf { x } _ { t } ; \theta _ { R } ) , \qquad \widehat { N } ( t + \Delta ) = F _ { \Delta } ^ { R } ( \mathbf { x } _ { t } ) = \mathbf { w } _ { R } \cdot \mathbf { h } _ { t } ^ { R } + b _ { R } , \qquad R = \mathrm { G R U ~ o r ~ L S T M } ,\tag{4}
$$

where $\mathcal { R } _ { R }$ denotes the recurrent processing of the input sequence, $\mathbf { h } _ { t } ^ { R }$ is the resulting hidden state, and $\theta _ { R }$ collects the learned weights and biases of the recurrent network. Both architectures carry information from one observation to the next through their internal states. A GRU uses update and reset gates, whereas an LSTM additionally maintains a cell state controlled by input, forget, and output gates. Thus, unlike ridge, MLP, and SINDy, for which temporal information is supplied explicitly through selected delays, GRU and LSTM learn a representation of the past internally through recurrence. We consider shallow networks with one recurrent layer and, as a robustness test, deep (i.e., stacked) architectures of increasing depth.

For ridge regression and SINDy, the record is divided chronologically into 70% training and 30% test samples. Neural-network models use a 70%/15%/15% training/validation/test split, with the validation interval used for early stopping and model selection. Forecast skill is measured relative to persistence and zero-anomaly climatology according to

$$
R _ { p } = { \frac { \mathrm { R M S E } _ { \mathrm { m o d e l } } } { \mathrm { R M S E } _ { \mathrm { p e r s i s t e n c e } } } } , \qquad R _ { c } = { \frac { \mathrm { R M S E } _ { \mathrm { m o d e l } } } { \mathrm { R M S E } _ { \mathrm { c l i m a t o l o g y } } } } ,\tag{5}
$$

so that values below unity indicate improvement over the corresponding reference. Persistence provides the natural short-lead benchmark, although its skill deteriorates with lead. Zero-anomaly climatology instead provides a fixed reference whose RMSE measures the typical amplitude of the observed anomalies, so that $R _ { c } < 1$ indicates model errors smaller than this intrinsic variability. Neural-network results are summarized across independent random initializations rather than selecting individual realizations according to test performance.

Direct forecasts in the “open-loop” configuration use observed delayed coordinates independently at each target time. To forecast beyond the final observation in July 2026, we instead iterate the one-month models recursively in “closed loop,” replacing unavailable observations in the delayed state by previous model predictions. Models with similar direct skill can behave diferently under iteration, so recursive behavior is examined using completed historical El Ni˜no events, defined by $N ( t ) \geq 0 . 5 ^ { \circ } \mathrm { C }$ for at least five consecutive months. For SINDy, these historical hindcasts provide a robustness diagnostic among polynomial degrees having comparable direct skill and motivate the degree-one map used for recursion.

For GRU and LSTM, architecture is selected through pseudo-real-time hindcasts. At each historical El Ni˜no event, the network is trained on all admissible input sequences available up to the forecast origin, with the final 15% of these data reserved for validation and early stopping, and is then iterated recursively over the subsequent three months. The input sequence length is fixed, but the number of training sequences increases for later historical events as the available record expands. Each hindcast nevertheless remains out of sample relative to the data used to train the corresponding network. The three-month hindcast errors are combined across historical events using weights that decrease with the delayed-state distance between each forecast origin and the July 2026 initialization. The resulting similarity-weighted RMSE favors architectures that predict historical events accurately while giving greater weight to states resembling the current initialization; the architecture minimizing this RMSE is selected. After architecture selection, the chosen networks are retrained using the full record available through July 2026, with the same validation procedure, before producing the current recursive forecast.

## 3 Results

Forecast skill improves rapidly as delayed observations are added and then saturates with a small number of coordinates. The best ridge models at forecast leads $\Delta = 1 , 3 ,$ and 6 months contain 4, 6, and 5 coordinates, respectively (Table 1), with lag sets $\{ 0 , 1 , 3 , 4 \}$ $\{ 0 , 3 , 6 , 9 , 1 2 , 1 5 \}$ , and {0, 1, 3, 9, 15} months. Short delays of a few months contain substantial forecast-relevant information, while delays of 9–15 months provide additional information at the longer leads. Beyond a few coordinates, however, the improvement is small. As forecast lead increases, improvement over persistence remains substantial $( R _ { p } < 1 )$ whereas skill relative to zero-anomaly climatology $( R _ { c } < 1 )$ progressively decreases. At 9–12 months the forecasts continue to outperform persistence but provide little improvement over climatology, with $R _ { c }$ close to unity. We therefore focus below on leads through six months, for which the delayed representation retains appreciable skill relative to both references.

<table><tr><td rowspan="2"> $\Delta$  [months]</td><td rowspan="2">m</td><td rowspan="2"> $\tau$  [months]</td><td colspan="2">Ridge</td><td colspan="2">MLP</td></tr><tr><td> $R _ { p }$ </td><td> $R _ { c }$ </td><td> $R _ { p }$ </td><td> $R _ { c }$ </td></tr><tr><td>1</td><td>4</td><td>{0, 1, 3, 4}</td><td>0.875</td><td>0.245</td><td>0.898</td><td>0.252</td></tr><tr><td>3</td><td>6</td><td>{0, 3, 6, 9, 12, 15}</td><td>0.805</td><td>0.532</td><td>0.808</td><td>0.533</td></tr><tr><td>6</td><td>5</td><td>{0, 1, 3, 9, 15}</td><td>0.757</td><td>0.821</td><td>0.756</td><td>0.819</td></tr></table>

Table 1: Direct forecast skill using ridge-selected delayed coordinates. The number of coordinates m and lag set $\tau$ are selected by ridge regression. MLP results are medians over ten independent initializations.

Replacing the linear ridge map by an MLP acting on the same coordinates does not systematically improve the forecast (Table 1). In contrast, an instantaneous MLP using only $N ( t )$ gives $R _ { p } = 1 . 0 0 9 , 0 . 9 5 8$ , and 0.863 at forecast leads $\Delta = 1 , 3$ , and 6 months, respectively. Thus the substantial gain comes from the delayed representation rather than from nonlinear flexibility of the forecast map.

SINDy allows polynomial interactions among the delayed coordinates while retaining a sparse explicit map. Direct SINDy forecasts outperform persistence and climatology at forecast leads $\Delta = 1 , 3$ , and 6 months, but provide no systematic evidence that greater nonlinear complexity improves forecast skill. For the $\Delta \ : = \ : 1$ month forecast, which is subsequently used for recursive prediction, the ridge-selected delayed state is

$$
\mathbf { z } ( t ) = \big ( N ( t ) , N ( t - 1 ) , N ( t - 3 ) , N ( t - 4 ) \big ) ,\tag{6}
$$

where, hereafter, integer time shifts are understood to be in months. The consistency of ridge, MLP, and SINDy indicates that much of the useful predictive information resides in a compact delayed representation of the scalar record rather than in the complexity of the map acting on it.

Recursive forecasting places a stronger constraint on model complexity. For $\mathrm { S I N D y }$ polynomial degrees one, two, and three have similar direct skill at forecast lead $\Delta = 1$ month,

$$
R _ { p } = 0 . 8 9 2 , \qquad 0 . 8 9 7 , \qquad 0 . 9 1 6 ,
$$

respectively, so direct skill alone provides little basis for selecting among them. Their behavior under recursive iteration, however, is markedly diferent. Similarity-weighted three-month RMSEs over the completed historical El Ni˜no events are $0 . 5 1 4 ^ { \circ } \mathrm { C } , 1 2 5 . 6 ^ { \circ } \mathrm { C }$ and $1 2 . 3 ^ { \circ } \mathrm { C }$ for degrees one, two, and three, respectively. The large errors of the nonlinear maps result from occasional recursive instabilities; for example, a degree-two hindcast initialized in October 1951 evolves from $1 . 2 1 ^ { \circ } \mathrm { C }$ to 3.00, 17.82, and $1 0 3 2 . 4 2 ^ { \circ } \mathrm { C }$ over the subsequent three months. The historical recursive robustness test therefore favors $p = 1$

![](images/188ea7be3b6b81866d6470ccc4c69cb27289f9fde24ca70c8721473c6db6e54d.jpg)

![](images/456953cb9dee0b29d80538eab71883a2ac6439e5c67c55e236a0c9e9afd346d5.jpg)

![](images/ecbbf6e4246c4e44a0a3c3953c52b21630d6897d658259d103c4ca47a0f14a14.jpg)  
Figure 1: Historical three-month recursive SINDy hindcasts obtained by iterating the $\Delta = 1$ month map. (a) Event with the smallest hindcast RMSE. (b) Event with the largest hindcast RMSE. (c) Mean observed and forecast evolution over the 20 historical El Ni˜no events as a function of time from the forecast origin; error bars denote one standard deviation across events. Forecast origins are selected using the same delayed-state similarity criterion employed for the July 2026 initialization.

Refitting the selected structure using all observations through July 2026 gives the explicit $\Delta = 1$ month map

$$
\begin{array} { c } { { \widehat { N } ( t + 1 ) = 0 . 0 0 4 6 6 7 + 1 . 0 8 2 9 4 6 N ( t ) - 0 . 0 6 0 2 9 2 N ( t - 1 ) } } \\ { { - 0 . 0 2 7 0 0 1 N ( t - 3 ) - 0 . 1 0 3 9 6 5 N ( t - 4 ) . } } \end{array}\tag{7}
$$

Thus, although polynomial libraries through degree three were allowed, the model favored for recursive forecasting is a linear recurrence involving only the four delayed coordinates identified above.

Figure 1 illustrates the historical recursive performance of this map. The best threemonth hindcast occurs for the 1986–88 event, with an RMSE of $0 . 0 5 4 ^ { \circ } \mathrm { C } .$ , whereas the largest error occurs for the 1951 event, with an RMSE of $0 . 9 2 7 ^ { \circ } \mathrm { C }$ . Across all 20 events, the mean forecast closely follows the mean observed evolution. At recursive forecast horizons of one, two, and three months, the mean signed errors (biases) across the 20 events are $0 . 0 8 4 , - 0 . 0 1 5$ , and $- 0 . 0 0 9 ^ { \circ } \mathrm { C }$ , respectively, while the corresponding RMSEs, which measure the magnitude of the forecast errors, are 0.318, 0.489, and $0 . 6 5 0 { } ^ { \circ } \mathrm { C }$ The recursive map therefore exhibits little systematic over- or underprediction across the historical events, although the magnitude of individual-event errors increases with forecast horizon.

The recurrent-network experiments test the same recursive problem without prescribing the temporal representation through selected delayed coordinates. GRU and LSTM networks use five-month input sequences $N ( t - 4 ) , \ldots , N ( t )$ , spanning the same maximum four-month history as (6) but additionally including $N ( t - 2 )$ . In the pseudo-real-time historical experiments, the sequence length remains fixed while the number of available training sequences increases with the forecast date as the training record expands. Each historical hindcast remains out of sample relative to the observations used to train the corresponding network.

<table><tr><td>Model</td><td>Selected complexity</td><td>RMSE [°C]</td></tr><tr><td>SINDy</td><td> $p = 1$ </td><td>0.514</td></tr><tr><td>GRU</td><td> $h = 3 2$ </td><td>0.530</td></tr><tr><td>LSTM</td><td> $h = 1 6$ </td><td>0.524</td></tr><tr><td>Deep GRU</td><td>[64, 32, 16, 8, 4, 2]</td><td>0.622</td></tr><tr><td>Deep LSTM</td><td>[32, 16, 8, 4, 2]</td><td>0.617</td></tr></table>

Table 2: Historical three-month recursive model comparison. RMSE denotes the similarityweighted error over completed historical El Ni˜no events, with greater weight assigned to forecast-origin delayed states closer to the July 2026 initialization. Recurrent-network forecasts are summarized over independent random initializations.

Among shallow GRUs with $h = 2 , 4 , 8 , 1 6 , 3 2$ , and 64 hidden units, historical recursive skill is best for $h = 3 2$ , with a similarity-weighted three-month RMSE of $0 . 5 3 0 ^ { \circ } \mathrm { C }$ (Table 2). The corresponding LSTM experiment selects $h = 1 6$ , with RMSE $0 . 5 2 4 ^ { \circ } \mathrm { C } .$ . These errors are close to the $0 . 5 1 4 ^ { \circ } \mathrm { C }$ obtained with the degree-one SINDy map. Allowing the temporal representation itself to be learned internally therefore provides no appreciable improvement in historical recursive skill. The best deep GRU and LSTM architectures have somewhat larger similarity-weighted errors, $0 . 6 2 2 ^ { \circ } \mathrm { C }$ and $0 . 6 1 7 ^ { \circ } \mathrm { C }$ , respectively.

The distinction between shallow and deep recurrent networks is not uniform across events: in both families, 11 of the 20 historical events favor the selected shallow model and 9 favor the deep model. The shallow models nevertheless have lower similarity-weighted aggregate errors and perform better for five of the six historical forecast origins having the smallest delayed-state distance from the July 2026 initialization. These results favor the shallow architectures for the prospective forecast, while the deep architectures provide a useful contrast for assessing dependence on recurrent architecture.

The historical experiments reinforce the direct-forecast results. A sparse linear recurrence on explicitly selected delayed coordinates performs comparably to recurrent networks that construct their temporal representations internally, while neither polynomial nonlinearity nor increased recurrent complexity provides a systematic gain. The principal improvement in forecast skill therefore arises from representing the recent history of the index rather than from increasing the complexity of the map acting on that information.

As a prospective application, the selected models are retrained using all observations available through July 2026 and iterated recursively for three months. SINDy and the selected shallow GRU and LSTM give closely agreeing forecasts, with the anomaly remaining near its July value through August–September before weakening in October. The deep recurrent models instead predict more rapid weakening, illustrating the dependence of the prospective forecast on recurrent architecture. Figure 2 places the preferred forecasts in the context of the 20 completed historical El Ni˜no events; among those events, the predicted evolution is most similar to the 1965–66 El Ni˜no.

Historical El Nin\~o events centered at peak  
![](images/b2dda9f5190bc0fbbcb8d94e1f9866b17311472203e78dedf0fd15442027630c.jpg)  
Figure 2: Prospective three-month forecasts initialized in July 2026, shown together with completed historical El Ni˜no events centered at their observed maxima (gray). The observed 2025–26 evolution is shown in black.

These results indicate that much of the forecast-relevant information in the Ni˜no-3.4 record is captured by a small number of past monthly observations. Direct forecast skill saturates with a compact delayed representation, and an MLP provides essentially no improvement over ridge regression on the same coordinates. SINDy likewise provides no systematic evidence that polynomial nonlinearity improves direct forecast skill, while historical recursive tests favor a linear recurrence. GRU and LSTM networks, which instead learn their temporal representations internally, provide no appreciable improvement in historical recursive skill, while deep architectures have larger aggregate historical errors than their shallow counterparts. The consistency across these distinct model classes supports a compact predictive representation of Ni˜no-3.4 evolution over the short-to-intermediate forecast leads and recursive horizons considered here.

## 4 Conclusions

The Ni˜no-3.4 record contains substantial predictive information in a small number of delayed observations. Ridge regression shows that this information saturates with a compact set of coordinates spanning short and longer memory scales, while MLP and SINDy models show no systematic gain from additional nonlinear complexity. Historical recursive experiments show that GRU and LSTM networks that learn their temporal representations internally do not improve appreciably on the explicit delayed-coordinate SINDy model.

Recursive robustness tests favor a simple linear SINDy recurrence over higher-degree polynomial maps. Pseudo-real-time hindcasts select shallow GRU and LSTM architectures, which have lower similarity-weighted historical errors than the selected deep architectures. As a prospective application, the SINDy and shallow recurrent models are applied to the developing 2026 event and give closely agreeing short-term forecasts, while the deep recurrent models illustrate the sensitivity of this forecast to recurrent architecture. Over the direct forecast leads and recursive forecast horizons considered here, a compact representation of the recent history of Ni˜no-3.4 appears more consequential for prediction than increasing model complexity.

## Acknowledgments

This work was initiated during the workshop Oceanograf´ıa en un Clima Cambiante: Procesos F´ısicos, Biol´ogicos y Pesqueros desde la Teor´ıa, la Observaci´on y la Modelaci´on, held at Universidad del B´ıo-B´ıo, Concepci´on, Chile, in August 2026. The author gratefully acknowledges the hospitality of Universidad del B´ıo-B´ıo and the workshop organizers.

## Conflict of Interest

The author declares no conflicts of interest.

## Availability Statement

The monthly Ni˜no-3.4 anomaly record used in this study is publicly available from the NOAA Climate Prediction Center at https://www.cpc.ncep.noaa.gov/data/indices/ detrend.nino34.ascii.txt. MATLAB codes used for the analysis are available from the author upon reasonable request.

## References

Battisti, D. and Hirst, A. (1989). Interannual variability in a tropical atmosphere-ocean model: Influence of the basic state, ocean geometry and nonlinearity. J. Atmos. Sci., 46:1,687–1,712.

Beron-Vera, F. J., Olascoaga, M. J., and Miron, P. (2026). Loop Current extension as an efective delayed dynamical system. Nonlinear Dynamics, submitted. arXiv:2605.30603.

Brunton, S. L., Proctor, J. L., and Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proc. Nat. Acad. Sci. USA, 113:3932–3937.

Chen, Y., Chen, X., Jin, Y., Zhao, Y., Dong, J., Gan, Y., and Bi, H. (2025). The hidden predictor of multi-year ENSO predictions revealed by deep learning. Journal of Geophysical Research: Oceans, 130(7):e2025JC022394.

Cho, K., van Merri¨enboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724–1734.

Clarke, A. J. (2014). El Ni˜no physics and El Ni˜no predictability. Annual Review of Marine Science, 6:79–99.

Dijkstra, H. A. and Burgers, G. (2002). Fluid dynamics of El Ni˜no variability. Annual Review of Fluid Mechanics, 34:531–558.

Dijkstra, H. A., Petersik, P., Hern´andez-Garc´ıa, E., and L´opez, C. (2019). The application of machine learning techniques to improve El Ni˜no prediction skill. Frontiers in Physics, 7:153.

Ham, Y.-G., Kim, J.-H., and Luo, J.-J. (2019). Deep learning for multi-year ENSO forecasts. Nature, 573(7775):568–572.

Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9:1735–1780.

Hoerl, A. E. and Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1):55–67.

Huang, B., Yin, X., Boyer, T., Liu, C., Menne, M., Rao, Y. D., Smith, T., Vose, R., and Zhang, H.-M. (2025a). Extended reconstructed sea surface temperature, version 6 (ERSSTv6). part i: An artificial neural network approach. Journal of Climate, 38(4):1105–1121.

Huang, B., Yin, X., Boyer, T., Liu, C., Menne, M., Rao, Y. D., Smith, T., Vose, R., and Zhang, H.-M. (2025b). Extended reconstructed sea surface temperature, version 6 (ERSSTv6). part ii: Upgrades on quality control and large-scale filter. Journal of Climate, 38(4):1123–1136.

Jin, F.-F. (1997). An equatorial ocean recharge paradigm for ENSO. part i: Conceptual model. Journal of the Atmospheric Sciences, 54(7):811–829.

Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088):533–536.

Suarez, M. J. and Schopf, P. S. (1988). A delayed action oscillator for ENSO. Journal of the Atmospheric Sciences, 45:3283–3287.

Takens, F. (1981). Detecting strange attractors in turbulence, page 366–381. Springer Berlin Heidelberg.

Wang, G.-G., Cheng, H., Zhang, Y., and Yu, H. (2023). ENSO analysis and prediction using deep learning: A review. Neurocomputing, 520:216–229.