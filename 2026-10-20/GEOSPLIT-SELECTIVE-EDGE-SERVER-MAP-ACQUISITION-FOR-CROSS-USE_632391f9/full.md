# GEOSPLIT: SELECTIVE EDGE–SERVER MAP ACQUISITION FOR CROSS-USER TRAVEL-MODE RECOGNITION

Zepu Wang<sup>1,⋆</sup>, Luyao Niu<sup>2,⋆</sup>, Ruolin Li<sup>2</sup>, Jeff(Xuegang) Ban<sup>1,†</sup>

<sup>1</sup>University of Washington <sup>2</sup>University of Southern California

zepu@uw.edu, luyaoniu@usc.edu, ruolinl@usc.edu, banx@uw.edu

## ABSTRACT

Networked mobility sensing couples local trajectory inference with remote geospatial context, making communication decisions part of the prediction system. We present GeoSplit, an edge–server framework for travel-mode recognition that separates the utility of map evidence, its training-time transfer, and its selective acquisition at runtime. On GeoLife with 10% labels, we evaluate these components under user-day and userdisjoint grouping. For unseen users, aligned full-map context raises macro-F1 from 0.7154 to 0.7864, while map-informed distillation yields a motion-only student at 0.7225. Through a compact query interface, entropy gating achieves 0.7729 macro-F1 versus 0.7738 for all queries, querying 39.5% of trajectories and reducing application payload by 60.5%. Under user-day grouping, selective and all-query scores are similarly close at a 21.6% query rate. Learned value estimation and probability fusion offer no consistent improvement. These results illustrate communication-efficient edge–server inference for an intelligent transportation sensing task.

Index Terms— Networked mobility sensing, edge–server inference, selective communication, travel-mode recognition, user generalization

## 1. INTRODUCTION

Mobile sensing systems infer travel modes from GPS trajectories for passive surveys and mobility analytics [1, 2]. Motion features may confuse modes with similar movement patterns [3, 4]; geographic context from transit stops and railway lines can help resolve them [5]. When this context resides at a remote map service, recognition quality and communication cost become coupled, especially for participants absent from training. The deployment question is whether shared spatial information is worth its application payload for a particular trajectory.

An edge device can query the server for every trajectory or acquire context selectively. The latter requires estimating query value: local predictive uncertainty need not match the expected reduction in loss from server assistance. Remote context can correct one prediction and harm another. Thus, gains from always-available maps do not establish an efficient selective policy.

Related work studies task-oriented communication [6], cost-aware feature acquisition [7], selective prediction and deferral [8, 9], and distillation from privileged information [10, 11]. These mechanisms suggest three questions for networked mobility sensing: how useful is aligned remote context, how much can be transferred offline, and when should the device communicate? The last question matters for unseen users because uncertainty estimates can degrade under distribution shift [12].

We study these questions with GeoSplit, an edge–server framework that predicts locally from quantized motion features and requests compact map context for selected trajectories. We compare aligned and mismatched maps and motion-only students trained with different teachers, then evaluate query gates against application payload. Pipelines are fitted independently under user-day and user-disjoint grouping. At 10% labels, aligned maps improve unseen-user macro-F1 from 0.7154 to 0.7864, while a map-informed motion-only student reaches 0.7225. Entropy gating reaches 0.7729 versus 0.7738 for all compact queries while querying 39.5% of trajectories. This separates the utility of remote evidence from its selective acquisition.

## 2. COMMUNICATION-AWARE MAP ACQUISITION

## 2.1. Local evidence and the query interface

Figure 1 summarizes GeoSplit. For a segmented GPS trajectory $X = \{ ( \ell _ { t } , \tau _ { t } ) \} _ { t = 1 } ^ { T }$ with positions $\ell _ { t } ,$ timestamps $\tau _ { t } .$ , and mode $Y \in \{ 1 , \ldots , 5 \}$ , the device extracts motion statistics $z = \phi ( X )$ and computes $p _ { 0 } ( y \mid Q ( z ) )$ ) using a training-fitted quantizer $Q .$ . A query sends quantized motion and a position summary s(X) to a server holding map M:

$$
\begin{array} { c } { { u = \mathrm { e n c o d e } \big ( Q ( z ) , s ( X ) \big ) , } } \\ { { p _ { 1 } ( y ) = h _ { 1 } \big ( Q ( z ) , \widehat { m } ( s ( X ) , M ) \big ) . } } \end{array}\tag{1}
$$

Here $\widehat { m }$ extracts map features from transmitted geometry. Both predictors use the same decoded motion; the server adds spatial context and returns a one-byte class, $\widehat { Y } _ { 1 } = \arg \operatorname* { m a x } _ { y } p _ { 1 } ( y )$ . A gate $q ( I )$ using only local information selects $\widehat { Y } _ { 1 }$ when $q = 1$ and $\widehat { Y } _ { 0 } = \arg \operatorname* { m a x } _ { y } p _ { 0 } ( y )$ otherwise, before observing remote features.

![](images/9c41de588c4922944b372dfb4afa0abde64ccb0a8dc6676e2ed00fe625ec5613.jpg)  
Fig. 1. GeoSplit: local motion prediction with gated map-server assistance. Selected trajectories transmit compact motion and geometry; the server returns a class label. Offline calibration and distillation support evaluation of map utility, training-time transfer, and runtime acquisition.

## 2.2. Query value and selection

Let $R _ { 1 } ( X )$ count request-plus-reply application bytes, with zero query cost for local inference. We consider

$$
\operatorname* { m i n } _ { \boldsymbol { q } } \mathcal { R } ( \boldsymbol { q } ) = \mathbb { E } [ \ell ( Y , \widehat { Y } _ { \boldsymbol { q } } ) ] \quad \mathrm { s . t . } \quad \mathbb { E } [ \boldsymbol { q } ( I ) R _ { 1 } ( X ) ] \leq B .\tag{2}
$$

For fixed predictors, conditional query value gives

$$
\begin{array} { r l } & { \boldsymbol { v } ( I ) = \mathbb { E } [ \ell ( \boldsymbol { Y } , \widehat { Y } _ { 0 } ) - \ell ( \boldsymbol { Y } , \widehat { Y } _ { 1 } ) \mid I ] , } \\ & { \mathcal { R } ( \boldsymbol { q } ) = \mathcal { R } ( 0 ) - \mathbb { E } [ \boldsymbol { q } ( I ) \boldsymbol { v } ( I ) ] . } \end{array}\tag{3}
$$

Under zero–one loss, $v ( I )$ is the probability of correcting a local error minus that of introducing one. The pointwise Lagrangian rule queries when $v ( I ) > \lambda \mathbb { E } [ R _ { 1 } ( X ) \mid I ] \ [ 7 , 9 ]$ We evaluate macro-F1 alongside cost; it is not the additive loss in (2).

The reference gate thresholds local predictive entropy:

$$
q ( X ) = \mathbf { 1 } \left[ - \sum _ { c = 1 } ^ { 5 } p _ { 0 } ( c ) \log p _ { 0 } ( c ) > \theta \right] .\tag{4}
$$

Alternatives use confidence, margin, random selection, or a learned value score. Thresholds target 10, 25, 50, and 75% queries within the training partition and remain fixed on outer folds, where realized rates and costs may differ.

The value regressor uses local motion, class probabilities, entropy, margin, and confidence. Its target is local minus blended-query log-loss on labeled out-of-fold predictions, clipped to $[ - 5 , 5 ]$ . The blend is $\widetilde { p } _ { 1 } = ( 1 - \alpha ) p _ { 0 } + \alpha p _ { 1 }$ , with $\alpha \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ selected by calibration log-loss. Gates are compared with both direct and blended predictions. Primary cost comparisons use class-only replies; exploratory fusion requires probabilities and is not assigned this reply cost.

## 2.3. Motion and position encoding

Each of 110 motion statistics uses a four-bit empirical-quantile code: 55 bytes plus a four-byte header. Fine queries encode at most 11 points, selected by maximum-geometric-deviation refinement on a 50 m grid, in at most 36 bytes plus a seven-byte header. A two-byte frame and one-byte reply give a 105-byte total maximum. Coarse queries use at most four uniformly selected points on a 100 m grid and 81 bytes overall.

Map features comprise bus-stop and station distances, proximity ratios at 50 and 100 m, and railway-segment distances. The full-map reference uses original trajectories and adds median stopping-center distances, with a sentinel and missingcontext indicator. Classifiers are trained on their respective feature definitions, so full-map and compact-query comparisons reflect both geometry and feature differences. Geometry was selected in an earlier held-out comparison and is fixed here.

## 3. EVALUATION DESIGN

## 3.1. Data and participant protocols

We classify five GeoLife [3] modes: walk, bike, bus, car (including taxi), and subway. Annotations define segment boundaries. Restricting complete trajectories to Beijing yields 8,731 segments from 64 users. Grouped cross-validation uses only the 6,277-segment, 62-user training pool of an earlier 6,277/695/1,759 training/development/test user-day partition.

Two five-fold outer protocols control participant overlap. User-day grouping keeps each user’s same-day observations together and merges days linked by crossing trajectories; users can appear in multiple folds. User-disjoint grouping holds out entire users [13]. Each protocol independently fits models and acquisition rules, so comparisons also reflect differences in routes and sample composition.

A fixed stratified draw labels 10% of each outer-training pool. Three inner group folds generate out-of-fold predictions, refitting quantizers within each inner-training fold. Labeled groups are split 60/40 for acquisition training and calibration. Calibration selects temperature $T \in \{ 0 . 5 , 1 , 2 , 4 \}$ [14], fusion coefficients, and query thresholds; outer labels are used only for evaluation. A 100%-label control repeats the informationsource comparisons.

All map comparisons use a Geofabrik OpenStreetMap [15] snapshot dated September 19, 2026, buffered around Beijing: 23,930 bus-stop elements, 712 rail/subway stations, and 16,186 railway ways. Thus contemporary maps classify historical trajectories.

## 3.2. Spatial and transfer controls

Spatial controls include map-only prediction, motion plus a centroid on a 1 km grid, map lookup after a 1 km eastward trajectory shift, and map features shuffled separately within training and evaluation partitions. Aligned, shifted, and shuffled contexts use matched classifiers.

For offline transfer [10, 11], identical motion-only students regress local-teacher or full-map-teacher probabilities, then clip and normalize their outputs. Both use all outer-training trajectories, the same label budget, and cross-fitted teacher targets; labeled samples average these targets with one-hot labels. The teacher input is the only difference. The supervised motion-only predictor provides a further reference.

## 3.3. Models, metrics, and communication cost

Histogram gradient-boosting classifiers use 150 iterations, at most 15 leaves, minimum leaf size 15, learning rate 0.05, and $L _ { 2 } = 1$ . The value regressor uses 100 iterations, seven leaves, minimum leaf size 20, and $L _ { 2 } = 1 0 ;$ students use five regressors with classifier settings. We report pooled outer-fold macro-F1 and paired 95% intervals from 2,000 user-bootstrap resamples, conditional on fitted models.

Table 1. Spatial evidence and training-time transfer with 10% labels. Entries are pooled outer-fold macro-F1. Both students use only motion at inference.
<table><tr><td>Input or model</td><td>User-day</td><td>Unseen user</td></tr><tr><td>Motion only</td><td>0.8377</td><td>0.7154</td></tr><tr><td>Map only</td><td>0.6726</td><td>0.5602</td></tr><tr><td>Motion + coarse location</td><td>0.8404</td><td>0.7059</td></tr><tr><td>Motion + full map</td><td>0.8592</td><td>0.7864</td></tr><tr><td>Motion + shifted map</td><td>0.8388</td><td>0.7122</td></tr><tr><td>Motion + shuffled map</td><td>0.8336</td><td>0.7142</td></tr><tr><td>Local-teacher student</td><td>0.8264</td><td>0.6922</td></tr><tr><td>Map-teacher student</td><td>0.8458</td><td>0.7225</td></tr></table>

Payload includes requests and class replies, assuming shared models and codebooks. Central reporting adds one application byte per trajectory; query totals exclude survey metadata and reporting transport. We also consider fixed perquery overheads of 64 and 256 bytes. Local preloading requires a 2.01 MB compressed task map and a 16.8 kB motion codebook. Costs measure application bytes and storage, not wireless latency or energy.

## 4. RESULTS

## 4.1. Aligned remote context adds evidence

Aligned full-map context improves macro-F1 from 0.8377 to 0.8592 under user-day grouping and from 0.7154 to 0.7864 for unseen users (Table 1). Paired gains are 0.0215 [0.0119, 0.0438] and 0.0711 [0.0317, 0.1023], respectively; the larger unseen-user gain accompanies a lower motion-only score.

Coarse location does not reproduce this improvement, and shifted or shuffled maps largely remove it, supporting the value of spatial correspondence beyond the tested regional cue. Fullmap gains persist at 100% labels: 0.8843 to 0.9061 under user-day grouping and 0.8051 to 0.8364 for unseen users.

## 4.2. Offline transfer leaves a gap to runtime context

Map-teacher students outperform local-teacher students by 0.0195 and 0.0303 under user-day and user-disjoint grouping. For unseen users, their score of 0.7225 exceeds supervised motion-only prediction by 0.0071 but remains below the fullmap score of 0.7864. Thus the evaluated distillation procedure transfers only part of the map benefit, motivating runtime context access.

## 4.3. Selective communication approaches all-query performance

Uncertainty gates improve the observed recognition–payload tradeoff over random queries (Fig. 2). All fine queries reach

![](images/30f0e6f54c310dba6d3b22fe14bb4fadce30e386fd16d1399254ae7eb78a3efb.jpg)

![](images/a0d0ee577bdd19516c7dace57a702e8043173086b3aef299b526d8494dc33106.jpg)  
Fig. 2. Recognition–payload tradeoffs with 10% labels. Policies share predictors within each independently fitted protocol. Thresholds target 10/25/50/75% queries on inner calibration data; costs are realized outer-fold payloads. All-query scores use the fine interface.

Table 2. Observed operating points with 10% labels. Entropy percentages are calibration targets; q and mean bytes per trajectory are measured on outer folds. Query policies use the fine-query interface.
<table><tr><td>Grouping</td><td>Policy</td><td>q (%)</td><td>Bytes</td><td>F1</td></tr><tr><td rowspan="2">User-day</td><td>Local only All fine</td><td>0.0 100.0</td><td>0.00 104.39</td><td>0.8377 0.8535</td></tr><tr><td>Entropy 25% Entropy 50%</td><td>21.6 46.4</td><td>22.46 48.46</td><td>0.8539 0.8535</td></tr><tr><td rowspan="4">Unseen user</td><td>Local only</td><td>0.0</td><td>0.00</td><td>0.7154</td></tr><tr><td>All fine</td><td>100.0</td><td>104.39</td><td>0.7738</td></tr><tr><td>Entropy 25%</td><td>19.0</td><td>19.80</td><td>0.7638</td></tr><tr><td>Entropy 50%</td><td>39.5</td><td>41.21</td><td>0.7729</td></tr></table>

0.8535 under user-day grouping and 0.7738 for unseen users. Selection is compared against these same compact-query predictors.

At the 25% calibration target, user-day entropy gating queries 21.6% of trajectories and scores 0.8539 versus 0.8535 for all fine queries (Table 2). Payload falls from 104.39 to 22.46 bytes per trajectory, a 78.5% reduction; the paired F1 difference is 0.0004 [−0.0030, 0.0024].

For unseen users, the same target yields 19.0% queries and 0.7638 macro-F1. The 50% target yields 39.5% queries and 0.7729 versus 0.7738 for all fine queries, with a paired difference of −0.0009 [−0.0048, 0.0050]. Payload falls to 41.21 bytes, a 60.5% reduction. These are observed operating points, not minimum-budget estimates or equivalence guarantees.

All fine queries correct 279 local errors but introduce 135 for unseen users. This supports assessing the value of replacing a prediction, as in (3), rather than uncertainty alone.

## 4.4. Learned value and fusion offer no consistent advantage

At the 25% target, value gating scores 0.8455 and 0.7345 under user-day and user-disjoint grouping, below entropy’s 0.8539 and 0.7638, at higher payloads of 23.74 and 23.41 bytes. The ordering persists with matched blended predictions, including the value-target predictor. The evaluated regressor therefore offers no advantage over entropy under this training budget.

Fusion raises user-day entropy macro-F1 from 0.8539 to 0.8563 but lowers unseen-user F1 from 0.7638 to 0.7423. Calibration-log-loss selection thus does not consistently improve outer-fold macro-F1; acquisition and fusion require separate evaluation.

## 5. DISCUSSION AND CONCLUSION

GeoSplit shows that aligned map evidence improves recognition for unseen users, while offline distillation captures only part of its benefit. Selective compact queries nearly match all-query performance (0.7729 versus 0.7738 macro-F1) at a 39.5% query rate and 60.5% lower measured application payload. The user-day setting offers a lower-rate operating point, but participant overlap makes it a different deployment question from the unseen-user evaluation. The observed closeness does not establish equivalence for other participants or maps. These savings assume on-demand map access; a locally preloaded 2.01 MB map remains an alternative. The evaluation is limited to one dataset, annotation-based segmentation, contemporary maps, and one label draw per fold; bootstrap intervals are conditional on fitted models. Networked mobility sensing should therefore assess remote evidence, offline transfer, and runtime communication together under participantdisjoint evaluation.

## 6. REFERENCES

[1] Nadine Schuessler and Kay W. Axhausen, “Processing raw data from global positioning systems without additional information,” Transportation Research Record: Journal ofthe Transportation Research Board, vol. 2105, no. 1, pp. 28–36, 2009.

[2] Linlin Wu, Biao Yang, and Peng Jing, “Travel mode detection based on GPS raw data collected by smartphones: A systematic review of the existing methodologies,” Information, vol. 7, no. 4, pp. 67, 2016.

[3] Yu Zheng, Quannan Li, Yukun Chen, Xing Xie, and Wei-Ying Ma, “Understanding mobility based on GPS data,” in Proceedings ofthe 10th international conference on Ubiquitous computing, 2008, pp. 312–321.

[4] Sina Dabiri and Kevin Heaslip, “Inferring transportation modes from GPS trajectories using a convolutional neural network,” Transportation Research Part C: Emerging Technologies, vol. 86, pp. 360–371, 2018.

[5] Ji Li, Xin Pei, Xuejiao Wang, Danya Yao, Yi Zhang, and Yun Yue, “Transportation mode identification with GPS trajectory data and GIS information,” Tsinghua Science and Technology, vol. 26, no. 4, pp. 403–416, 2021.

[6] Jiawei Shao, Yuyi Mao, and Jun Zhang, “Learning taskoriented communication for edge inference: An information bottleneck approach,” IEEE Journal on Selected Areas in Communications, vol. 40, no. 1, pp. 197–211, Jan. 2022.

[7] Hajin Shim, Sung Ju Hwang, and Eunho Yang, “Why pay more when you can pay less: A joint learning framework for active feature acquisition and classification,” 2017, arXiv:1709.05964.

[8] Yonatan Geifman and Ran El-Yaniv, “SelectiveNet: A deep neural network with an integrated reject option,” in Proceedings ofthe 36th International Conference on Machine Learning. 2019, vol. 97 of Proceedings of Machine Learning Research, pp. 2151–2159, PMLR.

[9] Hussein Mozannar and David Sontag, “Consistent estimators for learning to defer to an expert,” in Proceedings ofthe 37th International Conference on Machine Learning. 2020, vol. 119 of Proceedings of Machine Learning Research, pp. 7076–7087, PMLR.

[10] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean, “Distilling the knowledge in a neural network,” 2015, arXiv:1503.02531.

[11] David Lopez-Paz, Leon Bottou, Bernhard Sch ´ olkopf, and¨ Vladimir Vapnik, “Unifying distillation and privileged information,” in International Conference on Learning Representations, 2016.

[12] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek, “Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift,” in Advances in Neural Information Processing Systems. 2019, vol. 32, Curran Associates, Inc.

[13] Sohrab Saeb, Luca Lonini, Arun Jayaraman, David C. Mohr, and Konrad P. Kording, “The need to approximate the use-case in clinical machine learning,” GigaScience, vol. 6, no. 5, 2017.

[14] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger, “On calibration of modern neural networks,” in Proceedings of the 34th International Conference on Machine Learning. 2017, vol. 70 of Proceedings of Machine Learning Research, pp. 1321–1330, PMLR.

[15] M. Haklay and P. Weber, “OpenStreetMap: Usergenerated street maps,” IEEE Pervasive Computing, vol. 7, no. 4, pp. 12–18, 2008.