# Learning and Forecasting Multiscale Spatiotemporal Epidemic Dynamics with Lagrangian–Eulerian Graph Neural ODEs

Jinyi Duan<sup>1</sup>, Qian Li<sup>3\*</sup>, Biao Tang<sup>1,2\*</sup>

<sup>1</sup> School of Mathematics and Statistics, Xi’an Jiaotong University, Xi’an, 710049, People’s Republic of China .

<sup>2</sup> Center for Intersection of Mathematics and Life Sciences, Xi’an Jiaotong University, Xi’an, 710049, People’s Republic of China . <sup>3</sup> School of Mathematics and Statistics, Xi’an University of Technology, Xi’an, 710054, People’s Republic of China .

\*Corresponding author(s). E-mail(s): liqian@xaut.edu.cn; biaotang@xjtu.edu.cn; Contributing authors: DJY1111@stu.xjtu.edu.cn;

## Abstract

Accurate medium- to long-range epidemic forecasting requires extrapolating high-dimensional, heterogeneous and nonstationary dynamics from partial observations. Yet hybrid models often leave implicit which observations inform which epidemic processes and how learned graph interactions modify system state. We introduce LE-EpiGNODE, which replaces generic message passing with process-specific, mechanism-guided neural operators: age-contact information drives Lagrangian infection generation, whereas mobility drives conservative Eulerian population transport within a structure-preserving universal diferential equation. We evaluated LE-EpiGNODE across 14 controlled synthetic mechanisms spanning time-modulated, state-dependent, nonlinear and coupled dynamics, three complementary COVID-19 settings, 21 comparator models and five targeted ablations. Forecasting gains tended to become more pronounced at longer horizons; at the longest horizon, LE-EpiGNODE achieved the lowest WMAPE in all three real-world datasets, reducing error by 17.7–40.0% relative to the best baseline. Controlled ablations linked these gains to the Lagrangian–Eulerian decomposition and process-aligned priors, while synthetic ground truth showed that learned operators and induced state fluxes track known generating dynamics. These results support mechanism-organized graph neural dynamics as a route to reliable long-horizon extrapolation with directly testable, process-level mechanistic interpretation beyond trajectory accuracy.

## Introduction

Reliable epidemic forecasts are critical for anticipating healthcare demand, allocating public-health resources, and adapting interventions before transmission conditions change. Yet accurate medium- to long-range prediction remains dificult because epidemic trajectories emerge from high-dimensional, nonstationary, and multiscale dynamics. Age-dependent contact patterns shape local infection generation [1, 2], whereas population mobility reorganizes spatial connectivity and redistributes epidemiological states [3]. These heterogeneous processes are coupled through a shared epidemic state, so changes at one scale can reshape transmission at another. Meanwhile, the underlying system is only partially observed, with incidence, contact, and mobility data providing complementary but incomplete views of latent states and interactions. Long-horizon forecasts therefore require models that extrapolate these coupled dynamics, rather than simply extend observed epidemic trajectories [4, 5].

Existing approaches address complementary parts of this problem. Mechanistic age-structured and metapopulation models explicitly organize contact-mediated transmission and mobility-driven coupling [6–8], but their interaction functions become dificult to specify as heterogeneity, dimensionality, and temporal variation increase. Graph-based spatiotemporal models instead learn nonlinear relational structure directly from data [9–13]. Hybrid models have connected these directions at progressively deeper levels: MepoGNN and M-Graphormer infer parameters, contact-related quantities, or propagation graphs that enter predefined epidemic dynamics [14, 15], whereas EARTH incorporates a learnable transmission graph into an epidemiology-aware Neural ODE [16]. More generally, universal diferential equations (UDEs) embed trainable unresolved functions within partially known governing equations [17]. In epidemic modeling, this paradigm has been used to learn dificult-to-specify spatiotemporal heterogeneous processes and unresolved nonlinear functions within compartmental dynamics, further supporting mechanistic reconstruction and longer-term prediction [18, 19]. These developments establish that neural flexibility and mechanistic epidemic structure can be combined at parameter, graph, and continuous-dynamics levels.

Despite this progress, two related issues remain. Heterogeneous observations are often combined within shared transmission representations, learned graphs, or parameter-estimation modules, leaving the correspondence between an information source and the physical process it directly constrains largely implicit. Mechanistic knowledge is also commonly introduced through predicted parameters, compartmental equations, or global training constraints, while the semantics and state action of graph messages themselves remain generic. Moving from discrete graph models to Neural ODEs or UDEs does not by itself resolve this issue: a continuous neural vector field can still obscure which physical process a learned interaction represents and how that interaction is permitted to modify the epidemic state. This distinction is consequential rather than merely interpretive. Age-structured contact and mobility constrain processes with fundamentally diferent state actions [2, 20]: infection generation creates compartmental transitions, whereas population transport redistributes existing states under conservation [21]. If both are absorbed into a common latent interaction mechanism, diferent learned components may compensate for one another while reproducing similar trajectories. Such compensation can remain hidden within the training regime but need not persist during extrapolation beyond it [22]; consequently, trajectory accuracy alone does not ensure dynamically faithful long-horizon prediction. The remaining methodological challenge is therefore to align heterogeneous information with process-specific operators and to let epidemic mechanisms determine the physical semantics and admissible state action of their graph messages.

These limitations motivate the two complementary design routes summarized in Fig. 1a. From the graph-learning side, spatiotemporal message passing can be extended toward continuous neural interactions, but these models still require governing-dynamics guidance to determine what a learned interaction means and how it may update the epidemic state. From the mechanistic side, compartmental ODEs can be extended to age- and spatially structured UDEs, but high-dimensional, state-dependent couplings across population groups and regions require a scalable mechanism for learning heterogeneous relational structure. The two routes therefore expose complementary deficiencies and converge on the same operator-level requirement: graph neural learning is needed to represent unresolved heterogeneous interactions, while structured governing dynamics are needed to determine the physical meaning and admissible state action of those learned interactions.

![](images/2c4cb955877dc5a41ee2803cd0d0da12dfa3fa16a68125a6f5a785d68f74c269.jpg)  
Fig. 1: Conceptual motivation and framework of LE-EpiGNODE. (a), Conceptual synthesis motivating LE-EpiGNODE. From the graph-learning side, flexible spatiotemporal message passing progresses toward continuous neural interactions but requires governing-dynamics guidance to define message semantics and admissible state updates. From the mechanistic side, compartmental models progress toward age- and spatially structured UDEs with learnable unresolved functions but require scalable graph neural learning to represent high-dimensional heterogeneous couplings. These reciprocal requirements converge on mechanism-guided graph neural operators embedded within structured governing dynamics. (b), LE-EpiGNODE instantiates this convergence through a Lagrangian infection-generation operator and an Eulerian population-transport operator embedded in a structured UDE. Graph neural modules learn state- and time-dependent interaction strengths, while the epidemic mechanism specifies process semantics, admissible state updates, and conservation structure.

We address this requirement with LE-EpiGNODE, a Lagrangian–Eulerian Epidemic Graph Neural Ordinary Diferential Equation framework (Fig. 1b). Mechanistic information enters at three connected levels: it defines distinct relational topologies for the two epidemic processes, assigns heterogeneous observations to the corresponding pathways, and prescribes how messages from each pathway may modify the epidemic state. Age-contact information informs a Lagrangian infection-generation operator, whereas mobility information informs an Eulerian population-transport operator. Both operators are nonlinear, state- and time-dependent components of the governing vector field and interact through a shared epidemic state. Lagrangian messages generate infections from infectious sources through prescribed compartmental transitions, whereas Eulerian messages conservatively redistribute the full compartmental state across regions. Graph neural networks therefore learn unresolved interaction strengths and temporal responses, while the epidemic mechanism fixes what each message represents and what state transformation it is permitted to induce. This mechanism-organized, operator-level message passing is the central methodological distinction of LE-EpiGNODE.

## Results

We evaluated LE-EpiGNODE across 14 controlled synthetic generation conditions, three complementary real-world COVID-19 settings, 21 mechanistic, sequence, graphbased and hybrid comparators, and five targeted ablations. The synthetic systems provide known generating dynamics for controlled tests of long-horizon extrapolation and dynamical fidelity, whereas the real-world datasets test age-structured, mobility-driven and coupled age–space forecasting under partial observation. Detailed data-generation, dataset, mobility-input and baseline configurations are provided in Supplementary Tables 1–8.

## Long-horizon forecasting across heterogeneous epidemic dynamics

We first examined whether the Lagrangian–Eulerian formulation can maintain forecasting accuracy as predictions extend beyond the observed epidemic phase. The central question is whether the learned operators can extrapolate changing interaction dynamics, including explicit temporal modulation, endogenous state feedback and nonlinear responses, rather than only continue the local trajectory near the forecast origin.

Synthetic epidemic forecasting. The synthetic benchmark deliberately varies dynamical form as well as structural scale. Its 14 generation conditions comprise four Lagrangian infection-generation mechanisms—the Baseline Mass-Action Contact Mechanism (BMCM), Time-Decaying Contact Mechanism (TDCM), Infection-Avoidant Contact Mechanism (IACM) and Nonlinear Mass-Action Contact Mechanism (NMCM)—and two Eulerian population-transport mechanisms, Baseline Risk-Aware Gravity Mobility (BRGM) and Controlled Risk-Aware Gravity Mobility (CRGM), together with their eight pairwise multiscale combinations. These mechanisms span baseline mass-action, explicit temporal modulation, endogenous state feedback and nonlinear responses across isolated and coupled dynamical settings. Representative trajectories of the isolated mechanisms are shown in Supplementary Fig. 1, while their complementary age- and region-level signatures in the multiscale systems are illustrated in Supplementary Fig. 2. Within each structural setting, all models were trained using the same training period and forecast origin and were evaluated at horizons of 1, 4, 12 and 20 weeks. This common protocol probes the transition from short-term trajectory continuation to long-horizon extrapolation under qualitatively diferent forms of epidemic nonstationarity and multiscale coupling; complete mechanism definitions and generation settings are provided in Supplementary Methods, section 1.1, and Supplementary Tables 1–3.

Figure 2 shows representative fitting and forecasting trajectories under distinct Lagrangian and Eulerian interaction mechanisms, while Extended Data Fig. 1 extends the analysis to all eight paired mechanisms in the multiscale setting. Because population structure and the observation protocol are held fixed within each structural setting, diferences in temporal response and post-peak behavior reflect the prescribed interaction dynamics. Table 1 summarizes the corresponding WMAPE comparisons across forecast horizons, while Extended Data Table 1 provides the complete MAE and WMAPE results for all comparator models.

Across the 12 setting–horizon combinations, LE-EpiGNODE achieved the lowest WMAPE in 10 cases. Here, the best-performing baseline refers to the comparator with the lowest WMAPE within the same epidemic setting and forecast horizon. In the Lagrangian setting, LE-EpiGNODE achieved the lowest error at 1, 4 and 12 weeks; at 20 weeks, LE-EpiGNODE and MepoSIR were nearly tied, with WMAPE values of

12.45% and 12.21%, respectively. In the Eulerian setting, LE-EpiGNODE achieved the lowest WMAPE at all four horizons. At 20 weeks, its WMAPE was 21.11%, compared with 61.63% for SIR, the best-performing baseline at that horizon, representing a 65.7% relative reduction in WMAPE. Across all comparator models evaluated at this horizon, the relative reduction ranged from 65.7% to 84.0%, indicating a substantial separation from the full baseline set under mobility-driven long-horizon extrapolation. In the multiscale setting, LE-EpiGNODE’s ranking improved with horizon, reaching first place at 4, 12 and 20 weeks. At 20 weeks, its WMAPE was 13.36%, compared with 25.52% for MepoSIR, the best-performing baseline at that horizon, representing a 47.6% relative reduction in WMAPE.

These results reveal an overall horizon-dependent pattern. The relative advantage of LE-EpiGNODE was larger at the medium and long horizons than at the shortest horizon in the Eulerian and multiscale settings. At the shortest horizon in the multiscale setting, several sequence and spatiotemporal graph baselines achieved lower errors. The strong performance of MepoSIR in the comparatively simple Lagrangian setting also shows that a well-matched mechanistic inductive bias can remain efective. This pattern is consistent with the central design hypothesis of LE-EpiGNODE, while the structural ablations below provide a more direct test of its Lagrangian–Eulerian organization.

Real-world epidemic forecasting. We next examined whether the forecasting performance observed in the controlled systems extends to real epidemic observations. We evaluated $\mathrm { L E - E p i G N O D E }$ on three COVID-19 datasets representing complementary epidemic settings: age-structured dynamics in Ontario, Canada; mobility-coupled spatial dynamics across three major cities in Hubei, China, without age stratification; and coupled age–space dynamics in Spain. All models used only observations available before the forecast origin. Dataset construction, forecasting windows and implementation details are described in Methods. Figure 3 presents representative fitting and forecasting trajectories across the three epidemic settings. Table 1 summarizes the corresponding forecasting performance, while Extended Data Table 2 provides the complete MAE and WMAPE comparisons across all evaluated forecast horizons and comparator models.

LE-EpiGNODE achieved the lowest WMAPE at every evaluated horizon in Ontario and Hubei. In Spain, LE-EpiGNODE ranked fourth at 7 days and third at 14 days, then ranked first at both 49 and 93 days. At the longest horizon in Ontario, LE-EpiGNODE achieved a WMAPE of 16.20%, compared with 21.76% for MCM-SEIR, the best-performing baseline at that horizon, representing a 25.6% relative reduction in WMAPE. In Hubei, the corresponding values were 15.87% for LE-EpiGNODE and 26.45% for MepoGNN, representing a 40.0% relative reduction. In Spain, LE-EpiGNODE achieved 20.43% compared with 24.81% for MCM-SEIR, representing a 17.7% relative reduction. Spain’s stronger relative performance at longer horizons parallels the synthetic multiscale pattern.

Forecasting in Hubei and Spain additionally requires population transport to be extrapolated beyond the observation window. The auxiliary mobility network predicts future origin–destination (OD) mobility indices from the evolving epidemic state and time without using future OD observations. Its WMAPE increases to 40.15% at 26 days in Hubei and 29.06% at 93 days in Spain (Extended Data Table 3; Extended Data Figs. 2 and 3), whereas the corresponding epidemic-forecast WMAPE values remain at 15.87% and 20.43%. This separation suggests that the Eulerian pathway can retain forecast-relevant transport structure despite imperfect pointwise prediction of future OD mobility values. Because mobility and epidemic WMAPE quantify diferent prediction targets, this comparison is interpreted qualitatively rather than as a direct error ratio.

![](images/4e8222edcf107f27ffcfa78727c82deff4b6484d01d38112bfe11b486abed1cb.jpg)

![](images/4bf69cb24dc54919a72dcd615174d42267c91a54f9e83535f840d0bcc9633fb1.jpg)

![](images/9e19d7dc247dcbc6aeda88a2bce7508b26063a32ba8f2578dd00358d4a57a8bd.jpg)

![](images/d0794ad12c064b044c0ffe3eab7549f46c5d3ebb0877352a9ee9d0b6e7724cf5.jpg)

![](images/c46446cab46510482c95d776d928419ab913f065e0cbb19f77a250a6e2554788.jpg)

![](images/7d7b0bd3cdb86f2335492a220405e71eed71408b1e415d80a37c79dee0cd61fc.jpg)

![](images/48e924ad632e0ce047cbbd6274cebc8d509d78f3ebff3b6b124cba819ac31763.jpg)

![](images/aa1041bc55f40a0c44d47f1e81e993e362c1f5e8991fc49f67de5fe7346b35ea.jpg)  
Fig. 2: Fitting and forecasting trajectories in the synthetic scenarios. (a), Lagrangian scenario, where infection generation is governed by age-structured transmission dynamics within regions. (b), Eulerian scenario, where population transport couples regions while local infection generation and removal remain active. The vertical dashed line marks the forecast origin, and the gray-shaded region denotes the out-of-sample forecasting period. Colored envelopes show post-hoc normal-reference residual bands centered on the model predictions. All models are trained using only observations before the forecast origin.

## Process decomposition and aligned priors drive forecasting gains

To identify which components underlie the long-horizon forecasting gains, we evaluated five targeted ablations of the Lagrangian–Eulerian decomposition, the two dynamical pathways and their corresponding priors, as defined in Methods. We first examined these efects in the controlled synthetic systems and then tested whether the same process–information correspondence persisted in the real-world datasets. Table 2 reports the complete MAE and WMAPE results across all evaluated forecast horizons for both synthetic and real-world settings.

Table 1: Forecasting performance across synthetic and real-world settings. Values are WMAPE (%); lower values indicate better forecasting. Each WMAPE entry is reported as LE-EpiGNODE / best-performing baseline for the corresponding setting and forecast horizon, with the lower value shown in bold. The corresponding baseline model and the rank of LE-EpiGNODE among all evaluated models are also reported. Shaded rows denote the longest evaluated horizon, for which the relative WMAPE reduction of LE-EpiGNODE against the best-performing baseline is shown; negative values indicate higher error for LE-EpiGNODE. Complete MAE and WMAPE results across all comparator models and forecast horizons are provided in Extended Data Tables 1 and 2.
<table><tr><td>Setting</td><td>Horizon</td><td>WMAPE</td><td>(%)</td><td>Best baseline</td><td>LE rank Reduction (%)</td></tr><tr><td colspan="6">(a) Synthetic scenarios</td></tr><tr><td rowspan="4"></td><td>Lagrangian 1 week</td><td>5.67 / 6.31</td><td>DCRNN</td><td></td><td>1</td></tr><tr><td>4 weeks</td><td>5.84 / 7.91</td><td>DCRNN</td><td>1</td><td></td></tr><tr><td>12 weeks</td><td>8.45</td><td>9.19 MepoSIR</td><td>1</td><td></td></tr><tr><td>20 weeks</td><td>12.45 / 12.21</td><td>MepoSIR</td><td>2</td><td>-2.0</td></tr><tr><td rowspan="3">Eulerian</td><td>1 week</td><td>13.41 / 26.82</td><td></td><td>Graph WaveNet</td><td>1</td></tr><tr><td>4 weeks</td><td>16.44</td><td>44.91 AGCRN</td><td></td><td>1</td></tr><tr><td>12 weeks</td><td>18.32</td><td>58.12 SIR</td><td>1</td><td></td></tr><tr><td rowspan="3">Multiscale</td><td>20 weeks</td><td>21.11</td><td>61.63 SIR</td><td>1</td><td>65.7</td></tr><tr><td>1 week</td><td>17.16</td><td>/13.23 DCRNN</td><td>6</td><td></td></tr><tr><td>4 weeks</td><td>13.50 27.17</td><td>DCRNN</td><td>1</td><td></td></tr><tr><td></td><td>12 weeks</td><td>10.65</td><td>23.04 MepoSIR</td><td>1</td><td></td></tr><tr><td></td><td>20 weeks</td><td>13.36 / 25.52</td><td>MepoSIR</td><td>1</td><td>47.6</td></tr><tr><td colspan="6">(b) Real-world COVID-19 datasets</td></tr><tr><td rowspan="3">Ontario</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>7 days</td><td>3.43 / 7.42 4.49</td><td></td><td>M-Graphormer</td><td>1</td></tr><tr><td>14 days 35 days</td><td>12.17 18.33</td><td>MCM-SEIR MCM-SEIR</td><td>1 1</td><td></td></tr><tr><td rowspan="3">Hubei</td><td>67 days</td><td>7.80 16.20</td><td>21.76</td><td>MCM-SEIR</td><td>1</td></tr><tr><td></td><td></td><td>19.22</td><td></td><td>25.6</td></tr><tr><td>7 days 14 days</td><td>17.15 8.07 14.41</td><td>AGCRN MepoGNN</td><td>1</td><td></td></tr><tr><td rowspan="3">Spain</td><td>26 days</td><td>15.87 / 26.45</td><td>MepoGNN</td><td>1 1</td><td>40.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>7 days 14 days</td><td>9.61 / 3.46 9.66 / 6.09</td><td>EINN EINN</td><td>4</td><td></td></tr><tr><td rowspan="3"></td><td>49 days</td><td></td><td>MepoGNN</td><td>3 1</td><td></td></tr><tr><td></td><td>14.09 / 17.63</td><td></td><td></td><td></td></tr><tr><td>93 days</td><td>20.43 / 24.81</td><td>MCM-SEIR</td><td>1</td><td>17.7</td></tr></table>

Synthetic-system ablations. Removing the Lagrangian–Eulerian structure consistently degraded long-horizon forecasting in the synthetic systems (Table 2). In the multiscale setting, w/o LE structure increased the 20-week WMAPE from 13.36% to 58.26%. Removing the pathway corresponding to the represented process had a similarly pronounced efect: WMAPE increased from 12.45% to 50.13% without $\mathcal { G } _ { L }$ in the Lagrangian setting and from 21.11% to 78.27% without $\mathcal { G } _ { E }$ in the Eulerian setting. Removing either pathway also degraded performance when the two processes were coupled. The particularly large deterioration after removing the Eulerian pathway is consistent with the synthetic mobility mechanisms, in which transport is both epidemic-state dependent and, for CRGM, explicitly time modulated while remaining population conserving. In the multiscale setting, the large gap between the full model and w/o LE structure further supports separating infection generation from transport when two processes with diferent state actions evolve simultaneously, rather than asking one generic graph vector field to represent both.

The process-aligned priors provided additional gains. Removing the age-contact prior increased the 20-week WMAPE from 12.45% to 72.08% in the Lagrangian setting, whereas removing OD-informed mobility pretraining increased it from 21.11% to 44.57% in the Eulerian setting. The same variants also increased errors in the multiscale setting. Because the contact and OD priors enter diferent pathways, this process-matched deterioration supports the value of information–process alignment: each prior contributes most directly when assigned to the dynamical process it is intended to constrain.

![](images/3c7879186c93ea06324434a24732cd25897d0e967f574d5047b7fd1f3eccf7b3.jpg)  
Fig. 3: Representative fitting and forecasting trajectories of LE-EpiGNODE and selected baselines on the real-world COVID-19 datasets. (a), Ontario. (b), Hubei. (c), Spain. The vertical dashed line marks the forecast origin, and the gray-shaded region denotes the out-of-sample forecasting period. Models are trained using observations before the forecast origin. Ontario, Hubei, and Spain correspond to the Lagrangian, Eulerian, and multiscale propagation settings, respectively.

Real-world ablations. The real-world experiments showed a similar correspondence between processes and information sources (Table 2), with each pathway contributing in accordance with the dynamical role it represents in the given setting. At the longest horizon, w/o LE structure increased WMAPE from 16.20% to 34.78% in Ontario, from 15.87% to 45.45% in Hubei and from 20.43% to 55.43% in Spain. Removing $\mathcal { G } _ { L }$ had the largest efect in age-structured Ontario, increasing WMAPE to 58.20%, whereas removing $\mathcal { G } _ { E }$ increased WMAPE to 26.06% in mobility-coupled Hubei. Both pathways contributed in Spain, where age-dependent infection generation and spatial transport act in the same system: removing $\mathcal { G } _ { L }$ or $\mathcal { G } _ { E }$ increased WMAPE from 20.43% to 35.86% or 23.44%, respectively.

Table 2: Ablation performance across synthetic and real-world settings. Each cell reports MAE / WMAPE (%); lower values indicate better forecasting. The complete LE-EpiGNODE model is highlighted in bold. Ablations targeting the processes and priors emphasized in each setting are shown. A dash denotes a forecast horizon that was not evaluated in that setting.
<table><tr><td colspan="7">(a) Synthetic scenarios</td></tr><tr><td>Variant</td><td>1 week</td><td></td><td>4 weeks</td><td colspan="2">12 weeks</td><td colspan="2">20 weeks</td></tr><tr><td colspan="8">Lagrangian</td></tr><tr><td>LE-EpiGNODE</td><td>44.38</td><td>5.67</td><td>51.09</td><td>5.84</td><td>58.66</td><td>8.45</td><td>63.30 12.45</td></tr><tr><td>w/o LE structure</td><td>69.48</td><td>8.87</td><td>101.01</td><td>11.54</td><td>214.94</td><td>30.97</td><td>374.75 73.74</td></tr><tr><td> $\mathrm { w } / \mathrm { o } ~ \mathcal { G } _ { L }$ </td><td>176.15</td><td>22.49</td><td>185.64</td><td>21.22</td><td>207.70</td><td>29.93</td><td>254.75 50.13</td></tr><tr><td> $\mathrm { w } / \mathrm { o } e _ { u v } ^ { L }$ </td><td>54.58</td><td>6.97</td><td>58.87 / 6.73</td><td></td><td>128.53</td><td>/18.52</td><td>366.32 2 / 72.08</td></tr><tr><td colspan="8">Eulerian</td></tr><tr><td>LE-EpiGNODE</td><td>152.42</td><td>13.41</td><td>241.97</td><td>16.44</td><td>171.32</td><td>18.32</td><td>123.12 21.11</td></tr><tr><td>w/o LE structure</td><td>350.17</td><td>30.80</td><td>725.73</td><td>49.30</td><td>761.46</td><td>81.44</td><td>751.05 128.76</td></tr><tr><td> $\mathrm { w } \mathrm { / o } \mathcal { G } _ { E }$ </td><td>606.44</td><td>53.34</td><td>1009.28</td><td>68.56</td><td>723.92</td><td>77.43</td><td>456.54 78.27</td></tr><tr><td>w/o MLPmob pretraining</td><td>680.31</td><td>59.83</td><td>633.91</td><td>43.06</td><td>398.60</td><td>42.63</td><td>259.98 44.57</td></tr><tr><td colspan="8">Multiscale</td></tr><tr><td>LE-EpiGNODE</td><td>123.68</td><td>17.16</td><td>136.39</td><td>13.50</td><td>113.54</td><td>10.65</td><td>110.01 13.36</td></tr><tr><td>w/o LE structure</td><td>156.55</td><td>21.72</td><td>347.94</td><td>34.45</td><td>477.26</td><td>44.75</td><td>479.65 58.26</td></tr><tr><td> $\mathrm { w } \mathrm { / o } \ g _ { \underline { { L } } }$ </td><td>303.89</td><td>42.16</td><td>376.24</td><td>37.25</td><td>340.62</td><td>31.94</td><td>272.62 33.12</td></tr><tr><td> $\mathbf { w } / \mathrm { o } \ e _ { u v } ^ { L }$ </td><td>163.46</td><td>22.68</td><td>185.74</td><td>18.39</td><td>165.42</td><td>/15.51</td><td>142.63 /17.33</td></tr><tr><td>w/o GE</td><td>427.17</td><td>59.27</td><td>680.95</td><td>67.41</td><td>805.11</td><td>75.50</td><td>635.77 77.23</td></tr><tr><td>w/o  $\mathrm { M L P _ { m o b } }$  pretraining</td><td>228.77</td><td>31.74</td><td>290.25</td><td>28.73</td><td>288.06 27.01</td><td>222.20</td><td>26.99</td></tr><tr><td colspan="8">(b) Real-world COVID-19 datasets Ontario</td></tr><tr><td colspan="8"></td></tr><tr><td>Variant</td><td>7 days</td><td></td><td>14 days</td><td></td><td>35 days</td><td></td><td>67 days</td></tr><tr><td>LE-EpiGNODE w/o LE structure</td><td>128.57</td><td>3.43</td><td>165.71</td><td>4.49 36.52</td><td>190.60 7.80 835.93</td><td>248.66 533.63</td><td>16.20 34.78</td></tr><tr><td> $\mathrm { w } / \mathrm { o } ~ \mathcal { G } _ { L }$ </td><td>985.10 2028.34</td><td>26.31 54.18</td><td>1348.30 2102.24</td><td>56.93</td><td>1346.87</td><td>34.21 55.11</td><td>893.13 58.20</td></tr><tr><td> $\mathrm { w } / \mathrm { o } e _ { u v } ^ { L }$ </td><td>408.94/</td><td>10.92</td><td>400.34</td><td>10.84</td><td>593.86</td><td>/24.30</td><td>430.92</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>28.08</td></tr><tr><td colspan="8">Hubei</td></tr><tr><td colspan="8">Variant 7 days</td></tr><tr><td>LE-EpiGNODE</td><td>88.04</td><td>/17.15</td><td>14 days 58.25</td><td>8.07</td><td>26 days 100.34</td><td>15.87</td><td></td></tr><tr><td>w/o LE structure</td><td>117.18</td><td>22.82</td><td>116.65</td><td>16.16</td><td>287.37</td><td>45.45</td><td></td></tr><tr><td>w/o GE</td><td>105.46</td><td>20.54</td><td>84.11</td><td>11.65</td><td>164.80</td><td>26.06</td><td></td></tr><tr><td>w/o MI  $\mathrm { . P _ { m o b } }$  pretraining</td><td>123.33</td><td>24.02</td><td> $2 2 5 . 7 3 ^ { ' } / 3 1 . 2 8$ </td><td></td><td>203.72</td><td>32.22</td><td></td></tr><tr><td colspan="8">Spain</td></tr><tr><td>Variant</td><td>7 days</td><td></td><td>14 days</td><td></td><td>49 days</td><td></td><td>93 days</td></tr><tr><td>LE-EpiGNODE</td><td>54.78</td><td>9.61</td><td>54.63</td><td>9.66</td><td>43.89</td><td>14.09 31.74</td><td>37.60 20.43 101.98 55.43</td></tr><tr><td>w/o LE structure  $\mathrm { w } \mathrm { / o } \ g _ { \underline { { L } } }$ </td><td>104.06 160.33</td><td>18.26</td><td>126.40</td><td>22.36</td><td>98.86</td><td>32.06</td><td>65.97 35.86</td></tr><tr><td></td><td></td><td>28.13</td><td>153.58</td><td>27.16</td><td>99.87</td><td></td><td></td></tr><tr><td> $\mathrm { w } / \mathrm { o } e _ { u v } ^ { L }$ </td><td>110.94 132.73</td><td>19.47 23.29</td><td>100.91 123.99</td><td>17.85 21.93</td><td>59.70 58.30</td><td>/19.17 18.72</td><td>43.12 23.43 43.13 23.44</td></tr><tr><td> $\mathrm { w } \mathrm { / o } \ g _ { E }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { w / o \ M L P _ { m o b } }$  pretraining</td><td>132.73</td><td>23.29</td><td>113.18</td><td>20.02</td><td>58.08</td><td>18.65</td><td>42.62 23.16</td></tr></table>

The efects of the priors followed the same correspondence. Removing the agecontact prior increased the longest-horizon WMAPE from 16.20% to 28.08% in Ontario, while removing mobility pretraining increased it from 15.87% to 32.22% in Hubei. Both priors also improved forecasting in Spain, with smaller individual efects than in the settings that more directly represented the corresponding process.

## Process-specific operators enable dynamical validation and mechanistic interpretation beyond trajectory accuracy

Trajectory-level accuracy measures output fidelity, whereas LE-EpiGNODE also exposes internal dynamical quantities with explicit process semantics and mechanismdefined state actions. This makes it possible to test whether apparently accurate trajectories or accumulated structures are supported by faithful evolving dynamics. We therefore evaluate these quantities against synthetic ground truth and then examine their process-level organization in real-world forecasting, where the true generating operators are unobserved.

Dynamical fidelity in synthetic systems. The three evaluation levels mirror the construction of LE-EpiGNODE. The instantaneous process-specific operators $\mathcal { F } ^ { b }$ with $b \in \{ L , E \}$ denoting the Lagrangian or Eulerian branch, quantify the interaction strengths learned for each dynamical process by the neural components; the corresponding state fluxes Φ quantify the dynamical efect of those interactions after the mechanism-defined state action is applied; and the temporally integrated Lagrangian and Eulerian interaction fluxes (LIFs and EIFs) summarize cumulative process organization over the evaluation interval. The hierarchy $\mathcal { F } ^ { b } ( t )  \Phi ( t )  \mathrm { L I F / E I F }$ links learned interaction strengths, their instantaneous dynamical efects and cumulative process organization. WMAPE measures relative magnitude error, PCC measures agreement in the temporal and edge-wise variation of instantaneous operators and fluxes, and SRCC measures rank agreement of integrated interaction structures. Lower WMAPE and higher PCC or SRCC therefore indicate stronger agreement with the known generating dynamics; complete definitions are provided in Methods.

Figure 4a–d compares the true and inferred integrated interaction structures, while Extended Data Table 4 reports the complete quantitative dynamical-fidelity metrics. Complete edge-wise operator trajectories across the synthetic mechanisms are provided in Supplementary Figs. 3–21. LE-EpiGNODE showed strong agreement with the known generating dynamics at both instantaneous and accumulated levels across the three synthetic settings. In the Lagrangian setting, $\mathcal { F } ^ { L } \mathrm { - P C C , \Phi ^ { L } \mathrm { - P C C } }$ and LIF-SRCC were 0.980, 0.959 and 0.971, respectively. The corresponding values in the Eulerian setting were 0.996, 0.981 and 0.923. In the multiscale setting, the model retained strong agreement for both pathways: $\mathcal { F } ^ { L } \mathrm { { \mathrm { - } P C C } }$ and $\Phi ^ { L } \mathrm { - P C C }$ were 0.936 and 0.920 for the Lagrangian branch, while $\mathcal { F } ^ { E } \mathrm { \mathrm { - P C C } }$ and $\Phi ^ { E _ { \mathrm { - } } } \mathrm { P C C }$ were 0.999 and 0.990 for the Eulerian branch; the corresponding LIF-SRCC and EIF-SRCC values were 0.835 and 0.998.

The prior-removal variants provide a stricter test of the information–process alignment and reveal an important distinction between accumulated structure and evolving dynamics. Removing the age-contact prior left the LIF-SRCC unchanged at 0.971 in the Lagrangian setting, but increased the $\Phi ^ { L }$ -WMAPE from 25.89% to 196.68% and reduced the $\Phi ^ { L } \mathrm { - P C C }$ from 0.959 to 0.074. Thus, an apparently preserved cumulative interaction ranking can coexist with a severe loss of agreement in instantaneous flux variation. This discrepancy is particularly relevant for time-modulated and statedependent mechanisms, for which tracking the temporal evolution of the operator is a stricter requirement than reproducing its integrated ordering. Removing OD-informed mobility pretraining caused a broader deterioration in the Eulerian setting, increasing the $\mathcal { F } ^ { E _ { - } } \mathrm { W M A P E }$ from 6.93% to 53.49% and reducing the EIF-SRCC from 0.923 to 0.448. These results show that process-aligned priors constrain not only trajectory forecasts but also the evolving dynamical quantities represented by their corresponding pathways. More generally, the defined process semantics and state actions make these discrepancies observable as explicit dynamical quantities within LE-EpiGNODE.

Process-level interpretation in real epidemics. Because the generating operators are unavailable for the COVID-19 datasets, the real-world analysis uses the same operator- and flux-level quantities validated in synthetic systems to inspect how the fitted model organizes age-related and spatial information during forecasting. The LIF and EIF maps in Fig. 4e–h summarize the node-level composition of cumulative interactions along the two dynamical pathways. LIF is aggregated over the spatial dimension to characterize the relative organization of cumulative infection-generation flux across age groups, whereas EIF is aggregated over the age dimension to characterize the relative organization of cumulative infectious-state transport flux across regions. The resulting normalized profiles therefore provide complementary views of the age and spatial structures learned by LE-EpiGNODE.

The Lagrangian patterns reveal heterogeneous interactions across age groups. In Ontario, messages from several source groups are directed predominantly towards the [20, 40) group, while the ≥ 60 group also provides substantial input to multiple targets. Spain similarly exhibits unequal interaction strengths across age groups. Age-specific diferences in social contacts, susceptibility, infectivity and household exposure are known to shape the age distribution of COVID-19 transmission [1, 23–25], providing relevant epidemiological context for interpreting these age-related message patterns.

The Eulerian patterns show clearer spatial directionality. In Hubei, the learned fluxes form a Wuhan-centered structure, with surrounding cities receiving strong cumulative messages originating from Wuhan. Population movements from Wuhan played an important role in shaping the early spatial distribution of COVID-19 across China [26, 27]. In Spain, central regions produce stronger outward messages towards other regions, where regional mobility, multiple epidemic introductions, lockdowns and spatial spillovers afected the development of the first epidemic wave [28, 29].

The LIF and EIF maps characterize information propagation within LE-EpiGNODE. Accordingly, we interpret them as process-level explanations of how age-related and spatial information contributes to the model’s forecasts, while contact rates, individual transmission events and causal transmission pathways remain outside the scope of this analysis. Their value is that the same quantities whose dynamical fidelity is testable against synthetic ground truth remain inspectable in real forecasts, providing a consistent bridge from controlled process validation to real-world model interpretation.

## Discussion

LE-EpiGNODE was motivated by a mismatch between the organization of epidemic processes and the way heterogeneous information is often incorporated into forecasting models. Age-contact patterns and population mobility are not interchangeable covariates: they constrain infection generation and population transport, which act on the epidemic state in fundamentally diferent ways and may both vary with epidemic conditions and time. Mechanistic models preserve these process roles but can become restrictive when high-dimensional interactions are nonlinear or nonstationary, whereas flexible graph models can learn heterogeneous dependencies without explicitly fixing what a learned interaction represents or how it is allowed to modify the state. LE-EpiGNODE addresses this gap by embedding separate Lagrangian infectiongeneration and Eulerian population-transport operators within a structured UDE, aligning age-contact and mobility information with their corresponding pathways while retaining mechanism-defined state actions. The results indicate that this organization is particularly useful when forecasting moves beyond local trajectory continuation and requires extrapolation of evolving multiscale dynamics.

The forecasting results show broad gains that tend to become more pronounced at longer prediction horizons. LE-EpiGNODE achieved the lowest WMAPE at the longest horizon in all three real-world datasets. Its relative advantage increased from the shortest to the longest horizon in the Eulerian synthetic setting and in Hubei, while the multiscale synthetic setting and Spain shifted from weaker short-horizon rankings to first place at longer horizons. In Ontario, the relative WMAPE reduction was larger at intermediate horizons than at the longest horizon. In the Lagrangian synthetic setting, LE-EpiGNODE was slightly less accurate than MepoSIR at 20 weeks. Taken together, these results indicate that $\mathrm { L E - E p i G N O D E }$ remains competitive near the observed regime while its advantage tends to become more consistent when forecasting requires extrapolation of evolving state-dependent, time-varying and coupled multiscale interaction dynamics.

![](images/30c3a903b4dd6d49b8ede6d9a0e62d94e6f308ca2a62a376cc129b73c5746795.jpg)

![](images/daea8cb5b23b7ec7534536c154f60a7d6702cdc09bd458e6ca75b789f3ce7f0d.jpg)

![](images/e2ce425d8132864eb6e6801990aefc5291f29ab37380ee7a49b8d8a50a82c273.jpg)

![](images/9fa5de41d0ac271a56cffd1019390e0b288666a55e74ab2ba5e56214282fccc2.jpg)  
Fig. 4: Integrated interaction fluxes for dynamical validation and processlevel interpretation. (a), LIF in the Lagrangian synthetic scenario. (b), EIF in the Eulerian synthetic scenario. (c), LIF in the multiscale synthetic scenario. (d), EIF in the multiscale synthetic scenario. For synthetic systems, true and inferred fluxes are integrated over the full simulation horizon and min–max normalized for visualization. (e), LIF in Ontario. (f), EIF in Hubei. (g), EIF in Spain. (h), LIF in Spain. For real-world datasets, LIF is aggregated over regions and EIF over age groups to characterize age- and region-level interaction structure, respectively. Row and column normalization show the target distribution from each source and the source composition of each target. Diagonal conservation terms are excluded from EIF.

The ablation and process-level analyses jointly clarify the contribution of mechanism-organized message passing. Replacing the Lagrangian–Eulerian formulation with a generic Graph Neural ODE, removing individual pathways, or omitting their aligned contact and OD priors degraded forecasting in process-dependent ways across synthetic and real-world settings. These results support separating infection generation from conservative transport and aligning each information source with the process it constrains. Beyond forecasting, mechanism-defined message semantics and state actions make learned interaction strengths, induced state fluxes and integrated interaction structures directly testable against synthetic ground truth. LE-EpiGNODE showed strong agreement with the generating dynamics, whereas removing the contact prior left the cumulative Lagrangian ranking nearly unchanged despite a sharp deterioration in instantaneous flux agreement. This discrepancy shows that agreement in accumulated structure does not guarantee fidelity of evolving dynamics, particularly under time-modulated or state-dependent mechanisms. The $\mathcal { F } ^ { b } \to \Phi \to \mathrm { L I F } / \mathrm { E I F }$ hierarchy thus connects learned interactions, their dynamical efects and cumulative process organization, enabling direct dynamical validation in controlled systems and process-level model interpretation in real epidemics, where the true generating operators remain unobserved.

Several limitations remain. First, latent epidemic states and learned process fluxes may be only partially identifiable from reported incidence, particularly when observations are sparse, delayed or afected by changes in reporting. Additional epidemiological and behavioral measurements could provide stronger constraints on the hidden dynamics. Second, the auxiliary mobility network is pretrained on historical OD observations and frozen during epidemic-model training. Although its outputs remain dependent on epidemic state and time, its fixed parameters may limit adaptation to abrupt mobility changes caused by interventions or behavioral shifts. Third, the synthetic mechanisms are controlled and deliberately stylized, whereas the realworld evaluation is restricted to three COVID-19 settings; broader validation across pathogens, intervention regimes, spatial scales and surveillance systems is therefore needed.

The central methodological contribution of LE-EpiGNODE is mechanismorganized operator learning: heterogeneous observations are mapped to the physical processes they inform, graph messages are given explicit process semantics, and the governing dynamics constrain the state transformations those messages can induce while neural modules learn unresolved state- and time-dependent interaction strengths. This design is mechanism-informed in construction and makes the learned dynamics interpretable and, in controlled systems, directly testable against known generating processes. The resulting operator-level organization links flexible continuous-time learning with mechanistic structure, supports long-horizon epidemic extrapolation, and enables process-level mechanistic interpretation beyond trajectory accuracy. In real data, these quantities are used for model interpretation rather than causal mechanism estimation. More broadly, the same principle may be useful for multiscale dynamical systems in which diferent data streams constrain processes with distinct physical state actions.

## Methods

## LE-EpiGNODE dynamical formulation

LE-EpiGNODE formulates multiscale epidemic dynamics as a structured universal diferential equation (UDE) in which unresolved infection-generation and populationtransport interactions are represented by process-specific graph neural operators (Fig. 1b). The mechanistic backbone defines the relational role and admissible state action of each pathway, while the neural components learn their state- and time-dependent interaction strengths.

We consider a spatial epidemic system consisting of M regions, each divided into A age groups. For region k and age group $j ,$ , let $S _ { j } ^ { k } ( t ) , I _ { j } ^ { k } ( t )$ and $R _ { j } ^ { k } ( t )$ denote the susceptible, infectious and removed populations, respectively. The population size of age group i in region k is denoted by $P _ { i } ^ { k } ( t ) = S _ { i } ^ { k } ( t ) + I _ { i } ^ { k } ( t ) + R _ { i } ^ { k } ( t )$ . Let $C _ { i j } ^ { k }$ represent the contact intensity from infectious-source age group i to susceptible-target age group j in region $k , \beta ( t )$ denote a potentially time-varying transmission coeficient, $\gamma _ { j } ^ { k }$ denote the removal rate, $\eta _ { j }$ denote the mobility tendency of age group j, and $L ^ { l k }$ denote the transport intensity from region l to region k. A representative age-structured spatial epidemic system is

$$
\left\{ \begin{array} { l } { { \displaystyle { \frac { \mathrm { d } S _ { j } ^ { k } } { \mathrm { d } t } } = - \sum _ { i = 1 } ^ { A } \beta ( t ) C _ { i j } ^ { k } S _ { j } ^ { k } \frac { I _ { i } ^ { k } } { P _ { i } ^ { k } } + \sum _ { l = 1 } ^ { M } \eta _ { j } L ^ { l k } S _ { j } ^ { l } } , }  \\ { { \displaystyle { \frac { \mathrm { d } I _ { j } ^ { k } } { \mathrm { d } t } } = \sum _ { i = 1 } ^ { A } \beta ( t ) C _ { i j } ^ { k } S _ { j } ^ { k } \frac { I _ { i } ^ { k } } { P _ { i } ^ { k } } - \gamma _ { j } ^ { k } I _ { j } ^ { k } + \sum _ { l = 1 } ^ { M } \eta _ { j } L ^ { l k } I _ { j } ^ { l } , } } \\ { { \displaystyle { \frac { \mathrm { d } R _ { j } ^ { k } } { \mathrm { d } t } } = \gamma _ { j } ^ { k } I _ { j } ^ { k } + \sum _ { l = 1 } ^ { M } \eta _ { j } L ^ { l k } R _ { j } ^ { l } } . }  \end{array} \right.\tag{1}
$$

The diagonal entries of the transport operator account for outflow, and $\begin{array} { r } { \sum _ { k = 1 } ^ { M } L ^ { l k } = 0 } \end{array}$ ensures conservation of the transported population from each origin region l. In the observation model used below, case reporting is assumed to coincide with the $I  R$ transition, so the reported-incidence rate is identified with the removal flow $\gamma _ { j } ^ { k } I _ { j } ^ { k } ( t )$

Equation (1) separates two relational processes with diferent state actions: contact-mediated infection generation changes compartmental state, whereas population transport redistributes existing states across regions without changing compartmental identity. LE-EpiGNODE retains this mechanistic organization but replaces the unresolved, potentially nonstationary interaction functions with learnable operators. The resulting formulation preserves compartment-transition direction, explicit dependence of infection generation on infectious source states, and population conservation during spatial transport.

Accordingly, LE-EpiGNODE follows the universal diferential equation paradigm [17]:

$$
\frac { \mathrm { d } X } { \mathrm { d } t } = f _ { \mathrm { m e c h } } \big ( X ; \mathcal { F } ^ { L } ( X , t ) , \mathcal { F } ^ { E } ( X , t ) \big ) ,
$$

where $f _ { \mathrm { m e c h } }$ denotes the mechanism-defined governing vector field that encodes the retained compartmental organization and mechanistic constraints, while $\mathcal { F } ^ { L }$ and $\mathcal { F } ^ { E }$ represent the learnable Lagrangian infection-generation and Eulerian populationtransport operators, respectively. Unlike a single generic neural correction, the learnable operators are organized according to processes with distinct mechanistic roles, while the governing mechanisms determine their admissible state transformations.

To represent the two processes, we define a shared age–space node set in which each region–age combination corresponds to one node:

$$
v = ( k , j ) \in \mathcal { V } , \qquad k \in \{ 1 , \dots , M \} , \qquad j \in \{ 1 , \dots , A \} .
$$

The total number of nodes is $| \nu | = M A$ . For node $\boldsymbol { v } = ( k , j )$ , the epidemic state is

$$
\boldsymbol { X } _ { v } ( t ) = \left( \boldsymbol { X } _ { v } ^ { S } ( t ) , \boldsymbol { X } _ { v } ^ { I } ( t ) , \boldsymbol { X } _ { v } ^ { R } ( t ) \right) ^ { \top } = \left( \boldsymbol { S } _ { j } ^ { k } ( t ) , \boldsymbol { I } _ { j } ^ { k } ( t ) , \boldsymbol { R } _ { j } ^ { k } ( t ) \right) ^ { \top } .
$$

We construct two graph views over the shared node set V. The Lagrangian graph $\mathcal { G } _ { L } = ( \nu , \mathcal { E } _ { L } )$ represents within-region age-dependent interactions, with

$$
\mathcal { E } _ { L } = \{ ( k , i )  ( k , j ) : k \in \{ 1 , \ldots , M \} , \ i , j \in \{ 1 , \ldots , A \} \} .
$$

An edge $u = ( k , i ) \to v = ( k , j )$ indicates that infectious individuals in age group i can contribute to infection generation in age group j within the same region.

The Eulerian graph $\mathcal { G } _ { E } = ( \nu , \mathcal { E } _ { E } )$ represents mobility-driven population transport, with

$$
\mathcal { E } _ { E } = \{ ( l , j )  ( k , j ) : l \neq k , \ l , k \in \{ 1 , \ldots , M \} , \ j \in \{ 1 , \ldots , A \} \} .
$$

An edge $u = ( l , j ) \to v = ( k , j )$ represents the movement of individuals in age group j from region l to region k. Although $\mathcal { G } _ { L }$ and $\mathcal { G } _ { E }$ share the same nodes and epidemic states, they define distinct relational supports for the two dynamical processes: $\mathcal { G } _ { L }$ supports infection-generating interactions, whereas $\mathcal { G } _ { E }$ supports spatial redistribution of existing epidemic states. Their process identity is therefore fixed before the interaction strengths are learned.

## Process-specific graph neural operators in the UDE formulation

We parameterize the unresolved interactions on these two graph supports using separate state- and time-dependent neural operators. The Lagrangian operator $\mathcal { F } ^ { L }$ is informed by age-contact structure and learns infection-generation interaction strengths over $\mathcal { G } _ { L }$ , whereas the Eulerian operator $\mathcal { F } ^ { E }$ is informed by historical origin–destination mobility and learns population-transport strengths over $\mathcal { G } _ { E }$ . Their outputs remain coupled through the shared epidemic state, while the mechanistic vector field maps them to diferent state actions: infection-generating compartment transitions for Lagrangian messages and zero-sum redistribution of the full compartmental state for Eulerian messages.

Lagrangian infection-generation operator. The Lagrangian operator models contact-driven infection generation within each region. Following the infection term in Eq. (1), it depends on the infectious-source state, the susceptible-target state, the age-contact structure and continuous time. Age-contact information is introduced as a structural prior whose contribution can vary with the evolving epidemic state.

For a directed Lagrangian edge $u ~ = ~ ( k , i ) ~  ~ v ~ = ~ ( k , j )$ , the contact-prior representation is

$$
e _ { u v } ^ { L } = W _ { L } C _ { i j } ^ { k } ,
$$

where $C _ { i j } ^ { k }$ denotes the contact intensity from source age group i to target age group $j$ in region $k ,$ and $W _ { L }$ is a learnable projection into the latent edge space. Given the source state, target state, contact prior and time, the Lagrangian interaction strength

is defined as

$$
\mathcal { F } _ { u v } ^ { L } ( t ) = \frac { \mathrm { s o f t p l u s } [ \mathrm { M L P } _ { L } ( X _ { u } ( t ) \parallel X _ { v } ( t ) \parallel e _ { u v } ^ { L } \parallel t ) ] } { P _ { u } ( t ) } , \qquad u  v \in \mathcal { E } _ { L } ,\tag{2}
$$

where ∥ denotes feature concatenation and $P _ { u } ( t ) = X _ { u } ^ { S } ( t ) + X _ { u } ^ { I } ( t ) + X _ { u } ^ { R } ( t )$ is the population associated with the source node $u = ( k , i )$ . The softplus transformation ensures a non-negative interaction strength. Equation (2) uses the observed contact structure as a prior rather than a fixed transmission coeficient, allowing the efective interaction to vary with the source state, target state and time.

The infection-generation flux along edge $u  v$ is

$$
\Phi _ { u v } ^ { L } ( t ) = \mathcal { F } _ { u v } ^ { L } ( t ) X _ { u } ^ { I } ( t ) .
$$

Here, $\mathcal { F } _ { u v } ^ { L } ( t )$ represents the efective interaction strength, while multiplication by $X _ { u } ^ { I } ( t )$ preserves the explicit dependence of infection generation on the infectious-source population.

Eulerian population-transport operator. The Eulerian operator models the redistribution of existing epidemic states across regions. Regional populations and spatial separation define a gravity-type representation of spatial connectivity [30], while epidemic activity and time allow transport intensity to vary with transmission conditions [20]. Historical origin–destination observations are used to inform the auxiliary mobility function; the corresponding pretraining procedure is described below.

For an Eulerian edge $u = ( l , j ) \to v = ( k , j )$ , the spatial representation between regions l and k is

$$
e _ { l k } ^ { E } = \frac { ( N ^ { l } ) ^ { \kappa _ { 1 } } ( N ^ { k } ) ^ { \kappa _ { 2 } } } { \exp ( \lambda d _ { l k } ) } , \qquad l , k \in \{ 1 , \ldots , M \} ,
$$

where $N ^ { l }$ and $N ^ { k }$ are the fixed reference total populations of the source and destination regions, $d _ { l k }$ is their geographical distance or generalized transport impedance, and $\kappa _ { 1 }$ $\kappa _ { 2 }$ and λ control the population and distance efects.

Regional epidemic activity is constructed from the current infectious states and the removal rates:

$$
\bar { I } ^ { k } ( t ) = \frac { 1 } { N ^ { k } } \sum _ { j = 1 } ^ { A } \gamma _ { j } ^ { k } X _ { ( k , j ) } ^ { I } ( t ) ,
$$

where $\bar { I } ^ { k } ( t )$ represents the normalized reported-incidence activity implied by the infectious states in region k. The auxiliary mobility network takes a dataset-specific input vector $z _ { l k } ( t )$ that combines source and destination epidemic activity with the spatial representation and the temporal or regional features used in each dataset. The predicted OD mobility value is

$$
\hat { \mathcal { M } } ^ { l k } ( t ) = \mathrm { s o f t p l u s } \left[ \mathrm { M L P } _ { \mathrm { m o b } } \left( z _ { l k } ( t ) \right) \right] .\tag{3}
$$

For the synthetic and Hubei settings, $z _ { l k } ( t )$ contains the source and destination epidemic activities, the gravity-type spatial representation $e _ { l k } ^ { E }$ and normalized time. For Spain, additional periodic temporal features and source and destination region identities are included. The complete dataset-specific input definitions are provided in Supplementary Methods, section 1.2, and summarized in Supplementary Table 5. The pretraining and forecasting use of M $\mathrm { . P _ { m o b } }$ are specified in the learning procedure below.

The predicted OD mobility value is converted into the Eulerian populationtransport operator, whose of-diagonal entries are efective per-capita transport rates, as

$$
\mathcal { F } _ { u v } ^ { E } ( t ) = \left\{ \begin{array} { l l } { \displaystyle \eta _ { j } \mathrm { s o f t p l u s } \left( \alpha \frac { \hat { M } ^ { l k } ( t ) } { N ^ { l } } + \epsilon \right) , } & { u = ( l , j ) , v = ( k , j ) , l \neq k , } \\ { - \displaystyle \sum _ { w \neq u } \mathcal { F } _ { u w } ^ { E } ( t ) , } & { u = v , } \end{array} \right.\tag{4}
$$

where $\eta _ { j } \geq 0$ denotes the relative mobility tendency of age group $j ,$ while α and ϵ are model calibration parameters that map population-normalized mobility observations to efective transport rates. The mobility data and their construction are described in the corresponding source studies [31, 32]. The diagonal entry is defined as the negative sum of the outgoing interactions, giving

$$
\sum _ { v } \mathcal { F } _ { u v } ^ { E } ( t ) = 0 .
$$

This zero-sum constraint allows the Eulerian operator to redistribute existing epidemic states without creating or removing population mass. The corresponding infectiousstate transport flux is

$$
\Phi _ { u v } ^ { E } ( t ) = \mathcal { F } _ { u v } ^ { E } ( t ) X _ { u } ^ { I } ( t ) .
$$

The two fluxes therefore have diferent meanings: $\Phi _ { u v } ^ { L } ( t )$ represents newly generated infections associated with an infectious source, whereas $\Phi _ { u v } ^ { E } ( t )$ represents the spatial transport of an existing infectious population.

Coupled epidemic dynamics. The process-specific operators are coupled through the shared epidemic state, while fixed transition vectors determine how their learned interaction strengths act on the compartments. We define

$$
b _ { S I } = \left[ \begin{array} { c } { { - 1 } } \\ { { 1 } } \\ { { 0 } } \end{array} \right] , \qquad b _ { I R } = \left[ \begin{array} { c } { { 0 } } \\ { { - 1 } } \\ { { 1 } } \end{array} \right] ,
$$

where $b _ { S I }$ represents the transition from susceptible to infectious states and $b _ { I R }$ represents the transition from infectious to removed states.

Combining Lagrangian infection generation, the self-loop compartment transition and Eulerian population transport, we denote the complete state- and time-dependent operator by $\mathcal { F } = \mathcal { F } _ { \boldsymbol { \theta } } ( \boldsymbol { X } ( t ) , t )$ . Its action on the epidemic state at node v is

$$
\frac { \mathrm { d } X _ { v } } { \mathrm { d } t } = ( \mathcal { F } X ) _ { v } = \underbrace { \left( \sum _ { u \in \mathcal { N } _ { L } ( v ) \cup \{ v \} } \mathcal { F } _ { u v } ^ { L } ( t ) X _ { u } ^ { I } ( t ) \right) b _ { S I } } _ { \mathrm { L a g r a n g i a n ~ m e s s a g e ~ p a s s i n g } }  \\  + \underbrace { \gamma _ { v } X _ { v } ^ { I } ( t ) b _ { I R } } _ { \mathrm { s e l f - l o o p ~ m e s s a g e } } + \underbrace { \sum _ { u \in \mathcal { N } _ { E } ( v ) \cup \{ v \} } \mathcal { F } _ { u v } ^ { E } ( t ) X _ { u } ( t ) } _ { \mathrm { E u l e r i a n ~ m e s s a g e ~ p a s s i n g } } .\tag{5}
$$

Here, $\mathcal { N } _ { L } ( v )$ and $\mathcal { N } _ { E } ( v )$ denote the incoming neighbors of node v in the Lagrangian and Eulerian graphs, respectively, and $\gamma _ { v } = \gamma _ { j } ^ { k }$ for $\boldsymbol { v } = ( k , j )$ . The three terms correspond to infection generation, local compartmental progression and conservative spatial transport. Equation (5) therefore separates what is learned from how the state is updated: $\mathcal { F } ^ { L }$ and $\mathcal { F } ^ { E }$ learn edge-wise interaction strengths, whereas the transition vectors and zero-sum transport structure fix their mechanism-defined state actions.

The transition vectors satisfy

$$
\mathbf { 1 } ^ { \top } b _ { S I } = \mathbf { 1 } ^ { \top } b _ { I R } = 0 , \qquad \mathbf { 1 } = \left[ \begin{array} { l } { 1 } \\ { 1 } \\ { 1 } \end{array} \right] ,
$$

and the Eulerian operator satisfies $\begin{array} { r } { \sum _ { v } \mathcal { F } _ { u v } ^ { E } ( t ) = 0 . } \end{array}$ . Consequently, neither the compartment transitions nor spatial transport change the total population represented by the system. Population conservation is therefore encoded directly in the vector field rather than imposed through an additional penalty term.

Let $f _ { \theta } ( X , t )$ denote the complete vector field given by the right-hand side of Eq. (5), equivalently $f _ { \theta } ( X , t ) = \mathcal { F } _ { \theta } ( X , t ) X$ . The epidemic trajectory is obtained by

$$
X ( t ) = X ( t _ { 0 } ) + \int _ { t _ { 0 } } ^ { t } f _ { \theta } ( X ( \tau ) , \tau ) \ \mathrm { d } \tau .
$$

This notation distinguishes the complete nonlinear vector field from the processspecific operators $\mathcal { F } ^ { L }$ and $\mathcal { F } ^ { E }$

SEIR extension for real-world epidemics. The formulation above uses an SIR backbone for notational clarity and for the controlled synthetic experiments, where a minimal compartmental structure helps isolate the efects of infection generation and population transport. The Lagrangian–Eulerian decomposition itself is not tied to SIR. For the real-world COVID-19 analyses, we extend the same operator construction to an SEIR backbone by introducing an exposed compartment while leaving the two graph pathways unchanged. For node $v ,$

$$
X _ { v } ( t ) = \big ( X _ { v } ^ { S } ( t ) , X _ { v } ^ { E } ( t ) , X _ { v } ^ { I } ( t ) , X _ { v } ^ { R } ( t ) \big ) ^ { \top } , \qquad P _ { v } ( t ) = X _ { v } ^ { S } ( t ) + X _ { v } ^ { E } ( t ) + X _ { v } ^ { I } ( t ) + X _ { v } ^ { R } ( t ) ,
$$

with compartment-transition direction vectors

$$
b _ { S E } = ( - 1 , 1 , 0 , 0 ) ^ { \top } , \qquad b _ { E I } = ( 0 , - 1 , 1 , 0 ) ^ { \top } , \qquad b _ { I R } = ( 0 , 0 , - 1 , 1 ) ^ { \top } .
$$

The Lagrangian messages now generate susceptible-to-exposed transitions, the selfloop message describes exposed-to-infectious and infectious-to-removed progression, and the Eulerian messages transport the complete SEIR state across regions. The resulting dynamics are

$$
\begin{array} { r l } { \displaystyle \frac { \mathrm { d } X _ { v } } { \mathrm { d } t } = ( \mathcal { F } X ) _ { v } = } & { \underbrace { \left( \sum _ { u \in \mathcal { N } _ { L } ( v ) \cup \{ v \} } \mathcal { F } _ { u v } ^ { L } ( t ) X _ { u } ^ { I } ( t ) \right) b _ { S E } } _ { \mathrm { L a g r a n g i a n ~ m e s s a g e ~ p a s s i n g } } } \\ & { + \underbrace { \mathcal { F } X _ { v } ^ { E } ( t ) b _ { E I } + \gamma _ { v } X _ { v } ^ { I } ( t ) b _ { I R } } _ { \mathrm { s e l f - l o o p ~ m e s s a g e } } + \underbrace { \sum _ { u \in \mathcal { N } _ { E } ( v ) \cup \{ v \} } \mathcal { F } _ { u v } ^ { E } ( t ) X _ { u } ( t ) } _ { \mathrm { E u l e r i a n ~ m e s s a g e ~ p a s s i n g } } . } \end{array}\tag{6}
$$

Because the compartment-transition vectors sum to zero and the Eulerian operator remains zero-sum, the SEIR extension preserves the same population-conservation property as the SIR formulation. Dataset-specific values of the exposed-to-infectious transition rate σ are given with the real-world data settings below.

## Learning structured epidemic dynamics

Epidemic surveillance provides only partial observations of the underlying dynamics. In the observation model used here, reported incidence is identified with the $I  R$ transition flow, whereas the remaining compartment states and the operator-level interaction strengths remain latent. LE-EpiGNODE therefore learns the continuoustime dynamics indirectly through this reporting channel embedded in the structured vector field.

Optimizing both operators solely from reported-case trajectories can lead to ambiguity because diferent combinations of infection generation and population transport may produce similar observations. We therefore use a two-stage learning procedure. The auxiliary mobility network is first pretrained using historical origin–destination observations. The epidemic model is then optimized from reported cases while the pretrained mobility-network parameters remain fixed. This procedure transfers mobility information into the Eulerian pathway before the two processes are coupled through the epidemic trajectory.

OD-informed mobility pretraining. Let ϑ denote the parameters of the auxiliary mobility network ${ \mathrm { M L P } } _ { \mathrm { m o b } }$ , and let $\theta$ denote the remaining trainable parameters of the epidemic model, including the contact-prior projection, Lagrangian infection network, Eulerian transport correction parameters, removal rates and initial epidemic state.

During the first stage, ${ \mathrm { M L P } } _ { \mathrm { m o b } }$ is pretrained using the OD observations available within the training period. Because the latent infectious states are unavailable at this stage, the regional epidemic-activity input is constructed from observed reported cases:

$$
\bar { I } _ { \mathrm { o b s } } ^ { k } ( t _ { n } ) = \frac { 1 } { N ^ { k } \Delta t _ { n } } \sum _ { j = 1 } ^ { A } Y _ { \mathrm { o b s } , n , j } ^ { k } , \qquad \Delta t _ { n } = t _ { n } - t _ { n - 1 } ,
$$

where $Y _ { \mathrm { o b s } , n , j } ^ { k }$ is the reported case count for age group $j$ in region k over $[ t _ { n - 1 } , t _ { n } ]$ . The resulting population-normalized average reporting rate is used at observation times during mobility pretraining, with time units consistent with the epidemic dynamics. The mobility network is optimized by matching the predicted OD mobility value in Eq. (3) to the observed OD mobility value:

$$
\mathcal { L } _ { \mathrm { m o b } } ( \boldsymbol { \vartheta } ) = \frac { 1 } { | \mathcal { D } _ { \mathrm { m o b } } | } \sum _ { ( \boldsymbol { t } , \boldsymbol { l } , \boldsymbol { k } ) \in \mathcal { D } _ { \mathrm { m o b } } } \left[ \hat { \mathcal { M } } ^ { l \boldsymbol { k } } ( \boldsymbol { t } ) - \mathcal { M } ^ { l \boldsymbol { k } } ( \boldsymbol { t } ) \right] ^ { 2 } , \qquad l \neq k ,\tag{7}
$$

where $\mathcal { M } ^ { l k } ( t )$ denotes the dataset-specific OD mobility observation from region l to region $k ,$ and $\mathcal { D } _ { \mathrm { m o b } }$ is the set of observations used for pretraining. The optimized parameters are denoted by $\vartheta ^ { \ast }$ and remain fixed during subsequent epidemic-model training.

During epidemic training and forecasting, the regional epidemic activity supplied to the frozen mobility network is generated from the evolving infectious states:

$$
\bar { I } _ { \mathrm { d y n } } ^ { k } ( t ) = \frac { 1 } { N ^ { k } } \sum _ { j = 1 } ^ { A } \hat { \gamma } _ { j } ^ { k } \hat { X } _ { ( k , j ) } ^ { I } ( t ) ,
$$

where $\hat { \gamma } _ { j } ^ { k }$ is the learned removal rate and $\hat { X } _ { ( k , j ) } ^ { I } ( t )$ is the predicted infectious state. This instantaneous activity has the same units as the interval-averaged activity used during mobility pretraining. The epidemic-activity component of $z _ { l k } ( t )$ is therefore

constructed from

$$
\bar { I } _ { * } ^ { k } ( t ) = \left\{ \begin{array} { l l } { \bar { I } _ { \mathrm { o b s } } ^ { k } ( t _ { n } ) , } & { \mathrm { d u r i n g ~ m o b i l i t y ~ p r e t r a i n i n g } , t = t _ { n } , } \\ { \bar { I } _ { \mathrm { d y n } } ^ { k } ( t ) , } & { \mathrm { d u r i n g ~ e p i d e m i c ~ t r a i n i n g ~ a n d ~ f o r e c a s t i n g } . } \end{array} \right.
$$

During the second stage, the activity components of $z _ { l k } ( t )$ in $\operatorname { E q . } \left( 3 \right)$ are generated from $\bar { I } _ { \mathrm { d y n } } ^ { k } ( t )$ , while the remaining dataset-specific spatial, temporal and regional features are constructed as described in Supplementary Methods, section 1.2. Although $\vartheta ^ { \ast }$ remains fixed, the predicted transport intensities continue to vary with the evolving epidemic states and time. The pretrained network therefore acts as an OD-informed mobility prior rather than a fixed sequence of future mobility observations.

Epidemic training objective. Reported cases provide an indirect observation of the latent epidemic dynamics through the $I  R$ transition flow. With time measured in days, the predicted number of newly reported cases for age group $j$ in region k over $[ t _ { n - 1 } , t _ { n } ]$ is approximated by

$$
\hat { Y } _ { n , j } ^ { k } \approx \hat { \gamma } _ { j } ^ { k } \hat { X } _ { ( k , j ) } ^ { I } ( t _ { n } ) \Delta t _ { n } , \qquad \Delta t _ { n } = t _ { n } - t _ { n - 1 } = 1 ,
$$

where $\hat { \gamma } _ { j } ^ { k }$ is the learned removal rate and $\hat { X } _ { ( k , j ) } ^ { I } ( t _ { n } )$ is the predicted infectious state at $t _ { n }$ .

Given the observed number of newly reported cases $Y _ { \mathrm { o b s } , n , j } ^ { k }$ the epidemic observation loss is

$$
\mathcal { L } _ { \mathrm { e p i } } ( \theta ) = \frac { 1 } { | \mathcal { D } _ { \mathrm { e p i } } | } \sum _ { ( n , j , k ) \in \mathcal { D } _ { \mathrm { e p i } } } \left( \hat { Y } _ { n , j } ^ { k } - Y _ { \mathrm { o b s } , n , j } ^ { k } \right) ^ { 2 } ,\tag{8}
$$

where $\mathcal { D } _ { \mathrm { e p i } }$ denotes the set of epidemic observations used for training.

The initial epidemic state is estimated jointly with the model parameters. To place additional weight on consistency with the first observation time, we define

$$
\mathcal { L } _ { \mathrm { i c } } ( \theta ) = \frac { 1 } { | \mathcal { V } | } \sum _ { ( k , j ) \in \mathcal { V } } \left( \hat { Y } _ { 1 , j } ^ { k } - Y _ { \mathrm { o b s } , 1 , j } ^ { k } \right) ^ { 2 } .
$$

The epidemic model is optimized according to

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { m a i n } } ( \theta ) = \mathcal { L } _ { \mathrm { e p i } } ( \theta ) + \lambda _ { \mathrm { i c } } \mathcal { L } _ { \mathrm { i c } } ( \theta ) , \qquad \vartheta = \vartheta ^ { * } ,\tag{9}
$$

where $\lambda _ { \mathrm { i c } }$ controls the contribution of the initial-observation term and $\vartheta ^ { \ast }$ denotes the fixed parameters of the pretrained mobility network. The compartment-transition directions and population-conservation condition are encoded directly in $\mathcal { F }$ and therefore do not require additional penalty terms.

Optimization and forecasting. During training, the predicted epidemic trajectory is obtained by numerically integrating Eq. (5) for the SIR formulation or Eq. (6) for the real-world SEIR formulation. Gradients of the objective in Eq. (9) are propagated through the numerical solution to optimize the epidemic parameters and initial state jointly, while the pretrained mobility parameters $\vartheta ^ { \ast }$ remain fixed.

During forecasting, the learned dynamics are integrated forward from the forecast origin using the evolving predicted state. The Lagrangian operator updates infection-generation interactions according to the epidemic state, age-contact prior and time. The Eulerian operator updates transport intensities through $z _ { l k } ( t )$ , whose epidemic-activity components are generated internally from $\bar { I } _ { \mathrm { d y n } } ^ { k } ( t )$ during forecasting. LE-EpiGNODE therefore generates future epidemic trajectories without requiring future reported cases or future OD observations.

## Data

We use controlled synthetic datasets and real-world COVID-19 datasets for complementary purposes. The synthetic datasets provide known generating mechanisms and ground-truth operators for evaluating forecasting performance, structural contributions and dynamical fidelity under controlled conditions. The real-world datasets test forecasting from partial observations in settings involving age structure, population mobility or both.

Synthetic datasets. Synthetic data are generated by simulating the SIR-based Lagrangian–Eulerian system described above from day 0 to day 200, with daily sampling. We consider three structural settings. The Lagrangian setting contains one region and four age groups, isolating age-dependent infection generation without interregional transport. The Eulerian setting contains four regions and one age group, isolating mobility-driven spatial coupling without age stratification. The multiscale setting contains four regions and four age groups and couples both processes within the same epidemic state space.

Within each setting, we vary the generating functions while keeping population sizes, initial conditions, basic epidemiological parameters and observation-noise rules fixed. The four Lagrangian mechanisms—Baseline Mass-Action Contact Mechanism (BMCM), Time-Decaying Contact Mechanism (TDCM), Infection-Avoidant Contact Mechanism (IACM) and Nonlinear Mass-Action Contact Mechanism (NMCM)—span baseline mass-action, explicit temporal modulation, infection-dependent feedback and nonlinear state dependence. The two Eulerian mechanisms—Baseline Risk-Aware Gravity Mobility (BRGM) and Controlled Risk-Aware Gravity Mobility (CRGM)— represent epidemic-risk-aware transport with and without additional explicit temporal control. Their eight pairwise combinations define the multiscale systems, yielding 14 controlled generation conditions in total. This design tests whether a single learned formulation can extrapolate changing process dynamics rather than fit only a fixed mechanistic form. Complete mechanism definitions and generation settings are provided in Supplementary Methods, section 1.1, and Supplementary Tables 1–3.

For each condition, the dataset contains daily epidemic observations and the corresponding contact structure, with directed OD mobility observations additionally retained when transport is present. Gaussian observation noise is added to both epidemic and OD observations, with standard deviation set to 5% of the corresponding trajectory maximum. Because the generating equations are known, latent states, instantaneous Lagrangian and Eulerian operators, and their state fluxes are retained as ground truth for dynamical-recovery analyses. Complete population and contact structures, mechanism-specific parameters and noise settings are provided in Supplementary Methods, section 1.1, and Supplementary Tables 1–3.

Real-world COVID-19 datasets. Real-world validation uses three daily COVID-19 datasets representing complementary settings: age-structured transmission in Ontario, mobility-driven spatial dynamics in Hubei, and coupled age–space dynamics in Spain. Unlike the synthetic datasets, their latent states and generating operators are unobserved; we therefore use the real-world datasets to evaluate forecasting from partial observations using the SEIR extension in Eq. (6). Supplementary Table 4 summarizes the epidemic series, process-aligned information and evaluation settings.

Ontario uses age-stratified reported cases from the Ontario Data Catalogue covering 28 November 2021 to 27 February 2022, represented as one region with four age groups: [0, 20), [20, 40), [40, 60) and ≥ 60. An Ontario-specific contact matrix informs the Lagrangian pathway [2], and no interregional transport is included. The exposedto-infectious rate is set to $\sigma = ( 3 . 6 1 \mathrm { d a y s } ) ^ { - 1 }$ , using an Omicron incubation-period estimate as a modeling approximation [33].

Hubei uses daily reported cases for Wuhan, Xiaogan and Huanggang from 16 January to 25 February 2020, obtained from the DXY-COVID-19-Data and Wuhan-2019-nCoV repositories, without age stratification. Daily directed OD indices from Baidu mobility data hosted by Harvard Dataverse are used to pretrain the mobility network, while city populations and distances define the gravity representation. The exposed-to-infectious rate is set to $\sigma = ( 5 . 2 \mathrm { d a y s } ) ^ { - 1 }$ , using an early-pandemic incubation-period estimate as a modeling approximation [34].

Spain uses daily reported cases from the Spanish National Centre for Epidemiology covering 1 March to 21 June 2020, aggregated from NUTS-1 regions into four spatial nodes (R1: ES1+ES2; R2: ES3+ES4; R3: ES5; R4: ES6) and three age groups: [0, 50), [50, 70) and ≥ 70. Population-adjusted synthetic contact matrices inform the Lagrangian pathway [35]. Daily OD mobility observations obtained from Zenodo, together with regional populations and spatial separation, inform the Eulerian pathway. The exposed-to-infectious rate is set to $\sigma = ( 5 . 2 \mathrm { d a y s } ) ^ { - 1 }$ , using an early-pandemic incubation-period estimate as a modeling approximation [34].

All models use identical training periods, forecast origins and evaluation horizons within each dataset. Dataset-specific temporal and regional inputs to the auxiliary mobility predictor are described in Supplementary Methods, section 1.2, and summarized in Supplementary Table 5.

## Evaluation design

The evaluation addresses three questions: whether LE-EpiGNODE improves longhorizon forecasting, which structural components and process-aligned priors contribute to these gains, and whether the learned operators reproduce known generating dynamics in controlled systems. Synthetic experiments use pre-peak observations for training and evaluate forecast horizons of 1, 4, 12 and 20 weeks. For the real-world datasets, training uses 25 days for Ontario, 15 days for Hubei and 20 days for Spain, with forecast horizons of $7 / 1 4 / 3 5 / 6 7 , 7 / 1 4 / 2 6$ and $7 / 1 4 / 4 9 / 9 3$ days, respectively (Supplementary Table 4). Within each setting, all models use the same epidemic observations, forecast origins and evaluation horizons.

Baseline models. We compared LE-EpiGNODE with four groups of baselines representing diferent approaches to epidemic forecasting. Mechanistic baselines include SIR [36], MepoSIR [6] and the mobility-coupled metapopulation SEIR model (MCM-SEIR). General forecasting baselines include LSTM [37], GRU [38], Neural ODE [39], and the spatiotemporal graph models STGCN [40], DCRNN [41], AGCRN [42], Graph WaveNet [43], MTGNN [44] and Cola-GNN [10]. Epidemic-specific graph baselines include EpiGNN [12], Epi-Cola-GNN [45] and CovidGNN [46], while hybrid mechanistic-learning baselines include STAN [47], EpiGCN (adapted) [48], EARTH [16], MepoGNN [14], M-Graphormer [15] and EINN [49].

Synthetic experiments compare mechanistic and general forecasting baselines under controlled generating dynamics, while real-world experiments additionally include epidemic-specific graph and hybrid models. Model architectures, input configurations, hyperparameters and implementation details, including the continuous-time adaptation of EpiGCN, are provided in Supplementary Methods, section 1.3, and Supplementary Tables 6–8.

Ablation settings. We consider five ablations to assess the contributions of the Lagrangian–Eulerian structure and the priors aligned with each pathway. w/o LE structure replaces the structured formulation with a general Graph Neural ODE vector field. $w / o ~ \mathcal { G } _ { L }$ and $w / o ~ \mathcal { G } _ { E }$ remove the Lagrangian infection-generation and Eulerian population-transport pathways, respectively.

To evaluate the process-aligned priors, $w / o \ e _ { u v } ^ { L }$ removes the age-contact representation while retaining the remaining Lagrangian architecture and state inputs. $w / o$ ${ \mathrm { M L P } } _ { \mathrm { m o b } }$ pretraining retains the mobility network but replaces OD-supervised pretraining and parameter freezing with random initialization and joint optimization under the epidemic observation objective.

All variants use the same epidemic observations, training periods, forecast origins, evaluation horizons and epidemic-training settings as the complete model, except for the specified ablation.

Evaluation metrics. We evaluate LE-EpiGNODE from two complementary perspectives: epidemic forecasting and dynamical consistency. Forecasting performance is measured by mean absolute error (MAE) and weighted mean absolute percentage error (WMAPE):

$$
\mathrm { M A E } = \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \left| \hat { y } _ { q } - y _ { q } \right| , \quad \quad \mathrm { W M A P E } = \frac { \sum _ { q = 1 } ^ { Q } \left| \hat { y } _ { q } - y _ { q } \right| } { \sum _ { q = 1 } ^ { Q } \left| y _ { q } \right| } \times 1 0 0 \% ,
$$

where $y _ { q }$ and $\hat { y } _ { q }$ are the observed and predicted values, respectively, and $Q$ is the number of evaluated observations. MAE measures absolute prediction error, whereas WMAPE normalizes the total absolute error by the magnitude of the observed trajectory.

For synthetic systems, known generating dynamics allow direct evaluation of the instantaneous operators $\mathcal { F } ^ { L }$ and $\mathcal { F } ^ { E }$ and state fluxes Φ. WMAPE measures magnitude error, while the Pearson correlation coeficient (PCC) measures agreement in variation:

$$
\mathrm { P C C } ( a , \hat { a } ) = \frac { \sum _ { q = 1 } ^ { Q } ( a _ { q } - \bar { a } ) ( \hat { a } _ { q } - \overline { { { \hat { a } } } } ) } { \sqrt { \sum _ { q = 1 } ^ { Q } ( a _ { q } - \bar { a } ) ^ { 2 } } \sqrt { \sum _ { q = 1 } ^ { Q } ( \hat { a } _ { q } - \overline { { { \hat { a } } } } ) ^ { 2 } } } ,
$$

where $a _ { q }$ and $\hat { a } _ { q }$ are corresponding true and inferred operator or flux entries across evaluated edges and time points. Lower WMAPE values and PCC values closer to 1 indicate better agreement in magnitude and variation, respectively.

To assess cumulative interaction structure, we define the integrated Lagrangian interaction flux (LIF) and integrated Eulerian interaction flux (EIF) over $[ t _ { 0 } , t _ { T } ]$

$$
\mathrm { L I F } _ { u v } = \int _ { t _ { 0 } } ^ { t _ { T } } \Phi _ { u v } ^ { L } ( t ) \mathrm { d } t , \qquad \mathrm { E I F } _ { u v } = \int _ { t _ { 0 } } ^ { t _ { T } } \Phi _ { u v } ^ { E } ( t ) \mathrm { d } t .
$$

Agreement between true and inferred integrated structures is measured by the Spearman rank correlation coeficient (SRCC):

$$
\mathrm { S R C C } ( \boldsymbol { a } , \hat { \boldsymbol { a } } ) = \mathrm { P C C } \left( \mathrm { r a n k } ( \boldsymbol { a } ) , \mathrm { r a n k } ( \hat { \boldsymbol { a } } ) \right) .
$$

LIF-SRCC is computed over all entries of the integrated Lagrangian flux matrix, whereas EIF-SRCC excludes diagonal entries determined by population conservation. SRCC values closer to 1 indicate stronger agreement in the ranking of cumulative interactions.

For real-world datasets, operator and flux ground truth is unavailable, so integrated fluxes are instead aggregated and normalized for process-level interpretation:

$$
A _ { i j } ^ { L , \mathrm { r o w } } = \frac { \sum _ { k } \mathrm { L I F } _ { ( k , i ) , ( k , j ) } } { \sum _ { j ^ { \prime } } \sum _ { k } \mathrm { L I F } _ { ( k , i ) , ( k , j ^ { \prime } ) } } , \qquad A _ { i j } ^ { L , \mathrm { c o l } } = \frac { \sum _ { k } \mathrm { L I F } _ { ( k , i ) , ( k , j ) } } { \sum _ { i ^ { \prime } } \sum _ { k } \mathrm { L I F } _ { ( k , i ^ { \prime } ) , ( k , j ) } } ,
$$

and

$$
A _ { l k } ^ { E , \mathrm { r o w } } = \frac { \sum _ { j } \mathrm { E I F } _ { ( l , j ) , ( k , j ) } } { \sum _ { k ^ { \prime } \neq l } \sum _ { j } \mathrm { E I F } _ { ( l , j ) , ( k ^ { \prime } , j ) } } , \qquad A _ { l k } ^ { E , \mathrm { c o l } } = \frac { \sum _ { j } \mathrm { E I F } _ { ( l , j ) , ( k , j ) } } { \sum _ { l ^ { \prime } \neq k } \sum _ { j } \mathrm { E I F } _ { ( l ^ { \prime } , j ) , ( k , j ) } } , \qquad l \neq k .
$$

LIF aggregates infection-generation fluxes over regions to characterize age-to-age interactions, whereas EIF aggregates infectious-state transport fluxes over age groups to characterize region-to-region interactions. Row normalization gives the relative target distribution from each source, and column normalization gives the relative source composition of each target. Diagonal conservation terms are excluded from EIF.

## Data availability

The original empirical data sources used in this study are publicly accessible. COVID-19 case data for Ontario were obtained from the Ontario Data Catalogue. Case data for Hubei were obtained from the DXY-COVID-19-Data and Wuhan-2019-nCoV repositories. Case data for Spain were obtained from the Spanish National Centre for Epidemiology. The corresponding OD mobility data for Hubei and Spain were obtained from Harvard Dataverse and Zenodo, respectively. The synthetic datasets were generated using the mechanisms summarized in Methods, with the complete generation equations and parameter settings reported in Supplementary Methods, section 1.1, and Supplementary Tables 1–3.

Extended Data Table 1 — Forecasting performance on synthetic scenarios. Each cell reports MAE / WMAPE (%); the best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Model</td><td colspan="2">1 week</td><td colspan="2">4 weeks</td><td colspan="2">12 weeks</td><td colspan="2">20 weeks</td></tr><tr><td colspan="11">Lagrangian</td></tr><tr><td>LE-EpiGNODE</td><td>44.38</td><td>5.67</td><td>51.09</td><td>/5.84</td><td></td><td>58.66</td><td>8.45</td><td>63.30</td><td>12.45</td></tr><tr><td>SIR</td><td>270.78</td><td>34.57</td><td>299.29</td><td>34.21</td><td></td><td>238.84</td><td>34.42</td><td>186.39</td><td>36.68</td></tr><tr><td>MepoSIR</td><td>72.70</td><td>9.28</td><td>73.16</td><td>8.36</td><td></td><td>63.80</td><td>9.19</td><td>62.04</td><td>12.21</td></tr><tr><td>LSTM</td><td>70.16</td><td>8.96</td><td>119.28</td><td></td><td>13.63</td><td>220.36</td><td>31.75</td><td>373.05</td><td>73.40</td></tr><tr><td>GRU</td><td>50.56</td><td>6.45</td><td>87.05</td><td>9.95</td><td></td><td>215.85</td><td>31.10</td><td>383.14</td><td>75.39</td></tr><tr><td>Neural ODE</td><td>87.01</td><td>11.11</td><td>112.97</td><td></td><td>12.91</td><td>226.10</td><td>32.58</td><td>385.50</td><td>75.85</td></tr><tr><td>STGCN</td><td>64.11</td><td>8.18</td><td>102.69</td><td></td><td>11.74</td><td>204.36</td><td>29.45</td><td>364.85</td><td>71.79</td></tr><tr><td>DCRNN</td><td>49.47</td><td>6.31</td><td>69.23</td><td>7.91</td><td></td><td>201.67</td><td>29.06</td><td>374.82</td><td>73.75</td></tr><tr><td>AGCRN</td><td>64.15</td><td>8.19</td><td>92.19</td><td></td><td>10.54</td><td>210.09</td><td>30.27</td><td>376.15</td><td>74.01</td></tr><tr><td>GraphWaveNet</td><td>73.80</td><td>9.42</td><td>126.12</td><td></td><td>14.41</td><td>196.60</td><td>28.33</td><td>343.10</td><td>67.51</td></tr><tr><td>MTGNN</td><td>64.24</td><td>8.20</td><td>100.21</td><td></td><td>11.45</td><td>217.10</td><td>31.28</td><td>380.89</td><td>74.95</td></tr><tr><td colspan="10">Eulerian</td></tr><tr><td>LE-EpiGNODE</td><td>152.42</td><td>13.41</td><td>241.97</td><td></td><td>16.44</td><td>171.32</td><td>18.32</td><td>123.12</td><td>21.11</td></tr><tr><td>SIR</td><td>470.21</td><td>41.36</td><td>716.53</td><td></td><td>48.67</td><td>543.40</td><td>58.12</td><td>359.46</td><td>61.63</td></tr><tr><td>MepoSIR</td><td>647.62</td><td>56.96</td><td>739.74</td><td></td><td>50.25</td><td>567.70</td><td>60.72</td><td>365.66</td><td>62.69</td></tr><tr><td>LSTM</td><td>382.45</td><td>33.64</td><td>766.90</td><td></td><td>52.09</td><td>726.86</td><td>77.74</td><td>686.99</td><td>117.78</td></tr><tr><td>GRU</td><td>362.86</td><td>31.91</td><td>715.99</td><td>48.64</td><td></td><td>753.75</td><td>80.62</td><td>751.00</td><td>128.75</td></tr><tr><td>Neural ODE</td><td>424.93</td><td>37.37</td><td>802.98</td><td>54.54</td><td></td><td>754.31</td><td>80.68</td><td>702.46</td><td>120.43</td></tr><tr><td>STGCN</td><td>438.29</td><td>38.55</td><td>808.82</td><td></td><td>54.94</td><td>756.45</td><td>80.90</td><td>707.87</td><td>121.36</td></tr><tr><td>DCRNN</td><td>338.83</td><td>29.80</td><td>729.31</td><td></td><td>49.54</td><td>692.15</td><td>74.03</td><td>640.38</td><td>109.79</td></tr><tr><td>AGCRN</td><td>323.60</td><td>28.46</td><td>661.11</td><td></td><td>44.91</td><td>703.84</td><td>75.28</td><td>605.89</td><td>103.88</td></tr><tr><td>GraphWaveNet</td><td>304.94</td><td>26.82</td><td>665.80</td><td></td><td>45.23</td><td>742.35</td><td>79.40</td><td>770.63</td><td>132.12</td></tr><tr><td>MTGNN</td><td>342.14</td><td>30.09</td><td>707.47</td><td></td><td>48.06</td><td>691.90</td><td>74.00</td><td>709.36</td><td>121.62</td></tr><tr><td colspan="10">Multiscale</td></tr><tr><td>LE-EpiGNODE</td><td></td><td>123.68</td><td>/ 17.16</td><td>136.39</td><td>13.50</td><td>113.54</td><td>10.65</td><td>110.01</td><td>13.36</td></tr><tr><td>SIR</td><td>417.17</td><td>57.88</td><td>576.00</td><td></td><td>57.02</td><td>576.04</td><td>54.02</td><td>445.71</td><td>54.14</td></tr><tr><td>MepoSIR</td><td>218.34</td><td>30.29</td><td>283.60</td><td>28.08</td><td></td><td>245.70</td><td>23.04</td><td>210.09</td><td>25.52</td></tr><tr><td>LSTM</td><td>121.39</td><td>16.84</td><td>351.85</td><td>34.83</td><td></td><td>477.24</td><td>44.75</td><td>450.43</td><td>54.72</td></tr><tr><td>GRU</td><td>130.92</td><td>18.16</td><td>365.54</td><td>36.19</td><td></td><td>487.18</td><td>45.68</td><td>456.17</td><td>55.41</td></tr><tr><td>Neural ODE</td><td>183.23</td><td>25.42</td><td>398.87</td><td>39.49</td><td></td><td>511.79</td><td>47.99</td><td>494.10</td><td>60.02</td></tr><tr><td>STGCN</td><td>109.80</td><td>15.23</td><td>297.87</td><td>29.49</td><td></td><td>445.47</td><td>41.77</td><td>466.88</td><td>56.71</td></tr><tr><td>DCRNN</td><td>95.33</td><td>13.23</td><td>274.44</td><td>27.17</td><td></td><td>437.31</td><td>41.01</td><td>466.42</td><td>56.66</td></tr><tr><td>AGCRN</td><td>126.42</td><td>17.54</td><td>301.52</td><td></td><td>29.85</td><td>421.37</td><td>39.51</td><td>448.22</td><td>54.45</td></tr><tr><td>GraphWaveNet</td><td>117.47</td><td>16.30</td><td>330.10</td><td></td><td>32.68</td><td>460.83</td><td>43.21</td><td>454.27</td><td>55.18</td></tr><tr><td>MTGNN</td><td>108.14</td><td>15.00</td><td>285.70</td><td></td><td>28.28</td><td>423.83</td><td>39.74</td><td>446.42</td><td>54.23</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Extended Data Table 2 — Forecasting performance on the real-world COVID-19 datasets. Each cell reports MAE / WMAPE (%); the best and secondbest results are shown in bold and underlined, respectively.
<table><tr><td colspan="6">Ontario</td></tr><tr><td>Model</td><td>7 days</td><td>14 days</td><td>35 days</td><td></td><td>67 days</td></tr><tr><td>LE-EpiGNODE</td><td>128.57 /3.43</td><td>165.71 /4.49</td><td>190.60</td><td>/7.80</td><td>248.66 /16.20</td></tr><tr><td>MCM-SEIR</td><td>364.02 9.72</td><td>449.45 12.17</td><td>447.98</td><td>18.33</td><td>333.83 21.76</td></tr><tr><td>DCRNN</td><td>697.37 18.63</td><td>803.41 21.76</td><td>594.27</td><td>24.32</td><td>415.31 27.06</td></tr><tr><td>AGCRN</td><td>1255.42 33.53</td><td>1546.00 41.87</td><td>932.27</td><td>38.15</td><td>584.65 38.10</td></tr><tr><td>MTGNN</td><td>708.40 18.92</td><td>826.26 22.38</td><td>504.55</td><td>20.65</td><td>365.74 23.83</td></tr><tr><td>EpiGNN</td><td>609.43 16.28</td><td>751.34 20.35</td><td>581.71</td><td>23.80</td><td>530.21 34.55</td></tr><tr><td>Cola-GNN</td><td>870.40 23.25</td><td>1053.45 28.53</td><td>659.56</td><td>26.99</td><td>418.94 27.30</td></tr><tr><td>CovidGNN</td><td>1349.77 36.05</td><td>1601.26 43.37</td><td>995.45</td><td>40.73</td><td>608.55 39.66</td></tr><tr><td>MepoGNN</td><td>547.73 14.63</td><td>576.04 15.60</td><td>515.62</td><td>21.10</td><td>480.28 31.30</td></tr><tr><td>M-Graphormer</td><td>277.70 /7.42</td><td>493.73 /13.37</td><td>608.23</td><td>24.89</td><td>791.57 51.58</td></tr><tr><td>EpiGCN</td><td>870.47 23.25</td><td>969.13 /26.25</td><td>694.34</td><td>28.41</td><td>540.18 35.20</td></tr><tr><td>STAN</td><td>882.64 23.58</td><td>989.26 / 26.79</td><td>738.46</td><td>30.22</td><td>623.62 40.64</td></tr><tr><td>EINN</td><td>552.13 14.75</td><td>637.52 /17.27</td><td>570.03</td><td>23.32</td><td>374.45 24.40</td></tr><tr><td>Epi-Cola-GNN EARTH</td><td>874.91 23.37</td><td>861.65 23.34</td><td>608.58</td><td>24.90</td><td>377.05 24.57</td></tr><tr><td>503.39</td><td>/13.45</td><td>610.27 / 16.53</td><td>620.73</td><td>25.40</td><td>416.66 27.15</td></tr><tr><td colspan="6">Hubei</td></tr><tr><td>Model</td><td>7 days</td><td>14 days</td><td></td><td>26 days</td><td></td><td></td></tr><tr><td>LE-EpiGNODE</td><td>88.04 /17.15</td><td>58.25</td><td>/8.07</td><td>100.34 /15.87</td><td></td><td></td></tr><tr><td>MCM-SEIR DCRNN</td><td>120.45 /23.46 140.26 27.32</td><td>197.96</td><td>/27.43</td><td>488.72 /77.29 288.36 45.60</td><td></td><td></td></tr><tr><td>AGCRN</td><td>98.68 19.22</td><td>123.22 107.81</td><td>/17.07 / 14.94</td><td>291.94 46.17</td><td></td><td></td></tr><tr><td>MTGNN</td><td>146.23 28.48</td><td>131.28</td><td>/18.19</td><td>326.15 51.58</td><td></td><td></td></tr><tr><td>EpiGNN</td><td>146.70 28.57</td><td>124.16</td><td>/17.20</td><td>280.76 44.40</td><td></td><td></td></tr><tr><td>Cola-GNN</td><td>126.63 24.66</td><td>116.85</td><td>/ 16.19</td><td>256.29</td><td>40.53</td><td></td></tr><tr><td>CovidGNN</td><td>138.60 26.99</td><td>122.71</td><td>/ 17.00</td><td>250.55</td><td>39.62</td><td></td></tr><tr><td>MepoGNN</td><td>105.72 /20.59</td><td>104.00</td><td>/14.41</td><td>167.27</td><td>26.45</td><td></td></tr><tr><td>M-Graphormer</td><td>130.34 /25.39</td><td>188.64</td><td>26.14</td><td>244.12</td><td>38.61</td><td></td></tr><tr><td>EpiGCN</td><td>106.79 20.80</td><td>168.90</td><td>23.40</td><td>302.04</td><td>47.76</td><td></td></tr><tr><td>STAN</td><td>185.62</td><td>217.64</td><td>30.16</td><td>194.13</td><td>30.70</td><td></td></tr><tr><td>EINN</td><td>36.15</td><td>186.81</td><td>25.88</td><td>280.60</td><td>44.37</td><td></td></tr><tr><td></td><td>103.84 20.22</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Epi-Cola-GNN EARTH</td><td>105.28 20.50</td><td>178.32</td><td>24.71</td><td>316.56</td><td>50.06</td><td></td></tr><tr><td></td><td>144.79</td><td>28.20</td><td>198.63 /27.52</td><td>391.77</td><td>61.95</td><td></td></tr><tr><td colspan="7">Spain</td></tr><tr><td>Model</td><td>7 days</td><td></td><td>14 days</td><td>49 days</td><td></td><td>93 days</td></tr><tr><td>LE-EpiGNODE</td><td>54.78 / 9.61</td><td></td><td>54.63 / 9.66</td><td>43.89</td><td>/14.09</td><td>37.60 20.43</td></tr><tr><td>MCM-SEIR</td><td>92.71 16.27</td><td>104.42</td><td>/18.47</td><td>71.43</td><td>22.93</td><td>45.64 24.81</td></tr><tr><td>DCRNN</td><td>73.10 /12.83</td><td>84.93</td><td>/15.02</td><td>62.21</td><td>19.97</td><td>98.84 53.72</td></tr><tr><td>AGCRN</td><td>72.84 / 12.78</td><td>96.90</td><td>/17.14</td><td>120.47</td><td>38.67</td><td>155.10 84.30</td></tr><tr><td>MTGNN</td><td>56.83 /9.97</td><td>75.41</td><td>/13.34</td><td>72.10</td><td>23.15</td><td>82.27 44.71</td></tr><tr><td>EpiGNN</td><td>67.92 11.92</td><td>100.13</td><td>/17.71</td><td>115.12</td><td>/36.96</td><td>134.18 72.93</td></tr><tr><td>Cola-GNN</td><td>96.90 /17.00</td><td>117.50</td><td>20.78</td><td>101.28</td><td>32.51</td><td>114.12 62.03</td></tr><tr><td>CovidGNN</td><td>85.78 /15.05</td><td>116.24</td><td>20.56</td><td>137.93</td><td>/44.28</td><td>190.14 103.34</td></tr><tr><td>MepoGNN</td><td>39.89 / 7.00</td><td>46.82</td><td>8.28</td><td>54.92</td><td>17.63</td><td>53.09 28.86</td></tr><tr><td>M-Graphormer</td><td>146.31 25.67</td><td>146.28</td><td>25.87</td><td>92.73</td><td>29.77</td><td>83.40 45.33</td></tr><tr><td>EpiGCN</td><td>40.66 7.13</td><td>55.71</td><td>/9.85</td><td>84.62</td><td>27.17</td><td>64.01 34.79</td></tr><tr><td>STAN</td><td>70.06 12.29</td><td>86.56</td><td>15.31</td><td>144.56</td><td>46.41</td><td>221.38 120.32</td></tr><tr><td>EINN</td><td>19.70</td><td>3.46 34.46</td><td>6.09</td><td>68.97</td><td>22.14</td><td>57.62 31.32</td></tr><tr><td>Epi-Cola-GNN</td><td>103.15</td><td>18.10 99.67</td><td>17.63</td><td>76.47</td><td>24.55</td><td>63.01 34.24</td></tr><tr><td>EARTH</td><td>107.51</td><td>18.86 119.49</td><td>/ 21.13</td><td>155.51</td><td>49.92</td><td>181.46 98.63</td></tr></table>

Extended Data Table 3 — OD mobility fitting and forecasting errors. Each entry reports MAE / WMAPE (%) for the directed OD mobility values in Hubei and Spain. Errors are computed on each dataset’s original mobility scale; MAE values are not directly comparable across datasets.
<table><tr><td colspan="6">Hubei</td></tr><tr><td>Output</td><td>Training</td><td>7 days</td><td>14 days</td><td>26 days</td></tr><tr><td>OD mobility</td><td>0.0033 / 0.99</td><td>0.0120 / 28.75</td><td>0.0138 / 33.80</td><td>0.0161 / 40.15</td></tr></table>

<table><tr><td colspan="7">Spain</td></tr><tr><td>Output</td><td>Training</td><td>7 days</td><td>14 days</td><td>49 days</td><td>93 days</td><td></td></tr><tr><td>OD mobility</td><td>7303.85 / 12.25</td><td>4270.26 / 16.67</td><td></td><td>5753.13 / 24.24</td><td>6249.38 / 24.21</td><td>9649.42 / 29.06</td></tr></table>

Extended Data Table 4 — Dynamical fidelity on synthetic data. ${ \mathcal { F } } ^ { b } .$ WMAPE/PCC and Φ-WMAPE/PCC evaluate the instantaneous process-specific operator and state flux, respectively, where $b \in \{ L , E \}$ denotes the branch listed in each row. IF-SRCC denotes LIF-SRCC for the Lagrangian branch and EIF-SRCC for the Eulerian branch. Lower WMAPE and higher PCC or SRCC indicate stronger agreement with the known generating dynamics.
<table><tr><td>Scenario</td><td>Branch</td><td>Variant</td><td>Fb-WMAPE (%)</td><td>Φ-WMAPE (%)</td><td> $\mathcal { F } ^ { b } \mathbf { - P C C }$ </td><td>Φ-PCC</td><td>IF-SRCC</td></tr><tr><td rowspan="2">Lagrangian L</td><td rowspan="2"></td><td>LE-EpiGNODE</td><td>19.79</td><td>25.89</td><td>0.980</td><td>0.959</td><td>0.971</td></tr><tr><td> $\mathrm { w } / \mathrm { o } e _ { u v } ^ { L }$ </td><td>48.44</td><td>196.68</td><td>0.849</td><td>0.074</td><td>0.971</td></tr><tr><td rowspan="2">Eulerian</td><td rowspan="2">E</td><td>LE-EpiGNODE</td><td>6.93</td><td>22.58</td><td>0.996</td><td>0.981</td><td>0.923</td></tr><tr><td> $\mathrm { w / o \ M L P _ { m o b } }$  pretraining</td><td>53.49</td><td>79.37</td><td>0.683</td><td>0.664</td><td>0.448</td></tr><tr><td rowspan="6">Multiscale</td><td rowspan="3">L</td><td>LE-EpiGNODE</td><td>26.45</td><td>34.29</td><td>0.936</td><td>0.920</td><td>0.835</td></tr><tr><td> $\mathrm { w } / \mathrm { o } e _ { u v } ^ { L }$ </td><td>30.78</td><td>42.36</td><td>0.931</td><td>0.872</td><td>0.851</td></tr><tr><td> $\mathrm { w / o \ M L P _ { m o b } }$  pretraining</td><td>39.27</td><td>54.06</td><td>0.886</td><td>0.778</td><td>0.849</td></tr><tr><td rowspan="3">E</td><td> $\mathrm { L E - E p i G N O D E }$ </td><td>5.66</td><td>16.15</td><td>0.999</td><td>0.990</td><td>0.998</td></tr><tr><td> $\mathrm { w } / \mathrm { o } e _ { u v } ^ { L }$ </td><td>5.80</td><td>21.70</td><td>0.999</td><td>0.980</td><td>0.997</td></tr><tr><td> $\mathrm { w / o \ M L P _ { m o b } }$  pretraining</td><td>6.65</td><td>25.42</td><td>0.998</td><td>0.980</td><td>0.997</td></tr></table>

![](images/e4bdf7ac9f2fa602bf05216c83199faecdb3e03ef7fc57f6c457cf5caf8ce4c3.jpg)  
Extended Data Fig. 1 — Fitting and forecasting trajectories across all coupled multiscale synthetic mechanisms. (a), Multiscale scenarios combining the four Lagrangian infection-generation mechanisms (BMCM, TDCM, IACM and NMCM) with BRGM transport. (b), Corresponding multiscale scenarios combining the same four Lagrangian mechanisms with CRGM transport. The vertical dashed line marks the forecast origin, and the gray-shaded region denotes the out-of-sample forecasting period. Colored envelopes show post-hoc normal-reference residual bands centered on the model predictions. All models are trained using only observations before the forecast origin.

![](images/f330200fc60e512b2813127d6ca8c224a53a336c13b5e6a6a3c6155edc407546.jpg)  
Extended Data Fig. 2 — OD mobility extrapolation in Hubei. Observed and predicted trajectories are shown for all directed OD edges. The vertical dashed line marks the forecast origin. The auxiliary mobility predictor uses only OD observations available before the forecast origin. The overall $\dot { R } ^ { 2 }$ displayed in the plot summarizes the complete fitting and forecasting interval; forecasting MAE and WMAPE are reported separately in Extended Data Table 3.

![](images/a6f57e5f805252024d8fffff9b2d7e2246f1bdb924aa42ebff80f4dbf3351372.jpg)  
Extended Data Fig. 3 — OD mobility extrapolation in Spain. Observed and predicted trajectories are shown for all directed OD edges. The vertical dashed line marks the forecast origin. The auxiliary mobility predictor uses only OD observations available before the forecast origin. The overall ${ \check { R } } ^ { 2 }$ displayed in the plot summarizes the complete fitting and forecasting interval; forecasting MAE and WMAPE are reported separately in Extended Data Table 3.

## References

[1] Monod, M. et al. Age groups that sustain resurging COVID-19 epidemics in the United States. Science 371, eabe8372 (2021). URL https://doi.org/10.1126/ science.abe8372.

[2] Mistry, D. et al. Inferring high-resolution human mixing patterns for disease modeling. Nature Communications 12, 323 (2021). URL https://doi.org/10. 1038/s41467-020-20544-y.

[3] Chang, S. et al. Mobility network models of COVID-19 explain inequities and inform reopening. Nature 589, 82–87 (2021). URL https://doi.org/10.1038/ s41586-020-2923-3.

[4] Rahmandad, H., Xu, R. & Ghafarzadegan, N. Enhancing long-term forecasting: Learning from COVID-19 models. PLOS Computational Biology 18, e1010100 (2022). URL https://doi.org/10.1371/journal.pcbi.1010100.

[5] Kendall, M., Ferretti, L., Wymant, C. et al. Drivers of epidemic dynamics in real time from daily digital COVID-19 measurements. Science 385, eadm8103 (2024). URL https://doi.org/10.1126/science.adm8103.

[6] Colizza, V., Pastor-Satorras, R. & Vespignani, A. Reaction–difusion processes and metapopulation models in heterogeneous networks. Nature Physics 3, 276– 282 (2007). URL https://doi.org/10.1038/nphys560.

[7] Belik, V., Geisel, T. & Brockmann, D. Natural human mobility patterns and spatial spread of infectious diseases. Physical Review X 1, 011001 (2011). URL https://doi.org/10.1103/PhysRevX.1.011001.

[8] Arenas, A. et al. Modeling the spatiotemporal epidemic spreading of COVID-19 and the impact of mobility and social distancing interventions. Physical Review X 10, 041055 (2020). URL https://doi.org/10.1103/PhysRevX.10.041055.

[9] Liu, Z., Wan, G., Prakash, B. A., Lau, M. S. Y. & Jin, W. A review of graph neural networks in epidemic modeling. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 6577–6587 (Association for Computing Machinery, 2024). URL https://doi.org/10.1145/3637528.3671455.

[10] Deng, S., Wang, S., Rangwala, H., Wang, L. & Ning, Y. Cola-GNN: Crosslocation attention based graph neural networks for long-term ILI prediction. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management, 245–254 (Association for Computing Machinery, 2020). URL https://doi.org/10.1145/3340531.3411975.

[11] Panagopoulos, G., Nikolentzos, G. & Vazirgiannis, M. Transfer graph neural networks for pandemic forecasting. Proceedings of the AAAI Conference on Artificial Intelligence 35, 4838–4845 (2021). URL https://doi.org/10.1609/aaai.v35i6. 16616.

[12] Xie, F., Zhang, Z., Li, L., Zhou, B. & Tan, Y. EpiGNN: Exploring spatial transmission with graph neural network for regional epidemic forecasting. In Machine Learning and Knowledge Discovery in Databases: Research Track, Vol. 13718 of Lecture Notes in Computer Science, 469–485 (Springer, 2023). URL https://doi.org/10.1007/978-3-031-26422-1 29.

[13] Qiu, M., Tan, Z. & Bao, B.-K. MSGNN: Multi-scale spatio-temporal graph neural network for epidemic forecasting. Data Mining and Knowledge Discovery 38, 2348–2376 (2024). URL https://doi.org/10.1007/s10618-024-01035-w.

[14] Cao, Q. et al. MepoGNN: Metapopulation epidemic forecasting with graph neural networks. In Machine Learning and Knowledge Discovery in Databases: Research Track, Vol. 13718 of Lecture Notes in Computer Science, 453–468 (Springer, 2023). URL https://doi.org/10.1007/978-3-031-26422-1 28.

[15] Wang, X. & Jin, Z. Multi-region infectious disease prediction modeling based on spatio-temporal graph neural network and the dynamic model. PLOS Computational Biology 21, e1012738 (2025). URL https://doi.org/10.1371/journal.pcbi. 1012738.

[16] Wan, G. et al. EARTH: Epidemiology-aware neural ODE with continuous disease transmission graph. In Singh, A. et al. (eds) Proceedings of the 42nd International Conference on Machine Learning, Vol. 267 of Proceedings of Machine Learning Research, 61929–61941 (PMLR, 2025). URL https://proceedings.mlr.press/v267/ wan25b.html.

[17] Rackauckas, C. et al. Universal diferential equations for scientific machine learning. arXiv preprint arXiv:2001.04385v1 (2020). URL https://arxiv.org/abs/ 2001.04385v1.

[18] Tang, B. et al. Managing spatio-temporal heterogeneity of susceptibles by embedding it into an homogeneous model: A mechanistic and deep learning study. PLOS Computational Biology 20, e1012497 (2024).

[19] Ma, K. et al. Integrating Kolmogorov-Arnold networks with ordinary diferential equations for eficient, interpretable, and robust deep learning: Epidemiology of infectious diseases as a case study. Infectious Disease Modelling 11, 603–618 (2026).

[20] Nouvellet, P. et al. Reduction in mobility and COVID-19 transmission. Nature Communications 12, 1090 (2021). URL https://doi.org/10.1038/ s41467-021-21358-2.

[21] G´omez-Garde˜nes, J., Soriano-Pa˜nos, D. & Arenas, A. Critical regimes driven by recurrent mobility patterns of reaction–difusion processes in networks. Nature Physics 14, 391–395 (2018). URL https://doi.org/10.1038/s41567-017-0022-7.

[22] Gallo, L., Frasca, M., Latora, V. & Russo, G. Lack of practical identifiability may hamper reliable predictions in COVID-19 epidemic models. Science Advances 8, eabg5234 (2022). URL https://doi.org/10.1126/sciadv.abg5234.

[23] Caetano, C. et al. Identifying the main drivers of transmission in the early phase of the COVID-19 pandemic in Portugal. Scientific Reports 14, 30689 (2024). URL https://doi.org/10.1038/s41598-024-76604-6.

[24] Hu, S. et al. Infectivity, susceptibility, and risk factors associated with SARS-CoV-2 transmission under intensive contact tracing in Hunan, China. Nature Communications 12, 1533 (2021). URL https://doi.org/10.1038/s41467-021-21710-6.

[25] Bi, Q. et al. Insights into household transmission of SARS-CoV-2 from a population-based serological survey. Nature Communications 12, 3643 (2021).

URL https://doi.org/10.1038/s41467-021-23733-5.

[26] Jiang, J. & Luo, L. Influence of population mobility on the novel coronavirus disease (COVID-19) epidemic: Based on panel data from Hubei, China. Global Health Research and Policy 5, 30 (2020). URL https://doi.org/10.1186/ s41256-020-00151-6.

[27] Jia, J. S. et al. Population flow drives spatio-temporal distribution of COVID-19 in China. Nature 582, 389–394 (2020). URL https://doi.org/10.1038/ s41586-020-2284-y.

[28] Mazzoli, M. et al. Interplay between mobility, multi-seeding and lockdowns shapes COVID-19 local impact. PLOS Computational Biology 17, e1009326 (2021). URL https://doi.org/10.1371/journal.pcbi.1009326.

[29] Orea, L. & Alvarez, I. C. How efective has the Spanish lockdown been to battle<sup>´</sup> COVID-19? a spatial analysis of the coronavirus propagation across provinces. Health Economics 31, 154–173 (2022). URL https://doi.org/10.1002/hec.4437.

[30] Simini, F., Gonz´alez, M. C., Maritan, A. & Barab´asi, A.-L. A universal model for mobility and migration patterns. Nature 484, 96–100 (2012). URL https: //doi.org/10.1038/nature10856.

[31] Yuan, Z. et al. Modelling the efects of Wuhan’s lockdown during COVID-19, China. Bulletin of the World Health Organization 98, 484–494 (2020). URL https://doi.org/10.2471/BLT.20.254045.

[32] Ponce-de Leon, M. et al. COVID-19 Flow-Maps an open geographic information system on COVID-19 and human mobility for Spain. Scientific Data 8, 310 (2021). URL https://doi.org/10.1038/s41597-021-01093-5.

[33] Galmiche, S. et al. SARS-CoV-2 incubation period across variants of concern, individual factors, and circumstances of infection in France: A case series analysis from the ComCor study. The Lancet Microbe 4, e409–e417 (2023). URL https: //doi.org/10.1016/S2666-5247(23)00005-8.

[34] Li, Q., Guan, X., Wu, P. et al. Early transmission dynamics in Wuhan, China, of novel coronavirus–infected pneumonia. New England Journal of Medicine 382, 1199–1207 (2020). URL https://doi.org/10.1056/NEJMoa2001316.

[35] Prem, K. et al. Projecting contact matrices in 177 geographical regions: An update and comparison with empirical data for the COVID-19 era. PLOS Computational Biology 17, e1009098 (2021). URL https://doi.org/10.1371/journal. pcbi.1009098.

[36] Kermack, W. O. & McKendrick, A. G. A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society A 115, 700–721 (1927). URL https://doi.org/10.1098/rspa.1927.0118.

[37] Hochreiter, S. & Schmidhuber, J. Long short-term memory. Neural Computation 9, 1735–1780 (1997). URL https://doi.org/10.1162/neco.1997.9.8.1735.

[38] Cho, K. et al. Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Moschitti, A., Pang, B. & Daelemans, W. (eds) Proceedings of the 2014 Conference on Empirical Methods in Natural Language

Processing (EMNLP), 1724–1734 (Association for Computational Linguistics, Doha, Qatar, 2014). URL https://aclanthology.org/D14-1179/.

[39] Chen, R. T. Q., Rubanova, Y., Bettencourt, J. & Duvenaud, D. K. Neural ordinary diferential equations. In Advances in Neural Information Processing Systems, Vol. 31, 6572–6583 (Curran Associates, Inc., 2018). URL https://proceedings.neurips.cc/paper/2018/hash/ 69386f6bb1dfed68692a24c8686939b9-Abstract.html.

[40] Yu, B., Yin, H. & Zhu, Z. Spatio-temporal graph convolutional networks: A deep learning framework for trafic forecasting. In Lang, J. (ed.) Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, 3634– 3640 (International Joint Conferences on Artificial Intelligence Organization, 2018). URL https://doi.org/10.24963/ijcai.2018/505.

[41] Li, Y., Yu, R., Shahabi, C. & Liu, Y. Difusion convolutional recurrent neural network: Data-driven trafic forecasting. In International Conference on Learning Representations (2018). URL https://openreview.net/forum?id=SJiHXGWAZ.

[42] Bai, L., Yao, L., Li, C., Wang, X. & Wang, C. Adaptive graph convolutional recurrent network for trafic forecasting. In Advances in Neural Information Processing Systems, Vol. 33, 17804–17815 (Curran Associates, Inc., 2020). URL https://proceedings.neurips.cc/paper/2020/hash/ ce1aad92b939420fc17005e5461e6f48-Abstract.html.

[43] Wu, Z., Pan, S., Long, G., Jiang, J. & Zhang, C. Graph WaveNet for deep spatial–temporal graph modeling. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, 1907–1913 (International Joint Conferences on Artificial Intelligence Organization, 2019). URL https: //doi.org/10.24963/ijcai.2019/264.

[44] Wu, Z. et al. Connecting the dots: Multivariate time series forecasting with graph neural networks. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 753–763 (Association for Computing Machinery, 2020). URL https://doi.org/10.1145/3394486.3403118.

[45] Liu, M., Liu, Y. & Liu, J. Epidemiology-aware deep learning for infectious disease dynamics prediction. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, 4084–4088 (Association for Computing Machinery, 2023). URL https://doi.org/10.1145/3583780.3615139.

[46] Kapoor, A. et al. Examining COVID-19 forecasting using spatio-temporal GNNs. In Proceedings of the 16th International Workshop on Mining and Learning with Graphs (MLG) (2020). URL https://www.mlgworkshop.org/2020/papers/ MLG2020 paper 26.pdf.

[47] Gao, J. et al. STAN: Spatio-temporal attention network for pandemic prediction using real-world evidence. Journal of the American Medical Informatics Association 28, 733–743 (2021). URL https://doi.org/10.1093/jamia/ocaa322.

[48] Han, Z., Xi, Y., Xia, T., Liu, Y. & Li, Y. Devil in the landscapes: Inferring epidemic exposure risks from street view imagery. In Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems, Article 53 (Association for Computing Machinery, 2023). URL https://doi.org/ 10.1145/3589132.3625596.

[49] Rodr´ıguez, A., Cui, J., Ramakrishnan, N., Adhikari, B. & Prakash, B. A. EINNs: Epidemiologically-informed neural networks. Proceedings ofthe AAAI Conference on Artificial Intelligence 37, 14453–14460 (2023). URL https://doi.org/10.1609/ aaai.v37i12.26690.

## Acknowledgements

This research was partially supported by the National Natural Science Foundation of China (grant numbers 12522124 and 12371502). B.T. was also supported by the Young Talent Support Plan of Xi’an Jiaotong University.

## Author contributions

J.D., Q.L. and B.T. designed the study. J.D. wrote the code, conducted the numerical experiments and analyzed the data. All authors discussed the results and contributed to writing the manuscript.

## Competing interests

The authors declare no competing interests.

Correspondence and requests for materials should be addressed to Qian Li or Biao Tang.

## Supplementary Information

## S1 Supplementary Methods

## S1.1 Synthetic Data Generation

This section provides the complete settings for synthetic data generation, including the unified dynamical equations, scenario-level parameters, contact and mobility structures, mechanism-specific parameters, and observation noise. The controlled design comprises three structural settings—Lagrangian infection generation, Eulerian population transport, and coupled multiscale dynamics—and 14 generation conditions in total: four Lagrangian mechanisms, two Eulerian mechanisms, and their eight pairwise multiscale combinations. These conditions span baseline mass-action, explicitly timemodulated, state-dependent, nonlinear, and coupled multiscale interaction dynamics. All systems are sampled daily over 0–200 days. Within each structural setting, population sizes, initial states, basic epidemiological parameters, and observation-noise rules are held fixed so that diferences among trajectories primarily reflect the prescribed interaction mechanisms.

## S1.1.1 Dynamical System

Let $S _ { j } ^ { k } ( t ) , I _ { j } ^ { k } ( t )$ , and $R _ { j } ^ { k } ( t )$ denote the susceptible, infectious and removed populations of age group $j \in \{ 1 , \dotsc , A \}$ in region $k \in \{ 1 , \ldots , M \}$ , respectively. The simulated trajectories are generated by the following coupled Lagrangian–Eulerian system:

$$
\left\{ \begin{array} { l l } { \displaystyle \frac { d S _ { j } ^ { k } } { d t } = - \sum _ { i } \mathcal { F } _ { i \to j } ^ { L } ( t ) I _ { i } ^ { k } + \sum _ { l } \mathcal { F } _ { l \to k , j } ^ { E } ( t ) S _ { j } ^ { l } , } \\ { \displaystyle \frac { d I _ { j } ^ { k } } { d t } = \sum _ { i } \mathcal { F } _ { i \to j } ^ { L } ( t ) I _ { i } ^ { k } + \sum _ { l } \mathcal { F } _ { l \to k , j } ^ { E } ( t ) I _ { j } ^ { l } - \gamma I _ { j } ^ { k } , } \\ { \displaystyle \frac { d R _ { j } ^ { k } } { d t } = \gamma I _ { j } ^ { k } + \sum _ { l } \mathcal { F } _ { l \to k , j } ^ { E } ( t ) R _ { j } ^ { l } . } \end{array} \right.\tag{S1}
$$

Here, $\mathcal { F } _ { i  j } ^ { L } ( t )$ represents the Lagrangian infection efect from age group i to age group $j ,$ whose specific formulation is provided in Supplementary Table $1 ; \gamma$ denotes the removal rate of infectious individuals. In the multiscale scenario, the Lagrangian interaction incorporates spatial heterogeneity through the region-specific contact matrix $C _ { k } .$ For simplicity, the region index is not explicitly included in $\mathcal { F } _ { i  j } ^ { L } ( t )$ . Here, i indexes the infectious source and $j$ the susceptible target. The current source-group population is $P _ { i } ^ { k } ( t ) = S _ { i } ^ { k } ( t ) + I _ { i } ^ { k } ( t ) + R _ { i } ^ { k } ( t )$

For Eulerian mobility, $\mathcal { M } ^ { l k } ( t )$ denotes the OD mobility value from region l to region $k , \ N ^ { l }$ denotes the population size of source region $l ,$ and $\eta _ { j }$ represents the relative mobility propensity of age group $j .$ The Eulerian mobility rate between regions for age group j is defined as

$$
\mathcal { F } _ { l  k , j } ^ { E } ( t ) = \{ \begin{array} { l l } { \displaystyle \frac { \eta _ { j } \mathcal { M } ^ { l k } ( t ) } { N ^ { l } } , } & { l \neq k , } \\ { - \displaystyle \sum _ { m \neq l } \mathcal { F } _ { l  m , j } ^ { E } ( t ) , } & { l = k . } \end{array}\tag{S2}
$$

By this definition, for any source region l and age group $j ,$ we have

$$
\sum _ { k } \mathcal { F } _ { l  k , j } ^ { E } ( t ) = 0 .
$$

Supplementary Table 1: Synthetic dynamical mechanisms. Functional forms and dynamical classes used to generate the Lagrangian, Eulerian and multiscale synthetic systems. The eight multiscale conditions combine each Lagrangian mechanism with each Eulerian mechanism. Here, $N ^ { l }$ and $N ^ { k }$ are the fixed reference total populations of the source and destination regions, respectively, and $N _ { 0 }$ is the reference population used to scale the gravity terms.
<table><tr><td>Mechanism</td><td>Dynamical class</td><td>Functional form</td></tr><tr><td colspan="3">Lagrangian mechanisms</td></tr><tr><td>Baseline Mass-Action Contact Mechanism (BMCM)</td><td>Baseline mass-action</td><td> $\mathcal { F } _ { i  j } ^ { L } ( t ) = \beta C _ { i j } ^ { k } \frac { S _ { j } ^ { k } ( t ) } { P _ { i } ^ { k } ( t ) }$ </td></tr><tr><td>Time-Decaying Contact Mechanism (TDCM)</td><td>Explicitly time-modulated</td><td> $\mathcal { F } _ { i  j } ^ { L } ( t ) = \beta e ^ { - r _ { L } t } C _ { i j } ^ { k } \frac { S _ { j } ^ { k } ( t ) } { P _ { i } ^ { k } ( t ) }$ </td></tr><tr><td>Infection-Avoidant Contact Mechanism (IACM)</td><td>State-dependent feedback</td><td> $\mathcal { F } _ { i  j } ^ { L } ( t ) = \beta e ^ { - \alpha _ { I } I _ { i } ^ { k } ( t ) } C _ { i j } ^ { k } \frac { S _ { j } ^ { k } ( t ) } { P _ { i } ^ { k } ( t ) }$ </td></tr><tr><td>Nonlinear Mass-Action Contact Mechanism (NMCM)</td><td>Nonlinear state dependence</td><td> $\mathcal { F } _ { i  j } ^ { L } ( t ) = \beta [ I _ { i } ^ { k } ( t ) ] ^ { q _ { I } } C _ { i j } ^ { k } \frac { [ S _ { j } ^ { k } ( t ) ] ^ { p _ { S } } } { P _ { i } ^ { k } ( t ) }$ </td></tr><tr><td colspan="3">Eulerian mechanisms</td></tr><tr><td>Baseline Risk-Aware Gravity Mobility (BRGM)</td><td>State-dependent mobility</td><td> $\mathcal { M } _ { \mathrm { B R G M } } ^ { l k } = \kappa \left( \frac { N ^ { l } } { N _ { 0 } } \right) ^ { a } \left( \frac { N ^ { k } } { N _ { 0 } } \right) ^ { b } e ^ { - \lambda d _ { l k } - \rho _ { s } \bar { I } ^ { l } - \rho _ { d } \bar { I } ^ { k } }$ </td></tr><tr><td>Controlled Risk-Aware Gravity Mobility (CRGM)</td><td>State- and time- modulated mobility</td><td> $\mathcal { M } _ { \mathrm { C R G M } } ^ { l k } = \mathcal { M } _ { \mathrm { B R G M } } ^ { \mathit { i k } } e ^ { - \mathit { r } _ { E } t }$ </td></tr><tr><td colspan="3">Multiscale mechanisms</td></tr><tr><td>Lagrangian-Eulerian coupled mechanisms</td><td>Coupled multiscale dynamics</td><td> $\{ \mathrm { B M C M , T D C M , I A C M , N M C M } \}$   $\times \{ \mathrm { B R G M } , \mathrm { C R G M } \}$ </td></tr></table>

Therefore, the Eulerian term only describes the redistribution of the same age group across diferent regions, without introducing additional population increase or decrease.

The three synthetic scenarios are obtained by diferent configurations of Equation (S1). The Lagrangian scenario sets $M = 1$ and disables Eulerian mobility. The Eulerian scenario sets $A = 1$ and fixes the local contact intensity to 1. The multiscale scenario sets $M = A = 4$ and simultaneously activates both Lagrangian contact and Eulerian mobility mechanisms.

## S1.1.2 Mechanism Definitions

Supplementary Table 1 summarizes the Lagrangian infection mechanisms and Eulerian mobility mechanisms used for synthetic data generation. The multiscale scenario combines the four Lagrangian mechanisms with the two Eulerian mechanisms in all possible pairs. Regional epidemic activity is defined as

$$
\bar { I } _ { * } ^ { k } ( t ) = \frac { \gamma } { N ^ { k } } \sum _ { j = 1 } ^ { A } I _ { j } ^ { k } ( t ) ,
$$

where $N ^ { k }$ is the fixed reference total population of region k. The infectious states and population denominator are expressed on the same population scale.

The mechanisms are not intended to reproduce one particular epidemic outbreak. Instead, they provide controlled systems with known generating rules for evaluating forecasting and process-level dynamical fidelity across qualitatively diferent interaction dynamics. Together, the 14 conditions cover a stationary baseline, explicit temporal modulation, endogenous state-dependent feedback, nonlinear state dependence, and coupled age–space dynamics. The Lagrangian mechanisms vary infectiongeneration interactions driven by age-specific contact structure within regions, whereas the Eulerian mechanisms vary conservative population transport across regions.

## Lagrangian Infection Mechanisms.

The Lagrangian mechanisms are used to generate baseline, explicitly time-modulated, state-dependent, and nonlinear variations in local infection-generation intensity.

Baseline Mass-Action Contact Mechanism (BMCM) represents the basic transmission process under a stable contact environment. This mechanism assumes that age-specific contact patterns remain approximately unchanged over time. Infection generation is primarily determined by the predefined contact structure, without considering additional behavioral adjustments or intervention efects, and therefore serves as the baseline scenario for other non-stationary transmission mechanisms.

Time-Decaying Contact Mechanism (TDCM) introduces explicit temporal modulation through a gradually decreasing interaction intensity. This stylized mechanism represents settings in which efective contacts decline over time because of interventions, increasing risk awareness, or behavioral change.

Infection-Avoidant Contact Mechanism (IACM) introduces endogenous statedependent feedback through infection-responsive contact adjustment. As infection levels increase, efective interaction intensity changes with the evolving epidemic state. Nonlinear Mass-Action Contact Mechanism (NMCM) introduces nonlinear state dependence in infection generation, providing a controlled setting in which interaction strength responds nonlinearly to susceptible and infectious states.

## Eulerian Mobility Mechanisms.

The Eulerian mechanisms are used to generate state-dependent and explicitly time-modulated redistribution of epidemic states through interregional population transport.

Baseline Risk-Aware Gravity Mobility (BRGM) defines state-dependent mobility in which interregional transport varies with source- and destination-region epidemic activity as well as population and distance efects.

Controlled Risk-Aware Gravity Mobility (CRGM) adds explicit temporal modulation to the risk-aware mobility process, providing a controlled representation of progressively strengthened travel restrictions or other time-dependent mobility suppression.

## S1.1.3 Scenario-level Settings

The basic parameters of the three scenarios are summarized in Supplementary Table 2. Here, M and A denote the number of regions and age groups, respectively, and η represents the relative mobility propensity of diferent age groups. During synthetic data generation, the compartment states $( S , I , R )$ and the observation $\gamma I$ are scaled $1 0 ^ { - 4 }$ in the Lagrangian and multiscale settings and by $1 0 ^ { - 5 }$ in the Eulerian setting. The population sizes and initial conditions reported below are given in individuals.

Within each structural setting, the population sizes and initial infection states are fixed across generation conditions. For the Lagrangian scenario,

$$
N ^ { L } = 1 0 ^ { 3 } \left[ 8 2 \ 1 0 2 \ 1 3 0 \ 6 5 \right] , \qquad I ^ { L } ( 0 ) = \left[ 1 9 0 0 \ 1 0 0 0 \ 1 2 0 0 \ 1 3 0 0 \right] .
$$

Supplementary Table 2: Synthetic scenario settings. Shared configurations used to generate the Lagrangian, Eulerian and multiscale systems.
<table><tr><td>Setting</td><td>Lagrangian</td><td>Eulerian</td><td>Multiscale</td></tr><tr><td>Graph size</td><td> $M = 1 , \ A = 4$ </td><td> $M = 4 , \ A = 1$ </td><td> $M = 4 , \ A = 4$ </td></tr><tr><td>Simulation horizon</td><td> $0 { - } 2 0 0 ~ \mathrm { d a y s }$ </td><td>0–200 days</td><td>0–200 days</td></tr><tr><td>Sampling interval</td><td>1 day</td><td>1 day</td><td>1 day</td></tr><tr><td>Transmission rate  $\beta$ </td><td>0.01</td><td>0.20</td><td>0.02</td></tr><tr><td>Removal rate γ</td><td>0.02</td><td>0.08</td><td>0.021</td></tr><tr><td>Mobility propensity  $\eta$ </td><td></td><td>[1]</td><td>[0.106, 0.685, 0.156, 0.053]</td></tr><tr><td>Gravity exponents  $( a , b )$ </td><td></td><td>(1, 1)</td><td>(1, 1)</td></tr><tr><td>Distance decay λ</td><td></td><td>0.012</td><td>0.012</td></tr><tr><td>Gravity scale κ</td><td></td><td>3000</td><td>500000</td></tr><tr><td>Reference population  $N _ { 0 }$ </td><td></td><td> $1 0 ^ { 5 }$ </td><td> $1 0 ^ { 6 }$ </td></tr><tr><td>Risk coefficients  $( \rho _ { s } , \rho _ { d } )$ </td><td></td><td>(6,10)</td><td>(6,10)</td></tr><tr><td>ODE solver</td><td>LSODA</td><td>LSODA</td><td>LSODA</td></tr></table>

For the Eulerian scenario,

$$
N ^ { E } = 1 0 ^ { 3 } \left[ \begin{array} { l } { 1 3 0 } \\ { 1 0 0 } \\ { 9 0 } \\ { 8 0 } \end{array} \right] , \qquad I ^ { E } ( 0 ) = \left[ \begin{array} { l } { 5 0 0 } \\ { 0 } \\ { 0 } \\ { 0 } \end{array} \right] .
$$

For the multiscale scenario,

$$
N ^ { M } = 1 0 ^ { 3 } \left[ ^ { 1 2 0 ~ 1 6 5 ~ 1 9 0 ~ 1 1 5 } _ { 1 1 0 ~ 1 5 5 } \right] { } ~ ,
$$

$$
\pmb { I } ^ { M } ( 0 ) = \left[ \begin{array} { c c c c } { 1 9 0 0 } & { 1 0 0 0 } & { 1 2 0 0 } & { 1 3 0 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right] .
$$

The rows and columns of the population matrices and initial infection matrices correspond to regions and age groups, respectively. All scenarios satisfy

$$
\pmb { R } ( 0 ) = 0 , \qquad \pmb { S } ( 0 ) = \pmb { N } - \pmb { I } ( 0 ) .
$$

The initial infections in the Eulerian and multiscale scenarios are concentrated in the first region, providing a clear initial spatial transmission condition and enabling the investigation of interregional redistribution patterns generated by diferent mobility mechanisms.

## S1.1.4 Contact and Mobility Structures

The Lagrangian scenario adopts a fixed baseline contact matrix

$$
C ^ { L } = \left[ { 2 . 8 8 ~ 3 . 0 0 ~ 2 . 5 4 ~ 3 . 1 3 } \right] .
$$

Supplementary Table 3: Mechanism-specific parameters. Parameter values used by the explicitly time-modulated, state-dependent and nonlinear synthetic mechanisms.
<table><tr><td>Scenario</td><td>TDCM:  $r _ { L }$ </td><td>IACM:  $\alpha _ { I }$ </td><td>NMCM:  $( p _ { S } , q _ { I } )$ </td><td>CRGM:  $r _ { E }$ </td></tr><tr><td>Lagrangian</td><td>0.01</td><td>0.10</td><td>(1.0, 0.4)</td><td></td></tr><tr><td>Eulerian</td><td></td><td></td><td></td><td>0.30</td></tr><tr><td>Multiscale</td><td>0.0035</td><td>0.045</td><td>(1.0, 0.2)</td><td>0.03</td></tr></table>

BMCM, TDCM, IACM, and NMCM all generate corresponding Lagrangian infection efects based on this contact structure.

In the Eulerian scenario, each region contains only one age group, and the local contact intensity is uniformly set to 1.0. The interregional distance matrix is defined as

$$
D ^ { E } = \left[ { 1 2 0 \ { 1 2 0 \ 2 1 0 \ 2 6 0 } } \right] .
$$

The multiscale scenario adopts a region-dependent baseline contact matrix

$$
C _ { k } = s _ { k } C _ { 0 } , \qquad [ s _ { 1 } , s _ { 2 } , s _ { 3 } , s _ { 4 } ] = [ 1 . 0 0 , 0 . 9 8 , 0 . 9 5 , 0 . 9 2 ] ,
$$

where

$$
C _ { 0 } = \left[ \begin{array} { c c c c c c } { 1 . 6 0 0 } & { 1 . 4 6 0 } & { 1 . 3 9 5 } & { 1 . 5 8 5 } \\ { 1 . 5 5 0 } & { 1 . 5 0 0 } & { 1 . 4 2 0 } & { 1 . 5 6 5 } \\ { 1 . 6 0 5 } & { 1 . 5 8 0 } & { 1 . 4 5 0 } & { 1 . 6 5 5 } \\ { 1 . 6 1 5 } & { 1 . 4 5 0 } & { 1 . 3 7 0 } & { 1 . 5 0 0 } \end{array} \right] .
$$

The corresponding interregional distance matrix is

$$
D ^ { M } = \left[ 1 2 0 \ 0 \ 1 3 0 \ 2 2 0 \right] .
$$

The Lagrangian scenario does not include Eulerian mobility. In the Eulerian and multiscale scenarios, $\mathcal { M } ^ { l k } ( t )$ is generated by BRGM or CRGM defined in Supplementary Table 1, respectively, and is converted into the corresponding Eulerian mobility rate $\mathcal { F } _ { l  k , j } ^ { E } ( t )$ for each age group according to Equation (S2).

## S1.1.5 Mechanism-specific Parameters

In addition to the scenario-level shared parameters described above, TDCM, IACM, NMCM, and CRGM introduce additional mechanism-specific parameters. Their values are provided in Supplementary Table 3. BMCM and BRGM serve as the baseline mechanisms for Lagrangian and Eulerian dynamics, respectively, and do not introduce additional temporal modulation parameters.

The multiscale scenario combines the four Lagrangian mechanisms {BMCM, TDCM, IACM, NMCM} with the two Eulerian mechanisms {BRGM, CRGM} in all possible pairs, resulting in eight coupled Lagrangian–Eulerian mechanism combinations. All combinations share the same population sizes, initial states, basic epidemiological parameters, baseline contact structures, and region distance matrices.

Therefore, the systematic diferences among generated trajectories mainly arise from the adopted dynamical mechanisms.

## S1.1.6 Observation Noise

To avoid making the simulated observations identical to the deterministic numerical trajectories, we add mild observation noise to the infection states and OD mobility data, respectively. The noise is applied only at the observation level and does not participate in the state evolution described by Equation (S1).

For the noise-free infection sequence $I _ { v } ( t )$ of node $v ,$ the standard deviation of Gaussian noise is set to 5% of its maximum value over the entire simulation period:

$$
\begin{array} { r } { \widetilde { I } _ { v } ( t ) = \operatorname* { m a x } \left\{ 0 , I _ { v } ( t ) + \epsilon _ { v } ( t ) \right\} , \qquad \epsilon _ { v } ( t ) \sim \mathcal { N } \left( 0 , [ 0 . 0 5 I _ { v , \operatorname* { m a x } } ] ^ { 2 } \right) , } \end{array}
$$

where

$$
I _ { v , \operatorname* { m a x } } = \operatorname* { m a x } _ { t } I _ { v } ( t ) .
$$

The corresponding observed case count is calculated as

$$
\widetilde { Y } _ { v } ( t ) = \gamma \widetilde { I _ { v } } ( t ) .
$$

The values of $S _ { v } ( t )$ and $R _ { v } ( t )$ remain the results of noise-free numerical integration.

For the Eulerian and multiscale scenarios, noise is further independently added to each directed OD mobility sequence. Let $\mathcal { M } ^ { l k } ( t )$ denote the noise-free OD mobility value from region l to region k. Then,

$$
\widetilde { \mathcal { M } } ^ { l k } ( t ) = \operatorname* { m a x } \left\{ 0 , \mathcal { M } ^ { l k } ( t ) + \epsilon _ { l k } ( t ) \right\} , \qquad \epsilon _ { l k } ( t ) \sim \mathcal { N } \left( 0 , [ 0 . 0 5 \mathcal { M } _ { \mathrm { m a x } } ^ { l k } ] ^ { 2 } \right) , \quad l \neq k ,
$$

where

$$
\mathcal { M } _ { \mathrm { m a x } } ^ { l k } = \operatorname* { m a x } _ { t } \left| \mathcal { M } ^ { l k } ( t ) \right| .
$$

The noisy OD matrices preserve zero diagonal entries and are used to reconstruct the Eulerian mobility rates for each age group according to Equation (S2). The infection-state noise and OD mobility noise use fixed random seeds 42 and 43, respectively, to ensure reproducible comparisons across diferent mechanisms.

Therefore, within the same scenario, all generated datasets share identical initial conditions, basic parameters, and observation perturbation rules. Their primary dynamical diferences arise from the predefined Lagrangian mechanisms, Eulerian mechanisms, or their combinations.

## S1.1.7 Synthetic Data Overview

Supplementary Fig. 1 presents infection trajectories generated by the isolated Lagrangian and Eulerian mechanisms. The Lagrangian setting contrasts baseline massaction, explicitly time-modulated, state-dependent and nonlinear infection-generation dynamics, whereas the Eulerian setting contrasts BRGM with the additional temporal modulation introduced by CRGM. Supplementary Fig. 2 extends this overview to the eight pairwise Lagrangian–Eulerian combinations in the coupled multiscale setting, showing their complementary age–time and city–time infection-intensity patterns.

Because population sizes, initial states, basic epidemiological parameters and observation rules are held fixed within each structural setting, systematic diferences across these trajectories and heatmaps primarily reflect the prescribed interaction mechanisms.

![](images/8161ce595a65ae6398f5a219a611a28ff8b3eadb324018865dee9a3e618a8dca.jpg)

![](images/b8709e6175a0483fcdfc6117c62160ce47042bece254a7796697d1ca667ace86.jpg)

![](images/470c7ee69583470410dc562f4c4f89fbe7254ea933e13466f10c7545ca0a84a1.jpg)

![](images/5c9292ed73108cfb106c68a5d3ac0a3e401872388695aece4446bdd1db5410a9.jpg)  
(a) Four Lagrangian infection-generation mechanisms.

![](images/aac86954879c4a97f04a3c597e25d78291a2cd7e55bdb2b86eaf106b7b4c8fce.jpg)

![](images/a235464f5e3d87e205847ee213fb953b06cabd7de884d37cc9bd8550199e0fe3.jpg)  
(b) Two Eulerian population-transport mechanisms.

Supplementary Fig. 1: Synthetic trajectories in the isolated Lagrangian and Eulerian settings. Heatmaps show normalized infectious-state trajectories $\bar { I } _ { j } ^ { k } ( t )$ generated by the controlled interaction mechanisms. Within each structural setting, population sizes, initial states, basic epidemiological parameters and observation rules are held fixed so that diferences among trajectories primarily reflect the prescribed dynamical mechanisms.

![](images/b589f0bb3e0c341915ef8004f91e687005743f54be52b03e70762ff3e1431419.jpg)

![](images/a4674b310e9f95fa69599d8db81f4a40c044f4cb0c1d0e62379f270025d8e50c.jpg)  
Supplementary Fig. 2: Multiscale ground-truth signatures of the coupled synthetic mechanisms. Normalized infection-intensity patterns are shown for the eight pairwise combinations of four Lagrangian mechanisms (BMCM, TDCM, IACM and NMCM) and two Eulerian mechanisms (BRGM and CRGM). The left block shows age–time patterns across four age groups, whereas the right block shows the corresponding city–time patterns across four cities. In both blocks, rows correspond to the four Lagrangian mechanisms and columns correspond to BRGM and CRGM. Comparisons across rows under the same Eulerian mechanism highlight Lagrangian mechanism-specific dynamical signatures, whereas comparisons between BRGM and CRGM within the same row highlight the additional efects associated with the Eulerian mobility mechanism. Together, these patterns illustrate the complementary ageand region-level signatures generated by the coupled multiscale mechanisms.

Supplementary Table 4: Data and evaluation settings for the three realworld COVID-19 datasets.
<table><tr><td>Setting</td><td>Ontario</td><td>Hubei</td><td>Spain</td></tr><tr><td>Epidemic setting State structure</td><td>Age structured</td><td>Mobility driven</td><td>Coupled age-space 4 regions × 3 age</td></tr><tr><td></td><td>1 region × 4 age groups</td><td>3 regions × 1 age group</td><td>groups</td></tr><tr><td>Epidemic observations</td><td>Age-stratified daily cases</td><td>City-level daily cases</td><td>Region-age daily cases</td></tr><tr><td>Process-aligned information</td><td>Age-contact matrix</td><td>Daily OD mobility; population and distance</td><td>Age-contact matrices; daily OD mobility; population and distance</td></tr><tr><td>Observation period</td><td>28 Nov. 2021–27 Feb. 2022</td><td>16 Jan. 2020–25 Feb. 2020</td><td>1 Mar. 2020–21 Jun. 2020</td></tr><tr><td>Training</td><td>25 days</td><td>15 days</td><td>20 days</td></tr><tr><td>Forecasting</td><td>7 / 14 / 35 / 67 days</td><td>7 / 14 / 26 days</td><td>7 / 14 / 49 / 93 days</td></tr></table>

## S1.2 Real-world Data and Mobility Predictor Settings

The observation structures and evaluation settings of the three real-world datasets are summarized in Supplementary Table 4. This section supplements the input feature settings of the auxiliary mobility predictor ${ \mathrm { M L P } } _ { \mathrm { m o b } }$ across settings involving Eulerian transport. All such experiments use the same auxiliary mobility-predictor architecture:

$$
\hat { \mathcal { M } } ^ { l k } ( t ) = \mathrm { s o f t p l u s } \left[ \mathrm { M L P } _ { \mathrm { m o b } } \left( z _ { l k } ( t ) ; \vartheta \right) \right] ,\tag{S3}
$$

where $z _ { l k } ( t )$ denotes the dataset-specific input vector defined below. The predictor estimates directed OD mobility values using epidemic activity, spatial gravity priors and temporal information, with regional identity encodings additionally included for Spain.

## S1.2.1 Common Input Representation

For a directed edge $l  k ,$ the basic input is defined as

$$
z _ { l k } ^ { \mathrm { b a s e } } ( t ) = \bar { I } _ { * } ^ { l } ( t ) \parallel \bar { I } _ { * } ^ { k } ( t ) \parallel e _ { l k } ^ { E } \parallel t _ { \mathrm { n o r m } } ,
$$

where $\bar { I } _ { * } ^ { l } ( t )$ and $\bar { I } _ { * } ^ { k } ( t )$ denote the source and destination epidemic activities, using observed reporting activity during mobility pretraining and model-generated reporting activity during epidemic training and forecasting. The term $e _ { l k } ^ { E }$ is the gravity prior constructed from population and spatial distance information, and $t _ { \mathrm { n o r m } }$ represents the normalized temporal coordinate.

## S1.2.2 Hubei Input Setting

The Hubei experiment directly adopts the above basic input representation:

$$
z _ { l k } ^ { \mathrm { H B } } ( t ) = \bar { I } _ { * } ^ { l } ( t ) \parallel \bar { I } _ { * } ^ { k } ( t ) \parallel e _ { l k } ^ { E } \parallel t _ { \mathrm { n o r m } } .
$$

The Hubei OD sequences mainly exhibit low-frequency variations. Therefore, no additional periodic temporal features or regional identity features are introduced.

Supplementary Table 5: Mobility-predictor inputs. Input features used by the auxiliary mobility network in the synthetic, Hubei and Spain datasets.
<table><tr><td>Input feature</td><td>Synthetic</td><td>Hubei</td><td>Spain</td></tr><tr><td>Source epidemic intensity</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Destination epidemic intensity</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Gravity prior</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Normalized time</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Weekly periodic encoding</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Source region identity</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Destination region identity</td><td>No</td><td>No</td><td>Yes</td></tr></table>

## S1.2.3 Spain Input Setting

The Spain OD sequences exhibit more apparent short-period fluctuations. To enhance the ability of the auxiliary mobility network to capture such temporal patterns, we adopt a training-window normalized and clipped temporal coordinate

$$
\tau ( t ) = \operatorname* { m i n } \left. \operatorname* { m a x } \left( \frac { t } { T _ { \mathrm { t r a i n } } - 1 } , 0 \right) , 1 \right. ,
$$

and incorporate weekly periodic sine and cosine features:

$$
\psi _ { \mathrm { E S } } ( t ) = \left[ \tau ( t ) , \sin \left( \frac { 2 \pi t } { 7 } \right) , \cos \left( \frac { 2 \pi t } { 7 } \right) \right] .
$$

In addition, considering the stable spatial heterogeneity among diferent aggregated regions, we further include one-hot identity encodings $r _ { l }$ and $r _ { k }$ for the source and destination regions. Therefore, the input representation for the Spain experiment is

$$
z _ { l k } ^ { \mathrm { E S } } ( t ) = \bar { I } _ { * } ^ { l } ( t ) \parallel \bar { I } _ { * } ^ { k } ( t ) \parallel e _ { l k } ^ { E } \parallel \psi _ { \mathrm { E S } } ( t ) \parallel r _ { l } \parallel r _ { k } .
$$

## S1.2.4 Input Feature Summary

Supplementary Table 5 summarizes the mobility-predictor inputs for the synthetic, Hubei and Spain datasets. The synthetic Eulerian and multiscale experiments use the same basic representation as Hubei: source and destination epidemic activity, the gravity prior and normalized time, without periodic temporal features or regional identity encodings.

## S1.3 Baseline Model Configurations

This section summarizes the implementation details, model architectures, and final hyperparameters of the 21 comparator models used across the main experiments. The comparator set spans mechanistic models, general sequence and spatiotemporal forecasting models, epidemic-specific graph models, and hybrid mechanistic–learning models; not every comparator is used in every setting, as detailed below. Supplementary Table 6 describes the experimental roles and study-specific implementations; Supplementary Table 7 reports the final configuration ranges used across the 14 synthetic generation conditions; Supplementary Table 8 reports the specific configurations on the three real-world datasets, Ontario, Hubei, and Spain. All configurations correspond to the runs producing the main results rather than to predefined hyperparameter search spaces.

In the synthetic experiment table, numerical ranges represent the minimum and maximum values of the configurations actually adopted across the 14 generation conditions. In the real-world experiment table, unless otherwise specified, three values separated by slashes correspond to Ontario, Hubei, and Spain, respectively. For deep learning baselines, Hidden/layers denotes the hidden dimension and number of network layers, Dropout/batch denotes the dropout rate and batch size, and Input window denotes the number of historical observation days used to construct each prediction sample. Model-specific settings summarize model-specific parameters that cannot be included in the above general categories, including ODE solver settings, graph convolution orders, embedding dimensions, and propagation depths. All deep learning baselines use direct 7-day prediction outputs. M-Graphormer is a paper-level reproduction based on the descriptions in the original paper, and EpiGCN is an adaptation of the original EpiGCN graph convolution module for continuous-time prediction.

Supplementary Table 6: Baseline implementations. Experimental roles and study-specific configurations of the comparison models.
<table><tr><td>Model</td><td>Role</td><td>Implementation in this study</td></tr><tr><td>SIR</td><td>Synthetic baseline</td><td>Independent nodewise SIR dynamics with nonnegative initial compartments and node-specific transmission and removal parameters fitted to the complete training trajectory.</td></tr><tr><td>MepoSIR</td><td>Synthetic baseline</td><td>Contact-coupled metapopulation SIR using the observed age-contact matrix, region-shared age-specific transmission parameters and node-specific removal parameters.</td></tr><tr><td>LSTM</td><td>Synthetic baseline</td><td>Stacked LSTM encoder with a direct 7-day regression head, trained on permitted node histories and rolled out</td></tr><tr><td>GRU</td><td>Synthetic baseline</td><td>autoregressively in 7-day blocks. Stacked GRU encoder with the same direct 7-day prediction and blockwise autoregressive rollout protocol as LSTM.</td></tr><tr><td>Neural ODE</td><td>Synthetic baseline</td><td>MLP vector field integrated by fixed-step RK4, with the historical window encoded as the initial latent state and decoded into a 7-day trajectory.</td></tr><tr><td>STGCN</td><td>Synthetic baseline</td><td>Temporal-Chebyshev graph convolution blocks using spatial supports constructed from permitted contact, distance and training-period mobility information.</td></tr><tr><td>DCRNN</td><td>Synthetic and real baseline</td><td>Diffusion-convolutional GRU encoder with a single autoregressive decoder and scheduled sampling, generating seven future days per block.</td></tr><tr><td>AGCRN</td><td>Synthetic and real baseline</td><td>Adaptive graph-convolutional recurrence with learned node embeddings and node-specific parameter generation, using only permitted node histories.</td></tr><tr><td>Graph WaveNet</td><td>Synthetic baseline</td><td>Dilated gated temporal convolutions with residual, skip and adaptive graph-propagation paths, followed by a direct 7-day output head.</td></tr><tr><td>MTGNN</td><td>Synthetic and real baseline</td><td>Adaptive graph construction, multiscale dilated temporal convolutions and bidirectional MixProp with a direct multi-step head and dataset-specific graph settings.</td></tr><tr><td>MCM-SEIR</td><td>Real baseline</td><td>Contact- and mobility-coupled metapopulation SEIR using observed contact matrices, a conservative OD-derived mobility generator and fixed dataset-specific incubation rates.</td></tr><tr><td>EpiGNN</td><td>Real baseline</td><td>Historical GRU encoding, epidemic-aware adaptive adjacency and stacked graph propagation, with the original single-target</td></tr><tr><td>Cola-GNN</td><td>Real baseline</td><td>output replaced by a direct 7-day head. Recurrent temporal encoding with a learned dynamic correlation graph and graph propagation, extended from single-step to direct 7-day prediction without future-value</td></tr><tr><td>CovidGNN</td><td>Real baseline</td><td>feedback. Spatio-temporal graph convolution over region-time relations, trained with strictly training-contained sliding windows</td></tr><tr><td>MepoGNN</td><td>Real baseline</td><td>instead of the original whole-graph train/test mask. Graph WaveNet-style temporal blocks, bidirectional mobility propagation, rate heads and metapopulation SIR recursion using training-period OD supports and calibrated physical</td></tr><tr><td>M-Graphormer Real baseline</td><td></td><td>readouts. Paper-level reproduction combining gated TCNs, Graphormer attention, structural encodings, gated dense connections and metapopulation recursion with a direct 7-day head and</td></tr><tr><td>EpiGCN</td><td>Real baseline</td><td>node-level calibration. EpidemicGCN susceptible-infected-removed message propagation used as a continuous-time vector field with node calibration, excluding unavailable street-view features.</td></tr><tr><td>STAN</td><td>Real baseline</td><td>Multi-head GAT and GRU temporal encoding with dual data and epidemic heads, using historical γI, population and causal first differences in place of unavailable compartment observations.</td></tr><tr><td>EINN</td><td>Real baseline</td><td>Bidirectional GRU, masked self-attention, Fourier time features, an SEIRm residual and knowledge distillation, trained on expanding prefixes with a direct 7-day target on γI.</td></tr><tr><td>Epi-Cola-GNNReal baseline</td><td></td><td>Cola-GNN data and epidemic-dynamics heads, both extended from a single target to seven jointly optimized outputs.</td></tr><tr><td>EARTH</td><td>Real baseline</td><td>Controlled temporal path, continuous graph dynamics, epidemic multi-view representations and attention fusion with a direct 7-day output head.</td></tr></table>

<table><tr><td colspan="6">Supplementary Table 7: Synthetic baseline configurations. Ranges summarize the minimum and maximum hyperparameter values used across the 14 synthetic generation conditions, rather than hyperparameter search spaces.</td></tr><tr><td>Model</td><td>Learning rate</td><td></td><td>Hidden/layersDropout/batch</td><td></td><td>Input window Model-specific setting</td></tr><tr><td>SIR</td><td>1.6341 × 10−4-2.5972 × 10−2 N/A</td><td></td><td>N/A</td><td>N/A</td><td>βinit = 0.01703–0.48997; γinit = 0.005572–0.19871</td></tr><tr><td>MepoSIR</td><td>1.1078 × 10−4-4.5878 × 10−2 N/A</td><td></td><td>N/A</td><td>N/A</td><td>βinit = 0.01113–0.78695; γinit = 0.005099–0.36765</td></tr><tr><td>LSTM</td><td>1.6092 × 10−4–4.5127 × 10−3 32–96 / 1–3</td><td></td><td>0-0.3 / 4-16</td><td>7-14</td><td>stacked LSTM encoder with a direct multi-step regression head</td></tr><tr><td>GRU</td><td>1.0611 × 10−4–3.3113 × 10−3 64–96 / 1–3</td><td></td><td>0-0.3 / 4-16</td><td>7-28</td><td>stacked GRU encoder with a direct multi-step regression head</td></tr><tr><td>Neural ODE</td><td>1.0744 × 10−4–4.9943 × 10−3 16–96 / 1–3</td><td></td><td>0-0.3 / 4-16</td><td>7-28</td><td>Fixed-step RK4 solver; step size 0.5–1.0</td></tr><tr><td>STGCN</td><td>2.1638 × 10−4–3.3285 × 10−3 16–96 / 1–3</td><td></td><td>0-0.3 / 4-16</td><td>7-28</td><td>Chebyshev order 2–4; temporal kernel 3–5</td></tr><tr><td>DCRNN</td><td>1.3736 × 10−4-4.8628 × 10−3 16–96 / 1–3</td><td></td><td>0-0.3 / 4-16</td><td>7-28</td><td>diffusion step 1–3</td></tr><tr><td>AGCRN</td><td>1.0276 × 10−4–3.2996 × 10−3 16–96 / 1–3</td><td></td><td>0-0.3 / 4-16</td><td>7-28</td><td>embedding dimension 8–32; Chebyshev order 2-4</td></tr><tr><td>Graph WaveNet</td><td>2.5721 × 10−4–3.0792 × 10−3 16–96 / 1–3</td><td></td><td>0-0.3 / 4-16</td><td>7-28</td><td>adaptive graph propagation with dilated temporal convolu- tion</td></tr><tr><td>MTGNN</td><td>2.8603 × 10−4–3.5913 × 10−3 16–96 / 1–3</td><td></td><td>0.1-0.3 / 4-8</td><td>7-28</td><td>embedding dimension 8–32; graph depth 1-3</td></tr></table>

<table><tr><td colspan="6">Supplementary Table 8: Real-world baseline configurations. Final model architectures and hyperparameters for Ontario, Hubei and Spain.</td></tr><tr><td>Model</td><td>Learning rate</td><td>Hidden/layers</td><td>Dropout/batch</td><td>Input window</td><td>Model-specific setting</td></tr><tr><td>MCM-SEIR</td><td>0.014/0.005/0.020</td><td>N/A</td><td>N/A</td><td>N/A</td><td>βinit = 0.104/0.080/0.300; γinit = 0.13/0.17/0.08</td></tr><tr><td>DCRNN</td><td>0.0085/0.004/0.001056</td><td>16/1; 32/1; 16/1</td><td>0.35/8; 0/8; 0.277/8</td><td>10/3/10</td><td>diffusion step 1/2/3</td></tr><tr><td>AGCRN</td><td>0.0001349/0.003133/0.002033</td><td>64/1; 64/1; 32/3</td><td>0.0258/8; 0.387/8; 0.183/8</td><td>10/3/10</td><td>embedding 8/16/32; Chebyshev order 2/2/3</td></tr><tr><td>MTGNN</td><td>0.003008/0.009526/0.0001996</td><td>16/1; 32/2; 32/1</td><td>0.252/8; 0.153/8; 0.133/8</td><td>14/5/7</td><td>embedding 16/16/32; graph depth 1/2/1</td></tr><tr><td>EpiGNN</td><td>0.0009058/0.003611/0.002874</td><td>64/3; 64/3; 32/1</td><td>0.219/8; 0.0332/8; 0.238/8</td><td>18/5/13</td><td>epidemic-aware adaptive adjacency with a direct 7-day head</td></tr><tr><td>Cola-GNN</td><td>0.0001962/0.003817/0.002995</td><td>16/3; 32/1; 16/2</td><td>0.106/8; 0.382/8; 0.116/8</td><td>10/7/7</td><td>recurrent temporal encoder and dynamic correla- tion graph</td></tr><tr><td>CovidGNN</td><td>0.0002901/0.005485/0.006149</td><td>32/1; 64/3; 64/3</td><td>0.146/8; 0.389/8; 0.186/8</td><td>10/7/10</td><td>causal windowed spatio-temporal graph construc- tion</td></tr><tr><td>MepoGNN</td><td>0.005707/0.001907/0.001042</td><td>16/2; 64/2; 16/2</td><td>0.0691/8; 0.338/4; 0.0926/8</td><td>7/3/5</td><td>absolute/anchored/absolute physical readout</td></tr><tr><td>M- Graphormer</td><td>0.002/0.0007216/0.001</td><td>16/2; 32/3; 16/2</td><td>0.20/8; 0.0162/4; 0.05/4</td><td>10/3/7</td><td>attention heads 4/4/2; Graphormer blocks 2/1/3</td></tr><tr><td>EpiGCN</td><td>0.004/0.002/0.004</td><td>16/2; 32/3; 32/2</td><td>0.10/8; 0.05/4; 0.10/8</td><td>10/3/7</td><td>ODE substeps 2/4/2; anchored node-calibrated readout</td></tr><tr><td>STAN</td><td>0.003/0.003/0.001</td><td>16/2; 32/2; 32/2</td><td>0.05/8; 0.05/8; 0.10/4</td><td>10/7/7</td><td>GAT heads 1/2/2; GAT hidden 16/32/32</td></tr><tr><td>EINN</td><td>0.003290/0.002/0.002</td><td>16/3; 32/2; 32/2</td><td>0.152/4; 0.05/4; 0.05/4</td><td>14/3/7</td><td>Fourier features 20/12/12; ODE substeps 1/2/2</td></tr><tr><td>Epi-Cola- GNN</td><td>0.006/0.003/0.006</td><td>16/1; 32/1; 32/1</td><td>0/8; 0.05/8; 0/8</td><td>14/3/10</td><td>temporal channels 8/16/8</td></tr><tr><td>EARTH</td><td>0.004/0.003/0.001</td><td>64/1; 32/1; 32/1</td><td>0/8; 0/8; 0.10/4</td><td>14/3/7</td><td>epidemic ODE substeps 2/1/1</td></tr></table>

## S2 Supplementary Results

## S2.1 Dynamical Fidelity: Complete Operator Trajectories

This section provides the complete reference and inferred trajectories of the instantaneous Lagrangian operator $\mathcal { F } ^ { L } ( t )$ and Eulerian operator $\mathcal { F } ^ { E } ( t )$ across the synthetic generation mechanisms. Blue solid lines denote the reference generating operators, and red dashed lines denote the inferred operators. The trajectories are organized as $4 \times 4$ grids of directed-edge interactions. In the Lagrangian and Eulerian scenarios, each subfigure shows one grid for a specified mechanism; in the multiscale scenarios, each figure contains four grids corresponding to Regions 1–4 for $\mathcal { F } ^ { L } ( t )$ or Age groups 1–4 for $\mathcal { F } ^ { E } ( t )$ . The $\mathcal { F } ^ { L } ( t )$ grids represent directed infection-generation interactions among age groups within a region, whereas the $\mathcal { F } ^ { E } ( t )$ grids represent directed population-transport interactions among regions for a given age group.

Overall, the inferred operators reproduce the principal temporal patterns and relative edge structure of the reference dynamics. Agreement is generally stronger for $\mathcal { F } ^ { E } ( t )$ , for which the major state-dependent and time-modulated transport trends are closely tracked. The inferred $\mathcal { F } ^ { L } ( t )$ trajectories also capture the characteristic evolution of the diferent infection-generation mechanisms, but larger discrepancies in magnitude or turning points are more apparent during the early stage and in rapidly varying segments. These deviations may arise because early epidemic observations provide weaker constraints on instantaneous infection-generation strengths, while compensation between latent initial states and interaction strengths, smooth neural parameterizations, and Lagrangian–Eulerian coupling in the multiscale systems may also contribute.

![](images/9f86fb7869913acc3a6f4f2d9542730763ccbd2965766df41ffe57a52c39712f.jpg)  
(b) TDCM.  
Supplementary Fig. 3: Complete ground-truth and inferred $\mathcal { F } ^ { L } ( t )$ trajectories for BMCM and TDCM in the Lagrangian scenario. Blue solid lines denote the reference generating operators, and red dashed lines denote the inferred operators.

![](images/083ed3066f404ed428d9ed55bf1a416f493d030cc37351eac6f31a9e1d833e65.jpg)  
(a) IACM.

![](images/87d552ee5007959b14e22d2395a974d341807668d610d57cd0b3af6c2daef5dc.jpg)  
(b) NMCM.  
Supplementary Fig. 4: Complete ground-truth and inferred $\mathcal { F } ^ { L } ( t )$ trajectories for IACM and NMCM in the Lagrangian scenario. Blue solid lines denote the reference generating operators, and red dashed lines denote the inferred operators.

![](images/c5e779d270c56bd16f3c3ac4147610f5778b318cb8a19ded20bd6c8aa04d1a3a.jpg)  
(b) CRGM.  
Supplementary Fig. 5: Complete ground-truth and inferred $\mathcal { F } ^ { E } ( t )$ trajectories for BRGM and CRGM in the Eulerian scenario. Blue solid lines denote the reference generating operators, and red dashed lines denote the inferred operators.

## S2.1.2 Eulerian Scenario

## S2.1.3 Multiscale Scenario

For each coupled mechanism, the $\mathcal { F } ^ { L } ( t )$ and $\mathcal { F } ^ { E } ( t )$ trajectories are presented as separate four-panel figures. For $\mathcal { F } ^ { L } ( t )$ , panels $\mathrm { ( a ) - ( d ) }$ correspond to Regions 1–4; for $\mathcal { F } ^ { E } ( t )$ , panels (a)–(d) correspond to $\mathrm { A g e }$ groups 1–4. Each operator figure is arranged in a $2 \times 2$ landscape layout so that all four panels and the figure legend appear on a single page.

![](images/2fa71bbca699f3d36d614ca9afb42dc77f24675a07420d34c17d29b370fe46fa.jpg)  
<sub>Complete</sub> <sub>ground-truth</sub> <sub>and</sub> <sub>inferredFL(t)trajectories</sub> <sub>for</sub> <sub>BMC</sub>M<sup>–BRGM</sup> <sup>in</sup> <sup>the</sup> <sup>multiscale</sup> <sub>ence</sub> <sub>generating</sub> <sub>operators,</sub> <sub>and</sub> <sub>re</sub>d <sup>dashed</sup> <sup>lines</sup> <sup>denote</sup> <sup>the</sup> <sup>inf</sup>

![](images/ce0b397958b13b6cb0ccba08629162d76666971e394aff4d5749289d314894db.jpg)

![](images/ac64b4ca10d9a0fbda9a094b9822d072ad488df6e2f573f5d08fd014f2d1a3e7.jpg)  
<sub>Complete</sub> <sub>ground-truth</sub> <sub>and</sub> <sub>inferredFL(t)trajectories</sub> <sub>for</sub> <sub>BMC</sub>M<sup>–CRGM</sup> <sup>in</sup> <sup>the</sup> <sup>multiscale</sup> <sub>ence</sub> <sub>generating</sub> <sub>operators,</sub> <sub>and</sub> <sub>re</sub>d <sup>dashed</sup> <sup>lines</sup> <sup>denote</sup> <sup>the</sup> <sup>inf</sup>

![](images/ddff7e8104eb516c79ace22742799065f1036a7d3f4aa0db02f05feededb69ed.jpg)  
<sub>Complete</sub> <sub>ground-truth</sub> <sub>and</sub> <sub>inferredFE(t)trajectories</sub> <sub>for</sub> <sub>BMC</sub>M<sup>–CRGM</sup> <sup>in</sup> <sup>the</sup> <sup>multis</sup> <sub>eference</sub> <sub>generating</sub> <sub>operators,</sub> <sub>and</sub> <sub>re</sub>d <sup>dashed</sup> <sup>lines</sup> <sup>denote</sup> <sup>the</sup> <sup>in</sup>

![](images/0407f65a8c598dba234398d1815634f548bc42af2996c213b57058925947efa2.jpg)

![](images/4fbe8a68745d96d79c45263b3081b4406d19a02252df364ecc69d4180b2ff9d9.jpg)

![](images/d526b0f10bc6db7466dde16fc18087259bea47eff701ae241c5d0c3ec395e395.jpg)  
<sub>:Complete</sub> <sub>ground-truth</sub> <sub>and</sub> <sub>inferredFL(t)trajectories</sub> <sub>for</sub> <sub>TDCM–C</sub><sup>RGM</sup> <sup>in</sup> <sup>the</sup> <sup>multis</sup> <sub>reference</sub> <sub>generating</sub> <sub>operators,</sub> <sub>and</sub> <sub>re</sub>d <sup>dashed</sup> <sup>lines</sup> <sup>denote</sup> <sup>the</sup> <sup>in</sup>

![](images/ca59eb3f8b9711dd50c44b3281bc915d5bc02e44347f81645ce3efcec2ddb8dc.jpg)  
<sub>:Complete</sub> <sub>ground-truth</sub> <sub>and</sub> <sub>inferredFE(t)trajectories</sub> <sub>for</sub> <sub>TDCM–C</sub><sup>RGM</sup> <sup>in</sup> <sup>the</sup> <sup>multis</sup> <sub>reference</sub> <sub>generating</sub> <sub>operators,</sub> <sub>and</sub> <sub>re</sub>d <sup>dashed</sup> <sup>lines</sup> <sup>denote</sup> <sup>the</sup> <sup>in</sup>

![](images/a468cfc20ccc9f91e1b4a2f97edc6bd0bed1464f608cf62ae7df23f90a104865.jpg)  
<sub>Complete</sub> <sub>ground-truth</sub> <sub>and</sub> <sub>inferredFL(t)trajectories</sub> <sub>for</sub> <sub>IACM–B</sub><sup>RGM</sup> <sup>in</sup> <sup>the</sup> <sup>multiscale</sup> <sub>ence</sub> <sub>generating</sub> <sub>operators,</sub> <sub>and</sub> <sub>re</sub>d <sup>dashed</sup> <sup>lines</sup> <sup>denote</sup> <sup>the</sup> <sup>inf</sup>

![](images/89a86da9a12a259e61d51019d708d715de4eb7d7a4987cd26616aeb383f7ebb9.jpg)

![](images/31e1c703de432ba7dee7487c6cc915fc58996ef0192850f2867efa41f2dbe1be.jpg)  
<sub>Complete</sub> <sub>ground-truth</sub> <sub>and</sub> <sub>inferredFL(t)trajectories</sub> <sub>for</sub> <sub>IACM–C</sub><sup>RGM</sup> <sup>in</sup> <sup>the</sup> <sup>multiscale</sup> <sub>ence</sub> <sub>generating</sub> <sub>operators,</sub> <sub>and</sub> <sub>re</sub>d <sup>dashed</sup> <sup>lines</sup> <sup>denote</sup> <sup>the</sup> <sup>inf</sup>

![](images/dbae8d93f11f13f78fdff9f22f39b5175241c22f9c69ad740a15db095706c514.jpg)  
<sub>Complete</sub> <sub>ground-truth</sub> <sub>and</sub> <sub>inferredFE(t)trajectories</sub> <sub>for</sub> <sub>IACM–C</sub><sup>RGM</sup> <sup>in</sup> <sup>the</sup> <sup>multiscale</sup> <sub>ence</sub> <sub>generating</sub> <sub>operators,</sub> <sub>and</sub> <sub>re</sub>d <sup>dashed</sup> <sup>lines</sup> <sup>denote</sup> <sup>the</sup> <sup>inf</sup>

![](images/7a8e985b9d028798cf54cf92d854f0e14d2f48a706f2dea4b8285f8df9487a43.jpg)

![](images/0a2bfabecb318a9b90a33c62493ce98c363b08c0f0f0acad80571ad7889cb619.jpg)

![](images/c4be7d10b388b58cbc0e3e3bc36137867fe1df64bda4f50ee83af1e1e1d1f234.jpg)

![](images/fae62919e62451bc493c7ee333816c80a455b921cfc3b3d9aadf13d124630c26.jpg)  
<sub>Complete</sub> <sub>ground-truth</sub> <sub>and</sub> <sub>inferredFE(t)trajectories</sub> <sub>for</sub> <sub>NMC</sub>M<sup>–CRGM</sup> <sup>in</sup> <sup>the</sup> <sup>multis</sup> <sub>reference</sub> <sub>generating</sub> <sub>operators,</sub> <sub>and</sub> <sub>re</sub>d <sup>dashed</sup> <sup>lines</sup> <sup>denote</sup> <sup>the</sup> <sup>in</sup>