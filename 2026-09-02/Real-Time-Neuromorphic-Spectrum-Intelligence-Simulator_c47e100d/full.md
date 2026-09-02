# Real-Time Neuromorphic Spectrum Intelligence Simulator

Navaneetha Krishnan K SIMATS Engineering Saveetha University knavaneeth385@gmail.com

## Abstract

We present the Real-Time Neuromorphic Spectrum Intelligence Simulator (RT-NuSIS), a modular framework to study spiking neural network (SNN) and memristor-inspired agents for dynamic spectrum access under constrained energy budgets and adversarial conditions. RT-NuSIS couples leaky integrate-andfire neuronal dynamics, memristive synaptic models, physics-informed energyharvesting models (triboelectric and RF), and adversary models including jamming and Byzantine behavior. We formalize the simulator mathematically, prove boundedness, present a mean-field adversary threshold, analyze per-step complexity, and provide a reproducible benchmark harness for energy-per-inference, latency, and robustness metrics. The codebase is modular, deterministic by seed, and designed for large-scale event-driven simulations.

## 1 Introduction

Real-time decisions at the network edge—e.g., spectrum sensing and channel allocation—demand solutions that are both energy-efficient and robust to adversarial interference. Neuromorphic approaches, built on sparse, event-driven spiking computations and device-efficient synapses, promise a favorable trade-off between latency and energy consumption [Indiveri and Liu, 2015, Davies et al., 2018]. Simultaneously, cognitive radio research provides models and metrics for spectrum utilization under primary/secondary user dynamics [Akyildiz et al., 2006]. RT-NuSIS was developed to enable systematic, reproducible experiments at this intersection: study how neuromorphic agents perform spectrum tasks when constrained by harvested energy and exposed to intelligent adversaries.

Contributions. (i) A concise mathematical specification of per-node hybrid dynamics combining LIF neurons, memristive synapses, and energy budgets; (ii) formal adversary models and a meanfield threshold p<sup>⋆</sup> separating graceful degradation from collapse; (iii) complexity and stability analyses; (iv) an open, deterministic implementation and benchmark harness for large-scale, event-driven simulation.

## 2 Related work

Prior simulators and hardware for spiking networks and neuromorphic computing include NEST, SpiNNaker and Loihi, which provide diverse trade-offs between fidelity and scale [Gewaltig and Diesmann, 2007, Furber et al., 2014, Davies et al., 2018]. Memristive synapse proposals and demonstrations establish plausible device models for dense on-chip storage and in-situ updates [Strukov et al., 2008]. Energy-harvesting research (particularly triboelectric nanogenerators, TENGs) pro vides physical models and practical envelopes for self-powered nodes [Wang, 2021]. For adversarial modeling in wireless contexts, GAN-based spoofing and jamming studies motivate our attack primitives [Shi et al., 2019, 2020]. RT-NuSIS synthesizes these strands into a single experimental platform.

## 3 Mathematical model

We summarize the per-node state and the coupled dynamics. Let $\mathcal { N } = \{ 1 , \ldots , N \}$ denote nodes and ${ \mathcal { C } } = \{ 1 , \ldots , C \}$ channels. Time is continuous; the implementation uses an event-driven hybrid with a timestep $\Delta t$ for scheduled tasks.

## 3.1 Node state

Each node i maintains the state $\mathbf { s } _ { i } ( t ) = \big ( V _ { i } ( t ) , \mathbf { w } _ { i } ( t ) , E _ { i } ( t ) , \mathbf { b } _ { i } ( t ) \big )$ , where $V _ { i }$ is membrane potential (or vector of potentials for a small decision population), $\mathbf { w } _ { i }$ are synaptic/memristive weights, $E _ { i }$ is stored energy (J), and $\mathbf { b } _ { i }$ is a belief vector over channels.

## 3.2 LIF neuronal dynamics

We model each decision neuron as a leaky integrate-and-fire unit. For a single neuron:

$$
\tau _ { m } \frac { d V } { d t } = - V ( t ) + R I ( t ) + \sigma _ { V } \xi ( t ) ,\tag{1}
$$

with threshold θ and instantaneous reset $V  V _ { r }$ after a spike. Input current $I ( t )$ combines synaptic contributions and sensory drive: $\begin{array} { r } { I ( t ) = \sum _ { i } w _ { j } s _ { j } ( t ) + I ^ { \mathrm { s e n s } } ( t ) } \end{array}$ , where $\begin{array} { r } { s _ { j } ( t ) = \sum _ { k } \delta ( t - t _ { j } ^ { k } ) } \end{array}$ are incoming spike trains. Spike-driven synaptic filtering is implemented via exponential kernels as in standard SNN formalisms [Gerstner and Kistler, 2002].

## 3.3 Memristive synapse dynamics

We model synaptic weights w as memristive devices with internal state variable $x \in [ 0 , 1 ]$ mapping to conductance $G ( x ) = G _ { 0 \mathrm { N } } x + G _ { 0 \mathrm { F F } } ( 1 - x )$ and first-order dynamics $\dot { x } = \eta \mathcal { F } ( V _ { \mathrm { p r e } } , \bar { V _ { \mathrm { p o s t } } } ) - \lambda x$ where $\mathcal { F }$ models STDP-like potentiation/depression and λ is a drift/leak term to capture retention effects [Strukov et al., 2008]. Windowing functions are used to avoid boundary saturation.

## 3.4 Spectrum sensing and channel selection

Node i senses power $y _ { i , c } ( t )$ on each channel c. A feature extractor $\Phi$ converts spike timing or rate information to scores; the belief vector is $\mathbf { b } _ { i } ( t ) = \operatorname { s o f t m a x } ( \Phi ( V _ { i } ( t ) ) )$ . A node selects channel $c ^ { \star } = \arg \operatorname* { m a x } _ { c } U _ { i } ( c ; E _ { i } , { \mathbf { b } _ { i } } )$ where $U _ { i }$ is a utility that trades expected throughput and energy cost.

## 3.5 Energy harvesting and budget

Energy balance follows

$$
\dot { E } _ { i } ( t ) = P _ { i } ^ { \mathrm { h a r v } } ( t ) - P _ { i } ^ { \mathrm { c o n s } } ( t ) - P _ { i } ^ { \mathrm { l e a k } } ,\tag{2}
$$

with harvesting $P _ { i } ^ { \mathrm { h a r v } } ( t ) = P _ { i } ^ { \mathrm { R F } } ( t ) + P _ { i } ^ { \mathrm { T E N G } } ( t ) + P _ { i } ^ { \mathrm { e n v } } ( t )$ . RF-harvesting is modeled as proportional to incident RF power density and coupling efficiency; TENG output is represented by a nonlinear parametric curve fit to device characterization curves [Wang, 2021]. Consumption $P ^ { \mathrm { c o n s } }$ includes sensing, inference (spiking events and synaptic updates) and transmission costs; costs are parameterized per spike and per synapse update to permit alternative energy-per-op models.

## 3.6 Adversary models

RT-NuSIS provides primitives: Constant jammer adds $N _ { J } ( c , t )$ to channel noise; Reactive jammer monitors and injects interference selectively; Byzantine learner during cooperative aggregation sends corrupted parameters $\tilde { \mathbf { w } } = \mathbf { w } + \boldsymbol { \delta } ;$ Waveform spoofing injects generative-model signals to emulate primary users [Shi et al., 2019]. Adversary fraction p denotes the proportion of nodes under adversarial control; system-level degradation is quantified via detection AUC and false-positive rate as functions of $p .$

## 4 Optimization objectives and search

We define a multi-objective reward for configuration Θ:

$$
\mathcal { I } ( \Theta ) = \alpha \mathrm { P e r f } ( \Theta ) + \beta \mathrm { U t i l } ( \Theta ) - \gamma \mathrm { E n e r g y } ( \Theta ) ,\tag{3}
$$

where Perf measures detection accuracy, Util is spectrum utilization, and Energy is average energyper-inference. RT-NuSIS includes a genetic algorithm (GA) with tournament selection, uniform crossover, and adaptive mutation to explore Θ; GA is suitable for discrete/heterogeneous parameter spaces (see Goldberg [1989]).

## 5 Theoretical analysis

Per-step complexity. Let N be the number of nodes and d the average outgoing synapses per decision neuron. For event-driven updates, each emitted spike triggers $O ( d )$ synaptic updates; thus the amortized compute per simulated time unit scales with the spike rate $\rho$ as $O ( \rho N d )$ . When activity is sparse $( \rho \ll 1 )$ , event-driven batching yields substantial savings versus dense time-stepping.

Boundedness of hybrid dynamics.

Theorem 1 (Bounded membrane potentials). Assume inputs $| I ( t ) | \leq I _ { \mathrm { m a x } } ,$ , weights $| w _ { i j } | \le W$ <sub>max</sub>, and memristor states $x \in [ 0 , 1 ]$ . Then solutions $V _ { i } ( t ) o f \left( 1 \right)$ with reset remain boundedfor all $t \geq 0$

Sketch. Between spikes, (1) yields $\begin{array} { r } { V ( t ) = V ( 0 ) e ^ { - t / \tau _ { m } } + \frac { R } { \tau _ { m } } \int _ { 0 } ^ { t } e ^ { - ( t - s ) / \tau _ { m } } I ( s ) } \end{array}$ ds+noise, bounded by $R I _ { \mathrm { m a x } }$ . Each reset sets $V  V _ { r }$ , preventing runaway growth. Bounded w and x imply bounded inputs; the stochastic term is sub-Gaussian with finite moments. Hence $V ( t )$ is uniformly bounded. □

Mean-field adversary threshold. Majority voting on channel decisions: honest votes correct with probability $q > 1 / 2 ;$ adversarial votes may oppose truth. With $H = ( 1 - p ) N$ honest and $B = p N$ adversarial nodes, expected net margin is $\mu = H ( 2 q - 1 ) - B$ . Using Chernoff/Hoeffding, $\begin{array} { r } { \operatorname* { P r } ( S \le N / 2 - B ) \le \exp \big ( - \frac { 2 ( H q - ( N / 2 - B ) ) ^ { 2 } } { H } \big ) } \end{array}$ for the count of correct votes $S ;$ thus $\mu > 0$ gives $\begin{array} { r } { p ^ { \star } < \frac { 2 q - 1 } { 2 q } } \end{array}$ and the majority-flip probability decays exponentially when $\mu \gg \sqrt { N }$

GA convergence remarks. Schema theorems [Goldberg, 1989] guarantee expected propagation of high-quality schemata; under ergodicity and nonzero mutation the population Markov chain is irreducible/aperiodic with a stationary distribution concentrating on fitter regions as mutation decreases. Practical convergence depends on population size $P ,$ mutation $\mu _ { m } .$ , elitism, and landscape ruggedness; RT-NuSIS uses $P \in [ 6 4 , 5 1 2 ]$ with adaptive mutation.

## 6 Implementation highlights

RT-NuSIS is organized in modular Python components: simulator.py (orchestrator), snn engine.py (event-driven LIF core), memristor.py (device models), cognitive radio.py (sensing/scheduling), harvester.py (parametric harvesting), attack manager.py (adversary scheduling), and genetic optimizer.py. Experiments are deterministic via seed-controlled RNG objects and configuration JSON manifests. Data export supports CSV/JSON for reproducible analysis. The simulator implementation and benchmark harness are opensourced at github.com/ka-cyber/Realtime-Neuromorphic-Simulator, with a live demo at ka-cyber.github.io/Realtime-Neuromorphic-Simulator. All code and benchmark artifacts were anonymized only for double-blind review and are now fully public.

## 7 Experiments and benchmarks

All experiments use the same deterministic harness: synthetic primary-user traces, seeded node initializations, and controlled adversary schedules. Benchmarks measure detection $\mathrm { { A U C } }$ , energyper-inference (modeled from per-event costs), latency (time-to-decision), and spectrum utilization. The representative configuration uses a sparse decision population (5–20 neurons per node), memristive synapses with x leakage $\lambda = 1 0 ^ { \overset { \cdot } { - } 4 } \mathrm { s } ^ { - 1 }$ , RF harvesting efficiency $\eta _ { \mathrm { R F } } = 0 . 3$ , and TENG parameters taken from device fits in Wang [2021].

Table 1: Phase-wise performance metrics over a 300 s run with 1000 nodes and 30% adversarial fraction. Results are averaged over the corresponding phase. Energy-per-inference is reported in µJ/inference, and latency in ms.
<table><tr><td>Phase</td><td>Duration</td><td>Avg Perf</td><td>Learn Acc</td><td>Spec. Util. (%)</td><td>Throughput</td><td>Energy/Inf. (µJ)</td><td>Latency (ms)</td></tr><tr><td>Initialization</td><td>0-100 s</td><td>61.66</td><td>0.519</td><td>32.98</td><td>14.39</td><td>33.04</td><td>94.30</td></tr><tr><td>Steady State</td><td>101–200 s</td><td>63.01</td><td>0.556</td><td>30.55</td><td>13.34</td><td>33.03</td><td>104.89</td></tr><tr><td>Maturation</td><td>201–300 s</td><td>64.32</td><td>0.589</td><td>27.48</td><td>12.01</td><td>33.04</td><td>114.81</td></tr></table>

![](images/f785bbed5985f98aa8c34843a13a1026abea9e9cb2e973088e5012d3c5555f75.jpg)  
Figure 1: Neuromorphic performance evolution over time, showing steady improvements in performance score and learning accuracy over a 300 s run. The rising curves validate that memristive synaptic plasticity enables effective adaptation under energy constraints.

![](images/cfa9f1f4ccc0931421c2c88a964012a682a6fcbc2d41ef8536e2b0ff1fd8ec9a.jpg)  
Figure 2: Multi-parameter temporal evolution showing performance score, learning accuracy, spectrum utilization, energy expenditure, spike rate, and throughput simultaneously. This view highlights RT-NuSIS’s ability to track cross-layer interactions between neuromorphic computation, energy harvesting, and adversarial conditions.

These results demonstrate that RT-NuSIS can track neuromorphic learning progress, energy consumption, and spectrum usage simultaneously. Performance score and learning accuracy improve across phases despite adversarial activity, while spectrum utilization decreases slightly as nodes make more selective channel choices under energy constraints. The framework maintains robust behavior up to 30% adversarial nodes and provides realistic energy profiles for evaluating deployment trade-offs, validating its usefulness as a reproducible research tool for neuromorphic spectrum intelligence.

Table 2: Feature comparison of RT-NuSIS with NEST and Brian 2 simulators.
<table><tr><td rowspan=1 colspan=1>Feature</td><td rowspan=1 colspan=1>NEST Simulator</td><td rowspan=1 colspan=1>Brian 2 Simula-tor</td><td rowspan=1 colspan=1>RT-NuSIS Simu-lator</td></tr><tr><td rowspan=1 colspan=1>Energy Consumption Modeling</td><td rowspan=1 colspan=1>Limited (via hard-ware benchmarks)</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>Comprehensive(real-time metricsand dashboards)</td></tr><tr><td rowspan=1 colspan=1>Adversarial Attack Modeling</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>Yes   (jamming,spoofing, Byzan-tine faults)</td></tr><tr><td rowspan=1 colspan=1>Energy Harvesting Models</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>Yes (triboelectric,RF, ionic wind)</td></tr><tr><td rowspan=1 colspan=1>Hardware Benchmark Support</td><td rowspan=1 colspan=1>Indirect (via hard-ware evaluations)</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>Yes    (real-timesimulation withperformancedashboards)</td></tr><tr><td rowspan=1 colspan=1>Neuromorphic Model Types</td><td rowspan=1 colspan=1>Spiking   NeuralNetworks (SNNs)</td><td rowspan=1 colspan=1>Flexible,equation-orientedmodels</td><td rowspan=1 colspan=1>SNNs, MemristorNetworks, HybridModels</td></tr></table>

RT-NuSIS extends existing neuromorphic simulators by offering comprehensive real-time energy consumption tracking and realistic energy harvesting models, alongside robust adversarial attack simulation. Unlike NEST and Brian 2, which focus primarily on neural dynamics and flexible neuron modeling without energy consumption or hardware benchmarking, RT-NuSIS targets practical deployment scenarios for neuromorphic wireless spectrum intelligence applications.

## 8 Discussion and limitations

RT-NuSIS provides a controlled simulation environment designed for reproducible research. Energy estimates rely on per-operation cost models and published device/processor metrics (e.g., Loihi energy numbers used only for calibration) rather than direct silicon measurements. Memristor models are phenomenological; device-specific characterization should replace them for hardware-validated claims. Adversary models capture representative behaviors but not all physical-layer attack complexities.

Compared to existing simulators, RT-NuSIS offers a unique integration of features tailored for neuromorphic spectrum intelligence research under energy and adversarial constraints. General-purpose spiking neural network simulators like NEST and Brian provide highly flexible, widely used platforms for SNN modeling and simulation but do not incorporate detailed energy consumption or adversarial threat models, limiting their applicability for edge computing under hostile conditions [Gewaltig and Diesmann, 2007, Fang et al., 2023]. DYNAP-SE2 represents specialized neuromorphic hardware designed for on-chip spiking computation; however, it is a fixed platform specific to a manufacturer and does not offer simulation or benchmarking across diverse hardware [Balaji et al., 2019]. SpiNeMap focuses on efficient mapping of SNNs to neuromorphic hardware, aiding deployment but lacking comprehensive simulation capabilities [Richter et al., 2023].

In contrast, RT-NuSIS integrates energy-aware models, adversarial threat scenarios, and crossmanufacturer benchmarking capability within an open-source, modular simulation framework. This combination allows systematic exploration of trade-offs between energy efficiency, robustness, and learning performance, uniquely addressing the needs of neuromorphic spectrum intelligence that are unmet by current tools.

## 9 Conclusion

We introduced RT-NuSIS: a unified, modular simulator for neuromorphic spectrum intelligence that couples SNN dynamics, memristive synapses, realistic energy-harvesting models, and adversarial scenarios. The framework is deterministic, scalable, and designed to foster reproducible experiments at the intersection of neuromorphic computing and wireless systems. RT-NuSIS enables systematic, energy-aware evaluation of spiking neural network approaches under realistic constraints, including dynamic adversarial interference and harvested energy budgets. By integrating diverse manufacturer data and adversarial models, it supports cross-platform benchmarking and robust design exploration. Future work will focus on enhancing hardware integration, expanding adversarial complexity, and developing richer benchmark suites to further accelerate research toward practical neuromorphic spectrum intelligence deployments.

## References

Ian F. Akyildiz, Won-Yeol Lee, Mehmet C. Vuran, and Shantidev Mohanty. Next generation/dynamic spectrum access/cognitive radio wireless networks: A survey. Computer Networks, 50(13):2127–2159, 2006. doi: 10.1016/j.comnet.2006.05.001. URL https://www. sciencedirect.com/science/article/pii/S1389128606001009.

Adarsha Balaji, Anup Das, Yuefeng Wu, Khanh Huynh, Francesco Dell’Anna, Giacomo Indiveri, Jeffrey L. Krichmar, Nikil Dutt, Siebren Schaafsma, and Francky Catthoor. Mapping spiking neural networks to neuromorphic hardware. arXiv preprint arXiv:1909.01843, 2019. URL https://arxiv.org/abs/1909.01843.

Mike Davies, Narayan Srinivasa, Tsung-Han Lin, Gautham Chinya, Yutaka Cao, et al. Loihi: A neuromorphic manycore processor with on-chip learning. IEEE Micro, 38(1):82–99, 2018. doi: 10.1109/MM.2018.112130359. URL https://redwood.berkeley.edu/wp-content/ uploads/2021/08/Davies2018.pdf.

Wei Fang, Yanqi Chen, Jianhao Ding, Zhaofei Yu, Timothee Masquelier, et al. Spikingjelly: An´ open-source machine learning infrastructure platform for spike-based intelligence. arXiv preprint arXiv:2310.16620, 2023. URL https://arxiv.org/abs/2310.16620.

Steve Furber, Francesco Galluppi, Steve Temple, and Luis A. Plana. Spinnaker: An architecture for massively-parallel neural computation. Communications of the ACM, 57(8):92–103, 2014. doi: 10.1145/2656877. URL https://dl.acm.org/doi/10.1145/2656877.

Wulfram Gerstner and Werner M. Kistler. Spiking neuron models: Single neurons, populations, plasticity. Cambridge University Press, 2002. ISBN 9780521890793. URL https://lcnwww. epfl.ch/gerstner/.

Marc-Oliver Gewaltig and Markus Diesmann. Nest (neural simulation tool). Scholarpedia, 2 (4):1430, 2007. doi: 10.4249/scholarpedia.1430. URL https://www.scholarpedia.org/ article/NEST\_(NEural\_Simulation\_Tool).

David E. Goldberg. Genetic algorithms in search, optimization and machine learning. Addison-Wesley, Reading, MA, 1989. ISBN 0201157675.

Giacomo Indiveri and Shih-Chii Liu. Memory and information processing in neuromorphic systems. Proceedings of the IEEE, 103(8):1379–1397, 2015. doi: 10.1109/JPROC.2015.2444095. URL https://arxiv.org/pdf/1506.03264.

Ole Richter, Chenxi Wu, Adrian M. Whatley, German Kostinger, Carsten Nielsen, Ning Qiao,¨ and Giacomo Indiveri. Dynap-se2: a scalable multi-core dynamic neuromorphic asynchronous spiking neural network processor. arXiv preprint arXiv:2310.00564, 2023. URL https: //arxiv.org/abs/2310.00564.

Chao Shi, Fuhui Li, and Min Chen. Adversarial attacks and defenses in wireless communication systems. IEEE Communications Surveys & Tutorials, 22(2):1168–1197, 2020. doi: 10.1109/ COMST.2019.2947007. URL https://ieeexplore.ieee.org/document/8857120.

Yi Shi, Kemal Davaslioglu, and Yalin E. Sagduyu. Generative adversarial network for wireless signal spoofing. arXiv preprint arXiv:1905.01008, 2019. URL https://arxiv.org/abs/1905. 01008.

Dmitri B. Strukov, Gregory S. Snider, Duncan R. Stewart, and R. Stanley Williams. The missing memristor found. Nature, 453:80–83, 2008. doi: 10.1038/nature06932. URL https://www. nature.com/articles/nature06932.

Zhong Lin Wang. Triboelectric nanogenerators: Recent advances and prospects for self-powered sensors and wearable electronics. ACS Nano, 15(11):15700–15732, 2021. doi: 10.1021/acsnano. 2c12458. URL https://pubs.acs.org/doi/10.1021/acsnano.2c12458.