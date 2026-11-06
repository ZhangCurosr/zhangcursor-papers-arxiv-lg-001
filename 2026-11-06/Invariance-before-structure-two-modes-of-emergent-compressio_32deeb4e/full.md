# Invariance before structure: two modes of emergent compression in representation learning

I-X Centre for AI in Science Department of Electrical and Electronics Engineering Imperial College London London, UK, SW7 2AZ h.rajpal15@imperial.ac.uk

Dan Goodman Department of Electrical and Electronics Engineering Imperial College London London, UK, SW7 2AZ d.goodman@imperial.ac.uk

## Abstract

Learned representations show striking invariance across biological and artificial neural networks, with different substrates and variations driven by drift, heterogeneity and initialisation. To understand this invariance, we propose Emergent Adaptive Representation Learning (EARL), a framework that distinguishes two modes of emergence in learned representations: state-space compression, where a learned representation gains predictive power over its own future beyond what the network’s units can predict individually, and parameter-space compression, where networks with different micro-level parameters converge to the same representation. We formalise both as Fisher sensitivities and introduce ensemble-level informationtheoretic measures for each. In a recurrent reservoir with a trained bottleneck, parameter-space compression rises within the first few epochs of training and partly independently of the task, whereas state-space compression rises about 30 epochs later and only when the network begins to learn the temporal structure of the task. These results suggest that invariance to the substrate and task-specific emergent structure are acquired in distinct phases of learning.

## 1 Introduction

From local synaptic plasticity to global gradient-based optimisation, biological and artificial neural networks learn through very different mechanisms [1–3], yet both these overparameterised systems are able to learn similar representations at the level of coordinated population activity [4–6]. In both, the same representation can be realised by very different micro-level configurations. In the brain, population representations remain stable despite representational drift, synaptic turnover and homeostatic recruitment among the underlying neurons [7–10], and similar representations arise across circuits with heterogeneous parameters, within and across individuals and species [11, 12]. In artificial networks, models trained from different initialisations and architectures often converge to similar representations when performing a common task [13, 14]. Understanding how a learned representation becomes independent of its substrate is central to explaining why different brains and models represent the same stimuli similarly [15, 16].

We approach this question through emergence, the property of complex systems whose collective behaviour cannot be reduced to that of their components [17, 18], which recent information-theoretic measures make quantifiable [19–21]. Because emergence concerns information carried by the whole that is not present in any of its parts, it directly addresses the key property of learned representations implicated in substrate invariance. Such measures have characterised collective behaviour from bird flocks to ecosystems and organoids [20, 22, 23], neural oscillations and conscious states [24–26], and representations learned by deep networks [27–29]. In a classic reservoir computer, optimising the reservoir hyperparameters for prediction yields causally emergent representations [30]. In our previous work on a reservoir optimised for representation learning with a trained bottleneck, which we study here, the degree of emergence of the learned representation predicted generalisation to unseen rotations and dynamical systems [31]. Distilling insights from various examples of emergence, we argue that emergent representations can arise through two distinct forms of compression, of the network’s states and of its parameters. We introduce the Emergent Adaptive Representation Learning (EARL) framework, which formalises how both modes describe different aspects of generalisable and robust representation learning. We also propose empirically tractable information-theoretic measures for each. In a reservoir with a trained bottleneck, both measures show a transient, task-independent change during the initial fitting phase, followed by a robust rise later in training when the network learns the temporal structure of the task.

## 2 Emergent Adaptive Representation Learning (EARL) framework

Emergent properties typically provide a compact description of the underlying system [32, 33, 18]. We distinguish two kinds of compression that can give rise to them (Fig. 1).

## 2.1 State-space compression

In state-space compression, a coarse-grained description of the system’s states predicts its future, or the task-relevant variables, better than its microscopic components do [33, 20]. In physics, this corresponds to coarse-graining under the renormalisation group, which integrates out irrelevant degrees of freedom [34, 35]. In learning systems, it is related to the information bottleneck, which retains predictive information while discarding the rest of the input [36, 37] and has been linked to better in-task generalisation and robustness to input noise [38, 39].

## 2.2 Parameter-space compression

![](images/01a315e4fbc1ee49667a283550af8a37b01943c2d09caec1f3e753a9408e9ce4.jpg)

In parameter-space compression, macroscopic behaviour depends on only a few combinations of the microscopic parameters: many models in physics and biology are sensitive to a few stiff parameter combinations and insensitive to the rest [40]. Such overparameterised systems often exhibit universality, where many microscopic models share one macroscopic description [41]. In the brain, this corresponds to degeneracy, where disparate circuit parameters and continual synaptic turnover yield the same function [42, 11, 43]. In artificial networks, it underlies the convergence of representations across initialisations and model architectures [13, 14].

## 2.3 Formalism

Consider a neural network whose micro-state $X _ { t } \in \mathbb { R } ^ { N }$ , the activity of its N units, evolves under micro-level parameters θ, and a learned

Emergent Macro

Figure 1: The two modes of compression. A State space: a stable macro configuration emerges as microscopic fluctuations are compressed away. B Parameter space: the same configuration emerges from different models of microscopic interactions.

representation $V _ { t } = f _ { w } ( X _ { t } )$ with trainable weights w that encodes a task-relevant latent variable $L _ { t } .$ , such as the variables generating a simulated dataset or relevant behavioural variables recorded in an experiment. Here θ collects the parameters that define the substrate and are held fixed across realisations of the network, such as architectural hyperparameters or fixed weights. Both modes can be phrased as Fisher sensitivities of the representation (Appendix B); we instead use mutual-information analogues, which can be estimated across an ensemble from recorded activity without access to gradients [44].

## 3 Measuring state-space and parameter-space compression

## 3.1 State-space compression as causal emergence

A representation that compresses the state space should predict its own future better than its microvariables do individually. We therefore quantify state-space compression with the practical criterion for causal emergence of Rosas et al. [20], applied to the learned representation:

$$
\Psi = I ( V _ { t } ; V _ { t + 1 } ) - \sum _ { i = 1 } ^ { N } I ( X _ { t } ^ { i } ; V _ { t + 1 } )\tag{1}
$$

The first term is the self-predictability of the representation and the second sums the information each unit carries about the representation’s future. Following Sas et al. [45], we add a second-order correction, which adds back the redundancy shared by pairs of units and so minimally accounts for the information double-counted in the micro sum; this correction has previously been applied to neural recordings [24]. With many units, higher-order redundancies are not accounted for. All terms are Gaussian mutual-information estimates (in nats), computed per sequence and averaged over sequences and models. $\Psi > 0$ is sufficient but not necessary for emergence [20, 21]. Because the micro sum still over-counts information shared by many units, Ψ is typically negative in large networks, and we interpret relative changes: an increase in Ψ means the representation’s future is more predictable from the representation itself and less attributable to any single unit.

## 3.2 Parameter-space compression as degeneracy

Parameter-space compression asks the converse question: whether networks with different micro-level parameters $\dot { \theta }$ arrive at the same representation. Following the idea of parameter-space compression of Machta et al. [41], we train an ensemble across θ and measure how much information a task-relevant feature of the representation, $F ( V )$ , retains about $\theta ,$ relative to a matched feature of the micro-level activity, $F ( X )$

$$
\Lambda = 1 - \frac { I \big ( { \boldsymbol { \theta } } ; { \boldsymbol { F } } ( V ) \mid \nu \big ) } { I \big ( { \boldsymbol { \theta } } ; { \boldsymbol { F } } ( X ) \mid \nu \big ) }\tag{2}
$$

We take $F ( V ) = I ( V ; L )$ , the Gaussian mutual information between the learned and the true latent, which is invariant to invertible linear transformations of V and $L$ and defined for any task with a latent variable. The micro feature $F ( X )$ consists of summary statistics of fixed random projections of the micro-state; if the micro-level dynamics are not trained, the denominator is constant across training. Here $\nu$ collects nuisance variables shared across $\theta ,$ namely the input sequence and the initialisation of the trained weights; both features are centred across θ within each value of $\nu ,$ so Λ measures information about θ beyond what ν already determines. $\Lambda = 1$ means the representation carries no detectable information about $\theta ; \Lambda = 0$ means it carries as much as the micro-level activity does.

We estimate each $I ( \theta ; \cdot )$ with a held-out Barber–Agakov lower bound [46], treating the sampled settings of $\boldsymbol { \theta }$ as equiprobable classes and fitting and scoring on disjoint sequences. Every estimate is reported alongside a null computed with shuffled θ labels. The number of random projections and the choice of estimator shift the level of Λ but not its trend (Appendix C).

## 4 Model setup

We use the reservoir–bottleneck architecture from our previous work [31]: a fixed random recurrent reservoir of $N = 5 0 0$ leaky-tanh units, a trainable 10-dimensional linear bottleneck $Z _ { t } = W _ { z } X _ { t } + b _ { z }$ and a ridge-regression readout that predicts the next input using the bottleneck states (Appendix $\mathbf { A } )$ In this model, $X _ { t }$ are the $N = 5 0 { \bar { 0 } }$ reservoir states, $\bar { V } _ { t }$ the top three principal components of the bottleneck activity $Z _ { t }$ (fitted on the training sequences), $L _ { t }$ the true three-dimensional attractor state, and θ the reservoir hyperparameters. The micro feature $F ( X )$ is the time mean and log-variance of 24 fixed random projections of $X _ { t } ;$ because the reservoir is fixed, $I ( \theta ; F ( X ) )$ is constant over training. Inputs are random orthogonal 10-dimensional embeddings of three-dimensional attractors; the bottleneck is trained by backpropagation through the ridge readout to minimise one-step prediction error on 12 training sequences (Sprott B, Rössler, Lorenz), and evaluated on 12 sequences with unseen rotations of the same attractors and 12 sequences of held-out attractors (Chen, Sprott A). Each ensemble shares one reservoir draw and varies θ (spectral radius, input scaling, bias scaling, leak rate and ridge penalty) over 100 settings drawn log-uniformly within ±30% of the Optuna optimum [47], with sparsity fixed. Every setting is trained from the same four bottleneck initialisations. We report 20 replicate ensembles (5 reservoir draws × 4 data seeds), trained for 100 epochs on noise-free input (noisy input shown in Appendix E), and two matched controls with 10 replicates each, in which the training target is either time-permuted within its sequence or taken from another sequence.

![](images/887e0979d71a07995905f692fee0f20b477d53a84e089da71cf9289364a44400.jpg)  
Figure 2: Learning dynamics on noise-free input. A One-step prediction error. B State-space compression Ψ. C Parameter-space compression Λ. Solid: mean of 20 replicate ensembles (band: 10th–90th percentile) on training attractors, unseen rotations of the training attractors and held-out attractors; dashed/dotted: time-permuted and cross-sequence target controls (10 matched replicates each). Triangles: half-change epochs (from epoch 0 for Λ, from the minimum for Ψ).

## 5 Results

Training reduces the one-step prediction error monotonically on all three evaluation sets, while the learned representation becomes increasingly aligned with the true latent (Fig. 2A; Appendix D). State-space compression is non-monotonic (Fig. 2B): Ψ first decreases, reaching its minimum by about epoch 10, and then rises for the rest of training, reaching half of its rise by about epoch 40 on all three sets, consistent with our previous findings [31]. As expected for $N = 5 0 0$ units, Ψ is negative throughout.

Parameter-space compression instead rises from the start of training (Fig. 2C): Λ increases from about 0.78 to 0.96 on the training sequences $( \Delta \Lambda = 0 . 1 7 7 \pm 0 . 0 2 0$ , mean ± s.d.; similar on the other sets), and increases on every set (two-sided sign test across replicates, $p = 1 . 9 \times 1 0 ^ { - 6 } )$ . Λ reaches half of its total change by about epoch 8 and Ψ half of its rise by about epoch 40 (triangles in Fig. 2B,C), so parameter-space compression leads state-space compression by $\Delta T \approx 3 2$ epochs (difference of median half-change epochs), and Λ is never later than Ψ (sign test, $p \leq 1 . 5 \times \bar { 1 0 ^ { - 5 } } )$

Both controls reproduce the early phase but not the increase in later stages. With a time-permuted or cross-sequence target, Ψ still dips and Λ still rises to about 0.85, but Ψ does not recover and Λ plateaus (Fig. 2B,C, dashed and dotted lines). Over training, models trained on the true target gain more than their matched controls (paired sign test, $p = 0 . 0 0 2$ for each control and set): by about 115 nats in Ψ and 0.09–0.10 in Λ on the training sequences. Results with noisy input are qualitatively the same (Appendix E). Together, these results separate the two modes: both show a task-independent early phase, in which Ψ dips and Λ makes its first rise with or without the task, whereas the recovery of state-space compression and the continued gain in parameter-space compression appear only when the network learns the temporal structure of the task.

## 6 Discussion

We show that training may first make the representation invariant to the substrate, on which task learning then builds a self-predictive, distributed representation. Because Λ compares models through what their representations encode rather than by aligning the representations directly, it also offers an information-theoretic measure of representational universality across substrates.

Both measures are relative: Ψ is negative throughout and its trend depends on the temporal sampling of the data, and Λ requires a task-relevant latent and measures invariance across models rather than the robustness of a single model. The measures extend to other tasks by replacing $V _ { t + 1 }$ in Ψ and $F ( V )$ in Λ with task-appropriate quantities. We have previously observed the non-monotonic trajectory of Ψ in hippocampal and prefrontal recordings during learning [31]; measuring parameter-space compression across animals, and in datasets with representational drift [7, 8], is a natural next step.

## Acknowledgments and Disclosure of Funding

H.R. was funded by the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship, supported by Schmidt Sciences, LLC.

## References

[1] Timothy P Lillicrap, Adam Santoro, Luke Marris, Colin J Akerman, and Geoffrey Hinton. Backpropagation and the brain. Nature Reviews Neuroscience, 21(6):335–346, 2020.

[2] Jeffrey C Magee and Christine Grienberger. Synaptic plasticity forms and functions. Annual review of neuroscience, 43(1):95–117, 2020.

[3] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by backpropagating errors. nature, 323(6088):533–536, 1986.

[4] Shreya Saxena and John P Cunningham. Towards the neural population doctrine. Current opinion in neurobiology, 55:103–111, 2019.

[5] Saurabh Vyas, Matthew D Golub, David Sussillo, and Krishna V Shenoy. Computation through neural population dynamics. Annual review ofneuroscience, 43(1):249–275, 2020.

[6] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530, 2016.

[7] Laura N Driscoll, Noah L Pettit, Matthias Minderer, Selmaan N Chettih, and Christopher D Harvey. Dynamic reorganization of neuronal activity patterns in parietal cortex. Cell, 170(5):986–999, 2017.

[8] Daniel Deitch, Alon Rubin, and Yaniv Ziv. Representational drift in the mouse visual cortex. Current biology, 31(19):4327–4339, 2021.

[9] Michael E Rule, Timothy O’Leary, and Christopher D Harvey. Causes and consequences of representational drift. Current opinion in neurobiology, 58:141–147, 2019.

[10] Takahiro Noda, Eike Kienle, Jens-Bastian Eppler, Dominik F Aschauer, Matthias Kaschube, Yonatan Loewenstein, and Simon Rumpel. Homeostasis of a representational map in the neocortex. Nature Neuroscience, 28(7):1533–1545, 2025.

[11] Eve Marder and Jean-Marc Goaillard. Variability, compensation and homeostasis in neuron and network function. Nature Reviews Neuroscience, 7(7):563–574, 2006.

[12] Mostafa Safaie, Joanna C Chang, Junchol Park, Lee E Miller, Joshua T Dudman, Matthew G Perich, and Juan A Gallego. Preserved neural dynamics across animals performing similar behaviour. Nature, 623 (7988):765–771, 2023.

[13] Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John Hopcroft. Convergent learning: Do different neural networks learn the same representations? arXiv preprint arXiv:1511.07543, 2015.

[14] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pages 3519–3529. PMlR, 2019.

[15] Ilia Sucholutsky, Lukas Muttenthaler, Adrian Weller, Andi Peng, Andreea Bobu, Been Kim, Bradley C Love, Christopher J Cueva, Erin Grant, Iris Groen, et al. Getting aligned on representational alignment. arXiv preprint arXiv:2310.13018, 2023.

[16] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The platonic representation hypothesis. arXiv preprint arXiv:2405.07987, 2024.

[17] Philip W Anderson. More is different: Broken symmetry and the nature of the hierarchical structure of science. Science, 177(4047):393–396, 1972.

[18] David C Krakauer, Melanie Mitchell, and John W Krakauer. Large language models and emergence: A complex systems perspective. Philosophical Transactions ofthe Royal Society A: Mathematical, Physical and Engineering Sciences, 384(2320):20250014, 2026.

[19] Erik P Hoel, Larissa Albantakis, and Giulio Tononi. Quantifying causal emergence shows that macro can beat micro. Proceedings ofthe National Academy ofSciences, 110(49):19790–19795, 2013.

[20] Fernando E Rosas, Pedro AM Mediano, Henrik J Jensen, Anil K Seth, Adam B Barrett, Robin L Carhart Harris, and Daniel Bor. Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data. PLoS computational biology, 16(12):e1008289, 2020.

[21] Pedro AM Mediano, Fernando E Rosas, Andrea I Luppi, Henrik J Jensen, Anil K Seth, Adam B Barrett, Robin L Carhart-Harris, and Daniel Bor. Greater than the parts: a review of the information decomposition approach to causal emergence. Philosophical transactions. Series A, Mathematical, physical, and engineering sciences, 380(2227):20210246, 2022.

[22] Hardik Rajpal, Clem von Stengel, Pedro AM Mediano, Fernando E Rosas, Eduardo Viegas, Pablo A Marquet, and Henrik J Jensen. Information dynamics and the emergence of high-order individuality in ecosystems. Communications Biology, 8(1):1231, 2025.

[23] Thomas F Varley, Vaibhav P Pai, Caitlin Grasso, Jeantine Lunshof, Michael Levin, and Josh Bongard. Identification of brain-like complex information architectures in embryonic tissue of xenopus laevis organoids. Communicative & Integrative Biology, 18(1):2568307, 2025.

[24] Hardik Rajpal, Pedro AM Mediano, Madalina I Sas, Henrik J Jensen, and Fernando E Rosas. Quantifying the emergence of population-level activity in neuronal systems. bioRxiv, pages 2026–02, 2026.

[25] Andrea I Luppi, Pedro AM Mediano, Fernando E Rosas, Judith Allanson, John D Pickard, Guy B Williams, Michael M Craig, Paola Finoia, Alexander RD Peattie, Peter Coppola, et al. Reduced emergent character of neural dynamics in patients with a disrupted connectome. NeuroImage, 269:119926, 2023.

[26] Borjan Milinkovic, Anil K Seth, Lionel Barnett, Olivia Carter, and Thomas Andrillon. Emergent multiscal organisation of neural dynamics fragments in anaesthesia. Imaging Neuroscience, 4:IMAG–a, 2026.

[27] Christos Kaplanis, Pedro Mediano, and Fernando Rosas. Learning causally emergent representations. In NeurIPS 2023 workshop: Information-Theoretic Principles in Cognitive Systems, 2023.

[28] David McSharry, Christos Kaplanis, Fernando E Rosas, and Pedro A Mediano. Learning diverse causally emergent representations from time series data. Advances in Neural Information Processing Systems, 37: 119547–119572, 2024.

[29] Mingzhe Yang, Zhipeng Wang, Kaiwei Liu, Yingqi Rong, Bing Yuan, and Jiang Zhang. Finding emergence in data by maximizing effective information. National Science Review, 12(1):nwae279, 2025.

[30] Hanna M Tolle, Andrea I Luppi, Anil K Seth, and Pedro AM Mediano. Evolving reservoir computers reveal bidirectional coupling between predictive power and emergent dynamics. Patterns, 7(3), 2026.

[31] Hardik Rajpal and Dan Goodman. Emergent generalization by representation learning in artificial neural networks. arXiv preprint arXiv:2607.10430, 2026.

[32] Jessica C Flack. Coarse-graining as a downward causation mechanism. Philosophical transactions. Series A, Mathematical, physical, and engineering sciences, 375(2109):20160338, 2017.

[33] Cosma Rohilla Shalizi and James P Crutchfield. Computational mechanics: Pattern and prediction, structure and simplicity. Journal ofstatistical physics, 104(3):817–879, 2001.

[34] Kenneth G Wilson and John Kogut. The renormalization group and the ϵ expansion. Physics reports, 12 (2):75–199, 1974.

[35] Maciej Koch-Janusz and Zohar Ringel. Mutual information, neural networks and the renormalization group. Nature Physics, 14(6):578–582, 2018.

[36] Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000.

[37] Alessandro Achille and Stefano Soatto. Emergence of invariance and disentanglement in deep representations. Journal ofMachine Learning Research, 19(50):1–34, 2018.

[38] Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. Deep variational information bottleneck. arXiv preprint arXiv:1612.00410, 2016.

[39] Kenji Kawaguchi, Zhun Deng, Xu Ji, and Jiaoyang Huang. How does information bottleneck help deep learning? In International conference on machine learning, pages 16049–16096. PMLR, 2023.

[40] Ryan N Gutenkunst, Joshua J Waterfall, Fergal P Casey, Kevin S Brown, Christopher R Myers, and James P Sethna. Universally sloppy parameter sensitivities in systems biology models. PLoS computational biology, 3(10):e189, 2007.

[41] Benjamin B Machta, Ricky Chachra, Mark K Transtrum, and James P Sethna. Parameter space compression underlies emergent theories and predictive models. Science, 342(6158):604–607, 2013.

[42] Gerald M Edelman and Joseph A Gally. Degeneracy and complexity in biological systems. Proceedings of the national academy ofsciences, 98(24):13763–13768, 2001.

[43] Alessio Attardo, James E Fitzgerald, and Mark J Schnitzer. Impermanence of dendritic spines in live adult ca1 hippocampus. Nature, 523(7562):592–596, 2015.

[44] Nicolas Brunel and Jean-Pierre Nadal. Mutual information, fisher information, and population coding. Neural computation, 10(7):1731–1757, 1998.

[45] Madalina I Sas, Fernando E Rosas, Hardik Rajpal, Daniel Bor, Henrik J Jensen, and Pedro AM Mediano. Improved estimators of causal emergence for large systems. arXiv preprint arXiv:2601.00013, 2025.

[46] David Barber and Felix Agakov. The IM algorithm: A variational approach to information maximization. In Advances in Neural Information Processing Systems, volume 16, 2003.

[47] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2623–2631, 2019.

## A Model architecture

![](images/9fc8ebe100681b1989596ca1f5b5d7e8700f8e44af1d1ea48a2f0d15d6a0847d.jpg)  
Figure 3: Reservoir–bottleneck architecture, adapted from our previous work [31]. Random orthogo nal embeddings of the attractor latent $L _ { t }$ drive a fixed random reservoir with states $X _ { t } ;$ a trainable linear bottleneck maps $X _ { t }$ to $Z _ { t }$ , whose top three principal components form the learned representation $V _ { t } ;$ a ridge readout predicts the next input. The bottleneck is the only trained component.

## B Fisher-information formulation

We treat the micro-state as evolving under a transition kernel $p _ { \theta } ( X _ { t + 1 } \mid X _ { t } )$ , with external input treated as a source of stochasticity and marginalised, and read out the representation with additive noise, $V _ { t } = f _ { w } ( X _ { t } ) + \varepsilon _ { t } , \varepsilon _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ ; the noise makes the conditional densities below well defined.

State space. The sensitivity of the representation’s future to the current micro-state is

$$
F _ { \mathrm { s t a t e } } = \mathbb { E } \left[ \nabla _ { X _ { t } } \log p _ { \theta } ( V _ { t + 1 } \mid X _ { t } ) \nabla _ { X _ { t } } \log p _ { \theta } ( V _ { t + 1 } \mid X _ { t } ) ^ { \top } \right] = \sigma ^ { - 2 } \mathbb { E } \left[ J _ { t } ^ { \top } J _ { t } \right] , \qquad J _ { t } = \frac { \partial V _ { t + 1 } } { \partial X _ { t } } .\tag{3}
$$

This mirrors the micro term of Ψ. A constant representation trivially has $F _ { \mathrm { s t a t e } } = 0$ , so compression means that sensitivity concentrates on the few directions of $X _ { t }$ that predict the target while $I ( \bar { V } _ { t } ; V _ { t + 1 } )$ is preserved.

Parameter space. The sensitivity of the encoding of the latent to the micro-level parameters is

$$
F _ { \mathrm { p a r a m } } = \mathbb { E } \big [ \nabla _ { \theta } \log p _ { \theta , w ^ { * } ( \theta ) } ( V _ { t } \mid L _ { t } ) \nabla _ { \theta } \log p _ { \theta , w ^ { * } ( \theta ) } ( V _ { t } \mid L _ { t } ) ^ { \top } \big ] , \qquad \nabla _ { \theta } = \partial _ { \theta } + \Big ( \frac { \partial w ^ { * } } { \partial \theta } \Big ) ^ { \top } \partial _ { w } .\tag{4}
$$

We condition on the true latent $L _ { t } .$ , matching $F ( V ) = I ( V ; L )$ , rather than on $X _ { t }$ , which would block the dependence of $V$ on θ through the micro-state dynamics. The total derivative includes the adaptation of the trained weights $w ^ { \ast } ( \theta ) ^ { \intercal }$ : degeneracy means that training compensates for changes in θ, making this Fisher small relative to that of the micro-level activity. Λ is an ensemble-level informationtheoretic analogue of this comparison. In the low-noise limit Fisher and mutual information are related [44], which motivates the substitution in the main text. Unlike sloppy-model analyses [40, 41], we do not examine the eigenvalue spectrum of $F _ { \mathrm { p a r a m } } .$

For the reservoir model of Section $4 , X _ { t }$ are the reservoir states, w the bottleneck weights, θ the reservoir hyperparameters, and $L _ { t }$ the true attractor state.

## C Estimator details and robustness checks

The class-conditional model in the Barber–Agakov bound is Gaussian with covariance shrinkage $0 . 5 ;$ we fit both a plain and a rank-normal Gaussian and report the larger bound, since both are valid lower bounds. Λ is reported only when $I ( \theta ; F ( X ) ) > 0 . 2$ bits, a negative held-out $I ( \theta ; F ( V ) ) $ is clipped to zero, and $\Lambda = 1 - \operatorname* { m a x } ( I ( \theta ; F ( V ) ) , 0 ) / \operatorname* { m a x } ( I ( \theta ; F ( X ) ) , I ( \theta ; F ( V ) ) )$ , so that $\Lambda \in [ 0 , 1 ]$ . The choice of micro feature and of bound changes the level of Λ but not its rise over training, and the information about θ retained by the representation stays above the shuffled-label null throughout (Fig. 4).

![](images/2c7fe2fb055f0a09fa0384349ca16dbc8ec5791982206ceab538ccc728c010c0.jpg)

![](images/29f56b9fc675507d92c4fdf00091c94310456885727cf0b9fe9831f1233df47d.jpg)

![](images/47088e0637fe62d22f553fb5aff4d80a774478913b9b3ae3cd1463d00d8426b8.jpg)  
Figure 4: Robustness of parameter-space compression (noise-free input, training sequences). A Λ with the micro feature $F ( X )$ built from k random projections of the reservoir states $( k = 2 4$ in the main text); dashed line: shuffled-label null, $\Lambda = 1$ . B Λ with the Gaussian Barber–Agakov bound used in the main text and with a multinomial-logistic bound. C The numerator of Λ, $I ( \theta ; F ( V ) )$ with $F ( V ) = I ( V ; L )$ , and the same bound with shuffled θ labels. Both choices shift the level of Λ but not its rise, and ${ \dot { I } } ( \theta ; F ( V ) )$ stays above its null, so the approach of Λ towards 1 is not an estimator floor.

## D Alignment with the true latent

![](images/a3c610b3f3f2f27661fce55f54d021460e8bb23f772f8af3b9f286746fa3b336.jpg)  
Figure 5: Performance and alignment for noise-free input, with lines and bands as in Fig. 2. A One-step prediction error (as Fig. 2A). B Alignment of the learned with the true latent, $I ( V ; L )$

## E Noisy input

![](images/0b96defa56c6564e2ef6ea07e5972986f08db2ef6f6296c458093ad88e25391e.jpg)  
Figure 6: As Fig. 2, for models trained and evaluated on noisy input $( \sigma _ { y } = 0 . 1 )$ . Λ approaches 1 in trained runs and plateaus near 0.95 in controls; the ordering of the two modes and the differences from the controls are preserved.