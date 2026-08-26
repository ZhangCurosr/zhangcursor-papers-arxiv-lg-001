# Single State Update Predictive Coding training for Time Series Forecasting and Anomaly Detection

Matteo Cardoni<sup>[0000−0002−2759−5310]</sup> and Sam Leroux<sup>[0000−0003−3792−5026]</sup>

IDLab, Department of Information and Technology, Ghent University—imec, 9052 Ghent, Belgium matteo.cardoni@ugent.be (Corresponding author), sam.leroux@ugent.be

Abstract. Predictive Coding (PC) is a neural learning paradigm that enables parallelizable neural network layer updates. However, the main bottleneck of PC Networks (PCN) is the sequential backwards error propagation. To tackle this, we introduce a training technique that pairs a Generative PCN with a support Encoding PCN. The two PCNs are trained in parallel to match their neural activations, without sequential propagation. We apply this to time series anomaly detection and show that our approach results in more stable, continuous, online learning.

Keywords: Predictive Coding · Online learning · Anomaly detection

Introduction and related work: Predictive coding (PC) [1–3] is a learning paradigm that emerged in recent years as a low-complexity alternative for Backpropagation-based training [4]. Its main attraction is its parallelizable layerwise updates which, in contrast to Backpropagation [5], requires no sequential layer updates. PC training pairs each layer’s activation with an additional state of the same dimension [6]. For each layer l, it involves the minimization of an energy function between the states $h _ { l }$ and the activations $\mu _ { l }$ [6, 7]. The minimization is performed by iteratively optimizing the states and, subsequently, the weights. Despite PC’s parallelizable updates property, Predictive Coding Networks (PCN)’s layer updates require an error signal that propagates sequentially from the output throughout all the layers. Commonly, PCNs are initialized with feedforward initialization [6–8], which involves the states initialization with the activation values. Despite accelerating the convergence [8, 9], it requires at least as many inference steps as the network depth [8, 10] and causes a vanishing update propagation [8]. To overcome the requirement for multiple state update iterations, we introduce a Guided PC training technique that pairs a Generative PCN (G-PCN) with an Encoding PCN (E-PCN), which guides and accelerates the states update by enforcing layer-wise activations matching.Unlike prior state-update acceleration methods [10–14], our solution provides parallel and hierarchical updates across the network depth, using batched data.

Proposed solution: The goal for the G-PCN, of L layers, is to predict the next time-step data $x ^ { t } = h _ { L } ^ { t }$ with $\hat { x } ^ { t } = \mu _ { L } ^ { t }$ . Taking inspiration from temporal Predictive Coding [15], the G-PCN uses $h _ { 0 } ^ { t - 1 }$ to predict the same state at time t $\left( h _ { 0 } ^ { t } \right)$ , and $x ^ { t }$ from it. Training the G-PCN and the E-PCN to produce matching activations implies the minimization of two types of energies: the Internal Energy $\begin{array} { r } { \mathcal { I } = \frac { 1 } { 2 } \sum _ { l } ( h _ { l } - \mu _ { l } ) ^ { 2 } } \end{array}$ between the states and the activations of the G-PCN, and the Guiding Energy $\begin{array} { r } { \mathcal { G } = \frac { 1 } { 2 } \sum _ { l } ( \mu _ { l } - \gamma _ { l } ^ { 2 } ) } \end{array}$ , between the activations of the two PCNs.

For each time frame, our technique follows 4 steps. (i) First, both the G-PCN and the E-PCN perform feedforward initialization, as displayed in Figure 1. The G-PCN maps $h _ { 0 } ^ { \dot { t } - 1 }$ to $\mu _ { 0 } ^ { t }$ via the layer $\theta _ { 0 }$ , which performs temporal mapping. The subsequent layers have the role of a PC decoder [6], mapping the encoded data to $\hat { x } ^ { t }$ . In parallel, the E-PCN performs feedforward initialization taking $x ^ { t }$ as input, acting as an encoder. (ii) As depicted in Figure 2, the first part of the state update happens in the second step, where all the states of both the PCNs are updated in parallel to minimize the $\mathcal { G }$ and $\mathcal { T } _ { L } = ( \hat { x } ^ { t } - x ^ { t } ) ^ { 2 }$ . (iii) The third step involves, for every layer, the classic (Vanilla) PC state update with respect to $\mathcal { T } ,$ displayed in Figure 3. It is performed only by the G-PCN as it uses $\hat { x ^ { t } }$ as reference, ensuring that the E-PCN converges to the same representation. The second and third steps compose a unified, single-iteration state update for the G-PCN. (iv) The fourth step, that is the weight update of both the models, is displayed in Figure 4. The G-PCN weights are updated to minimize $\mathcal { T } ,$ , as in Vanilla PC. In parallel, the E-PCN weights are updated to minimize ${ \mathcal { G } } .$

Overall, the E-PCN is trained to match the representation of the G-PCN, acting as support, similarly to a teacher model [16]. The G-PCN, instead, uses $x ^ { t }$ as reference to propagate its updates with the help of the E-PCN, similarly to a student network.

Experiments and results: We apply this training technique to the task of continuous, online learning for time series anomaly detection. We created a dataset of MNIST digits [17] moving on a black background. We train a 5-layer MLP to predict the next frame, while batching it with previous frames. We use the Mean Squared Error (MSE) between the predicted and the actual frame as anomaly score. Starting from the initial dynamics (diagonal movement of 1 pixel), diferent anomalies have been introduced in the form of abrupt changes in X and/or Y direction or digit value. After an anomaly occurs, the model needs to retrain to consider the new sequence dynamics as normal. Figure 5 shows the anomaly scores over time. We compare traditional (Vanilla) PC training, performing 5 inference steps, with G-PCN. As the model has no prior knowledge, the anomaly scores are high at the start of training. Both techniques are capable of quickly learning the initial dynamics. The periodic spikes are caused by bouncing against the wall and changing direction. After the anomaly is introduced, the Guided PC can quickly recover to treat the new behavior as normal while the vanilla PC diverges. This is probably due to a slower and vanishing error signal propagation of Vanilla PC [8].

Conclusion: We introduced a PC training technique that relies two PC models to eliminate the sequential state update bottleneck, typical of PC, and enhances training stability. Using a single unified state update step, all layers states get updated in parallel. As PC usually requires a high number of inference steps, this technique is promising in terms of edge devices online learning.

![](images/b62827e7339ec942d4337f13c991cea8cbd990a60ed1d1f1a30bf599632551d8.jpg)  
Fig. 1: Guided PC training step 1. Parallel feedforward initialization of the G-PCN and the E-PCN. The initializations follow a sequential order: from $h _ { 0 } ^ { t - 1 }$ to $\hat { x } ^ { t }$ for the G-PCN and form $x ^ { t }$ to $\gamma _ { 0 } ^ { t }$ for the E-PCN. The G-PCN maps $h _ { 0 } ^ { \check { t } - 1 }$ to $\mu _ { 0 } ^ { t }$ via the layer $\theta _ { 0 } .$ , that acts as temporal mapping layer.

![](images/e0ed669125b19331e28cb431f5aaf3eab3071eaa93c64d580cd87bb2c3705d6d.jpg)  
Fig. 2: Guided PC training step 2. G-PCN and E-PCN parallel state update, to minimize the Guiding Energy  and the Inernal Energy  at the last layer. All the the states $h _ { 0 } ^ { t } , \ldots , h _ { L - 1 } ^ { t }$ and $k _ { L - 1 } ^ { t } , \ldots , k _ { 0 } ^ { t }$ receive a non-zero update.

![](images/43d6872eb3748586aa4335917ad420f6d425e8b2727ccfca3e787d7b56bd3cbe.jpg)  
Fig. 3: Guided PC training step 3. G-PCN state update to minimize the Internal Energy . The states $h _ { 0 } ^ { t } , \ldots , h _ { L - 1 } ^ { t }$ already received a non-zero update from step 2 (Figure 2). Therefore, the update with respect to will be non-zero for all $h _ { 0 } ^ { t } , \ldots , h _ { L - 1 } ^ { t }$

![](images/759868825bc8a60103484d3223492963c189cca265f93d3d2417d72f8e309cb7.jpg)  
Fig. 4: Guided PC training step 4. G-PCN and E-PCN parallel weight update. The G-PCN weights are updated to minimize , while the E-PCN weights are updated to minimize . In this way, the E-PCN is trained to match the G-PCN representation, acting as a support network.

Inversion of X and Y direction  
![](images/97deadf8a3887858df1b8763cd5c907b6a809dd01c4b7d63473cab829a53fcf7.jpg)

(a) X and Y directions inversion at frame 635. The later MSE spikes that occur when the digit is close to that position demonstrate the adaptation to the new normality. Inversion of Y direction  
![](images/c828487b85ef4faefbcb033d80eaeac759943baf4718e395278be4bcd0eb202c.jpg)  
(b) Y direction inversion at frame 635. The bounces become more frequent after the anomaly as the contact on the vertical and horizontal borders are not synchronous anymore.

Digit change  
![](images/77588c9351653438c3a3af0e5a84800e1498098a2232aa780f6be1f01e848870.jpg)  
(c) Digit change at frame 635, while maintaining the same direction.

Fig. 5: Average accuracy expressed as Mean Squared Error between the ground truth images $x ^ { t }$ and the predicted images ${ \hat { x } } ^ { t }$ , in sequences with three diferent anomaly types. 100 experiments have been averaged per each category. The semitransparent area indicates the standard deviation per time frame. The Guided PC obtains better stability with respect to anomalies, despite performing only one unified state update step

## References

1. Rajesh Rao and Dana Ballard. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field efects. Nature neuroscience, 2:79–87, 02 1999.

2. Karl Friston. A theory of cortical responses. Philosophical transactions of the Royal Society of London. Series B, Biological sciences, 360:815–36, 04 2005.

3. Karl Friston and Stefan Kiebel. Predictive coding under the free-energy principle. Philosophical transactions of the Royal Society of London. Series B, Biological sciences, 364:1211–21, 05 2009.

4. Beren Millidge, Tommaso Salvatori, Yuhang Song, Rafal Bogacz, and Thomas Lukasiewicz. Predictive coding: Towards a future of deep learning beyond backpropagation? arXiv preprint arXiv:2202.09467, 2022.

5. David E Rumelhart, Geofrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. nature, 323(6088):533–536, 1986.

6. Luca Pinchetti, Chang Qi, Oleh Lokshyn, Gaspard Olivers, Cornelius Emde, Mufeng Tang, Amine M’Charrak, Simon Frieder, Bayar Menzat, Rafal Bogacz, Thomas Lukasiewicz, and Tommaso Salvatori. Benchmarking predictive coding networks – made simple. arXiv preprint arXiv:2407.01163, 2025.

7. Björn van Zwol, Ro Jeferson, and Egon L. van den Broek. Predictive coding networks and inference learning: Tutorial and survey. arXiv preprint arXiv:2407.04117, 2024.

8. Cédric Goemaere, Gaspard Oliviers, Rafal Bogacz, and Thomas Demeester. Error optimization: Overcoming exponential signal decay in deep predictive coding networks. arXiv preprint arXiv:2505.20137, 2025.

9. James C. R. Whittington and Rafal Bogacz. An approximation of the error backpropagation algorithm in a predictive coding network with local hebbian synaptic plasticity. Neural Computation, 29(5):1229–1262, 05 2017.

10. Luca Pinchetti, Simon Frieder, Thomas Lukasiewicz, and Tommaso Salvatori. Faster predictive coding networks via better initialization. arXiv preprint arXiv:2601.20895, 2026.

11. Davide Casnici, Martin Lefebvre, Justin Dauwels, and Charlotte Frenkel. Accelerated predictive coding networks via direct kolen-pollack feedback alignment. arXiv preprint arXiv:2602.15571, 2026.

12. Aleksandrs Baskakovs, Sylvain Estebe, Kenneth Enevoldsen, Kristofer Nielbo, Chris Mathys, and Nicolas Legrand. Closed-form predictive coding via hierarchical gaussian filters. arXiv preprint arXiv:2605.20293, 2026.

13. Nick Alonso, Jef Krichmar, and Emre Neftci. Understanding and improving optimization in predictive coding networks. arXiv preprint arXiv:2305.13562, 2023.

14. Tommaso Salvatori, Yuhang Song, Yordan Yordanov, Beren Millidge, Zhenghua Xu, Lei Sha, Cornelius Emde, Rafal Bogacz, and Thomas Lukasiewicz. A stable, fast, and fully automatic learning algorithm for predictive coding networks. arXiv preprint arXiv:2212.00720, 2024.

15. Mufeng Tang, Helen Barron, and Rafal Bogacz. Sequential memory with temporal predictive coding. Advances in neural information processing systems, 36:44341– 44355, 2023.

16. Chengming Hu, Xuan Li, Dan Liu, Xi Chen, Ju Wang, and Xue Liu. Teacher-student architecture for knowledge learning: A survey. arXiv preprint arXiv:2210.17332, 2022.

17. Li Deng. The mnist database of handwritten digit images for machine learning research [best of the web]. IEEE Signal Processing Magazine, 29(6):141–142, 2012.

18. Sung-Cheol Kim, Adith S. Arun, Mehmet Eren Ahsen, Robert Vogel, and Gustavo Stolovitzky. The fermi–dirac distribution provides a calibrated probabilistic output for binary classifiers. Proceedings of the National Academy of Sciences, 118(34):e2100761118, 2021.

19. Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2017.

## Appendix

Data batching: In order to provide the PCNs with suficient examples, every input data was composed as a batch of 128 frames $s ^ { t } = x ^ { t } , x ^ { t - 1 } , \ldots x ^ { \mathsf { { \bar { t } } - 1 2 7 } }$ . The training began at $t = 0$ with $s ^ { 0 } = x ^ { 0 } , \mathbf { 0 } , \dots , \mathbf { 0 }$ . At each new time-step, every $x ^ { t }$ pushed of 1 time-step, $x ^ { t - 1 2 7 }$ is discarded and the new frame is assigned to $x ^ { t }$ As a consequence, the G-PCN state $h _ { 0 } ^ { t - 1 }$ also consists of a batch of 128 elements, each initialized to 0 at $t = 0$

Training details: To avoid excessive updates when the G-PCN already learned a good forecasting capability, an additive gaussian noise with variance $\sigma = 1 0 ^ { - 4 }$ has been added to $h _ { 0 } ^ { \bar { t } - 1 }$ . Moreover, the weights learning rates were modulated with respect to the ratio between the Average Mean Squared Error (MSE) between the generated frame and the input frame at time t (MSE<sup>t</sup>) and the same MSE at time $0 ~ ( \mathrm { M S E } ^ { 0 } )$ ). Specifically, a Fermi-Dirac function [18] has been used, passing as ${ \mathrm { x - v a l u e ~ 1 - } } { \frac { \mathrm { a v g . ~ M S E } ^ { t } } { { \mathrm { a v g . ~ M S E } } ^ { 0 } } } .$ . A learning rate α is modulated as $\alpha _ { \mathrm { m o d } } = \frac { \alpha } { \mathrm { e x p } \left( \left( 1 - \frac { \mathrm { a v g . ~ M S E } ^ { t } } { \mathrm { a v g . ~ M S E ^ { 0 } } } - \mu \right) / K T \right) + 1 }$ , using $\mu = 0 . 9$ and $K T = 0 . 0 5$ . These parameters allow the weights learning rates to stay close to their maximum value unless for a low MSE<sup>t</sup>, when the weights learning rates drop to avoid undue updates.

All experiments have been performed using the PCX framework [6]

Architectural details: The architectural details for the G-PCN and the E-PCN are reported in Table 1. Both PCNs have MLP architectures. The Vanilla PCN shares the same architecture with the G-PCN. None of the models use biases in their Dense layers, to maximize the causality between $h _ { 0 } ^ { t - 1 }$ and $\hat { x } ^ { t }$ tanh has been used as non-linearity to keep the neural activities in a contained range, helping convergence.

Hyperparameters details: Stochastic Gradient Descent was used for all the state optimizers. Adam [19] was used for all the weights optimizers PCs. Table 2 displays the hyperparameters used for grid search design space exploration. For the Vanilla PC, a higher number of hyperparameters was used, as only two optimizers are used (instead of 5 for the Guided PC). Table 3 displays the learning rates that have were to produce the results we presented. To choose the best performing configuration for Guided PC and Vanilla PC, the exploration began performing one experiment per configuration and anomaly type, using the first image of the MNIST training set. The average MSE between $x ^ { t }$ and $\hat { x } ^ { t }$ was evaluated. Only configurations producing the last 100 frames with average $\mathrm { M S E } ^ { t } < = 1 5 \%$ average $\mathrm { M S E ^ { 0 } }$ were considered. They were subsequently ordered in order of increasing variance, as we wanted ${ \hat { x } } ^ { t }$ to match $x ^ { t }$ without diference spikes. The configuration that provided the lower variance among the three anomaly types was selected and used for 100 experiments with diferent MNIST digits and initialization seeds, and is summarized in Table 3. Both the Adam optimizers $\beta _ { 1 }$ and $\beta _ { 2 }$ coeficient are 0, which signifies that an absence of momentum is favorable when learning abrupt changes, as are the anomalies and bouncing.

Table 1: Architectural details for the G-PCN and the E-PCN. The G-PCN forward direction goes in increasing order, while the E-PCN one goes in decreasing direction (see Figure 1). Every cell of the table lists the Dense layers dimensions, the layer connections and the non-linearity applied to the activation.
<table><tr><td></td><td>G-PCN</td><td>E-PCN</td></tr><tr><td>level -1</td><td>(512, 512)  $\dot { h } _ { 0 } ^ { t - 1 } \to \dot { h _ { 0 } ^ { t } }$  tanh</td><td></td></tr><tr><td>levels 0–2</td><td> $( 5 1 2 , 5 1 2 )$   $h _ { i } ^ { t } \to \mu _ { i + 1 } ^ { t }$  tanh</td><td> $( 5 1 2 , 5 1 2 )$   $k _ { i + 1 } ^ { t }  \gamma _ { i } ^ { t }$  tanh</td></tr><tr><td>level 3</td><td>(512, 4096)  $h _ { 3 } ^ { t } \to \hat { x } ^ { t }$  linear</td><td>(4096, 512)  $x ^ { t }  \gamma _ { 2 } ^ { t }$  tanh</td></tr></table>

Table 2: Hyperparameters used for design space grid search. The $\beta _ { 1 }$ and $\beta _ { 2 }$ coeficient for the Adam optimizers have been explored in couples of the same value (using the same for both the Guided PC optimizers).
<table><tr><td></td><td>SGD learning rates</td><td>Adam learning rates</td><td>Adam  $\beta _ { 1 } , \beta _ { 2 }$ </td></tr><tr><td>Guided PC</td><td>[1e-2, 5e-2, 1e-1]</td><td>[1e-4, 2.5e-4, 5e-4]</td><td rowspan="2">[(0, 0), (0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]</td></tr><tr><td>Vanilla PC</td><td> $[ \mathrm { 1 e - 4 } , \mathrm { 2 . 5 e - 4 } , \mathrm { 5 e - 4 } ,$  1e-3, 5e-3, 1e-2, 5e-2, 1e-1]</td><td>[1e-3, 5e-3, 1e-2, 5e-2, 1e-1]</td></tr></table>

Table 3: Learning rates used for Guided PC and Vanilla PC.
<table><tr><td></td><td colspan="2">Internal Energy learning rates</td><td colspan="2">Guiding Energy learning rates</td></tr><tr><td></td><td>States</td><td>Weights</td><td>States</td><td>Weights</td></tr><tr><td>G-PCN</td><td>5e-2</td><td>2.5e-4</td><td>1e-1</td><td></td></tr><tr><td>E-PCN</td><td>一</td><td></td><td>1e-1</td><td>1e-4</td></tr><tr><td>Vanilla-PCN</td><td>5e-2</td><td>2.5e-4</td><td>1</td><td></td></tr></table>

Table 4: Training time per frame averaged for 150.000 frames, from 100 experiments of 1500 frames each ( standard deviation). The required training times are similar for the two techniques because the Guided PC implementation cannot still leverage the parallelization potential.
<table><tr><td>Guided PC</td><td>Vanilla PC</td></tr><tr><td> $7 . 3 \pm 7 3 . 4 ~ \mathrm { m s }$ </td><td> $7 . 2 \pm 8 5 . 2 \ \mathrm { m s }$ </td></tr></table>

Table 4 summarizes the training times, averaged for 150.000 training frames, from 100 experiments of 1500 frames each, on NVIDIA Tesla v100-SXM3-32gb GPU. The reason why the Guided PC is does not require less time than the Vanilla PC is that, at the moment, our implementation cannot leverage on the parallelizability potential.