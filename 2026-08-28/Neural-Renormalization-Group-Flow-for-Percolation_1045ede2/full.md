# Neural Renormalization Group Flow for Percolation

A. Alvez,<sup>1</sup> L. Camagna,<sup>1</sup> S. Chibbaro,<sup>1,</sup> <sup>2</sup> C. Furtlehner,<sup>2,</sup> <sup>1</sup> F.P. Landes,<sup>1,</sup> <sup>2</sup> G. Manzan,<sup>2,</sup> <sup>1</sup> and L. Mensi<sup>1</sup>

<sup>1</sup>LISN, AO team, Bˆat 660 Universit´e Paris-Saclay, Orsay Cedex 91405

<sup>2</sup>Inria Saclay - Tau team, Bˆat 660 Universit´e Paris-Saclay, Orsay Cedex 91405

Machine learning ofers a possible route to data-driven real-space renormalization when the relevant observables are nonlocal and dificult to prescribe explicitly. We explore this idea for twodimensional site percolation developping a supervised, scale-shared neural architecture. The model recursively applies the same learned coarse-graining rule across scales, producing a latent field from which the crossing probability is predicted, while a corresponding fine-graining decoder reconstructs the largest-cluster mask. Trained only on small lattices, the model extrapolates to substantially larger systems, recovers the spanning cluster with high fidelity, and produces observables obeying the expected finite-size scaling near the critical point. We observe that to get such performance it is key that the learned latent representation exhibits critical fluctuations and scale-dependent flows consistent with the renormalization-group structure of percolation.

Introduction The use of machine learning (ML) in physics has led to important progress across many domains [1]. Yet physical datasets difer from generic image or text datasets in a crucial respect: their relevant structures are often constrained by symmetries, conservation laws, causal relations, and long-range correlations. These constraints are not merely useful prior information; in many systems they determine which observables are physically meaningful. This motivates hybrid approaches in which neural architectures are endowed with appropriate physical information, rather than relying exclusively on generic data-driven models [2].

Scale-invariant systems provide a particularly stringent test case. Processes such as avalanches [3], earthquakes [4], and solar flares [5] exhibit power-law statistics and correlations across a broad range of scales. Learning such systems requires more than detecting local patterns: the model must represent how observables transform under coarse graining, possibly with nontrivial anomalous dimensions. This requirement is closely related to known limitations of standard neural architectures, including spectral bias [6], and has recently been emphasized in the context of scale-invariant regression and extrapolation [7]. In such problems, successful extrapolation is expected only when the architecture or representation captures the underlying self-similarity [8].

Percolation is a central problem in statistical physics [9–11], and ofers a minimal and well-controlled setting in which to probe this issue [12, 13]. In twodimensional site percolation, the existence of a spanning cluster is a global connectivity property, while critical configurations are fractal and governed by universal scaling laws. Recent ML studies have shown that standard convolutional neural networks can roughly identify the percolation threshold or infer the occupation density, but they are not able to recognize the percolating cluster itself [14–18]. This suggests that the standard networks may learn density-like local proxies rather than the scaledependent connectivity structure that defines the physical transition.

This observation connects naturally with the renormalization group (RG), whose purpose is precisely to describe how relevant degrees of freedom evolve under changes of scale [19–21]. Several works have explored the relation between deep learning and RG, including mappings between restricted Boltzmann machines and variational RG [22], information-theoretic neural RG schemes [23], normalizing-flow-based RG approaches [24], neural ODE for functional RG approaches [25] when the Hamiltonian is given and more recently, generative perspectives [26–29]. However, much of this literature focuses on unsupervised learning or on generative coarsegraining, often using the two-dimensional Ising model as a testbed [30]. Capturing the correct critical exponents and anomalous scaling behavior seems to remain out of reach for present approaches. From this perspective, RG provides more than an analogy with deep learning. It suggests a concrete architectural principle: the same local transformation should be applied recursively across scales, so that the representation evolves through a hierarchy of coarse-grained variables. For critical percolation, such a representation should preserve the information relevant to spanning connectivity while discarding microscopic details that are irrelevant under coarse graining. This is precisely the type of inductive bias that is absent from standard convolutional architectures, where filters are local but the learned representation is not constrained to transform consistently across scales.

In this Letter, we investigate this idea in a supervised setting. We consider two related tasks for twodimensional site percolation: predicting whether a configuration percolates and reconstructing the correspond ing spanning cluster. To this purpose, we introduce an original scale-shared encoder-decoder architecture in which the same learned coarse-graining (respectively finegraining) map is applied at every scale. Our results, going well beyond previously considered benchmarks, show that the resulting model learns a finite-size scaling order parameter, extrapolates to lattice sizes much larger than those used during training, and reconstructs the nonlocal cluster structure responsible for percolation. In addition, the critical exponents of the theory are faithfully reproduced by the model predictions and remarkably, these performances seem to rely crucially on the presence of an accurate RG flow in the latent space.

![](images/09edeba1ff91fb8922a54aa95e766fd233c86f36b405addce7fb6f268801337c.jpg)  
FIG. 1. Site-percolation configuration at criticality (left), mask of the spanning cluster (center), and prediction of the neural network (right). The lattice size is $L = 1 0 2 4$ , which lies outside the training range and therefore probes finite-size extrapolation.

Percolation model. We consider two-dimensional site percolation on a square lattice of linear size L. Each site is independently occupied with probability p (also called permeability) and empty otherwise. Occupied nearestneighbor sites belong to the same cluster. A configuration is said to percolate if at least one occupied cluster spans the system either horizontally or vertically. A representative critical configuration, together with the corresponding spanning-cluster mask and the prediction of our model, is shown in Fig. 1. Theoretical results have been obtained in the thermodynamic limit, analytically or numerically [31, 32]; site percolation undergoes a continuous geometric phase transition at the critical occupation probability $p _ { c } \simeq 0 . 5 9 2 7 4 6$ , for the square lattice. Denoting by $\Pi ( p , L )$ the probability that a system of size L percolates, one has lim $L \to \infty \Pi ( p , L ) = \Theta ( p - p _ { c } )$ . Near criticality, the correlation length diverges as

$$
\xi ( p ) \sim | p - p _ { c } | ^ { - \nu } , \qquad \nu = \frac 4 3 .\tag{1}
$$

The order parameter $P _ { \infty }$ , defined as the probability that a site belongs to the infinite cluster, scales for $p > p _ { c }$ as

$$
P _ { \infty } ( p ) \sim ( p - p _ { c } ) ^ { \beta } , \qquad \beta = \frac { 5 } { 3 6 } .\tag{2}
$$

For finite systems, the transition is rounded and the relevant scaling variable is $( p - p _ { c } ) L ^ { 1 / \nu }$ . Thus, $\Pi ( p , L ) \simeq$ $\mathcal { F } \left( ( p - p _ { c } ) L ^ { 1 / \nu } \right)$ while the mass of the critical spanning cluster scales as $M ( L ) \sim L ^ { d _ { f } }$ , with fractal dimension $\begin{array} { r } { d _ { f } = 2 - \frac { \beta } { \nu } = \frac { 9 1 } { 4 8 } } \end{array}$ . These finite-size scaling laws make percolation a natural benchmark for testing whether a neural network has learned a scale-dependent connectivity observable rather than a local density proxy.

![](images/65432625d9e611617d244a21dfee6be8da634971f900b6b068079ffbe5f37360.jpg)  
FIG. 2. Scale-shared RG-inspired encoder-decoder architecture. The encoder applies the same learned coarse-graining map $f _ { \theta }$ at every scale until the lattice is represented by a single latent vector, from which a readout $g _ { \theta }$ predicts the percolation probability. The decoder applies the same learned finegraining map $f _ { \theta } ^ { \prime }$ at every scale, combining coarse information with encoder features to reconstruct the largest-cluster mask.

Methods. We introduce a scale-shared encoderdecoder architecture designed to mimic the hierarchical structure of a real-space renormalization procedure. The central constraint is that the same learned coarsegraining rule is applied at every scale. As a consequence, the number of parameters is independent of the lattice size $L ,$ while larger systems are processed by applying the same transformation more times. This is a variant of a U-Net [33] that we specifically adapt to the RG setting. The architecture is sketched in Fig. 2.

Given a percolation configuration $x \in \{ 0 , 1 \} ^ { L \times L }$ , with $L = 2 ^ { k }$ , the encoder first lifts the binary field to a $C -$ channel latent representation $h ^ { 0 }$ . It then constructs a hierarchy of coarse-grained fields

$$
h ^ { \ell + 1 } = f _ { \theta } ( h ^ { \ell } ) , \qquad \ell = 0 , \dots , k - 1 ,\tag{3}
$$

![](images/76195c23cd1288c6e03dc049d7ac9a3034b1e979d1ce7b66f549afaf8e2f6324.jpg)  
FIG. 3. Finite-size scaling of the learned percolation observables. Top Panel: predicted crossing probability $\widehat { \Pi } ( p , L )$ as a function of $p ,$ with inset showing the collapse as a function of $x = ( p - p _ { c } ) L ^ { 1 / \nu }$ . Bottom Panel: predicted order parameter $\hat { P } _ { \infty } ( p , L )$ obtained from the decoder output, compared with the target cluster mass fraction. The inset shows the behavior above the critical point and the corresponding estimate of $\beta .$

where $h ^ { \ell } \in \mathbb { R } ^ { \left( \frac { L } { 2 ^ { \ell } } \right) ^ { 2 } \times C }$ has spatial size $L / 2 ^ { \ell }$ (pictured as red blocks in the sketch). The map $f _ { \theta }$ is local and reduces the linear resolution by a factor two at each step, while keeping the number of channels fixed. After k iterations, the whole configuration is represented by a single vector $h ^ { k } \in \mathbb { R } ^ { C }$ . A readout map g<sub>θ</sub> then predicts the probability that the sample percolates,

$$
\Pi _ { \theta } = \sigma \left( g _ { \theta } ( h ^ { k } ) \right) ,\tag{4}
$$

where $\sigma$ is the sigmoid function, giving for percolation a scalar neural order parameter [34]. The encoder is trained using a binary cross-entropy loss on the percolation label.

The decoder is used to reconstruct the largest-cluster mask, which is non empty even for non-spanning configurations. It starts from the coarsest representation to proceed toward reproducing the fine-grained details. Denoting the decoder variables by $z ^ { \ell }$ (blue blocks in the sketch), we set $z ^ { k } = h ^ { k }$ and iterate

$$
z ^ { \ell } = f _ { \phantom { \prime } \theta } ^ { \prime } \left( h ^ { \ell } , \mathcal { U } z ^ { \ell + 1 } \right) , \qquad \ell = k - 1 , \dots , 0 ,\tag{5}
$$

where denotes a fixed upsampling operator. At each scale, the decoder combines the information propagated from coarser scales with the encoder representation at the same resolution. The same fine-graining rule $f _ { \theta } ^ { \prime }$ is used at every scale. A final readout predicts the mask

$$
p _ { \theta , i } = \sigma \left( g _ { \theta } ^ { \prime } ( z _ { i } ^ { 0 } ) \right) ,\tag{6}
$$

where $p _ { \theta , \astrosun }$ <sub>i</sub> is the predicted probability that site i belongs to the largest cluster. The decoder is trained with a pixel-wise binary cross-entropy loss on the true mask.

This construction difers from a standard convolutional encoder-decoder in its explicit scale sharing: the same coarse- and fine-graining maps are reused across all levels of the hierarchy. This makes the architecture naturally applicable to lattice sizes larger than those seen during training. The proposed architecture is original and is further described in the End Matter.

Results. We train the model on a mixed-size ensemble of square lattices with $L = 2 ^ { k } , k = 2 , \dots , 6$ , using $2 \times 1 0 ^ { 4 }$ samples per size. The trained model is then evaluated on $k = 2 , \ldots , 1 0$ , corresponding to $L = 1 0 2 4 , \mathrm { i . e }$ . a linear size 16 times larger than the largest training lattice. Samples are drawn with values of $p$ near criticality, using the finite-size scaling variable

$$
x = ( p - p _ { c } ) L ^ { 1 / \nu } ,\tag{7}
$$

with $p _ { c } = 0 . 5 9 2 7 4 6$ and $\nu = 4 / 3$ . For each size, the range of values of $p$ sampled is chosen to cover the rounded transition region, where the crossing probability satisfies approximately $0 . 1 < \Pi ( p , L ) < 0 . 9$ This avoids trivial examples far from criticality. This combination of sizes and p’s makes both the classification and reconstruction tasks sensitive to long-range connectivity. We first examine whether the network output has the expected finite-size scaling structure. Let $\widehat { \Pi } ( p , L )$ denote the empirical average of the predicted probability of percolation for samples of size L and occupation probability p. As shown in Fig. 3, $\widehat { \Pi } ( p , L )$ develops a sharp transition near the known critical point. When plotted as a function of $x = ( p - p _ { c } ) L ^ { 1 / \nu }$ , the curves obtained for diferent lattice sizes collapse onto a common scaling function, including sizes not seen during training. A finite-size scaling fit $p _ { c } ^ { \mathrm { f i t } } \simeq 0 . 5 9 2 9 5$ and $\nu ^ { \mathrm { { \hat { f i t } } } } \simeq 1 . 3 3 { \mathrm { { \tilde { 8 3 } } } }$ , quite close to the twodimensional percolation values. This indicates that the network has learned a dimensionless crossing/no-crossing observable, rather than a size-dependent decision boundary.

The decoder output gives access to a spatially resolved estimate of the largest-cluster mask. For $p \ > \ p _ { c } ,$ , the predicted cluster mass fraction $\begin{array} { r } { \hat { P } _ { \infty } ( p , L ) = 1 / L ^ { 2 } \sum _ { i } \hat { p } _ { i } } \end{array}$ provides a finite-size estimate of the percolation order parameter. As shown in Fig. 3, this quantity follows the expected growth above the transition. A fit of $\hat { P } _ { \infty } ( p , L )$ above $p _ { c }$ at the largest tested size gives an effective exponent $\beta ^ { \mathrm { f i t } } \simeq 0 . 1 7 9 \pm 0 . 0 0 6$ , while the scaling of $\hat { P } _ { \infty } ( p _ { c } , L )$ gives a fractal dimension $d _ { f } \simeq 1 . 8 8 7$ , equivalently $\beta / \nu \simeq 2 - d _ { f }$ . These estimates are close to the exact values $\beta = 5 / \bar { 3 6 } \approx 0 . 1 3 9$ and $d _ { f } = 9 1 / 4 8 \approx 1 . 8 9 6$ within the finite-size and learning errors of the experiment.

![](images/186d09e2da19497200bc0bbbe7772c2b90aa9f8c6705e35a0269b4354a41e6f7.jpg)  
FIG. 4. Classification and reconstruction performance as a function of the rescaled occupation probability $x ~ = ~ ( p ~ -$ $p _ { c } ) L ^ { 1 / \nu }$ Top: encoder accuracy for the percolation label. Bottom: decoder accuracy for the largest-cluster mask. The largest size used during training is $L = 6 4$ , while the test sizes extend up to $L = 1 0 2 4$ . For each value of x, the reported value is the median over $N = 5 0 0 0$ test samples. Error bars indicate bootstrap uncertainty over test samples. Insets show the performance at criticality as a function of $L .$

To corroborate quantitatively these findings, we then evaluate the supervised tasks directly. Figure 4 reports the classification accuracy of the encoder and the reconstruction accuracy of the decoder across system sizes and values of $x .$ The performances of this architecture are quite striking. The model maintains high classification accuracy across the critical window, including in the extrapolation regime $L > 6 4$ . As expected, the accuracy is lowest near $p _ { c }$ , where microscopic changes can alter the global connectivity label. By contrast, as shown in SM a standard U-Net achieves comparable performance on lattice sizes represented during training, but it deteriorates sharply when extrapolating to larger systems. Thus, successful interpolation of the percolation transition does not by itself guarantee scale generalization; this comparison supports scale sharing as the key inductive bias enabling extrapolation across system sizes.

For cluster reconstruction, the encoder-decoder is trained directly from scratch using a pixel-wise binary cross-entropy loss against the largest-cluster mask. This strategy outperforms freezing a classifier-pretrained encoder, likely because the classification task emphasizes the existence of a spanning cluster, whereas the reconstruction task requires identifying the largest cluster in both percolating and non-percolating configurations. The reconstruction task is more stringent than binary classification: a classifier may exploit finite-size texture gradients, whereas reconstructing the cluster requires identifying a nonlocal object across scales. The example shown in Fig. 1 confirms that the decoder preserves the coarse connectivity of the spanning structure and refines its geometry down to the microscopic scale, even for $L = 1 0 2 4$

Together, these observations show that the scaleshared architecture learns more than an occupationdensity proxy. Its global output obeys finite-size scaling, its decoder recovers the spatial structure of the largest cluster, and both properties persist beyond the training range.

Latent RG flow. We finally analyze how the scaling behavior is represented internally by the encoder. Let

$$
\Phi = h ^ { k } \in \mathbb { R } ^ { C }\tag{8}
$$

be the coarsest latent vector obtained after $k = \log _ { 2 } L$ applications of the same learned coarse-graining map. Since increasing L amounts to applying the same map more times, the family of distributions $\rho _ { \theta , L } ( \Phi \mid p )$ provides an empirical finite-dimensional representation of the learned flow with scale. The covariance matrix $\Sigma ( \Phi \mid p , L )$ of this field already indicates that its distribution carries relevant information about it. For fixed p and $L ,$ we estimate the covariance of Φ and define the latent susceptibility

$$
\overline { { \Sigma } } _ { \theta } ( p , L ) = \mathrm { T r } \left[ \Sigma ( \Phi \mid p , L ) \right] ,\tag{9}
$$

which measures sample-to-sample fluctuations of the coarse-grained representation. As shown in Fig. $5 ,$ $\Sigma _ { \theta } ( p , L )$ peaks near the transition. Its normalized profiles collapse under finite-size rescaling, with fitted values $p _ { c } ^ { \mathrm { f i t } } \simeq 0 . 5 9 2 9 5$ and $\nu ^ { \mathrm { f i t } } \simeq 1 . 4 1 5 8$ Thus, the latent representation itself carries critical fluctuations with the expected scaling structure. This provides an independent confirmation that the network has learned the whole structure of the transition.

To visualize the learned flow, we project Φ onto the leading principal components computed from critical samples at the largest training size. Figure 5 shows that the critical latent distribution is a mixture of six components associated with distinct connectivity patterns: one for non-percolating configurations, one for percolating configurations in both directions, and four components for percolating configurations in two or three directions and all the rotations of such patterns. This organization indicates that the network has learned a representation sensitive not only to the existence of percolation, but also to the topology of boundary connections.

![](images/350fb54f863b6e4bfa1bd865e51c3ca5a71b09423fb6618c1b88d645bc1e7afb.jpg)

![](images/29c96027170d6d6c2aa8b2d40867e3e695ca64a188301b89003a967602f85af5.jpg)  
FIG. 5. Top:Projection of the coarsest latent representation Φ onto principal components computed from critical samples at the largest training size. Points show critical samples. Colors correspond to distinct connectivity patterns, including nonpercolating configurations (light blue), configurations spanning in both directions (orange), and other percolating patterns related by rotations (green). Six clusters are clearly apparent. Arrows indicate the motion of the distribution $\Phi ( p , L )$ ’s centroid as the lattice size increases, for values of $p$ below and above $p _ { c } .$ providing an empirical visualization of the learned scale flow. Bottom: Finite-size scaling of the normalized latent susceptibility $\bar { \Sigma } ( p , L )$ as function of $p$ with its collapse inset as function of $x = ( p - p c ) L ^ { 1 / \nu }$

Changing p away from $p _ { c }$ deforms this latent distribution. As $L$ increases, the centroid of the projected distribution $\Phi ( p , L )$ drifts for $p > p _ { c }$ [resp. $p < p _ { c } ]$ , toward the percolating [resp. non-percolating] sector materialized by the stable fixed point $\Phi ^ { \triangleright } \ [ \mathrm { r e s p . } \ \bar { \Phi } ^ { \triangle } ]$ . The critical distribution lies near the separatrix between these two trends, with an unstable centroid fixed point $\Phi ^ { \star }$ . This behavior is consistent with the RG picture of an unstable critical fixed point separating two stable phases, although the present analysis should be understood as a finite-dimensional learned proxy rather than a direct reconstruction of the full RG transformation.

Conclusion. We have shown that a scale-shared encoder-decoder architecture can learn critical percolation observables in a supervised setting and extrapolate to lattice sizes larger than those used during training. The model does not only classify percolation: it reconstructs the spatial cluster structure and produces observables whose finite-size scaling are in agreement with that of two-dimensional percolation. Our method goes beyond phases classification with a black-box scalar neural order parameter [34], by learning a neural order parameter in the form of a latent field following the RG structure of a critical system. This further provides an empirical visualization of a learned scale flow between non-percolating and percolating sectors. These results point out that RG-inspired weight sharing is a key inductive bias for learning scale-invariant systems, especially when the relevant observables are nonlocal. A more complete reconstruction of the learned RG dynamics, including symmetry constraints and a controlled treatment of latent-field normalization, remains an important direction for future work.

Acknowledgments Authors wish to thank Raoul Santachiara and Francesco Chippari for insightful discussions on percolation; We also warmly thank Beatriz Seoane and Aurelien Decelle for fruitful discussions. We acknowledge financial support by the French ANR grant Scalp (ANR-24-CE23-1320).

[1] G. Carleo, I. Cirac, K. Cranmer, L. Daudet, M. Schuld, N. Tishby, L. Vogt-Maranto, and L. Zdeborov´a. Machine learning and the physical sciences. Reviews of Modern Physics, 91(4):045002, 2019.

[2] M.M. Bronstein, J. Bruna, T. Cohen, and P. Veliˇckovi´c. Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478, 2021.

[3] O. Ramos. Scale invariant avalanches: a critical confusion. arXiv preprint arXiv:1104.4991, 2011.

[4] Z. Olami, H.J.S. Feder, and K. Christensen. Selforganized criticality in a continuous, nonconservative cellular automaton modeling earthquakes. Phys.Rev.Lett, 68(8):1244, 1992.

[5] D. Hamon, M. Nicodemi, and H.J. Jensen. Continuously driven OFC: A simple model of solar flare statistics. Astronomy & Astrophysics, 387(1):326–334, 2002.

[6] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville. On the spectral bias of neural networks. In ICML, pages 5301–5310. PMLR, 2019.

[7] A. Alvez, C. Furtlehner, and F. Landes. Learning and extrapolating scale-invariant processes. Physical Review E, 114(2):025301, 2026.

[8] I. Sosnovik, M. Szmaja, and A. Smeulders. Scaleequivariant steerable networks. arXiv preprint arXiv:1910.11093, 2019.

[9] D. Staufer and A. Aharony. Introduction to Percolation Theory. Taylor & Francis, London, 2 edition, 1994.

[10] Armin Bunde and Shlomo Havlin, editors. Fractals and Disordered Systems. Springer-Verlag, Berlin and New

York, 1991.

[11] G.R. Grimmett. Percolation, volume 321 of Grundlehren der mathematischen Wissenschaften. Springer, Berlin, 2 edition, 1999.

[12] A. Bunde and S. Havlin. Fractals and disordered systems. Springer Science & Business Media, 2012.

[13] J. Cardy. Scaling and renormalization in statistical physics, volume 5. Cambridge university press, 1996.

[14] W. Zhang, J. Liu, and T.-C. Wei. Machine learning of phase transitions in the percolation and XY models. Physical Review E, 99(3):032142, 2019.

[15] J. Shen, W. Li, S. Deng, and T. Zhang. Supervised and unsupervised learning of directed percolation. Physical Review E, 103(5):052140, 2021.

[16] S. Cheng, F. He, H. Zhang, K.-D. Zhu, and Y. Shi. Machine learning percolation model. arXiv preprint arXiv:2101.08928, 2021.

[17] D. Bayo, A. Honecker, and R.A. R¨omer. The percolating cluster is invisible to image recognition with deep learning. New Journal of Physics, 25(11):113041, 2023.

[18] D. Bayo, B. C¸ ivitcio˘glu, J.J. Webb, A. Honecker, and R.A. R¨omer. Machine learning of phases and structures for model systems in physics. Journal of the Physical Society of Japan, 94(3):031002, 2025.

[19] G.I. Barenblatt. Scaling, volume 34. Cambridge University Press, 2003.

[20] Kenneth G Wilson and John Kogut. The renormalization group and the ϵ expansion. Physics reports, 12(2):75–199, 1974.

[21] N. Goldenfeld. Lectures on phase transitions and the renormalization group. CRC Press, 2018.

[22] P. Mehta and D.J. Schwab. An exact mapping between the variational renormalization group and deep learning. arXiv preprint arXiv:1410.3831, 2014.

[23] M. Koch-Janusz and Z. Ringel. Mutual information, neural networks and the renormalization group. Nature

Physics, 14(6):578–582, 2018.

[24] S.-H. Li and L. Wang. Neural network renormalization group. Phys.Rev.Lett., 121(26):260601, 2018.

[25] D. Di Sante, M. Medvidovi´c, A. Toschi, G. Sangiovanni, C. Franchini, A.M. Sengupta, and A.J. Millis. Deep learning the functional renormalization group. Physical review letters, 129(13):136402, 2022.

[26] W. Hou and Y.-Z. You. Machine learning renormalization group for statistical physics. ML: Science and Technology, 4(4):045010, 2023.

[27] K. Masuki and Y. Ashida. Generative difusion model with inverse renormalization group flows. arXiv preprint arXiv:2501.09064, 2025.

[28] A. Sheshmani, Y.-Z. You, B. Buyukates, A. Ziashahabi, and S. Avestimehr. Renormalization group flow, optimal transport, and difusion-based generative model. Phys.Rev. E, 111(1):015304, 2025.

[29] T. Marchand, M. Ozawa, G. Biroli, and S. Mallat. Multiscale data-driven energy estimation and generation. Physical Review X, 13(4):041038, 2023.

[30] M. Kaur, Y.W. Li, D. Perera, and D.P. Landau. Can machine learning truly decode phase transitions? a deep dive into the ising model with competing interactions. Physical Review E, 114(1):014104, 2026.

[31] J.L. Cardy. Critical percolation in finite geometries. Journal of Physics A: Mathematical and General, 25(4):L201–L206, 1992.

[32] M. E. J. Newman and R. M. Zif. Eficient Monte Carlo algorithm and high-precision results for percolation. Physical Review Letters, 85(19):4104–4107, 2000.

[33] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241, 2015.

[34] J. Carrasquilla and R.G. Melko. Machine learning phases of matter. Nature physics, 13(5):431–434, 2017.

## END MATTER

Architectural details. We describe here the local parametrization of the scale-shared encoder–decoder used in the experiments. The same encoder rule $f _ { \theta }$ and decoder rule $f _ { \theta } ^ { \prime }$ are applied at every scale. Consequently, the number of trainable parameters does not depend on the lattice size L.

Encoder. The coarse-graining map in Eq. (3) is written as

$$
f _ { \theta } ( h ^ { \ell } ) = \phi _ { \theta } \left( \mathrm { R e L U } \circ \mathrm { C o n v } _ { \theta } ( h ^ { \ell } ) \right) ,\tag{10}
$$

where ReLU denotes the Rectified Linear Unit nonlinearity, applied element-wise. The first operation is a standard two-dimensional convolution with kernel size $3 \times 3 .$ , stride 1, mirror padding at boundaries, and C input and output channels. Its role is to let neighboring blocks exchange information before the actual coarse-graining step. This operation is important in practice, since treating $2 \times 2$ blocks independently would lose information at block boundaries. The second operation, $\phi _ { \theta } .$ , performs the block collapse. The field $\operatorname { C o n v } _ { \theta } ( h ^ { \ell } )$ is partitioned into non-overlapping $2 \times 2$ patches. On each patch, the $4 C$ input values are mapped to one C-dimensional vector by the same two-layer multilayer perceptron (MLP),

$$
\phi _ { \theta } : \mathbb { R } ^ { 4 C }  \mathbb { R } ^ { C } .\tag{11}
$$

In all experiments we use $C = 3 2$ channels, ReLU activations and hidden width 64. Since the $2 \times 2$ patches are reduced to a single site, each application of $f _ { \theta }$ divides the number of sites by 4 and halves the linear system size. Iterating this rule $k = \log _ { 2 } L$ times maps the original configuration to a single latent vector $h ^ { k } \in \mathbb { R } ^ { C }$ . The final encoder readout $g _ { \theta }$ is a small multilayer perceptron (one hidden layer of hidden width 64): $\Pi _ { \theta } = \sigma \left( g _ { \theta } ( h ^ { k } ) \right)$ , where $\Pi _ { \theta }$ is interpreted as the predicted probability that the configuration percolates. The encoder is trained with a binary cross-entropy loss on the percolation label.

Decoder. The decoder starts from the coarsest representation $z ^ { k } = h ^ { k }$ and refines it scale by scale. At each level, the coarse decoder representation is first upsampled by copying each coarse pixel onto the corresponding $2 \times 2$ block,

$$
\mathcal { U } \boldsymbol { z } ^ { \ell + 1 } \in \mathbb { R } ^ { C \times ( L / 2 ^ { \ell } ) \times ( L / 2 ^ { \ell } ) } .\tag{12}
$$

It is then concatenated with the encoder feature $h ^ { \ell }$ at the same resolution. The fine-graining rule $z ^ { \ell } = f _ { \boldsymbol { \theta } } ^ { \prime } \left( h ^ { \ell } , \mathcal { U } z ^ { \ell + 1 } \right)$ $\mathrm { ( E q . ~ 5 ) }$ is applied independently on each $2 \times 2$ patch. It applies a linear map to the $_ { 8 C }$ values contained in the concatenated patch to output $_ { 4 C }$ values, which are reshaped into a refined C-channel $2 \times 2$ block, followed by a ReLU nonlinearity. Equivalently, $f _ { \theta } ^ { \prime }$ may be viewed as a local $2 \times 2$ (stride 2) convolution followed by a nonlinearity. The skip connection through $h ^ { \ell }$ plays the same role as in a U-Net: it reintroduces information recorded by the encoder at resolution $\ell ,$ which cannot be recovered from the coarser representation alone. The diference with a standard U-Net is that the same rule $f _ { \theta } ^ { \prime }$ is shared across all scales. At the final scale, the decoder output is combined with the original binary configuration x and mapped linearly to a four-pixel output on each $2 \times 2$ block. This gives the mask prediction (6) where $p _ { \theta , i }$ can be interpreted as the predicted probability that site i belongs to the largest cluster. The decoder is trained with a pixel-wise binary cross-entropy loss against the largest-cluster mask.

Parameter count. For $C = 3 2$ and hidden width 64, the scale-shared encoder rule $f _ { \theta }$ contains 19,584 parameters, while the classifier readout $g _ { \theta }$ contains 2,177 parameters. The decoder fine-graining rule $f _ { \theta } ^ { \prime }$ contains 32,896 parameters and the final decoder readout $g _ { \theta } ^ { \prime }$ contains 532 parameters. The full encoder–decoder therefore contains 55,189 trainable parameters. Since all local rules are shared across scales, this number is the same for all lattice sizes.