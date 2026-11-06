# Geometric Relevance Is Not Sufficient: Evaluating Focused Post-Training Supervision in Diffusion Docking

Michael Brown Maryam Rahimimovassagh Mehdi Yazdani-Jahromi Sanjeda Sara Jennifer Ivan Garibay University of Central Florida Orlando, FL, USA

## Abstract

Diffusion-based molecular docking requires learning a corrective score field across a broad corruption distribution, yet individual pose errors can differ substantially in their structural consequences. This creates a post-training conflict between preserving broad score-field coverage and concentrating supervision on corrections that appear more relevant to docking geometry. Using DiffDock [Corso et al., 2023], a fixed-budget study compares broad continuation with torsional supervision focused on Cartesian consequence or conditional target consistency, and with a broad/focused mixture. Evaluation separates fixed score fidelity, generated-pose quality, best-of-10 sampling capacity, and confidence-selected Top-1 docking performance. At < 5 Å, broad continuation had the highest mean Top-1 success among trained policies, and focused or mixed supervision showed no consistent advantage under the tested budget. The starting model had better fixed score fidelity, whereas the evaluated broad-continuation model had higher mean Top-1 success with the same aggregate best-of-10 recovery. Focusing supervision on structurally consequential or more consistently specified corrections did not consistently improve returned-pose success over broad continuation.

## 1 Introduction

Accurate protein–ligand binding-pose prediction supports structure-based drug discovery by enabling interaction modeling, hit identification, and optimization [Zhu et al., 2024, Alcaide et al., 2024]. DiffDock separates pose generation from confidence-based selection by sampling ligand translations, rotations, and torsions and then ranking the resulting poses [Corso et al., 2023, 2024, Khiari et al., 2026]. A near-native pose may therefore be generated without being selected for downstream use, so docking evaluation distinguishes structural accuracy from physical validity and interaction recovery [Buttenschoen et al., 2023, Errington et al., 2024, Morehead et al., 2026].

A score model learns a time-dependent corrective field over states produced by a corruption process, so its training objective averages error across examples, perturbations, and diffusion times [Song et al., 2021, Corso et al., 2023]. Torsional errors act in Cartesian space through atoms displaced around each rotatable bond. Equal angular errors can therefore have different consequences for the ligand pose depending on the bond-axis geometry and the atoms affected [Corso et al., 2023, Jing et al., 2022]. Cartesian consequence and conditional target consistency provide distinct bases for focusing torsional supervision. These requirements collide during fixed-budget post-training. Broad continuation preserves coverage of the learned score-field distribution, while focused supervision concentrates gradient pressure on corrections with greater Cartesian consequence or more consistent targets [Song et al., 2021, Corso et al., 2023]. A geometric diagnosis can identify a difficult or consequential state without specifying the corrective denoising direction required by the reverse process [Jing et al., 2022, Corso et al., 2023]. The decision is whether limited post-training should preserve broad coverage or redirect torsional supervision.

This study examines this decision in DiffDock through a matched five-epoch post-training experiment [Corso et $\mathrm { { a l . } }$ , 2023, 2024], with supporting ablations of geometric relevance, directional action, and torsion-priority criteria detailed in the appendix. Evaluation separates fixed score fidelity from generated-pose success, best-of-10 sampling capacity, and confidence-selected Top-1 docking performance [Buttenschoen et al., 2023, Errington et al., 2024]. No focused or mixed policy showed a consistent advantage over broad continuation on the primary $< 5 \mathring \mathrm { A }$ endpoints. The startingcheckpoint versus post-training EMA comparison showed higher mean Top-1 success alongside worse fixed score fidelity and unchanged aggregate best-of-10 recovery.

## 2 Competing Objectives and Evaluation

DiffDock represents a docking state as ${ z = ( \mathbf { t } , R , \theta ) }$ , with translational, rotational, and torsional coordinates corrupted across diffusion time [Corso et al., 2023]. Prior diffusion work reallocates denoising emphasis across noise levels or timesteps [Song et al., 2021, Karras et al., 2022, Choi et al., 2022, Hang et al., 2023, Nichol and Dhariwal, 2021]; here, the corrupted-state stream is fixed and torsional loss is reallocated across ligand edges using allocation masks and intrinsic or Cartesian metrics. Focused post-training does not alter the sampled complexes, perturbation stream, translation objective, or rotation objective. Instead, it changes the torsional contribution to the objective. Le A denote the torsion-allocation policy and let $Q \in \{ I , \widetilde { G } \}$ denote the torsional metric. For the corresponding torsional loss $\ell _ { \mathrm { t o r } } ^ { A , Q }$

$$
\begin{array} { r l r } & { } & { L _ { A , Q } ( \phi ) = \mathbb { E } _ { z \sim \mu } \left[ \ell _ { \mathrm { t r } } + \lambda _ { R } \ell _ { \mathrm { r o t } } + \lambda _ { T } \ell _ { \mathrm { t o r } } ^ { A , Q } \right] , } \\ & { } & { \nabla _ { \phi } L _ { A , Q } - \nabla _ { \phi } L _ { \mathrm { a l l } , I } = \lambda _ { T } \mathbb { E } _ { z \sim \mu } \left[ \nabla _ { \phi } \left( \ell _ { \mathrm { t o r } } ^ { A , Q } - \ell _ { \mathrm { t o r } } ^ { \mathrm { a l l } , I } \right) \right] . } \end{array}\tag{1}
$$

Here, $A =$ all and $Q = I$ denote broad intrinsic continuation. Because the corrupted-state exposure and non-torsional objectives are held fixed, changing A or $Q$ changes the torsional contribution to the post-training gradient. The competing policies therefore differ in how a fixed corrective budget is allocated rather than in the states presented to the model.

The evaluation measures are separated because candidate generation and confidence-based pose selection are distinct stages in diffusion docking pipelines [Corso et al., 2023, 2024, Zhu et al., 2024]. For $N$ complexes, S generated poses, RMSDs $d _ { i j } ,$ , confidence scores $q _ { i j }$ , and RMSD threshold $^ { a , }$ let

$$
\begin{array} { r l r } {  { d _ { i j } = \mathrm { R M S D } ( \widehat { x } _ { i j } , x _ { i } ^ { * } ) . } } \\ & { } & \\ & { S _ { \mathrm { g e n } } ( a ) = \frac { 1 } { N S } \sum _ { i , j } \mathbf { 1 } [ d _ { i j } < a ] , } \\ & { } & \\ & { S _ { \mathrm { b e s t } } ( a ) = \frac { 1 } { N } \sum _ { i } \mathbf { 1 } [ \operatorname* { m i n } _ { j } d _ { i j } < a ] , } \\ & { } & \\ & { } & { j _ { i } ^ { * } = \arg \operatorname* { m a x } _ { j } q _ { i j } , \qquad S _ { \mathrm { T o p 1 } } ( a ) = \frac { 1 } { N } \sum _ { i } \mathbf { 1 } [ d _ { i , j _ { i } ^ { * } } < a ] . } \end{array}\tag{2}
$$

$( S _ { \mathrm { g e n } } )$ measures generated-pose quality, $( S _ { \mathrm { b e s t } } )$ measures sampling capacity, and $\left( S _ { \mathrm { T o p 1 } } \right)$ measures confidence-selected utility [Corso et al., 2024, Zhu et al., 2024], whereas fixed score fidelity is evaluated separately on common validation perturbations. Accordingly, better score fidelity does not imply better sampling capacity, and better sampling capacity does not imply a better confidenceselected pose. The primary $< 5 \mathring \mathrm { A }$ threshold evaluates coarse pose recovery; supporting $< 2 \mathring \mathrm { A }$ results assess stricter structural accuracy (Appendix G). The relevant comparison in the next section is whether geometric information that identifies difficult pose states also points to the corrective supervision needed to improve them.

## 3 Geometric Diagnosis and Corrective Action

In these probes, low-rank tail energy is the squared Frobenius residual of the best rank-k approximation to a smoothed ligand–receptor proximity matrix. Across the evaluated PDBBind and MOAD slices [Liu et al., 2015, Hu et al., 2005], higher tail energy was associated with larger final RMSD and a greater prevalence of poses above 5 Å RMSD (Appendix Fig. 1a; Appendix Fig. 2). The signal therefore identifies geometrically difficult corrupted poses. The motivation for testing both the statistic and its operational role is supported by equivariant-network work showing that geometric treatment can affect numerical stability and downstream physical error [Meng et al., 2024]. Its downstream meaning, however, depended on how it was operationalized. Direct ranking, noisedependent gating, and contextual fusion did not preserve the same relationship with RMSD, and some gating transformations reversed the ordering induced by the raw statistic. Contextual fusion ranked highest in the aggregate descriptive comparison of the tested signal formulations. The corresponding method-family, cutoff, and weighting comparisons are reported in Appendix B.

The geometric rationale for focusing supervision is more direct at the torsional level. Internal ligand flexibility can be represented through torsional coordinates [Jing et al., 2022, Corso et al., 2023], but equal angular errors do not imply equal Cartesian perturbations. For an atom i at perpendicular distance $\rho _ { i }$ from a torsional rotation axis, an angular error $\Delta \theta$ produces a chord displacement whose squared magnitude is

$$
\begin{array} { r l } & { \| \Delta \mathbf { x } _ { i } \| ^ { 2 } = 4 \rho _ { i } ^ { 2 } \sin ^ { 2 } \biggl ( \displaystyle \frac { \Delta \theta } { 2 } \biggr ) \approx \rho _ { i } ^ { 2 } ( \Delta \theta ) ^ { 2 } , } \\ & { \mathrm { R M S D } ^ { 2 } \approx \Delta \theta ^ { \top } G _ { \mathrm { r a w } } ( \pmb \theta ) \Delta \theta , \qquad G _ { \mathrm { r a w } } ( \pmb \theta ) = \frac { 1 } { n _ { \mathrm { a t o m } } } J ^ { \top } P J . } \end{array}\tag{3}
$$

The first relation gives the single-torsion consequence, while the local metric $G _ { \mathrm { r a w } } ( \pmb { \theta } )$ extends the same principle across coupled torsional coordinates. Here, J maps infinitesimal torsional changes into Cartesian displacements and $P$ removes rigid-body components. The derivation of the chord-displacement identity and the rigid-aligned pullback metric in Equation 3, together with its finite-difference check, is given in Appendix D. Torsions can therefore differ in their Cartesian consequence even when their angular errors are comparable. This provides a geometric basis for testing focused torsional supervision, but it does not specify which corrective update the diffusion model should learn.

That limitation is visible in the directional probes. Reverse diffusion requires a directional denoising field over corrupted translational, rotational, and torsional states [Song et al., 2021, Corso et al., 2023]. Geometry-derived finite-difference directions showed weak alignment with translation score targets and no consistent alignment with rotation score targets (Appendix Fig. 1c; Appendix Fig. 3). Auxiliary geometry-loss gradients likewise showed weak or conflicting alignment with score-matching gradients across the evaluated parameter groups. An RMSD-improvement oracle and a score-aligned axis reference provide distinct analysis-only comparisons: the former uses immediate RMSD reduction, while the latter selects the tested signed axis with greatest agreement with the score target. Direct diagnostic directions were much less aligned with score targets than the score-informed axis reference. Full finite-difference, gradient-alignment, oracle-disagreement, and rotation-window analyses are reported in Appendix C. These probes motivate testing supervision allocation; they do not identify the cause of the trained policies’ outcomes. The next section tests Cartesian consequence and conditional target consistency through torsional loss metrics and allocation masks; the probe directions are not used as training targets.

## 4 Torsion Supervision Priorities and Study Design

Conditional target consistency supplies a second basis for allocating torsional supervision. At noise level $s ,$ let $\mathbb { E } _ { \mathrm { n e a r } }$ and $\mathbb { E } _ { \mathrm { r a n d o m } }$ denote empirical averages over nearest-neighbor and random pairs of scalar torsion score targets for group i. Then

$$
r _ { i } ( s ) = \frac { \mathbb { E } _ { \mathrm { n e a r } } [ | y - y ^ { \prime } | ] } { \mathbb { E } _ { \mathrm { r a n d o m } } [ | y - y ^ { \prime } | ] } , \qquad c _ { i } = \mathbf { 1 } \biggl [ \operatorname* { m a x } _ { s \in \mathcal { S } } r _ { i } ( s ) \leq 0 . 5 \biggr ] .\tag{4}
$$

The absolute differences are taken in score-target space without periodic angle wrapping; low $r _ { i } ( s )$ indicates that nearby represented states provide more similar targets than randomly paired states, and $c _ { i }$ defines the edge-level training mask by requiring this relation across all probed noise levels. An edge is target-consistent when its neighboring targets are at most half as dissimilar as randomly paired targets at every probed noise level; under squared-error supervision, target variation conditional on a represented state contributes to conditional variance, so the measured ratio serves as a proxy for where corrections may be less consistently specified. Construction of $r _ { i } ( s )$ , score-target pairing, noise-level aggregation, threshold selection, and mask generation are detailed in Appendix D. At the training-graph level, the consistency mask retained 61.2% of PDBBind and 49.5% of MOAD torsion edges; count-matched controls retained the same per-graph counts while changing assignments where possible.

<table><tr><td>Policy</td><td>Fixed loss</td><td> $\mathrm { G e n e r a t e d } < 5 \mathring { \mathrm { A } }$ </td><td> $\mathbf { B e s t - o f - 1 0 < 5 \Delta \mathring { A } }$ </td><td> $\mathrm { T o p - } 1 < 5 \mathring { \mathbf { A } }$ </td></tr><tr><td>Starting model</td><td>1.6034</td><td>45.76</td><td>86.87</td><td>34.34</td></tr><tr><td>Broad intrinsic</td><td>1.6546</td><td>47.14</td><td>86.87</td><td>40.07</td></tr><tr><td>Broad Cartesian</td><td>1.6559</td><td>46.84</td><td>85.86</td><td>38.38</td></tr><tr><td>Count-matched intrinsic</td><td>1.6532</td><td>47.10</td><td>87.21</td><td>37.37</td></tr><tr><td>Count-matched Cartesian</td><td>1.6538</td><td>46.90</td><td>85.52</td><td>38.05</td></tr><tr><td>Consistent intrinsic</td><td>1.6519</td><td>46.84</td><td>85.86</td><td>38.38</td></tr><tr><td>Consistent Cartesian</td><td>1.6520</td><td>46.97</td><td>85.86</td><td>37.71</td></tr><tr><td>Broad/focused mixture</td><td>1.6529</td><td>46.94</td><td>86.20</td><td>39.06</td></tr></table>

Table 1: Main post-training outcomes. Docking columns are percentages; fixed score loss is lower-isbetter. The starting model is one checkpoint; trained-policy values are three-seed means.

The two allocation priorities were compared on 403 torsion-edge groups at four noise levels (1,612 time-indexed qualification rows), with Cartesian consequence measured by a static $9 0 °$ cross-pair displacement proxy and high consequence defined by a 1.0 Å threshold. This screening proxy is distinct from the pose-dependent matrix $G _ { \mathrm { r a w } }$ ; loss weighting uses its trace-normalized implementation ${ \widetilde { G } } .$ . The resulting criteria overlapped but produced crossed assignments: 279 high-consequence rows were inconsistent, whereas 645 consistent rows had low Cartesian consequence (Appendix Fig. 1b). The full cross-tabulation is reported in Appendix D. The underlying continuous measures had weak Pearson correlation (0.124). Geometry and target consistency therefore supply distinct allocation information rather than interchangeable rankings. Full threshold, fresh-bank, symmetry, periodicity, and qualification audits are reported in Appendix D

Each policy receives five additional training epochs from the same previously trained DiffDock model. $\mathrm { ~ A ~ 3 ~ } \times \mathrm { ~ 2 ~ }$ design crosses allocation—all torsions, a count-matched subset, or the consistency mask— with the intrinsic angular metric I or trace-normalized Cartesian metric $\widetilde { G }$ derived from Equation 3. All-I is broad continuation, and the all-versus-consistent cells form the core $2 \times 2$ comparison. Countmatched controls preserve subset size while changing assignments where possible. Consistent versus Count-matched tests which torsions are selected; $\widetilde { G }$ versus I tests how their errors are measured; differences between these effects test interaction. Policy definitions are given in Appendix A.

A final reconciliation policy combines broad intrinsic supervision with focused Cartesian supervision:

$$
L _ { \mathrm { m i x } } = { \frac { 1 } { 2 } } L _ { \mathrm { a l l } , I } + { \frac { 1 } { 2 } } L _ { \mathrm { c o n s i s t e n t } , { \widetilde G } } .\tag{5}
$$

The mixture is a separate seventh policy, not a factorial cell. Its exact half-weighted implementation and policy configuration are given in Appendix A, while the mask construction and qualification are detailed in Appendix D. Having defined the competing priorities, the next section traces their consequences from the score objective through pose generation to confidence-based selection.

## 5 Post-Training Results and Implications

Across three training seeds, Cartesian weighting, target-consistency allocation beyond count-matched controls, their factorial combination, and the separate broad/focused mixture showed no consistent advantage over broad continuation on the primary $< 5 \mathring \mathrm { A }$ endpoints (Table 1; Appendix Fig. 5). For the broad/focused mixture minus broad continuation, descriptive 95% paired-complex bootstrap intervals were [−0.61, 0.17] percentage points for generated $< 5 \mathring \mathrm { A }$ success and [−4.38, 2.36] percentage points for Top-1 success. The unaveraged starting checkpoint retained lower fixed score loss than the broad-continuation EMA state, which had higher mean generated-pose and Top-1 success. For broad minus starting, the corresponding descriptive paired-complex bootstrap intervals were [0.17, 2.56] and [−1.68, 13.47] percentage points (Table 9); the Top-1 interval includes zero. Aggregate best-of 10 recovery was unchanged, but equal aggregate recovery does not establish that the same complexes contained successful candidates. This source-to-broad comparison includes checkpoint averaging and does not isolate an effect of continuation. $\mathrm { A t } < 2 \mathring \mathrm { A }$ , the mean Top-1 difference was +0.67 percentage points (Appendix Table 11).

Additional visual summaries and supporting details for the endpoint separation, seed-level policy contrasts, and flexibility changes are provided in Appendices F and G.

Ligand flexibility further separated these effects (Appendix Fig. 5c). For complexes with 1–5 torsions $( n = 6 4 )$ , the broad-continuation EMA state had higher mean generation, best-of-10 recovery, and Top-1 success than the starting checkpoint. For complexes with 6+ torsions $( n = 3 1 )$ , mean generated success and best-of-10 recovery were lower while Top-1 success was higher. Broad continuation had the highest observed mean $< 5 \mathring \mathrm { A }$ Top-1 success among trained policies, while the starting checkpoint retained the lowest fixed score loss and sampling differences varied with flexibility. Intervention activity and paired contrasts are reported in Appendices E–F; stratified values and evaluation-stability checks are reported in Appendices G–H.

## 6 Discussion and Limitations

Cartesian consequence and target consistency motivated focused supervision, but the tested policies did not consistently improve < 5 Å returned-pose success over broad continuation within five epochs. The directional probes show that identifying difficult states need not recover corrective directions; they do not establish the cause of the trained policies’ outcomes. Generation and confidence-based selection remain distinct stages [Corso et al., 2023, 2024]. Top-1 measures returned-pose performance and compatibility with the fixed selector. For the starting and broad-continuation states, that selector performed below the expected uniform-random success rate at both reported thresholds (Appendix G).

The study evaluates whole-pose outcomes without assigning failures to pocket placement or ligand conformation or measuring pairwise diversity among the ten generated poses; these separate analyses remain future work.

In higher-flexibility complexes, the broad EMA state had lower sampling recovery but higher mean Top-1 success than the starting checkpoint [Jing et al., 2022, Zhu et al., 2024]. RMSD and confidence selection do not establish physical validity or interaction recovery [Buttenschoen et al., 2023, Errington et al., 2024]; PoseBusters evaluation and external confirmation remain future work. Alternative allocation rules, adaptive weighting, longer post-training, or joint adaptation of generation and confidence selection could change the result; confidence-based feedback has coupled these stages in docking [Corso et al., 2024]. Evidence from one architecture, 99 complexes, three training seeds, and one sampling seed leaves generalization to other models, datasets, and training budgets unresolved. The design problem is geometric supervision with broad score-field coverage [Song et al., 2021, Corso et al., 2023].

## Acknowledgments and Disclosure of Funding

The authors gratefully acknowledge Niloofar Yousefi for her mentorship and support.

## References

Eric Alcaide, Zhifeng Gao, Guolin Ke, Yaqi Li, Linfeng Zhang, Hang Zheng, and Gengmo Zhou. Uni-Mol Docking V2: Towards realistic and accurate binding pose prediction, 2024. URL https://arxiv.org/abs/2405.11769.

Helen M. Berman, John Westbrook, Zukang Feng, Gary Gilliland, T. N. Bhat, Helge Weissig, Ilya N. Shindyalov, and Philip E. Bourne. The protein data bank. Nucleic Acids Research, 28(1):235–242, 2000. doi: 10.1093/nar/28.1.235. URL https://doi.org/10.1093/nar/28.1.235.

Martin Buttenschoen, Garrett M. Morris, and Charlotte M. Deane. PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences, 2023. URL https://arxiv.org/abs/2308.05777.

Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon. Perception prioritized training of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11472– 11481, 2022. URL https://openaccess.thecvf.com/content/CVPR2022/html/Choi\_ Perception\_Prioritized\_Training\_of\_Diffusion\_Models\_CVPR\_2022\_paper.html.

Gabriele Corso, Hannes Stärk, Bowen Jing, Regina Barzilay, and Tommi Jaakkola. Diffdock: Diffusion steps, twists, and turns for molecular docking. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.01776.

Gabriele Corso, Arthur Deng, Benjamin Fry, Nicholas Polizzi, Regina Barzilay, and Tommi Jaakkola. Deep confident steps to new pockets: Strategies for docking generalization. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2402.18396.

David Errington, Constantin Schneider, Cédric Bouysset, and Frédéric A. Dreyer. Assessing interaction recovery of predicted protein–ligand poses, 2024. URL https://arxiv.org/abs/2409. 20227.

Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffusion training via min-snr weighting strategy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7441–7451, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Hang\_Efficient\_ Diffusion\_Training\_via\_Min-SNR\_Weighting\_Strategy\_ICCV\_2023\_paper.html.

Liegi Hu, Mark L. Benson, Richard D. Smith, Michael G. Lerner, and Heather A. Carlson. Binding MOAD (mother of all databases). Proteins: Structure, Function, and Bioinformatics, 60(3): 333–340, 2005. doi: 10.1002/prot.20512. URL https://doi.org/10.1002/prot.20512.

Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, and Tommi Jaakkola. Torsional diffusion for molecular conformer generation. In Advances in Neural Information Processing Systems, volume 35, 2022. doi: 10.52202/ 068431-1760. URL https://papers.nips.cc/paper\_files/paper/2022/hash/ 994545b2308bbbbc97e3e687ea9e464f-Abstract-Conference.html.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems, volume 35, pages 26565–26577, 2022. doi: 10. 52202/068431-1926. URL https://papers.nips.cc/paper\_files/paper/2022/hash/ a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html.

Sofiene Khiari, Amr H. Mahmoud, and Markus A. Lill. AgenticPosesRanker: An agentic AI framework for physically grounded ranking of protein–ligand docking poses, 2026. URL https: //arxiv.org/abs/2605.03707.

Zhihai Liu, Yan Li, Li Han, Jie Li, Jie Liu, Zhixiong Zhao, Wei Nie, Yuchen Liu, and Renxiao Wang. Pdb-wide collection of binding data: current status of the PDBbind database. Bioinformatics, 31 (3):405–412, 2015. doi: 10.1093/bioinformatics/btu626. URL https://doi.org/10.1093/ bioinformatics/btu626.

Ziqiao Meng, Liang Zeng, Zixing Song, Tingyang Xu, Peilin Zhao, and Irwin King. Towards geometric normalization techniques in SE(3) equivariant graph neural networks for physical dynamics simulations. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pages 5981–5989, 2024. doi: 10.24963/ijcai.2024/661. URL https: //www.ijcai.org/proceedings/2024/661.

Alex Morehead, Nabin Giri, Jian Liu, Pawan Neupane, and Jianlin Cheng. Assessing the potential of deep learning for protein–ligand docking, 2026. URL https://arxiv.org/abs/2405.14108.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8162–8171. PMLR, 2021. URL https://proceedings.mlr.press/v139/nichol21a.html.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. URL https://arxiv.org/abs/2011.13456.

Jintao Zhu, Zhonghui Gu, Jianfeng Pei, and Luhua Lai. DiffBindFR: An SE(3) equivariant network for flexible protein–ligand docking. Chemical Science, 15:7926–7942, 2024. doi: 10.1039/D3SC06803J. URL https://pubs.rsc.org/en/content/articlehtml/2024/sc/ d3sc06803j.

Geometric relevance creates pressure without a unique corrective rule

![](images/94d001bd6f8c424e7baabc153bd6d9cf36dd89c4db624fa1faebd7fcf5c43a6e.jpg)

![](images/3fdd5d51597b12b5a997c2b36896d7a29f3806e7d46cfdfb014d0e29fbcc7b7b.jpg)

![](images/b21130bca8d6945398aa692d795244b85c5683d75b31eb455e86e8cc7ce693e6.jpg)  
Figure 1: (a) Bad-pose rates for all states and the highest tail-energy decile; (b) crossed Cartesianconsequence and target-consistency assignments; and (c) direct diagnostic and analysis-only reference alignment with the score target.

## A Experimental Definitions and Fixed Evaluation

This appendix defines the quantities, procedures, and fixed choices used in the analyses. The starting model is the inference-selected checkpoint from a 200-epoch full DiffDock score-model training run on the combined PDBBind, Binding MOAD, and PDBSidechain training stream. PDBSidechain is a project-derived preprocessing of protein structures from the RCSB Protein Data Bank source snapshot dated 2021-08-02 [Berman et al., 2000]. The starting model uses the six-layer equivariant graph architecture recorded for the run, with 32 scalar and 8 vector channels, second-order sphericalharmonic features, sinusoidal diffusion-time embeddings, and separate translation, rotation, and torsion score outputs. It is evaluated without additional score-model training and serves as the reference for every post-training policy.

## A.1 Policies and training stream

The training stream contains PDBBind, MOAD, and PDBSidechain examples. The allocation rules act on PDBBind and MOAD torsion edges; PDBSidechain remains an all-torsion background in every trained policy. The same translation and rotation supervision is retained across policies. A torsion-edge group is the ordered list of rotatable ligand edges for one graph. If a graph has m such edges, $e \in \mathbb { R } ^ { \bar { m } }$ denotes the vector of sigma-normalized torsion prediction errors for that graph. Let $I _ { m }$ be the $m \times m$ identity matrix. The trace-normalized local Cartesian matrix $\widetilde { G }$ is the training implementation of the pose-dependent matrix $G _ { \mathrm { r a w } }$ derived in Equation 3; its trace is rescaled to m before it is used in a full-graph Cartesian loss. Let $C = \operatorname { d i a g } ( c _ { 1 } , \cdot \cdot \cdot , c _ { m } )$ be the binary conditionalconsistency mask and let $C _ { \pi } = \Pi ^ { \mathsf { T } } C \Pi$ be the fixed within-graph count-matched permutation of that mask. The displayed quadratic forms below identify the error terms used by the policies. Their graph-level reductions divide by the number of supervised torsions, and masked Cartesian forms are trace-normalized after masking.

The starting model and seven trained-policy definitions are summarized in Table 2.

For a policy with allocation A and metric Q, the torsional contribution is represented by $\ell _ { \mathrm { t o r } } ^ { A , Q }$ in Equation 1. The all-torsion intrinsic form is $e ^ { \mathsf { T } } I _ { m } e ;$ the all-torsion Cartesian form is $e ^ { \mathsf { T } } \widetilde { G } e ;$ and the two masked forms are $e ^ { \mathsf { T } } C e$ and $e ^ { \mathsf { T } } C G C e$ , with $C _ { \pi }$ substituted for C in the count-matched policies. The starting model is the unchanged checkpoint against which the seven post-training policies are evaluated. The mixture is a separate policy: it combines the broad intrinsic objective with the consistent Cartesian objective after the factorial cells have been defined.

## A.2 Training and checkpoint rules

Each trained policy uses five post-training epochs, corresponding to 13,000 optimizer steps, from the same starting model. The three training seeds are 20260812, 20260813, and 20260814. An exponential-moving-average (EMA) checkpoint is the running parameter average saved during training; the epoch-5 EMA checkpoint is the prespecified trained model for evaluation, with no later best-epoch selection. All 21 trained policy instances completed five epochs, saw 13,000 training batches, skipped zero training batches, and produced an epoch-5 EMA checkpoint. Within-policy objective changes document learning under each objective, but policy-specific training losses are not used as cross-policy evidence because the allocation and metric definitions change their scale and terms.

<table><tr><td>Policy</td><td>Allocation</td><td>Torsional form or objec- tive</td><td>Role in the comparison</td></tr><tr><td>Starting model</td><td>none</td><td>none</td><td>Unchanged reference model</td></tr><tr><td>Broad intrinsic</td><td>all torsions</td><td>I</td><td>Broad continuation</td></tr><tr><td>Broad Cartesian</td><td>all torsions</td><td>G</td><td>Cartesian weighting without selection</td></tr><tr><td>Count-matched intrin- sic</td><td>permuted subset</td><td> $C _ { \pi }$ </td><td>Generic sparse-allocation control</td></tr><tr><td>Count-matched Carte- sian</td><td>permuted subset</td><td> $C _ { \pi } \widetilde { G } C _ { \pi }$ </td><td>Sparse Cartesian control</td></tr><tr><td>Consistent intrinsic</td><td>consistency mask C</td><td>C</td><td>Target-consistency allocation</td></tr><tr><td>Consistent Cartesian</td><td>consistency mask C</td><td> $C \widetilde G C$ </td><td>Joint allocation and Cartesian weighting</td></tr><tr><td>Broad/focused mix- ture</td><td>broad plus C</td><td> $\scriptstyle { \frac { 1 } { 2 } } L _ { \mathrm { a l l } , I }$   $\scriptstyle { \frac { 1 } { 2 } } L _ { \mathrm { c o n s i s t e n t } , \widetilde { G } }$ </td><td>Reconciliation policy</td></tr></table>

Table 2: Definitions of the starting model and seven trained policies. The starting model is evaluated once; each trained policy is evaluated with three training seeds. The count-matched mask preserves the number of selected torsions within each graph while changing their assignments.

The recorded optimizer configuration uses Adam with learning rate $1 0 ^ { - 4 }$ , batch size 8, one gradient accumulation step, zero weight decay, EMA rate 0.999, four data-loader workers, and deterministic training. No learning-rate scheduler or explicit learning-rate restart is configured.

The starting reference was evaluated using the original unaveraged inference-selected checkpoint. Each trained policy began from those source weights, used a newly initialized Adam optimizer, and was evaluated with the same prespecified epoch-5 EMA convention. Source-to-policy endpoint differences therefore include both parameter updates and checkpoint averaging, while trained-policy comparisons share the same optimizer initialization rule and EMA evaluation convention.

## A.3 Reproducibility and availability

The software environment is specified by the accompanying environment.yml and requirements.txt declarations. The modified DiffDock implementation and analysis scripts used for training, evaluation, and appendix summaries are retained with the project materials, together with derived result summaries, split identifiers, and evaluation manifests. No verified hardware snapshot identifying the GPU model was captured in the experiment records. Raw PDBBind and MOAD data are not redistributed because access is governed by their licenses; the corresponding split identifiers, derived evidence, and provenance manifests are retained.

## A.4 Fixed evaluation

Let A be the set of evaluation complexes, $N = | { \mathcal { A } } | = 9 9$ , and S = 10 the number of generated poses per complex. For complex $i , \widehat { x } _ { i j }$ is generated pose j, x<sup>∗</sup> is the reference pose, $d _ { i j } = \mathrm { R M S D } ( \widehat { x } _ { i j } , x _ { i } ^ { * } )$ and $q _ { i j }$ is the confidence score assigned to that pose. The fixed-confidence Top-1 index is $j _ { i } ^ { * } =$ arg max $q _ { i j }$ . A complex is torsion-eligible when its ligand has at least one rotatable bond; the reported 0, 1–5, and 6+ strata count those bonds. Generated-pose success, best-of-10 recovery, and Top-1 success use the definitions in Equation 2.

The downstream evaluation uses the PDBBind time-split test set containing 99 complexes, ten generated poses per complex, sampling seed 20260820, and a schedule of 20 diffusion steps with 19 actual updates. A separately trained PDBBind confidence model, selected at its recorded epoch-75 checkpoint, ranks candidates for every score-model checkpoint and is not retrained during this comparison. The confidence model was trained separately using a different score-model distribution and was held fixed for the starting checkpoint and all post-training policies. Because the score models generate different candidate sets, a fixed confidence model can still select different poses; Top-1 therefore measures both returned-pose performance and compatibility between the candidate distribution and the selector. Fixed score fidelity is measured on the PDBBind no-ligand-overlap validation split using one shared perturbation bank generated with seed 20260820. A perturbation bank is a fixed collection of corrupted validation states, including their diffusion times, rigid perturbations, torsion perturbations, and score targets. The bank is reused for the starting model and every trained policy.

<table><tr><td>Slice</td><td>n</td><td>Spearman ρ</td><td>Baseline bad</td><td>High-tail bad</td><td>Risk difference</td></tr><tr><td>PDBBind train</td><td>1000</td><td>0.171</td><td>0.555</td><td>0.900</td><td>0.345</td></tr><tr><td>PDBBind validation</td><td>197</td><td>0.253</td><td>0.579</td><td>0.850</td><td>0.271</td></tr><tr><td>PDBBind test, no receptor overlap</td><td>48</td><td>0.164</td><td>0.521</td><td>0.800</td><td>0.279</td></tr><tr><td>MOAD train</td><td>1000</td><td>0.155</td><td>0.607</td><td>0.880</td><td>0.273</td></tr></table>

Table 3: Definition and summary of the $k = 1$ tail-energy association across the PDBBind train, validation, and test slices and MOAD train slice. These are diagnostic summaries, not trained-policy comparisons.

For checkpoint $\phi ,$ the common fixed score fidelity is

$$
L _ { \mathrm { f i x e d } } ( \phi ) = \frac { 1 } { | \mathcal { F } | } \sum _ { \xi \in \mathcal { F } } \ell ( \phi ; \xi ) ,\tag{6}
$$

where $\mathcal { F }$ is the shared set of corrupted validation states. The lower value is closer to the fixed reference score objective. It is not used as a substitute for the generated-pose, best-of-10, or Top-1 endpoints defined in Equation 2.

The evaluation contract prespecified generated-pose and confidence-selected Top-1 success below 5 Å as co-primary endpoints, with $2 \mathring \mathrm { A }$ results retained as a stricter secondary analysis. Generated-pose success averages the fraction of the ten poses below this threshold; best-of-10 recovery records whether at least one pose per complex is below it; and Top-1 success records whether the pose selected by the fixed confidence model is below it. Conditional confidence-selection rate is the fraction of complexes for which Top-1 is below $5 \mathring \mathrm { A }$ among complexes with at least one generated pose below 5 Å. The torsion-count strata and $5 \mathring \mathrm { A }$ conditional-selection analysis are supporting results in Appendix G.

Policy means are averages over the three training seeds. The starting model has one evaluation because it has no training seed in this comparison. Paired-complex bootstrap intervals use 10,000 resamples and are descriptive; the three training seeds are used to assess direction consistency and effect size rather than to claim seed-level statistical significance. These intervals condition on the evaluated checkpoints and sampling seed; they resample complexes, not training or sampling seeds. They therefore do not quantify full uncertainty from retraining or resampling candidate sets. A twopercentage-point absolute docking-rate change with a matched seed direction was the prespecified materiality threshold.

## B Geometric State Diagnosis

For each corrupted state, the smoothed ligand–receptor proximity matrix is denoted by H. Its best rank-k approximation is $H _ { k }$ , and the low-rank tail energy is

$$
T _ { k } = \| H - H _ { k } \| _ { F } ^ { 2 } = \sum _ { r > k } \sigma _ { r } ( H ) ^ { 2 } .
$$

Here $\left\| \cdot \right\| _ { F }$ is the Frobenius norm and $\sigma _ { r } ( H )$ is the rth singular value. The reported $k = 1$ statistic is therefore the proximity-pattern energy left outside the leading component. A high-tail decile is the largest tenth of states ranked by $T _ { 1 }$ . The bad indicator in this appendix is final RMSD at least $5 \mathring \mathrm { A } ;$ the risk difference is the bad-pose rate in the high-tail decile minus the bad-pose rate in the full slice. Spearman correlation measures rank association between $T _ { k }$ and final RMSD.

The slice-level tail-energy associations are summarized in Table 3.

The method-family analysis compares three uses of the same geometric signal: standalone tail-energy ranking, noise-dependent gating, and contextual fusion. A method view is one candidate statistic or transformation evaluated on the same probe rows. Candidate percentiles summarize the relative position of a view within the tested candidate set; tie-aware ranking preserves ties rather than treating them as independent evidence. The 11 contextual-fusion, five sigma-restricted, and two standalone candidates were evaluated as correlated views. Their method-family and weighting summaries are descriptive supporting analyses; representative signal-role values are shown in Table 4.

High Tail Energy Finds Bad Poses  
![](images/f3b1cb60b06d4fd58389c64ae4b0651de6ed60800dda447a34aeb994b6e8a256.jpg)

Figure 2: Tail-energy association with difficult docking states in the 1,000-graph PDBBind training slice. Bars show pose-outcome rates across the full slice and within the highest 10% and 5% of tail energy. This diagnostic relationship does not establish a corrective training rule.
<table><tr><td>Signal use</td><td>PDBBind train</td><td>PDBBind val</td><td>PDBBind test</td><td>MOAD train</td></tr><tr><td>Standalone tail energy</td><td>0.171</td><td>0.253</td><td>0.164</td><td>0.155</td></tr><tr><td>Contextual fusion</td><td>0.238</td><td>0.293</td><td>0.127</td><td>0.233</td></tr><tr><td>Translation-gated fusion</td><td>-0.252</td><td>-0.255</td><td>-0.318</td><td>-0.213</td></tr><tr><td>Rotation-gated fusion</td><td>-0.230</td><td>-0.230</td><td>-0.308</td><td>-0.187</td></tr></table>

Table 4: Representative Spearman associations between signal variants and final RMSD. The two noise-dependent gated variants reverse the positive association observed for standalone tail energy and contextual fusion across the four evaluated slices.

## C Directional and Gradient Diagnostics

The directional probes evaluated one corrupted state after small perturbations along a finite candidatedirection set. For the rotation probes, the 26-direction set contained the signed coordinate axes, face diagonals, and cube diagonals. For candidate direction b, the probe recorded the change in the geometric signal, alignment with the score target and model prediction, and the resulting RMSD. If $d _ { 0 }$ is the RMSD after the unmodified probe and $d _ { b }$ is the RMSD after candidate b, the pure RMSD reference selected

$$
b _ { \mathrm { R M S D } } = \arg \operatorname* { m a x } _ { b } \Delta d _ { b } , \qquad \Delta d _ { b } = d _ { 0 } - d _ { b } .\tag{7}
$$

For vectors u and v, directional alignment is cos $( u , v ) = u ^ { \mathsf { T } } v / ( \| u \| \| v \| )$ when both norms are nonzero. The pure RMSD reference uses the crystal pose only to identify the tested direction with the greatest immediate geometric improvement; it is not a trainable target and is unavailable to the docking model during prediction. A rank-selected direction is the candidate with the largest observed decrease in the diagnostic statistic. The score-aligned axis reference selects among tested signed axes using agreement with the score target; it is the analysis-only reference used for the alignment rows in Table 5 and Panel (c) of Figure 1. A support-constrained oracle retains candidates passing the rank-energy and target-alignment screens, then selects the candidate maximizing the product of diagnostic decrease and nonnegative cosine with the score target. The support-constrained choice was compared with the rank-selected diagnostic direction, while $b _ { \mathrm { R M S D } }$ served as the pure RMSD reference. A free-sign reference is a post-hoc upper-bound comparison that permits the sign of a tested axis to be chosen using the score-target direction.

<table><tr><td>Direction choice</td><td>Component</td><td>n</td><td>Mean cosine</td><td>Positive fraction</td></tr><tr><td>Direct finite difference</td><td>Translation</td><td>1,574</td><td>0.147</td><td>59.6%</td></tr><tr><td>Direct finite difference</td><td>Rotation</td><td>1,548</td><td>-0.035</td><td>46.1%</td></tr><tr><td>Energy-ranked</td><td>Translation</td><td>1,574</td><td>0.118</td><td>59.8%</td></tr><tr><td>Energy-ranked</td><td>Rotation</td><td>1,548</td><td>-0.048</td><td>46.3%</td></tr><tr><td>Score-aligned axis reference</td><td>Translation</td><td>1,574</td><td>0.599</td><td>92.2%</td></tr><tr><td>Score-aligned axis reference</td><td>Rotation</td><td>1,548</td><td>0.496</td><td>84.7%</td></tr><tr><td>Free-sign reference</td><td>Translation</td><td>1,574</td><td>0.835</td><td>100.0%</td></tr><tr><td>Free-sign reference</td><td>Rotation</td><td>1,548</td><td>0.831</td><td>100.0%</td></tr></table>

Table 5: Alignment of tested direction choices with the corresponding score target. The diagnostic choices remain weakly aligned, whereas the score-aligned axis and free-sign rows use analysis-only information and serve as references rather than deployable corrective rules. The pure RMSD reference is defined separately.

![](images/297d1eb48b3893a5bd8ff8522b3f7cbba5d198dead6bdd10ab2fcb1b131fb6db.jpg)  
Figure 3: Directional diagnostic and analysis-only reference comparison. Direct diagnostic directions are compared with the score-aligned axis reference used in the alignment summary. The analysis-only reference is unavailable to the docking model and is used to assess diagnostic-to-action agreement.

The oracle probe skipped the final singleton batch of the 197-graph PDBBind validation slice, leaving 196 graphs. In this probe using the 26-direction set, the rank-selected and support-constrained choices agreed in 2 cases (1.0%), and a support-constrained choice was available in 97 cases (49.5%). Within those available cases, the support-constrained choice produced a positive immediate RMSD change in 56.7% of cases, compared with 51.5% for the rank-selected choice. These values describe local directional probes rather than full docking performance; their role is to show that the geometric diagnostic did not reliably recover a single pose-improving direction. The measured alignment summary is given in Table 5; candidate-level direction records and the additional gradient and rotation-window summaries are retained as supporting audit material.

The gradient table reports cosine alignment between the gradient of a gated geometric objective and the gradient of a score objective. A parameter group is the global model, the translation head, the rotation head, the torsion head, or the shared and remaining parameters. Rows with no valid batches are not interpreted as zero alignment. The rotation-window analysis partitions the single fusion-oracle trajectory into early and late epoch windows; its quantities are descriptive trajectory summaries rather than replicated training results. Representative gradient values are shown in Table 6.

<table><tr><td>Gate cutoff</td><td>Reference score loss</td><td>Global cosine</td><td>Rotation-head cosine</td><td>Batches</td></tr><tr><td>2.0</td><td>Rotation component</td><td>0.026</td><td>-0.064</td><td>20</td></tr><tr><td>2.0</td><td>Full score loss</td><td>-0.044</td><td>-0.064</td><td>20</td></tr><tr><td>3.0</td><td>Rotation component</td><td>0.026</td><td>-0.158</td><td>20</td></tr><tr><td>3.0</td><td>Full score loss</td><td>-0.009</td><td>-0.158</td><td>20</td></tr></table>

Table 6: Representative mean cosine alignments between a gated geometric objective and score losses over 20 valid batches. Near-zero global values and negative rotation-head values show weak or conflicting alignment; rows with no valid batches are omitted.

## D Torsion Priority Qualification

The qualification set contained 403 torsion-edge groups evaluated at four noise levels $s \in$ {0.1, 0.3, 0.6, 0.9}. For edge i, let $\mathcal { U } _ { i }$ be atoms moved by the torsional rotation and $\nu _ { i }$ the atoms left fixed.

## D.1 Derivation of the Cartesian metric

For a point at perpendicular distance $\rho _ { i }$ from a torsional axis, write its position relative to a point on the axis as $u _ { i } + v _ { i }$ , where $u _ { i }$ is parallel to the axis and $v _ { i }$ is perpendicular. A rotation by $\Delta \theta$ leaves $u _ { i }$ unchanged and gives

$$
\lVert x _ { i } ^ { \prime } - x _ { i } \rVert ^ { 2 } = \lVert R ( \Delta \theta ) v _ { i } - v _ { i } \rVert ^ { 2 } = 2 \lVert v _ { i } \rVert ^ { 2 } \big ( 1 - \cos { \Delta \theta } \big ) = 4 \rho _ { i } ^ { 2 } \sin ^ { 2 } \bigg ( \frac { \Delta \theta } { 2 } \bigg ) .
$$

For small $\Delta \theta$ , this is $\rho _ { i } ^ { 2 } ( \Delta \theta ) ^ { 2 } + O ( ( \Delta \theta ) ^ { 4 } )$ , which gives the first relation in Equation 3. For simultaneous small changes $\Delta \theta$ , the Cartesian displacement satisfies

$$
\Delta { \bf x } = J \Delta \pmb { \theta } + O \big ( \| \Delta \pmb { \theta } \| ^ { 2 } \big ) , \qquad \mathrm { R M S D } ^ { 2 } = \frac { 1 } { n _ { \mathrm { a t o m } } } \| P \Delta { \bf x } \| ^ { 2 } ,
$$

where $J$ is the Cartesian Jacobian with one torsional column per edge and $P$ is the symmetric idempotent projector that removes local rigid translation and rotation components. Substitution and $P ^ { \mathsf { T } } \bar { P _ { \mathrm { } } } = P$ yield

$$
\mathrm { R M S D } ^ { 2 } \approx \frac { 1 } { n _ { \mathrm { a t o m } } } \Delta \boldsymbol { \theta } ^ { \top } \boldsymbol { J } ^ { \top } \boldsymbol { P } \boldsymbol { J } \Delta \boldsymbol { \theta } = \Delta \boldsymbol { \theta } ^ { \top } \boldsymbol { G } _ { \mathrm { r a w } } ( \boldsymbol { \theta } ) \Delta \boldsymbol { \theta } .
$$

The diagonal entries of $G _ { \mathrm { r a w } }$ encode individual torsional consequence, while off-diagonal entries encode coupled Cartesian displacement. The implementation trace-normalizes this matrix to form $\widetilde { G }$ before applying the graph-level training loss, as defined in Appendix $\operatorname { A } ;$ the qualification calculation uses the unnormalized geometric form above. Finite-difference validation over 13,248 direction tests gave a 90th-percentile relative error of $3 . 2 \times 1 0 ^ { - 4 }$ for the analytic local form and $3 . 6 \times 1 0 ^ { - 3 }$ for the full local approximation. The off-diagonal Frobenius contribution reached 0.70 at the 90th percentile, so the coupled terms are retained rather than replaced by diagonal weights.

The static cross-pair displacement proxy rotates $\mathcal { U } _ { i }$ by $\pi / 2$ around the edge axis and computes

$$
d _ { \mathrm { c r o s s } } ( i ) = \left[ \frac { 1 } { | \mathcal { U } _ { i } | | \mathcal { V } _ { i } | } \sum _ { u \in \mathcal { U } _ { i } } \sum _ { v \in \mathcal { V } _ { i } } \left( \| x _ { u } ^ { \prime } - x _ { v } ^ { \prime } \| - \| x _ { u } - x _ { v } \| \right) ^ { 2 } \right] ^ { 1 / 2 } .
$$

High Cartesian consequence means $d _ { \mathrm { c r o s s } } ( i ) \geq 1 . 0 \mathring \mathrm { A }$ . This static screen is used only for the qualification cross-tabulation; it is distinct from the pose-dependent matrix $G _ { \mathrm { r a w } } ;$ loss weighting uses $\widetilde { G } .$

For the consistency calculation, the numerator in Equation 4 is the mean absolute difference between each score target and its nearest-neighbor target; the denominator is the mean absolute difference over randomly paired score targets. Periodic angle wrapping is used for angular perturbation handling elsewhere, but not for this score-target ratio. A row is consistent when $r _ { i } ( s ) \leq 0 . 5 $ ; the edge-level training mask $c _ { i }$ is one only when this holds at every probed noise level. Table 7 counts time-indexed (i, s) rows, not final edge-level mask values.

<table><tr><td></td><td>Consistent row</td><td>Inconsistent row</td></tr><tr><td>High Cartesian consequence</td><td>349</td><td>279</td></tr><tr><td>Low Cartesian consequence</td><td>645</td><td>339</td></tr></table>

Table 7: Crossed supervision priorities among 1,612 time-indexed qualification observations from 403 torsion-edge groups and four noise levels. Rows use the static $9 0 °$ cross-pair screen and the score-target consistency threshold defined in the text.

<table><tr><td>Source</td><td>Processed</td><td>Multi-torsion</td><td>Metric states</td><td>Direction tests</td><td>Graph failures</td></tr><tr><td>PDBBind</td><td>192</td><td>173</td><td>1656</td><td>13248</td><td>0</td></tr></table>

Table 8: Coverage and finite-difference checks for the local Cartesian metric on the PDBBind qualification set. These checks qualify the metric approximation and do not establish downstream training benefit.

The off-diagonal cells show the crossed assignments between Cartesian consequence and target consistency. The continuous measures have Pearson correlation 0.124. Fresh-bank prediction uses k nearest represented states or a local-5 estimator to predict torsional targets; lower prediction error indicates better agreement with the observed target. Symmetry strata use the fraction of duplicate atoms on the rotated side, with low symmetry at most 0.25 and elevated symmetry above 0.25. Periodic-wrapping diagnostics compare raw and wrapped angular updates. The allocation audit reports the selected edge fraction as selected edges divided by all torsion edges, while lookup failure means that a runtime graph could not be matched to its fixed allocation key.

The metric coverage and finite-difference checks are summarized in Table 8.

At the qualification-set level, the 90-degree screen yielded the cross-tabulation above. At the full training-graph level, the locked 0.5 consistency threshold selected 61.2% of PDBBind torsion edges and 49.5% of MOAD torsion edges, with zero allocation lookup failures. On the independent perturbation bank, label agreement was 85.3% for PDBBind and 90.8% for MOAD; Cohen’s kappa was 0.692 and 0.814, respectively. These quantities qualify the allocation rule and do not establish that it improves docking.

## E Intervention Activity

An intervention is active when it changes the torsional loss term or allocation while the run completes the specified training exposure. Within-policy objective response is the change from the first to the last logged training epoch for the same objective; it is not a cross-policy ranking. For a parameter group $^ { g , }$ let $\Delta \phi _ { g } ^ { ( a ) }$ and $\Delta \phi _ { g } ^ { ( b ) }$ be flattened checkpoint changes from the common starting model. Update similarity is

$$
\cos _ { g } ( a , b ) = \frac { \left( \Delta \phi _ { g } ^ { ( a ) } \right) ^ { \top } \Delta \phi _ { g } ^ { ( b ) } } { \| } \Delta \phi _ { g } ^ { ( a ) } \| \| \Delta \phi _ { g } ^ { ( b ) } \| .
$$

The training-integrity records show five epochs, 13,000 batches, zero skipped batches, and an epoch-5 EMA checkpoint for every trained instance. Every policy’s own training objective declined over its logged five-epoch trajectory. The metric ablation provides an independent activity check: aligned-versus-permuted update cosine was approximately 0.35 in the torsion head, compared with approximately 0.99 in the translation and rotation heads. Thus the metric change redirected the torsional part of the update even when the non-torsional heads changed similarly.

## F Full Matched Post-Training Study

The matched-study results contain one starting checkpoint and seven trained policies. A policy mean is the arithmetic mean across its three training seeds; the accompanying standard deviation is the sample standard deviation across those seeds. The starting row has no training-seed standard deviation because it is one fixed checkpoint. A paired seed contrast is policy A minus policy B for the same training seed. A paired-complex contrast uses the same generated complex and subtracts the two endpoint indicators before bootstrapping across complexes.

<table><tr><td>Contrast</td><td colspan="2">Generated &lt; 5 Å</td><td colspan="2"> $\mathrm { B e s t - o f } { \mathrm { - } } 1 0 < 5 \mathring { \mathrm { A } }$ </td><td colspan="2"> $\mathrm { T o p - } 1 < 5 \mathring { \mathbf { A } }$ </td></tr><tr><td>Broad — starting</td><td></td><td> $+ 1 . 3 8 \ [ 0 . 1 7 , \ 2 . 5 6 ]$ </td><td></td><td> $+ 0 . 0 0 \left[ - 4 . 3 8 , \ 4 . 3 8 \right]$ </td><td></td><td> $+ 5 . 7 2 [ - 1 . 6 8 , 1 3 . 4 7 ]$ </td></tr><tr><td> $\mathbf { A l l - G } - \mathbf { A l l - I }$ </td><td></td><td> $- 0 . 3 0 \ [ \dot { - } 0 . 7 1 , \ 0 . 1 \dot { 0 } ]$ </td><td></td><td> $- 1 . 0 1 \ \dot { [ - 2 . 6 9 , 0 . 3 4 ] }$ </td><td></td><td> $- 1 . 6 8 \mathsf { \bar { \Phi } } [ - 5 . 3 9 , \ 1 . 6 8 ]$ </td></tr><tr><td> $\mathbf { C o u n t - G - C o u n t - I }$ </td><td></td><td>-0.20 [−0.61, 0.17]</td><td></td><td> $- 1 . 6 8 [ \dot { - } 3 . 7 0 , - 0 . 3 \dot { 4 } ]$ </td><td></td><td> $+ 0 . 6 7 \ : [ - 3 . 0 3 , \ : 4 . 0 4 ]$ </td></tr><tr><td>Consistent-I — Count-I</td><td></td><td> $- 0 . 2 7 \left[ - 0 . 6 1 , 0 . 0 3 \right]$ </td><td></td><td> $- 1 . 3 5 \ : [ - 2 . 6 9 , \ : - 0 . 3 4 ]$ </td><td></td><td> $+ 1 . 0 1 \left\{ - 3 . 0 3 , 5 . 0 5 \right\}$ </td></tr><tr><td>Consistent-G — Count-G</td><td></td><td>+0.07 [-0.40, 0.54]</td><td></td><td> $+ 0 . 3 \dot { 4 } \left[ - 0 . 6 7 , 1 . 6 8 \right]$ </td><td></td><td> $- 0 . 3 4 \left[ - 3 . 0 3 , 2 . 3 6 \right]$ </td></tr><tr><td>Consistent-G — Consistent-I</td><td></td><td> $+ 0 . 1 3 \left. - 0 . 2 4 , 0 . 5 1 \right.$ </td><td></td><td> $+ 0 . 0 0 \left[ - 1 . 0 1 , 1 . 0 1 \right]$ </td><td></td><td> $- 0 . 6 7 \left[ - 4 . 0 4 , 2 . 6 9 \right]$ </td></tr><tr><td>Mixture — All-I</td><td></td><td> $- 0 . 2 0 \left[ - 0 . 6 1 , \ : 0 . 1 7 \right]$ </td><td></td><td> $- 0 . 6 7 \left[ - 1 . 6 8 , 0 . 0 0 \right]$ </td><td></td><td> $- 1 . 0 1 \ \dot { [ - 4 . 3 8 , \ 2 . 3 6 ] }$ </td></tr></table>

Table 9: Descriptive 95% paired-complex percentile bootstrap intervals for the predeclared allcomplex contrasts. Entries are percentage-point differences, with the first policy minus the second; intervals use 10,000 resamples and are not seed-level significance intervals.

<table><tr><td>Policy</td><td>Stratum</td><td>n</td><td>Generated</td><td>Best-of-10</td><td>Top-1</td></tr><tr><td>Starting model</td><td>all</td><td>99</td><td>45.76</td><td>86.87</td><td>34.34</td></tr><tr><td>Broad intrinsic</td><td>all</td><td>99</td><td>47.14</td><td>86.87</td><td>40.07</td></tr><tr><td>Mixture</td><td>all</td><td>99</td><td>46.94</td><td>86.20</td><td>39.06</td></tr><tr><td>Starting model</td><td>1–5 torsions</td><td>64</td><td>53.91</td><td>90.63</td><td>42.19</td></tr><tr><td>Broad intrinsic</td><td>1–5 torsions</td><td>64</td><td>56.35</td><td>92.19</td><td>47.92</td></tr><tr><td>Mixture</td><td>1–5 torsions</td><td>64</td><td>56.20</td><td>92.19</td><td>50.00</td></tr><tr><td>Starting model</td><td>6+ torsions</td><td>31</td><td>31.94</td><td>80.65</td><td>22.58</td></tr><tr><td>Broad intrinsic</td><td>6+ torsions</td><td>31</td><td>30.22</td><td>75.27</td><td>26.88</td></tr><tr><td>Mixture</td><td>6+ torsions</td><td>31</td><td>29.89</td><td>73.12</td><td>20.43</td></tr></table>

Table 10: Flexibility-stratified percentages on the 99-complex PDBBind test evaluation at the 5 Å threshold. Values use ten generated poses per complex; the four zero-torsion complexes are retained in the all-complex result but are not interpreted as a subgroup.

The primary practical contrast is the broad/focused mixture minus broad intrinsic continuation. Its generated $< 5 \mathring \mathrm { A }$ differences by seed were −0.30, 0.00, and −0.30 percentage points; its Top-1 differences were −4.04, +2.02, and −1.01 points. The primary reference contrast is broad intrinsic continuation minus the starting model: generated success changed by +0.91, +2.32, and +0.91 points, while Top-1 changed by +10.10, +5.05, and +2.02 points. The same paired arithmetic was applied to every predeclared threshold, endpoint, stratum, and policy pair, with 10,000-resample descriptive bootstrap intervals.

Relative to the starting checkpoint, the broad-intrinsic mean fixed loss increased by 0.0512, comprising +0.0292 in translation loss, +0.0224 in rotation loss, and −0.0004 in torsion loss. The records therefore do not identify the increase as a torsion-specific effect or attribute it to the learning rate, optimizer state, schedule, or checkpoint averaging.

The integrated endpoint and seed-matched outcome summaries are shown in Figures 4 and 5.

## G Sampling, Confidence Selection, and Flexibility

The flexibility strata are defined by the number of rotatable ligand bonds: zero, 1–5, and 6+; torsioneligible is the union of the latter two strata. Generated success is averaged over poses, best-of-10 is averaged over complexes, and Top-1 is averaged over confidence-selected complexes, as defined in Equation 2. All threshold variants use the same generated poses and confidence scores; only the RMSD indicator threshold changes.

The $5 \mathring \mathrm { A }$ flexibility-stratified endpoints are summarized in Table 10.

Supporting $2 \mathring \mathrm { A }$ endpoints for all policies are listed in Table 11.

For each complex with at least one generated pose below 5 Å, conditional confidence-selection rate records whether the confidence-selected pose is also below 5 Å. It measures selection among recoverable complexes, not the probability of generating a recoverable pose. In the high-flexibility stratum, broad continuation reduced best-of-10 recovery by 5.38 points relative to the starting model while increasing Top-1 by 4.30 points; the mixture was 6.45 points below broad continuation in Top-1.

![](images/f984ce1b4f00c601741089afe5288cfa2bd461473a1abdde5e5d18a62599512c.jpg)  
Figure 4: Integrated summary of the post-training endpoint separation. (a) The starting model and broad intrinsic continuation differ in fixed score loss, aggregate best-of-10 recovery, and confidenceselected Top-1 success; the starting model is one checkpoint and the broad values are three-seed means. (b) Seed-matched changes for the factorial policies are shown relative to broad intrinsic continuation; each dot is one training seed, and the broad/focused mixture is shown separately because it is not a factorial cell. (c) Source-to-broad changes are shown by torsional flexibility stratum. The figure re-expresses the quantities reported in the matched-study and flexibility summaries.

![](images/673e0199bf776834f6904122bb6794d2b1b338d36473d20906fc96a802a9760b.jpg)

![](images/9ba0a0b42651955c669f24af89ddce8841da58c544917211b2c84f70bdc4b824.jpg)

![](images/62430c2433c65224ae00409509888beb6c32e93b5bf7eb35d746b13fe8bb7233.jpg)  
Figure 5: Matched post-training outcomes. (a) Fixed score loss; (b) aggregate generated-pose, best-of-10, and confidence-selected Top-1 success; and (c) the same docking endpoints for the 1–5 and 6+ torsion strata. Bars show the starting model and three-seed policy means; error bars show seed standard deviations where available. The common PDBBind evaluation uses ten poses per complex and a 5 Å threshold.

For a uniform random choice among the $S$ generated poses, the expected success for complex i is

$$
\mathbb { E } _ { J \sim \mathrm { U n i f o r m } \{ 1 , \dots , S \} } { \bf 1 } [ d _ { i J } < a ] = \frac { 1 } { S } \sum _ { j = 1 } ^ { S } { \bf 1 } [ d _ { i j } < a ] .\tag{8}
$$

<table><tr><td>Policy</td><td>Generated &lt; 2 Å</td><td> $\mathrm { B e s t - o f - } 1 0 < 2 \mathring { \mathrm { A } }$ </td><td> $\mathrm { T o p - } 1 < 2 \mathring { \mathrm { A } }$ </td></tr><tr><td>Starting model</td><td>17.98</td><td>49.49</td><td>14.14</td></tr><tr><td>Broad intrinsic</td><td>18.99</td><td>48.82</td><td>14.81</td></tr><tr><td>Broad Cartesian</td><td>19.12</td><td>49.16</td><td>15.15</td></tr><tr><td>Count-matched intrin- sic</td><td>18.89</td><td>49.16</td><td>14.14</td></tr><tr><td>Count-matched Carte-</td><td>19.02</td><td>49.49</td><td>15.49</td></tr><tr><td>sian Consistent intrinsic</td><td>18.92</td><td>49.49</td><td>15.15</td></tr><tr><td>Consistent Cartesian</td><td>18.96</td><td>49.49</td><td>14.81</td></tr><tr><td>Broad/focused mix- ture</td><td>19.16</td><td>49.49</td><td>14.81</td></tr></table>

Table 11: Supporting 2 Å docking endpoints on the 99-complex PDBBind test evaluation. Values are percentages; trained-policy values are three-seed means. The 5 Å threshold remains primary.

<table><tr><td>Policy</td><td>Threshold</td><td>Random reference</td><td>Confidence Top-1</td><td>Top-1 minus reference</td></tr><tr><td>Starting model</td><td>2Å</td><td>17.98</td><td>14.14</td><td>-3.84</td></tr><tr><td>Broad intrinsic</td><td>2Å</td><td>18.99</td><td>14.81</td><td>-4.18</td></tr><tr><td>Starting model</td><td>5Å</td><td>45.76</td><td>34.34</td><td>-11.42</td></tr><tr><td>Broad intrinsic</td><td>5Å</td><td>47.14</td><td>40.07</td><td>-7.07</td></tr></table>

Table 12: Confidence-selected success compared with the expected success from uniformly selecting one of ten generated poses. Values are percentages; the random reference equals generated-pose success.

Averaging over complexes gives exactly $S _ { \mathrm { g e n } } ( a )$ , so every generated-pose value is also the corresponding uniform-random reference. The resulting source-versus-broad confidence-selection gaps at both reported thresholds are shown in Table 12.

Original DiffDock reported 35.0% Top-1 success below 2 Å with ten generated poses and 61.7% below 5 Å on its 363-complex PDBBind test set [Corso et al., 2023]. The present starting model reached 14.14% and 34.34%, respectively, while broad continuation reached 14.81% and 40.07%. These values are not a direct benchmark because the test-set size, score-model training run, model capacity, and evaluation setup differ.

## H Robustness and Evaluation Stability

An evaluation outlier is a generated pose with RMSD greater than 1,000 Å. Across the 22 evaluated checkpoints, 63 of 21,780 poses exceeded this value and occurred in eight complexes. The completecase sensitivity analysis removes every complex associated with an outlier from every policy before recomputing the paired contrasts. After this removal, the mixture-minus-broad differences were −0.26 points for generated $< 5 \mathring \mathrm { A }$ and −1.10 points for Top-1. The numerical failures therefore do not account for the primary policy conclusion.

Training-time monitoring validation resamples diffusion time and pose perturbations when the validation examples are transformed. A validation bank replay holds the checkpoint fixed and compares two independently sampled collections of corrupted states. In the replay, the mean score loss changed from 1.690 to 1.508, the median from 1.318 to 1.166, and the mean diffusion time changed from 0.475 to 0.519. This is a bank-difficulty comparison, not evidence of parameter improvement. The fixed validation bank used for the final policy comparison is separate and common to all checkpoints.

The isolated convolution replay examines one complex under a normal and a failing corrupted pose. A layer value is the maximum absolute activation recorded at that module. The failing pose increased the pose-readout maximum from 2.65 to $5 . 1 8 \times 1 0 ^ { 1 7 }$ , with progressive amplification through the convolution stack; the source checkpoint also failed on that perturbation. This is an inherited numerical limitation and is not assigned as the cause of the aggregate RMSD result.