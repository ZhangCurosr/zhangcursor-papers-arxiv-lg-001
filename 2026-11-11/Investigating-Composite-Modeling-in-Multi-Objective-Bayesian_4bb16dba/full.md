# Investigating Composite Modeling in Multi-Objective Bayesian Optimization for Scientific Design

Ricky Ye<sup>1∗</sup> Sahas Goli<sup>1∗</sup> Muhammad Ali<sup>1∗</sup> Poompol Buathong<sup>2</sup> Peter I. Frazier<sup>3</sup>

<sup>1</sup>Department of Computer Science, Cornell University <sup>2</sup>Center for Applied Mathematics, Cornell University <sup>3</sup>School of Operations Research and Information Engineering, Cornell University {rjy42,sg2636,ma2379,pb482,pf98}@cornell.edu

## Abstract

Scientific design often requires balancing competing objectives that are expensive to evaluate through experiments or simulations. Multi-objective Bayesian optimization (MOBO) addresses this challenge by using probabilistic models to guide the selection of promising designs. In many applications, tunable inputs map to observable intermediate responses, which are then combined through known functions to compute the final objectives. Prior work on Bayesian optimization of composite functions shows that learning from these intermediate responses, which we call Composite modeling, can substantially improve sampling efficiency in single-objective optimization, compared to methods that condition only on blackbox observations of the objective function. The benefits of Composite modeling in MOBO, however, remain understudied. In this work, we study the benefits of Composite modeling in MOBO and investigate the factors driving variation in its gains. We first establish single-objective conditions under which intermediate observations provide no additional predictive information beyond objective values, where composite and black box methods use the same warped Gaussian process prior. Controlled numerical experiments with versions of four MOBO methods across eight synthetic and application-inspired scientific-design benchmarks examine how these conditions relate to optimization gains in the multi-objective setting, where the black box approach models the objective functions with a standard multioutput Gaussian process. Both investigations support the claim that composite modeling tends to help more when the function mapping intermediate responses to final objectives is non-bijective and non-affine.

## 1 Introduction

Scientific design often involves balancing competing goals. A material, for example, may need to perform well while also being inexpensive. Evaluating a design can require a costly experiment. We formulate this problem as mi $\mathfrak { n } _ { \mathbf { x } \in \mathcal { X } } \mathbf { f } ( \mathbf { x } ) = ( f _ { 1 } ( \mathbf { x } ) , \ldots , f _ { N } ( \mathbf { x } ) )$ , where $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ is the design space. Multi-objective Bayesian optimization (MOBO) seeks favorable tradeoffs with few evaluations [1, 2], using Gaussian process (GP) models [3] and an acquisition function to select designs.

In many scientific applications, tunable inputs map to observable intermediate responses, which are then combined through known functions to compute the final objectives. These objectives have the composite form $f _ { i } ( \mathbf { x } ) = g _ { i } \big ( \mathbf { h } _ { i } ( \mathbf { x } ) \big )$ for $i = 1 , \ldots , N$ where $\mathbf { h } _ { i } : \mathcal { X }  \mathbb { R } ^ { k _ { i } }$ is an expensive black-box function returning $k _ { i }$ intermediate responses and $g _ { i } : \mathbb { R } ^ { k _ { i } } $ R is known and inexpensive to evaluate. For single objectives, we write $f , g$ and h in place of $f _ { 1 } , g _ { 1 }$ and $\mathbf { h } _ { 1 }$ . For example, in color formulation, dye concentrations determine measured color coordinates, which known functions convert into color differences from a target [4]. Matching under different lighting can create competing objectives [5].

Bayesian optimization (BO) of composite functions [6] models intermediate responses and propagates posterior samples through the known mappings. This can substantially improve single-objective sampling efficiency, with extensions to high-dimensional outputs and function networks [7, 8, 9, 10]. Composite modeling has also been applied to MOBO through smooth Tchebycheff scalarization (STCH) [11, 12], but the factors governing its benefits remain understudied.

We investigate the benefits of composite modeling through three inference approaches. Composite places a $\mathrm { G P }$ prior on the intermediate responses and conditions on their observations. BlackBox-GP places a GP prior directly on the objectives and conditions on objective observations. BlackBox-Induced uses the objective prior induced by Composite’s prior through the known mappings, but conditions only on objective observations. BlackBox-GP corresponds directly to how BO is usually implemented in practice while BlackBox-Induced supports a precise comparison of the benefits provided by the additional information provided by intermediate responses while holding the prior distribution fixed.

We find evidence that composite modeling tends to perform as well or better than black box approaches and provides the largest benefits when the functions $g _ { i }$ are non-bijective and non-affine. First, we analyze the single-objective case and show that Composite and BlackBox-Induced create equivalent posterior distributions when $g$ is bijective or when $g$ is affine and the prior on h has separable covariance. This suggests composite approaches are equal to black box ones in these settings. Second, experiments comparing Composite with BlackBox-GP on both single and multi-objective problems show that Composite provides a positive mean final-hypervolume gain in 15 of 16 comparisons and that the problems where Composite provides the biggest benefits are non-affine and non-bijective $g _ { i }$

## 2 Problem Setting

Consider objectives $f _ { i } ( \mathbf { x } ) = g _ { i } ( \mathbf { h } _ { i } ( \mathbf { x } ) ) , i = 1 , \ldots , N$ , where $\mathbf { h } _ { i } : \mathcal { X }  \mathbb { R } ^ { k _ { i } }$ is unknown and $g _ { i } : \mathbb { R } ^ { k _ { i } } $ R is known. The intermediate responses may be shared across objectives. We consider both single-objective $( N = 1 )$ and multi-objective $( N > 1 )$ settings. For the single-objective analysis, we omit the subscript and write $f ( \mathbf { x } ) = g ( \mathbf { h } ( \mathbf { x } ) )$ .

We consider three ways of performing inference in this setting, one that conditions on observations of $\mathbf { h } ( \mathbf { x } )$ and two that condition only on observations of $f ( \mathbf { x } )$ . The first method, called Composite, aligns with composite BO [6]: we put a multi-output $\mathrm { G P }$ on $\mathbf { h } ( \mathbf { x } )$ and condition on observations of $\mathbf { h } ( \mathbf { x } )$ The second method, called $B l a c k B o x – G P$ , aligns with classical BO [13, 14]: we put a GP on $f ( \mathbf { x } )$ and condition on observations of $f ( \mathbf { x } )$ . This treats $f ( \mathbf { x } )$ as a black box, supposing that we cannot observe the inner function $\mathbf { h } ( \mathbf { x } )$ . The third method, called BlackBox-Induced, supports theoretical analysis: we start with the same $\mathrm { G P }$ prior on $\mathbf { h } ( \mathbf { x } )$ as Composite, induce a corresponding prior on $f ( \mathbf { x } )$ through the known mapping $^ { g , }$ and condition on observations of $f ( \mathbf { x } )$ . Thus, this method uses the same prior as Composite but conditions on the same black box information as BlackBox-GP. This is a form of warped GP prior [15] with a fixed warping function. Appendix A describes how to perform posterior inference under this induced prior.

Below, using theory and experiments, we analyze whether observing $\mathbf { h } ( \mathbf { x } )$ (in the Composite method) provides a benefit over observing only $f ( \mathbf { x } )$ (the two BlackBox methods) and how this depends on the choice of $g .$ We focus specifically on whether $g$ is a bijection or is affine, finding support for the claim that conditioning on $\mathbf { h } ( \mathbf { x } )$ does not provide a benefit when $g$ is a bijection or is affine, and often does provide a benefit when g is neither bijective nor affine.

Table 1: Controlled synthetic experiments. Final regret reductions (single-objective) and final hypervolume gains (multi-objective) for Composite relative to BlackBox-GP, both in percentage (%) (mean ± SE over 20 paired runs). Positive values favor Composite. Bold marks unadjusted $p < 0 . 0 5$ (one-sided paired t-tests for improvement).
<table><tr><td rowspan="2">Outer mapping g</td><td rowspan="2">Bijective?</td><td colspan="2">Single objective</td><td colspan="2">Multi-objective</td></tr><tr><td>Final-RG reduction (%)</td><td> $p$ </td><td>Final-HV gain (%)</td><td> $p$ </td></tr><tr><td>Identity</td><td>Yes</td><td> $+ 1 8 . 1 \pm 2 9 . 2$ </td><td>0.29</td><td> $+ 0 . 2 \pm 0 . 9$ </td><td>0.40</td></tr><tr><td>Cubic  $\mathbf { \check { \boldsymbol { h } } } ^ { 3 }$ </td><td>Yes</td><td> $+ 1 7 . 1 \pm 2 1 . 1$ </td><td>0.24</td><td> $+ 0 . 9 \pm 1 . 6$ </td><td>0.29</td></tr><tr><td>Square  $\overline { { { h ^ { 2 } } } }$ </td><td>No</td><td> $\overline { { { \bf \nabla } + { \bf 9 9 . 8 } \pm { \bf 0 . 1 } } }$ </td><td>&lt; 0.01</td><td> $\mathbf { \sigma } + \mathbf { 2 . 9 \pm 1 . 0 }$ </td><td>&lt; 0.01</td></tr><tr><td>Shifted square  $( h - 3 ) ^ { 2 }$ </td><td>No</td><td> $\mathbf { + 9 9 . 6 \pm 0 . 5 }$ </td><td>&lt; 0.01</td><td> $+ 1 . 0 \pm 0 . 8$ </td><td>0.12</td></tr><tr><td>Three-root cubic</td><td>No</td><td> $\mathbf { + 4 5 . 3 \pm 1 6 . 4 }$ </td><td>0.03</td><td> ${ \bf + 2 . 5 \pm 1 . 2 }$ </td><td>0.03</td></tr><tr><td>Cosine cos h</td><td>No</td><td> $\mathbf { + 9 6 . 3 \pm 4 . 1 }$ </td><td>&lt; 0.01</td><td> $- 1 . 5 \pm 0 . 9$ </td><td>0.95</td></tr></table>

## 3 Theoretical Analysis

We first compare Composite and BlackBox-Induced in a single-objective case theoretically.

Proposition 1 (Posterior equivalence). At fixed evaluation locations $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n }$ , let $\cal H \_ =$ $( \mathbf { h } ( \mathbf { x } _ { 1 } ) , \ldots , \mathbf { h } ( \mathbf { x } _ { n } ) )$ , and $F = ( f ( \mathbf { x } _ { 1 } ) , \dots , f ( \mathbf { x } _ { n } ) )$ . Suppose h is drawn from a GP prior with fixed known mean function and kernel. Suppose also that either $( I ) g$ is bijective on the possible intermediate responses, so each objective value uniquely determines its intermediate response; or (2) $g ( \mathbf { z } ) = \mathbf { a } ^ { \intercal } \mathbf { z } + b$ for some fixed a and b, and the GP prior on h has separable covariance. Then, for everyfixed $\mathbf { x } \in \mathcal { X }$ and $t \in \mathbb { R } , \mathbb { P } ( f ( \mathbf { x } ) \leq t \mid H ) = \mathbb { P } ( f ( \mathbf { x } ) \leq t \mid F )$ with probability one.

Technical conditions and the proof are provided in Appendix B. Under the Proposition 1, which applies to Identity and Cubic experiments below, Composite and BlackBox-Induced yield the same posterior over f when they share a GP prior on h. Consequently, acquisition rules determined solely by the posterior produce the same decisions and final posterior distribution. In these problems, differences observed between Composite and BlackBox-GP are due to the prior distribution, Monte Carlo noise, or small implementation differences, since BlackBox-GP and BlackBox-Induced condition on the same information.

## 4 Experimental Analysis

Controlled Synthetic Experiments: We fix scalar intermediate functions $h : [ 0 , 1 ] ^ { 6 } \to$ R and vary the known outer mapping g (See Appendix C for h construction). We compare BlackBox-GP and Composite methods, each of which fits its own $\mathrm { G P }$ models and share the initial design, acquisition rule, and evaluation budget. Single-objective experiments use LogEI [13, 16] and we report mean percentage of final $\log _ { 1 0 }$ simple regret (Final-RG) reduction for Composite relative to BlackBox-GP with standard error (SE). For multi-objective, we use $f _ { 1 } ( { \bf x } ) = g ( h _ { 1 } ( { \bf x } ) )$ , and $f _ { 2 } ( \mathbf { x } ) = h _ { 2 } ( \mathbf { x } )$ where $h _ { 1 }$ and $h _ { 2 }$ are independent scalar intermediates, modeled with independent GPs. The $f _ { 2 }$ identity mapping isolates composition in $f _ { 1 }$ . We use LogEHVI [17, 16] and report mean percentage of final hypervolume (Final-HV) gain against a shared reference point for Composite relative to BlackBox-GP with SE. Higher metrics favor Composite. Single-objective starts with 5 initial points and two-objective starts with 10; both then sequentially evaluate 40 additional points based on their acquisition rules. Appendix D details comparison metrics. Four function draws and five initial designs per draw yield 20 paired runs.

Table 1 shows significant single-objective gains for all four selected non-bijective mappings, while for multi-objective cases, only square and three-root cubic yield significant hypervolume gains. Neither selected bijective mapping shows a significant gain in either setting. Square gives the largest gain of the six in both settings, but the two scales are not comparable: its single-objective gain is a 99.8% reduction in final regret, against a hypervolume gain under 3%. Hypervolume optimization requires discovering a set of trade-offs rather than a single optimum, which may limit Composite’s gains under a fixed evaluation budget. Larger-budget experiments could test this explanation.

Table 2: Scientific-design experiments. Final hypervolume gains (%, mean ± SE) for Composite relative to BlackBox-GP over 10 paired trials per benchmark (80 for Synthetic 1). Positive values favor Composite. Bold marks unadjusted $p < 0 . 0 5$ (one-sided paired t-tests for improvement). Affine marks affine outer mappings.
<table><tr><td colspan="4">Low-dimensional</td><td colspan="4">High-dimensional</td></tr><tr><td>Benchmark</td><td>Method</td><td>Final-HV gain (%)</td><td>p</td><td>Benchmark</td><td>Method</td><td>Final-HV gain (%)</td><td>p</td></tr><tr><td rowspan="2">Color formulation [5]</td><td>LogEHVI</td><td> $\mathbf { + 1 3 . 8 \pm 1 . 6 ~ < 0 . 0 1 }$ </td><td></td><td rowspan="2">Synthetic 2 [18]</td><td>Sp-STCH</td><td> ${ \bf + 3 . 6 \pm 1 . 8 }$ </td><td>0.038</td></tr><tr><td>STCH</td><td> $+ 8 . 7 \pm 1 . 7 \ < \bf { 0 . 0 1 }$ </td><td></td><td>MORBO</td><td> ${ \bf + 5 . 3 \pm 2 . 6 }$ </td><td>0.036</td></tr><tr><td rowspan="2">Synthetic 1 [19]</td><td>LogEHVI</td><td> $+ 1 2 . 7 \pm 0 . 6 < 0 . 0 1$ </td><td></td><td rowspan="2">Lake management [20]</td><td>Sp-STCH</td><td> $+ 8 . 7 \pm \mathbf { 0 . 7 } \ < \mathbf { 0 . 0 1 }$ </td><td></td></tr><tr><td>STCH</td><td> $\mathbf { + 7 . 5 \pm 0 . 4 ~ < 0 . 0 1 }$ </td><td></td><td>MORBO</td><td> ${ \bf + 4 . 1 \pm 1 . 5 }$ </td><td>0.012</td></tr><tr><td rowspan="2">Truss design (Affine) [21]</td><td>LogEHVI</td><td> $\mathbf { + 0 . 8 \pm 0 . 1 ~ < 0 . 0 1 }$ </td><td></td><td rowspan="2">Power flow (Affine) [22] Sp-STCH</td><td></td><td> $+ 0 . 2 \pm 0 . 2 \qquad 0 . 1 7$ </td><td></td></tr><tr><td>STCH</td><td> $\mathbf { + 0 . 7 \pm 0 . 3 }$ </td><td>0.022</td><td>MORBO</td><td> $\mathbf { + 1 . 1 \pm 0 . 2 \ \ < 0 . 0 1 }$ </td><td></td></tr><tr><td rowspan="2">Nanoparticle design [23]</td><td>LogEHVI</td><td> $+ 0 . 2 \pm 0 . 2$ </td><td>0.17</td><td rowspan="2" colspan="4"></td></tr><tr><td>STCH</td><td> $+ 0 . 3 \pm 0 . 4$ </td><td>0.24</td></tr><tr><td rowspan="2">Reaction optimization [24] LogEHVI</td><td></td><td> $+ 0 . 2 \pm 0 . 2$ </td><td>0.17</td><td rowspan="2" colspan="3"></td></tr><tr><td>STCH</td><td> $- 0 . 4 \pm 0 . 9$ </td><td>0.67</td></tr></table>

Scientific-Design Experiments: We compare BlackBox-GP and Composite versions of four MOBO methods on eight benchmarks as listed in Table 2 (see Appendix F for details). All have at least one non-bijective outer mapping; only truss design and power flow have affine outer mappings. Low-dimensional problems use LogEHVI [17, 16] and STCH [11, 12] methods with 5 initial and 40 adaptive evaluations; high-dimensional problems use STCH with a spherical kernel (sp-STCH) [25] and MORBO [26] with 20 initial and 100 adaptive evaluations. Appendices G and H detail methods and implementation. Composite models intermediates with independent GPs. Two methods share the initial design, acquisition rule, reference point, and budget. We report mean percentage gains in Final-HV ± SE over 10 different trials for all problems, except Synthetic 1, which uses 80 trials.

Table 2 shows that Composite improved mean final hypervolume in 15 of 16 comparisons, with the largest gains on color formulation (8.7–13.8%) and Synthetic 1 (7.5–12.7%). Benefits extend to high-dimensional problems: Synthetic 2 (3.6–5.3%) and lake management (4.1–8.7%). Composite provides statistically significant gains on the two affine benchmarks. This might seem to contradict Proposition 1: for affine g, BlackBox-Induced also places a GP prior on f, yet observing intermediates provides no additional predictive information under that shared prior. However, the GP priors used by BlackBox-Induced and BlackBox-GP need not coincide. In our implementation, each intermediate has its own fitted kernel, so the proposition’s separable-covariance condition generally fails. Combining the kernels fitted to individual intermediates induces a richer kernel family over f than BlackBox-GP allows, potentially explaining these gains. Nanoparticle design and reaction optimization show small gains despite non-affine, non-bijective mappings, so these properties alone do not explain performance. Both have correlated intermediates that we model with independent GPs; modeling these dependencies may improve performance. Synthetic 2 also shows modest gains despite non-affine mappings. Limited data in this high-dimensional problem may hinder both approaches. Independently varying input dimension and evaluation budget could clarify their effects.

## 5 Conclusion

We investigate when modeling intermediate responses improves MOBO through theory and experiments. Under a shared prior, intermediate observations add no predictive information when the outer mapping is bijective or affine with separable covariance. In the controlled experiments, large single-objective regret reductions did not translate into comparably large hypervolume gains. In scientific-design benchmarks, Composite improved mean final hypervolume in almost all cases, with the largest gains on problems with non-affine, non-bijective outer mappings, although these properties alone do not ensure improvement. Further work should examine how modeling choices such as kernels and dependencies among responses, and input dimension relative to evaluation budget jointly determine these gains.

## Acknowledgement

We would like to thank the anonymous reviewers for their constructive feedback and suggestions. This material is based on work supported by the AI Research Institutes program supported by the NSF and Intel Corporation under NSF award DMR-2433348. PB gratefully acknowledges financial support from the Development and Promotion of Science and Technology Talents Project (DPST) scholarship program of the Institute for the Promotion of Teaching Science and Technology (IPST), Ministry of Education, Thailand.

## References

[1] Abhijith M Gopakumar, Prasanna V Balachandran, Dezhen Xue, James E Gubernatis, and Turab Lookman. Multi-objective optimization for materials discovery via adaptive design. Scientific reports, 8(1):3738, 2018.

[2] Joshua Knowles. Parego: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. IEEE transactions on evolutionary computation, 10(1):50–66, 2006.

[3] Carl Edward Rasmussen and Christopher KI Williams. Gaussian processes for machine learning. Number 3. MIT press Cambridge, MA, 2006.

[4] Tobias Ginsburg, Kyle Hippe, Ryan Lewis, Aileen Cleary, Doga Ozgulbas, Rory Butler, Casey Stone, Abraham Stroka, Rafael Vescovi, and Ian Foster. Exploring benchmarks for selfdriving labs using color matching. In Proceedings of the SC’23 Workshops of the International Conference on High Performance Computing, Network, Storage, and Analysis, pages 2147– 2152, 2023.

[5] Sabrine Chaouch, Ali Moussa, and Neji Ladhari. Color formulation of cotton fabrics using multi-objective ant colony optimization. Journal ofNatural Fibers, 19(17):15459–15474, 2022.

[6] Raul Astudillo and Peter Frazier. Bayesian optimization of composite functions. In International Conference on Machine Learning, pages 354–363. PMLR, 2019.

[7] Wesley J Maddox, Maximilian Balandat, Andrew G Wilson, and Eytan Bakshy. Bayesian optimization with high-dimensional outputs. Advances in neural information processing systems, 34:19274–19287, 2021.

[8] Wesley Maddox, Qing Feng, and Max Balandat. Optimizing high-dimensional physics simulations via composite bayesian optimization. arXiv preprint arXiv:2111.14911, 2021.

[9] Raul Astudillo and Peter Frazier. Bayesian optimization of function networks. Advances in neural information processing systems, 34:14463–14475, 2021.

[10] P. Buathong, J. Wan, R. Astudillo, S. Daulton, M. Balandat, and P. I. Frazier. Bayesian optimization of function networks with partial evaluations. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 4752–4784. PMLR, 2024.

[11] Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, and Qingfu Zhang. Smooth tchebycheff scalarization for multi-objective optimization. arXiv preprint arXiv:2402.19078, 2024.

[12] TM Nogueira Pires, RP Cardoso Coelho, and FM Andrade Pires. Composite bayesian optimisation for multi-objective problems with smooth tchebycheff scalarisation. Structural and Multidisciplinary Optimization, 69(2):34, 2026.

[13] Donald R Jones, Matthias Schonlau, and William J Welch. Efficient global optimization of expensive black-box functions. Journal ofGlobal optimization, 13(4):455–492, 1998.

[14] Peter I Frazier. Bayesian optimization. In Recent advances in optimization and modeling of contemporary problems, pages 255–278. Informs, 2018.

[15] Edward Snelson, Zoubin Ghahramani, and Carl Rasmussen. Warped gaussian processes. Advances in neural information processing systems, 16, 2003.

[16] Sebastian Ament, Samuel Daulton, David Eriksson, Maximilian Balandat, and Eytan Bakshy. Unexpected improvements to expected improvement for bayesian optimization. Advances in neural information processing systems, 36:20577–20612, 2023.

[17] Samuel Daulton, Maximilian Balandat, and Eytan Bakshy. Differentiable expected hypervolume improvement for parallel multi-objective bayesian optimization. Advances in neural information processing systems, 33:9851–9864, 2020.

[18] S. Surjanovic and D. Bingham. Virtual library of simulation experiments: Test functions and datasets. Retrieved November 8, 2026, from http://www.sfu.ca/\~ssurjano.

[19] Kalyanmoy Deb, Lothar Thiele, Marco Laumanns, and Eckart Zitzler. Scalable multi-objective optimization test problems. In Proceedings of the 2002 congress on evolutionary computation. CEC’02 (Cat. No. 02TH8600), volume 1, pages 825–830. IEEE, 2002.

[20] Victoria L Ward, Riddhi Singh, Patrick M Reed, and Klaus Keller. Confronting tipping points: Can multi-objective evolutionary algorithms discover pollution control tradeoffs given environmental thresholds? Environmental Modelling & Software, 73:27–43, 2015.

[21] Ryoji Tanabe and Hisao Ishibuchi. An easy-to-use real-world multi-objective optimization problem suite. Applied Soft Computing, 89:106078, 2020.

[22] Abhishek Kumar, Guohua Wu, Mostafa Z Ali, Qizhang Luo, Rammohan Mallipeddi, Ponnuthurai Nagaratnam Suganthan, and Swagatam Das. A benchmark-suite of real-world constrained multi-objective optimization problems and some baseline results. Swarm and Evolutionary Computation, 67:100961, 2021.

[23] Samuel Kim, Peter Y Lu, Charlotte Loh, Jamie Smith, Jasper Snoek, and Marin Soljaciˇ c. Deep´ learning for bayesian optimization of scientific problems with high-dimensional structure. arXiv preprint arXiv:2104.11667, 2021.

[24] Kobi C Felton, Jan G Rittig, and Alexei A Lapkin. Summit: benchmarking machine learning methods for reaction optimisation. Chemistry-Methods, 1(2):116–122, 2021.

[25] Colin Doumont, Donney Fan, Natalie Maus, Jacob R Gardner, Henry Moss, and Geoff Pleiss. We still don’t understand high-dimensional bayesian optimization. arXiv preprint arXiv:2512.00170, 2025.

[26] Samuel Daulton, David Eriksson, Maximilian Balandat, and Eytan Bakshy. Multi-objective bayesian optimization over high-dimensional search spaces. In Uncertainty in Artificial Intelligence, pages 507–517. PMLR, 2022.

[27] Paul Kubelka and Franz Munk. Ein beitrag zur optik der farbanstriche. Z. tech. Phys, 12:593– 601, 1931.

[28] Chris Wyman, Peter-Pike Sloan, and Peter Shirley. Simple analytic approximations to the cie xyz color matching functions. J. Comput. Graph. Tech, 2(2):1–11, 2013.

[29] Stephen R Carpenter, Donald Ludwig, and William A Brock. Management of eutrophication for lakes subject to potentially irreversible change. Ecological applications, 9(3):751–771, 1999.

[30] Maximilian Balandat, Brian Karrer, Daniel Jiang, Samuel Daulton, Ben Letham, Andrew G Wilson, and Eytan Bakshy. Botorch: A framework for efficient monte-carlo bayesian optimization. Advances in neural information processing systems, 33:21524–21538, 2020.

## Appendix

## A Induced Prior and Inference for BlackBox-Induced

At fixed evaluation locations $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } .$ let $\begin{array} { r l r } { H } & { { } = } & { \left( \mathbf { h } ( \mathbf { x } _ { 1 } ) , \ldots , \mathbf { h } ( \mathbf { x } _ { n } ) \right) } \end{array}$ and ${ \cal F } \quad = \quad$ $( f ( \mathbf { x } _ { 1 } ) , \ldots , f ( \mathbf { x } _ { n } ) )$ . BlackBox-Induced and Composite share the vector-valued GP prior h ∼ $\mathcal { G P } ( \mathbf { m } _ { h } , K _ { h } )$ , where $\mathbf { m } _ { h } ( \mathbf { x } ) = \mathbb { E } [ \mathbf { h } ( \mathbf { x } ) ]$ is the mean vector and $K _ { h } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \mathrm { C o v } ( \mathbf { h } ( \mathbf { x } ) , \mathbf { h } ( \mathbf { x } ^ { \prime } ) )$ is the matrix-valued covariance kernel. This prior induces a prior on $f = g \circ \mathbf { h }$ . BlackBox-Induced conditions only on the objective observations $F .$ . Its posterior conditions on $g ( \mathbf { h } ( \mathbf { x } _ { j } ) ) = F _ { j }$ . For bijective $^ { g , }$ this reduces to standard GP conditioning on $\begin{array} { r } { { \bf h } ( { \bf x } _ { j } ) = g ^ { - 1 } ( F _ { j } ) } \end{array}$ . For general non-bijective $g ,$ , inference integrates over intermediate responses consistent with $F$ and need not have a closed form. One numerical approach is to repeatedly draw the latent intermediate responses $H$ from their conditional distribution given $F$ under the shared GP prior. For each draw of $H$ , condition the GP on those responses, sample $\mathbf { h } ( \mathbf { x } _ { * } )$ at a new input $\mathbf { x } _ { * }$ , and apply $g$ to obtain a posterior predictive sample of $f ( \mathbf { x } _ { * } )$ . Averaging these samples estimates the posterior mean $\mathbb { E } [ f ( \mathbf { x } _ { * } ) \mid F ]$

For affine $g ( \mathbf { z } ) = \mathbf { a } ^ { \top } \mathbf { z } + b ,$ the induced prior of f is a GP with mean ${ \mathbf { a } } ^ { \top } { \mathbf { m } } _ { h } ( { \mathbf { x } } ) + b$ and kernel $\mathbf { a } ^ { \top } K _ { h } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \mathbf { a } .$ . Inference therefore uses standard GP conditioning directly on $F$ . If the intermediate responses are modeled as independent GPs, then $K _ { h } ( { \bf x } , { \bf x } ^ { \prime } )$ is diagonal. Its jth diagonal entry, $k _ { j } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = [ K _ { h } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) ] _ { j j }$ , is the covariance kernel of the jth intermediate response $h _ { j }$ . The induced kernel of $f$ therefore simplifies to

$$
\mathbf { a } ^ { \top } K _ { h } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \mathbf { a } = \sum _ { j } a _ { j } ^ { 2 } k _ { j } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) .
$$

A sum of Matérn or RBF kernels with different lengthscales is generally not a single kernel of the same family. Thus, separately fitted BlackBox-GP and BlackBox-Induced priors need not coincide.

## B Proof of Proposition 1

Throughout, the evaluation locations $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n }$ are fixed, and H and $F$ contain noiseless observations. Both conditional distributions are computed under a single prior on h, with $f = g \circ h ;$ in particular, the prior on $f$ is the one induced by this construction.

For the bijective case, we additionally assume that $g$ is measurable and bijective on a measurable set containing the intermediate responses with probability one, and that its inverse on the corresponding image is measurable. For the affine case, the mean function and covariance parameters of the GP prior on h are fixed and known.

Prooffor the bijective case. Since g is measurable, the objective observations

$$
F = \left( g ( \mathbf { h } ( \mathbf { x } _ { 1 } ) ) , \ldots , g ( \mathbf { h } ( \mathbf { x } _ { n } ) ) \right)
$$

are a measurable function of H. Conversely, since g is bijective with measurable inverse,

$$
H = { \big ( } g ^ { - 1 } ( f ( \mathbf { x } _ { 1 } ) ) , \dots , g ^ { - 1 } ( f ( \mathbf { x } _ { n } ) ) { \big ) }
$$

is a measurable function of F. Thus, each dataset can be recovered from the other, and neither provides additional information when the other is known.

Consequently, under the same prior on $h ,$ for every $x \in \mathcal { X }$ and $t \in \mathbb { R }$

$$
\begin{array} { r l } & { \mathbb { P } ( f ( \mathbf { x } ) \leq t \mid H ) = \mathbb { P } ( f ( \mathbf { x } ) \leq t \mid H , F ) } \\ & { \qquad = \mathbb { P } ( f ( \mathbf { x } ) \leq t \mid F ) \quad \mathrm { a l m o s t ~ s u r e l y } . } \end{array}
$$

Hence the conditional CDFs coincide.

Proof for the affine case. Suppose $g ( \mathbf { z } ) = \mathbf { a } ^ { \intercal } \mathbf { z } + b$ and h has a $\mathrm { G P }$ prior with mean $\mathbf { m } _ { \mathbf { h } } ( \mathbf { x } )$ and separable covariance

$$
\operatorname { C o v } ( \mathbf { h } ( \mathbf { x } ) , \mathbf { h } ( \mathbf { x } ^ { \prime } ) ) = k _ { \mathbf { h } } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) B ,
$$

where k is a scalar covariance kernel and B is a fixed positive semidefinite matrix. Then f is a GP with mean $m _ { f } ( { \mathbf { x } } ) = { \mathbf { a } } ^ { \top } { \mathbf { m } } _ { \mathbf { h } } ( { \mathbf { x } } ) + b$ and covariance

$$
k _ { f } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = c k _ { \mathbf { h } } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) , \qquad c = \mathbf { a } ^ { \top } B \mathbf { a } .
$$

${ \mathrm { I f ~ } } c = 0 .$ f is deterministic under the prior, so the result holds immediately. Assume henceforth that $c > 0 .$

Fix a prediction location x and define

$$
( K _ { \mathbf { h } } ) _ { i j } = k _ { \mathbf { h } } ( \mathbf { x } _ { i } , \mathbf { x } _ { j } ) , \qquad \mathbf { k } _ { \mathbf { h } } = \bigl ( k _ { \mathbf { h } } ( \mathbf { x } _ { 1 } , \mathbf { x } ) , \dots , k _ { \mathbf { h } } ( \mathbf { x } _ { n } , \mathbf { x } ) \bigr ) ^ { \top } .
$$

Write $K _ { \mathbf { h } } ^ { \dagger }$ for the Moore–Penrose pseudoinverse of $K _ { \mathbf { h } }$ , which equals $K _ { \mathbf { h } } ^ { - 1 }$ when $K _ { \mathbf { h } }$ is nonsingular. Conditioning the vector-valued GP on H gives

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } [ { \bf h } ( { \bf x } ) \mid H ] = { \bf m } _ { \bf h } ( { \bf x } ) + \sum _ { i = 1 } ^ { n } ( K _ { { \bf h } } ^ { \dagger } { \bf k } _ { \bf h } ) _ { i } \big ( { \bf h } ( { \bf x } _ { i } ) - { \bf m } _ { \bf h } ( { \bf x } _ { i } ) \big ) , } \\ & { \displaystyle \mathrm { C o v } ( { \bf h } ( { \bf x } ) \mid H ) = \big ( k _ { \bf h } ( { \bf x } , { \bf x } ) - { \bf k } _ { \bf h } ^ { \top } K _ { { \bf h } } ^ { \dagger } { \bf k } _ { \bf h } \big ) B . } \end{array}
$$

Applying g therefore yields a Gaussian conditional distribution for $f ( \mathbf { x } )$ with mean and variance

$$
\begin{array} { r l } & { \mu _ { H } ( \mathbf { x } ) = m _ { f } ( \mathbf { x } ) + \mathbf { k } _ { \mathbf { h } } ^ { \top } K _ { \mathbf { h } } ^ { \dag } \big ( F - \mathbf { m } _ { f } ( \mathbf { X } ) \big ) , } \\ & { v _ { H } ( \mathbf { x } ) = c \big ( k _ { \mathbf { h } } ( \mathbf { x } , \mathbf { x } ) - \mathbf { k } _ { \mathbf { h } } ^ { \top } K _ { \mathbf { h } } ^ { \dag } \mathbf { k } _ { \mathbf { h } } \big ) , } \end{array}
$$

where $\mathbf { m } _ { f } ( \mathbf { X } ) = ( m _ { f } ( \mathbf { x } _ { 1 } ) , \dots , m _ { f } ( \mathbf { x } _ { n } ) ) ^ { \top }$

Alternatively, conditioning the scalar GP f directly on F gives

$$
\begin{array} { r l } & { \mu _ { F } ( { \bf x } ) = m _ { f } ( { \bf x } ) + ( c { \bf k _ { h } } ) ^ { \top } ( c K _ { \bf h } ) ^ { \dagger } \left( F - { \bf m } _ { f } ( { \bf X } ) \right) = \mu _ { H } ( { \bf x } ) , } \\ & { v _ { F } ( { \bf x } ) = c k _ { \bf h } ( { \bf x } , { \bf x } ) - ( c { \bf k _ { h } } ) ^ { \top } ( c K _ { \bf h } ) ^ { \dagger } ( c { \bf k _ { h } } ) = v _ { H } ( { \bf x } ) , } \end{array}
$$

using $( c K _ { \mathbf { h } } ) ^ { \dagger } = c ^ { - 1 } K _ { \mathbf { h } } ^ { \dagger } .$

Both conditional distributions are Gaussian with identical means and variances. Hence, for every $t \in \mathbb { R }$

$$
\mathbb { P } ( f ( \mathbf { x } ) \leq t \mid H ) = \mathbb { P } ( f ( \mathbf { x } ) \leq t \mid F ) \quad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

## C Details of the construction of h function in controlled experiments

In the controlled experiments, for each h, we randomly generate h using 1024 random Fourier features approximating a Matern-5/2 GP with lengthscale 0.5. For each problem, this h function is randomly generated four times.

## D Comparison metrics

## D.1 Single-objective metric

Let $f ^ { \star } = \operatorname* { m i n } _ { \mathbf { x } } { f ( \mathbf { x } ) }$ be the optimum, computed numerically for every controlled mapping, and let

$$
r _ { T } ^ { v , s } = \Big ( \operatorname* { m i n } _ { 1 \leq t \leq T } f ( \mathbf { x } _ { t } ^ { v , s } ) \Big ) - f ^ { \star }
$$

be the simple regret of variant $v \in \{ \mathrm { B l a c k B o x \mathrm { - } G P } .$ , Composite} in paired trial s within evaluation budget T. Since we minimize, smaller values indicate better performance. Regret varies over many orders of magnitude, so we average it on the $\log _ { 1 0 }$ scale:

$$
\bar { r } _ { T } ^ { v } = 1 0 ^ { m _ { T } ^ { v } } , \qquad m _ { T } ^ { v } = { \frac { 1 } { S } } \sum _ { s = 1 } ^ { S } \log _ { 1 0 } { r _ { T } ^ { v , s } } ,
$$

where $m _ { T } ^ { v }$ is the BlackBox-GP or Composite column of Table E.1. We measure final performance by

$$
\mathrm { F i n a l - R G r e d u c t i o n } ( \% ) = 1 0 0 \frac { \overline { { r } } _ { T } ^ { \mathrm { B l a c k B o x - G P } } - \overline { { r } } _ { T } ^ { \mathrm { C o m p o s i t e } } } { \overline { { r } } _ { T } ^ { \mathrm { B l a c k B o x - G P } } } .\tag{D.1}
$$

Positive reductions favor Composite.

## D.2 Multi-objective metric

Let $\mathcal { P } \subset \mathbb { R } ^ { N }$ be the set of nondominated objective vectors observed so far for a minimization problem. A vector y dominates $\mathbf { y } ^ { \prime } \mathrm { i f } y _ { i } \leq y _ { i } ^ { \prime }$ for every i and the inequality is strict for at least one objective. Given a fixed reference point r worse than every vector in $\mathcal { P }$ , the dominated hypervolume (HV) is

$$
\mathrm { H V } ( { \mathcal { P } } ; \mathbf { r } ) = \lambda _ { N } \left( \bigcup _ { \mathbf { y } \in { \mathcal { P } } } \prod _ { i = 1 } ^ { N } [ y _ { i } , r _ { i } ] \right) ,\tag{D.2}
$$

where $\lambda _ { N }$ denotes N-dimensional volume. Larger HV indicates better objective values or broader coverage of favorable tradeoffs. When an observed vector is not componentwise better than the reference point, it contributes no volume. We use the same fixed reference point for all methods and trials in a given setting. Let $\mathrm { H V } _ { t } ^ { v , s }$ denote the dominated hypervolume after t total evaluations for variant $v \in$ {BlackBox-GP, Composite} in paired trial s. The mean HV across S trials is

$$
\overline { { \mathrm { H V } } } _ { t } ^ { v } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \mathrm { H V } _ { t } ^ { v , s } .
$$

We measure final performance by the relative improvement in mean HV at the total evaluation budget $T \colon$

$$
\mathrm { F i n a l - H V } \ \mathrm { g a i n } ( \mathcal { Y } _ { 0 } ) = 1 0 0 \frac { \overline { { \mathrm { H V } } } _ { T } ^ { \mathrm { C o m p o s i t e } } - \overline { { \mathrm { H V } } } _ { T } ^ { \mathrm { B l a c k B o x - G P } } } { \overline { { \mathrm { H V } } } _ { T } ^ { \mathrm { B l a c k B o x - G P } } } .\tag{D.3}
$$

## D.3 Significance

The two arms share the function draw, the initial design, the acquisition rule and the evaluation budget, so runs are paired. All reported p-values are one-sided paired t-tests for a positive mean gain, computed on the per-run differences $\mathrm { l o g } _ { 1 0 } ^ { \bullet } r _ { T } ^ { \mathrm { B l a c k B o x - G P } , s } - \mathrm { l o g } _ { 1 0 } ^ { \bullet } r _ { T } ^ { \mathrm { C o m p o s i t e } , s }$ and $\mathrm { H V } _ { T } ^ { \mathrm { C o m p o s i t e } , s } -$ $\mathrm { H V } _ { T } ^ { \mathrm { \bar { B } l a c k B o x - G P } , s }$ . All tables report mean ± one standard error over the paired runs. Eq. (D.1) and Eq. (D.3) are functions of the arm means rather than averages of per-run percentages, so their standard errors are obtained from the standard error of the mean paired difference by the delta method.

## E Additional results for controlled experiments

Table 1 of the main text reports six representative outer mappings. This appendix gives every mapping we considered, under the metrics of Appendix D. Table E.1 reports results for the single-objective cases while Table E.2 reports results for multi-objective (two-objective) cases. Both settings use four function draws and five initial designs per draw, giving 20 paired runs per problem, and within each comparison the two methods (BlackBox-GP and Composite) share the initial design, the acquisition rule and the evaluation budget.

Single objective. Of the five mappings that are bijective on the range h attains, none reaches significance; of the eleven that are not, every one does, with $p \leq 0 . 0 3 0$ throughout. Bijectivity therefore separates the two groups perfectly in this setting.

Percentage reductions near 100% can obscure substantial differences in final regret. For the two-root quadratic and four-root quartic, reductions of 99.96% and 99.53% correspond to approximately 2600-fold and 210-fold reductions relative to BlackBox-GP, respectively. Table E.1 therefore also reports gains in $\log _ { 1 0 }$ regret to distinguish improvements of different orders of magnitude. The 100% ceiling also explains the narrow standard errors on the saturated rows.

Multi-objective (two objectives). The pattern transfers only partially. None of the five bijective mappings reaches significance, as in the single-objective case, but only five of the fourteen nonbijective settings do, and the effect sizes are percent-level rather than orders of magnitude. The ordering also fails to carry over: the three- and five-root polynomials give the two weakest singleobjective gains but both yield significant hypervolume gains, including the largest, while the four-root quartic and the shifted squares at $( h - 1 ) ^ { 2 }$ and $( h - 2 ) ^ { 2 }$ all exceed +99.5% in the single-objective grid yet fall below $+ 0 . 5 \%$ here. Hypervolume optimization requires discovering and refining a set of tradeoffs rather than locating a single optimum, and under a fixed budget that additional difficulty appears to dilute the benefit of modelling intermediates.

Table E.1: Controlled single-objective experiments, all 16 outer mappings. Values in BlackBox-GP and Composite columns are the final $\log _ { 1 0 }$ simple regret from each method; Reduction is their paired difference, so positive favors Composite. Final-RG reduction is that same difference as a percentage, via Eq. (D.1). All values are means ± one standard error over the 20 paired runs, the percentage column by the delta method, and p is a one-sided paired t-test for positive mean reduction. Bold marks significance at the unadjusted 0.05 level.
<table><tr><td>Outer mapping g</td><td>BlackBox-GP</td><td>Composite</td><td>Reduction</td><td>Final-RG reduction (%)</td><td>p</td></tr><tr><td colspan="6">Bijective on the range h attains</td></tr><tr><td>Exponential  $e ^ { h }$ </td><td> $- 2 . 5 9 \pm 0 . 0 9$ </td><td> $- 2 . 7 4 \pm 0 . 1 4$ </td><td> $+ 0 . 1 5 \pm 0 . 1 2$ </td><td> $+ 2 8 . 9 1 \pm 1 9 . 0 1$ </td><td>0.11</td></tr><tr><td>Identity h</td><td> $- 1 . 0 3 \pm 0 . 1 8$ </td><td> $- 1 . 1 1 \pm 0 . 1 4$ </td><td> $+ 0 . 0 9 \pm 0 . 1 6$ </td><td> $+ 1 8 . 1 0 \pm 2 9 . 2 4$ </td><td>0.29</td></tr><tr><td>Cubic  $\mathbf { \dot { \boldsymbol { h } } ^ { 3 } }$ </td><td> $+ 0 . 0 9 \pm 0 . 1 1$ </td><td> $+ 0 . 0 1 \pm 0 . 1 0$ </td><td> $+ 0 . 0 8 \pm 0 . 1 1$ </td><td> $+ 1 7 . 1 0 \pm 2 1 . 0 7$ </td><td>0.23</td></tr><tr><td>Hyperbolic sine sinh h</td><td> $+ 0 . 2 4 \pm 0 . 1 1$ </td><td> $+ 0 . 1 6 \pm 0 . 0 9$ </td><td> $+ 0 . 0 8 \pm 0 . 1 0$ </td><td> $+ 1 7 . 0 0 \pm 1 9 . 5 8$ </td><td>0.22</td></tr><tr><td>Quintic  $h ^ { 5 }$ </td><td> $+ 0 . 7 5 \pm 0 . 1 5$ </td><td> $+ 0 . 7 9 \pm 0 . 0 8$ </td><td> $- 0 . 0 5 \pm 0 . 1 6$ </td><td> $- 1 1 . 6 2 \pm 4 0 . 3 0$ </td><td>0.62</td></tr><tr><td colspan="6">Not bijective</td></tr><tr><td>Two-root quadratic</td><td> ${ \bf - 4 . 9 7 \pm 0 . 2 8 }$ </td><td> $\mathbf { - 8 . 3 9 \pm 0 . 3 0 }$ </td><td> $\mathbf { + 3 . 4 2 \pm 0 . 4 1 }$ </td><td> $\mathbf { + 9 9 . 9 6 \pm 0 . 0 4 }$ </td><td></td></tr><tr><td>Shifted square  $( h - 1 ) ^ { 2 }$ </td><td> $\mathbf { - 5 . 3 6 \pm 0 . 2 4 }$ </td><td> $\mathbf { - 8 . 2 2 \pm 0 . 2 4 }$ </td><td> $\mathbf { + 2 . 8 6 \pm 0 . 3 0 }$ </td><td> $\mathbf { + 9 9 . 8 6 \pm 0 . 0 9 }$ </td><td>&lt;0.01 &lt;0.01</td></tr><tr><td>Square  $h ^ { \dot { 2 } }$ </td><td> ${ \bf - 5 . 1 4 \pm 0 . 2 5 }$ </td><td> $\mathbf { - 7 . 9 5 \pm 0 . 2 6 }$ </td><td> $\mathbf { + 2 . 8 1 \pm 0 . 4 0 }$ </td><td></td><td></td></tr><tr><td> $( h - 2 ) ^ { 2 }$ </td><td> $\mathbf { - 5 . 6 2 \pm 0 . 2 2 }$ </td><td> ${ \bf - 8 . 4 0 \pm 0 . 2 8 }$ </td><td> $\mathbf { + 2 . 7 8 \pm 0 . 3 5 }$ </td><td> $\mathbf { + 9 9 . 8 4 \pm 0 . 1 4 }$ </td><td>&lt;0.01</td></tr><tr><td>Shifted square  $( h - 0 . 5 ) ^ { 2 }$ </td><td> ${ \bf - 5 . 4 0 \pm 0 . 2 8 }$ </td><td> $\mathbf { - 7 . 8 1 \pm 0 . 1 4 }$ </td><td> $\mathbf { + 2 . 4 1 \pm 0 . 2 5 }$ </td><td> $\mathbf { + 9 9 . 8 3 \pm 0 . 1 3 }$ </td><td>&lt;0.01</td></tr><tr><td>Shifted square Shifted square  $( h - 3 ) ^ { 2 }$ </td><td> $\mathbf { - 4 . 6 9 \pm 0 . 3 6 }$ </td><td> $\mathbf { - 7 . 0 5 \pm 0 . 4 9 }$ </td><td> $\mathbf { + 2 . 3 6 \pm 0 . 4 6 }$ </td><td> $\mathbf { + 9 9 . 6 1 \pm 0 . 2 2 }$ </td><td>&lt;0.01</td></tr><tr><td>Four-root quartic</td><td> $\mathbf { - 4 . 8 0 \pm 0 . 2 1 }$ </td><td> $\mathbf { - 7 . 1 3 \pm 0 . 1 7 }$ </td><td> $\mathbf { + 2 . 3 3 \pm 0 . 2 0 }$ </td><td> $\mathbf { + 9 9 . 5 6 \pm 0 . 4 6 }$ </td><td>&lt;0.01</td></tr><tr><td>Cosine cos h</td><td> $\mathbf { - 4 . 3 6 \pm 0 . 4 1 }$ </td><td> $\mathbf { - 5 . 7 9 \pm 0 . 4 3 }$ </td><td> $\mathbf { + 1 . 4 3 \pm 0 . 4 8 }$ </td><td> $\mathbf { + 9 9 . 5 3 \pm 0 . 2 2 }$   $\mathbf { + 9 6 . 3 1 \pm 4 . 0 6 }$ </td><td>&lt;0.01</td></tr><tr><td>Cosine cos 2h</td><td> $- { \bf 4 . 9 4 } \pm { \bf 0 . 3 2 }$ </td><td> $\mathbf { - 6 . 0 0 \pm 0 . 2 0 }$ </td><td> $\mathbf { + 1 . 0 6 \pm 0 . 3 5 }$ </td><td> $\mathbf { + 9 1 . 2 7 \pm 7 . 0 8 }$ </td><td>&lt;0.01</td></tr><tr><td>Five-root quintic</td><td> $\mathbf { + 1 . 6 9 \pm 0 . 0 4 }$ </td><td> $\mathbf { + 1 . 2 7 \pm 0 . 1 6 }$ </td><td> $\mathbf { + 0 . 4 2 \pm 0 . 1 4 }$ </td><td> $\mathbf { + 6 2 . 3 6 \pm 1 2 . 3 3 }$ </td><td>&lt;0.01 &lt;0.01</td></tr><tr><td></td><td></td><td></td><td> $\mathbf { + 0 . 2 6 \pm 0 . 1 3 }$ </td><td></td><td></td></tr><tr><td>Three-root cubic</td><td> $\mathbf { + 0 . 5 4 \pm 0 . 0 9 }$ </td><td> $\mathbf { + 0 . 2 8 \pm 0 . 1 1 }$ </td><td></td><td> $\mathbf { + 4 5 . 3 2 \pm 1 6 . 4 4 }$ </td><td>0.03</td></tr></table>

Table E.2: Controlled two-objective experiments, all 19 distinct settings of $g _ { 1 }$ . Values in BlackBox-GP and Composite columns are the final hypervolume; Gain is their paired difference, so positive favors Composite. Final-HV gain is that same difference as a percentage, via Eq. (D.3). All values are means ± one standard error over the 20 paired runs, the percentage column by the delta method, and p is a one-sided paired t-test for positive mean gain. The BlackBox-GP and Composite columns have the largest standard errors because each mapping pools four function draws whose hypervolumes differ substantially; that spread cancels in the paired Gain. Bold marks significance at the unadjusted 0.05 level.
<table><tr><td>Outer mapping g1</td><td>BlackBox-GP</td><td>Composite</td><td>Gain</td><td>Final-HV gain (%)</td><td>p</td></tr><tr><td colspan="6">Bijective on the range h attains</td></tr><tr><td>Cubic  $h ^ { 3 }$ </td><td> $7 1 . 4 9 \pm 1 1 . 1 7$ </td><td> $7 2 . 1 2 \pm 1 1 . 2 2$ </td><td> $+ 0 . 6 3 \pm 1 . 1 1$ </td><td> $+ 0 . 8 8 \pm 1 . 5 6$ </td><td>0.29</td></tr><tr><td>Exponential  $e ^ { h }$ </td><td> $7 1 . 1 7 \pm 1 1 . 2 2$ </td><td> $7 1 . 5 8 \pm 1 1 . 0 9$ </td><td> $+ 0 . 4 1 \pm 0 . 9 2$ </td><td> $+ 0 . 5 8 \pm 1 . 2 9$ </td><td>0.33</td></tr><tr><td>Quintic  $h ^ { 5 }$ </td><td> $7 1 . 9 5 \pm 1 1 . 1 1$ </td><td> $7 2 . 2 1 \pm 1 0 . 9 7$ </td><td> $+ 0 . 2 6 \pm 1 . 4 4$ </td><td> $+ 0 . 3 6 \pm 2 . 0 0$ </td><td>0.43</td></tr><tr><td>Identity  $h$ </td><td> $7 0 . 9 3 \pm 1 1 . 1 5$ </td><td> $7 1 . 1 0 \pm 1 1 . 3 0$ </td><td> $+ 0 . 1 6 \pm 0 . 6 5$ </td><td> $+ 0 . 2 3 \pm 0 . 9 2$ </td><td>0.40</td></tr><tr><td>Hyperbolic sine sinh h</td><td> $7 1 . 4 6 \pm 1 1 . 2 5$ </td><td> $7 1 . 1 9 \pm 1 1 . 2 1$ </td><td> $- 0 . 2 7 \pm 0 . 5 3$ </td><td> $- 0 . 3 8 \pm 0 . 7 5$ </td><td>0.69</td></tr><tr><td colspan="6">Not bijective</td></tr><tr><td>Five-root quintic</td><td> $\mathbf { 7 2 . 1 3 \pm 1 1 . 2 7 }$ </td><td> $\mathbf { 8 0 . 5 1 \pm 1 0 . 0 4 }$ </td><td> $\mathbf { + 8 . 3 8 \pm 3 . 2 3 }$ </td><td> $\mathbf { + 1 1 . 6 2 \pm 4 . 4 8 }$ </td><td>0.01</td></tr><tr><td>Square  $h ^ { 2 ^ { ' } }$ </td><td> $\mathbf { 6 9 . 8 8 \pm 1 0 . 8 7 }$ </td><td> $\mathbf { 7 1 . 9 1 \pm 1 1 . 1 6 }$ </td><td> $\mathbf { + 2 . 0 3 \pm 0 . 7 0 }$ </td><td> $\mathbf { + 2 . 9 0 \pm 1 . 0 1 }$ </td><td>&lt;0.01</td></tr><tr><td>Shifted square  $( h$  − 0.5)2</td><td> ${ \bf 7 0 . 9 8 \pm 1 1 . 0 3 }$ </td><td> $\mathbf { 7 2 . 9 4 \pm 1 1 . 2 8 }$ </td><td> $\mathbf { + 1 . 9 6 \pm 0 . 8 0 }$ </td><td> $\mathbf { + 2 . 7 6 \pm 1 . 1 3 }$ </td><td>0.01</td></tr><tr><td>Three-root cubic</td><td> ${ \bf 7 0 . 7 2 \pm 1 1 . 1 2 }$ </td><td> $\mathbf { 7 2 . 5 2 \pm 1 1 . 0 4 }$ </td><td> $\mathbf { + 1 . 8 0 \pm 0 . 8 6 }$ </td><td> $\mathbf { + 2 . 5 4 \pm 1 . 2 2 }$ </td><td>0.03</td></tr><tr><td>Two-root quadratic</td><td> $7 1 . 5 6 \pm 1 1 . 2 3$ </td><td> $7 2 . 8 8 \pm 1 1 . 2 3 $ </td><td> $+ 1 . 3 2 \pm 0 . 8 1$ </td><td> $+ 1 . 8 5 \pm 1 . 1 3$ </td><td>0.06</td></tr><tr><td>Cosine cos 2h</td><td> ${ \bf 7 0 . 5 4 \pm 1 1 . 0 5 }$ </td><td> ${ \bf 7 1 . 3 9 \pm 1 1 . 0 8 }$ </td><td> $\mathbf { + 0 . 8 5 \pm 0 . 4 5 }$ </td><td> $\mathbf { + 1 . 2 0 \pm 0 . 6 4 }$ </td><td>0.04</td></tr><tr><td>Shifted square  $( h - 3 ) ^ { 2 }$ </td><td> $7 0 . 7 9 \pm 1 1 . 1 7$ </td><td> $7 1 . 5 0 \pm 1 1 . 1 8$ </td><td> $+ 0 . 7 0 \pm 0 . 5 8$ </td><td> $+ 1 . 0 0 \pm 0 . 8 3$ </td><td>0.12</td></tr><tr><td>Three-root, spread 0.5</td><td> $7 0 . 7 8 \pm 1 0 . 9 5$ </td><td> $7 1 . 3 1 \pm 1 0 . 9 5$ </td><td> $+ 0 . 5 3 \pm 0 . 5 7$ </td><td> $+ 0 . 7 5 \pm 0 . 8 0$ </td><td>0.18</td></tr><tr><td>Four-root, spread 0.5</td><td> $7 1 . 1 5 \pm 1 1 . 0 8$ </td><td> $7 1 . 6 3 \pm 1 1 . 2 2$ </td><td> $+ 0 . 4 7 \pm 0 . 5 5$ </td><td> $+ 0 . 6 6 \pm 0 . 7 8$ </td><td>0.20</td></tr><tr><td>Shifted square  $( h - 1 ) ^ { 2 }$ </td><td> $7 1 . 5 2 \pm 1 1 . 2 9$ </td><td> $7 1 . 8 7 \pm 1 1 . 1 9$ </td><td> $+ 0 . 3 5 \pm 0 . 6 5$ </td><td> $+ 0 . 4 9 \pm 0 . 9 0$ </td><td>0.30</td></tr><tr><td>Shifted square  $( h - 2 ) ^ { 2 }$ </td><td> $7 1 . 0 9 \pm 1 1 . 0 2$ </td><td> $7 1 . 3 9 \pm 1 1 . 1 9$ </td><td> $+ 0 . 3 0 \pm 0 . 8 4$ </td><td> $+ 0 . 4 2 \pm 1 . 1 8$ </td><td>0.36</td></tr><tr><td>Six-root, spread 1.0</td><td> $7 1 . 6 1 \pm 1 1 . 1 9$ </td><td> $7 1 . 6 6 \pm 1 1 . 1 4$ </td><td> $+ 0 . 0 5 \pm 1 . 0 6$ </td><td> $+ 0 . 0 7 \pm 1 . 4 8$ </td><td>0.48</td></tr><tr><td>Four-root quartic</td><td> $7 1 . 9 7 \pm 1 1 . 4 0$ </td><td> $7 1 . 9 5 \pm 1 1 . 2 8$ </td><td> $- 0 . 0 1 \pm 0 . 8 5$ </td><td> $- 0 . 0 2 \pm 1 . 1 8$ </td><td>0.51</td></tr><tr><td>Cosine cos h</td><td></td><td></td><td> $- 1 . 0 5 \pm 0 . 6 3$ </td><td></td><td></td></tr><tr><td></td><td> $7 1 . 8 6 \pm 1 1 . 1 8$ </td><td> $7 0 . 8 1 \pm 1 1 . 1 5$ </td><td></td><td> $1 . 4 6 \pm 0 . 8 7$ </td><td>0.95</td></tr></table>

## F Benchmark definitions

This section specifies the exact deterministic oracle used in the experiments. All design vectors x below are normalized to $[ 0 , 1 ] ^ { d }$ , and all final objectives are minimized. Here d is the input dimension, K is the number of scalar intermediate responses retained by the implementation, and N is the number of final objectives.

## F.1 Multi-illuminant color formulation: $d = 6 , K = 6 , N = 2$

This analytical simulator represents six hypothetical dye loadings in an opaque textile coating. It is motivated by multi-objective color formulation under multiple illuminants [5], but is not a measured dataset and does not reproduce that paper’s objective equations. Wavelengths are $\lambda \in$ {400, 410, . . . , 700} nm. Define the Gaussian spectral bump of center $\mu$ and width $\sigma$

$$
B ( \lambda ; \mu , \sigma ) = \exp \left[ - \frac { 1 } { 2 } \left( \frac { \lambda - \mu } { \sigma } \right) ^ { 2 } \right] .\tag{F.1}
$$

The substrate absorption-to-scattering ratio is $q _ { 0 } ( \lambda ) = 0 . 0 1 2 + 0 . 0 0 8 B ( \lambda ; 4 3 0 , 7 5 )$ . The six fixed dye increments are

$$
\begin{array} { r l } & { q _ { 1 } ( \lambda ) = 1 . 7 5 B ( \lambda ; 6 1 0 , 3 4 ) + 0 . 2 2 B ( \lambda ; 4 4 5 , 5 2 ) , } \\ & { q _ { 2 } ( \lambda ) = 1 . 5 5 B ( \lambda ; 5 3 5 , 3 1 ) + 0 . 1 8 B ( \lambda ; 6 5 0 , 4 2 ) , } \\ & { q _ { 3 } ( \lambda ) = 1 . 6 5 B ( \lambda ; 4 4 8 , 2 8 ) + 0 . 1 6 B ( \lambda ; 5 8 5 , 6 5 ) , } \\ & { q _ { 4 } ( \lambda ) = 1 . 2 0 B ( \lambda ; 4 9 5 , 4 8 ) + 0 . 3 8 B ( \lambda ; 6 3 2 , 2 7 ) , } \\ & { q _ { 5 } ( \lambda ) = 1 . 3 5 B ( \lambda ; 5 7 5 , 4 1 ) + 0 . 2 4 B ( \lambda ; 4 2 0 , 2 4 ) , } \\ & { q _ { 6 } ( \lambda ) = 0 . 5 5 + 0 . 2 2 B ( \lambda ; 5 1 0 , 1 1 5 ) . } \end{array}\tag{F.2}
$$

For design x, the dye concentrations and total absorption-to-scattering ratio are

$$
c _ { m } ( { \bf x } ) = 0 . 8 5 x _ { m } , \qquad q ( \lambda ; { \bf x } ) = q _ { 0 } ( \lambda ) + \sum _ { m = 1 } ^ { 6 } c _ { m } ( { \bf x } ) q _ { m } ( \lambda ) .\tag{F.3}
$$

The opaque Kubelka–Munk inverse [27] gives

$$
R ( \lambda ; { \bf x } ) = 1 + q ( \lambda ; { \bf x } ) - \sqrt { q ( \lambda ; { \bf x } ) ^ { 2 } + 2 q ( \lambda ; { \bf x } ) } .\tag{F.4}
$$

Let $\overline { { \mathbf { c } } } ( \lambda ) = ( \bar { x } ( \lambda ) , \bar { y } ( \lambda ) , \bar { z } ( \lambda ) )$ denote the analytic CIE 1931 color-matching functions of Wyman et al. [28], and write $\Lambda _ { c } = \{ 4 0 0 , 4 1 0 , \ldots , 7 0 0 \}$ nm. For $\lambda _ { \mathrm { m } } = 1 0 ^ { - 9 } \lambda$ meters, define the normalized blackbody spectrum

$$
\widetilde { I } _ { T } ( \lambda ) = \frac { 1 } { \lambda _ { \mathrm { m } } ^ { 5 } \left[ \exp \left( \frac { c _ { 2 } } { \lambda _ { \mathrm { m } } T } \right) - 1 \right] } , \qquad I _ { T } ( \lambda ) = \frac { \widetilde { I } _ { T } ( \lambda ) } { \operatorname* { m a x } _ { \lambda ^ { \prime } \in \Lambda _ { c } } \widetilde { I } _ { T } ( \lambda ^ { \prime } ) } ,\tag{F.5}
$$

where $c _ { 2 } ~ = ~ 1 . 4 3 8 7 7 6 8 7 7 \times 1 0 ^ { - 2 } ~ \mathrm { m K }$ . For illuminant $j \in \{ D , A \}$ , with $T _ { D } = 6 5 0 4 \mathrm { ~ K ~ }$ and $T _ { A } = 2 8 5 6 \mathrm { ~ K ~ }$ , define

$$
\kappa _ { j } = \left[ \sum _ { \lambda \in \Lambda _ { c } } I _ { T _ { j } } ( \lambda ) \bar { y } ( \lambda ) \right] ^ { - 1 } .\tag{F.6}
$$

The tristimulus coordinates and perfect-reflector white point are

$$
\big ( X _ { j } ( { \mathbf x } ) , Y _ { j } ( { \mathbf x } ) , Z _ { j } ( { \mathbf x } ) \big ) = \kappa _ { j } \sum _ { \lambda \in \Lambda _ { c } } R ( \lambda ; { \mathbf x } ) I _ { T _ { j } } ( \lambda ) \overline { { \mathbf c } } ( \lambda ) ,\tag{F.7}
$$

$$
\left( X _ { n , j } , Y _ { n , j } , Z _ { n , j } \right) = \kappa _ { j } \sum _ { \lambda \in \Lambda _ { c } } I _ { T _ { j } } ( \lambda ) \overline { { \mathbf { c } } } ( \lambda ) .\tag{F.8}
$$

The corresponding CIE Lab coordinates are

$$
L _ { j } ^ { * } ( \mathbf { x } ) = 1 1 6 \phi \bigg ( \frac { Y _ { j } ( \mathbf { x } ) } { Y _ { n , j } } \bigg ) - 1 6 ,\tag{F.9}
$$

$$
a _ { j } ^ { * } ( \mathbf { x } ) = 5 0 0 \left[ \phi \bigg ( \frac { X _ { j } ( \mathbf { x } ) } { X _ { n , j } } \bigg ) - \phi \bigg ( \frac { Y _ { j } ( \mathbf { x } ) } { Y _ { n , j } } \bigg ) \right] ,\tag{F.10}
$$

$$
b _ { j } ^ { * } ( \mathbf { x } ) = 2 0 0 \left[ \phi \bigg ( \frac { Y _ { j } ( \mathbf { x } ) } { Y _ { n , j } } \bigg ) - \phi \bigg ( \frac { Z _ { j } ( \mathbf { x } ) } { Z _ { n , j } } \bigg ) \right] ,\tag{F.11}
$$

where $\delta = 6 / 2 9$ and

$$
\begin{array} { r } { \phi ( t ) = \left\{ { t } ^ { 1 / 3 } , \quad \begin{array} { l l } { t > \delta ^ { 3 } , } \\ { t / ( 3 \delta ^ { 2 } ) + 4 / 2 9 , } \end{array} \right. } \end{array}\tag{F.12}
$$

The objective-specific intermediate responses are

$$
\begin{array} { r l } & { { \bf h } _ { 1 } ( { \bf x } ) = \big ( L _ { D } ^ { * } ( { \bf x } ) , a _ { D } ^ { * } ( { \bf x } ) , b _ { D } ^ { * } ( { \bf x } ) \big ) , } \\ & { { \bf h } _ { 2 } ( { \bf x } ) = \big ( L _ { D } ^ { * } ( { \bf x } ) , a _ { D } ^ { * } ( { \bf x } ) , b _ { D } ^ { * } ( { \bf x } ) , L _ { A } ^ { * } ( { \bf x } ) , a _ { A } ^ { * } ( { \bf x } ) , b _ { A } ^ { * } ( { \bf x } ) \big ) . } \end{array}\tag{F.13}
$$

These contain six distinct retained scalar responses because the daylight coordinates are shared.

The unattainable target pigment has

$$
\begin{array} { r } { q _ { \star } ( \lambda ) = q _ { 0 } ( \lambda ) + 0 . 9 2 B ( \lambda ; 4 8 2 , 1 9 ) + 0 . 7 4 B ( \lambda ; 6 2 2 , 2 5 ) + 0 . 1 8 B ( \lambda ; 5 5 0 , 7 2 ) . } \end{array}\tag{F.14}
$$

The target ratio $q _ { \star } ( \lambda )$ is converted to reflectance using Eq. (F.4), then to illuminant-specific XYZ and CIE Lab coordinates using the equations above. The resulting target vectors are $\mathbf { t } _ { j } = ( L _ { \star , j } ^ { \ast } , a _ { \star , j } ^ { \ast } , b _ { \star , j } ^ { \ast } )$ for $j \in \{ D , A \}$ . For a generic argument $\mathbf { y } ,$ , define

$$
g _ { 1 } ( \mathbf { y } ) = \frac { \| \mathbf { y } _ { 1 : 3 } - \mathbf { t } _ { D } \| _ { 2 } } { 6 0 } , \qquad g _ { 2 } ( \mathbf { y } ) = \frac { \| ( \mathbf { y } _ { 4 : 6 } - \mathbf { t } _ { A } ) - ( \mathbf { y } _ { 1 : 3 } - \mathbf { t } _ { D } ) \| _ { 2 } } { 1 8 } .\tag{F.15}
$$

The objectives are $f _ { i } ( { \bf x } ) = g _ { i } ( { \bf h } _ { i } ( { \bf x } ) )$ . Thus $f _ { 1 }$ is daylight color error and $f _ { 2 }$ is a metamericmismatch proxy: it penalizes how the sample-minus-target Lab error changes between illuminants. The nonlinear norms erase the direction of the six-dimensional color errors. The hypervolume reference is (1.05, 1.05).

## F.2 DTLZ2 (Synthetic 1): $d = 6 , K = 4 , N = 2$

The synthetic two-objective DTLZ2 problem [19] is

$$
q ( { \bf x } ) = \sum _ { j = 2 } ^ { 6 } ( x _ { j } - 0 . 5 ) ^ { 2 } , \qquad \theta ( { \bf x } ) = { \frac { \pi x _ { 1 } } { 2 } } ,\tag{F.16}
$$

$$
\mathbf { h } _ { 1 } ( \mathbf { x } ) = { \big ( } q ( \mathbf { x } ) , \cos \theta ( \mathbf { x } ) { \big ) } , \qquad \mathbf { h } _ { 2 } ( \mathbf { x } ) = { \big ( } q ( \mathbf { x } ) , \sin \theta ( \mathbf { x } ) { \big ) } ,\tag{F.17}
$$

$$
g _ { 1 } ( \mathbf { y } ) = ( 1 + y _ { 1 } ) y _ { 2 } , \qquad g _ { 2 } ( \mathbf { y } ) = ( 1 + y _ { 1 } ) y _ { 2 } , \qquad f _ { i } ( \mathbf { x } ) = g _ { i } ( \mathbf { h } _ { i } ( \mathbf { x } ) ) .\tag{F.18}
$$

The repeated q deliberately gives each objective an independently modeled component group, avoiding implicit multi-task transfer. The Pareto set has $x _ { j } = 0 . 5$ for $j = 2 , \ldots , 6$ and the front satisfies $f _ { 1 } ^ { 2 } + f _ { 2 } ^ { 2 } = 1 , f _ { 1 } , f _ { 2 } \geq 0$ . For reference point (2.5, 2.5) the exact maximum hypervolume is $2 . 5 ^ { 2 } - \pi / 4 .$

## F.3 Four-bar truss RE21: $d = 4 , K = 8 , N = 2$

Four bars are joined into a frame carrying a load at one joint, and the design chooses how thick each bar is. Thicker bars sag less but use more material, giving the problem a tradeoff. This problem follows the RE21 formulation in the real-world multi-objective suite [21]. The four cross-sectional areas are

$$
\mathbf { a } ( \mathbf { x } ) = \mathbf { a } ^ { \mathrm { m i n } } + \mathbf { x } \odot ( \mathbf { a } ^ { \mathrm { m a x } } - \mathbf { a } ^ { \mathrm { m i n } } ) , \quad \mathbf { a } ^ { \mathrm { m i n } } = ( 1 , \sqrt { 2 } , \sqrt { 2 } , 1 ) , \quad \mathbf { a } ^ { \mathrm { m a x } } = ( 3 , 3 , 3 , 3 )\tag{F.19}
$$

in $\mathrm { c m ^ { 2 } }$ , where each entry of x sets that bar’s area as a fraction of the way between its minimum and maximum. The lower bounds are the thinnest each bar can be without exceeding the material’s stress limit of $1 0 \mathrm { k N / c m ^ { 2 } }$ under the load $F = 1 0 \mathrm { k N }$ , so every design in the box is already safe. Each bar is $L = 2 0 0$ cm long and the material’s stiffness (elastic modulus) is $E = 2 \times 1 0 ^ { 5 } ~ \mathrm { k N / c m ^ { 2 } }$ . The intermediate responses are

$$
\begin{array} { r l } & { { \bf h } _ { 1 } ( { \bf x } ) = \left( 2 a _ { 1 } ( { \bf x } ) , \sqrt { 2 } a _ { 2 } ( { \bf x } ) , \sqrt { a _ { 3 } ( { \bf x } ) } , a _ { 4 } ( { \bf x } ) \right) , } \\ & { { \bf h } _ { 2 } ( { \bf x } ) = \left( \frac { 2 } { a _ { 1 } ( { \bf x } ) } , \frac { 2 \sqrt { 2 } } { a _ { 2 } ( { \bf x } ) } , - \frac { 2 \sqrt { 2 } } { a _ { 3 } ( { \bf x } ) } , \frac { 2 } { a _ { 4 } ( { \bf x } ) } \right) . } \end{array}\tag{F.20}
$$

Each entry corresponds to a bar’s contribution to total material in $\mathbf { h } _ { 1 }$ , and to the sag at the loaded joint in $\mathbf { h } _ { 2 }$ . Raw volume and loaded-joint displacement are

$$
V ( \mathbf { y } ) = L \sum _ { j = 1 } ^ { 4 } y _ { j } , \qquad D ( \mathbf { y } ) = \frac { F L } { E } \sum _ { j = 1 } ^ { 4 } y _ { j } .\tag{F.21}
$$

They are affinely normalized using exact extrema over the box:

$$
V _ { \mathrm { m i n } } = 1 2 3 7 . 8 4 1 4 , \qquad V _ { \mathrm { m a x } } = 2 9 9 4 . 9 3 8 3 ,\tag{F.22}
$$

$$
D _ { \operatorname* { m i n } } = 0 . 0 0 2 7 6 1 4 2 , \qquad D _ { \operatorname* { m a x } } = 0 . 0 5 0 5 7 1 9 1 ,\tag{F.23}
$$

$$
g _ { 1 } ( \mathbf { y } ) = \frac { V ( \mathbf { y } ) - V _ { \operatorname* { m i n } } } { V _ { \operatorname* { m a x } } - V _ { \operatorname* { m i n } } } , \qquad g _ { 2 } ( \mathbf { y } ) = \frac { D ( \mathbf { y } ) - D _ { \operatorname* { m i n } } } { D _ { \operatorname* { m a x } } - D _ { \operatorname* { m i n } } } .\tag{F.24}
$$

The final objectives are $f _ { i } ( { \bf x } ) = g _ { i } ( { \bf h } _ { i } ( { \bf x } ) )$ . The reference is (1.05, 1.05). Unlike the color and DTLZ2 reductions, the outer map is essentially a signed sum plus affine scaling, consistent with the small, though significant, gains in Table 2.

## F.4 Latent Langermann–Ackley (Synthetic 2): $d = 6 0 0 , K = 1 2 , N = 2$

Let $A \in \mathbb { R } ^ { 6 \times 6 0 0 }$ have fixed orthonormal rows generated once with seed 18,271, and define the dense latent vector

$$
\begin{array} { r } { { \bf z } ( { \bf x } ) = 8 A ( { \bf x } - \frac { 1 } { 2 } { \bf 1 } ) . } \end{array}\tag{F.25}
$$

The intermediate responses are objective-specific copies, $\mathbf { h } _ { 1 } ( \mathbf { x } ) = \mathbf { h } _ { 2 } ( \mathbf { x } ) = \mathbf { z } ( \mathbf { x } ) \in \mathbb { R } ^ { 6 }$ . Thus the implementation retains $K = 1 2$ scalar responses, which prevents its independent GP implementation from sharing the six latent observations across objectives. For centers

$$
C = \left[ \begin{array} { c c c c c c c } { 1 . 6 } & { - 0 . 9 } & { 0 . 7 } & { 1 . 2 } & { - 1 . 4 } & { 0 . 5 } \\ { - 1 . 3 } & { 1 . 5 } & { - 0 . 8 } & { 0 . 6 } & { 1 . 0 } & { - 1 . 1 } \\ { 0 . 8 } & { 1 . 1 } & { 1 . 4 } & { - 1 . 5 } & { - 0 . 5 } & { 0 . 9 } \\ { - 0 . 7 } & { - 1 . 6 } & { 1 . 0 } & { 1 . 4 } & { 0 . 8 } & { - 0 . 6 } \\ { 1 . 2 } & { 0 . 6 } & { - 1 . 5 } & { - 0 . 9 } & { 1 . 3 } & { 1 . 0 } \end{array} \right] , \quad \mathbf { c } = ( 1 , 2 , 5 , 2 , 3 ) ,\tag{F.26}
$$

define $r _ { \ell } ( \mathbf { y } ) = \| \mathbf { y } - C _ { \ell } \| _ { 2 } ^ { 2 }$ and $\begin{array} { r } { C _ { \Sigma } = \sum _ { \ell = 1 } ^ { 5 } c _ { \ell } = 1 3 } \end{array}$ . The two known mappings are

$$
\begin{array} { l } { { g _ { 1 } ( { \bf y } ) = \frac { C _ { \Sigma } - \sum _ { \ell = 1 } ^ { 5 } c _ { \ell } \exp [ - r _ { \ell } ( { \bf y } ) / \pi ] \cos [ \pi r _ { \ell } ( { \bf y } ) ] } { 2 C _ { \Sigma } } , } } \\ { { g _ { 2 } ( { \bf y } ) = \frac { - 2 0 \exp [ - 0 . 2 \sqrt { \frac { 1 } { 6 } \sum _ { j } y _ { j } ^ { 2 } } ] - \exp [ \frac { 1 } { 6 } \sum _ { j } \cos ( 2 \pi y _ { j } ) ] + 2 0 + e } { 2 0 + e - e ^ { - 1 } } . } } \end{array}\tag{F.27}
$$

(F.28)

The objectives are $f _ { i } ( { \bf x } ) = g _ { i } ( { \bf h } _ { i } ( { \bf x } ) )$ $f _ { 1 }$ is a bounded Langermann-style landscape and $f _ { 2 }$ is normalized Ackley. Their optima conflict: Ackley is minimized at ${ \bf z = 0 }$ , whereas the Langermann centers are displaced. The reference point is (1.1, 1.1).

## F.5 Shallow-lake management: $d = 1 0 0 , K = 2 1 , N = 2$

A town releases phosphorus into a nearby lake each year for 100 years. Releasing more is economically useful, but phosphorus builds up in the lake, and past a threshold the lake flips into a polluted state that is very hard to reverse. The design chooses the 100 yearly release amounts. Each $x _ { t }$ in x controls annual phosphorus release $u _ { t } = 0 . 1 x _ { t }$ over 100 years. The lake is simulated under 100 fixed common-random-number inflow scenarios (s). For scenario s, writing $P _ { t } ^ { ( s ) }$ for the phosphorus in the lake water at year t, we have $P _ { 0 } ^ { ( s ) } = 0$ and for $t = 1 , \ldots , 9 9$

$$
P _ { t } ^ { ( s ) } = 0 . 5 8 P _ { t - 1 } ^ { ( s ) } + \frac { ( P _ { t - 1 } ^ { ( s ) } ) ^ { 2 } } { 1 + ( P _ { t - 1 } ^ { ( s ) } ) ^ { 2 } } + u _ { t - 1 } + \xi _ { t - 1 } ^ { ( s ) } ,\tag{F.29}
$$

where $\xi _ { t } ^ { ( s ) } \sim \mathrm { L o g N o r m a l } ( - 3 . 5 2 , 0 . 1 0 5 ^ { 2 } )$ was drawn once with seed 20,240,820. The first term is the carryover of phosphorus from last year, the third term is the amount the town chose to release last year, and the fourth is uncontrolled inflow from nature. The second term accounts for phosphorus that settled into the lake bed and later returns to the water. Below a critical level the lake clears itself faster than this recycling refills it; above that level it does not, and pollution keeps rising even if releases stop. That threshold in the lake’s phosphorus level is the tipping point. Fixing scenarios makes the oracle deterministic while retaining scenario-averaged variability in the physical response. Let $\bar { P } _ { t } = 1 0 0 ^ { - 1 } \sum _ { s } P _ { t } ^ { ( s ) }$ . The first intermediate response contains 20 five-year block peaks,

$$
h _ { 1 , b } ( \mathbf { x } ) = \frac { 1 } { 2 . 4 } \operatorname* { m a x } _ { t = 5 ( b - 1 ) , \ldots , 5 b - 1 } \bar { P } _ { t } , \quad b = 1 , \ldots , 2 0 .\tag{F.30}
$$

The second intermediate response is normalized discounted utility,

$$
h _ { 2 , 1 } ( \mathbf { x } ) = \frac { \sum _ { t = 0 } ^ { 9 9 } 0 . 9 8 ^ { t } u _ { t } } { 0 . 1 \sum _ { t = 0 } ^ { 9 9 } 0 . 9 8 ^ { t } } .\tag{F.31}
$$

Thus $\mathbf { h } _ { 1 } ( \mathbf { x } ) = \left( h _ { 1 , 1 } , \ldots , h _ { 1 , 2 0 } \right)$ and $\mathbf { h } _ { 2 } ( \mathbf { x } ) = \left( h _ { 2 , 1 } \right)$ . The known mappings, minimized objectives, and reference point are

$$
g _ { 1 } ( \mathbf { y } ) = \operatorname* { m a x } _ { b = 1 , \ldots , 2 0 } y _ { b } , \qquad g _ { 2 } ( \mathbf { y } ) = 1 - y _ { 1 } , \qquad f _ { i } ( \mathbf { x } ) = g _ { i } ( \mathbf { h } _ { i } ( \mathbf { x } ) ) , \qquad \mathbf { r } = ( 1 . 0 5 , 1 . 0 5 ) .\tag{F.32}
$$

This is a deterministic two-objective slice of the shallow-lake dynamics of Carpenter et al. [29] as used in multi-objective robustness studies [20]; it is not the original four-objective stochastic benchmark. The last decision $u _ { 9 9 }$ contributes to utility but, following the implemented indexing, not to the 100 recorded lake states. The composite representation retains which five-year period controls water quality before the outer maximum discards that information.

## F.6 RGB-selective multilayer nanoparticle: $d = 6 , K = 6 , N = 3$

Light partially reflects at every boundary between two different materials, so a particle whose layers alternate between two materials produces several reflections that interfere, reinforcing at some wavelengths and canceling at others. Choosing the layer thicknesses therefore chooses which colors the particle scatters, and the goal is a particle that scatters one target color band and little else. This electromagnetic design problem chooses a core radius and five concentric shell thicknesses. For normalized $\mathbf { x } \in [ 0 , 1 ] ^ { 6 }$ , the implemented dimensions are $t _ { j } ( \mathbf { x } ) = 3 0 + 4 0 x _ { j }$ nm. The layers alternate between silica, with relative permittivity 2.04, and a wavelength-dependent titanium-dioxide model. For wavelength λ measured in nanometers,

$$
\epsilon _ { \mathrm { T i O _ { 2 } } } ( \lambda ) = 5 . 9 1 3 + \frac { 0 . 2 4 4 1 } { 1 0 ^ { - 6 } \lambda ^ { 2 } - 0 . 0 8 0 3 } .\tag{F.33}
$$

This formula returns the relative permittivity of titanium dioxide at wavelength λ, which increases toward the blue end of the spectrum. The surrounding medium has relative permittivity 1.77. The simulator evaluates the concentric-sphere Mie calculation for orders 1–12 and both polarizations on the discrete wavelength grid

$$
\Lambda _ { s } = \{ 3 5 0 + 2 m : m = 0 , \ldots , 2 0 0 \} \ \mathrm { n m } .\tag{F.34}
$$

Mie theory gives the exact scattering from a layered sphere as an infinite series, which the implementation truncates at twelve orders. Given the layer thicknesses $t _ { j } ( \mathbf { x } )$ , the layer and medium permittivities, and the wavelength grid $\Lambda _ { s }$ , the simulator returns a scattering value at each wavelength. The design enters through the layer radii $\begin{array} { r } { r _ { j } ( { \bf x } ) = \sum _ { k < j } t _ { k } ( { \bf x } ) } \end{array}$ , which determine the interference conditions at each interface. The resulting scattering spectrum $S ( \lambda ; { \bf r } ( { \bf x } ) , \epsilon ( \lambda ) ;$ ) therefore depends on the design through the interface radii ${ \bf r ( x ) } = ( r _ { 1 } ( { \bf x } ) , \ldots , r _ { 6 } ( { \bf x } ) )$ and on the material properties through $\boldsymbol { \epsilon } ( \lambda ) = ( \epsilon _ { 1 } ( \lambda ) , \ldots , \epsilon _ { 7 } ( \lambda ) )$ . Specifically, $\epsilon _ { 1 } = \epsilon _ { 3 } = \epsilon _ { 5 } = 2 . 0 4$ for silica, $\epsilon _ { 2 } ( \lambda ) = \epsilon _ { 4 } ( \lambda ) = \epsilon _ { 6 } ( \lambda ) = \epsilon _ { \mathrm { T i O } _ { 2 } } ( \lambda )$ , and $\epsilon _ { 7 } = 1 . 7 7$ represents the surrounding medium; $r _ { j } ( \mathbf { x } )$ is the boundary between regions j and $j + 1$ . It is normalized by the particle cross-sectional area $\pi r _ { 6 } ( { \bf x } ) ^ { 2 }$ . For brevity, we denote this spectrum by $S ( \lambda ; \mathbf { x } )$ below. For target bands $B _ { 1 } = [ 4 0 0 , 5 0 0 )$ $B _ { 2 } = [ 5 0 0 , 6 0 0 )$ , and $B _ { 3 } = [ 6 0 0 , 7 0 0 )$ nm, the implemented in-band and out-of-band responses are the discrete sums

$$
I _ { b } ( { \bf x } ) = \sum _ { \lambda \in \Lambda _ { s } \cap B _ { b } } S ( \lambda ; { \bf x } ) , \qquad O _ { b } ( { \bf x } ) = \sum _ { \lambda \in \Lambda _ { s } \setminus B _ { b } } S ( \lambda ; { \bf x } ) .\tag{F.35}
$$

Here $I _ { b }$ is the scattering that lands inside the target band and $O _ { b }$ is everything else, so the objective below is the fraction of the particle’s scattering that misses the target band, and smaller values mean a more color-selective particle. The intermediate response, known mapping, and minimized objective for band b are

$$
\mathbf { h } _ { b } ( \mathbf { x } ) = { \big ( } I _ { b } ( \mathbf { x } ) , O _ { b } ( \mathbf { x } ) { \big ) } , \qquad g _ { b } ( \mathbf { y } ) = { \frac { y _ { 2 } } { y _ { 1 } + y _ { 2 } } } , \qquad f _ { b } ( \mathbf { x } ) = g _ { b } ( \mathbf { h } _ { b } ( \mathbf { x } ) ) , \quad b = 1 , 2 , 3 .\tag{F.36}
$$

The particle parameterization and scattering simulator follow Kim et al. [23]; the three RGBselectivity objectives are constructed in this work. When $g _ { b }$ is applied to GP posterior samples, the implementation clamps both components to be nonnegative and floors their sum at $1 0 ^ { - 1 2 }$ . Each objective therefore owns a disjoint component pair. The ratio is nonlinear and scale-invariant, but retaining the three $\mathbf { h } _ { b }$ doubles the independent output models from three to six without exposing a component shared across objectives. The reference point is (2.5, 2.5, 2.5).

## F.7 Summit SNAr reaction: $d = 4 , K = 8 , N = 2$

This case implements the mechanistic reaction benchmark distributed with Summit [24]. The starting material is a molecule carrying two replaceable fluorine atoms, and the reagent pyrrolidine displaces one of them. Because either fluorine can be replaced, the reaction yields the desired product, a variant in which the substitution occurred at the other position, and—if a second pyrrolidine reacts—a byproduct with both fluorines replaced.

Four normalized decisions set how long the mixture stays in the reactor, $\tau \in [ 0 . 5 , 2 . 0 ]$ min; how much pyrrolidine is supplied per unit of starting material, $e \in [ 1 , 5 ]$ moles per mole; the concentration of starting material entering the reactor, $c _ { \mathrm { i n } } \in [ 0 . 1 , 0 . 5 ] \mathrm { m o l } \dot { \mathrm { L } } ^ { - 1 } ;$ and the temperature, $T \in [ 3 0 , 1 2 0 ] ^ { \circ } \mathrm { C }$

The state $C = ( C _ { 0 } , \ldots , C _ { 4 } )$ holds the molar concentrations of starting material, pyrrolidine, product, variant, and byproduct, with molar masses $M = ( 1 5 9 . 0 9 , 7 1 . 1 2 , 2 1 0 . 2 1 , 2 1 0 . 2 1 , 2 6 1 . 3 3 ) \mathrm { g m o l ^ { - 1 } }$ From $C ( 0 ) = ( c _ { \mathrm { i n } } , e c _ { \mathrm { i n } } , 0 , 0 , 0 )$ , a fixed 256-step RK4 solve integrates the five-species plug-flow model (a tube reactor without back-mixing) over the residence time. The four rate constants follow the Arrhenius law, $k _ { j } = 0 . 6 k _ { j } ^ { \mathrm { r e f } } \exp [ - E _ { j } \bar { R } ^ { - 1 } ( T _ { \mathrm { K } } ^ { - 1 } - T _ { \mathrm { r e f } } ^ { - 1 } ) ]$ , with activation energy $E _ { j }$ in kJ mol<sup>−1</sup>, gas constant $R = 8 . 3 1 4 \times 1 0 ^ { - 3 } { \mathrm { ~ k J } } { \mathrm { m o l } } ^ { - 1 } { \mathrm { K } } ^ { - 1 } , T _ { \mathrm { K } } = T + 2 7 3 . 7 1$ , and $T _ { \mathrm { r e f } } = 3 6 3 . 7 1 \mathrm { K } ;$ the $( k ^ { \mathrm { r e f } } , E )$ pairs for $a , b , c ,$ d are (57.9, 33.3), (2.70, 35.3), (0.865, 38.9), and $( 1 . 6 3 , 4 4 . 8 )$ . With

$$
\begin{array} { r l r l r l r l r } { r _ { a } = k _ { a } C _ { 0 } C _ { 1 } , } & { { } } & { r _ { b } = k _ { b } C _ { 0 } C _ { 1 } , } & { { } } & { r _ { c } = k _ { c } C _ { 1 } C _ { 2 } , } & { { } } & { r _ { d } = k _ { d } C _ { 1 } C _ { 3 } , } \end{array}\tag{F.37}
$$

the concentration derivatives are

$$
\begin{array} { r l r l } & { \dot { C } _ { 0 } = - r _ { a } - r _ { b } , \qquad } & & { \dot { C } _ { 1 } = - r _ { a } - r _ { b } - r _ { c } - r _ { d } , } \\ & { \dot { C } _ { 2 } = r _ { a } - r _ { c } , \qquad } & & { \dot { C } _ { 3 } = r _ { b } - r _ { d } , } \\ & { \dot { C } _ { 4 } = r _ { c } + r _ { d } . } \end{array}\tag{F.38}
$$

Let $F ( \mathbf { x } ) = 5 / \tau ( \mathbf { x } )$ mL $\cdot \mathrm { m i n ^ { - 1 } }$ be the flow through the 5 mL reactor, and let $\mathbf { C } ^ { \mathrm { { o u t } } } ( \mathbf { x } )$ denote the outlet concentrations produced by the RK4 integration. The retained intermediate responses are

$$
\begin{array} { r } { { \bf h } _ { 1 } ( { \bf x } ) = \big ( C _ { 2 } ^ { \mathrm { o u t } } ( { \bf x } ) , F ( { \bf x } ) \big ) , \qquad { \bf h } _ { 2 } ( { \bf x } ) = \big ( C _ { 0 } ^ { \mathrm { o u t } } ( { \bf x } ) , \dots , C _ { 4 } ^ { \mathrm { o u t } } ( { \bf x } ) , F ( { \bf x } ) \big ) . } \end{array}\tag{F.39}
$$

Thus product concentration and flow occur in both intermediate groups and are modeled twice. For $\mathbf { y } = ( y _ { 1 } , y _ { 2 } ) \in \mathbb { R } ^ { 2 }$ , the first known mapping is one minus normalized space-time yield,

$$
g _ { 1 } ( { \bf y } ) = 1 - \mathrm { c l i p } \bigg ( { \frac { 1 2 M _ { 2 } y _ { 1 } y _ { 2 } } { 1 3 0 0 0 } } , 0 , 1 \bigg ) .\tag{F.40}
$$

For $\mathbf { y } = ( y _ { 1 } , \dots , y _ { 6 } ) \in \mathbb { R } ^ { 6 }$ , define

$$
W ( { \bf y } ) = M _ { 0 } y _ { 1 } + M _ { 1 } y _ { 2 } + M _ { 3 } y _ { 4 } + M _ { 4 } y _ { 5 } ,\tag{F.41}
$$

where $W$ is the outlet waste-mass concentration. The E-factor is

$$
E ( \mathbf { y } ) = \frac { 0 . 7 8 9 y _ { 6 } + 1 0 ^ { - 3 } y _ { 6 } W ( \mathbf { y } ) } { 1 0 ^ { - 3 } M _ { 2 } y _ { 3 } y _ { 6 } } , \qquad g _ { 2 } ( \mathbf { y } ) = \frac { \mathrm { m i n } \{ E ( \mathbf { y } ) , 1 0 0 0 \} } { 5 0 0 } .\tag{F.42}
$$

The final objectives are $f _ { i } ( { \bf x } ) = g _ { i } ( { \bf h } _ { i } ( { \bf x } ) )$ . The implementation clamps modeled concentrations to nonnegative values, floors modeled flow at $1 0 ^ { - 8 }$ , and floors the E-factor denominator at $1 0 ^ { - 1 2 }$ before applying these mappings.

The composition is nonlinear and mechanistic but offers the surrogate little: both outer maps are smooth over the attained states, product concentration and flow are modeled twice, and flow does not affect $f _ { 2 }$ at all. The reference point is (2.5, 2.5).

## F.8 RCM40 power flow: $d = 3 4 , K = 2 9 , N = 2$

RCM40 is the two-objective optimal-power-flow problem from the real-world constrained multiobjective suite [22]. It involves setting the voltages on an electrical grid so that both wasted power and reactive loading are minimized. We only use the implemented hypercube constraints and not the original inequality constraints.

The grid is the standard IEEE-14 test network; its 14 nodes are called buses. The optimizer receives $\mathbf { x } \in [ 0 , 1 ] ^ { 3 4 }$ . Its first 26 coordinates are mapped affinely to [−1, 1] and used as the real and imaginary parts of the voltages at buses 2–14. Bus 1 is the slack bus and is fixed at $V _ { 1 } = 1$ . The final eight coordinates are not read by the implemented oracle, leaving 26 active dimensions.

The admittance matrix $Y = G + \mathrm { i } B \in \mathbb { C } ^ { 1 4 \times 1 4 }$ encoding how strongly each pair of buses is electrically connected, is known in advance. The real matrices G and $B$ contain the conductive and reactive parts of that connection. Multiplication yields the per-bus current and power vectors $I , S \in \mathbb { C } ^ { 1 4 }$ from the voltage vector $V \in \mathbb { C } ^ { 1 4 }$

$$
I = Y V , \qquad S = V \odot \overline { { I } } , \qquad P = \Re S , \qquad Q = \Im S ,\tag{F.43}
$$

where $\odot$ denotes entrywise multiplication, $\cdot ^ { - }$ denotes complex conjugation, and ℜ and ℑ return the real and imaginary parts. Thus $P \in \mathbb { R } ^ { 1 4 }$ and $Q \in \mathbb { R } ^ { 1 4 }$ are the real- and reactive-power injections at the buses in per-unit coordinates.

The intermediate quantities we retain and their known maps are

$$
\begin{array} { l } { { \displaystyle { \bf h } _ { 1 } ( { \bf x } ) = ( P _ { 1 } , \ldots , P _ { 1 4 } ) } , \qquad g _ { 1 } ( { \bf y } ) = \displaystyle \sum _ { j = 1 } ^ { 1 4 } y _ { j } , \qquad } \\ { ~ } \\ { { \displaystyle { \bf h } _ { 2 } ( { \bf x } ) = ( \Im I _ { 2 } , Q _ { 1 } , \ldots , Q _ { 1 4 } ) } , \qquad g _ { 2 } ( { \bf y } ) = - y _ { 1 } + \displaystyle \sum _ { j = 3 } ^ { 1 5 } y _ { j } , \qquad } \\ { { \displaystyle f _ { i } ( { \bf x } ) = g _ { i } \big ( { \bf h } _ { i } ( { \bf x } ) \big ) } , \qquad \quad i = 1 , 2 } \end{array}\tag{F.44}
$$

Here $g _ { 1 }$ sums the real-power injections, giving total real-power loss. The second mapping gives $\begin{array} { r } { f _ { 2 } = - \Im I _ { 2 } + \sum _ { j = 2 } ^ { 1 4 } Q _ { j } ; } \end{array}$ the retained value $Q _ { 1 }$ receives zero weight. Both mappings are linear with constant Jacobians. Their coefficients are all one for $g _ { 1 }$ and are −1, 0, or 1 for $g _ { 2 }$ . The reference point is (200, 500).

## G Considered MOBO methods

For objectives $f _ { i } ( { \bf x } ) = g _ { i } ( { \bf h } _ { i } ( { \bf x } ) )$ , let $X _ { t } ~ = ~ ( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { t } )$ denote the evaluated designs, $F _ { t } ~ =$ $( \mathbf { f } ( \mathbf { x } _ { j } ) ) _ { j = 1 } ^ { t }$ their objective observations, and $H _ { t } \mathbf { \Phi } = ( \mathbf { h } ( \mathbf { x } _ { j } ) ) _ { j = 1 } ^ { t }$ the retained scalar intermediate responses, collected in h. The two modeling approaches differ in the posterior samples supplied to the acquisition rule. BlackBox- $. G P$ fits independent GPs to the final objectives, giving samples

$$
\widetilde { f } _ { i } ( \cdot ) \sim p _ { i } ( f _ { i } ( \cdot ) \mid X _ { t } , F _ { t , i } ) , \qquad i = 1 , \ldots , N .\tag{G.1}
$$

Composite instead fits independent GPs to the K retained scalar intermediate responses and propagates their posterior samples through the known mappings:

$$
\widetilde { h } _ { k } ( \cdot ) \sim p _ { k } \left( h _ { k } ( \cdot ) \mid X _ { t } , H _ { t , k } \right) , \quad k = 1 , \ldots , K , \qquad \widetilde { f } _ { i } ( \mathbf { x } ) = g _ { i } \left( \widetilde { \mathbf { h } } _ { i } ( \mathbf { x } ) \right) .\tag{G.2}
$$

Here $F _ { t , i }$ and $H _ { t , k }$ denote the corresponding observed output columns, and $\widetilde { \mathbf { h } } _ { i }$ collects the responses used by $g _ { i }$ . Shared or duplicated responses are retained as specified in Appendix F. The induced Composite objective posterior need not be Gaussian. Both variants use the same acquisition rule and observed objectives to measure improvement; MORBO applies these modeling choices locally within trust regions. All definitions below use the minimization convention.

## G.1 Log Expected Hypervolume Improvement (LogEHVI) [17, 16]

Expected hypervolume improvement (EHVI) [17] selects designs according to their expected contribution to the hypervolume of the observed Pareto front. Let $\mathcal { P }$ denote the nondominated objective vectors observed so far and r the fixed reference point. For our minimization problems, the hypervolume improvement from evaluating x is

$$
\begin{array} { r } { \Delta \mathrm { H V } ( \mathbf { x } ) = \mathrm { H V } ( \mathcal { P } \cup \{ \mathbf { f } ( \mathbf { x } ) \} ; \mathbf { r } ) - \mathrm { H V } ( \mathcal { P } ; \mathbf { r } ) . } \end{array}\tag{G.3}
$$

EHVI averages this improvement over the posterior distribution of $\mathbf { f } \left( \mathbf { x } \right)$ . Its logarithmic counterpart targets

$$
\alpha _ { \mathrm { L o g E H V I } } ( \mathbf { x } ) = \log \mathbb { E } [ \Delta \mathrm { H V } ( \mathbf { x } ) ] .\tag{G.4}
$$

Taking the logarithm preserves the maximizers whenever EHVI is positive. However, directly comput ing EHVI and then taking its logarithm can suffer from numerical underflow and vanishing gradients. We therefore use BoTorch’s qLogEHVI implementation [16], which evaluates a numerically stable, smooth approximation in log space. We set $q = 1$ , selecting one design at each iteration.

## G.2 Smooth Tchebycheff scalarization (STCH) [11, 12]

Smooth Tchebycheff scalarization combines the N objectives into the single scalar objective $s _ { \mu } ( \mathbf { f } ( \mathbf { x } ) ; \mathbf { w } )$ ), where w specifies the tradeoff among the objectives. For weights $w _ { i } ~ \geq ~ 0$ with $\textstyle \sum _ { i = 1 } ^ { N } w _ { i } = 1$ , reference vector $\mathbf { z } ^ { \ast }$ , and temperature $\mu > 0$ , define

$$
s _ { \mu } ( \mathbf { f } ( \mathbf { x } ) ; \mathbf { w } ) = \mu \log \left[ \sum _ { i = 1 } ^ { N } \exp \left( \frac { w _ { i } ( f _ { i } ( \mathbf { x } ) - z _ { i } ^ { * } ) } { \mu } \right) \right] .\tag{G.5}
$$

Let $s _ { \mathbf { w } } ^ { \mathrm { b e s t } }$ be the smallest scalarized value observed in the branch for weight w. Its LogEI acquisition targets

$$
\alpha _ { \mathrm { L o g E I } } ( \mathbf { x } ; \mathbf { w } ) = \log \mathbb { E } \left[ \left( s _ { \mathbf { w } } ^ { \mathrm { b e s t } } - s _ { \mu } ( \widetilde { \mathbf { f } } ( \mathbf { x } ) ; \mathbf { w } ) \right) _ { + } \right] ,\tag{G.6}
$$

where $( a ) _ { + } = \operatorname* { m a x } \{ a , 0 \}$ and the expectation is over the BlackBox-GP or Composite objective posterior. Maximizing this acquisition selects the next design for that weight. All STCH variants use temperature $\mu = 0 . 0 5$ and five fixed weights. For two objectives, $w _ { 1 }$ is equally spaced from 0.05 to 0.95 and $w _ { 2 } = 1 - w _ { 1 }$ ; for more objectives, deterministic normalized exponential draws cover the simplex. The five scalarizations with different weights share one Sobol initial design, evaluated only once, and then evolve independently. The BlackBox-GP STCH variant applies the negative of the scalarization in Eq. $( \mathbf { G } . 5 ) , - s _ { \mu } ( \widetilde { \mathbf { f } } ; \mathbf { w } )$ , to each objective posterior sample ${ \widetilde { \mathbf { f } } } .$ The Composite STCH variant first constructs $\widetilde { \mathbf { f } } = \left( g _ { i } ( \widetilde { \mathbf { h } } _ { i } ) \right) _ { i = 1 } ^ { N }$ and then applies the same $- s _ { \mu } ( \widetilde { \mathbf { f } } ; \mathbf { w } )$ . Both variants maximize qLogEI (q = 1) [16] relative to the best scalarized value observed for that weight vector. Low-dimensional branches receive eight adaptive evaluations each, while the high-dimensional STCH branches receive twenty. For reporting only, their points are interleaved in round-robin order so the HV trace does not depend on an arbitrary ordering of the five weights.

## G.3 Spherical-linear STCH (Sp-STCH) [25]

Sp-STCH retains the STCH weights, independent branches, composition, and qLogEI steps above, but replaces every standard output GP with the following spherical-linear model. For each output, the implementation centers x at $( 1 / 2 , \ldots , 1 / 2 )$ , divides coordinates by learned ARD scales, and applies a learned global radial scale before constructing

$$
P ( \mathbf { u } ) = \left[ \sqrt { b _ { 0 } } , \ \sqrt { b _ { 1 } } \frac { \mathbf { u } } { \| \mathbf { u } \| _ { 2 } } \right] , \qquad k ( \mathbf { u } , \mathbf { u } ^ { \prime } ) = P ( \mathbf { u } ) ^ { \top } P ( \mathbf { u } ^ { \prime } ) ,\tag{G.7}
$$

with the zero-radius direction defined as zero. A softmax constrains $b _ { 0 } + b _ { 1 } = 1$ and $b _ { 0 } , b _ { 1 } > 0$ The GP has a constant mean, a learned Gaussian likelihood with a log-normal noise prior, and standardized outputs, following the original spherical-linear implementation [25]. Posterior objective samples from these models are scalarized using Eq. (G.5), and the next design maximizes the LogEI acquisition in Eq. (G.6). Thus, only the surrogate changes relative to STCH.

## G.4 Multi-Objective Regional Bayesian Optimization (MORBO) [26]

MORBO maintains multiple adaptive trust regions $\mathcal { T } _ { \ell } \subseteq \mathcal { X }$ and fits local GP models using observations near each region. For a candidate pool $\mathcal { C } _ { \ell } \subset \mathcal { T } _ { \ell } .$ , a Thompson sample $\widetilde { \mathbf { f } } _ { \ell }$ assigns a sampled objective vector to each candidate. The sampled hypervolume improvement relative to a current nondominated set $\mathcal { P }$ is

$$
\Delta \widetilde { \mathrm { H V } } _ { \ell } ( \mathbf { x } ; \mathcal { P } ) = \mathrm { H V } \Big ( \mathcal { P } \cup \{ \widetilde { \mathbf { f } } _ { \ell } ( \mathbf { x } ) \} ; \mathbf { r } \Big ) - \mathrm { H V } ( \mathcal { P } ; \mathbf { r } ) , \qquad \mathbf { x } \in \mathcal { C } _ { \ell } .\tag{G.8}
$$

Coordinated batch selection uses these sampled improvements across regions, accounting for candidates already selected for the batch. Successful regions expand, while unsuccessful regions shrink or restart.

The BlackBox-GP MORBO variant fits local GPs to final objectives, whereas the Composite MORBO variant fits local GPs to the retained scalar entries of $\mathbf { h } _ { i } ( \mathbf { x } )$ and maps each posterior sample $\widetilde { \mathbf { h } } _ { i }$ through $g _ { i }$ before scoring candidate hypervolume improvement. Both variants begin at the same points and use the same region state, shared past observations, and batch size. There are five trust regions, each initialized with side length 0.8, minimum 0.01, and maximum 1.6. Candidate pools contain 512 raw points. Each local output GP uses a Matérn-5/2 ARD kernel with length scales constrained to [0.05, 4], output scale, and a Gaussian likelihood with a Gamma noise prior. Candidates are then selected via lower-tail winsorization and coordinated Thompson Sampling/LogEHVI.

## H Optimization configurations

Protocol. Each matched BlackBox-GP/Composite pair on the benchmarks uses the same trial seed and an identical scrambled-Sobol initial design, and every benchmark comparison is repeated over 10 paired trials, with 80 trials for Synthetic 1. One evaluation at a design returns all objectives, or all retained components, at once; applying the known composition to those components does not consume a further evaluation. Hypervolume is recomputed after each recorded design against the benchmark’s fixed reference point.

Models. All models are double-precision BoTorch and GPyTorch GPs [30]. Standard and spherical models are fit with fit\_gpytorch\_mll using exact or summed marginal log likelihoods; the configured kernel and likelihood priors enter that same fitting objective.

Surrogate models. The LogEHVI and STCH output GPs use a scaled Matérn 5/2 kernel with ARD in the controlled experiments and a dimension-scaled RBF kernel with ARD in the scientific-design benchmarks. Sp-STCH and MORBO set their covariance explicitly, as Appendices G.3 and G.4 describe. Table H.1 gives both configurations; absolute gains should be compared within a setting rather than across the two.

Table H.1: Surrogate and software configuration for the LogEHVI and STCH output GPs in each setting. Both fit hyperparameters by fit\_gpytorch\_mll in double precision with a Standardize outcome transform.
<table><tr><td></td><td>Controlled experiments</td><td>Scientific-design benchmarks</td></tr><tr><td>BoTorch</td><td>0.9.5</td><td>0.18.0</td></tr><tr><td>GPyTorch</td><td>1.11</td><td>1.15.2</td></tr><tr><td>PyTorch</td><td>2.11.0</td><td>2.12.0</td></tr><tr><td>Covariance</td><td>ScaleKernel(Matern 5/2), ARD</td><td>RBFKernel, ARD, no outputscale</td></tr><tr><td>Lengthscale prior</td><td>Gamma(3.0, 6.0)</td><td>LogNormal(√2 + − log d, √3)</td></tr><tr><td>Lengthscale constraint</td><td></td><td>≥ 0.025</td></tr><tr><td>Outputscale prior</td><td>Gamma(2.0, 0.15)</td><td></td></tr></table>

Evaluation budgets. Table H.2 gives the budget for each regime. The scalarization methods spend their adaptive budget across five fixed weight vectors rather than in one sequence, which is why the per-weight counts multiply out to the same totals as the sequential methods.

Table H.2: Initial design size and adaptive evaluation budget per trial. The high-dimensional regime covers L–A, Lake, and RCM40.
<table><tr><td>Regime</td><td>Method</td><td>Initial</td><td>Adaptive</td><td>Adaptive budget spent as</td></tr><tr><td rowspan="2">Low-dimensional</td><td>LogEHVI</td><td>5</td><td>40</td><td>sequential, one candidate per step</td></tr><tr><td>STCH</td><td>5</td><td>40</td><td>5 weights × 8 evaluations</td></tr><tr><td rowspan="2">High-dimensional</td><td>Sp-STCH</td><td>20</td><td>100</td><td>5 weights × 20 evaluations</td></tr><tr><td>MORBO</td><td>20</td><td>100</td><td>20 batches × 5 candidates</td></tr></table>

Acquisition optimization. LogEHVI, STCH, and Sp-STCH maximize their acquisition function with BoTorch’s optimize\_acqf, which seeds a multistart search from raw Sobol candidates and refines it with SciPy’s L-BFGS-B [30]. LogEHVI and STCH use 128 raw candidates with 8 restarts, and Sp-STCH uses 256 raw candidates with 10 restarts. MORBO does not use L-BFGS-B: it sequentially selects twenty batches of five candidates from a discrete Sobol pool of 512 points drawn inside its trust regions.