# ActMOF: Mapping Reticular Synthesis through Active Learning

Hongrui Sheng1,2, Shutian Liang3, Reid A. Coyle1, Shyam Chand Pal1, Peter Walther1, Bin Feng1,4, Xinyu Ding1,5, Zhiling Zheng1,4,\*

1Department of Chemistry, Washington University, St. Louis, Missouri 63130, United States

2Department of Chemistry, Fudan University, Shanghai 200438, China

3Department of Energy, Environmental and Chemical Engineering, Washington University, St. Louis, Missouri 63130, United States

4Institute of Materials Science and Engineering, Washington University, St. Louis, Missouri 63130, United States

5Department of Chemistry, Imperial College London, London W12 0BZ, United Kingdom

\*Corresponding author. Email: z.z@wustl.edu

## Abstract

Reticular chemistry allows metal-organic frameworks (MOFs) to be designed from their building units, yet their synthesis remains constrained by sparse experimental data, noisy crystallization outcomes, and the absence of benchmarks for comparing active-learning strategies. While data-driven approaches such as Bayesian optimization (BO) can improve synthesis success rates, selecting appropriate optimizers beforehand for mapping MOF crystallization landscapes remains largely empirical. Here we present ActMOF, an experimentgrounded benchmark for comparing search strategies in aluminum-carboxylate MOF synthesis. ActMOF is emulated from experimentally measured microwave syntheses with each encoded condition assigned a crystallinity index calculated based on diffraction peak intensity and full width at half-maximum. By benchmarking policies guided by Gaussian processes (GP), tree ensembles and large language models (LLM), ActMOF placed the GP family among the leading policies for active learning in sparse synthesis landscapes. It then demonstrated transferability by guiding the microwave synthesis of six reticularly related aluminum MOFs. All six campaigns gave crystalline products within a fixed budget of 12 experiments each, and established a reusable testbed for AI-driven MOF synthesis.

Keywords: Reticular synthesis, metal-organic frameworks, large language models, active learning, crystallinity

## Introduction

The discovery of crystalline reticular materials is an experimental science governed by many variables that are individually simple yet collectively nontrivial.1-6 Reticular synthesis, the assembly of molecular building units into predetermined ordered frameworks held together by strong bonds,6 makes the structure of a metal-organic framework (MOF) a matter of design.7-12 In its isoreticular form, the net and the secondary building unit (SBU) are kept while the linker is lengthened or functionalized, as first shown for the IRMOF series and later extended to the large-pore IRMOF-74 family; 13,14 in its multivariate (MTV) form, 15,16 several linkers share one underlying topology.15,17 Despite progress in rational design, the synthesis of the intended crystalline phase depends on concentration, modulator identity and amount, solvent environment, time, temperature, and nucleation kinetics, 18–24 and as reticular space expands, how to navigate synthesis becomes as pressing as what to synthesize. In light of this complexity, the most consequential decisions in MOF synthesis are often made not at the selection of framework based on properties alone,25,26 but at the level of synthesizable materials verification and reaction condition discovery.27-34 Hitherto, this condition-search problem has been addressed by chemical intuition24,31,35, literature analogy13,36, and basic design-of-experiments37,38 campaigns.2,39–41 A chemist can rationally design the net, node, and linker, yet still face a narrow crystallization window in which a small change in acidity, dilution, or thermal dose transforms a crystalline product into amorphous solid, side phase, or unreacted precursor.32,34,42–45

A human-centered machine learning (ML) strategy for synthesis should therefore be judged in a manner aligned with how chemists actually make decisions and provide early confidence for navigating vast synthesis space.46-48 Active learning (AL) offers an iterative strategy that mimics how chemists explore chemical space by refining experimental choices in response to prior outcomes.49-53 Concretely, active learning proceeds as an iterative predict-synthesizemeasure-update loop: a surrogate model, a computationally cheap predictive model, proposes the condition or batch of conditions expected to be most informative or most likely to improve crystallinity, that proposal is then evaluated, either against the experiment-grounded emulative landscape in the dry lab or by real microwave synthesis and powder X-ray diffraction (PXRD) measurement in the wet lab, and the observed outcome is fed back to update the surrogate before the next round of proposals (Figure 1a). Active learning therefore operates on both sides of this work: dry-lab active learning is used to benchmark the search behavior of candidate policies against the emulative landscape, while wet-lab active learning subsequently guides and revises real microwave synthesis campaigns for MOFs whose crystallization behavior lies outside the original benchmark. In lieu of being outcome-oriented, MOF synthesis screening is inherently process-driven, wherein chemists' confidence must be accumulated throughout the search itself. 31,54 A practical closed-loop workflow should accordingly provide (i) early promising signs, as each iteration consumes reagents, instrument time, and human attention55-57; (ii) steady improvement, because a modest but visible change in crystallinity can keep a researcher willing to continue the campaign52,53,58; and (iii) interpretable suggestions59–61, as a proposed reaction condition that conflicts strongly with chemical intuition will often be deprioritized by synthetic chemists unless supported by compelling chemical or statistical rationale.

Herein, we report ActMOF as an experiment-grounded benchmark emulated from 95 measured microwave syntheses spanning various synthesis parameters, on which synthesis hypotheses can be quantitatively compared. Built on a prior microwave synthesis campaign,33 this dataset expanded into a hypothetical reaction landscape of 6,101,172 encoded synthesis conditions by combining experimental reaction parameters, PXRD-derived outputs, machine-learning emulation, and local experimental-neighbor rules (Results and Supporting Information, Section S2). We then investigated Gaussian processes (GPs),52,53,62,63 statistical ensembles,33,52,64–66 large language model (LLM)-only suggestions,67–70 and hybrid GP-LLM workflows67,71–73 under an identical active-learning protocol to identify which model suggestions of reaction conditions best navigate the sparse crystallization landscapes of MOFs. To this end, the selected strategies are subsequently transferred, following the isoreticular principle of keeping the rod-shaped Al-OH SBU while changing the linker, 6,13,74–77 to CAU-2375, MIL-16078, CAU-1079,80, MIL-53-muc81, MTV-MOF-30382,83, and $\mathrm { M T V - M I L - 1 6 0 ^ { 7 4 , 8 4 , 8 5 } }$ whose microwave synthesis has not been systematically optimized. This application therefore supports the transferability of benchmark-guided active learning across related reticular synthesis systems and sheds light on how statistical optimization and chemical intuition can be synergistically integrated for actual MOF crystallinity discovery.

![](images/8590dab32509270997a25e4db87638248ddf97837973345415e39015cd6ab436.jpg)  
Figure 1: Overview of the ActMOF active-learning benchmark. (a) Closed-loop active-learning workflow, in which proposed conditions are evaluated on the synthesis landscape (dry lab) or by microwave synthesis and PXRD (wet lab). (b) Structure of MOF-321, [Al(OH)(PZVDC)], with its linker H2PZVDC and rod-shaped Al-SBU. (c) The 95 measured MOF-321 syntheses across the five encoded reaction parameters, colored by experimental q. (d) Definition of q as peak height over full width at half-maximum (FWHM) of the diagnostic reflection, shown on a representative PXRD pattern. For illustrative clarity: Al is represented by deep blue octahedra; O, pink; C, light gray; N, green. H atoms are omitted.

## Results and Discussion

At the onset of our study, we selected aluminum-based MOF-32133 (Figure 1b) as the model reaction system because it adopts reproducible and programmable microwave synthesis, 86–88 has strong relevance to rich carboxylate-based reticular chemistry, , 7,9,89,90 and carries a reaction space too vast for manual brute-force screening. In particular, the encoded synthesis space contains five variables (Figure 1c and Supporting Information, Table S5), which include aluminum precursor amount, modulator amount, added solvent volume, microwave reaction time, and microwave reaction temperature, adding up to 95 deduplicated data entries.

The primary experimental observable is PXRD crystallinity. In this study, we define the quality score q as the ratio of the intensity (I) of a diagnostic diffraction peak to its full width at half-maximum (FWHM, Figure 1d). The enhanced crystalline formation is associated with stronger and sharper diffraction features, especially at low angles where framework periodicity is readily visible,2,4,91–93 and, for MOF chemists, the crystallinity index q is usually considered as a first-stage synthesis metric, suitable for ranking crystallization outcomes during screening, whereas stability and sorption performance remain necessary downstream validations (Supporting Information, Table S3).2,9,41,94,95

Starting from a set of 95 unique measured reaction outcomes, we developed a rule-guided statistical emulator (Supporting Information, Section S2.4), in which crystallization neighborhoods were defined using a Chebyshev-distance rule over the five encoded synthesis variables, 96-100 to construct a grid of 6,101,172 candidate conditions. In essence, we note that ActMOF is not intended to declare the experimental outcome of every unmeasured condition (Supporting Information, Section S2.5). Inherently, one should consider it as an experiment-grounded reaction landscape designed to preserve the qualitative structure of the synthesis campaign, including sparse success, local neighborhoods of useful conditions, and broad low-quality regions, which are necessary for comparing AL search strategies. The emulator itself is built from random forests, so tree-based policies, which share this model family, might find the landscape unusually easy; in practice, they gained no measurable advantage over the Gaussian-process policies (Supporting Information, Section S4.4).

The global structure of ActMOF is shown in Figure 2. High-q regions, where top 1% q has a threshold of 42,463 and 0.1% q is higher than 100,288, were enriched in comparatively low metal loading, low modulator amount, long microwave heating time, and elevated temperature, with added water mattering less significantly (see Figure 2b). In particular, in a chemically interpretable sense, the resulting landscape suggests an insufficient base amount may leave the linker protonated, moderate dilution may improve nucleation, and an inappropriate thermal dose may produce amorphous solid or side phases. In fact, the sparse-success landscape and highly skewed distribution101 of crystallinity index q is consistent with the nature of MOF crystallization where most conditions are poor and the experimenter often encounters abrupt transitions between no product, poorly crystalline product, side phases, and the desired crystalline phase. In contrast to computational property-optimization tasks, 102–107 where smooth movement along a continuous surface gradually improves performance, such difference reveals the type of rugged basins of crystalline products that AL needs to navigate and has direct consequences for which optimizer should be adopted in the laboratory (Supporting Information, Section S2.6).

With the dataset in hand, we compared 39 policies (Supporting Information, Table S13) under conditions that mimic practical synthesis screening. Each run was initialized with three starting experiments drawn at random and different for each method and run, followed by iterative batches of three proposed experiments (Supporting Information, Section S3.1). The total budget was 120 experiments, corresponding to 39 active-learning iterations after initialization. The objective was the cumulative best q found by the search (Figure 3), and the surrogate models were trained on logarithmic scale to reduce the undue influence of rare high-q values and emphasize relative crystalline enhancement across the synthesis landscape. In addition, we evaluated the trajectory of improvement, hit rates for the top 1% and top 0.1% of the landscape, and normalized regret (Supporting Information, Section S4.1).

Notably, in practical laboratory MOF screening, a bench chemist rarely continues a campaign out of faith in asymptotic convergence after over a hundred experiments; rather, confidence must be accumulated from the first few batches of PXRD patterns. To this end, cumulative best-q trajectories in ActMOF distinguish the approaches showing early confidence with those demonstrating high terminal performance. Once q exceeds top 0.1% threshold (\~100,000), the product already corresponds to sharp and intense diffraction features that many synthetic chemists would regard as operationally successful, even if subsequent optimization can still refine crystal quality.

In the first five iterations, probability of improvement (PI), which scores a candidate by its chance of exceeding the current best, behaved slightly more exploitatively than expected improvement (EI), which scores it by the size of the gain it is expected to deliver. In the benchmark runs, the best-q-so-far of PI led that of EI at each of these iterations. This lead, however, came from the starting batches rather than from the acquisition rule: 7 of the 20 PI starting batches already contained a crystalline product against 1 of 20 for EI, and when both rules were rerun from identical starts, their early trajectories overlapped (Supporting Information, Figure S19). Conceptually, in the early stage of MOF microwave synthesis, uncertainty alone is not a reliable proxy for reaction promise in a sparse crystallinity landscape. A condition far from previous experiments may be statistically uncertain yet chemically unproductive, because it lies outside the window for metal hydrolysis, linker deprotonation and framework nucleation. For MOF synthesis, where the first goal is to locate a robust crystalline window before fine optimization, acquisition rules that weight predicted crystallinity over predictive variance are therefore advantageous, a property PI and EI share and one that separates both from uncertainty-driven sampling. The two nonetheless differ in what they reward. PI treats every improvement over the current best as equal, however small; EI asks how large that improvement is expected to be. On the certain map, where consulting a point is free, this difference barely matters. When each query is a microwave synthesis whose q carries a \~30% repeat-loading scatter, it does. With noise added, PI lost 8.8% of its final best q and EI 0.3% (Supporting Information, Figure S20) and EI, the criterion least affected by noise, was taken forward.

![](images/c55e112bec27908409a773aafce84e94e2ba179ac11baa80d10bb9f3c7f65790.jpg)

![](images/e1e85e80579eeb90181ccadd8a5ba72728c8c3fccbe42faca66e2d57c6dbd426.jpg)  
Figure 2: Synthesis landscape and parameter distributions across q tiers. (a) Uniform Manifold Approximation and Projection (UMAP) of a 1% stratified random subsample of \~6.1M enumerated synthesis conditions, colored by q percentile tier. Zero-crystallinity conditions $( q = 0 )$ are shaded by temperature-metal quadrant, split at the dataset medians: $\mathrm { T } < 1 0 0 \ ^ { \circ } \mathrm { C }$ (Low T) or $\geq 1 0 0 ~ ^ { \circ } \mathrm { C }$ (High T), and metal < 1.66 equiv (Low Metal) or ≥ 1.66 equiv (High Metal). Blue points mark the 95 experimental ground-truth conditions. (b) Ridgeline kernel density estimates (KDE) of each q tier along the five synthesis parameters, from a 2% random subsample within each tier; each curve is normalized to its maximum. The favorable window narrows with increasing tier stringency. Metal and modulator equivalents are mmol of $\mathrm { A l C l _ { 3 } { \cdot } 6 H _ { 2 } O }$ and $\mathrm { N a O H }$ respectively, per 1 mmol of linker; added solvent is deionized water. UMAP and KDE settings are given in Supporting Information, Section S2.6.

The ActMOF benchmark also supports the continued use of common GP kernels for MOF synthesis. Matérn-based ${ \mathrm { G P s } } ,$ 108-–110 especially those coupled to PI or EI acquisition functions, remained among the most reliable AL performers in ActMOF, which is encouraging because these kernels are already widely used in vanilla Bayesian optimization.111 This outcome is chemically reasonable because the ActMOF landscape contains local neighborhoods of related crystallization behavior, while still retaining abrupt changes caused by coupled concentration, base, solvent, time, and temperature effects. In this regard, a Matérn kernel is well matched to a landscape that is neither fully smooth nor purely discontinuous, although radial basis function (RBF) and rational-quadratic kernels performed comparably (Supporting Information, Table S15). Figure 4 follows the Matérn-5/2 EI policy as it settles on long hold times and high temperatures; its learned length scales rank the metal amount as the most sensitive variable and added water as nearly irrelevant (Supporting Information, Figure S10). At the same time, tree-based uncertainty models provided a complementary advantage. Paired with EI, a conformal random forest matched the Matérn-5/2 GP hit rate for the top-1% region within the first 7 iterations, and a tree-ensemble surrogate (BART-like ExtraTrees) reached the top-1% region in a median of 4 iterations (Supporting Information, Section S4.2). This is consistent with their ability to capture threshold-like rules such as minimum temperature for framework formation, moderate dilution improving nucleation, or narrow metal/base windows. An alternating GP-LLM proposer stayed with the leading GP curves without overtaking them, and LLM-only proposals lagged both in mean best q; language models are therefore kept as a control in Figure 3 (Supporting Information, Table S15). Thus, ActMOF shows that statistical models and rule-like ensemble models can both be useful, but for different chemical reasons.

![](images/b6c41342ba6a9844d6411d80b467c845a6bb5e0defd854eb59c483e37a23c31b.jpg)  
Figure 3: Benchmark comparison of active-learning policies on the ActMOF landscape. Cumulative best q is plotted against iteration (three random starting experiments, then 39 batches of three; 120 experiments in total) and averaged over five runs per policy, each run starting from its own random experiments. Shaded bands are 95% confidence intervals of the mean for random search and Matérn-5/2 PI. Shown are 16 of the 39 policies: Gaussian processes (Matérn-5/2 or Matérn-3/2 kernel; PI or EI), GP-LLM hybrids, tree ensembles (conformal random forest; Bayesian additive regression tree (BART)-like ExtraTrees), local approximate GPs (laGP), linear bandits (LinUCB), LLM-only proposers (GPT-5.5, GPT-4o, GPT-5-mini) and random search (Supporting Information, Table S13). The right axis normalizes each curve to the highest plotted q. Inset: iterations 0–5 for Matérn-5/2 PI and EI

Having identified a leading search behavior in ActMOF, we next asked whether the benchmarkguided strategy could be transferred to new MOF syntheses. To this end, we selected expectedimprovement Matérn GP (Supporting Information, Section S5.1) for experimental microwave synthesis of CAU-10, CAU-23, MIL-160, MIL-53-muc, MTV-MIL-160, and MTV-MOF-303 (Figure 5). None of the six reticularly related MOFs has been systematically optimized under microwave conditions.75,78,112 These materials were further chosen because they preserve a common aluminum-based reticular chemistry while introducing chemically meaningful differences in organic linker solubility, hydrophilicity, and carboxylate angularity.74,82–85 Accordingly, they pose the isoreticular question for synthesis that with the SBU kept and the linker changed, whether a benchmark-guided optimizer can transfer without assuming that the same reaction coordinate remains optimal.

GP Matérn52 + El: Selected Conditions by Iteration  
![](images/193a1b88a0f7da519dd44f5ae101770a16006a0811cb524fd3efdba4d53fb318.jpg)  
Figure 4: Selected experimental conditions and cumulative best q for a Matérn-5/2 Gaussian process with expected improvement (batch size = 3; mean of five runs). Heatmap rows correspond to the five decision variables (metal, modulator, additional solvent, reaction time, temperature); color encodes the normalized selected value at each iteration. The pink curve with right axis shows best q so far; the shaded band is the 95% confidence interval of the mean.

Notably, rather than starting each new MOF crystallinity optimization from scratch, the validation workflow used the MOF-321 data behind ActMOF as a calibrated transfer prior, where a reference GP first learns synthesis trends from the 95 measured MOF-321 syntheses, including the 28 without product, then a residual GP adapts those trends to the measured PXRD responses of the MOF candidate (Supporting Information, Section S5.2). Even the three conditions used to initialize each MOF's campaign were themselves proposed from this MOF-321- anchored prior rather than drawn at random, and each subsequent round refits the recalibration between the reference prediction and the accumulating real measurements for that specific MOF, reshaping the local landscape as evidence arrives (Supporting Information, Section S5.3). On the MOF-321 landscape, starting from these three prior conditions placed every run in the top 1%; from there the leading policies climbed to the top 0.1% in 65–100% of runs by iteration 13, while random search from the same start remained at its starting q in the median (Supporting Information, Table S17). This design directly tests whether ActMOF teaches a transferable search principle across various MOFs with underexplored microwave reaction conditions rather than a memorized numerical crystallinity surface of one particular MOF. We note that such transferability is particularly important for the two multivariate reticular systems, MTV-MIL-160 and MTV-MOF-303, because the coexistence of multiple linker environments inherently amplifies the complexity of nucleation and crystal growth, and microwave reaction of multivariate reticular materials remains far less established than that of their single-linker analogues. The ability to initialize such a campaign from a calibrated prior, while allowing new experimental residuals to rapidly overwrite system-specific errors, provides a practical route for reticular synthesis in which linker variation does not require a completely de novo search.

Indeed, the transfer-learning workflow proved effective for MTV-MOF-303 (Figure 5), which represents one of the two multivariate cases among the six validation systems. Starting from three prior-informed experiments, the search reached a phase-matched MTV-MOF-303 pattern within the 12-experiment budget, while best q then rose only modestly (1.23-fold) over the remaining rounds. Importantly, the search did not merely reproduce the optimal MOF-321 synthesis condition: its best condition lies inside the MOF-321 top-1% tier but at no MOF-321 optimum (Supporting Information, Table S11). Rather, the transferred prior accelerated identification of productive crystallization neighborhoods, whereas the residual adaptation layer corrected for framework-specific linker behavior and nucleation kinetics from that prior. Each proposal is returned as a laboratory understandable prescription, including metal precursor amount, modulator equivalents, solvent volume, reaction time and temperature, so a chemist can accept and annotate it before the run (Supporting Information, Tables S20–S25). A parallel campaign on MTV-MIL-160 followed the same budget and likewise ended in agreement with the simulated phase, further supporting transfer of the benchmark-guided policy to multivariate reticular chemistry.

To further demonstrate the broader applicability of benchmark-guided AL, the same transfer strategy was then applied to CAU-10, CAU-23, MIL-160, and MIL-53-muc. Gain within each campaign is the ratio of the best $q$ in the 12-run budget to the best q among that campaign's own three starting experiments. Because those starts were already prior-informed, this is a conservative denominator; the average increase ratio across the six frameworks is 2.2-fold (Supporting Information, Table S19). CAU-23 never exceeded its starting best q, and that starting solid already matched literature quality, whose Brunauer-Emmett-Teller (BET) surface area has reached 1173 $\mathrm { m ^ { 2 } / g }$ within $6 \%$ of the reported value for this framework75 (Supporting Information, Table S26). The remaining campaigns still moved the needle, about 1.8-fold for MIL-160, MIL-53-muc and MTV-MIL-160, and 5.4-fold for CAU-10. The self-correction had a visible price once: for CAU-10 and CAU-23, the first recalibration inverted the MOF-321 trends, and the round-2 batch at $6 0 { - } 6 5 \ ^ { \circ } \mathrm { C }$ gave little or no product before the calibration recovered in the next round (Supporting Information, Section S5.3). In every case the final PXRD matched the simulated target phase (Figure 5). Early confidence therefore arrives with the first batch, including the multivariate starts: all 18 starting experiments crystallized, as did 67 of the 72 experiments overall (Supporting Information, Figure S24); steady improvement is visible on the next few PXRD patterns rather than only at the end of the budget.

Chemically, all six best recipes stayed in the window that MOF-321 had mapped, with little metal (0.66–0.83 equiv $\mathrm { A l C l _ { 3 } { \cdot } 6 H _ { 2 } O ) }$ , little base (1.5–1.75 equiv NaOH) and long holds of 50–60 min, so the general rules of Al-carboxylate crystallization carried over (Supporting Information, Table S19). Notably, the two frameworks containing isophthalate, CAU-10 and MTV-MIL-160, converged on the same recipe at $1 4 0 ~ ^ { \circ } \mathrm { C }$ , whereas those built from five-membered heteroaromatic linkers (MIL-160, CAU-23 and MTV-MOF-303) and from linear muconate (MIL-53-muc) peaked at $1 0 0 { \mathrm { - 1 1 0 ~ } } ^ { \circ } \mathrm { C } ,$ on either side of the $1 2 0 ~ ^ { \circ } \mathrm { C }$ reported for MOF-321.33,76 This split follows the linker rather than the rod-shaped SBU. The cis-only chains of CAU-10 and MIL-160 fall into opposite groups, whereas the trans-only and alternating cis-trans chains of MIL-53-muc and CAU-23 fall together.75 Merely copying the MOF-321 optimum would likely have missed both groups; the transfer prior instead carried over the shared stoichiometry and left each campaign to find its own temperature. These optima were reached with 50–60 min of microwave heating, compared with 12 h of solvothermal synthesis for $\mathrm { C A U - 1 0 ^ { 7 9 } }$ and 24 h and 6 h of reflux for $\mathrm { M I L - 1 6 0 ^ { 8 0 } }$ and CAU-23,75 respectively (Supporting Information, Section S6).

![](images/6b8a755284f7fa4b6d0b044623f6dfc5a20ac6feb5917f8e66399005fbd2e6b3.jpg)

![](images/af32bd164cfe66e957739c8c89281f1ab36d78adfe43ea219805aa87f4e19f28.jpg)  
Figure 5: PXRD patterns for the six MOF systems used to validate transfer of the benchmarkguided active-learning policy (CAU-10, CAU-23, MIL-53-muc, MIL-160, MTV-MIL-160, and MTV-MOF-303). Crystal structures are shown above each column. For every system, representative experimental patterns from low-, medium-, and high-q outcomes encountered during the search are compared with the corresponding simulated reference pattern shown at the bottom of each column. Selected reflections in the high-q traces are indexed. For illustrative clarity: Al is represented by deep blue octahedra; O, pink; C, light gray; N, green; S, yellow. H atoms are omitted.

Because the policy was chosen on a single benchmark landscape, these campaigns serve as a falsifiable test of transfer. A policy that had merely memorized the MOF-321 optimum would have missed frameworks whose best temperatures lie on either side of it. That the same policy instead located linker-specific conditions for six targets under one encoded reaction scheme supports the central premise of this work, that an experiment-grounded benchmark can select model suggestions that transfer across related reticular syntheses, provided the synthesis variables keep a comparable chemical meaning.

## Concluding Remarks

Considering the continued expansion of automated experimentation, foundation models, and AI-guided decision-making in chemistry, it is increasingly important to move beyond isolated trial-and-error attempts toward the conversion of crystallization outcomes into transferable synthetic knowledge in a quantitative and interpretable manner. ActMOF advances this direction by introducing a crystallinity-guided reaction benchmark in which encoded synthesis conditions are derived from experimentally measured PXRD outcomes of an aluminum carboxylate-based MOF, thereby providing a reproducible platform for studying how optimization algorithms navigate sparse-success reticular synthesis spaces. In this regard, our work shows that efficient MOF discovery is facilitated by (i) early confidence within the first few experimental batches, (ii) controlled exploitation through Matérn GP models with expected-improvement acquisition, and (iii) suggestions returned as inspectable laboratory prescriptions rather than latent indices. It is envisioned that such benchmarks will move synthetic chemistry from reporting optimal recipes toward reporting transferable search principles. The transferable element is therefore not a fixed recipe but an adaptive prior as framework-specific evidence accumulates. The validation across CAU-10, CAU-23, MIL-160, MIL-53-muc, MTV-MIL-160 and MTV-MOF-303 suggests a future in which multivariate and isoreticular structure discovery can be pursued through adaptive, benchmark-guided synthesis intelligence.

## Associated Content

Data and Software Availability. The ActMOF dataset and benchmark code are available at https://github.com/deep-synthesis/ActMOF. An interactive web demonstration for recommending synthesis conditions is available at https://actmof.streamlit.app/.

Supporting Information. The Supporting Information follows the main text of this document (from page S1). It contains materials and methods, construction and validation of the ActMOF benchmark, benchmarking protocols and statistics, transfer-learning framework, synthesis details, PXRD patterns and N2 sorption data.

## Author Information

## Corresponding Author

Zhiling Zheng – Department of Chemistry; Institute of Materials Science and Engineering, Washington University, St. Louis, Missouri 63130, United States; orcid.org/0000-0001-6090-2258. Email: z.z@wustl.edu

## Authors

Hongrui Sheng – Department of Chemistry, Washington University, St. Louis, Missouri 63130, United States; Department of Chemistry, Fudan University, Shanghai 200438, China; orcid.org/0009-0008-7444-2692

Shutian Liang – Department of Energy, Environmental and Chemical Engineering, Washington University, St. Louis, Missouri 63130, United States

Reid A. Coyle – Department of Chemistry, Washington University, St. Louis, Missouri 63130, United States

Shyam Chand Pal – Department of Chemistry, Washington University, St. Louis, Missouri 63130, United States

Peter Walther – Department of Chemistry, Washington University, St. Louis, Missouri 63130, United States

Bin Feng – Department of Chemistry, Washington University, St. Louis, Missouri 63130, United States; Institute of Materials Science and Engineering, Washington University, St. Louis, Missouri 63130, United States

Xinyu Ding – Department of Chemistry, Washington University, St. Louis, Missouri 63130, United States; Department of Chemistry, Imperial College London, London W12 0BZ, United Kingdom

## Author Contributions

Z.Z. conceptualized the study and designed the methodology. H.S. performed sample preparation, PXRD and BET measurements, sorted the benchmark code, carried out data analysis, and prepared the manuscript and Supporting Information. S.L. prepared samples, collected PXRD data, and contributed to the Supporting Information. R.C. assisted with sample preparation, PXRD measurements, and code review. S.C.P. assisted with BET measurements and analysis. P.W. assisted with benchmark code and sample preparation. B.F. participated in discussions and contributed to data visualization. X.D. assisted with sample preparation. Z.Z. provided critical review and discussion and finalized the manuscript.

The manuscript was written through contributions of all authors. All authors have given approval to the final version of the manuscript.

## Conflicts of interest

There are no conflicts to declare.

## Acknowledgment

Z.Z. acknowledges funding and extends thanks for the support provided by the EQT Foundation through the Breakthrough Science Grant (No. 059994), Thinking Machines through the Tinker Research Grant, the Royal Society of Chemistry through the Researcher Collaborations grant (GR0042769), and Washington University in St. Louis through the Startup Grant. The authors thank Prof. Nan Lin (Department of Statistics and Data Science, Washington University in St. Louis) for insightful discussions on strengthening the statistical basis of the ActMOF landscape. H.S. thanks Chenglin Li (Washington University in St. Louis) for building the initial version of the interactive web application and Zeyuan Li (Tsinghua University) for testing the reliability of the ActMOF landscape.

## References

(1) Lyu, H.; Ji, Z.; Wuttke, S.; Yaghi, O. M. Digital Reticular Chemistry. Chem 2020, 6 (9), 2219– 2241.https://doi.org/10.1016/j.chempr.2020.08.008

(2) Gropp, C.; Canossa, S.; Wuttke, S.; Gándara, F.; Li, Q.; Gagliardi, L.; Yaghi, O. M. Standard Practices of Reticular Chemistry. ACS Cent. Sci. 2020, 6 (8), 1255–1273. https://doi.org/10 .1021/acscentsci.0c00592

(3) Feng, L.; Wang, K.-Y.; Powell, J.; Zhou, H.-C. Controllable Synthesis of Metal-Organic Frameworks and Their Hierarchical Assemblies. Matter 2019, 1 (4), 801–824. https://doi.org/10.1 016/j.matt.2019.08.022

(4) Howarth, A. J.; Peters, A. W.; Vermeulen, N. A.; Wang, T. C.; Hupp, J. T.; Farha, O. K. Best Practices for the Synthesis, Activation, and Characterization of Metal-Organic Frameworks Chem. Mater. 2017, 29 (1), 26–39. https://doi.org/10.1021/acs.chemmater.6b02626

(5) Jiang, H.; Benzaria, S.; Alsadun, N.; Jia, J.; Czaban-Jóźwiak, J.; Guillerm, V.; Shkurenko, A.; Thiam, Z.; Bonneau, M.; Maka, V. K.; Chen, Z.; Ameur, Z. O.; O'Keeffe, M.; Eddaoudi, M. Merged-Nets Enumeration for the Systematic Design of Multicomponent Reticular Structures. Science 2024, 386 (6722), 659–666. https://doi.org/10.1126/science.ads7866

(6) Yaghi, O. M.; O'Keeffe, M.; Ockwig, N. W.; Chae, H. K.; Eddaoudi, M.; Kim, J. Reticular Synthesis and the Design of New Materials. Nature 2003, 423 (6941), 705–714. https://doi.or g/10.1038/nature01650

(7) Moosavi, S. M.; Nandy, A.; Jablonka, K. M.; Ongari, D.; Janet, J. P.; Boyd, P. G.; Lee, Y.; Smit, B.; Kulik, H. J. Understanding the Diversity of the Metal-Organic Framework Ecosystem. Nat. Commun. 2020, 11 (1), 4068. https://doi.org/10.1038/s41467-020-17755-8

(8) Anderson, R.; Gómez-Gualdrón, D. A. Increasing Topological Diversity during Computational "Synthesis" of Porous Crystals: How and Why. CrystEngComm 2019, 21 (10), 1653–1665. https : //doi.org/10.1039/C8CE01637B

(9) Furukawa, H.; Cordova, K. E.; O'Keeffe, M.; Yaghi, O. M. The Chemistry and Applications of Metal-Organic Frameworks. Science 2013, 341 (6149), 1230444. https://doi.org/10.1126/sc ience.1230444

(10) Duan, C.; Nandy, A.; Chand Pal, S.; Yang, X.; Gao, W.; Du, Y.; Kraß, H.; Kang, Y.; Bernales, V.; Ye, Z.; Pyle, T.; Yang, R.; Gu, Z.; Schwaller, P.; Ma, S.; Sun, S.; Aspuru-Guzik, A.; Moosavi, S. M.; Wexler, R.; Zheng, Z. The Rise of Generative AI for Metal-Organic Framework Design and Synthesis. Matter 2026, 9 (5), 102748. https://doi.org/10.1016/j.matt.2026.102748

(11) Kim, B.; Lee, S.; Kim, J. Inverse Design of Porous Materials Using Artificial Neural Networks. Sci. Adv. 2020, 6 (1), eaax9324. https://doi.org/10.1126/sciadv.aax9324

(12) Zhou, H.-C.; Long, J. R.; Yaghi, O. M. Introduction to Metal–Organic Frameworks. Chem. Rev. 2012, 112 (2), 673–674. https://doi.org/10.1021/cr300014x

(13) Eddaoudi, M.; Kim, J.; Rosi, N.; Vodak, D.; Wachter, J.; O'Keeffe, M.; Yaghi, O. M. Systematic Design of Pore Size and Functionality in Isoreticular MOFs and Their Application in Methane Storage. Science 2002, 295 (5554), 469–472. https://doi.org/10.1126/science.1067208

(14) Deng, H.; Grunder, S.; Cordova, K. E.; Valente, C.; Furukawa, H.; Hmadeh, M.; Gándara, F.; Whalley, A. C.; Liu, Z.; Asahina, S.; Kazumori, H.; O'Keeffe, M.; Terasaki, O.; Stoddart, J. F.; Yaghi, O. M. Large-Pore Apertures in a Series of Metal-Organic Frameworks. Science 2012, 336 (6084), 1018–1023. https://doi.org/10.1126/science.1220131

(15) Helal, A.; Yamani, Z. H.; Cordova, K. E.; Yaghi, O. M. Multivariate Metal-Organic Frameworks. Natl. Sci. Rev. 2017, 4 (3), 296–298. https://doi.org/10.1093/nsr/nwx013

(16) Canossa, S.; Ji, Z.; Gropp, C.; Rong, Z.; Ploetz, E.; Wuttke, S.; Yaghi, O. M. System of Sequences in Multivariate Reticular Structures. Nat. Rev. Mater. 2023, 8 (5), 331–340. https://doi.org/ 10.1038/s41578-022-00482-5

(17) Deng, H.; Doonan, C. J.; Furukawa, H.; Ferreira, R. B.; Towne, J.; Knobler, C. B.; Wang, B.; Yaghi, O. M. Multiple Functional Groups of Varying Ratios in Metal-Organic Frameworks. Science 2010, 327 (5967), 846–850. https://doi.org/10.1126/science.1181761

(18) Glasby, L. T.; Gubsch, K.; Bence, R.; Oktavian, R.; Isoko, K.; Moosavi, S. M.; Cordiner, J. L.; Cole, J. C.; Moghadam, P. Z. DigiMOF: A Database of Metal–Organic Framework Synthesis Information Generated via Text Mining. Chem. Mater. 2023, 35 (11), 4510–4524. https://do i.org/10.1021/acs.chemmater.3c00788

(19) Kang, Y.; Lee, W.; Bae, T.; Han, S.; Jang, H.; Kim, J. Harnessing Large Language Models to Collect and Analyze Metal–Organic Framework Property Data Set. J. Am. Chem. Soc. 2025, 147 (5), 3943–3958. https://doi.org/10.1021/jacs.4c11085

(20) Azbell, T. J.; Pitt, T. A.; Jerozal, R. T.; Mandel, R. M.; Milner, P. J. Simplifying the Synthesis of Metal–Organic Frameworks. Acc. Mater. Res. 2023, 4 (10), 867–878. https://doi.org/10.1 021/accountsmr.3c00121

(21) Pruyn, T. M.; Aswad, A.; Khan, S. T.; Huang, J.; Black, R.; Moosavi, S. M. MOF-ChemUnity: Literature-Informed Large Language Models for Metal-Organic Framework Research. J. Am. Chem. Soc. 2025, 147 (47), 43474–43486. https://doi.org/10.1021/jacs.5c11789

(22) Luo, Y.; Bag, S.; Zaremba, O.; Cierpka, A.; Andreo, J.; Wuttke, S.; Friederich, P.; Tsotsalas, M. MOF Synthesis Prediction Enabled by Automatic Data Mining and Machine Learning. Angew. Chem. Int. Ed. 2022, 61 (19), e202200242. https://doi.org/10.1002/anie.202200242

(23) Park, S.; Kim, B.; Choi, S.; Boyd, P. G.; Smit, B.; Kim, J. Text Mining Metal–Organic Framework Papers. J. Chem. Inf. Model. 2018, 58 (2), 244–251. https://doi.org/10.1021/acs.jcim.7b0 0608

(24) Zheng, Z.; Zhang, O.; Borgs, C.; Chayes, J. T.; Yaghi, O. M. ChatGPT Chemistry Assistant for Text Mining and the Prediction of MOF Synthesis. J. Am. Chem. Soc. 2023, 145 (32), 18048– 18062.https://doi.org/10.1021/jacs.3c05819

(25) Ma, B.; Qin, N.; Yan, Q.; Zhou, W.; Zhang, S.; Wang, X.; Bao, L.; Lu, X. Advancing Metal Organic Framework and Covalent Organic Framework Design via the Digital-Intelligent Paradigm. Digit. Discov. 2026, 5 (2), 523–547. https://doi.org/10.1039/D5DD00401B

(26) Zheng, Z.; Rampal, N.; Inizan, T. J.; Borgs, C.; Chayes, J. T.; Yaghi, O. M. Large Language

Models for Reticular Chemistry. Nat. Rev. Mater. 2025, 10 (5), 369–381. https://doi.org/10 .1038/s41578-025-00772-8

(27) Banerjee, R.; Phan, A.; Wang, B.; Knobler, C.; Furukawa, H.; O'Keeffe, M.; Yaghi, O. M. High-Throughput Synthesis of Zeolitic Imidazolate Frameworks and Application to CO2 Capture. Science 2008, 319 (5865), 939–943. https://doi.org/10.1126/science.1152516

(28) Livas, C. G.; Trikalitis, P. N.; Froudakis, G. E. MOFSynth: A Computational Tool toward Synthetic Likelihood Predictions of MOFs. J. Chem. Inf. Model. 2024, 64 (21), 8193–8200. https: //doi.org/10.1021/acs.jcim.4c01298

(29) Mao, J.; Jiang, N.; Darù, A.; Filatov, A. S.; Burch, J. E.; Hofmann, J.; Vornholt, S. M.; Chapman, K. W.; Anderson, J. S.; Ferguson, A. L. Structure and Synthesizability of Iron-Sulfur Metal-Organic Frameworks. J. Am. Chem. Soc. 2025, 147 (21), 17651–17667. https://doi.org/10.1 021/jacs.4c16341

(30) Niyongabo Rubungo, A.; Fajardo-Rojas, F.; Gómez-Gualdrón, D. A.; Dieng, A. B. Highly Accurate and Fast Prediction of MOF Free Energy via Machine Learning. J. Am. Chem. Soc. 2025, 147 (52), 48035–48045. https://doi.org/10.1021/jacs.5c13960

(31) Moosavi, S. M.; Chidambaram, A.; Talirz, L.; Haranczyk, M.; Stylianou, K. C.; Smit, B. Capturing Chemical Intuition in Synthesis of Metal-Organic Frameworks. Nat. Commun. 2019, 10 (1), 539. https://doi.org/10.1038/s41467-019-08483-9

(32) Rong, Z.; Chen, Z.; Luong, F.; Chheda, S.; Luong, H. T. N.; Zheng, Z.; Greco, K.; Alghamdi, A. A.; Bui, K. H.; Jaffrelot Inizan, T.; Nguyen-Dang, T.; Pham, H. H.; Le, D. D.; Sauer, J.; Phung, V. B. T.; Chayes, J. T.; Borgs, C.; Boley, M.; El Ghaoui, L.; Yaghi, O. M. Algorithmic Iterative Reticular Synthesis of Zeolitic Imidazolate Framework Crystals. Nat. Synth. 2025, 5 (3), 409–419. https://doi.org/10.1038/s44160-025-00939-9

(33) Zheng, Z.; Zhang, O.; Nguyen, H. L.; Rampal, N.; Alawadhi, A. H.; Rong, Z.; Head-Gordon, T.; Borgs, C.; Chayes, J. T.; Yaghi, O. M. ChatGPT Research Group for Optimizing the Crystallinity of MOFs and COFs. ACS Cent. Sci. 2023, 9 (11), 2161–2170. https://doi.org/10.1021/acsc entsci.3c01087

(34) Domingues, N. P.; Moosavi, S. M.; Talirz, L.; Jablonka, K. M.; Ireland, C. P.; Ebrahim, F. M.; Smit, B. Using Genetic Algorithms to Systematically Improve the Synthesis Conditions of Al-PMOF. Commun. Chem. 2022, 5 (1), 170. https://doi.org/10.1038/s42004-022-00785-2

(35) Wu, X.; Jiang, J. Interrogating the Synthetic Likelihood of Metal–Organic Frameworks: A Digital Discovery Perspective. Chem. Sci. 2026, 17 (26), 12635–12646. https://doi.org/10.1039/D6 SC02765B

(36) Garibay, S. J.; Cohen, S. M. Isoreticular Synthesis and Modification of Frameworks with the UiO-66 Topology. Chem. Commun. 2010, 46 (41), 7700–7702. https://doi.org/10.1039/C0Cc 02990D

(37) Biemmi, E.; Christian, S.; Stock, N.; Bein, T. High-Throughput Screening of Synthesis Parameters in the Formation of the Metal-Organic Frameworks MOF-5 and HKUST-1. Microporous Mesoporous Mater. 2009, 117 (1–2), 111–117. https://doi.org/10.1016/j.micromeso.2008 .06.040

(38) Richezzi, M.; Donnarumma, P. R.; Howarth, A. J. Studying the Significance of the Parameters Involved in the Synthesis of Y-UiO-66 to Improve Product Yield. Mater. Adv. 2026, 7 (5), 2621– 2628. https://doi.org/10.1039/D5MA01198A

(39) Han, Z.; Yang, Y.; Rushlow, J.; Huo, J.; Liu, Z.; Hsu, Y.-C.; Yin, R.; Wang, M.; Liang, R.; Wang, K.-Y.; Zhou, H.-C. Development of the Design and Synthesis of Metal–Organic Frameworks (MOFs) – from Large Scale Attempts, Functional Oriented Modifications, to Artificial Intelligence (AI) Predictions. Chem. Soc. Rev. 2025, 54 (1), 367–395. https://doi.org/10.1039/D4CS0043 2A

(40) Lamberti, F.; Mazzariol, C.; Spolaore, F.; Ceccato, R.; Salmaso, L.; Gross, S. Design of Experiment: A Rational and Still Unexplored Approach to Inorganic Materials' Synthesis. Sustain Chem. 2022, 3 (1), 114–130. https://doi.org/10.3390/suschem3010009

(41) Achenbach, B.; Yurdusen, A.; Stock, N.; Maurin, G.; Serre, C. Synthetic Aspects and Characterization Needs in MOF Chemistry – from Discovery to Applications. Adv. Mater. 2025, 37 (52), 2411359. https://doi.org/10.1002/adma.202411359

(42) Chong, K. X.; Alsabia, Q. A.; Ye, Z.; McDaniel, A.; Baumgardner, D.; Xiao, D.; Sun, S. Con-

trolling Metal-Organic Framework Crystallization via Computer Vision and Robotic Handling. J. Mater. Chem. A 2025, 13 (33), 27279–27289. https://doi.org/10.1039/D5TA03199K

(43) Boström, H. L. B.; Emmerling, S.; Heck, F.; Koschnick, C.; Jones, A. J.; Cliffe, M. J.; Al Natour, R.; Bonneau, M.; Guillerm, V.; Shekhah, O.; Eddaoudi, M.; Lopez-Cabrelles, J.; Furukawa, S.; Romero-Angel, M.; Martí-Gastaldo, C.; Yan, M.; Morris, A. J.; Romero-Muñiz, I.; Xiong, Y.; Platero-Prats, A. E.; Roth, J.; Queen, W. L.; Mertin, K. S.; Schier, D. E.; Champness, N. R.; Yeung, H. H.-M.; Lotsch, B. V. How Reproducible Is the Synthesis of Zr-Porphyrin Metal– Organic Frameworks? An Interlaboratory Study. Adv. Mater. 2024, 36 (15), 2304832. https: //doi.org/10.1002/adma.202304832

(44) Chen, F. E.; Pitt, T. A.; Okong'o, D. J.; Wetherbee, L. G.; Fuentes-Rivera, J. J.; Milner, P. J. A Structure-Activity Study of Aromatic Acid Modulators for the Synthesis of Zirconium-Based Metal–Organic Frameworks. Chem. Mater. 2022, 34 (7), 3383–3394. https://doi.org/10.102 1/acs.chemmater.2c00241

(45) Neubauer, F.; Endo, K.; Bender, F.; Ciftci, E.; Hansen, N.; Krause, S.; Uekermann, B.; Pleiss, J. Data-Model-Driven Management and Analysis of MOF Synthesis Data. February 2, 2026. https://doi.org/10.26434/chemrxiv.10001842/v1

(46) Chen, L.; Lu, Z.; Chen, L.; Hou, L.; Zhang, D. Chemist-Guided Human–AI Workflow for Covalent Organic Framework Synthesis. J. Am. Chem. Soc. 2026, 148 (7), 7440–7449. https://doi.or g/10.1021/jacs.5c20068

(47) Butler, K. T.; Davies, D. W.; Cartwright, H.; Isayev, O.; Walsh, A. Machine Learning for Molecular and Materials Science. Nature 2018, 559 (7715), 547–555. https://doi.org/10.1038/s4 1586-018-0337-2

(48) Sanchez-Lengeling, B.; Aspuru-Guzik, A. Inverse Molecular Design Using Machine Learning: Generative Models for Matter Engineering. Science 2018, 361 (6400), 360–365. https://doi.org/ 10.1126/science.aat2663

(49) Eyke, N. S.; Green, W. H.; Jensen, K. F. Iterative Experimental Design Based on Active Machine Learning Reduces the Experimental Burden Associated with Reaction Screening. React. Chem. Eng. 2020, 5 (10), 1963–1972. https://doi.org/10.1039/D0RE00232A

(50) Shim, E.; Tewari, A.; Cernak, T.; Zimmerman, P. M. Machine Learning Strategies for Reaction Development: Toward the Low-Data Limit. J. Chem. Inf. Model. 2023, 63 (12), 3659–3668. https://doi.org/10.1021/acs.jcim.3c00577

(51) Lookman, T.; Balachandran, P. V.; Xue, D.; Yuan, R. Active Learning in Materials Science with Emphasis on Adaptive Sampling Using Uncertainties for Targeted Design. Npj Comput. Mater. 2019, 5 (1), 21. https://doi.org/10.1038/s41524-019-0153-8

(52) Shields, B. J.; Stevens, J.; Li, J.; Parasram, M.; Damani, F.; Alvarado, J. I. M.; Janey, J. M.; Adams, R. P.; Doyle, A. G. Bayesian Reaction Optimization as a Tool for Chemical Synthesis. Nature 2021, 590 (7844), 89–96. https://doi.org/10.1038/s41586-021-03213-y

(53) Burger, B.; Maffettone, P. M.; Gusev, V. V.; Aitchison, C. M.; Bai, Y.; Wang, X.; Li, X.; Alston, B. M.; Li, B.; Clowes, R.; Rankin, N.; Harris, B.; Sprick, R. S.; Cooper, A. I. A Mobile Robotic Chemist. Nature 2020, 583 (7815), 237–241. https://doi.org/10.1038/s41586-020-2442-2

(54) Zheng, Z.; Rong, Z.; Rampal, N.; Borgs, C.; Chayes, J. T.; Yaghi, O. M. A GPT-4 Reticular Chemist for Guiding MOF Discovery. Angew. Chem. Int. Ed. 2023, 62 (46), e202311983. https: //doi.org/10.1002/anie.202311983

(55) McDonald, M. A.; Koscher, B. A.; Canty, R. B.; Zhang, J.; Ning, A.; Jensen, K. F. Bayesian Optimization over Multiple Experimental Fidelities Accelerates Automated Discovery of Drug Molecules. ACS Cent. Sci. 2025, 11 (2), 346–356. https://doi.org/10.1021/acscentsci.4c0 1991

(56) Florit, F.; Nandiwale, K. Y.; Armstrong, C. T.; Grohowalski, K.; Diaz, A. R.; Mustakis, J.; Guinness, S. M.; Jensen, K. F. Dynamic Flow Experiments for Bayesian Optimization of a Single Process Objective. React. Chem. Eng. 2025, 10 (3), 656–666. https://doi.org/10.1039/D4RE 00543K

(57) Aqeeli, M.; Leelawat, T.; Shorthouse, D. Novelty-Aware Evolutionary Bayesian Optimisation for Multi-Objective Discovery Science. Digit. Discov. 2026, 5 (6), 2684–2700. https://doi.org/10 .1039/D6DD00134C

(58) Torres, J. A. G.; Lau, S. H.; Anchuri, P.; Stevens, J. M.; Tabora, J. E.; Li, J.; Borovika, A.;

Adams, R. P.; Doyle, A. G. A Multi-Objective Active Learning Platform and Web App for Reaction Optimization. J. Am. Chem. Soc. 2022, 144 (43), 19999–20007. https://doi.org/10 .1021/jacs.2c08592

(59) Nair, A. S.; Foppa, L.; Scheffler, M. Interpretable Bayesian Optimization for Catalyst Discovery. Faraday Discuss. 2026, 267, 298–311. https://doi.org/10.1039/D5FD00159E

(60) Kumar, A.; Pant, K. K.; Upadhyayula, S.; Kodamana, H. Multiobjective Bayesian Optimization Framework for the Synthesis of Methanol from Syngas Using Interpretable Gaussian Process Models. ACS Omega 2023, 8 (1), 410–421. https://doi.org/10.1021/acsomega.2c04919

(61) Qin, H.; Hu, J.; Zhao, S.; Zhou, H.-Q.; Liao, W.-M.; Chung, L.-H.; Wu, Y.; He, J. InterpretableMachine Learning Unveils Hydroxyl/AminoSynergy and Guides Discovery of Optimal MOF Photocatalysts for HydrogenEvolution. J. Am. Chem. Soc. 2026, 148 (29), 30998–31010. https://doi.org/10.1021/jacs.6c05998

(62) Shao, Z.-W.; Zhang, Z.; Kuang, Y.; Xiong, C.; Yang, J.; Wu, W.; Liu, Y.; Xiong, L.; Duan, X.; Liu, C. Bayesian Optimized Crystallization of a Hydroxamate-Functionalized Covalent Organic Framework for Enhanced Uranyl Uptake. Small 2025, 21 (18), 2411788. https://doi.org/10 .1002/sml1.202411788

(63) Nambiar, A. M. K.; Breen, C. P.; Hart, T.; Kulesza, T.; Jamison, T. F.; Jensen, K. F. Bayesian Optimization of Computer-Proposed Multistep Synthetic Routes on an Automated Robotic Flow Platform. ACS Cent. Sci. 2022, 8 (6), 825–836. https://doi.org/10.1021/acscentsci.2c002 07

(64) Xie, Y.; Zhang, C.; Deng, H.; Zheng, B.; Su, J.-W.; Shutt, K.; Lin, J. Accelerate Synthesis of Metal–Organic Frameworks by a Robotic Platform and Bayesian Optimization. ACS Appl. Mater. Interfaces 2021, 13 (45), 53485–53491. https://doi.org/10.1021/acsami.1c16506

(65) Du, Y.; Du, D. Bayesian Optimization Enhanced Neural Networks for Predicting Metal-Organic Framework Morphology: A ZIF-8 Synthesis Case Study. Mater. Lett. 2025, 380, 137738. https: //doi.org/10.1016/j.matlet.2024.137738

(66) Norinder, U.; Carlsson, L.; Boyer, S.; Eklund, M. Introducing Conformal Prediction in Predictive Modeling. A Transparent and Flexible Alternative to Applicability Domain Determination. J. Chem. Inf. Model. 2014, 54 (6), 1596–1603. https://doi.org/10.1021/ci5001168

(67) Zheng, Z.; Florit, F.; Jin, B.; Wu, H.; Li, S.-C.; Nandiwale, K. Y.; Salazar, C. A.; Mustakis, J. G.; Green, W. H.; Jensen, K. F. Integrating Machine Learning and Large Language Models to Advance Exploration of Electrochemical Reactions. Angew. Chem. Int. Ed. 2025, 64 (6), e202418074. https://doi.org/10.1002/anie.202418074

(68) Boiko, D. A.; MacKnight, R.; Kline, B.; Gomes, G. Autonomous Chemical Research with Large Language Models. Nature 2023, 624 (7992), 570–578. https://doi.org/10.1038/s41586-023 -06792-0

(69) Zhang, Y.; Han, Y.; Chen, S.; Yu, R.; Zhao, X.; Liu, X.; Zeng, K.; Yu, M.; Tian, J.; Zhu, F.; Yang, X.; Jin, Y.; Xu, Y. Large Language Models to Accelerate Organic Chemistry Synthesis. Nat. Mach. Intell. 2025, 7 (7), 1010–1022. https://doi.org/10.1038/s42256-025-01066-y

(70) Mahjour, B.; Hoffstadt, J.; Cernak, T. Designing Chemical Reaction Arrays Using Phactor and ChatGPT. Org. Process Res. Dev. 2023, 27 (8), 1510–1516. https://doi.org/10.1021/acs.op rd.3c00186

(71) Cissé, A.; Cooper, M. E.; Zhu, M.; Evangelopoulos, X.; Cooper, A. I. Can We Automate Scientific Reasoning in Closed-Loop Experiments Using Large Language Models? Digit. Discov. 2026, 5 (3), 1132–1160. https://doi.org/10.1039/D5DD00520E

(72) Ranković, B.; Schwaller, P. BoChemian: Large Language Model Embeddings for Bayesian Optimization of Chemical Reactions. In NeurIPS 2023 Workshop on Adaptive Experimental Design and Active Learninq in the Real World; 2023

(73) Patel, R. A.; Li, M.; Chang, C.-F.; De Lescure, L.; Chauvin, P.; Cherney, A.; Moayedpour, S.; Jager, S.; Jangjou, Y. Distilling and Exploiting Quantitative Insights from Large Language Models for Enhanced Bayesian Optimization of Chemical Reactions. Digit. Discov. 2026, 5 (5). 2042-2051.https://doi.org/10.1039/D6DD00052E

(74) Schlüsener, C.; Xhinovci, M.; Ernst, S.-J.; Schmitz, A.; Tannert, N.; Janiak, C. Solid-Solution Mixed-Linker Synthesis of Isoreticular Al-Based MOFs for an Easy Hydrophilicity Tuning in Water-Sorption Heat Transformations. Chem. Mater. 2019, 31 (11), 4051–4062. https://doi.

org/10.1021/acs.chemmater.9b00617

(75) Lenzen, D.; Zhao, J.; Ernst, S.-J.; Wahiduzzaman, M.; Ken Inge, A.; Fröhlich, D.; Xu, H.; Bart, H.-J.; Janiak, C.; Henninger, S.; Maurin, G.; Zou, X.; Stock, N. A Metal–Organic Framework for Efficient Water-Based Ultra-Low-Temperature-Driven Cooling. Nat. Commun. 2019, 10 (1), 3025.https://doi.org/10.1038/s41467-019-10960-0

(76) Hanikel, N.; Kurandina, D.; Chheda, S.; Zheng, Z.; Rong, Z.; Neumann, S. E.; Sauer, J.; Siepmann, J. I.; Gagliardi, L.; Yaghi, O. M. MOF Linker Extension Strategy for Enhanced Atmospheric Water Harvesting. ACS Cent. Sci. 2023, 9 (3), 551–557. https://doi.org/10.1021/ac scentsci.3c00018

(77) Rosi, N. L.; Kim, J.; Eddaoudi, M.; Chen, B.; O'Keeffe, M.; Yaghi, O. M. Rod Packings and Metal-Organic Frameworks Constructed from Rod-Shaped Secondary Building Units. J. Am. Chem. Soc. 2005, 127 (5), 1504–1518. https://doi.org/10.1021/ja045123o

(78) Cadiau, A.; Lee, J. S.; Damasceno Borges, D.; Fabry, P.; Devic, T.; Wharmby, M. T.; Martineau, C.; Foucher, D.; Taulelle, F.; Jun, C.-H.; Hwang, Y. K.; Stock, N.; De Lange, M. F.; Kapteijn, F.; Gascon, J.; Maurin, G.; Chang, J.-S.; Serre, C. Design of Hydrophilic Metal Organic Framework Water Adsorbents for Heat Reallocation. Adv. Mater. 2015, 27 (32), 4775–4780. https://doi. org/10.1002/adma.201502418

(79) Reinsch, H.; Van Der Veen, M. A.; Gil, B.; Marszalek, B.; Verbiest, T.; De Vos, D.; Stock, N. Structures, Sorption Characteristics, and Nonlinear Optical Properties of a New Series of Highly Stable Aluminum MOFs. Chem. Mater. 2013, 25 (1), 17–26. https://doi.org/10.1021/cm30 25445

(80) Tannert, N.; Jansen, C.; Nießing, S.; Janiak, C. Robust Synthesis Routes and Porosity of the Al-Based Metal–Organic Frameworks Al-Fumarate, CAU-10-H and MIL-160. Dalton Trans. 2019, 48 (9), 2967–2976. https://doi.org/10.1039/C8DT04688C

(81) Matemb Ma Ntep, T. J.; Reinsch, H.; Hügenell, P. P. C.; Ernst, S.-J.; Hastürk, E.; Janiak, C. Designing a New Aluminium Muconate Metal–Organic Framework (MIL-53-Muc) as a Methanol Adsorbent for Sub-Zero Temperature Heat Transformation Applications. J. Mater. Chem. A 2019, 7 (43), 24973-24981. https://doi.org/10.1039/C9TA07465A

(82) Hanikel, N.; Pei, X.; Chheda, S.; Lyu, H.; Jeong, W.; Sauer, J.; Gagliardi, L.; Yaghi, O. M. Evolution of Water Structures in Metal-Organic Frameworks for Improved Atmospheric Water Harvesting. Science 2021, 374 (6566), 454–459. https://doi.org/10.1126/science.abj0890

(83) Zheng, Z.; Hanikel, N.; Lyu, H.; Yaghi, O. M. Broadly Tunable Atmospheric Water Harvesting in Multivariate Metal–Organic Frameworks. J. Am. Chem. Soc. 2022, 144 (49), 22669–22675. https://doi.org/10.1021/jacs.2c09756

(84) Xu, M.; Wang, X.; Chen, J.; Sun, M.; Zhang, X.; Fan, W.; Wang, T.; Sun, D. Multivariate MOF-303/MIL-160 Balancing the Trade-off between Capacity and Selectivity in CO2/CH4 Separation. Dalton Trans. 2026, 55 (7), 2812–2816. https://doi.org/10.1039/D6DT00085A

(85) Schlüsener, C.; Jordan, D. N.; Xhinovci, M.; Ntep, T. J. M. M.; Schmitz, A.; Giesen, B.; Janiak, C. Probing the Limits of Linker Substitution in Aluminum MOFs through Water Vapor Sorption Studies: Mixed-MOFs Instead of Mixed-Linker CAU-23 and MIL-160 Materials. Dalton Trans. 2020, 49 (22), 7373–7383. https://doi.org/10.1039/D0DT01044H

(86) Ni, Z.; Masel, R. I. Rapid Production of Metal-Organic Frameworks via Microwave-Assisted Solvothermal Synthesis. J. Am. Chem. Soc. 2006, 128 (38), 12394–12395. https://doi.org/10 .1021/ja0635231

(87) Klinowski, J.; Paz, F. A. A.; Silva, P.; Rocha, J. Microwave-Assisted Synthesis of Metal–Organic Frameworks. Dalton Trans. 2010, 40 (2), 321–330. https://doi.org/10.1039/C0DT00708K

(88) Głowniak, S.; Szczęśniak, B.; Choma, J.; Jaroniec, M. Advances in Microwave Synthesis of Nanoporous Materials. Adv. Mater. 2021, 33 (48), 2103477. https://doi.org/10.1002/ad ma.202103477

(89) Freund, R.; Canossa, S.; Cohen, S. M.; Yan, W.; Deng, H.; Guillerm, V.; Eddaoudi, M.; Madden, D. G.; Fairen-Jimenez, D.; Lyu, H.; Macreadie, L. K.; Ji, Z.; Zhang, Y.; Wang, B.; Haase, F.; Wöll, C.; Zaremba, O.; Andreo, J.; Wuttke, S.; Diercks, C. S. 25 Years of Reticular Chemistry. Angew. Chem. Int. Ed. 2021, 60 (45), 23946–23974. https://doi.org/10.1002/anie.202101644

(90) Bhasin, H.; Kashyap, P.; Fernandes, P.; Mishra, D. Multi-Topic Carboxylates as Versatile Building Blocks for the Design and Synthesis of Multifunctional MOFs Based on Alkaline Earth, Main

Group and Transition Metals. Comments Inorg. Chem. 2023, 43 (4), 257–304. https://doi.or g/10.1080/02603594.2022.2121279

(91) Martí-Rujas, J. Structural Elucidation of Microcrystalline MOFs from Powder X-Ray Diffraction. Dalton Trans. 2020, 49 (40), 13897–13916. https://doi.org/10.1039/D0DT02802A

(92) Abid, H. R.; Azhar, M. R.; Iglauer, S.; Rada, Z. H.; Al-Yaseri, A.; Keshavarz, A. Physicochemical Characterization of Metal Organic Framework Materials: A Mini Review. Heliyon 2024, 10 (1), e23840. https://doi.org/10.1016/j.heliyon.2023.e23840

(93) Feng, B.; Wang, B.; Lv, L.; Zhang, M.; Chen, Z.; Pan, F.; Li, S. Interpreting X-Ray Diffraction Patterns of Metal–Organic Frameworks via Generative Artificial Intelligence. J. Am. Chem. Soc. 2026, 148 (1), 869–878. https://doi.org/10.1021/jacs.5c16416

(94) Clayson, I. G.; Hewitt, D.; Hutereau, M.; Pope, T.; Slater, B. High Throughput Methods in the Synthesis, Characterization, and Optimization of Porous Materials. Adv. Mater. 2020, 32 (44), 2002780. https://doi.org/10.1002/adma.202002780

(95) Howarth, A. J.; Liu, Y.; Li, P.; Li, Z.; Wang, T. C.; Hupp, J. T.; Farha, O. K. Chemical, Thermal and Mechanical Stabilities of Metal–Organic Frameworks. Nat. Rev. Mater. 2016, 1 (3), 15018. https://doi.org/10.1038/natrevmats.2015.18

(96) Deza, M. M.; Deza, E. Encyclopedia of Distances. In Encyclopedia of Distances; Springer Berlin Heidelberg: Berlin, Heidelberg, 2009; pp 1–583. https://doi.org/10.1007/978-3-642-00234 -2\_1

(97) Klove, T.; Lin, T.-T.; Tsai, S.-C.; Tzeng, W.-G. Permutation Arrays Under the Chebyshev Distance. IEEE Trans. Inf. Theory 2010, 56 (6), 2611–2617. https://doi.org/10.1109/TIT.2010 .2046212

(98) Kononova, O.; Huo, H.; He, T.; Rong, Z.; Botari, T.; Sun, W.; Tshitoyan, V.; Ceder, G. Text-Mined Dataset of Inorganic Materials Synthesis Recipes. Sci. Data 2019, 6 (1), 203. https: //doi.org/10.1038/s41597-019-0224-1

(99) Raccuglia, P.; Elbert, K. C.; Adler, P. D. F.; Falk, C.; Wenny, M. B.; Mollo, A.; Zeller, M.; Friedler, S. A.; Schrier, J.; Norquist, A. J. Machine-Learning-Assisted Materials Discovery Using Failed Experiments. Nature 2016, 533 (7601), 73–76. https://doi.org/10.1038/nature17439

(100) Strieth-Kalthoff, F.; Sandfort, F.; Kühnemund, M.; Schäfer, F. R.; Kuchen, H.; Glorius, F. Machine Learning for Chemical Reactivity: The Importance of Failed Experiments. Angew. Chem. Int. Ed. 2022, 61 (29), e202204647. https://doi.org/10.1002/anie.202204647

(101) Jia, X.; Lynch, A.; Huang, Y.; Danielson, M.; Lang'at, I.; Milder, A.; Ruby, A. E.; Wang, H.; Friedler, S. A.; Norquist, A. J.; Schrier, J. Anthropogenic Biases in Chemical Reaction Data Hinder Exploratory Inorganic Synthesis. Nature 2019, 573 (7773), 251–255. https://doi.org/ 10.1038/s41586-019-1540-5

(102) Osaro, E.; Colón, Y. J. Intelligent Screening of Porous Materials: A Review of Active-Learning Approaches in MOF Research. Chem. Phys. Rev. 2025, 6 (4), 041307. https://doi.org/10.1 063/5.0295283

(103) Jose, A.; Devijver, E.; Jakse, N.; Poloni, R. Informative Training Data for Efficient Property Prediction in Metal–Organic Frameworks by Active Learning. J. Am. Chem. Soc. 2024, 146 (9), 6134-6144. https://doi.org/10.1021/jacs.3c13687

(104) Wei, Z.; Jing, S.; Lü, X.; Wang, W.; Mele, G.; Yuan, K.; Ni, W.; Jiang, Z.-Y.; Guo, Y. Advancing Metal Organic Frameworks Screening and Remediation Condition Optimization for Removing Persistent Organic Pollutant via Integrated Machine Learning and Theoretical Calculation Insights. J. Colloid Interface Sci. 2026, 714, 140221. https://doi.org/10.1016/j.jcis.2026.140221

(105) Thaler, S.; Mayr, F.; Thomas, S.; Gagliardi, A.; Zavadlav, J. Active Learning Graph Neural Networks for Partial Charge Prediction of Metal-Organic Frameworks via Dropout Monte Carlo. Npj Comput. Mater. 2024, 10 (1), 86. https://doi.org/10.1038/s41524-024-01277-8

(106) Mukherjee, K.; Osaro, E.; J. Colón, Y. Active Learning for Efficient Navigation of Multi-Component Gas Adsorption Landscapes in a MOF. Digit. Discov. 2023, 2 (5), 1506–1521. https://doi.org/10.1039/D3DD00106G

(107) Schultz, L. E.; Gantzler, N.; Bobbitt, N. S.; Sava Gallis, D. F.; Dingreville, R. Graph Neural Network-Based Multi-Objective Bayesian Optimization for Enhanced Screening of Metal-Organic Frameworks with Optimal Separation Performance. J. Mater. Chem. A 2026, 14 (18), 10836– 10853. https://doi.org/10.1039/D5TA09133K

(108) Williams, C. K.; Rasmussen, C. E. Gaussian Processes for Machine Learning; MIT press Cambridge, MA, 2006; Vol. 2.

(109) Stein, M. L. Interpolation of Spatial Data: Some Theory for Kriging; Springer Science & Business Media, 1999.

(110) Snoek, J.; Larochelle, H.; Adams, R. P. Practical Bayesian Optimization of Machine Learning Algorithms. Adv. Neural Inf. Process. Syst. 2012, 25.

(111) Shahriari, B.; Swersky, K.; Wang, Z.; Adams, R. P.; De Freitas, N. Taking the Human Out of the Loop: A Review of Bayesian Optimization. Proc. IEEE 2016, 104 (1), 148–175. https: //doi.org/10.1109/JPR0C.2015.2494218

(112) Rodríguez-Miguel, P.; Jiménez, A.; Misol, A.; Trujillano, R.; Muñoz, H. J.; Gil, A.; Vicente, M. Á. Microwave-Assisted Rapid Synthesis of CAU23 and MIL53 MOFs from an Unusual Aluminum Source and Their Application in CO2 Capture. Chem. Eng. J. 2026, 543, 178648. https://doi. org/10.1016/j.cej.2026.178648

# Supporting Information

# ActMOF: Mapping Reticular Synthesis through Active Learning

Hongrui Sheng1,2, Shutian Liang3, Reid A. Coyle1, Shyam Chand Pal1, Peter Walther1, Bin Feng1,4, Xinyu Ding1,5, Zhiling Zheng1,4,\*

1Department of Chemistry, Washington University, St. Louis, Missouri 63130, United States   
2Department of Chemistry, Fudan University, Shanghai 200438, China

3Department of Energy, Environmental and Chemical Engineering, Washington University, St. Louis, Missouri 63130, United States

4Institute of Materials Science and Engineering, Washington University, St. Louis, Missouri 63130, United States

5Department of Chemistry, Imperial College London, London W12 0BZ, United Kingdom

\*Corresponding author. Email: z.z@wustl.edu

## Contents

S1 Materials and Methods S3   
S2 The ActMOF Landscape S3   
S2.1 Crystallinity Index q S3   
S2.2 Anchor Experiments S6   
S2.3 Variable Space S8   
S2.4 Emulator S10   
S2.5 Reliability S11   
S2.6 Structure S14   
S2.7 Validation Optima S16   
S3 Search Strategies S17   
S3.1 Protocol . S18   
S3.2 Gaussian Processes S20   
S3.3 Acquisition Functions S21   
S3.4 LLM Policies S22   
S3.5 Hybrid GP-LLM Policies S28   
S3.6 All Strategies S29   
S4 Benchmark Results S32   
S4.1 Metrics S32   
S4.2 All Strategies, n = 5 S33   
S4.3 Leading Group, n = 20 . S37   
S4.4 Robustness to the Construction of the Landscape S37   
S4.5 PI versus EI S40   
S4.6 Warm Start S41   
S4.7 Modeling the No-Product Region S44   
S5 Transfer Learning S45   
S5.1 Desktop Application S45   
S5.2 Reference and Residual GP S46   
S5.3 Calibration S47   
S5.4 Six Campaigns S47   
S6 Synthesis of the Validation MOFs S49   
S7 PXRD and Campaign Trajectories S50   
S8 Nitrogen Sorption S57   
References S58

## Section S1. Materials and Methods

Chemicals. All chemicals were purchased commercially and used without further purification (Table S1). The MOF-321 syntheses of Section S2.2 are taken from ref. [1] and were not repeated here.

Table S1: Chemicals used for the validation syntheses, with molar masses M. Deionized (DI) water was used throughout.
<table><tr><td>Chemical</td><td>M (g mol−¹) Use</td><td></td><td>Supplier</td><td>Purity</td></tr><tr><td>Aluminium chloride hexahy- drate,  $\mathrm { A l C l _ { 3 } \cdot 6 H _ { 2 } O }$ </td><td>241.43</td><td>Al source (all MOFs)</td><td>AK Scien- tific</td><td>≥99%</td></tr><tr><td>Isophthalic acid</td><td>166.13</td><td>linker, CAU-10, MTV- MIL-160</td><td>Ambeed</td><td>≥99.99%</td></tr><tr><td>Thiophene-2,5-dicarboxylic acid</td><td>172.16</td><td>linker, CAU-23, MTV- Ambeed MOF-303</td><td></td><td>≥97%</td></tr><tr><td>trans,trans-Muconic acid</td><td>142.11</td><td>linker, MIL-53-muc</td><td>A2B</td><td>≥97%</td></tr><tr><td>Furan-2,5-dicarboxylic acid</td><td>156.09</td><td>linker, MIL-160, MTV- MIL-160</td><td>Ambeed</td><td>≥99.30%</td></tr><tr><td>1H-Pyrazole-3,5-dicarboxylic acid hydrate</td><td>174.11</td><td>linker, MTV-MOF-303</td><td>Ambeed</td><td>≥99.75%</td></tr><tr><td>Sodium hydroxide Nitrogen</td><td></td><td>40.00 base (modulator) cooling of the</td><td>A2B mi- Linde</td><td>≥97% Grade 4.8</td></tr></table>

Microwave reactor. All syntheses were carried out in an Anton Paar Monowave 400 microwave reactor in G10 (“Glass 10") vials with a magnetic stir bar, at 850 W and 600 rpm stirring, at the temperature and hold time of each experiment (Section S2.3). After the hold time the vial was cooled to $5 0 ~ ^ { \circ } \mathrm { C }$ with nitrogen. The reaction mixture was transferred to a centrifuge tube, and the solid was isolated by centrifugation (6800 rpm, acceleration setting 9) and washed with deionized (DI) water (twice), ethanol (twice) and acetone (once). The washed solid was measured by PXRD directly (Sections S2.1 and S7); the samples for $\mathrm { N _ { 2 } }$ sorption were activated (Section S8). No unexpected or unusually high safety hazards were encountered.

## Section S2. The ActMOF Landscape

ActMOF is built in three steps: a crystallinity index q read from PXRD (Section S2.1); 95 measured MOF-321 syntheses (Section S2.2) placed in a five-dimensional encoded condition space (Section S2.3); and a rule-guided emulator that fills the remaining 6.1 million conditions (Section S2.4). Sections S2.5-S2.7 examine where the emulated landscape can be trusted, what it looks like and where the six validation optima lie on it.

## S2.1. Crystallinity Index q

(i) Definition. Following ref. [1], q is the intensity I of the diagnostic PXRD reflection divided by its full width at half maximum,

$$
q = \frac { I } { \mathrm { F W H M } } .\tag{S1}
$$

I is the intensity at the maximum of the diagnostic reflection (Table S2), and $\mathrm { F W H M } = x _ { 2 } - x _ { 1 }$ where x1 and x2 are the 2θ positions on either side of the maximum at which the intensity has fallen to $I / 2$ A pattern without this reflection gives $q = 0 \mathrm { \ ( F W H M } = 3 0 ^ { \circ }$ is entered as a placeholder that marks $q = 0 ; q$ is not computed from it). Because the intensity depends on the structure factor of the reflection, q is compared only between products of the same phase measured under the same protocol. The diagnostic reflection is the strongest reflection of the simulated pattern below $1 0 ^ { \circ } ~ 2 \theta$ (Table S2). Because it arises from the largest lattice spacing of the framework (about 10 Å), it reports the long-range order of the porous lattice.

Table S2: Strongest simulated reflection below $1 0 ^ { \circ } ~ 2 \theta$ (pymatgen, Cu $\mathrm { K } \alpha , \lambda = 1 . 5 4 0 6 \textup { \AA }$ , used as the diagnostic reflection. MTV-MIL-160 and MTV-MOF-303 are simulated from the MIL-160 and MOF-303 structures.
<table><tr><td>Framework</td><td>2θ (°)</td><td> $d \ ( \mathrm { \AA } )$ </td><td>hkl</td></tr><tr><td>CAU-10</td><td>8.30</td><td>10.65</td><td>(200)</td></tr><tr><td>CAU-23</td><td>8.43</td><td>10.48</td><td>(101)</td></tr><tr><td>MIL-53-muc</td><td>8.40</td><td>10.52</td><td>(110)</td></tr><tr><td>MIL-160</td><td>8.39</td><td>10.54</td><td>(200)</td></tr><tr><td>MTV-MIL-160</td><td>8.39</td><td>10.54</td><td>(200)</td></tr><tr><td>MTV-MOF-303</td><td>8.73</td><td>10.12</td><td>(011)</td></tr></table>

(ii) Sample mounting. All samples were prepared and measured by one procedure. About one spatula of the washed powder was placed in a 1.5 mL Eppendorf tube, acetone was added to the 1 mL mark and the suspension was mixed. Four drops were placed with a dropper on the center of a Rigaku silicon sample holder (Si(510), 5 mm × 0.2 mm well, part no. 1016621) and left to dry; the solid was then scraped into the well, filling it completely, and pressed with the spatula. Patterns were recorded as described in Section S7. To measure the mounting error directly, one powder (MIL-160, reflux 4 h) was loaded and measured, emptied from the holder and loaded again, five times in all: q varied with a coefficient of variation of 30.7% (I 19.1% FWHM 19.0%; n = 5; Figure S1). A single q value is therefore resolved to about ±30%, whereas q of the crystalline anchors spans three orders of magnitude (200–247,107).

![](images/9de6ba6fd66d3133b6bdb21ad14b791ece6002f1fa9476acf5ef5fddd48849f1.jpg)

![](images/688727b8eb7409fc7c78ae0c2383aa2cca0ce0d711bdbf9f9a7712646454a090.jpg)

![](images/9beea68fb32afec2e48b842ce9000f96bfd56dcde5bd56319ee44e4bb5e685dd.jpg)

![](images/ddffffd0d332b79e224a747d087b2b609676f99ff1f3ed6119acebd525c4e5b3.jpg)  
Figure S1: Repeated loading of one MIL-160 powder (reflux, 4 h): loaded, measured, emptied and loaded again, five loadings. (a) Diagnostic reflection; (b) q, (c) intensity I, (d) FWHM for each loading. Line: mean; band: ±1 s.d.; $\mathrm { C V } = \mathrm { s . d . / m e a n }$

(iii) Choice of reflection: CAU-23. The highest-q CAU-23 product (R00001, $q = 2 2 , 2 5 4 )$ is phase-pure CAU-23 and porous, with a BET area of $\mathrm { 1 1 7 3 ~ m ^ { 2 } g ^ { - 1 } }$ , 6% below the $1 2 5 0 ~ \mathrm { m } ^ { 2 } \mathrm { g } ^ { - 1 }$ reported for CAU-23 made by reflux2 (Section S8). Our own CAU-23 reflux product (6 h) was mistakenly left heating overnight afterwards and converted to another phase (strongest peak at $1 3 . 4 ^ { \circ } , S _ { \mathrm { B E T } } = 5 6 ~ \mathrm { m ^ { 2 } g ^ { - 1 } }$ ; Figure S2). Read at that peak, its q (20,354) would equal that of the microwave optimum, so $q$ is read only after the target phase is confirmed, at the reflection listed in Table S2.

b  
![](images/8143c8d5a7efca659286542f566252c8104e1a90638d6b479eb700cd365b3cab.jpg)

![](images/a6b12e4057902a8253a7d66af99b107d93a825a71ae6517488bd8284727d3e19.jpg)  
Figure S2: CAU-23 products. (a) Normalized PXRD patterns, offset: microwave optimum R00001 (MW), reflux products after 2 and 4 h, and the 6 h reflux product that was afterwards heated overnight (one scan and five loadings), below the pattern simulated from the CAU-23 structure (CCDC 1878820). Dashed lines: position of the diagnostic reflection in the simulated CAU-23 pattern (8.4°) and the 13.4° peak of the phase formed on overnight heating. (b) Diagnostic region on a common absolute scale. q and $S _ { \mathrm { B E T } }$ of these samples: Section S2.1 and Table S26.

(iv) Reference scale: MIL-160. To place q on a known scale, MIL-160 was also made by the conventional reflux route[3] (4 and 6 h; five loadings each). The best microwave product of the campaign (R00007, 55 min, $1 0 0 \ ^ { \circ } \mathrm { C } , q = 4 4 , 3 9 5 )$ lies within the range of both reflux products $( q = 6 0 , 1 0 6 \pm 8 , 2 4 6$ and $^ { 3 6 , 4 0 1 \pm 5 , 3 3 7 }$ , mean ± s.e.m.; Figure S3), and its BET area is 80% of that of the 6 h product (Section S8).

(v) q and porosity. Table S3 collects q and the BET area of the four samples for which both were measured. The two microwave optima, which have the highest $q$ of their campaigns, are porous, with 94% (CAU-23) and 63% (MIL-160) of the literature BET area, and the MIL-160 sample of higher BET area (reflux) has a q in the same range as the microwave sample. The CAU-23 product that had converted to another phase has almost no CAU-23 reflection left and is almost non-porous. Within one phase, a high q at the diagnostic reflection therefore goes with a porous product; $q$ ranks samples of one phase and is not a measure of surface area.

Table S3: q at the diagnostic reflection (Table S2) and BET area of the samples measured by both methods. $S _ { \mathrm { B E T } }$ in $\mathrm { m ^ { 2 } g ^ { - 1 } } ~ ( p / p _ { 0 } ~ 0 . 0 5 \mathrm { - 0 . 1 0 } ,$ Section S8); q of the reflux products, mean of five loadings.
<table><tr><td>Framework</td><td>Sample</td><td>q</td><td> $S _ { \mathrm { B E T } }$ </td><td>Lit.  $S _ { \mathrm { B E T } }$ </td></tr><tr><td>CAU-23</td><td>microwave optimum, R00001</td><td>22,254</td><td>1173</td><td>1250[2]</td></tr><tr><td rowspan="3">MIL-160</td><td>reflux 6 h, mistakenly heated overnight</td><td>1,722</td><td>56</td><td></td></tr><tr><td>microwave optimum, R00007</td><td>44,395</td><td>729</td><td>1150[4]</td></tr><tr><td>reflux, 6 h</td><td>36,401</td><td>908</td><td></td></tr></table>

![](images/29c1a4ad6a96bbe455b43160209f3791cb22a93cb883ed69a041253345450ff1.jpg)  
Figure S3: q of the MIL-160 microwave optimum R00007 (star, one loading) against reflux products after 4 and 6 h (dots, five loadings each; black, mean ± s.e.m.). The microwave value lies inside the range of the five loadings of both reflux products.

## S2.2. Anchor Experiments

The measured data are the microwave syntheses of MOF-321 (MOF-LA2-1; $\mathrm { A l ^ { 3 + } }$ , H2PZVDC, NaOH) reported in ref. [1], reused here. We call these measured conditions anchor experiments, or anchors: the emulator is trained on them and keeps their measured values (Section S2.4) so they anchor the emulated landscape to experiment. Table S4 lists all 96 recorded entries. Entries 26 and 31 are identical in condition and outcome, so 95 unique conditions remain: 67 crystalline $( q > 0 )$ and 28 without product $( q = 0 )$ , with q up to 247,107 (entry 73). The entries and their metadata (recording order, conditions, measured I and FWHM) are also provided as a data file in the GitHub repository.

Table S4: All 96 recorded MOF-321 entries. Encoded vector x as in Table S5. †Entries 26 and 31 are identical; the data set was deduplicated to 95 unique conditions.
<table><tr><td> $\mathrm { N o . }$ </td><td>Encoded x</td><td>Al (mL)</td><td>NaOH (mL)</td><td>H₂O (mL)</td><td>t (min)</td><td> $T ~ ( ^ { \circ } \mathrm { C } )$ </td><td></td><td>I FWHM (°)</td><td>q</td></tr><tr><td>1</td><td>(35, 5, 22, 9, 14)</td><td>3.5</td><td>0.5</td><td>2.2</td><td>45</td><td>70</td><td>0</td><td>30</td><td>0</td></tr><tr><td>2</td><td>(68, 7, 15, 12, 24)</td><td>6.8</td><td>0.7</td><td>1.5</td><td>60</td><td>120</td><td>0</td><td>30</td><td>0</td></tr><tr><td>3</td><td>(12, 9, 1, 7, 29)</td><td>1.2</td><td>0.9</td><td>0.1</td><td>35</td><td>145</td><td>785</td><td>2.2</td><td>357</td></tr><tr><td>4</td><td>(36, 10, 24, 3, 18)</td><td>3.6</td><td>1.0</td><td>2.4</td><td>15</td><td>90</td><td>0</td><td>30</td><td>0</td></tr><tr><td>5</td><td>(55, 11, 8, 2, 15)</td><td>5.5</td><td>1.1</td><td>0.8</td><td>10</td><td>75</td><td>0</td><td>30</td><td>0</td></tr><tr><td>6</td><td>(69, 6, 23, 1, 20)</td><td>6.9</td><td>0.6</td><td>2.3</td><td>5</td><td>100</td><td>605</td><td>30</td><td>0</td></tr><tr><td>No.</td><td>Encoded x</td><td>Al (mL)</td><td>NaOH (mL)</td><td>H₂O (mL)</td><td>t (min)</td><td>T (°C)</td><td></td><td>I FWHM (°)</td><td>q</td></tr><tr><td>7</td><td>(14, 12, 2, 11, 15)</td><td>1.4</td><td>1.2</td><td>0.2</td><td>55</td><td>75</td><td>0</td><td>30</td><td>0</td></tr><tr><td>8</td><td>(42, 7, 27, 6, 17)</td><td>4.2</td><td>0.7</td><td>2.7</td><td>30</td><td>85</td><td>0</td><td>30</td><td>0</td></tr><tr><td>9</td><td>(9, 14, 7, 4, 11)</td><td>0.9</td><td>1.4</td><td>0.7</td><td>20</td><td>55</td><td>182</td><td>30</td><td>0</td></tr><tr><td>10</td><td>(45, 6, 11, 8, 28)</td><td>4.5</td><td>0.6</td><td>1.1</td><td>40</td><td>140</td><td>0</td><td>30</td><td>0</td></tr><tr><td>11</td><td>(27, 12, 1, 2, 25)</td><td>2.7</td><td>1.2</td><td>0.1</td><td>10</td><td>125</td><td>880</td><td>3.9</td><td>226</td></tr><tr><td>12</td><td>(41, 14, 5, 3, 16)</td><td>4.1</td><td>1.4</td><td>0.5</td><td>15</td><td>80</td><td>0</td><td>30</td><td>0</td></tr><tr><td>13</td><td>(5, 5, 0, 1, 30)</td><td>0.5</td><td>0.5</td><td>0.0</td><td>555</td><td>150</td><td>1,457</td><td>0.32</td><td>4,553</td></tr><tr><td>14</td><td>(5, 15, 0, 1, 30)</td><td>0.5</td><td>1.5</td><td>0.0</td><td></td><td>150</td><td>0</td><td>30</td><td>0</td></tr><tr><td>15</td><td>(75, 15, 0, 1, 30)</td><td>7.5</td><td>1.5</td><td>0.0</td><td></td><td>150</td><td>0</td><td>30</td><td>0</td></tr><tr><td>16</td><td>(16, 7, 22, 3, 20)</td><td>1.6</td><td>0.7</td><td>2.2</td><td>15</td><td>100</td><td>2,345</td><td>0.37</td><td>6,338</td></tr><tr><td>17</td><td>(16, 9, 23, 4, 21)</td><td>1.6</td><td>0.9</td><td>2.3</td><td>20</td><td>105</td><td>960</td><td>1.1</td><td>873</td></tr><tr><td>18</td><td>(15, 7, 12, 11, 19)</td><td>1.5</td><td>0.7</td><td>1.2</td><td>55</td><td>95</td><td>16,751</td><td>0.22</td><td>76,141</td></tr><tr><td>19</td><td>(6, 5, 29, 11, 27)</td><td>0.6</td><td>0.5</td><td>2.9</td><td>55</td><td>135</td><td>7,575</td><td>0.18</td><td>42,083</td></tr><tr><td>20</td><td>(62, 7, 26, 11, 26)</td><td>6.2</td><td>0.7</td><td>2.6</td><td>55</td><td>130</td><td>0</td><td>30</td><td>00</td></tr><tr><td>21</td><td>(5, 15, 2, 11, 16)</td><td>0.5</td><td>1.5</td><td>0.2</td><td>55</td><td>80</td><td>0</td><td>30</td><td></td></tr><tr><td>22</td><td>(33, 10, 7, 11, 18)</td><td>3.3</td><td>1.0</td><td>0.7</td><td>55</td><td>90</td><td>8,060</td><td>0.25</td><td>32,240</td></tr><tr><td>23</td><td>(9, 6, 1, 12, 19)</td><td>0.9</td><td>0.6</td><td>0.1</td><td>60</td><td>95</td><td>1,318</td><td>0.4</td><td>3,295</td></tr><tr><td>24</td><td>(48, 14, 3, 12, 23)</td><td>4.8</td><td>1.4</td><td>0.3</td><td>60</td><td>115</td><td>920</td><td>4.6</td><td>200</td></tr><tr><td>25</td><td>(19, 8, 12, 11, 16)</td><td>1.9</td><td>0.8</td><td>1.2</td><td>55</td><td>80</td><td>0</td><td>30</td><td>0</td></tr><tr><td>26†</td><td>(15, 7, 14, 11, 24)</td><td>1.5</td><td>0.7</td><td>1.4</td><td>55</td><td>120</td><td>23,969</td><td>0.21</td><td>114,138</td></tr><tr><td>27</td><td>(12, 8, 10, 11, 18)</td><td>1.2</td><td>0.8</td><td>1.0</td><td>55</td><td>90</td><td>7,305</td><td>0.15</td><td>48,700</td></tr><tr><td>28</td><td>(15, 7, 13, 11, 18)</td><td>1.5</td><td>0.7</td><td>1.3</td><td>55</td><td>90</td><td>7,703</td><td>0.27</td><td>28,530</td></tr><tr><td>29</td><td>(14, 7, 11, 11, 20)</td><td>1.4</td><td>0.7</td><td>1.1</td><td>55</td><td>100</td><td>12,434</td><td>0.24</td><td>51,808</td></tr><tr><td>30</td><td>(7, 6, 20, 11, 15)</td><td>0.7</td><td>0.6</td><td>2.0</td><td>55</td><td>75</td><td>0</td><td>30</td><td>0</td></tr><tr><td>31†</td><td>(15, 7, 14, 11, 24)</td><td>1.5</td><td>0.7</td><td>1.4</td><td>55</td><td>120</td><td>23,969</td><td>0.21</td><td>114,138</td></tr><tr><td>32</td><td>(15, 6, 10, 11, 14)</td><td>1.5</td><td>0.6</td><td>1.0</td><td>55</td><td>70</td><td>0</td><td>30</td><td>0</td></tr><tr><td>33</td><td>(15, 7, 10, 11, 12)</td><td>1.5</td><td>0.7</td><td>1.0</td><td>55</td><td>60</td><td>0</td><td>30</td><td>0</td></tr><tr><td>34</td><td>(15, 8, 12, 11, 19)</td><td>1.5</td><td>0.8</td><td>1.2</td><td>55</td><td>95</td><td>1,103</td><td>0.8</td><td>1,379</td></tr><tr><td>35</td><td>(9, 8, 13, 7, 20)</td><td>0.9</td><td>0.8</td><td>1.3</td><td>35</td><td>100</td><td>0</td><td>30</td><td>0</td></tr><tr><td>36</td><td>(21, 6, 13, 12, 19)</td><td>2.1</td><td>0.6</td><td>1.3</td><td>60</td><td>95</td><td>17,269</td><td>0.21</td><td>82,233</td></tr><tr><td>37</td><td>(6, 6, 12, 11, 20)</td><td>0.6</td><td>0.6</td><td>1.2</td><td>55</td><td>100</td><td>1,077</td><td>0.48</td><td>2,244</td></tr><tr><td>38</td><td>(15, 8, 9, 12, 19)</td><td>1.5</td><td>0.8</td><td>0.9</td><td>60</td><td>95</td><td>903</td><td>2</td><td>452</td></tr><tr><td>39</td><td>(75, 12, 6, 11, 18)</td><td>7.5</td><td>1.2</td><td>0.6</td><td>55</td><td>90</td><td>0</td><td>30</td><td>0</td></tr><tr><td>40</td><td>(25, 13, 14, 7, 24)</td><td>2.5</td><td>1.3</td><td>1.4</td><td>35</td><td>120</td><td>0</td><td>30</td><td>0</td></tr><tr><td>41</td><td>(17, 13, 16, 12, 24)</td><td>1.7</td><td>1.3</td><td>1.6</td><td>60</td><td>120</td><td>0</td><td>30</td><td>0</td></tr><tr><td>42</td><td>(38, 13, 16, 8, 24)</td><td>3.8</td><td>1.3</td><td>1.6</td><td>40</td><td>120</td><td>0</td><td>30</td><td>0</td></tr><tr><td>43</td><td>(65, 5, 0, 12, 24)</td><td>6.5</td><td>0.5</td><td>0.0</td><td>60</td><td>120</td><td>0</td><td>30</td><td>0</td></tr><tr><td>44</td><td>(74, 7, 23, 1, 16)</td><td>7.4</td><td>0.7</td><td>2.3</td><td>5</td><td>80</td><td>0</td><td>30</td><td>0</td></tr><tr><td>45</td><td>(5, 7, 21, 12, 20)</td><td>0.5</td><td>0.7</td><td>2.1</td><td>60</td><td>100</td><td>0</td><td>30</td><td>0</td></tr><tr><td>46</td><td>(15, 7, 16, 12, 29)</td><td>1.5</td><td>0.7</td><td>1.6</td><td>60</td><td>145</td><td>31,082</td><td>0.2</td><td>155,410</td></tr><tr><td>47 48</td><td>(16, 7, 15, 11, 18) (62, 10, 12, 10, 17)</td><td>1.6 6.2</td><td>0.7 1.0</td><td>1.5 1.2</td><td>55 50</td><td>90 85</td><td>12,791 0</td></tr><tr><td> $\mathrm { N o . }$ </td><td>Encoded x</td><td>Al (mL)</td><td>NaOH (mL)</td><td> $\mathrm { H _ { 2 } O }$  (mL)</td><td>t (min)</td><td> $T ~ ( ^ { \circ } \mathrm { C } )$ </td><td></td><td>I FWHM (°)</td><td>q</td></tr><tr><td>67</td><td>(13, 6, 25, 12, 24)</td><td>1.3</td><td>0.6</td><td>2.5</td><td>60</td><td>120</td><td>23,658</td><td>0.28</td><td>84,493</td></tr><tr><td>68</td><td>(13, 7, 11, 11, 23)</td><td></td><td>1.3</td><td>0.7</td><td>1.1 55</td><td>115</td><td>17,558</td><td>0.26</td><td>67,531</td></tr><tr><td>69</td><td>(14, 7, 7, 11, 26)</td><td>1.4</td><td>0.7</td><td>0.7</td><td>55</td><td>130</td><td>7,418</td><td>0.29</td><td>25,579</td></tr><tr><td>70</td><td>(15, 6, 23, 12, 24)</td><td>1.5</td><td>0.6</td><td>2.3</td><td>60</td><td>120</td><td>32,110</td><td>0.3</td><td>107,033</td></tr><tr><td>71</td><td>(15, 5, 8, 12, 23)</td><td>1.5</td><td>0.5</td><td>0.8</td><td>60</td><td>115</td><td>20,121</td><td>0.21</td><td>95,814</td></tr><tr><td>72</td><td>(15, 6, 14, 12, 21)</td><td>1.5</td><td>0.6</td><td>1.4</td><td>60</td><td>105</td><td>19,118</td><td>0.25</td><td>76,472</td></tr><tr><td>73</td><td>(13, 5, 10, 11, 24)</td><td>1.3</td><td>0.5</td><td>1.0</td><td>55</td><td>120</td><td>34,595</td><td>0.14</td><td>247,107</td></tr><tr><td>74</td><td>(17, 6, 11, 11, 22)</td><td>1.7</td><td>0.6</td><td>1.1</td><td>55</td><td>110</td><td>17,880</td><td>0.22</td><td>81,273</td></tr><tr><td>75</td><td>(19, 5, 2, 12, 24)</td><td>1.9</td><td>0.5</td><td>0.2</td><td>60</td><td>120</td><td>6,978</td><td>0.16</td><td>43,613</td></tr><tr><td>76</td><td>(6, 6, 19, 12, 30)</td><td>0.6</td><td>0.6</td><td>1.9</td><td>60</td><td>150</td><td>0</td><td>30</td><td>0</td></tr><tr><td>77</td><td>(25, 7, 18, 11, 24)</td><td>2.5</td><td>0.7</td><td>1.8</td><td>55</td><td>120</td><td>8,241</td><td>0.26</td><td>31,696</td></tr><tr><td>78</td><td>(27, 7, 27, 12, 26)</td><td>2.7</td><td>0.7</td><td>2.7</td><td>60</td><td>130</td><td>7,910</td><td>0.23</td><td>34,391</td></tr><tr><td>79</td><td>(15, 5, 8, 11, 23)</td><td>1.5</td><td>0.5</td><td>0.8</td><td>55</td><td>115</td><td>26,277</td><td>0.22</td><td>119,441</td></tr><tr><td>80</td><td>(15, 5, 11, 10, 21)</td><td>1.5</td><td>0.5</td><td>1.1</td><td>50</td><td>105</td><td>18,112</td><td>0.29</td><td>62,455</td></tr><tr><td>81</td><td>(24, 5, 30, 12, 24)</td><td>2.4</td><td>0.5</td><td>3.0</td><td>60</td><td>120</td><td>4,394</td><td>0.31</td><td>14,174</td></tr><tr><td>82</td><td>(13, 7, 15, 12, 25)</td><td>1.3</td><td>0.7</td><td>1.5</td><td>60</td><td>125</td><td>27,452</td><td>0.24</td><td>114,383</td></tr><tr><td>83</td><td>(16, 7, 11, 12, 24)</td><td>1.6</td><td>0.7</td><td>1.1</td><td>60</td><td>120</td><td>16,348</td><td>0.28</td><td>58,386</td></tr><tr><td>84</td><td>(14, 6, 15, 12, 26)</td><td>1.4</td><td>0.6</td><td>1.5</td><td>60</td><td>130</td><td>30,033</td><td>0.23</td><td>130,578</td></tr><tr><td>85</td><td>(14, 7, 15, 11, 26)</td><td>1.4</td><td>0.7</td><td>1.5</td><td>55</td><td>130</td><td>28,764</td><td>0.29</td><td>99,186</td></tr><tr><td>86</td><td>(18, 7, 10, 12, 23)</td><td>1.8</td><td>0.7</td><td>1.0</td><td>60</td><td>115</td><td>28,299</td><td>0.22</td><td>128,632</td></tr><tr><td>87</td><td>(10, 7, 13, 12, 29)</td><td>1.0</td><td>0.7</td><td>1.3</td><td>60</td><td>145</td><td>1,082</td><td>0.41</td><td>2,639</td></tr><tr><td>88</td><td>(23, 7, 2, 12, 19)</td><td>2.3</td><td>0.7</td><td>0.2</td><td>60</td><td>95</td><td>20,334</td><td>0.27</td><td>75,311</td></tr><tr><td>89</td><td>(18, 7, 1, 4, 19)</td><td>1.8</td><td>0.7</td><td>0.1</td><td>20</td><td>95</td><td>19,300</td><td>0.32</td><td>60,313</td></tr><tr><td>90</td><td>(19, 6, 4, 12, 25)</td><td>1.9</td><td>0.6</td><td>0.4</td><td>60</td><td>125</td><td>12,874</td><td>0.21</td><td>61,305</td></tr><tr><td>91</td><td>(17, 7, 10, 12, 25)</td><td>1.7</td><td>0.7</td><td>1.0</td><td>60</td><td>125</td><td>28,836</td><td>0.26</td><td>110,908</td></tr><tr><td>92</td><td>(16, 6, 6, 5, 23)</td><td>1.6</td><td>0.6</td><td>0.6</td><td>25</td><td>115</td><td>16,918</td><td>0.31</td><td>54,574</td></tr><tr><td>93</td><td>(20, 6, 0, 12, 23)</td><td>2.0</td><td>0.6</td><td>0.0</td><td>60</td><td>115</td><td>11,668</td><td>0.3</td><td>38,893</td></tr><tr><td>94</td><td>(17, 7, 20, 11, 21)</td><td>1.7</td><td>0.7</td><td>2.0</td><td>55</td><td>105</td><td>26,346</td><td>0.28</td><td>94,093</td></tr><tr><td>95</td><td>(24, 7, 22, 8, 21)</td><td>2.4</td><td></td><td>0.7</td><td>2.2</td><td>40</td><td>105 17,895 11,657</td><td>0.31</td><td>57,726</td></tr><tr><td>96</td><td>(25, 7, 19, 10, 27)</td><td></td><td>2.5</td><td>0.7</td><td>1.9</td><td>50</td><td>135</td><td></td><td>0.34 34,285</td></tr></table>

## S2.3. Variable Space

Every condition is a vector of five integers ${ \bf x } = ( x _ { 1 } , \dots , x _ { 5 } )$ converted to laboratory quantities by

$$
V _ { \mathrm { A l } } = 0 . 1 x _ { 1 } { \mathrm { ~ m L } } , \qquad x _ { 1 } \in \{ 5 , \ldots , 7 5 \} ,\tag{S2}
$$

$$
V _ { \mathrm { N a O H } } = 0 . 1 x _ { 2 } \ \mathrm { m L } , \qquad x _ { 2 } \in \{ 5 , \ldots , 1 5 \} ,\tag{S3}
$$

$$
V _ { \mathrm { H _ { 2 } O } } = 0 . 1 x _ { 3 } \ \mathrm { m L } , \qquad x _ { 3 } \in \{ 0 , \ldots , 3 0 \} ,\tag{S4}
$$

$$
t = 5 x _ { 4 } \mathrm { m i n } , x _ { 4 } \in \{ 1 , \ldots , 1 2 \} ,\tag{S5}
$$

$$
T = 5 x _ { 5 } ^ { \circ } \mathrm { C } , \qquad x _ { 5 } \in \{ 1 0 , \ldots , 3 0 \} ,\tag{S6}
$$

where $V _ { \mathrm { A l } }$ and $V _ { \mathrm { N a O H } }$ are volumes of $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and NaOH stock solutions, both 100 mg mL−1 in DI water, and $V _ { \mathrm { H _ { 2 } O } }$ is additional DI water (Table S5). One step of $x _ { 1 }$ or $x _ { 2 }$ is therefore 10 mg of $\mathrm { A l C l _ { 3 } \cdot 6 H _ { 2 } O \ ( 0 . 0 4 1 }$ mmol) or 10 mg of NaOH (0.25 mmol). The variables are coded as integers because the syntheses are set up in fixed increments (0.1 mL of a stock solution, 5 min, $5 ~ ^ { \circ } \mathrm { C } )$ : with integer codes, every point of the grid is a recipe that can be dispensed exactly, all search methods choose from the same finite set of conditions, and two conditions are the same experiment exactly when their codes are equal. The full Cartesian product,

$$
7 1 \times 1 1 \times 3 1 \times 1 2 \times 2 1 = 6 , 1 0 1 , 1 7 2 ,\tag{S7}
$$

is the ActMOF candidate set. Every search policy and every wet-lab campaign uses the same encoding, so any proposal is directly a recipe. The anchors cover only part of this space (Figure S4): the first entries are spread over the whole range of every variable and mostly give no

product, whereas the later ones concentrate at low $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and NaOH volumes, long hold times and $9 0 { - } 1 5 0 \ ^ { \circ } \mathrm { C } ,$ where the original campaign found its products. Of the 71 $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ volumes on the grid, 36 were used at least once.

Table S5: Encoded synthesis variables.
<table><tr><td></td><td>Index Program variable</td><td>Laboratory quantity</td><td>Integer range</td><td>Step</td><td>Physical range</td></tr><tr><td> $x _ { 1 }$ </td><td>metal_amount</td><td> $\mathrm { A l C l _ { 3 } \cdot 6 H _ { 2 } O , 1 0 0 ~ m g m L ^ { - 1 } }$ </td><td>5-75</td><td>0.1 mL</td><td>0.5–7.5 mL</td></tr><tr><td> $x _ { 2 }$ </td><td>modulator</td><td> $\mathrm { N a O H , 1 0 0 ~ m g m L ^ { - 1 } }$ </td><td>5-15</td><td>0.1 mL</td><td>0.5-1.5 mL</td></tr><tr><td> $x _ { 3 }$ </td><td>add_solvent</td><td>Additional DI water</td><td>0-30</td><td>0.1 mL</td><td>0-3.0 mL</td></tr><tr><td> $x _ { 4 }$ </td><td>reaction_time</td><td>Microwave hold time</td><td>1-12</td><td>5 min</td><td>5–60 min</td></tr><tr><td> $x _ { 5 }$ </td><td></td><td>reaction_temperature Microwave temperature</td><td>10-30</td><td> $\mathrm { 5 ~ ^ { \circ } C }$ </td><td> $5 0 { - } 1 5 0 ~ ^ { \circ } \mathrm { C }$ </td></tr></table>

![](images/c280b7ded724a33ae279d28fc608c22925d23f3fc71c6f10680074b56dba8605.jpg)  
Figure S4: The anchor experiments in the order in which they were recorded (entries 1-96, Table S4). (a) q of every entry, with the top-1% and top-0.1% thresholds of the landscape (Table S10). (b-f) The five synthesis variables of every entry; the vertical axes span the full range of the grid. Dotted line: end of the first 60 entries, used in Sections S2.5 and S4.4.

## S2.4. Emulator

The emulator predicts I and FWHM for any x in three stages (Figure S5).

(i) Random forests. Two independent RandomForestRegressor models (scikit-learn), one for the measured peak height and one for the FWHM, take the five encoded variables as input and give $\hat { I } _ { \mathrm { R F } } ( \mathbf { x } )$ and $\hat { F } _ { \mathrm { R F } } ( { \bf x } ) ; q$ itself is not a training target. Training samples without product carry the placeholder FWHM $= 3 0 ^ { \circ }$ (Table S4). Settings and five-fold cross-validation are given in Table S6. The FWHM model has two tasks, and its overall error mixes them: the 28 samples without product sit at $3 0 ^ { \circ }$ and the 67 crystalline ones mostly at $0 . 2 { - } 0 . 4 ^ { \circ }$ (median 0.26°), so the RMSE of $7 . 3 ^ { \circ }$ mainly reflects how well the two groups are told apart. Taken separately, all 28 held-out samples without product are recognized as such $( \hat { I } < 3 0 0 0$ or $\hat { F } > 3 ^ { \circ } )$ , 55 of the 67 held-out crystalline samples are predicted as products, and for these 55 the median error of the FWHM is $0 . 0 6 ^ { \circ }$ (21% of the measured value).

Table S6: Random-forest settings (scikit-learn defaults unless stated) and five-fold crossvalidation (KFold, shuffled, random\_state = 42; mean $\pm \mathrm { \ s . d . }$ over folds).
<table><tr><td>Setting</td><td>Value</td><td></td><td> $R ^ { 2 }$ </td><td>RMSE</td></tr><tr><td>n_estimators</td><td>400</td><td>Peak height</td><td> $0 . 6 4 \pm 0 . 1 3$ </td><td> $6 , 6 8 5 \pm 1 , 5 3 5$ </td></tr><tr><td>max_depth</td><td>None</td><td>FWHM</td><td> $0 . 6 6 \pm 0 . 2 0$ </td><td> $7 . 3 \pm 2 . 1 ^ { \circ \ a }$ </td></tr><tr><td>min_samples_split/leaf</td><td> $2 ~ / ~ 1$ </td><td></td><td></td><td></td></tr><tr><td>max_features</td><td>1.0 (all five)</td><td></td><td></td><td></td></tr><tr><td>bootstrap</td><td>True</td><td></td><td></td><td></td></tr><tr><td>random_state</td><td>42</td><td></td><td></td><td></td></tr></table>

aDominated by the samples without product (placeholder $\mathrm { F W H M } = 3 0 ^ { \circ } )$ ; for crystalline samples predicted as products, the median error is $0 . 0 6 ^ { \circ }$ (see text).

(ii) Local neighbor blending. Close to an anchor, the forest prediction is pulled toward the measured values. Distances between conditions are counted in grid steps, as the largest difference in any one variable (Chebyshev distance):

$$
d _ { \infty } ( \mathbf { x } , \mathbf { x } _ { i } ) = \operatorname* { m a x } _ { k } \left| x _ { k } - x _ { i , k } \right| .\tag{S8}
$$

$\mathcal { N } _ { 1 } ( \mathbf { x } )$ denotes the set of anchors within one step of x in every variable $( d _ { \infty } \leq 1 )$ , and $\overline { { I } } _ { \mathcal { N } _ { 1 } }$ their mean measured intensity. If this set contains at least one anchor (Rule 1), the prediction is 80% anchor mean and 20% forest. Otherwise, $\mathcal { N } _ { 2 } ( \mathbf { x } )$ denotes the set of anchors two steps away in one or two variables and within one step in the others, and the prediction is a 50:50 blend (Rule 2):

$$
\hat { I } = 0 . 8 \overline { { { I } } } _ { \mathcal { N } _ { 1 } } + 0 . 2 \hat { I } _ { \mathrm { R F } } \quad \mathrm { o r } \quad \hat { I } = 0 . 5 \overline { { { I } } } _ { \mathcal { N } _ { 2 } } + 0 . 5 \hat { I } _ { \mathrm { R F } } ,\tag{S9}
$$

and likewise for FWHM. If neither set contains an anchor, the forest prediction is kept.

(iii) Hard boundaries.

$$
q _ { \mathrm { f i n a l } } = \left\{ { 0 , \atop { \hat { I } / \hat { F } , \mathrm { o t h e r w i s e } . } } \right.\tag{S10}
$$

These boundaries apply to emulator output only, not to measured data.

![](images/56e9085daf261f2fb40ccf8eea6f2cc8b7db65adfd3f03f316533bfa94b4432f.jpg)  
Figure S5: Construction of the emulator in three stages: random-forest prediction, blending with nearby anchors, hard thresholds. $d _ { \infty }$ , Chebyshev distance (grid steps) to the nearest anchor; Rule 2 applies when one or two variables are two steps away and the others within one step.

## S2.5. Reliability

The emulator is not used to predict any single reaction in one shot. A search benchmark needs two things from it: that conditions are ranked as experiments would rank them, and that the productive region is not inflated. We therefore judge the search strategies by their behavior on the landscape and by six wet-lab campaigns, rather than treating the fitted landscape as a definitive physical model of MOF-321 crystallization. Both requirements were tested on the measured data.

Leave-one-out test. Each anchor was removed in turn, the emulator was refit on the other 94 (five forest random states), and the removed condition was predicted. The error is measured on the logarithmic scale and averaged over the five forest random states $s ,$

$$
e _ { \mathrm { l o g } } ( i ) = \frac { 1 } { 5 } \sum _ { s = 1 } ^ { 5 } \left| \log ( 1 + q _ { \mathrm { p r e d } , i , s } ) - \log ( 1 + q _ { \mathrm { t r u e } , i } ) \right| ;\tag{S11}
$$

$e _ { \mathrm { l o g } } = 0$ is a perfect prediction; a prediction of 20,000 for a measured q of 10,000 (or the reverse) gives $e _ { \mathrm { l o g } } = 0 . 6 9$ , and a prediction of no product for a measured q of 10,000 gives $e _ { \mathrm { l o g } } = 9 . 2 $ Table S7 lists the result for every anchor. All 28 anchors without product were predicted as $q = 0$ . For the 67 crystalline anchors, predicted and measured q rank alike (Spearman $\rho = 0 . 7 8 )$ 57% are predicted between half and twice the measured $q \ ( e _ { \mathrm { l o g } } \ \leq \ 0 . 6 9 )$ , and 9 are wrongly predicted as no product. The errors are one-sided: the emulator can miss part of the productive region, which makes the benchmark harder, but it never predicted a product where none was measured.

Occurrence of prediction errors. Two quantities tell whether a prediction is backed by data. The first is $d _ { \mathrm { m i n } }$ , the Chebyshev distance to the nearest other anchor (Eq. S8). The second measures how the five forests (fitted with five random states) vote at this condition: $B _ { I }$ is the fraction of them whose predicted intensity lies below the intensity threshold $( \hat { I } < 3 0 0 0 )$ $B _ { F }$ the fraction whose predicted FWHM lies above the width threshold $( { \hat { F } } > 3 ^ { \circ } ;$ Eq. S10), and $B = \operatorname* { m a x } ( B _ { I } , B _ { F } )$ . B = 0 means that all five forests predict a product, $B = 1$ that all five predict none. A prediction is flagged when it is far from data and most forests predict no product:

$$
\mathrm { f l a g } ( \mathbf { x } ) = \mathbb { 1 } [ d _ { \mathrm { m i n } } ( \mathbf { x } ) > 3 ] \cdot \mathbb { 1 } [ B ( \mathbf { x } ) > 0 . 5 ] ,\tag{S12}
$$

where 1[·] is 1 if the condition in brackets holds and 0 otherwise. A flag is a warning, not an error: it marks a prediction of "no product" that no nearby measurement supports. The flagged anchors without product (Table S7) were all predicted correctly.

The large errors are confined to the flagged predictions (Figure S6, shaded region). The 49 crystalline anchors within three steps of another anchor have a median $e _ { \log }$ of 0.51, and none has $e _ { \mathrm { l o g } } > 5$ . Of the 18 that lie farther away, the 8 for which the forests predict a product are predicted as well (median 0.32); the 10 for which most forests predict no product are the 10 flagged ones, and all of them are large errors (median 7.99): these are isolated products that the emulator cannot know about once they are removed.

![](images/c0a9894c20533eb9a4a945397beb0f10f6f8e35f3b436a62c41054f15fefc41d.jpg)  
Figure S6: Leave-one-out predictions of the 95 anchors. Horizontal axis, $d _ { \mathrm { m i n } } .$ grid steps to the nearest other anchor; vertical axis, B, the share of the five forests that predict no product; color, error $e _ { \log }$ (light, small; dark, large). Shaded: flagged region of Eq. S12 $( d _ { \operatorname* { m i n } } > 3$ and $B > 0 . 5 )$ i.e. far from data and predicted as no product by most forests. Outlined and numbered (entry of Table S4): the ten predictions with $e _ { \mathrm { l o g } } > 5$ , all of them in the flagged region. $d _ { \mathrm { m i n } }$ and B take only a few values, so many anchors fall on the same position; they are drawn side by side to keep every one visible.

Table S7: Leave-one-out prediction of every anchor. $\mathrm { N o . , }$ entry of Table S4; $q ,$ measured; ${ \hat { q } } ,$ mean prediction of the five forests refitted without this anchor; $e _ { \mathrm { l o g } }$ , Eq. S11 (lower is better). \*Flagged by Eq. S12: far from the other anchors and predicted as no product by most forests. For an anchor without product $( q = 0 , \hat { q } = 0 )$ , the flag means that this correct prediction is not supported by a nearby measurement.
<table><tr><td> $\mathrm { N o . }$ </td><td>q</td><td> $\hat { q }$ </td><td> $e _ { \mathrm { l o g } }$ </td><td> $\mathrm { N o . }$ </td><td>q</td><td> $\hat { q }$ </td><td> $e _ { \mathrm { l o g } }$ </td></tr><tr><td> $1 ^ { * }$ </td><td>0</td><td>0</td><td>0.00</td><td>50</td><td>100,890</td><td>127,790</td><td>0.24</td></tr><tr><td> $2 ^ { * }$ </td><td>0</td><td>0</td><td>0.00</td><td>51</td><td>101,000</td><td>117,006</td><td>0.15</td></tr><tr><td>3*</td><td>357</td><td>0</td><td>5.88</td><td>52</td><td>207,261</td><td>104,582</td><td>0.68</td></tr><tr><td>4*</td><td>0</td><td>0</td><td>0.00</td><td>53</td><td>55,318</td><td>71,696</td><td>0.26</td></tr><tr><td> $5 ^ { * }$ </td><td>0</td><td>0</td><td>0.00</td><td>54</td><td>38,100</td><td>43,835</td><td>0.17</td></tr><tr><td> $6 ^ { * }$ </td><td>0</td><td>0</td><td>0.00</td><td>55</td><td>104,181</td><td>88,542</td><td>0.16</td></tr><tr><td>7*</td><td>0</td><td>0</td><td>0.00</td><td>56*</td><td>51,868</td><td>610</td><td>9.25</td></tr><tr><td> $8 ^ { * }$ </td><td>0</td><td>0</td><td>0.00</td><td>57</td><td>1,147</td><td>1,612</td><td>3.33</td></tr><tr><td> $9 ^ { * }$ </td><td>0</td><td>0</td><td>0.00</td><td>58</td><td>90,845</td><td>25,054</td><td>1.29</td></tr><tr><td> $1 0 ^ { * }$ </td><td>0</td><td>0</td><td>0.00</td><td>59</td><td>45,673</td><td>33,407</td><td>0.32</td></tr><tr><td> $\mathrm { N o . }$ </td><td>q</td><td>â</td><td> $e _ { \mathrm { l o g } }$ </td><td> $\mathrm { N o . }$ </td><td>q</td><td>â</td><td> $e _ { \mathrm { l o g } }$ </td></tr><tr><td>11*</td><td>226</td><td>0</td><td>5.42</td><td>60</td><td>50,021</td><td>64,823</td><td>0.26</td></tr><tr><td>12*</td><td>0</td><td>0</td><td>0.00</td><td>61</td><td>163,182</td><td>157,476</td><td>0.04</td></tr><tr><td>13*</td><td>4,553</td><td>0</td><td>8.42</td><td>62</td><td>171,271</td><td>157,035</td><td>0.09</td></tr><tr><td>14*</td><td>0</td><td>0</td><td>0.00</td><td>63</td><td>60,955</td><td>34,698</td><td>0.56</td></tr><tr><td>15*</td><td>0</td><td>0</td><td>0.00</td><td>64</td><td>7,770</td><td>12,987</td><td>0.51</td></tr><tr><td>16</td><td>6,338</td><td>6,922</td><td>0.09</td><td>65</td><td>80,348</td><td>90,515</td><td>0.14</td></tr><tr><td>17</td><td>873</td><td>2,066</td><td>0.86</td><td>66</td><td>50,709</td><td>15,857</td><td>1.16</td></tr><tr><td>18</td><td>76,141</td><td>19,113</td><td>1.38</td><td>67</td><td>84,493</td><td>114,560</td><td>0.30</td></tr><tr><td>19*</td><td>42,083</td><td>0</td><td>10.65</td><td>68</td><td>67,531</td><td>129,292</td><td>0.65</td></tr><tr><td>20*</td><td>0</td><td>0</td><td>0.00</td><td>69</td><td>25,579</td><td>93,364</td><td>1.29</td></tr><tr><td>21*</td><td>0</td><td>0</td><td>0.00</td><td>70</td><td>107,033</td><td>93,728</td><td>0.13</td></tr><tr><td>22*</td><td>32,240</td><td>0</td><td>10.38</td><td>71</td><td>95,814</td><td>116,891</td><td>0.20</td></tr><tr><td>23*</td><td>3,295</td><td>0</td><td>8.10</td><td>72</td><td>76,472</td><td>88,473</td><td>0.15</td></tr><tr><td>24*</td><td>200</td><td>0</td><td>5.30</td><td>73</td><td>247,107</td><td>106,067</td><td>0.85</td></tr><tr><td>25</td><td>0</td><td>0</td><td>0.00</td><td>74</td><td>81,273</td><td>122,612</td><td>0.41</td></tr><tr><td>26</td><td>114,138</td><td>111,644</td><td>0.02</td><td>75</td><td>43,612</td><td>49,088</td><td>0.12</td></tr><tr><td>27</td><td>48,700</td><td>3,761</td><td>2.57</td><td>76*</td><td>0</td><td>0</td><td>0.00</td></tr><tr><td>28</td><td>28,530</td><td>15,484</td><td>0.61</td><td>77</td><td>31,696</td><td>40,010</td><td>0.23</td></tr><tr><td>29</td><td>51,808</td><td>20,383</td><td>0.93</td><td>78</td><td>34,391</td><td>13,856</td><td>0.94</td></tr><tr><td>30*</td><td>0</td><td>0</td><td>0.00</td><td>79</td><td>119,441</td><td>95,194</td><td>0.23</td></tr><tr><td>32</td><td>0</td><td>0</td><td>0.00</td><td>80</td><td>62,455</td><td>71,463</td><td>0.13</td></tr><tr><td>33</td><td>0</td><td>0</td><td>0.00</td><td>81</td><td>14,174</td><td>78,046</td><td>1.70</td></tr><tr><td>34</td><td>1,379</td><td>11,958</td><td>2.16</td><td>82</td><td>114,383</td><td>112,761</td><td>0.01</td></tr><tr><td>35*</td><td>0</td><td>0</td><td>0.00</td><td>83</td><td>58,386</td><td>112,287</td><td>0.65</td></tr><tr><td>36</td><td>82,233</td><td>44,972</td><td>0.64</td><td>84</td><td>130,578</td><td>108,447</td><td>0.19</td></tr><tr><td>37*</td><td>2,244</td><td>0</td><td>7.72</td><td>85</td><td>99,186</td><td>116,373</td><td>0.16</td></tr><tr><td>38</td><td>452</td><td>12,821</td><td>3.34</td><td>86</td><td>128,632</td><td>80,176</td><td>0.47</td></tr><tr><td>39*</td><td>0</td><td>0</td><td>0.00</td><td>87*</td><td>2,639</td><td>0</td><td>7.88</td></tr><tr><td>40*</td><td>0</td><td>0</td><td>0.00</td><td>88</td><td>75,311</td><td>55,192</td><td>0.32</td></tr><tr><td>41*</td><td>0</td><td>0</td><td>0.00</td><td>89</td><td>60,312</td><td>11,694</td><td>1.64</td></tr><tr><td>42*</td><td>0</td><td>0</td><td>0.00</td><td>90</td><td>61,305</td><td>48,592</td><td>0.23</td></tr><tr><td>43*</td><td>0</td><td>0</td><td>0.00</td><td>91</td><td>110,908</td><td>64,697</td><td>0.54</td></tr><tr><td>44*</td><td>0</td><td>0</td><td>0.00</td><td>92</td><td>54,574</td><td>32,235</td><td>0.55</td></tr><tr><td>45*</td><td>0</td><td>0</td><td>0.00</td><td>93</td><td>38,893</td><td>45,767</td><td>0.16</td></tr><tr><td>46</td><td>155,410</td><td>45,962</td><td>1.23</td><td>94</td><td>94,093</td><td>81,203</td><td>0.15</td></tr><tr><td>47</td><td>42,637</td><td>16,230</td><td>1.00</td><td>95</td><td>57,726</td><td>14,716</td><td>1.38</td></tr><tr><td>48*</td><td>0</td><td>0</td><td>0.00</td><td>96</td><td>34,285</td><td>12,227</td><td>1.07</td></tr><tr><td>49</td><td>108,381</td><td>39,681</td><td>1.01</td><td></td><td></td><td></td><td></td></tr></table>

Data support across the landscape. Table S8 applies the same two quantities to the whole landscape. The productive region is well supported: every top-1% condition lies within 7 steps of an anchor (74% within 3), all five forests predict a product for every one of them, and none is flagged. The no-product region is different. It makes up 94% of the landscape, and 93% of it is flagged: these conditions are predicted as no product by most forests, but no measurement lies within three steps. The zeros of the landscape are therefore mostly extrapolations, not observations. They are consistent with the 28 measured failures, which are spread over the whole range of every variable (Figure S4) and were all predicted correctly when left out, but the exact extent of the no-product region depends on the emulator. For this reason the comparison of the strategies was repeated on 41 landscapes rebuilt with other choices (Section S4.4).

Prediction of later experiments. A map built from only the first 60 anchors in recording order (Table S4) predicted the 35 later ones, which it had not seen. It ranked them better than the q of the nearest earlier anchor did (Spearman ρ = 0.61 vs. 0.43) and again made only false zeros (7 of 34 crystalline anchors), never a false product. With more anchors the ranking improved steadily, from ρ = 0.48 at 40 anchors to 0.83 at 80 (Table S9). In absolute magnitude the nearest earlier anchor was closer (median $e _ { \mathrm { l o g } }$ 0.37 vs. 0.78), because the campaign stayed near earlier experiments; the benchmark relies on the ranking, not on the magnitude.

Table S8: Support of the emulated landscape by the anchors, by region. $d _ { \operatorname* { m i n } } ,$ grid steps to the nearest anchor; Within 3 steps, share of the conditions with $d _ { \operatorname* { m i n } } \le 3 \mathrm { : }$ Forests agree, share for which all five forests give the same answer (product or no product); Flagged, Eq. S12. Top x%: the x% of emulated conditions with the highest q (Table S10).
<table><tr><td>Region of the landscape</td><td>Conditions</td><td>Median  $d _ { \mathrm { m i n } }$ </td><td>Within 3 steps</td><td>Forests agree</td><td>Flagged</td></tr><tr><td>Top 0.1%</td><td>6,104</td><td>2</td><td>92%</td><td>100%</td><td>0%</td></tr><tr><td>Top 1%</td><td>61,013</td><td>3</td><td>74%</td><td>100%</td><td>0%</td></tr><tr><td>Top 5%</td><td>305,056</td><td>4</td><td>36%</td><td>99.8%</td><td>0%</td></tr><tr><td>All with product  $( q > 0 )$ </td><td>384,112</td><td>4</td><td>36%</td><td>96%</td><td>0.6%</td></tr><tr><td>No product  $( q = 0 )$ </td><td>5,716,965</td><td>6</td><td>6%</td><td>93%</td><td>93%</td></tr><tr><td>Full landscape</td><td>6,101,077</td><td>6</td><td>8%</td><td>93%</td><td>87%</td></tr></table>

Table S9: Prediction of later anchors from the first N anchors (recording order). Emulator: the landscape rebuilt from the first N anchors only, read at the condition of each later anchor. Nearest anchor: a simple comparison that assigns each later anchor the measured q of the closest of the first N anchors (Chebyshev distance; mean q if several are equally close). $\rho ,$ Spearman correlation between predicted and measured q (1, same order). False zeros: crystalline later anchors predicted as $q = 0$ by the emulator.
<table><tr><td> $N$ </td><td>Later anchors (crystalline)</td><td>emulator  $\rho ,$ </td><td> $\rho ,$  nearest anchor</td><td>False zeros</td></tr><tr><td>40</td><td>55 (49)</td><td>0.48</td><td>0.56</td><td>23/49</td></tr><tr><td>50</td><td>45 (44)</td><td>0.52</td><td>0.37</td><td>20/44</td></tr><tr><td>60</td><td>35 (34)</td><td>0.61</td><td>0.43</td><td>7/34</td></tr><tr><td>70</td><td>25 (24)</td><td>0.75</td><td>0.38</td><td>3/24</td></tr><tr><td>80</td><td>15 (15)</td><td>0.83</td><td>0.60</td><td>1/15</td></tr></table>

## S2.6. Structure

Sparsity. Only 6.3% of the landscape yields a crystalline product, and high q is rare even among these conditions: 1% of all conditions reach $q \geq 4 2 , 4 6 3$ and 0.1% reach $q \ge 1 0 0 , 2 8 8$ (Table S10). The anchors look far less sparse (29.5% zero, 46% above the top-1% threshold) because they come from an optimization campaign concentrated near the productive region (Figure S4).

Table S10: Quality tiers of the landscape and of the 95 anchors. Top x%: the x% of the 6,101,077 emulated conditions (anchor positions excluded) with the highest $q ;$ the threshold is the lowest $q$ in the tier. The same thresholds are used throughout the SI and the main text.
<table><tr><td>Tier</td><td>q threshold</td><td>Landscape</td><td>Fraction</td><td>Anchors</td></tr><tr><td>Top 0.01%</td><td>≥ 144,556</td><td>611</td><td>0.01%</td><td>5</td></tr><tr><td>Top 0.1%</td><td>≥ 100,288</td><td>6,104</td><td>0.10%</td><td>16</td></tr><tr><td>Top 1%</td><td>≥ 42,463</td><td>61,013</td><td>1.0%</td><td>44</td></tr><tr><td>Top 5%</td><td>≥ 4,167</td><td>305,056</td><td>5.0%</td><td>57</td></tr><tr><td>Non-zero</td><td>&gt; 0</td><td>384,112</td><td>6.3%</td><td>67</td></tr><tr><td>Zero</td><td>= 0</td><td>5,716,965</td><td>93.7%</td><td>28</td></tr></table>

Two-dimensional projections. Figure S7 shows, for all ten pairs of the five variables, the best q reachable when the other three variables are chosen freely. The crystalline region is one block: about 1-3 mL $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ , less than about 1 mL NaOH and at least $9 0 ~ ^ { \circ } \mathrm { C } ,$ at any amount of added DI water and hold time. The top-0.1% tier is a smaller block inside it, at 1.1-1.9 mL $\mathrm { A l C l _ { 3 } \cdot 6 H _ { 2 } O , 0 . 5 - 0 . 7 }$ mL NaOH, hold times of 40 min or more and $1 0 0 { - } 1 4 5 \ ^ { \circ } \mathrm { C } .$ again over most of the range of added DI water.

![](images/5a958bffc9d9c60cad8f3db8d6295bbf20e69f9ac0c551f231d220139c231276.jpg)  
Figure S7: The landscape on all ten two-variable projections: in each panel, the highest q over the other three variables, with the outlines of the top-1% (dashed) and top-0.1% (solid) tiers, drawn along the edges of the grid cells; gray, no product for any combination of the other three variables. Filled and open circles: anchors with $q > 0$ and $q = 0$

Chemical trends. $\mathrm { H i g h } { - } q$ conditions combine low $\mathrm { { A l C l _ { 3 } } \cdot 6 \mathrm { { H _ { 2 } O } \ ( 1 . 1 - 1 . 8 } }$ mL for all 16 top-0.1% anchors), low NaOH (0.5–0.7 mL), long hold times (55–60 min) and $1 1 5 { - } 1 4 5 ~ ^ { \circ } \mathrm { C }$ with added DI water mattering least (Figure S7). The base window is narrow: too little base leaves the linker protonated, while excess base is clearly detrimental, with 13 of 16 anchors at 1.0 mL NaOH or more giving no product and a single step from 0.7 to 0.8 mL (entries 18 and 34) lowering q from 76,141 to 1,379. All anchors at or below $7 5 ~ ^ { \circ } \mathrm { C }$ fail.

The top-0.1% tier is essentially one connected region (6,100 of its 6,104 grid points). Its surroundings are steep: of all pairs of neighboring conditions (one grid step apart in one variable) that include a top-1% condition, 29% belong to two different quality tiers, so a single grid step often changes the tier.

Settings of the embedding and density plots. The two-dimensional embedding of the landscape shown in the main text was computed with UMAP (n\_neighbors = 10, min\_dist = 0.2, Euclidean metric, random\_state = 42) on standardized variables, for 1% of the landscape sampled separately from $q = 0$ and $q > 0$ (61,012 conditions) plus the 95 anchors. The noproduct conditions are split at the medians of the embedded set $( 1 0 0 \ ^ { \circ } \mathrm { C } ; 4 . 0$ mL $\mathrm { A l C l _ { 3 } \cdot 6 H _ { 2 } O ) }$ The density curves are Gaussian kernel density estimates (Scott's rule) on a 2% random sample of each tier.

## S2.7. Validation Optima

Six further Al MOFs were optimized in the wet lab, each in 12 experiments that started from the MOF-321 data; the transfer-learning method and the campaigns are described in Section S5, the syntheses in Sections S6 and S7. Here we ask only where their best conditions lie on the MOF-321 landscape. We compare positions, not values: $q$ is comparable only within one phase (Section S2.1), so the $q$ that the landscape predicts for MOF-321 at a condition is not compared with the q measured there for another framework. For each validation MOF, the best-found condition $\mathbf { x } ^ { * }$ was located on the MOF-321 landscape by its Chebyshev distance to the nearest anchor $( d _ { 9 5 } )$ and to the top-1% and top-0.1% tiers (Figure S8, Table S11), measured to the nearest condition of the tier and to its centroid, the mean of the integer codes of all conditions in the tier (a point that need not lie on the grid). All six fall on grid points inside MOF-321's own top-1% tier and within 1-2 steps of its top-0.1% tier, whereas a condition drawn at random from the grid lies a median of 16 and 20 steps from the nearest condition of these tiers (20,000 random draws). The three shared starting conditions already sit in the top 1% (Table S11, bottom), so the transferred prior places each search in MOF-321's productive region from the start; the searches stayed there while re-learning q for each framework, and MTV-MOF-303 ended closer to the top-0.1% centroid than any starting point.

Table S11: Distance of each validation MOF's best condition, and of the three shared starting conditions, to MOF-321 anchors and top tiers (Chebyshev units).
<table><tr><td></td><td></td><td></td><td></td><td colspan="2">Top 1%</td><td colspan="2">Top 0.1%</td></tr><tr><td>MOF</td><td>Best ID</td><td>Best q</td><td> $d _ { 9 5 }$ </td><td>nearest</td><td>centroid</td><td>nearest</td><td>centroid</td></tr><tr><td>CAU-10</td><td>R00012</td><td>32,259</td><td>3</td><td>0</td><td>4.1</td><td>2</td><td>4.9</td></tr><tr><td>CAU-23</td><td>R00001</td><td>22,254</td><td>3</td><td>0</td><td>2.9</td><td>1</td><td>4.2</td></tr><tr><td>MIL-53-muc</td><td>R00008</td><td>58,064</td><td>2</td><td>0</td><td>2.9</td><td>1</td><td>4.2</td></tr><tr><td>MIL-160</td><td>R00007</td><td>44,395</td><td>2</td><td>0</td><td>3.9</td><td>1</td><td>5.2</td></tr><tr><td>MTV-MIL-160</td><td>R00007</td><td>30,990</td><td>3</td><td>0</td><td>4.1</td><td>2</td><td>4.9</td></tr><tr><td>MTV-MOF-303</td><td>R00009</td><td>53,648</td><td>2</td><td>0</td><td>2.2</td><td>1</td><td>3.9</td></tr><tr><td colspan="8">Shared starting conditions (landscape percentile)</td></tr><tr><td>R00001 (18,7,17,10,21)</td><td></td><td>99.78%</td><td>3</td><td>0</td><td>2.9</td><td>1</td><td>4.2</td></tr><tr><td>R00002 (23,6,15,10,20)</td><td></td><td>99.30%</td><td>2</td><td>0</td><td>6.2</td><td>4</td><td>7.9</td></tr><tr><td>R00003 (20,7,19,12,29)</td><td></td><td>99.44%</td><td>3</td><td>0</td><td>5.1</td><td>2</td><td>4.9</td></tr></table>

![](images/6e4eed384eba3e5b933c4bc39d446dbd529fd4361b92616d3ed0a986acffb40f.jpg)  
Figure S8: Best-found conditions of the six validation MOFs (open symbols) and the three shared starting conditions (crosses) on all ten two-variable projections of the MOF-321 landscape (highest q over the other three variables, with the outlines of the top-1% (dashed) and top-0.1% (solid) tiers), zoomed to the productive part of the grid. Symbols in the same grid cell are spread horizontally; the best CAU-23 condition is starting condition R00001.

## Section S3. Search Strategies

This section describes how the search policies explore the ActMOF landscape: the shared protocol (Section S3.1), Gaussian processes (Section S3.2) and the acquisition rules (Section S3.3), the LLM and hybrid policies (Sections S3.4–S3.5) and a list of all 39 strategies that were run, with a short description of the other model types (Section S3.6). The code is in the GitHub repository (benchmark/families/). The quantities used to compare strategies (quality tiers, iterations, hit rates) are defined in Table S14 (Section S4.1).

## S3.1. Protocol

Search loop. Each run starts from three experiments and then proposes three experiments per iteration. The landscape (Section S2.6) returns the q of each proposed condition, which is added to the history before the next iteration. A run has a budget of 120 experiments $( 3 + 3 9 \times 3 $ iterations 0–39). Every strategy was run five times (n = 5); all 39 strategies are shown in Figure S14. Table S12 lists the settings.

Starting experiments. The three starting experiments are drawn uniformly at random from the grid of Eq. S7 (function run\_single\_method, stage initial\_random). The random generator of run r of method m is seeded with

$$
s _ { m , r } = 4 2 + 1 0 , 0 0 0 r + h ( m ) ,\tag{S13}
$$

where $h ( m )$ is a hash of the method name: a fixed integer computed from the name by a hash function (stable\_int in the code). Because h(m) differs between methods, the starting experiments are not shared between methods, even within the same run index; this holds for all methods, including the LLM and hybrid policies. Each run of each strategy therefore starts from its own random conditions; 35 of the 195 starting batches (39 strategies, five runs each) already contain a crystalline product (Figure S9).

Independent and shared starting batches. Independent starts are the usual protocol of optimization benchmarks. Each strategy is averaged over its own set of starting situations, so no strategy is favored or penalized by one particular set of starts, and the spread over runs includes the effect of the start, as in a real campaign whose first experiments are not known in advance. The price is that, with few runs, two strategies can differ only because one of them drew luckier starts. Where this matters, in the early iterations and in the comparison of PI and EI, the runs were repeated with the same three starting conditions for every strategy (shared starts; Sections S4.5, S4.6 and S4.7), which removes this source of difference.

Before the first product. Because 94% of the landscape gives no product, a run usually begins with several experiments that all return q = 0. As long as all results of a run are the same, a model has nothing to learn from: it cannot tell one condition from another. In this phase every candidate receives the same score, and the three conditions of a batch are in effect drawn at random; only the rule of the next paragraph, which keeps them apart, still acts. A model is fitted from the first iteration in which two different results exist.

This phase matters for two reasons. First, until the first product appears, every model-based strategy behaves like random search, whatever its model and acquisition rule, so that differences between strategies in the first iterations mostly reflect luck (Section S4.5). Second, the phase can use up a large part of a short campaign: in the shared-start runs of Matérn-5/2 EI, half of the 20 runs needed more than 12 experiments to find their first product (Section S4.7), and in iterations 1–5 of the five Matérn-5/2 runs, 57 of the 75 EI queries and 24 of the 75 PI queries were chosen in this way. This is why the wet-lab campaigns, which have only 12 experiments start from a prior and not from random conditions (Section S4.6).

Candidate sampling and batch selection. Scoring all 6.1 million conditions in every iteration would be slow, so each iteration scores a random subset: 15,000 conditions drawn uniformly from the grid, excluding those already tested in the run. The two local approximate GPs (laGP) score 4,000 candidates to reduce the computing cost. Random search draws its three conditions directly; the LLM policies write their conditions without a candidate list (Section S3.4); the hybrid policies use the candidate list only in their GP iterations. Each candidate x receives a score a(x) from the acquisition function of the strategy (Section S3.3), which measures how useful it would be to test this condition next. The three best-scored candidates are often almost the same condition, because neighboring conditions get similar scores, and testing all three would waste two experiments. The batch is therefore built one condition at a time: the first is the best-scored candidate, and each further one maximizes the score plus a bonus for its distance from the conditions already chosen,

![](images/a5cdbee65f624ae3f7183d1a57ac81cf99db314027e79b754da546836499b6b3.jpg)  
Figure S9: Starting batches of all 39 strategies. Each cell is one run; runs 0-4; color gives the landscape percentile of the best of the three starting experiments; gray, no crystalline product in the starting batch.

$$
\widetilde { a } ( \mathbf { x } ) + \lambda \operatorname* { m i n } _ { \mathbf { x } ^ { \prime } \in B } \| \mathbf { x } - \mathbf { x } ^ { \prime } \| _ { 2 } , \qquad \lambda = 0 . 0 3 .\tag{S14}
$$

Here $\tilde { a }$ is the standardized score,

$$
\tilde { a } ( { \bf x } ) = \frac { a ( { \bf x } ) - \bar { a } } { s _ { a } } ,\tag{S15}
$$

where ā and $s _ { a }$ are the mean and the standard deviation of a over the candidates of the iteration; the standardization gives λ the same meaning for every acquisition function. B is the set of conditions already in the batch, and distances are measured with every variable scaled to [0, 1] over its range. With the small value $\lambda = 0 . 0 3$ , the bonus decides only between candidates of nearly equal score. One of the 39 strategies, "diversity EI" (Table S13), uses $\lambda = 0 . 1 0$ to test the effect of a stronger bonus. Before the batch is built, a random number of order $1 0 ^ { - 1 2 }$ is added to every score. It is far too small to change the order of different scores, but it puts candidates with exactly equal scores (for example, all candidates before the first product) in a random order, so that they are not chosen by their position in the list.

Table S12: Benchmarking protocol.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Starting experiments</td><td>3, uniform random on the grid; random seed Eq. S13, differs between methods</td></tr><tr><td>Batch size per iteration</td><td>3</td></tr><tr><td>Total budget</td><td>120 experiments  $( 3 + 3 9 \times 3 )$  , iterations 0–39  $n = 5$  for all 39 strategies; the leading group was extended to</td></tr><tr><td>Replicate runs</td><td> $n = 2 0$  (Section S4.3)</td></tr><tr><td>Model target</td><td> $y = \log ( 1 + q ) ;$  incumbent  $y ^ { + } = \operatorname* { m a x } y$  observed</td></tr><tr><td>Input scaling</td><td>each variable scaled to [0, 1] over its grid range</td></tr><tr><td>Candidates per iteration</td><td> $1 5 { , } 0 0 0$  untried grid points  $( \mathrm { l a G P : 4 , 0 0 0 } )$ </td></tr><tr><td>Batch construction</td><td>one condition at a time: standardized score plus distance bonus,  $\mathrm { E q . ~ S 1 4 , } ~ \lambda = 0 . 0 3$ </td></tr><tr><td>Exploration settings</td><td>EI, PI: smallest improvement that counts,  $\xi = 0 . 0 1$  on the log(1 + q) scale (about 1% in q); UCB: weight of the uncertainty,  $\beta = 2 ;$  quantile and CVaR: 90% level,  $\tau = 0 . 9$  (Section S3.3)</td></tr><tr><td>No product yet</td><td>no model until two different results exist; all candidates scored equally</td></tr><tr><td>Oracle</td><td>noise-free landscape q (Section S2.6); noisy variant in Section S4.5</td></tr></table>

## S3.2. Gaussian Processes

A Gaussian process (GP) is a flexible regression model that gives, for every untested condition, a predicted value $\mu ( \mathbf { x } )$ and an uncertainty σ(x) of that prediction. It assumes only that conditions close to each other give similar results, and it learns from the data how close is close, separately for every variable. Near tested conditions σ is small; far from them it is large. The acquisition functions (Section S3.3) use µ and σ to weigh refining a good region against exploring an unknown one.

The GP policies use scikit-learn's[5] GaussianProcessRegressor. The similarity of two conditions is given by a Matérn kernel $k _ { \nu } ( r )$ of their scaled distance

$$
r ^ { 2 } = \sum _ { j = 1 } ^ { 5 } \frac { ( x _ { j } - x _ { j } ^ { \prime } ) ^ { 2 } } { \ell _ { j } ^ { 2 } } ,\tag{S16}
$$

plus a noise term that lets the GP fit data that are not exactly smooth. The kernel parameter ν sets how smooth the GP assumes the response to be: $\nu = 5 / 2$ gives a smoother response, $\nu = 3 / 2$ allows sharper changes, as at the edge of a narrow crystallization window (RBF and rational-quadratic kernels were also run; Table S13). The length scale $\ell _ { j }$ is the change of variable j (as a fraction of its range) over which q is expected to change appreciably: a short length scale means that q is sensitive to that variable, a long one that it hardly matters. All parameters are refitted to the data in every iteration (fitting bounds and optimizer settings in the repository). The GP is fitted to $y = \log ( 1 + q )$ : the logarithm compresses the range of q (0 to $2 . 5 \times 1 0 ^ { 5 } )$ , sO that one very high value does not dominate the fit, and log(1+q) keeps $q = 0$ finite. RobustGaSP (Matérn-5/2 with estimated noise) and the local approximate GPs of the laGP package[6], which fit a small GP to the 10–60 tested conditions nearest to each candidate, were run through R.

Figure S10a illustrates the length scale with the same six experiments fitted twice. With a short length scale, the GP expects the response to change over short distances: it follows every point closely and is very uncertain between the points. With a long length scale, it draws one smooth curve through the points and is confident between them. Because one length scale is fitted for each of the five variables, the fitted values show which variables q is sensitive to (Figure S10b). After 120 experiments, $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ has the shortest length scale (median 0.09 of its range for Matérn-5/2), followed by temperature (0.40) and NaOH (0.48), whereas added

DI water and hold time have length scales longer than their whole range; for added DI water the length scale reaches the upper bound of the fit in four of five Matérn-5/2 runs. This agrees with the landscape (Figure S7): the metal and base amounts and the temperature decide whether a product forms; DI water hardly matters.

a  
![](images/5d6459b0d9a8400df68aa1ff6420d3272c533664470f5847282164be66876182.jpg)

b  
![](images/233f550736e99fe2ecb7a60294ea210955fc474895ef482ada4ad9486de1181b.jpg)

![](images/2b3e920e927a652fe1ebef42c35fb9d59133f2871dac102e23881a1c14fde15a.jpg)  
Figure S10: Length scales. (a) Illustration with one variable: the same six experiments fitted by a GP with a short and with a long length scale $\ell ;$ line, prediction $\mu ;$ band, $\mu \pm 2 \sigma$ . (b) Length scales learned by the Matérn- ${ \it - 5 / 2 }$ and Matérn- $- 3 / 2$ GPs with EI from the 120 experiments of each run $( n = 5 )$ , as a fraction of the range of each variable; one dot per run; line, median; dotted line, upper bound of the fit.

## S3.3. Acquisition Functions

With GP mean $\mu ( \mathbf { x } )$ and standard deviation $\sigma ( \mathbf { x } )$ on the log $( 1 + q )$ scale,

$$
\operatorname { P I } ( \mathbf { x } ) = \Phi ( z ) , \qquad z = { \frac { \mu ( \mathbf { x } ) - y ^ { + } - \xi } { \sigma ( \mathbf { x } ) } } ,\tag{S17}
$$

$$
\operatorname { E I } ( \mathbf { x } ) = \left( \mu ( \mathbf { x } ) - y ^ { + } - \xi \right) \Phi ( z ) + \sigma ( \mathbf { x } ) \phi ( z ) ,\tag{S18}
$$

as in efficient global optimization[7], where $y ^ { + }$ is the largest log $( 1 + q )$ observed so far in the run, $\xi = 0 . 0 1$ , and Φ and $\phi$ are the cumulative distribution function and the density of the standard normal distribution. PI is the probability that a candidate beats the best result so far by at least $\xi ;$ EI is the expected size of that improvement, counting a result below the best as zero. The other rules in Table S13 score candidates by an optimistic estimate of their result: UCB by $\mu + \beta \sigma$ with $\beta = 2 ;$ the upper quantile by the value that the result exceeds with 10% probability, $\mu + \Phi ^ { - 1 } ( \tau ) \sigma$ with $\tau = 0 . 9$ and the upper conditional value at risk (CVaR) by the mean of the best 10% of possible results, $\mu + \sigma \phi ( \Phi ^ { - 1 } ( \tau ) ) / ( 1 - \tau )$ . Thompson sampling draws, for every candidate, one random plausible result from its predicted distribution ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ and picks the highest draws, so that uncertain candidates are picked from time to time. Greedy $\mu$ picks the highest predictions and ignores the uncertainty.

What PI and EI reward. When $\mu$ exceeds $y ^ { + }$ , PI is highest for small $\sigma _ { \mathrm { { ; } } }$ so it rewards any improvement, however small; EI always increases with $\sigma$ at fixed $\mu ,$ because it also rewards the size of the possible gain. Section S4.5 compares the two rules.

GP-EI proposals. Figure S11 shows the 585 conditions proposed by Matérn-5/2 EI in runs 0–4. The early proposals (light) are spread over the whole grid; the later ones (dark) concentrate on the productive region. Overall, 59% of the proposals give a product, 39% reach the top-1% tier and 16% the top-0.1% tier

![](images/72dfc48b429eab6b32245c96d1d816aa75d5a93fbb658baa8ac9687484a70f71.jpg)  
Figure S11: Conditions proposed by Matérn-5/2 EI, runs 0-4, iterations 1-39, on all ten twovariable projections of the landscape (highest q over the other three variables; gray, no product). Color gives the iteration in which a condition was proposed, light (iteration 1) to dark (iteration 39); later iterations are drawn on top. Points are jittered within their grid cell.

## S3.4. LLM Policies

Models and settings. Three OpenAI models were used through the Chat Completions API with structured output: gpt-4o, gpt-5.5 and gpt-5-mini. The snapshots were those of August 2026: gpt-4o-2024-08-06, gpt-5.5-2026-04-23 and gpt-5-mini-2025-08-07. gpt-4o was run at temperature 0.2. The GPT-5 models do not accept a temperature and were run with the lowest reasoning effort (none for gpt-5.5, minimal for gpt-5-mini). A fixed seed was sent with each call: the seed of the run (Eq. S13) plus the number of the iteration. The iteration number is added because the seed of the run alone would give all 39 calls of a run the same seed; with it, every call has its own fixed seed and can be repeated on its own. Each model was run for 5

runs of 39 iterations.

Prompt. Every request has two messages. The system message is the same in every call and gives the chemistry, the objective and the integer encoding of the five variables (Box S1). The user message is a JSON object built from the history of the run (Box S2): the objective, the allowed ranges, the last 45 experiments of the run (conditions and q, oldest first) and the instructions to propose exactly three new experiments, “one exploitation condition, one local perturbation, and one exploration condition". The same messages were used for all three models and for the LLM iterations of the hybrids. Boxes S1 and S2 show the two messages in full; lines in green are our comments and are not part of the messages. They describe the set-up of the original MOF-321 syntheses (1 mmol linker, 35 mL vessel), not the validation syntheses. The model is given no list of candidates: it writes the integer codes of three new conditions itself guided by the previous experiments and the allowed ranges in the user message.

Box S1.System message.   
You are a metal-organic framework synthetic chemist who choose [sic] the next microwave synthesis   
experiments for aluminum dicarboxylate MOFs.   
The task:   
Choose exactly 3 new experiments from candidate\_pool.   
The goal is to maximize q\_final = Intensity\_final / FWHM\_final.   
Higher q\_final is better because it means stronger diffraction intensity and sharper peaks.   
Reaction background:   
Each experiment uses the same amount of organic linker: 1 mmol.   
The linker is a linear or V-shaped dicarboxylate linker.   
The metal precursor is AlCl3.6H20 stock solution (100 mg/mL) in DI water.   
The modulator/base is NaOH stock solution (100 mg/mL) in DI water.   
The reaction is run in a 35 mL Pyrex microwave pressure vessel.   
Very important:   
The five input variables are integer codes, not direct mmol/mL/minute/Celsius values.   
1. metal\_amount: integer 5 to 75 (encodes AlCl3.6H20 stock volume).   
2. modulator: integer 5 to 15 (encodes NaOH stock volume).   
3. add\_solvent: integer 0 to 30 (encodes extra DI water).   
4. reaction\_time: integer 1 to 12 (encodes microwave heating time).   
5. reaction\_temperature: integer 10 to 30 (encodes microwave heating temperature).   
Hard rules:   
- Do not invent conditions outside the stated integer ranges.   
- Do not repeat previously suggested conditions.   
- Output ONLY the five encoded integer variables (plus a short rationale) for each of the three new   
experiments - never decoded mL/mmol/min/Celsius values.

BoxS2. User message: the JSONobject sent withevery request. Angle brackets mark the values filled in from the run; previous\_experiments holds the 45most recent experiments of the run, repair\_context is empty except on a retry (One iteration of the loop).

```javascript
{
"job": "Suggest the next three MOF synthesis experiments for active learning.",
"objective": "Maximize q_final = Intensity_final / FWHM_final.",
"important_instruction": "Generate new synthesis parameter sets directly.",
"previous_experiments": [
// one object per experiment: the 45 most recent experiments of the run, oldest first
{
"metal_amount":"<metal_amount>",
"modulator": "<modulator>",
"add_solvent": "<add_solvent>",
"reaction_time":"<reaction_time>",
"reaction_temperature":"<reaction_temperature>",
"q_final": "<q_final>"
}
],
```

"allowed\_ranges": {   
"metal\_amount": "integer 5 to 75",   
"modulator": "integer 5 to 15",   
"add\_solvent": "integer 0 to 30",   
"reaction\_time": "integer 1 to 12",   
"reaction\_temperature": "integer 10 to 30"   
},   
"required\_behavior": [   
"Propose exactly three new experiments.",   
"Return only the five encoded integer variables for each experiment.",   
"Do not repeat previous experiments.",   
"Prefer high expected q\_final, but include diversity to learn the landscape.",   
"Use one exploitation condition, one local perturbation, and one exploration condition."   
1,   
"repair\_context": "" // empty, except on a retry (Box S4)   
}

Structure of one iteration. (i) Input. The model receives nothing but the two messages: no landscape, no candidate list and no GP output. The API keeps no state between calls; every call is a new request, so everything the model knows about the run reaches it through previous\_experiments. Each entry holds the five integer codes of an experiment and its q as a plain number (0 for no product), the same value every other strategy observes. The first request contains the three starting experiments; from iteration 16 on, the history exceeds 45 experiments and the oldest drop out of the message. This list is the model's only memory of the run: Box S3 shows a complete request and the answer to it. (ii) Proposal. The response is constrained by structured output to a fixed schema: a short strategy summary and exactly three conditions, each with the five integer codes and a rationale. The conditions are then checked by our program, not by the model. The program keeps the complete list of the conditions tested in the run, so it can also detect a repeat of an experiment that is no longer among the 45 sent to the model. A batch is rejected if a condition is out of range, repeats any earlier experiment of the run or is duplicated within the batch; the model is then called again with the same messages and the error in repair\_context (“Your previous answer was invalid: .. ."; Box S4). After three failed attempts, three random untried conditions are used. (iii) Feedback. The three accepted conditions are looked up on the landscape, as for every other strategy, and appended with their q to the history of the run; the next request is built from the updated history. Only conditions and q are sent back. The model's earlier rationales and strategy summaries are logged but not returned to it, and rejected answers do not enter the history. Random fallback conditions enter it like any other experiment. In the hybrids, GP and LLM iterations append to the same history, so the LLM sees the GP's experiments in the same form as its own.

BoxS3. Example request and response:gpt-5.5, run 0, iteration 5. The user message contains the15experimentsof the runso far (3starting experiments andthe12of iterations1–4); the response gives the three conditions with the model's rationales.

// REQUEST: model gpt-5.5, reasoning\_effort none, seed 1055994896, max\_completion\_tokens 1500   
// system message: see Box S1   
// user message:   
{   
"job": "Suggest the next three MOF synthesis experiments for active learning.",   
"objective": "Maximize q\_final = Intensity\_final / FWHM\_final.",   
"important\_instruction": "Generate new synthesis parameter sets directly.",   
"previous\_experiments": [   
{"metal\_amount": 30, "modulator": 7, "add\_solvent": 2, "reaction\_time": 3, "reaction\_temperature":   
13, "q\_final": 0.0},   
{"metal\_amount": 7, "modulator": 8, "add\_solvent": 26, "reaction\_time": 10, "reaction\_temperature":   
18, "q\_final": 0.0},   
{"metal\_amount": 56, "modulator": 10, "add\_solvent": 12, "reaction\_time": 12,   
"reaction\_temperature": 17, "q\_final": 0.0},   
{"metal\_amount": 45, "modulator": 10, "add\_solvent": 8, "reaction\_time": 8, "reaction\_temperature":   
25, "q\_final": 0.0},

```jsonl
{"metal_amount": 58, "modulator": 11, "add_solvent": 10, "reaction_time": 11
"reaction_temperature": 20, "q_final": 0.0},
{"metal_amount": 18, "modulator": 14, "add_solvent": 24, "reaction_time": 6, "reaction_temperature":
28, "q_final": 0.0},
{"metal_amount": 65, "modulator": 12, "add_solvent": 6, "reaction_time": 12, "reaction_temperature":
26, "q_final": 0.0},
{"metal_amount": 55, "modulator": 13, "add_solvent": 5, "reaction_time": 10, "reaction_temperature":
27, "q_final": 0.0},
{"metal_amount": 72, "modulator": 6, "add_solvent": 1, "reaction_time": 9, "reaction_temperature":
30, "q_final": 0.0},
{"metal_amount": 40, "modulator": 9, "add_solvent": 10, "reaction_time": 10, "reaction_temperature":
24, "q_final": 0.0},
{"metal_amount": 60, "modulator": 9, "add_solvent": 4, "reaction_time": 10, "reaction_temperature":
29, "q_final": 0.0},
{"metal_amount": 22, "modulator": 6, "add_solvent": 18, "reaction_time": 12, "reaction_temperature":
22, "q_final": 62309.0},
{"metal_amount": 20, "modulator": 6, "add_solvent": 20, "reaction_time": 12, "reaction_temperature":
23, "q_final": 55320.0},
{"metal_amount": 24, "modulator": 5, "add_solvent": 17, "reaction_time": 11, "reaction_temperature":
21, "q_final": 51432.0},
{"metal_amount": 32, "modulator": 6, "add_solvent": 22, "reaction_time": 12, "reaction_temperature":
20, "q_final": 0.0}
],
"allowed_ranges": {
"metal_amount": "integer 5 to 75",
"modulator": "integer 5 to 15",
"add_solvent": "integer 0 to 30",
"reaction_time": "integer 1 to 12",
"reaction_temperature": "integer 10 to 30"
},
"required_behavior": [
"Propose exactly three new experiments.",
"Return only the five encoded integer variables for each experiment.",
"Do not repeat previous experiments.",
"Prefer high expected q_final, but include diversity to learn the landscape.",
"Use one exploitation condition, one local perturbation, and one exploration condition."
1，
"repair_context": ""
'/ RESPONSE of the model
"strategy_summary": "Focus near the only high-q region observed: relatively low metal_amount, low
modulator, high add_solvent, long reaction_time, and moderate reaction_temperature. The batch
includes one exploitation point near the best condition, one small local perturbation to map
sensitivity, and one broader exploration point that stays chemically plausible while testing the edge
of the productive basin.",
"next conditions": [
{
"metal_amount": 21,
"modulator": 6,
"add_solvent": 19,
"reaction_time": 12,
"reaction_temperature": 22,
"rationale": "Exploitation condition centered between the two strongest prior hits, preserving low
modulator and maximum reaction_time while targeting the apparent optimum in dilution and
temperature."
},
{
"metal_amount": 25,
"modulator": 6,
"add_solvent": 18,
"reaction_time": 12,
"reaction_temperature": 21,
"rationale": "Local perturbation around the best-performing region to test slightly higher
metal_amount and nearby solvent/temperature values without repeating prior successful conditions."
},
{
"metal_amount": 16,
"modulator": 5,
"add_solvent": 22,
"reaction_time": 11,
"reaction_temperature": 24,
```

"rationale": "Exploration condition probing lower metal\_amount, minimum modulator, and higher   
dilution while maintaining long heating and moderate temperature to test whether the productive   
crystallization window extends outward."   
}   
1   
}   
// result of the three conditions on the landscape: q = 63,108, 17,817, 108,359; best q of the run   
// after iteration 5: 108,359

## Box S4. Example of a retry: gpt-5.5, run 2, iteration 30, from the log of the run.

// gpt-5.5, run 2, iteration 30. The request contains the 45 most recent of the 90 experiments of   
// the run.   
// ATTEMPT 1: one of the three proposed conditions was rejected by the validation:   
{"metal\_amount": 18, "modulator": 5, "add\_solvent": 21, "reaction\_time": 12, "reaction\_temperature":   
26}   
// rationale given by the model:   
"Exploitation: combines the best metal\_amount observed near 18 with the best lower add\_solvent region   
and temperature 26; this exact condition has not been run and should test whether the 127k q\_final   
basin peaks at metal\_amount 18 rather than 17."   
// This condition had already been queried in iteration 29 (experiment 88 of the run, q =   
// 127,236); it   
// was inside the 45-experiment window.   
// ATTEMPT 2: the same two messages are sent again, with the user-message field   
"repair\_context": "Your previous answer was invalid: Repeats an already-tried condition:   
{'metal\_amount': 18, 'modulator': 5, 'add\_solvent': 21, 'reaction\_time': 12, 'reaction\_temperature':   
26, 'rationale': 'Exploitation: combines the best metal\_amount observed near 18 with the best lower   
add\_solvent region and temperature 26; this exact condition has not been run and should test whether   
the 127k q\_final basin peaks at metal\_amount 18 rather than 17.'}"   
// Response to attempt 2 (accepted):   
{   
"strategy\_summary": "Exploit the current best basin at low modulator, maximum time, temperature 26,   
and reduced add\_solvent, while avoiding all previously run tuples. The set includes one near-maximum   
exploitation point, one local perturbation around the best observed condition, and one broader   
exploration toward still lower dilution to test whether the q\_final optimum continues beyond   
add\_solvent 21.",   
"next\_conditions": [   
{   
"metal\_amount": 18,   
"modulator": 5,   
"add\_solvent": 20,   
"reaction\_time": 12,   
"reaction\_temperature": 26,   
"rationale": "Exploitation: keeps the strongest-performing metal/modulator/time/temperature region   
and steps add\_solvent just below the best low-dilution results without repeating prior   
experiments."   
},   
{   
"metal\_amount": 17,   
"modulator": 5,   
"add\_solvent": 20,   
"reaction\_time": 12,   
"reaction\_temperature": 26,   
"rationale": "Local perturbation: directly tests whether the current best metal\_amount 17 condition   
benefits from one further decrease in add\_solvent from 21 to 20."   
},   
{   
"metal\_amount": 18,   
"modulator": 5,   
"add\_solvent": 19,   
"reaction\_time": 12,   
"reaction\_temperature": 26,   
"rationale": "Exploration: probes a more concentrated regime beyond the observed high-q basin to   
learn whether reduced add\_solvent continues improving crystallinity or begins to degrade phase   
quality."

```javascript
// Outcome on the landscape: q = 133,471, 147,961, 154,918; best q of the run after iteration 30:
// 154,918
```

Validity of the answers. gpt-5.5 returned a valid batch at the first attempt in 191 of 195 iterations and needed the random fallback once; in the hybrids, gpt-5.5 needed it in 2 (EI hybrid) and 0 (PI hybrid) of 95 LLM iterations. gpt-4o needed the fallback in 54 of 195 iterations (28%) and gpt-5-mini in 70 (36%), so part of their queries was random. Almost all rejected answers repeated an earlier experiment of the run (249 of 259 rejections for gpt-4o, all 379 for gpt-5- mini). A repeat can occur even when the earlier experiment is in the message: in Box S4, the rejected condition had been tested in the previous iteration.

Reproducibility. A language model writes its answer by sampling one word after another from a probability distribution, so the same request can give a different answer each time, even at the lowest temperature or reasoning effort and with a fixed seed (the seed is honored by the provider on a best-effort basis). This variation can be reduced but not removed. We therefore (i) recorded the snapshot behind each alias; (ii) fixed every sampling setting the models accept (temperature or reasoning effort, as above; at most 1,500 completion tokens) and sent a fixed seed with each call; (iii) started each campaign with a test call that stopped the run if the API had silently dropped the seed, the temperature or the reasoning effort, or if a GPT-5 model used more than 200 reasoning tokens; (iv) used the same system message and the same code to build the user message in the LLM-only and hybrid scripts; (v) drew the starting experiments from the fixed seed of the run (Eq. S13); and (vi) logged every iteration: the accepted conditions with their rationales, the strategy summary, every attempt with its error, whether the fallback was used the best-so-far and the cost. Because the user message is a deterministic function of the history, every request can be rebuilt from these logs, and the benchmark curves can be recomputed from the logged conditions without calling the API. Two gaps remain: the system\_fingerprint of the responses was not recorded, so a change of the serving backend within a snapshot cannot be excluded, and the random fallback was drawn without a fixed seed, so its conditions are known only from the logs. A rerun with the same snapshots and seeds should therefore give similar but not necessarily identical proposals; all reported numbers come from the logged runs.

LLM proposals. gpt-5.5 concentrates its queries on the productive region (Figure S12): of its 585 queries in runs 0–4, 81% crystallize, 60% reach the top-1% tier and 41% the top-0.1% tier, compared with 59%, 39% and 16% for Matérn-5/2 EI on the same runs (Figure S11). 87% of its queries use 0.5–0.7 mL NaOH, 83% at most 3 mL $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and nearly all (99.8%) at least 90 °C, i.e. the known window of Section S2.6. Its best-so-far nevertheless rises slowly after the first hits (Section S4.2, Table S15), because it rarely leaves this window to test conditions whose outcome is uncertain.

![](images/8089e399b90b96935d3caa4f5cb9ae7394b668fef28e27a4f0e10024b29909e4.jpg)  
Figure S12: Conditions proposed by gpt-5.5 (LLM only), runs 0–4, iterations 1–39, on all ten two-variable projections of the landscape (highest q over the other three variables; gray, no product). Color gives the iteration in which a condition was proposed, light (iteration 1) to dark (iteration 39); later iterations are drawn on top. Points are jittered within their grid cell. The same plot for Matérn-5/2 EI is Figure S11.

## S3.5. Hybrid GP-LLM Policies

The hybrid policies alternate between two decision-makers that share one history of experiments (Figure S13). In odd iterations, a Matérn-5/2 GP is fitted to all experiments of the run so far and picks three conditions from the 15,000 candidates with EI or PI, exactly as the GP-only policy. In even iterations, gpt-5.5 receives the same history (its 45 most recent experiments, including those chosen by the GP) with the prompt, window and validation of the LLM-only policy, and writes three conditions. After every iteration, the three conditions and their q are appended to the shared history, so the GP is refitted to the LLM's experiments and the LLM sees the GP's experiments in the next iteration. The alternation was chosen to combine the LLM's prior knowledge of productive conditions with the GP's model of the measured data.

To see which decision-maker moves the search forward, we counted in the logs of the five runs every iteration in which the best-so-far of a run increased, and noted who had chosen the batch of that iteration. The EI hybrid had 55 such improvements in its five runs, 42 in LLM iterations and 13 in GP iterations; the PI hybrid had 58, 38 in LLM iterations and 20 in GP iterations. The final best of the run came from an LLM iteration in all 5 runs of both hybrids. Within this design the LLM contributes most improvements, while the GP iterations mainly add smaller refinements.

![](images/b85cbf821add2e4f5f1a1f3ed8883744113534e2c42578f6deeb312f45043188.jpg)

b  
![](images/3ed72ae85caf4e7e24ea0bb6b40565761b357565cdef79d264542dadcd67d357.jpg)  
Figure S13: How the PI hybrid works. (a) Conditions chosen in each iteration, one row per variable: hue, which decision-maker chose the batch (purple, gpt-5.5, even iterations; pink, GP with PI, odd iterations; gray, random start); lightness, position of the chosen value in the range of the variable (light, low; dark, high), mean over the three conditions and five runs. (b) Mean q of each batch, same colors. Data: five runs of the PI hybrid in which every chosen condition was recorded.

## S3.6. All Strategies

Table S13 lists all 39 strategies run on the landscape; 16 of them, chosen as representatives of every class, are compared in the main text. All use the protocol of Table S12, the target log(1 + q), the same batch construction and, where a pool is used, the same candidate sampling. The metrics of all strategies are in Table S15.

Besides GPs, the strategies use other models that return a prediction $\mu$ and an uncertainty σ for every candidate; the acquisition rules of Section S3.3 are then applied in the same way. Two of these models are among the six strategies of Section S4.3. A random forest predicts by averaging many decision trees, each fitted to a resampled copy of the data, and the spread of the trees' predictions, $s _ { \mathrm { t r e e } } .$ serves as uncertainty. BART-like ExtraTrees (extremely randomized trees; “ExtraTrees EI" in the figures and tables) is a variant that chooses the split points of its trees at random; it is used here as a fast stand-in for Bayesian additive regression trees (BART). The conformal random forest adds to the tree spread the typical error of the forest on experiments it was not trained on, $\sigma _ { \mathrm { c o n f } } = ( s _ { \mathrm { t r e e } } ^ { 2 } + \hat { r } _ { 0 . 9 } ^ { 2 } ) ^ { 1 / 2 }$ , where $\hat { r } _ { 0 . 9 }$ is the 90% quantile of these errors. The bandit policies (LinUCB) describe y by a linear function of the five variables; a linear model cannot represent a narrow, closed high-q region, which is consistent with their results (Table S15). The settings of all models are given in Table S13 and in the GitHub repository.

Table S13: All 39 benchmarked strategies. Cand., number of candidate conditions scored per iteration (GP iterations only for the hybrids),
<table><tr><td>Method</td><td>Surrogate</td><td>Acquisition</td><td>Uncertainty</td><td>Settings</td><td>Cand.</td></tr><tr><td colspan="6">GP (kernel and acquisition variants)</td></tr><tr><td>Matérn-3/2 EI</td><td>GP, ARD + white noise, Matérn-3/2</td><td>EI</td><td>GP posterior s.d.</td><td>none</td><td>15,000</td></tr><tr><td>Matérn-5/2 EI</td><td>GP, ARD + white noise, Matérn-5/2</td><td>EI</td><td>GP posterior s.d.</td><td>none</td><td>15,000</td></tr><tr><td>RBF EI</td><td>GP, ARD + white noise, RBF</td><td>EI</td><td>GP posterior s.d.</td><td>none</td><td>15,000</td></tr><tr><td>RQ EI</td><td>GP, ARD + white noise, rational quadratic</td><td>EI</td><td>GP posterior s.d.</td><td>none</td><td>15,000</td></tr><tr><td>RobustGaSP EI</td><td>RobustGaSP (R), Matérn-5/2, nugget esti- mated</td><td>EI</td><td>GP posterior s.d.</td><td>max_eval = 60</td><td>15,000</td></tr><tr><td>Matérn-5/2 PI</td><td>GP, ARD + white noise, Matérn-5/2</td><td>PI</td><td>GP posterior s.d.</td><td>none</td><td>15,000</td></tr><tr><td>Matérn-5/2 UCB</td><td>GP, ARD + white noise, Matérn-5/2</td><td>UCB</td><td>GP posterior s.d.</td><td>β=2.0</td><td>15,000</td></tr><tr><td>Matérn-3/2 PI</td><td>GP, ARD + white noise, Matérn-3/2</td><td>PI</td><td>GP posterior s.d.</td><td>none</td><td>15,000</td></tr><tr><td>Matérn-3/2 UCB</td><td>GP, ARD + white noise, Matérn-3/2</td><td>UCB</td><td>GP posterior s.d.</td><td>β=2.0</td><td>15,000</td></tr><tr><td>Matérn-5/2 TS</td><td>GP, ARD + white noise, Matérn-5/2</td><td>Thompson</td><td>GP posterior s.d.</td><td>none</td><td>15,000</td></tr><tr><td>Matérn-3/2 TS</td><td>GP, ARD + white noise, Matérn-3/2</td><td>Thompson</td><td>GP posterior s.d</td><td>none</td><td>15,000</td></tr><tr><td>greedy µ</td><td>GP, ARD + white noise, Matérn-5/2</td><td>greedy µ</td><td>GP posterior s.d.</td><td>none</td><td>15,000</td></tr><tr><td>quantile 0.9</td><td>GP, ARD + white noise, Matérn-5/2</td><td>quantile τ</td><td>GP posterior s.d.</td><td>τ = 0.9</td><td>15,000</td></tr><tr><td>CVaR 0.9</td><td>GP, ARD + white noise, Matérn-5/2</td><td>CVaRτ</td><td>GP posterior s.d.</td><td>τ = 0.9</td><td>15,000</td></tr><tr><td>diversity EI (λ = 0.1)</td><td>GP, ARD + white noise, Matérn-5/2</td><td>EI</td><td>GP posterior s.d.</td><td>λ=0.1</td><td>15,000</td></tr><tr><td colspan="6">Tree and other ML surrogates</td></tr><tr><td>bootstrap RF EI</td><td>bootstrap random forest</td><td>EI</td><td>s.d. across trees</td><td>n_estimators = 300</td><td>15,000</td></tr><tr><td>conformal RF EI</td><td>random forest (300 trees), spread widened EI by OOB-error quantile</td><td></td><td> $\sqrt { s _ { \mathrm { t r e e } } ^ { 2 } + \hat { r } _ { 0 . 9 } ^ { 2 } }$ </td><td>n_estimators = 300</td><td>15,000</td></tr><tr><td>quantile gradient boosting EI</td><td>quantile gradient boosting</td><td>EI</td><td>10%/90% quantile spread</td><td>none</td><td>15,000</td></tr><tr><td>quantile gradient boosting q0.9</td><td>quantile gradient boosting</td><td>quantile τ</td><td>10%/90% quantile spread</td><td>τ =0.9</td><td>15,000</td></tr><tr><td>MLP ensemble EI</td><td>8 bootstrapped MLPs (32,16)</td><td>EI</td><td>s.d. across ensemble</td><td>n models = 8; α = 1e-3</td><td>15,000</td></tr><tr><td>ExtraTrees EI</td><td>BART-like ExtraTrees (300 trees)</td><td>EI</td><td>s.d. across trees</td><td>n estimators = 300</td><td>15,000</td></tr><tr><td colspan="6">Approximate and linear models</td></tr><tr><td>RFF GP EI</td><td>random Fourier features GP</td><td>EI</td><td>Bayesian linear posterior</td><td>n components = 512</td><td>15,000</td></tr><tr><td>Nyström GP EI</td><td>Nyström GP</td><td>EI</td><td>Bayesian linear posterior local GP posterior s.d.</td><td>n components = 120</td><td>15,000</td></tr><tr><td>laGP ALC EI</td><td>local approximate GP (laGP::aGP), ALC design</td><td>EI</td><td></td><td>start = 10; end = 60</td><td>4,000</td></tr><tr><td>laGP NN EI</td><td>local approximate GP (laGP::aGP), nearest EI neighbours</td><td></td><td>local GP posterior s.d.</td><td>start = 10; end = 60</td><td>4,000</td></tr><tr><td>Bayesian linear EI</td><td>Bayesian linear model</td><td>EI</td><td>Bayesian linear posterior</td><td>none</td><td>15,000</td></tr><tr><td>tures EI</td><td>ARD regression on random fea- ARD regression on RBF random features</td><td>EI</td><td>ARD posterior</td><td>n components = 256</td><td>15,000</td></tr><tr><td>Bandits LinUCB</td><td>contextual LinUCB (linear bandit)</td><td>UCB</td><td> $\hat { s } \sqrt { \phi ^ { \top } A ^ { - 1 } \phi }$ </td><td></td><td>15,000</td></tr><tr><td>LinUCB q0.9</td><td>contextual LinUCB (linear bandit)</td><td>quantile τ</td><td> $\hat { s } \sqrt { \phi ^ { \top } A ^ { - 1 } \phi }$ </td><td>β=2.0 τ = 0.9</td><td>15,000</td></tr><tr><td>chem. bandit UCB</td><td>chemistry-feature bandit, tail-weighted</td><td>UCB</td><td> $\sqrt { \phi ^ { \top } A ^ { - 1 } \phi }$ </td><td>β = 2.75; ridge = 1.0; success</td><td>15,000</td></tr><tr><td></td><td></td><td></td><td></td><td> $\mathrm { { w e i g h t } = 2 . 0 ; }$  top_weight = 4.0; top quantile = 0.8  $\mathrm { r i d g e } = 1 . 0 ;$  success</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>weight = 2.0; top_weight = 4.0; top_quantile = 0.8</td><td>15,000</td></tr><tr><td></td><td></td><td></td><td></td><td>τ = 0.9; ridge = 1.0; success weight = 2.0; top_weight = 4.0; top_quantile = 0.8</td><td>15,000</td></tr><tr><td></td><td></td><td>CVaRτ</td><td> $\sqrt { \phi ^ { \top } A ^ { - 1 } \phi }$ </td><td>τ = 0.9; ridge = 1.0; success weight = 2.0; top_weight = 4.0; top quantile = 0.8</td><td>15,000</td></tr><tr><td>GPT-4o</td><td>gpt-4o, structured JSON output</td><td>LLM pro-</td><td>none</td><td>temperature = 0.2; random seed none</td><td></td></tr><tr><td>GPT-5.5</td><td>gpt-5.5, structured JSON output</td><td>posal LLM</td><td></td><td>per call; window = 45</td><td></td></tr><tr><td>GPT-5-mini</td><td></td><td>pro- posal pro-</td><td>none</td><td>reasoning effort = none; random seed per call; window = 45</td><td>none</td></tr><tr><td></td><td>gpt-5-mini, structured JSON output</td><td>LLM posal</td><td>none</td><td>reasoning effort = minimal; ran- dom seed per call; window = 45</td><td>none</td></tr><tr><td>LLM-GP hybrid Matérn-5/2 EI + GPT-5.5</td><td>odd iterations GP Matérn-5/2 (EI), even it- EI / LLM</td><td></td><td>GP s.d. (GP iterations)</td><td>window = 45</td><td>15,000</td></tr><tr><td>Matérn-5/2 PI + GPT-5.5</td><td>erations gpt-5.5 odd iterations GP Matérn-5/2 (PI), even it- PI / LLM</td><td></td><td>GP s.d. (GP iterations)</td><td>window = 45</td><td>15,000</td></tr><tr><td>Random</td><td>erations gpt-5.5</td><td></td><td></td><td></td><td></td></tr><tr><td>random search</td><td>none</td><td>random</td><td>none</td><td>none</td><td>none</td></tr></table>

## Section S4. Benchmark Results

All metrics are computed from the best-so-far of each run on the noise-free landscape q, over the budget of 120 experiments (iterations 0–39). The comparison proceeds in two steps. First, all 39 strategies are compared with $n = 5$ runs (Section S4.2). This separates strong from weak strategies but leaves a large group of strong strategies that five runs cannot tell apart. Six strategies of this group are then run 20 times $( n = 2 0$ ; Section S4.3), and the remaining analyses (Sections S4.4–S4.7) use these six.

## S4.1. Metrics

Table S14 collects the quantities used in this section and in Sections S3 and S5.

Table S14: Quantities used to compare the strategies.
<table><tr><td>Quantity</td><td>Definition</td></tr><tr><td>Tier thresholds</td><td>Top 1%:  $q \geq 4 2 , 4 6 3 ;$  top 0.1%:  $q \geq 1 0 0 , 2 8 8 ;$  quantiles over the 6,101,077 emulated conditions (Table S10)</td></tr><tr><td>Iteration t</td><td>Three starting experiments, then three per iteration, so iteration t ends after 3 + 3t experiments;  $t = 3$  is 12 experiments (the wet-lab budget),  $t = 1 3$  is 42,  $t = 3 9$  is 120</td></tr><tr><td>Best-so-far  $q ^ { + } ( t )$ </td><td>Highest q among all experiments of a run up to iteration t (Eq. S19); higher is better</td></tr><tr><td>Hit rate  $H _ { 1 \% } ( t ) , H _ { 0 . 1 \% } ( t )$ </td><td>Fraction of the runs whose best-so-far has reached the tier by iteration t (Eq. S20); higher is better. Early hit rate:  $H _ { 1 \% } ( 3 )$ </td></tr><tr><td>Normalized regret R(t)</td><td>Mean distance of the best-so-far from the highest q on the  $\mathrm { g r i d } ,$  as a fraction of that value (Eq. S21); 0 if every run has found the best condition; lower is better</td></tr><tr><td>Query share</td><td>Fraction of proposed conditions that give a product or reach a tier (Sections S3.2 and S3.4); not a hit rate</td></tr><tr><td>Leading group</td><td>Strategies that beat random search and cannot be distinguished from the best strategy</td></tr><tr><td>Independent / shared starts</td><td>Independent: each strategy and run draws its own three starting conditions (Eq. S13). Shared: all strategies of a run start from the same three conditions</td></tr><tr><td>Bands in the figures</td><td>95% confidence interval of the mean over the runs  $( \mathrm { m e a n } \pm 1 . 9 6$  standard errors), in all figures Where a p value is given, it comes from a rank test, which compares</td></tr><tr><td>p values</td><td>the order of the values and assumes no particular distribution: Kruskal-Wallis, Dunn and Mann-Whitney tests for independent runs[8, 9], Wilcoxon signed-rank test for shared-start runs. When several pairs are tested, the p values are adjusted by Holm&#x27;s</td></tr><tr><td>Spearman  $\rho$ </td><td>method[19] Correlation between the rank orders of two sets of values (1, same order; 0, unrelated)</td></tr></table>

Best-so-far and hit rate. The best-so-far after iteration t of run $r$ is

$$
q _ { r } ^ { + } ( t ) = \operatorname* { m a x } _ { 1 \leq i \leq 3 + 3 t } q ( \mathbf { x } _ { r , i } ) ,\tag{S19}
$$

including the three starting experiments; iteration 3 corresponds to 12 experiments, the budget of a wet-lab campaign. With the tier thresholds of Table S10, $q _ { 1 \% } = 4 2 , 4 6 3$ and $q _ { 0 . 1 \% } = 1 0 0 , 2 8 8$

the hit rate is the fraction of the n runs whose best-so-far has reached the tier,

$$
H _ { \alpha } ( t ) = { \frac { 1 } { n } } \sum _ { r = 1 } ^ { n } \mathbb { 1 } \left[ q _ { r } ^ { + } ( t ) \geq q _ { \alpha } \right] ,\tag{S20}
$$

where 1[·] is 1 if the condition in brackets holds and 0 otherwise. The normalized regret is

$$
R ( t ) = \frac { 1 } { n } \sum _ { r = 1 } ^ { n } \frac { q _ { \mathrm { { m a x } } } - q _ { r } ^ { + } ( t ) } { q _ { \mathrm { { m a x } } } - q _ { \mathrm { { m i n } } } } ,\tag{S21}
$$

with $q _ { \mathrm { m a x } } = 2 4 7 , 1 0 7$ , the largest q on the grid, and $q _ { \mathrm { m i n } } = 0$ . The early hit rate is $H _ { 1 \% } ( 3 )$ , the hit rate after 12 experiments.

Random baseline. For reference, 120 random grid points contain a top-0.1% condition with probability 0.11; the 20 random-search runs gave 0.10.

## S4.2. All Strategies, $n = 5$

Figure S14 shows the best-so-far of all 39 strategies together, Figure S15 shows the same curves in groups, with every strategy named, and Table S15 lists the metrics; the $n = 5$ runs of every other figure and table are the same runs. Most full GPs, the tree ensembles and the two hybrids rise furthest above random search, whereas the bandits, the approximate and linear models and two of the three LLM-only policies stay closer to it. The top-1% hit rate after 7 iterations is 0.6 for conformal RF and for Matérn-5/2 EI (0.8 for Matérn-5/2 PI), and ExtraTrees reaches the top-1% tier in a median of 4 iterations.

Table S15: All 39 strategies, $n = 5$ . Mean and Median: best q after 120 experiments in thousands (mean $\pm \ \mathrm { s . d }$ . over the five runs, and median; higher is better). Hit rates (Eq. S20; higher is better): $H _ { 1 \% } ( 3 9 )$ and $H _ { 0 . 1 \% } ( 3 9 )$ , fraction of the five runs that have reached the top-1% and the top-0.1% tier after 120 experiments (iteration 39); $H _ { 0 . 1 \% } ( 1 3 )$ , the same after 42 experiments (iteration 13). Class as in Table S13. Ordered by median best $q .$
<table><tr><td>Strategy</td><td>Class</td><td></td><td>Mean Median</td><td></td><td> $H _ { 1 \% } ( 3 9 )$ </td><td></td><td> $H _ { 0 . 1 \% } ( 1 3 )$ </td><td> $H _ { 0 . 1 \% } ( 3 9 )$ </td></tr><tr><td>conformal RF EI</td><td>Tree / ML</td><td></td><td> $1 7 9 \pm 3 8$ </td><td></td><td>186</td><td>1.0</td><td>0.2</td><td>1.0</td></tr><tr><td>Matérn-3/2 UCB</td><td>GP</td><td></td><td> $1 8 1 \pm 2 6$ </td><td></td><td>183</td><td>1.0</td><td>0.4</td><td>1.0</td></tr><tr><td>Matérn-5/2 PI</td><td>GP</td><td></td><td> $1 8 4 \pm 2 6$ </td><td></td><td>176</td><td>1.0</td><td>0.4</td><td>1.0</td></tr><tr><td>GPT-5.5</td><td>LLM</td><td></td><td> $1 4 2 \pm 4 7$ </td><td></td><td>169</td><td>1.0</td><td>0.6</td><td>0.8</td></tr><tr><td>quantile gradient boosting EI</td><td></td><td>Tree / ML</td><td>152 ± 40</td><td></td><td>169</td><td>1.0</td><td>0.4</td><td>0.8</td></tr><tr><td>ExtraTrees EI</td><td></td><td>Tree / ML</td><td>169 ± 48</td><td></td><td>166</td><td>1.0</td><td>0.8</td><td>1.0</td></tr><tr><td>MLP ensemble EI</td><td>Tree / ML</td><td></td><td>146 ± 54</td><td></td><td>165</td><td>1.0</td><td>0.2</td><td>0.8</td></tr><tr><td>Matérn-5/2 EI</td><td>GP</td><td></td><td> $1 7 9 \pm 3 8$ </td><td></td><td>164</td><td>1.0</td><td>0.6</td><td>1.0</td></tr><tr><td>quantile 0.9</td><td>GP</td><td></td><td>160 ± 62</td><td></td><td>161</td><td>1.0</td><td>0.6</td><td>0.8</td></tr><tr><td>CVaR 0.9</td><td>GP</td><td></td><td>163 ± 32</td><td></td><td>160</td><td>1.0</td><td>0.2</td><td>1.0</td></tr><tr><td>Matérn-3/2 EI</td><td>GP</td><td></td><td>144 ± 52</td><td></td><td>160</td><td>1.0</td><td>0.4</td><td>0.8</td></tr><tr><td>Matérn-3/2 TS</td><td>GP</td><td></td><td>172 ± 48</td><td></td><td>159</td><td>1.0</td><td>0.2</td><td>1.0</td></tr><tr><td>RBF EI</td><td>GP</td><td></td><td>161 ± 26</td><td></td><td>158</td><td>1.0</td><td>0.4</td><td>1.0</td></tr><tr><td>greedy µ</td><td>GP</td><td></td><td>133 ± 55</td><td></td><td>156</td><td>1.0</td><td>0.6</td><td>0.6</td></tr><tr><td>Matérn-5/2 EI + GPT-5.5</td><td></td><td>LLM-GP hybrid 167 ± 58</td><td></td><td></td><td>155</td><td>1.0</td><td>0.4</td><td>1.0</td></tr><tr><td>RQ EI</td><td>GP</td><td></td><td>158 ± 58</td><td></td><td>155</td><td>1.0</td><td>0.2</td><td>0.8</td></tr><tr><td>Matérn-3/2 PI</td><td>GP</td><td></td><td>169 ± 40</td><td></td><td>155</td><td>1.0</td><td>0.4</td><td>1.0</td></tr><tr><td>Matérn-5/2 PI + GPT-5.5</td><td></td><td>LLM-GP hybrid</td><td> $1 7 2 \pm 4 9$ </td><td></td><td>153</td><td>1.0</td><td>0.2</td><td>1.0</td></tr><tr><td>RobustGaSP EI</td><td>GP</td><td></td><td> $1 3 6 \pm 3 5$ </td><td></td><td>150</td><td>1.0</td><td>0.0</td><td>0.8</td></tr><tr><td>bootstrap RF EI</td><td>Tree / ML</td><td></td><td> $1 5 5 \pm 2 5$ </td><td></td><td>150</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Matérn-5/2 TS</td><td>GP</td><td></td><td> $1 4 0 \pm 1 4$ </td><td></td><td>146</td><td>1.0</td><td>0.4</td><td>1.0</td></tr><tr><td>diversity EI (λ = 0.1)</td><td>GP</td><td></td><td> $1 7 4 \pm 4 5$ </td><td></td><td>144</td><td>1.0</td><td>0.2</td><td>1.0</td></tr><tr><td>chem. bandit TS</td><td>Bandit</td><td></td><td> $1 2 9 \pm 3 7$ </td><td></td><td>141</td><td>1.0</td><td>0.4</td><td>0.8</td></tr><tr><td>Matérn-5/2 UCB</td><td>GP</td><td></td><td> $1 2 6 \pm 2 0$ </td><td></td><td>122</td><td>1.0</td><td>0.2</td><td>1.0</td></tr><tr><td>laGP ALC EI</td><td></td><td>Approx. / linear</td><td> $1 1 6 \pm 3 8$ </td><td></td><td>116</td><td>1.0</td><td>0.4</td><td>0.8</td></tr><tr><td>Bayesian linear EI</td><td></td><td>Approx. / linear</td><td> $1 0 9 \pm 4$ </td><td></td><td>109</td><td>1.0</td><td>0.4</td><td>1.0</td></tr><tr><td>laĠP NN EI</td><td></td><td>Approx. / linear</td><td> $9 7 \pm 3 0$ </td><td></td><td>105</td><td>1.0</td><td>0.0</td><td>0.6</td></tr><tr><td>Strategy</td><td>Class</td><td></td><td></td><td>Mean Median</td><td></td><td> $H _ { 1 \% } ( 3 9 )$ </td><td> $H _ { 0 . 1 \% } ( 1 3 )$ </td><td> $H _ { 0 . 1 \% } ( 3 9 )$ </td></tr><tr><td>LinUCB q0.9</td><td>Bandit</td><td></td><td>74 ± 43</td><td></td><td>101</td><td>0.6</td><td>0.2</td><td>0.6</td></tr><tr><td>chem. bandit CVaR</td><td>Bandit</td><td></td><td>129 ± 88</td><td></td><td>99</td><td>1.0</td><td>0.4</td><td>0.4</td></tr><tr><td>LinUCB</td><td>Bandit</td><td></td><td>91 ± 21</td><td></td><td>94</td><td>1.0</td><td>0.0</td><td>0.4</td></tr><tr><td>quantile gradient boosting q0.9</td><td>Tree / ML</td><td></td><td>83 ± 55</td><td></td><td>82</td><td>0.8</td><td>0.0</td><td>0.2</td></tr><tr><td>GPT-4o</td><td>LLM</td><td></td><td>69 ± 42</td><td></td><td>73</td><td>0.6</td><td>0.0</td><td>0.2</td></tr><tr><td>random search</td><td>Random</td><td></td><td>63 ± 43</td><td></td><td>71</td><td>0.6</td><td>0.0</td><td>0.2</td></tr><tr><td>Nyström GP EI</td><td>Approx. / linear</td><td></td><td>96 ± 79</td><td></td><td>67</td><td>0.8</td><td>0.2</td><td>0.4</td></tr><tr><td>RFF GP EI</td><td>Approx. / linear</td><td></td><td>85 ± 40</td><td></td><td>63</td><td>1.0</td><td>0.2</td><td>0.4</td></tr><tr><td>GPT-5-mini</td><td>LLM</td><td></td><td>69 ± 41</td><td></td><td>58</td><td>0.8</td><td>0.0</td><td>0.4</td></tr><tr><td>chem. bandit q0.9</td><td>Bandit</td><td></td><td>79 ± 57</td><td></td><td>53</td><td>0.6</td><td>0.2</td><td>0.4</td></tr><tr><td>chem. bandit UCB</td><td>Bandit</td><td></td><td>67 ± 28</td><td></td><td>53</td><td>1.0</td><td>0.0</td><td>0.2</td></tr><tr><td>ARD regression on random features EI Approx. / linear</td><td></td><td></td><td>75 ± 60</td><td></td><td>49</td><td>0.8</td><td>0.2</td><td>0.2</td></tr></table>

Number of replicate runs. With five runs, the strategies at the top cannot be ranked: their run-to-run spread is as large as the differences between them (Figure S15). Six of them were therefore run 20 times (Section S4.3). They were chosen to cover the two model families of this group: the four Matérn GPs (both kernels, each with EI and PI), the family used in the wet lab and the two tree ensembles (conformal RF and BART-like ExtraTrees), which share the model family of the emulator and could profit from it (Section S4.4). The LLM and hybrid strategies were not extended because of the API cost.

![](images/e94cac81007f7b0f992c97f3178adde591a426c343f9b50086308c7cb80b60ab.jpg)

b  
![](images/d968bddb74db3e5c998bca079f99e7261aaa519b612ce97e92689ae1445d22a3.jpg)  
Figure S14: All 39 strategies, $n = 5$ . (a) Mean best-so-far $q$ of every strategy in one plot. Color, class of the strategy (blue, GP; green, tree and other ML; purple, approximate and linear; yellow, bandit; orange, LLM; pink, LLM–GP hybrid; black, random search); tint and line style distinguish the strategies of a class (legend, ordered by class and by final mean). (b) The same means as a heat map, one row per strategy, ordered by mean best $q$ at iteration 39; label color as in (a).

![](images/62488fac93fce41b66e83fd22d5f0f39c8e9552771020f52baf21f3391f1c5bb.jpg)

GP: PI, UCB and Thompson sampling (6)GP: other rules (4)  
![](images/b890bd309ad17b2efe5f83e7a18560c5c1fdea4802cb064084ab01250e5e4980.jpg)

![](images/96c4b4dae3854114795eccf6c1eac79ccf8c759eaf16d7fa301c25710b35e753.jpg)

Tree and other ML (6)  
![](images/d2b6dcb7d70600e438026f295cac9ac84a043e7f23983357105ddb3226ba039b.jpg)

![](images/6955a2664a5c92eda8d659f2585655642fec7f74913fd2be842cde93333f8c0b.jpg)

Bandit (6)  
![](images/724b4ffde10840047e7ef889ec2a2effe9c206ff09c5ce96bbc8fca4bc884100.jpg)

![](images/667d96b35dd2383e6f12e047bea4552ef8f2246b389f0bedfcfdf07cae46b538.jpg)  
Figure S15: The curves of Figure S14a in groups, every strategy named (number of strategies in parentheses; the 15 GP strategies in three panels): mean best-so-far q (line) and 95% CI of the mean (band), n = 5. Gray, random search, repeated in every panel; dashed and dotted lines, top-1% and top-0.1% thresholds.

## S4.3. Leading Group, $n = 2 0$

The six strategies (Matérn-5/2 and Matérn-3/2 GPs with EI and PI, conformal RF and BARTlike ExtraTrees) were run 20 times each, together with random search (Figure S16, Table S16).

With 20 runs the six strategies still cannot be told apart: their curves overlap throughout (Figure S16a), and a rank test finds no difference among them at iteration 13 $( p = 0 . 7 2 )$ or 39 $( p = 0 . 8 7 )$ . All six are far above random search (adjusted $p < 0 . 0 0 1$ at iteration 39). After 120 experiments, 85–100% of their runs have reached the top-0.1% tier, against 10% for random search. Their early hit rate $H _ { 1 \% } ( 3 )$ is 0.05–0.35 (random search 0.10), i.e. 12 experiments from a random start are rarely enough (Section S4.6).

![](images/3b2e18ed00f7190d621a45d7059da7bde7eebac45e397f507c935d8594c8ff88.jpg)

b  
![](images/6d8ad83093dc4b0f2bf0b0defef923337d68c8716ca325dff634ba084fa35fa0.jpg)  
Figure S16: The leading group with $n = 2 0$ runs. (a) Best-so-far q, mean and 95% CI; dashed and dotted lines, top-1% and top-0.1% thresholds. (b) Top-0.1% hit rate $H _ { 0 . 1 \% } ( t )$

Table S16: The six strategies and random search with $n = 2 0$ runs each. Mean and Median: best q after 120 experiments in thousands (mean ± s.d. over the 20 runs, and median; higher is better). $H _ { 1 \% } ( 3 )$ , fraction of the runs in the top-1% tier after 12 experiments; $H _ { 0 . 1 \% } ( 3 9 )$ , fraction in the top-0.1% tier after 120 experiments (higher is better). R(39), normalized regret after 120 experiments (Eq. S21; lower is better).
<table><tr><td>Strategy</td><td></td><td>Mean Median</td><td> $H _ { 1 \% } ( 3 )$ </td><td> $H _ { 0 . 1 \% } ( 3 9 )$ </td><td>R(39)</td></tr><tr><td>Matérn-5/2 PI</td><td> $1 6 9 \pm 2 5$ </td><td>167</td><td>0.25</td><td>1.00</td><td>0.32</td></tr><tr><td>Matérn-5/2 EI</td><td> $1 6 2 \pm 3 9$ </td><td>155</td><td>0.05</td><td>0.95</td><td>0.34</td></tr><tr><td>Matérn-3/2 PI</td><td> $1 6 8 \pm 3 4$ </td><td>157</td><td>0.10</td><td>1.00</td><td>0.32</td></tr><tr><td>Matérn-3/2 EI</td><td> $1 6 1 \pm 3 5$ </td><td>160</td><td>0.05</td><td>0.95</td><td>0.35</td></tr><tr><td>conformal RF EI</td><td> $1 4 6 \pm 6 2$ </td><td>159</td><td>0.15</td><td>0.85</td><td>0.41</td></tr><tr><td>ExtraTrees EI</td><td> $1 6 1 \pm 3 3$ </td><td>158</td><td>0.35</td><td>1.00</td><td>0.35</td></tr><tr><td>Random search</td><td> $6 3 \pm 3 0$ </td><td>61</td><td>0.10</td><td>0.10</td><td>0.75</td></tr></table>

## S4.4. Robustness to the Construction of the Landscape

Section S2.5 tests the emulator against measured outcomes. Here we ask whether the strategy comparison would change if the landscape had been built in another reasonable way. We rebuilt it 41 times, each time changing one choice of Section S2.4 and keeping everything else. The changes fall into five groups. (i) Forest random state (10 landscapes): the random forests were refitted with ten other random seeds, which tests the randomness of the fit itself. (ii) Anchor resampling (20): the forests were trained on a resampled set of anchors, made by drawing 95 times at random from the 95 anchors and putting each drawn anchor back, so that some anchors appear more than once and others not at all; this tests how much the landscape depends on individual experiments. (iii) Blending and threshold rules (8): one rule changed at a time, the Rule 1 blend to 60:40 or 100:0 instead of 80:20, the Rule 2 blend to 30:70 or 70:30 instead of 50:50, the intensity threshold to 2000 or 5000 instead of 3000, or the FWHM threshold to $2 ^ { \circ }$ or $5 ^ { \circ }$ instead of $3 ^ { \circ }$ . (iv) Emulator model (2): the random forests replaced by gradient boosting or by k-nearest neighbors. (v) Fewer anchors (1): only the first 60 anchors in recording order. Changing a choice moved the top-0.1% tier within the productive region but not out of it: 75–100% of the original top-0.1% conditions stayed in the top 1% of every rebuilt landscape.

On every landscape, the six strategies and random search were rerun with $n = 2 0$ (Figure S17). Random search was lowest on all 41 landscapes: the mean best $q$ of every one of the six strategies was at least 1.4 times that of random search, and all six beat it on every landscape (rank test, adjusted $p \leq 0 . 0 3 3 )$ . The order among the six, however, changed from landscape to landscape, and each of them ranked first on at least one (Figure S18). The separation from random search is a property of the landscape; the order among the six is not.

Emulator model and search policy. Because the emulator is a random forest, tree-based policies share its model family and might find the landscape easier than a real synthesis space. The effect exists but is small and runs both ways: the two tree policies ranked first and second on the gradient-boosting map, and two GPs on the k-nearest-neighbor map, whereas on the random-forest map used throughout this work the tree policies ranked fourth and sixth. These differences lie within the run-to-run spread, and the GP policies used in the wet lab do not depend on the emulator being a random forest.

![](images/ca0e559cdbbccde6cead68f89b38bb491e43b9faa1092ab9fb431044254ebdf5.jpg)  
Figure S17: Best-so-far q of the six strategies and random search on the original landscape and on each of the 41 rebuilt ones (mean of $n = 2 0 { \mathrm { ~ r u n s } } )$ Panel titles give the choice that was changed: forest random state 0–9, anchor resampling 1–20, one blending or threshold rule, the emulator model, or the first 60 anchors only. Dotted line, top-0.1% threshold of the landscape of the panel: every rebuilt landscape has its own q values, so the q that separates its best 0.1% of conditions differs from landscape to landscape.

![](images/1df365cb2461dd90b1530743f44b2c51f666853e6dc9ae4f51108d1d530c35f6.jpg)  
Figure S18: Summary of Figure S17: mean best q after 120 experiments $( n = 2 0 { \mathrm { ~ r u n s } } )$ of each strategy on the original landscape (star) and on the 41 rebuilt ones (dots, colored by the group of the change; number of landscapes in parentheses). Line, median over the 42 landscapes; shaded, random search.

## S4.5. PI versus EI

Independent-start runs. In the independent-start runs of $\mathrm { M a t e r n { - } 5 / 2 }$ , PI has a higher mean best-so-far than EI at every iteration from 0 to 5, both with $n = 5$ and with $n = 2 0$ (Figure S19a,b). The lead is present at iteration 0, before either rule has chosen a point, because 7 of the 20 PI starting batches contain a product against 1 of the 20 EI batches. PI also has a larger share of queries that crystallize (panel c; pooled over iterations 1–5, 38% against 11%).

Shared-start rerun. To separate the acquisition rule from the starting batch, we reran both rules with the production code but with a random seed that does not depend on the method, so that PI and EI start from the same three experiments and use the same stream of random numbers $( n = 2 0 )$ . The best-so-far curves then overlap (Figure S19d): EI leads at iterations 1, 2, 4 and 5 and PI at iteration 3. The early advantage of PI in the independent-start runs is therefore a consequence of its starting batches, not of the acquisition rule.

![](images/75354992c7ef7580cf953ed415da578cf01765c0534c3fab99f5666c084a3e9d.jpg)

![](images/db4ab926981c1f7be413a0886b3c1c26ebd8a9aabf915a5cea1b4b0d6f0bcb86.jpg)

![](images/d11e90598c73b2e93755e558dbc288c2c06b47c0ded769d2254f37a39efd5b2f.jpg)

![](images/bb06ef1005f4a42d7afde743fc83c839d9fe24c6e049afced40c4697fe98c438.jpg)  
Figure S19: Matérn-5/2 PI vs. EI in iterations 0–5. (a, b) Mean best-so-far q and 95% CI of the independent-start runs, $n = 5$ (runs 0–4) and $n = 2 0$ (c) Share of queries with a crystalline product, $n = 2 0$ . (d) Shared-start rerun: both rules start from the same three experiments, $n = 2 0$

Measurement noise. In the benchmark, the landscape returns the exact q of a condition. In the wet lab, one powder loaded and measured five times gave q values with a coefficient of variation of 30% (Section S2.1). To see how the two rules cope with this, the shared-start runs of $\mathrm { M a t e r n { - } 5 / 2 }$ PI and EI were repeated with a noisy landscape. Every time a strategy queries a condition, the landscape value $q$ is multiplied by a random factor $\varepsilon ,$ drawn anew for every query from a log-normal distribution with mean 1 and a coefficient of variation of 30%, which reproduces the measured loading-to-loading scatter; a condition without product still returns $q = 0$ . The strategy sees and learns from these noisy values only. The curves of Figure S20, however, are drawn with the noise-free landscape value of every condition the strategy chose: they show how good the chosen conditions really are, not how high the noisy readings happened to be. Noise lowers the mean best q at iteration 39 by 8.8% for PI (170,212 to 155,206) and by 0.3% for EI (150,068 to 149,545). PI is the more sensitive rule, as expected from its preference for small predicted gains.

![](images/acafa216484b8564de8719f02211c9aeea7aae8ac99f9745ad4e1ebe22b4f89e.jpg)

![](images/89ef8fba5bb2b4f0af91484f805f5d20199a959ee56d152bc8be1b9f7b08efec.jpg)  
Figure S20: Shared-start rerun of Matérn-5/2 PI and EI with a noise-free oracle (solid) and with 30% log-normal measurement noise (dashed); best-so-far on the true landscape q, mean and 95% CI, n = 20. (a) Iterations 0–39; (b) iterations 0–5 enlarged.

## S4.6. Warm Start

The wet-lab campaigns (Section S5) do not start from random conditions but from the three conditions proposed by the MOF-321 prior (R00001-R00003). These are the first batch that the desktop application proposes before any result of a new MOF has been entered: they are computed from the reference GP fitted to the 95 MOF-321 anchors (Sections S5.2 and S5.3) and are therefore the same for every new MOF to be synthesized. To separate what the starting point contributes from what the search contributes, we reran the six strategies and random search on the MOF-321 landscape with two warm starts: (i) the three prior conditions, identical in all runs; and (ii) three conditions drawn at random from the band between the top-1% and top-0.1% thresholds, one draw per run. The top 0.1% itself is excluded, so the start is never the answer. Figure S21 shows where these starting conditions lie, together with the random starts of Section S4.3. Each arm has 20 runs of 39 iterations, and within a run all strategies share the starting batch and the random seed. The cold-start curves are those of Section S4.3.

The prior conditions reach $q = 8 0 , 7 6 0$ on the landscape (top 1%). From there, random search barely improves: its median best q is still 80,760 after 39 iterations (Figure S22), and only 6 of 20 runs reach the top-0.1% tier (Table S17). The six strategies reach the top-0.1% tier in 65–100% of runs by iteration 13 and in all runs by iteration 39, and each of them is better than random search (paired rank test, Holm-adjusted $p \leq 0 . 0 0 2$ at iterations 13 and 39). The random top-1%–0.1% start gives almost the same curves as the prior start. The start therefore decides whether the search begins in the productive region, whatever the particular conditions; the climb from the top-1% to the top-0.1% tier is done by the search strategy, and this is the part of the campaign that the benchmark evaluates.

Compared with a cold start, a warm start saves roughly the first 10–15 iterations: with 12 experiments (iteration 3), at most 35% of cold-start runs reach even the top-1% tier (Table S16), against 100% of warm-start runs, of which 10–-40% are already in the top-0.1% tier. This is why the wet-lab campaigns, limited to 12 experiments, use the prior.

![](images/4654f26e32ff54ea9e6da737edfb73df85e574ca5310497aea70bf41414382e9.jpg)  
Figure S21: Starting conditions of the three start types on all ten two-variable projections of the MOF-321 landscape (highest q over the other three variables; gray, no product). Open circles, random starts (the 20 starting batches of Matérn-5/2 EI, n = 20); crosses, prior start R00001–R00003 (the same in all runs); orange, starts drawn from the top-1%–0.1% band (one batch of three per run). Points are jittered within their grid cell.

![](images/2ac393f4c8ab56a4deaab5e4ede9a646f5198b9d969640cced002e045a4d19c1.jpg)  
Figure S22: Warm start on the MOF-321 landscape. Mean best-so-far q of six strategies and random search from a random start (gray, Section S4.3), from the three conditions proposed by the MOF-321 prior (blue, with 95% CI) and from three random conditions in the top-1%–0.1% band (orange); $n = 2 0$ runs each. Dashed and dotted lines, top-1% and top-0.1% thresholds.

Table S17: Warm start vs. cold start $( n = 2 0$ each): top-0.1% hit rate $H _ { 0 . 1 \% } ( t )$ at iterations t = 3, 13 and 39 (12, 42 and 120 experiments; higher is better).
<table><tr><td></td><td></td><td colspan="3"> $H _ { 0 . 1 \% }$ </td></tr><tr><td>Strategy</td><td>Start</td><td>3</td><td>13</td><td>39</td></tr><tr><td>Matérn-5/2 EI</td><td>random</td><td>0.00</td><td>0.40</td><td>0.95</td></tr><tr><td></td><td>prior (R00001–3)</td><td>0.15</td><td>0.85</td><td>1.00</td></tr><tr><td></td><td>top 1%–0.1% band</td><td>0.10</td><td>0.80</td><td>1.00</td></tr><tr><td>Matérn-5/2 PI</td><td>random</td><td>0.05</td><td>0.50</td><td>1.00</td></tr><tr><td></td><td>prior (R00001–3)</td><td>0.20</td><td>1.00</td><td>1.00</td></tr><tr><td></td><td>top 1%–0.1% band</td><td>0.15</td><td>0.90</td><td>1.00</td></tr><tr><td>Matérn-3/2 EI</td><td>random</td><td>0.00</td><td>0.50</td><td>0.95</td></tr><tr><td></td><td>prior (R00001–3)</td><td>0.20</td><td>0.80</td><td>1.00</td></tr><tr><td></td><td>top 1%–0.1% band</td><td>0.05</td><td>0.55</td><td>1.00</td></tr><tr><td>Matérn-3/2 PI</td><td>random</td><td>0.00</td><td>0.40</td><td>1.00</td></tr><tr><td></td><td>prior (R00001–3)</td><td>0.40</td><td>0.95</td><td>1.00</td></tr><tr><td></td><td>top 1%–0.1% band</td><td>0.25</td><td>0.95</td><td>1.00</td></tr><tr><td>Conformal RF EI</td><td>random</td><td>0.05</td><td>0.35</td><td>0.85</td></tr><tr><td></td><td>prior (R00001–3)</td><td>0.10</td><td>0.65</td><td>1.00</td></tr><tr><td></td><td>top 1%–0.1% band</td><td>0.15</td><td>0.80</td><td>1.00</td></tr><tr><td>ExtraTrees EI</td><td>random</td><td>0.05</td><td>0.65</td><td>1.00</td></tr><tr><td></td><td>prior (R00001–3)</td><td>0.30</td><td>0.75</td><td>1.00</td></tr><tr><td></td><td>top 1%–0.1% band</td><td>0.10</td><td>1.00</td><td>1.00</td></tr><tr><td>Random search</td><td>random</td><td>0.00</td><td>0.00</td><td>0.10</td></tr><tr><td></td><td>prior (R00001–3)</td><td>0.00</td><td>0.10</td><td>0.30</td></tr><tr><td></td><td>top 1%–0.1% band</td><td>0.00</td><td>0.10</td><td>0.30</td></tr></table>

## S4.7. Modeling the No-Product Region

In all strategies above, an experiment without product enters the model as an ordinary value, $y = \log ( 1 + 0 ) = 0$ , on the same scale as the crystalline results. Because $9 4 \%$ of the landscape gives no product, one may ask whether the search would be faster if the question "product or no product" were modeled separately. This is an established idea in Bayesian optimization, known as optimization with unknown constraints or with failed experiments[13-15], and it has been applied in chemistry and in materials growth[16, 17]. We tested two policies of this kind against Matérn-5/2 EI.

(i) Product probability times EI. A GP classifier is fitted to all experiments of the run and gives, for every candidate, the probability $p ( \mathbf { x } )$ that it yields a product. A GP regression is fitted to the crystalline experiments only and gives $\operatorname { E I } _ { + } ( \mathbf { x } )$ , the expected improvement if a product forms. Candidates are scored by

$$
a ( { \bf x } ) = p ( { \bf x } ) \mathrm { E I } _ { + } ( { \bf x } ) ,\tag{S22}
$$

which is exactly the expected improvement of a model in which a condition gives no product with probability $1 - p$ and a normally distributed y otherwise.

(ii) Lowest observed value. An experiment without product is entered with the lowest y of the products found so far in the run instead of 0, and the usual GP with EI is used[17]; the failures then no longer lie far below all products. Before the first product, both policies follow the same rule as the baseline (Section S3.1), and all other settings are those of Table S12. Each policy was run 20 times with shared starts: in every run, the three policies begin with the same three experiments.

Neither policy was faster than the baseline (Figure S23, Table S18). The first product appeared after the same number of experiments in all three (median 12.5). This is so by construction: none of them has a model before a product exists. After 12 experiments the three policies had the same median best q. Later, the baseline was nominally ahead after 42 experiments and policy (ii) after 120, but none of the comparisons with the baseline that we had fixed beforehand (first product, share of productive queries in iterations 1-5, best q after 12 and after 42 experiments; paired rank tests) showed a difference (smallest adjusted $p = 0 . 3 4 )$ With 20 runs, small differences cannot be excluded. Policy (i) had one drawback: in 10 of its 20 runs it went 10 or more iterations without a new product after its first one (baseline: no run). The regression behind $\mathrm { E I _ { + } }$ has seen only products, so its expected improvement is largest in unexplored corners of the grid, and p lowers it there only moderately.

The absence of an early gain has the reason given in Section S3.1: the early search is limited by finding the first product, and before that a model of the no-product region has nothing to learn from either. In our setting this limit is removed by the prior (Section S4.6), not by the model. The plain GP with EI was therefore kept.

Table S18: Modeling the no-product region, shared starts, $n = 2 0$ . First product: median number of experiments until the first product. Median best q in thousands after 12, 42 and 120 experiments (higher is better). $\textstyle H _ { 0 . 1 \% } \colon$ runs in the top-0.1% tier after 42 and 120 experiments (higher is better). Stalled: runs with 10 or more iterations in a row without a new product after the first one (lower is better).
<table><tr><td></td><td></td><td colspan="3">Median best q</td><td colspan="2"> $H _ { 0 . 1 \% }$ </td><td></td></tr><tr><td>Policy</td><td>First product</td><td>12</td><td>42</td><td>120</td><td>42</td><td>120</td><td>Stalled</td></tr><tr><td>Matérn-5/2 EI (baseline)</td><td>12.5</td><td>1.1</td><td>92</td><td>155</td><td>9/20</td><td>18/20</td><td>0/20</td></tr><tr><td>(i) product probability × EI</td><td>12.5</td><td>1.1</td><td>78</td><td>152</td><td>8/20</td><td>17/20</td><td>10/20</td></tr><tr><td>(ii) lowest observed value</td><td>12.5</td><td>1.1</td><td>55</td><td>165</td><td>5/20</td><td>19/20</td><td>3/20</td></tr></table>

![](images/4069b87041849447288699b1e954d2d7b98ae754206670cc2e91fb58aed80ebd.jpg)  
Figure S23: Modeling the no-product region: Matérn-5/2 EI (baseline) and the two policies of Section S4.7, shared starts, n = 20. Mean best-so-far q and 95% CI of the mean. (a) Iterations 0–39; dashed and dotted lines, top-1% and top-0.1% thresholds. (b) Iterations 0–5 enlarged.

Choice of policy for the wet lab. The six strategies could not be told apart (Section S4.3), and their order changed with the construction of the landscape (Section S4.4). The wet-lab campaigns used a Matérn-5/2 GP with EI for two practical reasons. First, EI is robust to measurement noise: with 30% noise it lost 0.3% of its final best q, against 8.8% for PI (Figure S20), and with shared starts it progressed as fast as PI in the first iterations (Figure S19d). Second, the GP is easy to adjust: in the desktop application, the kernel (Matérn-5/2 or Matérn-3/2) and the acquisition function (EI or PI) are each one setting (Section S5.1), and the GP also provides the reference and residual models of the transfer prior (Section S5.2).

## Section S5. Transfer Learning

The six validation campaigns (Sections S6-S7) were run with the ActMOF desktop application (MOF Bayesian Optimization Student App, version 1.0.9). Its settings are those of the benchmark (Section S3.1) unless stated below.

Bayesian optimization of reaction conditions is established in chemical synthesis[10], and starting a new optimization from the data of earlier, related ones is known as transfer learning or multi-task Bayesian optimization|11]. The form used here, a rescaled reference model plus a GP for what the reference does not explain (Section S5.2), follows the model of Kennedy and O'Hagan for combining a cheap, approximate model with a few expensive data|12|: the expensive response is written as a constant times the cheap one plus a GP for the difference. Here the MOF-321 model takes the place of the cheap model and the few experiments on the new MOF that of the expensive data. The offset a and the restriction of the slope b described below are our additions for the case of very few experiments.

## S5.1. Desktop Application

The search space is the five encoded variables of Section S2.3 (metal\_amount, modulator, add\_solvent, reaction\_time, reaction\_temperature). The user chooses a Matérn-5/2 or Matérn-3/2 kernel and EI or PI; all six campaigns used Matérn-5/2 with EI, 15,000 random

candidates and the transfer prior with its default schedule. Each iteration proposes three experiments. The user enters the measured PXRD intensity and FWHM of the diagnostic reflection, and the program computes

$$
q = \mathrm { r o u n d } \left( \frac { I } { \mathrm { F W H M } } \right) ,\tag{S23}
$$

with $q = 0$ when no reflection is found (entered as $\mathrm { F W H M } = 3 0 )$ or $I \leq 0 ; q$ can also be entered directly.

Surrogate. The program uses its own compact GP implementation (NumPy, Cholesky factorization) instead of scikit-learn, so that it can be distributed as a single executable. Inputs are scaled to [0, 1] and targets are $y = \log ( 1 + q )$ , standardized before fitting. Unlike the benchmark GPs, the kernel uses one length scale l for all five variables (an isotropic Matérn kernel). In a campaign the model is fitted to only 3-12 results of the new MOF; with so few data, five separate length scales cannot be estimated reliably and tend to run to the bounds of the fit, whereas a single length scale is stable. l and the noise level are set by trying a fixed list of values and keeping the pair under which the measured data are most probable (highest marginal likelihood); this is repeated after every batch.

Candidates and batch. As in the benchmark, each iteration scores 15,000 untried conditions and picks three of them. Most candidates are drawn at random from the whole grid. Once results exist, up to 2,400 of them are instead taken from the neighborhood of the eight best experiments so far (300 around each, with every variable changed by at most three grid steps), so that the surroundings of the good results are always among the candidates. The three conditions are then chosen as in the benchmark, by EI $( \xi = 0 . 0 1 )$ with the distance bonus of Eq. S14 (λ = 0.03).

## S5.2. Reference and Residual GP

Instead of adding the MOF-321 data to the new project as pseudo-observations, the program uses them through a calibrated transfer prior.

Reference GP. A GP with the kernel above is fitted to the 95 measured MOF-321 syntheses of Section S2.2, built into the program (entries 26 and 31 of Table S4 are the same experiment and are counted once), with $y = \log ( 1 + q )$ . All 95 experiments are used, including the 28 without product, so the reference GP learns both where MOF-321 crystallizes and where it does not. Its predictive mean $\mu _ { \mathrm { r e f } } ( \mathbf { x } )$ and s.d. $\sigma _ { \mathrm { r e f } } ( \mathbf { x } )$ encode the MOF-321 synthesis trends that also anchor the landscape of Section S2.6.

Calibration. Let n be the number of experiments completed so far on the new MOF, with conditions $\mathbf { x } _ { i }$ and results $y _ { i } = \log ( 1 + q _ { i } ) , i = 1 , \ldots , n$ . The reference prediction is mapped onto the response of the new MOF by a straight line,

$$
y _ { i } \approx a + b \mu _ { \mathrm { r e f } } ( { \bf x } _ { i } ) .\tag{S24}
$$

Before any result $( n = 0 ) , a = 0$ and $b = 1$ : the reference is used as it is. With one result $( n = 1 )$ , only the offset is set, $a = y _ { 1 } - \mu _ { \mathrm { r e f } } ( { \bf x } _ { 1 } )$ $b = 1$ . With $n \geq 2$

$$
( a , b ) = \underset { a , b } { \mathrm { a r g } \mathrm { m i n } } \ \Big [ \sum _ { i = 1 } ^ { n } \big ( y _ { i } - a - b \mu _ { \mathrm { r e f } } ( \mathbf { x } _ { i } ) \big ) ^ { 2 } + \rho ( b - 1 ) ^ { 2 } \Big ] , \qquad \rho = \mathrm { m a x } ( 0 . 0 5 , 1 / n ) .\tag{S25}
$$

The first term is an ordinary least-squares fit of the line; the second penalizes slopes far from 1 and so keeps b near 1 while there are few results. In addition, b is limited to [-2,3]. The offset a absorbs differences in the intensity scale (instrument, reflection, framework), and b how strongly the MOF-321 trends carry over: $b \approx 1$ keeps them, $b \approx 0$ switches the prior off, and $b < 0$ inverts it.

Residual GP. The residuals

$$
r _ { i } = y _ { i } - \left( a + b \mu _ { \mathrm { r e f } } ( \mathbf { x } _ { i } ) \right)\tag{S26}
$$

are fitted by a second GP with the same kernel once $n \geq 3$ and the residuals are not all equal. The predictive model of the new MOF is

$$
\begin{array} { r l r l r } & { \mu ( { \bf x } ) = a + b \mu _ { \mathrm { r e f } } ( { \bf x } ) + \mu _ { r } ( { \bf x } ) , } & & { \sigma ( { \bf x } ) = \sqrt { \sigma _ { r } ( { \bf x } ) ^ { 2 } + \left( 0 . 3 5 \left| b \right| \sigma _ { \mathrm { r e f } } ( { \bf x } ) \right) ^ { 2 } + \left( 0 . 2 5 s _ { r } \right) ^ { 2 } } , } \end{array}\tag{S27}
$$

where $\mu _ { r }$ and $\sigma _ { r }$ are the residual-GP mean and $\mathrm { s . d }$ and $s _ { r }$ is the s.d. of the residuals (0.35 before any data). The factors 0.35 and 0.25 are fixed settings of the program; they were chosen by us and are not fitted to data. The factor 0.35 reduces the reference uncertainty so that, once results of the new MOF exist, the residual model is the main source of uncertainty; the last term, a quarter of the scatter of the residuals, keeps σ from falling to zero next to tested conditions. EI or PI is then computed from $\mu$ and $\sigma ,$ with the incumbent $y ^ { + }$ equal to the best log(1 + q) of the project; before the first result, $y ^ { + }$ is set to the 75th percentile of $\mu$ over the candidates, so that the first batch balances high predicted q and uncertainty.

## S5.3. Calibration

Duration of the prior. The calibrated prior is used for the first four batches (the program's default, $M = 4 )$ , i.e. for all 12 experiments of each campaign reported here.

Initial batch. Before any result exists, the proposal depends only on the reference GP and the fixed random seed of the program, so the first batch is the same for every new MOF: R00001 = (18, 7, 17, 10, 21), R00002 = (23, 6, 15, 10, 20) and R00003 = (20, 7, 19, 12, 29), i.e. 180/70/170 µL, 50 min, 105 °C; 230/60/150 µL, 50 min, $1 0 0 ~ ^ { \circ } \mathrm { C } ;$ and 200/70/190 µL, 60 min, ${ 1 4 5 } ^ { \mathrm { ~ \circ } } \mathrm { C \ ( A l C l _ { 3 } \cdot 6 H _ { 2 } O / N a O H / D I }$ water, at the 0.1 mmol scale of Section S6). These three conditions lie inside the MOF-321 top-1% tier (Table S11): the campaigns start from the MOF-321 prior, not from random conditions.

Recalibration. After each batch, $^ { a , }$ b and the residual GP are refitted to all results of the MOF; the values below are those recorded by the program. For MIL-53-muc, MIL-160 and MTV-MIL-160, b stays between 0.55 and 1.5: the MOF-321 trends carry over and the residual GP adjusts them locally. For MTV-MOF-303, b falls to 0.05 and 0.20 in rounds 2 and 3, i.e. the prior is largely switched off and the search relies on the residual GP. For CAU-10 and CAU-23, the six results after round 1 are anti-correlated with $\mu _ { \mathrm { r e f } }$ and b becomes negative (-2.0, the lower bound, and -0.38). The prior is then inverted: it predicts the highest q where MOF-321 gives none, and the round-2 batches of both MOFs (the same three conditions, at 60–65 °C) gave no or almost no product (Tables S20 and S21). With these results, b returns to 0.48 and 0.41 in round $^ { 3 , }$ and the next CAU-10 batch contains its best sample (R00012). The calibration thus corrects itself within one batch, at the cost of one poor batch.

## S5.4. Six Campaigns

Figure S24 shows every measured q and the best-so-far of the six campaigns, and Table S19 compares the best of the three starting experiments with the best of the 12 experiments of each campaign. The best q improved 1.23–5.40-fold for five MOFs; for CAU-23 the first starting condition remained the best. All best samples were found in the calibrated-prior phase (rounds 1-3), except that of CAU-23 (round 0). Of the 72 experiments within the first 12 of each MOF, 67 gave a crystalline product. The five without product are R00004 and R00007–R00009 of CAU-10 and R00009 of CAU-23; the other two round-2 experiments of CAU-23 gave only $q = 3 2 1$ and 117.

![](images/20a9ad25cbcdfd0df3c0ac24c774ba5a0487d484b172e861c12a8ce29f65d070.jpg)  
reference prior only (round 0) calibrated prior + residual GP (rounds 1-3) best so far  
Figure S24: Transfer-prior campaigns of the six validation MOFs. Points, measured q of each experiment in synthesis order, first 12 experiments, colored by the model that proposed it: reference GP only (gray, round 0) and calibrated prior with residual GP (blue, rounds 1–3). Line, best so far. Light vertical lines separate the rounds. Titles, best of the first 12 experiments divided by the best of R00001-R00003

Table S19: Best q of the three starting experiments and of the first 12 experiments, their ratio, and the best recipe $\mathrm { ( A l C l _ { 3 } \cdot 6 H _ { 2 } O }$ and NaOH stock solutions and added DI water in $\mu \mathrm { L }$ with 0.1 mmol linker; hold time; temperature; Section S6). Concentrations: amounts of $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and NaOH from the 100 mg mL−1 stock solutions (M in Table S1) divided by the total volume of the three liquids. $\mathrm { A l ^ { 3 + } { : } L { : } O H ^ { - } }$ , molar ratio per linker (L, 0.1 mmol in total, also for the two multivariate frameworks).
<table><tr><td></td><td colspan="2">Best q (ID)</td><td></td><td colspan="3">Best recipe</td><td colspan="4">Composition</td></tr><tr><td>MOF</td><td>Start</td><td>First 12 Ratio</td><td></td><td> $\mathrm { A l / N a O H / H _ { 2 } O }$  (µL)</td><td>t (min)</td><td>T (°C)</td><td> $[ \mathrm { A l } ^ { 3 + } ]$  (mM)</td><td>[OH] (mM)</td><td> $\mathrm { A l } ^ { 3 + } { \cdot } \mathrm { L } { : } \mathrm { O H }$  (mol)</td><td> $\mathrm { O H ^ { - } } ;$   $\mathrm { A l ^ { 3 + } }$ </td></tr><tr><td>CAU-10</td><td>5,978 (R3)</td><td>32,259 (R12)</td><td>5.40</td><td>200/70/180</td><td></td><td>60 140</td><td>184</td><td>389</td><td>0.83:1:1.75</td><td>2.11</td></tr><tr><td>CAU-23</td><td>22,254 (R1)</td><td>22,254 (R1)</td><td>1.00</td><td>180/70/170</td><td>50</td><td>105</td><td>178</td><td>417</td><td>0.75:1:1.75</td><td>2.35</td></tr><tr><td>MIL-53-muc</td><td>32,523 (R1)</td><td>58,064 (R8)</td><td>1.79</td><td>160/60/160</td><td>50</td><td>105</td><td>174</td><td>395</td><td>0.66:1:1.50</td><td>2.26</td></tr><tr><td>MIL-160</td><td>25,364 (R2)</td><td>44,395 (R7)</td><td>1.75</td><td>190/60/150</td><td>55</td><td>100</td><td>197</td><td>375</td><td>0.79:1:1.50</td><td>1.91</td></tr><tr><td>MTV-MIL-160</td><td>17,258 (R3)</td><td>30,990 (R7)</td><td>1.80</td><td>200/70/180</td><td>60</td><td>140</td><td>184</td><td>389</td><td>0.83:1:1.75</td><td>2.11</td></tr><tr><td>MTV-MOF-303 43,452 (R1)</td><td></td><td>53,648 (R9)</td><td>1.23</td><td>190/60/160</td><td></td><td>50 110</td><td>192</td><td>366</td><td>0.79:1:1.50</td><td>1.91</td></tr></table>

Arithmetic mean / geometric mean / median 2.16 / 1.83 / 1.77

Transferable and framework-specific features of the prior. The six campaigns suggest that the MOF-321 prior transfers the general crystallization rules of these Al-carboxylate syntheses, and that the optimum of each framework still has to be found by the search. All six best recipes (Table S19) lie in or next to the productive window of MOF-321 (Section S2.6). In equivalents relative to the linker, the top-0.1% MOF-321 anchors use 0.46–0.75 equiv $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and 1.25–1.75 equiv NaOH (55–60 min, $1 1 5 { - } 1 4 5 \ ^ { \circ } \mathrm { C } )$ , and the six best recipes $0 . 6 6 \mathrm { - } 0 . 8 3$ equiv $\mathrm { A l C l _ { 3 } \cdot 6 H _ { 2 } O }$ and 1.5–1.75 equiv NaOH (50–60 min, $1 0 0 { - } 1 4 0 \ ^ { \circ } \mathrm { C } )$ , i.e. little metal, little base, long hold times and high temperature. Within this window, however, the optima differ: CAU-10 and MTV-MIL-160 are best at $1 4 0 ~ ^ { \circ } \mathrm { C }$ and 60 min, CAU-23, MIL-53-muc and MIL-160 at 100– $1 0 5 ~ ^ { \circ } \mathrm { C }$ and 50–55 min, and MTV-MOF-303 at 110 °C and 50 min. None of the best conditions is one of the three starting conditions, except for CAU-23, and none coincides with a MOF-321 optimum: they lie inside the MOF-321 top-1% tier but 1-2 grid steps from its top-0.1% tier (Section S2.7). The calibration makes the same point: the slope b varies from framework to framework and between rounds (Section S5.3), so the MOF-321 trends are rescaled, partly switched off (MTV-MOF-303) or temporarily inverted (CAU-10, CAU-23) as results come in. and five of the six best samples were proposed after at least one recalibration.

## Section S6. Synthesis of the Validation MOFs

General procedure. All six frameworks were made by the same procedure; only the linker differs. The linker was weighed into a G10 vial, followed by DI water (the volume of the recipe), a small magnetic stir bar and the NaOH stock solution (100 mg mL-1 in DI water). The mixture was sonicated until the linker had dissolved. The $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ stock solution $( 1 0 0 \ \mathrm { m g } \mathrm { m L } ^ { - 1 }$ in DI water) was then added, and the mixture was heated in the microwave reactor and worked up as in Section S1. The volumes of the two stock solutions and of DI water, the hold time and the temperature of every synthesis are those proposed by the active-learning campaign; they are listed for each MOF, together with the encoded variables $\mathbf { x } ,$ in Tables S20–S25, and the best recipe of each framework in Table S19. The validation syntheses were run at one tenth of the scale of Table S5: 0.1 mmol of linker (for the two multivariate frameworks, 0.05 mmol of each of the two linkers, 1:1), with one step of x1, x2 or $x _ { 3 }$ equal to $1 0 ~ \mu \mathrm { L }$ instead of 0.1 mL. Molar masses are given in Table S1.

(i) CAU-10. CAU-10 was prepared from $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and isophthalic acid by the general procedure, with 0.1 mmol (16.6 mg) isophthalic acid; the conditions of the 12 syntheses are in Table S20.

(ii) CAU-23. CAU-23 was prepared from $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and thiophene-2,5-dicarboxylic acid by the general procedure, with 0.1 mmol (17.2 mg) thiophene-2,5-dicarboxylic acid; the conditions of the 12 syntheses are in Table S21.

(iii) MIL-53-muc. MIL-53-muc was prepared from $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and trans,trans-muconic acid by the general procedure, with 0.1 mmol (14.2 mg) trans,trans-muconic acid; the conditions of the 12 syntheses are in Table S22.

(iv) MIL-160. MIL-160 was prepared from $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and furan-2,5-dicarboxylic acid by the general procedure, with 0.1 mmol (15.6 mg) furan-2,5-dicarboxylic acid; the conditions of the 12 syntheses are in Table S23

(v) MTV-MIL-160. MTV-MIL-160 was prepared from $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and two linkers, furan-2,5-dicarboxylic acid and isophthalic acid by the general procedure, with 0.05 mmol (7.8 mg)

furan-2,5-dicarboxylic acid and 0.05 mmol (8.3 mg) isophthalic acid; the conditions of the 12 syntheses are in Table S24. Both linkers were weighed into the vial before the DI water and the NaOH solution were added (feed ratio 1:1).

(vi) MTV-MOF-303. MTV-MOF-303 was prepared from $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and two linkers, thiophene-2,5-dicarboxylic acid and 1H-pyrazole-3,5-dicarboxylic acid hydrate by the general procedure, with 0.05 mmol (8.6 mg) thiophene-2,5-dicarboxylic acid and 0.05 mmol (8.7 mg) 1H-pyrazole-3,5-dicarboxylic acid hydrate; the conditions of the 12 syntheses are in Table S25 Both linkers were weighed into the vial before the DI water and the $\mathrm { N a O H }$ solution were added (feed ratio 1:1).

## Section S7. PXRD and Campaign Trajectories

Instrument and data processing. PXRD patterns were collected on a Rigaku MiniFlex diffractometer (Cu Kα) from 3 to 30° 2θ with a step of $0 . 0 1 ^ { \circ }$ at $8 ^ { \circ } \operatorname* { m i n } ^ { - 1 }$ . The washed powders were loaded on silicon sample holders and pressed with a spatula as described in Section S2.1. The patterns were processed with a Python script (GitHub repository). In each pattern the diagnostic reflection of the target phase (Table S2) was selected, its maximum intensity was taken as $I ,$ and the FWHM was taken as the distance between the two 2θ positions at half of this maximum. I and FWHM were entered into the active-learning program, which computes q = I/FWHM (Eq. S23); a pattern without a reflection was entered as $I = 0$ , FWHM = 30 $( q = 0 )$

The figures and tables below show, for each MOF, the 12 syntheses of its campaign in synthesis order (R00001–R00012), one MOF per page. In each figure, one row of three patterns is one round: round 0, conditions proposed by the reference prior only; rounds 1-3, by the calibrated prior with residual GP (Section S5.3). The tables give the encoded variables x = $( x _ { 1 } , \ldots , x _ { 5 } )$ of Table S5, the recipe in laboratory units (stock solutions of $\mathrm { { A l C l } _ { 3 } \cdot 6 H _ { 2 } O }$ and NaOH, 100 $\mathrm { m g { m } L ^ { - 1 } }$ in DI water, and added DI water in $\mu \mathrm { L } ;$ Section S6) and the measured intensity, FWHM and $q = I / \mathrm { F W H M }$ of the diagnostic reflection; n.d., no reflection detected $( q = 0 )$ ; bold, best sample. The first three experiments are the same three conditions for every MOF, proposed by the MOF-321 reference prior before any result of that MOF was available. The best q of each campaign and the improvement over the starting experiments are summarized in Table S19 and Figure S24.

(i) CAU-10. Recipes and results in Table S20, patterns in Figure S25.  
Table S20: Recipes and PXRD results of CAU-10.
<table><tr><td colspan="3"></td><td colspan="3">Stock solution / DI water (µL)</td><td colspan="5"></td></tr><tr><td>Exp.</td><td>Round</td><td>Encoded x</td><td> $\mathrm { A l C l _ { 3 } } \cdot 6 \mathrm { H _ { 2 } O }$ </td><td>NaOH</td><td> $\mathrm { D I \ H _ { 2 } O }$ </td><td>Time (min)</td><td> $T ~ ( ^ { \circ } \mathrm { C } )$ </td><td></td><td>I FWHM (°)</td><td>q</td></tr><tr><td>R00001</td><td>0</td><td>(18, 7, 17, 10, 21)</td><td>180</td><td>70</td><td>170</td><td>50</td><td>105</td><td>2,140</td><td>1.31</td><td>1,634</td></tr><tr><td>R00002</td><td>0</td><td>(23, 6, 15, 10, 20)</td><td>230</td><td>60</td><td>150</td><td>50</td><td>100</td><td>2,148</td><td>1.28</td><td>1,678</td></tr><tr><td>R00003</td><td>0</td><td>(20, 7, 19, 12, 29)</td><td>200</td><td>70</td><td>190</td><td>60</td><td>145</td><td>1,614</td><td>0.27</td><td>5,978</td></tr><tr><td>R00004</td><td>1</td><td>(22, 7, 16, 12, 29)</td><td>220</td><td>70</td><td>160</td><td>60</td><td>145</td><td>0</td><td>n.d.</td><td>0</td></tr><tr><td>R00005</td><td>1</td><td>(19, 7, 17, 12, 29)</td><td>190</td><td>70</td><td>170</td><td>60</td><td>145</td><td>2,639</td><td>0.51</td><td>5,175</td></tr><tr><td>R00006</td><td>1</td><td>(21, 7, 18, 12, 29)</td><td>210</td><td>70</td><td>180</td><td>60</td><td>145</td><td>4,277</td><td>0.39</td><td>10,967</td></tr><tr><td>R00007</td><td>2</td><td>(21, 7, 15, 11, 12)</td><td>210</td><td>70</td><td>150</td><td>55</td><td>60</td><td>0</td><td>n.d.</td><td>0</td></tr><tr><td>R00008</td><td>2</td><td>(22, 7, 12, 10, 13)</td><td>220</td><td>70</td><td>120</td><td>50</td><td>65</td><td>0</td><td>n.d.</td><td>0</td></tr><tr><td>R00009</td><td>2</td><td>(25, 9, 14, 10, 13)</td><td>250</td><td>90</td><td>140</td><td>50</td><td>65</td><td>0</td><td>n.d.</td><td>0</td></tr><tr><td>R00010</td><td>3</td><td>(19, 7, 18, 12, 28)</td><td>190</td><td>70</td><td>180</td><td>60</td><td>140</td><td>3,666</td><td>0.64</td><td>5,728</td></tr><tr><td>R00011</td><td>3</td><td>(17, 7, 18, 12, 28)</td><td>170</td><td>70</td><td>180</td><td>60</td><td>140</td><td>5,176</td><td>0.48</td><td>10,783</td></tr><tr><td>R00012</td><td>3</td><td> $( \mathbf { 2 0 } , 7 , \mathbf { 1 8 } , \mathbf { 1 2 } , \mathbf { 2 8 } )$ </td><td>200</td><td>70</td><td>180</td><td>60</td><td>140</td><td>7,097</td><td>0.22</td><td>32,259</td></tr></table>

![](images/f8afe20e4611cc07452f5301a9cff9be9525c6790b609b97025d7cdb12a84ba3.jpg)  
Figure S25: PXRD patterns of CAU-10, R00001–R00012 (Table S20).

(ii) CAU-23. Recipes and results in Table S21, patterns in Figure S26.  
Table S21: Recipes and PXRD results of CAU-23.
<table><tr><td></td><td></td><td></td><td colspan="3">Stock solution / DI water</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Exp.</td><td>Round</td><td>Encoded x</td><td> $\mathrm { A l C l _ { 3 } } \cdot 6 \mathrm { H _ { 2 } O }$ </td><td>NaOH</td><td> $\mathrm { D I \ H _ { 2 } O }$ </td><td>Time (min)</td><td> $T ~ ( ^ { \circ } \mathrm { C } )$ </td><td></td><td>I FWHM (°)</td><td></td></tr><tr><td>R00001</td><td>0</td><td> $( \mathbf { 1 8 } , \mathbf { 7 } , \mathbf { 1 7 } , \mathbf { 1 0 } , \mathbf { 2 1 } )$ </td><td>180</td><td>70</td><td>170</td><td>50</td><td>105</td><td>8,430</td><td>0.379</td><td>22,254</td></tr><tr><td>R00002</td><td>0</td><td>(23, 6, 15, 10, 20)</td><td>230</td><td>60</td><td>150</td><td>50</td><td>100</td><td>3,551</td><td>0.607</td><td>5,846</td></tr><tr><td>R00003†</td><td>0</td><td>(20, 7, 17, 12, 29)</td><td>200</td><td>70</td><td>170</td><td>60</td><td>145</td><td>4,651</td><td>0.603</td><td>7,711</td></tr><tr><td>R00004</td><td>1</td><td>(16, 7, 18, 10, 21)</td><td>160</td><td>70</td><td>180</td><td>50</td><td>105</td><td>6,533</td><td>0.301</td><td>21,668</td></tr><tr><td>R00005</td><td>1</td><td>(19, 7, 19, 10, 21)</td><td>190</td><td>70</td><td>190</td><td>50</td><td>105</td><td>3,064</td><td>0.329</td><td>9,313</td></tr><tr><td>R00006</td><td>1</td><td>(18, 7, 19, 10, 22)</td><td>180</td><td>70</td><td>190</td><td>50</td><td>110</td><td>2,918</td><td>0.304</td><td>9,583</td></tr><tr><td>R00007</td><td>2</td><td>(21, 7, 15, 11, 12)</td><td>210</td><td>70</td><td>150</td><td>55</td><td>60</td><td>849</td><td>2.64</td><td>321</td></tr><tr><td>R00008</td><td>2</td><td>(22, 7, 12, 10, 13)</td><td>220</td><td>70</td><td>120</td><td>50</td><td>65</td><td>883</td><td>7.57</td><td>117</td></tr><tr><td>R00009</td><td>2</td><td>(25, 9, 14, 10, 13)</td><td>250</td><td>90</td><td>140</td><td>50</td><td>65</td><td>0</td><td>n.d.</td><td>0</td></tr><tr><td>R00010</td><td>3</td><td>(15, 6, 16, 10, 22)</td><td>150</td><td>60</td><td>160</td><td>50</td><td>110</td><td>3,540</td><td>0.401</td><td>8,830</td></tr><tr><td>R00011</td><td>3</td><td>(14, 7, 16, 10, 21)</td><td>140</td><td>70</td><td>160</td><td>50</td><td>105</td><td>5,283</td><td>0.317</td><td>16,671</td></tr><tr><td>R00012</td><td>3</td><td>(17, 6, 17, 11, 20)</td><td>170</td><td>60</td><td>170</td><td>55</td><td>100</td><td>4,145</td><td>0.598</td><td>6,934</td></tr></table>

†Synthesized with 170 $\mu \mathrm { L }$ instead of the proposed 190 $\mu \mathrm { L }$ DI water (pipetting error); the condition actually used is listed and was the one given to the model.

![](images/da7202812c8a4d398f15a3e7a614fdf06f0e6723528b003522fd1fcf6c27fef2.jpg)  
Figure S26: PXRD patterns of CAU-23, R00001–R00012 (Table S21).

(iii) MIL-53-muc. Recipes and results in Table S22, patterns in Figure S27.

Table S22: Recipes and PXRD results of MIL-53-muc.
<table><tr><td></td><td></td><td></td><td colspan="3">Stock solution / DI water (µL)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Exp.</td><td>Round</td><td>Encoded x</td><td>AlCl3  $\cdot 6 \mathrm { H _ { 2 } O }$ </td><td>NaOH</td><td>DI  $\mathrm { H _ { 2 } O }$ </td><td>Time (min)</td><td>T (°C)</td><td></td><td>I FWHM (°)</td><td>q</td></tr><tr><td>R00001</td><td>0</td><td>(18, 7, 17, 10, 21)</td><td>180</td><td>70</td><td>170</td><td>50</td><td>105</td><td>10,339</td><td>0.318</td><td>32,523</td></tr><tr><td>R00002</td><td>0</td><td>(23, 6, 15, 10, 20)</td><td>230</td><td>60</td><td>150</td><td>50</td><td>100</td><td>4,600</td><td>0.315</td><td>14,599</td></tr><tr><td>R00003</td><td>0</td><td>(20, 7, 19, 12, 29)</td><td>200</td><td>70</td><td>190</td><td>60</td><td>145</td><td>3,163</td><td>0.651</td><td>4,860</td></tr><tr><td>R00004</td><td>1</td><td>(19, 6, 19, 10, 21)</td><td>190</td><td>60</td><td>190</td><td>50</td><td>105</td><td>3,784</td><td>0.382</td><td>9,895</td></tr><tr><td>R00005</td><td>1</td><td>(18, 6, 15, 10, 22)</td><td>180</td><td>60</td><td>150</td><td>50</td><td>110</td><td>5,953</td><td>0.3</td><td>19,857</td></tr><tr><td>R00006</td><td>1</td><td>(16, 6, 16, 10, 20)</td><td>160</td><td>60</td><td>160</td><td>50</td><td>100</td><td>7,402</td><td>0.343</td><td>21,580</td></tr><tr><td>R00007</td><td>2</td><td>(19, 6, 16, 10, 20)</td><td>190</td><td>60</td><td>160</td><td>50</td><td>100</td><td>5,981</td><td>0.369</td><td>16,209</td></tr><tr><td>R00008</td><td>2</td><td>(16, 6, 16, 10, 21)</td><td>160</td><td>60</td><td>160</td><td>50</td><td>105</td><td>18,209</td><td>0.314</td><td>58,064</td></tr><tr><td>R00009</td><td>2</td><td>(18, 6, 16, 11, 22)</td><td>180</td><td>60</td><td>160</td><td>55</td><td>110</td><td>8,206</td><td>0.451</td><td>18,207</td></tr><tr><td>R00010</td><td>3</td><td>(20, 6, 17, 11, 20)</td><td>200</td><td>60</td><td>170</td><td>55</td><td>100</td><td>9,091</td><td>0.314</td><td>28,989</td></tr><tr><td>R00011</td><td>3</td><td>(14, 6, 17, 10, 21)</td><td>140</td><td>60</td><td>170</td><td>50</td><td>105</td><td>7,474</td><td>0.38</td><td>19,679</td></tr><tr><td>R00012</td><td>3</td><td>(22, 6, 16, 11, 21)</td><td>220</td><td>60</td><td>160</td><td>55</td><td>105</td><td>14,736</td><td>0.329</td><td>44,818</td></tr></table>

![](images/abbeffab856d8480985b3ba48643ddfb06b0d31cf4667da7af18c193ce03cf7c.jpg)  
Figure S27: PXRD patterns of MIL-53-muc, R00001–R00012 (Table S22).

(iv) MIL-160. Recipes and results in Table S23, patterns in Figure S28.

Table S23: Recipes and PXRD results of MIL-160.
<table><tr><td></td><td></td><td></td><td colspan="3">Stock solution / DI water</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Exp.</td><td>Round</td><td>Encoded x</td><td> $\mathrm { A l C l _ { 3 } } \cdot 6 \mathrm { H _ { 2 } O }$ </td><td>NaOH</td><td> $\mathrm { D I \ H _ { 2 } O }$ </td><td>Time (min)</td><td> $T ~ ( ^ { \circ } \mathrm { C } )$ </td><td></td><td>I FWHM (°)</td><td>q</td></tr><tr><td>R00001</td><td>0</td><td>(18, 7, 17, 10, 21)</td><td>180</td><td>70</td><td>170</td><td>50</td><td>105</td><td>2,326</td><td>0.32</td><td>7,269</td></tr><tr><td>R00002</td><td>0</td><td>(23, 6, 15, 10, 20)</td><td>230</td><td>60</td><td>150</td><td>50</td><td>100</td><td>5,580</td><td>0.22</td><td>25,364</td></tr><tr><td>R00003</td><td>0</td><td>(20, 7, 19, 12, 29)</td><td>200</td><td>70</td><td>190</td><td>60</td><td>145</td><td>2,804</td><td>0.25</td><td>11,216</td></tr><tr><td>R00004</td><td>1</td><td>(21, 6, 16, 10, 20)</td><td>210</td><td>60</td><td>160</td><td>50</td><td>100</td><td>5,207</td><td>0.27</td><td>19,285</td></tr><tr><td>R00005</td><td>1</td><td>(20, 6, 16, 10, 21)</td><td>200</td><td>60</td><td>160</td><td>50</td><td>105</td><td>5,491</td><td>0.25</td><td>21,964</td></tr><tr><td>R00006</td><td>1</td><td>(19, 6, 17, 11, 21)</td><td>190</td><td>60</td><td>170</td><td>55</td><td>105</td><td>5,811</td><td>0.27</td><td>21,522</td></tr><tr><td>R00007</td><td>2</td><td>(19, 6, 15, 11, 20)</td><td>190</td><td>60</td><td>150</td><td>55</td><td>100</td><td>9,767</td><td>0.22</td><td>44,395</td></tr><tr><td>R00008</td><td>2</td><td>(20, 7, 16, 12, 28)</td><td>200</td><td>70</td><td>160</td><td>60</td><td>140</td><td>4,813</td><td>0.26</td><td>18,512</td></tr><tr><td>R00009</td><td>2</td><td>(20, 6, 14, 10, 20)</td><td>200</td><td>60</td><td>140</td><td>50</td><td>100</td><td>5,199</td><td>0.29</td><td>17,928</td></tr><tr><td>R00010</td><td>3</td><td>(20, 6, 14, 11, 20)</td><td>200</td><td>60</td><td>140</td><td>55</td><td>100</td><td>4,459</td><td>0.234</td><td>19,072</td></tr><tr><td>R00011</td><td>3</td><td>(20, 6, 17, 11, 20)</td><td>200</td><td>60</td><td>170</td><td>55</td><td>100</td><td>4,579</td><td>0.23</td><td>19,874</td></tr><tr><td>R00012</td><td>3</td><td>(19, 6, 14, 11, 20)</td><td>190</td><td>60</td><td>140</td><td>55</td><td>100</td><td>5,043</td><td>0.319</td><td>15,799</td></tr></table>

![](images/db40e007bb4385152561cdb320f5948547d77aac9e1df3da546041481c302ddc.jpg)  
Figure S28: PXRD patterns of MIL-160, R00001–R00012 (Table S23).

(v) MTV-MIL-160. Recipes and results in Table S24, patterns in Figure S29.  
Table S24: Recipes and PXRD results of MTV-MIL-160.
<table><tr><td></td><td></td><td></td><td>Stock solution / DI water</td><td></td><td> $( \mu \mathrm { L } )$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Exp.</td><td>Round</td><td>Encoded x</td><td> $\mathrm { A l C l _ { 3 } } \cdot 6 \mathrm { H _ { 2 } O }$ </td><td>NaOH</td><td> $\mathrm { D I \ H _ { 2 } O }$ </td><td>Time (min)</td><td> $T ~ ( ^ { \circ } \mathrm { C } )$ </td><td></td><td>I FWHM (°)</td><td>q</td></tr><tr><td>R00001</td><td>0</td><td>(18, 7, 17, 10, 21)</td><td>180</td><td>70</td><td>170</td><td>50</td><td>105</td><td>3,043</td><td>0.306</td><td>9,951</td></tr><tr><td>R00002</td><td>0</td><td>(23, 6, 15, 10, 20)</td><td>230</td><td>60</td><td>150</td><td>50</td><td>100</td><td>1,386</td><td>0.383</td><td>3,615</td></tr><tr><td>R00003</td><td>0</td><td>(20, 7, 19, 12, 29)</td><td>200</td><td>70</td><td>190</td><td>60</td><td>145</td><td>6,142</td><td>0.356</td><td>17,258</td></tr><tr><td>R00004</td><td>1</td><td>(21, 7, 18, 12, 29)</td><td>210</td><td>70</td><td>180</td><td>60</td><td>145</td><td>6,892</td><td>0.233</td><td>29,554</td></tr><tr><td>R00005</td><td>1</td><td>(19, 7, 17, 12, 29)</td><td>190</td><td>70</td><td>170</td><td>60</td><td>145</td><td>5,147</td><td>0.246</td><td>20,948</td></tr><tr><td>R00006</td><td>1</td><td>(22, 7, 16, 12, 29)</td><td>220</td><td>70</td><td>160</td><td>60</td><td>145</td><td>4,006</td><td>0.25</td><td>16,030</td></tr><tr><td>R00007</td><td>2</td><td>(20, 7, 18, 12, 28)</td><td>200</td><td>70</td><td>180</td><td>60</td><td>140</td><td>7,608</td><td>0.245</td><td>30,990</td></tr><tr><td>R00008</td><td>2</td><td>(20, 7, 17, 12, 28)</td><td>200</td><td>70</td><td>170</td><td>60</td><td>140</td><td>4,893</td><td>0.225</td><td>21,737</td></tr><tr><td>R00009</td><td>2</td><td>(22, 7, 17, 12, 29)</td><td>220</td><td>70</td><td>170</td><td>60</td><td>145</td><td>4,202</td><td>0.253</td><td>16,589</td></tr><tr><td>R00010</td><td>3</td><td>(20, 6, 18, 10, 21)</td><td>200</td><td>60</td><td>180</td><td>50</td><td>105</td><td>5,902</td><td>0.248</td><td>23,779</td></tr><tr><td>R00011</td><td>3</td><td>(20, 7, 18, 12, 29)</td><td>200</td><td>70</td><td>180</td><td>60</td><td>145</td><td>6,129</td><td>0.211</td><td>29,020</td></tr><tr><td>R00012</td><td>3</td><td>(22, 7, 18, 12, 29)</td><td>220</td><td>70</td><td>180</td><td>60</td><td>145</td><td>5,466</td><td>0.24</td><td>22,813</td></tr></table>

![](images/130750120ba0a84267763a1edd848b3270ea648bf4c7b9ef0dae6390c53d7c99.jpg)

![](images/b65625725ddabdf9cafa067dcf0aea171e9ac044c248dd301e0cb52ddfa1701b.jpg)

![](images/49efd3ad3f26efa9dcfe75438503cf237ba291fb3dcd0627f281cc73903487da.jpg)

![](images/926a370176f336fd32c09c19654e5c4f7bbc6f7a23fe1cfece3101a52308998d.jpg)

![](images/6a3469143666c1662612e5c588b8a9a5aec1252c997609978e5b68cd7c87a1f2.jpg)

![](images/8246d1a70cc4255e9d5a576d38677f74c2be222b464efb5f82d4f57a6827fffe.jpg)

![](images/45a145059a4c22b002234f99c127205db38ce9a133a126cbcae1159b421fef9f.jpg)

![](images/266f19cdffe6a49477536878340ca7afbdba8aea40862b5460777624c5c3c8ac.jpg)

![](images/128d4f7cb44d31b9f771b6578a20a2a517f200c6e773f17cd5df80c27bdfe9cf.jpg)

![](images/cc134331fd62956d386f8895a2ab1abd41d1408ec91c285032bb936f25e5b282.jpg)

![](images/d885eef8dc5e4eae8607c4a128bf76a0f2cdb4e13f5107567ca793064fb1aa7b.jpg)

![](images/9be9c715b2c146af357b50c806728efae4551c7d51d8bbc42f613070f3a3df57.jpg)  
Figure S29: PXRD patterns of MTV-MIL-160, R00001–R00012 (Table S24).

![](images/07d9e95d931d0c21b5847257fd7b0a3cfd44d405789edaf71673a3583b5da1bf.jpg)

(vi) MTV-MOF-303. Recipes and results in Table S25, patterns in Figure S30.

Table S25: Recipes and PXRD results of MTV-MOF-303.
<table><tr><td></td><td></td><td></td><td colspan="3">Stock solution /DI water</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Exp.</td><td>Round</td><td>Encoded x</td><td>AlCl3  $\cdot 6 \mathrm { H _ { 2 } O }$ </td><td>NaOH</td><td>DI  $\mathrm { H _ { 2 } O }$ </td><td>Time (min)</td><td>T (°C)</td><td></td><td>I FWHM (°)</td><td>q</td></tr><tr><td>R00001</td><td>0</td><td>(18, 7, 17, 10, 21)</td><td>180</td><td>70</td><td>170</td><td>50</td><td>105</td><td>12,488</td><td>0.287</td><td>43,452</td></tr><tr><td>R00002</td><td>0</td><td>(23, 6, 15, 10, 20)</td><td>230</td><td>60</td><td>150</td><td>50</td><td>100</td><td>6,362</td><td>0.426</td><td>14,948</td></tr><tr><td>R00003</td><td>0</td><td>(20, 7, 19, 12, 29)</td><td>200</td><td>70</td><td>190</td><td>60</td><td>145</td><td>5,557</td><td>0.256</td><td>21,707</td></tr><tr><td>R00004</td><td>1</td><td>(21, 7, 16, 12, 28)</td><td>210</td><td>70</td><td>160</td><td>60</td><td>140</td><td>3,371</td><td>0.281</td><td>12,018</td></tr><tr><td>R00005</td><td>1</td><td>(20, 7, 17, 12, 28)</td><td>200</td><td>70</td><td>170</td><td>60</td><td>140</td><td>6,815</td><td>0.287</td><td>23,779</td></tr><tr><td>R00006</td><td>1</td><td>(20, 6, 17, 10, 22)</td><td>200</td><td>60</td><td>170</td><td>50</td><td>110</td><td>15,110</td><td>0.283</td><td>53,392</td></tr><tr><td>R00007</td><td>2</td><td>(18, 6, 18, 10, 22)</td><td>180</td><td>60</td><td>180</td><td>50</td><td>110</td><td>9,073</td><td>0.233</td><td>38,873</td></tr><tr><td>R00008</td><td>2</td><td>(17, 6, 18, 10, 22)</td><td>170</td><td>60</td><td>180</td><td>50</td><td>110</td><td>7,324</td><td>0.25</td><td>29,296</td></tr><tr><td>R00009</td><td>2</td><td>(19, 6, 16, 10, 22)</td><td>190</td><td>60</td><td>160</td><td>50</td><td>110</td><td>13,353</td><td>0.249</td><td>53,648</td></tr><tr><td>R00010</td><td>3</td><td>(19, 6, 17, 11, 21)</td><td>190</td><td>60</td><td>170</td><td>55</td><td>105</td><td>4,301</td><td>0.327</td><td>13,141</td></tr><tr><td>R00011</td><td>3</td><td>(20, 6, 18, 10, 22)</td><td>200</td><td>60</td><td>180</td><td>50</td><td>110</td><td>5,054</td><td>0.284</td><td>17,777</td></tr><tr><td>R00012</td><td>3</td><td>(19, 6, 16, 11, 21)</td><td>190</td><td>60</td><td>160</td><td>55</td><td>105</td><td>11,015</td><td>0.308</td><td>35,740</td></tr></table>

Figure S30: PXRD patterns of MTV-MOF-303, R00001–R00012 (Table S25).

## Section S8. Nitrogen Sorption

$\mathrm { N _ { 2 } }$ sorption was measured for two representative frameworks, CAU-23 and MIL-160: for each, the microwave optimum of the campaign and a reflux product for comparison. The isotherms were recorded at 77 K on a Micromeritics 3Flex after activation under vacuum at $1 2 0 ~ ^ { \circ } \mathrm { C }$ overnight. BET areas were obtained from the adsorption branch by a linear fit of

$$
{ \frac { p / p _ { 0 } } { n \left( 1 - p / p _ { 0 } \right) } } = { \frac { 1 } { n _ { \mathrm { m } } C } } + { \frac { C - 1 } { n _ { \mathrm { m } } C } } { \frac { p } { p _ { 0 } } } ,\tag{S28}
$$

where n is the adsorbed amount and $n _ { \mathrm { m } }$ the monolayer capacity; $n _ { \mathrm { m } }$ was converted to an area with the standard cross-sectional area of an adsorbed $\mathrm { N _ { 2 } }$ molecule, 0.162 $\mathrm { n m } ^ { 2 } [ 1 8 ]$ . Langmuir areas were obtained from the same points with the Langmuir equation,

$$
\frac { p / p _ { 0 } } { n } = \frac { 1 } { n _ { \mathrm { m } } K } + \frac { p / p _ { 0 } } { n _ { \mathrm { m } } } ,\tag{S29}
$$

where K is the Langmuir constant. The same range, $p / p _ { 0 } = 0 . 0 5 – 0 . 1 0$ (six points, $R ^ { 2 } \ge 0 . 9 9 9 7 )$ was used for every sample and both equations, so that the areas can be compared directly. The microwave samples are the best products of the active-learning campaigns (CAU-23 R00001, MIL-160 R00007); the reflux samples are the 6 h products of Section S2.1 (Table S26, Figure S31).

Table S26: BET and Langmuir areas from the adsorption branch at $p / p _ { 0 } = 0 . 0 5 – 0 . 1 0 ;$ areas in $\mathrm { m ^ { 2 } g ^ { - 1 } }$
<table><tr><td>Framework Sample</td><td></td><td>Mass  $\left( { \mathrm { m g } } \right)$ </td><td> $S _ { \mathrm { B E T } }$ </td><td> $S _ { \mathrm { L a n g m u i r } }$ </td><td>Lit.  $S _ { \mathrm { B E T } }$ </td></tr><tr><td>CAU-23</td><td>microwave, R00001</td><td>42.6</td><td>1173</td><td>1371</td><td>1250[2]</td></tr><tr><td>MIL-160</td><td>reflux 6 h, mistakenly heated overnight</td><td>57.2</td><td>56</td><td>65</td><td></td></tr><tr><td></td><td>microwave, R00007</td><td>34.3</td><td>729</td><td>850</td><td>1150[4]</td></tr><tr><td></td><td>reflux, 6 h</td><td>49.2</td><td>908</td><td>1058</td><td></td></tr></table>

![](images/59c1b1f55c841a4a7732af87bae12ae629acf097f01d53cc3ca8231ce5a128bb.jpg)

![](images/083ca995130ba098c118d15231a44f7dc83bc97be924e4e26b3cf8de0e7d33a4.jpg)  
Figure S31: $\mathrm { N _ { 2 } }$ isotherms at 77 K of the microwave optima and the 6 h reflux products of (a) CAU-23 (heated overnight after reflux) and (b) MIL-160. Filled: adsorption; open: desorption. Legend: SBET.

Results. The CAU-23 microwave optimum is microporous (type-I isotherm) with $S _ { \mathrm { B E T } } ~ { = }$ 1173 $\mathrm { m ^ { 2 } g ^ { - 1 } \ (  { S _ { L a n g m u i r } } = 1 3 7 1 \ m ^ { 2 } g ^ { - 1 } ) }$ , 94% of the literature BET value; the reflux product that was mistakenly heated overnight has converted into a different, nearly non-porous phase $( 5 6 ~ \mathrm { m } ^ { 2 } \mathrm { g } ^ { - 1 } )$ . For MIL-160, the microwave synthesis reached in 55 min 80% of the BET area obtained after 6 h of reflux and measured on the same instrument (729 against 908 m2 g−1; Langmuir, 850 against $\mathrm { 1 0 5 8 ~ m ^ { 2 } g ^ { - 1 } } )$ . The two microwave optima, which have the highest q of their campaigns, are thus porous frameworks. This supports the use of q as the target of the optimization (Section S2.1, Table S3).

## Data and Code Availability

Code and data are available in the GitHub repository, https://github.com/deep-synthesis /ActMOF.

## References

[1] Zheng, Z.; Zhang, O.; Nguyen, H. L.; Rampal, N.; Alawadhi, A. H.; Rong, Z.; Head-Gordon, T.; Borgs, C.; Chayes, J. T.; Yaghi, O. M. ChatGPT Research Group for Optimizing the Crystallinity of MOFs and COFs. ACS Cent. Sci. 2023, 9, 2161–2170. DOI: 10.1021/acscentsci.3c01087.

[2] Lenzen, D.; Zhao, J.; Ernst, S.-J.; Wahiduzzaman, M.; Ken Inge, A.; Fröhlich, D.; Xu, H.; Bart, H.-J.; Janiak, C.; Henninger, S.; Maurin, G.; Zou, X.; Stock, N. A Metal– Organic Framework for Efficient Water-Based Ultra-Low-Temperature-Driven Cooling. Nat. Commun. 2019, 10, 3025.

[3] Tannert, N.; Jansen, C.; Nießing, S.; Janiak, C. Robust Synthesis Routes and Porosity of the Al-Based Metal-Organic Frameworks Al-Fumarate, CAU-10-H and MIL-160. Dalton Trans. 2019, 48, 2967–2976.

[4] Permyakova, A.; Skrylnyk, O.; Courbon, E.; Affram, M.; Wang, S.; Lee, U.-H.; Valekar, A. H.; Nouar, F.; Mouchaham, G.; Devic, T.; De Weireld, G.; Chang, J.-S.; Steunou, N.; Frère, M.; Serre, C. Synthesis Optimization, Shaping, and Heat Reallocation Evaluation of the Hydrophilic Metal–Organic Framework MIL-160(Al). ChemSusChem 2017, 10, 1419–1426.

[5] Pedregosa, F.; Varoquaux, G.; Gramfort, A.; Michel, V.; Thirion, B.; Grisel, O.; Blondel, M.; Prettenhofer, P.; Weiss, R.; Dubourg, V.; Vanderplas, J.; Passos, A.; Cournapeau, D.; Brucher, M.; Perrot, M.; Duchesnay, E. Scikit-learn: Machine Learning in Python. J. Mach. Learn. Res. 2011, 12, 2825–2830.

[6] Gramacy, R. B.; Apley, D. W. Local Gaussian Process Approximation for Large Computer Experiments. J. Comput. Graph. Stat. 2015, 24, 561–578.

[7] Jones, D. R.; Schonlau, M.; Welch, W. J. Efficient Global Optimization of Expensive Black-Box Functions. J. Global Optim. 1998, 13, 455–492.

[8] Kruskal, W. H.; Wallis, W. A. Use of Ranks in One-Criterion Variance Analysis. J. Am Stat. Assoc. 1952, 47, 583–621.

[9] Dunn, O. J. Multiple Comparisons Using Rank Sums. Technometrics 1964, 6, 241–252.

[10] Shields, B. J.; Stevens, J.; Li, J.; Parasram, M.; Damani, F.; Alvarado, J. I. M.; Janey, J. M.; Adams, R. P.; Doyle, A. G. Bayesian Reaction Optimization as a Tool for Chemical Synthesis. Nature 2021, 590, 89–96.

[11] Swersky, K.; Snoek, J.; Adams, R. P. Multi-Task Bayesian Optimization. In Advances in Neural Information Processing Systems 26 (NIPS 2013); 2013; pp 2004–2012.

[12] Kennedy, M. C.; O'Hagan, A. Predicting the Output from a Complex Computer Code When Fast Approximations Are Available. Biometrika 2000, 87, 1–13.

[13] Gramacy, R. B.; Lee, H. K. H. Optimization under Unknown Constraints. In Bayesian Statistics 9; Bernardo, J. M., et al., Eds.; Oxford University Press, 2011; pp 229–256.

[14] Gelbart, M. A.; Snoek, J.; Adams, R. P. Bayesian Optimization with Unknown Constraints. In Proceedings of the 30th Conference on Uncertainty in Artificial Intelligence (UAI); AUAI Press, 2014; pp 250–259.

[15] Bachoc, F.; Helbert, C.; Picheny, V. Gaussian Process Optimization with Failures: Classification and Convergence Proof. J. Global Optim. 2020, 78, 483–506.

[16] Hickman, R. J.; Tom, G.; Zou, Y.; Aldeghi, M.; Aspuru-Guzik, A. Anubis: Bayesian Optimization with Unknown Feasibility Constraints for Scientific Experimentation. Digital Discovery 2025, 4, 2104–2122.

[17] Wakabayashi, Y. K.; Otsuka, T.; Krockenberger, Y.; Sawada, H.; Taniyasu, Y.; Yamamoto, H. Bayesian Optimization with Experimental Failure for High-Throughput Materials Growth. npj Comput. Mater. 2022, 8, 180.

[18] Thommes, M.; Kaneko, K.; Neimark, A. V.; Olivier, J. P.; Rodriguez-Reinoso, F.; Rouquerol, J.; Sing, K. S. W. Physisorption of Gases, with Special Reference to the Evaluation of Surface Area and Pore Size Distribution (IUPAC Technical Report). Pure Appl. Chem. 2015, 87, 1051–1069.

[19] Holm, S. A Simple Sequentially Rejective Multiple Test Procedure. Scand. J. Stat. 1979, 6, 65–70.