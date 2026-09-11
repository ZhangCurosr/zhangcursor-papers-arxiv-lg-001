# Dynamic language model representations for multi-objective reaction optimisation

Joshua W. Sin <sup>\*</sup> <sup>1</sup> <sup>2</sup> David Ming Segura <sup>\*</sup> <sup>1</sup> <sup>2</sup> <sup>3</sup> Bojana Rankovic´ <sup>\*</sup> <sup>2</sup> <sup>3</sup> Siu Lun Chau <sup>4</sup> Marius D. R. Lutz <sup>5</sup> Andrea Anelli <sup>5</sup> Ryan P. Burwood <sup>6</sup> Kurt Puntener¨ <sup>1</sup> Maximilian J. Notheis <sup>1</sup> Raphael Bigler <sup>1</sup> Philippe Schwaller <sup>2</sup> <sup>3</sup>

## Abstract

Optimising chemical reactions across multiple objectives, such as yield, selectivity, and safety, is central to chemical synthesis, and model-driven approaches depend critically on how reaction components are represented. Established featurisations are either chemically uninformative, as with one-hot encodings, or, as with molecular descriptors, do not readily extend across chemically distinct components. For structurally and functionally diverse components, it is therefore unclear what a shared representation should contain. Constructing such a representation is itself a challenging research undertaking that must be revisited for each new reaction system. Here we bypass this step by learning the reaction representation dynamically from text. Textual descriptions of reaction conditions are encoded by a fine-tuned language model trained jointly with Gaussian process surrogates, yielding task-adaptive representations within a multi-objective Bayesian optimisation loop. Across nickel- and palladium-catalysed cross-couplings in both sequential and parallel experimentation regimes, this approach reaches optimisation convergence in fewer experiments than descriptor libraries or one-hot encoding. Applied

prospectively to a palladium-catalysed cyanation spanning mixed ligand denticity and heterogeneous additives, and to a three-objective asymmetric hydrogenation across chiral iridium and ruthenium catalyst families, two rounds of highthroughput experimentation (192 reactions, under 3% of each design space) delivered conditions translating directly to gram scale in 94% and 84% isolated yield, the latter at 99.6% enantiomeric excess.

## 1. Introduction

Chemical reaction optimisation is essential to chemical synthesis, and optimising reaction conditions requires navigating vast combinatorial spaces of reaction components such as ligands, catalysts, and bases, while balancing multiple objectives such as yield, selectivity, and cost [1–5]. This challenge is particularly acute in pharmaceutical process and preclinical chemistry, where identifying robust conditions for active pharmaceutical ingredient (API) synthesis entails additional environmental, health, and safety considerations [6–9], and in academic settings where enabling novel transformations often requires extensive exploration of many diverse reaction parameters [10, 11].

Machine learning approaches have proved remarkably effective, with Bayesian optimisation emerging as the leading framework for data-efficient reaction optimisation. It has been applied in low-data regimes [12–19] and, more recently, in parallel high-throughput settings [20, 21]. Despite these algorithmic advances, how best to represent reactions for model-driven optimisation remains an open and consequential challenge.

Molecular descriptors have been the dominant featurisation strategy in reaction optimisation and prediction (Figure 1a). Pre-computed descriptor libraries such as Kraken [25] and the COSMO-RS [26] database provide readily available steric and electronic properties for established compound classes and have been widely adopted [15, 16, 20, 21, 27– 30]. Alternatively, tailored and often laborious feature ena Approaches for featurising chemical entities in model-driven reaction optimisation

![](images/59753aeb7b01c6b7ef5acd7264e1552b0b542385e95639d1aceddf3c24c075ec.jpg)  
Static representations which do not adapt to the optimisation task as more data is collected

b This approach: trainable large language model embeddings for reaction optimisation  
![](images/4d479125ebdbf096e1c6c3a3ca21a4dc6a2b1185bccb68bdf0033fe16782633b.jpg)  
Figure 1. Featurisation approaches for model-driven reaction optimisation. a, Conventional approaches for representing chemica entities. Left: molecular descriptors (e.g., Sterimol parameters, buried volume, and bite angle) incorporate physicochemical properties derived from density functional theory (DFT) [22] or extended tight-binding (xTB) [23], enabling chemical similarity-aware representations. However, descriptor sets are inflexible across entity classes, require domain-specific expertise for feature engineering, and can be computationally expensive to generate. Right: one-hot encoding assigns a binary indicator vector to each categorical component (e.g., different ligands, bases, and precursors), yielding sparse, high-dimensional representations that encode no chemical similarity. b, This approach: trainable large language model (LLM) embeddings for reaction optimisation. Textual descriptions of reaction conditions are encoded by a LoRA-fine-tuned [24] language model into dense, continuous embeddings. These embeddings are passed through objective-specific projection heads into independent Gaussian process (GP) surrogate models, and the entire architecture (LLM-GP) is trained jointly. The architecture is embedded within a multi-objective Bayesian optimisation cycle, with the reaction representations improving at each iteration, organising representations by reaction performance. This yields a unified, task-adaptive representation without requiring descriptor computation or domain-specific feature engineering.

gineering workflows, involving computationally expensive density functional theory (DFT) or semi-empirical calculations, conformer ensemble analysis, and reaction-specific mechanistic interrogation, can yield informative descriptors [22, 23, 31–37].

Despite their widespread use, descriptor-based approaches face fundamental limitations. Selecting which descriptors to compute relies on a priori chemical intuition about which molecular properties govern reactivity, a process that is biased by existing mechanistic understanding and must often be revisited for each new reaction system. More critically, descriptors are often not transferable across chemically distinct reaction components. Steric and electronic parameters for monodentate phosphines, for example, are only partially applicable to bidentate ligands, and reactions involving chemically heterogeneous components such as inorganic bases and organometallic additives may lack any meaningful shared descriptor space, making feature engineering increasingly prohibitive as chemical diversity grows. One-hot encoding offers a simpler alternative that is universally applicable and has shown competitive performance in some settings [12, 13, 38]. However, it produces sparse, high-dimensional representations that encode no chemical similarity, and its dimensionality scales unfavourably with the number of reaction components [20, 38] (Figure 1a).

Advances in large language models (LLMs) have shown strong performance across chemical tasks [39–47] and demonstrated that their learned embeddings can serve as dense, informative representations for predictive modelling [48–51]. Recent work has further shown that LLM representations can be adapted to specific optimisation tasks through joint fine-tuning with Gaussian process surrogates, demonstrated for single-objective, sequential optimisation [52]. Crucially, LLMs can encode arbitrarily complex chemical information into fixed-length representations directly from text, offering a naturally unified featurisation method across component classes. These important advances notwithstanding, a general framework for multi-objective reaction optimisation, applicable across chemically heterogeneous reaction components and validated across experimental regimes from low-data to highthroughput settings, has yet to be realised.

Here, we introduce Alice, a multi-objective chemical reaction optimisation framework that uses trainable LLM embeddings as a unified reaction representation (Figure 1b). Textual descriptions of reaction conditions are encoded by a low-rank adaptation (LoRA)-fine-tuned language model and passed through reaction objective-specific projection heads into independent Gaussian process surrogates, with the architecture trained end-to-end via joint marginal loglikelihood maximisation. This jointly trained architecture, hereafter referred to as the LLM-GP framework, produces dynamic embeddings that are refined at each optimisation iteration, adapting the reaction representation to the optimisation task at hand. We first benchmark the LLM-GP framework retrospectively against established descriptor libraries and one-hot encoding baselines across sequential low-data optimisation of nickel- and palladiumcatalysed cross-couplings and large-batch 96-well plate high-throughput experimentation campaigns, converging on high-performing conditions in fewer experiments across all settings.

We then apply the framework prospectively to two wet-lab campaigns executed on an automated high-throughput platform, each chosen for a form of chemical heterogeneity for which descriptor libraries are not readily available. In a palladium-catalysed cyanation, a single representation spans mono- and bidentate phosphine catalysts alongside additives ranging from elemental zinc to organic and inorganic bases; within this space the framework identified conditions using a lower-hazard cyanide source that gave >99% conversion and >99% selectivity, and 94% isolated yield on gram scale. In an asymmetric dynamic kinetic hydrogenation, the framework balanced conversion, diastereomeric excess (de), and enantiomeric excess (ee) across chiral catalysts spanning different metals, donor sets, and coordination geometries, delivering the target syn alcohol in 84% isolated yield and 99.6% ee on scale-up. Both campaigns used only two rounds of 96 experiments, sampling under 3% of their respective design spaces. Neither required descriptor computation or feature engineering, indicating that effective multi-objective optimisation is achievable in reaction spaces where constructing a shared descriptor representation would itself be a substantial undertaking.

## 2. Computational results

We benchmarked the LLM-GP framework against established and readily available featurisation methods in reaction optimisation: DFT descriptors from the Kraken [25] and COSMO RS [26] descriptor libraries, and one-hot encoding baselines. Descriptor coverage is uneven across component classes: Kraken supplies descriptors for the monophosphine ligands and COSMO-RS for the solvents, but no comparable library exists for the bases and precursors, which were therefore represented by one-hot encoding within the descriptor baseline. All three representations were embedded in the same multi-objective Bayesian optimisation loop, using the same acquisition function (qLogNParEGO) and an identical experimental budget (see Methods).

A set of reaction conditions is Pareto optimal when no other condition improves one objective without degrading another; these points define the Pareto front (Figure 2a). Optimisation performance was measured by hypervolume, the volume of objective space enclosed by the Pareto-optimal conditions identified so far. A larger hypervolume means the conditions found are both closer to the best achievable trade-offs and better spread across them, and we report it as a percentage of the value attained by the true Pareto front of each dataset. We compared methods by the mean number of iterations required to reach 90% and 95% of the maximum hypervolume, with standard error across 20 random seeds, as these thresholds represent practically relevant levels of convergence. For each dataset the optimisation budget was set by the number of iterations needed for at least one method to reach 95% of the maximum hypervolume on average. We additionally report the percentage of optimisation runs reaching each hypervolume threshold within the allocated experimental budget as a measure of robustness; these results are provided in Extended Data Section A. Among the pre-trained language models and pooling strategies evaluated, T5-base [53] with mean pooling consistently yielded the best optimisation performance and was used throughout (full ablation studies across model architectures and pooling strategies are provided in Extended Data Section B).

## 2.1. Sequential low-data reaction optimisation

We first evaluated our approach in sequential low-data optimisation regimes, where the primary goal is sample efficiency: minimising the number of experiments required to identify high-performing conditions (Figure 2b). We used two open-source multi-objective reaction datasets, each

404 experiments to optimise over in a chemical space of 59,040 combinations

384 experiments to optimise over in a chemical space of 88,000 combinations

![](images/4ca150f4e06465af9357875f5c5c71ae4a16ec00d10fa88392ed6a91901367ad.jpg)

![](images/53336fd511c2eee7c2a471f85a9c7538cf0b872ec52f8039e316c0c203c0c139.jpg)

![](images/3666199d5e94d01b0b23d455378855fba199f599c4ab5f80753b3fa7ca2116ec.jpg)

![](images/0d7e9b63fae2eb1be695c997c59f791a5051161330eeeed16a62f9d712124ccf.jpg)

![](images/bfa6e20bae28e7735aa9c9a5c9e614d160310dc733969298d02f3a3e1d030287.jpg)

![](images/471b17b0e05602e52e19845f8c6ae3a62d2ff99a183e5e13e7b6189d2a2e00d9.jpg)

![](images/d560ed44b633def525454b52882f347807383f543275144790478f9a5dd9dec4.jpg)

![](images/247f7ee7b851a2cddaa905b239f7ffd9a3a87a5ba482df82564d6070e3d7a590.jpg)  
Figure 2. Benchmarking multi-objective reaction optimisation across experimental regimes. a, Metrics for evaluating multi-objective optimisation. Left: the hypervolume indicator quantifies the volume of objective space dominated by the current Pareto-optimal set, providing a scalar measure of multi-objective performance. Right: convergence is assessed by tracking the normalised hypervolume over optimisation iterations; the number of iterations required to reach practical performance thresholds (90% and 95% of maximum hypervolume) serves as a comparative metric across featurisation methods. The percentage of optimisation runs (initialised with different random seeds) reaching each threshold is reported in Extended Data Section A. b, Sequential low-data reaction optimisation benchmarks. Two multi-objective case studies from published reaction datasets are evaluated: a nickel-catalysed Suzuki coupling [20] and a palladium catalysed Suzuki coupling [21]. Iterations required to reach 90% and 95% hypervolume thresholds are compared for the LLM-GP framework, DFT descriptor libraries (Kraken [25], COSMO-RS [26]), and one-hot encoding, repeated across 20 random seeds (see Methods for more details). Runs not reaching a threshold within the experimental budget were assigned the maximum budget. c, 96-well plate HTE reaction optimisation benchmark. A palladium-catalysed sulfonamide coupling [21] (virtual benchmark of 21,648 experiments) from a published dataset is optimised in large parallel batches of 96 experiments.

comprising experimental yield and selectivity as the optimisation objectives and presenting large combinatorial search spaces with diverse categorical reaction components: a nickel-catalysed Suzuki coupling (384 experiments from a space of 88,000 combinations, varying ligand, Ni-precursor, base, solvent, co-solvent, and temperature) [20], and a palladium-catalysed Suzuki coupling (404 experiments from 59,040 combinations, varying ligand, Pd-precursor, base, solvent, and co-solvent) [21]. Each campaign was initialised with 5 experiments selected using Sobol sampling to ensure diverse coverage of initial points across the search space (see Methods), after which the Bayesian optimisation loop selected one candidate per iteration. Iteration counts below refer to these optimisation iterations and exclude the initial experiments. On the nickel-catalysed Suzuki coupling, the LLM-GP framework reached the 90% hypervolume threshold in approximately 13 optimisation iterations on average, roughly half the number required by both the DFT descriptor-based (∼26 iterations) and one-hot encoding (∼24 iterations) baselines (Figure 2b). At the more stringent 95% threshold, the gap narrowed, though the framework retained a consistent advantage, converging in approximately 24 iterations compared to ∼28 for DFT descriptors and ${ \sim } 3 0 $ for one-hot encoding. On the palladium-catalysed Suzuki coupling benchmark, the framework achieved both the 90% and 95% thresholds in approximately 65 iterations on average, while the descriptor-based and one-hot encoding baselines required approximately 80 iterations each to reach the same performance levels (Figure 2b). Across both case studies, the LLM-GP framework consistently reached practical convergence thresholds in fewer experiments than either baseline, improving sample efficiency. Over 20 independent runs initialised with different random seeds, the framework also achieved a higher proportion of successful optimisations reaching both convergence thresholds within the allocated experimental budget, indicating more robust convergence (see Extended Data Section A).

## 2.2. Highly parallel HTE reaction optimisation

Modern high-throughput experimentation (HTE) platforms have combined parallel screening with data-driven optimisation [20, 54], enabling the exploration of large reaction spaces in compressed experimental timescales. We next sought to evaluate our approach in this batched optimisation setting, where reducing the number of HTE plate iterations directly translates to savings in experimental time. We used an open-source palladium-catalysed sulfonamide coupling dataset (a Buchwald–Hartwig-type C–N coupling with 21,648 virtual experiments, varying ligand, Pd-precursor, base, and solvent) [21] as a benchmark, optimising yield and selectivity in parallel batches of 96 experiments to simulate 96-well plate HTE campaigns (Figure 2c). Each campaign was initialised with a Sobol-sampled plate of 96 experiments, after which each iteration selected 96 conditions at once, so one optimisation iteration corresponds to one physical plate. The LLM-GP framework reached practical convergence thresholds in approximately 1.2 plate optimisation iterations beyond initialisation, whereas both the DFT descriptor-based and one-hot encoding baselines required approximately double (∼2.4) to achieve the same thresholds (Figure 2c). The framework also achieved a higher proportion of successful optimisation runs reaching both convergence thresholds, and results for 24- and 48-well plate batch sizes show the same trend (Extended Data Section A). Given that each 96-well plate campaign typically requires approximately one week of experimental and analytical time, this reduction translates directly into meaningful savings in time and resources. These results demonstrate that our approach offers consistent advantages across both sequential low-data regimes, where sample efficiency is crucial, and parallel large-batch settings increasingly adopted in pharmaceutical process development and academic high-throughput screening [55–57].

## 3. Prospective experimental case studies

Building on our retrospective benchmarks across conventional Buchwald–Hartwig and Suzuki–Miyaura datasets, we next sought to apply our framework prospectively to wet-lab reaction optimisation campaigns, moving beyond standard reaction spaces. Here, we targeted transformations containing chemically heterogeneous components that traditional reaction representations struggle to capture. In each campaign, we used our approach to identify high-performing conditions within the search space, iteratively generating ML predictions and conducting the suggested experiments with our HTE robotic platform (see Supplementary Information Section 1 for more details). All experimental data and characterisation of any isolated products are reported in the Supplementary Information.

## 3.1. Case study 1: Palladium-catalysed cyanation

Aryl nitriles are prevalent across pharmaceuticals, agrochemicals, and functional materials [58, 59], serving as compact, metabolically stable hydrogen-bond acceptors and hydroxyl and carboxyl isosteres [60]. Among synthetic routes accessing these motifs, the transition-metal-catalysed cyanation of aryl halides remains one of the most widely adopted and functionally tolerant strategies [61]. While substantial methodology development has produced a broad repertoire of candidate catalysts, additives, and cyanide sources, this very diversity introduces three intersecting sources of chemical heterogeneity that challenge traditional modelling approaches. First, both monodentate and bidentate phosphine ligands are competitive across cyanation substrate classes, yet descriptor libraries developed for monophosphines are not directly transferable to bisphosphines (and vice versa) thus fragmenting the feature space. Second, productive cyanation conditions often require additives spanning chemically disparate classes such as elemental reductants (e.g., Zn), organic bases (e.g., NEt ), and inorganic salts (e.g., KOAc), which resist straightforward unification within standard chemical feature spaces. Finally, the choice of cyanide source carries practical implications beyond reactivity. Classical sources such as NaCN and $\mathrm { Z n } ( \mathrm { C N } ) _ { 2 }$ are highly toxic, whereas safer but less reactive alternatives such as ${ \mathrm { K } } _ { 4 } [ \mathrm { F e } ( \mathrm { C N } ) _ { 6 } ]$ are increasingly preferred in industrial process settings despite their poor organic solubility, which necessitates biphasic reaction media [62, 63]. Together, these intersecting dimensions of heterogeneity in cyanation reactions pose a distinct challenge for conventional descriptor frameworks, highlighting the value of more flexible, unified representation strategies.

![](images/71d7922a79cce6a38db8fee0993974638f6bac645d2d36bca1e125215c4ebc4a.jpg)

![](images/24105d0f02525c8d99ebad889291a033fcc00682759b9a1b2e8bc1111e9b44fa.jpg)  
Figure 3. Case study 1: Palladium-catalysed cyanation. a, Search space for the cyanation of 1 to 2. Excluding reaction conditions with temperatures above the solvent boiling point results in 30,000 conditions (full search space in Extended Data Figure 7). Conversion (%) and selectivity (%) were maximised. Each reaction condition was encoded as a textual description, embedded by the joint LLM-GP framework, and model selected conditions were executed on an automated high-throughput experimentation (HTE) platform in 96-wel HTE plates (see Supplementary Information Section 2.1). b, Reaction objective distributions for HTE optimisation rounds, Plate 1 (Initialisation) and Plate 2 (Optimisation), with marginal histograms. Only conditions using ${ \mathrm { K } } _ { 4 } [ \mathrm { F e } ( \mathrm { C N } ) _ { 6 } ] .$ , the least toxic cyanide source, are shown on the plot. The step line traces the Pareto front among these conditions after Plate 1. c, Scale-up of the best conditions identified by optimisation, delivering 2 in 94% isolated yield at gram scale (see Supplementary Information Section 2.2).

We assembled a design space for the cyanation of methyl 3-bromo-5-fluorobenzoate, comprising 40 commercially available palladium catalysts spanning both mono- and bidentate ligand classes (e.g., [Pd(tBuXPhos)(allyl)]OTf, $\mathrm { \Delta [ P d ( D P P F ) C l _ { 2 } ] ) }$ , three cyanide sources of varying toxicity (NaCN: median lethal dose (mg/kg) $\begin{array} { r } { L D _ { 5 0 } = 3 . 6 , Z \mathrm { n } ( \mathbf { C N } ) _ { 2 } \mathrm { ; } } \end{array}$ $L D _ { 5 0 } = 5 4 , { \mathrm { K } } _ { 4 } [ \mathrm { F e } ( { \bf C N } ) _ { 6 } ] ; L D _ { 5 0 } = 3 6 1 3 )$ , five additive options (Zn, KOAc, NEt , K CO , or none), ten solvents, two co-solvent options, and three temperatures (Figure 3a). Conditions in which the reaction temperature exceeded the solvent boiling point were removed, giving a final design space of $3 0 { , } 0 0 0$ conditions. While constructing a bespoke descriptor representation across these heterogeneous components is in principle feasible, doing so would require laborious, trial-and-error feature engineering with no guarantee of success; our approach bypasses this bottleneck by providing a unified, out-of-the-box featurisation directly from textual descriptions, enabling immediate campaign execution without manual descriptor curation. The full search space is provided in Extended Data Figure 7.

Optimisation was initialised with a Sobol-sampled first plate of 96 experiments to probe the global design space. While the first round identified several high-performing conditions, the top hits relied exclusively on toxic cyanide sources (NaCN and $\mathrm { Z n } ( \mathrm { C N } ) _ { 2 } )$ , while ${ \mathrm { K } } _ { 4 } [ \mathrm { F e } ( \mathrm { C N } ) _ { 6 } ]$ yielded only mediocre outcomes with either poor conversion or selectivity (Figure 3b). To test whether high-performing conditions could be identified using a lower-hazard cyanide source, we fixed the cyanide source for round 2 to ${ \mathrm { K } } _ { 4 } [ \mathrm { F e } ( \mathrm { C N } ) _ { 6 } ]$ and re-optimised within this constrained domain. The second optimisation round successfully met this target, identifying multiple conditions achieving >99% conversion and >99% selectivity (Figure 3b). Notably, high-performing hits spanned both monodentate ([Pd(tBuXPhos)(allyl)]OTf) and bidentate ([Pd(Xantphos)(allyl)]Cl) ligands, exemplifying the benefit of jointly searching over both ligand families within a unified representation. To confirm that these milligram-scale HTE hits translate to preparative scale-up, we scaled the lead conditions ([Pd(Xantphos)(allyl)]Cl with $\mathrm { K } _ { 4 } [ \mathrm { F e } ( \mathrm { C N } ) _ { 6 } ] \cdot 3 \mathrm { H } _ { 2 } \mathrm { O }$ in $\mathrm { D M C } / \mathrm { H } _ { 2 } \mathrm { O } )$ to gram scale (Figure 3c). The reaction proceeded cleanly, delivering the target

## a Asymmetric hydrogenation reaction condition search space

![](images/075023ff2501c8edd4b9052be24044ad132e2229f5a066e8ac870c0191043e53.jpg)  
Figure 4. Case study 2: Asymmetric ketone hydrogenation. a, Search space for the dynamic kinetic hydrogenation of 3 to the targe syn-(1S, 2R) alcohol 4a, one of four accessible stereoisomers (4a–4d) arising from base-catalysed epimerisation between 3a and 3b. Conversion (%), de (syn) (%), and ee (syn) (%) were maximised simultaneously across 32 chiral iridium and ruthenium catalysts, 9 bases, 7 solvents, 2 base loadings and 2 temperatures, giving 8,064 conditions (full search space in Extended Data Figure 8). b, Reaction objectives across HTE optimisation rounds, Plate 1 (Initialisation) and Plate 2 (Optimisation). Positive de and ee denote excess of the target syn diastereomer and (1S, 2R) enantiomer respectively. Negative values denote the corresponding anti diastereomer or (1R, 2S) enantiomer, respectively. Syn-selective outcomes increased from 5 of 96 conditions in Plate 1 to 59 of 96 in Plate 2. c, Scale-up of the bes conditions identified by optimisation, delivering 4a in 84% isolated yield and 99.6% ee at gram scale (see Supplementary Information Section 3.2). [IrClH ((R/S) – DTB – SpiroPAP – 3-Me)] was the only catalyst pair in the campaign selective for the target syn isomer.

aryl nitrile with 94% isolated yield (see Supplementary Information Section 2). This scalable, low-hazard protocol was identified in just two iterative campaign rounds (one week per round) totalling 192 experiments, evaluating less than 1% of the 30,000-condition design space.

## 3.2. Case study 2: Asymmetric ketone hydrogenation

Having established the approach on reaction systems with multi-component heterogeneity, we next sought to evaluate its capacity to navigate stereoselective transformations. Catalytic asymmetric hydrogenation of prochiral ketones is among the most widely deployed enantioselective methods in pharmaceutical and agrochemical synthesis [64]. In particular, the dynamic kinetic hydrogenation of α-substituted aryl ketones, in which two contiguous stereocentres are set in a single step to yield up to four potential stereoisomers [65, 66], exemplifies a class of multi-objective optimisation problems that require balancing trade-offs across a Pareto front defined by conversion, diastereomeric excess (de), and enantiomeric excess (ee). Productive catalysts include chiral iridium and ruthenium complexes drawn from a structurally diverse pool of ligand motifs, spanning bidentate phosphines, mixed phosphine–nitrogen donors (P, N and P, N, N), and a variety of associated ancillary ligands and counter-ions. Representing this coordination diversity in a unified descriptor framework is non-trivial; physical featurisation schemes engineered to quantify the

3D chiral pocket of one ligand geometry or metal centre fail to generalise across fundamentally disparate coordination spheres. Jointly balancing three reaction objectives across this heterogeneous chiral space compounds the modelling challenge. We assembled a design space of 8,064 conditions for the asymmetric dynamic kinetic hydrogenation of 1,2-diphenylpropan-1-one towards the syn-(1S, 2R)-1,2- diphenylpropan-1-ol diastereomer, comprising 32 chiral iridium and ruthenium catalysts, 9 organic and inorganic bases, 7 solvents, varied base loading (mol%), and temperature (Figure 4a). The full search space is provided in Extended Data Figure 8.

Optimisation was initialised with a Sobol-sampled first HTE plate of 96 experiments spanning the full design space (Figure 4b). The first round of experiments revealed that the undesired anti diastereomer was favoured across the majority of conditions evaluated, yielding multiple hits with high anti selectivity (−99% de (syn)). In contrast, accessing the target syn diastereomer was more challenging: only 5 out of 96 experiments were syn selective, and out of the 32 chiral catalysts tested, only a single catalyst pair, [IrClH ((R/S) – DTB – SpiroPAP – 3-Me)], demonstrated activity towards the syn isomer. The top hit for the desired isomer in Plate 1 achieved quantitative conversion but moderate stereoselectivity (73.5% de (syn) and 86.7% ee (syn)) (Figure 4b). In achiral media, enantiomeric catalyst pairs yield mirror-image products, so each training observation was augmented with its reflected counterpart under a sign inversion of the enantiomeric excess (see Supplementary Information Section 3.3).

Guided by the updated LLM representations, the framework refocused optimisation towards syn-producing regions, increasing syn-selective outcomes from 5/96 to 59/96 conditions in Plate 2 (Figure 4b). The Pareto front was expanded towards higher stereoselectivity, with the most stereoselective conditions identified reaching 87.1% de (syn) and 93.2% ee (syn). The most balanced lead hit achieved >99% conversion, 80.5% de (syn), and 89.7% ee (syn). Identifying the productive catalyst family did not by itself resolve the optimisation problem: across the reaction conditions employing [IrClH<sub>2</sub>((R/S) – DTB – SpiroPAP – 3-Me)], diastereoselectivity spanned -67% to +87.1% de (syn), and in 12 instances its sign inverted upon changing temperature or base loading alone, consistent with the dynamic kinetic nature of the transformation. Systematically removing the highest-performing conditions from the Plate 1 training data did not prevent the model from recovering high-performing syn-selective conditions, indicating that the outcome did not depend on a small number of fortunate initial hits (see Supplementary Information Section 3.4). Translating the lead conditions identified from optimisation to preparative gram scale proceeded smoothly (Figure 4c): the lead conditions, [IrClH ((S) – DTB – SpiroPAP – 3-Me)] with DBU (50 mol%) in tAmOH at $6 0 ^ { \circ } \mathrm { C } ,$ delivered the target syn alcohol with the desired enantiomer in 84% isolated yield with 99.6% enantiomeric excess (see Supplementary Information Section 3). This scalable protocol was identified in two iterative campaign rounds (one week per round) using 192 experiments out of 8,064, corresponding to approximately 2.4% of the design space.

## 4. Outlook

This work demonstrates that dynamic language model representations provide a general route to multi-objective reaction optimisation, applicable across chemically diverse reaction systems. By encoding reaction components directly from textual descriptions, the framework searches jointly over chemical entities of different classes (e.g., ligand families, additives, and catalysts) within a single representation, without descriptors being selected or computed for each new system. Applied prospectively to two wet-lab campaigns, it identified conditions that translated directly to preparative scale within two rounds of high-throughput experimentation. Engineered descriptors are chemically interpretable, but their selection presupposes an understanding of which molecular properties govern reactivity in the system at hand. Learning the representation dynamically from text removes this requirement, allowing optimisation to begin before such understanding is established, where mechanistic insight could emerge from the conditions identified rather than being needed to find them. By removing expert-guided featurisation as a prerequisite, we anticipate this approach will extend model-guided optimisation to a wider range of chemistry, including biocatalysis and enzymatic reactions, and ultimately enable more efficient chemical processes.

## Methods

## Metrics for multi-objective optimisation problems

We consider the simultaneous maximisation of M objective functions $y _ { m } \colon \mathcal { X } \to \mathbb { R } , m = 1 , \ldots , M$ (e.g., reaction yield and selectivity), over a finite chemical design space $\mathcal { X } = \{ \mathbf { x } _ { 1 } , \dotsc , \mathbf { x } _ { N } \}$ , where each $\mathbf { x } _ { i }$ is a vector of reaction conditions. The aim is to identify the Pareto optimal set, also referred to as the Pareto front. We say that x<sup>′</sup> dominates x (written x<sup>′</sup> ≻ x) if $y _ { m } ( \mathbf x ^ { \prime } ) \geq y _ { m } ( \mathbf x )$ for all $m = 1 , \ldots , M$ and $y _ { j } ( \mathbf x ^ { \prime } ) > y _ { j } ( \mathbf x )$ for some $j \in \{ 1 , \dots , M \}$ . The Pareto optimal set is then defined as:

$$
\mathcal { P } ^ { * } = \left\{ \mathbf { x } \in \mathcal { X } \mid \nexists \mathbf { x } ^ { \prime } \in \mathcal { X } : \mathbf { x } ^ { \prime } \succ \mathbf { x } \right\}\tag{1}
$$

We seek to approximate ${ \mathcal { P } } ^ { * }$ using as few experimental iterations or evaluations as possible with multi-objective Bayesian optimisation. The quality of a candidate Pareto front $\mathcal { P } \subseteq \mathbb { R } ^ { M }$ , obtained from the current set of observations, is measured by its dominated hypervolume (Figure 2a) relative to a reference point $\mathbf { r } \in \bar { \mathbb { R } ^ { M } }$ , chosen such that $r _ { m } < p _ { m }$ for all $\mathbf { p } \in \mathcal { P }$ and $m = 1 , \ldots , M$ :

$$
\begin{array} { r } { \mathrm { H V } ( \mathcal { P } , \mathbf { r } ) = \mathrm { V o l } \Big ( \big \{ \mathbf { y } \in \mathbb { R } ^ { M } \mid \exists \mathbf { p } \in \mathcal { P } : } \\ { r _ { m } \leq y _ { m } \leq p _ { m } \forall m \big \} \Big ) } \end{array}\tag{2}
$$

where $\operatorname { V o l } ( \cdot )$ denotes the Lebesgue measure. For all benchmark datasets, the optimisation objectives are percentages bounded on [0, 100], and we set $\mathbf { r } = ( 0 , 0 )$ . The same reference point was used for all featurisation methods. Hypervolume is a widely used quality indicator for multi-objective optimisation, rewarding both convergence to the true Pareto front and diversity along it [13, 17, 67, 68]. Importantly, it is monotonic and Pareto-compliant [69, 70], meaning that if one candidate Pareto front strictly dominates another, it will achieve a strictly higher hypervolume. In this work, we report the normalised hypervolume percentage, defined as $\mathrm { H V } ( \mathcal { P } , \mathbf { r } ) / \mathrm { H V } ( \mathcal { P } ^ { * } , \mathbf { r } ) \times 1 0 0 \%$ , to evaluate the quality of candidate Pareto fronts $\mathcal { P }$ identified by optimisation algorithms relative to the true Pareto front ${ \mathcal { P } } ^ { * }$

## Representing reaction conditions

We benchmarked three featurisation approaches for representing reaction conditions.

One-hot encoding. As a universally applicable and inexpensive baseline that has shown competitive performance in prior work [12, 13, 38], each unique categorical variable (e.g., XPhos, dioxane, PhMe) was represented as a binary indicator vector, and all component vectors were concatenated to form the input representation (Figure 1a).

Molecular descriptors. Widely adopted as the primary featurisation strategy in reaction optimisation, molecular descriptors from established descriptor libraries were included as a chemically informative baseline [15, 16, 20, 21, 27–30]. Monophosphine ligands were represented using 190 DFT descriptors from the Kraken library [25], with Principal Component Analysis (PCA) [71] applied to retain components explaining 99% of the variance, reducing dimensionality while preserving information. Solvents were parameterised using four COSMOtherm-derived DFT descriptors from the COSMO-RS database [26]. The remaining categorical variables (e.g., bases and precursors), for which comparable descriptor libraries are not readily available, were represented using one-hot encoding.

Large language model representations. Natural language provides a flexible modality for encoding chemically heterogeneous reaction components without requiring componentclass-specific featurisation. Each reaction condition $\mathbf { x } _ { i } \in \mathcal { X }$ was represented as a textual prompt of component-value

pairs, for example:

Reaction condition: ligand: {ligand   
name} solvent: {solvent name}   
precursor: {precursor name} base:   
{base name}

with one pair per parameter varied in the design space.

A pre-trained language model $h _ { \phi }$ maps the tokenised prompt to a sequence of embedding vectors ${ \bf H } _ { i } ~ = ~ h _ { \phi } ( { \bf x } _ { i } ) ~ \in  $ $\mathbb { R } ^ { L \times d _ { \mathrm { e m b } } }$ where L is the sequence length and $d _ { \mathrm { e m b } }$ is the model’s hidden dimension. A pooling operator POOL: $\mathbb { R } ^ { L \times d _ { \mathrm { e m b } } }  \mathbb { R } ^ { d _ { \mathrm { e m b } } }$ aggregates the token-level representations into a fixed-dimensional embedding $\begin{array} { r l } { \mathbf { e } _ { i } } & { { } = } \end{array}$ POOL $( \mathbf { H } _ { i } ) \ \in \ \mathbb { R } ^ { d _ { \mathrm { e m b } } }$ to produce a sequence-level representation. We adopted pooling strategies based on each model’s architecture. For encoder-decoder models, only the encoder stack was used, and pooling was applied over its hidden states. The T5 encoder-decoder models (T5-base [53], T5-small, and T5-chem [72]) used mean pooling, which averages hidden states over non-padded tokens. We used last-token pooling for the decoder-only model Qwen2.5 [73], which takes the hidden state at the last non-padding position, as causal attention ensures that only this token has attended to the full input sequence [74]. Mean pooling was also evaluated for Qwen2.5. For the BART-base encoder-decoder model [75], we used mean pooling following the T5 models, and additionally evaluated CLS pooling as BART inherits a $< s >$ classification token. All models were accessed via the Hugging Face Transformers library [76]. Ablation studies over pre-trained language model architectures and pooling strategies across all benchmark datasets are presented in Extended Data Section B.

## Overview of optimisation loop

We initialised our Bayesian optimisation workflows using low-discrepancy Sobol sequences [77] to provide broad initial coverage of the design space, which were evaluated to form the initial training data. At each subsequent iteration of the optimisation loop, the surrogate model was fitted to all currently observed data, a batch of candidates was selected by optimising the acquisition function over the remaining design space, and the selected experiments were evaluated and added to the training set.

We used the qLogNParEGO acquisition function [68, 78, 79], a log acquisition function that conditions on noisy baseline observations and applies Chebyshev scalarisation to reduce the multi-objective problem into single-objective subproblems. For batch acquisition, candidates were selected in a greedy sequential fashion, with each selection conditioned on previously chosen pending points.

For surrogate models using fixed input representations (onehot encoding and molecular descriptors), we used an optimised Gaussian process (GP) configuration adapted from EDBO+ [13] and Minerva [20] following standard marginal likelihood optimisation. For our approach using an LLMbased deep kernel surrogate with learned input representations, the GP hyperparameters and language model parameters were optimised jointly as described below. All computational experiments were repeated over 20 random seeds to report statistical variation.

## LLM-based deep kernel Gaussian process surrogate

The LLM first maps tokenised reaction condition prompts to pooled embeddings $\mathbf { e } _ { i } \in \mathbb { R } ^ { d \mathrm { e m b } }$ . For each objective $m = 1 , \ldots , M$ , a trainable projection head $g _ { \phi _ { n } }$ transforms the language model embedding e<sub>i</sub> to an objective-specific representation $\mathbf { z } _ { i } ^ { ( m ) } = g _ { \phi _ { m } } ( \mathbf { e } _ { i } )$ . A Gaussian process (GP) then models the mapping from each projected representation to its corresponding objective value:

$$
\begin{array} { c } { y _ { m } ( \mathbf { x } _ { i } ) = f _ { m } ( \mathbf { z } _ { i } ^ { ( m ) } ) + \epsilon _ { m } , } \\ { f _ { m } \sim \mathcal { G P } ( \mu _ { m } , k _ { \theta _ { m } } ) , } \\ { \epsilon _ { m } \sim \mathcal { N } ( 0 , \sigma _ { m } ^ { 2 } ) } \end{array}\tag{3}
$$

where $f _ { m }$ is the latent function for objective m, modelled as a Gaussian process with constant mean function $\mu _ { m }$ and kernel function $k _ { \theta _ { m } }$ , and $\epsilon _ { m }$ is Gaussian observation noise with variance $\sigma _ { m } ^ { 2 }$ . We use the Matern-5/2 kernel:´

$$
k _ { \theta _ { m } } ( \mathbf { z } , \mathbf { z } ^ { \prime } ) = \sigma _ { f , m } ^ { 2 } \left( 1 + \frac { \sqrt { 5 } \rho } { \ell _ { m } } + \frac { 5 \rho ^ { 2 } } { 3 \ell _ { m } ^ { 2 } } \right) \exp \left( - \frac { \sqrt { 5 } \rho } { \ell _ { m } } \right) ,\tag{4}
$$

where $\sigma _ { f , m } ^ { 2 }$ is the signal variance, and $\ell _ { m }$ is the lengthscale. Together with the constant mean function $\mu _ { m }$ and noise variance $\sigma _ { m } ^ { 2 }$ , these constitute the GP hyperparameters $\theta _ { m } = \{ \mu _ { m } , \sigma _ { f , m } ^ { 2 } , \ell _ { m } , \sigma _ { m } ^ { 2 } \}$ . The kernel matrix for objective m is given by ${ \bf K } _ { m } = k _ { \theta _ { m } } ( { \bf Z } ^ { ( m ) } , { \bf Z } ^ { ( m ) } ) + \sigma _ { m } ^ { 2 } { \bf I }$ , where ${ \mathbf Z } ^ { ( m ) }$ collects the projected embeddings of all observed reaction conditions. The M GPs are queried jointly by the multi-objective acquisition function. The language model $h _ { \phi } .$ , projection heads $\{ g _ { \phi _ { m } } \} _ { m = 1 } ^ { M }$ , and GP hyperparameters $\lbrace \boldsymbol { \theta } _ { m } \rbrace _ { m = 1 } ^ { M }$ are jointly optimised by minimising the sum of negative marginal log-likelihoods:

$$
\mathcal { L } = - \sum _ { m = 1 } ^ { M } \log p ( \mathbf { y } _ { m } \mid \mathbf { Z } ^ { ( m ) } , \boldsymbol { \theta } _ { m } ) ,\tag{5}
$$

where the marginal log-likelihood for objective m is:

$$
\begin{array} { c } { \log p ( \mathbf { y } _ { m } \mid \mathbf { Z } ^ { ( m ) } , \boldsymbol { \theta } _ { m } ) = \displaystyle - \frac { 1 } { 2 } \bigg [ \tilde { \mathbf { y } } _ { m } ^ { \top } \mathbf { K } _ { m } ^ { - 1 } \tilde { \mathbf { y } } _ { m } } \\ { + \log \left| \mathbf { K } _ { m } \right| + n \log 2 \pi \bigg ] } \end{array}\tag{6}
$$

where n is the number of observations and $\tilde { \mathbf { y } } _ { m } = \mathbf y _ { m } - \mu _ { m } \mathbf { 1 }$ denotes the targets centred by the constant mean function. GP targets are standardised to zero mean and unit variance for training. Gradients from all M objectives are backpropagated through the kernel and projection heads to the shared LLM parameters $\phi ,$ where each projection head $g _ { \phi _ { m } }$ receives gradients only from its corresponding objective. This enables the LLM to learn representations that are jointly optimised for GP predictive performance across all objectives, while each projection head specialises for its target objective.

## Parameter-efficient fine-tuning of language models

The language model embeddings of reaction conditions evolve across BO iterations as the LLM is fine-tuned on accumulating observations. To do so efficiently, we apply parameter-efficient fine-tuning (PEFT) with Low-Rank Adaptation (LoRA) [24]. Rather than updating all of the pre-trained language model’s parameters, we use LoRA to update only a small subset. For a pre-trained weight matrix $\mathbf { W } _ { 0 } \in \mathbb { R } ^ { d \times k }$ , LoRA injects trainable low-rank decompositions such that the weight update takes the form:

$$
\begin{array} { r } { \mathbf { W } = \mathbf { W } _ { 0 } + \Delta \mathbf { W } , \quad \quad } { } \\ { \Delta \mathbf { W } = \frac { \alpha } { r } \mathbf { B } \mathbf { A } , \quad \mathrm { w i t h } \ \mathbf { B } \in \mathbb { R } ^ { d \times r } , \mathbf { A } \in \mathbb { R } ^ { r \times k } } \end{array}\tag{7}
$$

where $r \ll \operatorname* { m i n } ( d , k )$ is the rank and α is a scaling hyperparameter. We use LoRA to target a subset of linear projection matrices within the pre-trained language models, using rank r = 4 and α = 16.

## Computational implementation

All models, Bayesian optimisation workflows, and computational analyses were implemented in Python 3.10. We used PyTorch [80] (v2.8.0), GPyTorch [81] (v1.14), and BoTorch [82] (v0.13.0) to construct our multi-objective Bayesian optimisation algorithms. Pretrained language models were loaded using Hugging Face transformers [76] (v4.51.2), with Low-Rank Adaptation (LoRA) implemented using peft (v0.15.1) for parameter-efficient fine-tuning. We used PyTorch Lightning [83] (v2.5.4) for reproducible random seed management and workflow control. Scikit-learn [84] (v1.7.1) and NumPy [85] (v1.26.4) were utilised for machine learning pipelines and numerical data processing, including Principal Component Analysis (PCA) for descriptor dimensionality reduction. Experiment tracking and logging were conducted using Weights & Biases (wandb, v0.21.0) [86]. Matplotlib [87] (v3.10.1) and seaborn [88] (v0.13.2) were used for plotting and visualising all computational results presented in this work. All computations were run on the Roche HPC cluster, using NVIDIA A100 Tensor Core and Blackwell GPUs.

## Experimental procedures

High-throughput experimentation procedures, analytical methods, and scale-up syntheses and characterisation for the palladium-catalysed cyanation and asymmetric ketone hydrogenation reactions are provided in the Supplementary Information.

## Data availability

All benchmark datasets, reaction condition search spaces, and high-throughput experimentation (HTE) experimental data generated in this study are included in the manuscript, Supplementary Information, and on the accompanying public GitHub repository.

## Code availability

The custom code developed for this work is implemented in Python and is made available in a public GitHub repository under the Apache 2.0 licence: https://github.com/schwallergroup/alice

## Acknowledgements

J.W.S., R.P.B., K.P. and R.B. thank Roche and its Technology Innovation and Science (TIS) initiative for financial support. We thank the Global Internship Program in Innovation & Sustainability (IP2TIS) 2025 (https://careers. roche.com/global/en/ip2tis-program) for financial support of this project. D.M.S., B.R., and P.S. acknowledge support from NCCR Catalysis (grant no. 225147), a National Centre of Competence in Research funded by the Swiss National Science Foundation. D.M.S. was funded by the Swiss National Science Foundation (SNSF) [226509].

## Author contributions

J.W.S., B.R., R.B. and P.S. conceived the project. J.W.S., D.M.S. and B.R. developed the computational workflow with input from S.L.C., A.A., R.P.B. and P.S. J.W.S. and D.M.S. performed the computations and trained the models. J.W.S., M.J.N. and R.B. carried out the experimental work. J.W.S., K.P., M.J.N. and R.B. designed and planned the experimental case studies, with input from M.D.R.L. J.W.S., D.M.S. and B.R. analysed the results with help from

M.D.R.L., K.P., M.J.N., R.B. and P.S. S.L.C., A.A., R.P.B., K.P., M.J.N., R.B. and P.S. provided supervision. J.W.S. wrote the paper with input from all authors.

## Competing interests

J.W.S., D.M.S., M.D.R.L., A.A., R.P.B., K.P., M.J.N. and R.B. declare potential financial and non-financial conflicts of interest as full employees of F. Hoffmann-La Roche Ltd. The other authors declare no competing interests.

## References

[1] Connor J. Taylor, Alexander Pomberger, Kobi C. Felton, Rachel Grainger, Magda Barecka, Thomas W. Chamberlain, Richard A. Bourne, Christopher N. Johnson, and Alexei A. Lapkin. A Brief Introduction to Chemical Reaction Optimization. Chemical Reviews, 123(6):3089–3126, 2023. ISSN 0009-2665, 1520-6890. doi: 10.1021/acs.chemrev. 2c00798. URL https://pubs.acs.org/doi/ 10.1021/acs.chemrev.2c00798.

[2] Jonas Duker, Lukas Hebing, Samuel Leweke,¨ Rachel L. Nicholls, Maximilian Lubbesmeyer, Giulio¨ Volpin, Burkhard Konig, and Julius Hillenbrand.¨ Data science-assisted workflow for reaction optimization in process chemistry. Organic Process Research & Development, 30(1):175–188, January 2026. ISSN 1520-586X. doi: 10.1021/acs. oprd.5c00384. URL http://dx.doi.org/10. 1021/acs.oprd.5c00384.

[3] Matthew Ball, Dragos Horvath, Thierry Kogej, Mikhail Kabeshov, and Alexandre Varnek. Predicting reaction conditions: a data-driven perspective. Chemical Science, 16(38):17523–17541, 2025. ISSN 2041-6539. doi: 10.1039/d5sc03045e. URL http: //dx.doi.org/10.1039/D5SC03045E.

[4] David F. Nippa, Alexander J. Boddy, Kenneth Atz, Uwe Grether, Hayley Binch, and Rainer E. Martin. Accelerating compound synthesis in drug discovery: the role of digitalisation and automation. RSC Medicinal Chemistry, 16(12):5753–5764, 2025. ISSN 2632- 8682. doi: 10.1039/d5md00672d. URL http: //dx.doi.org/10.1039/D5MD00672D.

[5] Alexandre A. Schoepfer, Jan Weinreich, Ruben Laplaza, Jerome Waser, and Clemence Corminboeuf. Cost-informed bayesian reaction optimization. Digital Discovery, 3(11):2289–2297, 2024. ISSN 2635- 098X. doi: 10.1039/d4dd00225c. URL http: //dx.doi.org/10.1039/D4DD00225C.

[6] Tony Y. Zhang. Process Chemistry: The Science, Business, Logic, and Logistics. Chemical Reviews, 106(7):2583–2595, 2006. ISSN 0009-2665. doi: 10.1021/cr040677v. URL https://doi.org/10. 1021/cr040677v.

[7] Michael F. Lipton and Anthony G. M. Barrett. Introduction: Process chemistry. Chemical Reviews, 106(7):2581–2582, 2006. ISSN 1520-6890. doi: 10.1021/cr068400d. URL http://dx.doi.org/ 10.1021/cr068400d.

[8] Hui Zhao, Anne K. Ravn, Michael C. Haibach, Keary M. Engle, and Carin C. C. Johansson Seechurn. Diversification of pharmaceutical manufacturing processes: Taking the plunge into the non-pgm catalyst pool. ACS Catalysis, 14(13):9708–9733, 2024. ISSN 2155-5435. doi: 10.1021/ acscatal.4c01809. URL http://dx.doi.org/ 10.1021/acscatal.4c01809.

[9] Coby J. Clarke, Wei-Chien Tu, Oliver Levers, Andreas Brohl, and Jason P. Hallett. Green and sustainable¨ solvents in chemical processes. Chemical Reviews, 118(2):747–800, January 2018. ISSN 1520-6890. doi: 10.1021/acs.chemrev.7b00571. URL http://dx. doi.org/10.1021/acs.chemrev.7b00571.

[10] Benjamin J. S. Rowsell, Harry M. O’Brien, Gayathri Athavan, Patrick R. Daley-Dee, Johannes Krieger, Emma Richards, Karl Heaton, Ian J. S. Fairlamb, and Robin B. Bedford. The iron-catalysed suzuki coupling of aryl chlorides. Nature Catalysis, 7(11): 1186–1198, October 2024. ISSN 2520-1158. doi: 10.1038/s41929-024-01234-0. URL http://dx. doi.org/10.1038/s41929-024-01234-0.

[11] Connor P. Delaney, Eva Lin, Qinan Huang, Isaac F. Yu, Guodong Rao, Lizhi Tao, Ana Jed, Serena M. Fantasia, Kurt A. Puntener, R. David Britt, and John F.¨ Hartwig. Cross-coupling by a noncanonical mechanism involving the addition of aryl halide to cu(ii). Science, 381(6662):1079–1085, 2023. ISSN 1095- 9203. doi: 10.1126/science.adi9226. URL http:// dx.doi.org/10.1126/science.adi9226.

[12] Benjamin J. Shields, Jason Stevens, Jun Li, Marvin Parasram, Farhan Damani, Jesus I. Martinez Alvarado, Jacob M. Janey, Ryan P. Adams, and Abigail G. Doyle. Bayesian reaction optimization as a tool for chemical synthesis. Nature, 590(7844):89–96, 2021. ISSN 0028-0836, 1476-4687. doi: 10.1038/s41586-021-03213-y. URL https://www.nature.com/articles/ s41586-021-03213-y.

[13] Jose Antonio Garrido Torres, Sii Hong Lau, Pranay Anchuri, Jason M. Stevens, Jose E. Tabora, Jun Li, Alina Borovika, Ryan P. Adams, and Abigail G. Doyle. A Multi-Objective Active Learning Platform and Web App for Reaction Optimization. Journal of the American Chemical Society, 144(43):19999– 20007, 2022. ISSN 0002-7863, 1520-5126. doi: 10.1021/jacs.2c08592. URL https://pubs.acs. org/doi/10.1021/jacs.2c08592.

[14] Elena Braconi and Edouard Godineau. Bayesian optimization as a sustainable strategy for early-stage

process development? a case study of cu-catalyzed c–n coupling of sterically hindered pyrazines. ACS Sustainable Chemistry & Engineering, 11(28): 10545–10554, 2023. ISSN 2168-0485. doi: 10.1021/ acssuschemeng.3c02455. URL http://dx.doi. org/10.1021/acssuschemeng.3c02455.

[15] Natalie P. Romer, Daniel S. Min, Jason Y. Wang, Richard C. Walroth, Kyle A. Mack, Lauren E. Sirois, Francis Gosselin, Daniel Zell, Abigail G. Doyle, and Matthew S. Sigman. Data science guided multiobjective optimization of a stereoconvergent nickelcatalyzed reduction of enol tosylates to access trisubstituted alkenes. ACS Catalysis, 14(7):4699–4708, March 2024. ISSN 2155-5435. doi: 10.1021/ acscatal.4c00650. URL http://dx.doi.org/ 10.1021/acscatal.4c00650.

[16] Jamie A. Cadge, Cedric Lozano, Morgan T. Merriman, Paul Oblad, Matthew S. Sigman, and Sarah E. Reisman. A data science-guided approach for the development of nickel-catalyzed homo-diels–alder reactions. Journal of the American Chemical Society, 147(34):31175–31186, August 2025. ISSN 1520-5126. doi: 10.1021/jacs.5c09948. URL http: //dx.doi.org/10.1021/jacs.5c09948.

[17] Jiyizhe Zhang, Naoto Sugisawa, Kobi C. Felton, Shinichiro Fuse, and Alexei A. Lapkin. Multiobjective bayesian optimisation using q -noisy expected hypervolume improvement ( q nehvi) for the schotten–baumann reaction. Reaction Chemistry & Engineering, 9(3):706–712, 2024. ISSN 2058-9883. doi: 10.1039/d3re00502j. URL http://dx.doi. org/10.1039/D3RE00502J.

[18] Connor J. Taylor, Kobi C. Felton, Daniel Wigh, Mohammed I. Jeraal, Rachel Grainger, Gianni Chessari, Christopher N. Johnson, and Alexei A. Lapkin. Accelerated chemical reaction optimization using multi-task learning. ACS Central Science, 9(5): 957–968, April 2023. ISSN 2374-7951. doi: 10.1021/ acscentsci.3c00050. URL http://dx.doi.org/ 10.1021/acscentsci.3c00050.

[19] John H. Dunlap, Jeffrey G. Ethier, Amelia A. Putnam-Neeb, Sanjay Iyer, Shao-Xiong Lennon Luo, Haosheng Feng, Jose Antonio Garrido Torres, Abigail G. Doyle, Timothy M. Swager, Richard A. Vaia, Peter Mirau, Christopher A. Crouse, and Luke A. Baldwin. Continuous flow synthesis of pyridinium salts accelerated by multi-objective bayesian optimization with active learning. Chemical Science, 14 (30):8061–8069, 2023. ISSN 2041-6539. doi: 10. 1039/d3sc01303k. URL http://dx.doi.org/ 10.1039/D3SC01303K.

[20] Joshua W. Sin, Siu Lun Chau, Ryan P. Burwood, Kurt Puntener, Raphael Bigler, and Philippe Schwaller.¨ Highly parallel optimisation of chemical reactions through automation and machine intelligence. Nature Communications, 16(1):6464, 2025. ISSN 2041-1723. doi: 10.1038/s41467-025-61803-0. URL https://www.nature.com/articles/ s41467-025-61803-0.

[21] Remi Schlama, Joshua W. Sin, Ryan P. Bur-´ wood, Kurt Puntener, Raphael Bigler, and Philippe¨ Schwaller. Swarm intelligence for chemical reaction optimization. Chem, page 103035, April 2026. ISSN 2451-9294. doi: 10.1016/j.chempr.2026. 103035. URL http://dx.doi.org/10.1016/ j.chempr.2026.103035.

[22] W. Kohn, A. D. Becke, and R. G. Parr. Density functional theory of electronic structure. The Journal of Physical Chemistry, 100(31):12974–12980, January 1996. ISSN 1541-5740. doi: 10.1021/jp960669l. URL http://dx.doi.org/10.1021/jp960669l.

[23] Christoph Bannwarth, Sebastian Ehlert, and Stefan Grimme. Gfn2-xtb—an accurate and broadly parametrized self-consistent tight-binding quantum chemical method with multipole electrostatics and density-dependent dispersion contributions. Journal of Chemical Theory and Computation, 15(3): 1652–1671, February 2019. ISSN 1549-9626. doi: 10.1021/acs.jctc.8b01176. URL http://dx.doi. org/10.1021/acs.jctc.8b01176.

[24] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https://arxiv.org/ abs/2106.09685.

[25] Tobias Gensch, Gabriel dos Passos Gomes, Pascal Friederich, Ellyn Peters, Theophile Gaudin, Robert´ Pollice, Kjell Jorner, AkshatKumar Nigam, Michael Lindner-D’Addario, Matthew S. Sigman, and Alan´ Aspuru-Guzik. A Comprehensive Discovery Platform for Organophosphorus Ligands for Catalysis. Journal ofthe American Chemical Society, 144(3):1205– 1217, 2022. ISSN 0002-7863. doi: 10.1021/jacs. 1c09718. URL https://doi.org/10.1021/ jacs.1c09718.

[26] Laurianne Moity, Morgan Durand, Adrien Benazzouz, Christel Pierlot, Valerie Molinier, and´ Jean-Marie Aubry. Panorama of sustainable solvents using the COSMO-RS approach. Green Chemistry, 14(4):1132–1145, 2012. ISSN 1463-9270. doi: 10.1039/C2GC16515E. URL

https://pubs.rsc.org/en/content/ articlelanding/2012/gc/c2gc16515e.

[27] Derek M. Dalton, Richard C. Walroth, Caroline Rouget-Virbel, Kyle A. Mack, and F. Dean Toste. Utopia Point Bayesian Optimization Finds Condition-Dependent Selectivity for N-Methyl Pyrazole Condensation. Journal of the American Chemical Society, 146(23):15779–15786, 2024. ISSN 0002-7863. doi: 10.1021/jacs.4c01616. URL https://doi.org/ 10.1021/jacs.4c01616.

[28] Jamie A. Cadge, Sierra D. Hart, Richard C. Walroth, Kyle A. Mack, and Matthew S. Sigman. Bisphosphine ligand conformer selection to enhance descriptor database representation: improving statistical modelling outcomes. Chemical Science, 16(43): 20473–20485, 2025. ISSN 2041-6539. doi: 10. 1039/d5sc04691b. URL http://dx.doi.org/ 10.1039/D5SC04691B.

[29] Alexander S. Shved, Blake E. Ocampo, Elena S. Burlova, Casey L. Olen, N. Ian Rinehart, and Scott E. Denmark. molli: A general purpose python toolkit for combinatorial small molecule library generation, manipulation, and feature extraction. Journal ofChemical Information and Modeling, 64(21):8083–8090, October 2024. ISSN 1549-960X. doi: 10.1021/acs. jcim.4c00424. URL http://dx.doi.org/10. 1021/acs.jcim.4c00424.

[30] Melodie Christensen, Lars P. E. Yunker, Folarin Adedeji, Florian Hase, Lo¨ ¨ıc M. Roch, Tobias Gensch, Gabriel dos Passos Gomes, Tara Zepel, Matthew S. Sigman, Alan Aspuru-Guzik, and Ja-´ son E. Hein. Data-science driven autonomous process optimization. Communications Chemistry, 4 (1), August 2021. ISSN 2399-3669. doi: 10. 1038/s42004-021-00550-x. URL http://dx.doi. org/10.1038/s42004-021-00550-x.

[31] Mohammad H. Samha, Lucas J. Karas, David B. Vogt, Emmanuel C. Odogwu, Jennifer Elward, Jennifer M. Crawford, Janelle E. Steves, and Matthew S. Sigman. Predicting success in cu-catalyzed c–n coupling reactions using data science. Science Advances, 10(3), January 2024. ISSN 2375-2548. doi: 10.1126/sciadv.adn3478. URL http://dx.doi. org/10.1126/sciadv.adn3478.

[32] Jordan J. Dotson, Lucy van Dijk, Jacob C. Timmerman, Samantha Grosslight, Richard C. Walroth, Francis Gosselin, Kurt Puntener, Kyle A. Mack, and¨ Matthew S. Sigman. Data-driven multi-objective optimization tactics for catalytic asymmetric reactions using bisphosphine ligands. Journal of the

American Chemical Society, 145(1):110–121, December 2022. ISSN 1520-5126. doi: 10.1021/ jacs.2c08513. URL http://dx.doi.org/10. 1021/jacs.2c08513.

[33] Lucas W. Souza, Nathan D. Ricke, Braden C. Chaffin, Mike E. Fortunato, Shutian Jiang, Cihan Soylu, Thomas C. Caya, Sii Hong Lau, Katherine A. Wieser, Abigail G. Doyle, and Kian L. Tan. Applying active learning toward building a generalizable model for ni-photoredox cross-electrophile coupling of aryl and alkyl bromides. Journal of the American Chemical Society, 147(22):18747–18759, May 2025. ISSN 1520-5126. doi: 10.1021/jacs.5c02218. URL http: //dx.doi.org/10.1021/jacs.5c02218.

[34] Shivaani S. Gandhi, Giselle Z. Brown, Santeri Aikonen, Jordan S. Compton, Paulo Neves, Jesus I. Martinez Alvarado, Iulia I. Strambeanu, Kristi A. Leonard, and Abigail G. Doyle. Data science-driven discovery of optimal conditions and a condition-selection model for the chan–lam coupling of primary sulfonamides. ACS Catalysis, 15(3):2292–2304, January 2025. ISSN 2155-5435. doi: 10.1021/ acscatal.4c07972. URL http://dx.doi.org/ 10.1021/acscatal.4c07972.

[35] N. Ian Rinehart, Rakesh K. Saunthwal, Joel Wellauer,¨ Andrew F. Zahrt, Lukas Schlemper, Alexander S. Shved, Raphael Bigler, Serena Fantasia, and Scott E. Denmark. A machine-learning tool to predict substrateadaptive conditions for pd-catalyzed c–n couplings. Science, 381(6661):965–972, 2023. ISSN 1095-9203. doi: 10.1126/science.adg2114. URL http://dx. doi.org/10.1126/science.adg2114.

[36] Arnau Call, Andrea Palone, Jordan P. Liles, Natalie P. Romer, Jacquelyne A. Read, Josep M. Luis, Matthew S. Sigman, Massimo Bietti, and Miquel Costas. Understanding catalytic enantioselective c–h bond oxidation at nonactivated methylenes through predictive statistical modeling analysis. ACS Catalysis, 15(3):2110–2123, January 2025. ISSN 2155-5435. doi: 10.1021/acscatal.4c05659. URL http://dx. doi.org/10.1021/acscatal.4c05659.

[37] Blake E. Ocampo, Bilal Altundas, Matthew J. Bock, Sara Feiz, and Scott E. Denmark. Data-driven prediction of enantioselectivity for the sharpless asymmetric dihydroxylation: Model development and experimental validation. ACS Central Science, 11(9): 1640–1650, 2025. ISSN 2374-7951. doi: 10.1021/ acscentsci.5c00900. URL http://dx.doi.org/ 10.1021/acscentsci.5c00900.

[38] Bojana Rankovic, Ryan-Rhys Griffiths, Henry B.´ Moss, and Philippe Schwaller. Bayesian optimisa-

tion for additive screening and yield improvements – beyond one-hot encoding. Digital Discovery, 3 (4):654–666, 2024. ISSN 2635-098X. doi: 10. 1039/d3dd00096f. URL http://dx.doi.org/ 10.1039/D3DD00096F.

[39] Islambek Ashyrmamatov, Su Ji Gwak, Su-Young Jin, Ikhyeong Jun, Umit V. Ucak, Jay-Yoon Lee, and Juyong Lee. A survey on large language models in biology and chemistry. Experimental & Molecular Medicine, April 2026. ISSN 2092-6413. doi: 10.1038/s12276-025-01583-1. URL http://dx. doi.org/10.1038/s12276-025-01583-1.

[40] Alexandru Oarga, Matthew Hart, Andres M. Bran, Magdalena Lederbauer, and Philippe Schwaller. Scientific knowledge graph and ontology generation using open large language models. Digital Discovery, 5(3):1269–1279, 2026. ISSN 2635-098X. doi: 10. 1039/d5dd00275c. URL http://dx.doi.org/ 10.1039/D5DD00275C.

[41] Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D. White, and Philippe Schwaller. Augmenting large language models with chemistry tools. Nature Machine Intelligence, 6(5):525–535, May 2024. ISSN 2522-5839. doi: 10.1038/ s42256-024-00832-8. URL http://dx.doi. org/10.1038/s42256-024-00832-8.

[42] Yoel Zimmermann, Adib Bazgir, Alexander Al-Feghali, Mehrad Ansari, Joshua Bocarsly, L Catherine Brinson, Yuan Chiang, Defne Circi, Min-Hsueh Chiu, Nathan Daelman, Matthew L Evans, Abhijeet S Gangan, Janine George, Hassan Harb, Ghazal Khalighinejad, Sartaaj Takrim Khan, Sascha Klawohn, Magdalena Lederbauer, Soroush Mahjoubi, Bernadette Mohr, Seyed Mohamad Moosavi, Aakash Naik, Aleyna Beste Ozhan, Dieter Plessers, Aritra Roy, Fabian Schoppach, Philippe Schwaller, Carla¨ Terboven, Katharina Ueltzen, Yue Wu, Shang Zhu, Jan Janssen, Calvin Li, Ian Foster, and Ben Blaiszik. 32 examples of llm applications in materials science and chemistry: towards automation, assistants, agents, and accelerated scientific discovery. Machine Learning: Science and Technology, 6(3):030701, 2025. ISSN 2632-2153. doi: 10.1088/2632-2153/ ae011a. URL http://dx.doi.org/10.1088/ 2632-2153/ae011a.

[43] Jieyu Lu and Yingkai Zhang. Unified deep learning model for multitask reaction predictions with explanation. Journal of Chemical Information and Modeling, 62(6):1376–1387, March 2022. ISSN 1549-960X. doi: 10.1021/acs.jcim.1c01467. URL http://dx.doi. org/10.1021/acs.jcim.1c01467.

[44] Nawaf Alampara, Anagha Aneesh, Martino R˜ ´ıos-Garc´ıa, Adrian Mirza, Mara Schilling-Wilhelmi, Ali Asghar Aghajani, Meiling Sun, Gordan Prastalo, and Kevin Maik Jablonka. General-purpose models for the chemical sciences: Llms and beyond. Chemical Reviews, 126(4):2484–2549, February 2026. ISSN 1520-6890. doi: 10.1021/acs. chemrev.5c00583. URL http://dx.doi.org/ 10.1021/acs.chemrev.5c00583.

[45] Andres M. Bran, Theo A. Neukomm, Daniel Arm-´ strong, Zlatko Joncev, and Philippe Schwaller. Chem-ˇ ical reasoning in llms unlocks strategy-aware synthesis planning and reaction mechanism elucidation. Matter, page 102812, April 2026. ISSN 2590-2385. doi: 10.1016/j.matt.2026.102812. URL http://dx. doi.org/10.1016/j.matt.2026.102812.

[46] Daniel Armstrong, Zlatko Joncev, Andres M Bran,ˇ and Philippe Schwaller. Synthstrategy: Extracting and formalizing latent strategic insights from llms in organic chemistry, 2025. URL https://arxiv. org/abs/2512.01507.

[47] Ewa Wieczorek, Joshua W. Sin, Sara Tanovic, Matthew T. O. Holland, Liam Wilbraham, Victor Sebastian-P ´ erez, Anthony Bradley, Dominik Miketa,´ Paul E. Brennan, and Fernanda Duarte. Transfer learning for heterocycle retrosynthesis. Journal of Chemical Information and Modeling, 65(15): 7851–7861, 2025. ISSN 1549-960X. doi: 10.1021/ acs.jcim.4c02041. URL http://dx.doi.org/ 10.1021/acs.jcim.4c02041.

[48] Jerret Ross, Brian Belgodere, Vijil Chenthamarakshan, Inkit Padhi, Youssef Mroueh, and Payel Das. Large-scale chemical language representations capture molecular structure and properties. Nature Machine Intelligence, 4(12):1256–1264, December 2022. ISSN 2522-5839. doi: 10.1038/ s42256-022-00580-7. URL http://dx.doi. org/10.1038/s42256-022-00580-7.

[49] Seyone Chithrananda, Gabriel Grand, and Bharath Ramsundar. Chemberta: Largescale self-supervised pretraining for molecular property prediction, 2020. URL https://arxiv.org/abs/2010.09885.

[50] Kevin Maik Jablonka, Philippe Schwaller, Andres Ortega-Guerrero, and Berend Smit. Leveraging large language models for predictive chemistry. Nature Machine Intelligence, 6(2):161–169, February 2024. ISSN 2522-5839. doi: 10.1038/ s42256-023-00788-1. URL http://dx.doi. org/10.1038/s42256-023-00788-1.

[51] David Ming Segura, Jeremy Goumaz, Joshua W. Sin, Bojana Rankovic, and Philippe Schwaller. Bi-semantic´ chemical embedder for joint representation learning of smiles and natural language, 2026. URL https: //arxiv.org/abs/2608.03855.

[52] Bojana Rankovic, Ryan-Rhys Griffiths, and Philippe´ Schwaller. Large language models as uncertaintycalibrated optimizers for experimental discovery, 2025. URL https://arxiv.org/abs/2504. 06265.

[53] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 21(140):1–67, 2020. URL http://jmlr.org/papers/v21/ 20-074.html.

[54] Vincent Porte, Luca Hepp, Philipp Kollmus, Shizhao Lu, Eloisa Serrano, Daniela Blanco, and Marco Santagostino. Frugal sampling strategies for navigating complex reaction spaces. Organic Process Research & Development, April 2026. ISSN 1520-586X. doi: 10.1021/acs.oprd.6c00027. URL http://dx.doi. org/10.1021/acs.oprd.6c00027.

[55] Julian Gotz, Moritz K. Jackl, Chalupat Jindakun,¨ Alexander N. Marziale, Jer´ ome Andrˆ e, Daniel J.´ Gosling, Clayton Springer, Marco Palmieri, Marcel Reck, Alexandre Luneau, Cara E. Brocklehurst, and Jeffrey W. Bode. High-throughput synthesis provides data for predicting molecular properties and reaction success. Science Advances, 9(43), October 2023. ISSN 2375-2548. doi: 10.1126/sciadv.adj2314. URL http: //dx.doi.org/10.1126/sciadv.adj2314.

[56] Julian Gotz, Euan Richards, Iain A. Stepek, Yu Taka-¨ hashi, Yi-Lin Huang, Louis Bertschi, Bertran Rubi, and Jeffrey W. Bode. Predicting three-component reaction outcomes from 40, 000 miniaturized reactant combinations. Science Advances, 11(22), May 2025. ISSN 2375-2548. doi: 10.1126/ sciadv.adw6047. URL http://dx.doi.org/10. 1126/sciadv.adw6047.

[57] Babak Mahjour, Rui Zhang, Yuning Shen, Andrew McGrath, Ruheng Zhao, Osama G. Mohamed, Yingfu Lin, Zirong Zhang, James L. Douthwaite, Ashootosh Tripathi, and Tim Cernak. Rapid planning and analysis of high-throughput experiment arrays for reaction discovery. Nature Communications, 14(1), 2023. ISSN 2041-1723. doi: 10. 1038/s41467-023-39531-0. URL http://dx.doi. org/10.1038/s41467-023-39531-0.

[58] Mohan Neetha, C. M. A. Afsina, Thaipparambil Aneeja, and Gopinathan Anilkumar. Recent advances and prospects in the palladium-catalyzed cyanation of aryl halides. RSC Advances, 10(56): 33683–33699, 2020. ISSN 2046-2069. doi: 10. 1039/d0ra05960a. URL http://dx.doi.org/ 10.1039/d0ra05960a.

[59] Daniel T. Cohen and Stephen L. Buchwald. Mild palladium-catalyzed cyanation of (hetero)aryl halides and triflates in aqueous media. Organic Letters, 17 (2):202–205, January 2015. ISSN 1523-7052. doi: 10.1021/ol5032359. URL http://dx.doi.org/ 10.1021/ol5032359.

[60] Fraser F. Fleming, Lihua Yao, P. C. Ravikumar, Lee Funk, and Brian C. Shook. Nitrile-containing pharmaceuticals: Efficacious roles of the nitrile pharmacophore. Journal of Medicinal Chemistry, 53(22): 7902–7917, August 2010. ISSN 1520-4804. doi: 10.1021/jm100762r. URL http://dx.doi.org/ 10.1021/jm100762r.

[61] Pazhamalai Anbarasan, Thomas Schareina, and Matthias Beller. Recent developments and perspectives in palladium-catalyzed cyanation of aryl halides: synthesis of benzonitriles. Chemical Society Reviews, 40(10):5049, 2011. ISSN 1460-4744. doi: 10.1039/c1cs15004a. URL http://dx.doi.org/ 10.1039/c1cs15004a.

[62] Alexander M. Nauth and Till Opatz. Non-toxic cyanide sources and cyanating agents. Organic & Biomolecular Chemistry, 17(1):11–23, 2019. ISSN 1477-0539. doi: 10.1039/c8ob02140f. URL http: //dx.doi.org/10.1039/c8ob02140f.

[63] Nicolas A. Wilson, William M. Palmer, Meredith K. Slimp, Eric M. Simmons, Matthew V. Joannou, Jennifer Albaneze-Walker, Jacob M. Ganley, and Doug E. Frantz. Ni-catalyzed cyanation of (hetero)aryl electrophiles using the nontoxic cyanating reagent K<sub>4</sub>[Fe(CN)<sub>6</sub>]. ACS Catalysis, 15(8): 6459–6465, April 2025. ISSN 2155-5435. doi: 10.1021/acscatal.5c00158. URL http://dx.doi. org/10.1021/acscatal.5c00158.

[64] H.U. Blaser, F. Spindler, and M. Studer. Enantioselective catalysis in fine chemicals production. Applied Catalysis A: General, 221(1-2):119–143, November 2001. ISSN 0926-860X. doi: 10.1016/ s0926-860x(01)00801-8. URL http://dx.doi. org/10.1016/S0926-860X(01)00801-8.

[65] R. Noyori, T. Ikeda, T. Ohkuma, M. Widhalm, M. Kitamura, H. Takaya, S. Akutagawa, N. Sayo, T. Saito,

T. Taketomi, and H. Kumobayashi. Stereoselective hydrogenation via dynamic kinetic resolution. Journal of the American Chemical Society, 111(25):9134–9135, December 1989. ISSN 1520-5126. doi: 10.1021/ ja00207a038. URL http://dx.doi.org/10. 1021/ja00207a038.

[66] Taiga Yurino, Ryo Nishihara, Toshihisa Yasuda, Shuangli Yang, Noriyuki Utsumi, Takeaki Katayama, Noriyoshi Arai, and Takeshi Ohkuma. Asymmetric hydrogenation of <sub>α</sub>-alkyl-substituted β-keto esters and amides through dynamic kinetic resolution. Organic Letters, 26(14):2872–2876, January 2024. ISSN 1523-7052. doi: 10.1021/acs. orglett.3c04036. URL http://dx.doi.org/10. 1021/acs.orglett.3c04036.

[67] Samuel Daulton, Maximilian Balandat, and Eytan Bakshy. Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization. In Advances in Neural Information Processing Systems, volume 33, pages 9851–9864. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper\_files/paper/2020/hash/ 6fec24eac8f18ed793f5eaad3dd7977c-Abs html.

[68] Samuel Daulton, Maximilian Balandat, and Eytan Bakshy. Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement, 2021. URL http://arxiv.org/ abs/2105.08195.

[69] Andreia P. Guerreiro, Carlos M. Fonseca, and Lu´ıs Paquete. The Hypervolume Indicator: Problems and Algorithms. ACM Computing Surveys, 54(6):1–42, 2022. ISSN 0360-0300, 1557-7341. doi: 10.1145/ 3453474. URL http://arxiv.org/abs/2005. 00515.

[70] Charles Audet, Jean Bigeon, Dominique Cartier, Sebastien Le Digabel, and Ludovic Salomon. Per-´ formance indicators in multiobjective optimization. European Journal of Operational Research, 292(2): 397–422, 2021. ISSN 0377-2217. doi: 10.1016/ j.ejor.2020.11.016. URL http://dx.doi.org/ 10.1016/j.ejor.2020.11.016.

[71] Andrzej Mackiewicz and Waldemar Ratajczak.´ Principal components analysis (pca). Computers & Geosciences, 19(3):303–342, March 1993. ISSN 0098-3004. doi: 10.1016/0098-3004(93) 90090-r. URL http://dx.doi.org/10.1016/ 0098-3004(93)90090-R.

[72] Dimitrios Christofidellis, Giorgio Giannone, Jannis Born, Ole Winther, Teodoro Laino, and Matteo Manica. Unifying molecular and textual representations via multi-task language modelling. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 6140–6157. PMLR, 2023. URL https://proceedings.mlr. press/v202/christofidellis23a.html.

[73] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv <sub>act.</sub>preprint arXiv:2407.10671, 2024.

[74] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.

[75] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. CoRR, abs/1910.13461, 2019. URL http: //arxiv.org/abs/1910.13461.

[76] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtow-´ icz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Huggingface’s transformers: State-of-the-art natural language processing, 2019. URL https://arxiv.org/ abs/1910.03771.

[77] Sebastian Burhenne, Dirk Jacob, and Gregor P Henze. Sampling based on Sobol’ sequences for Monte Carlo techniques applied to building simulations. Proceedings of Building Simulation 2011: 12th Conference of International Building Performance Simulation Association, pages 1816–1823, 2011.

[78] Sebastian Ament, Samuel Daulton, David Eriksson, Maximilian Balandat, and Eytan Bakshy. Unexpected improvements to expected improvement for bayesian optimization, 2023. URL https://arxiv.org/ abs/2310.20708.

[79] Simone Pilon, Elia Savino, Oliver M. Bayley, Michael Vanzella, Miguel Claros, Petros Siasiaridis, Junsong Liu, Florian Lukas, Matteo Damian, Vasilis Tseliou, Niccolo Intini, Aidan Slattery, Jesus SanJos\` e-´ Orduna, Tim den Hartog, Ron A. H. Peters, Andrea F. G. Gargano, Francesco G. Mutti, and Timothy Noel. A flexible and affordable self-driving ¨ laboratory for automated reaction optimization. Nature Synthesis, April 2026. ISSN 2731-0582. doi: 10.1038/s44160-026-01053-0. URL http://dx. doi.org/10.1038/s44160-026-01053-0.

[80] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zach ¨ DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, highperformance deep learning library. In Proceedings ofthe 33rd International Conference on Neural Information Processing Systems, number 721, pages 8026– 8037. Curran Associates Inc., 2019.

[81] Jacob R. Gardner, Geoff Pleiss, David Bindel, Kilian Q. Weinberger, and Andrew Gordon Wilson. GPy-Torch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, pages 7587–7597. Curran Associates Inc., 2018.

[82] Maximilian Balandat, Brian Karrer, Daniel Jiang, [88] Samuel Daulton, Ben Letham, Andrew G Wilson, and Eytan Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. In Advances in Neural Information Processing Systems, volume 33, pages 21524–21538. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper\_files/paper/2020/hash/ f5b1b89d98b7286673128a5fb112cb9a-Abstract. html.

[83] William Falcon, Jirka Borovec, Adrian Walchli, Nic¨ Eggert, Justus Schock, Jeremy Jordan, Nicki Skafte, Ir1dXD, Vadim Bereznyuk, Ethan Harris, Tullie Murrell, Peter Yu, Sebastian Præsius, Travis Addair, Jacob Zhong, Dmitry Lipin, So Uchida, Shreyas Bapat, Hendrik Schroter, Boris Dayma, Alexey Karnachev,¨

Akshay Kulkarni, Shunta Komatsu, Martin.B, Jean-Baptiste SCHIRATTI, Hadrien Mary, Donal Byrne, Cristobal Eyzaguirre, cinjon, and Anton Bakhtin. PyTorchLightning/pytorch-lightning: 0.7.6 release, 2020. URL https://zenodo.org/records/ 3828935.

[84] Fabian Pedregosa, Gael Varoquaux, Alexandre Gram-¨ fort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Edouard Duchesnay. Scikit-learn: Machine Learn-<sup>´</sup> ing in Python. J. Mach. Learn. Res., 12:2825–2830, 2011. ISSN 1532-4435.

[85] Charles R. Harris, K. Jarrod Millman, Stefan J.´ van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernandez del R´ ´ıo, Mark Wiebe, Pearu Peterson, Pierre Gerard-Marchant, Kevin´ Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. Array programming with NumPy. Nature, 585 (7825):357–362, September 2020. doi: 10.1038/ s41586-020-2649-2. URL https://doi.org/ 10.1038/s41586-020-2649-2.

[86] Lukas Biewald et al. Experiment tracking with weights and biases. 2020.

[87] John D. Hunter. Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3):90– 95, 2007. ISSN 1558-366X. doi: 10.1109/MCSE.2007. 55. URL https://ieeexplore.ieee.org/ document/4160265.

Michael L. Waskom. Seaborn: Statistical data visualization. Journal of Open Source Software, 6 (60):3021, 2021. ISSN 2475-9066. doi: 10.21105/ joss.03021. URL https://joss.theoj.org/ papers/10.21105/joss.03021.

![](images/7f18032e2d22625d28017bdde21cfb486bd0f658388c4dbe47391694866cc7c0.jpg)  
Nickel-catalysed Suzuki coupling dataset

## A. Supplemental main text benchmarking studies

![](images/7ab1a0f8e33ddb234b265dbfeef4a529cf6254445dffdd3f6f8a8669ae28998a.jpg)  
Palladium-catalysed Suzuki coupling dataset

![](images/35f111fe1ecc431917a7fb1d61fdda7eff3ad88e68f1f1737a55b288af69f489.jpg)

![](images/464131c054fd75c2109a302a256f8a5fe17dc7314078490226926b0efea42d8e.jpg)  
LLM-GP DFT Descriptor databases One-hot encoding

Extended Data Figure 1. Sequential low-data optimisation benchmarks comparing different representation methods (the LLM-GP framework, DFT descriptor databases, and one-hot encoding) on the nickel-catalysed and palladium-catalysed Suzuki coupling datasets. Top panels: mean number of optimisation iterations required to reach 90% and 95% hypervolume thresholds (lower is better, error bars indicate standard error across 20 random seeds). Bottom panels: percentage of optimisation runs (initialised with different random seeds) reaching each threshold within the allocated experimental budget (higher is better).

Batch size 24 HTE optimisation

![](images/55297c34511b008f9df8de89df85e6689451bdbcce0690edeedfed845edefff8.jpg)  
Batch size 48 HTE optimisation

![](images/7c232ac36b54d3bcdd1e4de61c9dfd7b6c21a92bcbfc1ae7ad86a3a8a9af52e6.jpg)  
Batch size 96 HTE optimisation

![](images/7ab6c3b1d70854a32a7846939a0264b66b2c8f5e71d6d29e2bb4593afb8d7438.jpg)

![](images/c971abd214657f41615ac9b56c5966fd502fbe4471afb8c97c73b80be7954eb0.jpg)

![](images/491f818047b6ec05ce6d8e2b5f5a135a8a48121f7b96f58469991628c947d4ee.jpg)

![](images/0c87cf8360667ddc8fbee018081e06be85a9429b3bc0e0ce531953b2567d67e4.jpg)  
LLM-GP DFT Descriptor databases One-hot encoding

Extended Data Figure 2. High-throughput experimentation (HTE) optimisation benchmarks on the palladium-catalysed sulfonamide coupling dataset, comparing the LLM-GP framework, DFT descriptor databases, and one-hot encoding across 24-well (left), 48-well (middle), and 96-well (right) plate batch sizes. Top panels: mean number of plate iterations required to reach 90% and 95% hypervolume thresholds (lower is better, error bars indicate standard error across 20 random seeds). Bottom panels: percentage of optimisation runs (initialised with different random seeds) reaching each threshold within the allocated experimental budget (higher is better).

## B. Influence of different pre-trained models and pooling strategies

Nickel-catalysed Suzuki coupling dataset

![](images/15c747b84db7eae4f252e6f39d4eea470c61c38d5217ee2408070126255abdc7.jpg)

Palladium-catalysed Suzuki coupling dataset  
![](images/8573190f312ad0e338431582454f05417faa28739f635a4ef6946077faa64af9.jpg)

Extended Data Figure 3. Ablation studies over pre-trained language model architectures and pooling strategies on the nickel-catalysed (top) and palladium-catalysed (bottom) Suzuki coupling datasets. Mean number of optimisation iterations required to reach 90% and 95% hypervolume thresholds are compared across T5-chem (mean pooling), BART-base (mean and CLS pooling), T5-base and T5-small (mean pooling), and Qwen2.5-0.5B (mean and last token pooling). Error bars indicate standard error across 20 random seeds. T5-base with mean pooling consistently achieved practical hypervolume thresholds within the fewest iterations across all datasets and thresholds.

Nickel-catalysed Suzuki coupling dataset  
![](images/8a990ce27b8b75fc04ad0ca8b4e042929e8b31af859f4cb99f6bf2f510c6277d.jpg)

Palladium-catalysed Suzuki coupling dataset  
![](images/3d76b76f094591a1012f96723f5dcda0264560fe2ec7e5f93871896e2b67c1e5.jpg)  
Extended Data Figure 4. Percentage of optimisation runs (initialised with 20 different random seeds) reaching 90% and 95% hypervolume thresholds within the allocated experimental budget, across pre-trained language model architectures and pooling strategies on the nickel-catalysed (top) and palladium-catalysed (bottom) Suzuki coupling datasets. Models and pooling strategies are as described in the Methods.

Batch size 24 HTE optimisation

![](images/cf0e4964909ce3d8013696007a15c95986b4d5e81baf0678eab2cdfddbfc51b7.jpg)

Batch size 48 HTE optimisation  
![](images/1b237446a66a816269c21d2cda2ac6938897d29c528bdf602245d4f35dbe7255.jpg)

Batch size 96 HTE optimisation  
![](images/85613ecb2f5a908db920f654d1a37200ebaa7d1bf5c8dbe69795f506901f3b6c.jpg)

Extended Data Figure 5. Ablation studies over pre-trained language model architectures and pooling strategies for high-throughput experimentation (HTE) optimisation on the palladium-catalysed sulfonamide coupling dataset, across 24-well (top), 48-well (middle), and 96-well (bottom) plate batch sizes. The bar charts show the mean number of plate iterations required to reach 90% and 95% hypervolume thresholds. Models and pooling strategies are as described in the Methods. Error bars indicate standard error across 20 random seeds.

Batch size 24 HTE optimisation

![](images/732a2d72d611d422be637d02f2f08e2c4babe27cd6fd70fecd907b0e8c386828.jpg)

Batch size 48 HTE optimisation  
![](images/d689250fdf070f7c41576390325a2a3cf0db1b015f978545edc4f9afba938095.jpg)  
Batch size 96 HTE optimisation

![](images/0671db565a1cce07bccd128646abdc41f157531f7a7cd5ff1ed7f1b03f9fde3b.jpg)

Extended Data Figure 6. Percentage of optimisation runs (initialised with 20 different random seeds) reaching 90% and 95% hypervolume thresholds within the allocated experimental budget, across pre-trained language model architectures and pooling strategies for high throughput experimentation (HTE) optimisation benchmarks on the palladium-catalysed sulfonamide coupling dataset, across 24-well (top), 48-well (middle), and 96-well (bottom) plate batch sizes. Models and pooling strategies are as described in the Methods.

## C. Design spaces for prospective optimisation campaigns

![](images/938408be5ba3c343003210f66e18b32e5dad088ced6df2ca86576ed96dac3766.jpg)  
Extended Data Figure 7. Reaction condition search space for the prospective Pd-catalysed cyanation reaction comprising 40 mono- and bidentate Pd catalysts, 10 solvents, 3 cyanide sources, 2 co-solvents, 5 additives, and 3 temperatures.

![](images/ee44bb31c191dc7947a6a039b4cc22ac404035b7327f2c4decbca79fa2d0de6e.jpg)  
Extended Data Figure 8. Reaction condition search space for the prospective asymmetric ketone hydrogenation comprising 32 chiral catalysts, 7 solvents, 9 bases, 2 base mol%, and 2 temperatures.

# Supplementary Information Dynamic language model representations for multi-objective reaction optimisation

Joshua W. Sin<sup>1,2†</sup>, David Ming Segura<sup>1,2,3†</sup>, Bojana Rankovi´c<sup>2,3†</sup>, Siu Lun Chau<sup>4</sup>, Marius D. R. Lutz<sup>5</sup>, Andrea Anelli<sup>5</sup>, Ryan P. Burwood<sup>6</sup>, Kurt P¨untener<sup>1</sup>, Maximilian J. Notheis<sup>1</sup>, Raphael Bigler<sup>1</sup>, Philippe Schwaller<sup>2,3\*</sup>

<sup>1</sup>Process Chemistry & Catalysis, Synthetic Molecules Technical Development, F. Hofmann-La Roche AG, Basel, Switzerland

<sup>2</sup>Laboratory of Artificial Chemical Intelligence (LIAC), EPFL, Lausanne, Switzerland

<sup>3</sup>National Centre of Competence in Research (NCCR) Catalysis, EPFL, Lausanne, Switzerland

<sup>4</sup>Epistemic Intelligence & Computation Lab, College of Computing & Data Science, Nanyang Technological University, Singapore

<sup>5</sup>Roche Pharma Research and Early Development (pRED), F. Hofmann-La Roche AG, Basel, Switzerland

<sup>6</sup>Solid State Sciences, Synthetic Molecules Technical Development, F. Hofmann-La Roche AG, Basel, Switzerland

†These authors contributed equally to this work. \*Corresponding authors. Email: philippe.schwaller@epfl.ch

## Contents

1 High-throughput experimentation (HTE) platform 2   
2 Prospective application: Palladium-catalysed cyanation 4   
2.1 HTE experimental procedure 4   
2.2 Scale-up and synthesis of methyl 3-cyano-5-fluorobenzoate 4   
3 Prospective application: Asymmetric ketone hydrogenation 7   
3.1 HTE experimental procedure 7   
3.2 Scale-up and synthesis of (1S, 2R)-1,2-diphenylpropan-1-ol . 7   
3.3 Enantiomeric data augmentation 10   
3.4 Model sensitivity to initial training data 10

## 1 High-throughput experimentation (HTE) platform

We conducted all high-throughput experiments using a parallel experimentation platform customdesigned by UnchainedLabs (Supplementary Figure 1). This system comprises two interconnected Big Kahuna platforms, one of which is further integrated with a LiCONiC LiCotel system. The entire setup is encased within an LC Technology Solutions glove box featuring dual circulation systems, with one dedicated to solid dispensing and another to reaction execution. LC-MS analysis was performed using an ACQUITY UPLC I-Class system with QDa from Waters. For Chiral HPLC analyses, a Chiralcel OJ-3 column from Daicel was used, with ethanol and heptane as the mobile phase.

Solid components were dispensed with a Mettler balance in vial dispense mode, using SV hoppers for precursors and ligands and SV hoppers (≤ 15 mg) or 10 mL classic hoppers (>15 mg) for solid additives. For target dispense quantities < 0.4 mg, materials were dispensed as coated ChemBeads. Following automated solid dispensing, liquid components (substrates, liquid additives, and solvents) were manually added using an Eppendorf Multipette E3 single channel pipette (4987000010). Reactions were performed in standard 96-position parallel synthesis reaction blocks (Analytical Sales and Services, SKU: 96960), with V&P Scientific super tumble stir discs (VP 721F-1) for stirring, or in a Screening Pressure Reactor (SPR) from UnchainedLabs. Data analysis was performed using the HTE OS workflow described by Wuitschik et al. [1] HTE OS is integrated with a SpotFire application that enables tagging of LC-MS and HPLC signals into categories (e.g., “limiting SM”, “other SM”, “solvent”, “ignore peak”). During analysis, all peaks were tagged accordingly, with peaks corresponding to ligands and precatalyst components designated as “ignore peak”. For LCAP calculations, signals tagged as “other SM”, “solvent”, or “ignore peak” were excluded from consideration.

![](images/51df3a45c506f55df2f22af2e8ef28f3302de740478294ba889d8217d9daf65a.jpg)  
LiCotel stores up to 4,500 compounds (e.g., catalysts, substrates)

![](images/8e5b75a7c47e15103acbd78d67348d291eb4fa2497cdcf5c187dea915ae0ae4c.jpg)  
Solid Dosing Big Kahuna with two balances for precise (submg) solid dosing to vials or plates

![](images/968e0b70032f125f685865b5cf99872ac1cfd99a7803585dff8bf777c20970b5.jpg)  
Supplementary Figure 1: Schematic overview of the high-throughput experimentation (HTE) setup.

![](images/15fbf591d5e9ace586b12f6f5676d0f0c210baef2e00cbf23b3f3d155cf9b509.jpg)

## 2 Screening LC/MS

highly performant UPLC with short standard methods to measure up to 1,000 samples/d

![](images/afac23bffea8673cbcbab085a94197eb01f99a3389f8811408bc7b566b0ae66c.jpg)  
Reaction Execution Big Kahuna for accurate liquid dispensing and parallel reaction execution (-20 to 150 °C)

## 2 Prospective application: Palladium-catalysed cyanation

## 2.1 HTE experimental procedure

For all HTE plates, the following experimental procedure was performed. Palladium catalysts (1 mol%), cyanide sources, and solid additives were dispensed into 1 mL vials with stirring disks in a 96-well plate. Methyl 3-bromo-5-fluorobenzoate $( 1 9 . 0 \mathrm { \textmu L }$ , 129 µmol), solvents $\left( 2 4 0 ~ \mathrm { { \textmu L } } \right)$ , and cosolvent (120 µL) were added to the vials, followed by the addition of liquid additive $\mathrm { N E t _ { 3 } }$ (26.9 µL, 1.50 eq) where applicable. The plate was sealed and stirred (400 rpm) at the specified temperature for 20 h. $4 / 1 \ \mathrm { M e C N / H _ { 2 } O \ ( 4 0 0 \ \mu L ) }$ was added to each well and shaken for 20 min at room temperature. Samples were taken and analysed by Liquid Chromatography-Mass Spectrometry (LC-MS).

## 2.2 Scale-up and synthesis of methyl 3-cyano-5-fluorobenzoate

![](images/50b6a614ce06ab28b9d486485d51eac103d19a61acd8408d3f15d22182d8da29.jpg)

A nitrogen-flushed 100 mL Easymax flask with overhead stirring was charged with methyl 3-bromo-5- fluorobenzoate (3.17 mL, 21.5 mmol) and DMC (40 mL). In a glovebox, a solution of $\mathrm { K _ { 4 } F e ( C N ) _ { 6 } \cdot 3 H _ { 2 } O }$ (4.53 g, 10.7 mmol, 0.50 eq) in water (20 mL) was prepared and added to the Easymax flask under argon. The mixture was heated to $7 5 ^ { \circ } \mathrm { C }$ . Once the temperature was reached, [Pd(allyl)(Xantphos)]Cl (163.4 mg, 215 µmol, 1.0 mol%) was added and the mixture was stirred at $7 5 ~ ^ { \circ } \mathrm { C }$ for 4 h. The reaction was transferred with degassed 25% aq. NaCl solution (100 mL) to a separation funnel and extracted three times with degassed EtOAc $( 3 \times 1 0 0 ~ \mathrm { m L } )$ . The combined organic phases were washed with sat. aq. NaCl solution, dried over $\mathrm { N a _ { 2 } S O _ { 4 } }$ , filtered, and the solvent was removed using a rotary evaporator. The crude product was purified by flash column chromatography on silica gel (EtOAc:heptane = 0:100 to 30:70) over 20 min, 80 g, only UV active fractions collected), afording the product as a white solid. Yield: 3.61 g (94%).

$$
 \begin{array} { r l } & { ^ { 1 } \mathbf { H } \mathbf { \Delta } \mathbf { N } \mathbf { M } \mathbf { R } \left( \mathrm { 4 0 0 } \mathbf { \Delta } \mathrm { M } \mathbf { f } \mathbf { z } , \mathrm { C D C l } _ { 3 } \right) ; \mathbf { \Delta } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { 8 } - 8 . 1 3 \left( m , \mathrm { 1 H } , \mathbf { A r } H \right) , 8 . 0 0 - 7 . 9 5 \left( m , \mathrm { 1 H } , \mathbf { A r } H \right) , 7 . 5 8 - 7 . 5 3 \left( m , \mathrm { 1 } \right) } \\ &  \mathrm { 1 H } , \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta }  \end{array}
$$

$^ { 1 3 } { \bf C } \{ ^ { 1 } { \bf H } \}$ NMR (101 MHz, CDCl ): δ 164.0 $( d , \ ^ { 4 } J _ { \mathrm { F , C } } = 2 . 9$ Hz, CO Me), 162.1 $( d , \ ^ { 1 } J _ { \mathrm { F , C } } = 2 5 2 . 4$ Hz, arom.), 133.8 $( d , { } ^ { 3 } J _ { \mathrm { F , C } } = 7 . 7$ Hz, arom.), $1 2 9 . 2 \ : ( d , \ : ^ { 4 } J _ { \mathrm { F , C } } = 3 . 7 $ Hz, arom.), 123.1 $( d , { } ^ { 2 } J _ { \mathrm { F , C } } = 2 4 . 6 $ Hz, arom.), 121.4 $( d , \ ^ { 2 } J _ { \mathrm { F , C } } = 2 3 . 1 $ Hz, arom.), $1 1 6 . 7 \ : ( d , \ : ^ { 4 } J _ { \mathrm { F , C } } = 2 . 9 \ : \mathrm { H z } , \ : C \mathrm { N } )$ , 114.3 $( d , \ ^ { 3 } J _ { \mathrm { F , C } } = 8 . 8$ Hz, arom.), $5 3 . 0 \ \mathrm { ( O C H _ { 3 } ) }$

![](images/39d7bc82fd1b0b3611feecdc79fe673e26115c031e5b6f04e518c440fd60129d.jpg)  
Supplementary Figure 2: <sup>1</sup>H NMR (400 MHz, CDCl<sub>3</sub>) of methyl 3-cyano-5-fluorobenzoate

![](images/f730eb24f6b4574b848c539c80d1e02147d7b103994fd45a665794bc88b577ac.jpg)  
Supplementary Figure 3: $^ { 1 3 } \mathrm { C } \{ ^ { 1 } \mathrm { H } \}$ NMR (101 MHz, CDCl<sub>3</sub>) of methyl 3-cyano-5-fluorobenzoate

## 3 Prospective application: Asymmetric ketone hydrogenation

## 3.1 HTE experimental procedure

For all HTE plates, the following experimental procedure was performed inside a nitrogen filled glovebox. Iridium and ruthenium catalysts (1 mol%) and solid bases were dispensed into 1 mL vials in a 96-well plate. Stock solutions of 1,2-diphenylpropan-1-one (20 mg, 95.1 µmol in solvent (200 µL)) were prepared and added to the vials, followed by addition of liquid bases, where applicable. The plate was sealed inside the glovebox, placed in a screening pressure reactor (SPR) and shaken overnight at the indicated temperature under 20 bar $\operatorname { H } _ { 2 } .$ . The solvent was removed using a GeneVac and the residue redissolved in EtOH (200 µL). Samples were taken and analysed by chiral HPLC.

## 3.2 Scale-up and synthesis of (1S, 2R)-1,2-diphenylpropan-1-ol

![](images/bd51dcbf83497c691611bb048250cc62b2a48f74a25989a49766a3d6f3046c9d.jpg)

In an argon filled glovebox, $\mathrm { [ I r ( C l ) H _ { 2 } ( ( S ) }$ – DTB – SpiroPAP – 3-Me)] (93.1 mg, 95.1 µmol, 0.01 eq) and 1,2-diphenylpropan-1-one (2000 mg, 9.51 mmol, 1.0 eq) were added to a 50 mL autoclave. Degassed tAmOH (16.1 g, 20 mL, 19.2 eq) and DBU (724.0 mg, 711.2 µL, 4.76 mmol, 0.5 eq) were subsequently added to the reaction mixture. The autoclave was then sealed, pressurised with Ar (5 bar) and removed from the glove box. The autoclave was connected to a $\mathrm { H _ { 2 } }$ line, purged five times with $\mathrm { H _ { 2 } }$ (10 bar) and stirred at $6 0 ~ ^ { \circ } \mathrm { C }$ and 500 rpm under 20 bar of $\mathrm { H _ { 2 } }$ . Reaction progress was monitored via $\mathrm { H _ { 2 } }$ uptake, reaching completion within 4 h. The autoclave was allowed to cool to room temperature and carefully depressurised. The deep yellow reaction mixture was transferred with EtOAc (130 mL) to a separation funnel and washed sequentially with 1 M aqueous HCl solution (2 \* 50 mL), water (50 mL) and saturated aq. NaCl solution (50 mL). The organic phase was dried over $\mathrm { N a _ { 2 } S O _ { 4 } }$ , filtered, and concentrated under reduced pressure to aford a lightly brown oil.

This crude oil was purified by flash column chromatography, eluting with EtOAc/heptane $( 1 8 \% , \mathrm { v / v ) }$ Solvent removal under reduced pressure yielded 1,2-diphenylpropan-1-ol as a transparent, oily solid (1.99 g). The crude solid was then dissolved in pentane (10 mL) and allowed to crystallise by slow evaporation over 48 h at room temperature. The resulting crystal bearing oily residue on the surface was transferred into a 15 mL glass vial. Ice-cold pentane (3 mL) was added to submerge the crystal. The vial was gently swirled for 3–5 seconds, after which the supernatant was drawn of. The crystal was rinsed a second time with ice-cold pentane (3 mL) and immediately decanted. The purified crystal was transferred onto lint-free filter paper and was allowed to dry to constant mass at room temperature, afording the product. Yield: 1687.89 mg (83.6%).

<sup>1</sup>H NMR (400 MHz, DMSO – d<sub>6</sub>): δ 7.22 – 7.07 (m, 10H, ArH), 5.31 $( d , \ ^ { 3 } J _ { \mathrm { H , H } ^ { \prime } } = 4 . 8 $ Hz, 1H, (H)COH ), 4.64 (dd, <sup>3</sup>J<sub>H,H’</sub> = 4.6 Hz, $^ 3 J _ { \mathrm { H , H } } , = 6 . 4$ Hz, 1H, (H )COH), 2.96 $( p , \ ^ { 3 } J _ { \mathrm { H , H } } , = 6 . 9$ Hz, 1H, (H )CCH<sub>3</sub>), 1.24 $( d , \mathrm { } ^ { 3 } J _ { \mathrm { H , H } } , = 7 . 0$ Hz, 3H, $\mathrm { ( H ) C C } H _ { 3 } \mathrm { ) }$

V119360\_1\_\_ELN042419\_432\_1\_final\_sample\_dmso.jdx   
DMSO-d6   
16 H's

<sup>13</sup>C{<sup>1</sup>H} NMR (101 MHz, DMSO – d<sub>6</sub>): δ 145.41 (arom.), 145.18 (arom.), 128.52 (arom.), 128.23 (arom.), 127.95 (arom.), 126.99 (arom.), 126.90 (arom.), 126.25 (arom.), 77.49 (COH), 47.53 $\left( C \mathrm { C H _ { 3 } } \right)$ 16.97 $\mathrm { ( C C H _ { 3 } ) }$  
![](images/9e32d9eb79938fa8c1795e63e572f725260ba3f24194bb4c8ed8f1c7f07792ca.jpg)  
Supplementary Figure 4: <sup>1</sup>H NMR (400 MHz, DMSO – d<sub>6</sub>) of (1S, 2R)-1,2-diphenylpropan-1-ol

![](images/47b3e5880ca8de9a5d8ed19d52700844a9472dca348eb6c4366be13de3859cbe.jpg)  
Supplementary Figure 5: $^ { 1 3 } \mathrm { C } \{ ^ { 1 } \mathrm { H } \}$ NMR (101 ${ \mathrm { M H z } } ,$ , DMSO – d<sub>6</sub>) of (1S, 2R)-1,2-diphenylpropan-1-ol

## 3.3 Enantiomeric data augmentation

All solvents and bases in the design space are achiral, and all 32 chiral catalysts are present as enantiomeric pairs. A catalyst and its enantiomer are therefore expected to yield mirror-image product dis tributions under otherwise identical conditions. Conversion and diastereomeric excess are unchanged, while the enantiomeric excess changes sign. We exploited this symmetry as a data augmentation strategy. For each observed reaction, a reflected counterpart was added to the training data in which the catalyst was replaced by its enantiomer, the sign of ee (syn) inverted, and conversion and de (syn) retained. Augmented observations were used to fit the surrogate models only. The acquisition function was evaluated over the full design space excluding conditions already run experimentally.

## 3.4 Model sensitivity to initial training data

The first initialisation plate in the HTE reaction optimisation campaign contained only five conditions producing the target syn diastereomer. To assess how strongly the round-2 optimisation outcome depended on these observations, we repeated the optimisation with the top n conditions removed from the Plate 1 training data, for n = 1, 3, and 5, ranking conditions by de (syn).

In each case the surrogate models were refitted on the reduced dataset and the acquisition function was evaluated over the full design space, excluding conditions already queried. No new experiments were performed: suggested conditions for which experimental data were already available were compared against the best condition remaining in the ablated training set. Because only suggestions already measured in round 2 could be evaluated, the values below are lower bounds on the performance recoverable at each ablation level.

With the single most diastereoselective condition removed (n = 1), the best remaining training observation reached 47.9% de (syn). The model recovered conditions delivering 99.4% conversion, 75.4% de (syn), and 89.5% ee (syn), improving stereoselectivity. With the top three removed $( n = 3 )$ , the best remaining training observation reached 23.5% de (syn), and the model recovered conditions at 64.2% de (syn), again with higher enantioselectivity. Removing all five conditions $( n = 5 )$ that were syn-selective left no positive diastereoselectivity examples in the training data, yet the model still recovered conditions reaching 83.4% de (syn) and 93.3% ee (syn). Even with no syn-selective examples to learn from, the model identified conditions delivering high syn selectivity.

## References

[1] G. Wuitschik, V. Jost, T. Schindler, M. Jakubik, Organic Process Research & Development 2024, 28, 2875–2884, DOI 10.1021/acs.oprd.4c00160.